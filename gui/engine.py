"""GUI control layer over VoxTerm's engine.

Exposes the operations the GUI drives — start/stop recording (via VoxTerm's own
``AudioCapture``), the background transcribe+export job, and session history — as a
small thread-safe object the HTTP server calls. No transcription/diarization logic
lives here; recording reuses ``audio.capture.AudioCapture`` and the heavy lifting is
``gui.transcribe`` + ``gui.export`` (the reviewed, tested pipeline).

Core flow: record -> stop -> transcribe (robust, reuses the tested pipeline). The live
monitor (_live_loop) tails the in-progress WAV: VAD-chunked for batch backends
(_live_chunk_loop), or true word-by-word streaming via _live_stream_loop when the optional
sherpa-onnx backend is installed.
"""
from __future__ import annotations

import os
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import config  # noqa: E402
from audio.mix import mix_chunks  # noqa: E402
from gui._timefmt import fmt_hms  # noqa: E402
from gui import transcribe, export  # noqa: E402

OUT_DIR = Path(os.environ.get("VOXTERM_GUI_OUT_DIR") or (Path.home() / "voxterm-live"))
SR = config.SAMPLE_RATE


def _wav_header(data_len: int, sr: int = SR) -> bytes:
    """The canonical 44-byte PCM WAV header (mono, s16). ``data_len`` may be 0 as a
    placeholder while the file is still growing — it's patched on close. Tailers read raw
    PCM past byte 44 regardless, so a placeholder size never breaks live monitoring."""
    return (b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", data_len))


def _pcm_bytes(chunk: np.ndarray) -> bytes:
    """float32 [-1,1] → little-endian s16 bytes (the recording's PCM encoding)."""
    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Engine:
    def __init__(self, out_dir: Path = OUT_DIR):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cap = None        # mic (audio.capture.AudioCapture)
        self._sys = None        # system/loopback audio (audio.system_capture.SystemCapture)
        self._source = "mic"    # "mic" | "system" | "both" — chosen at start_recording
        # Recording streams straight to a growing WAV on disk (so the live monitor can tail
        # THIS recording, and so a long session doesn't sit entirely in RAM).
        self._rec_file = None
        self._rec_wav_path = None
        self._rec_bytes = 0
        self._poll_thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.recording = False
        self.level = 0.0
        self.started_at = None
        self.job = {"state": "idle"}  # idle | transcribing | done | error
        # live (near-real-time) monitor of an in-progress recording (reads the file)
        self._live = {"active": False, "wav": None, "lines": [], "partial": None}
        self._live_stop = threading.Event()
        self._live_thread = None
        self._stab = None  # PartialStabilizer for the in-progress (volatile) tail
        # remember the OS default input device once, so picking "System default" reverts a prior
        # explicit choice instead of leaving sd.default.device pinned to it
        self._sd_captured = False
        self._sd_default_in = None

    # ---- static option lists for the UI ----
    def models(self) -> list[str]:
        # Offer exactly what the TUI offers: every model the engine supports on this host.
        # config.py's platform branches + sherpa gate register them all in AVAILABLE_MODELS
        # (fw-* on Linux/Intel; qwen3 on Linux/Win when qwen-asr is installed; MLX qwen3/parakeet
        # on Apple Silicon; sherpa-* when the [streaming] extra is installed). Earlier this read
        # FASTER_WHISPER_MODELS and silently hid installed qwen3 on Linux — fixed to match the TUI.
        return sorted(config.AVAILABLE_MODELS)

    def default_model(self) -> str:
        # CPU-friendly default (fw-base on Linux/Intel, fw-small fallback; MLX on Apple Silicon) — not the raw
        # config.DEFAULT_MODEL, which is qwen3-0.6b when qwen-asr is installed (slow on CPU).
        return transcribe.gui_default_model()

    def languages(self) -> dict:
        return dict(config.AVAILABLE_LANGUAGES)

    def input_devices(self) -> list[dict]:
        """Microphones the user can pick from. Skips ALSA resampler/mixer plugins (noise) and
        de-dupes by name; index -1 means 'system default'."""
        out = [{"index": -1, "name": "System default"}]
        try:
            import sounddevice as sd
            skip = ("lavrate", "samplerate", "speex", "upmix", "vdownmix", "dmix", "surround", "jack", "null")
            seen = set()
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) <= 0:
                    continue
                name = (d.get("name") or "").strip()
                low = name.lower()
                if not name or name in seen or any(s in low for s in skip):
                    continue
                seen.add(name)
                out.append({"index": i, "name": name})
        except Exception:
            pass
        return out

    def warm(self) -> None:
        """Preload the default model + VAD + diarizer in the background so the first recording
        doesn't pay cold-start latency. Best-effort; called once at server startup."""
        threading.Thread(target=lambda: transcribe.preload(language="en"),
                          daemon=True, name="gui-warm").start()

    # ---- recording ----
    def start_recording(self, device: int | None = None, source: str = "mic") -> dict:
        with self._lock:
            if self.recording:
                return {"ok": True, "already": True}
            # source: "mic" (default), "system" (loopback / what's playing), or "both" (mixed).
            self._source = source if source in ("mic", "system", "both") else "mic"
            self._cap = None
            self._sys = None
            if self._source in ("mic", "both"):
                from audio.capture import AudioCapture
                try:                                   # tolerate a malformed device value from the client
                    dev = int(device) if device is not None else -1
                except (ValueError, TypeError):
                    dev = -1
                # steer AudioCapture's input to the chosen mic; -1 = system default. We mutate the
                # global sd.default.device, so remember the OS default once and restore it when the
                # user re-selects "System default" (otherwise a prior explicit choice stays pinned).
                try:
                    import sounddevice as sd
                    if not self._sd_captured:
                        cur0 = sd.default.device
                        self._sd_default_in = list(cur0)[0] if isinstance(cur0, (list, tuple)) else cur0
                        self._sd_captured = True
                    cur = sd.default.device
                    pair = list(cur) if isinstance(cur, (list, tuple)) else [cur, cur]
                    pair[0] = dev if dev >= 0 else self._sd_default_in
                    sd.default.device = tuple(pair)
                except Exception:
                    pass
                try:
                    self._cap = AudioCapture()
                    self._cap.start()
                except Exception as e:  # no input device / busy / permission
                    self._cap = None
                    self.recording = False
                    return {"ok": False, "error": f"could not open the microphone: {e}"}
            if self._source in ("system", "both"):
                try:
                    from audio.system_capture import SystemCapture
                    self._sys = SystemCapture()
                    self._sys.start()
                    if not self._sys.is_active:        # parec missing, no monitor source, SCK denied…
                        raise RuntimeError(self._sys.status_message or "system audio capture unavailable")
                except Exception as e:
                    try:
                        if self._cap:
                            self._cap.stop()
                    except Exception:
                        pass
                    self._cap = None
                    self._sys = None
                    self.recording = False
                    return {"ok": False, "error": f"could not capture system audio: {e}"}
            # open the growing WAV now (placeholder header, patched on stop) so the live
            # monitor can tail this very recording and click-Live follows what you're saying.
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._rec_wav_path = self.out_dir / f"{ts}-gui.wav"
            try:
                self._rec_file = open(self._rec_wav_path, "wb")
                self._rec_file.write(_wav_header(0))
                self._rec_file.flush()
            except OSError as e:
                try:
                    self._cap.stop()
                except Exception:
                    pass
                self._cap = None
                self.recording = False
                return {"ok": False, "error": f"could not open the recording file: {e}"}
            self._rec_bytes = 0
            self._stop.clear()
            self.recording = True
            self.started_at = time.time()
            self.level = 0.0
            self._poll_thread = threading.Thread(target=self._poll, daemon=True, name="gui-rec-poll")
            self._poll_thread.start()
            return {"ok": True, "wav": str(self._rec_wav_path)}

    @staticmethod
    def _drain(cap) -> list:
        """Drain a capture's queued chunks as float32 arrays ([] if absent/errored)."""
        if cap is None:
            return []
        try:
            chunks = cap.drain()
        except Exception:
            return []
        return [np.asarray(c, dtype=np.float32) for c in chunks if c is not None and len(c)]

    def _poll(self):
        while not self._stop.is_set():
            mic = self._drain(self._cap)
            sysa = self._drain(self._sys)
            fresh = mix_chunks(mic, sysa) if (mic and sysa) else (mic or sysa)
            if fresh:
                with self._lock:                       # serialize with stop's finalize
                    if self._rec_file:
                        for c in fresh:
                            b = _pcm_bytes(c)
                            self._rec_file.write(b)
                            self._rec_bytes += len(b)
                        self._rec_file.flush()         # make new audio visible to the live tailer
                last = fresh[-1]
                if len(last):
                    self.level = float(np.sqrt(np.mean(np.square(last))))
            time.sleep(0.066)  # ~15 Hz

    def stop_recording(self, model: str | None = None, language: str = "en", diarize: bool = True) -> dict:
        model = model or transcribe.gui_default_model()
        if not self.recording:
            return {"ok": False, "error": "not recording"}
        # Signal + join the poll thread WITHOUT holding self._lock (the poll thread takes
        # the lock to append, so holding it here would deadlock). Once joined, no more
        # appends can race the final drain/concat/clear.
        self._stop.set()
        # The live monitor is bound to this recording's lifetime: stop it before we finalize
        # the file, so its reader is gone and it can't re-decode the finalized WAV forever or
        # run inference concurrently with the post-stop transcribe job. Best-effort (idempotent).
        self.live_stop()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        with self._lock:
            self.recording = False
            try:
                mic = self._drain(self._cap)          # any frames still queued in either stream
                sysa = self._drain(self._sys)
                for c in (mix_chunks(mic, sysa) if (mic and sysa) else (mic or sysa)):
                    b = _pcm_bytes(c)
                    self._rec_file.write(b)
                    self._rec_bytes += len(b)
            except Exception:
                pass
            for cap in (self._cap, self._sys):
                try:
                    if cap:
                        cap.stop()
                except Exception:
                    pass
            wav = self._rec_wav_path
            n_bytes = self._rec_bytes
            patched = False
            try:                                  # patch the header with the real size → valid WAV
                self._rec_file.flush()
                self._rec_file.seek(0)
                self._rec_file.write(_wav_header(n_bytes))
                self._rec_file.flush()
                self._rec_file.close()
                patched = True
            except Exception as e:                # surface I/O failure — don't transcribe a broken file
                self.job = {"state": "error", "error": f"could not finalize recording: {e}"}
            self._rec_file = None
        if not patched:
            return {"ok": False, "error": "could not finalize the recording file"}
        if n_bytes < SR:  # < 0.5s of s16 mono (SR*2 bytes/s → 0.5s = SR bytes)
            try:
                wav.unlink()
            except OSError:
                pass
            self.job = {"state": "error", "error": "recording too short"}
            return {"ok": False, "error": "recording too short"}
        self.job = {"state": "transcribing", "frac": 0.0, "msg": "starting", "wav": str(wav)}
        # load + transcribe off the request thread (matches transcribe_existing)
        threading.Thread(
            target=lambda: self._do_transcribe(transcribe.load_wav_16k_mono(wav), model, language, str(wav), diarize),
            daemon=True, name="gui-transcribe").start()
        return {"ok": True, "wav": str(wav), "seconds": round(n_bytes / (SR * 2), 1)}

    def _do_transcribe(self, audio, model, language, wav, diarize: bool = True):
        try:
            def prog(frac, msg):
                self.job = {"state": "transcribing", "frac": round(frac, 3), "msg": msg, "wav": wav}
            r = transcribe.transcribe_audio(audio, self.out_dir, model=model, language=language, progress=prog, diarize=diarize)
            md_path, json_path, srt_path, vtt_path = export.export(Path(r["events_path"]), self.out_dir)
            stem = Path(r["transcript_path"]).stem.replace("-transcript", "")
            # The WAV is named at record time (<ts>-gui.wav) but the session stem is the
            # transcribe time, so they differ — link the audio under the stem so the GUI can
            # offer Download WAV / playback for this session (see audio_path).
            self._link_audio(wav, stem)
            self.job = {"state": "done", "wav": wav, **r,
                        "agent_md": str(md_path), "agent_json": str(json_path),
                        "agent_srt": str(srt_path), "agent_vtt": str(vtt_path),
                        "stem": stem}
        except Exception as e:
            self.job = {"state": "error", "error": f"{type(e).__name__}: {e}"}

    def transcribe_existing(self, wav_path: str, model: str | None = None, language: str = "en", diarize: bool = True) -> dict:
        """Transcribe an already-recorded WAV (e.g. a prior capture) in the background."""
        model = model or transcribe.gui_default_model()
        p = Path(wav_path)
        if not p.exists():
            return {"ok": False, "error": "no such file"}
        self.job = {"state": "transcribing", "frac": 0.0, "msg": "starting", "wav": str(p)}
        threading.Thread(target=lambda: self._do_transcribe(transcribe.load_wav_16k_mono(p), model, language, str(p), diarize),
                         daemon=True, name="gui-transcribe").start()
        return {"ok": True}

    def status(self) -> dict:
        with self._lock:                          # consistent snapshot vs the live thread's writes
            live = {"active": self._live["active"], "wav": self._live["wav"],
                    "lines": self._live["lines"][-120:], "partial": self._live.get("partial")}
        return {
            "recording": self.recording,
            "level": round(self.level, 4),
            "elapsed": round(time.time() - self.started_at, 1) if (self.recording and self.started_at) else 0,
            "job": self.job,
            "live": live,
        }

    # ---- live (near-real-time) monitor: tail an in-progress recording's file ----
    def _newest_wav(self):
        wavs = sorted(self.out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        return wavs[0] if wavs else None

    def live_start(self, wav: str | None = None) -> dict:
        with self._lock:
            if self._live_thread and self._live_thread.is_alive():
                return {"ok": False, "error": "live monitor is still stopping; try again"}
            if self._live["active"]:
                return {"ok": True, "already": True, "wav": self._live["wav"]}
            target = Path(wav) if wav else self._newest_wav()
            if not target or not target.exists():
                return {"ok": False, "error": "no recording to monitor"}
            self._live = {"active": True, "wav": str(target), "lines": [], "partial": None}
            from gui.stabilize import PartialStabilizer
            self._stab = PartialStabilizer()
            self._live_stop.clear()
            self._live_thread = threading.Thread(target=self._live_loop, args=(target,), daemon=True, name="gui-live")
            self._live_thread.start()
            return {"ok": True, "wav": str(target)}

    def live_stop(self) -> dict:
        self._live_stop.set()
        # Capture the thread into a local: a concurrent live_stop() (e.g. stop_recording calls
        # this while a separate /api/live/stop request races it) may null self._live_thread
        # between our checks. Operating on the local keeps this idempotent and crash-free —
        # joining the same Thread from two callers is safe.
        t = self._live_thread
        if t and t.is_alive():
            t.join(timeout=3)
            if t.is_alive():                     # still decoding — don't claim it stopped
                return {"ok": False, "error": "live monitor still stopping"}
        self._live_thread = None
        if isinstance(self._live, dict):
            self._live["active"] = False
            self._live["partial"] = None
        return {"ok": True}

    def _live_loop(self, wav: Path):
        # tail raw PCM of the (still-growing) WAV; dispatch to streaming or chunked transcription
        from gui.transcribe import _get_engines, gui_default_model
        from gui.eot import is_incomplete
        from audio.transcriber import SherpaStreamingTranscriber
        try:
            # Prefer the sherpa streaming backend for the live view when it's installed (opt-in)
            # — it streams word-by-word. Else fw-base (light, where it exists), else the platform
            # default (MLX on Apple Silicon). dedicated="live" → its OWN transcriber, never
            # sharing decode state with the post-stop batch job.
            if "sherpa-stream-en" in config.AVAILABLE_MODELS:
                live_model = "sherpa-stream-en"
            elif "fw-base" in config.AVAILABLE_MODELS:
                live_model = "fw-base"
            else:
                live_model = gui_default_model()   # CPU-aware default, never the raw qwen3-0.6b that's unusable on CPU
            tr, vad, _d = _get_engines(live_model, "en", dedicated="live")
        except Exception as e:
            with self._lock:
                self._live["lines"].append({"t": "", "text": f"(live engine error: {e})"})
                self._live["active"] = False
            return
        f = open(wav, "rb")
        f.seek(0, 2)  # tail from the CURRENT end — only NEW speech (true live, no slow backlog replay)
        abs_start = max(0, (f.tell() - 44) // 2)  # samples already recorded before we started (for timestamps)
        try:
            if isinstance(tr, SherpaStreamingTranscriber):
                self._live_stream_loop(tr, f, abs_start)
            else:
                self._live_chunk_loop(tr, vad, f, abs_start, is_incomplete)
        finally:
            f.close()
            with self._lock:
                self._live["active"] = False

    def _live_chunk_loop(self, tr, vad, f, abs_start, is_incomplete):
        """VAD-windowed transcription for batch backends (fw-*/MLX/qwen3/parakeet). The
        original live path, unchanged — finalize speech windows past a tail guard, merge
        mid-clause fragments, publish a LocalAgreement-stabilized volatile partial."""
        buf = np.zeros(0, dtype=np.float32)
        while not self._live_stop.is_set():
            self._live_stop.wait(8.0)
            data = f.read()
            if data:
                n = len(data) - (len(data) % 2)
                if n:
                    buf = np.concatenate([buf, np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0])
            if len(buf) >= SR * 2:
                segs = vad.get_speech_segments(buf, min_speech_ms=500, min_silence_ms=300, max_speech_s=6.0)
                guard = len(buf) - int(SR * 0.6)
                consumed = 0
                for (s, e) in segs:
                    if e > guard:
                        break
                    txt = (tr.transcribe(buf[s:e]).get("text") or "").strip()
                    if txt:
                        # lock only the brief dict mutation, never the slow transcribe/VAD, so
                        # status()/SSE sees a consistent snapshot without stalling.
                        with self._lock:
                            lines = self._live["lines"]
                            if lines and is_incomplete(lines[-1]["text"]):
                                lines[-1]["text"] = (lines[-1]["text"] + " " + txt).strip()
                            else:
                                lines.append({"t": fmt_hms((abs_start + s) / SR), "text": txt})
                            self._live["lines"] = lines[-200:]
                    consumed = e
                if consumed:
                    abs_start += consumed
                    buf = buf[consumed:]
                    if self._stab:
                        self._stab.reset()
            if self._stab is not None and len(buf) >= int(SR * 0.4):
                ptxt = (tr.transcribe(buf).get("text") or "").strip()
                st = self._stab.push(ptxt)
                partial = ({"t": fmt_hms(abs_start / SR), **st}
                           if (st["stable"] or st["volatile"]) else None)
                with self._lock:
                    self._live["partial"] = partial
            else:
                with self._lock:
                    self._live["partial"] = None

    def _live_stream_loop(self, tr, f, abs_start):
        """True word-by-word streaming for the sherpa-onnx backend. One persistent OnlineStream
        is fed the freshly-tailed PCM; the running decode is the volatile partial; sherpa's own
        endpoint detection finalizes a line. Tighter cadence than the chunked path for low
        latency. Same lock discipline on the live-state writes."""
        from audio.transcriber import is_hallucination
        tr.reset_dedup()           # tr is cached + reused across live sessions; clear stale dedup state
        rec = tr.recognizer
        st = rec.create_stream()
        fed = abs_start            # total samples fed (for the current line's start timestamp)
        line_start = abs_start
        while not self._live_stop.is_set():
            self._live_stop.wait(1.0)
            data = f.read()
            if data:
                n = len(data) - (len(data) % 2)
                if n:
                    frame = np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0
                    st.accept_waveform(16000, np.ascontiguousarray(frame))
                    fed += n // 2
                    while rec.is_ready(st):
                        rec.decode_stream(st)
            text = (rec.get_result(st) or "").strip()
            if rec.is_endpoint(st):
                final = (text.capitalize() if text.isupper() else text) if text else ""
                rec.reset(st)
                # parity with the chunked/batch backends: drop hallucinations + consecutive dupes
                if final and (is_hallucination(final, "en") or tr.is_duplicate(final)):
                    final = ""
                with self._lock:
                    if final:
                        self._live["lines"].append({"t": fmt_hms(line_start / SR), "text": final})
                        self._live["lines"] = self._live["lines"][-200:]
                    self._live["partial"] = None
                line_start = fed
            else:
                partial = ({"t": fmt_hms(line_start / SR), "stable": "",
                            "volatile": (text.capitalize() if text.isupper() else text)}
                           if text else None)
                with self._lock:
                    self._live["partial"] = partial

    # ---- session history ----
    def _session_dirs(self) -> list[Path]:
        dirs = [self.out_dir]
        try:
            dirs.append(Path(config.SESSIONS_DIR))
            dirs.append(Path(config.LIVE_DIR))
        except Exception:
            pass
        seen, uniq = set(), []
        for d in dirs:
            if d and d not in seen and d.exists():
                seen.add(d)
                uniq.append(d)
        return uniq

    def sessions(self) -> list[dict]:
        """All sessions across the known dirs, newest first, with which artifacts exist."""
        out = {}
        for d in self._session_dirs():
            for f in d.glob("*-transcript.md"):
                stem = f.stem[: -len("-transcript")]
                out.setdefault((d, stem), {"stem": stem, "dir": str(d), "mtime": f.stat().st_mtime})
                out[(d, stem)]["transcript"] = f.name
            for f in d.glob("*-agent.md"):
                stem = f.stem[: -len("-agent")]
                e = out.setdefault((d, stem), {"stem": stem, "dir": str(d), "mtime": f.stat().st_mtime})
                e["agent_md"] = f.name
                e["mtime"] = max(e.get("mtime", 0), f.stat().st_mtime)
            for f in d.glob("*-agent.json"):
                stem = f.stem[: -len("-agent")]
                e = out.setdefault((d, stem), {"stem": stem, "dir": str(d), "mtime": f.stat().st_mtime})
                e["agent_json"] = f.name
        items = sorted(out.values(), key=lambda x: x.get("mtime", 0), reverse=True)
        for it in items:
            it["title"] = self._session_title(it)
        return items

    def _session_title(self, entry: dict) -> str:
        """A clean, content-based title (the first spoken sentence) instead of a raw timestamp —
        ChatGPT-style. Reads only the head of the transcript file; cached by (path, mtime)."""
        import re
        fname = entry.get("transcript") or entry.get("agent_md")
        if not fname:
            return ""
        p = Path(entry["dir"]) / fname
        try:
            mt = p.stat().st_mtime
        except OSError:
            return ""
        cache = self.__dict__.setdefault("_title_cache", {})
        key = (str(p), mt)
        if key in cache:
            return cache[key]
        title = ""
        try:
            for s in p.read_text(encoding="utf-8")[:3000].splitlines():
                s = s.strip()
                if not s or s[0] in "#>-" or s.lower().startswith("voxterm"):
                    continue
                s = re.sub(r"^\*{0,2}\[[^\]]*\]\*{0,2}\s*", "", s)              # drop [timestamp]
                s = re.sub(r"^\*{0,2}[^*:]{1,30}\*{0,2}\s*(\(#\d+\))?\s*[:：]\s*", "", s)  # drop Speaker:
                s = re.sub(r"[*_`]", "", s).strip()
                if len(s) >= 2:                       # keep short first utterances ("Hello.", "Yes.")
                    s = re.sub(r"\s+", " ", s)
                    title = (s[:54].rstrip() + "…") if len(s) > 56 else s
                    break
        except Exception:
            title = ""
        cache[key] = title
        return title

    def _resolve(self, stem: str, suffix: str, only_dir: str | None = None) -> Path | None:
        # prevent traversal: stem must be a bare name
        if "/" in stem or ".." in stem:
            return None
        dirs = self._session_dirs()
        if only_dir:  # restrict to that dir IFF it's a known session dir (no traversal)
            od = Path(only_dir)
            dirs = [d for d in dirs if d == od]
        for d in dirs:
            p = d / f"{stem}{suffix}"
            if p.exists():
                return p
        return None

    # text artifacts a session owns (audio .wav is managed separately and never touched)
    _ARTIFACT_SUFFIXES = ["-transcript.md", "-agent.md", "-agent.json",
                          "-agent.srt", "-agent.vtt", "-events.jsonl"]

    def delete_session(self, stem: str, dir: str | None = None) -> dict:
        """Remove ONLY this session's text artifacts for ``stem``.

        Reuses _resolve's traversal guard (reject '/' or '..' in the stem) and resolves
        strictly within _session_dirs() (honoring the optional ``dir`` like _resolve's
        only_dir). Deletes only files that exist; never touches .wav audio or anything
        outside a known session dir. Returns the list of deleted filenames.
        """
        # SAME guard as _resolve: stem must be a bare name (no traversal)
        if "/" in stem or ".." in stem:
            return {"ok": False, "error": "bad stem", "deleted": []}
        deleted: list[str] = []
        for suffix in self._ARTIFACT_SUFFIXES:
            p = self._resolve(stem, suffix, only_dir=dir)  # resolves within known dirs only
            if p and p.is_file():
                try:
                    p.unlink()
                    deleted.append(p.name)
                except OSError:
                    pass
        return {"ok": True, "deleted": deleted}

    def read_artifact(self, stem: str, kind: str, dir: str | None = None) -> dict:
        suffix = {"transcript": "-transcript.md", "agent_md": "-agent.md", "agent_json": "-agent.json",
                  "srt": "-agent.srt", "vtt": "-agent.vtt"}.get(kind)
        if not suffix:
            return {"ok": False, "error": "bad kind"}
        p = self._resolve(stem, suffix, only_dir=dir)
        if not p:
            return {"ok": False, "error": "not found"}
        return {"ok": True, "stem": stem, "kind": kind, "path": str(p), "text": p.read_text(encoding="utf-8")}

    def summarize_session(self, stem: str, dir: str | None = None, template_id: str = "tldr",
                          model: str = "", custom_prompt: str = "") -> dict:
        """Summarize a saved session's transcript with the local LLM the TUI uses
        (MLX on Apple Silicon, or an ``ollama:<model>`` backend on any platform).
        Returns {"ok", "summary"} or a clear {"ok": False, "error"} when no backend is
        available — never raises to the handler. (read_artifact validates the stem.)"""
        art = self.read_artifact(stem, "transcript", dir=dir)
        if not art.get("ok"):
            art = self.read_artifact(stem, "agent_md", dir=dir)
        if not art.get("ok"):
            return {"ok": False, "error": "transcript not found"}
        body = (art.get("text") or "").strip()
        if not body:
            return {"ok": False, "error": "transcript is empty"}
        try:
            from summarizer import SummarizerError, get_summarizer, resolve_template
        except Exception as e:
            return {"ok": False, "error": f"summarizer unavailable: {e}"}
        try:
            summary = get_summarizer(model).summarize(body, resolve_template(template_id), custom_prompt)
            return {"ok": True, "summary": summary, "template": template_id}
        except SummarizerError as e:
            return {"ok": False, "error": str(e)}        # missing/unreachable backend — surfaced to the UI
        except Exception as e:
            return {"ok": False, "error": f"summarization failed: {e}"}

    def _link_audio(self, wav: str, stem: str) -> None:
        """Hardlink the source WAV to '<stem>-gui.wav' so audio_path() can find a session's
        audio by stem. The WAV is named at record time and the stem at transcribe time, so they
        differ; a hardlink costs no extra disk, survives deletion of the original name, and is
        never touched by delete_session (audio is intentionally kept). Best-effort."""
        try:
            src = Path(wav)
            dst = self.out_dir / f"{stem}-gui.wav"
            if not src.is_file() or dst.exists() or src.resolve() == dst.resolve():
                return
            try:
                os.link(src, dst)
            except OSError:        # cross-device, or a filesystem without hardlinks
                import shutil
                shutil.copy2(src, dst)
        except Exception:
            pass

    def audio_path(self, stem: str, dir: str | None = None) -> Path | None:
        """Locate the source WAV for a saved session, or None. New recordings are hardlinked to
        '<stem>-gui.wav' at transcribe time (_link_audio); for legacy sessions we fall back to
        the in-dir .wav whose mtime is closest to the transcript (recording + transcribe happen
        within seconds), bounded to a 1-hour window so we never return an unrelated file."""
        if "/" in stem or ".." in stem:
            return None
        for suffix in ("-gui.wav", ".wav"):           # direct link (the normal path)
            p = self._resolve(stem, suffix, only_dir=dir)
            if p and p.is_file():
                return p
        ref = self._resolve(stem, "-transcript.md", only_dir=dir) or self._resolve(stem, "-agent.json", only_dir=dir)
        if not ref:
            return None
        try:
            ref_mt = ref.stat().st_mtime
        except OSError:
            return None
        dirs = [d for d in self._session_dirs() if (not dir or d == Path(dir))]
        best, best_dt = None, 3600.0                  # accept only a match within 1 hour
        for d in dirs:
            for w in list(d.glob("*-gui.wav")) + list(d.glob("*.wav")):
                try:
                    dt = abs(w.stat().st_mtime - ref_mt)
                except OSError:
                    continue
                if dt < best_dt:
                    best, best_dt = w, dt
        return best

    def export_session(self, stem: str, kind: str, renames: dict | None = None, dir: str | None = None) -> dict:
        """Render a saved session to md/json/srt/vtt with the client's speaker renames applied.

        Rebuilds the doc from the events log via export.build() — the SAME path that produced the
        on-disk -agent.* artifacts — then renders with export.py's formatters. So a download
        byte-matches the on-disk file except for the (intentional) renames, and there is ONE
        formatter implementation (the client no longer reimplements it)."""
        render = {"md": export.render_md, "json": export.render_json,
                  "srt": export.to_srt, "vtt": export.to_vtt}.get(kind)
        ext = {"md": "-agent.md", "json": "-agent.json", "srt": ".srt", "vtt": ".vtt"}.get(kind)
        if not render:
            return {"ok": False, "error": "bad kind"}
        ev = self._resolve(stem, "-events.jsonl", only_dir=dir)
        if not ev:
            return {"ok": False, "error": "no events log for this session"}
        try:
            doc = export.build(export.load_events(ev), session_id=stem, source_stream=ev.name)
        except Exception as e:
            return {"ok": False, "error": f"export build failed: {e}"}
        renames = {str(k): str(v) for k, v in (renames or {}).items()}
        if renames:  # mirror the client view: rename local (non-peer) turns + speakers by id
            for t in doc.get("turns", []):
                if not t.get("peer") and str(t.get("speaker_id")) in renames:
                    t["speaker"] = renames[str(t["speaker_id"])]
            for sp in doc.get("speakers", []):
                if not sp.get("peer") and str(sp.get("id")) in renames:
                    sp["label"] = renames[str(sp["id"])]
        return {"ok": True, "text": render(doc), "filename": f"{stem}{ext}"}
