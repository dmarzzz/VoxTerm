#!/usr/bin/env python3
"""VOXTERM — Cyberpunk TUI Voice Transcription Engine"""

from __future__ import annotations

import gc
import sys
import os
import shutil
import subprocess
import json
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Internal runtime defaults — prevent known framework conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Limit PyTorch internal threads to prevent C++ runtime conflicts with MLX
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Suppress leaked-semaphore warning from multiprocessing resource tracker
# on hard exit (os._exit). SpeechBrain/PyTorch create semaphores that aren't
# cleaned up before our forced exit — harmless, OS reclaims them immediately.
import diagnostics
diagnostics.setup_faulthandler()
diagnostics.rotate_crash_logs()

import warnings
warnings.filterwarnings("ignore", message="resource_tracker", category=UserWarning)

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen
from textual import work

from widgets.header import CyberHeader
from widgets.waveform import WaveformWidget, _make_style
from widgets.transcript import TranscriptPanel
from widgets.tag_screen import SpeakerTagScreen
from widgets.profile_screen import SpeakerProfileScreen
from audio.capture import AudioCapture
from audio.buffer import AudioBuffer
from audio.system_capture import SystemCapture
from transcriber.engine import Qwen3Transcriber, WhisperTranscriber, FasterWhisperTranscriber
from diarization.proxy import DiarizationProxy
from speakers.store import SpeakerStore
from audio.vad import SileroVAD
from config import (
    SAMPLE_RATE, CHUNK_SIZE, WAVEFORM_FPS,
    SILENCE_THRESHOLD, SILENCE_TRIGGER_SECONDS,
    MAX_BUFFER_SECONDS, MIN_BUFFER_SECONDS,
    DEFAULT_MODEL, AVAILABLE_MODELS, QWEN3_MODELS, FASTER_WHISPER_MODELS,
    DEFAULT_LANGUAGE, AVAILABLE_LANGUAGES,
    LIVE_DIR,
)
from paths import SESSIONS_DIR, STATE_FILE


def _clipboard_cmd() -> list[str] | None:
    """Return the clipboard copy command for this platform."""
    if sys.platform == "darwin":
        return ["pbcopy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    if shutil.which("wl-copy"):
        return ["wl-copy"]
    return None


def _load_state() -> dict:
    """Load persisted state from disk. Returns {} on any failure."""
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    """Write state dict to disk. Silently ignores errors."""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


class ModelSelectScreen(ModalScreen):
    """Modal for selecting a whisper model."""

    DEFAULT_CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model-dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: heavy #6644cc;
        border-title-color: #aa66ff;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #model-list {
        height: auto;
        max-height: 14;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #model-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #model-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_model: str):
        super().__init__()
        self._current = current_model

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog") as dialog:
            dialog.border_title = "SELECT MODEL"
            options = []
            for name, repo in AVAILABLE_MODELS.items():
                label = f"  {'▸ ' if name == self._current else '  '}{name:12s}  {repo}"
                options.append(Option(label, id=name))
            yield OptionList(*options, id="model-list")
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="model-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#model-list", OptionList)
        for idx, name in enumerate(AVAILABLE_MODELS):
            if name == self._current:
                option_list.highlighted = idx
                break

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self):
        self.dismiss(None)


class LanguageSelectScreen(ModalScreen):
    """Modal for selecting transcription language."""

    DEFAULT_CSS = """
    LanguageSelectScreen {
        align: center middle;
    }
    #lang-dialog {
        width: 50;
        height: auto;
        max-height: 22;
        border: heavy #cc6644;
        border-title-color: #ffaa66;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #lang-list {
        height: auto;
        max-height: 16;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #lang-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #lang-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_lang: str | None):
        super().__init__()
        self._current = current_lang or "en"

    def compose(self) -> ComposeResult:
        with Vertical(id="lang-dialog") as dialog:
            dialog.border_title = "SELECT LANGUAGE"
            options = []
            for code, name in AVAILABLE_LANGUAGES.items():
                marker = "▸ " if code == self._current else "  "
                label = f"  {marker}{code:5s}  {name}"
                options.append(Option(label, id=code))
            yield OptionList(*options, id="lang-list")
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="lang-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#lang-list", OptionList)
        for idx, code in enumerate(AVAILABLE_LANGUAGES):
            if code == self._current:
                option_list.highlighted = idx
                break

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self):
        self.dismiss(None)


class HelpScreen(ModalScreen):
    """Modal showing all keyboard shortcuts."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-dialog {
        width: 48;
        height: auto;
        max-height: 20;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #help-content {
        height: auto;
        color: #c0c0c0;
    }
    #help-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("?", "dismiss", "Close", key_display="?"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog") as dialog:
            dialog.border_title = "KEYBOARD SHORTCUTS"
            yield Static(
                "[bold #00e5ff]R[/]       [#c0c0c0]Start / stop recording[/]\n"
                "[bold #00e5ff]T[/]       [#c0c0c0]Tag / name speakers[/]\n"
                "[bold #00e5ff]P[/]       [#c0c0c0]Speaker profiles[/]\n"
                "[bold #00e5ff]Ctrl+S[/]  [#c0c0c0]Save / copy transcript[/]\n"
                "[bold #00e5ff]S[/]       [#c0c0c0]Save / copy transcript[/]\n"
                "[bold #00e5ff]M[/]       [#c0c0c0]Switch transcription model[/]\n"
                "[bold #00e5ff]L[/]       [#c0c0c0]Switch language[/]\n"
                "[bold #00e5ff]C[/]       [#c0c0c0]Clear transcript[/]\n"
                "[bold #00e5ff]D[/]       [#c0c0c0]Toggle debug mode[/]\n"
                "[bold #00e5ff]Q[/]       [#c0c0c0]Quit[/]",
                id="help-content",
                markup=True,
            )
            yield Static(
                " [#607080]ESC[/] or [#607080]?[/] to close",
                id="help-hint",
                markup=True,
            )


class ExportScreen(ModalScreen):
    """Modal for exporting transcript to a destination."""

    DEFAULT_CSS = """
    ExportScreen {
        align: center middle;
    }
    #export-dialog {
        width: 50;
        height: auto;
        max-height: 12;
        border: heavy #44cc66;
        border-title-color: #66ff88;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #export-list {
        height: auto;
        max-height: 6;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #export-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #export-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="export-dialog") as dialog:
            dialog.border_title = "EXPORT TRANSCRIPT"
            yield OptionList(
                Option("  Save to file", id="file"),
                Option("  Copy to clipboard", id="clipboard"),
                Option("  Discard transcript", id="discard"),
                id="export-list",
            )
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="export-hint",
                markup=True,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self):
        self.dismiss(None)


class VoxTerm(App):
    """Cyberpunk voice transcription TUI."""

    CSS_PATH = "cyberpunk.tcss"
    TITLE = "VOXTERM"

    BINDINGS = [
        Binding("r", "toggle_recording", "Record/Pause"),
        Binding("t", "tag_speakers", "Tag"),
        Binding("p", "manage_profiles", "Profiles"),
        Binding("m", "switch_model", "Model"),
        Binding("l", "switch_language", "Language"),
        Binding("ctrl+s", "export_transcript", "Save"),
        Binding("s", "export_transcript", "Export"),
        Binding("d", "toggle_debug", "Debug"),
        Binding("c", "clear_transcript", "Clear"),
        Binding("?", "show_help", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, transcriber=None, model_name="qwen3-0.6b", language="en"):
        super().__init__()
        self.audio_capture = AudioCapture()
        self.system_capture = SystemCapture()
        self.audio_buffer = AudioBuffer()
        self.vad = SileroVAD()
        self.transcriber = transcriber or Qwen3Transcriber()
        self.diarizer = DiarizationProxy()
        self.speaker_store = SpeakerStore()
        self._model_name = model_name
        self._language = language
        self._is_qwen3 = model_name in QWEN3_MODELS
        self._recording = False
        self._had_speech = False
        self._silence_chunks = 0
        self._transcribing = threading.Event()  # set = busy, clear = idle
        self._transcribe_started: float = 0.0
        self._debug = False
        self._last_dbg: float = 0.0
        self._transcribe_count = 0
        self._model_loaded = transcriber is not None and transcriber.is_loaded
        self._diarizer_loaded = False
        self._system_audio_notified = False
        self._last_saved_at: float | None = None
        self._session_start = datetime.now()
        self._live_file: Path | None = None
        self._live_header_written = False
        # Maps session speaker_id → persistent profile_id (set on tagging)
        self._speaker_profile_map: dict[int, str] = {}
        # Active learning fatigue prevention
        self._prompt_times: list[float] = []        # timestamps of recent prompts
        self._prompt_confirmations: dict[int, int] = {}  # speaker_id → confirm count
        self._last_prompt_time: float = 0.0
        self._onboarding_shown = False

    def compose(self) -> ComposeResult:
        yield CyberHeader()
        with Vertical(id="main-container"):
            yield WaveformWidget()
            yield TranscriptPanel()
            yield Static(
                "  [bold #607080]● IDLE[/]    [#00ffcc]loading...[/]",
                id="telemetry",
                markup=True,
            )
        yield Static(
            " [bold #00e5ff]\\[R][/][#607080] Record  [/]"
            "[bold #00e5ff]\\[T][/][#607080] Tag  [/]"
            "[bold #00e5ff]\\[P][/][#607080] Profiles  [/]"
            "[bold #00e5ff]\\[?][/][#607080] Help[/]",
            id="footer-bar",
            markup=True,
        )

    def on_mount(self) -> None:
        # Open speaker profile store (fast — just SQLite + cache load)
        try:
            self.speaker_store.open()
            self.speaker_store.backup()
        except Exception:
            pass  # graceful degradation — ephemeral mode if DB fails

        if self._model_loaded:
            transcript = self.query_one(TranscriptPanel)
            transcript.system_message("VOXTERM engine online")
            transcript.system_message(f"model loaded: {self._model_name}")
            self._update_telemetry()
            self._start_audio_timer()
            self._load_diarizer()
        else:
            self.query_one(TranscriptPanel).system_message("initializing VOXTERM engine...")
            self._start_audio_timer()
            self._load_model()

    @property
    def _chunk_duration(self) -> float:
        return CHUNK_SIZE / SAMPLE_RATE

    def _update_telemetry(self):
        # Status dot
        if self._recording:
            status = "[bold #00ff88]● REC[/]"
        elif self._model_loaded:
            status = "[bold #607080]● IDLE[/]"
        else:
            status = "[bold #607080]● LOADING[/]"

        model_text = self._model_name if self._model_loaded else "loading..."
        lang_text = AVAILABLE_LANGUAGES.get(self._language, self._language) if self._language else "auto"

        spk_count = self.diarizer.num_speakers if self._diarizer_loaded else 0
        if spk_count > 0:
            tagged_count = len(self.diarizer.get_speaker_names())
            if tagged_count > 0:
                spk_text = f"    [#aa88ff]{tagged_count}/{spk_count} tagged[/]"
            else:
                spk_text = f"    [#aa88ff]{spk_count} speakers[/]"
        else:
            spk_text = ""

        # Auto-save indicator
        if self._last_saved_at is not None:
            ago = int(time.time() - self._last_saved_at)
            if ago < 60:
                saved_text = f"    [#00ff88]saved {ago}s ago[/]"
            elif ago < 3600:
                saved_text = f"    [#ffaa00]saved {ago // 60}m ago[/]"
            else:
                saved_text = f"    [#ff6600]saved {ago // 3600}h ago[/]"
        else:
            saved_text = ""

        self.query_one("#telemetry", Static).update(
            f"  {status}"
            f"    [#00ffcc]{model_text}[/]"
            f"    [#ffaa66]{lang_text}[/]"
            f"{spk_text}"
            f"{saved_text}"
        )

    def _start_audio_timer(self):
        self.set_interval(1.0 / WAVEFORM_FPS, self._process_audio, name="audio_timer")
        self.set_interval(1.0, self._refresh_telemetry, name="telemetry_timer")
        self.set_interval(60.0, self._periodic_gc, name="gc_timer")

    def _refresh_telemetry(self):
        """Periodic refresh for saved counter and recording timer."""
        if self._recording:
            self.query_one(CyberHeader).refresh()
        if self._last_saved_at is not None:
            self._update_telemetry()

    def _periodic_gc(self):
        """Prevent memory fragmentation during long sessions + memory watchdog."""
        gc.collect()

        # Memory watchdog: warn at 4GB, crash-dump at 6GB
        import resource as _resource
        rss_bytes = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_bytes / (1024 * 1024)
        if rss_mb > 6000:
            self._write_crash_dump(f"memory_watchdog: {rss_mb:.0f}MB")
            self.query_one(TranscriptPanel).system_message(
                f"high memory usage ({rss_mb:.0f}MB) — consider saving and restarting"
            )
        elif rss_mb > 4000 and self._debug:
            self.query_one(TranscriptPanel).system_message(
                f"[dbg] memory: {rss_mb:.0f}MB"
            )

    def _write_crash_dump(self, context: str, exc: BaseException | None = None):
        """Write a diagnostic dump to disk. Always runs, not gated by debug."""
        try:
            cache = _make_style.cache_info()
            try:
                entry_count = len(self.query_one(TranscriptPanel).get_entries())
            except Exception:
                entry_count = -1

            state = {
                "uptime_sec": (datetime.now() - self._session_start).total_seconds(),
                "recording": self._recording,
                "is_transcribing": self._transcribing.is_set(),
                "transcribe_count": self._transcribe_count,
                "model": self._model_name,
                "model_loaded": self._model_loaded,
                "diarizer_loaded": self._diarizer_loaded,
                "language": self._language,
                "had_speech": self._had_speech,
                "silence_chunks": self._silence_chunks,
                "sys_capture": f"active={self.system_capture.is_active} msg={self.system_capture.status_message}",
                "audio_buf_dur": f"{self.audio_buffer.duration:.2f}s",
                "style_cache": f"hits={cache.hits} misses={cache.misses} size={cache.currsize}/{cache.maxsize}",
                "transcript_entries": entry_count,
                "speakers": self.diarizer.num_speakers if self._diarizer_loaded else 0,
                "gc_counts": str(gc.get_count()),
            }
            diagnostics.write_crash_dump(context, exc, state)
        except Exception:
            pass  # crash dump must never itself crash the app

    def _process_audio(self):
        """Read audio chunks, update waveform, check transcription trigger."""
        try:
            self._process_audio_inner()
        except Exception as e:
            self._write_crash_dump("_process_audio", e)
            raise

    @staticmethod
    def _mix_chunks(mic: list[np.ndarray], sys: list[np.ndarray]) -> list[np.ndarray]:
        """Time-aligned addition of mic and system audio chunks."""
        mixed = []
        n = min(len(mic), len(sys))
        for i in range(n):
            mixed.append(np.clip(mic[i] + sys[i], -1.0, 1.0))
        mixed.extend(mic[n:])
        mixed.extend(sys[n:])
        return mixed

    def _process_audio_inner(self):
        waveform = self.query_one(WaveformWidget)

        if not self._recording:
            waveform.tick()
            return

        mic_chunks = self.audio_capture.drain()
        sys_chunks = self.system_capture.drain()
        chunks = self._mix_chunks(mic_chunks, sys_chunks) if sys_chunks else mic_chunks
        if not chunks:
            waveform.tick()
            return

        for chunk in chunks:
            waveform.push_samples(chunk)

            # Speech detection: Silero VAD (neural) with RMS fallback
            if self.vad.is_loaded:
                is_speech = self.vad.is_speech(chunk)
            else:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                is_speech = rms >= SILENCE_THRESHOLD

            if is_speech:
                self._silence_chunks = 0
                self._had_speech = True
                self.audio_buffer.append(chunk)
            else:
                self._silence_chunks += 1
                # Only buffer silence if we're in an active speech segment
                if self._had_speech:
                    self.audio_buffer.append(chunk)

        waveform.tick()

        # Check transcription trigger
        silence_duration = self._silence_chunks * self._chunk_duration
        buffer_duration = self.audio_buffer.duration

        if self._transcribing.is_set():
            # Graduated watchdog: warn → force-reset → disable
            elapsed = time.time() - self._transcribe_started if self._transcribe_started else 0
            if elapsed > 30:
                # Critical: transcriber appears hung — force-reset and warn
                self._transcribing.clear()
                self._write_crash_dump(f"transcription_hung_{elapsed:.0f}s")
                self.query_one(TranscriptPanel).system_message(
                    f"transcription timed out ({elapsed:.0f}s) — try a smaller model [M]"
                )
            elif elapsed > 15:
                # Force-reset and log
                self._transcribing.clear()
                self.query_one(TranscriptPanel).system_message(
                    f"[watchdog] reset after {elapsed:.0f}s"
                )
            elif elapsed > 8 and self._debug:
                self.query_one(TranscriptPanel).system_message(
                    f"[dbg] transcription slow: {elapsed:.0f}s"
                )
            return

        if self._debug:
            now = time.time()
            if buffer_duration > 0.5 and now - self._last_dbg > 3:
                self._last_dbg = now
                self.query_one(TranscriptPanel).system_message(
                    f"[dbg] buf={buffer_duration:.1f}s sil={silence_duration:.1f}s "
                    f"speech={self._had_speech}"
                )

        if self._had_speech and silence_duration > SILENCE_TRIGGER_SECONDS and buffer_duration > MIN_BUFFER_SECONDS:
            self._trigger_transcription()
        elif buffer_duration >= MAX_BUFFER_SECONDS:
            self._trigger_transcription()

    def _trigger_transcription(self):
        """Send accumulated audio to transcription worker."""
        self._transcribing.set()
        self._transcribe_started = time.time()
        self._silence_chunks = 0
        audio = self.audio_buffer.get_and_clear()
        if len(audio) < int(SAMPLE_RATE * MIN_BUFFER_SECONDS):
            self._transcribing.clear()
            return

        self._had_speech = False
        self._transcribe_audio(audio)

    @work(thread=True, group="transcription")
    def _transcribe_audio(self, audio: np.ndarray):
        try:
            if self._debug:
                duration = len(audio) / SAMPLE_RATE
                self.call_from_thread(
                    self.query_one(TranscriptPanel).system_message,
                    f"[dbg] transcribing {duration:.1f}s audio..."
                )
            # 1. Transcribe (Qwen3: ~100ms, Whisper: ~2-4s)
            # MLX runs in main process; PyTorch diarizer runs in subprocess.
            # No lock needed — they're in separate processes.
            result = self.transcriber.transcribe(audio)

            # Periodic GC to prevent MLX memory buildup
            self._transcribe_count += 1
            if self._transcribe_count % 20 == 0:
                gc.collect()

            text = result.get("text", "")

            # 2. Speaker ID via subprocess (non-blocking to main thread)
            #    Use SCD to split audio at speaker changes, diarize each segment
            speaker_label, speaker_id = "", 0
            confidence = ""
            if text and self._diarizer_loaded:
                try:
                    speaker_label, speaker_id = self.diarizer.identify(
                        audio.copy()
                    )

                    # Debug: show speaker assignment info
                    if self._debug:
                        dbg = getattr(self.diarizer, '_last_debug', {})
                        n_spk = dbg.get('debug_speakers', '?')
                        rms = dbg.get('debug_rms', '?')
                        self.call_from_thread(
                            self.query_one(TranscriptPanel).system_message,
                            f"[dbg] → {speaker_label} (id={speaker_id}) "
                            f"speakers={n_spk} rms={rms}",
                        )

                    # 3. Cross-session matching — if speaker stable and not yet matched
                    if (
                        speaker_id > 0
                        and self.speaker_store.is_open
                        and not self.diarizer.is_matched(speaker_id)
                        and speaker_id not in self._speaker_profile_map
                        and self.diarizer.is_speaker_stable(speaker_id)
                    ):
                        centroid = self.diarizer.get_session_centroid(speaker_id)
                        if centroid is not None:
                            match = self.speaker_store.classify_match(centroid)

                            if match.tier == "high":
                                # Auto-assign: name this speaker
                                speaker_label = match.name
                                confidence = "high"
                                self.diarizer.set_speaker_name(speaker_id, match.name)
                                self.diarizer.mark_matched(speaker_id)
                                self._speaker_profile_map[speaker_id] = match.profile_id

                                # Update profile if mature enough
                                if self.speaker_store.is_profile_mature(match.profile_id):
                                    seg_data = self.diarizer.get_segment_embeddings(speaker_id)
                                    for emb, dur in seg_data:
                                        if dur >= 3.0:
                                            self.speaker_store.update_profile_embedding(
                                                match.profile_id, emb,
                                                duration=dur, confirmed=False,
                                            )

                                # Apply color + confidence on main thread
                                self.call_from_thread(
                                    self._on_auto_recognition,
                                    speaker_id, match.name, match.color,
                                    "high", match.score,
                                )

                            elif match.tier == "medium":
                                # Suggest but don't auto-assign
                                speaker_label = f"{match.name}?"
                                confidence = "medium"
                                self.diarizer.mark_matched(speaker_id)

                                self.call_from_thread(
                                    self._on_auto_recognition,
                                    speaker_id, f"{match.name}?", match.color,
                                    "medium", match.score,
                                )

                except Exception:
                    pass

            if text:
                self.call_from_thread(
                    self._on_transcription, text, speaker_label, speaker_id,
                    confidence,
                )
        except Exception as e:
            self._write_crash_dump("_transcribe_audio", e)
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"transcription error: {e}"
            )
        finally:
            # ALWAYS unblock — even if worker is cancelled or crashes
            self._transcribing.clear()

    def _on_transcription(
        self, text: str, speaker: str = "", speaker_id: int = 0,
        confidence: str = "",
    ):
        self.query_one(TranscriptPanel).add_transcript(
            text, speaker, speaker_id, confidence=confidence,
        )
        self._append_live_transcript(text, speaker, speaker_id)
        self._update_telemetry()

        # First-use onboarding tip
        if (
            not self._onboarding_shown
            and speaker_id > 0
            and not self.speaker_store.get_all_profiles()
            and not self.diarizer.get_speaker_names()
        ):
            self._onboarding_shown = True
            self.query_one(TranscriptPanel).system_message(
                "tip: press [T] to name speakers — VoxTerm will remember them"
            )

    def _on_auto_recognition(
        self, speaker_id: int, name: str, color: str,
        tier: str, score: float,
    ):
        """Called on main thread when cross-session matching identifies a speaker."""
        transcript = self.query_one(TranscriptPanel)
        transcript.set_speaker_confidence(speaker_id, tier, score)
        # Rename all prior entries for this speaker in the transcript
        transcript.rename_speaker(speaker_id, name, color)

        # For MEDIUM matches: show confirmation prompt (if within budget)
        if tier == "medium" and self._can_prompt():
            self._show_confirm_prompt(speaker_id, name.rstrip("?"), score)

        self._update_telemetry()

    # ── active learning / fatigue ──────────────────────────────

    def _can_prompt(self) -> bool:
        """Check if we're within the active learning prompt budget."""
        now = time.time()
        session_elapsed = now - self._session_start.timestamp()

        # Max 5 prompts per 10-minute window
        cutoff = now - 600
        recent = [t for t in self._prompt_times if t > cutoff]
        if len(recent) >= 5:
            return False

        # After first 5 min: max 1 prompt per 2 minutes
        if session_elapsed > 300 and (now - self._last_prompt_time) < 120:
            return False

        return True

    def _show_confirm_prompt(self, speaker_id: int, name: str, score: float):
        """Show a non-blocking confirmation for a MEDIUM-confidence match."""
        pct = int(score * 100)
        transcript = self.query_one(TranscriptPanel)
        transcript.system_message(
            f"is this {name}? (~{pct}%) "
            f"press [T] to confirm or rename"
        )
        self._prompt_times.append(time.time())
        self._last_prompt_time = time.time()

    # ── background auto-save ───────────────────────────────────

    def _append_live_transcript(self, text: str, speaker: str, speaker_id: int):
        """Append a single transcript line to the live file on disk."""
        try:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            if self._live_file is None:
                fname = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
                self._live_file = LIVE_DIR / fname
                self._live_header_written = False

            with open(self._live_file, "a", encoding="utf-8") as f:
                if not self._live_header_written:
                    f.write(f"# VOXTERM Transcript\n\n")
                    f.write(f"- **Date:** {self._session_start.strftime('%Y-%m-%d')}\n")
                    f.write(f"- **Time:** {self._session_start.strftime('%H:%M:%S')}\n")
                    f.write(f"- **Model:** {self._model_name}\n\n---\n\n")
                    self._live_header_written = True

                ts = datetime.now().strftime("%H:%M:%S")
                if speaker:
                    f.write(f"**[{ts}]** **{speaker}:** {text}\n\n")
                else:
                    f.write(f"**[{ts}]** {text}\n\n")
            self._last_saved_at = time.time()
        except Exception:
            pass  # never block transcription on I/O failure

    @work(thread=True, group="model_loading")
    def _load_model(self):
        self.call_from_thread(
            self.query_one(TranscriptPanel).system_message,
            "loading whisper model (first run downloads ~461MB)..."
        )
        try:
            self.transcriber.load()
            self.call_from_thread(self._on_model_loaded)
        except Exception as e:
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"model load failed: {e}"
            )

    def _on_model_loaded(self):
        self._model_loaded = True
        transcript = self.query_one(TranscriptPanel)
        transcript.system_message(f"model loaded: {self._model_name}")
        if not self._diarizer_loaded:
            self._load_diarizer()
        else:
            transcript.system_message("press [R] to start recording")
        self._update_telemetry()

    @work(thread=True, group="diarizer_loading")
    def _load_diarizer(self):
        self.call_from_thread(
            self.query_one(TranscriptPanel).system_message,
            "loading speaker identification model..."
        )
        try:
            # Set up crash/restart callbacks for subprocess mode
            self.diarizer.on_subprocess_crash = self._on_diarizer_crash
            self.diarizer.on_subprocess_ready = lambda: self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                "speaker identification restarted"
            )
            self.diarizer.load()
            self.call_from_thread(self._on_diarizer_loaded)
        except Exception as e:
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"speaker ID unavailable: {e}"
            )
            self.call_from_thread(self._on_diarizer_fallback)

    def _on_diarizer_loaded(self):
        self._diarizer_loaded = True
        mode = "subprocess" if self.diarizer._mode == "subprocess" else "in-process"
        self.query_one(TranscriptPanel).system_message(
            f"speaker identification online ({mode})"
        )
        self.query_one(TranscriptPanel).system_message("press [R] to start recording")
        self._update_telemetry()

    def _on_diarizer_crash(self, crash_count: int):
        """Called from worker thread when diarizer subprocess crashes."""
        self._write_crash_dump(f"diarizer_subprocess_crash #{crash_count}")
        if self.diarizer._mode == "inprocess":
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                "speaker ID subprocess failed — using in-process fallback"
            )
        else:
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"speaker ID subprocess crashed — restarting ({crash_count}/{3})"
            )

    def _on_diarizer_fallback(self):
        self.query_one(TranscriptPanel).system_message("press [R] to start recording")

    # ── actions ─────────────────────────────────────────────────

    def action_toggle_recording(self):
        if not self._model_loaded:
            self.query_one(TranscriptPanel).system_message("model still loading, please wait...")
            return

        waveform = self.query_one(WaveformWidget)
        header = self.query_one(CyberHeader)
        transcript = self.query_one(TranscriptPanel)
        if self._recording:
            self._recording = False
            self.audio_capture.stop()
            self.system_capture.stop()
            waveform.set_recording(False)
            header.set_recording(False)
            transcript.system_message("recording paused")
        else:
            self._recording = True
            self.vad.reset()
            try:
                self.audio_capture.start()
                waveform.set_recording(True)
                header.set_recording(True)
                transcript.system_message("recording started")
            except Exception as e:
                self._recording = False
                waveform.set_recording(False)
                header.set_recording(False)
                transcript.system_message(
                    f"microphone error: {e} — grant Terminal mic access in System Settings > Privacy"
                )
                self._update_telemetry()
                return

            # Start system audio capture (non-fatal if unavailable)
            try:
                self.system_capture.start()
            except Exception:
                pass

            # Show system audio status once per session
            if (
                not self._system_audio_notified
                and self.system_capture.status_message
            ):
                transcript.system_message(self.system_capture.status_message)
                self._system_audio_notified = True
        self._update_telemetry()

    def action_switch_model(self):
        if self._transcribing.is_set():
            self.query_one(TranscriptPanel).system_message("wait for transcription to finish...")
            return
        was_recording = self._recording
        if was_recording:
            self._recording = False
            self.audio_capture.stop()

        def on_model_selected(model_key):
            if model_key is None or model_key == self._model_name:
                if was_recording:
                    self.action_toggle_recording()
                return
            self._swap_model(model_key)

        self.push_screen(ModelSelectScreen(self._model_name), on_model_selected)

    def action_switch_language(self):
        def on_lang_selected(lang_code):
            if lang_code is None or lang_code == self._language:
                return
            self._language = lang_code
            lang_name = AVAILABLE_LANGUAGES.get(lang_code, lang_code)
            # Update transcriber language if it's Qwen3
            if self._is_qwen3 and hasattr(self.transcriber, '_language'):
                self.transcriber._language = lang_code
            _save_state({"last_model": self._model_name, "last_language": lang_code})
            self.query_one(TranscriptPanel).system_message(f"language set to {lang_name}")
            self._update_telemetry()

        self.push_screen(LanguageSelectScreen(self._language), on_lang_selected)

    def action_tag_speakers(self):
        """Open speaker tagging modal."""
        if not self._diarizer_loaded:
            self.query_one(TranscriptPanel).system_message(
                "speaker identification not loaded yet"
            )
            return

        session_speakers = self.diarizer.get_all_session_speakers()
        if not session_speakers:
            self.query_one(TranscriptPanel).system_message("no speakers detected yet")
            return

        # Build speaker list for the modal
        speaker_names = self.diarizer.get_speaker_names()
        speakers = []
        for sid, seg_count in sorted(session_speakers.items()):
            name = speaker_names.get(sid, f"Speaker {sid}")
            color = self.diarizer.get_speaker_color(sid)
            tagged = sid in speaker_names
            speakers.append({
                "id": sid,
                "name": name,
                "color": color,
                "segments": seg_count,
                "tagged": tagged,
            })

        # Collect known names from persistent profiles for autocomplete
        known_names = list(self.speaker_store.get_profile_names().values())

        def on_tag_result(result):
            if result is None:
                return
            if "merge_source" in result:
                self._apply_speaker_merge(
                    result["merge_source"], result["merge_target"]
                )
            else:
                self._apply_speaker_tag(result["speaker_id"], result["name"])

        self.push_screen(
            SpeakerTagScreen(speakers, known_names=known_names),
            on_tag_result,
        )

    def _apply_speaker_tag(self, speaker_id: int, name: str):
        """Apply a speaker tag: update diarizer, transcript, and persistent store."""
        # 1. Update the in-session diarizer
        self.diarizer.set_speaker_name(speaker_id, name)

        # 2. Get the session embeddings for enrollment
        segment_data = self.diarizer.get_segment_embeddings(speaker_id)
        embeddings = [emb for emb, dur in segment_data if dur >= 3.0]
        durations = [dur for emb, dur in segment_data if dur >= 3.0]

        # Fall back to all embeddings if none pass the 3s filter
        if not embeddings and segment_data:
            embeddings = [emb for emb, dur in segment_data]
            durations = [dur for emb, dur in segment_data]

        # 3. Check if this name matches an existing profile
        existing_profiles = self.speaker_store.get_all_profiles()
        matched_profile = None
        for p in existing_profiles:
            if p.name.lower() == name.lower():
                matched_profile = p
                break

        if matched_profile:
            # Update existing profile with new embeddings
            profile_id = matched_profile.id
            color = matched_profile.color
            for emb, dur in zip(embeddings, durations):
                self.speaker_store.update_profile_embedding(
                    profile_id, emb, duration=dur, confirmed=True
                )
        elif embeddings:
            # Create a new profile
            color = self.diarizer.get_speaker_color(speaker_id)
            profile_id = self.speaker_store.create_profile(
                name=name,
                color=color,
                embeddings=embeddings,
                durations=durations,
            )
        else:
            # No embeddings available — just rename in-session
            color = None
            profile_id = None

        if profile_id:
            self._speaker_profile_map[speaker_id] = profile_id

        # 4. Update transcript display
        self.query_one(TranscriptPanel).rename_speaker(speaker_id, name, color)

        # 5. Feedback
        self.query_one(TranscriptPanel).system_message(f"tagged as {name}")
        self._update_telemetry()

    def _apply_speaker_merge(self, source_id: int, target_id: int):
        """Merge source session speaker into target."""
        transcript = self.query_one(TranscriptPanel)
        target_name = self.diarizer.get_speaker_name(target_id)
        target_color = self.diarizer.get_speaker_color(target_id)

        # Merge in-session: rename source entries to target
        transcript.rename_speaker(source_id, target_name, target_color)

        # Merge in diarizer (centroids, embeddings, cleanup)
        self.diarizer.merge_speakers(source_id, target_id)

        # If both have persistent profiles, merge those too
        source_pid = self._speaker_profile_map.get(source_id)
        target_pid = self._speaker_profile_map.get(target_id)
        if source_pid and target_pid and source_pid != target_pid:
            self.speaker_store.merge_profiles(source_pid, target_pid)
            self._speaker_profile_map.pop(source_id, None)

        transcript.system_message(
            f"merged Speaker {source_id} into {target_name}"
        )
        self._update_telemetry()

    def action_manage_profiles(self):
        """Open speaker profiles management screen."""
        profiles = self.speaker_store.get_all_profiles()

        def on_profile_result(result):
            if result is None:
                return
            action = result.get("action")
            if action == "rename":
                pid = result["profile_id"]
                new_name = result["name"]
                self.speaker_store.rename_profile(pid, new_name)
                self.query_one(TranscriptPanel).system_message(
                    f"profile renamed to {new_name}"
                )
                # Update in-session labels if this profile is active
                for sid, mapped_pid in self._speaker_profile_map.items():
                    if mapped_pid == pid:
                        self.diarizer.set_speaker_name(sid, new_name)
                        self.query_one(TranscriptPanel).rename_speaker(
                            sid, new_name
                        )
            elif action == "delete":
                pid = result["profile_id"]
                meta = next(
                    (p for p in profiles if p.id == pid), None
                )
                name = meta.name if meta else "unknown"
                self.speaker_store.delete_profile(pid)
                # Remove from session mapping
                self._speaker_profile_map = {
                    sid: p for sid, p in self._speaker_profile_map.items()
                    if p != pid
                }
                self.query_one(TranscriptPanel).system_message(
                    f"deleted profile: {name}"
                )
            elif action == "delete_all":
                self.speaker_store.delete_all_data()
                self._speaker_profile_map.clear()
                self.query_one(TranscriptPanel).system_message(
                    "all voice data deleted"
                )

        self.push_screen(SpeakerProfileScreen(profiles), on_profile_result)

    def _swap_model(self, model_key: str):
        self._model_loaded = False
        self._model_name = model_key
        self._update_telemetry()
        # Free old model memory before loading the new one
        self.transcriber._model = None
        self.query_one(TranscriptPanel).system_message(
            f"switching to {model_key} (may take a minute)..."
        )
        self._do_swap(model_key)

    @work(thread=True, exclusive=True, group="model_loading")
    def _do_swap(self, model_key: str):
        repo = AVAILABLE_MODELS[model_key]
        try:
            if model_key in QWEN3_MODELS:
                new_transcriber = Qwen3Transcriber(model=repo, language=self._language)
            elif model_key in FASTER_WHISPER_MODELS:
                new_transcriber = FasterWhisperTranscriber(model=repo, language=self._language)
            else:
                new_transcriber = WhisperTranscriber(model=repo)
            new_transcriber.load()
            self.call_from_thread(self._on_swap_done, new_transcriber, model_key)
        except Exception as e:
            self.call_from_thread(
                self._on_swap_error, f"model switch failed: {e}"
            )

    def _on_swap_done(self, transcriber, model_key):
        self.transcriber = transcriber
        self._model_name = model_key
        self._is_qwen3 = model_key in QWEN3_MODELS
        self._model_loaded = True
        _save_state({"last_model": model_key, "last_language": self._language})
        transcript = self.query_one(TranscriptPanel)
        transcript.system_message(f"model loaded: {model_key}")
        transcript.system_message("press [R] to start recording")
        self._update_telemetry()

    def _on_swap_error(self, msg: str):
        self.query_one(TranscriptPanel).system_message(msg)
        self._model_loaded = True
        self._update_telemetry()

    def action_export_transcript(self):
        """Open export modal to choose destination."""
        transcript = self.query_one(TranscriptPanel)
        if not transcript.get_entries():
            transcript.system_message("nothing to export")
            return

        def on_export_selected(destination):
            if destination is None:
                return
            if destination == "file":
                self._export_to_file()
            elif destination == "clipboard":
                self._export_to_clipboard()
            elif destination == "discard":
                self._discard_transcript()

        self.push_screen(ExportScreen(), on_export_selected)

    def _export_to_file(self):
        """Promote live file to final transcript."""
        transcript = self.query_one(TranscriptPanel)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        filename = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
        filepath = SESSIONS_DIR / filename

        # Write the full markdown (cleaner than the append-mode live file)
        md = transcript.get_markdown(self._model_name)
        filepath.write_text(md, encoding="utf-8")

        # Remove the live file since we promoted it
        if self._live_file and self._live_file.exists():
            self._live_file.unlink()

        entry_count = len(transcript.get_entries())
        self._start_new_session()
        transcript.system_message(f"exported {entry_count} entries → {filepath}")

    def _export_to_clipboard(self):
        """Copy transcript to clipboard."""
        transcript = self.query_one(TranscriptPanel)
        cmd = _clipboard_cmd()
        if cmd is None:
            transcript.system_message("no clipboard tool found (install xclip, xsel, or wl-copy)")
            return
        text = transcript.get_plain_text()
        entry_count = len(transcript.get_entries())
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            self._start_new_session()
            transcript.system_message(f"copied {entry_count} entries to clipboard")
        except Exception:
            transcript.system_message("clipboard copy failed")

    def _discard_transcript(self):
        """Discard transcript and delete the live file."""
        entry_count = len(self.query_one(TranscriptPanel).get_entries())
        # Delete the live auto-save file
        if self._live_file and self._live_file.exists():
            self._live_file.unlink()
        self._start_new_session()
        self.query_one(TranscriptPanel).system_message(
            f"discarded {entry_count} entries"
        )

    def _start_new_session(self):
        """Clear transcript and reset for a new session."""
        # Record session-speaker mappings before clearing
        self._record_session_stats()

        transcript = self.query_one(TranscriptPanel)
        transcript.clear()
        self.audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        self.vad.reset()
        if self._diarizer_loaded:
            self.diarizer.reset_session()
        self._speaker_profile_map.clear()
        self._prompt_times.clear()
        self._prompt_confirmations.clear()
        self._last_prompt_time = 0.0
        # Start fresh live file
        self._session_start = datetime.now()
        self._live_file = None
        self._live_header_written = False

    def _record_session_stats(self):
        """Record speaker-session mappings to persistent store."""
        if not self.speaker_store.is_open or not self._speaker_profile_map:
            return
        session_id = self._session_start.strftime("%Y-%m-%d_%H%M%S")
        session_speakers = self.diarizer.get_all_session_speakers()
        for sid, profile_id in self._speaker_profile_map.items():
            seg_count = session_speakers.get(sid, 0)
            try:
                self.speaker_store.record_session_speaker(
                    session_id, profile_id, sid, seg_count
                )
            except Exception:
                pass

    def action_show_help(self):
        self.push_screen(HelpScreen())

    def action_toggle_debug(self):
        self._debug = not self._debug
        state = "ON" if self._debug else "OFF"
        self.query_one(TranscriptPanel).system_message(f"debug mode {state}")

    def action_clear_transcript(self):
        """Clear display only — live file stays on disk as the record."""
        self.query_one(TranscriptPanel).clear()
        self.audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        self.vad.reset()
        if self._diarizer_loaded:
            self.diarizer.reset_session()
        self._speaker_profile_map.clear()

    def action_quit(self):
        # Live file already on disk — no extra save needed
        self._record_session_stats()
        self.audio_capture.stop()
        self.system_capture.stop()
        try:
            self.diarizer.shutdown()
        except Exception:
            pass
        try:
            self.speaker_store.close()
        except Exception:
            pass

        # Let Textual restore the terminal, then hard-exit before
        # Python's GC triggers C extension segfaults.
        # Silence stderr to suppress resource_tracker leaked semaphore warning.
        def _silent_exit():
            try:
                sys.stderr.close()
            except Exception:
                pass
            os._exit(0)
        threading.Timer(0.5, _silent_exit).start()
        self.exit()



if __name__ == "__main__":
    import argparse

    # Resolve defaults: saved preferences > config defaults
    _state = _load_state()
    _saved_model = _state.get("last_model")
    _saved_lang = _state.get("last_language")
    _default_model = _saved_model if _saved_model in AVAILABLE_MODELS else DEFAULT_MODEL
    _default_lang = _saved_lang if _saved_lang in AVAILABLE_LANGUAGES else DEFAULT_LANGUAGE

    parser = argparse.ArgumentParser(description="VOXTERM — Local Voice Transcription TUI")
    parser.add_argument(
        "-m", "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default=_default_model,
        help=f"Transcription model (default: {_default_model})",
    )
    parser.add_argument(
        "-l", "--language",
        choices=list(AVAILABLE_LANGUAGES.keys()),
        default=_default_lang,
        help=f"Transcription language (default: {_default_lang})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, repo in AVAILABLE_MODELS.items():
            tag = " (default)" if name == _default_model else ""
            if name in QWEN3_MODELS:
                backend = " [qwen3-asr]"
            elif name in FASTER_WHISPER_MODELS:
                backend = " [faster-whisper]"
            else:
                backend = " [whisper]"
            print(f"  {name:20s} → {repo}{backend}{tag}")
        sys.exit(0)

    model_repo = AVAILABLE_MODELS[args.model]
    model_name = args.model
    language = args.language

    # Pre-TUI setup: install BlackHole if Bluetooth output detected (macOS only)
    # Must happen before TUI launches — brew needs the live terminal for sudo
    if sys.platform == "darwin":
        try:
            from audio.platform import get_output_device_info
            from audio.blackhole import is_blackhole_installed
            dev_info = get_output_device_info()
            if dev_info.get("is_bluetooth") and not is_blackhole_installed():
                print(f"VOXTERM // Bluetooth output detected ({dev_info['name']})")
                print("Installing BlackHole for system audio capture...\n")
                result = subprocess.run(["brew", "install", "blackhole-2ch"])
                if result.returncode == 0:
                    print("\nBlackHole installed. Restarting audio service...")
                    subprocess.run(
                        ["sudo", "killall", "coreaudiod"],
                        timeout=15,
                    )
                    import time
                    time.sleep(2)  # Give CoreAudio time to restart and detect BlackHole
                    print("Audio service restarted.\n")
                else:
                    print("\nBlackHole install failed — system audio capture will be limited.\n")
        except Exception:
            pass

    print(f"VOXTERM // loading model ({model_name}) lang={language}...")
    print("(first run downloads the model, please wait)\n")
    if model_name in QWEN3_MODELS:
        transcriber = Qwen3Transcriber(model=model_repo, language=language)
    elif model_name in FASTER_WHISPER_MODELS:
        transcriber = FasterWhisperTranscriber(model=model_repo, language=language)
    else:
        transcriber = WhisperTranscriber(model=model_repo)
    transcriber.load()
    print("Model ready. Launching TUI...\n")

    # Prevent segfault: PortAudio/PyTorch/SpeechBrain C threads crash
    # during Python's shutdown when native objects are GC'd in random order.
    # atexit fires before finalizers; the finally block catches SystemExit.
    import atexit
    atexit.register(os._exit, 0)

    # Restore terminal on segfault so the shell doesn't get stuck in raw mode
    diagnostics.setup_signal_handlers()

    app = VoxTerm(transcriber=transcriber, model_name=model_name, language=language)

    # Global exception hooks — dump diagnostics on any uncaught crash
    diagnostics.setup_exception_hooks(app)

    try:
        app.run()
    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        os._exit(0)
