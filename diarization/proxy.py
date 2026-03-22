"""DiarizationProxy — subprocess-backed speaker diarization.

Drop-in replacement for DiarizationEngine. Same public API, but delegates
all PyTorch/SpeechBrain work to a child process so MLX and PyTorch never
share an address space (preventing C++ runtime segfaults).

Falls back to in-process DiarizationEngine if the subprocess fails
repeatedly (3 crashes within 60 seconds).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from config import DIARIZER_MAX_RESTARTS, DIARIZER_RESTART_WINDOW
from diarization.ipc import (
    MSG_ERROR, MSG_GET_CENTROID, MSG_GET_COLOR,
    MSG_GET_EMBEDDINGS, MSG_GET_NAME, MSG_GET_NAMES, MSG_GET_STATE,
    MSG_IDENTIFY, MSG_IDENTIFY_MULTI, MSG_IS_MATCHED, MSG_IS_STABLE,
    MSG_MARK_MATCHED, MSG_MERGE, MSG_NUM_SPEAKERS, MSG_READY,
    MSG_RESET, MSG_SET_NAME, MSG_SHUTDOWN,
    decode_array, encode_array, recv_msg, send_msg,
)

_WORKER_MODULE = "diarization.subprocess_worker"


class DiarizationProxy:
    """Subprocess-backed speaker diarization with same API as DiarizationEngine."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()  # serializes IPC calls
        self._loaded = False
        self._mode = "subprocess"  # "subprocess" or "inprocess"
        self._needs_respawn = False  # set on crash, handled outside lock

        # Crash tracking for fallback
        self._crash_times: list[float] = []

        # In-process fallback (lazy-loaded)
        self._engine = None
        self._engine_lock = threading.Lock()  # protects PyTorch calls in inprocess mode

        # Callback for crash notifications (set by app.py)
        self.on_subprocess_crash: callable | None = None
        self.on_subprocess_ready: callable | None = None
        self._last_debug: dict = {}  # debug info from last identify() call

        # T2/I2: cached values for non-blocking telemetry reads from main thread
        self._cached_num_speakers: int = 0
        self._cached_speaker_names: dict[int, str] = {}

    # ── lifecycle ─────────────────────────────────────────

    def load(self):
        """Spawn the diarizer subprocess and wait for it to be ready."""
        try:
            self._spawn()
            self._loaded = True
        except Exception:
            self._disable()

    def _spawn(self):
        """Start the subprocess worker."""
        project_root = str(Path(__file__).parent.parent)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # capture for diagnostics
            cwd=project_root,
        )

        # Wait for READY message (model loading can take 5-30s)
        resp = recv_msg(self._proc.stdout)
        if resp is None or resp.get("type") != MSG_READY:
            # Grab stderr for error context
            stderr_out = ""
            try:
                stderr_out = self._proc.stderr.read(2000).decode("utf-8", errors="replace")
            except Exception:
                pass
            self._kill()
            raise RuntimeError(
                f"Diarizer subprocess did not start: {stderr_out[:200]}"
            )

    def _kill(self):
        """Force-kill the subprocess."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def shutdown(self):
        """Cleanly stop the subprocess."""
        if self._mode in ("inprocess", "disabled"):
            return
        with self._lock:
            if self._proc is not None:
                try:
                    send_msg(self._proc.stdin, {"type": MSG_SHUTDOWN})
                    self._proc.wait(timeout=3)
                except Exception:
                    self._kill()
                self._proc = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── speaker identification (main API) ─────────────────

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, int]:
        if self._mode == "disabled":
            return "Speaker 1", 1
        if self._mode == "inprocess":
            with self._engine_lock:
                return self._engine.identify(audio, sample_rate)

        resp = self._call({
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": sample_rate,
        })
        if resp is None:
            return "Speaker 1", 1
        # Stash debug info from subprocess (if present)
        self._last_debug = {
            k: resp[k] for k in ("debug_rms", "debug_samples", "debug_speakers")
            if k in resp
        }
        # T2/I2: refresh cached telemetry so main-thread reads don't block
        self._cached_num_speakers = resp.get("debug_speakers", self._cached_num_speakers)
        return resp.get("label", "Speaker 1"), resp.get("speaker_id", 1)

    def identify_segments(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> list[tuple[str, int, int, int]]:
        """Identify speakers with speaker-change detection.

        Returns list of (label, speaker_id, start_sample, end_sample).
        """
        if self._mode == "disabled":
            return [("Speaker 1", 1, 0, len(audio))]
        if self._mode == "inprocess":
            with self._engine_lock:
                return self._engine.identify_segments(audio, sample_rate)

        resp = self._call({
            "type": MSG_IDENTIFY_MULTI,
            "audio": encode_array(audio),
            "sample_rate": sample_rate,
        })
        if resp is None:
            return [("Speaker 1", 1, 0, len(audio))]

        self._last_debug = {
            k: resp[k] for k in ("debug_speakers",)
            if k in resp
        }

        segments = resp.get("segments", [])
        if not segments:
            return [("Speaker 1", 1, 0, len(audio))]

        return [
            (
                seg.get("label", "Speaker 1"),
                seg.get("speaker_id", 1),
                seg.get("start_sample", 0),
                seg.get("end_sample", len(audio)),
            )
            for seg in segments
        ]

    # ── speaker queries ───────────────────────────────────

    def get_speaker_color(self, speaker_id: int) -> str:
        if self._mode == "disabled":
            return "#00ffcc"
        if self._mode == "inprocess":
            return self._engine.get_speaker_color(speaker_id)
        resp = self._call({"type": MSG_GET_COLOR, "speaker_id": speaker_id})
        if resp is None:
            return "#00ffcc"
        return resp.get("color", "#00ffcc")

    def get_speaker_name(self, speaker_id: int) -> str:
        if self._mode == "disabled":
            return f"Speaker {speaker_id}"
        if self._mode == "inprocess":
            return self._engine.get_speaker_name(speaker_id)
        resp = self._call({"type": MSG_GET_NAME, "speaker_id": speaker_id})
        if resp is None:
            return f"Speaker {speaker_id}"
        return resp.get("name", f"Speaker {speaker_id}")

    def get_speaker_names(self) -> dict[int, str]:
        if self._mode == "disabled":
            return {}
        if self._mode == "inprocess":
            return self._engine.get_speaker_names()
        resp = self._call({"type": MSG_GET_NAMES})
        if resp is None:
            return {}
        return {int(k): v for k, v in resp.get("names", {}).items()}

    @property
    def num_speakers(self) -> int:
        if self._mode == "disabled":
            return 0
        if self._mode == "inprocess":
            return self._engine.num_speakers
        resp = self._call({"type": MSG_NUM_SPEAKERS})
        if resp is None:
            return 0
        return resp.get("count", 0)

    # T2/I2: non-blocking cached accessors for main-thread telemetry
    @property
    def cached_num_speakers(self) -> int:
        """Return cached speaker count (non-blocking, no IPC)."""
        return self._cached_num_speakers

    @property
    def cached_speaker_names(self) -> dict[int, str]:
        """Return cached speaker names (non-blocking, no IPC)."""
        return self._cached_speaker_names.copy()

    def set_speaker_name(self, speaker_id: int, name: str) -> None:
        if self._mode == "disabled":
            return
        if self._mode == "inprocess":
            self._engine.set_speaker_name(speaker_id, name)
            return
        self._call({"type": MSG_SET_NAME, "speaker_id": speaker_id, "name": name})
        # Update cached names
        self._cached_speaker_names[speaker_id] = name

    # ── session state queries ─────────────────────────────

    def get_all_session_speakers(self) -> dict[int, int]:
        if self._mode == "disabled":
            return {}
        if self._mode == "inprocess":
            return self._engine.get_all_session_speakers()
        resp = self._call({"type": MSG_GET_STATE})
        if resp is None:
            return {}
        return {int(k): v for k, v in resp.get("session_speakers", {}).items()}

    def get_segment_embeddings(self, speaker_id: int) -> list[tuple[np.ndarray, float]]:
        if self._mode == "disabled":
            return []
        if self._mode == "inprocess":
            return self._engine.get_segment_embeddings(speaker_id)
        resp = self._call({"type": MSG_GET_EMBEDDINGS, "speaker_id": speaker_id})
        if resp is None:
            return []
        try:
            return [
                (decode_array(item["embedding"]), item["duration"])
                for item in resp.get("embeddings", [])
            ]
        except (ValueError, KeyError):
            return []

    def get_session_centroid(self, speaker_id: int) -> np.ndarray | None:
        if self._mode == "disabled":
            return None
        if self._mode == "inprocess":
            return self._engine.get_session_centroid(speaker_id)
        resp = self._call({"type": MSG_GET_CENTROID, "speaker_id": speaker_id})
        if resp is None:
            return None
        c = resp.get("centroid")
        if not c:
            return None
        try:
            return decode_array(c)
        except ValueError:
            return None

    def is_speaker_stable(self, speaker_id: int) -> bool:
        if self._mode == "disabled":
            return False
        if self._mode == "inprocess":
            return self._engine.is_speaker_stable(speaker_id)
        resp = self._call({"type": MSG_IS_STABLE, "speaker_id": speaker_id})
        if resp is None:
            return False
        return resp.get("stable", False)

    def mark_matched(self, speaker_id: int) -> None:
        if self._mode == "disabled":
            return
        if self._mode == "inprocess":
            self._engine.mark_matched(speaker_id)
            return
        self._call({"type": MSG_MARK_MATCHED, "speaker_id": speaker_id})

    def is_matched(self, speaker_id: int) -> bool:
        if self._mode == "disabled":
            return False
        if self._mode == "inprocess":
            return self._engine.is_matched(speaker_id)
        resp = self._call({"type": MSG_IS_MATCHED, "speaker_id": speaker_id})
        if resp is None:
            return False
        return resp.get("matched", False)

    def merge_speakers(self, source_id: int, target_id: int) -> None:
        if self._mode == "disabled":
            return
        if self._mode == "inprocess":
            self._engine.merge_speakers(source_id, target_id)
            return
        self._call({"type": MSG_MERGE, "source_id": source_id, "target_id": target_id})

    def reset_session(self):
        if self._mode == "disabled":
            return
        if self._mode == "inprocess":
            self._engine.reset_session()
            return
        self._call({"type": MSG_RESET})

    # ── IPC internals ─────────────────────────────────────

    def _call(self, msg: dict) -> dict | None:
        """Send a message and wait for the response. Thread-safe.

        Returns the response dict, or None on failure (subprocess crash).
        On crash, respawn happens OUTSIDE the lock to avoid UI freezes.
        """
        # Fast path: disabled or respawning — return None immediately
        if self._mode == "disabled" or self._needs_respawn:
            return None

        needs_crash_handling = False

        with self._lock:
            try:
                if self._proc is None or self._proc.stdin is None:
                    raise BrokenPipeError("no subprocess")
                send_msg(self._proc.stdin, msg)
                resp = recv_msg(self._proc.stdout)
                if resp is None:
                    raise BrokenPipeError("subprocess EOF")
                if resp.get("type") == MSG_ERROR:
                    return None
                return resp
            except (BrokenPipeError, OSError, ValueError):
                # Mark for crash handling but DON'T handle inside the lock —
                # _handle_crash sleeps and respawns, which would freeze the UI.
                self._kill()
                needs_crash_handling = True

        # Handle crash OUTSIDE the lock so other threads aren't blocked
        if needs_crash_handling:
            self._handle_crash()
        return None

    # ── crash recovery ────────────────────────────────────

    def _handle_crash(self):
        """Handle a subprocess crash. Respawn or disable.

        Called OUTSIDE _lock to avoid blocking the UI thread during sleep/respawn.
        """
        now = time.time()
        self._crash_times.append(now)

        # Prune old crash times outside the window
        cutoff = now - DIARIZER_RESTART_WINDOW
        self._crash_times = [t for t in self._crash_times if t > cutoff]

        if self.on_subprocess_crash:
            try:
                self.on_subprocess_crash(len(self._crash_times))
            except Exception:
                pass

        if len(self._crash_times) >= DIARIZER_MAX_RESTARTS:
            self._disable()
            return

        # Non-blocking respawn on a daemon thread so the transcription worker
        # isn't stuck for 2-30s waiting for model reload.
        self._needs_respawn = True
        threading.Thread(target=self._background_respawn, daemon=True).start()

    def _background_respawn(self):
        """Respawn subprocess on a background thread (non-blocking)."""
        time.sleep(1.0)
        with self._lock:
            try:
                self._spawn()
                self._needs_respawn = False
            except Exception:
                self._disable()
                return

        if self.on_subprocess_ready:
            try:
                self.on_subprocess_ready()
            except Exception:
                pass

    def _disable(self):
        """Disable diarization rather than importing PyTorch into the MLX process.

        The previous in-process fallback imported PyTorch, whose C++ runtime
        conflicts with MLX at the dynamic linker level — the exact crash that
        subprocess isolation was designed to prevent.  Instead we gracefully
        degrade: all calls return safe defaults until the app is restarted.
        """
        with self._lock:
            if self._mode == "disabled":
                return
            self._mode = "disabled"
            self._kill()
