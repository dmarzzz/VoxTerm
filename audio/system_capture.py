"""System audio capture via platform-specific backends.

On macOS: uses a Swift helper binary (ScreenCaptureKit) compiled on first use.
On other platforms: no-op (returns empty chunks). Future backends can be added.
"""

from __future__ import annotations

import os
import signal
import queue
import subprocess
import threading
import numpy as np
from pathlib import Path

from audio.platform import CURRENT_PLATFORM, Platform, has_swiftc, get_output_device_info
from config import SAMPLE_RATE, BIN_DIR

# 1024 samples * 4 bytes/float32 = 4096 bytes per chunk
_CHUNK_SAMPLES = 1024
_CHUNK_BYTES = _CHUNK_SAMPLES * 4

# Swift source lives next to this file
_SWIFT_SOURCE = Path(__file__).parent / "_macos_sck.swift"
_BINARY_PATH = BIN_DIR / "sck-helper"


class SystemCapture:
    """Captures system/desktop audio. Same interface as AudioCapture."""

    def __init__(self):
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._active = False
        self._unavailable = False
        self._status_message = ""
        self._bt_multi_output_active = False  # True if we created a multi-output device
        self._died_during_recording = False  # A2: set if helper dies while active

    # ── public API (matches AudioCapture) ────────────────────

    def start(self) -> None:
        if self._active:
            return
        self._died_during_recording = False  # reset from previous session
        if CURRENT_PLATFORM != Platform.MACOS:
            self._unavailable = True
            self._status_message = "system audio capture not supported on this platform"
            return

        # Kill any stale sck-helper from a prior crash so it releases the audio tap
        self._kill_stale_helpers()

        binary = self._ensure_binary()
        if binary is None:
            return

        try:
            self._proc = subprocess.Popen(
                [str(binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
        except OSError as e:
            self._unavailable = True
            self._status_message = f"failed to launch system audio helper: {e}"
            return

        self._active = True
        self._status_message = ""

        # Bluetooth detected — route audio through BlackHole if available
        try:
            dev_info = get_output_device_info()
            if dev_info.get("is_bluetooth"):
                from audio.blackhole import is_blackhole_installed, create_multi_output
                if is_blackhole_installed():
                    ok, msg, _ = create_multi_output()
                    if ok:
                        self._bt_multi_output_active = True
                    else:
                        self._status_message = (
                            "system audio limited with Bluetooth — "
                            "mic recording will continue normally"
                        )
                else:
                    self._status_message = (
                        "system audio limited with Bluetooth — "
                        "mic recording will continue normally"
                    )
        except Exception:
            pass

        # Reader thread: stdout → chunked numpy arrays → queue
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="sck-reader"
        )
        self._reader_thread.start()

        # Stderr monitor: capture error messages from helper
        threading.Thread(
            target=self._stderr_loop, daemon=True, name="sck-stderr"
        ).start()

    def stop(self) -> None:
        if self._proc is None:
            return

        # Send SIGTERM so the helper's signal handler can stop the SCStream
        # and release the CoreAudio tap before exiting
        try:
            self._proc.send_signal(signal.SIGTERM)
        except OSError:
            pass

        # Wait for clean shutdown (helper stops SCStream, then exits)
        try:
            self._proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            # Force kill as last resort
            try:
                self._proc.kill()
                self._proc.wait(timeout=1)
            except OSError:
                pass

        self._proc = None
        self._active = False

        # Teardown multi-output device if we created one
        if self._bt_multi_output_active:
            try:
                from audio.blackhole import destroy_multi_output
                destroy_multi_output()
            except Exception:
                pass
            self._bt_multi_output_active = False

        # Drain remaining items from queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def drain(self) -> list[np.ndarray]:
        chunks = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def died_during_recording(self) -> bool:
        """True if the SCK helper died while audio capture was active."""
        return self._died_during_recording

    # ── private ──────────────────────────────────────────────

    @staticmethod
    def _kill_stale_helpers() -> None:
        """Find and SIGTERM any orphaned sck-helper processes."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "sck-helper"],
                capture_output=True, text=True, timeout=3,
            )
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        except Exception:
            pass

    def _ensure_binary(self) -> Path | None:
        """Compile the Swift helper if needed. Returns binary path or None."""
        if not _SWIFT_SOURCE.exists():
            self._unavailable = True
            self._status_message = "system audio helper source not found"
            return None

        # Check if binary exists and is up-to-date
        if _BINARY_PATH.exists():
            src_mtime = _SWIFT_SOURCE.stat().st_mtime
            bin_mtime = _BINARY_PATH.stat().st_mtime
            if bin_mtime >= src_mtime:
                return _BINARY_PATH

        # Need to compile
        if not has_swiftc():
            self._unavailable = True
            self._status_message = (
                "system audio requires Swift compiler — "
                "run: xcode-select --install"
            )
            return None

        BIN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["swiftc", "-O", "-o", str(_BINARY_PATH), str(_SWIFT_SOURCE)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                self._unavailable = True
                err = result.stderr.strip()[:200] if result.stderr else "unknown error"
                self._status_message = f"failed to compile system audio helper: {err}"
                return None
        except subprocess.TimeoutExpired:
            self._unavailable = True
            self._status_message = "system audio helper compilation timed out"
            return None
        except OSError as e:
            self._unavailable = True
            self._status_message = f"compilation error: {e}"
            return None

        return _BINARY_PATH

    def _reader_loop(self) -> None:
        """Read raw PCM from helper stdout, chunk into 1024-sample blocks."""
        buf = bytearray()
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._active = False
            return

        try:
            while True:
                data = proc.stdout.read(_CHUNK_BYTES)
                if not data:
                    break  # EOF — helper exited

                buf.extend(data)
                while len(buf) >= _CHUNK_BYTES:
                    chunk_bytes = bytes(buf[:_CHUNK_BYTES])
                    del buf[:_CHUNK_BYTES]
                    chunk = np.frombuffer(chunk_bytes, dtype=np.float32).copy()
                    try:
                        self.queue.put_nowait(chunk)
                    except queue.Full:
                        # Drop oldest chunk to prevent memory growth
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.queue.put_nowait(chunk)
        except (OSError, ValueError):
            pass
        finally:
            was_active = self._active
            self._active = False
            # A2: flag if the helper died while we were actively recording
            if was_active:
                self._died_during_recording = True
            # Check exit code for permission errors
            if proc.poll() == 1:
                self._status_message = (
                    "Screen Recording permission required — "
                    "grant access in System Settings > Privacy & Security > Screen Recording"
                )

    def _stderr_loop(self) -> None:
        """Capture stderr from helper for diagnostics."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                msg = line.decode("utf-8", errors="replace").strip()
                if msg and not self._status_message:
                    self._status_message = msg
        except (OSError, ValueError):
            pass
