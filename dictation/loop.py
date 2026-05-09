"""Audio processing loop for dictation mode.

Mirrors the core logic from app.py _process_audio_inner / _trigger_transcription
but without Textual, diarization, or waveform rendering.  Runs in daemon threads.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from audio.buffer import AudioBuffer
from audio.capture import AudioCapture
from audio.vad import SileroVAD
from config import (
    CHUNK_SIZE,
    MAX_BUFFER_SECONDS,
    MIN_BUFFER_SECONDS,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    SILENCE_TRIGGER_SECONDS,
    WAVEFORM_FPS,
)
from dictation.injector import KeyboardInjector

log = logging.getLogger(__name__)


class DictationLoop:
    """Audio capture -> VAD -> buffer -> transcribe -> inject loop."""

    def __init__(
        self,
        transcriber,
        injector: KeyboardInjector,
        on_state_change: callable | None = None,
        hivemind_client=None,
    ):
        self.capture = AudioCapture()
        self.vad = SileroVAD()
        self.buffer = AudioBuffer()
        self.transcriber = transcriber
        self.injector = injector
        self._on_state_change = on_state_change or (lambda s: None)
        # Optional hivemind sink — segments mirrored on every transcribe.
        self._hivemind = hivemind_client
        self._session_start = time.time()

        self._active = False
        self._transcribing = threading.Event()
        self._transcribe_started: float = 0
        self._had_speech = False
        self._silence_chunks = 0
        self._chunk_duration = CHUNK_SIZE / SAMPLE_RATE

        self._thread: threading.Thread | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Start mic capture and audio processing."""
        if self._active:
            return
        self._active = True
        self._had_speech = False
        self._silence_chunks = 0
        self.vad.reset()
        self.buffer.clear()
        self.capture.start()
        self._on_state_change("listening")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("dictation started")

    def stop(self) -> None:
        """Stop capture and processing."""
        if not self._active:
            return
        self._active = False
        self.capture.stop()
        self.buffer.clear()
        self._on_state_change("idle")
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        log.info("dictation stopped")

    def _run(self) -> None:
        """Main audio loop — mirrors app.py _process_audio_inner without TUI."""
        interval = 1.0 / WAVEFORM_FPS

        while self._active:
            chunks = self.capture.drain()
            if not chunks:
                time.sleep(interval)
                continue

            for chunk in chunks:
                # Speech detection: Silero VAD (neural) with RMS fallback
                if self.vad.is_loaded:
                    is_speech = self.vad.is_speech(chunk)
                else:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    is_speech = rms >= SILENCE_THRESHOLD

                if is_speech:
                    self._silence_chunks = 0
                    self._had_speech = True
                    self.buffer.append(chunk)
                else:
                    self._silence_chunks += 1
                    if self._had_speech:
                        self.buffer.append(chunk)

            # Check transcription trigger
            silence_duration = self._silence_chunks * self._chunk_duration
            buffer_duration = self.buffer.duration

            # Watchdog: same graduated timeout as app.py:621-640
            if self._transcribing.is_set():
                elapsed = time.time() - self._transcribe_started if self._transcribe_started else 0
                if elapsed > 30:
                    log.error("transcription hung (%ds) — resetting", int(elapsed))
                    self._transcribing.clear()
                elif elapsed > 15:
                    log.warning("transcription slow (%ds) — resetting", int(elapsed))
                    self._transcribing.clear()
                time.sleep(interval)
                continue

            if (self._had_speech
                    and silence_duration > SILENCE_TRIGGER_SECONDS
                    and buffer_duration > MIN_BUFFER_SECONDS):
                self._trigger_transcription()
            elif buffer_duration >= MAX_BUFFER_SECONDS:
                self._trigger_transcription()

            time.sleep(interval)

    def _trigger_transcription(self) -> None:
        """Send accumulated audio to transcription worker."""
        if self._transcribing.is_set():
            return
        audio = self.buffer.get_and_clear()
        if len(audio) < int(SAMPLE_RATE * MIN_BUFFER_SECONDS):
            return

        self._had_speech = False
        self._silence_chunks = 0
        self._transcribing.set()
        self._transcribe_started = time.time()
        self._on_state_change("transcribing")
        threading.Thread(
            target=self._transcribe_worker, args=(audio,), daemon=True,
        ).start()

    def _transcribe_worker(self, audio: np.ndarray) -> None:
        """Transcribe audio and inject text into focused app."""
        try:
            result = self.transcriber.transcribe(audio)
            text = result.get("text", "").strip()
            if text:
                # Capitalize first letter of each segment
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
                self.injector.type_text(text + " ")
                log.info("injected: %s", text[:60])
                # Mirror to hivemind sink if configured. Dictation has
                # no diarization so speaker is the literal "user".
                if self._hivemind is not None:
                    try:
                        self._hivemind.add_segment(
                            time.time() - self._session_start, "user", text,
                        )
                    except Exception:
                        log.warning("hivemind add_segment failed", exc_info=True)
        except Exception as e:
            log.error("transcription error: %s", e)
        finally:
            self._transcribing.clear()
            if self._active:
                self._on_state_change("listening")
