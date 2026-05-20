import math
import queue
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from audio.noise_filter import HighPassFilter
from config import (
    SAMPLE_RATE, CHUNK_SIZE, DTYPE,
    NOISE_FILTER_ENABLED, NOISE_FILTER_CUTOFF_HZ, NOISE_FILTER_ORDER,
)


class AudioCapture:
    """Manages microphone input via sounddevice InputStream."""

    def __init__(self):
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream = None
        self._native_rate = None
        self._resample_up = None
        self._resample_down = None
        # Surface device-loss / permission-revoke conditions to the UI.
        # _device_error is set from the audio callback, read from the audio loop.
        self._device_error: str | None = None
        self._zero_chunks = 0  # consecutive all-silence chunks (heartbeat)
        # Streaming high-pass filter for AC hum / fan rumble / HVAC noise.
        # Cuts the low-frequency noise floor so VAD's RMS gate isn't held open
        # by background noise.
        self._noise_filter = (
            HighPassFilter(NOISE_FILTER_CUTOFF_HZ, SAMPLE_RATE, NOISE_FILTER_ORDER)
            if NOISE_FILTER_ENABLED else None
        )

    def _callback(self, indata, frames, time_info, status):
        # Surface portaudio status flags (input overflow, device loss, etc.)
        if status:
            self._device_error = str(status)

        # Mix to mono
        if indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0]
        # Resample if device rate differs from target
        if self._resample_up is not None:
            mono = resample_poly(mono, self._resample_up, self._resample_down)
        chunk = mono.astype(np.float32)

        # High-pass filter applied after resampling so the cutoff is meaningful
        # in the final SAMPLE_RATE domain (VAD + transcriber see filtered audio).
        # Runs before the zero-RMS heartbeat so a dead mic still reads as zero.
        if self._noise_filter is not None:
            chunk = self._noise_filter.filter(chunk)

        # Heartbeat: a healthy mic has noise floor > ~1e-6 even in a quiet
        # room. Sustained literal-zero RMS suggests the device is dead or
        # mic permission was revoked mid-recording (macOS substitutes silence).
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        if rms < 1e-9:
            self._zero_chunks += 1
            if self._zero_chunks >= 100 and self._device_error is None:
                # ~100 chunks ≈ 6s at 1024@16k — well past any natural pause
                self._device_error = (
                    "no audio signal — mic may be disconnected or permission revoked"
                )
        else:
            self._zero_chunks = 0

        try:
            self.queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(chunk)

    def start(self):
        self._device_error = None
        self._zero_chunks = 0
        # Reset filter state to avoid a click on resume.
        if self._noise_filter is not None:
            self._noise_filter.reset()
        dev_info = sd.query_devices(kind='input')
        self._device_name = dev_info['name']
        native_channels = dev_info['max_input_channels']
        self._native_rate = int(dev_info['default_samplerate'])
        # Compute resample ratio as integer up/down factors
        if self._native_rate != SAMPLE_RATE:
            g = math.gcd(SAMPLE_RATE, self._native_rate)
            self._resample_up = SAMPLE_RATE // g
            self._resample_down = self._native_rate // g
        else:
            self._resample_up = None
            self._resample_down = None
        self._stream = sd.InputStream(
            samplerate=self._native_rate,
            channels=native_channels,
            dtype=DTYPE,
            blocksize=int(CHUNK_SIZE * self._native_rate / SAMPLE_RATE),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def device_error(self) -> str | None:
        return self._device_error

    def drain(self) -> list[np.ndarray]:
        """Get all available chunks from the queue."""
        chunks = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks
