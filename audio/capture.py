import queue
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, CHUNK_SIZE, CHANNELS, DTYPE


class AudioCapture:
    """Manages microphone input via sounddevice InputStream."""

    def __init__(self):
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream = None
        self._device_error: str | None = None  # set on device-loss or error
        self._zero_count = 0  # consecutive zero-RMS chunks for heartbeat

    def _callback(self, indata, frames, time_info, status):
        # A1: check status for device errors (hot-unplug, permission loss)
        if status:
            self._device_error = str(status)

        chunk = indata[:, 0].copy()

        # A5: heartbeat — detect permission revocation or dead device
        # (stream reports active but all-zeros audio)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < 1e-10:
            self._zero_count += 1
            if self._zero_count >= 50:  # ~3 seconds of literal zeros
                if not self._device_error:
                    self._device_error = (
                        "no audio signal (device may be disconnected "
                        "or permission revoked)"
                    )
        else:
            self._zero_count = 0

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
        self._zero_count = 0
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_SIZE,
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
        """Returns device error message, or None if healthy."""
        return self._device_error

    def clear_error(self) -> None:
        self._device_error = None
        self._zero_count = 0

    def drain(self) -> list[np.ndarray]:
        """Get all available chunks from the queue."""
        chunks = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks
