import queue
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, CHUNK_SIZE, CHANNELS, DTYPE


class AudioCapture:
    """Manages microphone input via sounddevice InputStream."""

    def __init__(self):
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        try:
            self.queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(indata[:, 0].copy())

    def start(self):
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

    def drain(self) -> list[np.ndarray]:
        """Get all available chunks from the queue."""
        chunks = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks
