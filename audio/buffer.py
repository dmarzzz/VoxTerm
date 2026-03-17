import threading
import numpy as np
from config import SAMPLE_RATE


class AudioBuffer:
    """Thread-safe audio accumulator for transcription chunks."""

    def __init__(self):
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray):
        with self._lock:
            self._buffer.append(chunk)
            self._total_samples += len(chunk)

    def get_and_clear(self) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._buffer)
            self._buffer.clear()
            self._total_samples = 0
            return audio

    @property
    def duration(self) -> float:
        with self._lock:
            return self._total_samples / SAMPLE_RATE

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0
