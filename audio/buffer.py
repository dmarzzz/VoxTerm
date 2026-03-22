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


class PeerAudioBuffer:
    """Ring buffer for incoming peer audio — sequence-based, fixed max duration.

    Incoming UDP frames arrive by sequence number.  Gaps (lost frames) are
    filled with silence.  Oldest data is discarded when the buffer exceeds
    ``max_duration_sec``.
    """

    def __init__(self, max_duration_sec: float = 2.0, frame_samples: int = 320):
        self._max_samples = int(max_duration_sec * SAMPLE_RATE)
        self._frame_samples = frame_samples
        self._buffer = np.zeros(self._max_samples, dtype=np.int16)
        self._write_pos = 0
        self._last_seq = -1
        self._lock = threading.Lock()
        self.frames_received = 0
        self.gaps_filled = 0

    def write_frame(self, seq: int, pcm_int16: bytes) -> None:
        """Write a frame by sequence number, filling gaps with silence."""
        frame = np.frombuffer(pcm_int16, dtype=np.int16)
        n = len(frame)

        with self._lock:
            if self._last_seq >= 0 and seq > self._last_seq + 1:
                # Fill gap with silence
                gap = seq - self._last_seq - 1
                silence_samples = gap * self._frame_samples
                self._advance(np.zeros(silence_samples, dtype=np.int16))
                self.gaps_filled += gap

            self._advance(frame)
            self._last_seq = seq
            self.frames_received += 1

    def _advance(self, data: np.ndarray) -> None:
        """Append data to the ring buffer (caller holds lock)."""
        n = len(data)
        if n >= self._max_samples:
            # Overwrite entire buffer
            self._buffer[:] = data[-self._max_samples:]
            self._write_pos = 0
            return

        end = self._write_pos + n
        if end <= self._max_samples:
            self._buffer[self._write_pos:end] = data
        else:
            first = self._max_samples - self._write_pos
            self._buffer[self._write_pos:] = data[:first]
            self._buffer[:n - first] = data[first:]
        self._write_pos = end % self._max_samples

    def read(self, duration_sec: float) -> np.ndarray:
        """Read the most recent N seconds as float32 [-1, 1]."""
        n = min(int(duration_sec * SAMPLE_RATE), self._max_samples)
        with self._lock:
            start = (self._write_pos - n) % self._max_samples
            if start < self._write_pos:
                data = self._buffer[start:self._write_pos].copy()
            else:
                data = np.concatenate([
                    self._buffer[start:],
                    self._buffer[:self._write_pos],
                ])
        return data.astype(np.float32) / 32767.0

    @property
    def duration(self) -> float:
        return self._max_samples / SAMPLE_RATE

    def clear(self) -> None:
        with self._lock:
            self._buffer[:] = 0
            self._write_pos = 0
            self._last_seq = -1
