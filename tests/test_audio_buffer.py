"""Tests for audio/buffer.py."""

import threading
import time

import numpy as np
import pytest

from audio.buffer import AudioBuffer


class TestAppendAndDuration:

    def test_append_and_duration(self):
        buf = AudioBuffer()
        chunk = np.zeros(1024, dtype=np.float32)
        buf.append(chunk)
        buf.append(chunk)
        buf.append(chunk)
        # 3 * 1024 / 16000 = 0.192
        assert abs(buf.duration - 0.192) < 1e-6


class TestGetAndClear:

    def test_get_and_clear(self):
        buf = AudioBuffer()
        c1 = np.ones(1024, dtype=np.float32)
        c2 = np.full(1024, 2.0, dtype=np.float32)
        buf.append(c1)
        buf.append(c2)
        audio = buf.get_and_clear()
        assert len(audio) == 2048
        assert np.allclose(audio[:1024], 1.0)
        assert np.allclose(audio[1024:], 2.0)
        # Buffer should be empty after get_and_clear
        assert buf.duration == 0.0

    def test_empty_buffer(self):
        buf = AudioBuffer()
        audio = buf.get_and_clear()
        assert len(audio) == 0
        assert audio.dtype == np.float32


class TestConcurrentAppendClear:

    def test_concurrent_append_clear(self):
        """Two threads appending, one clearing for ~1 second. No crashes."""
        buf = AudioBuffer()
        stop = threading.Event()
        errors = []

        def appender():
            try:
                while not stop.is_set():
                    chunk = np.random.randn(1024).astype(np.float32)
                    buf.append(chunk)
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                while not stop.is_set():
                    buf.get_and_clear()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=appender),
            threading.Thread(target=appender),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()

        time.sleep(1.0)
        stop.set()

        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Errors during concurrent access: {errors}"
