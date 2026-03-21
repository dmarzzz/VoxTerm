"""Tests for the binary IPC protocol in diarization/ipc.py."""

import io

import numpy as np
import pytest

from diarization.ipc import encode_array, decode_array, send_msg, recv_msg


class TestEncodeDecodeArray:

    def test_encode_decode_array(self):
        rng = np.random.RandomState(42)
        arr = rng.randn(512).astype(np.float32)
        encoded = encode_array(arr)
        decoded = decode_array(encoded)
        assert decoded.dtype == np.float32
        assert np.allclose(arr, decoded)

    def test_encode_empty_array(self):
        arr = np.array([], dtype=np.float32)
        encoded = encode_array(arr)
        decoded = decode_array(encoded)
        assert decoded.dtype == np.float32
        assert len(decoded) == 0
        assert np.allclose(arr, decoded)


class TestSendRecvMessage:

    def test_send_recv_message(self):
        pipe = io.BytesIO()
        msg = {"type": "identify", "speaker_id": 3, "label": "Alice"}
        send_msg(pipe, msg)
        pipe.seek(0)
        received = recv_msg(pipe)
        assert received == msg

    def test_send_recv_large_message(self):
        """15 seconds of 16kHz audio encoded as a hex payload."""
        audio = np.random.randn(16000 * 15).astype(np.float32)
        msg = {"type": "identify", "audio": encode_array(audio)}
        pipe = io.BytesIO()
        send_msg(pipe, msg)
        pipe.seek(0)
        received = recv_msg(pipe)
        assert received is not None
        recovered = decode_array(received["audio"])
        assert np.allclose(audio, recovered)

    def test_recv_eof(self):
        pipe = io.BytesIO(b"")
        result = recv_msg(pipe)
        assert result is None

    def test_recv_truncated(self):
        """A partial header (fewer than 4 bytes) should return None."""
        pipe = io.BytesIO(b"\x05\x00")  # only 2 bytes of a 4-byte header
        result = recv_msg(pipe)
        assert result is None
