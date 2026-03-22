"""Tests for P2P UDP audio streaming."""

import threading
import time

import numpy as np
import pytest

from audio.buffer import PeerAudioBuffer
from network.audio_stream import AudioStreamer, SAMPLES_PER_FRAME
from network.crypto import derive_session_key


class TestAudioStreamer:
    def _make_key(self, code="TEST-CODE"):
        return derive_session_key(code)

    def test_send_recv_round_trip(self):
        """Send a frame from one streamer, receive on another."""
        key = self._make_key()
        node_id = b"sender1234567890"

        sender = AudioStreamer(node_id, key, udp_port=0)
        receiver = AudioStreamer(b"recvr12345678901", key, udp_port=0)

        received = threading.Event()
        result = {}

        def on_frame(nid, seq, ts, pcm_bytes):
            result["node_id"] = nid
            result["seq"] = seq
            result["ts"] = ts
            result["pcm"] = pcm_bytes
            received.set()

        receiver.on_frame_received = on_frame
        sender.start()
        receiver.start()

        try:
            pcm = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
            pcm[0] = 0.5  # identifiable sample

            sender.send_frame(
                pcm, seq=42, timestamp=100.5,
                peer_addrs=[("127.0.0.1", receiver.local_port)],
            )

            assert received.wait(timeout=5.0)
            assert result["seq"] == 42
            assert abs(result["ts"] - 100.5) < 1e-6
            # Verify audio content survived
            received_int16 = np.frombuffer(result["pcm"], dtype=np.int16)
            assert len(received_int16) == SAMPLES_PER_FRAME
            assert received_int16[0] == int(0.5 * 32767)  # float32→int16 conversion

        finally:
            sender.stop()
            receiver.stop()

    def test_wrong_key_dropped(self):
        """Frames encrypted with wrong key should be silently dropped."""
        key1 = self._make_key("CODE-AAAA")
        key2 = self._make_key("CODE-BBBB")

        sender = AudioStreamer(b"s" * 16, key1, udp_port=0)
        receiver = AudioStreamer(b"r" * 16, key2, udp_port=0)

        received = threading.Event()
        receiver.on_frame_received = lambda *args: received.set()

        sender.start()
        receiver.start()

        try:
            pcm = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
            sender.send_frame(pcm, 0, 0.0, [("127.0.0.1", receiver.local_port)])
            # Should NOT be received
            assert not received.wait(timeout=1.0)
            assert receiver.rx_dropped >= 1
        finally:
            sender.stop()
            receiver.stop()

    def test_multiple_frames(self):
        """Send multiple frames, verify all received in order."""
        key = self._make_key()
        sender = AudioStreamer(b"s" * 16, key, udp_port=0)
        receiver = AudioStreamer(b"r" * 16, key, udp_port=0)

        seqs = []
        done = threading.Event()

        def on_frame(nid, seq, ts, pcm_bytes):
            seqs.append(seq)
            if len(seqs) >= 5:
                done.set()

        receiver.on_frame_received = on_frame
        sender.start()
        receiver.start()

        try:
            pcm = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
            for i in range(5):
                sender.send_frame(pcm, seq=i, timestamp=float(i),
                                  peer_addrs=[("127.0.0.1", receiver.local_port)])
                time.sleep(0.01)

            assert done.wait(timeout=5.0)
            assert seqs == [0, 1, 2, 3, 4]
        finally:
            sender.stop()
            receiver.stop()

    def test_stats(self):
        key = self._make_key()
        sender = AudioStreamer(b"s" * 16, key, udp_port=0)
        receiver = AudioStreamer(b"r" * 16, key, udp_port=0)

        done = threading.Event()
        receiver.on_frame_received = lambda *a: done.set()
        sender.start()
        receiver.start()

        try:
            pcm = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
            sender.send_frame(pcm, 0, 0.0, [("127.0.0.1", receiver.local_port)])
            done.wait(timeout=5.0)
            assert sender.tx_count >= 1
            assert receiver.rx_count >= 1
        finally:
            sender.stop()
            receiver.stop()


class TestPeerAudioBuffer:
    def test_write_and_read(self):
        buf = PeerAudioBuffer(max_duration_sec=1.0, frame_samples=320)
        # Write one frame
        frame = np.ones(320, dtype=np.int16) * 1000
        buf.write_frame(0, frame.tobytes())
        assert buf.frames_received == 1

        # Read back
        audio = buf.read(0.02)  # 20ms = 320 samples
        assert len(audio) == 320
        assert audio[-1] == pytest.approx(1000 / 32767.0, abs=0.001)

    def test_gap_filling(self):
        buf = PeerAudioBuffer(max_duration_sec=1.0, frame_samples=320)
        frame = np.ones(320, dtype=np.int16) * 500
        buf.write_frame(0, frame.tobytes())
        # Skip seq 1, 2 — gap of 2 frames
        buf.write_frame(3, frame.tobytes())
        assert buf.gaps_filled == 2

    def test_ring_buffer_wraps(self):
        buf = PeerAudioBuffer(max_duration_sec=0.1, frame_samples=320)
        # 0.1s at 16kHz = 1600 samples. Write 10 frames of 320 = 3200 > 1600
        for i in range(10):
            frame = np.full(320, i * 100, dtype=np.int16)
            buf.write_frame(i, frame.tobytes())
        # Should still be readable without error
        audio = buf.read(0.1)
        assert len(audio) == 1600

    def test_clear(self):
        buf = PeerAudioBuffer(max_duration_sec=1.0, frame_samples=320)
        buf.write_frame(0, np.ones(320, dtype=np.int16).tobytes())
        buf.clear()
        audio = buf.read(0.02)
        assert np.all(audio == 0.0)
