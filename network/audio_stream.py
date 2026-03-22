"""UDP audio streaming — send/receive encrypted PCM frames between peers.

Each audio frame is a single UDP datagram containing 20ms of 16kHz 16-bit
mono PCM (~640 bytes payload).  Fire-and-forget: no retransmission,
lost frames are treated as silence.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Callable

import numpy as np

from config import P2P_AUDIO_FRAME_MS, SAMPLE_RATE
from network.crypto import encrypt_audio_frame, decrypt_audio_frame

log = logging.getLogger("p2p.audio")

# Samples per frame: 20ms at 16kHz = 320 samples
SAMPLES_PER_FRAME = int(SAMPLE_RATE * P2P_AUDIO_FRAME_MS / 1000)
# Max UDP datagram we expect (frame + overhead)
MAX_DATAGRAM = 2048


class AudioStreamer:
    """UDP audio frame sender/receiver.

    Usage::

        streamer = AudioStreamer(node_id_bytes, session_key, udp_port=9901)
        streamer.on_frame_received = lambda node_id, seq, ts, pcm: ...
        streamer.start()

        # Send 20ms frame to all peers
        streamer.send_frame(pcm_array, seq=42, timestamp=100.5,
                           peer_addrs=[("192.168.1.42", 9901)])

        streamer.stop()
    """

    def __init__(self, node_id: bytes, key: bytes, udp_port: int = 0):
        self._node_id = node_id
        self._key = key
        self._udp_port = udp_port
        self._sock: socket.socket | None = None
        self._running = False
        self._recv_thread: threading.Thread | None = None

        # Stats
        self.tx_count = 0
        self.rx_count = 0
        self.rx_dropped = 0
        self._last_drop_report_count = 0

        # Callback: (node_id_bytes, seq, timestamp, pcm_int16_bytes)
        self.on_frame_received: Callable[[bytes, int, float, bytes], None] | None = None

    @property
    def local_port(self) -> int:
        """The actual bound port (useful when binding to port 0)."""
        if self._sock:
            return self._sock.getsockname()[1]
        return 0

    def start(self) -> None:
        """Bind UDP socket and start receive thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self._udp_port))
        self._sock.settimeout(1.0)
        self._running = True

        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="p2p-audio-recv"
        )
        self._recv_thread.start()
        log.info("Audio streamer started on port %d", self.local_port)
        log.debug("UDP audio socket bound to port %d", self.local_port)

    def stop(self) -> None:
        """Stop receive thread and close socket."""
        self._running = False
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
            if self._recv_thread.is_alive():
                log.warning("Audio recv thread did not exit within 2s")
        if self._sock:
            try:
                self._sock.close()
            except Exception as exc:
                log.debug("Audio socket close error: %s", exc)
            self._sock = None
        log.info("Audio streamer stopped (tx=%d rx=%d dropped=%d)",
                 self.tx_count, self.rx_count, self.rx_dropped)

    def send_frame(
        self,
        pcm: np.ndarray,
        seq: int,
        timestamp: float,
        peer_addrs: list[tuple[str, int]],
    ) -> None:
        """Encrypt and send a single audio frame to all peers.

        Args:
            pcm: Audio samples as float32 numpy array (will be converted to int16).
            seq: Frame sequence number.
            timestamp: Sender's monotonic clock at capture time.
            peer_addrs: List of (ip, port) tuples.
        """
        if not self._sock or not peer_addrs:
            return

        # Convert float32 [-1, 1] to int16 for wire format
        pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        datagram = encrypt_audio_frame(self._key, self._node_id, seq, timestamp, pcm_bytes)

        for addr in peer_addrs:
            try:
                self._sock.sendto(datagram, addr)
            except Exception as exc:
                log.debug("UDP send failed to %s:%d: %s", addr[0], addr[1], exc)
        self.tx_count += 1

    def _recv_loop(self) -> None:
        """Receive and decrypt UDP audio frames."""
        while self._running:
            try:
                data, addr = self._sock.recvfrom(MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                break

            result = decrypt_audio_frame(self._key, data)
            if result is None:
                self.rx_dropped += 1
                if self.rx_dropped - self._last_drop_report_count >= 100:
                    log.warning("Audio frame drops: %d total (last 100 from %s:%d)",
                                self.rx_dropped, addr[0], addr[1])
                    self._last_drop_report_count = self.rx_dropped
                continue

            node_id, pcm_bytes, seq, timestamp = result
            self.rx_count += 1

            if self.on_frame_received:
                self.on_frame_received(node_id, seq, timestamp, pcm_bytes)
