"""Per-peer connection state for P2P sessions."""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field

from network.clock import ClockSync
from network.segments import MergedSegment

log = logging.getLogger("p2p.peer")


@dataclass
class PeerStats:
    """Counters for debug and monitoring."""

    tcp_tx: int = 0
    tcp_rx: int = 0
    udp_tx: int = 0
    udp_rx: int = 0
    udp_dropped: int = 0
    finals_rx: int = 0
    partials_rx: int = 0


@dataclass
class PeerConnection:
    """All state for one connected peer."""

    node_id: str
    display_name: str
    ip: str
    tcp_port: int
    udp_port: int
    sock: socket.socket | None = None
    clock: ClockSync = field(default_factory=ClockSync)
    state: str = "connecting"  # connecting → handshaking → connected → disconnected
    heartbeat_seq: int = 0
    last_heartbeat_recv: float = field(default_factory=time.monotonic)
    pending_partial: MergedSegment | None = None
    stats: PeerStats = field(default_factory=PeerStats)
    send_lock: threading.Lock = field(default_factory=threading.Lock)

    def is_alive(self, timeout: float = 5.0) -> bool:
        """Check if the peer is still alive based on heartbeat timeout."""
        return (time.monotonic() - self.last_heartbeat_recv) < timeout

    def set_state(self, new_state: str) -> None:
        """Update peer state with logging."""
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            log.debug("Peer %s: %s -> %s", self.display_name, old_state, new_state)

    def close(self) -> None:
        """Close the TCP socket if open."""
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError as exc:
                log.debug("Socket shutdown failed for %s: %s", self.display_name, exc)
            try:
                self.sock.close()
            except OSError as exc:
                log.debug("Socket close failed for %s: %s", self.display_name, exc)
            self.sock = None
        self.set_state("disconnected")
