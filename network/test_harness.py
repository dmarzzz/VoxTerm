"""Multi-peer simulation harness for stress testing.

Spins up N virtual peers in-process using threads and loopback sockets.
Each virtual peer can broadcast synthetic transcript segments.
Useful for testing mesh behavior, bandwidth scaling, and clock sync
convergence without needing multiple physical machines.

Usage::

    harness = PeerHarness(peer_count=5, session_code="test-bacon-horse")
    harness.start()

    # All peers are connected to each other
    # Broadcast from one peer
    harness.peers[0].mgr.broadcast_final("alice", 1, "hello", 0, 1, 0.9)

    # Check what peers received
    for peer in harness.peers:
        print(f"{peer.name}: {len(peer.received_finals)} finals")

    harness.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from network.session import SessionManager


@dataclass
class VirtualPeer:
    """A simulated VoxTerm peer for testing."""

    name: str
    node_id: str
    mgr: SessionManager
    received_finals: list = field(default_factory=list)
    received_partials: list = field(default_factory=list)
    connected_peers: list = field(default_factory=list)


class PeerHarness:
    """Spin up N virtual peers on loopback, fully meshed.

    Args:
        peer_count: Number of virtual peers to create.
        session_code: Shared session code for all peers.
    """

    def __init__(self, peer_count: int = 3, session_code: str = "test-bacon-horse"):
        self._session_code = session_code
        self._peer_count = peer_count
        self.peers: list[VirtualPeer] = []
        self._started = False

    def start(self) -> None:
        """Create all peers, connect them in a mesh, and start heartbeats."""
        # Create peers
        for i in range(self._peer_count):
            name = f"peer-{i}"
            node_id = f"node-{i:012d}"
            mgr = SessionManager(name, node_id=node_id, tcp_port=0)
            vp = VirtualPeer(name=name, node_id=node_id, mgr=mgr)

            # Wire callbacks
            def make_final_cb(peer):
                def cb(nid, msg):
                    peer.received_finals.append((nid, msg))
                return cb

            def make_partial_cb(peer):
                def cb(nid, msg):
                    peer.received_partials.append((nid, msg))
                return cb

            def make_connected_cb(peer):
                def cb(p):
                    peer.connected_peers.append(p.node_id)
                return cb

            mgr.on_final_received = make_final_cb(vp)
            mgr.on_partial_received = make_partial_cb(vp)
            mgr.on_peer_connected = make_connected_cb(vp)

            self.peers.append(vp)

        # First peer creates the session
        self.peers[0].mgr.create_session()
        self.peers[0].mgr._session_code = self._session_code
        from network.crypto import derive_session_key
        self.peers[0].mgr._session_key = derive_session_key(self._session_code)
        creator_port = self.peers[0].mgr._server_sock.getsockname()[1]

        # Other peers join via the first peer's port
        for vp in self.peers[1:]:
            vp.mgr.join_by_ip("127.0.0.1", creator_port, self._session_code)

        # Wait for connections to establish
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            # Each peer should be connected to at least 1 other peer
            all_connected = all(len(vp.connected_peers) >= 1 for vp in self.peers)
            if all_connected:
                break
            time.sleep(0.1)

        self._started = True

    def stop(self) -> None:
        """Shut down all peers."""
        for vp in reversed(self.peers):
            try:
                vp.mgr.leave_session()
            except Exception:
                pass
        self.peers.clear()
        self._started = False

    def broadcast_from(self, peer_index: int, text: str, seq: int = 1) -> None:
        """Have a specific peer broadcast a FINAL segment."""
        vp = self.peers[peer_index]
        vp.mgr.broadcast_final(
            speaker_name=vp.name,
            seq=seq,
            text=text,
            start_ts=time.monotonic(),
            end_ts=time.monotonic() + 1.0,
            confidence=0.9,
        )

    def wait_for_finals(self, count: int, timeout: float = 5.0) -> bool:
        """Wait until the total finals received across all peers reaches count."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            total = sum(len(vp.received_finals) for vp in self.peers)
            if total >= count:
                return True
            time.sleep(0.05)
        return False
