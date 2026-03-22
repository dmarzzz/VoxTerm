"""Tests for P2P peer lifecycle — connection, handshake, session management."""

import socket
import threading
import time

import pytest

from network.crypto import derive_session_key
from network.peer import PeerConnection
from network.session import SessionManager


class TestPeerConnection:
    def test_initial_state(self):
        peer = PeerConnection(
            node_id="abc",
            display_name="alice",
            ip="127.0.0.1",
            tcp_port=9900,
            udp_port=9901,
        )
        assert peer.state == "connecting"
        assert peer.heartbeat_seq == 0
        assert peer.is_alive(timeout=5.0)

    def test_is_alive_timeout(self):
        peer = PeerConnection(
            node_id="abc",
            display_name="alice",
            ip="127.0.0.1",
            tcp_port=9900,
            udp_port=9901,
        )
        # Force last_heartbeat_recv to the past
        peer.last_heartbeat_recv = time.monotonic() - 10.0
        assert not peer.is_alive(timeout=5.0)

    def test_close(self):
        peer = PeerConnection(
            node_id="abc",
            display_name="alice",
            ip="127.0.0.1",
            tcp_port=9900,
            udp_port=9901,
        )
        peer.close()
        assert peer.state == "disconnected"
        assert peer.sock is None


class TestSessionManagerLifecycle:
    def test_create_session(self):
        mgr = SessionManager("alice", node_id="node-a", tcp_port=0)
        try:
            code = mgr.create_session()
            assert code is not None
            assert len(code.split("-")) == 3  # word-word-word
            assert mgr.is_in_session
            assert mgr.session_code == code
            assert mgr.peer_count == 0
        finally:
            mgr.leave_session()

    def test_leave_session(self):
        mgr = SessionManager("alice", node_id="node-a", tcp_port=0)
        mgr.create_session()
        mgr.leave_session()
        assert not mgr.is_in_session
        assert mgr.session_code is None


@pytest.mark.timeout(10)
class TestTwoPeerConnection:
    """Integration test: two SessionManagers connect on loopback."""

    def test_two_peers_connect(self):
        connected_events = {"a": threading.Event(), "b": threading.Event()}

        mgr_a = SessionManager("alice", node_id="node-a", tcp_port=0)
        mgr_b = SessionManager("bob", node_id="node-b", tcp_port=0)

        try:
            code = mgr_a.create_session()
            port_a = mgr_a._server_sock.getsockname()[1]

            mgr_a.on_peer_connected = lambda p: connected_events["a"].set()
            mgr_b.on_peer_connected = lambda p: connected_events["b"].set()

            assert mgr_b.join_by_ip("127.0.0.1", port_a, code)

            # Wait for both sides to register the connection
            assert connected_events["a"].wait(timeout=5.0)
            assert connected_events["b"].wait(timeout=5.0)

            assert mgr_a.peer_count == 1
            assert mgr_b.peer_count == 1

            # Verify peer info
            peers_a = mgr_a.peers
            peer_b_from_a = list(peers_a.values())[0]
            assert peer_b_from_a.display_name == "bob"
            assert peer_b_from_a.state == "connected"

        finally:
            mgr_b.leave_session()
            mgr_a.leave_session()

    def test_wrong_session_code_rejected(self):
        mgr_a = SessionManager("alice", node_id="node-a", tcp_port=0)
        mgr_b = SessionManager("bob", node_id="node-b", tcp_port=0)

        try:
            code_a = mgr_a.create_session()
            port_a = mgr_a._server_sock.getsockname()[1]

            # Bob tries with wrong code
            result = mgr_b.join_by_ip("127.0.0.1", port_a, "WRONG-CODE")
            assert not result
            assert mgr_a.peer_count == 0

        finally:
            mgr_b.leave_session()
            mgr_a.leave_session()

    def test_bye_disconnects_peer(self):
        connected_a = threading.Event()
        disconnected_a = threading.Event()

        mgr_a = SessionManager("alice", node_id="node-a", tcp_port=0)
        mgr_b = SessionManager("bob", node_id="node-b", tcp_port=0)

        try:
            code = mgr_a.create_session()
            port_a = mgr_a._server_sock.getsockname()[1]

            mgr_a.on_peer_connected = lambda p: connected_a.set()
            mgr_a.on_peer_disconnected = lambda nid, name: disconnected_a.set()

            mgr_b.join_by_ip("127.0.0.1", port_a, code)
            connected_a.wait(timeout=5.0)

            # Bob leaves
            mgr_b.leave_session()

            # Alice should detect disconnect
            assert disconnected_a.wait(timeout=5.0)
            assert mgr_a.peer_count == 0

        finally:
            mgr_a.leave_session()

    def test_transcript_exchange(self):
        """Two peers exchange FINAL segments."""
        connected = {"a": threading.Event(), "b": threading.Event()}
        received_by_bob = threading.Event()
        received_by_alice = threading.Event()
        msg_at_bob = {}
        msg_at_alice = {}

        mgr_a = SessionManager("alice", node_id="node-a", tcp_port=0)
        mgr_b = SessionManager("bob", node_id="node-b", tcp_port=0)

        try:
            code = mgr_a.create_session()
            port_a = mgr_a._server_sock.getsockname()[1]

            mgr_a.on_peer_connected = lambda p: connected["a"].set()
            mgr_b.on_peer_connected = lambda p: connected["b"].set()

            def on_final_at_alice(node_id, msg):
                msg_at_alice.update(msg)
                received_by_alice.set()

            def on_final_at_bob(node_id, msg):
                msg_at_bob.update(msg)
                received_by_bob.set()

            mgr_a.on_final_received = on_final_at_alice  # Alice receives Bob's messages
            mgr_b.on_final_received = on_final_at_bob    # Bob receives Alice's messages

            mgr_b.join_by_ip("127.0.0.1", port_a, code)
            connected["a"].wait(timeout=5.0)
            connected["b"].wait(timeout=5.0)

            # Alice sends a final → Bob should receive it
            mgr_a.broadcast_final("alice", seq=1, text="hello from alice", start_ts=10.0, end_ts=12.0, confidence=0.95)
            assert received_by_bob.wait(timeout=5.0)
            assert msg_at_bob["text"] == "hello from alice"
            assert msg_at_bob["speaker_name"] == "alice"

            # Bob sends a final → Alice should receive it
            mgr_b.broadcast_final("bob", seq=1, text="hello from bob", start_ts=11.0, end_ts=13.0, confidence=0.90)
            assert received_by_alice.wait(timeout=5.0)
            assert msg_at_alice["text"] == "hello from bob"

        finally:
            mgr_b.leave_session()
            mgr_a.leave_session()
