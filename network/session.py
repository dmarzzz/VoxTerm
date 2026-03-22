"""Session manager — the central coordinator for P2P sessions.

Owns discovery, peer table, and all network threads.  Follows the same
pattern as ``diarization/proxy.py``: lock-serialized peer table access,
crash/disconnect handling outside locks, callbacks for UI notification.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Callable

from config import (
    P2P_HEARTBEAT_INTERVAL,
    P2P_HEARTBEAT_TIMEOUT,
    P2P_MAX_PEERS,
    P2P_PROTO_VERSION,
    P2P_TCP_PORT,
    P2P_UDP_PORT,
    SAMPLE_RATE,
    CHANNELS,
)
from network.clock import ClockSync
from network.crypto import (
    DecryptionError,
    derive_session_key,
    encrypt,
    decrypt,
    generate_session_code,
    send_encrypted_msg,
    recv_encrypted_msg,
)
from network.peer import PeerConnection
from network.protocol import (
    MSG_BYE,
    MSG_FINAL,
    MSG_HEARTBEAT,
    MSG_HEARTBEAT_ACK,
    MSG_HELLO,
    MSG_PARTIAL,
    build_bye,
    build_heartbeat,
    build_heartbeat_ack,
    build_hello,
    validate_message,
)

log = logging.getLogger("p2p.session")

# Encryption handshake tokens
_HANDSHAKE_HELLO = b"voxterm-hello"
_HANDSHAKE_ACK = b"voxterm-hello-ack"


class SessionManager:
    """Manages a P2P session: discovery, connections, message exchange.

    Usage::

        mgr = SessionManager("halcyon", node_id="abc123")

        # Create a new session
        code = mgr.create_session()
        print(f"Session code: {code}")

        # Or join an existing session
        mgr.join_session("VOXJ-7K3M")

        # Wire up callbacks
        mgr.on_peer_connected = lambda peer: ...
        mgr.on_final_received = lambda node_id, msg: ...

        # Broadcast transcripts
        mgr.broadcast_final("halcyon", seq=1, text="hello", ...)

        # Leave
        mgr.leave_session()
    """

    def __init__(
        self,
        display_name: str,
        node_id: str,
        tcp_port: int = P2P_TCP_PORT,
        udp_port: int = P2P_UDP_PORT,
    ):
        self._display_name = display_name
        self._node_id = node_id
        self._tcp_port = tcp_port
        self._udp_port = udp_port

        self._session_code: str | None = None
        self._session_key: bytes | None = None
        self._in_session = False

        self._peers: dict[str, PeerConnection] = {}
        self._lock = threading.Lock()

        self._server_sock: socket.socket | None = None
        self._running = False

        # Threads
        self._accept_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._read_threads: dict[str, threading.Thread] = {}

        # Callbacks (called from network threads — use call_from_thread in UI)
        self.on_peer_connected: Callable[[PeerConnection], None] | None = None
        self.on_peer_disconnected: Callable[[str, str], None] | None = None  # node_id, display_name
        self.on_final_received: Callable[[str, dict], None] | None = None  # node_id, msg
        self.on_partial_received: Callable[[str, dict], None] | None = None  # node_id, msg

    # ── properties ────────────────────────────────────────────

    @property
    def is_in_session(self) -> bool:
        return self._in_session

    @property
    def session_code(self) -> str | None:
        return self._session_code

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def peers(self) -> dict[str, PeerConnection]:
        with self._lock:
            return dict(self._peers)

    @property
    def peer_count(self) -> int:
        with self._lock:
            return len(self._peers)

    # ── session lifecycle ─────────────────────────────────────

    def create_session(self) -> str:
        """Create a new session, generate code, start listening."""
        code = generate_session_code()
        self._session_code = code
        self._session_key = derive_session_key(code)
        self._start_server()
        self._in_session = True
        log.info("Session created: %s", code)
        return code

    def join_session(self, session_code: str) -> None:
        """Join an existing session by code."""
        self._session_code = session_code
        self._session_key = derive_session_key(session_code)
        self._start_server()  # also listen for incoming connections
        self._in_session = True
        log.info("Joined session: %s", session_code)

    def join_by_ip(self, ip: str, port: int, session_code: str) -> bool:
        """Connect to a specific peer by IP (manual fallback / test entry point).

        Returns True if connection succeeded.
        """
        if not self._session_key:
            self._session_code = session_code
            self._session_key = derive_session_key(session_code)
        if not self._in_session:
            self._start_server()
            self._in_session = True

        return self._connect_to_peer(ip, port)

    def leave_session(self) -> None:
        """Send BYE to all peers and shut down."""
        self._running = False

        # Send BYE to all connected peers (outside lock)
        with self._lock:
            peers_snapshot = list(self._peers.values())
        for peer in peers_snapshot:
            self._send_to_peer(peer, build_bye(self._node_id))

        # Close all peer connections
        with self._lock:
            for peer in self._peers.values():
                peer.close()
            self._peers.clear()

        # Stop server
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        # Wait for threads
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=2.0)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        for t in list(self._read_threads.values()):
            if t.is_alive():
                t.join(timeout=2.0)
        self._read_threads.clear()

        self._in_session = False
        self._session_code = None
        self._session_key = None
        log.info("Left session")

    # ── broadcasting ──────────────────────────────────────────

    def broadcast_final(
        self,
        speaker_name: str,
        seq: int,
        text: str,
        start_ts: float,
        end_ts: float,
        confidence: float,
    ) -> None:
        """Send a FINAL transcript segment to all connected peers."""
        from network.protocol import build_final

        msg = build_final(self._node_id, speaker_name, seq, text, start_ts, end_ts, confidence)
        self._broadcast_tcp(msg)

    def broadcast_partial(
        self,
        speaker_name: str,
        seq: int,
        text: str,
        start_ts: float,
    ) -> None:
        """Send a PARTIAL transcript segment to all connected peers."""
        from network.protocol import build_partial

        msg = build_partial(self._node_id, speaker_name, seq, text, start_ts)
        self._broadcast_tcp(msg)

    def _send_to_peer(self, peer: PeerConnection, msg: dict) -> bool:
        """Send a message to a single peer, holding the send lock."""
        if not peer.sock or peer.state != "connected":
            return False
        try:
            with peer.send_lock:
                send_encrypted_msg(peer.sock, self._session_key, msg)
            peer.stats.tcp_tx += 1
            return True
        except Exception as exc:
            log.debug("Failed to send to %s: %s", peer.display_name, exc)
            return False

    def _broadcast_tcp(self, msg: dict) -> None:
        """Send a TCP message to all connected peers."""
        if not self._session_key:
            return
        with self._lock:
            peers_snapshot = list(self._peers.values())
        for peer in peers_snapshot:
            self._send_to_peer(peer, msg)

    # ── server ────────────────────────────────────────────────

    def _start_server(self) -> None:
        """Start the TCP server and background threads."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", self._tcp_port))
        self._server_sock.listen(P2P_MAX_PEERS)
        self._server_sock.settimeout(1.0)  # for clean shutdown
        self._running = True

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="p2p-accept")
        self._accept_thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="p2p-heartbeat")
        self._heartbeat_thread.start()

        log.info("TCP server listening on port %d", self._tcp_port)

    def _accept_loop(self) -> None:
        """Accept incoming TCP connections."""
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.info("Incoming connection from %s:%d", addr[0], addr[1])
            threading.Thread(
                target=self._handle_incoming,
                args=(conn, addr),
                daemon=True,
                name=f"p2p-incoming-{addr[0]}",
            ).start()

    def _handle_incoming(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        """Handle an incoming TCP connection: handshake then read loop."""
        try:
            conn.settimeout(5.0)
            if not self._do_handshake_server(conn):
                conn.close()
                return
            conn.settimeout(None)

            # Exchange HELLOs
            my_hello = build_hello(
                self._node_id, self._display_name,
                proto_v=P2P_PROTO_VERSION,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
            )
            send_encrypted_msg(conn, self._session_key, my_hello)
            their_hello = recv_encrypted_msg(conn, self._session_key)

            if not their_hello or their_hello.get("type") != MSG_HELLO:
                conn.close()
                return

            peer = PeerConnection(
                node_id=their_hello["node_id"],
                display_name=their_hello["display_name"],
                ip=addr[0],
                tcp_port=addr[1],
                udp_port=0,
                sock=conn,
                state="connected",
            )
            self._register_peer(peer)

        except Exception as exc:
            log.debug("Incoming connection failed: %s", exc)
            try:
                conn.close()
            except Exception:
                pass

    def _connect_to_peer(self, ip: str, port: int) -> bool:
        """Initiate an outgoing TCP connection to a peer."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((ip, port))

            if not self._do_handshake_client(sock):
                sock.close()
                return False
            sock.settimeout(None)

            # Exchange HELLOs
            my_hello = build_hello(
                self._node_id, self._display_name,
                proto_v=P2P_PROTO_VERSION,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
            )
            send_encrypted_msg(sock, self._session_key, my_hello)
            their_hello = recv_encrypted_msg(sock, self._session_key)

            if not their_hello or their_hello.get("type") != MSG_HELLO:
                sock.close()
                return False

            peer = PeerConnection(
                node_id=their_hello["node_id"],
                display_name=their_hello["display_name"],
                ip=ip,
                tcp_port=port,
                udp_port=0,
                sock=sock,
                state="connected",
            )
            self._register_peer(peer)
            return True

        except Exception as exc:
            log.debug("Failed to connect to %s:%d: %s", ip, port, exc)
            return False

    # ── encryption handshake ──────────────────────────────────

    def _do_handshake_server(self, conn: socket.socket) -> bool:
        """Server side: receive encrypted hello, send ack."""
        try:
            msg = recv_encrypted_msg(conn, self._session_key)
            if msg is None or msg.get("handshake") != "hello":
                return False
            send_encrypted_msg(conn, self._session_key, {"handshake": "ack"})
            return True
        except Exception:
            return False

    def _do_handshake_client(self, sock: socket.socket) -> bool:
        """Client side: send encrypted hello, expect ack."""
        try:
            send_encrypted_msg(sock, self._session_key, {"handshake": "hello"})
            msg = recv_encrypted_msg(sock, self._session_key)
            return msg is not None and msg.get("handshake") == "ack"
        except Exception:
            return False

    # ── peer management ───────────────────────────────────────

    def _register_peer(self, peer: PeerConnection) -> None:
        """Add a peer to the table and start its read loop."""
        with self._lock:
            if peer.node_id in self._peers:
                # Already connected — close the old connection
                old = self._peers[peer.node_id]
                old.close()
            self._peers[peer.node_id] = peer

        log.info("Peer connected: %s (%s)", peer.display_name, peer.node_id[:8])

        # Start read loop for this peer
        t = threading.Thread(
            target=self._read_loop,
            args=(peer,),
            daemon=True,
            name=f"p2p-read-{peer.node_id[:8]}",
        )
        self._read_threads[peer.node_id] = t
        t.start()

        if self.on_peer_connected:
            self.on_peer_connected(peer)

    def _remove_peer(self, peer: PeerConnection) -> None:
        """Remove a peer from the table (called outside lock for I/O safety)."""
        display_name = peer.display_name
        node_id = peer.node_id
        peer.close()

        with self._lock:
            # Only fire callback if this peer was still in the table
            # (prevents double-remove from heartbeat + read loop race)
            was_present = self._peers.pop(node_id, None) is not None
        self._read_threads.pop(node_id, None)

        if not was_present:
            return  # already removed by another thread

        log.info("Peer disconnected: %s (%s)", display_name, node_id[:8])

        if self.on_peer_disconnected:
            self.on_peer_disconnected(node_id, display_name)

    # ── read loop (per-peer) ──────────────────────────────────

    def _read_loop(self, peer: PeerConnection) -> None:
        """Per-peer TCP read loop. Runs in its own thread."""
        while self._running and peer.state == "connected":
            try:
                msg = recv_encrypted_msg(peer.sock, self._session_key)
            except Exception:
                msg = None

            if msg is None:
                # EOF or decryption failure
                break

            peer.stats.tcp_rx += 1
            self._dispatch_message(peer, msg)

        # Peer disconnected — clean up outside lock (proxy.py pattern)
        # This fires whether peer sent BYE or just dropped.
        self._remove_peer(peer)

    def _dispatch_message(self, peer: PeerConnection, msg: dict) -> None:
        """Route an incoming message to the appropriate handler."""
        msg_type = msg.get("type")

        if msg_type == MSG_HEARTBEAT:
            self._on_heartbeat(peer, msg)

        elif msg_type == MSG_HEARTBEAT_ACK:
            self._on_heartbeat_ack(peer, msg)

        elif msg_type == MSG_FINAL:
            peer.stats.finals_rx += 1
            if self.on_final_received:
                self.on_final_received(peer.node_id, msg)

        elif msg_type == MSG_PARTIAL:
            peer.stats.partials_rx += 1
            if self.on_partial_received:
                self.on_partial_received(peer.node_id, msg)

        elif msg_type == MSG_BYE:
            log.info("Received BYE from %s: %s", peer.display_name, msg.get("reason"))
            peer.state = "disconnected"

        else:
            log.debug("Unknown message type from %s: %s", peer.display_name, msg_type)

    # ── heartbeat ─────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Send heartbeats to all peers and check for timeouts."""
        while self._running:
            time.sleep(P2P_HEARTBEAT_INTERVAL)
            if not self._running:
                break

            with self._lock:
                peers_snapshot = list(self._peers.values())

            now = time.monotonic()
            dead_peers = []

            for peer in peers_snapshot:
                if peer.state != "connected":
                    continue

                # Send heartbeat
                peer.heartbeat_seq += 1
                hb = build_heartbeat(self._node_id, peer.heartbeat_seq, now)
                if not self._send_to_peer(peer, hb):
                    dead_peers.append(peer)
                    continue

                # Check timeout
                if not peer.is_alive(P2P_HEARTBEAT_TIMEOUT):
                    log.warning("Heartbeat timeout for %s", peer.display_name)
                    dead_peers.append(peer)

            # Remove dead peers outside lock
            for peer in dead_peers:
                self._remove_peer(peer)

    def _on_heartbeat(self, peer: PeerConnection, msg: dict) -> None:
        """Handle incoming heartbeat — respond with ack."""
        peer.last_heartbeat_recv = time.monotonic()
        ack = build_heartbeat_ack(
            self._node_id,
            echo_ts=msg["local_ts"],
            echo_node_id=msg["node_id"],
        )
        self._send_to_peer(peer, ack)

    def _on_heartbeat_ack(self, peer: PeerConnection, msg: dict) -> None:
        """Handle heartbeat ack — update clock sync."""
        peer.last_heartbeat_recv = time.monotonic()
        # We need the original send time, which is the echo_ts
        t1 = msg["echo_ts"]  # our original send time
        t2 = msg["local_ts"]  # their receive/respond time
        t3 = time.monotonic()  # now
        peer.clock.add_sample(t1, t2, t3)
