#!/usr/bin/env python3
"""End-to-end P2P integration test — runs the full stack locally.

Spawns two simulated VoxTerm nodes in one process. Each node has:
  - mDNS discovery (real Bonjour/Avahi)
  - TCP session manager (real sockets on loopback)
  - Transcript exchange (real messages)

Tests run in sequence, each building on the last. Prints a clear
pass/fail report at the end.

Usage:
    python3 test_p2p_e2e.py
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network.discovery import PeerDiscovery
from network.session import SessionManager
from network.segments import TranscriptAssembler
from network.crypto import generate_session_code, derive_session_key


# ── helpers ───────────────────────────────────────────────────

class SimNode:
    """A simulated VoxTerm node with discovery + session + assembler."""

    def __init__(self, name: str, node_id: str):
        self.name = name
        self.node_id = node_id
        self.discovery: PeerDiscovery | None = None
        self.session: SessionManager | None = None
        self.assembler = TranscriptAssembler()
        self.received_finals: list[dict] = []
        self.received_partials: list[dict] = []
        self.connected_peers: list[str] = []
        self.disconnected_peers: list[str] = []
        self._connected_event = threading.Event()
        self._final_event = threading.Event()

    def start_discovery(self, tcp_port: int):
        self.discovery = PeerDiscovery(self.node_id, self.name, tcp_port=tcp_port, udp_port=0)
        self.discovery.start()
        self.discovery.update_session_status(True)

    def start_session(self, code: str):
        self.session = SessionManager(self.name, node_id=self.node_id, tcp_port=0)
        self.session._session_code = code
        self.session._session_key = derive_session_key(code)
        self.session._start_server()
        self.session._in_session = True

        self.session.on_peer_connected = lambda p: self._on_connected(p)
        self.session.on_peer_disconnected = lambda nid, name: self._on_disconnected(nid, name)
        self.session.on_final_received = lambda nid, msg: self._on_final(nid, msg)
        self.session.on_partial_received = lambda nid, msg: self._on_partial(nid, msg)

        return self.session._server_sock.getsockname()[1]

    def _on_connected(self, peer):
        self.connected_peers.append(peer.display_name)
        self._connected_event.set()

    def _on_disconnected(self, nid, name):
        self.disconnected_peers.append(name)

    def _on_final(self, nid, msg):
        self.received_finals.append(msg)
        self._final_event.set()

    def _on_partial(self, nid, msg):
        self.received_partials.append(msg)

    def wait_connected(self, timeout=10):
        return self._connected_event.wait(timeout)

    def wait_final(self, timeout=5):
        return self._final_event.wait(timeout)

    def stop(self):
        if self.session and self.session.is_in_session:
            try:
                self.session.leave_session()
            except Exception:
                pass
        if self.discovery:
            try:
                self.discovery.stop()
            except Exception:
                pass


# ── test runner ───────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.tests: list[tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str = ""):
        self.tests.append((name, passed, detail))
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))

    def summary(self):
        total = len(self.tests)
        passed = sum(1 for _, p, _ in self.tests if p)
        failed = total - passed
        print()
        print("=" * 60)
        if failed == 0:
            print(f"\033[92mALL {total} TESTS PASSED\033[0m")
        else:
            print(f"\033[91m{failed} FAILED\033[0m, {passed} passed out of {total}")
        print("=" * 60)
        return failed == 0


def run_tests():
    print("=" * 60)
    print("P2P END-TO-END INTEGRATION TESTS")
    print("=" * 60)
    print()

    results = TestResult()
    code = generate_session_code()
    alice = SimNode("alice", "aaaa1111aaaa1111")
    bob = SimNode("bob", "bbbb2222bbbb2222")

    try:
        # ── Test 1: mDNS Discovery ────────────────────────────
        print("Phase 1: mDNS Discovery")

        alice_port = alice.start_session(code)
        bob_port = bob.start_session(code)

        alice_found = threading.Event()
        bob_found = threading.Event()
        alice_found_peers = []
        bob_found_peers = []

        # Wire callbacks BEFORE starting discovery to avoid race
        def make_found_cb(self_id, peers_list, event):
            def cb(pi):
                if pi.node_id != self_id:
                    peers_list.append(pi.display_name)
                    event.set()
            return cb

        alice.discovery = PeerDiscovery(alice.node_id, alice.name, tcp_port=alice_port, udp_port=0)
        alice.discovery.on_peer_found = make_found_cb(alice.node_id, alice_found_peers, alice_found)

        bob.discovery = PeerDiscovery(bob.node_id, bob.name, tcp_port=bob_port, udp_port=0)
        bob.discovery.on_peer_found = make_found_cb(bob.node_id, bob_found_peers, bob_found)

        alice.discovery.start()
        alice.discovery.update_session_status(True)
        bob.discovery.start()
        bob.discovery.update_session_status(True)

        a_ok = alice_found.wait(timeout=10)
        b_ok = bob_found.wait(timeout=10)

        results.record(
            "Alice discovers Bob via mDNS",
            a_ok,
            f"found: {alice_found_peers}" if a_ok else "timeout"
        )
        results.record(
            "Bob discovers Alice via mDNS",
            b_ok,
            f"found: {bob_found_peers}" if b_ok else "timeout"
        )

        if not (a_ok or b_ok):
            print("\n  !! mDNS discovery failed — cannot continue.")
            print("     Check: is Bonjour/avahi running? Firewall blocking UDP 5353?")
            return results.summary()

        # ── Test 2: TCP Connection ────────────────────────────
        print("\nPhase 2: TCP Connection")

        # Connect alice to bob directly
        target = alice.discovery.get_visible_peers()
        print(f"  Alice sees: {[(p.display_name, p.ip, p.tcp_port, p.node_id[:8]) for p in target]}")
        connected = False
        for pi in target:
            if pi.node_id == bob.node_id:
                print(f"  Alice connecting to {pi.display_name} at {pi.ip}:{pi.tcp_port}...")
                connected = alice.session.join_by_ip(pi.ip, pi.tcp_port, code)
                print(f"  Connection result: {connected}")
                break
        if not connected:
            print(f"  Alice could not find Bob in visible peers, trying bob's port directly...")
            connected = alice.session.join_by_ip("127.0.0.1", bob_port, code)
            print(f"  Direct connection result: {connected}")

        alice_connected = alice.wait_connected(timeout=5)
        bob_connected = bob.wait_connected(timeout=5)

        results.record(
            "Alice sees Bob connected",
            alice_connected,
            f"peers: {alice.connected_peers}" if alice_connected else "timeout"
        )
        results.record(
            "Bob sees Alice connected",
            bob_connected,
            f"peers: {bob.connected_peers}" if bob_connected else "timeout"
        )

        if not (alice_connected and bob_connected):
            print("\n  !! TCP connection failed — cannot continue transcript tests.")
            return results.summary()

        # ── Test 3: Transcript Exchange ───────────────────────
        print("\nPhase 3: Transcript Exchange")

        # Alice sends to Bob
        alice.session.broadcast_final(
            "alice", seq=1, text="hello from alice",
            start_ts=time.monotonic(), end_ts=time.monotonic() + 1.0,
            confidence=0.95,
        )
        bob_got = bob.wait_final(timeout=5)
        results.record(
            "Bob receives Alice's transcript",
            bob_got and len(bob.received_finals) > 0,
            f"text: {bob.received_finals[0]['text']}" if bob.received_finals else "nothing received"
        )

        # Bob sends to Alice
        bob.session.broadcast_final(
            "bob", seq=1, text="hello from bob",
            start_ts=time.monotonic(), end_ts=time.monotonic() + 1.0,
            confidence=0.90,
        )
        alice_got = alice.wait_final(timeout=5)
        results.record(
            "Alice receives Bob's transcript",
            alice_got and len(alice.received_finals) > 0,
            f"text: {alice.received_finals[0]['text']}" if alice.received_finals else "nothing received"
        )

        # ── Test 4: Rapid Burst ───────────────────────────────
        print("\nPhase 4: Rapid Burst (20 messages each direction)")

        bob.received_finals.clear()
        bob._final_event.clear()
        alice.received_finals.clear()
        alice._final_event.clear()

        ts = time.monotonic()
        for i in range(20):
            alice.session.broadcast_final("alice", seq=100+i, text=f"burst-a-{i}",
                                          start_ts=ts+i*0.1, end_ts=ts+i*0.1+0.05, confidence=0.9)
            bob.session.broadcast_final("bob", seq=100+i, text=f"burst-b-{i}",
                                        start_ts=ts+i*0.1+0.05, end_ts=ts+i*0.1+0.1, confidence=0.85)

        time.sleep(2)  # let messages propagate

        bob_burst = len(bob.received_finals)
        alice_burst = len(alice.received_finals)
        results.record(
            f"Bob received {bob_burst}/20 burst messages from Alice",
            bob_burst == 20,
            f"{bob_burst}/20"
        )
        results.record(
            f"Alice received {alice_burst}/20 burst messages from Bob",
            alice_burst == 20,
            f"{alice_burst}/20"
        )

        # ── Test 5: Clock Sync ────────────────────────────────
        print("\nPhase 5: Clock Sync")

        time.sleep(2)  # let heartbeats accumulate

        alice_peers = alice.session.peers
        if alice_peers:
            peer = list(alice_peers.values())[0]
            samples = peer.clock.sample_count
            offset = peer.clock.offset
            rtt = peer.clock.rtt
            results.record(
                f"Clock sync active ({samples} samples)",
                samples > 0,
                f"offset={offset*1000:.1f}ms rtt={rtt*1000:.1f}ms"
            )
            results.record(
                "Clock offset reasonable (<100ms on loopback)",
                abs(offset) < 0.1,
                f"{offset*1000:.1f}ms"
            )
        else:
            results.record("Clock sync active", False, "no peers in table")

        # ── Test 6: Graceful Disconnect ───────────────────────
        print("\nPhase 6: Disconnect")

        bob.session.leave_session()
        time.sleep(2)

        results.record(
            "Alice detects Bob disconnect",
            len(alice.disconnected_peers) > 0,
            f"disconnected: {alice.disconnected_peers}" if alice.disconnected_peers else "no disconnect detected"
        )

        final_peer_count = alice.session.peer_count
        results.record(
            "Alice peer count drops to 0",
            final_peer_count == 0,
            f"peer_count={final_peer_count}"
        )

    except Exception as exc:
        results.record("Unexpected exception", False, str(exc))
        import traceback
        traceback.print_exc()

    finally:
        alice.stop()
        bob.stop()

    return results.summary()


import pytest


@pytest.mark.e2e
def test_p2p_e2e():
    """Run the full E2E test suite as a pytest test."""
    assert run_tests(), "E2E test failed — see output above for details"


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
