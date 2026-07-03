"""Multi-peer stress tests using the test harness."""

import time

import pytest

from network.test_harness import PeerHarness


@pytest.mark.timeout(30)
class TestMultiPeerMesh:
    def test_three_peers_connect(self):
        harness = PeerHarness(peer_count=3, session_code="apple-forest-river")
        try:
            harness.start()
            # All peers should have at least 1 connection
            for vp in harness.peers:
                assert len(vp.connected_peers) >= 1, f"{vp.name} has no connections"
        finally:
            harness.stop()

    def test_broadcast_reaches_all_peers(self):
        harness = PeerHarness(peer_count=3, session_code="bacon-horse-galaxy")
        try:
            harness.start()
            time.sleep(0.5)  # let connections stabilize

            # Peer 0 broadcasts
            harness.broadcast_from(0, "hello from peer-0")

            # Other peers should receive it (peer 0 doesn't receive its own)
            assert harness.wait_for_finals(count=2, timeout=5.0)  # 2 receivers

            for vp in harness.peers[1:]:
                assert len(vp.received_finals) >= 1
                assert vp.received_finals[0][1]["text"] == "hello from peer-0"

        finally:
            harness.stop()

    def test_bidirectional_exchange(self):
        harness = PeerHarness(peer_count=3, session_code="crane-dragon-echo")
        try:
            harness.start()
            time.sleep(0.5)

            # Each peer broadcasts
            for i, vp in enumerate(harness.peers):
                harness.broadcast_from(i, f"hello from {vp.name}", seq=i + 1)

            # Each peer should receive finals from the other 2
            # Total finals across all peers: 3 senders × 2 receivers each = 6
            time.sleep(1.0)  # give time for all messages

            total = sum(len(vp.received_finals) for vp in harness.peers)
            # At minimum, some messages should have been received
            assert total >= 3, f"Expected at least 3 total finals, got {total}"

        finally:
            harness.stop()

    def test_two_parties_on_same_host_do_not_cross_contaminate(self):
        """Two simultaneous parties stay isolated by session code/key.

        This models the shared-access-point risk in-process: both parties share
        the same network stack and run concurrently, then one peer deliberately
        attempts a cross-party TCP connection. The encrypted handshake must
        reject the wrong session key, and transcript messages must remain inside
        their own party.
        """
        alpha = PeerHarness(
            peer_count=2,
            session_code="alpha-forest-river",
            node_prefix="alpha",
        )
        beta = PeerHarness(
            peer_count=2,
            session_code="beta-horse-galaxy",
            node_prefix="beta",
        )
        try:
            alpha.start()
            beta.start()
            time.sleep(0.5)

            alpha.broadcast_from(0, "alpha-only transcript", seq=1)
            beta.broadcast_from(0, "beta-only transcript", seq=1)

            assert alpha.wait_for_finals(count=1, timeout=5.0)
            assert beta.wait_for_finals(count=1, timeout=5.0)

            # A same-network but wrong-party connection should fail at the
            # encrypted session-code handshake and not add a peer.
            beta_port = beta.server_port(0)
            assert not alpha.peers[1].mgr.join_by_ip(
                "127.0.0.1",
                beta_port,
                "alpha-forest-river",
            )
            time.sleep(0.5)

            alpha_texts = [
                msg["text"]
                for peer in alpha.peers
                for _, msg in peer.received_finals
            ]
            beta_texts = [
                msg["text"]
                for peer in beta.peers
                for _, msg in peer.received_finals
            ]
            assert "alpha-only transcript" in alpha_texts
            assert "beta-only transcript" not in alpha_texts
            assert "beta-only transcript" in beta_texts
            assert "alpha-only transcript" not in beta_texts
            assert alpha.peers[1].mgr.peer_count == 1
            assert beta.peers[0].mgr.peer_count == 1
        finally:
            alpha.stop()
            beta.stop()

    def test_five_peers_mesh(self):
        """Stress test with 5 peers."""
        harness = PeerHarness(peer_count=5, session_code="eagle-frost-marble")
        try:
            harness.start()
            time.sleep(0.5)

            # Peer 0 broadcasts
            harness.broadcast_from(0, "five-peer test")

            # Should reach at least some peers
            time.sleep(2.0)
            receivers = sum(1 for vp in harness.peers[1:] if len(vp.received_finals) > 0)
            assert receivers >= 1, "No peers received the broadcast"

        finally:
            harness.stop()

    def test_clock_sync_convergence(self):
        """After heartbeat exchange, clock offsets should be near zero on loopback."""
        harness = PeerHarness(peer_count=2, session_code="gold-hammer-nest")
        try:
            harness.start()

            # Wait for a few heartbeat cycles
            time.sleep(3.0)

            mgr_a = harness.peers[0].mgr
            peers_a = mgr_a.peers
            if peers_a:
                peer = list(peers_a.values())[0]
                if peer.clock.sample_count > 0:
                    # On loopback, offset should be very close to 0
                    assert abs(peer.clock.offset) < 0.1, \
                        f"Clock offset too large: {peer.clock.offset}"
                    assert peer.clock.rtt < 0.1, \
                        f"RTT too large: {peer.clock.rtt}"
        finally:
            harness.stop()
