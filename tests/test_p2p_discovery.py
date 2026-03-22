"""Tests for P2P mDNS discovery — PeerInfo parsing and data classes."""

import pytest

from network.discovery import PeerInfo


class TestPeerInfo:
    def test_creation(self):
        p = PeerInfo(
            node_id="abc123",
            display_name="halcyon",
            ip="192.168.1.42",
            tcp_port=9900,
            udp_port=9901,
            in_session=False,
        )
        assert p.node_id == "abc123"
        assert p.display_name == "halcyon"
        assert p.ip == "192.168.1.42"
        assert p.tcp_port == 9900
        assert p.udp_port == 9901
        assert not p.in_session
        assert p.proto_v == 1

    def test_in_session_flag(self):
        p = PeerInfo("n", "alice", "1.2.3.4", 9900, 9901, in_session=True)
        assert p.in_session
