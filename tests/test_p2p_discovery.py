"""Tests for P2P mDNS discovery — PeerInfo parsing and data classes."""

import pytest

from network.discovery import PeerDiscovery, PeerInfo


def _props_as_str(props) -> dict:
    out = {}
    for k, v in props.items():
        key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
        val = v.decode() if isinstance(v, (bytes, bytearray)) else ("" if v is None else str(v))
        out[key] = val
    return out


class TestMdnsNeverBroadcastsSessionCode:
    """Security: the session code derives the AES party key, so it must NEVER
    appear in the mDNS TXT record. Only the non-secret party_id is advertised."""

    def test_txt_record_omits_session_code(self):
        disc = PeerDiscovery("node-1", "alice", tcp_port=9900, udp_port=9901)
        disc.update_group("myparty", in_session=True, party_id="pid-abc123", party_color="#0fc")
        props = _props_as_str(disc._build_service_info().properties)
        # no key named session_code (or anything carrying the code seed)
        assert "session_code" not in props
        # the non-secret group id IS advertised (peers need it to group/reconnect)
        assert props.get("party_id") == "pid-abc123"
        assert props.get("in_session") == "1"

    def test_update_group_takes_party_id_not_code(self):
        # The discovery API must not even accept a session_code to broadcast.
        import inspect
        sig = inspect.signature(PeerDiscovery.update_group)
        assert "party_id" in sig.parameters
        assert "session_code" not in sig.parameters

    def test_party_id_round_trips_without_a_code_field(self):
        disc = PeerDiscovery("node-2", "bob", tcp_port=9901, udp_port=9902)
        disc.update_group("g", in_session=True, party_id="pid-xyz", party_color="#abc")
        peer = PeerDiscovery._parse_service_info(disc._build_service_info())
        assert peer is not None
        assert peer.party_id == "pid-xyz"
        assert not hasattr(peer, "session_code")


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
