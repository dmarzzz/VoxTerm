"""Party host/join flow after removing the broadcast code.

Guards the security split: the session code (key material, shared out-of-band)
and the party_id (non-secret, broadcast for grouping) are independent — the code
is never derivable from anything advertised. Discovery network I/O is stubbed.
"""

from __future__ import annotations

import pytest

from network.discovery import PeerDiscovery
from network.party import PartyManager, PartyState


class _FakeWorkers:
    def cancel_group(self, *a, **k):
        pass


class _FakeApp:
    """Stand-in app: _party_* worker hooks become no-ops, so host_party()/
    join_party() run their synchronous logic without starting a real session."""

    def __init__(self):
        self.workers = _FakeWorkers()

    def call_from_thread(self, fn, *a, **k):
        pass

    def set_timer(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeConfig:
    def __init__(self):
        self._d = {"p2p_display_name": "tester"}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, value):
        self._d[key] = value


def _mgr():
    return PartyManager(_FakeApp(), _FakeConfig())


def test_host_party_returns_shareable_code_and_independent_party_id():
    p = _mgr()
    code = p.host_party()
    assert isinstance(code, str) and len(code.split("-")) == 3  # the out-of-band code
    assert p._party_id and p._party_id != code                  # broadcast id, NOT the code
    # the party_id must not contain or reveal the code
    assert code not in p._party_id


def test_join_party_adopts_given_party_id():
    p = _mgr()
    p.join_party("bacon-horse-galaxy", party_id="pid-from-discovery")
    assert p._party_id == "pid-from-discovery"


def test_discovered_parties_excludes_codes(monkeypatch):
    p = _mgr()

    class _Peer:
        def __init__(self, nid, pid):
            self.node_id = nid
            self.display_name = "n"
            self.group_name = "g"
            self.party_color = "#0fc"
            self.party_id = pid
            self.in_session = True

    class _Disc:
        def get_visible_peers(self):
            return [_Peer("a", "pid-1"), _Peer("b", "pid-1"), _Peer("c", "pid-2")]

    p._discovery = _Disc()
    parties = p.discovered_parties()
    by_id = {g["party_id"]: g for g in parties}
    assert set(by_id) == {"pid-1", "pid-2"}
    assert by_id["pid-1"]["peer_count"] == 2
    # the picker payload must never carry a session code
    for g in parties:
        assert "session_code" not in g and "code" not in g


class _PI:
    def __init__(self, node_id, party_id, in_session=True):
        self.node_id = node_id
        self.party_id = party_id
        self.in_session = in_session


def test_same_party_gate():
    # The mesh dial gate: only peers sharing our non-secret party_id (and not us,
    # and in-session) are dialed. This is what the connect / reconnect-sweep use.
    p = _mgr()
    p._node_id = "me"
    p._party_id = "pid-X"
    assert p._same_party(_PI("other", "pid-X")) is True
    assert p._same_party(_PI("other", "pid-Y")) is False          # different party
    assert p._same_party(_PI("me", "pid-X")) is False             # self
    assert p._same_party(_PI("other", "pid-X", in_session=False)) is False
    p._party_id = ""
    assert p._same_party(_PI("other", "")) is False               # no party -> never match


def test_join_with_empty_party_id_stays_solo_not_isolated():
    # Regression for the review's major finding: a join with no discovered
    # party_id must NOT mint a phantom id and sit silently isolated — it must
    # surface a failure and stay solo.
    p = _mgr()
    failures = []
    p.on_party_failed = lambda e: failures.append(e)
    p.join_party("bacon-horse-galaxy", party_id="")
    assert p._state == PartyState.SOLO
    assert p._session_mgr is None
    assert failures, "empty party_id must surface a message, not isolate silently"


def test_host_color_derives_from_party_id_not_code():
    # Security: the broadcast party_color must not be derived from the secret
    # code (that would leak bits of it). It comes from the non-secret party_id.
    from network.party import _party_color

    p = _mgr()
    p.host_party()
    assert p._color_pri == _party_color(p._party_id)[0]
