"""Regression test for the P2P mesh-heal fix.

PartyManager wired only `on_peer_found` (fired on NEW mDNS services). When a known
peer re-advertised as an *Updated* event — it flips `in_session` on joining the
party, or changes port/group — discovery fired `on_peer_updated`, which PartyManager
never assigned, so the event was silently dropped and a 3+ peer mesh kept a permanent
hole. This test asserts both callbacks run the same connect logic.

Discovery is stubbed to a no-op so the test exercises only the callback wiring — no
real Zeroconf/mDNS multicast (deterministic, fast, and safe in CI/sandboxes without
working multicast).
"""

from __future__ import annotations

import pytest

from network.party import PartyManager
from network.discovery import PeerDiscovery


class _FakeWorkers:
    def cancel_group(self, *a, **k):  # App.workers.cancel_group(app, group)
        pass


class _FakeApp:
    """Minimal stand-in: PartyManager only needs workers.cancel_group + call_from_thread.

    call_from_thread is a no-op here — the callback wiring under test happens
    synchronously in start_session_blocking, not via the UI thread.
    """

    def __init__(self):
        self.workers = _FakeWorkers()

    def call_from_thread(self, fn, *a, **k):
        pass

    def __getattr__(self, name):  # any other _party_* hook -> no-op
        return lambda *a, **k: None


class _FakeConfig:
    def __init__(self):
        self._d = {"p2p_display_name": "tester"}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, value):
        self._d[key] = value


@pytest.mark.timeout(15)
def test_party_wires_on_peer_updated_for_mesh_heal(monkeypatch):
    # Stub discovery network I/O so no real multicast is needed — the wiring under
    # test (callback assignment) happens regardless of whether mDNS actually starts.
    monkeypatch.setattr(PeerDiscovery, "start", lambda self: None)
    monkeypatch.setattr(PeerDiscovery, "update_group", lambda self, *a, **k: None)
    monkeypatch.setattr(PeerDiscovery, "update_port", lambda self, *a, **k: None)

    party = PartyManager(_FakeApp(), _FakeConfig())
    try:
        party.start_session_blocking("test-bacon-horse-galaxy", is_creator=True)
        disc = party._discovery
        assert disc is not None, "discovery was not started"
        assert disc.on_peer_found is not None
        # The fix: Updated mDNS events (in_session flip / port / group change) must run
        # the SAME gated connect logic, not be dropped on the floor.
        assert disc.on_peer_updated is not None, (
            "on_peer_updated is unwired — re-advertising peers leave a permanent mesh hole"
        )
        assert disc.on_peer_updated is disc.on_peer_found
    finally:
        try:
            if party._session_mgr is not None:
                party._session_mgr.leave_session()
        except Exception:
            pass
        party._stop_discovery()
