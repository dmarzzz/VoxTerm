"""Regression tests for the re-diarization app wiring: the generation guard
that drops stale relabels, and reset bumping the generation. Exercises the real
VoxTerm methods on a bare instance (no Textual app/event loop)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tui.app import VoxTerm


class _FakeTP:
    def __init__(self):
        self.relabel_calls = []
        self.messages = []

    def apply_relabels(self, updates):
        self.relabel_calls.append(updates)
        return len(updates)

    def system_message(self, *a, **k):
        self.messages.append(a)


def _bare_app():
    app = VoxTerm.__new__(VoxTerm)
    app._session_audio_chunks = []
    app._session_audio_len = 0
    app._session_audio_lock = threading.Lock()
    app._rediar_entries = []
    app._last_rediar_len = 0
    app._rediar_frozen = False
    app._rediar_gen = 0
    app._debug = False
    app._tp = _FakeTP()
    app.query_one = lambda *a, **k: app._tp
    app._update_telemetry = lambda: None
    return app


def test_reset_clears_state_and_bumps_generation():
    app = _bare_app()
    import numpy as np
    app._session_audio_chunks = [np.zeros(16000, np.float32)]
    app._session_audio_len = 16000
    app._rediar_entries = [{"idx": 0, "t0": 0.0, "t1": 1.0, "sid": 1}]
    app._last_rediar_len = 16000
    app._rediar_frozen = True

    app._reset_rediar_state()

    assert app._session_audio_chunks == []
    assert app._session_audio_len == 0
    assert app._rediar_entries == []
    assert app._last_rediar_len == 0
    assert app._rediar_frozen is False
    assert app._rediar_gen == 1  # bumped


def test_apply_drops_stale_relabels_after_reset():
    app = _bare_app()
    app._rediar_entries = [{"idx": 0, "t0": 0.0, "t1": 1.0, "sid": 1}]
    # a correction snapshotted at gen 0...
    updates = {0: (2, "Speaker 2", "#ff44aa")}
    # ...but the session was cleared/started anew (gen bumped) before it landed
    app._reset_rediar_state()  # gen -> 1
    app._apply_rediar_updates(updates, final=False, gen=0)
    assert app._tp.relabel_calls == []  # stale relabels dropped, transcript untouched


def test_apply_applies_fresh_relabels():
    app = _bare_app()
    app._rediar_entries = [{"idx": 0, "t0": 0.0, "t1": 1.0, "sid": 1}]
    updates = {0: (2, "Speaker 2", "#ff44aa")}
    app._apply_rediar_updates(updates, final=True, gen=app._rediar_gen)
    assert app._tp.relabel_calls == [updates]          # applied
    assert app._rediar_entries[0]["sid"] == 2          # corrected id adopted
    assert app._tp.messages                            # final -> system message shown


def test_record_session_audio_freezes_at_cap(monkeypatch):
    import numpy as np
    import config
    app = _bare_app()
    monkeypatch.setattr(config, "REDIAR_MAX_SESSION_SEC", 1.0)  # 1s cap
    # import-site constant in app.py is read at call time via config? It is bound
    # at import; assert behavior against the bound value instead.
    from tui import app as appmod
    cap_samples = int(appmod.REDIAR_MAX_SESSION_SEC * appmod.SAMPLE_RATE)
    base = app._record_session_audio(np.zeros(cap_samples + 10, np.float32))
    assert base == 0
    assert app._rediar_frozen is True
    # once frozen, further appends return None (no new spans, alignment preserved)
    assert app._record_session_audio(np.zeros(100, np.float32)) is None
