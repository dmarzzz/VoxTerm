"""Diarizer respawn must not hold the IPC lock during the model-load wait.

_call() detects a subprocess crash and respawns OUTSIDE _lock so the UI thread
isn't frozen. The bug: _handle_crash re-acquired _lock around the spawn, whose
READY wait (recv_msg, up to DIARIZER_STARTUP_TIMEOUT ~30s) blocks while the model
loads — so a concurrent _call() stalled for the whole load.

The test stubs the spawn at the Popen + recv_msg seam (which exists regardless of
how _spawn is factored), so it is a true regression guard: it FAILS against the old
lock-around-spawn structure and PASSES once the wait is moved off the lock.

Fallback-subprocess-mode only; the default ONNX 'direct' mode never spawns.
"""

from __future__ import annotations

import threading

import pytest

import audio.diarization.proxy as proxy_mod
from audio.diarization.proxy import DiarizationProxy


class _FakeProc:
    stdout = object()
    stderr = object()
    stdin = object()

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.mark.timeout(15)
def test_respawn_does_not_hold_lock_during_model_load(monkeypatch):
    # mode="subprocess" sets up the subprocess attrs; load() is NOT called, so no
    # real worker spawns. The spawn is faked at the Popen + recv_msg boundary.
    proxy = DiarizationProxy(mode="subprocess")
    proxy.on_subprocess_crash = None
    proxy.on_subprocess_ready = None

    recv_blocking = threading.Event()
    release = threading.Event()

    def fake_recv_msg(stream, timeout=None):
        # Stand in for the slow model-load READY wait.
        recv_blocking.set()
        assert release.wait(10), "release was never signaled"
        return {"type": proxy_mod.MSG_READY}

    monkeypatch.setattr(proxy_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(proxy_mod, "recv_msg", fake_recv_msg)

    t = threading.Thread(target=proxy._handle_crash, daemon=True)
    t.start()
    assert recv_blocking.wait(5), "respawn never reached the READY wait"

    # THE GUARD: while the model-load READY wait blocks, the IPC lock must be FREE.
    # The old code held _lock around the whole spawn, so this acquire would fail —
    # i.e. this assertion is what turns red on a regression to that structure.
    acquired = proxy._lock.acquire(blocking=False)
    assert acquired, "_lock held during model-load wait — a concurrent _call() would freeze"
    proxy._lock.release()

    release.set()
    t.join(timeout=5)
    assert not t.is_alive(), "_handle_crash did not finish"
    assert proxy._proc is not None, "respawned proc was not installed under the lock"
