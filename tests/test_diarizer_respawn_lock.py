"""Diarizer respawn must not hold the IPC lock during the model-load wait.

_call() deliberately detects a subprocess crash and respawns OUTSIDE _lock so the
UI thread isn't frozen. But _handle_crash previously re-acquired _lock around
_spawn(), whose READY wait blocks up to DIARIZER_STARTUP_TIMEOUT (~30s) for the
model to load — so a concurrent _call() stalled for the whole load. The fix runs
the spawn + READY wait on a local proc (no lock) and locks only the pointer swap.

This is fallback-subprocess-mode only; the default ONNX 'direct' mode never spawns.
"""

from __future__ import annotations

import threading

import pytest

from audio.diarization.proxy import DiarizationProxy


@pytest.mark.timeout(15)
def test_respawn_does_not_hold_lock_during_model_load(monkeypatch):
    # mode="subprocess" so __init__ sets up the subprocess attrs; load() is NOT
    # called, so no real worker is spawned.
    proxy = DiarizationProxy(mode="subprocess")
    proxy.on_subprocess_crash = None
    proxy.on_subprocess_ready = None

    spawn_entered = threading.Event()
    release = threading.Event()
    sentinel = object()

    def fake_spawn_proc():
        spawn_entered.set()
        # Stand in for the slow (up to ~30s) model-load READY wait.
        assert release.wait(10), "release was never signaled"
        return sentinel

    monkeypatch.setattr(proxy, "_spawn_proc", fake_spawn_proc)

    t = threading.Thread(target=proxy._handle_crash, daemon=True)
    t.start()
    assert spawn_entered.wait(5), "respawn never reached _spawn_proc"

    # THE ASSERTION: while the (simulated) model load is in progress, the IPC lock
    # must be FREE — otherwise a concurrent _call() would block for the full timeout.
    acquired = proxy._lock.acquire(blocking=False)
    assert acquired, "_lock held during model-load respawn — a concurrent _call() would freeze"
    proxy._lock.release()

    # Let the spawn finish; the new proc is installed under a brief lock.
    release.set()
    t.join(timeout=5)
    assert not t.is_alive(), "_handle_crash did not finish"
    assert proxy._proc is sentinel, "respawned proc was not installed"
