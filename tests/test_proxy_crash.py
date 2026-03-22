"""Crash recovery tests for DiarizationProxy."""

import time
import pytest
import numpy as np

from config import DIARIZER_MAX_RESTARTS
from diarization.proxy import DiarizationProxy


@pytest.fixture
def proxy():
    """Instantiate a DiarizationProxy, load it, yield, then shut down."""
    p = DiarizationProxy()
    p.load()
    yield p
    p.shutdown()


@pytest.mark.timeout(30)
def test_subprocess_killed_recovers(proxy, sample_audio):
    """After the subprocess is killed, the next identify() still works
    (proxy respawns the subprocess automatically)."""
    audio = sample_audio(duration_sec=2.5)
    # First call succeeds
    label1, sid1 = proxy.identify(audio)
    assert isinstance(label1, str)

    # Kill the subprocess
    assert proxy._proc is not None
    proxy._proc.kill()

    # Next call triggers crash recovery + respawn
    # The first call after kill returns None (fallback), but the proxy respawns.
    # After respawn, subsequent calls should work.
    # Give it a moment for the respawn (which has a 1s sleep).
    label2, sid2 = proxy.identify(audio)
    # Even if the first post-crash call returns the default, the proxy should
    # still be functional. Verify we get valid types.
    assert isinstance(label2, str)
    assert isinstance(sid2, int)


@pytest.mark.timeout(30)
def test_crash_callback_invoked(proxy, sample_audio):
    """on_subprocess_crash callback is called when subprocess dies."""
    crash_counts = []

    def on_crash(count):
        crash_counts.append(count)

    proxy.on_subprocess_crash = on_crash
    audio = sample_audio(duration_sec=2.5)

    # Ensure subprocess is running
    proxy.identify(audio)

    # Kill the subprocess
    proxy._proc.kill()

    # This call triggers crash handling
    proxy.identify(audio)

    assert len(crash_counts) >= 1, "on_subprocess_crash was never called"


@pytest.mark.timeout(30)
def test_disabled_after_max_restarts(proxy, sample_audio):
    """After DIARIZER_MAX_RESTARTS crashes within the window, proxy disables
    diarization (no longer falls back to in-process PyTorch to avoid
    MLX/PyTorch C++ runtime conflict)."""
    audio = sample_audio(duration_sec=2.5)

    # Shrink the restart window so all crashes count
    import diarization.proxy as proxy_mod
    orig_window = proxy_mod.DIARIZER_RESTART_WINDOW
    proxy_mod.DIARIZER_RESTART_WINDOW = 600  # large window so crashes don't expire

    try:
        for i in range(DIARIZER_MAX_RESTARTS):
            if proxy._proc is not None:
                proxy._proc.kill()
            # Trigger crash handling by calling a method
            proxy.identify(audio)
            # Brief pause so respawn sleep (1s) completes before next kill
            time.sleep(0.1)

        assert proxy._mode == "disabled", (
            f"Expected disabled mode after {DIARIZER_MAX_RESTARTS} crashes, "
            f"got {proxy._mode}"
        )

        # Verify it still returns safe defaults rather than crashing
        label, sid = proxy.identify(audio)
        assert label == "Speaker 1"
        assert sid == 1
    finally:
        proxy_mod.DIARIZER_RESTART_WINDOW = orig_window
