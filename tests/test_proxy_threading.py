"""Threading stress tests for DiarizationProxy."""

import threading
import pytest
import numpy as np

from diarization.proxy import DiarizationProxy


@pytest.fixture
def proxy():
    """Instantiate a DiarizationProxy, load it, yield, then shut down."""
    p = DiarizationProxy()
    p.load()
    yield p
    p.shutdown()


@pytest.mark.timeout(15)
def test_concurrent_identify_and_query(proxy, sample_audio):
    """Two threads — one calling identify(), the other calling
    get_all_session_speakers() — should not deadlock or raise."""
    audio = sample_audio(duration_sec=2.5)
    errors = []

    def identify_loop():
        try:
            for _ in range(5):
                label, sid = proxy.identify(audio)
                assert isinstance(label, str)
                assert isinstance(sid, int)
        except Exception as exc:
            errors.append(exc)

    def query_loop():
        try:
            for _ in range(5):
                speakers = proxy.get_all_session_speakers()
                assert isinstance(speakers, dict)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=identify_loop)
    t2 = threading.Thread(target=query_loop)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive(), "identify thread deadlocked"
    assert not t2.is_alive(), "query thread deadlocked"
    assert errors == [], f"Threads raised exceptions: {errors}"


@pytest.mark.timeout(15)
def test_tag_during_transcription(proxy, sample_audio):
    """Thread A calls identify(audio) while thread B calls
    get_all_session_speakers(). Both should complete without error.
    Thread B must get a dict (not None, not an exception)."""
    audio = sample_audio(duration_sec=2.5)
    errors = []
    results_b = []

    def thread_a():
        try:
            proxy.identify(audio)
        except Exception as exc:
            errors.append(exc)

    def thread_b():
        try:
            result = proxy.get_all_session_speakers()
            results_b.append(result)
        except Exception as exc:
            errors.append(exc)

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not ta.is_alive(), "thread A deadlocked"
    assert not tb.is_alive(), "thread B deadlocked"
    assert errors == [], f"Threads raised exceptions: {errors}"
    assert len(results_b) == 1, "thread B did not produce a result"
    assert isinstance(results_b[0], dict), (
        f"Expected dict from get_all_session_speakers, got {type(results_b[0])}"
    )
