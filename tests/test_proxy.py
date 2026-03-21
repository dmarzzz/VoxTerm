"""Integration tests for DiarizationProxy — spawns a real subprocess."""

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


@pytest.mark.timeout(30)
def test_proxy_loads_and_ready(proxy):
    """After load(), is_loaded must be True."""
    assert proxy.is_loaded is True


@pytest.mark.timeout(30)
def test_identify_returns_label_and_id(proxy, sample_audio):
    """identify() returns a (str, int) tuple."""
    audio = sample_audio(duration_sec=2.5, freq=440.0)
    label, speaker_id = proxy.identify(audio)
    assert isinstance(label, str)
    assert isinstance(speaker_id, int)


@pytest.mark.timeout(30)
def test_get_all_session_speakers_after_identify(proxy, sample_audio):
    """After at least one identify(), get_all_session_speakers() is non-empty."""
    audio = sample_audio(duration_sec=2.5)
    proxy.identify(audio)
    speakers = proxy.get_all_session_speakers()
    assert isinstance(speakers, dict)
    assert len(speakers) > 0


@pytest.mark.timeout(30)
def test_set_and_get_speaker_name(proxy, sample_audio):
    """set_speaker_name / get_speaker_name round-trips correctly."""
    audio = sample_audio(duration_sec=2.5)
    _, speaker_id = proxy.identify(audio)
    proxy.set_speaker_name(speaker_id, "Alice")
    assert proxy.get_speaker_name(speaker_id) == "Alice"


@pytest.mark.timeout(30)
def test_num_speakers(proxy, sample_audio):
    """After identify(), num_speakers is at least 1."""
    audio = sample_audio(duration_sec=2.5)
    proxy.identify(audio)
    assert proxy.num_speakers >= 1


@pytest.mark.timeout(30)
def test_get_speaker_color(proxy, sample_audio):
    """get_speaker_color returns a hex color string."""
    audio = sample_audio(duration_sec=2.5)
    _, speaker_id = proxy.identify(audio)
    color = proxy.get_speaker_color(speaker_id)
    assert isinstance(color, str)
    assert color.startswith("#")
    assert len(color) == 7  # e.g. "#00ffcc"


@pytest.mark.timeout(30)
def test_reset_session(proxy, sample_audio):
    """After reset_session(), num_speakers drops to 0."""
    audio = sample_audio(duration_sec=2.5)
    proxy.identify(audio)
    assert proxy.num_speakers >= 1
    proxy.reset_session()
    assert proxy.num_speakers == 0


@pytest.mark.timeout(30)
def test_is_stable_and_matched(proxy, sample_audio):
    """is_speaker_stable and is_matched return booleans."""
    audio = sample_audio(duration_sec=2.5)
    _, speaker_id = proxy.identify(audio)
    assert isinstance(proxy.is_speaker_stable(speaker_id), bool)
    assert isinstance(proxy.is_matched(speaker_id), bool)
