"""get_transcriber() centralizes the model_key -> backend selection.

The same QWEN3 / faster-whisper / MLX-whisper if/elif chain was hand-written in
three places (tui/app.py x2, dictation/app.py), risking drift when a backend is
added. This asserts the factory dispatches each model key to the right backend.
Construction is cheap (no model download — that happens in .load()).
"""

from __future__ import annotations

import pytest

from audio.transcriber import (
    FasterWhisperTranscriber,
    Qwen3Transcriber,
    get_transcriber,
)
from config import FASTER_WHISPER_MODELS, QWEN3_MODELS


def test_factory_dispatches_by_model_set():
    # Exercise whatever backends are populated on this platform.
    assert QWEN3_MODELS or FASTER_WHISPER_MODELS, "no transcriber backends configured"
    for key in QWEN3_MODELS:
        assert isinstance(get_transcriber(key), Qwen3Transcriber)
    for key in FASTER_WHISPER_MODELS:
        assert isinstance(get_transcriber(key), FasterWhisperTranscriber)


def test_factory_passes_language_through():
    key = next(iter(FASTER_WHISPER_MODELS or QWEN3_MODELS))
    t = get_transcriber(key, language="ja")
    assert t._language == "ja"


def test_factory_unknown_model_raises_keyerror():
    # Same failure mode as the inline AVAILABLE_MODELS[...] lookups it replaced.
    with pytest.raises(KeyError):
        get_transcriber("definitely-not-a-real-model")
