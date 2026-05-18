"""Tests for custom ASR keyword context plumbing."""

import sys
from types import SimpleNamespace

import numpy as np

sys.modules.setdefault("sounddevice", SimpleNamespace())
sys.modules.setdefault("scipy", SimpleNamespace())
sys.modules.setdefault(
    "scipy.signal",
    SimpleNamespace(resample_poly=lambda audio, up, down: audio),
)

from audio.transcriber import (
    Qwen3Transcriber,
    build_keyword_context,
    parse_custom_keywords,
)


def test_parse_custom_keywords_from_text_and_file(tmp_path):
    keyword_file = tmp_path / "keywords.txt"
    keyword_file.write_text("SUAVE\nMEV-Boost\nsuave\n", encoding="utf-8")

    keywords = parse_custom_keywords(
        ["Flashbots, MEV-Boost", "orderflow auction"],
        keyword_file,
    )

    assert keywords == ["Flashbots", "MEV-Boost", "orderflow auction", "SUAVE"]


def test_build_keyword_context():
    context = build_keyword_context(["Flashbots", "MEV-Boost"])

    assert context == "Custom vocabulary: Flashbots; MEV-Boost"


def test_qwen_mlx_transcriber_passes_keyword_context(monkeypatch):
    calls = {}

    def fake_transcribe(audio, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(text="Flashbots")

    monkeypatch.setitem(
        sys.modules,
        "mlx_qwen3_asr",
        SimpleNamespace(transcribe=fake_transcribe),
    )

    transcriber = Qwen3Transcriber(custom_keywords=["Flashbots", "MEV-Boost"])
    transcriber._use_mlx = True
    transcriber._model = object()

    result = transcriber.transcribe(np.ones(16000, dtype=np.float32))

    assert result["text"] == "Flashbots"
    assert calls["context"] == "Custom vocabulary: Flashbots; MEV-Boost"


def test_qwen_pytorch_transcriber_passes_keyword_context():
    calls = {}

    class FakeModel:
        def transcribe(self, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(text="MEV-Boost")]

    transcriber = Qwen3Transcriber(custom_keywords=["Flashbots", "MEV-Boost"])
    transcriber._use_mlx = False
    transcriber._model = FakeModel()

    result = transcriber.transcribe(np.ones(16000, dtype=np.float32))

    assert result["text"] == "MEV-Boost"
    assert calls["context"] == "Custom vocabulary: Flashbots; MEV-Boost"
