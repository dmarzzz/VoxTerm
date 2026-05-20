"""Ollama summarizer backend — factory dispatch, parsing, HTTP behavior.

Network is mocked so these run in CI without a live Ollama server.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import summarizer.engine as summarizer_engine
from summarizer.engine import (
    OllamaSummarizer,
    SummarizerError,
    get_summarizer,
)
from summarizer.prompts import resolve_template


def test_factory_dispatches_ollama_prefix():
    s = get_summarizer("ollama:qwen3:0.6b")
    assert isinstance(s, OllamaSummarizer)


def test_factory_rejects_default_mlx_on_intel_macos(monkeypatch):
    monkeypatch.setattr(summarizer_engine.sys, "platform", "darwin")
    monkeypatch.setattr(summarizer_engine.platform, "machine", lambda: "x86_64")
    summarizer_engine._cache.clear()

    with pytest.raises(SummarizerError, match="Apple Silicon macOS"):
        get_summarizer("")


def test_factory_caches_by_key():
    a = get_summarizer("ollama:llama3.2")
    b = get_summarizer("ollama:llama3.2")
    assert a is b


def test_model_and_host_parsing_default():
    s = OllamaSummarizer("qwen3:0.6b")
    # model keeps its own colon; host falls back to localhost
    assert s.model_name == "ollama:qwen3:0.6b@http://localhost:11434"


def test_explicit_host_after_at():
    s = OllamaSummarizer("llama3.2@http://192.168.1.5:11434")
    assert s.model_name == "ollama:llama3.2@http://192.168.1.5:11434"


def test_bare_host_from_env_gets_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.2:11434")
    s = OllamaSummarizer("mistral")
    assert s.model_name == "ollama:mistral@http://10.0.0.2:11434"


def test_empty_model_rejected():
    with pytest.raises(SummarizerError):
        OllamaSummarizer("")


def test_summarize_posts_chat_and_returns_content():
    captured = {}

    def fake_post(self, path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "  - a point\n"}}

    s = OllamaSummarizer("qwen3:0.6b")
    with patch.object(OllamaSummarizer, "_post", fake_post):
        out = s.summarize("transcript text", resolve_template("key_points"))

    assert out == "- a point"
    assert captured["path"] == "/api/chat"
    pl = captured["payload"]
    assert pl["model"] == "qwen3:0.6b"
    assert pl["stream"] is False
    assert pl["messages"][0]["role"] == "system"
    assert "transcript text" in pl["messages"][1]["content"]


def test_empty_response_raises():
    s = OllamaSummarizer("qwen3:0.6b")
    with patch.object(
        OllamaSummarizer, "_post", lambda self, p, d: {"message": {"content": ""}}
    ):
        with pytest.raises(SummarizerError, match="empty summary"):
            s.summarize("x", resolve_template("tldr"))


def test_unreachable_host_raises_clean_error():
    # port 9 (discard) refuses connections fast
    s = OllamaSummarizer("qwen3:0.6b@http://localhost:9")
    with pytest.raises(SummarizerError, match="Cannot reach Ollama"):
        s.summarize("x", resolve_template("tldr"))
