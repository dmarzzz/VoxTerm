"""Redaction engine — pure span logic + backend factory + Ollama flow.

The LLM is never invoked: span parsing, regex detection, and masking are
pure functions, and the Ollama path is mocked. Runs in CI with no model.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import redaction.engine as redaction_engine
from redaction.engine import (
    OllamaRedactor,
    RedactionError,
    apply_redactions,
    chunk_text,
    get_redactor,
    parse_spans,
    regex_spans,
)
from redaction.prompts import resolve_profile


# --- factory -------------------------------------------------------------

def test_factory_dispatches_ollama_prefix():
    r = get_redactor("ollama:qwen3:0.6b")
    assert isinstance(r, OllamaRedactor)


def test_factory_rejects_default_mlx_on_intel_macos(monkeypatch):
    monkeypatch.setattr(redaction_engine.sys, "platform", "darwin")
    monkeypatch.setattr(redaction_engine.platform, "machine", lambda: "x86_64")
    redaction_engine._cache.clear()
    with pytest.raises(RedactionError, match="Apple Silicon macOS"):
        get_redactor("")


def test_factory_caches_by_key():
    a = get_redactor("ollama:llama3.2")
    b = get_redactor("ollama:llama3.2")
    assert a is b


def test_empty_model_rejected():
    with pytest.raises(RedactionError):
        OllamaRedactor("")


# --- parse_spans ---------------------------------------------------------

def test_parse_spans_plain_array():
    out = '[{"text": "Alice", "type": "NAME"}, {"text": "bob@x.com", "type": "EMAIL"}]'
    assert parse_spans(out) == [("Alice", "NAME"), ("bob@x.com", "EMAIL")]


def test_parse_spans_tolerates_preamble_and_fences():
    out = 'Sure! Here you go:\n```json\n[{"text": "Acme", "type": "ORG"}]\n```'
    assert parse_spans(out) == [("Acme", "ORG")]


def test_parse_spans_coerces_unknown_type_to_other():
    assert parse_spans('[{"text": "x", "type": "BANANA"}]') == [("x", "OTHER")]


def test_parse_spans_skips_malformed_items():
    out = '[{"text": "Al", "type": "NAME"}, {"nope": 1}, {"text": "", "type": "NAME"}, 42]'
    assert parse_spans(out) == [("Al", "NAME")]


def test_parse_spans_garbage_returns_empty():
    assert parse_spans("the model refused to answer") == []
    assert parse_spans("") == []
    assert parse_spans("[not valid json}") == []


# --- regex_spans ---------------------------------------------------------

def test_regex_catches_structured_pii():
    text = (
        "mail me at jane.doe@example.com or call 415-555-0199, "
        "ssn 123-45-6789, see https://secret.example.com/x, host 10.0.0.2"
    )
    found = dict((t, ty) for t, ty in regex_spans(text))
    assert found.get("jane.doe@example.com") == "EMAIL"
    assert found.get("123-45-6789") == "ID"
    assert "https://secret.example.com/x" in found
    assert any(t == "10.0.0.2" for t, _ in regex_spans(text))
    assert any(ty == "PHONE" for _, ty in regex_spans(text))


def test_regex_ignores_plain_text_and_timestamps():
    # Colons separate timestamp fields, so a clock time isn't a phone run.
    found = regex_spans("the meeting at 12:34:56 went well")
    assert found == []


# --- apply_redactions ----------------------------------------------------

def test_apply_masks_verbatim_and_counts():
    text = "Alice met Alice and Bob."
    res = apply_redactions(text, [("Alice", "NAME"), ("Bob", "NAME")])
    assert res.redacted_text == "[NAME] met [NAME] and [NAME]."
    assert res.counts["NAME"] == 3
    assert res.total == 3


def test_apply_longest_first_prevents_partial_clobber():
    text = "Alice Smith spoke; Alice nodded."
    res = apply_redactions(text, [("Alice", "NAME"), ("Alice Smith", "NAME")])
    # "Alice Smith" masked as one unit first, then the lone "Alice".
    assert res.redacted_text == "[NAME] spoke; [NAME] nodded."
    assert res.counts["NAME"] == 2


def test_apply_skips_spans_not_present():
    text = "nothing sensitive here"
    res = apply_redactions(text, [("Hallucinated Name", "NAME")])
    assert res.redacted_text == text
    assert res.total == 0


def test_apply_drops_too_short_spans():
    text = "a b c"
    res = apply_redactions(text, [("a", "NAME"), ("", "NAME"), (" ", "OTHER")])
    assert res.redacted_text == text
    assert res.total == 0


# --- chunk_text ----------------------------------------------------------

def test_chunk_covers_all_lines_without_loss():
    text = "".join(f"line {i}\n" for i in range(200))
    chunks = chunk_text(text, 100)
    assert len(chunks) > 1
    assert "".join(chunks) == text  # nothing dropped, nothing duplicated


def test_chunk_hard_splits_one_huge_line():
    text = "x" * 250
    chunks = chunk_text(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_chunk_empty_text():
    assert chunk_text("", 100) == []


# --- end-to-end through a mocked Ollama backend --------------------------

def test_ollama_redact_identifies_then_masks():
    captured = {}

    def fake_post(self, path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": '[{"text": "Alice", "type": "NAME"}]'}}

    r = OllamaRedactor("qwen3:0.6b")
    with patch.object(OllamaRedactor, "_post", fake_post):
        res = r.redact(
            "Alice emailed bob@x.com about the deal.",
            resolve_profile("standard"),
        )

    # LLM-found NAME + regex-found EMAIL both masked.
    assert "[NAME]" in res.redacted_text
    assert "[EMAIL]" in res.redacted_text
    assert "Alice" not in res.redacted_text
    assert "bob@x.com" not in res.redacted_text
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["model"] == "qwen3:0.6b"


def test_ollama_redact_empty_model_reply_is_not_an_error():
    # An empty/[] reply means "no PII found" — valid, not a failure.
    r = OllamaRedactor("qwen3:0.6b")
    with patch.object(OllamaRedactor, "_post", lambda self, p, d: {"message": {"content": "[]"}}):
        res = r.redact("nothing sensitive at all", resolve_profile("standard"))
    assert res.total == 0
    assert res.redacted_text == "nothing sensitive at all"


def test_unreachable_host_raises_clean_error():
    r = OllamaRedactor("qwen3:0.6b@http://localhost:9")
    with pytest.raises(RedactionError, match="Cannot reach Ollama"):
        r.redact("x", resolve_profile("standard"))
