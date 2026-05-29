"""OpenAI Privacy Filter backend — label mapping, span extraction, dispatch.

The model is never loaded: the token classifier (`_classify`) is mocked, so
these run in CI without onnxruntime weights or optimum/transformers.
"""

from __future__ import annotations

from unittest.mock import patch

import redaction.engine as redaction_engine
from redaction.engine import get_redactor
from redaction.privacy_filter import (
    PrivacyFilterRedactor,
    map_label,
    spans_from_entities,
)


# --- label mapping -------------------------------------------------------

def test_map_label_known_and_bio_prefixes():
    assert map_label("private_person") == "NAME"
    assert map_label("B-private_email") == "EMAIL"
    assert map_label("I-secret") == "CREDENTIAL"
    assert map_label("account_number") == "ID"


def test_map_label_unknown_is_none():
    assert map_label("MISC") is None
    assert map_label("B-organization") is None
    assert map_label("") is None


# --- span extraction (pure) ---------------------------------------------

def test_spans_use_offsets_into_chunk():
    chunk = "Alice emailed bob@x.com"
    ents = [
        {"entity_group": "private_person", "score": 0.99, "start": 0, "end": 5},
        {"entity_group": "private_email", "score": 0.97, "start": 14, "end": 23},
    ]
    assert spans_from_entities(chunk, ents, 0.5) == [
        ("Alice", "NAME"),
        ("bob@x.com", "EMAIL"),
    ]


def test_spans_threshold_filters_low_confidence():
    chunk = "call Bob"
    ents = [{"entity_group": "private_person", "score": 0.30, "start": 5, "end": 8}]
    assert spans_from_entities(chunk, ents, 0.5) == []


def test_spans_skip_unknown_labels_and_fallback_to_word():
    chunk = "irrelevant"
    ents = [
        {"entity_group": "MISC", "score": 0.99, "start": 0, "end": 3},
        {"label": "secret", "score": 0.9, "word": "hunter2"},  # no offsets
    ]
    assert spans_from_entities(chunk, ents, 0.5) == [("hunter2", "CREDENTIAL")]


# --- detect / redact via a mocked classifier ----------------------------

def test_detect_maps_labels_and_adds_regex_backstop():
    r = PrivacyFilterRedactor()
    text = "Alice — SSN 123-45-6789"
    ents = [{"entity_group": "private_person", "score": 0.99, "start": 0, "end": 5}]
    with patch.object(PrivacyFilterRedactor, "_classify", lambda self, chunk: ents):
        spans = dict(r.detect(text))
    assert spans.get("Alice") == "NAME"          # from the model
    assert spans.get("123-45-6789") == "ID"      # from the regex backstop


def test_redact_masks_detected_spans():
    r = PrivacyFilterRedactor()
    text = "Alice met at the cafe."
    ents = [{"entity_group": "private_person", "score": 0.99, "start": 0, "end": 5}]
    with patch.object(PrivacyFilterRedactor, "_classify", lambda self, chunk: ents):
        res = r.redact(text)
    assert res.redacted_text == "[NAME] met at the cafe."
    assert res.counts.get("NAME") == 1


# --- factory dispatch ----------------------------------------------------

def test_factory_dispatches_privacy_filter():
    redaction_engine._cache.clear()
    assert isinstance(get_redactor("privacy-filter"), PrivacyFilterRedactor)


def test_factory_privacy_filter_custom_repo():
    redaction_engine._cache.clear()
    r = get_redactor("privacy-filter:some/fork")
    assert isinstance(r, PrivacyFilterRedactor)
    assert r.model_name == "privacy-filter:some/fork"
