"""Tests for the hallucination filter in transcriber/engine.py."""

import pytest

from transcriber.engine import _is_hallucination, _DeduplicatorMixin


class TestIsHallucination:

    def test_empty_text(self):
        assert _is_hallucination("") is False

    def test_single_char(self):
        assert _is_hallucination("a") is True

    def test_thanks_for_watching(self):
        assert _is_hallucination("Thanks for watching") is True

    def test_music_bracket(self):
        assert _is_hallucination("[music]") is True

    def test_non_latin_english(self):
        # Chinese characters with expected language "en" should be hallucination
        assert _is_hallucination("\u4f60\u597d\u4e16\u754c", expected_language="en") is True

    def test_chinese_with_chinese_lang(self):
        # Chinese characters with expected language "zh" should NOT be hallucination
        assert _is_hallucination("\u4f60\u597d\u4e16\u754c", expected_language="zh") is False

    def test_repetition(self):
        text = " ".join(["word"] * 8)
        assert _is_hallucination(text) is True

    def test_normal_sentence(self):
        assert _is_hallucination("The weather is nice today") is False


class TestDeduplicator:

    def test_deduplicator(self):
        dedup = _DeduplicatorMixin()
        dedup._init_dedup()
        text = "Hello world"
        assert dedup._is_duplicate(text) is False
        assert dedup._is_duplicate(text) is True
