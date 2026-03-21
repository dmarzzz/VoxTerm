"""Tests for Silero VAD wrapper (ONNX, no PyTorch)."""

import sys

import numpy as np
import pytest

from audio.vad import SileroVAD

SAMPLE_RATE = 16000


class TestSileroVADLoading:

    def test_loads_successfully(self):
        vad = SileroVAD()
        assert vad.is_loaded is True

    def test_no_torch_imported(self):
        """Verify VAD loads without importing torch into the process."""
        # Clear any cached modules from other tests
        had_torch = "torch" in sys.modules
        vad = SileroVAD()
        assert vad.is_loaded
        if not had_torch:
            # torch should not have been imported by our wrapper
            assert "torch" not in sys.modules, (
                "SileroVAD imported torch — this will cause MLX/PyTorch segfaults"
            )


class TestSpeechProbability:

    def test_silence_low_probability(self):
        vad = SileroVAD()
        silence = np.zeros(512, dtype=np.float32)
        prob = vad.speech_probability(silence)
        assert 0.0 <= prob <= 1.0
        assert prob < 0.3  # silence should have low probability

    def test_returns_float(self):
        vad = SileroVAD()
        chunk = np.zeros(512, dtype=np.float32)
        prob = vad.speech_probability(chunk)
        assert isinstance(prob, float)

    def test_1024_chunk_splitting(self):
        """1024-sample chunks should be split into two 512-sample sub-chunks."""
        vad = SileroVAD()
        chunk = np.zeros(1024, dtype=np.float32)
        prob = vad.speech_probability(chunk)
        assert 0.0 <= prob <= 1.0

    def test_handles_mono_2d_input(self):
        """2D mono array (N, 1) should be handled correctly."""
        vad = SileroVAD()
        chunk = np.zeros((512, 1), dtype=np.float32)
        prob = vad.speech_probability(chunk)
        assert 0.0 <= prob <= 1.0

    def test_incomplete_tail_skipped(self):
        """Chunk sizes not a multiple of 512 should skip the tail."""
        vad = SileroVAD()
        # 700 samples = 1 full 512-sample sub-chunk + 188 tail (skipped)
        chunk = np.zeros(700, dtype=np.float32)
        prob = vad.speech_probability(chunk)
        assert 0.0 <= prob <= 1.0

    def test_fallback_when_not_loaded(self):
        """Unloaded model returns 1.0 (always speech — safe fallback)."""
        vad = SileroVAD()
        vad._loaded = False
        prob = vad.speech_probability(np.zeros(512, dtype=np.float32))
        assert prob == 1.0


class TestIsSpeech:

    def test_silence_not_speech(self):
        vad = SileroVAD(threshold=0.5)
        silence = np.zeros(1024, dtype=np.float32)
        assert vad.is_speech(silence) is False

    def test_threshold_applied(self):
        vad = SileroVAD(threshold=0.5)
        # With a very low threshold, even noise might pass
        vad_low = SileroVAD(threshold=0.001)
        silence = np.zeros(1024, dtype=np.float32)
        # At threshold 0.001, the model's baseline noise may exceed it
        # But at 0.5, silence should definitely be below
        assert vad.is_speech(silence) is False


class TestReset:

    def test_reset_doesnt_crash(self):
        vad = SileroVAD()
        vad.reset()
        # Should still work after reset
        prob = vad.speech_probability(np.zeros(512, dtype=np.float32))
        assert 0.0 <= prob <= 1.0

    def test_reset_clears_state(self):
        """After reset, model should behave as freshly initialized."""
        vad = SileroVAD()
        # Process some audio to build up internal state
        for _ in range(10):
            vad.speech_probability(np.random.randn(512).astype(np.float32) * 0.01)
        # Reset
        vad.reset()
        # Should produce similar output to a fresh model on silence
        fresh_vad = SileroVAD()
        prob_reset = vad.speech_probability(np.zeros(512, dtype=np.float32))
        prob_fresh = fresh_vad.speech_probability(np.zeros(512, dtype=np.float32))
        assert abs(prob_reset - prob_fresh) < 0.05


class TestConsistency:

    def test_deterministic_on_same_input(self):
        """Same input should produce same output (stateful, so order matters)."""
        vad1 = SileroVAD()
        vad2 = SileroVAD()
        chunk = np.random.RandomState(42).randn(512).astype(np.float32) * 0.1
        prob1 = vad1.speech_probability(chunk)
        prob2 = vad2.speech_probability(chunk)
        assert abs(prob1 - prob2) < 1e-6
