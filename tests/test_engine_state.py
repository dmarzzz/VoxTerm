"""Tests for DiarizationEngine state management (no model loading)."""

import numpy as np
import pytest

from diarization.engine import DiarizationEngine

EMBEDDING_DIM = 192


class TestSpeakerNames:

    def test_speaker_name_set_get(self, mock_engine):
        mock_engine.set_speaker_name(1, "Alice")
        assert mock_engine.get_speaker_name(1) == "Alice"
        # Unset speaker falls back to default
        assert mock_engine.get_speaker_name(99) == "Speaker 99"

    def test_speaker_names_dict(self, mock_engine):
        mock_engine.set_speaker_name(1, "Alice")
        mock_engine.set_speaker_name(2, "Bob")
        names = mock_engine.get_speaker_names()
        assert names == {1: "Alice", 2: "Bob"}
        # Returned dict should be a copy
        names[3] = "Eve"
        assert 3 not in mock_engine.get_speaker_names()


class TestSpeakerColors:

    def test_speaker_color_cycling(self, mock_engine):
        palette = mock_engine._color_palette
        # Inject speakers and assign colors the same way the engine does
        for i in range(1, len(palette) + 2):
            idx = (i - 1) % len(palette)
            mock_engine._speaker_colors[i] = palette[idx]
        # First speaker gets palette[0], second gets palette[1], etc.
        assert mock_engine.get_speaker_color(1) == palette[0]
        assert mock_engine.get_speaker_color(2) == palette[1]
        # Wraps around
        assert mock_engine.get_speaker_color(len(palette) + 1) == palette[0]


class TestMergeSpeakers:

    def test_merge_speakers(self, mock_engine, random_embedding):
        emb_a = random_embedding(seed=10)
        emb_b = random_embedding(seed=20)

        # Inject two speakers with centroids and embeddings
        mock_engine._speaker_centroids[1] = emb_a.copy()
        mock_engine._speaker_centroids[2] = emb_b.copy()
        mock_engine._segment_embeddings[1] = [(emb_a.copy(), 2.0)]
        mock_engine._segment_embeddings[2] = [(emb_b.copy(), 3.0)]
        mock_engine._speaker_colors[1] = "#aaa"
        mock_engine._speaker_colors[2] = "#bbb"
        mock_engine._speaker_names[1] = "Alice"
        mock_engine._speaker_names[2] = "Bob"

        mock_engine.merge_speakers(source_id=2, target_id=1)

        # Source speaker should be removed
        assert 2 not in mock_engine._speaker_centroids
        assert 2 not in mock_engine._speaker_colors
        assert 2 not in mock_engine._speaker_names

        # Target should have both embeddings
        assert len(mock_engine._segment_embeddings[1]) == 2

        # Merged centroid is average then normalized
        merged = (emb_a + emb_b) / 2.0
        merged /= np.linalg.norm(merged)
        assert np.allclose(mock_engine._speaker_centroids[1], merged, atol=1e-5)


class TestResetSession:

    def test_reset_session(self, mock_engine, random_embedding):
        emb = random_embedding(seed=1)
        mock_engine._speaker_centroids[1] = emb
        mock_engine._speaker_colors[1] = "#fff"
        mock_engine._speaker_names[1] = "Alice"
        mock_engine._segment_embeddings[1] = [(emb, 2.0)]
        mock_engine._prev_centroids[1] = emb
        mock_engine._matched_speakers.add(1)
        mock_engine._next_id = 5

        mock_engine.reset_session()

        assert len(mock_engine._speaker_centroids) == 0
        assert len(mock_engine._speaker_colors) == 0
        assert len(mock_engine._speaker_names) == 0
        assert len(mock_engine._segment_embeddings) == 0
        assert len(mock_engine._prev_centroids) == 0
        assert len(mock_engine._matched_speakers) == 0
        assert mock_engine._next_id == 1
        assert mock_engine._last_speaker_id == 1


class TestSpeakerStability:

    def test_is_speaker_stable(self, mock_engine, random_embedding):
        emb = random_embedding(seed=42)
        # Stable requires >= 3 segments and small centroid movement
        mock_engine._segment_embeddings[1] = [
            (emb, 2.0), (emb, 2.0), (emb, 2.0),
        ]
        mock_engine._speaker_centroids[1] = emb.copy()
        # Previous centroid nearly identical (tiny perturbation)
        mock_engine._prev_centroids[1] = emb + 1e-6
        assert mock_engine.is_speaker_stable(1) is True

    def test_is_speaker_unstable(self, mock_engine, random_embedding):
        emb_a = random_embedding(seed=1)
        emb_b = random_embedding(seed=2)
        mock_engine._segment_embeddings[1] = [
            (emb_a, 2.0), (emb_a, 2.0), (emb_a, 2.0),
        ]
        mock_engine._speaker_centroids[1] = emb_a.copy()
        # Previous centroid is very different
        mock_engine._prev_centroids[1] = emb_b.copy()
        assert mock_engine.is_speaker_stable(1) is False


class TestSegmentEmbeddings:

    def test_segment_embeddings(self, mock_engine, random_embedding):
        emb1 = random_embedding(seed=10)
        emb2 = random_embedding(seed=20)
        mock_engine._segment_embeddings[1] = [(emb1, 2.0), (emb2, 3.5)]
        result = mock_engine.get_segment_embeddings(1)
        assert len(result) == 2
        assert np.allclose(result[0][0], emb1)
        assert result[0][1] == 2.0
        assert np.allclose(result[1][0], emb2)
        assert result[1][1] == 3.5
        # Non-existent speaker returns empty list
        assert mock_engine.get_segment_embeddings(999) == []


class TestCosineSim:

    def test_cosine_sim(self):
        # Same vector → 1.0
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(DiarizationEngine._cosine_sim(a, a) - 1.0) < 1e-6

        # Orthogonal → 0.0
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(DiarizationEngine._cosine_sim(a, b) - 0.0) < 1e-6

        # Opposite → -1.0
        c = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(DiarizationEngine._cosine_sim(a, c) - (-1.0)) < 1e-6


class TestTrimSilence:

    def test_trim_silence(self):
        sr = 16000
        # 1s silence + 1s tone + 1s silence = 3s total
        silence = np.zeros(sr, dtype=np.float32)
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.concatenate([silence, tone, silence])

        trimmed = DiarizationEngine._trim_silence(audio)
        # Trimmed audio should be shorter than original
        assert len(trimmed) < len(audio)
        # Trimmed audio should still contain the tone (most of the energy)
        rms_trimmed = float(np.sqrt(np.mean(trimmed ** 2)))
        rms_original = float(np.sqrt(np.mean(audio ** 2)))
        assert rms_trimmed > rms_original
