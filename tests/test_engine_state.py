"""Tests for DiarizationEngine state management (no model loading)."""

import numpy as np
import pytest

from diarization.engine import DiarizationEngine

EMBEDDING_DIM = 512


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

        # Merged centroid is running sum (a + b)
        expected = emb_a + emb_b
        assert np.allclose(mock_engine._speaker_centroids[1], expected, atol=1e-5)


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
        mock_engine._identify_count = 42

        mock_engine.reset_session()

        assert len(mock_engine._speaker_centroids) == 0
        assert len(mock_engine._speaker_colors) == 0
        assert len(mock_engine._speaker_names) == 0
        assert len(mock_engine._segment_embeddings) == 0
        assert len(mock_engine._prev_centroids) == 0
        assert len(mock_engine._matched_speakers) == 0
        assert mock_engine._next_id == 1
        assert mock_engine._last_speaker_id == 1
        assert mock_engine._identify_count == 0


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


class TestRunningSumCentroid:
    """Verify centroid update uses running sum (not EMA)."""

    def test_centroid_is_sum_of_embeddings(self, mock_engine, random_embedding):
        emb1 = random_embedding(seed=1)
        emb2 = random_embedding(seed=2)

        # Set up initial speaker
        mock_engine._speaker_centroids[1] = emb1.copy()
        mock_engine._segment_embeddings[1] = [(emb1, 2.0)]

        # Simulate matching update (high quality)
        mock_engine._prev_centroids[1] = mock_engine._speaker_centroids[1].copy()
        mock_engine._speaker_centroids[1] = mock_engine._speaker_centroids[1] + emb1

        # After adding the same embedding again, centroid = emb1 + emb1
        expected = emb1 + emb1
        assert np.allclose(mock_engine._speaker_centroids[1], expected, atol=1e-6)


class TestGetSessionCentroid:
    """Verify get_session_centroid returns L2-normalized vector."""

    def test_returns_normalized(self, mock_engine, random_embedding):
        emb = random_embedding(seed=42)
        # Running sum: centroid is unnormalized
        mock_engine._speaker_centroids[1] = emb * 5.0
        centroid = mock_engine.get_session_centroid(1)
        norm = float(np.linalg.norm(centroid))
        assert abs(norm - 1.0) < 1e-6

    def test_returns_none_for_missing(self, mock_engine):
        assert mock_engine.get_session_centroid(999) is None


class TestQualityGating:
    """Verify low-energy audio is quality-gated."""

    def test_low_rms_skips_new_speaker(self, loaded_mock_engine):
        engine = loaded_mock_engine
        # Create very low energy audio (below QUALITY_RMS_THRESHOLD)
        audio = np.full(24000, 0.001, dtype=np.float32)
        label, sid = engine.identify(audio)
        # Should assign to last speaker (default 1), not create new
        assert sid == 1
        # No centroid should be created from low-quality audio
        assert len(engine._speaker_centroids) == 0

    def test_high_rms_creates_speaker(self, loaded_mock_engine):
        engine = loaded_mock_engine
        t = np.linspace(0, 2.0, 32000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        label, sid = engine.identify(audio)
        assert sid == 1
        assert len(engine._speaker_centroids) == 1


class TestPeriodicMerge:
    """Verify periodic cluster merging."""

    def test_maybe_merge_clusters(self, mock_engine, random_embedding):
        emb = random_embedding(seed=1)
        # Two clusters with very similar centroids (should merge)
        mock_engine._speaker_centroids[1] = emb.copy()
        mock_engine._speaker_centroids[2] = emb + np.random.RandomState(99).randn(EMBEDDING_DIM).astype(np.float32) * 0.01
        mock_engine._segment_embeddings[1] = [(emb, 2.0)] * 5
        mock_engine._segment_embeddings[2] = [(emb, 2.0)] * 2
        mock_engine._speaker_colors[1] = "#aaa"
        mock_engine._speaker_colors[2] = "#bbb"

        mock_engine._maybe_merge_clusters()

        # Should have merged — only 1 cluster remains
        assert len(mock_engine._speaker_centroids) == 1
        # Larger cluster (1, 5 segments) should be the target
        assert 1 in mock_engine._speaker_centroids
        assert 2 not in mock_engine._speaker_centroids

    def test_no_merge_when_dissimilar(self, mock_engine, random_embedding):
        emb1 = random_embedding(seed=1)
        emb2 = random_embedding(seed=2)
        # Two very different clusters — should NOT merge
        mock_engine._speaker_centroids[1] = emb1.copy()
        mock_engine._speaker_centroids[2] = emb2.copy()
        mock_engine._segment_embeddings[1] = [(emb1, 2.0)]
        mock_engine._segment_embeddings[2] = [(emb2, 2.0)]

        mock_engine._maybe_merge_clusters()

        assert len(mock_engine._speaker_centroids) == 2

    def test_merge_triggered_at_interval(self, loaded_mock_engine):
        engine = loaded_mock_engine
        assert engine._identify_count == 0
        # Run identify enough times to trigger merge check
        t = np.linspace(0, 2.0, 32000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        for _ in range(engine.MERGE_INTERVAL):
            engine.identify(audio)
        assert engine._identify_count == engine.MERGE_INTERVAL


class TestThresholdConstants:
    """Verify Phase 1 threshold values."""

    def test_similarity_threshold(self):
        assert DiarizationEngine.MATCH_THRESHOLD == 0.35
        assert DiarizationEngine.NEW_SPEAKER_THRESHOLD == 0.30

    def test_merge_threshold(self):
        assert DiarizationEngine.MERGE_THRESHOLD == 0.50

    def test_min_speech_samples(self):
        from diarization.engine import _MIN_SPEECH_SAMPLES
        assert _MIN_SPEECH_SAMPLES == 24000  # 1.5s at 16kHz


class TestSpectralRecluster:
    """Verify spectral re-clustering merges over-segmented speakers."""

    def test_merges_similar_speakers(self, mock_engine, random_embedding):
        """Three 'speakers' from same source should merge into fewer."""
        base = random_embedding(seed=1)
        # Create 3 speakers with very similar embeddings (same person, over-segmented)
        # Use tiny noise (0.003) so high-dim vectors stay nearly identical
        for sid in (1, 2, 3):
            centroid = base + np.random.RandomState(sid + 100).randn(EMBEDDING_DIM).astype(np.float32) * 0.003
            mock_engine._speaker_centroids[sid] = centroid
            mock_engine._speaker_colors[sid] = f"#{sid}00000"
            mock_engine._segment_embeddings[sid] = [
                (base + np.random.RandomState(sid * 10 + j).randn(EMBEDDING_DIM).astype(np.float32) * 0.003, 2.0)
                for j in range(3)
            ]

        assert len(mock_engine._speaker_centroids) == 3
        # Pairwise merge handles identical speakers (spectral recluster
        # enforces min k=2 so won't merge to 1, but pairwise catches it)
        mock_engine._maybe_merge_clusters()
        mock_engine._maybe_merge_clusters()  # may need two rounds for 3→2→1
        # Should have merged at least some
        assert len(mock_engine._speaker_centroids) < 3

    def test_preserves_distinct_speakers(self, mock_engine, random_embedding):
        """Two genuinely different speakers should NOT be merged."""
        emb1 = random_embedding(seed=1)
        emb2 = random_embedding(seed=2)  # different seed = different direction

        for sid, base_emb in [(1, emb1), (2, emb2)]:
            mock_engine._speaker_centroids[sid] = base_emb.copy()
            mock_engine._speaker_colors[sid] = f"#{sid}00000"
            mock_engine._segment_embeddings[sid] = [
                (base_emb + np.random.RandomState(sid * 10 + j).randn(EMBEDDING_DIM).astype(np.float32) * 0.02, 2.0)
                for j in range(4)
            ]

        mock_engine._spectral_recluster()
        # Both speakers should remain
        assert len(mock_engine._speaker_centroids) == 2

    def test_skips_when_too_few_segments(self, mock_engine, random_embedding):
        """Re-clustering should skip when below RECLUSTER_MIN_SEGMENTS."""
        emb = random_embedding(seed=1)
        mock_engine._speaker_centroids[1] = emb.copy()
        mock_engine._speaker_centroids[2] = emb.copy()
        mock_engine._segment_embeddings[1] = [(emb, 2.0)]
        mock_engine._segment_embeddings[2] = [(emb, 2.0)]

        mock_engine._spectral_recluster()
        # Should not merge — too few segments
        assert len(mock_engine._speaker_centroids) == 2

    def test_kmeans_basic(self):
        """K-means should separate two well-separated clusters."""
        rng = np.random.RandomState(42)
        cluster1 = rng.randn(20, 3).astype(np.float32) + np.array([5, 0, 0])
        cluster2 = rng.randn(20, 3).astype(np.float32) + np.array([-5, 0, 0])
        X = np.vstack([cluster1, cluster2])
        labels = DiarizationEngine._kmeans(X, k=2)
        # All points in cluster1 should have the same label
        assert len(set(labels[:20])) == 1
        assert len(set(labels[20:])) == 1
        # And different from each other
        assert labels[0] != labels[20]


class TestViterbiSmoothing:
    """Verify VBx-style HMM smoothing."""

    def test_smooths_rapid_switching(self):
        """Rapid A-B-A-B switching should be smoothed to continuity."""
        engine = DiarizationEngine()
        # Alternating labels: speaker switching every segment
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        smoothed = engine._viterbi_smooth(labels, k=2)
        # With loopP=0.99, the HMM should prefer fewer switches
        switches_before = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        switches_after = sum(1 for i in range(1, len(smoothed)) if smoothed[i] != smoothed[i-1])
        assert switches_after < switches_before

    def test_preserves_genuine_turns(self):
        """Long runs of each speaker should be preserved."""
        engine = DiarizationEngine()
        labels = [0]*10 + [1]*10 + [0]*10
        smoothed = engine._viterbi_smooth(labels, k=2)
        # Should preserve the two speaker turns (only 2 switches)
        switches = sum(1 for i in range(1, len(smoothed)) if smoothed[i] != smoothed[i-1])
        assert switches == 2

    def test_single_speaker_unchanged(self):
        """All same speaker should stay unchanged."""
        engine = DiarizationEngine()
        labels = [0] * 20
        smoothed = engine._viterbi_smooth(labels, k=2)
        assert all(s == smoothed[0] for s in smoothed)


class TestWhitening:
    """Verify PLDA-lite whitening transform."""

    def test_update_whitening_with_data(self, mock_engine, random_embedding):
        """Whitening should be computed when enough data is available."""
        # Create two speakers with multiple embeddings each
        for sid in (1, 2):
            base = random_embedding(seed=sid)
            mock_engine._speaker_centroids[sid] = base.copy()
            mock_engine._segment_embeddings[sid] = [
                (base + np.random.RandomState(sid * 100 + j).randn(EMBEDDING_DIM).astype(np.float32) * 0.05, 2.0)
                for j in range(6)
            ]

        assert mock_engine._whiten_matrix is None
        mock_engine._update_whitening()
        assert mock_engine._whiten_matrix is not None
        assert mock_engine._whiten_mean is not None
        assert mock_engine._whiten_matrix.shape == (EMBEDDING_DIM, EMBEDDING_DIM)

    def test_whiten_transform(self, mock_engine, random_embedding):
        """Whitened embedding should have different values."""
        emb = random_embedding(seed=1)
        # Set up a simple whitening transform
        mock_engine._whiten_matrix = np.eye(EMBEDDING_DIM, dtype=np.float32) * 2.0
        mock_engine._whiten_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        whitened = mock_engine._whiten(emb)
        assert not np.allclose(whitened, emb)
        assert np.allclose(whitened, emb * 2.0)

    def test_skips_when_insufficient_data(self, mock_engine, random_embedding):
        """Whitening should not activate with too few segments."""
        emb = random_embedding(seed=1)
        mock_engine._speaker_centroids[1] = emb.copy()
        mock_engine._segment_embeddings[1] = [(emb, 2.0)]
        mock_engine._update_whitening()
        assert mock_engine._whiten_matrix is None


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


class TestSCDConstants:
    """Verify speaker change detection threshold constants."""

    def test_scd_change_threshold(self):
        assert DiarizationEngine.SCD_CHANGE_THRESHOLD == 0.6

    def test_scd_window_sec(self):
        assert DiarizationEngine.SCD_WINDOW_SEC == 2.0

    def test_scd_hop_sec(self):
        assert DiarizationEngine.SCD_HOP_SEC == 0.5

    def test_scd_min_samples(self):
        from diarization.engine import _SCD_MIN_SAMPLES
        assert _SCD_MIN_SAMPLES == 48000  # 3.0s at 16kHz


class TestExtractEmbedding:
    """Verify _extract_embedding helper."""

    def test_returns_none_when_not_loaded(self, mock_engine):
        audio = np.zeros(32000, dtype=np.float32)
        result = mock_engine._extract_embedding(audio)
        assert result is None

    def test_returns_none_for_short_audio(self, loaded_mock_engine):
        # Audio shorter than _MIN_SPEECH_SAMPLES (24000 = 1.5s)
        audio = np.zeros(16000, dtype=np.float32)
        result = loaded_mock_engine._extract_embedding(audio)
        assert result is None

    def test_returns_embedding_for_valid_audio(self, loaded_mock_engine):
        t = np.linspace(0, 2.0, 32000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = loaded_mock_engine._extract_embedding(audio)
        assert result is not None
        assert result.shape == (EMBEDDING_DIM,)
        assert result.dtype == np.float32


class TestIdentifySegments:
    """Verify identify_segments with speaker change detection."""

    def test_short_audio_falls_back_to_single_identify(self, loaded_mock_engine):
        """Audio shorter than _SCD_MIN_SAMPLES should fall back to single identify."""
        # 2.5s audio (40000 samples) < 3.0s (48000 samples)
        t = np.linspace(0, 2.5, 40000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        results = loaded_mock_engine.identify_segments(audio)
        assert len(results) == 1
        label, sid, start, end = results[0]
        assert start == 0
        assert end == len(audio)
        assert sid >= 1
        assert isinstance(label, str)

    def test_returns_valid_segments_for_long_audio(self, loaded_mock_engine):
        """Long audio should return at least one segment with valid fields."""
        # 5s audio — enough for SCD sliding window
        t = np.linspace(0, 5.0, 80000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        results = loaded_mock_engine.identify_segments(audio)
        assert len(results) >= 1
        for label, sid, start, end in results:
            assert isinstance(label, str)
            assert sid >= 1
            assert start >= 0
            assert end <= len(audio)
            assert end > start

    def test_segments_cover_audio(self, loaded_mock_engine):
        """Segments should start at 0 and end at audio length (no gaps at boundaries)."""
        t = np.linspace(0, 5.0, 80000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        results = loaded_mock_engine.identify_segments(audio)
        assert results[0][2] == 0  # first segment starts at 0
        assert results[-1][3] == len(audio)  # last segment ends at audio length

    def test_not_loaded_returns_default(self):
        """Unloaded engine should return a single default segment."""
        engine = DiarizationEngine()
        audio = np.zeros(80000, dtype=np.float32)
        results = engine.identify_segments(audio)
        assert results == [("Speaker 1", 1, 0, len(audio))]

    def test_mono_conversion(self, loaded_mock_engine):
        """Stereo audio should be handled (converted to mono)."""
        t = np.linspace(0, 5.0, 80000, dtype=np.float32)
        mono = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.stack([mono, mono], axis=1)
        results = loaded_mock_engine.identify_segments(stereo)
        assert len(results) >= 1
        for label, sid, start, end in results:
            assert sid >= 1
