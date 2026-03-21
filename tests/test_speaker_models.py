"""Tests for speakers/models.py — SpeakerProfile data model."""

import numpy as np
import pytest

from speakers.models import SpeakerMeta, SpeakerProfile

EMBEDDING_DIM = 192


@pytest.fixture
def _random_embedding():
    """Local helper — returns a factory for normalized 192-dim embeddings."""
    def _make(seed=None):
        rng = np.random.RandomState(seed)
        emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        return emb
    return _make


def _make_profile(**overrides) -> SpeakerProfile:
    defaults = dict(
        id="test-id",
        name="Test",
        color="#00ffcc",
        centroid=np.zeros(EMBEDDING_DIM, dtype=np.float32),
    )
    defaults.update(overrides)
    return SpeakerProfile(**defaults)


class TestSpeakerProfile:

    def test_add_exemplar_keeps_all_under_max(self, _random_embedding):
        profile = _make_profile()
        for i in range(5):
            profile.add_exemplar(_random_embedding(seed=i))
        assert len(profile.exemplars) == 5

    def test_add_exemplar_replaces_over_max(self, _random_embedding):
        profile = _make_profile()
        # First fill to MAX_EXEMPLARS (20)
        for i in range(20):
            profile.add_exemplar(_random_embedding(seed=i))
        assert len(profile.exemplars) == 20

        # Add 5 more — should stay at 20 via replacement
        for i in range(20, 25):
            profile.add_exemplar(_random_embedding(seed=i))
        assert len(profile.exemplars) == profile.MAX_EXEMPLARS  # 20

    def test_recompute_centroid(self, _random_embedding):
        profile = _make_profile()
        for i in range(3):
            profile.add_exemplar(_random_embedding(seed=i + 100))
        profile.recompute_centroid()

        norm = float(np.linalg.norm(profile.centroid))
        assert abs(norm - 1.0) < 1e-5, f"Centroid L2 norm should be ~1.0, got {norm}"

    def test_best_match_score_with_centroid(self, _random_embedding):
        emb = _random_embedding(seed=200)
        profile = _make_profile(centroid=emb.copy())

        # Matching with the same embedding should give a very high score
        score = profile.best_match_score(emb)
        assert score > 0.99, f"Expected near-perfect match, got {score}"

    def test_compute_quality(self, _random_embedding):
        profile = _make_profile()
        # Add several similar embeddings (close seeds give correlated RNG outputs)
        base = _random_embedding(seed=300)
        for i in range(5):
            # Small perturbation around base
            noise = np.random.RandomState(300 + i).randn(EMBEDDING_DIM).astype(np.float32) * 0.05
            emb = base + noise
            emb /= np.linalg.norm(emb) + 1e-10
            profile.add_exemplar(emb)

        quality = profile.compute_quality()
        assert quality > 0.5, f"Expected quality > 0.5 for similar embeddings, got {quality}"

    def test_to_meta(self, _random_embedding):
        emb = _random_embedding(seed=400)
        profile = _make_profile(
            id="meta-test-id",
            name="MetaTest",
            color="#ff44aa",
            centroid=emb,
            confirmed_count=7,
            auto_assigned_count=3,
            total_duration_sec=42.5,
            quality_score=0.85,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-02T00:00:00",
            last_seen_at="2026-01-03T00:00:00",
        )
        meta = profile.to_meta()

        assert isinstance(meta, SpeakerMeta)
        assert meta.id == "meta-test-id"
        assert meta.name == "MetaTest"
        assert meta.color == "#ff44aa"
        assert meta.confirmed_count == 7
        assert meta.auto_assigned_count == 3
        assert meta.total_duration_sec == 42.5
        assert meta.quality_score == 0.85
        assert meta.created_at == "2026-01-01T00:00:00"
        assert meta.updated_at == "2026-01-02T00:00:00"
        assert meta.last_seen_at == "2026-01-03T00:00:00"
