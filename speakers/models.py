"""Data models for persistent speaker profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class SpeakerMeta:
    """Lightweight metadata for UI display (no embedding data)."""

    id: str
    name: str
    color: str
    confirmed_count: int
    auto_assigned_count: int
    total_duration_sec: float
    quality_score: float
    created_at: str
    updated_at: str
    last_seen_at: str


MULTI_CENTROID_MIN = 15   # min exemplars before multi-centroid kicks in
MULTI_CENTROID_K = 3      # number of sub-centroids


@dataclass
class SpeakerProfile:
    """Full speaker profile including embedding data."""

    id: str
    name: str
    color: str

    centroid: np.ndarray                           # (192,) float32
    exemplars: list[np.ndarray] = field(default_factory=list)  # up to MAX_EXEMPLARS
    sub_centroids: list[np.ndarray] = field(default_factory=list)  # K sub-centroids

    confirmed_count: int = 0
    auto_assigned_count: int = 0
    total_duration_sec: float = 0.0
    quality_score: float = 0.0

    created_at: str = ""
    updated_at: str = ""
    last_seen_at: str = ""

    MAX_EXEMPLARS: int = field(default=20, repr=False)

    def add_exemplar(self, embedding: np.ndarray) -> None:
        """Add an exemplar, replacing the most redundant if full."""
        if len(self.exemplars) < self.MAX_EXEMPLARS:
            self.exemplars.append(embedding.copy())
        else:
            # Replace the exemplar closest to centroid (most redundant)
            sims = [
                float(np.dot(self.centroid, e) / (np.linalg.norm(self.centroid) * np.linalg.norm(e) + 1e-10))
                for e in self.exemplars
            ]
            idx = int(np.argmax(sims))
            self.exemplars[idx] = embedding.copy()

    def recompute_centroid(self) -> None:
        """Recompute centroid as L2-normalized mean of exemplars."""
        if not self.exemplars:
            return
        stacked = np.stack(self.exemplars)
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-10:
            mean /= norm
        self.centroid = mean.astype(np.float32)

        # Update sub-centroids if enough exemplars
        if len(self.exemplars) >= MULTI_CENTROID_MIN:
            self._compute_sub_centroids()
        else:
            self.sub_centroids = []

    def _compute_sub_centroids(self) -> None:
        """Compute K sub-centroids via simple k-means on exemplars."""
        k = min(MULTI_CENTROID_K, len(self.exemplars) // 3)
        if k < 2:
            self.sub_centroids = []
            return

        data = np.stack(self.exemplars)  # (N, 192)
        n = len(data)

        # Initialize with evenly spaced exemplars
        indices = np.linspace(0, n - 1, k, dtype=int)
        centers = data[indices].copy()

        # Run k-means for a few iterations
        for _ in range(10):
            # Assign each exemplar to nearest center
            # (N, K) similarity matrix
            sims = data @ centers.T  # cosine-ish (works well for unit-ish vectors)
            assignments = sims.argmax(axis=1)

            # Recompute centers
            new_centers = np.zeros_like(centers)
            for ci in range(k):
                mask = assignments == ci
                if mask.any():
                    mean = data[mask].mean(axis=0)
                    norm = np.linalg.norm(mean)
                    if norm > 1e-10:
                        mean /= norm
                    new_centers[ci] = mean
                else:
                    new_centers[ci] = centers[ci]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        self.sub_centroids = [c.astype(np.float32) for c in centers]

    def best_match_score(self, embedding: np.ndarray) -> float:
        """Compute best match score using sub-centroids if available."""
        if self.sub_centroids:
            scores = [
                float(np.dot(embedding, c) / (np.linalg.norm(embedding) * np.linalg.norm(c) + 1e-10))
                for c in self.sub_centroids
            ]
            return max(scores)
        # Fallback to single centroid
        return float(
            np.dot(embedding, self.centroid)
            / (np.linalg.norm(embedding) * np.linalg.norm(self.centroid) + 1e-10)
        )

    def compute_quality(self) -> float:
        """Compute quality score: 1 - mean pairwise cosine distance of exemplars."""
        if len(self.exemplars) < 2:
            return 0.0
        stacked = np.stack(self.exemplars)
        # Normalize
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = stacked / norms
        # Pairwise cosine similarities
        sims = normed @ normed.T
        n = len(sims)
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_sim = float(sims[mask].mean())
        return max(0.0, min(1.0, mean_sim))

    def detect_drift(self) -> float:
        """Check how far the centroid has drifted from the initial exemplars.

        Returns the cosine distance from the centroid of the first 5 exemplars
        (the "golden set") to the current centroid. > 0.20 suggests drift.
        """
        if len(self.exemplars) < 5:
            return 0.0
        golden = np.stack(self.exemplars[:5]).mean(axis=0)
        norm = np.linalg.norm(golden)
        if norm > 1e-10:
            golden /= norm
        sim = float(
            np.dot(golden, self.centroid)
            / (np.linalg.norm(golden) * np.linalg.norm(self.centroid) + 1e-10)
        )
        return max(0.0, 1.0 - sim)

    def to_meta(self) -> SpeakerMeta:
        return SpeakerMeta(
            id=self.id,
            name=self.name,
            color=self.color,
            confirmed_count=self.confirmed_count,
            auto_assigned_count=self.auto_assigned_count,
            total_duration_sec=self.total_duration_sec,
            quality_score=self.quality_score,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_seen_at=self.last_seen_at,
        )
