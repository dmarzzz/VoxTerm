"""Tests for speakers/store.py — persistent speaker profile storage."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from speakers.store import SpeakerStore


class TestSpeakerStore:

    def test_create_and_retrieve(self, in_memory_store: SpeakerStore, random_embedding):
        emb1 = random_embedding(seed=1)
        emb2 = random_embedding(seed=2)
        pid = in_memory_store.create_profile(
            name="Alice", color="#ff0000", embeddings=[emb1, emb2]
        )

        profiles = in_memory_store.get_all_profiles()
        assert len(profiles) == 1

        meta = profiles[0]
        assert meta.id == pid
        assert meta.name == "Alice"
        assert meta.color == "#ff0000"
        assert meta.confirmed_count == 2
        assert meta.total_duration_sec == 0.0

    def test_update_profile_embedding(self, in_memory_store: SpeakerStore, random_embedding):
        emb_init = random_embedding(seed=10)
        pid = in_memory_store.create_profile(
            name="Bob", color="#00ff00", embeddings=[emb_init]
        )

        # Grab the centroid before the update
        profiles_before = in_memory_store.get_all_profiles()
        centroid_before = in_memory_store._centroids[pid].copy()

        # Add a very different embedding to shift the centroid
        emb_new = random_embedding(seed=99)
        in_memory_store.update_profile_embedding(pid, emb_new)

        centroid_after = in_memory_store._centroids[pid]
        # Centroid should have moved
        assert not np.allclose(centroid_before, centroid_after, atol=1e-4)

    def test_rename_profile(self, in_memory_store: SpeakerStore, random_embedding):
        emb = random_embedding(seed=20)
        pid = in_memory_store.create_profile(
            name="OldName", color="#0000ff", embeddings=[emb]
        )

        in_memory_store.rename_profile(pid, "NewName")

        # Verify the in-memory cache reflects the new name
        meta = in_memory_store._profiles[pid]
        assert meta.name == "NewName"

        # Also verify via the public API
        profiles = in_memory_store.get_all_profiles()
        assert profiles[0].name == "NewName"

    def test_delete_profile(self, in_memory_store: SpeakerStore, random_embedding):
        emb = random_embedding(seed=30)
        pid = in_memory_store.create_profile(
            name="ToDelete", color="#aabbcc", embeddings=[emb]
        )
        assert len(in_memory_store.get_all_profiles()) == 1

        in_memory_store.delete_profile(pid)

        assert len(in_memory_store.get_all_profiles()) == 0

    def test_match_profiles(self, in_memory_store: SpeakerStore, random_embedding):
        # Create two profiles with very different embeddings
        emb_a = random_embedding(seed=40)
        emb_b = random_embedding(seed=41)
        pid_a = in_memory_store.create_profile(
            name="SpeakerA", color="#111111", embeddings=[emb_a]
        )
        pid_b = in_memory_store.create_profile(
            name="SpeakerB", color="#222222", embeddings=[emb_b]
        )

        # Match with an embedding identical to profile A
        results = in_memory_store.match_profiles(emb_a, top_k=3)
        assert len(results) >= 2
        # The first result should be profile A (highest score)
        assert results[0][0] == pid_a
        assert results[0][2] > results[1][2]

    def test_classify_match_low(self, in_memory_store: SpeakerStore, random_embedding):
        emb = random_embedding(seed=50)
        in_memory_store.create_profile(
            name="Lonely", color="#333333", embeddings=[emb]
        )

        # Create a near-orthogonal embedding to force a low match
        orthogonal = np.zeros(512, dtype=np.float32)
        orthogonal[0] = 1.0  # unit vector along dim 0
        # Subtract any projection onto emb to make it truly orthogonal
        orthogonal -= np.dot(orthogonal, emb) * emb
        norm = np.linalg.norm(orthogonal)
        if norm > 1e-10:
            orthogonal /= norm

        result = in_memory_store.classify_match(orthogonal)
        assert result.tier == "low"

    def test_is_profile_mature(self, in_memory_store: SpeakerStore, random_embedding):
        emb = random_embedding(seed=60)
        pid = in_memory_store.create_profile(
            name="Newbie", color="#444444", embeddings=[emb]
        )
        # Only 1 confirmed sample — should not be mature (needs >= 10)
        assert in_memory_store.is_profile_mature(pid) is False

    def test_session_speaker_recording(self, in_memory_store: SpeakerStore, random_embedding):
        emb = random_embedding(seed=70)
        pid = in_memory_store.create_profile(
            name="SessionTest", color="#555555", embeddings=[emb]
        )
        # Should not crash
        in_memory_store.record_session_speaker(
            session_id="sess-001", speaker_id=pid, local_id=1, segment_count=5
        )

    def test_delete_all_data(self, in_memory_store: SpeakerStore, random_embedding):
        for seed in range(80, 85):
            emb = random_embedding(seed=seed)
            in_memory_store.create_profile(
                name=f"Speaker{seed}", color="#666666", embeddings=[emb]
            )
        assert len(in_memory_store.get_all_profiles()) == 5

        in_memory_store.delete_all_data()

        assert len(in_memory_store.get_all_profiles()) == 0

    # ── file permission tests ─────────────────────────────────

    def test_export_db_permissions(self, in_memory_store: SpeakerStore, random_embedding, tmp_path):
        """Exported database files should be owner-only (0o600)."""
        emb = random_embedding(seed=100)
        in_memory_store.create_profile(name="ExportTest", color="#aaaaaa", embeddings=[emb])

        export_path = tmp_path / "exported.db"
        in_memory_store.export_db(export_path)

        assert export_path.exists()
        mode = export_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_backup_permissions(self, in_memory_store: SpeakerStore, tmp_path):
        """Backup files should be 0o600, backup directory should be 0o700."""
        backup_dir = tmp_path / ".backups"
        with patch("speakers.store.BACKUP_DIR", backup_dir):
            in_memory_store.backup()

        assert backup_dir.exists()
        dir_mode = backup_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700, f"Expected dir 0o700, got {oct(dir_mode)}"

        backup_files = list(backup_dir.glob("speakers_*.db"))
        assert len(backup_files) == 1
        file_mode = backup_files[0].stat().st_mode & 0o777
        assert file_mode == 0o600, f"Expected file 0o600, got {oct(file_mode)}"

    def test_export_db_chmod_failure_logs_warning(
        self, in_memory_store: SpeakerStore, random_embedding, tmp_path, caplog
    ):
        """When chmod fails on export, a warning should be logged."""
        emb = random_embedding(seed=101)
        in_memory_store.create_profile(name="LogTest", color="#bbbbbb", embeddings=[emb])

        export_path = tmp_path / "exported.db"

        with patch.object(Path, "chmod", side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING, logger="speakers.store"):
                in_memory_store.export_db(export_path)

        assert export_path.exists()
        assert "Could not set permissions" in caplog.text

    def test_backup_retightens_directory_on_early_return(self, in_memory_store: SpeakerStore, tmp_path):
        """Even when today's backup exists, directory permissions are re-tightened."""
        backup_dir = tmp_path / ".backups"
        with patch("speakers.store.BACKUP_DIR", backup_dir):
            # First backup creates the file
            in_memory_store.backup()
            # Loosen directory permissions to simulate external change
            backup_dir.chmod(0o755)
            # Second backup hits early return but should still tighten
            in_memory_store.backup()

        dir_mode = backup_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700, f"Expected dir 0o700 after re-tightening, got {oct(dir_mode)}"
