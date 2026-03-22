"""Tests for ConfigStore — typed settings persistence with merge semantics."""

import json
import tempfile
from pathlib import Path

import pytest

from config_store import ConfigStore


@pytest.fixture
def tmp_path_file(tmp_path):
    return tmp_path / "state.json"


class TestDefaults:
    def test_fresh_store_has_all_defaults(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        assert cs.get("last_model") == ""
        assert cs.get("last_language") == ""
        assert cs.get("audio_retention") is False
        assert cs.get("export_format") == "markdown"
        assert cs.get("summarization_model") == ""
        assert cs.get("summarization_strength") == "medium"

    def test_unknown_key_returns_none(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        assert cs.get("nonexistent") is None


class TestGetSet:
    def test_set_and_get(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.set("last_model", "qwen3-1.7b")
        assert cs.get("last_model") == "qwen3-1.7b"

    def test_set_bool(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.set("audio_retention", True)
        assert cs.get("audio_retention") is True

    def test_set_persists_to_disk(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.set("export_format", "txt")
        raw = json.loads(tmp_path_file.read_text())
        assert raw["export_format"] == "txt"

    def test_merge_semantics(self, tmp_path_file):
        """Setting one key preserves all others."""
        cs = ConfigStore(tmp_path_file)
        cs.set("last_model", "tiny")
        cs.set("last_language", "ja")
        assert cs.get("last_model") == "tiny"
        assert cs.get("last_language") == "ja"
        # Defaults still present
        assert cs.get("audio_retention") is False


class TestPersistence:
    def test_reload_from_disk(self, tmp_path_file):
        cs1 = ConfigStore(tmp_path_file)
        cs1.set("last_model", "turbo")
        cs1.set("summarization_strength", "high")

        cs2 = ConfigStore(tmp_path_file)
        assert cs2.get("last_model") == "turbo"
        assert cs2.get("summarization_strength") == "high"
        assert cs2.get("export_format") == "markdown"  # default


class TestBackwardCompat:
    def test_old_two_key_file(self, tmp_path_file):
        """Existing .state.json with only last_model + last_language loads fine."""
        tmp_path_file.write_text(json.dumps({
            "last_model": "small",
            "last_language": "de",
        }))
        cs = ConfigStore(tmp_path_file)
        assert cs.get("last_model") == "small"
        assert cs.get("last_language") == "de"
        assert cs.get("audio_retention") is False
        assert cs.get("export_format") == "markdown"

    def test_unknown_keys_preserved(self, tmp_path_file):
        """Keys not in schema are kept (future-proofing)."""
        tmp_path_file.write_text(json.dumps({"future_key": 42}))
        cs = ConfigStore(tmp_path_file)
        assert cs.get("future_key") == 42

    def test_corrupt_file_uses_defaults(self, tmp_path_file):
        tmp_path_file.write_text("not json!!")
        cs = ConfigStore(tmp_path_file)
        assert cs.get("last_model") == ""

    def test_missing_file_uses_defaults(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        assert cs.get("export_format") == "markdown"


class TestTypeValidation:
    def test_wrong_type_rejected_on_load(self, tmp_path_file):
        tmp_path_file.write_text(json.dumps({"audio_retention": "yes"}))
        cs = ConfigStore(tmp_path_file)
        assert cs.get("audio_retention") is False  # default, not "yes"

    def test_correct_type_accepted_on_load(self, tmp_path_file):
        tmp_path_file.write_text(json.dumps({"audio_retention": True}))
        cs = ConfigStore(tmp_path_file)
        assert cs.get("audio_retention") is True

    def test_wrong_type_rejected_on_set(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        with pytest.raises(TypeError, match="expected bool"):
            cs.set("audio_retention", "yes")

    def test_wrong_type_rejected_on_update(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        with pytest.raises(TypeError, match="expected str"):
            cs.update({"last_model": 123})


class TestAtomicWrite:
    def test_no_tmp_file_left(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.set("last_model", "test")
        assert not tmp_path_file.with_suffix(".tmp").exists()

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "state.json"
        cs = ConfigStore(deep)
        cs.set("last_model", "test")
        assert deep.exists()


class TestUpdate:
    def test_update_multiple_keys(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.update({"last_model": "qwen3-1.7b", "last_language": "ja"})
        assert cs.get("last_model") == "qwen3-1.7b"
        assert cs.get("last_language") == "ja"

    def test_update_single_disk_write(self, tmp_path_file):
        """update() writes to disk once, not per-key."""
        cs = ConfigStore(tmp_path_file)
        cs.update({"last_model": "turbo", "last_language": "fr"})
        raw = json.loads(tmp_path_file.read_text())
        assert raw["last_model"] == "turbo"
        assert raw["last_language"] == "fr"

    def test_update_preserves_existing(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.set("export_format", "txt")
        cs.update({"last_model": "turbo"})
        assert cs.get("export_format") == "txt"


class TestDataSnapshot:
    def test_data_returns_copy(self, tmp_path_file):
        cs = ConfigStore(tmp_path_file)
        cs.set("last_model", "turbo")
        snapshot = cs.data
        snapshot["last_model"] = "tampered"
        assert cs.get("last_model") == "turbo"  # not affected
