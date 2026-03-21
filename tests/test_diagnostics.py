"""Tests for diagnostics.py — crash dump writing and log rotation."""

import json
from pathlib import Path

import pytest

import diagnostics


@pytest.fixture(autouse=True)
def _patch_crash_dir(tmp_crash_dir, monkeypatch):
    """Redirect diagnostics.CRASH_DIR to a temporary directory for every test."""
    monkeypatch.setattr(diagnostics, "CRASH_DIR", tmp_crash_dir)


def test_write_crash_dump_creates_files(tmp_crash_dir):
    """write_crash_dump creates both a .log and a .json file."""
    diagnostics.write_crash_dump(
        context="test_write",
        exc=RuntimeError("boom"),
        state={"recording": True},
    )
    logs = list(tmp_crash_dir.glob("*.log"))
    jsons = list(tmp_crash_dir.glob("*.json"))
    assert len(logs) >= 1, "No .log file created"
    assert len(jsons) >= 1, "No .json file created"


def test_crash_dump_json_valid(tmp_crash_dir):
    """The .json crash dump is valid JSON with expected keys."""
    diagnostics.write_crash_dump(
        context="test_json",
        exc=ValueError("bad value"),
    )
    json_files = list(tmp_crash_dir.glob("*.json"))
    assert json_files, "No .json file created"
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "timestamp" in data
    assert "context" in data
    assert "error_type" in data
    assert "error_message" in data
    assert "peak_rss_mb" in data


def test_crash_dump_contains_context(tmp_crash_dir):
    """The .log crash dump contains the supplied context string."""
    diagnostics.write_crash_dump(
        context="test_ctx",
        exc=RuntimeError("ctx error"),
    )
    log_files = list(tmp_crash_dir.glob("*.log"))
    assert log_files, "No .log file created"
    content = log_files[0].read_text(encoding="utf-8")
    assert "test_ctx" in content


def test_rotate_keeps_max(tmp_crash_dir):
    """rotate_crash_logs prunes logs down to CRASH_LOG_MAX_COUNT."""
    from config import CRASH_LOG_MAX_COUNT

    # Create 60 dummy .log files
    for i in range(60):
        (tmp_crash_dir / f"2025-01-01_00{i:04d}.log").write_text(
            f"dummy log {i}", encoding="utf-8"
        )

    diagnostics.rotate_crash_logs()

    remaining = list(tmp_crash_dir.glob("*.log"))
    assert len(remaining) <= CRASH_LOG_MAX_COUNT, (
        f"Expected at most {CRASH_LOG_MAX_COUNT} logs, found {len(remaining)}"
    )


def test_rotate_preserves_faulthandler_log(tmp_crash_dir):
    """rotate_crash_logs never deletes faulthandler.log."""
    fh_log = tmp_crash_dir / "faulthandler.log"
    fh_log.write_text("faulthandler trace data", encoding="utf-8")

    # Create 60 other .log files to trigger rotation
    for i in range(60):
        (tmp_crash_dir / f"2025-01-01_00{i:04d}.log").write_text(
            f"dummy log {i}", encoding="utf-8"
        )

    diagnostics.rotate_crash_logs()

    assert fh_log.exists(), "faulthandler.log was deleted by rotation"
