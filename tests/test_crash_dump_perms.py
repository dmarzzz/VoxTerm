"""Crash artifacts must be owner-only, matching the codebase's 0o600/0o700 convention.

The crash dumps and faulthandler.log previously inherited the process umask (often
world-readable), unlike the speaker DB / backups which are locked to 0o600/0o700
(audio/speakers/store.py). They carry session/runtime metadata, so on a shared or
stolen machine a local reader shouldn't get them. This asserts the dir is 0o700 and
the dump files + faulthandler.log are not group/other-accessible.
"""

from __future__ import annotations

import os

import pytest

import diagnostics

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")


def test_crash_dump_files_are_owner_only(tmp_path, monkeypatch):
    crash_dir = tmp_path / "crashes"
    monkeypatch.setattr(diagnostics, "CRASH_DIR", crash_dir)

    diagnostics.write_crash_dump("unit-test", ValueError("boom"), {"recording": True})

    assert crash_dir.exists(), "crash dir not created"
    assert (crash_dir.stat().st_mode & 0o077) == 0, "crash dir is group/other-accessible"

    files = list(crash_dir.glob("*.log")) + list(crash_dir.glob("*.json"))
    assert files, "no crash dump was written"
    for f in files:
        assert (f.stat().st_mode & 0o077) == 0, f"{f.name} is group/other-readable"


def test_faulthandler_log_is_owner_only(tmp_path, monkeypatch):
    import faulthandler

    crash_dir = tmp_path / "crashes"
    monkeypatch.setattr(diagnostics, "CRASH_DIR", crash_dir)
    try:
        diagnostics.setup_faulthandler()
        fh = crash_dir / "faulthandler.log"
        assert fh.exists(), "faulthandler.log not created"
        assert (fh.stat().st_mode & 0o077) == 0, "faulthandler.log is group/other-readable"
    finally:
        # Restore global faulthandler state so we don't leak it into other tests.
        faulthandler.disable()
        if diagnostics._fault_file is not None:
            try:
                diagnostics._fault_file.close()
            except Exception:
                pass
            diagnostics._fault_file = None
