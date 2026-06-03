"""Detect & sync recordings off a plugged-in Rode USB recorder (issue #145).

A Rode (e.g. Wireless GO) with onboard recording mounts as a read-only USB
mass-storage volume. We can only copy down — never delete from the device.
The device has no real-time clock and writes no BWF timestamp, so dedup is by
content hash, not mtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, DEFAULT_MODEL, DEFAULT_LANGUAGE
from .core import transcribe_file, build_pipeline

RODE_GLOBS = ("*_Wireless_GO.WAV", "*_Wireless_GO.wav")
IMPORT_DIR = DATA_DIR / "imports"
MANIFEST = DATA_DIR / ".rode_imports.json"


def _mount_roots() -> list[Path]:
    user = os.environ.get("USER", "")
    roots = [Path("/media") / user, Path(f"/run/media/{user}"), Path("/Volumes")]
    return [r for r in roots if r.is_dir()]


def find_rode_recordings() -> list[Path]:
    """Return WAVs on any mounted Rode volume (deduped, sorted by name)."""
    found: set[Path] = set()
    for root in _mount_roots():
        for vol in root.iterdir():
            if not vol.is_dir():
                continue
            for pattern in RODE_GLOBS:
                found.update(vol.glob(pattern))
    return sorted(found)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def import_recordings(
    transcribe: bool = True,
    model_name: str = DEFAULT_MODEL,
    language: str | None = DEFAULT_LANGUAGE,
) -> list[tuple[Path, str, str | None]]:
    """Copy new recordings off the device, then transcribe.

    Dedup is by content hash, tracked separately from transcription: a file
    that was copied (e.g. via --no-transcribe) but not yet transcribed will be
    transcribed on a later run. Action per file is one of "import" (copied +
    transcribed), "copy" (copied only), "transcribe" (already copied, now
    transcribed), or "skip" (nothing to do).
    """
    recordings = find_rode_recordings()
    manifest = _load_manifest()
    results: list[tuple[Path, str, str | None]] = []
    pipeline = build_pipeline(model_name, language) if (transcribe and recordings) else None
    if recordings:
        IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    for src in recordings:
        digest = _sha256(src)
        entry = manifest.get(digest)
        copied = entry is not None

        if not copied:
            dest = IMPORT_DIR / src.name
            if dest.exists():
                dest = IMPORT_DIR / f"{digest[:8]}_{src.name}"
            shutil.copy2(src, dest)
            entry = {
                "name": src.name,
                "size": src.stat().st_size,
                "imported_at": datetime.now().isoformat(timespec="seconds"),
                "local": str(dest),
                "transcript": None,
            }
            manifest[digest] = entry
            MANIFEST.write_text(json.dumps(manifest, indent=2))

        if transcribe and not entry.get("transcript"):
            entry["transcript"] = str(transcribe_file(entry["local"], pipeline=pipeline))
            MANIFEST.write_text(json.dumps(manifest, indent=2))
            action = "import" if not copied else "transcribe"
        else:
            action = "copy" if not copied else "skip"

        results.append((src, action, entry.get("transcript")))

    return results
