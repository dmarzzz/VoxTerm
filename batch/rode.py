"""Import recordings from Rode USB recorders."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import DATA_DIR, SESSIONS_DIR
from .core import transcribe_file as core_transcribe_file

RODE_PATTERNS = ("*_Wireless_GO.WAV", "*_Wireless_GO.wav")
DEFAULT_IMPORT_DIR = DATA_DIR / "imports" / "rode"
DEFAULT_MANIFEST = DATA_DIR / ".rode_imports.json"


@dataclass(frozen=True)
class ImportResult:
    source_path: Path
    local_path: Path
    action: str
    sha256: str
    transcript_path: Path | None = None


def default_mount_roots() -> list[Path]:
    user = os.environ.get("USER", "")
    roots = [Path("/media") / user, Path("/run/media") / user, Path("/Volumes")]
    return [root for root in roots if root.is_dir()]


def find_rode_recordings(roots: Iterable[str | Path] | None = None) -> list[Path]:
    """Find Rode Wireless GO recording files under mounted volumes."""

    found: set[Path] = set()
    for root_value in roots or default_mount_roots():
        root = Path(root_value)
        if not root.is_dir():
            continue
        candidates = [root]
        try:
            candidates.extend(child for child in root.iterdir() if child.is_dir())
        except OSError:
            continue
        for candidate in candidates:
            for pattern in RODE_PATTERNS:
                found.update(path for path in candidate.glob(pattern) if path.is_file())
    return sorted(found, key=lambda path: (path.name, str(path)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _copy_destination(import_dir: Path, source: Path, digest: str) -> Path:
    dest = import_dir / source.name
    if dest.exists():
        dest = import_dir / f"{digest[:8]}_{source.name}"
    return dest


def import_recordings(
    *,
    roots: Iterable[str | Path] | None = None,
    import_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    transcribe: bool = True,
    model: str | None = None,
    language: str = "en",
    diarize: bool = True,
) -> list[ImportResult]:
    """Copy new Rode recordings and optionally transcribe them.

    Rode Wireless GO devices expose read-only mass storage and unreliable
    timestamps, so sync is copy-down only and deduped by sha256.
    """

    import_root = Path(import_dir) if import_dir is not None else DEFAULT_IMPORT_DIR
    manifest_file = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    transcript_dir = Path(out_dir) if out_dir is not None else SESSIONS_DIR
    manifest = _load_manifest(manifest_file)
    results: list[ImportResult] = []

    for source in find_rode_recordings(roots):
        digest = sha256_file(source)
        entry = manifest.get(digest)
        copied_now = False

        if not isinstance(entry, dict):
            import_root.mkdir(parents=True, exist_ok=True)
            local = _copy_destination(import_root, source, digest)
            shutil.copy2(source, local)
            copied_now = True
            entry = {
                "source_name": source.name,
                "size": source.stat().st_size,
                "sha256": digest,
                "imported_at": datetime.now().isoformat(timespec="seconds"),
                "local_path": str(local),
                "transcript_path": "",
            }
            manifest[digest] = entry
            _save_manifest(manifest_file, manifest)

        local_path = Path(entry["local_path"])
        transcript_path = Path(entry["transcript_path"]) if entry.get("transcript_path") else None

        if transcribe and transcript_path is None:
            result = core_transcribe_file(
                local_path,
                out_dir=transcript_dir,
                model=model,
                language=language,
                diarize=diarize,
            )
            transcript_path = Path(result["transcript_path"])
            entry["transcript_path"] = str(transcript_path)
            _save_manifest(manifest_file, manifest)
            action = "import" if copied_now else "transcribe"
        else:
            action = "copy" if copied_now else "skip"

        results.append(
            ImportResult(
                source_path=source,
                local_path=local_path,
                action=action,
                sha256=digest,
                transcript_path=transcript_path,
            )
        )

    return results
