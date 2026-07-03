"""Finalized transcript upload with a durable local retry queue.

This is intentionally separate from Hivemind live streaming. Hivemind mirrors
segments during a session; this module deals with the explicit save/export
moment, queues the finalized transcript payload, and retries without blocking
local persistence.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

QUEUE_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 8.0


class UploadMode(str, Enum):
    """User-visible destination mode for finalized transcript export."""

    LOCAL_ONLY = "local"
    LOCAL_AND_REMOTE = "local_remote"
    REMOTE_ONLY = "remote"

    @classmethod
    def parse(cls, value: str | None) -> "UploadMode":
        raw = (value or cls.LOCAL_ONLY.value).strip().lower()
        aliases = {
            "local-only": cls.LOCAL_ONLY.value,
            "local_only": cls.LOCAL_ONLY.value,
            "local+remote": cls.LOCAL_AND_REMOTE.value,
            "local-remote": cls.LOCAL_AND_REMOTE.value,
            "local_and_remote": cls.LOCAL_AND_REMOTE.value,
            "remote-only": cls.REMOTE_ONLY.value,
            "remote_only": cls.REMOTE_ONLY.value,
        }
        raw = aliases.get(raw, raw)
        try:
            return cls(raw)
        except ValueError:
            return cls.LOCAL_ONLY

    @property
    def saves_local(self) -> bool:
        return self in (self.LOCAL_ONLY, self.LOCAL_AND_REMOTE)

    @property
    def uploads_remote(self) -> bool:
        return self in (self.LOCAL_AND_REMOTE, self.REMOTE_ONLY)

    @property
    def label(self) -> str:
        if self is self.LOCAL_ONLY:
            return "Local only"
        if self is self.LOCAL_AND_REMOTE:
            return "Local + Remote"
        return "Remote only"


@dataclass(frozen=True)
class FlushResult:
    """Summary of one queue flush attempt."""

    attempted: int
    sent: int
    pending: int
    last_error: str = ""

    @property
    def ok(self) -> bool:
        return self.pending == 0 and not self.last_error


def default_queue_path() -> Path:
    try:
        from config import DATA_DIR  # type: ignore

        return Path(DATA_DIR) / ".upload-queue.jsonl"
    except Exception:
        return Path.home() / ".config" / "voxterm" / ".upload-queue.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_upload_record(
    transcript_markdown: str,
    metadata: dict[str, Any] | None = None,
    *,
    include_audio: bool = False,
    audio_path: Path | None = None,
) -> dict[str, Any]:
    """Build one queued upload record.

    Audio is opt-in and only included when a concrete file path is supplied.
    TUI recordings currently do not retain audio by default, so most records
    will carry transcript + metadata only.
    """

    meta = dict(metadata or {})
    audio: dict[str, Any] | None = None
    if include_audio:
        if audio_path and audio_path.exists() and audio_path.is_file():
            audio = {
                "filename": audio_path.name,
                "content_type": "audio/wav"
                if audio_path.suffix.lower() == ".wav"
                else "application/octet-stream",
                "base64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            }
            meta["audio_included"] = True
        else:
            meta["audio_included"] = False
            meta["audio_note"] = "audio upload requested, but no retained audio file was available"

    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "kind": "voxterm.finalized_transcript",
        "id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "attempts": 0,
        "last_error": "",
        "metadata": meta,
        "transcript_markdown": transcript_markdown,
        "audio": audio,
    }


class UploadQueue:
    """Small JSONL queue with atomic rewrite semantics."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_queue_path()
        self._lock = threading.Lock()

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._load_unlocked()

    def enqueue(self, record: dict[str, Any]) -> int:
        with self._lock:
            records = self._load_unlocked()
            records.append(record)
            self._replace_unlocked(records)
            return len(records)

    def replace(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self._replace_unlocked(records)

    def _load_unlocked(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except Exception as exc:
            log.warning("upload queue: could not read %s: %s", self.path, exc)
            return []

        records: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("upload queue: ignoring corrupt JSONL row")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _replace_unlocked(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        body = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        )
        tmp.write_text(body, encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


class TranscriptUploadClient:
    """Queue finalized transcripts and flush them to a configured endpoint."""

    def __init__(
        self,
        *,
        endpoint: str = "",
        auth_token: str = "",
        queue: UploadQueue | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint = endpoint.strip()
        self.auth_token = auth_token.strip()
        self.queue = queue or UploadQueue()
        self.timeout = timeout

    def enqueue(
        self,
        transcript_markdown: str,
        metadata: dict[str, Any] | None = None,
        *,
        include_audio: bool = False,
        audio_path: Path | None = None,
    ) -> int:
        record = build_upload_record(
            transcript_markdown,
            metadata,
            include_audio=include_audio,
            audio_path=audio_path,
        )
        return self.queue.enqueue(record)

    def flush(self) -> FlushResult:
        records = self.queue.load()
        if not records:
            return FlushResult(attempted=0, sent=0, pending=0)
        if not self.endpoint:
            return FlushResult(
                attempted=0,
                sent=0,
                pending=len(records),
                last_error="missing upload endpoint",
            )

        remaining: list[dict[str, Any]] = []
        attempted = 0
        sent = 0
        last_error = ""

        for record in records:
            attempted += 1
            record["attempts"] = int(record.get("attempts") or 0) + 1
            try:
                self._post(record)
                sent += 1
            except Exception as exc:
                last_error = str(exc)
                record["last_error"] = last_error
                remaining.append(record)

        self.queue.replace(remaining)
        return FlushResult(
            attempted=attempted,
            sent=sent,
            pending=len(remaining),
            last_error=last_error,
        )

    def _post(self, record: dict[str, Any]) -> None:
        payload = {
            key: value
            for key, value in record.items()
            if key not in {"attempts", "last_error"}
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "voxterm/finalized-transcript-upload",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                if status < 200 or status >= 300:
                    raise urllib.error.HTTPError(
                        self.endpoint,
                        status,
                        f"unexpected upload status {status}",
                        hdrs=response.headers,
                        fp=None,
                    )
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            raise RuntimeError(exc.reason) from exc
