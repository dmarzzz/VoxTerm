"""Append-only JSONL event log for out-of-process consumers.

VoxTerm's core stays in-process; this module exposes a tiny stream of
structured events so companion tools (LED matrices, OBS overlays, web
dashboards, etc.) can subscribe without VoxTerm having to know what they
do. One JSON object per line, written to a session-scoped file under
LIVE_DIR. Consumers tail the file.

Disabled by default — set VOXTERM_EVENTS=1 to enable. When disabled this
module is a no-op (the EventLogger never opens a file).

Event schema:
    {"t": <unix_seconds>, "kind": "<name>", ...fields}

Kinds emitted by VoxTerm today (see call sites in tui/app.py):
    text       — finalized transcription      fields: speaker, speaker_id, text, confidence, overlap
    peer_text  — peer transcript (party mode) fields: peer, speaker, text
    speaker    — speaker change               fields: speaker_id, label
    vad        — VAD on/off transition        fields: on (bool)
    amplitude  — chunk RMS (~15 Hz)           fields: rms (float 0..1)
    recording  — recording toggle             fields: on (bool)
    party      — party-session join/leave     fields: on (bool)
    session    — session start/end            fields: phase ("start"|"end")
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import IO, Optional

log = logging.getLogger(__name__)


class EventLogger:
    """Thread-safe append-only JSONL writer. Never raises into the caller —
    failures to write are logged once and then silently dropped, so a broken
    consumer or full disk can never take down a recording session."""

    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._fp: Optional[IO[str]] = None
        self._lock = threading.Lock()
        self._failed_once = False

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def open(self) -> None:
        if self._path is None or self._fp is not None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered append; tail-style consumers see each event as
            # soon as the newline lands.
            self._fp = open(self._path, "a", buffering=1, encoding="utf-8")
        except Exception as e:
            log.warning("event log: could not open %s — disabling: %s", self._path, e)
            self._fp = None
            self._path = None  # latch-disable; don't retry every event

    def emit(self, kind: str, **fields) -> None:
        if self._fp is None:
            return
        record = {"t": round(time.time(), 4), "kind": kind, **fields}
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as e:
            # Non-serialisable field — log once and drop the event.
            if not self._failed_once:
                log.warning("event log: dropping un-serialisable event %r: %s", kind, e)
                self._failed_once = True
            return
        with self._lock:
            try:
                self._fp.write(line + "\n")
            except Exception as e:
                if not self._failed_once:
                    log.warning("event log: write failed — disabling: %s", e)
                    self._failed_once = True
                try:
                    self._fp.close()
                except Exception:
                    pass
                self._fp = None

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:
                    pass
                self._fp = None


class NullEventLogger(EventLogger):
    """No-op variant used when VOXTERM_EVENTS is unset. Saves a branch in
    every call site at the cost of two extra Python class instances."""

    def __init__(self) -> None:
        super().__init__(path=None)

    def open(self) -> None: ...
    def emit(self, kind: str, **fields) -> None: ...
    def close(self) -> None: ...
