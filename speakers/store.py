"""Persistent speaker profile storage backed by SQLite.

Stores ECAPA-TDNN embeddings (192-dim float32) as encrypted BLOBs
(AES-256-CBC with HMAC-SHA256, key in macOS Keychain).  Uses WAL mode
for safe concurrent access from the worker thread and the Textual event loop.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import SpeakerMeta, SpeakerProfile
from . import crypto

log = logging.getLogger(__name__)

EMBEDDING_DIM = 192
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32

# Cross-session confidence thresholds
CROSS_SESSION_HIGH_BASE = 0.55   # base threshold for auto-assign
CROSS_SESSION_MEDIUM = 0.35      # below this → unknown
ADAPTIVE_BOOST = 0.15            # extra strictness for new profiles
ADAPTIVE_DECAY_RATE = 10         # how fast the boost decays with samples
CONFLICT_MARGIN = 0.05           # if top-2 are within this, treat as ambiguous
COLD_START_MIN_CONFIRMED = 10    # min confirmed before auto-updates allowed


@dataclass
class MatchResult:
    """Result of cross-session speaker matching."""
    tier: str            # "high", "medium", "low"
    profile_id: str      # best-match profile UUID (empty if low)
    name: str            # profile name (empty if low)
    score: float         # cosine similarity
    color: str           # profile color (empty if low)
    ambiguous: bool      # True if top-2 are too close (conflict)

# Default storage location (not synced by iCloud)
DEFAULT_DB_DIR = Path.home() / "Library" / "Application Support" / "voxterm"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / ".speakers.db"
BACKUP_DIR = DEFAULT_DB_DIR / ".backups"

_SCHEMA_VERSION = 1

_CREATE_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS speakers (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    color           TEXT NOT NULL DEFAULT '',
    centroid        BLOB NOT NULL,
    exemplars       BLOB NOT NULL DEFAULT X'',
    exemplar_count  INTEGER NOT NULL DEFAULT 0,
    confirmed_count       INTEGER NOT NULL DEFAULT 0,
    auto_assigned_count   INTEGER NOT NULL DEFAULT 0,
    total_duration_sec    REAL NOT NULL DEFAULT 0.0,
    quality_score         REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    notes           TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS session_speakers (
    session_id   TEXT NOT NULL,
    speaker_id   TEXT NOT NULL,
    local_id     INTEGER NOT NULL,
    segment_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, speaker_id),
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_speakers_speaker
    ON session_speakers(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speakers_last_seen
    ON speakers(last_seen_at);
"""


class SpeakerStore:
    """Persistent speaker profile storage with in-memory centroid cache."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._enc_key: bytes | None = None  # AES-256 key from Keychain

        # Hot caches — loaded eagerly on open()
        self._centroids: dict[str, np.ndarray] = {}   # profile_id → centroid
        self._profiles: dict[str, SpeakerMeta] = {}    # profile_id → metadata

    # ── lifecycle ────────────────────────────────────────────

    def open(self) -> None:
        """Open (or create) the database and load centroids into memory."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), timeout=5.0, check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

        # Load encryption key from macOS Keychain (auto-creates on first use)
        if crypto.is_available():
            self._enc_key = crypto.get_or_create_key()
            if self._enc_key:
                self._migrate_to_encrypted()

        self._load_all()

        # Set file permissions to owner-only (biometric data)
        try:
            self._db_path.chmod(0o600)
        except OSError:
            pass

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    # ── queries ──────────────────────────────────────────────

    def get_all_profiles(self) -> list[SpeakerMeta]:
        """Return metadata for all stored profiles."""
        with self._lock:
            return list(self._profiles.values())

    def get_profile(self, profile_id: str) -> SpeakerProfile | None:
        """Load a full profile including exemplars (lazy)."""
        if not self._conn:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (profile_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_profile(row)

    def get_profile_names(self) -> dict[str, str]:
        """Return {profile_id: name} for all profiles."""
        with self._lock:
            return {pid: m.name for pid, m in self._profiles.items()}

    def match_profiles(
        self, embedding: np.ndarray, top_k: int = 3
    ) -> list[tuple[str, str, float]]:
        """Match an embedding against all stored centroids.

        Returns [(profile_id, name, cosine_score)] sorted by score desc.
        """
        results = []
        with self._lock:
            if not self._centroids:
                return []
            for pid, centroid in self._centroids.items():
                score = self._cosine_sim(embedding, centroid)
                name = self._profiles[pid].name if pid in self._profiles else ""
                results.append((pid, name, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def classify_match(self, embedding: np.ndarray) -> MatchResult:
        """Match an embedding against profiles and classify confidence.

        Uses multi-centroid matching for profiles with >= 15 exemplars.
        Returns a MatchResult with tier="high", "medium", or "low".
        """
        matches = self.match_profiles(embedding, top_k=3)
        if not matches:
            return MatchResult("low", "", "", 0.0, "", False)

        best_pid, best_name, best_score = matches[0]

        # Look up profile metadata for adaptive threshold
        with self._lock:
            meta = self._profiles.get(best_pid)
        if not meta:
            return MatchResult("low", "", "", 0.0, "", False)

        # Refine score with multi-centroid matching if profile is mature
        num_samples = meta.confirmed_count + meta.auto_assigned_count
        if num_samples >= 15:
            profile = self.get_profile(best_pid)
            if profile and profile.sub_centroids:
                multi_score = profile.best_match_score(embedding)
                best_score = max(best_score, multi_score)

        # Adaptive high threshold: stricter with fewer samples
        effective_high = CROSS_SESSION_HIGH_BASE + ADAPTIVE_BOOST * math.exp(
            -num_samples / ADAPTIVE_DECAY_RATE
        )

        # Check for conflict: are the top two matches too close?
        ambiguous = False
        if len(matches) >= 2:
            second_score = matches[1][2]
            if best_score - second_score < CONFLICT_MARGIN:
                ambiguous = True

        # Classify
        if best_score >= effective_high and not ambiguous:
            tier = "high"
        elif best_score >= CROSS_SESSION_MEDIUM:
            tier = "medium"
        else:
            tier = "low"

        if tier == "low":
            return MatchResult("low", "", "", best_score, "", False)

        return MatchResult(
            tier=tier,
            profile_id=best_pid,
            name=best_name,
            score=best_score,
            color=meta.color,
            ambiguous=ambiguous,
        )

    def is_profile_mature(self, profile_id: str) -> bool:
        """Check if a profile has enough confirmed samples for auto-updates."""
        with self._lock:
            meta = self._profiles.get(profile_id)
        if not meta:
            return False
        return meta.confirmed_count >= COLD_START_MIN_CONFIRMED

    # ── mutations ────────────────────────────────────────────

    def create_profile(
        self,
        name: str,
        color: str,
        embeddings: list[np.ndarray],
        durations: list[float] | None = None,
    ) -> str:
        """Create a new speaker profile from a set of embeddings.

        Returns the new profile UUID.
        """
        if not self._conn:
            raise RuntimeError("SpeakerStore not open")
        if not embeddings:
            raise ValueError("At least one embedding is required to create a profile")

        profile_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Compute centroid from embeddings
        stacked = np.stack(embeddings)
        centroid = stacked.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm

        total_dur = sum(durations) if durations else 0.0
        max_exemplars = 20
        exemplar_blob = self._embeddings_to_blob(embeddings[:max_exemplars])
        centroid_blob = self._centroid_to_blob(centroid)

        with self._lock:
            self._conn.execute(
                """INSERT INTO speakers
                   (id, name, color, centroid, exemplars, exemplar_count,
                    confirmed_count, total_duration_sec,
                    created_at, updated_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id, name, color,
                    centroid_blob, exemplar_blob, len(embeddings),
                    len(embeddings), total_dur,
                    now, now, now,
                ),
            )
            self._conn.commit()

            # Update caches
            self._centroids[profile_id] = centroid
            self._profiles[profile_id] = SpeakerMeta(
                id=profile_id, name=name, color=color,
                confirmed_count=len(embeddings), auto_assigned_count=0,
                total_duration_sec=total_dur, quality_score=0.0,
                created_at=now, updated_at=now, last_seen_at=now,
            )

        return profile_id

    def update_profile_embedding(
        self,
        profile_id: str,
        embedding: np.ndarray,
        duration: float = 0.0,
        confirmed: bool = True,
    ) -> None:
        """Add an embedding to an existing profile and update its centroid."""
        if not self._conn:
            return

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (profile_id,)
            ).fetchone()
            if not row:
                return
            profile = self._row_to_profile(row)

            profile.add_exemplar(embedding)
            if confirmed:
                profile.confirmed_count += 1
            else:
                profile.auto_assigned_count += 1
            profile.total_duration_sec += duration
            profile.recompute_centroid()
            profile.quality_score = profile.compute_quality()

            now = datetime.now().isoformat()
            exemplar_blob = self._embeddings_to_blob(profile.exemplars)

            self._conn.execute(
                """UPDATE speakers SET
                       centroid = ?, exemplars = ?, exemplar_count = ?,
                       confirmed_count = ?, auto_assigned_count = ?,
                       total_duration_sec = ?, quality_score = ?,
                       updated_at = ?, last_seen_at = ?
                   WHERE id = ?""",
                (
                    self._centroid_to_blob(profile.centroid),
                    exemplar_blob, len(profile.exemplars),
                    profile.confirmed_count, profile.auto_assigned_count,
                    profile.total_duration_sec, profile.quality_score,
                    now, now,
                    profile_id,
                ),
            )
            self._conn.commit()

            self._centroids[profile_id] = profile.centroid.copy()
            if profile_id in self._profiles:
                meta = self._profiles[profile_id]
                meta.confirmed_count = profile.confirmed_count
                meta.auto_assigned_count = profile.auto_assigned_count
                meta.total_duration_sec = profile.total_duration_sec
                meta.quality_score = profile.quality_score
                meta.updated_at = now
                meta.last_seen_at = now

    def rename_profile(self, profile_id: str, name: str) -> None:
        """Rename a speaker profile."""
        if not self._conn:
            return
        now = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE speakers SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, profile_id),
            )
            self._conn.commit()
            if profile_id in self._profiles:
                self._profiles[profile_id].name = name
                self._profiles[profile_id].updated_at = now

    def update_color(self, profile_id: str, color: str) -> None:
        """Update a speaker's persistent color."""
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE speakers SET color = ? WHERE id = ?",
                (color, profile_id),
            )
            self._conn.commit()
            if profile_id in self._profiles:
                self._profiles[profile_id].color = color

    def delete_profile(self, profile_id: str) -> None:
        """Permanently remove a speaker profile (VACUUMs to scrub BLOB data)."""
        if not self._conn:
            return
        with self._lock:
            self._conn.execute("DELETE FROM speakers WHERE id = ?", (profile_id,))
            self._conn.execute(
                "DELETE FROM session_speakers WHERE speaker_id = ?", (profile_id,)
            )
            self._conn.commit()
            self._conn.execute("VACUUM")
            self._centroids.pop(profile_id, None)
            self._profiles.pop(profile_id, None)

    def merge_profiles(self, source_id: str, target_id: str) -> None:
        """Merge source profile into target. Source is deleted."""
        if not self._conn:
            return

        with self._lock:
            source_row = self._conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (source_id,)
            ).fetchone()
            target_row = self._conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (target_id,)
            ).fetchone()
            if not source_row or not target_row:
                return

            source = self._row_to_profile(source_row)
            target = self._row_to_profile(target_row)

            for emb in source.exemplars:
                target.add_exemplar(emb)
            target.confirmed_count += source.confirmed_count
            target.auto_assigned_count += source.auto_assigned_count
            target.total_duration_sec += source.total_duration_sec
            target.recompute_centroid()

            now = datetime.now().isoformat()
            exemplar_blob = self._embeddings_to_blob(target.exemplars)

            self._conn.execute(
                """UPDATE speakers SET
                       centroid = ?, exemplars = ?, exemplar_count = ?,
                       confirmed_count = ?, auto_assigned_count = ?,
                       total_duration_sec = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    self._centroid_to_blob(target.centroid),
                    exemplar_blob, len(target.exemplars),
                    target.confirmed_count, target.auto_assigned_count,
                    target.total_duration_sec, now,
                    target_id,
                ),
            )
            # Reassign session mappings (delete conflicts first)
            self._conn.execute(
                """DELETE FROM session_speakers WHERE speaker_id = ? AND session_id IN
                   (SELECT session_id FROM session_speakers WHERE speaker_id = ?)""",
                (target_id, source_id),
            )
            self._conn.execute(
                "UPDATE session_speakers SET speaker_id = ? WHERE speaker_id = ?",
                (target_id, source_id),
            )
            self._conn.execute("DELETE FROM speakers WHERE id = ?", (source_id,))
            self._conn.commit()

            self._centroids[target_id] = target.centroid.copy()
            self._centroids.pop(source_id, None)
            self._profiles.pop(source_id, None)
            if target_id in self._profiles:
                meta = self._profiles[target_id]
                meta.confirmed_count = target.confirmed_count
                meta.auto_assigned_count = target.auto_assigned_count
                meta.total_duration_sec = target.total_duration_sec
                meta.updated_at = now

    # ── session tracking ─────────────────────────────────────

    def record_session_speaker(
        self, session_id: str, speaker_id: str, local_id: int, segment_count: int = 0
    ) -> None:
        """Record that a speaker appeared in a session."""
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO session_speakers
                   (session_id, speaker_id, local_id, segment_count)
                   VALUES (?, ?, ?, ?)""",
                (session_id, speaker_id, local_id, segment_count),
            )
            self._conn.commit()

    # ── export / import / delete ─────────────────────────────

    def export_db(self, output_path: Path) -> None:
        """Export all speaker profiles as a portable SQLite file."""
        if not self._conn:
            return
        dst = sqlite3.connect(str(output_path))
        self._conn.backup(dst)
        dst.close()

    def import_db(self, input_path: Path, merge: bool = True) -> None:
        """Import profiles from an exported database.

        If merge=True, skip profiles whose ID already exists.
        If merge=False, overwrite everything.
        """
        if not self._conn:
            return
        src = sqlite3.connect(str(input_path))
        try:
            rows = src.execute("SELECT * FROM speakers").fetchall()
            with self._lock:
                for row in rows:
                    pid = row[0]
                    if merge:
                        existing = self._conn.execute(
                            "SELECT id FROM speakers WHERE id = ?", (pid,)
                        ).fetchone()
                        if existing:
                            continue
                    placeholders = ",".join(["?"] * len(row))
                    self._conn.execute(
                        f"INSERT OR REPLACE INTO speakers VALUES ({placeholders})",
                        row,
                    )
                self._conn.commit()
                self._load_all()
        finally:
            src.close()

    def delete_all_data(self) -> None:
        """Permanently delete all voice profiles and backups."""
        with self._lock:
            if self._conn:
                self._conn.execute("DELETE FROM session_speakers")
                self._conn.execute("DELETE FROM speakers")
                self._conn.commit()
                self._conn.execute("VACUUM")  # scrub encrypted bytes from free pages
                self._centroids.clear()
                self._profiles.clear()

        # Remove backup files (outside lock — filesystem ops)
        if BACKUP_DIR.exists():
            for f in BACKUP_DIR.glob("speakers_*.db"):
                try:
                    f.unlink()
                except OSError:
                    pass

    # ── backup ───────────────────────────────────────────────

    def backup(self) -> None:
        """Create a daily backup of the speaker database."""
        if not self._conn:
            return
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            backup_path = BACKUP_DIR / f"speakers_{today}.db"
            if backup_path.exists():
                return  # already backed up today

            dst = sqlite3.connect(str(backup_path))
            self._conn.backup(dst)
            dst.close()

            # Prune old backups, keep last 7
            backups = sorted(BACKUP_DIR.glob("speakers_*.db"))
            for old in backups[:-7]:
                old.unlink()
        except Exception:
            pass  # backup must never block the app

    # ── internals ────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        if not self._conn:
            return
        # Check if tables exist
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "speakers" not in tables:
            self._conn.executescript(_CREATE_SQL)
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            self._conn.commit()

    def _load_all(self) -> None:
        """Eagerly load all centroids and metadata into memory."""
        if not self._conn:
            return
        rows = self._conn.execute(
            """SELECT id, name, color, centroid,
                      confirmed_count, auto_assigned_count,
                      total_duration_sec, quality_score,
                      created_at, updated_at, last_seen_at
               FROM speakers"""
        ).fetchall()

        self._centroids.clear()
        self._profiles.clear()

        for row in rows:
            pid = row[0]
            centroid_blob = row[3]
            centroid = self._blob_to_centroid(centroid_blob)
            if centroid is not None:
                self._centroids[pid] = centroid

            self._profiles[pid] = SpeakerMeta(
                id=pid,
                name=row[1],
                color=row[2],
                confirmed_count=row[4],
                auto_assigned_count=row[5],
                total_duration_sec=row[6],
                quality_score=row[7],
                created_at=row[8],
                updated_at=row[9],
                last_seen_at=row[10],
            )

    def _row_to_profile(self, row: tuple) -> SpeakerProfile:
        """Convert a full DB row to a SpeakerProfile."""
        # Row order matches SELECT * FROM speakers
        (
            pid, name, color, centroid_blob, exemplars_blob, exemplar_count,
            confirmed_count, auto_assigned_count, total_duration_sec,
            quality_score, created_at, updated_at, last_seen_at,
            _tags, _notes,
        ) = row

        centroid = self._blob_to_centroid(centroid_blob)
        if centroid is None:
            centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        exemplars = self._blob_to_embeddings(exemplars_blob)

        return SpeakerProfile(
            id=pid,
            name=name,
            color=color,
            centroid=centroid,
            exemplars=exemplars,
            confirmed_count=confirmed_count,
            auto_assigned_count=auto_assigned_count,
            total_duration_sec=total_duration_sec,
            quality_score=quality_score,
            created_at=created_at,
            updated_at=updated_at,
            last_seen_at=last_seen_at,
        )

    def _encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt a BLOB if encryption is enabled."""
        if self._enc_key and plaintext:
            return crypto.encrypt_blob(self._enc_key, plaintext)
        return plaintext

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt a BLOB if encryption is enabled and data is encrypted."""
        if not data:
            return data
        if self._enc_key and crypto.is_encrypted(data):
            return crypto.decrypt_blob(self._enc_key, data)
        return data  # unencrypted (legacy or encryption disabled)

    def _embeddings_to_blob(self, embeddings: list[np.ndarray]) -> bytes:
        """Pack embeddings into an (optionally encrypted) BLOB."""
        if not embeddings:
            return b""
        raw = np.stack(embeddings).astype(np.float32).tobytes()
        return self._encrypt(raw)

    def _blob_to_embeddings(self, blob: bytes) -> list[np.ndarray]:
        """Unpack a BLOB into a list of 192-dim embeddings."""
        if not blob:
            return []
        raw = self._decrypt(blob)
        if len(raw) < EMBEDDING_BYTES:
            return []
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        n = len(arr) // EMBEDDING_DIM
        return [arr[i * EMBEDDING_DIM : (i + 1) * EMBEDDING_DIM] for i in range(n)]

    def _centroid_to_blob(self, centroid: np.ndarray) -> bytes:
        """Serialize a centroid into an (optionally encrypted) BLOB."""
        return self._encrypt(centroid.astype(np.float32).tobytes())

    def _blob_to_centroid(self, blob: bytes) -> np.ndarray | None:
        """Deserialize a centroid BLOB."""
        if not blob:
            return None
        raw = self._decrypt(blob)
        if len(raw) != EMBEDDING_BYTES:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def _migrate_to_encrypted(self) -> None:
        """Migrate existing unencrypted BLOBs to encrypted format."""
        if not self._conn or not self._enc_key:
            return
        rows = self._conn.execute(
            "SELECT id, centroid, exemplars FROM speakers"
        ).fetchall()
        migrated = 0
        for pid, centroid_blob, exemplars_blob in rows:
            needs_update = False
            new_centroid = centroid_blob
            new_exemplars = exemplars_blob

            if centroid_blob and not crypto.is_encrypted(centroid_blob):
                new_centroid = crypto.encrypt_blob(self._enc_key, centroid_blob)
                needs_update = True
            if exemplars_blob and not crypto.is_encrypted(exemplars_blob):
                new_exemplars = crypto.encrypt_blob(self._enc_key, exemplars_blob)
                needs_update = True

            if needs_update:
                self._conn.execute(
                    "UPDATE speakers SET centroid = ?, exemplars = ? WHERE id = ?",
                    (new_centroid, new_exemplars, pid),
                )
                migrated += 1
        if migrated:
            self._conn.commit()
            log.info("Encrypted %d speaker profile(s)", migrated)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
