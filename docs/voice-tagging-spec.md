# Voice Tagging & Progressive Speaker Recognition

> Feature Specification for VoxTerm
> Draft: 2026-03-20

---

## Table of Contents

1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Architecture](#3-architecture)
4. [Technical Specification](#4-technical-specification)
5. [Data Model & Storage](#5-data-model--storage)
6. [Recognition Pipeline](#6-recognition-pipeline)
7. [Progressive Learning System](#7-progressive-learning-system)
8. [UX Design](#8-ux-design)
9. [Roadmap & Implementation Plan](#9-roadmap--implementation-plan)
10. [Risks & Mitigations](#10-risks--mitigations)
11. [Appendix](#11-appendix)

---

## 1. Overview

### Problem

VoxTerm's diarization is ephemeral. Every session starts from scratch — speakers are labeled "Speaker 1", "Speaker 2", etc., with no memory of who they are. Users who regularly transcribe meetings with the same people must mentally map anonymous labels to real names every session.

### Solution

A **voice tagging and progressive speaker recognition** system that:

1. Lets users **manually tag** speakers by name during or after a session
2. **Persists** voice profiles (ECAPA-TDNN embeddings) across sessions
3. **Automatically recognizes** returning speakers with increasing accuracy
4. **Learns progressively** — tag a lot at first, less and less over time

### Design Principles

- **Local-first**: All voice profiles stored on-device, never transmitted
- **Non-disruptive**: The existing diarization hot path stays fast; persistence is additive
- **Graceful degradation**: If the profile DB is unavailable, fall back to current ephemeral behavior
- **User-initiated enrollment**: Profiles are only created by explicit user action, never automatically

---

## 2. Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | User can name/rename any detected speaker via a keyboard-driven modal | P0 |
| FR-2 | Speaker names persist across sessions and update all transcript entries | P0 |
| FR-3 | Voice embeddings are stored per-profile and used for cross-session recognition | P0 |
| FR-4 | System auto-recognizes returning speakers with confidence indicators | P0 |
| FR-5 | Three-tier confidence display: confirmed, suggested (?), unknown | P0 |
| FR-6 | User can merge two speaker clusters that represent the same person | P1 |
| FR-7 | User can manage a speaker profile library (view, edit, delete) | P1 |
| FR-8 | Speaker colors persist across sessions for recognized profiles | P1 |
| FR-9 | System prompts for confirmation on medium-confidence matches (active learning) | P1 |
| FR-10 | User can correct mis-identifications and the system updates accordingly | P1 |
| FR-11 | Profiles can be exported/imported for backup or device transfer | P2 |
| FR-12 | "Delete all voice data" wipes all biometric data completely | P2 |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Cross-session matching adds < 5ms to the per-chunk pipeline | < 5ms |
| NFR-2 | Profile DB supports up to 500 speakers without degradation | 500 speakers |
| NFR-3 | Startup time increase from loading profiles | < 50ms |
| NFR-4 | Profile storage footprint per speaker (centroid + 20 exemplars) | ~ 16 KB |
| NFR-5 | Total DB size for 100 speakers | < 2 MB |
| NFR-6 | No new Python dependencies (SQLite is stdlib) | 0 new deps |
| NFR-7 | Graceful fallback if DB is corrupt/missing | Ephemeral mode |

---

## 3. Architecture

### High-Level Data Flow

```
                          ┌─────────────────────────────┐
                          │    ~/Library/Application     │
                          │    Support/voxterm/          │
                          │    .speakers.db (SQLite)     │
                          └──────────┬──────────────────┘
                                     │
                            load on startup
                                     │
                                     ▼
┌──────────┐   audio    ┌──────────────────────┐  embedding   ┌───────────────────┐
│ Audio    │ ────────► │  DiarizationEngine    │ ──────────► │  SpeakerStore     │
│ Capture  │            │  (ECAPA-TDNN)         │  + metadata  │  (SQLite + cache) │
└──────────┘            │                       │              │                   │
                        │  identify()           │  ◄────────── │  match_profiles() │
                        │    + session clusters  │  profile     │                   │
                        │    + raw embeddings    │  matches     └───────────────────┘
                        └──────────┬────────────┘                       │
                                   │                                    │
                          (label, id, confidence)                       │
                                   │                                    │
                                   ▼                                    │
                        ┌──────────────────────┐               user tags│
                        │  TranscriptPanel     │ ◄─────────────────────┘
                        │  (display + tagging)  │   T key → modal
                        └──────────────────────┘
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| `DiarizationEngine` | In-session clustering (unchanged hot path) + per-segment embedding retention |
| `SpeakerStore` | SQLite persistence, centroid cache, profile CRUD, cross-session matching |
| `SpeakerTagScreen` | Modal for naming session speakers (new) |
| `SpeakerProfileScreen` | Modal for managing persistent profiles (new) |
| `TranscriptPanel` | Display with confidence indicators, re-render on rename |

### Threading Model

```
Worker Thread (@work)              Main Thread (Textual event loop)
─────────────────────              ─────────────────────────────────
identify(audio)
  → compute embedding
  → in-session clustering
  → retain raw embedding
  → cross-session matching
  → SQLite write (WAL)
  → call_from_thread(result) ────► update transcript display
                                   user presses T
                                     → SpeakerTagScreen
                                     → SpeakerStore.rename() (SQLite write)
                                     → TranscriptPanel.rename_speaker()
```

SQLite WAL mode allows concurrent reads (main thread) with a single writer (worker thread). A `threading.Lock` guards the in-memory centroid cache for multi-step read-modify-write operations.

---

## 4. Technical Specification

### 4.1 Per-Segment Embedding Retention

**This is the most critical change to the existing system.**

Currently, `identify()` computes an embedding and immediately merges it into the session centroid via EMA, discarding the raw embedding. The new system must retain per-segment embeddings for:
- Retroactive enrollment when a user tags a speaker
- Exemplar storage in persistent profiles
- Quality filtering
- Error correction (remove bad embeddings)

**Change to DiarizationEngine:**

```python
class DiarizationEngine:
    def __init__(self):
        # ... existing fields ...
        self._segment_embeddings: dict[int, list[tuple[np.ndarray, float]]] = {}
        #                         speaker_id → [(embedding, duration_sec), ...]
```

`identify()` appends `(embedding, duration)` to the speaker's list before the EMA update. Memory impact: ~800 bytes per segment, ~100 segments/session = ~80 KB. Cleared on `reset_session()`.

### 4.2 Cross-Session Matching

After in-session clustering **stabilizes** for a speaker, compare the session centroid against persistent profiles.

**Stabilization definition:** A session speaker is considered stable when:
- At least **3 segments** have been attributed to them, AND
- The cosine distance between the centroid after the last two segments is **< 0.05**

This prevents matching against noisy, early-session centroids that haven't converged.

### 4.3 Confidence Thresholds

| Tier | Cosine Similarity | Behavior | Display |
|------|-------------------|----------|---------|
| **HIGH** | >= `effective_high_threshold` | Auto-assign name silently | `Alice` (no decoration) |
| **MEDIUM** | >= 0.35 and < HIGH | Suggest name, prompt for confirmation | `Alice?` |
| **LOW** | < 0.35 | Unknown speaker | `Speaker 3` |

**Adaptive high threshold** (more conservative with fewer samples):

```
effective_high = 0.55 + 0.15 * exp(-num_embeddings / 10)
```

| Profile samples | Effective HIGH threshold |
|-----------------|------------------------|
| 1 | 0.686 |
| 5 | 0.641 |
| 10 | 0.605 |
| 20 | 0.570 |
| 30+ | ~0.555 |

This is intentionally slower to decay than the original proposal (changed from `/5` to `/10` per validation feedback) — new profiles must earn trust.

### 4.4 Profile Conflict Resolution

When multiple persistent profiles match a session speaker above HIGH threshold:
1. Use the **highest-scoring match**
2. If the top two scores are within **0.05** of each other, treat as MEDIUM confidence instead — prompt the user: `"Alice or Bob?"`
3. This prompt counts against the fatigue budget

### 4.5 Enrollment Strategy

**During accumulation (< 20 exemplars):** Running mean centroid
```python
centroid = (centroid * count + new_embedding) / (count + 1)
```

**Post-maturity (>= 20 exemplars):** EMA with alpha = 0.98
```python
centroid = 0.98 * centroid + 0.02 * new_embedding
```

**Exemplar management (max 20):**
- If < 20 exemplars: append
- If full: replace the exemplar with highest cosine similarity to centroid (most redundant)
- This maintains diversity across the speaker's voice variation

**Quality filters before accepting an embedding:**

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Duration | >= 3.0s speech | Sub-2.6s produces unreliable embeddings |
| RMS energy | >= 0.005 | Reject near-silence (existing threshold) |
| Single-speaker | No overlap detected | Overlapping speech blends embeddings |

> Note: SNR filtering was descoped. The RMS threshold combined with the duration filter provides sufficient quality gating without requiring a VAD-based SNR estimator.

### 4.6 Multi-Centroid Matching

When a profile has **>= 15 exemplars**, compute up to K=3 sub-centroids via k-means on stored exemplars. Re-cluster every 10 new embeddings (not 5 — stabilized per validation).

```python
match_score = max(cosine_sim(new_emb, c) for c in profile.sub_centroids)
```

This captures intra-speaker variation (calm vs animated, morning vs evening, different mics).

For profiles with < 15 exemplars, use single-centroid matching (current behavior).

### 4.7 Cold-Start Protection

During cold start (profile has < 10 confirmed exemplars):
- **Only user-confirmed embeddings** update the profile
- Auto-assigned embeddings are displayed but NOT added to the profile
- This prevents centroid poisoning from early mis-identifications

After 10 confirmed exemplars, auto-assigned HIGH-confidence embeddings may also update the profile.

---

## 5. Data Model & Storage

### 5.1 Storage Location

```
~/Library/Application Support/voxterm/
    .speakers.db              # SQLite WAL-mode database
    .speakers.db-wal          # WAL file (auto-managed)
    .speakers.db-shm          # shared memory (auto-managed)

~/Library/Application Support/voxterm/.backups/
    speakers_2026-03-20.db    # daily backup, keep last 7
```

**Why `~/Library/Application Support/` instead of `~/Documents/voxterm/`:**
- Not synced by iCloud (avoids WAL journal conflicts)
- Standard macOS location for application data
- Transcripts remain in `~/Documents/voxterm/` (user-visible output)
- Profiles are app data, not user documents

### 5.2 SQLite Schema

```sql
-- Schema version for migrations
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
);
INSERT INTO schema_version VALUES (1);

-- Speaker profiles
CREATE TABLE speakers (
    id              TEXT PRIMARY KEY,           -- UUID4
    name            TEXT NOT NULL DEFAULT '',
    color           TEXT NOT NULL DEFAULT '',   -- hex color, e.g. "#00ffcc"

    -- Embeddings (raw float32 bytes)
    centroid        BLOB NOT NULL,             -- 768 bytes (192 * float32)
    exemplars       BLOB NOT NULL DEFAULT X'', -- N * 768 bytes
    exemplar_count  INTEGER NOT NULL DEFAULT 0,

    -- Enrollment quality
    confirmed_count       INTEGER NOT NULL DEFAULT 0,
    auto_assigned_count   INTEGER NOT NULL DEFAULT 0,
    total_duration_sec    REAL NOT NULL DEFAULT 0.0,
    quality_score         REAL NOT NULL DEFAULT 0.0,

    -- Timestamps (ISO 8601)
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,

    -- User metadata
    tags            TEXT NOT NULL DEFAULT '[]',  -- JSON array
    notes           TEXT NOT NULL DEFAULT ''
);

-- Session-to-speaker mapping
CREATE TABLE session_speakers (
    session_id   TEXT NOT NULL,     -- e.g. "2026-03-17_100728"
    speaker_id   TEXT NOT NULL,     -- FK to speakers.id
    local_id     INTEGER NOT NULL,  -- session-local int (1, 2, 3...)
    segment_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, speaker_id),
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
);

CREATE INDEX idx_session_speakers_speaker ON session_speakers(speaker_id);
CREATE INDEX idx_speakers_last_seen ON speakers(last_seen_at);
```

### 5.3 Embedding BLOB Format

- **Centroid:** `embedding.astype(np.float32).tobytes()` → 768 bytes
- **Exemplars:** `np.stack(exemplars).astype(np.float32).tobytes()` → `N * 768` bytes
- **Read back:** `np.frombuffer(blob, dtype=np.float32).reshape(-1, 192)`

Raw bytes — no base64, no JSON encoding. 5x more compact than JSON representation.

### 5.4 SpeakerStore Class

```python
class SpeakerStore:
    """Persistent speaker profile storage backed by SQLite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._centroids: dict[str, np.ndarray] = {}   # hot cache: uuid → centroid
        self._profiles: dict[str, SpeakerMeta] = {}    # metadata cache
        self._lock = threading.Lock()                   # guards cache mutations

    # Lifecycle
    def open(self) -> None              # connect, migrate, load centroids
    def close(self) -> None             # commit, close connection

    # Queries
    def match_profiles(self, embedding: np.ndarray) -> list[tuple[str, str, float]]
        # Returns [(profile_id, name, score)] sorted by score desc

    # Mutations
    def create_profile(self, name: str, embeddings: list[np.ndarray], ...) -> str
    def update_profile(self, profile_id: str, embedding: np.ndarray, ...) -> None
    def rename_profile(self, profile_id: str, name: str) -> None
    def merge_profiles(self, source_id: str, target_id: str) -> None
    def delete_profile(self, profile_id: str) -> None  # includes VACUUM

    # Export/Import
    def export_db(self, output_path: Path) -> None
    def import_db(self, input_path: Path, merge: bool = True) -> None

    # Maintenance
    def backup(self) -> None            # daily backup via sqlite3 backup API
    def get_all_profiles(self) -> list[SpeakerMeta]  # for profile management UI
```

### 5.5 Concurrency

- `threading.Lock` around `_centroids` and `_profiles` cache reads/writes
- SQLite WAL mode: `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`
- Connection timeout: 5 seconds (handles rare write contention)
- File permissions: `0o600` on creation

### 5.6 Backup & Recovery

- On startup: `SpeakerStore.backup()` creates a daily snapshot via `sqlite3.Connection.backup()`
- Keep last 7 daily backups, prune older
- If `.speakers.db` fails to open → log warning, fall back to ephemeral mode
- Schema migrations via `schema_version` table with `ALTER TABLE ADD COLUMN` pattern

### 5.7 Privacy

- Embeddings are biometric data — stored with `0600` permissions
- `delete_profile()` calls `VACUUM` to scrub deleted BLOB bytes from disk
- "Delete all voice data" → `os.unlink(db_path)` + remove backups
- Export produces a standalone `.db` file for device transfer

---

## 6. Recognition Pipeline

### 6.1 Session Startup

```
App starts
  → SpeakerStore.open()
  → Load all centroids into memory (~75KB for 100 profiles)
  → DiarizationEngine.load() (ECAPA-TDNN model)
  → Begin recording
```

`SpeakerStore.open()` must complete **before** the diarizer starts processing audio.

### 6.2 Per-Chunk Flow (Hot Path)

```
Audio chunk arrives (every few seconds)
  │
  ├─ DiarizationEngine.identify(audio)
  │    ├─ Extract 192-dim ECAPA-TDNN embedding
  │    ├─ In-session clustering (cosine sim >= 0.30 → same speaker)
  │    ├─ EMA centroid update (alpha=0.95) — UNCHANGED
  │    ├─ NEW: Retain (embedding, duration) in _segment_embeddings[speaker_id]
  │    └─ Return (session_speaker_id, embedding)
  │
  ├─ Cross-session matching (if speaker is stable: >= 3 segments, centroid converged)
  │    ├─ SpeakerStore.match_profiles(session_centroid)
  │    ├─ Apply confidence tier (HIGH / MEDIUM / LOW)
  │    ├─ If HIGH and no conflict: auto-assign profile name
  │    ├─ If MEDIUM: mark as suggested, maybe prompt (within fatigue budget)
  │    └─ If LOW: leave as "Speaker N"
  │
  └─ Display result in TranscriptPanel with confidence indicator
```

### 6.3 User Tagging Flow

```
User presses T
  → SpeakerTagScreen opens
  → Shows all session speakers with current labels
  → User selects a speaker, types a name
  │
  ├─ If name matches existing profile:
  │    → Link session speaker to that profile
  │    → Add session embeddings (quality-filtered) to profile
  │    → Recompute centroid
  │
  ├─ If new name:
  │    → Create new profile in SpeakerStore
  │    → Seed with session embeddings (quality-filtered)
  │
  └─ Update all transcript entries for that speaker
     → TranscriptPanel.rename_speaker(speaker_id, new_name)
     → Re-render transcript
```

---

## 7. Progressive Learning System

### 7.1 Learning Phases

```
         Manual ──────────────────────────────► Automatic
          │                                         │
    Phase 1        Phase 2        Phase 3       Phase 4
   Cold Start    Early Learn     Growing       Mature
   0 tags        1-3 tags       4-10 tags      10+ tags
   ~0s data      ~5-15s         ~20-60s        ~1-2 min
     │              │               │              │
  All manual    Suggest "?"    Auto HIGH      Nearly all
  tagging       prompts        confirm MED    automatic
```

### 7.2 Active Learning Rules

| Rule | Value | Purpose |
|------|-------|---------|
| Max prompts per 10-min window | 5 | Prevent fatigue |
| Confirmations to "stop asking" per speaker/session | 3 | Diminishing returns |
| First 5 min of session | Max 1 prompt per unique speaker | Initial mapping |
| After 5 min | Max 1 prompt per 2 min | Steady state |
| Prompt response options | Confirm (Y) / Reject+Reassign (N) / Skip (S) | Cover all cases |

### 7.3 Profile Maturity Thresholds

| Milestone | Samples | Effect |
|-----------|---------|--------|
| Can suggest | 1+ | MEDIUM-tier suggestions enabled |
| Can auto-assign | 3+ confirmed | HIGH-tier auto-assignment enabled |
| Centroid strategy change | 20+ | Switch from running mean to EMA (alpha=0.98) |
| Multi-centroid enabled | 15+ exemplars | K-means sub-centroids for better matching |
| Cold-start protection lifted | 10+ confirmed | Auto-assigned embeddings may update profile |

---

## 8. UX Design

### 8.1 New Keybindings

| Key | Action | Context |
|-----|--------|---------|
| `T` | Open Speaker Tag modal | Main screen |
| `P` | Open Speaker Profiles library | Main screen |

Existing keys unchanged: R, M, L, S, C, D, Q, ?

### 8.2 Transcript Display Format

```
[14:23:01]  Alice        Hey, did you see the new release?
[14:23:05]  Bob? ~72%    Yeah, I was just looking at it.
[14:23:09]  Alice        The performance improvements are incredible.
[14:23:14]  Speaker 3 *  Can I join the conversation?
```

| Indicator | Meaning |
|-----------|---------|
| *(none)* | Manually tagged / confirmed identity |
| `~72%` | Auto-recognized with confidence score |
| `?` | Suggested name, awaiting confirmation |
| `*` | New unknown speaker this session |

### 8.3 Speaker Tag Modal (T)

```
┏━━ TAG SPEAKERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                     ┃
┃  #  SPEAKER          SEGMENTS  STATUS               ┃
┃  ─────────────────────────────────────               ┃
┃  1  ● Alice              12    tagged                ┃
┃ >2  ● Speaker 2           8    unknown  ← NEW       ┃
┃  3  ● Speaker 3           3    unknown               ┃
┃                                                      ┃
┃  ┌─────────────────────────────────────┐             ┃
┃  │ Name: Bob_                          │             ┃
┃  │ Suggestions: Bob Chen | Bob (work)  │             ┃
┃  └─────────────────────────────────────┘             ┃
┃                                                      ┃
┃  [ENTER] save  [TAB] next  [^M] merge  [ESC] close  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Workflow:** `T` → cursor on newest untagged → type name → `Enter` → done. Three actions.

- `Up/Down` or `J/K`: navigate speaker list
- `Enter`: focus name input (pre-filled with current label)
- `Input` uses `SuggestFromList` with names from profile library
- `Enter` in input: save name, return to list
- `Ctrl+M`: merge selected speaker into another (sub-modal)
- `Escape`: close

### 8.4 Speaker Profiles Screen (P)

```
┏━━ SPEAKER PROFILES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                               ┃
┃  NAME             SESSIONS  TOTAL TIME  LAST SEEN             ┃
┃  ──────────────────────────────────────────────────            ┃
┃ >● Alice              14     3h 42m     2026-03-20            ┃
┃  ● Bob Chen            7     1h 15m     2026-03-18            ┃
┃  ● Charlie             3       28m      2026-03-15            ┃
┃                                                               ┃
┃  ┌──────────────────────────────────────────────┐             ┃
┃  │  Alice                                       │             ┃
┃  │  Sessions: 14    Total: 3h 42m               │             ┃
┃  │  Samples: 847    Quality: 0.91               │             ┃
┃  │  Color: ■ #00ffcc                            │             ┃
┃  └──────────────────────────────────────────────┘             ┃
┃                                                               ┃
┃  [ENTER] edit  [^D] delete  [ESC] close                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### 8.5 Status Bar Enhancement

```
● REC    qwen3-0.6b    English    2/3 tagged    saved 3s ago
```

"2/3 tagged" = 2 of 3 detected speakers have been named.

### 8.6 Confirmation Prompt (Active Learning)

When a MEDIUM-confidence match is found and the fatigue budget allows:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Is this Bob Chen?   [Y] [N] [S]   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

Non-modal toast/notification — does not interrupt recording. Auto-dismisses after 10 seconds (treated as Skip).

---

## 9. Roadmap & Implementation Plan

### Phase 1: Core Tagging (MVP) — ~3-4 days

**Goal:** Users can name speakers and names persist across sessions.

| Step | Task | Files | Est. |
|------|------|-------|------|
| 1.1 | Create `SpeakerStore` class with SQLite schema, CRUD ops, centroid cache | `speakers/store.py` (new) | 4h |
| 1.2 | Add per-segment embedding retention to `DiarizationEngine.identify()` | `diarization/engine.py` | 2h |
| 1.3 | Wire `SpeakerStore.open()` into app startup (before diarizer) | `app.py` | 1h |
| 1.4 | Build `SpeakerTagScreen` modal (OptionList + Input) | `widgets/tag_screen.py` (new) | 3h |
| 1.5 | Add `rename_speaker()` + `_rerender()` to `TranscriptPanel` | `widgets/transcript.py` | 2h |
| 1.6 | Connect tagging flow: T key → modal → SpeakerStore.create/update → re-render | `app.py` | 2h |
| 1.7 | Add `T` keybinding + update help screen + status bar "X/Y tagged" | `app.py`, `cyberpunk.tcss` | 1h |
| 1.8 | Persist speaker color per profile | `speakers/store.py`, `widgets/transcript.py` | 1h |

**Deliverable:** Press `T`, name a speaker, see it in the transcript, close VoxTerm, reopen — names are still there.

### Phase 2: Auto-Recognition — ~3-4 days

**Goal:** Returning speakers are automatically identified.

| Step | Task | Files | Est. |
|------|------|-------|------|
| 2.1 | Implement `SpeakerStore.match_profiles()` with cosine matching | `speakers/store.py` | 2h |
| 2.2 | Add stabilization check (>= 3 segments, centroid converged < 0.05) | `diarization/engine.py` | 1h |
| 2.3 | Implement three-tier confidence system with adaptive thresholds | `speakers/store.py` | 2h |
| 2.4 | Cross-session matching integration in the worker pipeline | `app.py` | 3h |
| 2.5 | Confidence indicators in transcript display | `widgets/transcript.py` | 2h |
| 2.6 | Profile conflict resolution (top-2 margin check) | `speakers/store.py` | 1h |
| 2.7 | Quality filtering for enrollment (duration, energy) | `diarization/engine.py` | 1h |
| 2.8 | Cold-start protection (only confirmed embeddings < 10 samples) | `speakers/store.py` | 1h |
| 2.9 | Daily backup logic | `speakers/store.py` | 1h |

**Deliverable:** Tag Alice in session 1. Start session 2 — Alice is auto-recognized with a confidence score.

### Phase 3: Progressive Learning & Profiles — ~2-3 days

**Goal:** System improves over time; users can manage profiles.

| Step | Task | Files | Est. |
|------|------|-------|------|
| 3.1 | Active learning prompt (toast notification for MEDIUM matches) | `app.py`, `widgets/` (new toast) | 3h |
| 3.2 | Fatigue prevention system (budgets, cooldowns, per-speaker caps) | `speakers/store.py` | 2h |
| 3.3 | Build `SpeakerProfileScreen` (DataTable + detail pane) | `widgets/profile_screen.py` (new) | 3h |
| 3.4 | Merge speaker flow (Ctrl+M in tag modal) | `widgets/tag_screen.py`, `speakers/store.py` | 2h |
| 3.5 | Error correction: reassign segment to different speaker | `app.py`, `speakers/store.py` | 2h |
| 3.6 | Profile stats tracking (session count, total time, last seen) | `speakers/store.py` | 1h |

**Deliverable:** By session 5 with the same people, most speakers are auto-recognized. Profile library shows usage stats.

### Phase 4: Polish & Advanced — ~2 days

**Goal:** Robustness, multi-centroid matching, export/import.

| Step | Task | Files | Est. |
|------|------|-------|------|
| 4.1 | Multi-centroid matching (k-means at 15+ exemplars, K=3) | `speakers/store.py` | 2h |
| 4.2 | Running mean → EMA transition at 20+ exemplars | `speakers/store.py` | 1h |
| 4.3 | Export/import profiles via sqlite3 backup API | `speakers/store.py` | 2h |
| 4.4 | "Delete all voice data" option | `speakers/store.py`, `app.py` | 1h |
| 4.5 | Graceful degradation (DB unavailable → ephemeral fallback) | `app.py`, `speakers/store.py` | 1h |
| 4.6 | First-use onboarding tip ("Press T to name speakers") | `app.py` | 0.5h |
| 4.7 | Profile drift alert (centroid > 0.20 from golden set) | `speakers/store.py` | 1h |

**Total estimated effort: ~10-13 days**

### New Files

```
voxterm/
    speakers/                    # NEW package
        __init__.py
        store.py                 # SpeakerStore (SQLite, cache, matching)
        models.py                # SpeakerProfile dataclass, SpeakerMeta
    widgets/
        tag_screen.py            # NEW — SpeakerTagScreen modal
        profile_screen.py        # NEW — SpeakerProfileScreen modal
```

---

## 10. Risks & Mitigations

### Validated Risks (from architecture review)

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Centroid poisoning during cold start** | HIGH | Only accept user-confirmed embeddings until 10+ samples (Section 4.7) |
| **Thread safety on centroid cache** | MEDIUM | `threading.Lock` on all cache read-modify-write ops (Section 5.5) |
| **Retroactive enrollment data loss** | HIGH | Per-segment embedding retention in session (Section 4.1) |
| **Profile conflicts (multiple matches)** | MEDIUM | Top-2 margin check, degrade to MEDIUM if ambiguous (Section 4.4) |
| **iCloud sync corruption** | MEDIUM | Store DB in `~/Library/Application Support/` (Section 5.1) |
| **RichLog re-render flicker** | LOW | Acceptable for TUI; future ListView migration if needed |
| **First-session discoverability** | MEDIUM | One-time system message tip (Phase 4.6) |

### Architectural Constraints

- **No new dependencies.** SQLite is in Python stdlib. All ML models are already loaded.
- **Hot path budget: < 5ms.** Cross-session matching is O(8 * P) cosine ops — well within budget at 500 profiles.
- **Profiles are user-initiated only.** No automatic profile creation prevents unbounded growth.

---

## 11. Appendix

### A. Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `SIMILARITY_THRESHOLD` (in-session) | 0.30 | `diarization/engine.py` (unchanged) |
| `EMA_ALPHA` (in-session) | 0.95 | `diarization/engine.py` (unchanged) |
| `CROSS_SESSION_HIGH_BASE` | 0.55 | `speakers/store.py` (new) |
| `CROSS_SESSION_MEDIUM` | 0.35 | `speakers/store.py` (new) |
| `ADAPTIVE_DECAY_RATE` | 10 | `speakers/store.py` (new) |
| `ADAPTIVE_BOOST` | 0.15 | `speakers/store.py` (new) |
| `ENROLLMENT_EMA_ALPHA` | 0.98 | `speakers/store.py` (new) |
| `MAX_EXEMPLARS` | 20 | `speakers/store.py` (new) |
| `MIN_ENROLLMENT_DURATION` | 3.0s | `speakers/store.py` (new) |
| `STABILIZATION_SEGMENTS` | 3 | `speakers/store.py` (new) |
| `STABILIZATION_DELTA` | 0.05 | `speakers/store.py` (new) |
| `MULTI_CENTROID_MIN` | 15 | `speakers/store.py` (new) |
| `MULTI_CENTROID_K` | 3 | `speakers/store.py` (new) |
| `COLD_START_CONFIRMED_MIN` | 10 | `speakers/store.py` (new) |
| `CONFLICT_MARGIN` | 0.05 | `speakers/store.py` (new) |
| `MAX_PROMPTS_PER_10MIN` | 5 | `speakers/store.py` (new) |

### B. Research References

1. SpeechBrain ECAPA-TDNN — EER 0.80% on VoxCeleb1-O (192-dim, cosine scoring)
2. "Cosine or PLDA?" (Wang et al., Interspeech 2022) — cosine competitive with PLDA for large-margin embeddings
3. "Online Learning of Open-set Speaker ID" (Yoo et al., Interspeech 2022) — active learning for speaker identification
4. "Sub-center Modelling for Speaker Embeddings" (arxiv 2407.04291) — multi-centroid approach
5. "Investigating Confidence Estimation for Speaker Diarization" (Chowdhury et al., Interspeech 2024)
6. Resemblyzer study — minimum 2.63s for reliable speaker embeddings
7. Forensic ASR validation — ECAPA-TDNN robust across microphone types

### C. Memory Budget

| Component | Memory |
|-----------|--------|
| 100 profile centroids | 75 KB |
| SQLite connection | ~50 KB |
| Per-session embeddings (~100 segments) | ~80 KB |
| Profile metadata cache | ~20 KB |
| **Total overhead** | **~225 KB** |

Compare: ECAPA-TDNN model (~80 MB), Qwen3 model (~600 MB+). Negligible.
