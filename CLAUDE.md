# CLAUDE.md — VoxTerm Agent Guide

## What is this project?

VoxTerm is a local, offline voice transcription TUI for macOS Apple Silicon. It captures mic + system audio, transcribes speech in real-time, identifies speakers, and remembers voices across sessions.

**Stack**: MLX (Qwen3-ASR transcription on Metal GPU) · 3D-Speaker ERes2Net (512-dim speaker embeddings via ONNX) · Silero VAD (ONNX, speech detection) · Textual (TUI) · SQLite (speaker profiles) · sounddevice (mic) · Swift/ScreenCaptureKit (system audio) · 3D-Speaker LID (language identification)

## Architecture

```
MAIN PROCESS
├─ Main thread (Textual event loop)
│  ├─ 15fps audio timer: reads mic + system audio queues
│  ├─ Silero VAD (ONNX, no PyTorch): speech/silence detection per chunk
│  ├─ UI rendering, keybindings (R/T/P/M/L/S/C/D/Q)
│  └─ SQLite reads for profile display
│
├─ Worker thread (@work(thread=True), group="transcription")
│  ├─ MLX transcription (Qwen3-ASR or Whisper)
│  ├─ 3D-Speaker diarization (ONNX, in-process — no subprocess needed)
│  │  ├─ ERes2Net-large (512-dim embeddings, best accuracy: 0.52% EER)
│  │  ├─ Pure-numpy Fbank features (no PyTorch/torchaudio)
│  │  └─ Online cosine clustering with spectral re-clustering
│  ├─ Language identification (3D-Speaker LID, ONNX)
│  ├─ Cross-session speaker matching (SQLite writes)
│  └─ call_from_thread() → UI updates
│
SUBPROCESSES (fallback only — not used when ONNX models available)
├─ Diarizer subprocess (PyTorch/speakerlab)
│  ├─ Loads 3D-Speaker model via speakerlab (fallback if ONNX unavailable)
│  ├─ Receives audio over pipe, returns speaker ID + embedding
│  ├─ Owns all session state (centroids, names, embeddings)
│  └─ Auto-restarts on crash; falls back to in-process if repeated failures
│
└─ System audio subprocess (Swift/ScreenCaptureKit)
   ├─ Compiled on first use from _macos_sck.swift
   ├─ Streams raw PCM over stdout pipe
   └─ For Bluetooth: routes through BlackHole virtual device
```

**Why ONNX?** The primary speaker embedding model (3D-Speaker ERes2Net) is exported to ONNX and runs via onnxruntime in the main process — no PyTorch needed, no subprocess needed. This eliminates IPC overhead and crash recovery complexity. The PyTorch subprocess path is kept as a fallback for when ONNX models aren't available.

**Legacy process isolation**: MLX (Metal GPU) and PyTorch (CPU) have C++ runtimes that conflict when loaded in the same process. The fallback diarizer subprocess prevents this, but is no longer needed with the ONNX backend.

## File map

| File | Description |
|------|-------------|
| `app.py` | Main Textual app — audio loop, transcription pipeline, session management, modals |
| `config.py` | Constants: sample rate, models, colors, paths, thresholds |
| `cyberpunk.tcss` | Textual CSS theme |
| `diagnostics.py` | Crash reporting: faulthandler, signal handlers, crash dumps, log rotation |
| `audio/capture.py` | Mic input via sounddevice callback → queue |
| `audio/system_capture.py` | System audio via Swift subprocess + pipe reader threads |
| `audio/vad.py` | Silero VAD wrapper (ONNX, no PyTorch) — neural speech/silence detection |
| `audio/buffer.py` | Thread-safe audio accumulator (append/get_and_clear) |
| `audio/platform.py` | macOS platform detection (Bluetooth, output device info) |
| `audio/blackhole.py` | BlackHole virtual device integration for Bluetooth routing |
| `audio/_macos_sck.swift` | ScreenCaptureKit Swift helper source |
| `audio/_macos_aggregate.swift` | Multi-output device Swift helper source |
| `transcriber/engine.py` | Qwen3-ASR (primary) + mlx-whisper (fallback), hallucination filter, dedup |
| `diarization/fbank.py` | Pure-numpy Mel filterbank (Kaldi-compatible, no PyTorch) |
| `diarization/onnx_embedder.py` | ONNX-based speaker embedding extraction (3D-Speaker models) |
| `diarization/campplus.py` | CAM++ model architecture (legacy, vendored from WeSpeaker) |
| `diarization/cluster.py` | 3D-Speaker clustering algorithms: spectral (p-value pruning), AHC, auto-select |
| `diarization/engine.py` | Online speaker clustering with ONNX/PyTorch backend dispatch |
| `diarization/proxy.py` | DiarizationProxy — direct (ONNX), subprocess, or inprocess modes |
| `diarization/subprocess_worker.py` | Subprocess entry point: loads model, read-process-write loop |
| `diarization/ipc.py` | Binary IPC protocol for main↔subprocess communication |
| `lid/engine.py` | Language identification using 3D-Speaker LID models (ONNX) |
| `scripts/export_onnx.py` | Export 3D-Speaker models to ONNX format |
| `speakers/models.py` | SpeakerProfile, SpeakerMeta dataclasses, multi-centroid matching |
| `speakers/store.py` | SQLite persistence, cross-session matching, backup/restore |
| `widgets/waveform.py` | FFT pixel-shader oscilloscope with pitch-mapped color |
| `widgets/transcript.py` | RichLog transcript with speaker labels + confidence indicators |
| `widgets/header.py` | Recording indicator header bar |
| `widgets/tag_screen.py` | Speaker tagging modal (T key) |
| `widgets/profile_screen.py` | Speaker profile management modal (P key) |
| `tui/events.py` | Optional JSONL event stream for out-of-process consumers (gated by `VOXTERM_EVENTS=1`) |

## Data and debug paths

These are the locations to check when debugging. All under the user's home directory.

### User-visible data
| Path | Contents |
|------|----------|
| `~/Documents/voxterm/` | Exported transcript files (.md) |
| `~/Documents/voxterm/.live/` | Live auto-save during recording (append-mode .md) |
| `~/Documents/voxterm/.state.json` | Persisted preferences (last model, last language) |

### Debug and crash data
| Path | Contents |
|------|----------|
| `~/Documents/voxterm/.crashes/*.log` | Human-readable crash dumps (timestamp, uptime, stack trace, runtime state, memory stats) |
| `~/Documents/voxterm/.crashes/*.json` | Machine-readable crash dumps (same data, structured) |
| `~/Documents/voxterm/.crashes/faulthandler.log` | C-level segfault tracebacks from Python's faulthandler module |

### Application data
| Path | Contents |
|------|----------|
| `~/Library/Application Support/voxterm/.speakers.db` | SQLite speaker profiles — biometric voice embeddings (chmod 600, WAL mode) |
| `~/Library/Application Support/voxterm/.backups/` | Daily DB backups (7-day retention, `speakers_YYYY-MM-DD.db`) |
| `~/Documents/voxterm/.bin/sck-helper` | Compiled Swift ScreenCaptureKit helper binary |
| `~/Documents/voxterm/.bin/aggregate-helper` | Compiled Swift multi-output device helper binary |

### Model caches (managed by frameworks)
| Path | Contents |
|------|----------|
| `~/.cache/3dspeaker/eres2net_large/` | 3D-Speaker ERes2Net-large ONNX model (primary speaker embeddings) |
| `~/.cache/3dspeaker/campplus/` | 3D-Speaker CAM++ ONNX model (alternative speaker embeddings) |
| `~/.cache/3dspeaker/campplus_lid/` | 3D-Speaker CAM++ LID ONNX model (language identification) |
| `~/.cache/wespeaker/campplus_voxceleb/` | Legacy CAM++ speaker encoder (~28MB, PyTorch fallback) |
| `~/.cache/huggingface/` | MLX/Qwen3-ASR model weights (~600MB-1.5GB) |

## Debugging

### Debug mode
Press `D` in the TUI to toggle debug mode. Shows in the transcript panel:
- Buffer duration and silence duration every 3 seconds
- Audio duration before each transcription
- Watchdog reset events

### Crash investigation
1. Check `~/Documents/voxterm/.crashes/` for recent `.log` or `.json` files
2. Check `~/Documents/voxterm/.crashes/faulthandler.log` for C-level tracebacks (segfaults)
3. Crash dumps include: peak RSS, audio buffer duration, style cache stats, transcript entry count, speaker count, GC counters, model state

### Known issues
- **MLX/PyTorch segfault**: These C++ runtimes conflict in the same process. Fixed by running diarizer in a subprocess. If subprocess isolation fails, falls back to in-process mode with `threading.Lock` + `OMP_NUM_THREADS=1` + `torch.set_num_threads(1)`.
- **Shutdown segfault**: Python's GC collects C extension objects (PortAudio, PyTorch, SpeechBrain) in random order during shutdown, causing segfaults. Mitigated with `os._exit(0)` via atexit handler and finally block.
- **Resource tracker warning**: SpeechBrain/PyTorch create semaphores that aren't cleaned up before forced exit. Harmless — suppressed with `warnings.filterwarnings`.

## How to run

Dependencies are declared in `pyproject.toml` (hatchling). There is no `requirements.txt`. The `./voxterm` script in the repo root is the **end-user launcher** installed by `install.sh` to `~/.local/share/voxterm/.venv` — it does NOT auto-create a workspace venv. For development you must set up `.venv/` manually.

### Development setup (first time, in repo root)
```bash
python3 -m venv .venv
source .venv/bin/activate                  # zsh/bash
pip install --upgrade pip
pip install -e .                           # installs deps from pyproject.toml
pip install -r dev/requirements-dev.txt    # optional: test deps
```

### Running (inside activated venv)
```bash
python -m tui.app                  # default: qwen3-0.6b, English
python -m tui.app -m qwen3-1.7b   # larger model
python -m tui.app -l ja            # Japanese
python -m tui.app --list-models    # show all available models
python -m dictation.app            # dictation mode
```

> **Important**:
> - Run as a module (`python -m tui.app`), NOT as a script (`python tui/app.py`) — the package uses relative imports.
> - The `[project.scripts]` entry in `pyproject.toml` also exposes a `voxterm` console script after `pip install -e .`, so `voxterm` (from inside the venv) works too.
> - There is no `app.py` at the repo root; the entry point is the `tui` package's `app:main`.
> - Always verify `.venv/bin/python` exists before running. If it doesn't, run the dev setup above.

**Keybindings**: R(record) T(tag speakers) P(profiles) M(model) L(language) S(save) C(clear) D(debug) ?(help) Q(quit)

## Optional event stream for out-of-process consumers

VoxTerm can emit a JSONL event stream that companion tools (LED matrices, OBS overlays, dashboards) can subscribe to via file-tail. **Off by default** — no file is opened, no overhead. Enable with `VOXTERM_EVENTS=1`.

```sh
VOXTERM_EVENTS=1 ./voxterm
# events land at: ~/Documents/voxterm-transcripts/.live/<session>-events.jsonl
```

Each line is one JSON object: `{"t": <unix_seconds>, "kind": "<name>", ...fields}`. Documented kinds (one per line in the stream, in order of arrival):

| kind | fields | notes |
|---|---|---|
| `session` | `phase` ("start"\|"end"), `model`, `language` | bracket the run |
| `recording` | `on` (bool) | R-key toggles |
| `party` | `on` (bool) | P2P session join/leave |
| `vad` | `on` (bool) | edge-triggered |
| `amplitude` | `rms` (float 0..1) | every audio chunk, ~15 Hz |
| `speaker` | `speaker_id`, `label`, `color` (TUI hex like `#00ffcc`) | edge-triggered |
| `text` | `speaker`, `speaker_id`, `color`, `text`, `confidence`, `overlap` | per finalised transcription |
| `peer_text` | `peer`, `speaker`, `text` | from party mode |

The `color` field on speaker/text events is the same hex the TUI uses, so consumers can render with a palette identical to what you see on screen. Module: `tui/events.py`.

## Speaker profile database schema

```sql
speakers (
    id TEXT PK,              -- UUID
    name TEXT,               -- user-assigned name
    color TEXT,              -- hex color
    centroid BLOB,           -- 512-dim float32 (2048 bytes)
    exemplars BLOB,          -- up to 20 exemplars (N*2048 bytes)
    exemplar_count INTEGER,
    confirmed_count INTEGER, -- user-confirmed segments
    auto_assigned_count INTEGER,
    total_duration_sec REAL,
    quality_score REAL,      -- mean pairwise cosine similarity
    created_at TEXT,         -- ISO 8601
    updated_at TEXT,
    last_seen_at TEXT,
    tags TEXT,               -- JSON array (future use)
    notes TEXT               -- free-form (future use)
)

session_speakers (
    session_id TEXT,          -- YYYY-MM-DD_HHMMSS
    speaker_id TEXT FK,
    local_id INTEGER,        -- in-session speaker number
    segment_count INTEGER
)
```

## Cross-session matching thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| HIGH base | 0.55 | Auto-assign speaker name |
| Adaptive boost | +0.15 * exp(-samples/10) | Stricter with fewer samples |
| MEDIUM | 0.35 | Suggest with "?" indicator |
| Conflict margin | 0.05 | If top-2 within this, treat as ambiguous |
| Match threshold | 0.55 | Assign to existing speaker if above (stable phase) |
| Match threshold (discovery) | 0.70 | Stricter threshold during first 30 calls |
| New speaker threshold | 0.45 | Must be below this vs ALL centroids to create new speaker |
| Continuity bonus | 0.05 | Similarity boost for the most recent speaker |
| Conflict margin | 0.05 | If top-2 within this, prefer more established speaker |
| Cluster merge | 0.65 | Periodic pairwise merge threshold |
| Centroid EMA alpha | 0.30 | Weight for new embedding in EMA centroid update |
| Centroid update min sim | 0.50 | Min cosine sim to centroid before updating it |
| Max embeddings/speaker | 20 | Cap per-speaker embedding retention |
| Max segment order | 200 | Cap temporal segment history |
| Quality RMS gate | 0.003 | Min RMS energy for centroid updates |
| SCD change threshold | 0.4 | Cosine distance for speaker change detection |
| SCD window / hop | 2.0s / 0.5s | Sliding window for embedding-based SCD |
| HMM loopP | 0.99 | VBx-style self-transition probability (continuity prior) |
