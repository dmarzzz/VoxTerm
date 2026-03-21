# CLAUDE.md — VoxTerm Agent Guide

## What is this project?

VoxTerm is a local, offline voice transcription TUI for macOS Apple Silicon. It captures mic + system audio, transcribes speech in real-time, identifies speakers, and remembers voices across sessions.

**Stack**: MLX (Qwen3-ASR transcription on Metal GPU) · CAM++ (512-dim speaker embeddings on CPU) · Silero VAD (ONNX, speech detection) · Textual (TUI) · SQLite (speaker profiles) · sounddevice (mic) · Swift/ScreenCaptureKit (system audio)

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
│  ├─ Cross-session speaker matching (SQLite writes)
│  └─ call_from_thread() → UI updates
│
SUBPROCESSES
├─ Diarizer subprocess (PyTorch/CAM++)
│  ├─ Loads CAM++ model (~28MB, downloads on first use from ModelScope)
│  ├─ Receives audio over pipe, returns speaker ID + embedding
│  ├─ Owns all session state (centroids, names, embeddings)
│  └─ Auto-restarts on crash; falls back to in-process if repeated failures
│
└─ System audio subprocess (Swift/ScreenCaptureKit)
   ├─ Compiled on first use from _macos_sck.swift
   ├─ Streams raw PCM over stdout pipe
   └─ For Bluetooth: routes through BlackHole virtual device
```

**Why process isolation?** MLX (Metal GPU) and PyTorch (CPU) have C++ runtimes that conflict when loaded in the same process, causing segfaults. The diarizer runs in its own subprocess so they never share an address space.

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
| `diarization/campplus.py` | CAM++ model architecture (vendored from WeSpeaker, Apache 2.0) |
| `diarization/engine.py` | CAM++ online speaker clustering (runs inside subprocess) |
| `diarization/proxy.py` | DiarizationProxy — same API as engine, delegates to subprocess |
| `diarization/subprocess_worker.py` | Subprocess entry point: loads model, read-process-write loop |
| `diarization/ipc.py` | Binary IPC protocol for main↔subprocess communication |
| `speakers/models.py` | SpeakerProfile, SpeakerMeta dataclasses, multi-centroid matching |
| `speakers/store.py` | SQLite persistence, cross-session matching, backup/restore |
| `widgets/waveform.py` | FFT pixel-shader oscilloscope with pitch-mapped color |
| `widgets/transcript.py` | RichLog transcript with speaker labels + confidence indicators |
| `widgets/header.py` | Recording indicator header bar |
| `widgets/tag_screen.py` | Speaker tagging modal (T key) |
| `widgets/profile_screen.py` | Speaker profile management modal (P key) |

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
| `~/.cache/wespeaker/campplus_voxceleb/` | CAM++ speaker encoder (~28MB) |
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

```bash
python3 app.py                    # default: qwen3-0.6b, English
python3 app.py -m qwen3-1.7b     # larger model
python3 app.py -l ja              # Japanese
python3 app.py --list-models      # show all available models
./voxterm                         # launcher script
```

**Keybindings**: R(record) T(tag speakers) P(profiles) M(model) L(language) S(save) C(clear) D(debug) ?(help) Q(quit)

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
| Match threshold | 0.35 | Assign to existing speaker if above |
| New speaker threshold | 0.30 | Must be below this vs ALL centroids to create new speaker |
| Continuity bonus | 0.10 | Similarity boost for the most recent speaker |
| Conflict margin | 0.05 | If top-2 within this, prefer more established speaker |
| Cluster merge | 0.50 | Periodic pairwise merge threshold |
| Quality RMS gate | 0.003 | Min RMS energy for centroid updates |
| SCD change threshold | 0.4 | Cosine distance for speaker change detection |
| SCD window / hop | 2.0s / 0.5s | Sliding window for embedding-based SCD |
| HMM loopP | 0.99 | VBx-style self-transition probability (continuity prior) |
