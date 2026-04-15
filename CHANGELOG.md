# Changelog

All notable changes to VoxTerm are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-15

First public release. VoxTerm is a local, offline voice transcription TUI for
macOS Apple Silicon — mic + system audio in, speaker-tagged transcripts out,
nothing ever leaves your machine.

### Added

#### Core transcription
- Real-time transcription via MLX on Metal GPU with Qwen3-ASR (0.6B default,
  1.7B opt-in) and a mlx-whisper fallback (tiny → large-v3).
- Microphone capture via `sounddevice` and system audio capture via a
  compiled-on-first-use Swift/ScreenCaptureKit helper.
- Silero VAD (ONNX) for neural speech/silence gating — no PyTorch on the hot
  path.
- Hallucination filter and live deduplication for clean transcript output.

#### Speaker diarization
- Online speaker diarization powered by 3D-Speaker ERes2Net-large (512-dim
  embeddings, 0.52% EER) exported to ONNX and run in-process via
  `onnxruntime` — no subprocess, no PyTorch required on the primary path.
- Pure-NumPy Kaldi-compatible Mel filterbank (no PyTorch / torchaudio).
- Online cosine clustering with spectral re-clustering, HMM-style continuity
  prior, and VBx-style loop probability for stable speaker assignment.
- Legacy PyTorch subprocess path retained as a crash-isolated fallback for
  when ONNX models aren't available.

#### Persistent voice tagging
- Cross-session speaker recognition: tag once, recognized forever.
- SQLite-backed profile store (`~/Library/Application Support/voxterm/`) with
  WAL mode, 7-day rolling backups, and `chmod 600` on the DB.
- **Speaker embeddings are encrypted at rest** using AES-256-CBC; the key is
  stored in the macOS Keychain — zero user configuration.
- Multi-centroid matching with adaptive HIGH/MEDIUM thresholds, conflict
  margins, continuity bonuses, and periodic cluster merges.

#### Party Mode (P2P collaborative transcription)
- Press `N` to join or leave a party — no codes, no setup.
- LAN peer discovery via mDNS; auto-joins the nearest party or hosts one.
- AES-256-GCM-encrypted transcript sharing over TCP, with everyone seeing
  every join/leave (no silent surveillance).
- Each party gets a unique shared color; visual bloom effect on join.

#### Language identification
- 3D-Speaker CAM++ LID (ONNX) for automatic language detection.

#### TUI
- Cyberpunk-themed Textual UI with an FFT pixel-shader oscilloscope
  (pitch-mapped color), recording header, rainbow model names, and colored
  system-message highlights.
- Keybindings: `R` record, `N` party, `T` tag speakers, `O` profiles,
  `M` model, `L` language, `S` save, `C` clear, `E` transcript explorer,
  `D` debug, `?` help, `Esc` / `Q` quit.
- Transcript explorer modal (`E`) for browsing past sessions.
- Speaker tagging modal (`T`) and speaker profile manager (`O`).
- Instant TUI launch with model loading in the background.

#### Transcripts
- Auto-saved Markdown transcripts under `~/Documents/voxterm/`, with live
  append-mode output during recording in `.live/`.
- Persisted user preferences (last model, last language) in `.state.json`.

#### Installer & distribution
- One-line `curl | bash` installer (`install.sh`) that fetches the latest
  `v*` release tag, sets up a `pipx`-managed venv, and installs the
  `voxterm` launcher.
- GitHub Actions release workflow (`.github/workflows/release.yml`) that
  validates `config.py` VERSION matches the pushed tag before publishing a
  GitHub Release.

#### Diagnostics
- C-level `faulthandler` tracebacks, signal handlers, and crash dumps under
  `~/Documents/voxterm/.crashes/` — peak RSS, audio buffer duration, style
  cache stats, GC counters, transcript/speaker counts.
- Debug mode (`D`) surfaces buffer/silence durations and watchdog events
  live in the transcript panel.

### Fixed
- MLX + PyTorch C++ runtime segfault resolved by running the legacy
  diarizer in an isolated subprocess (ONNX path avoids the conflict
  entirely).
- Shutdown segfault during Python GC of C extensions — mitigated via
  `os._exit(0)` from atexit / finally.
- `fds_to_keep` multiprocessing crash on model load — patched
  `subprocess.Popen`, set spawn method, retry with `close_fds=False`.
- Quit hang and leaked-semaphore warning on exit.
- Diarization centroid drift and memory leak causing speaker collapse
  after ~45 minutes.
- Installer matched non-release tags (`benchmark-fixtures`, `onnx-models`);
  now restricted to `v*`.

### Security
- Voice biometric embeddings (speaker profiles) are encrypted at rest with
  AES-256-CBC; key held in macOS Keychain.
- Speaker database is `chmod 600` and kept in Application Support, never
  synced to `~/Documents`.
- All P2P transcript traffic is AES-256-GCM encrypted and stays on the LAN;
  no relay servers.
- `install.sh` pins to published `v*` release tags only.

### Platforms
- **Primary:** macOS on Apple Silicon (M1+), Python 3.9+.
- **Experimental:** Windows (Phase 1 support landed in #75); Linux plan
  drafted in `docs/`.

[Unreleased]: https://github.com/dmarzzz/VoxTerm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dmarzzz/VoxTerm/releases/tag/v0.1.0
