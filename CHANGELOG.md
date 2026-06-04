# Changelog

All notable changes to VoxTerm are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional cross-platform CPU **streaming ASR** backend (sherpa-onnx) behind the
  `[streaming]` extra, with model keys `sherpa-stream-en` (zipformer-20M, ultra-fast) and
  `sherpa-nemotron-en` (NeMo FastConformer-RNNT 0.6B, accurate). Fully additive — nothing
  changes when the extra is absent. See `docs/streaming-asr.md` and
  `docs/streaming-asr-benchmark.md`.

## [0.3.0] - 2026-06-03

### Added

- NVIDIA Parakeet ASR backend on Apple Silicon via `parakeet-mlx`: new
  `ParakeetTranscriber` plus two registry models — `parakeet-0.6b`
  (`parakeet-tdt-0.6b-v3`) and `parakeet-1.1b` (`parakeet-tdt-1.1b`).
  `parakeet-1.1b` matches the largest Qwen3 model's accuracy at ~12× the
  throughput in benchmarks (`docs/parakeet-asr-benchmark.md`, reproduce with
  `python -m dev.bench_asr`). Note: the requested cache-aware *streaming*
  model `nvidia/nemotron-speech-streaming-en-0.6b` can't run on this MLX
  stack yet (causal downsampling unsupported by `parakeet-mlx`; NeMo is
  CUDA-only) — the 0.6B TDT is its supported non-streaming sibling (#167).
- Per-peer audio frame-loss telemetry in the P2P debug overlay — surfaces the
  real UDP loss signal (sequence gaps filled with silence) as a per-peer gap
  rate instead of a permanently-zero counter (#162).
- Hivemind sink integration spec (`docs/specs/hivemind-sink-integration.md`):
  one client speaking two transcript-sink protocols (legacy non-TEE + attested
  TEE), selected by config (#154).

### Changed

- Transcriber backend selection is centralized in a single `get_transcriber()`
  factory, replacing three hand-written if/elif chains (#165).

### Fixed

- Pin `onnxruntime<1.24`: 1.24.0 introduced a ~11 MB-per-`session.run` heap leak
  (measured, not shape-driven) that drove multi-GB RSS growth over long
  diarization sessions; 1.23.x is the last leak-free release (#166).
- P2P mesh self-healing: wire `on_peer_updated` and add a periodic reconnect
  sweep so a peer that re-advertises (or leaves and rejoins) no longer leaves a
  permanent hole in a 3+ peer mesh (#161).
- Diarizer subprocess respawn runs off the IPC lock, so the up-to-30s model-load
  wait during crash recovery no longer freezes a concurrent call or the UI
  (#163).
- Diarization CMN mean is computed in float64 for a true zero-mean result, and
  the macOS test suite is stabilized (fbank precision + p2p e2e session key)
  (#156).
- Linux test suite passes and platform-support docs are accurate (macOS +
  Linux); macOS-Keychain-only tests are skipped on Linux (#159).

### Security

- Crash dumps and `faulthandler.log` are now owner-only (`0o600`/`0o700`),
  matching the speaker-DB convention — they carry session/runtime metadata a
  local reader on a shared machine shouldn't get (#164).

## [0.2.1] - 2026-05-16

### Fixed

- `voxterm update` returned `curl 404` against v0.2.0 because the
  release workflow didn't auto-attach `install.sh`. The release
  workflow now uploads `install.sh` as a release asset on every
  tag push, and the rendered release body links to the release-asset
  install URL directly (#129).
- Pressing `h` and selecting a sink had no visible in-TUI confirmation
  (the only signal was a hidden INFO line in `voxterm.log`). The
  hivemind menu now fires a textual toast and a SYS message in the
  transcript panel on connect / disconnect so the state change is
  immediately visible (#130).

## [0.2.0] - 2026-05-16

### Added

#### Hivemind transcript-sink integration
- New `h` keybinding opens a hivemind menu listing mDNS-discovered
  swf-node sinks on the LAN. ENTER on a sink toggles transcript push
  on/off for that sink. The choice persists to ConfigStore, so a
  returning user gets push re-enabled automatically without re-toggling.
- Push is gated by user opt-in by default: `voxterm` always runs the
  mDNS browser, but transcripts are buffered (not POSTed) until the
  user presses `h` and enables a sink. `--hivemind on` keeps the old
  always-push behavior for scripted/headless use.
- Pinning by sink pubkey: once a sink is enabled, a different sink
  showing up on the LAN won't get our transcripts by accident.
- Visible logging at every stage. Startup banner reports searching
  / found / pushing state; `voxterm.log` (with `VOXTERM_LOG_LEVEL=INFO`)
  shows discovery events, the first batch posted, and a heartbeat
  every Nth batch so a tail confirms ongoing health.

### Changed

- macOS default transcription model switched from `qwen3-1.7b` to
  `qwen3-0.6b` to match Linux/Windows. Smaller first-run download
  and lower memory footprint. The 1.7B variant is still available
  via `voxterm -m qwen3-1.7b`.

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
