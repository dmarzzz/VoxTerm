# Changelog

All notable changes to VoxTerm are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Local-LLM transcript redaction** (`X` key). Masks PII in a transcript
  using an on-device model and writes a `*-redacted.md` copy. The LLM only
  *identifies* verbatim sensitive spans (returned as JSON); the engine
  masks them by exact string replacement, so the transcript stays
  byte-for-byte intact and the model can't paraphrase or drop content. A
  deterministic regex pass backstops structured PII (emails, URLs, SSNs,
  phone/IP-like runs). Four profiles (standard / contacts-only /
  aggressive / custom) and the same MLX + Ollama backend split as
  summarization (`ollama:model[@host]` works off Apple Silicon).
  - **Review before write.** Because a small model *will* miss PII — and a
    miss written into a file named `-redacted` feels safe — nothing is
    written until you confirm. The found spans are shown as a toggle list
    (uncheck false positives, type to add missed ones) with a live preview;
    cancelling writes nothing.
  - **Original-file disposition.** You choose what happens to the
    unredacted live autosave: keep / replace (delete) / shred (best-effort
    overwrite + delete — not a forensic wipe on copy-on-write/SSD storage,
    and the UI says so). Defaults to keep and is not persisted.
  - **Disclosure tiers (the dial).** Redaction level is keyed to *audience*,
    not an abstract amount: RAW → INNER → ROOM → WORLD, a strictly-nested
    mask-policy over an expanded vocabulary (identifiers + sensitivity
    content-classes like substances/health/legal/finance). Detection finds
    everything in one pass; cycling the tier (⌃T on the review screen)
    re-filters what's masked with no re-inference, and the current tier is
    shown as a color-coded meter. ROOM keeps the work (projects, orgs) but
    strips people; WORLD strips proper nouns too; INNER strips only secrets.
    Persisted as `redaction_tier`. See #150 for the design + roadmap
    (egress-coupled auto-tier, always-on header badge, live outbound masking).
  - New `redaction/` package; mirrors `summarizer/`.

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
