# VOXTERM

> Local, real-time voice transcription. Speaker-aware. Network-optional.

![platform](https://img.shields.io/badge/macOS-Apple_Silicon-black)
![platform-linux](https://img.shields.io/badge/Linux-supported-brightgreen)
![platform-windows](https://img.shields.io/badge/Windows-phase_1-yellow)
![python](https://img.shields.io/badge/python-3.12+-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

```
+++ VOXTERM v0.1.0 // LOCAL VOICE TRANSCRIPTION ENGINE
┌─ WAVEFORM ─────────────────────────────────────────────────────┐
│       ▓▓░  ▓░░    ░ ▓▓     ░░░ ▓▒░░  ░░░ ▓▒    ░░░             │
│   ░  ▒▓▒▓ ▒▒▒▓▓ ░ ▓ ▓▓▓ ░ ▓ ▒▓▒▓▓▓▒░░▒▒░ ▒▓▒░  ▒▒▒░  ░         │
│ ▒▓▒▓▒▓▓▓▒▓▓▓▒▓▓▒▓▒▓▒▓▓▒▒▓▒▒▓▓▓▒▓▓▓▒▓▓▓▒▓▓▒▓▒▓▓▓▒▓▓▓▒▓▓▒        │
└────────────────────────────────────────────────────────────────┘
┌─ TRANSCRIPT // LIVE ───────────────────────────────────────────┐
│ [14:30:01]  Daniel  hey ron — did the sink land?               │
│ [14:30:04]  Ron     yeah, just merged                          │
│ [14:30:07]  HIVE    ✓ batch 0 · 2 segs → convent-box           │
│ [14:30:09]  Daniel  rad. testing now                           │
└────────────────────────────────────────────────────────────────┘
  ● REC  qwen3-1.7b [M]  English [L]  ⬢ HIVEMIND · convent-box
  [R] Record  [T] Tag  [E] Transcripts  [P] Party  [H] Hivemind  [?] Help
```

Mic in. Markdown out. Speakers tagged. Nothing leaves your machine unless you point it somewhere — and even then, only at peers you can see on your LAN.

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash
voxterm
```

Models download on first run (~600 MB – 1.5 GB depending on model).

<details>
<summary>Manual setup (developers)</summary>

```bash
git clone https://github.com/dmarzzz/VoxTerm.git
cd VoxTerm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./voxterm                    # launcher; uses .venv automatically
```

Entry point is `tui/app.py` (not `app.py` at the repo root).

</details>

---

## Privacy & storage

VoxTerm is **local first and private by default.** Everything runs on your machine. Nothing is ever sent to a server you didn't pick.

- **No audio is stored.** Microphone input is processed in real-time and discarded. Only text transcripts are saved.
- **Voice profiles are encrypted at rest.** Speaker embeddings (biometric data used to recognize voices across sessions) are encrypted with AES-256-CBC. The key lives in your macOS Keychain — zero config.
- **Transcripts are yours.** Auto-saved as markdown to `~/Documents/voxterm-transcripts/`. Never uploaded.
- **Network features are opt-in.** Party mode is LAN-only. Hivemind mode pushes only to a sink you explicitly pick from a list of advertised peers.
- **Delete everything anytime.** Press `O` → delete to permanently wipe all voice data from disk.

---

## Modes

### Solo (default)

```
  mic + system audio  ──►  VoxTerm  ──►  ~/Documents/voxterm-transcripts/*.md
```

Press `R` to record. The transcript writes live to disk so a crash never costs you a session.

### Party — same room, multiple laptops

```
  laptop A  ◄─── LAN mDNS ───►  laptop B
   VoxTerm  ◄── transcripts ──►  VoxTerm
   (you)                         (peer)
```

Each laptop transcribes its closest speaker best — combine them and the merged transcript beats any single mic. **Press `P`** to join. No codes, no setup. Auto-discovers peers on the LAN; auto-joins the nearest party or hosts one if none are running. Encrypted with AES-256-GCM. Everyone sees who joins and leaves.

See [`docs/party-mode-design.md`](docs/party-mode-design.md) for the protocol.

### Hivemind — push transcripts to a peer's swf-node

```
  VoxTerm  ──►  swf-node  ──►  alchemist + field-guide
   (push)        (signs)        (consume bundles)
```

Send live transcripts to a [swf-node](https://github.com/dmarzzz/searxng-wth-frnds) sink running on the LAN — the Shape Rotator program's "convent box". **Press `H`** to scan. Pick a sink; VoxTerm remembers it across launches. Batches go out every ~60s / 30 segments / EOF. Local files keep saving as before — hivemind is purely additive.

One-way push only. VoxTerm never reads from the network. The sink resigns and stores; downstream UIs read from their own swf-node.

---

## Voice tagging

VoxTerm learns and remembers speaker voices across sessions.

1. Record a conversation — speakers are detected as `Speaker 1`, `Speaker 2`, …
2. Press `T`, type a name, hit Enter.
3. Next session, VoxTerm auto-recognizes returning speakers.
4. The more you tag, the less you need to.

Press `O` to manage your speaker library (rename, delete, wipe all data).

---

## Keyboard

| Key | Action |
|-----|--------|
| `R` | Start / stop recording |
| `T` | Tag / name speakers |
| `E` | Browse saved transcripts |
| `S` / `^S` | Save / export transcript |
| `M` | Switch transcription model |
| `L` | Switch language |
| `P` | Party mode — join or leave |
| `H` | Hivemind — pick a transcript sink |
| `O` | Speaker profiles |
| `V` | Toggle merged transcript view (party mode) |
| `C` | Clear transcript |
| `D` | Toggle debug mode |
| `?` | Help |
| `Q` | Quit |

---

## Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen3-0.6b` | ~600 MB | Default on Linux. Fast, decent. |
| `qwen3-1.7b` | ~1.5 GB | Default on macOS. More accurate. |
| Whisper variants | tiny → large-v3 | macOS via `mlx-whisper`, Linux/Windows via `faster-whisper`. |

Switch live with `M`. Models download from Hugging Face on first use. Set up [llama-swap](llama-swap-config.example.yaml) to route through a local Ollama-compatible server instead.

---

## Architecture

```
audio/         capture, VAD, transcription, diarization, speaker store
network/       discovery + party + hivemind (mDNS + zeroconf)
tui/           Textual app, widgets, theme
dictation/     macOS dictation hotkey (⌘⇧D — types into the focused app)
docs/          design notes (party mode, p2p protocol, hivemind scoping)
tests/         pytest suite (~700 cases, mocked engines for CI)
config.py      single-file config + paths + ConfigStore
diagnostics.py crash dumps, faulthandler, log rotation
```

Stack: **MLX** (Qwen3-ASR / mlx-whisper on Metal GPU) · **3D-Speaker ERes2Net** (ONNX, 512-dim embeddings) · **Silero VAD** (ONNX) · **Textual** (TUI) · **SQLite** (speaker profiles) · **sounddevice** (mic) · **ScreenCaptureKit** (system audio on macOS).

The diarizer runs in-process via `onnxruntime` — no PyTorch on the hot path on macOS. (Linux/Windows fall back to the PyTorch `qwen-asr` path.)

---

## Status

| Platform | Status |
|---|---|
| macOS Apple Silicon (M1+) | **Primary.** All features. |
| Linux (x86_64 + arm64) | Supported. `faster-whisper` + optional `qwen-asr`. No system-audio capture (no `ScreenCaptureKit` analogue yet). |
| Windows | **Phase 1.** Same backends as Linux; some rough edges. |
| Mobile | Out of scope. |

Voxterm is `0.1.0`. Expect improvements; expect occasional rough edges. File an issue.

---

## Related

- [`searxng-wth-frnds`](https://github.com/dmarzzz/searxng-wth-frnds) — `swf-node`, the LAN-first peer search daemon. The hivemind sink lives here.
- [`shape-rotator-os`](https://github.com/dmarzzz/shape-rotator-wrld-knwldge-viz) — the alchemist + field-guide Electron apps that consume hivemind bundles.

---

## License

[MIT](LICENSE).
