# VOXTERM

Local real-time voice transcription TUI with speaker diarization and P2P collaborative transcription. Runs entirely offline — no cloud APIs, no audio stored.

![platform](https://img.shields.io/badge/platform-macOS_(Apple_Silicon)-black)
![version](https://img.shields.io/badge/version-0.1.0-blue)

## Privacy & Storage Policy

VoxTerm is **local first and private by default**. Everything runs on your machine. Nothing is ever sent to a server.

- **No audio is stored.** Microphone input is processed in real-time and discarded. Only text transcripts are saved.
- **Voice profiles are encrypted at rest.** Speaker embeddings (biometric data used to recognize voices across sessions) are encrypted with AES-256-CBC. The key lives in your macOS Keychain — zero config.
- **Transcripts are yours.** Auto-saved as markdown to `~/Documents/voxterm-transcripts/`. Never uploaded anywhere.
- **P2P stays on your LAN.** Party mode shares transcripts over your local network only. No relay servers.
- **Delete everything anytime.** Press `P` → delete to permanently wipe all voice data from disk.

## Install

One command:

```bash
curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash
```

Then run:

```bash
voxterm
```

Requires macOS with Apple Silicon (M1+) and Python 3.9+. Models download automatically on first use.

<details>
<summary>Manual setup (for developers)</summary>

```bash
git clone https://github.com/dmarzzz/VoxTerm.git
cd voxterm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m tui.app
```

</details>

## Controls

| Key | Action |
|-----|--------|
| `R` | Start/pause recording |
| `N` | Party mode — join or leave P2P sessions |
| `T` | Tag/name speakers |
| `P` | Speaker profiles |
| `M` | Switch transcription model |
| `L` | Switch language |
| `S` | Save/export transcript |
| `C` | Clear transcript |
| `D` | Toggle debug mode |
| `?` | Help |
| `Q` | Quit |

## Party Mode (P2P)

Multiple people in the same room can share transcripts over the local network. Each laptop captures its closest speaker best — the combined result is better than any single mic.

**Press N** to join the party. Press N again to leave. No codes, no configuration.

- Auto-discovers nearby VoxTerm peers via mDNS
- Auto-joins the nearest party, or hosts one if none found
- Each party gets a unique color — all peers see the same color
- Encrypted transcript sharing (AES-256-GCM)
- Everyone sees who joins and leaves — no silent surveillance

See [docs/party-mode-design.md](docs/party-mode-design.md) for the full design.

## Hivemind mode (transcript streaming to swf-node)

Voxterm can stream transcript batches to a "convent box" running
`swf-node --full --hivemind-sink`. The sink wraps each batch in a
signed bundle and propagates it on the LAN.

By default voxterm auto-discovers sinks via mDNS. To override:
```
voxterm --hivemind-sink-url=http://convent.local:7777
```

Or disable entirely:
```
voxterm --hivemind=off
```

Mode flag: `--hivemind=auto|on|off`
- `auto` (default): mDNS-discover; fall back to local logging only.
- `on`: require a sink (fails to start if discovery times out after 5s).
- `off`: never POST; everything stays local.

Your voxterm device gets a persistent UUID stored at
`~/Library/Application Support/voxterm/device_id` (macOS) or
`~/.config/voxterm/device_id` (Linux); it appears in every batch as
`origin_device`. (It is a stable identifier, not a cryptographic
identity — the convent sink signs the bundle.)

Cadence: voxterm flushes a batch every ~60s, every 30 segments, or on
exit, whichever comes first. Batches POST to
`<sink>/hivemind/transcripts` per spec §4.3.

See `SHAPE-ROTATOR-OS-SPEC.md` §3.5 and §4.3 in
`shape-rotator-wrld-knwldge-viz` for the wire format.

## Voice Tagging

VoxTerm learns and remembers speaker voices across sessions:

1. Record a conversation — speakers are detected as "Speaker 1", "Speaker 2", etc.
2. Press `T` to name them — type a name, press Enter
3. Next session, VoxTerm auto-recognizes returning speakers
4. The more you tag, the less you need to — the system learns over time

Press `P` to manage your speaker profile library (rename, delete, wipe all data).

## Models

- **qwen3-0.6b** (default) — fast, good for most use
- **qwen3-1.7b** — more accurate, larger
- Whisper variants (tiny through large-v3) available via `M` menu

Models download automatically on first use.

## Project Structure

```
audio/              Capture, VAD, transcription, diarization, speaker profiles
network/            P2P: discovery, sessions, party mode
tui/                App, widgets, theme
tests/              Test suite
docs/               Design docs and specs
config.py           Constants, paths, settings
```
