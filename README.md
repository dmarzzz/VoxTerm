# VOXTERM

Local real-time voice transcription TUI with speaker diarization. Runs entirely offline — no cloud APIs.

![voxterm](https://img.shields.io/badge/platform-macOS_(Apple_Silicon)-black)

## Setup

```bash
git clone https://github.com/dmarzzz/VoxTerm.git
cd voxterm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

Or use the launcher:
```bash
./voxterm
```

## Controls

| Key | Action |
|-----|--------|
| `R` | Start/pause recording |
| `T` | Tag/name speakers |
| `P` | Speaker profiles |
| `M` | Switch model |
| `L` | Switch language |
| `S` | Save transcript |
| `Y` | Copy to clipboard |
| `C` | Clear transcript |
| `Q` | Quit |

## Voice Tagging

VoxTerm can learn and remember speaker voices across sessions:

1. Record a conversation — speakers are detected as "Speaker 1", "Speaker 2", etc.
2. Press `T` to name them — type a name, press Enter
3. Next session, VoxTerm auto-recognizes returning speakers with confidence scores
4. The more you tag, the less you need to — the system learns over time

Press `P` to manage your speaker profile library (rename, delete, wipe all data).

## Models

- **qwen3-0.6b** (default) — fast, ~1.2GB
- **qwen3-1.7b** — more accurate, ~3.4GB
- Whisper variants (tiny through large-v3) available via `M` menu

Models download automatically on first use.

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.11+
- Microphone access

## Privacy & Voice Data

Voice tagging stores CAM++ speaker embeddings locally on your machine at `~/Library/Application Support/voxterm/.speakers.db`. These embeddings are biometric data — they can identify a person across recordings but cannot be used to reconstruct audio.

**What we do:**
- All processing is local and offline — no data ever leaves your machine
- Embedding BLOBs are **encrypted at rest** with AES-256-CBC + HMAC-SHA256
- Encryption key is auto-generated and stored in your **macOS Keychain** — zero config, no passwords
- Existing unencrypted databases are automatically migrated on first open
- Database file permissions are set to owner-only (`0600`)
- Daily backups with 7-day retention
- `Ctrl+X` in the profiles screen (`P`) permanently deletes all voice data with a `VACUUM` to scrub bytes from disk
- `.gitignore` excludes all speaker databases and transcripts from version control

**Known limitations:**
- **Metadata is not encrypted.** Speaker names, timestamps, and session history are stored in plaintext SQL columns. Only the biometric embedding BLOBs are encrypted. Someone with file access can see *who* spoke and *when*, but not derive a voiceprint.
- **Keychain access.** The encryption key lives in your macOS login Keychain. Any process running as your user *could* request it. FileVault provides the strongest defense for the at-rest case.

**What we explicitly avoid:**
- No subprocess calls for key management — the Security framework is called directly via `ctypes`, so the key never appears in `argv` or the process list
- No additional dependencies — CommonCrypto and Security.framework are built into macOS
