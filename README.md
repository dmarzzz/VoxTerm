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
| `M` | Switch model |
| `L` | Switch language |
| `S` | Save transcript |
| `Y` | Copy to clipboard |
| `C` | Clear transcript |
| `Q` | Quit |

## Models

- **qwen3-0.6b** (default) — fast, ~1.2GB
- **qwen3-1.7b** — more accurate, ~3.4GB
- Whisper variants (tiny through large-v3) available via `M` menu

Models download automatically on first use.

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.11+
- Microphone access
