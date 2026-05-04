# AGENTS.md - VoxTerm Agent Notes

Use `CLAUDE.md` for the full architecture map. This file captures integration notes that matter when changing VoxTerm.

## ShaderClaw Display Mode

VoxTerm display mode is optional. Core transcription must continue to work when ShaderClaw is not installed.

`voxterm --display`, `voxterm --display-only`, and the in-app `X` key launch ShaderClaw as the renderer for live transcript visuals. VoxTerm does not vendor ShaderClaw. It expects a local ShaderClaw checkout and auto-detects common locations:

- `../shader-claw3` next to the VoxTerm repo
- `~/shader-claw3`
- `~/Code/shader-claw3`
- `~/Developer/shader-claw3`
- `~/Projects/shader-claw3`
- `~/src/shader-claw3`

If ShaderClaw is elsewhere, set:

```bash
export VOXTERM_SHADERCLAW_DIR=/path/to/shader-claw3
```

Fresh setup for display mode:

```bash
cd ~
git clone https://github.com/G3993/ShaderClaw3.git shader-claw3
cd shader-claw3
npm install
```

Then from VoxTerm:

```bash
./voxterm --display
```

## Display Mode Contract

VoxTerm writes live transcripts to `config.LIVE_DIR`. ShaderClaw reads that directory through `VOXTERM_LIVE_DIR` and serves the display at `http://localhost:7778/voxterm`.

When touching this integration:

- Keep `display_mode.py` as the single VoxTerm-side launch surface.
- Keep missing ShaderClaw failures non-fatal and actionable.
- Do not make ShaderClaw a required dependency for normal VoxTerm startup.
- Run `./voxterm --display-only` as a quick smoke test when ShaderClaw is available.

Focused tests:

```bash
.venv/bin/python -m pytest -q tests/test_display_mode.py
```
