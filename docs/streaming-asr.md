# Streaming ASR (optional backend)

VoxTerm's optional `[streaming]` extra adds a **cross-platform, CPU-only, streaming** ASR
backend via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Unlike the default
faster-whisper (which transcribes in batches after you stop), these models decode
**word-by-word as you speak** — ideal for the live view.

It is 100% **opt-in and additive**: without the extra installed, nothing about VoxTerm
changes (the models simply don't appear).

## Install

```bash
pip install "voxterm[streaming]"
```

The wheel is CPU-only and ships for Linux, macOS (Apple Silicon), and Windows. *(There is no
Intel-macOS wheel, so the key is hidden there.)* sherpa-onnx statically links its own ONNX
Runtime, so it cannot conflict with VoxTerm's `onnxruntime` (used by the diarizer / VAD).

## Models

Two keys appear once the extra is installed (the model downloads to
`~/.cache/voxterm/sherpa/` on first use):

| key | model | character |
|---|---|---|
| `sherpa-stream-en` | streaming zipformer-20M (int8) | ~16× real-time on CPU, rough (small model) |
| `sherpa-nemotron-en` | NeMo FastConformer-RNNT 0.6B (int8) | near-`fw-base` accuracy, ~4× real-time, **streaming** |

See the measured [benchmark](./streaming-asr-benchmark.md).

## Use

- **GUI:** pick the model in the dropdown, then record/transcribe as usual. The **live
  transcript** view automatically prefers `sherpa-stream-en` (when installed) and streams the
  text in word-by-word, finalizing a line on a pause.
- **TUI:** pass the key like any model: `python -m tui.app -m sherpa-nemotron-en`.

The default model is unchanged (`fw-small` on Linux/Intel, MLX on Apple Silicon) — streaming
is something you opt into per use.

## How it works

`audio/transcriber.py:SherpaStreamingTranscriber` wraps sherpa-onnx's `OnlineRecognizer`
(transducer: encoder/decoder/joiner). Per-call `create_stream()` makes it a drop-in for the
existing chunked callers; `gui/engine.py:_live_stream_loop` drives a single persistent stream
for true streaming in the live view (endpoint detection finalizes each line). New model keys
are added to the registry in `audio/transcriber.py` (`_SHERPA_MODEL_URLS`) + gated in
`config.py` (surfaced only when `sherpa_onnx` is importable on a supported platform).

## Notes

- The bundled `sherpa-nemotron-en` is the **English** sibling of NVIDIA's
  `nemotron-3.5-asr-streaming` family; the multilingual `.nemo` checkpoint needs a custom
  ONNX export before it can be wired (the same backend would carry it).
- Reproduce the benchmark: `python scripts/bench_asr.py`. Browser e2e: `python scripts/gui_e2e.py`.
