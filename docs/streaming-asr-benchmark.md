# Streaming ASR benchmark — sherpa-onnx backends vs faster-whisper

VoxTerm's optional `[streaming]` extra adds a cross-platform, CPU-only **streaming** ASR
backend (sherpa-onnx). This compares it against the existing faster-whisper models on
accuracy (WER) and CPU speed (RTF), to justify the addition.

## Results

Host: Linux x86_64, CPU only (no GPU). 3 labeled clips (clean LibriSpeech-style read
speech, 28.2 s total) bundled with the zipformer model. Reproduce with
`python scripts/bench_asr.py`.

| backend (model key) | model | WER ↓ | RTF ↓ (CPU) | streaming | load |
|---|---|---|---|---|---|
| `fw-small` *(og default)* | faster-whisper small | **2.1%** | 0.642 | no (batch) | 2.4 s |
| `fw-base` | faster-whisper base | 5.1% | 0.176 | no (batch) | 8.4 s |
| `sherpa-nemotron-en` | NeMo FastConformer-RNNT 0.6B (int8) | 4.4% | 0.248 | **yes** | 4.4 s |
| `sherpa-stream-en` | zipformer-20M (int8) | 20.9% | **0.064** | **yes** | 1.0 s |

*WER is normalized (uppercase, alphanumerics only) so case/punctuation differences between
backends don't skew it. RTF = wall-clock ÷ audio-duration; lower is faster, <1.0 = faster than
real time.*

## Reading it

- **`fw-small` is the most accurate** (2.1%) and stays the default for the record→stop→
  transcribe (batch) path. But it's batch-only and the slowest here (RTF 0.64).
- **`sherpa-nemotron-en` is the streaming sweet spot:** near-`fw-base` accuracy (4.4%) with
  a healthy ~4× real-time CPU speed (RTF 0.25) **and** native word-by-word streaming — which
  is exactly what the live view wants and which faster-whisper can't do.
- **`sherpa-stream-en` (zipformer-20M) trades accuracy for raw speed:** ~16× real-time
  (RTF 0.064), but 20.9% WER — it's a 20M-param model. Good for ultra-low-latency / weak
  hardware where a rough live caption is fine.
- All three sherpa numbers come from the SAME optional backend; nothing changes for users
  who don't install the extra.

## Honest caveats

- **Tiny labeled set (3 clips / 28 s).** WER differences are within noise — treat WER as
  indicative, not a leaderboard. RTF is the reliable signal here. A rigorous WER pass would
  use full LibriSpeech test-clean (2620 utts), which is too slow to run 4× on CPU for this.
- Clean read speech only; no overlapping speakers, noise, or accents — real-room WER will be
  higher for every backend.
- RTF is single-clip per-call (`tr.transcribe`); the true *streaming* live path feeds frames
  incrementally, so perceived latency is lower than these batch-style RTF numbers suggest.
- nemotron-EN here is the **English sibling**; the multilingual nemotron-3.5 (`.nemo`) needs a
  custom ONNX export before it can be benchmarked (the same backend would carry it once exported).

## Methodology

`scripts/bench_asr.py`: for each installed backend, load via `get_transcriber(key)`, transcribe
each clip with `tr.transcribe()`, compute word-level edit-distance WER vs the bundled
`trans.txt` references, and time the transcribe calls for RTF. Backends absent (e.g. sherpa not
installed) are skipped.
