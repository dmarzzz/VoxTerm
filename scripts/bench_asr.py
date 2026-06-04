"""Benchmark VoxTerm ASR backends — WER (accuracy) + RTF (CPU speed).

Compares the og faster-whisper models against the optional sherpa-onnx streaming backends
(zipformer-20M, nemotron-0.6B) on a small set of labeled clips. WER is normalized
(uppercase, alphanumerics only) so case/punctuation differences between backends don't skew
it. RTF = wall-clock / audio-duration on CPU (lower = faster; <1.0 = faster than real time).

    python scripts/bench_asr.py            # all installed backends
    python scripts/bench_asr.py fw-small sherpa-nemotron-en   # a subset

Note: the labeled set is tiny (clean LibriSpeech-style read speech bundled with the
zipformer model), so WER differences are within noise — RTF is the meaningful signal.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from audio.transcriber import get_transcriber  # noqa: E402


def load_wav_16k_mono(path) -> "np.ndarray":
    """Load any WAV as float32 mono @ 16 kHz. Self-contained (no gui/ dependency) so the
    benchmark is shippable with the streaming backend. Needs soundfile + scipy."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        data = resample_poly(data, 16000, sr).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def _norm(s: str) -> list[str]:
    return re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / len(r)


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    want = args or ["fw-base", "fw-small", "sherpa-stream-en", "sherpa-nemotron-en"]
    models = [m for m in want if m in config.AVAILABLE_MODELS]
    missing = [m for m in want if m not in config.AVAILABLE_MODELS]
    if missing:
        print(f"(skipping unavailable: {missing} — install voxterm[streaming] for sherpa keys)")

    zip_dir = Path.home() / ".cache" / "voxterm" / "sherpa" / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    trans = zip_dir / "test_wavs" / "trans.txt"
    if not trans.exists():
        print("error: labeled clips not found; load a sherpa model once to fetch them.", file=sys.stderr)
        return 2
    refs = dict(line.split(" ", 1) for line in trans.read_text().splitlines() if " " in line)
    clips = [(zip_dir / "test_wavs" / n, t) for n, t in refs.items() if (zip_dir / "test_wavs" / n).exists()]

    rows = []
    for key in models:
        tr = get_transcriber(key)
        t0 = time.perf_counter()
        tr.load()
        load_s = time.perf_counter() - t0
        wers, audio_s, wall_s = [], 0.0, 0.0
        for wav, ref in clips:
            a = load_wav_16k_mono(wav)
            t1 = time.perf_counter()
            out = tr.transcribe(a)
            wall_s += time.perf_counter() - t1
            audio_s += len(a) / 16000
            wers.append(wer(ref, out.get("text", "")))
        rows.append((key, sum(wers) / len(wers), wall_s / audio_s, load_s))
        print(f"  done {key}: WER={rows[-1][1]:.1%} RTF={rows[-1][2]:.3f} load={load_s:.1f}s", flush=True)

    print("\n| backend | model | WER ↓ | RTF ↓ (CPU) | load |")
    print("|---|---|---|---|---|")
    for key, w, rtf, load_s in rows:
        print(f"| `{key}` | {config.AVAILABLE_MODELS[key]} | {w:.1%} | {rtf:.3f} | {load_s:.1f}s |")
    print(f"\n(clips: {len(clips)} labeled, {sum(len(load_wav_16k_mono(w))/16000 for w,_ in clips):.1f}s total; "
          f"host: {sys.platform}, CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
