#!/usr/bin/env python3
"""Benchmark the deferred re-diarization correction pass — programmatically,
no manual TUI recording required.

It runs the SAME pipeline the app does — the online path builds per-chunk
speaker labels (which tend to over-merge), then the whole-session re-diarization
correction (sherpa-onnx) re-clusters and relabels — and reports speaker counts
and DER BEFORE vs AFTER, across a threshold sweep.

Three ways to use it:

  # 1. Validate on the bundled AMI fixtures (ground truth -> real DER)
  python tests/benchmark_rediarization.py
  python tests/benchmark_rediarization.py --sweep            # threshold sweep

  # 2. Tune for YOUR voices with NO labeling: record a clip, then sweep.
  #    You know how many people were in the room — pick the threshold whose
  #    "corrected speakers" matches that count without over-splitting.
  python tests/benchmark_rediarization.py --record 60 --out mytest.wav
  python tests/benchmark_rediarization.py --wav mytest.wav --sweep

  # 3. With your own ground truth (RTTM, or a simple start,end,speaker CSV)
  python tests/benchmark_rediarization.py --wav clip.wav --rttm clip.rttm
  python tests/benchmark_rediarization.py --wav clip.wav --labels clip.csv

The corrector's models auto-download on first use (or set VOXTERM_SHERPA_MODELS_DIR).
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import benchmark_diarization as bd  # load_wav, parse_rttm, compute_der, get_speakers_at
from audio.diarization.rediarize import SessionRediarizer, reconcile_labels

SR = 16000
FIX = ROOT / "tests" / "fixtures" / "speakers"
PALETTE = ["#00ffcc", "#ff44aa", "#44ff44", "#ffaa00",
           "#aa88ff", "#ff6644", "#44ddff", "#ffff44"]
SWEEP = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


# ── label loading ────────────────────────────────────────────

def load_labels_csv(path: Path) -> list[dict]:
    """Parse a simple `start_sec,end_sec,speaker` CSV into ref segments."""
    segs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("start", "#")):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        segs.append({"start": float(parts[0]), "end": float(parts[1]), "speaker": parts[2]})
    return segs


# ── pipeline ─────────────────────────────────────────────────

def online_entries(audio: np.ndarray, engine):
    """Mirror the app's live path: identify_segments over 5s chunks -> entries+hyp."""
    entries, hyp = [], []
    cs = 5 * SR
    idx = 0
    for start in range(0, len(audio), cs):
        chunk = audio[start:start + cs]
        if len(chunk) < SR:
            continue
        for _, sid, s0, s1 in engine.identify_segments(chunk):
            t0, t1 = (start + s0) / SR, (start + s1) / SR
            entries.append({"idx": idx, "t0": t0, "t1": t1, "sid": sid})
            hyp.append((t0, t1, sid))
            idx += 1
    return entries, hyp


def corrected_hyp(audio, entries, threshold):
    """Run the re-diarization correction at `threshold`; return (hyp, n_clusters)."""
    rd = SessionRediarizer(threshold=threshold)
    if not rd.available:
        return None, 0
    global_segs = rd.rediarize(audio)
    if not global_segs:
        return None, 0
    n_clusters = len({c for _, _, c in global_segs})
    updates = reconcile_labels(global_segs, entries, {}, {}, PALETTE)
    new_sid = {e["idx"]: e["sid"] for e in entries}
    for idx, (sid, _, _) in updates.items():
        new_sid[idx] = sid
    hyp = [(e["t0"], e["t1"], new_sid[e["idx"]]) for e in entries]
    return hyp, n_clusters


def score(ref, hyp):
    """DER% via pyannote.metrics (optimal mapping) if available, else the
    internal scorer. Returns (der_pct, n_hyp_speakers)."""
    n_spk = len({h[2] for h in hyp}) if hyp else 0
    if ref is None or not ref:
        return None, n_spk
    try:
        from pyannote.core import Segment, Annotation
        from pyannote.metrics.diarization import DiarizationErrorRate
        r = Annotation()
        for s in ref:
            r[Segment(s["start"], s["end"])] = s["speaker"]
        h = Annotation()
        for i, (t0, t1, sid) in enumerate(hyp):
            h[Segment(t0, t1)] = str(sid)
        der = DiarizationErrorRate(collar=0.25)(r, h) * 100
        return der, n_spk
    except Exception:
        d = bd.compute_der(ref, hyp)
        return d["DER"] * 100, n_spk


# ── runners ──────────────────────────────────────────────────

def run_one(name, audio, ref, thresholds, engine):
    n_ref = len({s["speaker"] for s in ref}) if ref else None
    entries, hyp_before = online_entries(audio, engine)
    der_before, spk_before = score(ref, hyp_before)
    ref_str = f"ref {n_ref} spk" if n_ref else "no ground truth"
    print(f"\n{'─'*72}\n  {name}  ({len(audio)/SR:.0f}s, {ref_str})\n{'─'*72}")
    bstr = f"{der_before:5.1f}%" if der_before is not None else "   n/a"
    print(f"  ONLINE (live path):   speakers={spk_before:<2d}  DER={bstr}")
    print(f"  {'threshold':>9}  {'corrected_spk':>13}  {'DER':>7}  {'vs online':>10}")
    for t in thresholds:
        hyp, n_clusters = corrected_hyp(audio, entries, t)
        if hyp is None:
            print(f"  {t:>9.2f}  (re-diarizer unavailable — install sherpa-onnx / models)")
            continue
        der_after, spk_after = score(ref, hyp)
        astr = f"{der_after:5.1f}%" if der_after is not None else "   n/a"
        if der_before and der_after is not None and der_before > 0:
            delta = f"{(der_after-der_before)/der_before*100:+4.0f}%"
        else:
            delta = "    —"
        flag = ""
        if n_ref:
            flag = "  <- matches" if spk_after == n_ref else ""
        print(f"  {t:>9.2f}  {spk_after:>13d}  {astr}  {delta:>10}{flag}")


def record_wav(seconds: float, out: Path):
    import sounddevice as sd
    print(f"Recording {seconds:.0f}s at {SR}Hz on the default mic — speak now "
          f"(have everyone take a turn)...")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    pcm = (np.clip(audio.flatten(), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"Saved {out} ({seconds:.0f}s). Now run: "
          f"python tests/benchmark_rediarization.py --wav {out} --sweep")


def main():
    ap = argparse.ArgumentParser(description="Re-diarization correction benchmark")
    ap.add_argument("--wav", type=str, help="Your own 16kHz mono wav")
    ap.add_argument("--rttm", type=str, help="Ground-truth RTTM for --wav")
    ap.add_argument("--labels", type=str, help="Ground-truth CSV (start,end,speaker) for --wav")
    ap.add_argument("--file", type=str, help="Bundled fixture name (dev00, tst00)")
    ap.add_argument("--sweep", action="store_true", help="Sweep cluster thresholds")
    ap.add_argument("--threshold", type=float, help="Single threshold (default: config default)")
    ap.add_argument("--record", type=float, metavar="SECONDS", help="Record a test clip from the mic")
    ap.add_argument("--out", type=str, default="rediar_test.wav", help="Output wav for --record")
    args = ap.parse_args()

    if args.record:
        record_wav(args.record, Path(args.out))
        return

    if args.sweep:
        thresholds = SWEEP
    elif args.threshold is not None:
        thresholds = [args.threshold]
    else:
        from config import REDIAR_CLUSTER_THRESHOLD
        thresholds = [REDIAR_CLUSTER_THRESHOLD]

    print("=" * 72)
    print("RE-DIARIZATION CORRECTION BENCHMARK  (online live path vs corrected)")
    print("=" * 72)

    from audio.diarization.engine import DiarizationEngine
    engine = DiarizationEngine()
    engine.load()

    if args.wav:
        audio = bd.load_wav(Path(args.wav))
        ref = None
        if args.rttm:
            ref = bd.parse_rttm(Path(args.rttm), Path(args.rttm).stem)
        elif args.labels:
            ref = load_labels_csv(Path(args.labels))
        run_one(Path(args.wav).stem, audio, ref, thresholds, engine)
    else:
        names = [args.file] if args.file else ["dev00", "tst00"]
        for name in names:
            wav = FIX / f"{name}.wav"
            if not wav.exists():
                print(f"  SKIP {name}: {wav} not found")
                continue
            audio = bd.load_wav(wav)
            ref = bd.parse_rttm(FIX / f"{name}.rttm", name)
            run_one(name, audio, ref, thresholds, engine)

    print(f"\n{'='*72}")
    print("Tip: pick the lowest threshold whose corrected_spk matches reality")
    print("without over-splitting one person into several.")


if __name__ == "__main__":
    main()
