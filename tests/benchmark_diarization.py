#!/usr/bin/env python3
"""Full-pipeline diarization benchmark with DER scoring.

Runs audio through VoxTerm's actual pipeline (VAD → SCD → identify_segments)
and computes Diarization Error Rate against RTTM ground truth.

Usage:
    python3 tests/benchmark_diarization.py                    # all fixtures
    python3 tests/benchmark_diarization.py --file tst00       # single file
    python3 tests/benchmark_diarization.py --max-duration 60  # first 60s only
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "speakers"
SAMPLE_RATE = 16000


# ── audio loading ──────────────────────────────────────────


def load_wav(path: Path, max_duration: float | None = None) -> np.ndarray:
    """Load WAV file as float32 numpy array."""
    with open(path, "rb") as f:
        riff = f.read(4)
        assert riff == b"RIFF"
        f.read(4)
        assert f.read(4) == b"WAVE"

        sample_rate = 16000
        bits_per_sample = 16
        audio_format = 1

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                channels = struct.unpack("<H", fmt_data[2:4])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                if max_duration is not None:
                    max_bytes = int(max_duration * sample_rate * (bits_per_sample // 8))
                    chunk_size = min(chunk_size, max_bytes)
                raw = f.read(chunk_size)
                if audio_format == 3:  # float32
                    audio = np.frombuffer(raw, dtype=np.float32).copy()
                elif audio_format == 1 and bits_per_sample == 16:  # PCM int16
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    raise ValueError(f"Unsupported WAV format: {audio_format}/{bits_per_sample}")
                return audio
            else:
                f.read(chunk_size)

    raise ValueError(f"No data chunk in {path}")


# ── RTTM parsing ──────────────────────────────────────────


def parse_rttm(path: Path, file_id: str) -> list[dict]:
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            if parts[1] != file_id:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append({"speaker": speaker, "start": start, "end": start + duration})
    return segments


def get_speakers_at(segments: list[dict], time_sec: float) -> list[str]:
    return [s["speaker"] for s in segments if s["start"] <= time_sec < s["end"]]


# ── DER computation ───────────────────────────────────────


def compute_der(
    ref_segments: list[dict],
    hyp: list[tuple[float, float, int]],
    collar: float = 0.25,
) -> dict:
    """Compute Diarization Error Rate.

    Args:
        ref_segments: ground truth [{speaker, start, end}]
        hyp: hypothesis [(start_sec, end_sec, speaker_id)]
        collar: forgiveness collar in seconds around boundaries

    Returns dict with: DER, miss_rate, false_alarm_rate, confusion_rate,
                        n_ref_speakers, n_hyp_speakers
    """
    # Evaluate at 100ms resolution
    step = 0.1
    ref_speakers = sorted({s["speaker"] for s in ref_segments})
    if not ref_segments:
        return {"DER": 0, "miss_rate": 0, "fa_rate": 0, "confusion_rate": 0,
                "n_ref": 0, "n_hyp": 0}

    max_time = max(s["end"] for s in ref_segments)

    # Build reference label at each frame
    # Build hypothesis label at each frame
    n_frames = int(max_time / step) + 1

    total_speech = 0
    miss = 0
    false_alarm = 0
    confusion = 0

    # Build optimal mapping: hyp_id → ref_speaker
    # Collect co-occurrence counts
    cooccur: dict[tuple[int, str], int] = {}
    for t_idx in range(n_frames):
        t = t_idx * step
        ref_active = get_speakers_at(ref_segments, t)
        hyp_active = []
        for h_start, h_end, h_id in hyp:
            if h_start <= t < h_end:
                hyp_active.append(h_id)

        for h_id in hyp_active:
            for r_spk in ref_active:
                cooccur[(h_id, r_spk)] = cooccur.get((h_id, r_spk), 0) + 1

    # Greedy mapping: assign each hyp_id to its most common ref_speaker
    hyp_to_ref: dict[int, str] = {}
    used_ref: set[str] = set()
    # Sort by count descending for greedy best-first
    sorted_pairs = sorted(cooccur.items(), key=lambda x: -x[1])
    for (h_id, r_spk), count in sorted_pairs:
        if h_id not in hyp_to_ref and r_spk not in used_ref:
            hyp_to_ref[h_id] = r_spk
            used_ref.add(r_spk)

    # Score each frame
    for t_idx in range(n_frames):
        t = t_idx * step
        ref_active = set(get_speakers_at(ref_segments, t))
        hyp_active_ids = set()
        for h_start, h_end, h_id in hyp:
            if h_start <= t < h_end:
                hyp_active_ids.add(h_id)

        # Check collar: skip frames near reference boundaries
        near_boundary = False
        for seg in ref_segments:
            if abs(t - seg["start"]) < collar or abs(t - seg["end"]) < collar:
                near_boundary = True
                break
        if near_boundary:
            continue

        n_ref = len(ref_active)
        n_hyp = len(hyp_active_ids)

        if n_ref == 0 and n_hyp == 0:
            continue
        if n_ref > 0:
            total_speech += n_ref

        if n_ref > 0 and n_hyp == 0:
            miss += n_ref
        elif n_ref == 0 and n_hyp > 0:
            false_alarm += n_hyp
        else:
            # Map hyp IDs to ref speakers
            mapped_ref = {hyp_to_ref.get(h, f"__unknown_{h}") for h in hyp_active_ids}
            # Correct = intersection of ref and mapped hyp
            correct = len(ref_active & mapped_ref)
            # Miss = ref speakers not covered
            miss += max(0, n_ref - n_hyp)
            # Confusion = min(n_ref, n_hyp) - correct
            confusion += max(0, min(n_ref, n_hyp) - correct)
            # False alarm = extra hyp speakers beyond ref
            false_alarm += max(0, n_hyp - n_ref)

    if total_speech == 0:
        total_speech = 1

    return {
        "DER": (miss + false_alarm + confusion) / total_speech,
        "miss_rate": miss / total_speech,
        "fa_rate": false_alarm / total_speech,
        "confusion_rate": confusion / total_speech,
        "n_ref": len(ref_speakers),
        "n_hyp": len({h[2] for h in hyp}),
        "total_speech_frames": total_speech,
    }


# ── full pipeline benchmark ───────────────────────────────


def run_benchmark(
    audio: np.ndarray,
    ref_segments: list[dict],
    chunk_seconds: float = 5.0,
    use_vad: bool = True,
    use_scd: bool = True,
    use_multi: bool = False,
    max_duration: float | None = None,
) -> dict:
    """Run full diarization pipeline and compute DER."""
    from diarization.engine import DiarizationEngine
    from audio.vad import SileroVAD

    engine = DiarizationEngine()
    engine.load()

    vad = SileroVAD() if use_vad else None

    if max_duration:
        max_samples = int(max_duration * SAMPLE_RATE)
        audio = audio[:max_samples]

    duration = len(audio) / SAMPLE_RATE
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)

    hyp_segments: list[tuple[float, float, int]] = []
    t_start = time.time()

    # Process in chunks (simulating real-time)
    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        if len(chunk) < SAMPLE_RATE:  # skip < 1s
            continue

        # VAD: detect speech regions (for gating output, not trimming input)
        if vad and vad.is_loaded:
            speech_regions = vad.get_speech_segments(chunk)
            # Check if there's meaningful speech in this chunk
            total_speech = sum(e - s for s, e in speech_regions)
            if total_speech < len(chunk) * 0.25:
                continue  # skip chunks with < 25% speech

        # Multi-speaker (overlap-aware) or SCD or single identify
        if use_multi:
            results = engine.identify_multi(chunk)
            for label, sid, seg_start, seg_end in results:
                abs_start = (start + seg_start) / SAMPLE_RATE
                abs_end = (start + seg_end) / SAMPLE_RATE
                if vad and vad.is_loaded:
                    for vs, ve in speech_regions:
                        inter_start = max(seg_start, vs)
                        inter_end = min(seg_end, ve)
                        if inter_end > inter_start + SAMPLE_RATE // 4:
                            hyp_segments.append((
                                (start + inter_start) / SAMPLE_RATE,
                                (start + inter_end) / SAMPLE_RATE,
                                sid,
                            ))
                else:
                    hyp_segments.append((abs_start, abs_end, sid))
        elif use_scd:
            results = engine.identify_segments(chunk)
            for label, sid, seg_start, seg_end in results:
                abs_start = (start + seg_start) / SAMPLE_RATE
                abs_end = (start + seg_end) / SAMPLE_RATE
                # Only output where VAD says there's speech
                if vad and vad.is_loaded:
                    for vs, ve in speech_regions:
                        # Intersect hypothesis with speech region
                        inter_start = max(seg_start, vs)
                        inter_end = min(seg_end, ve)
                        if inter_end > inter_start + SAMPLE_RATE // 4:
                            hyp_segments.append((
                                (start + inter_start) / SAMPLE_RATE,
                                (start + inter_end) / SAMPLE_RATE,
                                sid,
                            ))
                else:
                    hyp_segments.append((abs_start, abs_end, sid))
        else:
            label, sid = engine.identify(chunk)
            abs_start = start / SAMPLE_RATE
            abs_end = end / SAMPLE_RATE
            if vad and vad.is_loaded:
                for vs, ve in speech_regions:
                    inter_start = max(0, vs)
                    inter_end = min(end - start, ve)
                    if inter_end > inter_start + SAMPLE_RATE // 4:
                        hyp_segments.append((
                            (start + inter_start) / SAMPLE_RATE,
                            (start + inter_end) / SAMPLE_RATE,
                            sid,
                        ))
            else:
                hyp_segments.append((abs_start, abs_end, sid))

    elapsed = time.time() - t_start
    rtf = elapsed / duration if duration > 0 else 0

    # Compute DER
    der = compute_der(ref_segments, hyp_segments)
    der["rtf"] = rtf
    der["elapsed_sec"] = elapsed
    der["audio_duration"] = duration
    der["n_chunks"] = len(range(0, len(audio), chunk_samples))
    der["n_hyp_segments"] = len(hyp_segments)

    return der


# ── main ──────────────────────────────────────────────────


BENCHMARKS = {
    "dev00": {"rttm_id": "dev00", "desc": "2 speakers, 30s"},
    "tst00": {"rttm_id": "tst00", "desc": "4 speakers, 30s, dense"},
    "ES2014c": {"rttm_id": "ES2014c", "desc": "4 speakers, 38min meeting"},
}


def main():
    parser = argparse.ArgumentParser(description="Diarization benchmark")
    parser.add_argument("--file", type=str, help="Run specific file only")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Max audio duration in seconds")
    parser.add_argument("--chunk", type=float, default=5.0,
                        help="Chunk size in seconds")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--no-scd", action="store_true")
    args = parser.parse_args()

    files = [args.file] if args.file else list(BENCHMARKS.keys())

    print("=" * 72)
    print("VOXTERM DIARIZATION BENCHMARK")
    print("=" * 72)

    for name in files:
        info = BENCHMARKS.get(name)
        if not info:
            print(f"Unknown file: {name}")
            continue

        wav_path = FIXTURES / f"{name}.wav"
        rttm_path = FIXTURES / f"{name}.rttm"
        if not wav_path.exists():
            print(f"  SKIP {name}: {wav_path} not found")
            continue
        if not rttm_path.exists():
            print(f"  SKIP {name}: {rttm_path} not found")
            continue

        desc = info["desc"]
        rttm_id = info["rttm_id"]

        print(f"\n{'─' * 72}")
        print(f"  {name}: {desc}")
        print(f"{'─' * 72}")

        audio = load_wav(wav_path, max_duration=args.max_duration)
        ref = parse_rttm(rttm_path, rttm_id)
        ref_speakers = sorted({s["speaker"] for s in ref})

        print(f"  Audio: {len(audio)/SAMPLE_RATE:.1f}s, "
              f"Ref speakers: {len(ref_speakers)} ({', '.join(ref_speakers)})")
        print(f"  Ref segments: {len(ref)}")
        print(f"  Config: chunk={args.chunk}s, VAD={'ON' if not args.no_vad else 'OFF'}, "
              f"SCD={'ON' if not args.no_scd else 'OFF'}")
        print()

        result = run_benchmark(
            audio, ref,
            chunk_seconds=args.chunk,
            use_vad=not args.no_vad,
            use_scd=not args.no_scd,
            max_duration=args.max_duration,
        )

        der_pct = result["DER"] * 100
        miss_pct = result["miss_rate"] * 100
        fa_pct = result["fa_rate"] * 100
        conf_pct = result["confusion_rate"] * 100

        print(f"  RESULTS:")
        print(f"    DER:        {der_pct:6.1f}%")
        print(f"    Miss:       {miss_pct:6.1f}%")
        print(f"    False Alarm:{fa_pct:6.1f}%")
        print(f"    Confusion:  {conf_pct:6.1f}%")
        print(f"    Speakers:   {result['n_ref']} ref → {result['n_hyp']} detected")
        print(f"    Segments:   {result['n_hyp_segments']} hypothesis")
        print(f"    RTF:        {result['rtf']:.3f} ({result['elapsed_sec']:.1f}s for "
              f"{result['audio_duration']:.1f}s audio)")

    print(f"\n{'=' * 72}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
