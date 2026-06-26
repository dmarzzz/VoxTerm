"""End-to-end: the deferred re-diarization correction lowers per-entry DER.

Builds transcript entries from the live online engine (the path that benchmarks
at ~71% DER), then runs the real SessionRediarizer + reconcile_labels over the
whole clip and checks the corrected per-entry labels score materially better.

Skipped unless sherpa-onnx and the segmentation+embedder models are resolvable
(point VOXTERM_SHERPA_MODELS_DIR / VOXTERM_SHERPA_EMBED at a model dir, e.g. the
.context/sherpa_models the dev eval downloads).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import benchmark_diarization as bd  # noqa: E402
from audio.diarization.rediarize import SessionRediarizer, reconcile_labels  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "speakers"
SR = 16000
_DEFAULT_MODELS = ROOT / ".context" / "sherpa_models"


def _ensure_model_env():
    if not os.environ.get("VOXTERM_SHERPA_MODELS_DIR") and _DEFAULT_MODELS.exists():
        os.environ["VOXTERM_SHERPA_MODELS_DIR"] = str(_DEFAULT_MODELS)
        for cand in ("eres2net_large.onnx", "eres2net_base.onnx"):
            if (_DEFAULT_MODELS / cand).exists():
                os.environ.setdefault("VOXTERM_SHERPA_EMBED", str(_DEFAULT_MODELS / cand))
                break


def _online_entries(audio, engine):
    """Mimic app tracking: identify_segments over 5s chunks -> entries + hyp."""
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


@pytest.mark.parametrize("name", ["dev00"])
def test_rediarization_lowers_der(name):
    _ensure_model_env()
    rd = SessionRediarizer()
    if not rd.available:
        pytest.skip("sherpa-onnx or models unavailable; set VOXTERM_SHERPA_MODELS_DIR")

    from audio.diarization.engine import DiarizationEngine
    audio = bd.load_wav(FIX / f"{name}.wav")
    ref = bd.parse_rttm(FIX / f"{name}.rttm", name)

    eng = DiarizationEngine()
    eng.load()
    entries, hyp_before = _online_entries(audio, eng)
    assert entries, "online path produced no entries"
    der_before = bd.compute_der(ref, hyp_before)["DER"]

    global_segs = rd.rediarize(audio)
    assert global_segs, "rediarize produced no segments"
    updates = reconcile_labels(global_segs, entries, {}, {},
                               ["#00ffcc", "#ff44aa", "#44ff44", "#ffaa00"])

    new_sid = {e["idx"]: e["sid"] for e in entries}
    for idx, (sid, _, _) in updates.items():
        new_sid[idx] = sid
    hyp_after = [(e["t0"], e["t1"], new_sid[e["idx"]]) for e in entries]
    der_after = bd.compute_der(ref, hyp_after)["DER"]

    # The correction must materially reduce DER (dev00 measured ~71% -> ~30%).
    assert der_after < der_before * 0.75, (
        f"{name}: DER not improved enough: before={der_before:.3f} after={der_after:.3f}"
    )
