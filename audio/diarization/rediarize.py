"""Deferred whole-session re-diarization correction pass.

The live online path (DiarizationEngine.identify) extracts a single blended
embedding per 1-3s ASR chunk and commits the speaker label permanently, which
benchmarks at ~67-70% DER on AMI fixtures. This module re-clusters the ENTIRE
accumulated session offline with a segmentation-first pipeline (sherpa-onnx:
pyannote-segmentation-3.0 powerset + a 3D-Speaker ERes2Net embedder + global
agglomerative clustering) and the app retroactively rewrites the transcript's
speaker labels. On the same fixtures this measured ~71% -> ~30% DER.

Two pieces:
  - ``SessionRediarizer``: thin, lazily-loaded wrapper over sherpa-onnx that
    turns whole-session audio into global (start, end, cluster_id) segments.
    Pure ONNX, no PyTorch, no HuggingFace token; sherpa-onnx statically links
    its own onnxruntime so it sidesteps VoxTerm's onnxruntime<1.24 leak pin.
  - ``reconcile_labels``: a pure function that maps the global clusters back
    onto existing transcript entries, preserving cross-session names/colors
    where a cluster lines up with an online speaker and minting fresh
    identities where the online path over-merged two real speakers.

The reconciliation is intentionally separate and dependency-free so it is unit
testable without any model or audio.
"""
from __future__ import annotations

import logging
import tarfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

from config import (
    REDIAR_CLUSTER_THRESHOLD, REDIAR_EMBED_URL, REDIAR_MODEL_CACHE,
    REDIAR_SEG_URL, SAMPLE_RATE,
)

log = logging.getLogger(__name__)

import os as _os

_SEG_SUBDIR = "sherpa-onnx-pyannote-segmentation-3-0"
_EMBED_FILENAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def _resolve_models(cache_dir: Path, auto_download: bool) -> tuple[Path, Path] | None:
    """Resolve (segmentation, embedding) ONNX paths, downloading if allowed.

    Honors VOXTERM_SHERPA_MODELS_DIR (a dir containing the seg subdir and an
    embedder .onnx) and VOXTERM_SHERPA_EMBED (an explicit embedder file) so the
    test/eval harness can point at pre-downloaded models without a network hop.
    Returns None if the models cannot be located or fetched.
    """
    override = _os.environ.get("VOXTERM_SHERPA_MODELS_DIR")
    base = Path(override) if override else Path(cache_dir)

    seg = base / _SEG_SUBDIR / "model.onnx"

    embed_override = _os.environ.get("VOXTERM_SHERPA_EMBED")
    if embed_override:
        embed = Path(embed_override)
    else:
        embed = base / _EMBED_FILENAME
        if not embed.exists():
            # fall back to any eres2net/embedder .onnx already in the dir
            cands = sorted(
                p for p in base.glob("*.onnx")
                if "segmentation" not in p.name and "model.onnx" != p.name
            )
            if cands:
                embed = cands[0]

    if seg.exists() and embed.exists():
        return seg, embed

    if not auto_download:
        return None
    # Download to a temp path then atomically rename, so an interrupted/partial
    # download never leaves a corrupt file at the final path (which would pass
    # the .exists() check above forever and silently disable re-diarization).
    try:
        base.mkdir(parents=True, exist_ok=True)
        if not seg.exists():
            log.info("rediarize: downloading segmentation model -> %s", seg.parent)
            tmp = base / "seg.tar.bz2.tmp"
            urllib.request.urlretrieve(REDIAR_SEG_URL, tmp)
            with tarfile.open(tmp, "r:bz2") as tf:
                tf.extractall(base)
            tmp.unlink(missing_ok=True)
        if not embed.exists():
            embed = base / _EMBED_FILENAME
            log.info("rediarize: downloading embedder -> %s", embed)
            tmp = embed.with_suffix(".onnx.tmp")
            urllib.request.urlretrieve(REDIAR_EMBED_URL, tmp)
            _os.replace(tmp, embed)  # atomic promote only on a complete download
    except Exception as e:  # pragma: no cover - network failure path
        log.warning("rediarize: model download failed: %s", e)
        for leftover in (base / "seg.tar.bz2.tmp", base / (_EMBED_FILENAME + ".tmp"),
                         (base / _EMBED_FILENAME).with_suffix(".onnx.tmp")):
            try:
                leftover.unlink(missing_ok=True)
            except Exception:
                pass
        return None
    return (seg, embed) if seg.exists() and embed.exists() else None


class SessionRediarizer:
    """Whole-session offline diarizer (sherpa-onnx). Lazily loaded, fail-soft."""

    def __init__(
        self,
        threshold: float = REDIAR_CLUSTER_THRESHOLD,
        cache_dir: Path | None = None,
        auto_download: bool = True,
    ):
        self._threshold = float(threshold)
        self._cache_dir = cache_dir or REDIAR_MODEL_CACHE
        self._auto_download = auto_download
        self._sd = None
        self._sample_rate = SAMPLE_RATE
        self._unavailable = False  # sticky: don't retry a known-bad load
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True if the diarizer is (or can be) loaded. Triggers a lazy load."""
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._sd is not None:
            return True
        if self._unavailable:
            return False
        with self._load_lock:
            if self._sd is not None:
                return True
            if self._unavailable:
                return False
            try:
                import sherpa_onnx  # noqa: F401
            except Exception as e:
                log.warning("rediarize: sherpa-onnx not importable (%s); "
                            "re-diarization disabled", e)
                self._unavailable = True
                return False
            models = _resolve_models(self._cache_dir, self._auto_download)
            if models is None:
                log.warning("rediarize: models unavailable; re-diarization disabled")
                self._unavailable = True
                return False
            seg, embed = models
            try:
                cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                            model=str(seg)
                        ),
                    ),
                    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embed)),
                    clustering=sherpa_onnx.FastClusteringConfig(
                        num_clusters=-1, threshold=self._threshold
                    ),
                    min_duration_on=0.3,
                    min_duration_off=0.5,
                )
                if not cfg.validate():
                    raise RuntimeError("sherpa-onnx diarization config invalid")
                self._sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
                self._sample_rate = self._sd.sample_rate
            except Exception as e:
                # A corrupt/partial cached model (e.g. an interrupted download)
                # fails to parse here. Remove the cached files so the NEXT launch
                # re-downloads cleanly instead of staying permanently disabled.
                log.warning("rediarize: failed to build diarizer (%s); "
                            "clearing cache for re-download next launch", e)
                if not _os.environ.get("VOXTERM_SHERPA_MODELS_DIR"):
                    for p in (embed, seg):
                        try:
                            Path(p).unlink(missing_ok=True)
                        except Exception:
                            pass
                self._unavailable = True
                return False
        log.info("rediarize: ready (seg=%s embed=%s thr=%.2f)",
                 seg.name, embed.name, self._threshold)
        return True

    def rediarize(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[tuple[float, float, int]]:
        """Re-diarize whole-session audio -> [(start_s, end_s, cluster_id)].

        Returns [] on any failure or if unavailable. Never raises.
        """
        if audio is None or len(audio) == 0:
            return []
        if not self._ensure_loaded():
            return []
        try:
            a = np.ascontiguousarray(audio, dtype=np.float32)
            if sample_rate != self._sample_rate:  # pragma: no cover - app feeds 16k
                log.warning("rediarize: expected %dHz got %dHz; skipping",
                            self._sample_rate, sample_rate)
                return []
            result = self._sd.process(a).sort_by_start_time()
            return [(float(s.start), float(s.end), int(s.speaker)) for s in result]
        except Exception as e:
            log.warning("rediarize: process failed: %s", e)
            return []


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def reconcile_labels(
    global_segs: list[tuple[float, float, int]],
    entries: list[dict],
    online_names: dict[int, str],
    online_colors: dict[int, str],
    palette: list[str],
) -> dict[int, tuple[int, str, str]]:
    """Map global clusters onto transcript entries -> {idx: (sid, label, color)}.

    Only entries whose speaker assignment CHANGES are returned (a relabel set).

    Args:
        global_segs: whole-session (start_s, end_s, cluster_id) from re-diarization.
        entries: per-entry [{"idx", "t0", "t1", "sid"}] in session-seconds.
        online_names/online_colors: current display name/color for each online sid.
        palette: hex colors for freshly-minted speaker identities.

    Reconciliation rules:
      - each entry adopts the cluster it most overlaps in time;
      - clusters are assigned a display identity in size order: the first
        cluster dominated by online speaker S inherits S's name+color (so
        cross-session names survive); a *second* cluster also dominated by S
        (i.e. the online path over-merged two real voices) gets a fresh
        "Speaker N" identity, splitting them apart.
    """
    if not global_segs or not entries:
        return {}

    # 1. each entry -> best-overlapping cluster
    entry_cluster: dict[int, int] = {}
    for e in entries:
        idx, t0, t1 = e["idx"], float(e["t0"]), float(e["t1"])
        best_c, best_ov = None, 0.0
        for s0, s1, c in global_segs:
            ov = _overlap(t0, t1, s0, s1)
            if ov > best_ov:
                best_ov, best_c = ov, c
        if best_c is not None and best_ov > 0.0:
            entry_cluster[idx] = best_c

    # 2. cluster -> {total duration, duration-weighted dominant online sid}
    sid_by_idx = {e["idx"]: int(e["sid"]) for e in entries}
    dur_by_idx = {e["idx"]: max(0.0, float(e["t1"]) - float(e["t0"])) for e in entries}
    cluster_total: dict[int, float] = {}
    cluster_sid_dur: dict[int, dict[int, float]] = {}
    for idx, c in entry_cluster.items():
        d = dur_by_idx[idx]
        cluster_total[c] = cluster_total.get(c, 0.0) + d
        cluster_sid_dur.setdefault(c, {})
        sid = sid_by_idx[idx]
        cluster_sid_dur[c][sid] = cluster_sid_dur[c].get(sid, 0.0) + d

    def dom_sid(c: int) -> int:
        d = cluster_sid_dur.get(c, {})
        return max(d, key=d.get) if d else 0

    # 3. assign display identities, biggest clusters first
    base_id = max([0, *sid_by_idx.values(), *online_names.keys()], default=0)
    next_new = base_id
    claimed: dict[int, int] = {}          # online sid -> cluster that took it
    disp_for_cluster: dict[int, tuple[int, str, str]] = {}
    for c in sorted(cluster_total, key=cluster_total.get, reverse=True):
        dom = dom_sid(c)
        if dom > 0 and dom not in claimed:
            claimed[dom] = c
            name = online_names.get(dom) or f"Speaker {dom}"
            color = online_colors.get(dom) or (palette[(dom - 1) % len(palette)] if palette else "")
            disp_for_cluster[c] = (dom, name, color)
        else:
            next_new += 1
            color = palette[(next_new - 1) % len(palette)] if palette else ""
            disp_for_cluster[c] = (next_new, f"Speaker {next_new}", color)

    # 4. emit relabels for entries whose sid changes
    out: dict[int, tuple[int, str, str]] = {}
    for idx, c in entry_cluster.items():
        disp_sid, name, color = disp_for_cluster[c]
        if disp_sid != sid_by_idx[idx]:
            out[idx] = (disp_sid, name, color)
    return out
