# Recommendations: Frontier Speaker Diarization for VoxTerm

> Synthesized from 9 research agents, 100+ web searches, 6 Phase 1 surveys + 3 Phase 2 deep dives.
> Date: 2026-03-21

---

## The Problem

2 real speakers → 5+ detected speakers. Over-segmentation dominates.

## Root Causes (from research)

1. **No VAD** — RMS energy threshold lets noise/breath create false speaker embeddings
2. **No quality gating** — every embedding (even from 0.5s of noise) can create a new cluster
3. **Threshold too eager for new speakers** — cosine sim 0.30 (= cosine distance 0.70) vs diart's tuned 1.517 (on [0,2] scale)
4. **EMA centroid update drifts** — alpha=0.95 means new data barely moves the centroid; diart uses running sum instead
5. **Once split, never merged** — no periodic cluster merging
6. **No minimum segment duration** — sub-second fragments create spurious clusters

## Ranked Improvements

### Tier 1: Quick Wins (< 1 day each, no new dependencies, massive impact)

| # | Change | Impact | Effort | Details |
|---|--------|--------|--------|---------|
| 1 | **Raise similarity threshold to 0.55-0.65** | Very High | 1 line | Current 0.30 is wildly too low. Working systems use 0.55-0.70 for ECAPA-TDNN. This alone will cut over-segmentation dramatically. |
| 2 | **Add minimum segment duration filter (1.5s)** | Very High | ~20 lines | Embeddings from < 1.5s audio are unreliable. Assign short segments to the most recent speaker instead of creating new clusters. Research shows 12% relative DER improvement. |
| 3 | **Replace EMA with running sum** | High | ~10 lines | diart uses `centroid += new_embedding` (cosine distance only cares about direction, so sum = stable centroid direction). No drift, no recency bias. Maximally stable. |
| 4 | **Add periodic cluster merging** | High | ~30 lines | Every ~30s, compute pairwise similarity between all centroids. Merge pairs exceeding 0.60. Directly fixes "once split, never merged." |
| 5 | **Add quality gating (rho_update)** | High | ~20 lines | Compute RMS energy of the speech portion. If below threshold, don't use the embedding for clustering (assign to nearest or skip). Prevents noise from corrupting centroids. |

**Expected combined impact of Tier 1: reduce 5+ detected speakers down to 2-3 for a 2-speaker conversation.**

### Tier 2: Silero VAD (1-2 days, adds 1 small dependency)

| # | Change | Impact | Effort | Details |
|---|--------|--------|--------|---------|
| 6 | **Replace RMS VAD with Silero VAD** | Very High | ~80 lines | 0.97 AUC vs ~0.73 for RMS. 1.8MB model, < 1ms per frame, MIT license, PyTorch. Process audio through Silero before buffering — only buffer speech frames. Eliminates the biggest source of bad embeddings. Needs 512-sample frames at 16kHz. |

**Silero VAD is the single highest-impact improvement according to benchmark research (VAD quality = 50-70% of total DER).** Placed in Tier 2 only because it requires a new dependency.

### Tier 3: Embedding Model Upgrade (2-3 days, changes embedding dimension)

| # | Change | Impact | Effort | Details |
|---|--------|--------|--------|---------|
| 7 | **Switch from ECAPA-TDNN to CAM++ (WeSpeaker)** | High | ~150 lines | 7.2M params (vs 14.7M), 512-dim (vs 192), 0.65% EER (vs 0.87%), 2.5x faster on CPU. Available via `wespeaker` pip package with ONNX support. **Migration required**: stored profiles use 192-dim embeddings, need a migration path (re-enroll or dual-model transition). |

### Tier 4: Advanced Clustering (3-5 days, significant architecture change)

| # | Change | Impact | Effort | Details |
|---|--------|--------|--------|---------|
| 8 | **Adopt diart-style constrained clustering** | Very High | ~300 lines | Overlap-aware embeddings, cannot-link constraints via Hungarian assignment, quality gating, conservative new-speaker threshold. This is the "do it right" approach. Requires a segmentation model (pyannote) alongside the embedding model. |
| 9 | **Add spectral clustering with eigengap** | High | ~100 lines | Use NME-SC for automatic speaker count estimation. Apply periodically (every 60s) to re-cluster all session embeddings. Corrects over-segmentation retroactively. Needs `spectralcluster` or `scipy`. |

### Tier 5: Probabilistic Scoring (1-2 weeks, needs training data)

| # | Change | Impact | Effort | Details |
|---|--------|--------|--------|---------|
| 10 | **Add PLDA scoring** | Medium-High | ~200 lines + training | Replaces raw cosine similarity with calibrated log-likelihood ratios. Properly models within-speaker variability, making thresholds consistent regardless of conditions. Needs a pre-trained PLDA model (can train on VoxCeleb embeddings). |
| 11 | **VBx-style speaker continuity prior** | Medium | ~150 lines | Add HMM transition model with loopP=0.99 (99% chance same speaker continues). Strongly discourages rapid speaker switching that causes over-segmentation. |

---

## Recommended Implementation Order

```
Phase 1: Quick wins (Tier 1)                    ← 1-2 days
    Raise threshold, min duration, running sum,
    periodic merging, quality gating

Phase 2: Silero VAD (Tier 2)                    ← 1-2 days
    Replace RMS with neural VAD

Phase 3: Evaluate                               ← 1 day
    Test with real conversations, measure improvement
    Decide if Tier 3+ is needed

Phase 4: Embedding upgrade (Tier 3)             ← 2-3 days
    CAM++ via WeSpeaker, profile migration

Phase 5: Advanced clustering (Tier 4)           ← 3-5 days
    diart-style or spectral clustering
```

**Phase 1 alone should fix the majority of the over-segmentation problem.** The 5 quick wins are all < 30 lines each and require zero new dependencies.

---

## Key Numbers from Research

| Metric | Current | After Tier 1 | After Tier 1+2 | Frontier |
|--------|---------|--------------|----------------|----------|
| Similarity threshold | 0.30 | 0.55-0.65 | 0.55-0.65 | Adaptive |
| VAD | RMS 0.012 | RMS 0.012 | Silero (0.97 AUC) | Silero + OSD |
| Min segment | 2.0s | 1.5s | 1.5s | 0.5s (with quality gate) |
| Centroid update | EMA 0.95 | Running sum | Running sum | Quality-gated sum |
| Cluster merging | Never | Every 30s | Every 30s | Continuous |
| Embedding model | ECAPA-TDNN 192d | ECAPA-TDNN 192d | ECAPA-TDNN 192d | CAM++ 512d |
| Embedding speed | ~100ms | ~100ms | ~100ms | ~40ms |

---

## Research Documents Index

| File | Contents |
|------|----------|
| `embedding-models.md` | 13 models compared: ECAPA-TDNN, CAM++, WavLM, TitaNet, ReDimNet, etc. |
| `online-diarization.md` | 12 online/streaming systems analyzed |
| `frameworks.md` | 9 frameworks compared: pyannote, diart, NeMo, SpeechBrain, etc. |
| `vad-and-overlap.md` | VAD systems (Silero, WebRTC, pyannote) + overlap detection |
| `clustering-algorithms.md` | 14 clustering/scoring algorithms with VoxTerm recommendations |
| `benchmarks.md` | SOTA DER numbers, benchmark datasets, production system insights |
| `deep-dive-diart.md` | diart source code analysis: exact algorithms, parameters, pseudocode |
| `deep-dive-algorithms.md` | Key papers: spectral clustering, VBx, quality gating, EEND |
| `deep-dive-silero-wespeaker.md` | Silero VAD + WeSpeaker CAM++ APIs, integration details |
