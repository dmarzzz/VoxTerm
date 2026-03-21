# Overlapping Speech Handling for Real-Time Speaker Diarization

Research date: 2026-03-21
Context: VoxTerm uses CAM++ (512-dim, PyTorch subprocess) + Silero VAD (ONNX). Problem: simultaneous speakers produce blended embeddings that get assigned to the wrong speaker or create phantom speakers.

---

## 1. Overlap-Aware Embedding Extraction (diart approach)

### 1.1 How pyannote/segmentation-3.0 enables overlap awareness

pyannote/segmentation-3.0 is a ~5.9 MB PyanNet model (SincNet + BiLSTM + feed-forward) that ingests 10 seconds of 16kHz mono audio and outputs a `(767 frames, 7 classes)` matrix. The 7 classes use "powerset" encoding:

| Class | Meaning |
|-------|---------|
| 0 | Non-speech |
| 1 | Speaker #1 only |
| 2 | Speaker #2 only |
| 3 | Speaker #3 only |
| 4 | Speakers #1 + #2 |
| 5 | Speakers #1 + #3 |
| 6 | Speakers #2 + #3 |

This means the model **natively detects overlap** -- it doesn't just say "speech/no speech" but tells you which speakers are active at each frame, including combinations.

### 1.2 CPU performance: Can it run in real-time?

**Benchmarks from diart (AMD Ryzen 9 CPU):**
- Segmentation-3.0 inference on 5-second chunk: **~11ms on CPU**, ~8ms on GPU
- Segmentation v1 (older): ~12ms CPU, ~8ms GPU

**Benchmark from the DIART optimization paper (Intel Xeon Gold 5215):**
- Full DIART pipeline (segmentation + embedding + clustering): **~65ms per chunk** mean latency on CPU

**CoreML on Apple Silicon (M1):**
- FluidAudio achieved **0.017 RTF (60x real-time)** using pyannote segmentation-3.0 converted to CoreML, beating Nvidia hardware

**Verdict: Yes, segmentation-3.0 runs comfortably in real-time on CPU.** At 11ms per 5-second chunk, it's far under the 500ms step budget. Even the full diart pipeline at 65ms total is well within real-time. On Apple Silicon, CoreML conversion makes it even faster.

### 1.3 ONNX export status

The ONNX community has published `onnx-community/pyannote-segmentation-3.0` on HuggingFace. The PyTorch model can be exported with:

```python
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    do_constant_folding=True,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
        "logits": {0: "batch_size", 1: "num_frames"},
    },
)
```

**Caveat:** The DIART optimization paper found that ONNX conversion of the *embedding* model actually **degraded latency by 40%** compared to PyTorch on their test hardware. However, the segmentation model is much smaller (~5.9 MB) and should benefit more from ONNX optimizations. Worth benchmarking on Apple Silicon specifically.

### 1.4 The Overlapped Speech Penalty (OSP) -- diart's key insight

diart's most important contribution is the **Overlapped Speech Penalty (OSP)**, which uses the segmentation model's frame-level posteriors to weight the embedding extraction:

```python
def overlapped_speech_penalty(segmentation, gamma=3, beta=10):
    probs = torch.softmax(beta * segmentation, dim=-1)
    weights = torch.pow(segmentation, gamma) * torch.pow(probs, gamma)
    weights[weights < 1e-8] = 1e-8
    return weights
```

**What this does:** For each frame and each local speaker:
- If speaker A dominates (activation 0.9, only speaker): weight = 0.9^3 * 1.0^3 = **0.729** (high)
- If speakers A and B overlap equally (both 0.7): weight = 0.7^3 * 0.5^3 = **0.043** (near zero)

The embedding is then extracted using these weights for statistics pooling. Overlapping frames contribute almost nothing. Each local speaker gets its **own** embedding from frames where only they are speaking.

### 1.5 Lighter alternatives to pyannote segmentation

**No widely available lightweight alternative exists** that provides both VAD and per-speaker overlap detection. The options are:

| Model | Size | Overlap detection | Real-time CPU? |
|-------|------|-------------------|----------------|
| pyannote/segmentation-3.0 | 5.9 MB | Yes (3 speakers, powerset) | Yes (~11ms/5s chunk) |
| Silero VAD v6 | 1.8 MB | No (binary speech/non-speech) | Yes (<1ms) |
| FireRedVAD | 2.2 MB | No | Yes |
| WebRTC VAD | 158 KB | No | Yes |

pyannote segmentation-3.0 is effectively the only viable option for real-time overlap-aware segmentation. At 5.9 MB and 11ms per chunk, it's already lightweight enough. There is no reason to seek a lighter alternative -- it's small and fast.

### 1.6 Integration path for VoxTerm

**Option A: Full diart-style pipeline (recommended)**

Replace the current single-embedding-per-chunk approach with:

1. Run pyannote/segmentation-3.0 on each 5-second chunk (11ms)
2. Apply OSP to get per-speaker weights
3. Extract CAM++ embeddings with weighted statistics pooling
4. Use constrained online clustering (Hungarian assignment for cannot-link)
5. Total latency: ~40-65ms per chunk on CPU (well within real-time)

This requires loading pyannote/segmentation-3.0 in the diarizer subprocess alongside CAM++. Both are PyTorch, so they coexist fine. The segmentation model adds only ~6 MB memory.

**Option B: Segmentation for overlap detection only**

Keep the current CAM++ embedding pipeline but add a binary overlap flag:

1. Run segmentation-3.0 on each chunk
2. Check if any overlap class (4, 5, or 6) is active above threshold
3. If overlap detected: skip centroid update (don't contaminate with blended embedding)
4. Still assign to nearest speaker for labeling, but don't update centroids

This is simpler but loses the per-speaker embedding separation that makes diart effective.

---

## 2. Speech Separation Models

### 2.1 Conv-TasNet

| Property | Value |
|----------|-------|
| Parameters | 5.1M |
| Architecture | Fully convolutional, temporal convolutional network (TCN) |
| Speakers | Fixed (trained for 2 or 3 speakers; must know in advance) |
| Causal latency | <= 2ms (causal variant, SI-SNRi up to 10.6 dB) |
| Non-causal quality | SI-SNRi ~15 dB |
| SpeechBrain | Available as `speechbrain/sepformer-wsj02mix` (but SepFormer, not Conv-TasNet) |
| License | Apache 2.0 (SpeechBrain) |

**CPU performance:** Conv-TasNet is explicitly designed for real-time on CPU. The causal variant achieves <= 2ms latency. The non-causal variant processes faster than real-time on a single CPU core.

**Critical limitation:** Must know the number of speakers in advance (trained for 2 or 3). Cannot handle unknown or variable speaker counts. This is a fundamental constraint of all supervised separation models.

### 2.2 SepFormer

| Property | Value |
|----------|-------|
| Parameters | 6.6M (small) to 26M (SOTA) |
| Architecture | Dual-path transformer |
| Speakers | Fixed (2 or 3) |
| Quality | Best-in-class for separation (SI-SNRi ~22 dB on WSJ0-2Mix) |
| CPU performance | ~153ms for 10 seconds of audio (marginal for real-time) |

**CPU performance:** SepFormer is 10x slower than TDANet on CPU. The small variant is borderline real-time.

### 2.3 TDANet (most CPU-efficient)

| Property | Value |
|----------|-------|
| Parameters | Not published (small) |
| Architecture | Bio-inspired encoder-decoder with top-down attention |
| Speakers | Fixed (2, based on LRS2-2Mix training) |
| CPU performance | **10% of SepFormer's CPU time** (so ~15ms for 10 seconds) |
| MACs | 5% of SepFormer |
| Pretrained | Available on HuggingFace (TDANetBest-2ms-LRS2, TDANetBest-4ms-LRS2) |

TDANet is the most efficient separation model available. On CPU, it runs at roughly 10% the cost of SepFormer. This makes it the only separation model that could plausibly run in real-time on CPU alongside the rest of the pipeline.

### 2.4 Unknown number of speakers

Recent research (2024-2025) addresses separation with unknown speaker counts:
- **Attractor-based methods** (2025) dynamically estimate speaker count during separation
- **Deflationary extraction** iteratively separates one speaker at a time and stops when residual energy drops below a threshold
- **Neural blind source separation** jointly separates and diarizes without knowing speaker count

However, none of these have production-ready pretrained models available, and their CPU performance is not benchmarked for real-time use.

### 2.5 Practical verdict on speech separation

**Speech separation is not recommended for VoxTerm's real-time pipeline** for several reasons:

1. **Fixed speaker count requirement** -- all practical models need to know 2 or 3 speakers
2. **Adds significant complexity** -- separate, re-embed, re-cluster each chunk
3. **Latency budget** -- even Conv-TasNet adds 2-15ms per chunk on top of existing pipeline
4. **Diminishing returns** -- diart's OSP achieves most of the benefit (ignoring overlap frames) without actually separating the signals
5. **Quality vs cost** -- separation quality degrades significantly in causal/real-time mode

**Exception:** Speech separation could be valuable as an **offline post-processing step** on identified overlap regions. After recording, run Conv-TasNet or TDANet on overlap segments only, re-embed the separated outputs, and refine speaker assignments.

---

## 3. Silero VAD v6 Overlap Detection

### 3.1 Capability

**Silero VAD is strictly binary speech/non-speech.** It outputs a single probability that the chunk contains speech. It has **no overlap detection capability whatsoever.**

From the Silero VAD documentation and independent research:
- Silero VAD "was not trained specifically for overlapped speech detection tasks"
- It is "a pre-trained enterprise-grade Voice Activity Detector" for distinguishing speech from silence
- The output is "the probability that the chunk is not silent" -- a single scalar, not per-speaker

### 3.2 Behavior during overlap

When two speakers talk simultaneously, Silero VAD will report high speech probability (it detects speech, just not overlap). This means:
- It won't filter out overlapping segments
- It won't flag them for special handling
- The overlapping audio passes through to the embedding pipeline as normal

### 3.3 What Silero VAD CAN contribute

Silero VAD still helps indirectly with overlap handling:
- Filtering out non-speech segments prevents noise from contaminating embeddings
- Cleaner embeddings mean centroids are more accurate
- More accurate centroids make it easier to detect blended embeddings (they'll be further from any clean centroid)

---

## 4. Practical Overlap Handling Without a Segmentation Model

### 4.1 Signal-level overlap detection (heuristics)

Several audio features correlate with overlapping speech, though none are reliable on their own:

**Spectral flatness:**
- Single-speaker speech has strong harmonic structure (low spectral flatness)
- Overlapping speech blurs harmonics, increasing spectral flatness
- Threshold: spectral flatness > 0.3-0.4 suggests multiple sources
- Problem: also triggered by noise, music, breathy speech

**Energy variance:**
- Single-speaker audio has relatively stable energy within a frame
- Overlapping speech creates energy fluctuations as voices interfere
- Can compute energy variance across sub-frames (e.g., 10ms windows within a 1.5s segment)
- Problem: not very discriminative in practice

**Zero-crossing rate (ZCR):**
- Overlapping voiced speech from two speakers increases ZCR due to beating between fundamental frequencies
- Problem: unreliable, many confounders

**Pitch tracking instability:**
- Track F0 contour across the segment
- Overlapping speakers cause pitch tracker to jump between two F0s or fail
- Sudden F0 discontinuities or pitch tracking failure can indicate overlap
- Problem: requires robust pitch tracker, computational cost

**Practical verdict:** Signal-level heuristics are **too unreliable for production use** as a primary overlap detector. However, they can serve as **cheap auxiliary signals** to increase confidence when combined with embedding-based detection (see 4.2).

### 4.2 Embedding-based overlap detection (recommended heuristic approach)

This is the most practical approach that doesn't require a segmentation model. The key insight: **a blended embedding from two speakers will be equidistant from both speakers' centroids.**

**Detection heuristic:**

```python
def detect_overlap(embedding, centroids, threshold=0.05):
    """Detect if an embedding likely contains overlapping speech.

    Returns True if the embedding is suspiciously equidistant from
    the top-2 centroids (likely a blend of both speakers).
    """
    similarities = {sid: cosine_sim(embedding, c) for sid, c in centroids.items()}
    if len(similarities) < 2:
        return False

    sorted_sims = sorted(similarities.values(), reverse=True)
    top1, top2 = sorted_sims[0], sorted_sims[1]

    # Key signal: top-2 are close AND neither is a confident match
    gap = top1 - top2
    is_ambiguous = gap < threshold
    is_weak_match = top1 < MATCH_THRESHOLD  # e.g., 0.55

    return is_ambiguous and is_weak_match
```

**Why this works:** When speaker A and speaker B overlap:
- The embedding is a weighted average of both voice characteristics
- Cosine similarity to centroid_A will be moderate (not high, because B's voice contaminates)
- Cosine similarity to centroid_B will also be moderate
- The gap between top-1 and top-2 will be small (< 0.05)
- Neither similarity will be above the confident match threshold

**What to do when overlap is detected:**

1. **Don't update any centroid** -- the blended embedding would corrupt both
2. **Assign to the most recent speaker** for labeling (temporal continuity)
3. **Optionally mark the segment** in the transcript (e.g., with a "[+]" indicator)
4. **Track overlap frequency** -- if many consecutive segments are ambiguous, increase caution

**Additional signals to strengthen detection:**

```python
def overlap_confidence(embedding, centroids, audio_rms, spectral_flatness):
    """Multi-signal overlap confidence score."""
    # Embedding signal
    sims = sorted([cosine_sim(embedding, c) for c in centroids.values()], reverse=True)
    emb_signal = 1.0 - (sims[0] - sims[1]) if len(sims) >= 2 else 0.0

    # Spectral signal (high flatness = more overlap-like)
    spec_signal = min(spectral_flatness / 0.4, 1.0)

    # Energy signal (unusually high RMS for this speaker = possible overlay)
    energy_signal = min(audio_rms / (2 * typical_rms), 1.0)

    # Weighted combination
    return 0.6 * emb_signal + 0.25 * spec_signal + 0.15 * energy_signal
```

### 4.3 Discarding overlap segments (centroid protection)

**This is the simplest and most effective immediate improvement.** Even without detecting overlap explicitly, VoxTerm already has the `CONFLICT_MARGIN` (0.05) check:

```python
# Current code in engine.py:
if top_score - sec_score < self.CONFLICT_MARGIN:
    # Ambiguous -- prefer more established speaker
```

The improvement: **when the top-2 are within CONFLICT_MARGIN, also skip centroid update:**

```python
if top_score - sec_score < self.CONFLICT_MARGIN:
    # Ambiguous segment -- likely overlap or speaker boundary
    sid = best_id  # assign for labeling
    skip_centroid_update = True  # DON'T contaminate centroid
```

This single change prevents the worst damage from overlapping speech: centroid drift toward blended embeddings that then cause further misassignment.

### 4.4 Summary of embedding-based approach

| Signal | Reliability | Cost | Use |
|--------|------------|------|-----|
| Top-2 similarity gap < 0.05 | High | Free (already computed) | Primary overlap indicator |
| Neither top-1 above MATCH_THRESHOLD | High | Free | Confirms ambiguity |
| Spectral flatness > 0.35 | Medium | ~0.1ms (FFT) | Auxiliary confirmation |
| RMS energy > 2x speaker's typical | Medium | Free | Auxiliary confirmation |
| F0 tracking failure | Medium | ~1ms | Optional confirmation |

---

## 5. What Production Systems Do for Overlap

### 5.1 WhisperX

WhisperX uses pyannote segmentation for diarization but **acknowledges that overlapping speech is "not handled particularly well"** by either Whisper or WhisperX.

Their approach:
- Run pyannote segmentation to get per-speaker timelines
- Align Whisper's transcription segments to pyannote's speaker segments
- For overlapping regions, **assign to the speaker with the longest intersection duration**
- This is essentially a "pick one speaker" strategy -- no actual separation

### 5.2 AssemblyAI

AssemblyAI's latest diarization (2025) introduces:
- **In-house speaker embedding model** trained with advanced data augmentation specifically targeting noisy and overlapping scenarios
- **Higher resolution processing** (16kHz input vs previous 8kHz)
- **Shorter segment handling** -- accurate for segments as short as 250ms
- **30% improvement** in noisy/far-field scenarios
- Their approach focuses on making embeddings **robust to overlap** rather than separating overlapped speech

Key insight: AssemblyAI's strategy is to make the embedding model itself handle overlap better, not to add a separate overlap detection step.

### 5.3 Deepgram

Deepgram's approach:
- **Per-word diarization** -- assigns speaker labels at the word level, not segment level
- Can "differentiate even when there are overlapping speakers"
- Processes audio in **100ms chunks** for low latency
- Uses a **language-agnostic diarization model** trained on 100,000+ voices
- Processes audio **10x faster** than competitors

Their word-level approach inherently handles some overlap: even in overlapping regions, individual words can often be attributed to specific speakers because the overlap rarely spans every single word.

### 5.4 pyannote community-1 (2025)

The latest open-source approach introduces "exclusive diarization mode":
- **Only one speaker (the most likely to be transcribed)** is active at any given time
- This "dramatically simplifies alignment between STT word timestamps and speaker labels"
- Eliminates "jitter caused by overlapping speech or short backchannels"

This is a pragmatic recognition that for transcription use cases, **the best handling of overlap is to pick one speaker and move on.** The transcription engine (Whisper, Qwen3-ASR) will transcribe whoever is dominant anyway.

### 5.5 Common patterns across production systems

All production systems converge on similar strategies:

1. **Nobody does real-time speech separation in production** -- it's too expensive and fragile
2. **Robust embeddings beat overlap detection** -- make embeddings that work despite overlap
3. **Pick the dominant speaker** -- for transcription, assign overlapping regions to whoever speaks the loudest/longest
4. **Post-processing > real-time correction** -- fix overlap issues after the fact, not during streaming
5. **Word-level > segment-level** -- finer granularity naturally reduces overlap impact

---

## 6. Recommended Implementation Plan for VoxTerm

### Phase 1: Embedding-based overlap protection (immediate, no new dependencies)

**Changes to `diarization/engine.py`:**

1. **Skip centroid update when ambiguous.** When top-2 similarity gap < CONFLICT_MARGIN (0.05), assign to best speaker but don't update centroid. This is a 3-line change.

2. **Track and flag potential overlap.** Add a return field indicating overlap confidence. When the top-2 gap is small and neither exceeds MATCH_THRESHOLD, the segment is likely overlapped.

3. **Add spectral flatness as auxiliary signal.** Cheap to compute (single FFT), provides additional confidence for overlap detection.

**Expected impact:** Prevents centroid contamination from blended embeddings. Won't detect all overlaps but will stop the worst damage (phantom speaker creation, centroid drift).

### Phase 2: pyannote segmentation-3.0 for overlap-aware embeddings (1-2 days)

**Architecture change:**

Load pyannote/segmentation-3.0 (5.9 MB) in the diarizer subprocess alongside CAM++. Both are PyTorch, no conflict.

**Pipeline change:**

```
Current:  audio -> CAM++ -> embedding -> clustering
New:      audio -> segmentation-3.0 -> OSP weights -> CAM++ weighted pooling -> clustering
```

1. Run segmentation-3.0 on each chunk (~11ms on CPU)
2. Apply Overlapped Speech Penalty to get per-speaker frame weights
3. Extract CAM++ embeddings using weighted statistics pooling
4. Each local speaker gets its own clean embedding
5. Use Hungarian assignment for constrained clustering

**This is the diart approach adapted for VoxTerm's architecture.** Total additional latency: ~15-20ms per chunk (segmentation + OSP computation). Well within budget.

**Note:** pyannote/segmentation-3.0 requires accepting HuggingFace terms (even though MIT licensed). The ONNX community version (`onnx-community/pyannote-segmentation-3.0`) may avoid this.

### Phase 3: Exclusive diarization mode (future)

Adopt pyannote community-1's approach: for transcription purposes, output only the dominant speaker per frame. This eliminates the overlap problem for the transcript entirely.

Implementation: after clustering, if multiple speakers are active in a frame, keep only the one with highest activation. This produces a clean, non-overlapping speaker timeline that aligns perfectly with Qwen3-ASR's output.

---

## 7. Key Takeaways

1. **pyannote/segmentation-3.0 runs in real-time on CPU** (~11ms per 5-second chunk). The "too slow for CPU" concern from our earlier research applies to the *full diarization pipeline*, not the segmentation model alone.

2. **diart's Overlapped Speech Penalty is the single most impactful technique** for handling overlap. It doesn't separate speakers -- it just ignores overlapping frames when extracting embeddings. Simple and effective.

3. **Speech separation is not practical for real-time** due to fixed speaker count requirements and latency. Reserve it for offline post-processing.

4. **Silero VAD has zero overlap detection capability.** It's binary speech/non-speech only.

5. **The cheapest effective defense is to skip centroid updates on ambiguous segments.** When the top-2 centroids are within 0.05 cosine similarity, the embedding is likely blended -- don't let it corrupt centroids.

6. **Production systems don't separate overlapping speech either.** They pick the dominant speaker and move on. For transcription, this is the right trade-off.

---

## 8. Sources

### pyannote segmentation and diarization
- [pyannote/segmentation-3.0 -- HuggingFace](https://huggingface.co/pyannote/segmentation-3.0)
- [onnx-community/pyannote-segmentation-3.0 -- HuggingFace](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)
- [pyannote-onnx -- GitHub](https://github.com/pengzhendong/pyannote-onnx)
- [pyannote community-1 -- pyannoteAI blog](https://www.pyannote.ai/blog/community-1)
- [pyannote/speaker-diarization-community-1 -- HuggingFace](https://huggingface.co/pyannote/speaker-diarization-community-1)

### diart framework
- [diart -- GitHub](https://github.com/juanmc2005/diart)
- [DIART optimization paper (2024)](https://arxiv.org/html/2408.02341v1)
- [Near-Real-Time Speaker Diarization on CoreML](https://inference.plus/p/low-latency-speaker-diarization-on)

### Speech separation
- [Conv-TasNet paper](https://arxiv.org/abs/1809.07454)
- [TDANet -- GitHub](https://github.com/JusperLee/TDANet)
- [TDANet paper (ICLR 2023)](https://openreview.net/pdf?id=fzberKYWKsI)
- [SpeechBrain Conv-TasNet docs](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.conv_tasnet.html)
- [Advances in Speech Separation survey (2025)](https://arxiv.org/html/2508.10830v1)
- [Attractor-based separation for unknown speakers (2025)](https://arxiv.org/abs/2505.16607)

### Silero VAD
- [Silero VAD -- GitHub](https://github.com/snakers4/silero-vad)

### Production systems
- [WhisperX -- GitHub](https://github.com/m-bain/whisperX)
- [AssemblyAI diarization update](https://www.assemblyai.com/blog/speaker-diarization-update)
- [Deepgram diarization docs](https://developers.deepgram.com/docs/diarization)
- [Deepgram next-gen diarization](https://deepgram.com/learn/nextgen-speaker-diarization-and-language-detection-models)
- [Best speaker diarization models comparison (2026)](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)

### Overlap detection research
- [Joint Training of Speaker Embedding + OSD (2024)](https://arxiv.org/abs/2411.02165)
- [Overlap-aware low-latency online speaker diarization](https://arxiv.org/abs/2109.06483)
- [CPU-only alternative to pyannote 3.1](https://towardsai.net/p/machine-learning/towards-approximate-fast-diarization-a-cpu-only-alternative-to-pyannote-3-1)
- [Similarity Measurement of Segment-level Speaker Embeddings](https://sites.duke.edu/dkusmiip/files/2022/11/Similarity-Measurement-of-Segment-level-Speaker-Embeddings-in-Speaker-Diarization.pdf)
