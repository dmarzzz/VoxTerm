# Deep Dive: diart -- Online Speaker Diarization Framework

**Source:** https://github.com/juanmc2005/diart
**Paper:** "Overlap-Aware Low-Latency Online Speaker Diarization Based on End-to-End Local Segmentation" (Coria, Bredin, Ghannay, Rosset -- ASRU 2021)
**Research date:** 2026-03-21

---

## 1. Architecture Overview

diart is a streaming speaker diarization pipeline that processes audio in a rolling buffer
updated every 500ms. It combines three components:

1. **Speaker Segmentation Model** -- detects who is speaking when (frame-level posteriors)
2. **Overlap-Aware Speaker Embedding** -- extracts per-speaker embeddings, down-weighting overlap frames
3. **Constrained Incremental Online Clustering** -- maps local speakers to global speaker centroids

The pipeline runs on 5-second audio chunks with 500ms step (hop). Latency is adjustable
between 500ms and 5s. Multiple overlapping predictions are aggregated via a delayed
Hamming-weighted average before binarization.

```
Audio Stream
    |
    v
[5s rolling buffer, 500ms hop]
    |
    v
[Segmentation Model] --> seg: (frames, local_speakers)
    |
    v
[Overlap-Aware Embedding] --> emb: (local_speakers, emb_dim)
    |                               using seg as weights
    v
[Online Clustering] --> SpeakerMap (local -> global)
    |
    v
[Delayed Aggregation + Binarize] --> Annotation
```

---

## 2. The Full Pipeline (`SpeakerDiarization.__call__`)

Pseudocode from the actual source (`src/diart/blocks/diarization.py`):

```
def __call__(waveforms):
    batch = stack(waveforms)                     # (batch, samples, 1)
    segmentations = self.segmentation(batch)      # (batch, frames, local_spk)
    embeddings = self.embedding(batch, segmentations)  # (batch, local_spk, emb_dim)

    for wav, seg, emb in zip(waveforms, segmentations, embeddings):
        # Attach sliding window timestamps to segmentation
        seg = SlidingWindowFeature(seg, sliding_window_from(wav))

        # CORE: update clustering state, get permuted segmentation
        permuted_seg = self.clustering(seg, emb)

        # Buffer management
        self.chunk_buffer.append(wav)
        self.pred_buffer.append(permuted_seg)

        # Aggregate overlapping predictions (Hamming-weighted average)
        agg_prediction = self.pred_aggregation(self.pred_buffer)
        agg_prediction = self.binarize(agg_prediction)  # tau_active threshold

        # Shift timestamps if needed
        # Evict oldest buffer entry when full
```

Key insight: the segmentation model outputs K local speaker tracks per chunk (K=3 for
pyannote/segmentation). These local tracks must be mapped to global speakers each step.
The clustering module handles this mapping.

---

## 3. Overlap-Aware Speaker Embedding

### 3.1 The Problem

Standard x-vector/ECAPA-TDNN embeddings use statistics pooling that treats all frames
equally. When two speakers overlap, the embedding becomes contaminated -- a blend of
both speakers. This causes clustering to create spurious centroids.

### 3.2 The Solution: Overlapped Speech Penalty (OSP)

diart modifies the statistics pooling weights to penalize overlapping frames.

**Source:** `src/diart/blocks/embedding.py` -- class `OverlappedSpeechPenalty`

```python
def overlapped_speech_penalty(segmentation, gamma=3, beta=10):
    # segmentation shape: (batch, frames, speakers)
    probs = torch.softmax(beta * segmentation, dim=-1)
    weights = torch.pow(segmentation, gamma) * torch.pow(probs, gamma)
    weights[weights < 1e-8] = 1e-8
    return weights
```

**What this does, step by step:**

1. `softmax(beta * seg)` -- Sharpens the segmentation posteriors. With beta=10,
   this makes the softmax very peaky. If one speaker dominates a frame, its softmax
   output approaches 1.0. If two speakers are equally active, both get ~0.5.

2. `seg^gamma * probs^gamma` -- Multiplies the raw activation by the sharpened
   probability, both raised to gamma=3. This creates a compound penalty:
   - A frame where speaker A has activation 0.9 and is the only one speaking:
     probs[A] ~ 1.0, weight = 0.9^3 * 1.0^3 = 0.729 (high)
   - A frame where speakers A and B both have activation 0.7:
     probs[A] ~ 0.5, weight = 0.7^3 * 0.5^3 = 0.343 * 0.125 = 0.043 (very low)

3. Floor at 1e-8 to avoid division by zero.

**Effect:** Overlapping speech frames get near-zero weight in embedding extraction.
Low-confidence frames (low activation) also get downweighted. Only high-confidence,
single-speaker frames contribute significantly to the embedding.

### 3.3 Embedding Extraction Flow

```python
class OverlapAwareSpeakerEmbedding:
    def __call__(self, waveform, segmentation):
        weights = self.osp(segmentation)          # Apply overlap penalty
        raw_emb = self.embedding(waveform, weights)  # Weighted stats pooling
        return self.normalize(raw_emb)            # L2 normalize
```

The embedding model receives the per-speaker weights. Internally, the waveform is
replicated K times (once per local speaker), and the embedding model performs
weighted statistics pooling using these weights. Each local speaker gets its own
embedding computed from frames where only that speaker is confidently active.

### 3.4 Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| gamma     | 3       | Exponent for overlap penalty (higher = more aggressive) |
| beta      | 10      | Softmax temperature for sharpening (higher = sharper) |
| norm      | 1       | L2 normalization target |
| normalize_embedding_weights | False | Whether to min-max normalize weights |

### 3.5 Default Models

- **Segmentation:** `pyannote/segmentation` (12ms CPU, 8ms GPU) or `pyannote/segmentation-3.0` (11ms CPU, 8ms GPU)
- **Embedding:** `pyannote/embedding` (26ms CPU, 12ms GPU) -- 512-dim ECAPA-TDNN style
- Also supports: `hbredin/wespeaker-voxceleb-resnet34-LM` (ONNX, 48ms CPU, 15ms GPU), SpeechBrain, NVIDIA TitaNet

---

## 4. Constrained Incremental Online Clustering

This is the core anti-over-segmentation mechanism.

**Source:** `src/diart/blocks/clustering.py` -- class `OnlineSpeakerClustering`

### 4.1 State

```python
self.centers: np.ndarray          # (max_speakers, emb_dim) -- centroid matrix
self.active_centers: set[int]     # indices of known speaker centroids
self.blocked_centers: set[int]    # reserved but unavailable slots
self.max_speakers: int = 20      # hard cap on global speakers
```

### 4.2 The Three Hyperparameters (Anti-Over-Segmentation)

| Parameter   | Default | Tuned (DIHARD III) | Range      | Description |
|-------------|---------|-------------------|------------|-------------|
| tau_active  | 0.6     | 0.555             | [0.0, 1.0] | Min peak activation for a local speaker to be considered "active" |
| rho_update  | 0.3     | 0.422             | [0.0, 1.0] | Min mean activation for a speaker to be "long enough" for centroid creation/update |
| delta_new   | 1.0     | 1.517             | [0.0, 2.0] | Min cosine distance to all existing centroids to create a new speaker |

**How each prevents over-segmentation:**

- **tau_active** acts as a voice activity gate. Local speakers whose peak segmentation
  activation is below this threshold are IGNORED entirely -- no embedding computed,
  no assignment attempted. This filters out phantom speakers from noise.

- **rho_update** acts as a quality gate. Even if a speaker is "active" (passes tau_active),
  their embedding is only used to UPDATE an existing centroid or CREATE a new centroid
  if their mean activation across the chunk exceeds rho_update. Short bursts of speech
  don't modify centroids. This prevents transient noise from corrupting centroids.

- **delta_new** is the NEW SPEAKER GATE. A new global speaker centroid is only created
  when the cosine distance between the local speaker's embedding and ALL existing
  centroids exceeds delta_new. With cosine distance range [0, 2], the DIHARD III
  value of 1.517 is very conservative -- a new speaker must be extremely different
  from everyone seen so far. This is the primary over-segmentation preventer.

### 4.3 The `identify()` Algorithm -- Complete Pseudocode

```
def identify(segmentation, embeddings):
    embeddings = embeddings.numpy()

    # STEP 1: Determine which local speakers are "active" and "long"
    active_speakers = {s : max(seg[:, s]) >= tau_active}
    long_speakers   = {s : mean(seg[:, s]) >= rho_update}
    # Also filter out speakers with NaN embeddings
    active_speakers = active_speakers intersect non_nan_speakers

    # STEP 2: First chunk -- initialize centroids
    if no centroids exist:
        for each active speaker s:
            create new centroid initialized to embeddings[s]
        return hard_map(active -> new centroids)

    # STEP 3: Compute distance matrix (local speakers x global centroids)
    dist_matrix = cdist(embeddings, self.centers, metric="cosine")
    #   shape: (num_local_speakers, max_speakers)

    # STEP 4: Apply cannot-link constraints
    # Mask out INACTIVE local speakers (rows) and INACTIVE global centroids (cols)
    dist_map = SpeakerMap(dist_matrix, MinimizationObjective)
    dist_map = dist_map.unmap_speakers(
        inactive_local_speakers,    # rows set to +inf
        inactive_global_centers     # cols set to +inf
    )

    # STEP 5: Apply delta_new threshold -- remove assignments above threshold
    valid_map = dist_map.unmap_threshold(delta_new)
    # Any (local, global) pair with cosine distance >= delta_new is invalidated

    # STEP 6: Solve optimal assignment (Hungarian algorithm)
    # scipy.optimize.linear_sum_assignment on the masked distance matrix
    # This finds the min-cost bipartite matching
    optimal = linear_sum_assignment(valid_map.mapping_matrix)

    # STEP 7: Handle unmatched speakers
    missed_speakers = active speakers not assigned by valid_map

    for each missed speaker s:
        if num_free_centers > 0 AND s is "long" (passes rho_update):
            # CREATE NEW CENTROID
            flag s for new center creation
        else:
            # FALL BACK: assign to closest free existing centroid
            # (even if distance > delta_new, forced assignment)
            preferences = sort all global speakers by distance
            free = preferences not already assigned
            if free exists:
                assign s -> closest free global speaker

    # STEP 8: Update existing centroids (RUNNING SUM, not EMA)
    for each (local_spk, global_spk) in valid assignments:
        if local_spk is "long" AND local_spk was NOT a missed speaker:
            self.centers[global_spk] += embeddings[local_spk]
            # NOTE: This is += (accumulation), NOT weighted average!

    # STEP 9: Create new centroids for flagged speakers
    for each flagged speaker s:
        center_idx = next_free_center_position()
        self.centers[center_idx] = embeddings[s]    # Initialize to embedding
        self.active_centers.add(center_idx)
        assign s -> center_idx

    return final_speaker_map
```

### 4.4 Cannot-Link Constraints (How They Work)

Cannot-link constraints in diart are implemented **implicitly** through the architecture:

1. The segmentation model outputs K separate local speaker tracks (typically K=3).
   Each track corresponds to a distinct local speaker.

2. When computing the distance matrix, each local speaker gets its OWN embedding
   (extracted with overlap-aware weighting specific to that speaker's activation).

3. The Hungarian algorithm (linear_sum_assignment) enforces a **one-to-one matching**
   constraint: each local speaker maps to a DIFFERENT global centroid.

4. By definition, two local speakers from the same chunk CANNOT be assigned to the
   same global speaker. This is the cannot-link constraint.

The key insight: the segmentation model already knows these are different speakers
within the chunk. The one-to-one constraint on the assignment matrix preserves this.

### 4.5 Centroid Update Mechanism -- RUNNING SUM (Critical Difference from EMA)

**diart uses running sum, NOT exponential moving average (EMA).**

```python
def update(self, assignments, embeddings):
    for local_spk, global_spk in assignments:
        self.centers[global_spk] += embeddings[local_spk]
```

This is a pure accumulation: `center = center + new_embedding`.

**Why this works with cosine distance:**

- Cosine distance only cares about the DIRECTION of the vector, not its magnitude.
- Adding embeddings together creates a vector that points toward the mean direction
  of all accumulated embeddings (like a resultant vector in physics).
- This is mathematically equivalent to computing the centroid direction, because
  cosine_distance(sum_of_vectors, query) ranks the same as
  cosine_distance(mean_of_vectors, query).

**Implications:**
- All past observations contribute equally (no recency bias like EMA).
- The centroid is very stable -- it takes many contradicting observations to shift it.
- Early embeddings have the same influence as late embeddings.
- This is BETTER for preventing over-segmentation: a brief moment of noise won't
  shift the centroid enough to cause a new speaker to be detected.

**Quality gating (rho_update):**
- Centroid updates are ONLY applied when the local speaker passes the rho_update
  threshold (mean activation >= rho_update across the chunk).
- This means only chunks where the speaker talks for a substantial portion contribute
  to centroid updates. Brief utterances don't corrupt centroids.

### 4.6 New Speaker vs. Existing Speaker Decision

The decision tree:

```
For each local speaker in current chunk:
  1. Is max(activation) >= tau_active?
     NO  -> Ignore this speaker entirely
     YES -> Continue

  2. Compute cosine distance to ALL active centroids
  3. Run Hungarian assignment with delta_new threshold

  4. Was this speaker assigned to an existing centroid (dist < delta_new)?
     YES -> Map to that centroid
            If mean(activation) >= rho_update: update centroid via +=
     NO  -> Speaker is "missed" (too far from all centroids)

  5. For missed speakers:
     Is there room for a new centroid (< max_speakers)?
     AND Is mean(activation) >= rho_update? (quality gate)
       YES -> Create new centroid initialized to this embedding
       NO  -> Force-assign to closest available existing centroid
              (even above delta_new threshold)
```

**Critical detail:** A new speaker requires passing BOTH:
- delta_new distance threshold (far enough from all existing centroids)
- rho_update quality threshold (speaking long enough to be trustworthy)

This dual-gating is extremely conservative about creating new speakers.

---

## 5. Prediction Aggregation

### 5.1 Delayed Aggregation

Predictions from overlapping chunks are combined before output. The latency parameter
controls how many overlapping windows are aggregated:

```
num_overlapping_windows = round(latency / step)
```

With default step=0.5s and latency=0.5s, there's 1 window (no aggregation).
With latency=5s (max), there are 10 overlapping windows averaged together.

### 5.2 Hamming-Weighted Average

The default aggregation strategy applies a Hamming window to each overlapping
prediction buffer, then averages. This:
- Gives highest weight to the center of each chunk (most reliable prediction)
- Downweights edges of chunks (less context available)
- Smooths predictions over time, reducing jitter

### 5.3 Binarization

After aggregation, the continuous predictions are binarized using tau_active
as the threshold to produce the final Annotation (who speaks when).

---

## 6. Configuration Defaults

Complete default configuration from `SpeakerDiarizationConfig`:

```python
SpeakerDiarizationConfig(
    segmentation = None,              # defaults to pyannote/segmentation
    embedding = None,                 # defaults to pyannote/embedding
    duration = 5,                     # chunk duration in seconds
    step = 0.5,                       # hop between chunks in seconds
    latency = None,                   # defaults to step (0.5s) if None
    tau_active = 0.6,                 # activity threshold
    rho_update = 0.3,                 # quality gate for centroid updates
    delta_new = 1.0,                  # new speaker distance threshold
    gamma = 3,                        # overlap penalty exponent
    beta = 10,                        # overlap penalty softmax temperature
    max_speakers = 20,                # hard cap on speakers
    normalize_embedding_weights = False,
    device = None,                    # auto-detect CPU/CUDA
    sample_rate = 16000,              # input sample rate
)
```

### 6.1 Dataset-Specific Tuned Values

| Parameter   | Default | DIHARD III | Tuning Range |
|-------------|---------|-----------|--------------|
| tau_active  | 0.6     | 0.555     | [0.0, 1.0]  |
| rho_update  | 0.3     | 0.422     | [0.0, 1.0]  |
| delta_new   | 1.0     | 1.517     | [0.0, 2.0]  |

Hyperparameter optimization uses Optuna with TPE (Tree-structured Parzen Estimator)
sampler. The `diart.tune` CLI automates this.

---

## 7. How diart Prevents Over-Segmentation -- Summary

diart uses FIVE independent mechanisms to prevent over-segmentation:

### 7.1 Overlap-Aware Embeddings
Overlap frames are penalized via `seg^gamma * softmax(beta*seg)^gamma`.
This prevents contaminated embeddings that would create spurious centroids.

### 7.2 Activity Gating (tau_active)
Speakers below the activation threshold are completely invisible to clustering.
The segmentation model's ghost speakers (inactive tracks) never create centroids.

### 7.3 Quality Gating (rho_update)
Only speakers with sustained activity (mean activation >= rho_update) can:
- Create new centroids
- Update existing centroids
Brief noise bursts are ignored.

### 7.4 Conservative New-Speaker Threshold (delta_new)
The cosine distance must exceed delta_new (default 1.0, tuned to 1.517 on DIHARD III)
for a new centroid to be created. On a [0, 2] scale, 1.517 means the embedding must
be nearly orthogonal to all existing centroids. This makes new speaker creation very rare.

### 7.5 Cannot-Link Constraints via Hungarian Assignment
The one-to-one bipartite matching ensures local speakers from the same chunk
map to different global speakers, preventing the segmentation model's separation
from being undone by clustering.

### 7.6 Running-Sum Centroids
By accumulating embeddings rather than using EMA, centroids are maximally stable.
They represent the average direction of ALL observations, making them resistant
to transient noise.

---

## 8. Comparison: diart vs VoxTerm's Current Approach

| Aspect | diart | VoxTerm (current) |
|--------|-------|-------------------|
| Embedding extraction | Overlap-aware weighted pooling | Raw ECAPA-TDNN on full segment |
| Centroid update | Running sum (all history equally weighted) | EMA (exponential decay) |
| New speaker decision | Cosine dist > delta_new + quality gate | Simple threshold |
| Cannot-link | Hungarian one-to-one assignment | None |
| Quality gating | rho_update on mean activation | None |
| Activity gating | tau_active on peak activation | VAD-based |
| Prediction smoothing | Hamming-weighted multi-window aggregation | None |
| Over-segmentation | 5 independent mechanisms | 1 mechanism (threshold) |

### 8.1 Key Takeaways for VoxTerm

1. **Overlap-aware embeddings are critical.** Without them, overlapping speech produces
   blended embeddings that create phantom speakers. VoxTerm should implement weighted
   pooling using segmentation posteriors.

2. **Running sum > EMA for centroids.** EMA introduces recency bias that makes centroids
   drift. Running sum with cosine distance is inherently stable. Consider switching.

3. **Dual-gated new speaker creation.** Requiring BOTH high distance AND sustained speech
   is far more conservative than a single threshold. VoxTerm should add a quality gate.

4. **Cannot-link constraints matter.** Even simple one-to-one matching (via Hungarian
   algorithm) prevents the clustering from merging speakers the segmentation correctly
   separated.

5. **delta_new = 1.517 is very high.** On cosine distance [0, 2], this means a new speaker
   must be nearly opposite from all existing ones. VoxTerm's threshold is likely too low.

6. **Multi-window aggregation smooths noise.** Averaging predictions across multiple
   overlapping windows with Hamming weighting reduces single-chunk errors.

---

## 9. Source File Index

All paths relative to `https://github.com/juanmc2005/diart/tree/main/src/diart/`:

| File | Contains |
|------|----------|
| `blocks/clustering.py` | `OnlineSpeakerClustering` -- the core algorithm |
| `blocks/diarization.py` | `SpeakerDiarization`, `SpeakerDiarizationConfig` -- pipeline orchestration |
| `blocks/embedding.py` | `OverlapAwareSpeakerEmbedding`, `OverlappedSpeechPenalty` |
| `blocks/segmentation.py` | `SpeakerSegmentation` -- model wrapper |
| `blocks/aggregation.py` | `DelayedAggregation`, `HammingWeightedAverageStrategy` |
| `blocks/base.py` | `Pipeline`, `PipelineConfig`, `HyperParameter` definitions |
| `blocks/utils.py` | `Binarize` -- threshold-based annotation conversion |
| `mapping.py` | `SpeakerMap`, `SpeakerMapBuilder` -- Hungarian assignment |
| `functional.py` | `overlapped_speech_penalty()`, `normalize_embeddings()` |
| `models.py` | `EmbeddingModel`, `SegmentationModel` -- model loading |
| `optim.py` | `Optimizer` -- Optuna-based hyperparameter tuning |
