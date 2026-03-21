# Deep Dive: Speaker Diarization Algorithms to Prevent Over-Segmentation

> Research conducted 2026-03-21 for VoxTerm.
> Problem: VoxTerm over-segments speakers (2 real speakers detected as 5+).
> Goal: Identify practical algorithmic improvements with implementation details.

---

## Table of Contents

1. [Spectral Clustering with Eigengap (NME-SC)](#1-spectral-clustering-with-eigengap-nme-sc)
2. [VBx: Bayesian HMM over X-Vectors](#2-vbx-bayesian-hmm-over-x-vectors)
3. [Sticky HDP-HMM: Self-Transition Bias](#3-sticky-hdp-hmm-self-transition-bias)
4. [Quality-Gated Embedding Updates](#4-quality-gated-embedding-updates)
5. [Periodic Cluster Merging](#5-periodic-cluster-merging)
6. [Diart: Online Streaming Diarization](#6-diart-online-streaming-diarization)
7. [Pyannote Powerset Segmentation](#7-pyannote-powerset-segmentation)
8. [EEND: End-to-End Neural Diarization](#8-eend-end-to-end-neural-diarization)
9. [UIS-RNN with Chinese Restaurant Process](#9-uis-rnn-with-chinese-restaurant-process)
10. [Minimum Segment Duration Filtering](#10-minimum-segment-duration-filtering)
11. [Cosine Similarity Threshold Calibration](#11-cosine-similarity-threshold-calibration)
12. [Multiscale Diarization (MSDD)](#12-multiscale-diarization-msdd)
13. [Practical Recommendations for VoxTerm](#13-practical-recommendations-for-voxterm)

---

## 1. Spectral Clustering with Eigengap (NME-SC)

**Paper:** Park et al., "Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap" (IEEE SPL 2020, [arXiv:2003.02405](https://arxiv.org/abs/2003.02405))

**Code:** [tango4j/Auto-Tuning-Spectral-Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering), [wq2012/SpectralCluster](https://github.com/wq2012/SpectralCluster) (Google's implementation)

### Core Algorithm (Step by Step)

1. **Compute affinity matrix** from raw cosine similarity between speaker embeddings:
   ```
   A_ij = cos(w_i, w_j) = (w_i . w_j) / (||w_i|| * ||w_j||)
   ```
   Key insight: use raw cosine similarity directly, no kernel or scaling parameter needed.

2. **Sweep p-percentile threshold** over range `p in [1, floor(N/4)]`:
   - For each row in A, keep the p largest values as 1, zero out the rest (binarization)
   - Symmetrize: `A_p = max(A_p, A_p^T)`

3. **Compute unnormalized Laplacian** for each p:
   ```
   L_p = D_p - A_p     (D_p = degree matrix)
   ```

4. **SVD to get eigenvalues**, then compute eigengap vector:
   ```
   e_p = [lambda_{p,2} - lambda_{p,1}, lambda_{p,3} - lambda_{p,2}, ..., lambda_{p,N} - lambda_{p,N-1}]
   ```

5. **Compute Normalized Maximum Eigengap (NME)**:
   ```
   g_p = max(e_p) / (lambda_{p,N} + epsilon)       epsilon = 1e-10
   r(p) = p / g_p
   ```

6. **Select optimal p** that minimizes r(p):
   ```
   p_hat = argmin_p r(p)
   ```
   The ratio r(p) is a proxy for DER -- lower is better.

7. **Estimate speaker count** from the eigengap at optimal p:
   ```
   k = argmax(e_{p_hat})
   ```

8. **Extract spectral embeddings** from the k smallest eigenvectors.

9. **K-means clustering** on the spectral embeddings to produce final labels.

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `epsilon` | 1e-10 | Prevents division by zero in NME |
| `max_speakers` | 8 | Upper bound on speaker count search |
| `p range` | [1, N/4] | Search space for p-percentile |
| `distance` | cosine | Affinity metric |

### How It Prevents Over-Segmentation

- The NME criterion automatically finds the "right" number of clusters by identifying the largest gap in the eigenvalue spectrum.
- Unlike manual threshold tuning, NME self-calibrates: when there are truly 2 speakers, the eigengap at k=2 will be much larger than at k=5, so the algorithm correctly identifies 2.
- The p-percentile sweep ensures the affinity matrix sparsity is also auto-tuned.

### Computational Cost

- **Complexity:** O(N^2) for affinity matrix, O(N^2 * N/4) for p-sweep with SVD
- **Real-time feasible?** Yes for moderate N (< 500 segments). The SpectralCluster library handles streaming with incremental updates.
- **Benchmarks:** CALLHOME = 7.24% speaker error with oracle VAD

### Python Implementation Complexity

- ~50 lines using SpectralCluster library (which wraps scipy/sklearn)
- ~200 lines from scratch using numpy + scipy.linalg

### Training Data Required?

**No.** Works out-of-the-box with any speaker embeddings. No supervised training needed.

### Quick Implementation

```python
from spectralcluster import SpectralClusterer

clusterer = SpectralClusterer(
    min_clusters=1,
    max_clusters=8,
    custom_dist="cosine"
)
# X is (num_segments, embedding_dim) numpy array
labels = clusterer.predict(X)
```

Or with Google's pre-configured settings:
```python
from spectralcluster import configs
labels = configs.icassp2018_clusterer.predict(X)
```

---

## 2. VBx: Bayesian HMM over X-Vectors

**Paper:** Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization" (Computer Speech and Language 2021, [arXiv:2012.14952](https://arxiv.org/abs/2012.14952))

**Code:** [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx)

**Tutorial:** [desh2608.github.io/2022-04-10-gbo-vb/](https://desh2608.github.io/2022-04-10-gbo-vb/)

### Core Algorithm (Step by Step)

1. **Extract x-vectors** from audio segments (typically 1.5s windows, 0.75s shift).

2. **Initialize with AHC** (Agglomerative Hierarchical Clustering) to get initial speaker assignments. For files > 30 min, use random initialization (faster).

3. **Define Bayesian HMM generative model:**
   - Speakers are hidden HMM states
   - X-vectors are emissions from Gaussian distributions
   - Speaker means are drawn from a zero-centered Gaussian prior:
     ```
     p(x_t | y_s) = N(x_t; V * y_s, I)
     ```
     where V is the eigenvoice matrix and y_s is the speaker latent variable.

4. **Variational Bayes inference** via ELBO optimization:
   ```
   ELBO(q) = E_q[log p(X|Z)] - KL(q(Z) || p(Z))
   ```
   Uses mean-field approximation: `q(Z) = prod_j q_j(z_j)`

5. **Alternating updates** (similar to EM / forward-backward):
   - **E-step:** Update q(Z) -- speaker assignments using forward-backward algorithm
   - **M-step:** Update q(Y) -- speaker parameters given assignments

6. **Iterate** until ELBO converges (typically 5-20 iterations).

7. **Output** final speaker labels from converged q(Z).

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Fa` | 0.3 (general), 1.0 (CALLHOME) | Acoustic scaling factor. Counteracts the independence assumption between observations. **Most important parameter.** |
| `Fb` | 17 - 17.18 | Speaker regularization coefficient. Higher Fb = fewer output speakers. **Directly controls over-segmentation.** |
| `loopP` | 0.99 | Self-loop probability in HMM. Values < 1 = HMM mode. Value = 1 = GMM mode (no temporal constraints). |

### How It Prevents Over-Segmentation

1. **Bayesian prior on speaker parameters** (zero-centered Gaussian) regularizes the solution -- extra clusters are penalized because they consume prior probability mass.
2. **Fb parameter** directly controls the penalty for creating new speakers. Higher Fb = stronger preference for fewer speakers. This is the primary knob for fighting over-segmentation.
3. **HMM transition model** (loopP = 0.99) strongly favors speaker continuity -- once in a speaker state, the model stays there with 99% probability per frame.
4. **Variational inference** naturally prunes unnecessary clusters during optimization.

### Computational Cost

- **Complexity:** O(K * T * D) per iteration, where K = speakers, T = frames, D = embedding dim
- **Real-time feasible?** The VBx inference itself is fast (seconds for minutes of audio). The bottleneck is x-vector extraction.
- **Caveat:** AHC initialization can be slow for files > 30 minutes.

### Python Implementation Complexity

- ~100 lines to implement the VB update loop (given embeddings and eigenvoice matrix)
- ~300 lines for the full pipeline including AHC initialization
- The BUTSpeechFIT/VBx repo is a complete reference implementation

### Training Data Required?

**Partially.** The eigenvoice matrix V is typically pre-trained on speaker verification data (e.g., VoxCeleb). However, the VBx inference itself is unsupervised. Pre-trained models are available in the repo.

---

## 3. Sticky HDP-HMM: Self-Transition Bias

**Paper:** Fox et al., "A Sticky HDP-HMM with Application to Speaker Diarization" (Annals of Applied Statistics, 2011)

**PDF:** [MIT DSpace](https://dspace.mit.edu/bitstream/handle/1721.1/79665/sticky_hdphmm_final_submission.pdf)

### Core Algorithm

The sticky HDP-HMM augments the standard Hierarchical Dirichlet Process HMM with a self-transition bias parameter kappa:

1. **Standard HDP-HMM:** Each speaker state has a transition distribution drawn from a Dirichlet process. The number of speaker states is theoretically unbounded (nonparametric).

2. **Sticky modification:** Add extra probability mass kappa to the diagonal of the transition matrix:
   ```
   pi_j ~ DP(alpha + kappa, (alpha * beta + kappa * delta_j) / (alpha + kappa))
   ```
   where delta_j is a point mass at state j (self-transition).

3. **Effect:** For positive kappa, the prior probability of self-transitions is boosted. This makes the model "sticky" -- it prefers to stay in the current speaker state rather than switching.

4. **Inference:** Gibbs sampling or variational inference to find posterior over states and parameters.

### Key Parameters

| Parameter | Effect |
|-----------|--------|
| `kappa` | Self-transition bias strength. Higher kappa = more sticky = fewer transitions = less over-segmentation |
| `alpha` | DP concentration parameter. Controls prior over transition distribution |
| `gamma` | Top-level DP concentration. Controls tendency to create new speakers |

### How It Prevents Over-Segmentation

- Without kappa, the standard HDP-HMM tends to create too many rapid state transitions, over-segmenting the audio.
- The kappa parameter adds inertia: the model needs stronger evidence to switch speakers.
- The nonparametric nature still allows discovering new speakers when warranted, but the sticky prior suppresses spurious transitions.

### Computational Cost

- Gibbs sampling is expensive: O(K^2 * T) per iteration, many iterations needed (hundreds to thousands).
- **Not real-time feasible** in its original formulation.
- Variational inference variants are faster but less principled.

### Python Implementation Complexity

- ~500-1000 lines from scratch (Gibbs sampler + HDP bookkeeping)
- Available in pyhsmm library
- **Recommendation:** Use VBx instead -- it achieves a similar effect more efficiently via loopP parameter.

### Training Data Required?

**No** for the HDP-HMM structure. Yes for the observation model (speaker embeddings still need pre-trained extractors).

---

## 4. Quality-Gated Embedding Updates

**Paper:** "Investigating Confidence Estimation Measures for Speaker Diarization" ([arXiv:2406.17124](https://arxiv.org/html/2406.17124v1))

### Core Concept

Not all speaker embeddings are equally reliable. Short segments, overlapping speech, background noise, and reverberation produce low-quality embeddings that can cause clustering errors (including over-segmentation). Quality-gating means: measure embedding quality, then down-weight or discard poor embeddings.

### Confidence Scoring Methods

#### Method 1: Cosine Similarity Score
```
confidence(x) = mean(cos_sim(x, centroid_of_assigned_speaker))
```
Simple but effective baseline. Embeddings far from their cluster centroid are flagged as low-confidence.

#### Method 2: Silhouette Score (Best Performer)
```
s(x) = (b(x) - a(x)) / max(a(x), b(x))
```
Where:
- `a(x)` = distance to assigned speaker centroid (compactness)
- `b(x)` = distance to nearest competing speaker centroid (separation)

Range: [-1, +1]. Higher = more confident assignment.

**This was the strongest performer** across all systems and datasets tested.

#### Method 3: Local Confidence Score
1. Compute initial confidence as cosine similarity to centroid.
2. Remove embeddings > 2 standard deviations from centroid.
3. Recompute centroid without outliers.
4. Recompute confidence with cleaned centroid.
5. Iterate until convergence.

This outlier-resistant approach stabilizes cluster centroids.

### Quality-Gating for Over-Segmentation Prevention

Key finding from the paper:
- **~30% of diarization errors occur in the lowest ~10% confidence segments.**
- **~55% of diarization errors occur in the lowest ~30% confidence segments.**

Practical implication: by discarding or down-weighting the worst 10-30% of embeddings before/during clustering, you can eliminate a large fraction of errors.

### Implementation for VoxTerm

```python
import numpy as np
from sklearn.metrics import silhouette_samples

def quality_gate_embeddings(embeddings, labels, threshold=-0.1):
    """Remove low-quality embeddings before re-clustering."""
    scores = silhouette_samples(embeddings, labels, metric='cosine')
    mask = scores > threshold
    return embeddings[mask], labels[mask], scores

def weighted_centroid(embeddings, scores):
    """Compute quality-weighted speaker centroid."""
    weights = np.clip(scores, 0, 1)
    return np.average(embeddings, axis=0, weights=weights)
```

### Computational Cost

- Silhouette score: O(N * K) where N = embeddings, K = clusters
- **Real-time feasible:** Yes, very lightweight
- **Implementation:** ~20-30 lines of Python

### Training Data Required?

**No.** Works entirely on the embeddings themselves.

---

## 5. Periodic Cluster Merging

**Context:** Agglomerative Hierarchical Clustering (AHC) and post-processing refinement.

**Key References:**
- [Introduction to Speech Processing, Aalto University](https://speechprocessingbook.aalto.fi/Recognition/Speaker_Diarization.html)
- [Park et al., "A review of speaker diarization"](https://sail.usc.edu/publications/files/Park-Diarization-CSL2022.pdf)

### Core Algorithm

#### AHC-Based Merging (Bottom-Up)

1. Start with each speech segment as its own cluster.
2. Compute pairwise distance matrix between all clusters (using BIC, cosine, or PLDA scoring).
3. Merge the two most similar clusters.
4. Update the distance matrix.
5. Repeat until stopping criterion is met.

**Stopping criteria:**
- BIC-based: stop when `max(delta_BIC) < 0` (no merge improves BIC)
- Threshold-based: stop when `max_similarity < threshold`
- Fixed count: stop when target number of clusters reached

#### Post-Processing Refinement (3-stage)

1. **Refine speech/non-speech segmentation:** Re-estimate VAD boundaries using speaker models.
2. **Assimilate short speech segments:** Merge segments shorter than minimum duration into nearest speaker cluster.
3. **Fuse same-speaker clusters:** After initial clustering, compute inter-cluster similarity and merge clusters that likely belong to the same speaker.

#### Resegmentation (Viterbi Refinement)

After clustering:
1. Train a GMM for each speaker using assigned segments.
2. Run Viterbi decoding over the full audio with the speaker GMMs.
3. Re-assign segments to speakers based on Viterbi path.
4. Iterate 2-3 times.

This corrects boundary errors and merges accidentally-split speakers.

### How It Prevents Over-Segmentation

- **Periodic re-merging** catches clusters that should have been merged but weren't (e.g., due to noisy embeddings early in the conversation).
- **Short segment assimilation** removes micro-segments that create phantom speakers.
- **Viterbi resegmentation** enforces temporal smoothness, reducing speaker fragmentation.

### Key Parameters

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| `min_segment_duration` | 2.5 seconds | Minimum speech segment length |
| `merge_threshold` | Dataset-dependent | Similarity threshold for cluster merging |
| `n_resegmentation_iters` | 2-3 | Viterbi refinement iterations |

### Computational Cost

- AHC: O(N^2 log N)
- Viterbi: O(K * T) per iteration
- **Real-time feasible?** AHC yes (with incremental updates). Viterbi is batch-only.

### Python Implementation Complexity

- AHC: ~50 lines using scipy.cluster.hierarchy
- Short segment filtering: ~20 lines
- Viterbi resegmentation: ~100-150 lines

### Training Data Required?

**No** for AHC and filtering. GMM-based resegmentation is unsupervised (trains on the audio itself).

---

## 6. Diart: Online Streaming Diarization

**Paper:** Coria et al., "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation" (IEEE ASRU 2021)

**Code:** [juanmc2005/diart](https://github.com/juanmc2005/diart)

**JOSS Paper:** [10.21105/joss.05266](https://joss.theoj.org/papers/10.21105/joss.05266)

### Core Algorithm (Step by Step)

1. **Audio streaming:** Process audio in rolling buffer of 5 seconds, updated every 500ms (step size).

2. **Local segmentation:** Run pyannote segmentation model on the 5s buffer to detect speaker activity and overlaps. Output: frame-level speaker probabilities for up to 3 local speakers.

3. **Overlap-aware embedding extraction:** Use modified statistics pooling that down-weights frames where overlapping speakers are detected. This produces cleaner single-speaker embeddings.

4. **Incremental clustering:** Compare new embeddings to existing speaker centroids.
   - If similarity > threshold: assign to existing speaker and update centroid
   - If similarity < threshold: create new speaker cluster

5. **Cannot-link constraints:** From the segmentation output, if two local speakers are detected in the same frame, they cannot be the same global speaker. This prevents merging.

6. **Output:** Emit RTTM annotations in real-time as speakers are identified.

### Key Parameters

| Parameter | Default | DIHARD III | AMI | VoxConverse | Description |
|-----------|---------|------------|-----|-------------|-------------|
| `tau_active` | 0.6 | 0.555 | 0.507 | 0.576 | Speech activity threshold (speaker is active if probability > tau_active) |
| `rho_update` | 0.3 | 0.422 | 0.006 | 0.915 | Centroid update threshold (only update centroid if confidence > rho_update) |
| `delta_new` | 1.0 | 1.517 | 1.057 | 0.648 | New speaker threshold (create new cluster if max similarity < delta_new) |
| `latency` | 5s | -- | -- | -- | Buffer duration. Lower = faster but less accurate |

### How It Prevents Over-Segmentation

- **delta_new threshold** is the primary knob. Higher delta_new = harder to create new speakers = fewer false clusters.
- **rho_update** prevents low-quality embeddings from corrupting centroids. Low rho_update (like AMI's 0.006) means almost all embeddings update centroids; high rho_update (like VoxConverse's 0.915) means only very confident embeddings update.
- **Cannot-link constraints** from the segmentation model prevent two actually-different speakers from being merged, but don't create false splits.
- **Centroid averaging** over time stabilizes cluster representations.

### Computational Cost

| Component | CPU Time | GPU Time |
|-----------|----------|----------|
| Segmentation | 11-12ms | 8ms |
| Embedding | 26-150ms | 12-29ms |
| Clustering | < 1ms | < 1ms |
| **Total per step** | **~40-160ms** | **~20-40ms** |

**Real-time feasible?** Yes. The 500ms step size gives ample budget. GPU recommended for comfort.

### Python Implementation Complexity

- ~20 lines for basic usage with diart library
- ~200-300 lines to reimplement the core pipeline from scratch
- Requires pyannote models (segmentation + embedding)

```python
from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference

pipeline = SpeakerDiarization()
mic = MicrophoneAudioSource()
inference = StreamingInference(pipeline, mic, do_plot=True)
prediction = inference()
```

### Training Data Required?

**Pre-trained models needed** (pyannote segmentation + embedding). But these are freely available on HuggingFace. No fine-tuning required for basic usage.

---

## 7. Pyannote Powerset Segmentation

**Model:** [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**Paper:** "Powerset multi-class cross entropy loss for neural speaker diarization" (INTERSPEECH 2023)

**Code:** [FrenchKrab/IS2023-powerset-diarization](https://github.com/FrenchKrab/IS2023-powerset-diarization)

### How Powerset Encoding Works

Instead of treating speaker diarization as a multi-label problem (each frame can have multiple binary speaker labels), the powerset approach enumerates all possible speaker combinations as distinct classes:

| Class | Meaning |
|-------|---------|
| 0 | Non-speech |
| 1 | Speaker #1 only |
| 2 | Speaker #2 only |
| 3 | Speaker #3 only |
| 4 | Speakers #1 + #2 |
| 5 | Speakers #1 + #3 |
| 6 | Speakers #2 + #3 |

7 classes total for max 3 speakers per chunk, max 2 simultaneous speakers.

### Model Architecture

- **Input:** 10 seconds of mono 16kHz audio (160,000 samples)
- **Output:** (num_frames, 7) matrix of class probabilities
- **Parameters:** ~1M (lightweight)
- **Trained on:** AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, VoxConverse

### How It Prevents Over-Segmentation

- The powerset encoding creates a joint multi-class classification problem rather than independent binary decisions. This means the model jointly reasons about all speaker combinations, reducing inconsistent predictions.
- Standard multi-label approaches can independently activate too many speaker channels. Powerset forces mutual exclusivity among configurations.
- The model is trained with cross-entropy loss over the 7 classes, so it learns to output one configuration per frame (with proper calibration of "non-speech" vs speaker states).

### Conversion to Multi-Label

```python
from pyannote.audio.utils.powerset import Powerset

max_speakers_per_chunk = 3
max_speakers_per_frame = 2
to_multilabel = Powerset(max_speakers_per_chunk, max_speakers_per_frame).to_multilabel

multilabel = to_multilabel(powerset_encoding)
```

### Computational Cost

- ~8ms per 10s chunk on GPU
- **Real-time feasible:** Absolutely. This is one of the fastest components.
- 13.6M+ monthly downloads on HuggingFace (very mature)

### Python Implementation Complexity

Using the model: ~10 lines. The model itself is pre-trained and available via HuggingFace.

### Training Data Required?

The model comes **pre-trained**. Fine-tuning on custom data is optional (see the IS2023-powerset-diarization repo for instructions).

---

## 8. EEND: End-to-End Neural Diarization

**Papers:**
- Fujita et al., "End-to-End Neural Speaker Diarization with Permutation-Free Objectives" (2019)
- Horiguchi et al., "End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors" (2020)

**Code:** [hitachi-speech/EEND](https://github.com/hitachi-speech/EEND), [BUTSpeechFIT/EEND](https://github.com/BUTSpeechFIT/EEND)

### Core Algorithm

1. **Feature extraction:** Compute log-mel filterbanks from audio.

2. **Encoder:** Process features through a deep network (BLSTM, Transformer, or Conformer) to produce frame-level representations.

3. **Attractor computation:** An LSTM-based encoder-decoder generates "attractor" vectors, one per speaker detected. The number of attractors determines the number of speakers.

4. **Speaker activity:** Compute dot product between each frame representation and each attractor to get per-speaker, per-frame activation probabilities.

5. **Permutation-invariant training (PIT):** During training, find the assignment of output channels to reference speakers that minimizes the loss. This handles the label permutation problem.

### How It Prevents Over-Segmentation

- The attractor mechanism implicitly estimates speaker count. If only 2 speakers exist, the LSTM decoder should produce exactly 2 attractors.
- The model is trained end-to-end to minimize DER, so it directly optimizes against both over-segmentation and under-segmentation.
- However, EEND can struggle with recordings very different from training data.

### Computational Cost

- **Transformer EEND:** O(T^2) for self-attention. Not feasible for long recordings without chunking.
- **Streaming variants (FS-EEND, LS-EEND):** O(T) with causal attention. Real-time feasible.
- **Practical limitation:** Requires GPU for reasonable speed.

### Python Implementation Complexity

- ~1000+ lines for a full EEND implementation
- Using pre-trained models: ~50 lines

### Training Data Required?

**Yes, extensive.** EEND requires supervised training with ground-truth diarization labels. This is a significant barrier. Pre-trained models exist but may not generalize to VoxTerm's domain.

---

## 9. UIS-RNN with Chinese Restaurant Process

**Paper:** Zhang et al., "Fully Supervised Speaker Diarization" (ICASSP 2019, [arXiv:1810.04719](https://arxiv.org/abs/1810.04719))

**Code:** [google/uis-rnn](https://github.com/google/uis-rnn)

### Core Algorithm

1. **Extract d-vectors (speaker embeddings)** for each speech segment.

2. **Model each speaker** with a parameter-sharing RNN. RNN states for different speakers interleave in time.

3. **Distance-dependent CRP (ddCRP)** determines whether each new segment belongs to an existing speaker or a new speaker:
   ```
   P(new speaker) = crp_alpha / (crp_alpha + sum of distances to existing speakers)
   ```
   When segments are close in time and similar in embedding space, they are likely the same speaker.

4. **Online decoding:** Process segments sequentially, making speaker assignment decisions in real-time.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `crp_alpha` | Controls probability of creating new speaker. Lower = fewer new speakers = less over-segmentation. Can be learned from training data. |

### How It Prevents Over-Segmentation

- The CRP naturally regularizes the number of speakers. Creating a new speaker cluster requires "enough" evidence (controlled by crp_alpha).
- The distance-dependent variant further discourages new speakers when existing speakers are a good match.
- The RNN models speaker-specific patterns, reducing confusion between speakers.

### Computational Cost

- O(T * K) per segment, where K = current number of speakers
- **Real-time feasible:** Yes, online decoding is efficient.

### Performance

- 7.6% DER on NIST SRE 2000 CALLHOME (better than spectral clustering)

### Training Data Required?

**Yes, fully supervised.** Requires training data with speaker labels. This is a barrier for custom deployments.

---

## 10. Minimum Segment Duration Filtering

### Core Concept

After initial diarization, filter out speaker segments shorter than a minimum duration. These short fragments are usually errors (micro-segments from noisy embeddings, brief misclassifications).

### Algorithm

```python
def filter_short_segments(segments, min_duration=2.5):
    """
    segments: list of (speaker_id, start_time, end_time)

    For each segment shorter than min_duration:
      1. If neighbors have the same speaker -> merge with neighbors
      2. If neighbors differ -> assign to the speaker of the longer neighbor
      3. If isolated -> remove entirely
    """
    filtered = []
    for seg in segments:
        duration = seg[2] - seg[1]
        if duration >= min_duration:
            filtered.append(seg)
        else:
            # Merge into adjacent segment
            assign_to_neighbor(seg, filtered)
    return merge_consecutive(filtered)
```

### Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `min_duration` | 2.5 seconds | Segments shorter than this are reassigned |
| `min_duration_on` (pyannote) | 0.0 - 0.5s | Minimum speech duration for VAD |
| `min_duration_off` (pyannote) | 0.0 - 0.5s | Minimum silence between speech segments |

### How It Prevents Over-Segmentation

- Eliminates micro-segments that create phantom speakers.
- A 12% relative DER improvement was observed on AMI MDM data with alternative minimum duration constraints.
- Short segments (< 1 second) have measurably worse speaker assignment accuracy.

### Computational Cost

- O(N) where N = number of segments. Negligible.
- **Real-time feasible:** Absolutely.

### Implementation Complexity

- ~20-30 lines of Python

### Training Data Required?

**No.**

---

## 11. Cosine Similarity Threshold Calibration

### Core Concept

When using cosine similarity to compare embeddings and decide if two segments belong to the same speaker, the threshold is critical. Too low = over-merge (under-segmentation). Too high = over-segment.

### Calibration Methods

#### Method 1: Equal Error Rate (EER) Threshold
1. Compute cosine similarity for all known same-speaker pairs and different-speaker pairs.
2. Plot FAR (False Acceptance Rate) and FRR (False Rejection Rate) curves.
3. Find the threshold where FAR = FRR. This is the EER threshold.

#### Method 2: Sigmoid Calibration
Convert raw cosine similarity to probability:
```
P(same speaker) = sigmoid(alpha * cos_sim + beta)
```
Tune alpha and beta on validation data. Alpha controls steepness, beta controls threshold.

#### Method 3: Logistic Regression
Train a simple logistic regression model on pairs of embeddings to predict same/different speaker from similarity scores. Most accurate but requires labeled data.

### Typical Threshold Values

| Model / System | Threshold | Notes |
|----------------|-----------|-------|
| General (ECAPA-TDNN) | ~0.25 | Good default across datasets |
| Strict verification | ~0.6 | High precision, lower recall |
| WavLM on VoxCeleb1 | ~0.86 | Model-specific |

**Critical insight:** The optimal threshold is highly model-specific and dataset-specific. Always calibrate on your own data if possible.

### How It Prevents Over-Segmentation

- A properly calibrated threshold prevents the system from splitting one speaker into multiple clusters.
- If the threshold is too high (too strict), embeddings from the same speaker that vary slightly (due to pitch changes, emotion, etc.) get assigned to different clusters.
- For VoxTerm's over-segmentation problem, the threshold is likely too high (too strict about same-speaker matching).

### Computational Cost

- O(1) per comparison. Negligible.

### Training Data Required?

- EER method: Yes, needs labeled pairs
- Default threshold (0.25): No, works out-of-the-box
- Sigmoid calibration: Needs small validation set

---

## 12. Multiscale Diarization (MSDD)

**Blog:** [NVIDIA Developer Blog](https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/)

### Core Concept

Instead of extracting embeddings from a single segment length (e.g., 1.5s), extract from multiple lengths simultaneously (e.g., 0.5s, 1.0s, 1.5s, 3.0s). Then dynamically combine them.

### How It Works

1. **Multi-scale embedding extraction:** For each time step, extract embeddings from segments of varying lengths centered on that point.

2. **Scale weight learning:** A 1D CNN processes the multi-scale embeddings to learn importance weights for each scale at each time step.

3. **Context vectors:** Compute cosine similarity between embeddings and cluster representatives at each scale, weighted by learned scale weights.

4. **LSTM decoder:** Feeds weighted context vectors through multi-layer LSTM to produce per-speaker probabilities.

### How It Prevents Over-Segmentation

- Short segments provide temporal precision but noisy speaker identity.
- Long segments provide robust speaker identity but poor temporal precision.
- By dynamically combining scales, the system uses long-segment evidence to stabilize speaker identity while using short-segment evidence for precise boundaries.
- This reduces the noise from short-segment embeddings that would otherwise cause cluster fragmentation.

### Performance

- ~60% DER reduction compared to single-scale baselines on two-speaker datasets.
- Implemented in [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) speaker diarization pipeline.

### Computational Cost

- Proportional to number of scales (typically 3-5x single-scale)
- **Real-time feasible:** Yes with GPU. The inference is parallelizable across scales.

### Training Data Required?

**Yes** for the MSDD decoder (requires diarization labels for training). Pre-trained models available in NeMo.

---

## 13. Practical Recommendations for VoxTerm

### Priority Order (Impact vs. Implementation Effort)

| Priority | Technique | Impact | Effort | Needs Training? |
|----------|-----------|--------|--------|-----------------|
| **1** | Minimum segment duration filtering | High | Trivial (~20 lines) | No |
| **2** | Cosine similarity threshold calibration | High | Low (~10 lines) | No (use 0.25 default) |
| **3** | Spectral clustering with NME eigengap | Very High | Low (~50 lines with library) | No |
| **4** | Quality-gated embedding updates | High | Low (~30 lines) | No |
| **5** | VBx with Fb tuning | Very High | Medium (~100 lines or use library) | Pre-trained eigenvoice |
| **6** | Diart streaming pipeline | High | Medium (use diart library) | Pre-trained pyannote |
| **7** | Periodic cluster merging (AHC post-processing) | Medium | Medium (~50-100 lines) | No |
| **8** | EEND | High | High (needs training) | Yes |

### Specific Recommendations for VoxTerm's 2-to-5+ Problem

1. **Immediate fix (Priority 1+2):** Add minimum segment duration of 2.5s and lower the cosine similarity threshold. These two changes alone could reduce 5 detected speakers back to 2-3.

2. **Replace clustering (Priority 3):** Switch from whatever current clustering to NME-SC spectral clustering. The auto-tuning eigengap will correctly identify 2 speakers in most cases. Use the `spectralcluster` library.

3. **Add quality gating (Priority 4):** Before clustering, compute silhouette scores for each embedding. Discard the bottom 10% -- these are the embeddings causing phantom clusters.

4. **Add speaker continuity (Priority 5):** The VBx HMM transition model (loopP=0.99) prevents rapid speaker switching. Even without the full VBx pipeline, adding a transition penalty helps.

5. **For streaming (Priority 6):** If moving to real-time streaming, adopt diart's architecture with carefully tuned delta_new (start at 1.0, increase to reduce over-segmentation).

### Key Formulas to Implement

```python
# 1. NME for speaker count estimation
g_p = max(eigengaps) / (max_eigenvalue + 1e-10)
r_p = p / g_p
optimal_p = argmin(r)
num_speakers = argmax(eigengaps[optimal_p])

# 2. Silhouette-based quality gating
s_i = (b_i - a_i) / max(a_i, b_i)
keep = s_i > -0.1  # discard clearly misassigned embeddings

# 3. VBx-style transition penalty
# HMM transition: P(stay) = 0.99, P(switch) = 0.01 / (K-1)
transition_matrix = np.full((K, K), 0.01 / (K - 1))
np.fill_diagonal(transition_matrix, 0.99)

# 4. Minimum segment filter
segments = [s for s in segments if s.duration >= 2.5]
```

---

## Sources

### Spectral Clustering / NME-SC
- [arXiv:2003.02405 - Auto-Tuning Spectral Clustering](https://arxiv.org/abs/2003.02405)
- [GitHub: tango4j/Auto-Tuning-Spectral-Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)
- [GitHub: wq2012/SpectralCluster](https://github.com/wq2012/SpectralCluster)
- [GitHub: tango4j/Python-Speaker-Diarization](https://github.com/tango4j/Python-Speaker-Diarization)
- [scikit-learn SpectralClustering docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
- [SpeechBrain diarization module](https://speechbrain.readthedocs.io/en/v1.0.2/API/speechbrain.processing.diarization.html)
- [Self-Tuning Spectral Clustering for Speaker Diarization](https://arxiv.org/html/2410.00023)

### VBx / Bayesian HMM
- [arXiv:2012.14952 - VBx paper](https://arxiv.org/abs/2012.14952)
- [GitHub: BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx)
- [GBO Notes: Variational Bayes and VBx](https://desh2608.github.io/2022-04-10-gbo-vb/)
- [GitHub: BUTSpeechFIT/DVBx](https://github.com/BUTSpeechFIT/DVBx)
- [VBx Software Page](https://speech.fit.vut.cz/software/vbhmm-x-vectors-diarization)

### Sticky HDP-HMM
- [Sticky HDP-HMM Paper (MIT)](https://dspace.mit.edu/bitstream/handle/1721.1/79665/sticky_hdphmm_final_submission.pdf)
- [Transition Cost Analysis in Speaker Diarization](https://link.springer.com/article/10.1186/s13636-021-00196-6)

### Quality / Confidence Estimation
- [arXiv:2406.17124 - Confidence Estimation for Diarization](https://arxiv.org/html/2406.17124v1)
- [Cosine Scoring with Uncertainty](https://arxiv.org/html/2403.06404v1)
- [PLDA vs Cosine Scoring](https://www.isca-archive.org/interspeech_2022/wang22r_interspeech.pdf)

### Diart / Online Diarization
- [GitHub: juanmc2005/diart](https://github.com/juanmc2005/diart)
- [JOSS Paper: Diart](https://joss.theoj.org/papers/10.21105/joss.05266)
- [Diart Documentation](https://diart.readthedocs.io/en/latest/autoapi/diart/index.html)

### Pyannote Powerset
- [pyannote/segmentation-3.0 on HuggingFace](https://huggingface.co/pyannote/segmentation-3.0)
- [IS2023 Powerset Diarization](https://github.com/FrenchKrab/IS2023-powerset-diarization)
- [pyannote-audio GitHub](https://github.com/pyannote/pyannote-audio)

### EEND
- [GitHub: hitachi-speech/EEND](https://github.com/hitachi-speech/EEND)
- [GitHub: BUTSpeechFIT/EEND](https://github.com/BUTSpeechFIT/EEND)
- [GitHub: Audio-WestlakeU/FS-EEND](https://github.com/Audio-WestlakeU/FS-EEND)
- [Emergent Mind: EEND topic](https://www.emergentmind.com/topics/end-to-end-neural-diarization-eend)

### UIS-RNN / Chinese Restaurant Process
- [arXiv:1810.04719 - Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719)
- [Google Research Blog: Online Speaker Diarization](https://research.google/blog/accurate-online-speaker-diarization-with-supervised-learning/)
- [GitHub: DonkeyShot21/uis-rnn-sml](https://github.com/DonkeyShot21/uis-rnn-sml)

### Multiscale Diarization
- [NVIDIA Blog: Multiscale Speaker Diarization](https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/)
- [NeMo Speaker Diarization Issue](https://github.com/NVIDIA-NeMo/NeMo/issues/7558)

### General Speaker Diarization
- [Aalto University: Introduction to Speech Processing - Speaker Diarization](https://speechprocessingbook.aalto.fi/Recognition/Speaker_Diarization.html)
- [Park et al.: A Review of Speaker Diarization](https://sail.usc.edu/publications/files/Park-Diarization-CSL2022.pdf)
- [Speaker Diarization Wikipedia](https://en.wikipedia.org/wiki/Speaker_diarisation)
- [AssemblyAI: What is Speaker Diarization (2026 Guide)](https://www.assemblyai.com/blog/what-is-speaker-diarization-and-how-does-it-work)
- [Speaker Verification Using Cosine Similarity](https://www.iosrjournals.org/iosr-jce/papers/Vol26-issue1/Ser-1/C2601011926.pdf)
- [NeMo Cosine Similarity Discussion](https://github.com/NVIDIA-NeMo/NeMo/discussions/10430)
- [WavLM Speaker Verification Model](https://huggingface.co/microsoft/wavlm-base-sv)
