# Online/Streaming Speaker Diarization: Research Survey

**Date:** 2026-03-21
**Purpose:** Evaluate online speaker diarization systems to replace VoxTerm's current approach (ECAPA-TDNN + cosine similarity + EMA centroid update) which over-segments 2 real speakers into 5+ detected speakers.
**Constraints:** Online/streaming, <200ms latency per segment, local/offline, Python/PyTorch on CPU.

---

## Table of Contents

1. [Problem Analysis: Why Over-Segmentation Occurs](#1-problem-analysis)
2. [Diart (Overlap-Aware Online Diarization)](#2-diart)
3. [NVIDIA Streaming Sortformer](#3-nvidia-streaming-sortformer)
4. [Google UIS-RNN](#4-google-uis-rnn)
5. [Turn-to-Diarize](#5-turn-to-diarize)
6. [Google SpectralCluster (Multi-Stage Streaming)](#6-google-spectralcluster)
7. [SCDiar (Speaker Change Detection + Diarization)](#7-scdiar)
8. [VBx (Variational Bayes HMM Clustering)](#8-vbx)
9. [NVIDIA NeMo MSDD (Multi-Scale Diarization Decoder)](#9-nvidia-nemo-msdd)
10. [Lightweight Block-Online K-Means Approach](#10-lightweight-block-online-k-means)
11. [Whisper-Based Real-Time System](#11-whisper-based-real-time-system)
12. [Online Speaker Change Detector (Alumae)](#12-online-speaker-change-detector)
13. [Comparative Summary](#13-comparative-summary)
14. [Recommendations for VoxTerm](#14-recommendations-for-voxterm)

---

## 1. Problem Analysis

### Why VoxTerm's Current Approach Over-Segments

VoxTerm extracts ECAPA-TDNN embeddings per transcription segment (1-15s variable length), compares via cosine similarity to session centroids (threshold 0.30), and updates centroids via EMA (alpha 0.95). The primary failure mode is that 2 real speakers routinely become 5+ detected speakers.

**Root causes identified through research:**

1. **Embedding variability on short/variable-length segments.** ECAPA-TDNN embeddings degrade significantly on segments shorter than ~2 seconds. Channel attention and squeeze-excitation blocks help, but short segments produce noisy embeddings that scatter in the embedding space, easily exceeding the distance threshold.

2. **Fixed cosine similarity threshold is fragile.** A single threshold (0.30) cannot account for within-speaker variability caused by emotional shifts, volume changes, crosstalk, or environmental noise. Research consistently shows the threshold is the most critical and most fragile parameter.

3. **EMA centroid update is too aggressive.** With alpha=0.95, each new embedding contributes 5% to the centroid. A single noisy embedding can drift the centroid enough that the next legitimate embedding from the same speaker exceeds the threshold, triggering a new speaker.

4. **No overlap awareness.** When two speakers overlap, the embedding is a blend of both voices. Without overlap detection, this blended embedding matches neither centroid and spawns a new speaker.

5. **No minimum duration or quality filtering.** Very short speech bursts produce unreliable embeddings that should be filtered before clustering decisions.

6. **No cannot-link constraints.** Without awareness of which segments co-occur (overlap), the system can assign two simultaneously-active segments to the same speaker, or worse, create phantom speakers from blended audio.

### Key Insight from Literature

The research consensus is that **raw embedding + threshold clustering is the weakest approach**. Every competitive system adds at least one of: segmentation-guided weighting, overlap-aware constraints, multi-scale embeddings, minimum duration filtering, or probabilistic speaker models. The diart system specifically addresses all of these.

---

## 2. Diart (Overlap-Aware Online Diarization)

**The strongest candidate for VoxTerm's use case.**

### Source
- GitHub: https://github.com/juanmc2005/diart
- Paper: "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation" (Coria et al., 2021, ASRU)
- JOSS Paper: https://joss.theoj.org/papers/10.21105/joss.05266
- PyPI: `pip install diart`

### Architecture

Diart combines three components into a streaming pipeline:

1. **Segmentation Model** (pyannote/segmentation or segmentation-3.0): Processes a rolling audio buffer and outputs per-frame speaker activity probabilities for up to K_max=4 local speakers. This is the key differentiator -- it detects *who is speaking when* at frame level, including overlaps.

2. **Embedding Model** (pyannote/embedding or alternatives): Extracts speaker embeddings, but with overlap-aware weighting. The statistics pooling is modified so that frames where multiple speakers are active receive lower weight:
   ```
   w_f = (s_f * softmax_k(beta * s_f))^gamma
   ```
   where beta=10, gamma=3. This dramatically reduces embedding contamination from overlapping speech.

3. **Constrained Incremental Clustering**: Maps local speakers to global speakers using the Hungarian algorithm, with cannot-link constraints preventing two co-occurring local speakers from being assigned to the same global speaker.

### Rolling Buffer Mechanism

- Default: 5-second rolling buffer, updated every 500ms (the "step")
- Latency is adjustable from 500ms to 5s
- Each step: the segmentation model runs on the full buffer, but only the new 500ms contributes to clustering decisions
- Previous buffer content provides acoustic context for better segmentation

### Critical Hyperparameters

| Parameter | What it controls | Default | Effect on over-segmentation |
|-----------|-----------------|---------|----------------------------|
| `tau_active` | Minimum speaker activity probability to be considered active | 0.555 | Higher = fewer false speakers; too high = missed speech |
| `rho_update` | Minimum duration of speech before centroid update is applied | 0.422 | Higher = only longer segments update centroids; prevents noisy short segments from corrupting |
| `delta_new` | Distance threshold for creating a new speaker (0-2 range) | 1.517 | Higher = harder to create new speakers = less over-segmentation; too high = merges distinct speakers |

**Tuned values per dataset:**

| Dataset | tau | rho | delta |
|---------|-----|-----|-------|
| DIHARD III | 0.555 | 0.422 | 1.517 |
| AMI | 0.507 | 0.006 | 1.057 |
| VoxConverse | 0.576 | 0.915 | 0.648 |

### How It Prevents Over-Segmentation

1. **Overlap-aware embeddings**: Downweights frames with multiple speakers, producing cleaner per-speaker embeddings
2. **Cannot-link constraints**: Segmentation model identifies co-occurring speakers, preventing them from being merged into the same cluster
3. **rho_update filtering**: Short speech segments don't update centroids, preventing centroid drift
4. **delta_new threshold**: Requires substantial distance before creating a new speaker
5. **Ensemble-like aggregation**: Multiple buffer positions see the same speech, providing redundancy that smooths out single-segment noise

### Latency Characteristics

Model latencies measured on AMD Ryzen 9 CPU (no GPU):
- pyannote/segmentation: **12ms**
- pyannote/segmentation-3.0: **11ms**
- pyannote/embedding: **26ms**
- SpeechBrain ECAPA-TDNN: **41-42ms**
- Total pipeline per step: **~40-60ms on CPU** (well within 200ms budget)

With 500ms step size, the algorithmic latency is 500ms. Can be reduced to the step size.

### Truly Online?

**Yes.** Processes audio incrementally with a rolling buffer. No future lookahead beyond the buffer. The buffer provides past context, not future information.

### Open Source

Fully open source (MIT license). Requires accepting pyannote model licenses on HuggingFace.

### Hyperparameter Optimization

Includes built-in Optuna-based optimizer:
```python
from diart.optim import Optimizer
optimizer = Optimizer("/wav/dir", "/rttm/dir", "/output/dir")
optimizer(num_iter=100)
```

### Relevance to VoxTerm

**HIGH.** Diart directly solves VoxTerm's over-segmentation problem through overlap-aware embeddings and constrained incremental clustering. It runs on CPU within latency budget. The main integration question is how to feed VoxTerm's audio segments into diart's rolling buffer architecture (diart expects a continuous audio stream, not pre-segmented chunks).

---

## 3. NVIDIA Streaming Sortformer

### Source
- Blog: https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- Paper: "Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering" (Medennikov et al., Interspeech 2025)
- arXiv: https://arxiv.org/abs/2507.18446
- Framework: NVIDIA NeMo

### Architecture

End-to-end neural diarization model (not a pipeline):
1. **Convolutional pre-encoder**: Compresses raw audio features
2. **Conformer blocks**: Process compressed features with local+global attention
3. **Transformer blocks**: Analyze conversational context
4. **Arrival-Order Speaker Cache (AOSC)**: Memory buffer storing frame-level acoustic embeddings of previously observed speakers

The model directly outputs frame-level speaker labels without separate embedding extraction or clustering steps.

### AOSC Speaker Cache

The key innovation: AOSC tracks all speakers previously detected in the audio stream. When processing a new audio chunk:
1. Current chunk speakers are compared against cached speaker embeddings
2. Consistent speaker IDs are maintained across chunks
3. New speakers are added to the cache when detected
4. Speakers are ordered by their arrival time (first appearance)

### Latency

- Minimum tested latency: **0.32 seconds** with "highly competitive performance"
- Performance degrades gracefully as latency decreases
- Chunk-wise processing with FIFO queue and input buffer

### How It Handles Over-Segmentation

The end-to-end model learns to identify speakers holistically rather than relying on threshold-based decisions. The AOSC cache provides persistent speaker memory, preventing the system from re-creating known speakers.

### Open Source

Available through NVIDIA NeMo framework. **However, requires GPU for practical inference.** The conformer+transformer architecture is computationally expensive and not designed for CPU-only deployment.

### Relevance to VoxTerm

**LOW-MEDIUM.** State-of-the-art accuracy but requires GPU. The end-to-end architecture is not easily adaptable to CPU-only constraints. Would require significant model compression or distillation to be practical.

---

## 4. Google UIS-RNN

### Source
- GitHub: https://github.com/google/uis-rnn
- Paper: "Fully Supervised Speaker Diarization" (Zhang et al., ICASSP 2019)
- arXiv: https://arxiv.org/abs/1810.04719

### Architecture

Unbounded Interleaved-State Recurrent Neural Network:
1. Each speaker is modeled by a parameter-sharing RNN (GRU)
2. RNN states for different speakers interleave in time
3. Integrated with a distance-dependent Chinese Restaurant Process (ddCRP)

### How It Works

Three probability components are combined:
1. **p(z_t | z_{t-1})**: Speaker change probability at each time step
2. **Speaker assignment via ddCRP**: New speaker probability follows a Chinese Restaurant Process -- the probability of a new speaker is constant (analogous to "new table" probability), while existing speaker probability is proportional to how many segments they already have
3. **Sequence generation**: GRU generates embedding sequences; each speaker's GRU hidden state is maintained and updated

### Online Decoding

**Yes, truly online.** Decodes sequentially: at each time step, decides whether the current embedding belongs to an existing speaker or a new one, then updates the corresponding GRU hidden state. No future information needed.

### How It Prevents Over-Segmentation

The ddCRP prior naturally penalizes creating too many speakers. The probability of creating a new speaker is a fixed constant (tunable), while existing speakers accumulate probability mass proportional to their segment count. This "rich get richer" property inherently resists over-segmentation.

### Latency

Immediate per-segment decisions. The bottleneck is the embedding extraction, not the clustering.

### Open Source

Yes (Apache 2.0 license). PyTorch implementation. **Note:** The open-source version is "slightly different than the internal one" used for published results.

### Limitations

- Requires supervised training data (speaker-labeled embeddings)
- Performance depends heavily on embedding quality
- The open-source implementation lacks production polish
- Last updated 2020; not actively maintained

### Relevance to VoxTerm

**MEDIUM.** The ddCRP-based speaker number estimation is elegant and directly addresses over-segmentation. Could be used as a drop-in replacement for the current cosine-threshold clustering. However, requires training data and the implementation is dated. The core idea (probabilistic new-speaker gating via CRP) could be adapted without the full UIS-RNN.

---

## 5. Turn-to-Diarize

### Source
- Paper: "Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection" (Xia et al., 2021)
- arXiv: https://arxiv.org/abs/2109.11641

### Architecture

Two-stage pipeline:
1. **Transformer Transducer**: Detects speaker turns (change points) during ASR. Speaker turn tokens are inserted into the transcript during recognition.
2. **Constrained Spectral Clustering**: Speaker embeddings are extracted per turn segment, then clustered with constraints derived from detected turns.

### Key Innovation

Instead of clustering all frames or fixed-length segments, the system:
- Uses ASR to find natural speaker boundaries
- Extracts one embedding per speaker turn (much sparser than frame-level)
- Applies must-link/cannot-link constraints from turn detection to guide clustering

### Constraint Matrix

- **Must-link**: Consecutive segments without a detected turn must be the same speaker
- **Cannot-link**: Segments separated by a detected turn should be different speakers (with confidence weighting)

### Online Capability

**Semi-online.** Speaker turn detection happens in real-time during ASR streaming. However, the spectral clustering step is not inherently online -- it can be applied incrementally but works best with a larger window of turns.

### Latency

Per-turn processing. Latency is primarily determined by turn length. Computational cost of clustering is greatly reduced due to turn sparsity (far fewer embeddings to cluster than frame-level approaches).

### How It Prevents Over-Segmentation

Must-link constraints from within-turn continuity prevent unnecessary splits. The system only considers speaker changes at detected turn boundaries.

### Open Source

The spectral clustering component is available via Google's SpectralCluster library. The transformer transducer speaker turn detection model is not publicly available.

### Relevance to VoxTerm

**MEDIUM.** VoxTerm already has ASR segments from Qwen3-ASR. The constraint matrix concept could be adapted: consecutive segments without detected speaker changes could be must-linked. However, the full Turn-to-Diarize system requires a specialized ASR model with speaker turn tokens.

---

## 6. Google SpectralCluster (Multi-Stage Streaming)

### Source
- GitHub: https://github.com/wq2012/SpectralCluster
- Papers: ICASSP 2018 (refined Laplacian), ICASSP 2022 (multi-stage streaming)

### Architecture

Python implementation of spectral clustering with streaming support:

1. **Affinity Matrix Refinement**: Before clustering, the cosine similarity matrix undergoes:
   - Gaussian blur (smoothing)
   - Row-wise thresholding (sparsification)
   - Diagonal cropping (self-similarity normalization)

2. **Multi-Stage Clustering**: `MultiStageClusterer` enables streaming via `streaming_predict()`:
   - Feed one embedding at a time
   - Returns cluster labels for ALL inputs (including corrections to previous predictions)
   - Retroactive correction is key: early mistakes are fixed as more evidence arrives

3. **Auto-Tuning**: The `AutoTune` class optimizes the `p_percentile` parameter for row-wise thresholding, reducing sensitivity to manual threshold selection.

### How It Prevents Over-Segmentation

- **Affinity refinement** cleans up the similarity matrix before clustering
- **Auto-tuning** adapts thresholds to the data distribution
- **Retroactive corrections** fix early over-segmentation as more data arrives
- **Constraint support**: Can incorporate must-link/cannot-link matrices from external sources

### Latency

The `streaming_predict` method processes one embedding at a time with immediate return. However, spectral clustering internally recomputes the full eigendecomposition, which scales O(n^3) with the number of embeddings seen so far. This becomes prohibitive for long sessions.

### Truly Online?

**Partially.** The streaming interface is online, but the underlying spectral clustering grows in cost. The multi-stage approach mitigates this by using hierarchical pre-clustering (`max_spectral_size` parameter).

### Open Source

Yes (Apache 2.0). Pure Python. **Note:** "This is not the original C++ implementation used by the papers...consider this repo as a 'demonstration' of the algorithms."

### Relevance to VoxTerm

**MEDIUM-HIGH.** The affinity matrix refinement and auto-tuning could directly improve VoxTerm's clustering without changing the overall architecture. The streaming_predict API with retroactive corrections is exactly the right interface. Main concern: computational scaling for long sessions.

---

## 7. SCDiar (Speaker Change Detection + Diarization)

### Source
- Paper: "SCDiar: a streaming diarization system based on speaker change detection and speech recognition" (2025)
- arXiv: https://arxiv.org/html/2501.16641

### Architecture

Three integrated components:
1. **CIF-based ASR**: Continuous Integrate-and-Fire for acoustic-to-token alignment
2. **Speaker Change Detection (SCD)**: BiLSTM + 1D CNN on token-level features, detecting speaker boundaries via peak detection
3. **Speaker Diarization (SD)**: Segment-token similarity matrix with cross-attention decoder

### Key Innovation: Segment-Token Similarity Matrix

Instead of simple cosine similarity between segment embeddings and centroids:
- Preserves individual token contributions within segments
- Addresses the problem that "short utterances often provide insufficient speaker information"
- Combines local token-level and global segment-level information through multi-scale concatenation
- Uses cross-attention decoder layers for refined speaker embeddings

### Representative Segment Selection

Instead of using all segments per speaker, selects the single best representative segment through bounded least-squares optimization. This converts many-to-one mapping to one-to-one, reducing noise in the speaker cache.

### Over-Segmentation Prevention

1. Minimum token threshold (10 tokens) for segment consideration
2. Cosine distance filtering to avoid near-duplicate speakers in cache
3. Optimization scheme balancing segment quality with distinctiveness

### Streaming Capability

**True streaming.** Maintains an incrementally-updated speaker embedding cache. Processes within VAD chunk boundaries (max 15s default).

### Latency

Real-time factors: ASR=0.072, SCD=0.004, SD=0.009 (total ~0.085 on GPU). Well below real-time.

### Performance

- AISHELL-4 (4-8 speakers): 3.42% cpWER online vs 2.13% offline
- Performance stabilizes after 10 seconds of streaming input

### Open Source

Paper is public; code availability unclear.

### Relevance to VoxTerm

**LOW-MEDIUM.** The segment-token similarity matrix and representative segment selection are interesting ideas. However, the system is tightly coupled to CIF-based ASR (not Qwen3-ASR). The token-level processing approach is architecturally incompatible with VoxTerm's segment-level pipeline.

---

## 8. VBx (Variational Bayes HMM Clustering)

### Source
- GitHub: https://github.com/BUTSpeechFIT/VBx
- Paper: "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization" (Landini et al., Computer Speech & Language, 2021)

### Architecture

Models speaker diarization as a Hidden Markov Model:
- **Hidden states**: Speaker identities
- **Emissions**: Speaker embeddings (x-vectors)
- **Transitions**: Speaker change probabilities
- **Inference**: Variational Bayes (VB) -- iterative approximate inference

### How It Works

1. Initialize with an over-estimated number of speaker states
2. VB inference iteratively:
   - Re-estimates speaker means (centroids) given current assignments
   - Re-estimates assignments given current speaker means
   - Prunes empty/near-empty speakers automatically
3. The HMM structure encodes temporal continuity: speakers tend to continue speaking (high self-transition probability)

### How It Prevents Over-Segmentation

1. **HMM transition probabilities**: High self-transition probability naturally enforces minimum segment duration -- rapid speaker switching is penalized
2. **VB pruning**: Speakers with too few assigned segments are automatically eliminated during inference
3. **Principled speaker count estimation**: The number of speakers emerges from the data rather than being thresholded
4. **Fb parameter**: Controls the probability of creating new speakers; tunable to trade off over- vs under-segmentation

### Online Capability

**Primarily offline.** Standard VBx requires the full sequence of embeddings. However, the iterative VB algorithm could theoretically be applied in a sliding-window fashion with warm-starting from previous iterations.

### Latency

Offline: processes the full recording. Not designed for real-time. The VB iterations add significant computational overhead.

### Open Source

Yes. Python implementation available.

### Relevance to VoxTerm

**MEDIUM (for ideas, not direct integration).** VBx is offline, but its key ideas are directly applicable:
- **HMM transition model**: Adding a speaker persistence prior (speakers tend to continue) would reduce VoxTerm's over-segmentation
- **Automatic speaker count via VB pruning**: More principled than threshold-based new-speaker detection
- **The Fb parameter**: Provides a cleaner control knob than raw cosine distance threshold

---

## 9. NVIDIA NeMo MSDD (Multi-Scale Diarization Decoder)

### Source
- Blog: https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/
- Paper: "Multi-scale Speaker Diarization with Dynamic Scale Weighting" (Park et al., 2022)
- Framework: NVIDIA NeMo

### Architecture

Addresses the fundamental single-scale trade-off:
- **Long segments** (1.5-3.0s): Good speaker characteristics, poor temporal resolution
- **Short segments** (0.25-0.5s): Good temporal resolution, poor speaker characteristics

MSDD solution:
1. Extract TitaNet embeddings at multiple scales: windows of [1.5, 1.25, 1.0, 0.75, 0.5] seconds
2. 1-D CNN calculates adaptive importance weights for each scale at each time step
3. Cosine similarities between input and cluster-average embeddings, weighted by scale factors
4. Multi-layer LSTM processes context vectors to output per-speaker existence probabilities

### How It Prevents Over-Segmentation

- Short segments alone cause embedding degradation and over-segmentation
- MSDD dynamically downweights short-scale embeddings when they're unreliable
- Long-scale embeddings anchor the speaker identity while short-scale provides temporal precision
- Result: 0.25s temporal resolution (3x improvement) without over-segmentation

### Performance

60% reduction in DER on two-speaker datasets compared to single-scale baselines.

### Online Capability

**Currently offline only.** Streaming identified as future research direction. Requires full-sequence LSTM processing.

### Relevance to VoxTerm

**MEDIUM (for the multi-scale idea).** VoxTerm could extract embeddings at multiple segment lengths and weight them dynamically. This is conceptually simple to implement: for each audio segment, extract embeddings from overlapping windows of different lengths, then weight the similarity scores. However, the full MSDD architecture requires GPU.

---

## 10. Lightweight Block-Online K-Means Approach

### Source
- Paper: "A lightweight approach to real-time speaker diarization: from audio toward audio-visual data streams" (EURASIP Journal on Audio, Speech, and Music Processing, 2024)

### Architecture

1. **SE-ResNet-34**: Modified residual network with squeeze-and-excitation blocks for speaker embedding extraction
2. **VAD**: Neural voice activity detection to filter non-speech
3. **Block-online k-means**: Clustering applied to incoming embedding blocks with look-ahead

### How It Works

- Audio is processed in blocks
- Speaker embeddings are extracted per block using SE-ResNet-34 with cached buffers (avoiding redundant computation on overlapping regions)
- Embeddings are smoothed with a moving average (context of t +/- 40 frames) to reduce sensitivity
- Block-online k-means assigns speakers, with a look-ahead mechanism that delays decisions slightly for improved accuracy

### Latency

- CPU real-time factor: **< 0.1** (processes audio 10x faster than real-time on CPU)
- Total latency: **~5.5 seconds** (includes look-ahead buffer)

### Over-Segmentation Prevention

- Moving average smoothing of embeddings reduces frame-to-frame variability
- Block-level processing amortizes noise across multiple frames
- Look-ahead provides additional context for better clustering decisions

### Open Source

Paper is public; specific code availability unclear.

### Relevance to VoxTerm

**MEDIUM.** The embedding smoothing via moving average is a simple technique VoxTerm could adopt immediately. The 5.5s latency exceeds VoxTerm's budget, but the smoothing concept applies at any latency. The SE-ResNet-34 architecture is not directly usable (VoxTerm uses ECAPA-TDNN), but the caching strategy for overlapping buffers is applicable.

---

## 11. Whisper-Based Real-Time System

### Source
- Paper: "Real-time multilingual speech recognition and speaker diarization system based on Whisper segmentation" (PMC, 2024)

### Architecture

1. Audio blocks extracted every 0.5s from continuous stream
2. Whisper processes buffered audio, generating speech segments (natural speaker-homogeneous segments)
3. ECAPA-TDNN extracts 192-dimensional speaker embeddings from 80-dim MFCCs
4. Incremental clustering with cosine distance

### Centroid Update Mechanism

Very similar to VoxTerm's approach, but with important differences:

```
d(e, c_i) = e . c_i / (||e|| ||c_i||)
```

- If distance to nearest centroid < delta_new (threshold = 0.7): update centroid as c_s <- c_s + e (additive, not EMA)
- If no similar centroid exists and speaker slots remain: create new centroid
- When maximum speakers reached: assign to closest without updating

**Note the threshold of 0.7** (cosine similarity), which is much higher than VoxTerm's 0.30 (cosine distance). This is the same metric inverted: cosine_distance = 1 - cosine_similarity, so 0.7 similarity = 0.3 distance. VoxTerm's threshold appears correctly calibrated in absolute terms.

### Key Difference from VoxTerm

Whisper naturally segments overlapping speech into distinct pieces per speaker. This preprocessing step eliminates the overlap contamination problem that plagues VoxTerm's segment-level approach.

### Performance

- Processing ratio: 3% of speech duration (extremely fast)
- Two speakers: 2.68% WDER
- System stabilizes after ~600 words

### Relevance to VoxTerm

**HIGH (for diagnostics).** This system uses nearly identical components (ECAPA-TDNN, cosine distance, 0.7 similarity threshold) but achieves much better results. The key difference is Whisper's speaker-homogeneous segmentation vs. VoxTerm's Qwen3-ASR segments which may span speaker changes. This suggests VoxTerm's over-segmentation may be partly caused by segments containing mixed-speaker audio.

---

## 12. Online Speaker Change Detector (Alumae)

### Source
- GitHub: https://github.com/alumae/online_speaker_change_detector

### Architecture

Three-stage neural network:
1. Several 1-D convolutional layers (speech encoder)
2. Multi-layer LSTM (models speaker change patterns)
3. Softmax output layer

### How It Works

- Produces **10 decisions per second** (100ms step)
- Special cross-entropy training that tolerates small timestamp errors
- Softmax outputs are "very peaky" -- directly usable for turn detection without post-processing
- Binary: change / no-change at each 100ms step

### Limitations

- Trained on Estonian broadcast data (domain-specific)
- Detects *that* a change occurred, not *who* the speakers are
- Must be combined with a separate speaker embedding + clustering system

### Relevance to VoxTerm

**LOW-MEDIUM.** Could serve as a preprocessing step: only extract new embeddings and make clustering decisions at detected change points. Between change points, simply extend the current speaker label. This would dramatically reduce over-segmentation from within-turn embedding variability. However, the Estonian-trained model would need retraining.

---

## 13. Comparative Summary

| System | Online? | CPU Feasible? | Latency | Handles Overlap? | Over-seg Prevention | Open Source? | VoxTerm Fit |
|--------|---------|---------------|---------|-------------------|---------------------|-------------|-------------|
| **Diart** | Yes | Yes (40-60ms) | 500ms-5s | Yes (core feature) | Overlap weighting, rho filter, delta threshold, constraints | Yes (MIT) | **BEST** |
| Streaming Sortformer | Yes | No (GPU required) | 320ms+ | Yes | End-to-end learned | Yes (NeMo) | Low |
| UIS-RNN | Yes | Yes | Per-segment | No | ddCRP prior | Yes (Apache) | Medium |
| Turn-to-Diarize | Semi | Partial | Per-turn | No | Must-link constraints | Partial | Medium |
| SpectralCluster | Streaming | Yes | Per-embedding | No | Affinity refinement, auto-tune | Yes (Apache) | Medium-High |
| SCDiar | Yes | No (GPU) | ~85ms (GPU) | Partial | Min tokens, cache filtering | Unclear | Low |
| VBx | No (offline) | Yes | N/A | No | HMM transitions, VB pruning | Yes | Medium (ideas) |
| NeMo MSDD | No (offline) | No (GPU) | N/A | Yes | Multi-scale weighting | Yes (NeMo) | Medium (ideas) |
| Block-Online K-Means | Yes | Yes (RTF<0.1) | 5.5s | No | Moving average smoothing | Unclear | Medium |
| Whisper-Based | Yes | Partial | ~5s | Whisper handles | Whisper segmentation | Paper only | High (diagnostics) |
| Alumae SCD | Yes | Yes | 100ms | No | Change-point gating | Yes | Low-Medium |

---

## 14. Recommendations for VoxTerm

### Primary Recommendation: Integrate Diart

Diart is the best fit for VoxTerm's constraints:

1. **Directly solves over-segmentation** through overlap-aware embedding extraction and constrained incremental clustering
2. **Runs on CPU** within the 200ms latency budget (measured ~40-60ms per step)
3. **Fully open source** with active maintenance
4. **Built-in hyperparameter optimization** via Optuna
5. **Streaming API** with microphone source, RTTM output, and WebSocket support

**Integration approach:**
- Feed VoxTerm's audio stream directly into diart's `MicrophoneAudioSource` or custom `AudioSource`
- Use diart's speaker labels instead of VoxTerm's current ECAPA-TDNN + cosine clustering
- Tune `tau_active`, `rho_update`, and `delta_new` for VoxTerm's typical audio conditions
- The segmentation model replaces VoxTerm's need for separate VAD

**Key concern:** Diart expects a continuous audio stream via a rolling buffer, while VoxTerm currently processes discrete segments from Qwen3-ASR. Integration requires either:
- (a) Running diart in parallel on the raw audio stream (recommended), or
- (b) Adapting diart's buffer to accept VoxTerm's segments

### Quick Wins (Can Apply Immediately Without Diart)

If full diart integration is too invasive, these techniques from the research can be applied to the existing pipeline:

1. **Raise delta_new (new speaker threshold) significantly.** VoxTerm's 0.30 cosine distance threshold is aggressive. Research systems use 0.648-1.517 for delta_new. Start by raising to 0.50-0.70.

2. **Add minimum duration filter (rho_update).** Do not update centroids from segments shorter than ~0.5 seconds. Short segments produce unreliable embeddings.

3. **Smooth embeddings with moving average.** Before comparing to centroids, average the current embedding with recent embeddings from the same speaker (window of 3-5 segments). This reduces single-segment noise.

4. **Add speaker persistence prior.** After assigning a speaker, bias the next segment toward the same speaker (e.g., reduce the effective distance by 10-20% for the previously-active speaker). This encodes the observation that speakers tend to continue.

5. **Reduce EMA alpha from 0.95 to 0.80-0.85.** Each new embedding contributes 15-20% instead of 5%, making centroids more stable (less sensitive to a single noisy embedding drifting the centroid).

   **Wait -- this is backwards.** Lower alpha means *more* weight on new embeddings. For stability, *raise* alpha toward 0.98-0.99, or switch to a simple running mean of all embeddings for a speaker (which is equivalent to alpha approaching 1.0 as more embeddings arrive).

6. **Cap maximum speakers.** If VoxTerm typically has 2-4 speakers, set a maximum. When reached, assign new embeddings to the closest existing centroid without creating new speakers.

### Medium-Term: Adopt SpectralCluster's Streaming Predict

Google's SpectralCluster `streaming_predict()` API provides:
- Retroactive correction of previous speaker labels as new evidence arrives
- Affinity matrix refinement (Gaussian blur + row-wise thresholding)
- Auto-tuning of clustering thresholds
- Constraint matrix support (can integrate turn detection constraints)

This could replace VoxTerm's current cosine-threshold + EMA approach with minimal architectural change while providing significantly better clustering quality.

### Long-Term: Speaker Persistence via HMM Prior (from VBx)

Add a simple HMM-inspired speaker transition model:
- Track the current speaker state
- New speaker transitions require overcoming a transition penalty (analogous to VBx's transition probability)
- Self-transitions (same speaker continues) are favored by default
- This single addition would dramatically reduce the rapid label-flipping that causes over-segmentation

---

## Key Papers and References

1. Coria et al., "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation," ASRU 2021. https://arxiv.org/abs/2109.06483
2. Coria, "Diart: A Python Library for Real-Time Speaker Diarization," JOSS 2024. https://joss.theoj.org/papers/10.21105/joss.05266
3. Medennikov et al., "Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization," Interspeech 2025. https://arxiv.org/abs/2507.18446
4. Zhang et al., "Fully Supervised Speaker Diarization" (UIS-RNN), ICASSP 2019. https://arxiv.org/abs/1810.04719
5. Xia et al., "Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection," 2021. https://arxiv.org/abs/2109.11641
6. Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx)," Computer Speech & Language 2021. https://arxiv.org/abs/2012.14952
7. Park et al., "Multi-scale Speaker Diarization with Dynamic Scale Weighting," 2022. https://arxiv.org/abs/2203.15974
8. Dawalatabad et al., "ECAPA-TDNN Embeddings for Speaker Diarization," Interspeech 2021. https://arxiv.org/abs/2104.01466
9. "A Review of Common Online Speaker Diarization Methods," 2024. https://arxiv.org/abs/2406.14464
10. "SCDiar: a streaming diarization system based on speaker change detection," 2025. https://arxiv.org/abs/2501.16641
11. "A lightweight approach to real-time speaker diarization," EURASIP JASM 2024. https://link.springer.com/article/10.1186/s13636-024-00382-2
12. Wang, "SpectralCluster: Spectral Clustering for Speaker Diarization." https://github.com/wq2012/SpectralCluster
