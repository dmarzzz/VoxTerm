# Speaker Diarization: Benchmarks, State of the Art, and Engineering Guide

Research compiled 2026-03-21. Covers results through early 2026.

---

## Table of Contents

1. [DER Metric Explained](#1-der-metric-explained)
2. [Major Benchmarks and Datasets](#2-major-benchmarks-and-datasets)
3. [State-of-the-Art Systems](#3-state-of-the-art-systems)
4. [Benchmark Results Summary](#4-benchmark-results-summary)
5. [Architectural Approaches: EEND vs Clustering vs Hybrid](#5-architectural-approaches-eend-vs-clustering-vs-hybrid)
6. [Online / Streaming Diarization](#6-online--streaming-diarization)
7. [Error Analysis: What Goes Wrong](#7-error-analysis-what-goes-wrong)
8. [Number of Speakers Estimation](#8-number-of-speakers-estimation)
9. [Overlap Handling](#9-overlap-handling)
10. [Production System Design](#10-production-system-design)
11. [Practical Engineering Recommendations](#11-practical-engineering-recommendations)
12. [What "Good" Looks Like for VoxTerm](#12-what-good-looks-like-for-voxterm)
13. [Sources](#13-sources)

---

## 1. DER Metric Explained

**Diarization Error Rate (DER)** is the primary evaluation metric. It is the sum of three error types, normalized by the total reference speech duration:

```
DER = (Missed Speech + False Alarm + Speaker Confusion) / Reference Duration * 100
```

| Error Type        | Cause                                    | Pipeline Stage    |
|-------------------|------------------------------------------|-------------------|
| **Missed Speech** | System fails to detect speech            | VAD / Segmentation|
| **False Alarm**   | System labels non-speech as speech       | VAD / Segmentation|
| **Speaker Confusion** | Speech assigned to the wrong speaker | Clustering / Assignment |

**Evaluation protocol matters enormously.** Results vary dramatically based on:

- **Collar**: A forgiveness window (typically 0.25s) around reference boundaries. "No collar" (0.0s) is the strictest and most realistic evaluation. Some older papers report with 0.25s collar, which inflates results.
- **Overlap scoring**: Whether overlapped speech regions are included. Including overlap is harder but more realistic.
- **Oracle VAD**: Whether ground-truth speech/non-speech is provided. Oracle VAD removes missed speech and false alarm errors entirely, leaving only confusion -- not realistic.
- **Oracle speaker count**: Whether the true number of speakers is given. Production systems must estimate this.

The strictest (and most informative) protocol: **no collar, overlap scored, no oracle VAD, no oracle speaker count**. All modern benchmarks use this.

---

## 2. Major Benchmarks and Datasets

### CALLHOME

- **Domain**: Telephone conversations (narrowband, 8kHz upsampled)
- **Languages**: Multilingual -- English, Mandarin, Japanese, German, Spanish, Arabic
- **Speakers per recording**: Mostly 2 (some 3-6)
- **Overlap**: ~12.6% of speech time
- **Characteristics**: Casual conversational speech, frequent interruptions, telephone channel distortion
- **Why it matters**: The classic diarization benchmark. Easy-to-moderate difficulty.

### AMI Meeting Corpus

- **Domain**: Meeting recordings (100 hours)
- **Microphone configurations**: IHM (individual headset mix), SDM (single distant microphone), array
- **Speakers per recording**: Typically 4
- **Overlap**: ~15.9% of speech time
- **Characteristics**: Controlled meeting rooms, multiple mic setups allow comparing near-field vs far-field
- **Why it matters**: Standard meeting diarization benchmark. SDM variant is much harder than IHM.

### DIHARD (I, II, III)

- **Domain**: Extremely diverse -- web video, meeting speech, clinical interviews, restaurant conversation, child language, audiobooks, court proceedings, etc.
- **Speakers**: Varies widely (2 to 10+)
- **Characteristics**: Deliberately difficult. Diverse acoustic conditions, background noise, reverberation, many speakers, varied recording quality.
- **Why it matters**: Stress-tests diarization systems. Closest to "real world chaos."
- **Track 1**: Reference VAD provided (only confusion measured)
- **Track 2**: No reference VAD (full pipeline evaluated -- the hard track)

### VoxConverse

- **Domain**: YouTube videos -- political debates, news broadcasts, panel discussions, entertainment
- **Speakers**: High speaker count (often 5-20+)
- **Characteristics**: Challenging in-the-wild audio with music, laughter, applause, variable recording quality
- **Why it matters**: Tests scalability to many speakers in uncontrolled conditions.

### AliMeeting

- **Domain**: Mandarin meeting recordings
- **Configurations**: Near-field and far-field
- **Speakers**: 2-4 per session
- **Overlap**: Higher overlap ratios than AMI
- **Why it matters**: Non-English meeting benchmark, tests far-field robustness.

### AISHELL-4

- **Domain**: Mandarin meeting/conference recordings
- **Speakers**: 4-8 per recording
- **Characteristics**: Real conference recordings with distant microphones
- **Why it matters**: Large-scale Chinese meeting benchmark.

---

## 3. State-of-the-Art Systems

### PyannoteAI (Commercial, 2024-2025)

- **Type**: Cloud-based commercial API
- **Architecture**: Modular pipeline -- segmentation (pyannote/segmentation-3.0) + speaker embedding + clustering
- **Input**: 16kHz mono audio
- **Overall DER**: 11.2% (across multilingual benchmark suite)
- **Strengths**: Best overall accuracy, stable across speaker counts, works across languages
- **Limitations**: Commercial/cloud-only; the open-source pyannote 3.1 is less accurate

### Pyannote 3.1 (Open Source)

- **Type**: Open-source modular pipeline (pure PyTorch)
- **Architecture**: Speaker segmentation model + speaker embedding model + agglomerative clustering
- **Key DER results** (no collar, overlap scored):

| Dataset          | DER%  | FA%  | Miss% | Conf% |
|------------------|-------|------|-------|-------|
| AISHELL-4        | 12.2  | 3.8  | 4.4   | 4.0   |
| VoxConverse v0.3 | 11.3  | 4.1  | 3.4   | 3.8   |
| AMI (IHM)        | 18.8  | 3.6  | 9.5   | 5.7   |
| DIHARD 3 (Full)  | 21.7  | 6.2  | 8.1   | 7.3   |
| CALLHOME pt2     | 28.5  | --   | --    | --    |
| REPERE phase 2   | 7.8   | 1.8  | 2.6   | 3.5   |

- **Strengths**: Well-documented, easy to deploy, pure PyTorch, supports num_speakers/min_speakers/max_speakers hints
- **Weaknesses**: ~2x slower on CPU than v3.0; higher DER than commercial version

### DiariZen (Open Source, BUT/FIT)

- **Type**: Open-source hybrid pipeline
- **Architecture**: WavLM Large (self-supervised) + Conformer layers for EEND segmentation, integrated with pyannote clustering
- **Model efficiency**: 80% structured pruning reduces WavLM from 316.6M to 63.3M parameters (MACs from 17.8G to 3.8G/sec)
- **Key DER results** (no collar, overlap scored):

| Dataset          | Pyannote 3.1 | DiariZen-Large-s80-v2 |
|------------------|-------------:|----------------------:|
| AMI-SDM          | 22.4         | **13.9**              |
| AISHELL-4        | 12.2         | **10.1**              |
| AliMeeting far   | 24.4         | **10.8**              |
| MSDWild          | 25.3         | **15.8**              |
| DIHARD3 full     | 21.7         | **14.5**              |
| RAMC             | 22.2         | **11.0**              |
| VoxConverse      | 11.3         | **9.1**               |

- **Overall DER**: 13.3% (competitive open-source alternative)
- **Strengths**: Dramatically outperforms pyannote 3.1 on difficult datasets; excellent with 5+ speakers (DER=7.1%); WavLM provides rich acoustic representations
- **Weaknesses**: Requires GPU for reasonable speed; more complex setup

### EEND-TA (End-to-End, Research)

- **Type**: End-to-end neural diarization
- **Architecture**: Conformer encoder + combiner + Transformer decoder with Conversational Summary Vector
- **Training**: Scaled pre-training with up to 8 speakers, 80,000+ hours simulated from LibriSpeech
- **Key DER results**:

| Dataset          | EEND-TA (fine-tuned) |
|------------------|---------:|
| DIHARD III       | **14.49** |
| AMI-Mix          | 11.04    |
| AMI-SDM          | 15.33    |
| AliMeeting-far   | 11.41    |
| AliMeeting-near  | 8.55     |
| VoxConverse      | 14.29    |
| CALLHOME         | 17.24    |
| AISHELL-4        | 12.21    |

- **Speed**: Processes 158 hours of test audio in 97 seconds on GPU
- **Strengths**: Single unified model, no separate clustering step, handles overlap natively
- **Weaknesses**: Degrades with >4 speakers; memory-intensive for long recordings

### NVIDIA Streaming Sortformer v2

- **Type**: End-to-end streaming diarization
- **Architecture**: Fast-Conformer encoder (NEST, 17 layers) + 18-layer Transformer + sigmoid outputs for up to 4 speakers per frame
- **Training data**: ~5,000 hours (real + simulated)
- **Streaming mechanism**: Arrival-Order Speaker Cache (AOSC) for consistent speaker tracking across chunks
- **Speed**: 214.3x real-time factor (extremely fast)
- **Strengths**: Purpose-built for streaming; low latency; fast inference
- **Weaknesses**: Limited to 4 speakers; performance degrades beyond that; less accurate than offline systems

### VBx (Variational Bayes HMM over x-vectors)

- **Type**: Clustering-based (offline)
- **Architecture**: x-vector extraction + Bayesian HMM clustering
- **Key DER results**:
  - CALLHOME: 4.42% (forgiving protocol) / 14.21% (fair protocol)
  - DIHARD II: 18.19% (dev) / 18.55% (eval)
- **Strengths**: Mathematically principled; robust; open-source; used as initialization in many hybrid systems
- **Weaknesses**: Cannot handle overlapped speech natively; requires separate overlap detector

### Google Turn-to-Diarize

- **Type**: Online/streaming, on-device
- **Architecture**: Transformer Transducer detects speaker turns via special `<st>` token during transcription; speaker embeddings computed per turn; constrained clustering
- **Clustering strategy**: AHC for short sequences, spectral clustering with eigen-gap for medium, AHC pre-clustering + spectral for long
- **Strengths**: Designed for mobile/on-device; leverages ASR turn detection to reduce clustering cost
- **Application**: Google Recorder app

### Microsoft Azure Real-Time Diarization

- **Type**: Cloud-based streaming API
- **Architecture**: Single-channel streaming diarization
- **Features**: Real-time speaker labels (GUEST1, GUEST2, etc.); improved model with ~3% WDER improvement
- **Advanced**: Audio-visual meeting transcription combining face tracking, face recognition, speaker recognition, and sound source localization

---

## 4. Benchmark Results Summary

### Cross-System Comparison (2025 Benchmarking Study)

Evaluated with the strictest protocol: no collar, overlap scored, no oracle VAD, no oracle speaker count.

**Overall DER across all datasets:**

| System               | Type         | Overall DER | Speed      |
|----------------------|-------------|:-----------:|:----------:|
| PyannoteAI           | Commercial  | **11.2%**   | Cloud      |
| DiariZen Large v2    | Open source | **13.3%**   | GPU        |
| Sortformer v2-stream | Open source | ~14-15%     | 214x RT    |
| Pyannote 3.1         | Open source | ~18-22%     | GPU/CPU    |

**Performance by language:**

| System           | Mandarin | English | German | Japanese | Spanish |
|------------------|:--------:|:-------:|:------:|:--------:|:-------:|
| PyannoteAI       | 10.0     | 6.6     | 8.3    | 13.8     | 14.3    |
| DiariZen         | 10.1     | 7.0     | 11.6   | 15.6     | 19.1    |
| SF v2-stream     | 9.4      | 14.1    | 9.6    | 12.7     | 21.1    |

**Performance by speaker count:**

| System           | 1 spk | 2 spk | 3 spk | 4 spk | 5+ spk |
|------------------|:-----:|:-----:|:-----:|:-----:|:------:|
| PyannoteAI       | 2.7   | 9.9   | 9.1   | 10.1  | 6.6    |
| DiariZen         | 2.3   | 11.4  | 10.3  | 12.7  | 7.1    |
| SF v2-stream     | 4.7   | 10.4  | 14.1  | 13.2  | 22.7   |

Key observation: Sortformer v2 degrades sharply with 5+ speakers due to its 4-speaker training limit. DiariZen and PyannoteAI handle many speakers gracefully.

### DIHARD III Challenge Results (Top 5)

**Track 1 (with reference VAD):**

| Rank | Team           | DER%  |
|------|----------------|------:|
| 1    | USTC-NELSLIP   | 13.45 |
| 2    | Hitachi-JHU    | 14.09 |
| 3    | Clova          | 15.40 |
| 4    | DKU-Duke-Lenovo| 15.43 |
| 5    | BUT            | 15.46 |

**Track 2 (no reference VAD -- full pipeline):**

| Rank | Team           | DER%  |
|------|----------------|------:|
| 1    | USTC-NELSLIP   | 19.37 |
| 2    | Hitachi-JHU    | 20.01 |
| 3    | DKU-Duke-Lenovo| 21.63 |
| 4    | team_f         | 23.86 |
| 5    | Clova          | 24.31 |

The ~6% gap between Track 1 and Track 2 demonstrates how much VAD quality matters.

### VoxConverse / VoxSRC Challenge Results

| System        | Year | DER%  | Notes                              |
|---------------|------|------:|-------------------------------------|
| ByteDance     | 2021 | 5.15  | Eval set, best submission           |
| BUCEA         | 2022 | 5.48  | Eval set                            |
| DiariZen v2   | 2025 | 9.1   | No collar, overlap scored           |
| Pyannote 3.1  | 2024 | 11.3  | No collar, overlap scored           |

Note: VoxSRC challenge numbers may use different scoring protocols than the 2025 benchmarking study.

---

## 5. Architectural Approaches: EEND vs Clustering vs Hybrid

### Clustering-Based (Traditional Pipeline)

```
Audio -> VAD -> Segmentation -> Embedding Extraction -> Clustering -> Output
```

**Advantages:**
- Robust across diverse acoustic conditions
- Handles arbitrary number of speakers
- Works well for long recordings
- Each component can be independently optimized and debugged
- Mature, well-understood

**Disadvantages:**
- Cannot natively handle overlapped speech (needs separate overlap detector)
- Separately optimized modules may not reach global optimum
- More complex pipeline with multiple failure points

**Representative systems:** Pyannote 3.1, VBx, Google Turn-to-Diarize

### End-to-End Neural Diarization (EEND)

```
Audio -> Single Neural Network -> Frame-level speaker activity -> Output
```

**Advantages:**
- Handles overlapped speech natively (multi-label per frame)
- End-to-end optimization of DER
- Simpler pipeline (single model)
- Often faster inference

**Disadvantages:**
- Struggles with more than ~4 speakers (fixed output dimension)
- Memory-intensive for long recordings (>10 minutes)
- Requires large amounts of simulated training data
- Unknown/large speaker count at inference is problematic

**Representative systems:** EEND-TA, Sortformer

### Hybrid (Best of Both Worlds)

```
Audio -> Local EEND (short chunks with overlap handling) -> Global Clustering -> Output
```

**Advantages:**
- EEND handles overlap within local windows
- Clustering handles arbitrary speaker count and long recordings
- Currently achieves best results in practice

**Disadvantages:**
- More complex than either approach alone
- Requires careful design of chunk boundaries and merging

**Representative systems:** DiariZen, pyannote (with powerset segmentation)

### Current Consensus

Hybrid approaches dominate. The field has converged on: **local neural segmentation (handling overlap) + global clustering (handling speaker count and long recordings)**. This is what pyannote and DiariZen both implement, with DiariZen using WavLM for richer local representations.

---

## 6. Online / Streaming Diarization

### Latency-Accuracy Tradeoff

Online systems systematically underperform offline equivalents because they have less context. Key findings:

- **Sweet spot latency**: Performance improves as latency increases, but plateaus. The sweet spot varies by system:
  - ~3 seconds for some lightweight approaches
  - ~5.5 seconds for CPU-based approaches with near-offline quality
  - ~15 seconds for some block-online systems
- **Beyond the sweet spot**: Marginal accuracy gains diminish rapidly
- **Buffer size for ASR alignment**: 100-250ms audio buffers provide the best balance of word accuracy and responsiveness

### Streaming Approaches

| Approach                  | Latency    | Quality vs Offline | Notes                             |
|---------------------------|:----------:|:------------------:|-----------------------------------|
| Block-online (large)      | 10-30s     | -2-5% DER          | Process audio in large chunks     |
| Block-online (small)      | 3-5s       | -5-10% DER         | Smaller chunks, more errors       |
| Frame-level streaming     | <1s        | -10-15% DER        | Real-time, significant quality loss|
| Retrospective refinement  | Variable   | Near offline        | Fast initial labels, refined later|

### Key Streaming Challenges

1. **Local optimum problem**: Without the full recording, clustering may group speakers incorrectly early on, and these errors propagate
2. **Speaker emergence**: A new speaker appearing mid-stream forces re-evaluation of all previous assignments
3. **Computational cost**: Spectral clustering has O(n^3) complexity; long streams require chunking strategies

### Practical Streaming Solutions

- **Arrival-Order Speaker Cache (AOSC)**: Used by NVIDIA Sortformer. Maintains a memory buffer of detected speakers, enabling consistent tracking across chunks.
- **Rolling buffer**: Process a sliding window (e.g., 500ms updates to a 5-15 second buffer) with incremental clustering.
- **Dual-label strategy**: Emit fast preliminary labels, then refine with a reconciliation pass.

---

## 7. Error Analysis: What Goes Wrong

### Dominant Error Types

From the 2025 benchmarking study: **Missed speech is the dominant failure case across all diarization models.** This is primarily due to boundary detection errors (detecting speech onset/offset incorrectly) rather than entirely missing short utterances.

Breakdown of typical DER contributions (from pyannote 3.1 on representative datasets):

| Error Type        | Typical Contribution | Primary Cause                        |
|-------------------|:--------------------:|--------------------------------------|
| Missed Speech     | 30-50% of total DER  | VAD/segmentation boundary errors     |
| Speaker Confusion | 25-40% of total DER  | Clustering errors, similar voices    |
| False Alarm       | 15-30% of total DER  | Non-speech classified as speech      |

### What Makes Diarization Fail

**Audio Quality Factors:**
- Reverberation and room acoustics (far-field is much harder than near-field: AMI IHM 18.8% vs AMI SDM 22.4%)
- Background noise (especially non-stationary: music, TV, other conversations)
- Telephone/compressed audio (bandwidth limitations mask speaker discriminative features)
- Variable recording levels across speakers

**Speaker Factors:**
- Similar-sounding speakers (same gender, age, dialect)
- Very short utterances (<1 second) -- insufficient embedding quality
- Speakers with dramatically different volume levels
- Code-switching (language switching within a conversation)

**Structural Factors:**
- Overlapping speech (15-20% of meeting speech; one of the largest error contributors)
- Very many speakers (performance degrades significantly above 4-6 speakers for most systems)
- Long silences followed by speaker changes
- Rapid back-and-forth turn-taking

### Over-Segmentation

Modern diarization systems **intentionally over-segment** as a first step -- it is easier to merge incorrectly split speakers than to disentangle incorrectly merged ones. The merging/clustering step then reduces the speaker count.

**Over-segmentation becomes a problem when:**
- The clustering threshold is set too conservatively (refuses to merge)
- Speaker embeddings from the same person are inconsistent (due to noise, emotion changes, or short segments)
- The speaker count estimator overestimates
- Domain mismatch between training and inference data

**Under-segmentation (merging distinct speakers) is harder to recover from** and generally worse for downstream applications.

---

## 8. Number of Speakers Estimation

Estimating the number of speakers is one of the hardest sub-problems. Most production systems use one of:

### Spectral Clustering with Eigen-Gap

The standard approach. After building an affinity matrix from speaker embeddings:
1. Compute eigenvalues of the Laplacian matrix
2. Find the maximum gap between consecutive eigenvalues
3. The number of clusters = position of the maximum gap

**Normalized Maximum Eigengap (NME)**: An improvement that auto-tunes both the number of clusters and the affinity matrix threshold without development set tuning. Achieved 17% relative improvement in speaker error rate on CALLHOME.

### Providing Hints

Most production pipelines (including pyannote) support:
- `num_speakers=N` -- exact count if known
- `min_speakers=M, max_speakers=N` -- bounds to constrain estimation

**Providing even a rough bound (e.g., min_speakers=2, max_speakers=4) significantly improves accuracy** by preventing the estimator from going wildly wrong.

### How Systems Handle Unknown Speaker Count

| System           | Method                                      |
|------------------|---------------------------------------------|
| Pyannote 3.1     | Agglomerative clustering with learned threshold |
| DiariZen         | Same as pyannote (shared clustering)         |
| VBx              | Bayesian HMM (speaker count as latent)       |
| EEND-TA          | Attractor-based (learns speaker slots)       |
| Sortformer       | Fixed at 4 outputs; cannot estimate          |
| Turn-to-Diarize  | Eigen-gap spectral clustering                |

---

## 9. Overlap Handling

Overlapping speech accounts for 12-20% of speech in meetings and conversations. It is one of the largest contributors to DER.

### Approaches

**End-to-end (EEND/powerset):** Predicts per-frame activity for each speaker simultaneously. Handles overlap natively by allowing multiple speakers to be active on the same frame. This is the approach used by pyannote's segmentation model (powerset training) and DiariZen's WavLM+Conformer EEND module.

**Separate overlap detector:** Traditional clustering pipelines add a separate overlap detection step. When overlap is detected, the system assigns the region to the two most likely speakers. Used by VBx and earlier pyannote versions.

**Hybrid selection:** Some systems (e.g., for the MISP 2025 challenge) adaptively choose between VBx (better for low overlap) and EEND (better for high overlap) based on estimated overlap ratio.

**Graph-based methods:** Graph neural networks that assign multiple community labels per node, enabling overlap detection as a clustering output.

### Practical Impact

- Without overlap handling: DER increases by 5-15% absolute on meeting data
- With good overlap handling: The gap between overlap-scored and non-overlap-scored DER narrows to 2-5%
- Modern systems (pyannote 3.1, DiariZen) handle overlap reasonably well out of the box

---

## 10. Production System Design

### Typical Production Pipeline

```
Audio Ingestion
    |
    v
Preprocessing (resample to 16kHz mono, normalize)
    |
    v
Speaker Diarization (segmentation + embedding + clustering)
    |
    v
Speech Recognition (ASR)
    |
    v
Alignment & Transcript Formatting (assign words to speakers)
    |
    v
Post-Processing (merge short segments, smooth speaker labels)
    |
    v
Downstream (summarization, analytics, etc.)
```

### Key Production Considerations

1. **Diarization errors propagate downstream.** A transcript with perfect WER but wrong speaker labels can break application logic. Speaker confusion is often worse than missed speech for downstream consumers.

2. **Batch vs streaming architecture:**
   - **Batch** (offline): Process entire recording at once. Clustering sees full conversation context. Better accuracy. Suitable for recordings, podcasts, call center review.
   - **Streaming** (online): Process in chunks with speaker cache. Lower accuracy but real-time. Required for live meetings, voice assistants.

3. **Multi-channel advantage:** When you have information about which device each audio segment came from, diarize each endpoint separately to avoid cross-device speaker confusion.

4. **Post-processing matters:** Simple heuristics like minimum segment duration (e.g., discard speaker segments < 0.5s) and merge-nearby-same-speaker can meaningfully improve perceived quality even if DER changes marginally.

5. **DiarizationLM (Google, 2024):** Uses LLMs to post-process diarization output, correcting speaker labels using linguistic context from the transcript. A promising direction for reducing confusion errors.

---

## 11. Practical Engineering Recommendations

### What Matters Most (Ranked)

1. **Audio quality and preprocessing** -- Clean 16kHz mono audio with good SNR matters more than model choice. Noise reduction and normalization yield disproportionate gains.

2. **VAD quality** -- VAD errors account for 50-70% of DER (missed speech + false alarm). A good VAD is the single most impactful component. The ~6% DER gap between DIHARD III Track 1 (oracle VAD) and Track 2 (system VAD) demonstrates this.

3. **Speaker count hints** -- If you know the approximate number of speakers, provide min/max bounds. This eliminates the hardest estimation problem and can improve DER by 5-10% relative.

4. **Embedding model quality** -- Self-supervised models (WavLM) dramatically outperform older embeddings. DiariZen's gains over pyannote 3.1 come largely from WavLM's richer representations.

5. **Overlap handling** -- Use a system with native overlap support (pyannote 3.1+, DiariZen, EEND-TA). This matters more for meeting/conversation scenarios (15-20% overlap) than for interviews/podcasts (5% overlap).

6. **Clustering algorithm and threshold** -- VBx or agglomerative clustering with a well-tuned threshold. The threshold controls the over-segmentation/under-segmentation tradeoff.

### Common Mistakes to Avoid

- **Ignoring preprocessing**: Feeding compressed, noisy, or non-16kHz audio directly
- **Not providing speaker count hints** when you have them (even rough bounds help)
- **Using models out-of-domain** without any adaptation (e.g., a model trained on meetings applied to phone calls)
- **Ignoring overlap** in evaluation -- report DER with overlap scored for honest results
- **Over-tuning on one dataset** -- systems tuned to one benchmark often fail on others
- **Expecting <10% DER on hard data** -- DIHARD-style diverse audio will always be harder than clean meetings

### Accuracy Expectations by Scenario

| Scenario                              | Realistic DER  | Notes                                 |
|---------------------------------------|:--------------:|---------------------------------------|
| Clean audio, 2 speakers, known count  | 5-10%          | Near-production quality               |
| Clean audio, 2-3 speakers, unknown    | 8-15%          | Good for most applications            |
| Meeting room, 4 speakers, near-field  | 12-20%         | Overlap and crosstalk add errors      |
| Meeting room, 4 speakers, far-field   | 15-25%         | Reverberation hurts significantly     |
| Diverse/noisy audio, unknown speakers | 20-30%         | Expect manual correction needed       |
| In-the-wild audio, many speakers      | 25-40%+        | Very challenging; all systems struggle|

### Performance Scale

- **DER < 10%**: Excellent. Production-ready for most use cases.
- **DER 10-15%**: Good. Usable with light manual review.
- **DER 15-25%**: Fair. Noticeable errors; may need correction for critical work.
- **DER > 25%**: Challenging audio or system limitations. Extensive manual correction likely needed.

---

## 12. What "Good" Looks Like for VoxTerm

Given VoxTerm's architecture (local offline transcription with ECAPA-TDNN for speaker embeddings):

### Realistic Targets

- **2-speaker conversations with clean audio**: Target DER 10-15%. Achievable with ECAPA-TDNN embeddings + reasonable clustering.
- **3-4 speaker meetings**: Target DER 15-22%. Overlap handling and speaker count estimation become critical.
- **Unknown/many speakers**: DER 20-30%. Expect degradation; provide user controls (min/max speakers).

### Highest-Impact Improvements (in order)

1. **VAD quality**: Ensure the VAD (whether from the ASR model or a dedicated module) is well-tuned. This is the single largest lever.
2. **Speaker count bounds**: Allow users to specify approximate speaker count. Even "2-4 speakers" helps enormously.
3. **Minimum segment duration**: Post-process to merge or discard very short speaker segments (<0.3-0.5s). This reduces over-segmentation artifacts.
4. **Clustering threshold tuning**: The agglomerative clustering threshold (or equivalent) is the most sensitive hyperparameter. It controls the over-segmentation/under-segmentation balance.
5. **Overlap-aware segmentation**: If using a segmentation model that supports it (e.g., pyannote's powerset approach), enable overlap detection.
6. **Self-supervised embeddings**: If feasible, replacing ECAPA-TDNN with WavLM-based embeddings (as DiariZen does) could yield significant accuracy gains, though at higher compute cost.

### Key Tradeoff for VoxTerm

Since VoxTerm is local/offline, it can process the entire recording in batch mode. This is a significant advantage over streaming systems. Batch processing allows:
- Global clustering over all speaker embeddings (better accuracy)
- Retrospective speaker count estimation
- Post-processing refinement passes

The tradeoff is between accuracy and latency-to-first-result. For a TUI application, consider showing partial results quickly (streaming-style) while refining in the background.

---

## 13. Sources

### Benchmarking Studies
- [Benchmarking Diarization Models (Lanzendorfer et al., 2025)](https://arxiv.org/html/2509.26177v1)
- [SDBench: A Comprehensive Benchmark Suite for Speaker Diarization](https://arxiv.org/html/2507.16136v2)
- [Open-Source Speaker Diarization Benchmark - Picovoice](https://picovoice.ai/docs/benchmark/speaker-diarization/)
- [Best Speaker Diarization Models Compared 2026](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)

### Systems and Models
- [Pushing the Limits of End-to-End Diarization (EEND-TA)](https://arxiv.org/abs/2509.14737)
- [DiariZen GitHub](https://github.com/BUTSpeechFIT/DiariZen)
- [pyannote/speaker-diarization-3.1 on HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [NVIDIA Streaming Sortformer](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/)
- [NVIDIA Streaming Sortformer v2.1 on HuggingFace](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [VBx: Variational Bayes HMM over x-vectors](https://github.com/BUTSpeechFIT/VBx)
- [Google Turn-to-Diarize](https://research.google/pubs/turn-to-diarize-online-speaker-diarization-constrained-by-transformer-transducer-speaker-turn-detection/)
- [Google Recorder Speaker Labeling](https://research.google/blog/who-said-what-recorders-on-device-solution-for-labeling-speakers/)
- [Microsoft Real-Time Diarization GA](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-general-availability-of-real-time-diarization/4147556)
- [DiarizationLM (Google, 2024)](https://arxiv.org/html/2401.03506v10)
- [SpeakerLM: End-to-End with Multimodal LLMs](https://arxiv.org/html/2508.06372v2)

### Challenges and Datasets
- [DIHARD III Challenge Results](https://dihardchallenge.github.io/dihard3/results.html)
- [DIHARD III Overview Paper](https://dihardchallenge.github.io/dihard3/docs/third_dihard_overview.pdf)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [AMI Diarization Setup (BUT)](https://github.com/BUTSpeechFIT/AMI-diarization-setup)
- [VoxSRC Challenge / VoxConverse (ByteDance 2021)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/ByteDance_diarization.pdf)
- [DISPLACE Challenge 2024](https://displace2024.github.io/)
- [Papers With Code: Speaker Diarization](https://paperswithcode.com/task/speaker-diarization)
- [Awesome Diarization (curated resource list)](https://github.com/wq2012/awesome-diarization)

### Methods and Techniques
- [Auto-Tuning Spectral Clustering with NME](https://arxiv.org/abs/2003.02405)
- [Bayesian HMM clustering of x-vectors (VBx theory paper)](https://arxiv.org/abs/2012.14952)
- [Integrating EEND and Clustering-Based Diarization](https://ar5iv.labs.arxiv.org/html/2010.13366)
- [A Review of Common Online Speaker Diarization Methods](https://arxiv.org/html/2406.14464v1)
- [BER: Balanced Error Rate for Speaker Diarization](https://ar5iv.labs.arxiv.org/html/2211.04304)

### Guides and Overviews
- [What Is Speaker Diarization? (MarkTechPost 2025 Guide)](https://www.marktechpost.com/2025/08/21/what-is-speaker-diarization-a-2025-technical-guide-top-9-speaker-diarization-libraries-and-apis-in-2025/)
- [Speaker Diarization Overview (Encord)](https://encord.com/blog/speaker-diarization/)
- [Speaker Diarization Overview (La Javaness)](https://lajavaness.medium.com/speaker-diarization-an-introductory-overview-c070a3bfea70)
- [Introduction to Speech Processing - Speaker Diarization (Aalto)](https://speechprocessingbook.aalto.fi/Recognition/Speaker_Diarization.html)
- [What is Speaker Diarization (AssemblyAI)](https://www.assemblyai.com/blog/what-is-speaker-diarization-and-how-does-it-work)
- [Gladia: What is Diarization?](https://www.gladia.io/blog/what-is-diarization)
