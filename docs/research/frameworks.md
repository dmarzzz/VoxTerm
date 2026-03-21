# Speaker Diarization Frameworks Research

**Date:** 2026-03-21
**Context:** VoxTerm needs local, real-time, CPU-based speaker diarization. Current system uses ECAPA-TDNN with naive cosine clustering, causing over-segmentation (2 speakers detected as 5+).

---

## Table of Contents

1. [pyannote.audio](#1-pyannoteaudio)
2. [diart](#2-diart)
3. [NVIDIA NeMo](#3-nvidia-nemo)
4. [SpeechBrain](#4-speechbrain)
5. [WhisperX](#5-whisperx)
6. [simple_diarizer](#6-simple_diarizer)
7. [Resemblyzer](#7-resemblyzer)
8. [LinTO Diarization](#8-linto-diarization)
9. [diarize (FoxNoseTech)](#9-diarize-foxnosetech)
10. [Comparison Matrix](#10-comparison-matrix)
11. [Recommendations for VoxTerm](#11-recommendations-for-voxterm)

---

## 1. pyannote.audio

- **URL:** https://github.com/pyannote/pyannote-audio
- **Latest Version:** 4.0.4 (February 7, 2026)
- **License:** MIT (code), CC-BY-4.0 (community-1 model)
- **Stars:** 8,000+
- **HuggingFace downloads:** 45 million/month

### Architecture

The pipeline has three stages:

1. **Speaker Segmentation** -- PyanNet model (SincNet + LSTM + Linear). Processes 10-second chunks. Uses powerset multi-class classification with 7 output classes: non-speech, speaker 1, speaker 2, speaker 3, and three overlap combinations. Eliminates detection threshold hyperparameter. Segmentation model is ~1.5M parameters (~6 MB ONNX).
2. **Speaker Embedding** -- WeSpeaker ResNet34-LM model (~26.7 MB). Produces 256-dimensional embeddings per segment.
3. **Speaker Clustering** -- VBx (Bayesian HMM clustering of x-vector sequences). Groups segments into speakers.

Two pipeline variants:
- **community-1** (open-source, CC-BY-4.0): Latest open-source model, significantly better than 3.1.
- **precision-2** (commercial): Even better accuracy, self-hosted option.

### Accuracy (DER %)

| Dataset           | Legacy 3.1 | Community-1 | Precision-2 |
|-------------------|-----------|-------------|-------------|
| AISHELL-4         | 12.2      | 11.7        | 11.4        |
| AliMeeting (ch1)  | 24.5      | 20.3        | 15.2        |
| AMI (IHM)         | 18.8      | 17.0        | 12.9        |
| DIHARD 3          | 21.4      | 20.2        | 14.7        |
| VoxConverse v0.3  | 11.2      | 11.2        | 8.5         |

### Real-Time Capability

- **GPU:** Real-time factor ~2.5% on V100 (processes 1 hour in ~1.5 minutes). Very fast.
- **CPU:** Real-time factor 0.5-1.0 (processes 1 hour in 30-60 minutes). Too slow for real-time use on CPU alone.
- **Not designed for streaming.** Processes full recordings. The clustering stage requires seeing all audio.

### Memory Usage

- **v3.x on GPU:** ~2.6 GB VRAM peak
- **v4.0 on GPU:** ~9.5 GB VRAM peak (6x regression, reported bug)
- **RAM:** 8-16 GB recommended; scales with audio length and speaker count
- **CPU-only:** Works but very slow

### Over-Segmentation Handling

- Community-1 brings "significant improvement in speaker counting and assignment" vs 3.1.
- Powerset formulation handles overlapping speech natively (reduces confusion errors).
- VBx clustering uses Bayesian HMM which is more robust than naive clustering.
- Can constrain with `num_speakers`, `min_speakers`, `max_speakers` parameters.
- New "exclusive speaker diarization" mode outputs only one speaker per frame.

### Dependencies

- Python 3.8+, PyTorch, torchaudio, ffmpeg
- HuggingFace account + access token required
- Pure PyTorch (no onnxruntime needed since v3.1)

### Component Reuse

Individual models (segmentation, embedding) can be used standalone. The `pyannote.core` and `pyannote.metrics` libraries are independent.

### Key Takeaway for VoxTerm

Best-in-class accuracy but NOT suitable for real-time CPU streaming out of the box. The segmentation and embedding models could be extracted and used with a custom streaming pipeline (which is exactly what diart does).

---

## 2. diart

- **URL:** https://github.com/juanmc2005/diart
- **Latest Version:** 0.9+ (active development)
- **License:** MIT
- **Published:** JOSS 2024 (Journal of Open Source Software)

### Architecture

Specifically designed for real-time/online speaker diarization. Implements the paper "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation."

Pipeline:
1. **Rolling audio buffer** updated every 500ms (configurable step size)
2. **Local segmentation** on the current buffer using pyannote segmentation model
3. **Speaker embedding** extraction for detected speakers
4. **Incremental online clustering** that improves as conversation progresses
5. **Cannot-link constraints** from segmentation prevent wrongful merges

### Real-Time Latency

- Adjustable between 500ms and 5s
- **GPU latency:** Segmentation 8-12ms, Embedding 12-29ms per step
- **CPU latency:** 26-150ms per step depending on embedding model
- **At 500ms step:** Easily real-time on CPU with lighter models

### Supported Embedding Models

Eight models supported:
- pyannote/embedding (default)
- WeSpeaker variants (ResNet, etc.)
- SpeechBrain ECAPA-TDNN, X-vector, ResNet
- NeMo models
- Compatible with pyannote 3.x+ segmentation models

### Accuracy

DER benchmarks reported on DIHARD III with various hyperparameter configurations (tau, rho, delta). Performance varies by latency setting -- lower latency = higher DER. Systematic analysis on AMI, DIHARD, VoxConverse datasets. Accuracy is generally worse than offline pyannote (expected trade-off for real-time).

### Memory Usage

Not explicitly documented, but uses the same underlying models as pyannote. The incremental clustering adds minimal overhead. The rolling buffer approach inherently limits memory usage compared to full-file processing.

### Over-Segmentation Handling

- **Cannot-link constraints** derived from segmentation prevent two local speakers from being wrongfully merged during incremental clustering.
- Incremental clustering gets more accurate over time as more data is observed.
- Hyperparameters (tau, rho, delta) can be tuned to reduce over-segmentation.

### Dependencies

- Python 3.8+, PyTorch
- ffmpeg < 4.4, PortAudio 19.6.X, libsndfile >= 1.2.2
- pyannote.audio (for default models)
- Optional: sounddevice for microphone input

### Key Takeaway for VoxTerm

**This is the most directly relevant framework for VoxTerm.** It is specifically built for real-time streaming diarization. It wraps pyannote models in an online pipeline with incremental clustering. Can run on CPU at 500ms latency. The architecture is very close to what VoxTerm needs. Main concern: depends on pyannote models (HuggingFace token required, model licensing).

---

## 3. NVIDIA NeMo

- **URL:** https://github.com/NVIDIA-NeMo/NeMo
- **Latest Version:** 2.7.1 (March 20, 2026)
- **License:** Apache 2.0
- **Stars:** 12,000+

### Architecture

Two diarization approaches:

**A) Cascaded (Pipelined) System:**
1. **VAD:** MarbleNet-3x2 (voice activity detection)
2. **Speaker Embedding:** TitaNet (23M parameters for large variant, depth-wise separable Conv1D)
3. **Clustering Module:** Non-trainable, groups embeddings into clusters
4. **Multi-Scale Diarization Decoder (MSDD):** Neural post-processing that refines clustering using multi-scale segments. Applies learned scale weights (CNN or attention based).

**B) End-to-End System:**
- **Sortformer:** Transformer encoder-based, generates speaker labels directly from audio.
- **Streaming Sortformer:** Processes in overlapping chunks with Arrival-Order Speaker Cache (AOSC).

### Model Sizes

- **TitaNet-Large:** 23M parameters, trained on VoxCeleb1/2, Fisher, Switchboard, LibriSpeech, SRE
- **TitaNet-Small:** Available for CPU/small GPU inference
- **MarbleNet:** Smaller model for VAD

### Accuracy (DER %)

| Dataset          | Oracle VAD, Known Speakers | Oracle VAD, Unknown Speakers |
|------------------|--------------------------|-----------------------------|
| NIST SRE 2000    | 6.73                     | 5.38                        |
| AMI Lapel        | 2.03                     | 2.03                        |
| AMI MixHeadset   | 1.73                     | 1.89                        |
| CH109            | 1.19                     | 1.63                        |

Note: These results use Oracle VAD which gives unrealistically good numbers. Real-world DER is higher.

### Real-Time Capability

- Streaming Sortformer enables adjustable latency-accuracy trade-off.
- The cascaded pipeline is designed for batch/offline processing.
- **GPU-focused.** CPU inference is possible but NeMo is primarily optimized for NVIDIA GPUs.

### Over-Segmentation Handling

- MSDD uses multi-scale analysis to refine cluster boundaries.
- Spectral clustering with eigenvalue analysis for automatic speaker count estimation.

### Dependencies

- Python 3.12+, PyTorch 2.6+
- NVIDIA GPU recommended (but not strictly required for inference)
- Heavy dependency tree (full NeMo toolkit is large)
- `pip install nemo-toolkit[all]`

### Key Takeaway for VoxTerm

Overkill for VoxTerm. NeMo is a massive framework designed for NVIDIA GPU infrastructure. The TitaNet embedding model is excellent (could be used standalone), but the full pipeline is too heavy for local CPU real-time use. The Streaming Sortformer is interesting but GPU-centric.

---

## 4. SpeechBrain

- **URL:** https://github.com/speechbrain/speechbrain
- **Latest Version:** 1.0+ (January 2024 major release)
- **License:** Apache 2.0
- **Stars:** 9,000+

### Architecture

SpeechBrain is a general-purpose speech toolkit, not a dedicated diarization framework. It provides building blocks:

1. **Speaker Embedding Models:** ECAPA-TDNN, X-vectors, ResNet
2. **Diarization Recipe (AMI dataset):** Embedding extraction + spectral clustering
3. **VAD:** Various options
4. **Processing utilities:** Segment merging, spectral clustering, DER computation

The AMI diarization recipe uses:
- ECAPA-TDNN for embeddings
- Oracle VAD (ground-truth speech regions)
- Spectral clustering on cosine similarity affinity matrix

### Model Sizes

- ECAPA-TDNN: ~6M parameters (~20 MB)
- X-vector: ~4.2M parameters
- Models available on HuggingFace

### Accuracy

Competitive speaker verification (EER) but no standardized diarization DER benchmarks prominently published. The recipe is research-oriented.

### Real-Time Capability

Not designed for real-time. No streaming pipeline. Future plans mention "real-time, streamable, small-footprint Conversational AI."

### Over-Segmentation Handling

Spectral clustering with eigengap analysis can estimate speaker count, but no dedicated over-segmentation mitigation beyond clustering quality.

### Dependencies

- Python 3.8+, PyTorch
- HuggingFace Hub for pretrained models
- Moderate dependency footprint

### Key Takeaway for VoxTerm

VoxTerm already uses SpeechBrain's ECAPA-TDNN for embeddings. The main issue (over-segmentation) is not in the embedding model but in the clustering. SpeechBrain's spectral clustering recipe could replace naive cosine clustering, but the real value here is better clustering algorithms, not a different embedding model.

---

## 5. WhisperX

- **URL:** https://github.com/m-bain/whisperX
- **Latest Version:** Active development (migrated to pyannote v4)
- **License:** BSD-2-Clause
- **Stars:** 13,000+

### Architecture

Three-stage pipeline for transcription + diarization:
1. **Transcription:** OpenAI Whisper (via faster-whisper backend) with batch inference
2. **Alignment:** wav2vec2 forced alignment for word-level timestamps
3. **Speaker Assignment:** pyannote-audio diarization, then word-speaker alignment

### Diarization Details

WhisperX does NOT implement its own diarization. It delegates entirely to pyannote (community-1 model). It then assigns speaker labels to transcribed words by correlating diarization segments with word timestamps.

### Real-Time Capability

**Not real-time.** Processes complete audio files. Batch-oriented (70x real-time transcription speed on GPU for the ASR part).

### Memory

< 8GB GPU for Whisper large-v2 with beam_size=5. CPU mode supported but slow.

### Over-Segmentation

Inherits pyannote's behavior. No additional handling.

### Key Takeaway for VoxTerm

Not relevant for the diarization problem specifically. WhisperX is a transcription-first tool that wraps pyannote. VoxTerm already has its own ASR (Qwen3-ASR). No unique diarization contribution.

---

## 6. simple_diarizer

- **URL:** https://github.com/cvqluu/simple_diarizer
- **Latest Version:** 0.0.13 (December 12, 2022 -- inactive)
- **License:** GPL-3.0
- **Stars:** ~350

### Architecture

Minimal pipeline:
1. **VAD:** Silero VAD
2. **Speaker Embeddings:** SpeechBrain models (ECAPA-TDNN or X-vector, selectable)
3. **Clustering:** Agglomerative Hierarchical Clustering (AHC) or Spectral Clustering (SC)

### Usage

```python
from simple_diarizer.diarizer import Diarizer
diar = Diarizer(embed_model='ecapa', cluster_method='sc')
segments = diar.diarize(wav_path, num_speakers=2)
```

### Real-Time Capability

None. Offline/batch only.

### Over-Segmentation Handling

Requires `num_speakers` parameter or automatic estimation via eigengap. No sophisticated over-segmentation mitigation.

### Key Takeaway for VoxTerm

**Inactive project (no updates since late 2022).** GPL-3.0 license is restrictive. Uses the same ECAPA-TDNN + spectral clustering approach that VoxTerm could implement directly. The main value is code simplicity as a reference implementation. However, the spectral clustering approach here could be studied for VoxTerm's clustering improvement.

---

## 7. Resemblyzer

- **URL:** https://github.com/resemble-ai/Resemblyzer
- **Latest Version:** 0.1.3 (project appears dormant, ~50 total commits)
- **License:** Apache 2.0
- **Stars:** ~2,500

### Architecture

- **Voice Encoder:** GE2E (Generalized End-to-End) loss model, based on the Google paper
- **Embedding:** 256-dimensional vectors
- **Architecture:** 3-layer LSTM + linear projection (relatively small model, ~few MB)
- **Diarization:** Demo-level. Compares continuous embeddings against reference speaker profiles.

### Performance

- ~1000x real-time on GTX 1080
- Runs on CPU
- Not benchmarked on standard diarization datasets

### Real-Time Capability

Fast enough for real-time embedding extraction. But the diarization demo is naive (requires reference audio for each speaker, simple cosine similarity).

### Over-Segmentation Handling

None. The "diarization" is just sliding-window cosine similarity against known speaker profiles. No clustering.

### Key Takeaway for VoxTerm

The GE2E voice encoder is small, fast, and CPU-friendly. However, the embedding quality is significantly below modern ECAPA-TDNN / WeSpeaker models. The diarization approach (reference-based) is fundamentally different from what VoxTerm needs for unknown speakers. **Not recommended** as a diarization solution, but the model is a reference for lightweight embedding extraction.

---

## 8. LinTO Diarization

- **URL:** https://github.com/linto-ai/linto-diarization
- **Latest Version:** Active (301 commits)
- **License:** AGPLv3 (copyleft, restrictive)

### Architecture

Docker-based microservice wrapping multiple diarization backends:
- **PyAnnote** (recommended)
- **simple_diarizer**
- **PyBK** (deprecated)

Deployment: HTTP API server or Celery worker with Redis message broker.

### Speaker Identification

Supports known-speaker identification when reference audio samples are provided, using Qdrant vector database for embedding storage.

### Real-Time Capability

Not real-time. Service-oriented (submit audio, get results).

### Over-Segmentation

Exposes `speaker_count` and `max_speaker` parameters to constrain output.

### Key Takeaway for VoxTerm

This is an API wrapper, not a diarization engine. The underlying diarization is pyannote or simple_diarizer. AGPLv3 license is problematic. The speaker identification with reference embeddings in Qdrant is an interesting pattern for VoxTerm's voice tagging feature.

---

## 9. diarize (FoxNoseTech)

- **URL:** https://github.com/FoxNoseTech/diarize
- **Latest Version:** 0.1.1 (March 6, 2026 -- very new)
- **License:** Apache 2.0
- **Stars:** New project

### Architecture

CPU-optimized four-stage pipeline:
1. **VAD:** Silero VAD (MIT license)
2. **Speaker Embeddings:** WeSpeaker ResNet34-LM via ONNX Runtime (Apache 2.0) -- 256-dimensional
3. **Speaker Count Estimation:** GMM + BIC (Bayesian Information Criterion)
4. **Clustering:** scikit-learn spectral clustering

All neural inference runs through ONNX Runtime for CPU optimization.

### Accuracy

~10.8% DER on VoxConverse dev set (weighted). Compare:
- pyannote community-1: ~11.2%
- pyannote precision-2: ~8.5%

### Performance

- **8x faster than real-time on CPU** (RTF 0.12)
- 10-minute meeting processed in ~75 seconds
- pyannote CPU RTF: 0.86 (7x slower)

### Real-Time Capability

**Not streaming** (processes full files). Streaming is on the roadmap.

### Over-Segmentation Handling

GMM + BIC for speaker count estimation. No explicit over-segmentation mitigation beyond clustering.

### Dependencies

- PyTorch, torchaudio, soundfile, scikit-learn, ONNX Runtime
- All permissive licenses (MIT, Apache 2.0, BSD)
- No HuggingFace token required

### Key Takeaway for VoxTerm

**Very interesting new project.** Demonstrates that using WeSpeaker ResNet34-LM via ONNX + spectral clustering achieves near-pyannote accuracy at 7x faster CPU speed. The specific model (WeSpeaker ResNet34-LM) and ONNX approach could be directly adopted by VoxTerm. However, it is batch-only (no streaming yet) and very new (v0.1.1).

---

## 10. Comparison Matrix

| Framework         | Real-Time | CPU Viable | DER (VoxConverse) | License     | Model Size | Over-Seg Handling | Active |
|-------------------|-----------|-----------|-------------------|-------------|------------|-------------------|--------|
| pyannote 4.0      | No        | Slow      | ~11.2%            | MIT/CC-BY   | ~33 MB     | VBx + powerset    | Yes    |
| diart             | **Yes**   | **Yes**   | Higher (online)   | MIT         | ~33 MB*    | Cannot-link       | Yes    |
| NeMo              | Streaming | GPU pref  | ~2% (oracle VAD)  | Apache 2.0  | ~90 MB+    | MSDD multi-scale  | Yes    |
| SpeechBrain       | No        | Yes       | Not published     | Apache 2.0  | ~20 MB     | Spectral cluster  | Yes    |
| WhisperX          | No        | Slow      | Inherits pyannote | BSD-2       | 1+ GB      | Inherits pyannote | Yes    |
| simple_diarizer   | No        | Yes       | Not published     | GPL-3.0     | ~20 MB     | Minimal           | Dead   |
| Resemblyzer       | Embed only| Yes       | Not benchmarked   | Apache 2.0  | ~5 MB      | None              | Dead   |
| LinTO             | No        | Wrapper   | Inherits backend  | AGPLv3      | Varies     | Parameter limits  | Yes    |
| diarize (FoxNose) | No        | **Yes**   | ~10.8%            | Apache 2.0  | ~27 MB     | GMM+BIC           | New    |

*diart uses pyannote models by default

---

## 11. Recommendations for VoxTerm

### The Core Problem

VoxTerm's over-segmentation issue (2 speakers -> 5+ detected) is a **clustering problem**, not an embedding problem. The ECAPA-TDNN embeddings are likely fine. The fix should focus on:

1. Better clustering algorithm (replace naive cosine similarity)
2. Better speaker count estimation
3. Constraints to prevent over-splitting

### Option A: Adopt diart (Recommended for Real-Time)

**Why:** Diart is the only framework specifically designed for real-time streaming diarization. It wraps pyannote's segmentation model with incremental online clustering and cannot-link constraints that directly address over-segmentation.

**How:** Replace VoxTerm's current clustering with diart's `SpeakerDiarization` pipeline. Configure with 500ms step for real-time performance.

**Concerns:**
- Depends on pyannote models (HuggingFace token, CC-BY-4.0 license for community-1)
- CPU latency 26-150ms per step (should be OK for 500ms steps)
- Adds pyannote.audio as a dependency

### Option B: Steal the Clustering (Most Practical)

Instead of adopting a full framework, extract the key improvements:

1. **Replace cosine clustering with spectral clustering** (scikit-learn, no new dependencies). Both simple_diarizer and diarize (FoxNoseTech) use this successfully.
2. **Add VBx (Bayesian HMM clustering)** from pyannote. This is the clustering method that gives pyannote its accuracy advantage.
3. **Add speaker count estimation via eigengap or GMM+BIC** instead of fixed thresholds.
4. **Switch embedding model to WeSpeaker ResNet34-LM via ONNX** for faster CPU inference and better embeddings. The diarize (FoxNoseTech) project proves this achieves ~10.8% DER.

### Option C: pyannote Powerset Segmentation + Custom Clustering

Use pyannote's segmentation model (PyanNet, ~6 MB) as a local speaker segmentation frontend instead of pure embedding-based clustering. The powerset approach inherently handles overlapping speech and outputs per-frame speaker activity. Then use a lightweight clustering method to link segments across chunks.

**Advantage:** The segmentation model is small and fast. It handles the hardest part (local who-is-speaking detection) with a neural model trained on diverse data.

**Concern:** Still requires pyannote dependency and HuggingFace token.

### Priority Recommendation

**Short term (immediate fix):** Replace naive cosine clustering with spectral clustering + eigengap speaker count estimation. This requires only scikit-learn (already likely a dependency) and directly addresses over-segmentation. Look at diarize (FoxNoseTech) source code for implementation reference.

**Medium term:** Evaluate switching from ECAPA-TDNN to WeSpeaker ResNet34-LM via ONNX for faster, more accurate embeddings on CPU.

**Long term:** If real-time accuracy still insufficient, adopt diart's incremental clustering approach with cannot-link constraints, or integrate pyannote's powerset segmentation model.
