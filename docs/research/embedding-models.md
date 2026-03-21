# Speaker Embedding Models: Research Report

**Date:** 2026-03-21
**Goal:** Find a replacement for ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, 192-dim) that reduces over-segmentation in VoxTerm's real-time diarization pipeline.

**Core problem:** Same speaker produces inconsistent embeddings across segments due to noise, short utterances, and emotion changes, causing spectral clustering to split 2 real speakers into 5+.

**Constraints:** CPU-only (Apple Silicon M-series), <200ms per 2-5s clip, <500MB model, offline/local.

---

## Quick Comparison Table

| Model | Arch | Params | Emb Dim | Vox1-O EER% | FLOPs/GMACs | CPU RTF | ONNX | Short-Utt Robust | Source |
|---|---|---|---|---|---|---|---|---|---|
| **ECAPA-TDNN** (current) | TDNN+SE+Res2 | 14.7M | 192 | 0.80-0.89 | 3.96G | 0.033 | Yes (SpeechBrain) | Poor | speechbrain/spkrec-ecapa-voxceleb |
| **CAM++** | D-TDNN+context-mask | 7.2M | 512 | 0.65-0.73 | 1.72G | 0.013 | Yes (WeSpeaker) | Moderate | wenet-e2e/wespeaker, modelscope/3D-Speaker |
| **ReDimNet-B1** | Hybrid 1D/2D CNN | 2.2M | ~192 | 0.73-0.85 | 0.54G | ~0.01* | Yes | Unknown | IDRnD/redimnet |
| **ReDimNet-B2** | Hybrid 1D/2D CNN | 4.7M | ~192 | 0.52-0.57 | 0.90G | ~0.01* | Yes | Unknown | IDRnD/redimnet |
| **ERes2NetV2** | Dual-branch Res2Net | 17.8M | 192 | 0.61 | 12.6G | ~0.04* | Yes (3D-Speaker) | **Good** (designed for it) | modelscope/3D-Speaker |
| **ERes2Net-base** | Res2Net+SE | 6.6M | 192 | 0.84 | 5.16G | ~0.02* | Yes (3D-Speaker) | Moderate | modelscope/3D-Speaker |
| **TitaNet-S** | DepthSepConv1D+SE | 6.4M | 192 | 1.15 | N/A | N/A | Yes (NeMo) | Moderate | nvidia/speakerverification_en_titanet_large |
| **TitaNet-M** | DepthSepConv1D+SE | 13.4M | 192 | 0.81 | N/A | N/A | Yes (NeMo) | Moderate | nvidia/speakerverification_en_titanet_large |
| **TitaNet-L** | DepthSepConv1D+SE | 25.3M | 192 | 0.66-0.68 | N/A | N/A | Yes (NeMo) | Good | nvidia/speakerverification_en_titanet_large |
| **WeSpeaker ResNet34-LM** | ResNet34 | 6.3-6.7M | 256 | 0.72-1.05 | 6.84G | 0.032 | Yes (WeSpeaker) | Moderate | pyannote/wespeaker-voxceleb-resnet34-LM |
| **WeSpeaker ResNet293-LM** | Deep ResNet | ~25M+ | 256 | 0.45 | High | ~0.06* | Yes (WeSpeaker) | Good | Wespeaker/wespeaker-voxceleb-resnet293-LM |
| **ECAPA2** | Hybrid 1D+2D+fwSE | ~13M* | ~192* | ~0.41* | N/A | N/A | No (JIT only) | **Excellent** (designed for it) | Jenthe/ECAPA2 |
| **SPK2VEC** | Transformer (raw wave) | 2.5M | N/A | 1.22 | N/A | RTF 0.0002 | No | Moderate | Paper only |
| **ReDimNet2-B6** | Improved hybrid 1D/2D | 12.3M | N/A | 0.29 | 13G | N/A | Yes | Unknown | IDRnD/redimnet |
| **x-vector** (legacy) | TDNN | ~4.2M | 512 | 3.0+ | Low | Fast | Yes (SpeechBrain) | **Very poor** | speechbrain/spkrec-xvect-voxceleb |

*Values marked with ~ are estimated from paper context or parameter counts; * indicates values inferred but not explicitly confirmed in sources.*

**CPU RTF** = Real-Time Factor on single-thread CPU. Lower = faster. RTF of 0.013 means 1 second of audio processed in 13ms.

---

## Detailed Model Notes

### 1. CAM++ (Top Recommendation)

**Architecture:** Dense-TDNN backbone with context-aware masking and multi-granularity pooling.

**Why it stands out for VoxTerm:**
- **2.5x faster** than ECAPA-TDNN on CPU (RTF 0.013 vs 0.033)
- **51% fewer parameters** than ECAPA-TDNN (7.2M vs 14.7M)
- **56% fewer FLOPs** (1.72G vs 3.96G)
- **Better accuracy** (0.65-0.73% EER vs 0.80-0.89%)
- ONNX export available via WeSpeaker
- Used in production at scale (WeSpeaker toolkit, 3D-Speaker)

**Integration path:** WeSpeaker Python package or ONNX runtime. Available as `wespeaker` pip package with `wespeaker.load_model("cam++")`.

**Estimated model size:** ~28-30 MB (based on 7.2M float32 params)

**Concerns:** 512-dim embeddings are larger than current 192-dim. May need to verify stability on very short (<2s) noisy segments specifically.

**Key paper:** Wang et al., "CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking" (Interspeech 2023). https://arxiv.org/abs/2303.00332

**Pretrained:** https://github.com/wenet-e2e/wespeaker | https://github.com/modelscope/3D-Speaker

---

### 2. ECAPA2 (Best Robustness, Needs Evaluation)

**Architecture:** Hybrid 1D (TDNN-like) + 2D (ResNet-like) with frequency-wise Squeeze-and-Excitation (fwSE). Uses 16-head attention pooling.

**Why it stands out for VoxTerm:**
- **Explicitly designed for short-utterance robustness** -- training strategy includes short utterance sampling (10.9% relative improvement on short-duration test set Vox1-S)
- Injects global context via fwSE for spatially-invariant features
- State-of-the-art on VoxCeleb1-O and VoxCeleb1-E
- Significantly fewer parameters than competing SOTA models
- Frame/channel attention maps can double as weakly supervised VAD logits

**Integration path:** HuggingFace model `Jenthe/ECAPA2`. TorchScript JIT format (no ONNX yet). Supports CPU inference natively.

**Concerns:**
- CC-BY-NC-4.0 license (non-commercial use only)
- No ONNX export yet -- JIT only
- Exact parameter count and inference speed benchmarks not publicly documented
- Newer model, less production validation than CAM++ or WeSpeaker models

**Key paper:** Desplanques et al., "ECAPA2: A Hybrid Neural Network Architecture and Training Strategy for Robust Speaker Embeddings" (ASRU 2023). https://arxiv.org/abs/2401.08342

**Pretrained:** https://huggingface.co/Jenthe/ECAPA2

---

### 3. ERes2NetV2 (Best Short-Duration Design)

**Architecture:** Dual-branch feature fusion. Bottom-up dual-stage branch + bottleneck local feature fusion branch. Pruned from ERes2Net for efficiency.

**Why it stands out for VoxTerm:**
- **Specifically designed for short-duration verification** -- benchmarked at 2s and 3s segments
- EER at 2s: 1.48%, at 3s: 0.98%, full: 0.61%
- 192-dim embeddings (drop-in compatible with current system)
- Outperforms ECAPA-TDNN (0.61% vs 0.82%) with fewer params (17.8M vs 20.8M)

**Concerns:**
- 12.6G FLOPs is **3x higher** than ECAPA-TDNN -- may be too slow for real-time CPU
- Larger model may exceed latency budget on CPU
- Available through modelscope/3D-Speaker (less mainstream than WeSpeaker/SpeechBrain)

**Key paper:** Chen et al., "ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational Efficiency" (Interspeech 2024). https://arxiv.org/abs/2406.02167

**Pretrained:** https://github.com/modelscope/3D-Speaker

---

### 4. ReDimNet-B1/B2 (Lightest Option)

**Architecture:** Novel dimension-reshaping between 1D and 2D feature maps. Uses ConvNeXt/transformer 1D blocks + ResNet/ConvNeXt 2D blocks. Attentive statistics pooling.

**Why it stands out for VoxTerm:**
- **Extremely lightweight:** B1 is 2.2M params / 0.54 GMACs; B2 is 4.7M / 0.90 GMACs
- B1 achieves 0.73% EER (comparable to CAM++) at 1/3 the compute
- B2 achieves 0.52% EER -- better than ECAPA-TDNN at 1/4 the compute
- Designed as Pareto-optimal across model size vs accuracy tradeoffs
- Input: 72-dim Mel filterbank features (standard)

**Integration path:** GitHub IDRnD/redimnet. PyTorch weights available. Likely easy ONNX export.

**Concerns:**
- Relatively new architecture (2024) with limited production deployment reports
- No published data on robustness to short/noisy segments specifically
- Two-stage training (pretrain + large-margin fine-tuning) -- would need to use pretrained weights
- ReDimNet2 (improved version, 2025) has even better results but is very new

**Key paper:** "Reshape Dimensions Network for Speaker Recognition" (2024). https://arxiv.org/abs/2407.18223

**Pretrained:** https://github.com/IDRnD/redimnet

---

### 5. WeSpeaker ResNet34-LM (Production Default / pyannote 3.1)

**Architecture:** Standard ResNet34 with large-margin fine-tuning.

**Why it matters:**
- **Default embedding model in pyannote/speaker-diarization-3.1** -- the most widely used diarization pipeline
- Battle-tested in production
- 256-dim embeddings
- ONNX export available
- ~6.3-6.7M params, reasonable inference speed (RTF 0.032)

**Concerns:**
- EER 0.72-1.05% -- not top accuracy
- FLOPs are relatively high (6.84G) despite small param count
- No specific short-utterance optimizations
- pyannote 3.1 has moved to pure PyTorch (away from ONNX) for both segmentation and embedding

**Pretrained:** https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

---

### 6. TitaNet-S/M (NVIDIA NeMo Ecosystem)

**Architecture:** 1D depth-wise separable convolutions with Squeeze-and-Excitation layers. Channel attention statistics pooling. Scalable width (256/512/1024 channels).

**Why it matters:**
- Part of NVIDIA NeMo ecosystem (good documentation, active maintenance)
- TitaNet-S (6.4M) is competitive at 1.15% EER
- TitaNet-M (13.4M) hits 0.81% EER
- 192-dim embeddings (compatible)
- Trained on diverse data (VoxCeleb, Fisher, Switchboard, LibriSpeech, SRE)

**Concerns:**
- NeMo framework is heavy/complex to integrate standalone
- No published CPU RTF benchmarks
- TitaNet-S accuracy (1.15%) is notably worse than CAM++ (0.65%)
- Extracting the embedding model from NeMo for standalone use requires some effort

**Pretrained:** https://huggingface.co/nvidia/speakerverification_en_titanet_large

---

### 7. SPK2VEC (Ultralight Transformer)

**Architecture:** Lightweight transformer operating on raw waveform with trainable front-end.

**Why it matters:**
- Only 2.5M params, 47MB total size
- RTF 0.0002 -- essentially instant
- EER 1.22% on VoxCeleb1
- Multilingual training (English + Arabic)

**Concerns:**
- 1.22% EER is decent but not competitive with CAM++ or ReDimNet
- Academic paper only -- no readily available pretrained weights on HuggingFace/GitHub
- Raw waveform input may behave differently from Mel-based models
- Limited validation outside paper

**Key paper:** "SPK2VEC: Advanced speaker embeddings transformer model using hybrid supervised and self-supervised contrastive learning" (2026). https://www.sciencedirect.com/science/article/abs/pii/S1568494626003492

---

## Why ECAPA-TDNN Over-Segments

The current model (speechbrain/spkrec-ecapa-voxceleb, 192-dim) has known weaknesses relevant to real-time diarization:

1. **Short utterance sensitivity:** ECAPA-TDNN's attentive statistics pooling degrades on <2s segments. The model was trained/validated primarily on longer utterances (3-8s). When segments are 1-2s (common in real conversation), embeddings become noisy.

2. **No short-utterance training strategy:** Unlike ECAPA2 and ERes2NetV2, the SpeechBrain ECAPA-TDNN checkpoint does not use short-utterance augmentation during training.

3. **Single-scale features:** ECAPA-TDNN uses multi-layer feature aggregation but at a single frequency scale. Models like ERes2NetV2 (multi-scale) and CAM++ (multi-granularity pooling) capture more robust representations.

4. **Noise sensitivity of x-vector lineage:** While much improved over x-vectors, ECAPA-TDNN still inherits some instability in SNR < 15dB conditions (x-vectors drop to 55% P@1 at 15dB babble noise).

5. **Embedding dimension:** 192-dim may be too compact to capture speaker variability across emotional/prosodic changes. CAM++ uses 512-dim; pyannote default uses 256-dim.

---

## Recommendations (Ranked)

### Tier 1: Primary Candidates (Test These First)

**1. CAM++ via WeSpeaker** -- Best overall tradeoff
- Drop-in replacement feasible
- 2.5x faster CPU inference than current ECAPA-TDNN
- Better accuracy (0.65% vs 0.80% EER)
- Production-proven (WeSpeaker toolkit, used in VoxSRC challenges)
- ONNX export for maximum CPU speed
- Integration: `pip install wespeaker` or load ONNX directly
- Estimated model size: ~28MB

**2. ReDimNet-B2** -- Lightest with top accuracy
- 4.7M params, 0.90 GMACs -- fastest option with <0.6% EER
- Would need to validate short-utterance stability empirically
- Integration: PyTorch weights from GitHub, straightforward ONNX export
- Estimated model size: ~18MB

### Tier 2: Worth Testing If Tier 1 Still Over-Segments

**3. ECAPA2** -- Specifically designed for robustness
- Best theoretical choice for the over-segmentation problem
- Short-utterance training strategy directly addresses VoxTerm's issue
- BUT: CC-BY-NC license, no ONNX, less production validation
- Test if CAM++/ReDimNet still over-segment

**4. ERes2NetV2** -- Best short-duration benchmarks
- Proven at 2s segments (1.48% EER)
- BUT: 12.6G FLOPs may be too slow for real-time CPU
- Test if latency is acceptable on M-series

### Tier 3: Alternatives

**5. WeSpeaker ResNet34-LM** -- Safe, battle-tested
- pyannote 3.1 default -- if diarization quality is acceptable there, it will work here
- Not the best accuracy but well-validated in diarization specifically

**6. TitaNet-M** -- Good if already using NeMo
- Solid accuracy but NeMo integration overhead may not be worth it

---

## Suggested Testing Protocol

1. **Collect test clips:** Record 5-10 minutes of 2-speaker conversation with natural interruptions, varied emotion, and some background noise.
2. **Extract embeddings** from each model for 2s, 3s, and 5s segments.
3. **Measure:**
   - Cosine similarity of same-speaker segments (should be >0.7)
   - Cosine distance of different-speaker segments (should be >0.3 separation)
   - Standard deviation of same-speaker embeddings (lower = more stable)
   - Clustering result: does spectral clustering find the correct speaker count?
4. **Measure latency:** Time per embedding extraction on Apple Silicon CPU.
5. **Compare:** CAM++ vs ReDimNet-B2 vs current ECAPA-TDNN as baseline.

---

## Key Sources

- WeSpeaker toolkit: https://github.com/wenet-e2e/wespeaker
- 3D-Speaker (CAM++, ERes2Net): https://github.com/modelscope/3D-Speaker
- ReDimNet: https://github.com/IDRnD/redimnet
- ECAPA2: https://huggingface.co/Jenthe/ECAPA2
- CAM++ paper: https://arxiv.org/abs/2303.00332
- ERes2NetV2 paper: https://arxiv.org/abs/2406.02167
- ReDimNet paper: https://arxiv.org/abs/2407.18223
- ECAPA2 paper: https://arxiv.org/abs/2401.08342
- Comparison study (MDPI): https://www.mdpi.com/2076-3417/14/4/1329
- TitaNet paper: https://arxiv.org/abs/2110.04410
- SpeechBrain ECAPA-TDNN: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- pyannote diarization 3.1: https://huggingface.co/pyannote/speaker-diarization-3.1
- SPK2VEC paper: https://www.sciencedirect.com/science/article/abs/pii/S1568494626003492
- VoxCeleb retrospective: https://robots.ox.ac.uk/~vgg/publications/2024/Huh24/huh24.pdf
