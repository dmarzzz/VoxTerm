# VAD and Overlapped Speech Detection Research

Research date: 2026-03-21

## Context

VoxTerm currently uses RMS energy thresholding (threshold=0.012) for VAD. This causes
over-segmentation: noise, breath, and background sounds get embedded as speaker segments,
creating false speaker clusters. This document evaluates neural VAD systems and overlap
detection approaches as replacements.

Constraints:
- Real-time (< 10ms per frame)
- Offline/local, no cloud APIs
- Python (PyTorch or ONNX on CPU)
- macOS Apple Silicon

---

## 1. VAD Systems Comparison

### 1.1 Silero VAD

| Property | Value |
|---|---|
| Source | [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| Architecture | CNN + LSTM (quantized), proprietary design |
| Model size | ~1.8 MB (JIT) / ~2.2 MB (ONNX) |
| Parameters | Not publicly disclosed (quantized model) |
| Sample rates | 8 kHz, 16 kHz |
| Frame sizes | 512, 1024, 1536 samples @16kHz (32ms, 64ms, 96ms) |
| Latency | < 1ms per chunk on single CPU thread; ONNX can be 4-5x faster |
| License | MIT |
| Overlap detection | No |
| Version | v6.2.1 (Feb 2026) |

**Accuracy (from official wiki, multi-domain validation set of 17 hours):**

| Metric | Silero v6 | Silero v5 | WebRTC VAD | Commercial VAD |
|---|---|---|---|---|
| ROC-AUC (31.25ms segments) | **0.97** | 0.96 | 0.73 | 0.93 |
| Speech accuracy | 0.92 | 0.91 | -- | 0.87 |

Tested across: ESC-50, AliMeeting, Earnings calls, MSDWild, AISHELL-4, VoxConverse,
Libriparty (combined 17h multi-domain set).

**API (3 lines to integrate):**
```python
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
model = load_silero_vad()
speech_timestamps = get_speech_timestamps(wav_tensor, model, return_seconds=True)
```

**Dependencies:** PyTorch >= 1.12.0, torchaudio >= 0.12.0 (I/O only), optionally onnxruntime >= 1.16.1.

**Verdict:** Best overall choice for VoxTerm. Extremely simple API, sub-millisecond latency,
high accuracy, MIT license, and we already have PyTorch as a dependency. The 0.97 AUC on
multi-domain data is excellent. Trained on 6000+ languages. Does not detect overlap.

---

### 1.2 WebRTC VAD (py-webrtcvad)

| Property | Value |
|---|---|
| Source | [github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad) |
| Architecture | Gaussian Mixture Model (GMM), traditional signal processing |
| Model size | ~158 KB (C library) |
| Parameters | N/A (not a neural model) |
| Sample rates | 8, 16, 32, 48 kHz |
| Frame sizes | 10, 20, or 30 ms |
| Latency | << 1ms per chunk |
| License | BSD (WebRTC license) |
| Overlap detection | No |

**Accuracy:**

| Metric | WebRTC VAD | Silero VAD v6 |
|---|---|---|
| ROC-AUC | 0.73 | 0.97 |
| TPR @ 5% FPR | ~50% | ~87.7% |

**API:**
```python
import webrtcvad
vad = webrtcvad.Vad(aggressiveness)  # 0-3
is_speech = vad.is_speech(frame_bytes, sample_rate)
```

**Dependencies:** Only C extension (pip install webrtcvad). No ML framework needed.

**Verdict:** Extremely lightweight and fast, but accuracy is poor -- ROC-AUC of 0.73 means
it frequently confuses speech with noise (exactly the problem we already have with RMS).
It uses energy/spectral features similar to our current approach. Not recommended as a
replacement for our RMS thresholding since it would not substantially improve the
false-cluster problem.

---

### 1.3 TEN VAD

| Property | Value |
|---|---|
| Source | [github.com/TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) |
| Architecture | Undisclosed lightweight neural network |
| Model size | 306 KB (Linux .so), 731 KB (macOS .framework) |
| Sample rate | 16 kHz only |
| Frame sizes | 160 or 256 samples (10ms or 16ms) |
| Latency | RTF 0.005-0.057 depending on platform; ~0.016 on Apple M1 |
| License | Apache 2.0 (with additional conditions); some derived code BSD |
| Overlap detection | No |

**Accuracy (from official benchmarks):**
Claims "superior precision-recall compared to both WebRTC VAD and Silero VAD" on
LibriSpeech, GigaSpeech, and DNS Challenge datasets. However, independent benchmarks
(FireRedVAD comparison on FLEURS-VAD-102) show:

| Metric | TEN VAD | Silero VAD | WebRTC |
|---|---|---|---|
| F1 | 95.19% | 95.95% | 52.30% |
| False Alarm Rate | 15.47% | 9.41% | 2.83% |
| Miss Rate | 2.95% | 3.95% | 64.15% |

**API:**
```python
from ten_vad import TenVad
vad = TenVad()
result = vad.process(audio_frame)
```

**Dependencies:** NumPy, SciPy. Optional: onnxruntime. Python bindings currently
optimized for Linux x64; macOS support via C library only.

**Verdict:** Extremely small and fast. Lower latency than Silero for speech-to-non-speech
transitions (important for conversational AI). However: (1) Python bindings are
Linux-only, macOS requires C integration, (2) slightly lower F1 than Silero on independent
benchmarks, (3) higher false alarm rate than Silero. Would require extra work to integrate
on macOS with Python. Not the best fit for VoxTerm currently.

---

### 1.4 FireRedVAD

| Property | Value |
|---|---|
| Source | [github.com/FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) |
| Architecture | DFSMN (Deep Feed-forward Sequential Memory Network) |
| Model size | 2.2 MB (float32), ~588K parameters |
| Sample rate | 16 kHz |
| Modes | Non-streaming VAD, Streaming VAD, Audio Event Detection |
| License | Apache 2.0 |
| Overlap detection | No (but has AED for speech/singing/music classification) |

**Accuracy (FLEURS-VAD-102, 102 languages, 9443 files):**

| Metric | FireRedVAD | Silero VAD | TEN VAD | FunASR VAD | WebRTC |
|---|---|---|---|---|---|
| F1 | **97.57%** | 95.95% | 95.19% | 90.91% | 52.30% |
| AUC-ROC | **99.60%** | 97.99% | 97.81% | -- | -- |
| False Alarm Rate | **2.69%** | 9.41% | 15.47% | 44.03% | 2.83% |
| Miss Rate | 3.62% | 3.95% | 2.95% | 0.42% | 64.15% |

**API:**
```python
# pip install fireredvad
from fireredvad import FireRedVAD
vad = FireRedVAD(mode="streaming")  # or "non-streaming"
results = vad.process(audio_file)
```

**Dependencies:** PyTorch, ONNX Runtime, NumPy, SciPy. Python 3.10+.

**Verdict:** Highest accuracy of any open-source VAD tested (97.57% F1, 99.60% AUC-ROC).
Very recent release (March 2026). Supports streaming mode. Same model size as Silero.
However: (1) very new project, less battle-tested, (2) requires Python 3.10+,
(3) latency per frame not explicitly benchmarked in their docs. The DFSMN architecture
is efficient and should be real-time capable. Worth evaluating alongside Silero.

---

### 1.5 pyannote VAD (via segmentation-3.0)

| Property | Value |
|---|---|
| Source | [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| Architecture | PyanNet: SincNet + Bi-LSTM + Feed-forward (Powerset encoding) |
| Model size | ~5 MB (segmentation-3.0); ~17 MB (segmentation v1) |
| Parameters | ~1M estimated (based on architecture: SincNet + 2x BiLSTM-128 + 2x FF-128) |
| Input | 10 seconds mono audio @ 16 kHz |
| Output | 7-class matrix: non-speech, 3 speakers, 3 speaker-pair overlaps |
| License | MIT |
| Overlap detection | **Yes** (built-in, up to 2 simultaneous speakers per frame) |

**Accuracy:** State-of-the-art for speaker diarization across AISHELL-4, AliMeeting, AMI,
DIHARD, VoxConverse, etc. DER typically 15-25% on challenging multi-speaker datasets.
Specific VAD-only precision/recall not separately published.

**CPU Performance:** Real-time factor 0.5-1.0 on CPU for full diarization pipeline
(30-60 minutes to process 1 hour). ONNX conversion can provide ~2x speedup.
**Not suitable for real-time streaming on CPU.**

**API:**
```python
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

model = Model.from_pretrained("pyannote/segmentation-3.0",
                               use_auth_token="HF_TOKEN")
pipeline = VoiceActivityDetection(segmentation=model)
pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
vad = pipeline("audio.wav")
```

**Dependencies:** pyannote.audio 3.0+, PyTorch, HuggingFace account + token (even though
MIT licensed, requires accepting terms on HF).

**Verdict:** The only system that provides both VAD and overlap detection in a single model.
However, it is too slow for real-time streaming on CPU (RTF 0.5-1.0). Best used for
offline/batch processing. The HuggingFace token requirement adds friction. Not suitable
as a drop-in replacement for our real-time VAD, but could be used for offline
post-processing or overlap-aware re-segmentation.

---

### 1.6 Cobra VAD (Picovoice)

| Property | Value |
|---|---|
| Source | [picovoice.ai/platform/cobra](https://picovoice.ai/platform/cobra/) |
| Architecture | Proprietary lightweight DNN |
| Latency | RTF as low as 0.005 (AMD CPU) |
| License | Proprietary (requires API key, free tier available) |
| Overlap detection | No |

**Accuracy:** Claims highest AUC of all tested VADs. ~90% TPR at 5% FPR (vs Silero ~87.7%,
WebRTC ~50%).

**Verdict:** Not suitable -- proprietary, requires API key, vendor lock-in. VoxTerm is
fully offline/local. Mentioned only for benchmark context.

---

## 2. VAD Recommendation Summary

| System | AUC-ROC | F1 | Latency/frame | Size | Overlap | Real-time? | Recommended? |
|---|---|---|---|---|---|---|---|
| **Silero VAD v6** | 0.97 | ~96% | < 1ms | 1.8 MB | No | Yes | **Yes (primary)** |
| **FireRedVAD** | 0.996 | 97.6% | TBD | 2.2 MB | No | Likely | **Yes (evaluate)** |
| TEN VAD | ~0.978 | 95.2% | < 1ms | 731 KB | No | Yes | No (macOS Python issue) |
| pyannote seg-3.0 | -- | -- | 500-1000ms/s | 5 MB | **Yes** | **No** | No (too slow) |
| WebRTC VAD | 0.73 | 52% | << 1ms | 158 KB | No | Yes | No (poor accuracy) |
| Cobra VAD | highest | ~90% TPR | < 1ms | -- | No | Yes | No (proprietary) |

**Primary recommendation: Silero VAD v6.**
- Drop-in replacement for RMS thresholding
- Sub-millisecond latency, well within our < 10ms budget
- 0.97 AUC-ROC, dramatically better than RMS energy
- MIT license, already uses PyTorch (which VoxTerm has)
- Extremely simple 3-line API
- Battle-tested: used in WhisperX, LiveKit, and many production systems

**Secondary evaluation: FireRedVAD.**
- Even higher accuracy (97.57% F1 vs ~96%), but very new (March 2026)
- Streaming mode available
- Worth benchmarking latency on Apple Silicon before committing

---

## 3. Overlapped Speech Detection

### 3.1 Approaches

There are two fundamentally different approaches to handling overlapped speech:

**A. Binary Overlap Detection (OSD)**
- Detects *whether* overlap is occurring (yes/no per frame)
- Does not separate the speakers
- Used to: flag overlapped regions for exclusion from embedding, or trigger
  re-segmentation
- Models: pyannote segmentation-3.0, dedicated OSD classifiers

**B. Speech Separation**
- Separates the mixed signal into individual speaker streams
- Each stream can then be independently embedded and clustered
- Models: Conv-TasNet (5.1M params), SepFormer (6.6M-26M params)
- Much more computationally expensive

### 3.2 Binary Overlap Detection Models

#### pyannote segmentation-3.0 (OSD mode)

The same model used for VAD can detect overlap. It outputs a 7-class powerset:
{non-speech, spk1, spk2, spk3, spk1+spk2, spk1+spk3, spk2+spk3}.

By taking the second-maximum over speaker activations per frame, overlap regions
are identified.

```python
from pyannote.audio.pipelines import OverlappedSpeechDetection
pipeline = OverlappedSpeechDetection(segmentation=model)
pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
osd = pipeline("audio.wav")
# Returns Timeline of overlapped regions
```

- **Accuracy:** State-of-the-art for OSD; trained on 9 major diarization datasets
- **Limitation:** Too slow for real-time CPU (RTF 0.5-1.0)
- **Use case for VoxTerm:** Offline post-processing pass to identify and handle
  overlap regions after initial segmentation

#### Dedicated lightweight OSD

No widely-available standalone lightweight OSD model exists that runs in real-time
on CPU. Most OSD research uses full segmentation models (like pyannote) or
SSL-based encoders (WavLM) that are too large for real-time CPU use.

### 3.3 Speech Separation Models

| Model | Parameters | Real-time CPU? | Quality |
|---|---|---|---|
| Conv-TasNet | 5.1M | Yes (low latency, efficient) | Good (SI-SNRi ~15 dB) |
| SepFormer (small) | 6.6M | Marginal (153ms for 10s on CPU) | Better |
| SepFormer (SOTA) | 26M | No | Best |
| pyannote speech-separation-ami-1.0 | -- | No (GPU recommended) | Good |

Conv-TasNet is the most feasible for CPU real-time separation, but adds significant
complexity to the pipeline and requires knowing the number of speakers in advance
(typically assumes 2).

### 3.4 Practical Approach for VoxTerm

Given our constraints (real-time, CPU, Apple Silicon), the recommended approach is:

1. **Use Silero VAD** to replace RMS thresholding for speech/non-speech detection.
   This alone should dramatically reduce false speaker clusters caused by noise.

2. **Do NOT attempt real-time overlap detection or separation.** The computational
   cost is too high for CPU real-time processing, and the complexity is not justified
   for the current use case.

3. **If overlap handling is needed later**, consider:
   - **Offline OSD pass:** After recording, run pyannote segmentation-3.0 to identify
     overlapped regions. Exclude those regions from speaker clustering (do not embed
     overlapped frames). This is the simplest and most effective approach.
   - **Overlap-aware embedding:** Use context-aware masking (CAM) or similar techniques
     to make speaker embeddings more robust to partial overlap. Research shows that
     combining embeddings from both noisy and enhanced speech improves verification
     accuracy.
   - **Speech separation as post-processing:** Run Conv-TasNet on identified overlap
     regions only (not the entire stream), then re-embed the separated outputs.

---

## 4. Speech Enhancement for Speaker Embeddings

An alternative to overlap detection is improving the robustness of speaker embeddings
themselves:

- **DNN-based embeddings (x-vectors, ECAPA-TDNN)** are inherently more robust to noise
  and reverberation than i-vectors
- **Context-Aware Masking (CAM)** enables the embedding network to focus on the target
  speaker and blur unrelated noise
- **Dual-embedding fusion:** Extract embeddings from both the original (noisy) audio
  and a speech-enhanced version, then combine them for more robust speaker verification
- **Pre-trained speech enhancement** models (e.g., using wav2vec 2.0, HuBERT features)
  can clean audio before embedding extraction

For VoxTerm's immediate needs, the simplest improvement is replacing RMS VAD with
Silero VAD to ensure only genuine speech frames are passed to ECAPA-TDNN for embedding.
This avoids embedding noise/breath as speaker characteristics.

---

## 5. Integration Plan for VoxTerm

### Phase 1: Replace RMS VAD with Silero VAD (immediate)

**What changes:**
- Remove RMS energy thresholding (threshold=0.012)
- Load Silero VAD model at startup (~1.8 MB, one-time cost)
- For each audio chunk, call `model(chunk, sample_rate)` to get speech probability
- Only pass chunks with speech probability > threshold (default 0.5) to the
  diarization/embedding pipeline

**Expected impact:**
- Eliminate false clusters from noise/breath/background sounds
- Reduce over-segmentation by ~50-80% (based on AUC improvement from 0.73-class
  to 0.97-class detector)
- Add < 1ms latency per frame (well within 10ms budget)

**New dependency:** `pip install silero-vad` (pulls no new deps beyond PyTorch)

### Phase 2: Offline overlap handling (future, if needed)

- After recording completes, run pyannote segmentation-3.0 on the full audio
- Identify overlapped regions
- Exclude overlapped frames from speaker clustering
- Optionally re-segment overlapped regions using separated speaker signals

---

## 6. Key References and Sources

### VAD Systems
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad) -- MIT license, primary recommendation
- [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [py-webrtcvad GitHub](https://github.com/wiseman/py-webrtcvad) -- WebRTC VAD Python bindings
- [TEN VAD GitHub](https://github.com/TEN-framework/ten-vad) -- Apache 2.0, lightweight
- [TEN VAD HuggingFace](https://huggingface.co/TEN-framework/ten-vad)
- [FireRedVAD GitHub](https://github.com/FireRedTeam/FireRedVAD) -- Apache 2.0, SOTA accuracy
- [FireRedVAD HuggingFace](https://huggingface.co/FireRedTeam/FireRedVAD)
- [Picovoice VAD Comparison 2025](https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/)
- [Picovoice VAD Comparison 2026](https://picovoice.ai/blog/best-voice-activity-detection-vad/)

### Overlap and Segmentation
- [pyannote/segmentation-3.0 HuggingFace](https://huggingface.co/pyannote/segmentation-3.0) -- MIT, VAD+OSD
- [pyannote/overlapped-speech-detection HuggingFace](https://huggingface.co/pyannote/overlapped-speech-detection)
- [pyannote-audio GitHub](https://github.com/pyannote/pyannote-audio) -- full diarization toolkit
- [Powerset diarization paper (Plaquet & Bredin, INTERSPEECH 2023)](https://huggingface.co/pyannote/segmentation-3.0)

### Speech Separation
- [Conv-TasNet paper](https://arxiv.org/abs/1809.07454) -- 5.1M params, real-time capable
- [SepFormer](https://www.emergentmind.com/topics/sepformer) -- dual-path transformer
- [pyannote speech-separation-ami-1.0](https://huggingface.co/pyannote/speech-separation-ami-1.0)

### Speaker Embedding Robustness
- [X-Vectors: Robust DNN Embeddings](https://www.researchgate.net/publication/327812023_X-Vectors_Robust_DNN_Embeddings_for_Speaker_Recognition)
- [Context-Aware Masking for Speaker Verification](https://cs.nju.edu.cn/lwj/paper/ICASSP21_CAM.pdf)
- [Noise-Robust Speaker Verification Framework (2025)](https://arxiv.org/abs/2508.18913)

### Surveys and Benchmarks
- [Comprehensive VAD Survey (2024)](https://www.researchgate.net/publication/384131916)
- [Lightweight Real-Time VAD (MagicNet)](https://arxiv.org/abs/2405.16797)
- [One Voice Detector to Rule Them All (The Gradient)](https://thegradient.pub/one-voice-detector-to-rule-them-all/)
- [DIART: Optimized Real-Time Diarization](https://arxiv.org/html/2408.02341v1)
- [Near-Real-Time Speaker Diarization on CoreML](https://inference.plus/p/low-latency-speaker-diarization-on)
