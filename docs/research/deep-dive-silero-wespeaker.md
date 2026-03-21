# Deep Dive: Silero VAD & WeSpeaker/CAM++

Research for replacing VoxTerm's RMS-based VAD with Silero VAD and potentially
upgrading from SpeechBrain ECAPA-TDNN to WeSpeaker CAM++.

---

## 1. Silero VAD

### 1.1 Overview

| Property | Value |
|---|---|
| Version | 6.2.1 (Feb 2026) |
| Model size | ~2 MB (JIT), ~1.8 MB (ONNX) |
| AUC | 0.97 |
| Latency | < 1 ms per chunk on single CPU thread |
| Languages | 6000+ (language-agnostic) |
| License | MIT |
| Sample rates | 8000 Hz, 16000 Hz |
| Runtime | PyTorch JIT or ONNX Runtime |

### 1.2 Installation

```bash
pip install silero-vad
```

Dependencies:
- `torch>=1.12.0`
- `torchaudio>=0.12.0` (for audio I/O only)
- `onnxruntime>=1.16.1` (optional, for ONNX mode)

System requirements: Python 3.8+, 1 GB+ RAM, CPU with AVX/AVX2/AVX-512/AMX.

### 1.3 Model Loading

**Preferred (pip package):**
```python
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

model = load_silero_vad()          # PyTorch JIT (default)
model = load_silero_vad(onnx=True) # ONNX Runtime
```

**Alternative (torch.hub):**
```python
import torch
torch.set_num_threads(1)

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
```

### 1.4 Input Requirements

| Parameter | Requirement |
|---|---|
| Sample rate | 8000 Hz or 16000 Hz only |
| Format | `torch.Tensor`, float32, mono |
| Chunk size (16 kHz) | **512 samples** (32 ms) |
| Chunk size (8 kHz) | **256 samples** (32 ms) |
| Batch dimension | `(batch_size, num_samples)` |

**Converting int16 mic input to float32:**
```python
import numpy as np

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound
```

### 1.5 Frame-Level Probability API (Direct Model Call)

The model is callable and returns a speech probability per frame:

```python
# model(audio_chunk, sample_rate) -> speech probability [0.0, 1.0]
prob = model(torch.from_numpy(audio_float32), 16000).item()
```

**Input:** `(batch, 512)` tensor at 16 kHz or `(batch, 256)` at 8 kHz.
**Output:** Speech probability float in [0, 1].

The model is **stateful** -- it maintains internal hidden states `(2, batch_size, 128)`
plus a context buffer. Call `model.reset_states()` between independent audio streams.

### 1.6 Batch API: `get_speech_timestamps()`

Process a complete audio file and get speech segment boundaries:

```python
@torch.no_grad()
def get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    visualize_probs: bool = False,
    progress_tracking_callback: Callable[[float], None] = None,
    neg_threshold: float = None,
    window_size_samples: int = 512,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> list[dict]:
```

**Returns:** `[{'start': 10240, 'end': 52480}, ...]` (samples) or seconds if
`return_seconds=True`.

**Parameter reference:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | 0.5 | Probability above this = speech. Tune per dataset. |
| `neg_threshold` | `max(threshold - 0.15, 0.01)` | Below this = non-speech (hysteresis). |
| `sampling_rate` | 16000 | Must be 8000 or 16000. |
| `min_speech_duration_ms` | 250 | Discard speech segments shorter than this. |
| `max_speech_duration_s` | inf | Force-split segments longer than this. |
| `min_silence_duration_ms` | 100 | Silence gap needed to split segments. |
| `speech_pad_ms` | 30 | Pad each segment by this on both sides. |
| `return_seconds` | False | Return timestamps in seconds vs samples. |
| `window_size_samples` | 512 | Chunk size for internal processing. |
| `visualize_probs` | False | Plot probability curve (matplotlib). |

### 1.7 Streaming API: `VADIterator`

For real-time frame-by-frame streaming with speech start/end events:

```python
class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):

    def reset_states(self):
        """Reset model state AND iterator state. Call between audio streams."""

    @torch.no_grad()
    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """Process one audio chunk.

        Returns:
            dict with 'start' key  -- when speech begins
            dict with 'end' key    -- when speech ends
            None                   -- no state change
        """
```

**Usage example:**
```python
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad()
vad_iterator = VADIterator(model, sampling_rate=16000)

WINDOW = 512  # 32 ms at 16 kHz

for i in range(0, len(wav), WINDOW):
    chunk = wav[i : i + WINDOW]
    if len(chunk) < WINDOW:
        break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict)  # {'start': 1.024} or {'end': 3.456}

vad_iterator.reset_states()
```

### 1.8 Real-Time Streaming with PyAudio

Complete working example for microphone input:

```python
import numpy as np
import torch
import pyaudio

torch.set_num_threads(1)

model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, VADIterator, _) = utils

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
NUM_SAMPLES = 512  # 32 ms chunks

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, channels=CHANNELS,
    rate=SAMPLE_RATE, input=True,
    frames_per_buffer=NUM_SAMPLES
)

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    return sound.squeeze()

while True:
    audio_chunk = stream.read(NUM_SAMPLES)
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    confidence = model(torch.from_numpy(audio_float32), 16000).item()
    # confidence is in [0.0, 1.0]; > 0.5 typically means speech
```

### 1.9 ONNX Runtime Details

The `OnnxWrapper` class used internally:

```python
class OnnxWrapper:
    def __init__(self, path, force_onnx_cpu=False):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(path, ...)
        self.reset_states()

    def __call__(self, x, sr: int):
        # Validates: x.shape[-1] must be 512 (16kHz) or 256 (8kHz)
        # Prepends context (64 samples at 16kHz, 32 at 8kHz)
        # ONNX inputs: {'input': audio, 'state': hidden, 'sr': sample_rate}
        # Returns: speech probability tensor
```

ONNX models:
- `silero_vad.onnx` -- opset 16 (default), supports 8 kHz + 16 kHz
- `silero_vad_16k_op15.onnx` -- opset 15, 16 kHz only

### 1.10 Thread Safety & Concurrency

**The model is NOT thread-safe.** It maintains internal mutable state (`_state`,
`_context`, `_last_sr`, `_last_batch_size`).

Recommended approaches for concurrent use:
1. **One model instance per thread/stream** (simplest, recommended for VoxTerm)
2. Per-request initialization (higher overhead)
3. ProcessPoolExecutor with separate model instances
4. For ONNX: pass state tensors explicitly between calls (state is returned as output)

Per the maintainer: "The only really correct way is to have a separate
process/worker and to pass state to this worker back-and-forth."

**For VoxTerm:** Since we have a single audio stream, one model instance on the
audio worker thread is sufficient. Call `model.reset_states()` if the stream is
restarted.

### 1.11 Memory & Initialization

| Metric | Value |
|---|---|
| Model file | ~2 MB |
| RAM at load | ~50-100 MB (PyTorch runtime overhead) |
| RAM per inference | Negligible (in-place state update) |
| Load time (JIT) | ~1-2 s (first load includes torch JIT compile) |
| Load time (ONNX) | ~0.2-0.5 s |
| Inference time | < 1 ms per 32 ms chunk |

### 1.12 Tuning Recommendations for VoxTerm

For a transcription TUI processing live microphone input:

```python
vad_iterator = VADIterator(
    model,
    threshold=0.5,            # Start here; lower to 0.3-0.4 for quiet speakers
    sampling_rate=16000,
    min_silence_duration_ms=300,  # Longer than default -- avoids splitting
                                  # mid-sentence pauses for transcription
    speech_pad_ms=50,          # Slightly more padding to avoid clipping word edges
)
```

For `get_speech_timestamps` on buffered audio:
- `min_speech_duration_ms=500` -- avoids sending tiny fragments to ASR
- `min_silence_duration_ms=500` -- natural sentence-level segmentation
- `speech_pad_ms=100` -- generous padding for ASR accuracy

---

## 2. WeSpeaker / CAM++

### 2.1 Overview

WeSpeaker is a speaker embedding learning toolkit from WeNet. It supports
multiple architectures including ResNet, ECAPA-TDNN, and CAM++.

| Property | Value |
|---|---|
| Repository | github.com/wenet-e2e/wespeaker |
| License | Apache 2.0 |
| Python package | `pip install git+https://github.com/wenet-e2e/wespeaker.git` |
| Runtime package | `pip install wespeakerruntime` (ONNX only) |
| Default sample rate | 16000 Hz |
| Feature extraction | 80-dim Fbank, 25 ms window, 10 ms shift |

### 2.2 CAM++ Model Specifications

From the original paper (Wang et al., 2023, arXiv:2303.00332):

| Property | CAM++ | ECAPA-TDNN (512) | ResNet34 |
|---|---|---|---|
| Parameters | **7.18M** | 14.66M | 6.70M |
| FLOPs | **1.15-1.72G** | 3.96G | 6.84G |
| Embedding dim | **512** (default) | 192 or 512 | 256 |
| RTF (CPU, 1 thread) | **0.013** | 0.033 | 0.032 |
| VoxCeleb1-O EER | **0.73%** | 0.89% | 0.97% |
| VoxCeleb1-E EER | 0.89% | 1.07% | 1.03% |
| VoxCeleb1-H EER | 1.76% | 1.98% | 1.88% |

Key advantages of CAM++ over ECAPA-TDNN:
- **51% fewer parameters** (7.18M vs 14.66M)
- **2.5x faster inference** (RTF 0.013 vs 0.033)
- **18% lower EER** on VoxCeleb1-O (0.73% vs 0.89%)

Architecture: D-TDNN backbone with 3 blocks (12, 24, 16 layers), 2D conv
front-end with 4 residual blocks, context-aware multi-granularity masking/pooling.

### 2.3 WeSpeaker VoxCeleb v2 Benchmark (All Models)

Best results with large-margin fine-tuning + AS-Norm + QMF:

| Model | Params | FLOPs | vox1-O EER | vox1-E EER | vox1-H EER |
|---|---|---|---|---|---|
| XVEC-TSTP-emb512 | 4.61M | 0.53G | -- | -- | -- |
| CAM++ | 7.18M | 1.15G | 0.659% | 0.803% | 1.569% |
| ResNet34 | 6.70M | 6.84G | -- | -- | -- |
| ResNet152 | 25.62M | 17.28G | -- | -- | -- |
| ResNet221 | 23.86M | 11.64G | -- | -- | -- |
| ResNet293 (emb256) | 26.30M | 14.77G | 0.425% | 0.641% | 1.146% |
| ECAPA-512 | 6.19M | 1.04G | -- | -- | -- |
| ECAPA-1024 | 14.65M | 2.65G | -- | -- | -- |
| SimAM-ResNet100 (VoxBlink2) | -- | -- | 0.202% | 0.421% | 0.795% |

### 2.4 Available Pretrained Models in Hub

WeSpeaker's Python package Hub provides these named models:

| Key | Archive | Description |
|---|---|---|
| `"chinese"` | cnceleb_resnet34.tar.gz | ResNet34 on CNCeleb |
| `"english"` | voxceleb_resnet221_LM.tar.gz | ResNet221-LM on VoxCeleb |
| `"campplus"` | campplus_cn_common_200k.tar.gz | CAM++ on 200K Chinese speakers |
| `"eres2net"` | eres2net_cn_commom_200k.tar.gz | ERes2Net on 200K Chinese |
| `"vblinkp"` | voxblink2_samresnet34.zip | SimAM-ResNet34 (VoxBlink2 pretrain) |
| `"vblinkf"` | voxblink2_samresnet34_ft.zip | Same + VoxCeleb2 finetune |
| `"w2vbert2_mfa"` | w2v-BERT2-MFA-LM | W2V-BERT2.0 with MFA |

Additional models available on ModelScope/HuggingFace (not in default Hub):
- CAM++_LM (VoxCeleb, with large-margin fine-tuning)
- ECAPA-TDNN 512/1024 variants
- ResNet34/152/221/293 variants

**Important:** The `"campplus"` Hub model is trained on **Chinese** data (200K
speakers from cn_common). For English/multilingual use, download the VoxCeleb-
trained CAM++ model from ModelScope or use `load_model_local()` with a custom
model directory.

### 2.5 Python API

```python
import wespeaker

# Load a named model (downloads automatically)
model = wespeaker.load_model('english')           # ResNet221-LM
model = wespeaker.load_model('campplus')           # CAM++ (Chinese)
model = wespeaker.load_model_local('/path/to/dir') # Custom model dir
                                                    # (needs avg_model.pt + config.yaml)

# Configure
model.set_device('cpu')       # 'cpu', 'cuda', 'cuda:0', 'mps'
model.set_resample_rate(16000)
model.set_vad(True)           # Internal Silero VAD for speech detection

# Extract embedding (returns numpy array)
embedding = model.extract_embedding('audio.wav')

# Batch extraction (Kaldi wav.scp format)
utt_names, embeddings = model.extract_embedding_list('wav.scp')

# Speaker similarity (returns float in [0, 1])
score = model.compute_similarity('audio1.wav', 'audio2.wav')

# Speaker diarization
result = model.diarize('audio.wav')

# Speaker registration & recognition
model.register('alice', 'alice_enrollment.wav')
model.register('bob', 'bob_enrollment.wav')
result = model.recognize('unknown.wav')  # {'name': 'alice', 'confidence': 0.87}
```

### 2.6 Speaker Class Internals

Key implementation details from `wespeaker/cli/speaker.py`:

```python
class Speaker:
    def __init__(self, model_dir: str):
        self.model = load_model_pt(model_dir)  # Loads avg_model.pt
        self.vad = load_silero_vad()            # Built-in Silero VAD!
        self.table = {}                         # Speaker registry
        self.resample_rate = 16000
        self.apply_vad = False
        self.device = torch.device('cpu')

        # Diarization defaults
        self.diar_min_duration = 0.255   # seconds
        self.diar_window_secs = 1.5
        self.diar_period_secs = 0.75
        self.diar_frame_shift = 10       # ms
        self.diar_batch_size = 32
        self.diar_subseg_cmn = True
```

Note: WeSpeaker already uses Silero VAD internally for optional speech detection
before embedding extraction.

### 2.7 CAM++ Architecture Source (from WeSpeaker)

From `wespeaker/models/campplus.py`:

```python
class CAMPPlus(nn.Module):
    # Default configuration:
    #   embed_dim=512        -- output embedding dimension
    #   growth_rate=32       -- channel growth in dense blocks
    #   init_channels=128    -- initial channel count
    #   pooling_func='TSTP'  -- temporal statistics pooling

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, F) => (B, F, T)
        x = self.head(x)         # FCM front-end (ResNet-style 2D conv)
        x = self.xvector(x)      # D-TDNN + CAM blocks => embedding
        return x                  # (B, embed_dim)
```

Input: `(batch, time_steps, 80)` -- 80-dim Fbank features.
Output: `(batch, 512)` -- L2-normalized speaker embedding.

### 2.8 ONNX Inference Path

**Option A: wespeakerruntime (lightweight, ONNX-only)**
```bash
pip install wespeakerruntime
```

```python
import wespeakerruntime as wespeaker

speaker = wespeaker.Speaker(lang='en')  # Downloads ONNX model
# or: speaker = wespeaker.Speaker(onnx_path='/path/to/model.onnx')

embedding = speaker.extract_embedding('audio.wav')
score = speaker.compute_cosine_score(emb1, emb2)
```

Configuration: resample_rate=16000, 80 Mel bins, 25 ms frame, 10 ms shift, CMN.

**Option B: WeSpeaker CLI with ONNX**
```bash
python wespeaker/bin/infer_onnx.py \
    --onnx_path /path/to/model.onnx \
    --wav_path audio.wav
```

**Option C: Export your own ONNX model**
WeSpeaker supports JIT and ONNX export from checkpoint models. The exported
`.onnx` files can be loaded directly in ONNX Runtime.

### 2.9 Input Requirements

| Property | Value |
|---|---|
| Sample rate | 16000 Hz (resampled internally) |
| Format | WAV, single-channel (mono) |
| Features | 80-dim Fbank (computed internally) |
| Window | 25 ms frame, 10 ms shift |
| Min duration | ~0.5 s practical minimum (shorter is unreliable) |
| Recommended duration | > 3 s for best accuracy |
| LM models | Optimized for > 3 s audio segments |

### 2.10 Thread Safety

WeSpeaker's `Speaker` class is **NOT thread-safe** -- it holds mutable model
state, speaker registry, and VAD state. For concurrent use:
- Use separate `Speaker` instances per thread
- Or protect access with a mutex

For VoxTerm (single audio stream): one instance on the audio worker thread is
sufficient.

### 2.11 ONNX Model Size Estimates

| Model | Checkpoint (.pt) | ONNX (.onnx) est. |
|---|---|---|
| CAM++ (7.18M params) | ~28 MB | ~28 MB |
| ECAPA-TDNN-512 (6.19M) | ~25 MB | ~25 MB |
| ResNet34 (6.70M) | ~27 MB | ~27 MB |
| ResNet221 (23.86M) | ~95 MB | ~95 MB |

### 2.12 Comparison: Current VoxTerm vs CAM++ Upgrade

| Property | Current (SpeechBrain ECAPA) | CAM++ (WeSpeaker) |
|---|---|---|
| Model | speechbrain/spkrec-ecapa-voxceleb | CAM++ VoxCeleb |
| Parameters | 14.66M | 7.18M |
| Embedding dim | 192 | 512 |
| VoxCeleb1-O EER | 0.89% | 0.73% |
| CPU inference RTF | ~0.033 | ~0.013 |
| Model size | ~20 MB | ~28 MB |
| Dependencies | speechbrain, HuggingFace | wespeaker or wespeakerruntime |
| Load complexity | High (HF compat hacks) | Low (direct PyTorch or ONNX) |

---

## 3. Integration Plan for VoxTerm

### 3.1 Silero VAD Integration

**Replace RMS-based VAD in the audio pipeline:**

Currently, VoxTerm likely uses an energy/RMS threshold to detect speech. Silero
VAD provides far superior accuracy (0.97 AUC) with comparable latency.

**Recommended approach:**

```python
import torch
from silero_vad import load_silero_vad

class SileroVADDetector:
    """Drop-in replacement for RMS-based VAD."""

    def __init__(self, threshold=0.5, sample_rate=16000):
        torch.set_num_threads(1)
        self.model = load_silero_vad(onnx=True)  # Faster load, lower memory
        self.sample_rate = sample_rate
        self.threshold = threshold

    def is_speech(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """Check if a 512-sample (32ms) chunk contains speech.

        Returns (is_speech, probability).
        """
        tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        prob = self.model(tensor, self.sample_rate).item()
        return prob > self.threshold, prob

    def reset(self):
        self.model.reset_states()
```

**Key consideration:** The model expects exactly 512 samples at 16 kHz.
VoxTerm's audio chunks may be larger -- either:
1. Process in 512-sample sub-chunks and aggregate probabilities, or
2. Use `get_speech_timestamps()` on the full chunk buffer

### 3.2 CAM++ Integration

**Replace SpeechBrain ECAPA-TDNN in `diarization/engine.py`:**

```python
import wespeaker

class DiarizationEngine:
    def load(self):
        # Option A: Use WeSpeaker Python package
        self._model = wespeaker.load_model('campplus')  # or load_model_local()
        self._model.set_device('cpu')

        # Option B: Use ONNX runtime (lighter dependencies)
        import wespeakerruntime
        self._model = wespeakerruntime.Speaker(onnx_path='campplus.onnx')

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        # WeSpeaker expects file paths, not tensors directly.
        # For in-memory audio, either:
        # 1. Save to temp file (simple but I/O overhead)
        # 2. Use extract_embedding_from_pcm() internal method
        # 3. Compute fbank features manually and call model forward()
        pass
```

**Embedding dimension change:** Current code uses 192-dim embeddings (SpeechBrain
ECAPA). CAM++ produces 512-dim embeddings. This affects:
- `_speaker_centroids` storage (trivial -- just larger arrays)
- Cosine similarity computation (unchanged algorithm)
- Any saved/persisted embeddings (need migration or re-enrollment)
- Memory for stored profiles: 512 * 4 bytes = 2 KB per profile (negligible)

### 3.3 Dependency Comparison

| Dependency | Current | With Silero + WeSpeaker |
|---|---|---|
| speechbrain | Required | **Remove** |
| huggingface_hub | Required (compat hacks) | **Remove** |
| torch | Required | Required (shared) |
| torchaudio | Required | Required (shared) |
| silero-vad | -- | **Add** (~2 MB model) |
| wespeaker | -- | **Add** (~28 MB model) |
| onnxruntime | -- | Optional (for ONNX mode) |

Net effect: Remove speechbrain + huggingface_hub complexity; add silero-vad +
wespeaker (much simpler, fewer compat issues).

---

## Sources

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD PyPI](https://pypi.org/project/silero-vad/)
- [Silero VAD PyTorch Hub](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- [Silero VAD Wiki: Examples](https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies)
- [Silero VAD Streaming Notebook](https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb)
- [Silero VAD Advanced Usage (DeepWiki)](https://deepwiki.com/snakers4/silero-vad/5-advanced-usage)
- [Silero VAD State Discussion](https://github.com/snakers4/silero-vad/discussions/572)
- [WeSpeaker GitHub](https://github.com/wenet-e2e/wespeaker)
- [WeSpeaker Pretrained Models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- [WeSpeaker Python Package Docs](https://github.com/wenet-e2e/wespeaker/blob/master/docs/python_package.md)
- [wespeakerruntime PyPI](https://pypi.org/project/wespeakerruntime/)
- [CAM++ Paper (arXiv:2303.00332)](https://arxiv.org/html/2303.00332)
- [CAM++ ONNX toolkit](https://github.com/lovemefan/campplus)
- [FunASR CAM++ HuggingFace](https://huggingface.co/funasr/campplus)
