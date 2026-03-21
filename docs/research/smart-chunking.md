# Smart Audio Chunking/Segmentation for Speaker Diarization

> Research date: 2026-03-21
> Problem: VoxTerm uses fixed silence-duration triggering (0.8s silence -> split, 8s max buffer).
> This is speaker-unaware -- it splits on pauses regardless of whether the speaker changed.
> Goal: Identify chunking strategies that account for speaker changes, optimize embedding quality, and reduce diarization error.

---

## Current VoxTerm Chunking Pipeline

```
Audio Stream (1024-sample chunks @ 16kHz = 64ms each)
    |
    v
[Silero VAD: is this chunk speech?]
    |
    +--> Speech:  append to audio_buffer, reset silence counter
    +--> Silence: increment silence counter, append if mid-speech
    |
    v
[Trigger decision]
    |
    +--> silence_duration > 0.8s AND buffer > 1.0s  --> transcribe
    +--> buffer_duration >= 8.0s                     --> transcribe (force)
    |
    v
[Worker thread: transcribe + diarize full buffer as single chunk]
```

Problems with this approach:
1. A single buffer may contain multiple speakers -- the embedding is contaminated
2. A speaker change mid-sentence (no pause) is never detected
3. Long monologues are force-split at 8s regardless of sentence boundaries
4. Short back-and-forth exchanges (< 0.8s pauses) are merged into one buffer
5. The entire buffer gets ONE embedding -- no temporal resolution within the buffer

---

## 1. Speaker Change Detection (SCD)

### 1.1 BIC-Based Change Detection

**How it works:** Slide a window across the audio. At each position, split the window
into two halves. Fit a single Gaussian to the whole window and separate Gaussians to
each half. If two Gaussians explain the data significantly better than one (measured by
BIC), declare a speaker change.

**Algorithm:**
```
Given window of MFCC features X[t-W:t+W]:
    X_left  = X[t-W:t]
    X_right = X[t:t+W]

    # Fit Gaussians
    mu_full, sigma_full = fit_gaussian(X)
    mu_left, sigma_left = fit_gaussian(X_left)
    mu_right, sigma_right = fit_gaussian(X_right)

    # BIC criterion
    L_full  = log_likelihood(X, mu_full, sigma_full)
    L_split = log_likelihood(X_left, mu_left, sigma_left) +
              log_likelihood(X_right, mu_right, sigma_right)

    delta_BIC = L_split - L_full - lambda * penalty_term
    # penalty_term = 0.5 * (d + d*(d+1)/2) * log(N)
    # d = feature dimension, N = num frames, lambda = tuning parameter

    if delta_BIC > 0:
        speaker_change_at(t)
```

**Parameters:**
- Window size W: typically 2-4 seconds (1-2s per half)
- Step size: 100-500ms
- Lambda (penalty): 1.0-1.5 (higher = fewer false alarms)
- Features: 13-19 MFCCs, optionally with deltas

**Pros:**
- No training data required
- Unsupervised, works with any audio
- Well-understood mathematically
- Very cheap computationally (~1ms per window on CPU)

**Cons:**
- Poor temporal resolution (needs 2-4s of context per side)
- Cannot detect changes within overlapping speech
- Sensitive to lambda tuning
- Outperformed by neural approaches by a large margin

**Practical assessment for VoxTerm:** BIC is obsolete for new systems. Neural VAD +
embedding distance achieves much better results with comparable compute. BIC's main
value is as a baseline to beat.

**Numpy implementation (~40 lines):**
```python
def bic_change_detect(features: np.ndarray, window: int = 200,
                       step: int = 50, penalty: float = 1.3) -> list[int]:
    """Detect speaker changes via BIC on MFCC features.

    features: (T, D) array of frame-level features
    window: total window in frames (100 frames = 1s at 10ms shift)
    Returns: list of frame indices where changes are detected
    """
    changes = []
    half = window // 2
    d = features.shape[1]
    pen = 0.5 * (d + d * (d + 1) / 2) * np.log(window)

    for t in range(half, len(features) - half, step):
        X = features[t - half:t + half]
        X_l = features[t - half:t]
        X_r = features[t:t + half]

        # Log-likelihood under single vs split Gaussians
        L_full = _gaussian_ll(X)
        L_split = _gaussian_ll(X_l) + _gaussian_ll(X_r)

        delta = L_split - L_full - penalty * pen
        if delta > 0:
            changes.append(t)

    return changes

def _gaussian_ll(X: np.ndarray) -> float:
    n, d = X.shape
    cov = np.cov(X.T) + np.eye(d) * 1e-6
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * n * (d * np.log(2 * np.pi) + logdet)
```

### 1.2 Neural Speaker Change Detection

**pyannote segmentation-3.0** is the state-of-the-art neural SCD model. Rather than
detecting change points explicitly, it outputs frame-level speaker activity for up to
3 speakers per 10-second window. Speaker changes are implicit in the output: when
speaker A's activation drops and speaker B's rises, that is a change.

**Architecture:** PyanNet -- a neural network that processes 10s audio chunks at
16kHz and outputs (num_frames, 7) predictions, where the 7 classes are:
- Non-speech
- Speaker 1 only
- Speaker 2 only
- Speaker 3 only
- Speakers 1+2 overlap
- Speakers 1+3 overlap
- Speakers 2+3 overlap

**Temporal resolution:** 16ms per frame (625 frames for 10s).

**Latency:** 11ms on CPU per 10s chunk (segmentation-3.0).

**How to detect speaker changes from the output:**
```python
# seg shape: (frames, speakers) after powerset-to-multilabel conversion
# Each frame has per-speaker activation in [0, 1]

def find_speaker_changes(seg: np.ndarray, threshold: float = 0.5) -> list[int]:
    """Find frame indices where the dominant speaker changes."""
    # Binarize
    active = seg > threshold  # (frames, speakers)
    # Find dominant speaker per frame (argmax of activation)
    dominant = np.argmax(seg, axis=1)
    # Change = dominant speaker differs from previous frame
    changes = np.where(np.diff(dominant) != 0)[0] + 1
    return changes.tolist()
```

**SCDiar approach (2025):** Uses a BiLSTM + 1D-CNN (kernel size 3) on token-level
ASR features. Outputs a probability sequence, applies peak detection with threshold
theta_scd. Minimum 10 tokens per segment to prevent unreliable short segments.

**Practical assessment for VoxTerm:** The pyannote segmentation model is the best
option but requires either the pyannote dependency (heavy, gated model) or an ONNX
export (via pyannote-onnx). The model is MIT-licensed. If adding pyannote is too heavy,
embedding-distance-based detection (section 1.3) is the best self-contained alternative.

### 1.3 Embedding-Distance-Based Change Detection

**The most practical approach for VoxTerm.** Uses the existing CAM++ embedding model
to detect speaker changes by comparing embeddings from adjacent sliding windows.

**Algorithm:**
```
Maintain a sliding window of audio (e.g., 2s window, 500ms hop).

For each new position:
    emb_current = extract_embedding(audio[t-2s : t])
    emb_previous = extract_embedding(audio[t-2.5s : t-0.5s])

    distance = 1.0 - cosine_similarity(emb_current, emb_previous)

    if distance > change_threshold:
        speaker_change_detected_at(t - 0.5s)  # change within the overlap region
```

**Key parameters:**
- Window size: 1.5-3.0s (must be long enough for reliable embeddings)
- Hop/step: 250-500ms
- Change threshold: 0.3-0.5 cosine distance (empirical, depends on model)

**Practical considerations:**
- Each embedding extraction costs ~40ms on CPU with CAM++ (RTF 0.013)
- With 500ms hop and 2s window: 2 extractions per second = ~80ms/s of compute
- At 16kHz, 2s window = 32000 samples = 400 Fbank frames

**Refinement -- multi-scale detection:**
Compute embeddings at multiple scales (1.5s, 2.5s, 3.5s) and combine:
```python
def detect_change_multiscale(audio, t, model, threshold=0.4):
    distances = []
    for window_sec in [1.5, 2.5, 3.5]:
        w = int(window_sec * 16000)
        hop = int(0.5 * 16000)
        if t < w + hop:
            continue
        emb_curr = model.extract(audio[t-w:t])
        emb_prev = model.extract(audio[t-w-hop:t-hop])
        dist = 1.0 - cosine_sim(emb_curr, emb_prev)
        distances.append(dist)
    # Change if majority of scales agree
    return np.mean(distances) > threshold
```

**Practical assessment for VoxTerm:** This is the best fit. It reuses the existing
CAM++ model, requires no new dependencies, and adds ~80ms/s of compute on CPU.
Can run in the diarizer subprocess. The main downside is that 1.5s minimum window
means changes can only be detected with 1.5s granularity.

### 1.4 Energy/Pitch-Based Change Detection

**How it works:** Track energy (RMS) and pitch (F0) contours. Speaker changes often
coincide with:
- Energy dips (both speakers pause briefly)
- Pitch discontinuities (different speakers have different F0 ranges)
- Spectral centroid shifts

**Algorithm:**
```python
def energy_pitch_change(audio, sr=16000, frame_ms=25, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    # Frame-level energy
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_len]**2))
        for i in range(0, len(audio) - frame_len, hop_len)
    ])

    # Smooth and differentiate
    kernel = np.ones(5) / 5
    energy_smooth = np.convolve(energy, kernel, mode='same')
    energy_delta = np.abs(np.diff(energy_smooth))

    # Change points = local maxima of energy derivative above threshold
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(energy_delta, height=np.mean(energy_delta) * 2,
                          distance=int(500 / hop_ms))  # min 500ms between changes
    return peaks
```

**Practical assessment for VoxTerm:** Very cheap (~0.1ms per second of audio) but
unreliable. Works for clean turn-taking (one person stops, another starts) but fails
for overlapping speech, back-channel responses, or speakers with similar energy levels.
Useful only as a cheap pre-filter to reduce the number of embedding extractions needed.

---

## 2. Sliding Window with Overlap (diart-Style)

### 2.1 How diart's Sliding Window Works

diart processes audio as a continuous stream with a 5s rolling buffer and 500ms hop:

```
Time:  0s        5s        5.5s       6s        6.5s
       |---------|
              |---------|
                     |---------|
                            |---------|

Each window: 5s of audio -> segmentation model -> embedding extraction -> clustering
Step: 500ms between windows
Overlap: 4.5s (90% overlap between consecutive windows)
```

**Per-step processing (every 500ms):**
1. Segmentation model runs on the FULL 5s buffer -> (frames, local_speakers) posteriors
2. Overlap-aware penalty weights frames -> clean per-speaker embeddings
3. Hungarian assignment maps local speakers to global centroids
4. Prediction appended to rolling buffer of predictions
5. Overlapping predictions aggregated via Hamming-weighted average
6. Binarized at tau_active threshold -> final speaker labels

**Why 5s windows?** The segmentation model (PyanNet) was trained on 5s/10s chunks.
Shorter windows degrade its ability to distinguish speakers. The model needs
acoustic context to reliably identify speaker patterns.

### 2.2 Aggregation of Overlapping Windows

The key to smooth predictions is combining overlapping window outputs:

```
Window 1 prediction: [A=0.9, B=0.1] [A=0.8, B=0.2] [A=0.7, B=0.3] ...
Window 2 prediction:          [A=0.85, B=0.15] [A=0.75, B=0.25] [A=0.3, B=0.7] ...
                              ^^^overlap zone^^^

Hamming weights:     [0.08]   [0.29]   [0.54]   [0.77]   [0.93]   [1.0]  ...
                     ^edge                                          ^center

Aggregated = sum(hamming_weight * prediction) / sum(hamming_weight)
```

The Hamming window gives highest weight to the CENTER of each prediction window
(where the model has the most context and is most confident), and lowest weight
to the EDGES. When multiple windows overlap, the edge of one window overlaps with
the middle of another -- the more confident prediction dominates.

**Number of overlapping windows:**
```
num_overlapping = round(latency / step)
```
- latency=0.5s (min), step=0.5s -> 1 window (no aggregation)
- latency=2.5s, step=0.5s -> 5 windows averaged
- latency=5.0s (max), step=0.5s -> 10 windows averaged

More overlap = smoother, more accurate predictions = more latency.

### 2.3 Computational Cost

Measured on AMD Ryzen 9 CPU (single thread):

| Component              | CPU Time | GPU Time |
|------------------------|----------|----------|
| Segmentation (pyannote)| 11-12ms  | 8ms      |
| Embedding (pyannote)   | 26ms     | 12ms     |
| Clustering (numpy)     | < 1ms    | < 1ms    |
| **Total per 500ms step** | **~40ms** | **~21ms** |

**CPU duty cycle:** 40ms every 500ms = 8% CPU utilization. Easily real-time.

**VoxTerm adaptation:** Without the pyannote segmentation model, using only CAM++
embeddings on sliding windows:
- CAM++ embedding extraction: ~40ms per 2s window
- With 500ms hop: ~80ms per second of audio = 8% CPU
- This is feasible in the diarizer subprocess

### 2.4 Simplified diart-Style Pipeline for VoxTerm

A practical adaptation without pyannote segmentation:

```
Audio Stream
    |
    v
[Silero VAD: frame-level speech probabilities]
    |
    v
[Sliding window: 2s window, 500ms hop]
    |
    +--> For each window with speech:
    |      emb = CAM++.extract(window_audio)
    |      speaker_id = assign_to_cluster(emb)
    |
    v
[Temporal smoothing: median filter over last 3-5 assignments]
    |
    v
[Output: (start_time, end_time, speaker_id) per segment]
```

This trades off the segmentation model's overlap awareness for simplicity.
Still a major improvement over the current single-embedding-per-buffer approach.

---

## 3. VAD-Guided Segmentation

### 3.1 Using Silero VAD for Smart Splitting

VoxTerm already uses Silero VAD for speech/silence detection. The key insight is
to use it for SEGMENTATION rather than just gating:

**Current flow:**
```
Silero VAD -> binary speech/silence -> buffer speech -> split on 0.8s silence
```

**Improved flow:**
```
Silero VAD -> speech probability per 32ms frame
    |
    v
[Speech segment detection: start/end boundaries]
    |
    v
[Split at speech boundaries, NOT fixed silence duration]
    |
    v
[Each speech segment -> separate embedding + diarization]
```

### 3.2 Silero VADIterator for Streaming

The `VADIterator` class provides speech start/end events in real-time:

```python
# State machine:
# 1. Waiting for speech: prob < threshold -> continue
# 2. Speech starts: prob >= threshold -> emit {'start': sample_idx}
# 3. In speech: prob stays high -> continue
# 4. Tentative end: prob < (threshold - 0.15) -> set temp_end
# 5. Confirmed end: silence > min_silence_duration_ms -> emit {'end': sample_idx}
#    OR speech resumes -> cancel tentative end, continue

# Key parameters:
threshold = 0.5           # speech onset
neg_threshold = 0.35      # speech offset (hysteresis prevents flapping)
min_silence_duration_ms = 100  # minimum silence to confirm end-of-speech
speech_pad_ms = 30        # padding on both sides of detected speech
```

### 3.3 Smart Segmentation with Silero

Using `get_speech_timestamps()` on the audio buffer before diarization:

```python
from silero_vad import get_speech_timestamps

def segment_buffer(audio: np.ndarray, model, sr: int = 16000):
    """Split audio buffer into speech segments using Silero VAD.

    Returns list of (start_sample, end_sample) tuples.
    """
    import torch
    wav = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=sr,
        min_speech_duration_ms=500,     # ignore < 500ms speech
        min_silence_duration_ms=300,    # need 300ms silence to split
        speech_pad_ms=100,              # pad segments for clean edges
        max_speech_duration_s=6.0,      # force-split at 6s
    )
    return [(ts['start'], ts['end']) for ts in timestamps]
```

### 3.4 How This Changes VoxTerm's Pipeline

**Before (current):**
```
8s buffer -> 1 transcription -> 1 embedding -> 1 speaker label
```

**After (VAD-guided):**
```
8s buffer -> VAD segments -> N speech segments
    |
    +--> Segment 1 (2.3s) -> transcribe -> embed -> speaker A
    +--> Segment 2 (0.8s) -> transcribe -> too short for embed, assign to recent speaker
    +--> Segment 3 (3.1s) -> transcribe -> embed -> speaker B
```

**Benefits:**
- Each segment is more likely to contain a single speaker
- Short back-channel responses ("yeah", "uh-huh") are separated
- Silence between speakers is excluded from embeddings (cleaner)
- Natural speech boundaries respected, not arbitrary 0.8s silence

**Implementation concern:** Multiple shorter transcription calls vs one long one.
Qwen3-ASR handles short segments well (unlike Whisper which hallucinates on < 1s).
The overhead of multiple calls is offset by shorter per-call latency.

---

## 4. Adaptive Chunking

### 4.1 Concept

Short chunks when speakers are changing frequently, longer chunks during monologues.
The chunk duration adapts to the conversational dynamics.

### 4.2 Algorithm

```python
class AdaptiveChunker:
    """Dynamically adjust chunk duration based on speaker change frequency."""

    def __init__(self, min_chunk: float = 1.5, max_chunk: float = 8.0,
                 target_chunk: float = 3.0, change_cooldown: float = 2.0):
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.target_chunk = target_chunk
        self.change_cooldown = change_cooldown

        self._current_chunk_target = target_chunk
        self._last_change_time = 0.0
        self._change_count_window = []  # timestamps of recent changes
        self._window_duration = 30.0    # look-back window for change rate

    def update(self, time_now: float, speaker_changed: bool):
        """Call after each diarization decision."""
        if speaker_changed:
            self._last_change_time = time_now
            self._change_count_window.append(time_now)

        # Prune old changes
        cutoff = time_now - self._window_duration
        self._change_count_window = [
            t for t in self._change_count_window if t > cutoff
        ]

        # Compute change rate (changes per minute)
        elapsed = min(time_now, self._window_duration)
        if elapsed > 0:
            change_rate = len(self._change_count_window) / (elapsed / 60.0)
        else:
            change_rate = 0

        # Adapt chunk target
        if change_rate > 10:   # rapid switching (> 10 changes/min)
            self._current_chunk_target = self.min_chunk
        elif change_rate > 5:  # moderate switching
            self._current_chunk_target = 2.0
        elif change_rate > 2:  # normal conversation
            self._current_chunk_target = self.target_chunk
        else:                  # monologue
            self._current_chunk_target = self.max_chunk

    @property
    def chunk_target(self) -> float:
        """Current target chunk duration in seconds."""
        return self._current_chunk_target
```

### 4.3 Integration with VoxTerm

Replace the fixed MAX_BUFFER_SECONDS with the adaptive chunker:

```python
# In _process_audio_inner:
if self._had_speech and silence_duration > SILENCE_TRIGGER_SECONDS \
   and buffer_duration > MIN_BUFFER_SECONDS:
    self._trigger_transcription()
elif buffer_duration >= self._chunker.chunk_target:  # was: MAX_BUFFER_SECONDS
    self._trigger_transcription()
```

### 4.4 Combining Adaptive Chunking with Speaker Change Detection

The most powerful approach: use embedding-distance SCD to trigger splits DURING
a buffer, and adaptive chunking to set the maximum:

```python
def should_split(audio_buffer, current_duration, chunker, last_embedding, model):
    """Decide whether to split the current buffer."""

    # 1. Silence-based split (existing behavior)
    if silence_duration > 0.8 and current_duration > 1.0:
        return True

    # 2. Adaptive max duration
    if current_duration >= chunker.chunk_target:
        return True

    # 3. Speaker change detection (embedding distance)
    if current_duration >= 3.0 and last_embedding is not None:
        # Extract embedding from last 2s of buffer
        recent_audio = audio_buffer[-32000:]  # last 2s
        recent_emb = model.extract(recent_audio)
        distance = 1.0 - cosine_sim(recent_emb, last_embedding)
        if distance > 0.4:  # speaker changed
            return True

    return False
```

---

## 5. Optimal Chunk Sizes for Speaker Embeddings

### 5.1 The Duration-Quality Tradeoff

Speaker embeddings degrade significantly below certain durations. Research consensus:

| Duration   | Embedding Quality | Recommendation |
|------------|-------------------|----------------|
| < 0.5s     | Very poor         | Do not use for clustering. Assign to most recent speaker. |
| 0.5 - 1.0s | Poor              | Usable for same-speaker continuity, not for new speaker decisions. |
| 1.0 - 1.5s | Marginal          | Can match to existing centroids but should NOT create new speakers. |
| 1.5 - 3.0s | Good              | Reliable for matching. Can create new speakers with caution. |
| 3.0 - 5.0s | Optimal           | Best accuracy/latency tradeoff. Use for enrollment and centroid updates. |
| 5.0 - 10s  | Diminishing returns| Slightly better but not worth the latency. Risk of multi-speaker contamination. |
| > 10s      | Plateau / risk    | Likely to contain speaker changes, defeating the purpose. |

### 5.2 CAM++ Specific Guidance

From WeSpeaker documentation and benchmarks:

- **Minimum practical duration:** 0.5s (below this, too few Fbank frames for pooling)
- **Recommended duration:** > 3s for best accuracy
- **LM (large-margin) models:** Optimized specifically for > 3s segments
- **Architecture detail:** TSTP (Temporal Statistics Pooling) aggregates across time.
  With 10ms frame shift, a 1.5s segment = 150 frames. The mean/std statistics
  become more reliable with more frames.
- **RTF 0.013:** A 3s segment takes ~40ms to process on single CPU thread

### 5.3 VBx x-Vector Extraction Parameters

From the VBx reference implementation:
- **Segment length (seg_len):** 144 frames = 1.44s at 10ms shift (default)
- **Segment hop (seg_jump):** 24 frames = 0.24s
- **Overlap:** 83% (144-24=120 frames overlap)
- **Tail handling:** If remaining > 10 frames, extract one more segment

This means VBx uses 1.44s segments with heavy overlap -- much shorter than the
"3s optimal" for single embeddings, because VBx aggregates across many overlapping
embeddings via its HMM.

### 5.4 NVIDIA MSDD Multi-Scale Windows

MSDD extracts embeddings at FIVE scales simultaneously:
- 1.5s, 1.25s, 1.0s, 0.75s, 0.5s

A 1D-CNN dynamically weights these scales at each time step:
- During stable monologue: longer scales dominate (better speaker ID)
- During speaker transitions: shorter scales dominate (better temporal precision)
- Result: 0.25s temporal resolution without over-segmentation
- DER improvement: 60% reduction vs single-scale

### 5.5 Sweet Spot for VoxTerm

Given VoxTerm's constraints (single embedding per chunk, no segmentation model,
CPU inference, < 200ms latency):

**Recommended: 2.5-4.0s chunks as the target range.**

Rationale:
- 2.5s minimum gives CAM++ enough frames for reliable embeddings
- 4.0s keeps latency reasonable (transcription starts within 4s of silence)
- Force-split at 6-8s for long monologues
- Do NOT create new speakers from chunks < 1.5s

**With sliding window approach (future):**
- 2s windows with 500ms hop
- Each window gets its own embedding
- Cluster assignment uses median of last 3-5 windows
- Effective latency: 2.5-4.5s (depending on aggregation depth)

---

## 6. Two-Pass Approach

### 6.1 What Is It?

First pass: transcribe audio with fixed/simple chunking optimized for ASR quality.
Second pass: re-segment and re-diarize the same audio using speaker-aware strategies.

### 6.2 Common in Production?

**Yes.** Most production speaker diarization systems use a two-pass or multi-pass
architecture:

1. **whisper-diarization (MahmoudAshraf97):**
   - Pass 1: Whisper transcription -> word-level timestamps
   - Pass 2: MarbleNet VAD -> speech segmentation -> TitaNet embeddings -> clustering
   - Pass 3: CTC forced alignment -> word-to-speaker mapping

2. **pyannote speaker-diarization-3.0:**
   - Pass 1: Segmentation model on sliding windows (10s chunks, every 16ms)
   - Pass 2: Embedding extraction per speaker per chunk
   - Pass 3: Global clustering + aggregation

3. **Dimitriadis i-vector system:**
   - Fast pass: immediate labels from current active window
   - Accurate pass: refined labels via reconciliation of adjacent windows

4. **VBx pipeline:**
   - Pass 1: x-vector extraction on sliding windows (1.44s, 0.24s hop)
   - Pass 2: AHC initialization
   - Pass 3: VB-HMM refinement

### 6.3 Two-Pass Design for VoxTerm

```
PASS 1: Transcription-Optimized Chunking
    |
    Audio buffer fills until: 0.8s silence OR 8s max
    |
    v
    Qwen3-ASR transcribes the full buffer -> text + word timestamps
    |
    v
PASS 2: Diarization-Optimized Re-Segmentation
    |
    Same audio buffer is re-analyzed:
    |
    a) Silero VAD identifies speech segments within the buffer
    b) Word timestamps from Qwen3 refine segment boundaries
    c) Segments > 1.5s get their own embedding via CAM++
    d) Segments < 1.5s are assigned to the nearest neighboring segment's speaker
    e) Embeddings are clustered against session centroids
    |
    v
    Result: (text, speaker_id) pairs at finer temporal resolution
```

**Benefits of two-pass for VoxTerm:**
- Transcription quality is unchanged (same chunking as today)
- Diarization gets speaker-homogeneous segments (better embeddings)
- No latency penalty: both passes run on the same audio buffer
- The second pass is cheap: VAD + 1-3 embedding extractions per buffer

### 6.4 Implementation Sketch

```python
def _transcribe_and_diarize(self, audio: np.ndarray):
    """Two-pass: transcribe first, then diarize with better segmentation."""

    # PASS 1: Transcribe the full buffer
    text = self.transcriber.transcribe(audio)
    if not text.strip():
        return

    # PASS 2: Re-segment for diarization
    speech_segments = silero_segment(audio, self.vad_model)

    results = []
    for start, end in speech_segments:
        seg_audio = audio[start:end]
        seg_duration = (end - start) / SAMPLE_RATE

        if seg_duration >= 1.5:
            # Full diarization
            label, speaker_id = self.diarizer.identify(seg_audio)
        else:
            # Too short -- assign to most recent speaker
            label, speaker_id = self._last_speaker_label, self._last_speaker_id

        results.append({
            'text': extract_text_for_segment(text, start, end),  # needs alignment
            'speaker': label,
            'speaker_id': speaker_id,
        })

    return results
```

### 6.5 The Alignment Problem

The main challenge with two-pass: mapping Qwen3's text output back to the
VAD-derived segments. Options:

1. **Timestamp-based:** Qwen3-ASR provides word-level timestamps. Use these to
   assign words to VAD segments. This is the cleanest approach.

2. **Proportional split:** Divide the text proportionally by the duration of each
   VAD segment. Crude but simple.

3. **Re-transcribe per segment:** Transcribe each VAD segment independently with
   Qwen3. Most accurate but 3-5x more compute.

4. **Hybrid:** Use Qwen3's full-buffer transcription for the text, but only use
   VAD segments for speaker assignment. Each segment gets a speaker label, and
   the full text is displayed with speaker change markers.

Option 4 is the most practical for VoxTerm: it preserves the current transcription
quality while adding sub-buffer speaker resolution.

---

## 7. Synthesis: Recommended Approach for VoxTerm

### Phase 1: VAD-Guided Splitting (minimal change, high impact)

Replace the fixed silence-duration trigger with Silero VAD segment boundaries:

```python
# In _trigger_transcription:
audio = self.audio_buffer.get_and_clear()

# Use Silero to find speech segments within the buffer
speech_segments = self.vad.get_speech_timestamps(audio)

if len(speech_segments) <= 1:
    # Single segment or no segments -- process as before
    self._transcribe_audio(audio)
else:
    # Multiple speech segments -- split and process each
    for seg in speech_segments:
        seg_audio = audio[seg['start']:seg['end']]
        self._transcribe_audio(seg_audio)
```

**Impact:** Each diarization embedding now comes from a single speech segment
rather than a buffer containing mixed speakers separated by pauses.
**Effort:** ~30 lines changed.

### Phase 2: Embedding-Distance Change Detection (medium change, high impact)

Add intra-buffer speaker change detection using CAM++ embeddings:

```python
def _split_on_speaker_changes(self, audio: np.ndarray) -> list[np.ndarray]:
    """Split audio at detected speaker change points."""
    window_samples = int(2.0 * SAMPLE_RATE)  # 2s window
    hop_samples = int(0.5 * SAMPLE_RATE)     # 500ms hop

    if len(audio) < window_samples * 2:
        return [audio]  # too short to detect changes

    embeddings = []
    positions = []
    for start in range(0, len(audio) - window_samples, hop_samples):
        end = start + window_samples
        emb = self.diarizer.extract_embedding(audio[start:end])
        embeddings.append(emb)
        positions.append(start + window_samples // 2)

    # Find change points: large distance between consecutive embeddings
    change_points = []
    for i in range(1, len(embeddings)):
        dist = 1.0 - cosine_sim(embeddings[i], embeddings[i-1])
        if dist > 0.4:
            change_points.append(positions[i])

    # Split at change points
    segments = []
    prev = 0
    for cp in change_points:
        if cp - prev >= int(1.5 * SAMPLE_RATE):  # min 1.5s segment
            segments.append(audio[prev:cp])
            prev = cp
    segments.append(audio[prev:])
    return segments
```

**Impact:** Speaker changes within a single buffer are detected and split.
Each resulting segment gets its own embedding, dramatically improving
diarization accuracy for conversational audio.
**Effort:** ~80 lines + new IPC message for extract_embedding.

### Phase 3: Adaptive Chunking (polish)

Replace the fixed 8s MAX_BUFFER_SECONDS with the adaptive chunker from
section 4.2. During rapid speaker exchanges, the buffer limit drops to
1.5-2.0s, ensuring each chunk contains a single speaker. During
monologues, it rises to 6-8s for better transcription context.

**Impact:** Reduces multi-speaker contamination of long buffers.
**Effort:** ~40 lines.

### Phase 4: Two-Pass (future)

Full two-pass with Qwen3 word timestamps for text-to-segment alignment.
This is the most accurate but requires changes to how transcript entries
are displayed (sub-buffer speaker labels).

**Impact:** Best possible diarization accuracy within VoxTerm's architecture.
**Effort:** ~200 lines.

---

## Key Numbers Summary

| Parameter | Current | Recommended | Source |
|-----------|---------|-------------|--------|
| Silence trigger | 0.8s fixed | VAD speech boundaries | Silero VAD |
| Max buffer | 8.0s fixed | 1.5-8.0s adaptive | Adaptive chunker |
| Min embedding duration | 1.5s (24000 samples) | 1.5s (unchanged, good) | Research consensus |
| Optimal embedding duration | Any (full buffer) | 2.5-4.0s | CAM++/ECAPA benchmarks |
| Speaker change detection | None | Embedding distance, 0.4 threshold | Sliding window SCD |
| Change detection window | N/A | 2.0s | Embedding quality sweet spot |
| Change detection hop | N/A | 500ms | diart precedent |
| Segments per buffer | 1 | 1-4 (VAD-guided) | VAD segmentation |
| Embeddings per second | 0.1-0.3 | 0.5-2.0 | Sliding window |

---

## Sources

1. Bredin & Laurent, "End-to-end speaker segmentation for overlap-aware resegmentation," 2021. [arXiv:2104.04045](https://arxiv.org/abs/2104.04045)
2. Coria et al., "Overlap-aware low-latency online speaker diarization," ASRU 2021. [arXiv:2109.06483](https://arxiv.org/abs/2109.06483)
3. Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx)," 2021. [arXiv:2012.14952](https://arxiv.org/abs/2012.14952)
4. Park et al., "Multi-scale Speaker Diarization with Dynamic Scale Weighting," 2022. [arXiv:2203.15974](https://arxiv.org/abs/2203.15974)
5. SCDiar, "A streaming diarization system based on speaker change detection," 2025. [arXiv:2501.16641](https://arxiv.org/abs/2501.16641)
6. "A Review of Common Online Speaker Diarization Methods," 2024. [arXiv:2406.14464](https://arxiv.org/abs/2406.14464)
7. Silero VAD, https://github.com/snakers4/silero-vad
8. diart, https://github.com/juanmc2005/diart
9. VBx, https://github.com/BUTSpeechFIT/VBx
10. WeSpeaker CAM++, https://github.com/wenet-e2e/wespeaker
11. whisper-diarization, https://github.com/MahmoudAshraf97/whisper-diarization
12. pyannote segmentation-3.0, https://huggingface.co/pyannote/segmentation-3.0
13. Alumae online SCD, https://github.com/alumae/online_speaker_change_detector
14. Wang et al., "CAM++: A Fast and Efficient Network for Speaker Verification," 2023. [arXiv:2303.00332](https://arxiv.org/abs/2303.00332)
