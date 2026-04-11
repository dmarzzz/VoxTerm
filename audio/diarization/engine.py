"""Speaker diarization via 3D-Speaker embeddings + online cosine clustering.

Identifies the dominant speaker in each audio chunk by extracting a speaker
embedding and comparing it to a running set of speaker centroids.
New speakers are created automatically when similarity falls below a threshold.

Models (via 3D-Speaker):
  - ONNX backend (default): ERes2Net-large (512-dim) or CAM++ (512-dim)
    No PyTorch required — runs in-process alongside MLX.
  - PyTorch backend (fallback): loaded via speakerlab in subprocess.
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

_MIN_SPEECH_SAMPLES = 16000   # 1.0 s at 16 kHz — shorter → unreliable embeddings
MAX_SPEAKERS = 8              # hard cap on simultaneous speaker clusters
_SCD_MIN_SAMPLES = 48000     # 3.0 s at 16 kHz — minimum audio length for SCD


class DiarizationEngine:
    """Online speaker identification using ECAPA-TDNN embeddings."""

    MATCH_THRESHOLD = 0.55        # cosine sim above this → assign to existing speaker
    MATCH_THRESHOLD_DISCOVERY = 0.70  # stricter threshold during discovery phase
    NEW_SPEAKER_THRESHOLD = 0.45  # must be below this vs ALL centroids to create new speaker
    CONTINUITY_BONUS = 0.05       # small bias toward keeping the same speaker across short turns
    CONFLICT_MARGIN = 0.05        # if top-2 within this → prefer more established speaker
    MERGE_THRESHOLD = 0.65        # pairwise cosine sim above this → merge clusters
    QUALITY_RMS_THRESHOLD = 0.003 # min RMS energy for quality-gated centroid update
    MERGE_INTERVAL = 5            # check for cluster merges every N identify() calls
    RECLUSTER_INTERVAL = 8        # spectral re-clustering every N identify() calls
    RECLUSTER_MIN_SEGMENTS = 4    # min total segments before re-clustering kicks in
    LOOP_PROB = 0.99              # VBx-style HMM self-transition probability
    WHITEN_MIN_SEGMENTS = 8       # min segments before PLDA-lite whitening kicks in
    SCD_CHANGE_THRESHOLD = 0.6    # cosine distance above this → speaker change detected
    SCD_WINDOW_SEC = 2.0          # sliding window duration for SCD embedding extraction
    SCD_HOP_SEC = 0.5             # hop between consecutive SCD windows
    CENTROID_EMA_ALPHA = 0.3      # EMA weight for new embedding when updating centroids
    CENTROID_UPDATE_MIN_SIM = 0.50  # min cosine sim to centroid before updating it
    MAX_EMBEDDINGS_PER_SPEAKER = 20  # cap per-speaker embedding retention
    MAX_SEGMENT_ORDER = 200       # cap temporal segment history
    DISCOVERY_PHASE_CALLS = 30    # number of identify() calls for discovery phase

    def __init__(self):
        self._model = None
        self._onnx_embedder = None  # OnnxSpeakerEmbedder (when backend="onnx")
        self._backend = "pytorch"   # "onnx", "pytorch", or "mock"
        self._segmentation = None   # pyannote segmentation for overlap-aware embeddings
        self._loaded = False
        self._speaker_centroids: dict[int, np.ndarray] = {}
        self._next_id = 1
        self._last_speaker_id = 1
        self._speaker_colors: dict[int, str] = {}
        self._speaker_names: dict[int, str] = {}
        # Per-segment embedding retention: speaker_id → [(embedding, duration_sec)]
        self._segment_embeddings: dict[int, list[tuple[np.ndarray, float]]] = {}
        # Stabilization tracking: speaker_id → previous centroid (for convergence check)
        self._prev_centroids: dict[int, np.ndarray] = {}
        # Tracks which speakers have been cross-session matched already
        self._matched_speakers: set[int] = set()
        # Periodic merge tracking
        self._identify_count = 0
        # Temporal segment order for HMM smoothing: [(speaker_id, embedding)]
        self._segment_order: list[tuple[int, np.ndarray]] = []
        # PLDA-lite: cached whitening transform (updated periodically)
        self._whiten_matrix: np.ndarray | None = None
        self._whiten_mean: np.ndarray | None = None
        # Overlap metadata from last identify() call
        self._last_identify_meta: dict = {}
        self._color_palette = [
            "#00ffcc",   # cyan
            "#ff44aa",   # pink
            "#44ff44",   # green
            "#ffaa00",   # amber
            "#aa88ff",   # lavender
            "#ff6644",   # coral
            "#44ddff",   # sky
            "#ffff44",   # yellow
        ]

    # ── lifecycle ─────────────────────────────────────────

    # Legacy CAM++ URL (WeSpeaker) — used only if speakerlab is unavailable
    _LEGACY_MODEL_URL = (
        "https://modelscope.cn/models/"
        "iic/speech_campplus_sv_en_voxceleb_16k/resolve/master/"
        "campplus_voxceleb.bin"
    )

    def load(self, backend: str | None = None):
        """Load the speaker embedding model.

        Args:
            backend: "onnx" or "pytorch". If None, reads from config.
        """
        import os
        if os.environ.get("VOXTERM_MOCK_ENGINE"):
            self._model = _MockEmbeddingModel()
            self._backend = "mock"
            self._loaded = True
            return

        if backend is None:
            from config import SPEAKER_MODEL_BACKEND
            backend = SPEAKER_MODEL_BACKEND

        if backend == "onnx":
            self._load_onnx()
        else:
            self._load_pytorch()

        # Load segmentation model for overlap-aware embeddings (optional)
        try:
            from .segmentation import SpeakerSegmentation
            self._segmentation = SpeakerSegmentation()
            if not self._segmentation.is_loaded:
                self._segmentation = None
        except Exception:
            self._segmentation = None

        self._loaded = True

    def _load_onnx(self) -> None:
        """Load 3D-Speaker model via ONNX Runtime (no PyTorch)."""
        from config import SPEAKER_MODEL_NAME
        from .onnx_embedder import OnnxSpeakerEmbedder

        self._onnx_embedder = OnnxSpeakerEmbedder(model_name=SPEAKER_MODEL_NAME)
        self._onnx_embedder.load()
        self._backend = "onnx"
        log.info("Diarization engine using ONNX backend (%s)", SPEAKER_MODEL_NAME)

    def _load_pytorch(self) -> None:
        """Load speaker model via PyTorch (subprocess-safe path)."""
        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        try:
            # Prefer speakerlab (3D-Speaker) if available
            self._load_pytorch_speakerlab()
        except ImportError:
            # Fallback to vendored CAM++ (WeSpeaker)
            self._load_pytorch_legacy()

        self._backend = "pytorch"

    def _load_pytorch_speakerlab(self) -> None:
        """Load 3D-Speaker model via speakerlab package."""
        import torch
        from config import SPEAKER_MODEL_NAME
        from .onnx_embedder import ONNX_MODELS

        from diarization.export_onnx import _find_checkpoint, _create_model, MODEL_CONFIGS

        if SPEAKER_MODEL_NAME not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {SPEAKER_MODEL_NAME}")

        config = MODEL_CONFIGS[SPEAKER_MODEL_NAME]
        model_id = config["modelscope_id"]
        revision = config.get("revision")

        from modelscope.hub.snapshot_download import snapshot_download
        if revision is not None:
            model_dir = snapshot_download(model_id, revision=revision)
        else:
            model_dir = snapshot_download(model_id)
        model = _create_model(config)
        ckpt_path = _find_checkpoint(__import__("pathlib").Path(model_dir))
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        log.info("Diarization engine using PyTorch/speakerlab backend (%s)", SPEAKER_MODEL_NAME)

    def _load_pytorch_legacy(self) -> None:
        """Load vendored CAM++ from WeSpeaker (legacy fallback)."""
        import torch
        from .campplus import CAMPPlus

        model_path = self._ensure_legacy_model()
        self._model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        log.info("Diarization engine using PyTorch/legacy CAM++ backend")

    @classmethod
    def _ensure_legacy_model(cls) -> str:
        """Download legacy CAM++ weights on first use, cache locally."""
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "wespeaker" / "campplus_voxceleb"
        model_path = cache_dir / "campplus_voxceleb.bin"
        if model_path.exists():
            return str(model_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(cls._LEGACY_MODEL_URL, str(model_path))
        return str(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_last_identify_meta(self) -> dict:
        """Return metadata from the last identify() call.

        Returns dict with:
            is_overlap: bool — True if overlapping speech detected
            overlap_speakers: list[int] — speaker IDs involved in overlap
        """
        return self._last_identify_meta.copy()

    # ── speaker identification ────────────────────────────

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, int]:
        """Identify the dominant speaker in an audio chunk.

        Returns (label, speaker_id):
            label      – custom name or "Speaker 1", "Speaker 2", …
            speaker_id – integer key (1-based)
        """
        if not self._loaded:
            log.warning("identify() called but diarization model not loaded — returning Speaker 1")
            return "Speaker 1", 1
        if self._backend == "pytorch" and self._model is None:
            log.warning("identify() called but PyTorch model is None — returning Speaker 1")
            return "Speaker 1", 1
        if self._backend == "onnx" and self._onnx_embedder is None:
            log.warning("identify() called but ONNX embedder is None — returning Speaker 1")
            return "Speaker 1", 1

        # Ensure mono float32
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Trim leading/trailing silence for cleaner embeddings
        audio = self._trim_silence(audio)

        # If trimmed audio is too short, reuse last known speaker
        if len(audio) < _MIN_SPEECH_SAMPLES:
            sid = self._last_speaker_id
            label = self._speaker_names.get(sid, f"Speaker {sid}")
            return label, sid

        # Quality gate: check RMS energy of speech portion
        rms = float(np.sqrt(np.mean(audio ** 2)))
        is_high_quality = rms >= self.QUALITY_RMS_THRESHOLD

        # Overlap-aware embedding: use segmentation to weight frames
        # so overlapping speech doesn't contaminate the embedding
        if self._segmentation is not None and len(audio) >= _MIN_SPEECH_SAMPLES:
            # Check if segmentation detects any active speaker at all
            activation = self._segmentation.segment(audio)
            active_speakers = self._segmentation.get_active_speakers(activation)
            if not active_speakers:
                # Segmentation says nobody is speaking — treat as silence
                sid = self._last_speaker_id
                label = self._speaker_names.get(sid, f"Speaker {sid}")
                return label, sid
            embedding = self._extract_overlap_aware_from_activation(
                audio, activation, active_speakers, sample_rate
            )
            if embedding is None:
                embedding = self._extract_embedding_raw(audio, sample_rate)
        else:
            embedding = self._extract_embedding_raw(audio, sample_rate)

        if embedding is None:
            sid = self._last_speaker_id
            label = self._speaker_names.get(sid, f"Speaker {sid}")
            return label, sid

        # PLDA-lite: use whitened embeddings if available
        emb_for_sim = self._whiten(embedding) if self._whiten_matrix is not None else embedding

        # Compare against existing centroids with continuity bias
        scores: list[tuple[float, int]] = []
        for sid, centroid in self._speaker_centroids.items():
            c_for_sim = self._whiten(centroid) if self._whiten_matrix is not None else centroid
            score = self._cosine_sim(emb_for_sim, c_for_sim)
            # Continuity bias: boost the most recent speaker
            if sid == self._last_speaker_id:
                score += self.CONTINUITY_BONUS
            scores.append((score, sid))
        scores.sort(reverse=True)

        best_score = scores[0][0] if scores else -1.0
        best_id = scores[0][1] if scores else None

        # Populate debug info for UI
        self._last_debug = {
            'debug_speakers': len(self._speaker_centroids),
            'debug_rms': f"{rms:.4f}",
            'debug_best_score': f"{best_score:.3f}",
            'debug_scores': [(f"{s:.3f}", sid) for s, sid in scores[:3]],
        }

        # Ambiguity check: if top-2 are within CONFLICT_MARGIN, prefer
        # the speaker with more segments (more established)
        if len(scores) >= 2:
            top_score, top_id = scores[0]
            sec_score, sec_id = scores[1]
            if top_score - sec_score < self.CONFLICT_MARGIN:
                top_count = len(self._segment_embeddings.get(top_id, []))
                sec_count = len(self._segment_embeddings.get(sec_id, []))
                if sec_count > top_count:
                    best_score, best_id = sec_score, sec_id

        # Detect likely overlap: embedding similar to 2+ speakers is probably blended
        is_ambiguous = False
        if len(scores) >= 2:
            gap = scores[0][0] - scores[1][0]
            # Top-2 are very close — could be overlapping speakers or ambiguous
            if gap < self.CONFLICT_MARGIN:
                is_ambiguous = True

        # Store overlap metadata for UI consumption
        overlap_speakers = []
        if is_ambiguous and len(scores) >= 2:
            overlap_speakers = [scores[0][1], scores[1][1]]
        self._last_identify_meta = {
            "is_overlap": is_ambiguous,
            "overlap_speakers": overlap_speakers,
        }

        # Discovery phase: use stricter match threshold during early session
        # to aggressively discover who's in the room, then relax
        in_discovery = self._identify_count < self.DISCOVERY_PHASE_CALLS
        effective_match_threshold = (
            self.MATCH_THRESHOLD_DISCOVERY if in_discovery
            else self.MATCH_THRESHOLD
        )

        # Adaptive new-speaker threshold
        n_existing = len(self._speaker_centroids)
        can_create_new = is_high_quality
        adaptive_new_threshold = self.NEW_SPEAKER_THRESHOLD
        if not in_discovery and n_existing >= 2:
            # Stricter threshold per extra speaker (post-discovery)
            adaptive_new_threshold = max(0.10, self.NEW_SPEAKER_THRESHOLD - 0.05 * (n_existing - 2))
        # After discovery + many speakers: freeze speaker creation
        if not in_discovery and self._next_id > 4:
            total_created = self._next_id - 1
            freeze_threshold = max(0.05, 0.20 - 0.03 * (total_created - 4))
            if best_score > freeze_threshold:
                can_create_new = False

        # Update debug info with effective threshold
        self._last_debug['debug_phase'] = 'discovery' if in_discovery else 'stable'
        self._last_debug['debug_threshold'] = f"{effective_match_threshold:.2f}"

        if best_score >= effective_match_threshold and best_id is not None:
            sid = best_id
            should_update = is_high_quality and not is_ambiguous

            if should_update:
                # Gate: only update centroid if embedding is close enough
                cur_centroid = self._speaker_centroids[sid]
                cur_norm = np.linalg.norm(cur_centroid)
                if cur_norm > 1e-10:
                    sim_to_centroid = self._cosine_sim(embedding, cur_centroid)
                else:
                    sim_to_centroid = 1.0  # first embedding, always accept
                if sim_to_centroid >= self.CENTROID_UPDATE_MIN_SIM:
                    self._prev_centroids[sid] = cur_centroid.copy()
                    # EMA update: centroid = (1-α) * normalized_centroid + α * embedding
                    normed_centroid = cur_centroid / (cur_norm + 1e-10)
                    alpha = self.CENTROID_EMA_ALPHA
                    new_centroid = (1.0 - alpha) * normed_centroid + alpha * embedding
                    # Re-normalize to unit length
                    new_norm = np.linalg.norm(new_centroid)
                    if new_norm > 1e-10:
                        new_centroid = new_centroid / new_norm
                    self._speaker_centroids[sid] = new_centroid
        elif best_score >= adaptive_new_threshold and best_id is not None:
            # Uncertain zone: assign to closest but skip centroid update
            sid = best_id
        elif n_existing >= MAX_SPEAKERS and best_id is not None:
            # Cap reached — assign to closest existing speaker
            sid = best_id
        elif can_create_new:
            # New speaker: below adaptive threshold vs ALL centroids
            sid = self._next_id
            # Store normalized centroid (EMA expects unit-length centroids)
            emb_norm = np.linalg.norm(embedding)
            self._speaker_centroids[sid] = embedding.copy() / (emb_norm + 1e-10)
            idx = (sid - 1) % len(self._color_palette)
            self._speaker_colors[sid] = self._color_palette[idx]
            self._next_id += 1
        else:
            # Low-quality audio, no existing match — assign to last speaker
            sid = self._last_speaker_id
            label = self._speaker_names.get(sid, f"Speaker {sid}")
            return label, sid

        # Retain per-segment embedding for later enrollment (capped to prevent leak)
        duration_sec = len(audio) / sample_rate
        if sid not in self._segment_embeddings:
            self._segment_embeddings[sid] = []
        self._segment_embeddings[sid].append((embedding.copy(), duration_sec))
        if len(self._segment_embeddings[sid]) > self.MAX_EMBEDDINGS_PER_SPEAKER:
            self._segment_embeddings[sid] = self._segment_embeddings[sid][-self.MAX_EMBEDDINGS_PER_SPEAKER:]

        # Track temporal order for HMM smoothing (capped to prevent leak)
        self._segment_order.append((sid, embedding.copy()))
        if len(self._segment_order) > self.MAX_SEGMENT_ORDER:
            self._segment_order = self._segment_order[-self.MAX_SEGMENT_ORDER:]

        self._last_speaker_id = sid

        # Periodic maintenance
        self._identify_count += 1

        # End of discovery phase: re-cluster and refresh centroids to fix
        # contamination from mixed-speaker audio during early session
        if self._identify_count == self.DISCOVERY_PHASE_CALLS:
            log.info("Discovery phase complete (%d speakers). Re-clustering.",
                     len(self._speaker_centroids))
            self._refresh_centroids()
            self._spectral_recluster()

        if self._identify_count % self.RECLUSTER_INTERVAL == 0:
            self._spectral_recluster()
        # Refresh centroids and check merges on every merge cycle
        if self._identify_count % self.MERGE_INTERVAL == 0:
            self._refresh_centroids()
            self._maybe_merge_clusters()

        label = self._speaker_names.get(sid, f"Speaker {sid}")
        return label, sid

    def _extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Extract a 512-dim speaker embedding without clustering logic.

        Alias for _extract_embedding_raw (used by SCD and other callers).
        """
        return self._extract_embedding_raw(audio, sample_rate)

    def _extract_embedding_raw(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Extract a speaker embedding via the active backend.

        Returns None if audio is too short or model not loaded.
        """
        if not self._loaded:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        if self._backend == "onnx" and self._onnx_embedder is not None:
            return self._onnx_embedder.extract(audio, sample_rate)

        # PyTorch path
        if self._model is None:
            return None

        feats = self._compute_fbank(audio, sample_rate)
        if feats is None:
            return None

        import torch
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        embedding = self._model(feats_t).squeeze().cpu().numpy()
        return embedding

    def _compute_fbank(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Compute 80-dim Fbank features with CMN.

        Uses pure-numpy fbank when ONNX backend is active,
        torchaudio when PyTorch backend is active.
        """
        if self._backend == "onnx":
            from .fbank import compute_fbank
            feats = compute_fbank(audio, sample_rate=sample_rate)
            return feats if feats.shape[0] > 0 else None

        import torch
        import torchaudio

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0) * (1 << 15)
        feats = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
            sample_frequency=sample_rate, window_type='hamming',
            use_energy=False,
        )
        feats = feats - feats.mean(dim=0)  # CMN
        return feats.numpy()

    def _extract_weighted_embedding(
        self,
        audio: np.ndarray,
        frame_weights: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Extract embedding with feature-level weighting from segmentation."""
        if not self._loaded:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        # ONNX path: use the embedder's built-in weighted extraction
        if self._backend == "onnx" and self._onnx_embedder is not None:
            return self._onnx_embedder.extract_weighted(audio, frame_weights, sample_rate)

        # PyTorch path
        if self._model is None:
            return None

        feats = self._compute_fbank(audio, sample_rate)
        if feats is None:
            return None

        # Upsample segmentation weights (~17ms) to Fbank frame level (~10ms)
        n_fbank = feats.shape[0]
        seg_dur = 270 / sample_rate
        fbank_dur = 160 / sample_rate
        fbank_weights = np.ones(n_fbank, dtype=np.float32)
        for i in range(n_fbank):
            seg_idx = min(int(i * fbank_dur / seg_dur), len(frame_weights) - 1)
            fbank_weights[i] = frame_weights[seg_idx]

        feats = feats * fbank_weights[:, None]

        import torch
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        embedding = self._model(feats_t).squeeze().cpu().numpy()
        return embedding

    def _extract_overlap_aware(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Extract overlap-aware speaker embedding using segmentation."""
        if self._segmentation is None:
            return None
        activation = self._segmentation.segment(audio)
        active_speakers = self._segmentation.get_active_speakers(activation)
        if not active_speakers:
            return None
        return self._extract_overlap_aware_from_activation(
            audio, activation, active_speakers, sample_rate
        )

    def _extract_overlap_aware_from_activation(
        self,
        audio: np.ndarray,
        activation: np.ndarray,
        active_speakers: list[dict],
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Extract embedding from clean (non-overlapping) frames.

        Uses pre-computed segmentation activation to crop audio to frames
        where only the dominant speaker is active.
        """
        # Find the dominant local speaker (highest mean activation)
        dominant = max(active_speakers, key=lambda s: s["mean_activation"])
        spk_idx = dominant["speaker_idx"]

        # Find frames where ONLY the dominant speaker is active (no overlap)
        n_frames = activation.shape[0]
        frame_samples = 270  # samples per segmentation frame
        clean_chunks: list[np.ndarray] = []

        for f in range(n_frames):
            if activation[f, spk_idx] < 0.5:
                continue
            other_active = any(
                activation[f, j] > 0.3
                for j in range(activation.shape[1])
                if j != spk_idx
            )
            if other_active:
                continue
            s = f * frame_samples
            e = min(s + frame_samples, len(audio))
            if e > s:
                clean_chunks.append(audio[s:e])

        if not clean_chunks:
            return None

        clean_audio = np.concatenate(clean_chunks)
        if len(clean_audio) < _MIN_SPEECH_SAMPLES:
            return None

        return self._extract_embedding_raw(clean_audio, sample_rate)

    def identify_segments(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> list[tuple[str, int, int, int]]:
        """Identify speakers in an audio buffer, splitting at speaker changes.

        Uses sliding-window embedding comparison to detect speaker change
        points, then runs identify() on each sub-segment.

        Returns list of (label, speaker_id, start_sample, end_sample).
        Falls back to single identify() when audio is too short for SCD.
        """
        if not self._loaded:
            log.warning("identify_segments() called but model not loaded — returning Speaker 1")
            return [("Speaker 1", 1, 0, len(audio))]

        # Ensure mono float32
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Too short for sliding-window SCD — fall back to single identify
        if len(audio) < _SCD_MIN_SAMPLES:
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        # 1. Extract embeddings with sliding window
        window_samples = int(self.SCD_WINDOW_SEC * sample_rate)
        hop_samples = int(self.SCD_HOP_SEC * sample_rate)

        embeddings: list[np.ndarray] = []
        positions: list[int] = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            window = audio[start:start + window_samples]
            emb = self._extract_embedding(window, sample_rate)
            if emb is not None:
                embeddings.append(emb)
                positions.append(start)

        if len(embeddings) < 2:
            # Too few valid windows — fall back to single identify
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        # 2. Find speaker change points via cosine distance between consecutive windows
        change_points = [0]
        for i in range(1, len(embeddings)):
            dist = 1.0 - self._cosine_sim(embeddings[i], embeddings[i - 1])
            if dist > self.SCD_CHANGE_THRESHOLD:
                change_points.append(positions[i])
        change_points.append(len(audio))

        # 3. Merge short segments with neighbors to ensure min duration
        # Always start at 0 and end at len(audio)
        merged_points = [0]
        for cp in change_points[1:-1]:  # skip first (0) and last (len(audio))
            if cp - merged_points[-1] >= _MIN_SPEECH_SAMPLES:
                merged_points.append(cp)
            # else: skip this change point (absorb into current segment)
        merged_points.append(len(audio))

        # 4. Identify each segment
        results: list[tuple[str, int, int, int]] = []
        for i in range(len(merged_points) - 1):
            seg_start = merged_points[i]
            seg_end = merged_points[i + 1]
            seg_audio = audio[seg_start:seg_end]
            if len(seg_audio) >= _MIN_SPEECH_SAMPLES:
                label, sid = self.identify(seg_audio, sample_rate)
                results.append((label, sid, seg_start, seg_end))

        if not results:
            # All sub-segments too short — fall back to full-buffer identify
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        return results

    def identify_multi(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> list[tuple[str, int, int, int]]:
        """Identify ALL active speakers in an audio chunk using segmentation.

        Unlike identify() which returns one speaker, this method detects
        overlapping speakers and returns a segment for each active speaker,
        including overlapping time regions.

        Returns list of (label, speaker_id, start_sample, end_sample).
        Multiple entries can cover the same time range (= overlap).
        Falls back to single identify() when segmentation is unavailable.
        """
        if not self._loaded:
            return [("Speaker 1", 1, 0, len(audio))]

        if audio.ndim > 1:
            audio = audio[:, 0]

        if self._segmentation is None or len(audio) < _MIN_SPEECH_SAMPLES:
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        # Run segmentation to get per-frame, per-speaker activation
        activation = self._segmentation.segment(audio)
        active_speakers = self._segmentation.get_active_speakers(activation)

        if not active_speakers:
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        # For each active local speaker, extract a clean embedding
        # from their solo frames (no overlap)
        frame_samples = 270  # segmentation frame step
        local_data: list[dict] = []  # per local speaker: embedding, frames, etc.

        for spk_info in active_speakers:
            spk_idx = spk_info["speaker_idx"]

            active_frames: list[int] = []
            solo_chunks: list[np.ndarray] = []

            for f in range(activation.shape[0]):
                if activation[f, spk_idx] < 0.5:
                    continue
                active_frames.append(f)
                other_active = any(
                    activation[f, j] > 0.3
                    for j in range(activation.shape[1])
                    if j != spk_idx
                )
                if not other_active:
                    s = f * frame_samples
                    e = min(s + frame_samples, len(audio))
                    if e > s:
                        solo_chunks.append(audio[s:e])

            if not active_frames:
                continue

            # Extract embedding using feature-level weighting:
            # Compute Fbank features for full audio, then weight by
            # this speaker's activation (overlap frames get low weight)
            emb = self._extract_weighted_embedding(
                audio, activation[:, spk_idx], sample_rate
            )
            if emb is None:
                emb = self._extract_embedding_raw(audio, sample_rate)
            if emb is None:
                continue

            local_data.append({
                "spk_idx": spk_idx,
                "embedding": emb,
                "active_frames": active_frames,
            })

        if not local_data:
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        # One-to-one assignment: each local speaker maps to a DIFFERENT global speaker
        # Sort by embedding quality (more solo frames = better embedding)
        # Greedy: best match first, remove used global speakers
        used_global: set[int] = set()
        results: list[tuple[str, int, int, int]] = []

        # Compute all scores: (local_idx, global_sid, score)
        all_scores: list[tuple[int, int, float]] = []
        for li, ld in enumerate(local_data):
            for sid, centroid in self._speaker_centroids.items():
                score = float(self._cosine_sim(ld["embedding"], centroid))
                all_scores.append((li, sid, score))
        all_scores.sort(key=lambda x: -x[2])  # best scores first

        # Greedy one-to-one matching
        assigned_local: set[int] = set()
        local_to_global: dict[int, int] = {}

        for li, sid, score in all_scores:
            if li in assigned_local or sid in used_global:
                continue
            if score >= self.MATCH_THRESHOLD:
                local_to_global[li] = sid
                assigned_local.add(li)
                used_global.add(sid)

        # Unmatched local speakers: create new global speakers
        for li, ld in enumerate(local_data):
            if li in assigned_local:
                continue
            if len(self._speaker_centroids) < MAX_SPEAKERS:
                sid = self._next_id
                emb_n = np.linalg.norm(ld["embedding"])
                self._speaker_centroids[sid] = ld["embedding"].copy() / (emb_n + 1e-10)
                idx = (sid - 1) % len(self._color_palette)
                self._speaker_colors[sid] = self._color_palette[idx]
                self._next_id += 1
                local_to_global[li] = sid
                assigned_local.add(li)
                used_global.add(sid)
            else:
                # Assign to closest unused global speaker
                best_s, best_sid = -1.0, None
                for sid, centroid in self._speaker_centroids.items():
                    if sid in used_global:
                        continue
                    sc = float(self._cosine_sim(ld["embedding"], centroid))
                    if sc > best_s:
                        best_s, best_sid = sc, sid
                if best_sid is not None:
                    local_to_global[li] = best_sid
                    assigned_local.add(li)
                    used_global.add(best_sid)

        # Build results
        for li, ld in enumerate(local_data):
            sid = local_to_global.get(li)
            if sid is None:
                continue
            start_frame = min(ld["active_frames"])
            end_frame = max(ld["active_frames"]) + 1
            start_sample = start_frame * frame_samples
            end_sample = min(end_frame * frame_samples, len(audio))
            label = self._speaker_names.get(sid, f"Speaker {sid}")
            results.append((label, sid, start_sample, end_sample))

            # Store embedding for this speaker (capped to prevent leak)
            duration_sec = (end_sample - start_sample) / sample_rate
            if sid not in self._segment_embeddings:
                self._segment_embeddings[sid] = []
            self._segment_embeddings[sid].append((ld["embedding"].copy(), duration_sec))
            if len(self._segment_embeddings[sid]) > self.MAX_EMBEDDINGS_PER_SPEAKER:
                self._segment_embeddings[sid] = self._segment_embeddings[sid][-self.MAX_EMBEDDINGS_PER_SPEAKER:]
            self._segment_order.append((sid, ld["embedding"].copy()))
        # Cap segment order after processing all local speakers
        if len(self._segment_order) > self.MAX_SEGMENT_ORDER:
            self._segment_order = self._segment_order[-self.MAX_SEGMENT_ORDER:]

        if not results:
            label, sid = self.identify(audio, sample_rate)
            return [(label, sid, 0, len(audio))]

        self._identify_count += 1
        if self._identify_count % self.RECLUSTER_INTERVAL == 0:
            self._spectral_recluster()
            self._refresh_centroids()
        elif self._identify_count % self.MERGE_INTERVAL == 0:
            self._maybe_merge_clusters()

        return results

    def get_speaker_color(self, speaker_id: int) -> str:
        """Return the hex colour assigned to a speaker."""
        return self._speaker_colors.get(
            speaker_id,
            self._color_palette[0],
        )

    @property
    def num_speakers(self) -> int:
        return len(self._speaker_centroids)

    # ── speaker names ──────────────────────────────────────

    def set_speaker_name(self, speaker_id: int, name: str) -> None:
        """Assign a custom name to a session speaker."""
        self._speaker_names[speaker_id] = name

    def get_speaker_name(self, speaker_id: int) -> str:
        """Return the custom name or default 'Speaker N'."""
        return self._speaker_names.get(speaker_id, f"Speaker {speaker_id}")

    def get_speaker_names(self) -> dict[int, str]:
        """Return all custom speaker name mappings."""
        return dict(self._speaker_names)

    def get_segment_embeddings(self, speaker_id: int) -> list[tuple[np.ndarray, float]]:
        """Return retained (embedding, duration) pairs for a session speaker."""
        return list(self._segment_embeddings.get(speaker_id, []))

    def get_all_session_speakers(self) -> dict[int, int]:
        """Return {speaker_id: segment_count} for all session speakers."""
        return {
            sid: len(embs) for sid, embs in self._segment_embeddings.items()
        }

    def get_session_centroid(self, speaker_id: int) -> np.ndarray | None:
        """Return the current session centroid for a speaker (L2-normalized)."""
        c = self._speaker_centroids.get(speaker_id)
        if c is None:
            return None
        norm = float(np.linalg.norm(c))
        if norm < 1e-10:
            return c.copy()
        return c / norm

    def is_speaker_stable(self, speaker_id: int) -> bool:
        """Check if a session speaker's centroid has stabilized.

        Stable when >= 3 segments AND centroid movement < 0.05 cosine distance.
        """
        seg_count = len(self._segment_embeddings.get(speaker_id, []))
        if seg_count < 3:
            return False
        prev = self._prev_centroids.get(speaker_id)
        curr = self._speaker_centroids.get(speaker_id)
        if prev is None or curr is None:
            return False
        delta = 1.0 - self._cosine_sim(prev, curr)
        return delta < 0.05

    def mark_matched(self, speaker_id: int) -> None:
        """Mark a speaker as already cross-session matched (skip future matching)."""
        self._matched_speakers.add(speaker_id)

    def is_matched(self, speaker_id: int) -> bool:
        """Check if a speaker has already been cross-session matched."""
        return speaker_id in self._matched_speakers

    def merge_speakers(self, source_id: int, target_id: int) -> None:
        """Merge source speaker into target within the current session."""
        # Count segments before moving (for weighted centroid merge)
        target_n = len(self._segment_embeddings.get(target_id, []))
        source_embs = self._segment_embeddings.pop(source_id, [])
        source_n = len(source_embs)

        # Move embeddings
        if target_id not in self._segment_embeddings:
            self._segment_embeddings[target_id] = []
        self._segment_embeddings[target_id].extend(source_embs)
        if len(self._segment_embeddings[target_id]) > self.MAX_EMBEDDINGS_PER_SPEAKER:
            self._segment_embeddings[target_id] = self._segment_embeddings[target_id][
                -self.MAX_EMBEDDINGS_PER_SPEAKER:
            ]

        # Merge centroids: weighted average by segment count, re-normalize
        target_c = self._speaker_centroids.get(target_id)
        source_c = self._speaker_centroids.pop(source_id, None)
        if target_c is not None and source_c is not None:
            total = target_n + source_n
            if total > 0:
                merged = (target_n * target_c + source_n * source_c) / total
            else:
                merged = (target_c + source_c) / 2.0
            norm = np.linalg.norm(merged)
            if norm > 1e-10:
                merged = merged / norm
            self._speaker_centroids[target_id] = merged

        # Clean up source state
        self._speaker_colors.pop(source_id, None)
        self._speaker_names.pop(source_id, None)
        self._prev_centroids.pop(source_id, None)
        self._matched_speakers.discard(source_id)

    # ── session management ────────────────────────────────

    def reset_session(self):
        """Clear all session speakers (for a new conversation)."""
        self._speaker_centroids.clear()
        self._speaker_colors.clear()
        self._speaker_names.clear()
        self._segment_embeddings.clear()
        self._prev_centroids.clear()
        self._matched_speakers.clear()
        self._next_id = 1
        self._last_speaker_id = 1
        self._identify_count = 0
        self._segment_order.clear()
        self._whiten_matrix = None
        self._whiten_mean = None

    # ── internals ─────────────────────────────────────────

    def _whiten(self, embedding: np.ndarray) -> np.ndarray:
        """Apply PLDA-lite whitening transform to an embedding."""
        if self._whiten_matrix is None or self._whiten_mean is None:
            return embedding
        return (embedding - self._whiten_mean) @ self._whiten_matrix

    def _update_whitening(self) -> None:
        """Estimate within-speaker covariance and compute whitening transform.

        PLDA-lite: uses the session's own embeddings to estimate within-speaker
        scatter.  After whitening, cosine similarity better separates speakers
        because within-speaker variability becomes isotropic.
        """
        # Need enough data from multiple speakers
        speakers_with_data = {
            sid: embs for sid, embs in self._segment_embeddings.items()
            if len(embs) >= 2
        }
        total = sum(len(e) for e in speakers_with_data.values())
        if len(speakers_with_data) < 2 or total < self.WHITEN_MIN_SEGMENTS:
            return

        # Compute within-speaker scatter matrix
        dim = None
        scatter = None
        count = 0
        for sid, emb_list in speakers_with_data.items():
            embs = np.stack([e for e, _d in emb_list])
            if dim is None:
                dim = embs.shape[1]
                scatter = np.zeros((dim, dim), dtype=np.float64)
            mean = embs.mean(axis=0)
            centered = embs - mean
            scatter += centered.T @ centered
            count += len(embs)

        if count < 2 or scatter is None or dim is None:
            return
        scatter /= count

        # Regularize to prevent singular matrix
        scatter += np.eye(dim) * 1e-4

        # Whitening: W = Σ^{-1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(scatter)
        eigvals = np.maximum(eigvals, 1e-6)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Compute global mean for centering
        all_embs = []
        for emb_list in self._segment_embeddings.values():
            for emb, _d in emb_list:
                all_embs.append(emb)
        global_mean = np.mean(np.stack(all_embs), axis=0)

        self._whiten_matrix = W.astype(np.float32)
        self._whiten_mean = global_mean.astype(np.float32)

    def _refresh_centroids(self) -> None:
        """Recompute centroids from high-confidence embeddings to combat drift.

        Only uses embeddings that are self-consistent (similar to the
        speaker's other embeddings), filtering out noise/overlap contamination.
        """
        for sid in list(self._speaker_centroids.keys()):
            emb_list = self._segment_embeddings.get(sid, [])
            if len(emb_list) < 3:
                continue
            # Use the most recent embeddings (up to 30)
            recent = [e for e, _d in emb_list[-30:]]
            embs = np.stack(recent)

            # Compute mean embedding direction
            mean_emb = embs.mean(axis=0)
            mean_norm = np.linalg.norm(mean_emb)
            if mean_norm < 1e-10:
                continue
            mean_emb /= mean_norm

            # Filter: keep only embeddings within 0.3 cosine sim of the mean
            # This removes outliers (overlap contamination, noise)
            filtered = []
            for emb in recent:
                sim = float(self._cosine_sim(emb, mean_emb))
                if sim > 0.3:
                    filtered.append(emb)

            if len(filtered) < 2:
                continue

            # New centroid = normalized mean of filtered embeddings
            new_centroid = np.stack(filtered).mean(axis=0)
            norm = np.linalg.norm(new_centroid)
            if norm > 1e-10:
                new_centroid = new_centroid / norm
            self._speaker_centroids[sid] = new_centroid

    def _viterbi_smooth(self, labels: list[int], k: int) -> list[int]:
        """Apply VBx-style Viterbi smoothing with speaker continuity prior.

        Uses a transition matrix with loopP self-transition probability
        to discourage rapid speaker switching in the label sequence.
        """
        n = len(labels)
        if n < 2 or k < 2:
            return labels

        loop_p = self.LOOP_PROB
        switch_p = (1.0 - loop_p) / (k - 1)

        # Log transition matrix
        log_trans = np.full((k, k), np.log(switch_p + 1e-30))
        np.fill_diagonal(log_trans, np.log(loop_p))

        # Emission: soft assignment from k-means labels (1.0 for assigned, 0.1 for others)
        log_emit = np.full((n, k), np.log(0.1))
        for i, lbl in enumerate(labels):
            log_emit[i, lbl] = np.log(0.9)

        # Viterbi forward pass
        log_prior = np.full(k, np.log(1.0 / k))
        V = np.zeros((n, k))
        backptr = np.zeros((n, k), dtype=int)
        V[0] = log_prior + log_emit[0]

        for t in range(1, n):
            for j in range(k):
                scores = V[t - 1] + log_trans[:, j]
                backptr[t, j] = int(np.argmax(scores))
                V[t, j] = scores[backptr[t, j]] + log_emit[t, j]

        # Backtrack
        smoothed = [0] * n
        smoothed[-1] = int(np.argmax(V[-1]))
        for t in range(n - 2, -1, -1):
            smoothed[t] = backptr[t + 1, smoothed[t + 1]]

        return smoothed

    def _spectral_recluster(self) -> None:
        """Re-cluster all session embeddings using 3D-Speaker algorithms.

        Uses auto_cluster() which selects AHC for small sample counts (< 40)
        and spectral clustering with p-value pruning for larger sets.
        Runs every RECLUSTER_INTERVAL identify() calls.
        """
        from config import (
            CLUSTER_AHC_MAX_SAMPLES, CLUSTER_AHC_THRESHOLD,
            CLUSTER_SPECTRAL_PVAL_BETA,
        )
        from .cluster import auto_cluster

        # Collect all segment embeddings with their speaker assignments
        all_embs: list[np.ndarray] = []
        seg_speaker_ids: list[int] = []
        for sid, emb_list in self._segment_embeddings.items():
            for emb, _dur in emb_list:
                all_embs.append(emb)
                seg_speaker_ids.append(sid)

        n = len(all_embs)
        if n < self.RECLUSTER_MIN_SEGMENTS:
            return

        current_speakers = list(self._speaker_centroids.keys())
        if len(current_speakers) < 2:
            return

        X = np.stack(all_embs)  # (N, embed_dim)

        # Run 3D-Speaker clustering (auto-selects AHC vs spectral)
        labels = auto_cluster(
            X,
            max_speakers=min(MAX_SPEAKERS, len(current_speakers)),
            threshold=CLUSTER_AHC_THRESHOLD,
            p_value_beta=CLUSTER_SPECTRAL_PVAL_BETA,
            ahc_max_samples=CLUSTER_AHC_MAX_SAMPLES,
        )

        # VBx-style HMM smoothing: apply Viterbi with loopP to reduce rapid switching
        k = int(labels.max()) + 1
        if k < 2:
            return
        if k >= len(current_speakers):
            return  # clustering says we have the right count (or more)
        if len(self._segment_order) >= n:
            labels = self._viterbi_smooth(labels.tolist(), k)

        # Build mapping: new_label → set of original speaker_ids
        label_to_sids: dict[int, set[int]] = {}
        for i, lbl in enumerate(labels):
            label_to_sids.setdefault(lbl, set()).add(seg_speaker_ids[i])

        # For each new cluster that spans multiple old speakers, merge them
        for _lbl, sids_in_cluster in label_to_sids.items():
            if len(sids_in_cluster) < 2:
                continue
            # Keep the speaker with the most segments as target
            sid_list = sorted(
                sids_in_cluster,
                key=lambda s: len(self._segment_embeddings.get(s, [])),
                reverse=True,
            )
            target = sid_list[0]
            for source in sid_list[1:]:
                if source in self._speaker_centroids:
                    self.merge_speakers(source, target)

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iter: int = 20) -> list[int]:
        """Simple k-means clustering (numpy only, no scipy dependency)."""
        n = X.shape[0]
        # Initialize centroids via k-means++ seeding
        rng = np.random.RandomState(42)
        centroids = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dists / (dists.sum() + 1e-10)
            centroids.append(X[rng.choice(n, p=probs)])
        centroids_arr = np.stack(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.stack(
                [np.sum((X - c) ** 2, axis=1) for c in centroids_arr]
            )  # (k, n)
            new_labels = dists.argmin(axis=0)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids_arr[j] = X[mask].mean(axis=0)

        return labels.tolist()

    def _maybe_merge_clusters(self) -> None:
        """Merge speaker clusters whose centroids are too similar.

        Compares all centroid pairs; merges the smaller cluster into the larger
        when cosine similarity exceeds MERGE_THRESHOLD.
        """
        sids = list(self._speaker_centroids.keys())
        if len(sids) < 2:
            return

        # Find the most similar pair
        best_sim, best_pair = -1.0, None
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                sim = self._cosine_sim(
                    self._speaker_centroids[sids[i]],
                    self._speaker_centroids[sids[j]],
                )
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (sids[i], sids[j])

        if best_sim < self.MERGE_THRESHOLD or best_pair is None:
            return

        # Merge smaller cluster into larger (by segment count)
        a, b = best_pair
        count_a = len(self._segment_embeddings.get(a, []))
        count_b = len(self._segment_embeddings.get(b, []))
        if count_a >= count_b:
            target, source = a, b
        else:
            target, source = b, a
        self.merge_speakers(source, target)

    @staticmethod
    def _trim_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
        """Trim leading/trailing silence from audio."""
        window = 1600  # 100 ms
        n = len(audio)
        if n < window * 4:
            return audio

        start, end = 0, n
        for i in range(0, n - window, window):
            if np.sqrt(np.mean(audio[i:i + window] ** 2)) > threshold:
                start = i
                break
        for i in range(n - window, window, -window):
            if np.sqrt(np.mean(audio[i:i + window] ** 2)) > threshold:
                end = i + window
                break

        if end > start:
            return audio[start:end]
        return audio

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class _MockEmbeddingModel:
    """Lightweight stand-in for speaker embedding model (testing only).

    Returns deterministic embeddings derived from the audio content
    so that different audio produces different speakers.
    """

    def __call__(self, feats):
        import torch
        from config import SPEAKER_EMBEDDING_DIM
        # Derive embedding from feature content for deterministic but varied results
        feat_np = feats.squeeze().numpy()
        rng = np.random.RandomState(int(abs(feat_np[:100].sum()) * 1000) % 2**31)
        emb = rng.randn(SPEAKER_EMBEDDING_DIM).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        return torch.tensor(emb).unsqueeze(0)
