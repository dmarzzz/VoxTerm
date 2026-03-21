"""Speaker diarization via ECAPA-TDNN embeddings + online cosine clustering.


Identifies the dominant speaker in each audio chunk by extracting a speaker
embedding and comparing it to a running set of speaker centroids.  New
speakers are created automatically when similarity falls below a threshold.

Requires: speechbrain (pip install speechbrain)
Model:    speechbrain/spkrec-ecapa-voxceleb (~20 MB, downloads on first use)
"""

from __future__ import annotations

import numpy as np

_MIN_SPEECH_SAMPLES = 32000   # 2.0 s at 16 kHz — shorter → unreliable embeddings
MAX_SPEAKERS = 8              # hard cap on simultaneous speaker clusters


class DiarizationEngine:
    """Online speaker identification using ECAPA-TDNN embeddings."""

    SIMILARITY_THRESHOLD = 0.30   # cosine sim above this → same speaker
    EMA_ALPHA = 0.95              # centroid update rate (higher = slower adapt)

    def __init__(self):
        self._model = None
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

    def load(self):
        """Load the ECAPA-TDNN speaker encoder (blocks, ~2-5 s)."""
        import os
        if os.environ.get("VOXTERM_MOCK_ENGINE"):
            self._model = _MockEcapaModel()
            self._loaded = True
            return

        import torch
        # Keep torch on CPU to avoid Metal/GPU conflicts with MLX
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        # Single-threaded inference — prevents OpenMP/MKL threads from
        # conflicting with MLX's Metal runtime (common segfault cause)
        torch.set_num_threads(1)

        # Fix torchaudio compat: newer versions removed list_audio_backends
        try:
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                torchaudio.list_audio_backends = lambda: ["default"]
        except ImportError:
            pass

        # Fix huggingface_hub compat: newer versions removed 'use_auth_token'
        # and newer speechbrain looks for 'custom.py' which old model repos lack
        import huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
        _orig_download = huggingface_hub.hf_hub_download
        def _patched_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            try:
                return _orig_download(*args, **kwargs)
            except EntryNotFoundError:
                # Model repo lacks custom.py — that's fine, return None
                return None
        huggingface_hub.hf_hub_download = _patched_download

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise RuntimeError(
                "speechbrain is required for speaker diarization. "
                "Install it with:  pip install speechbrain"
            )

        cache_dir = __import__("pathlib").Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
        # Use local cache as source if available (avoids HF download issues)
        if (cache_dir / "hyperparams.yaml").exists():
            source = str(cache_dir)
        else:
            source = "speechbrain/spkrec-ecapa-voxceleb"

        self._model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(cache_dir),
            run_opts={"device": "cpu"},
        )
        # Warm up to ensure model is fully initialized
        dummy = torch.zeros(1, 16000)
        self._model.encode_batch(dummy)
        del dummy
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── speaker identification ────────────────────────────

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, int]:
        """Identify the dominant speaker in an audio chunk.

        Returns (label, speaker_id):
            label      – custom name or "Speaker 1", "Speaker 2", …
            speaker_id – integer key (1-based)
        """
        if not self._loaded or self._model is None:
            return "Speaker 1", 1

        import torch

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

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # Extract 192-dim embedding on CPU
        embedding = self._model.encode_batch(waveform).squeeze().cpu().numpy()
        del waveform

        # Compare against existing centroids
        best_score, best_id = -1.0, None
        for sid, centroid in self._speaker_centroids.items():
            score = self._cosine_sim(embedding, centroid)
            if score > best_score:
                best_score, best_id = score, sid

        if best_score >= self.SIMILARITY_THRESHOLD and best_id is not None:
            # Save previous centroid for stabilization tracking
            self._prev_centroids[best_id] = self._speaker_centroids[best_id].copy()
            # Update centroid with exponential moving average
            self._speaker_centroids[best_id] = (
                self.EMA_ALPHA * self._speaker_centroids[best_id]
                + (1 - self.EMA_ALPHA) * embedding
            )
            sid = best_id
        elif len(self._speaker_centroids) >= MAX_SPEAKERS and best_id is not None:
            # Cap reached — assign to closest existing speaker
            sid = best_id
        else:
            # New speaker
            sid = self._next_id
            self._speaker_centroids[sid] = embedding
            idx = (sid - 1) % len(self._color_palette)
            self._speaker_colors[sid] = self._color_palette[idx]
            self._next_id += 1

        # Retain per-segment embedding for later enrollment
        duration_sec = len(audio) / sample_rate
        if sid not in self._segment_embeddings:
            self._segment_embeddings[sid] = []
        self._segment_embeddings[sid].append((embedding.copy(), duration_sec))

        self._last_speaker_id = sid
        label = self._speaker_names.get(sid, f"Speaker {sid}")
        return label, sid

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
        """Return the current session centroid for a speaker."""
        return self._speaker_centroids.get(speaker_id)

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
        # Move embeddings
        source_embs = self._segment_embeddings.pop(source_id, [])
        if target_id not in self._segment_embeddings:
            self._segment_embeddings[target_id] = []
        self._segment_embeddings[target_id].extend(source_embs)

        # Merge centroids
        target_c = self._speaker_centroids.get(target_id)
        source_c = self._speaker_centroids.pop(source_id, None)
        if target_c is not None and source_c is not None:
            merged = (target_c + source_c) / 2.0
            norm = float(np.linalg.norm(merged))
            if norm > 1e-10:
                merged /= norm
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

    # ── internals ─────────────────────────────────────────

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


class _MockEcapaModel:
    """Lightweight stand-in for SpeechBrain EncoderClassifier (testing only).

    Returns deterministic 192-dim embeddings derived from the audio content
    so that different audio produces different speakers.
    """

    def encode_batch(self, waveform):
        import torch
        # Derive embedding from audio content for deterministic but varied results
        audio_np = waveform.squeeze().numpy()
        rng = np.random.RandomState(int(abs(audio_np[:100].sum()) * 1000) % 2**31)
        emb = rng.randn(192).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        return torch.tensor(emb).unsqueeze(0).unsqueeze(0)
