"""Speaker diarization via ECAPA-TDNN embeddings + online cosine clustering.

Identifies the dominant speaker in each audio chunk by extracting a speaker
embedding and comparing it to a running set of speaker centroids.  New
speakers are created automatically when similarity falls below a threshold.

Requires: speechbrain (pip install speechbrain)
Model:    speechbrain/spkrec-ecapa-voxceleb (~20 MB, downloads on first use)
"""

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
        import torch
        # Keep torch on CPU to avoid Metal/GPU conflicts with MLX
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)

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
            label      – "Speaker 1", "Speaker 2", …
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
            return f"Speaker {sid}", sid

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

        self._last_speaker_id = sid
        return f"Speaker {sid}", sid

    def get_speaker_color(self, speaker_id: int) -> str:
        """Return the hex colour assigned to a speaker."""
        return self._speaker_colors.get(
            speaker_id,
            self._color_palette[0],
        )

    @property
    def num_speakers(self) -> int:
        return len(self._speaker_centroids)

    # ── session management ────────────────────────────────

    def reset_session(self):
        """Clear all session speakers (for a new conversation)."""
        self._speaker_centroids.clear()
        self._speaker_colors.clear()
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
