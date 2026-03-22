"""Resemblyzer GE2E d-vector speaker embedding backend.

256-dim embeddings, ~17 MB model. Uses the GE2E (Generalized End-to-End)
LSTM-based d-vector model. Very lightweight.

Install: pip install resemblyzer
"""

from __future__ import annotations

import numpy as np

from diarization.backends import EmbeddingBackend, register_backend

_MIN_SPEECH_SAMPLES = 24000  # 1.5 s at 16 kHz


class ResemblyzerBackend(EmbeddingBackend):

    @property
    def name(self) -> str:
        return "resemblyzer"

    @property
    def embed_dim(self) -> int:
        return 256

    def __init__(self) -> None:
        self._encoder = None
        self._loaded = False

    def load(self) -> None:
        try:
            from resemblyzer import VoiceEncoder
        except ImportError:
            raise ImportError(
                "Resemblyzer backend requires: pip install resemblyzer"
            )

        self._encoder = VoiceEncoder(device="cpu")
        self._loaded = True

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if not self._loaded or self._encoder is None:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        from resemblyzer import preprocess_wav

        # Resemblyzer expects float64 at 16 kHz
        wav = audio.astype(np.float64)

        # Resample if needed
        if sample_rate != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)

        wav = preprocess_wav(wav, source_sr=16000)

        embedding = self._encoder.embed_utterance(wav)

        # L2 normalize (resemblyzer already normalizes, but be safe)
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding.astype(np.float32)


register_backend("resemblyzer", ResemblyzerBackend)
