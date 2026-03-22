"""Pluggable speaker embedding backends for diarization.

Each backend wraps a different speaker encoder model and exposes a uniform
interface: load the model, extract a fixed-dim embedding from raw audio.

Usage:
    backend = get_backend("campplus")
    backend.load()
    embedding = backend.extract(audio_float32, sample_rate=16000)
    print(backend.embed_dim)  # e.g. 512
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class EmbeddingBackend(abc.ABC):
    """Abstract base for speaker embedding extractors."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'campplus', 'ecapa_tdnn')."""

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the output embedding vector."""

    @abc.abstractmethod
    def load(self) -> None:
        """Download (if needed) and load the model. May block for seconds."""

    @abc.abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Extract a speaker embedding from raw mono float32 audio.

        Returns an L2-normalized numpy array of shape (embed_dim,), or None
        if the audio is too short / model not loaded.
        """


# ── registry ───────────────────────────────────────────

_BACKENDS: dict[str, type[EmbeddingBackend]] = {}

# Friendly names for display
BACKEND_INFO: dict[str, dict] = {
    "campplus": {
        "label": "CAM++ (WeSpeaker)",
        "dim": 512,
        "size": "~28 MB",
        "package": "torch, torchaudio",
    },
    "ecapa_tdnn": {
        "label": "ECAPA-TDNN (SpeechBrain)",
        "dim": 192,
        "size": "~60 MB",
        "package": "speechbrain",
    },
    "titanet": {
        "label": "TitaNet (NVIDIA NeMo)",
        "dim": 192,
        "size": "~100 MB",
        "package": "nemo_toolkit[asr]",
    },
    "resemblyzer": {
        "label": "GE2E d-vector (Resemblyzer)",
        "dim": 256,
        "size": "~17 MB",
        "package": "resemblyzer",
    },
    "pyannote": {
        "label": "WeSpeaker (pyannote.audio)",
        "dim": 512,
        "size": "~200 MB",
        "package": "pyannote.audio",
    },
}


def register_backend(name: str, cls: type[EmbeddingBackend]) -> None:
    _BACKENDS[name] = cls


def get_backend(name: str) -> EmbeddingBackend:
    """Instantiate an embedding backend by name.

    Raises ValueError if the backend is unknown or its dependencies are missing.
    """
    # Lazy-import all backend modules so registration happens on demand
    _ensure_registered()

    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown diarization backend '{name}'. Available: {available}"
        )
    return _BACKENDS[name]()


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    _ensure_registered()
    return sorted(_BACKENDS.keys())


_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    # Import each backend module — they call register_backend() at import time
    from diarization.backends import campplus_backend  # noqa: F401
    from diarization.backends import speechbrain_backend  # noqa: F401
    from diarization.backends import nemo_backend  # noqa: F401
    from diarization.backends import resemblyzer_backend  # noqa: F401
    from diarization.backends import pyannote_backend  # noqa: F401
