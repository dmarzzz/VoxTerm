"""pyannote.audio speaker embedding backend.

512-dim embeddings using pyannote's bundled WeSpeaker model, ~200 MB.
This uses pyannote's embedding extraction pipeline (not the full
diarization pipeline).

Install: pip install pyannote.audio
Note: Some pyannote models require a Hugging Face token. Set
HF_TOKEN or HUGGING_FACE_HUB_TOKEN env var if needed.
"""

from __future__ import annotations

import numpy as np

from diarization.backends import EmbeddingBackend, register_backend

_MIN_SPEECH_SAMPLES = 24000  # 1.5 s at 16 kHz


class PyannoteEmbeddingBackend(EmbeddingBackend):

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def embed_dim(self) -> int:
        return 512

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> None:
        try:
            from pyannote.audio import Model, Inference
        except ImportError:
            raise ImportError(
                "pyannote embedding backend requires: pip install pyannote.audio"
            )

        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        # pyannote/wespeaker-voxceleb-resnet34-LM is freely available
        model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        )
        self._model = Inference(model, window="whole")
        self._loaded = True

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if not self._loaded or self._model is None:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        import torch

        # pyannote Inference expects a dict with "waveform" and "sample_rate"
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, samples) — batch=1, channel=1
        inp = {"waveform": waveform.squeeze(0), "sample_rate": sample_rate}

        embedding = self._model(inp)

        # embedding is a numpy array of shape (embed_dim,)
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding


register_backend("pyannote", PyannoteEmbeddingBackend)
