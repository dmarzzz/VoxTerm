"""NVIDIA NeMo TitaNet speaker embedding backend.

192-dim embeddings, ~100 MB model. Uses the pretrained TitaNet-Large
model from NVIDIA NeMo, trained on VoxCeleb1+2.

Install: pip install nemo_toolkit[asr]
"""

from __future__ import annotations

import numpy as np

from diarization.backends import EmbeddingBackend, register_backend

_MIN_SPEECH_SAMPLES = 24000  # 1.5 s at 16 kHz


class NeMoTitaNetBackend(EmbeddingBackend):

    @property
    def name(self) -> str:
        return "titanet"

    @property
    def embed_dim(self) -> int:
        return 192

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> None:
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError:
            raise ImportError(
                "NeMo TitaNet backend requires: pip install nemo_toolkit[asr]"
            )

        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        self._model = EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
        self._model.eval()
        self._model = self._model.cpu()
        self._loaded = True

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if not self._loaded or self._model is None:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        import torch

        # NeMo expects (batch, samples) at 16 kHz
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        if sample_rate != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)

        # get_embedding returns (batch, embed_dim)
        _, embedding = self._model.forward(
            input_signal=waveform, input_signal_length=lengths
        )
        embedding = embedding.squeeze().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding


register_backend("titanet", NeMoTitaNetBackend)
