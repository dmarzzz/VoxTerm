"""SpeechBrain ECAPA-TDNN speaker embedding backend.

192-dim embeddings, ~60 MB model. Uses the pretrained SpeechBrain
ECAPA-TDNN model trained on VoxCeleb1+2.

Install: pip install speechbrain
"""

from __future__ import annotations

import numpy as np

from diarization.backends import EmbeddingBackend, register_backend

_MIN_SPEECH_SAMPLES = 24000  # 1.5 s at 16 kHz


class SpeechBrainECAPABackend(EmbeddingBackend):

    @property
    def name(self) -> str:
        return "ecapa_tdnn"

    @property
    def embed_dim(self) -> int:
        return 192

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "SpeechBrain ECAPA-TDNN backend requires: pip install speechbrain"
            )

        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(
                __import__("pathlib").Path.home()
                / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
            ),
            run_opts={"device": "cpu"},
        )
        self._loaded = True

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if not self._loaded or self._model is None:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        import torch

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # Resample if needed (SpeechBrain expects 16 kHz)
        if sample_rate != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        embedding = self._model.encode_batch(waveform).squeeze().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding


register_backend("ecapa_tdnn", SpeechBrainECAPABackend)
