"""CAM++ speaker embedding backend (WeSpeaker, VoxCeleb-trained).

This is the original/default backend used by VoxTerm.
512-dim embeddings, ~28 MB model, runs on CPU via PyTorch.
"""

from __future__ import annotations

import numpy as np

from diarization.backends import EmbeddingBackend, register_backend

_MIN_SPEECH_SAMPLES = 24000  # 1.5 s at 16 kHz

_MODEL_URL = (
    "https://modelscope.cn/models/"
    "iic/speech_campplus_sv_en_voxceleb_16k/resolve/master/"
    "campplus_voxceleb.bin"
)


class CAMPPlusBackend(EmbeddingBackend):

    @property
    def name(self) -> str:
        return "campplus"

    @property
    def embed_dim(self) -> int:
        return 512

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> None:
        import os
        if os.environ.get("VOXTERM_MOCK_ENGINE"):
            self._model = _MockModel()
            self._loaded = True
            return

        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        from diarization.campplus import CAMPPlus

        model_path = self._ensure_model()
        self._model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._loaded = True

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if not self._loaded or self._model is None:
            return None
        if len(audio) < _MIN_SPEECH_SAMPLES:
            return None

        feats = self._compute_fbank(audio, sample_rate)
        if feats is None:
            return None

        import torch
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        embedding = self._model(feats_t).squeeze().cpu().numpy()
        return embedding

    @staticmethod
    def _compute_fbank(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        import torch
        import torchaudio
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0) * (1 << 15)
        feats = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
            sample_frequency=sample_rate, window_type="hamming",
            use_energy=False,
        )
        feats = feats - feats.mean(dim=0)
        return feats.numpy()

    @classmethod
    def _ensure_model(cls) -> str:
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "wespeaker" / "campplus_voxceleb"
        model_path = cache_dir / "campplus_voxceleb.bin"
        if model_path.exists():
            return str(model_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(_MODEL_URL, str(model_path))
        return str(model_path)

    # Expose fbank for weighted embedding extraction in engine
    def compute_fbank(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        return self._compute_fbank(audio, sample_rate)

    def extract_from_fbank(self, feats: np.ndarray) -> np.ndarray | None:
        if not self._loaded or self._model is None:
            return None
        import torch
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        embedding = self._model(feats_t).squeeze().cpu().numpy()
        return embedding


class _MockModel:
    def __call__(self, feats):
        import torch
        feat_np = feats.squeeze().numpy()
        rng = np.random.RandomState(int(abs(feat_np[:100].sum()) * 1000) % 2**31)
        emb = rng.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        return torch.tensor(emb).unsqueeze(0)


register_backend("campplus", CAMPPlusBackend)
