"""Pyannote segmentation model for overlap-aware speaker detection.

Uses the ONNX export of pyannote/segmentation-3.0 to detect per-frame
speaker activity within audio chunks.  Provides overlap-aware weights
for embedding extraction (diart's Overlapped Speech Penalty).

Model: onnx-community/pyannote-segmentation-3.0 (~6 MB, MIT license)
Input: (1, 1, num_samples) at 16 kHz
Output: (1, num_frames, 7) powerset logits at ~17ms frame resolution

Powerset classes:
  0=NO_SPEAKER, 1=SPK1, 2=SPK2, 3=SPK3,
  4=SPK1+2, 5=SPK1+3, 6=SPK2+3
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Powerset → per-speaker mapping
# Each local speaker appears in these powerset classes:
_SPK_CLASSES = {
    0: [1, 4, 5],  # spk1, spk1+2, spk1+3
    1: [2, 4, 6],  # spk2, spk1+2, spk2+3
    2: [3, 5, 6],  # spk3, spk1+3, spk2+3
}

_FRAME_STEP_SAMPLES = 270  # ~16.9ms at 16kHz


class SpeakerSegmentation:
    """Pyannote segmentation-3.0 via ONNX Runtime.

    Detects up to 3 local speakers per chunk with frame-level activation
    and overlap detection.
    """

    # Overlap-aware weighting (from diart)
    OSP_GAMMA = 3    # exponent for overlap penalty
    OSP_BETA = 10    # softmax temperature for sharpening

    # Activity thresholds (from diart DIHARD-III tuning)
    TAU_ACTIVE = 0.55   # min peak activation to consider a speaker "active"
    RHO_UPDATE = 0.3    # min mean activation for centroid update eligibility

    def __init__(self):
        self._session = None
        self._loaded = False
        try:
            self._load()
        except Exception:
            pass

    def _load(self) -> None:
        import onnxruntime as ort

        model_path = self._find_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._loaded = True

    @staticmethod
    def _find_model() -> str:
        """Find or download the segmentation ONNX model."""
        cache_dir = Path.home() / ".cache" / "pyannote" / "segmentation-3.0"
        model_path = cache_dir / "model.onnx"
        if model_path.exists():
            return str(model_path)

        # Download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                "onnx-community/pyannote-segmentation-3.0",
                "onnx/model.onnx",
            )
            # Copy to our cache location
            cache_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(downloaded, model_path)
            return str(model_path)
        except Exception:
            pass

        # Try locating via huggingface cache directly
        spec = importlib.util.find_spec("huggingface_hub")
        if spec and spec.origin:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            for p in hf_cache.rglob("model.onnx"):
                if "pyannote-segmentation" in str(p):
                    return str(p)

        raise FileNotFoundError(
            "pyannote segmentation model not found. "
            "Install huggingface_hub and run once to download."
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def segment(self, audio: np.ndarray) -> np.ndarray:
        """Run segmentation on an audio chunk.

        Args:
            audio: (num_samples,) float32 mono at 16kHz

        Returns:
            Per-speaker activation matrix (num_frames, 3) with values in [0, 1].
            Each column is a local speaker's activation over time.
        """
        if not self._loaded:
            return np.zeros((1, 3), dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]

        inp = audio.reshape(1, 1, -1)
        logits = self._session.run(None, {"input_values": inp})[0][0]

        # Softmax over powerset classes
        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Convert powerset → per-speaker activation
        n_frames = probs.shape[0]
        activation = np.zeros((n_frames, 3), dtype=np.float32)
        for spk_idx, class_indices in _SPK_CLASSES.items():
            for cls in class_indices:
                activation[:, spk_idx] += probs[:, cls]

        return activation

    def get_active_speakers(
        self, activation: np.ndarray,
    ) -> list[dict]:
        """Identify active local speakers from activation matrix.

        Returns list of dicts with:
            speaker_idx: local speaker index (0, 1, 2)
            mean_activation: mean activity level
            peak_activation: peak activity level
            is_long: whether mean > RHO_UPDATE (quality gate)
        """
        speakers = []
        for i in range(activation.shape[1]):
            peak = float(activation[:, i].max())
            mean = float(activation[:, i].mean())
            if peak >= self.TAU_ACTIVE:
                speakers.append({
                    "speaker_idx": i,
                    "mean_activation": mean,
                    "peak_activation": peak,
                    "is_long": mean >= self.RHO_UPDATE,
                })
        return speakers

    def overlap_aware_weights(
        self, activation: np.ndarray,
    ) -> np.ndarray:
        """Compute diart's Overlapped Speech Penalty (OSP) weights.

        Returns (num_frames, 3) weights where overlap frames are penalized.
        Single-speaker frames get high weight; overlap frames get near-zero.
        """
        gamma = self.OSP_GAMMA
        beta = self.OSP_BETA

        # Sharpened softmax across speakers
        scaled = beta * activation
        scaled -= scaled.max(axis=-1, keepdims=True)
        exp_scaled = np.exp(scaled)
        sharp_probs = exp_scaled / (exp_scaled.sum(axis=-1, keepdims=True) + 1e-10)

        # OSP: activation^gamma * sharp_probs^gamma
        weights = np.power(activation, gamma) * np.power(sharp_probs, gamma)
        weights = np.maximum(weights, 1e-8)
        return weights

    def weighted_embedding_mask(
        self,
        activation: np.ndarray,
        audio_samples: int,
    ) -> list[tuple[int, np.ndarray]]:
        """Generate per-speaker sample-level masks for weighted embedding.

        For each active local speaker, returns a weight array over audio samples
        that can be used to mask the audio before embedding extraction.

        Returns list of (speaker_idx, weights_per_sample) tuples.
        """
        osp_weights = self.overlap_aware_weights(activation)
        active_speakers = self.get_active_speakers(activation)

        n_frames = activation.shape[0]
        results = []

        for spk_info in active_speakers:
            idx = spk_info["speaker_idx"]
            frame_weights = osp_weights[:, idx]

            # Upsample frame weights to sample level
            sample_weights = np.repeat(frame_weights, _FRAME_STEP_SAMPLES)
            # Trim or pad to match audio length
            if len(sample_weights) > audio_samples:
                sample_weights = sample_weights[:audio_samples]
            elif len(sample_weights) < audio_samples:
                sample_weights = np.pad(
                    sample_weights,
                    (0, audio_samples - len(sample_weights)),
                )

            results.append((idx, sample_weights))

        return results
