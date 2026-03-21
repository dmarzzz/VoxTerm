"""Silero VAD wrapper using ONNX Runtime (no PyTorch required).

Provides neural voice activity detection as a drop-in replacement for
RMS energy thresholding.  Uses the Silero VAD ONNX model directly via
onnxruntime with pure-numpy state management, so PyTorch is never
imported in the main process (avoiding MLX/PyTorch C++ runtime conflicts).

Requires: pip install silero-vad onnxruntime
"""

from __future__ import annotations

import numpy as np

from config import SAMPLE_RATE, VAD_THRESHOLD

_CHUNK_SAMPLES = 512     # Silero ONNX expects exactly 512 samples at 16 kHz
_CONTEXT_SIZE = 64       # 64-sample context prepended to each chunk
_STATE_SHAPE = (2, 1, 128)  # hidden state: 2 layers × 1 batch × 128 dims


class SileroVAD:
    """Neural VAD using Silero's ONNX model (no PyTorch dependency).

    Accepts 1024-sample chunks (VoxTerm's standard chunk size) and splits
    them into two 512-sample sub-chunks for Silero.  Returns the max
    speech probability across sub-chunks.
    """

    def __init__(self, threshold: float = VAD_THRESHOLD):
        self.threshold = threshold
        self._session = None
        self._state: np.ndarray = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context: np.ndarray = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)
        self._loaded = False

        try:
            self._load()
        except Exception:
            pass  # graceful fallback — is_speech() always returns True

    def _load(self) -> None:
        """Load the Silero ONNX model via onnxruntime."""
        import onnxruntime

        # Locate the ONNX model file bundled with the silero-vad package
        model_path = self._find_model_path()

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._loaded = True

    @staticmethod
    def _find_model_path() -> str:
        """Find the silero_vad.onnx file from the installed package.

        Uses importlib.util to locate the package directory without importing
        silero_vad (which would transitively import torch).
        """
        import importlib.util
        spec = importlib.util.find_spec("silero_vad")
        if spec is None or spec.origin is None:
            raise ImportError("silero-vad package not found")
        from pathlib import Path
        return str(Path(spec.origin).parent / "data" / "silero_vad.onnx")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def speech_probability(self, chunk: np.ndarray) -> float:
        """Return speech probability for an audio chunk.

        Accepts any chunk length that is a multiple of 512.  For VoxTerm's
        1024-sample chunks, processes two 512-sample sub-chunks and returns
        the max probability.

        Falls back to 1.0 (always speech) if the model isn't loaded.
        """
        if not self._loaded:
            return 1.0

        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]

        # Process in 512-sample sub-chunks
        max_prob = 0.0
        for offset in range(0, len(chunk), _CHUNK_SAMPLES):
            sub = chunk[offset:offset + _CHUNK_SAMPLES]
            if len(sub) < _CHUNK_SAMPLES:
                break  # skip incomplete tail
            prob = self._infer(sub)
            if prob > max_prob:
                max_prob = prob

        return max_prob

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Check whether an audio chunk contains speech."""
        return self.speech_probability(chunk) >= self.threshold

    def get_speech_segments(
        self,
        audio: np.ndarray,
        min_speech_ms: int = 500,
        min_silence_ms: int = 300,
        max_speech_s: float = 6.0,
    ) -> list[tuple[int, int]]:
        """Find speech segment boundaries in an audio buffer.

        Returns list of (start_sample, end_sample) tuples for each speech
        region, split at silence gaps >= min_silence_ms and capped at
        max_speech_s.  Uses a fresh model state (does not affect streaming).
        """
        if not self._loaded:
            # Fallback: return the whole buffer as one segment
            return [(0, len(audio))]

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Save and reset state for independent batch processing
        saved_state = self._state.copy()
        saved_context = self._context.copy()
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)

        # Get per-frame probabilities (512 samples = 32ms per frame)
        probs: list[float] = []
        for offset in range(0, len(audio), _CHUNK_SAMPLES):
            sub = audio[offset:offset + _CHUNK_SAMPLES]
            if len(sub) < _CHUNK_SAMPLES:
                break
            probs.append(self._infer(sub))

        # Restore state
        self._state = saved_state
        self._context = saved_context

        if not probs:
            return [(0, len(audio))]

        # Convert to speech/silence regions
        frame_samples = _CHUNK_SAMPLES  # 512
        min_speech_frames = max(1, int(min_speech_ms / 32))
        min_silence_frames = max(1, int(min_silence_ms / 32))
        max_speech_frames = int(max_speech_s * SAMPLE_RATE / frame_samples)

        segments: list[tuple[int, int]] = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for i, p in enumerate(probs):
            if p >= self.threshold:
                if not in_speech:
                    speech_start = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # End of speech segment
                        speech_end = i - silence_count + 1
                        if speech_end - speech_start >= min_speech_frames:
                            segments.append((
                                speech_start * frame_samples,
                                min(speech_end * frame_samples, len(audio)),
                            ))
                        in_speech = False
                        silence_count = 0

            # Force-split long segments
            if in_speech and (i - speech_start) >= max_speech_frames:
                segments.append((
                    speech_start * frame_samples,
                    min((i + 1) * frame_samples, len(audio)),
                ))
                in_speech = False

        # Handle trailing speech
        if in_speech:
            speech_len = len(probs) - speech_start
            if speech_len >= min_speech_frames:
                segments.append((
                    speech_start * frame_samples,
                    len(audio),
                ))

        return segments if segments else [(0, len(audio))]

    def reset(self) -> None:
        """Reset internal model state (call between recording sessions)."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)

    def _infer(self, sub_chunk: np.ndarray) -> float:
        """Run ONNX inference on a single 512-sample sub-chunk."""
        # Prepend context (matching Silero's OnnxWrapper behavior)
        x = np.concatenate([self._context, sub_chunk.reshape(1, -1)], axis=1)

        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        out, new_state = self._session.run(None, ort_inputs)

        # Update state for next call
        self._state = new_state
        self._context = x[:, -_CONTEXT_SIZE:]

        return float(out.squeeze())
