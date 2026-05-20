"""Stateful audio filters for background-noise rejection.

The primary use case is attenuating AC hum, fan rumble, and HVAC noise — all
of which sit below ~150 Hz and inflate the apparent RMS energy that the VAD's
RMS gate (SILENCE_THRESHOLD) uses. Without filtering, an AC running in the
room can hold the gate continuously open and cause spurious transcriptions
(or, paradoxically, drown out softer speech in the same chunk).

Speech intelligibility lives at 1-3 kHz; even male fundamentals start around
80 Hz, so a high-pass at 100 Hz leaves voice essentially untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


class HighPassFilter:
    """N-th order Butterworth high-pass that keeps state across chunks.

    Designed for streaming mic input: call .filter(chunk) for each chunk;
    internal state ensures no boundary discontinuities between chunks.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int, order: int = 2) -> None:
        nyquist = sample_rate * 0.5
        normalized = cutoff_hz / nyquist
        if not 0.0 < normalized < 1.0:
            raise ValueError(
                f"high-pass cutoff {cutoff_hz} Hz invalid for sample rate {sample_rate} Hz"
            )
        self._sos = butter(order, normalized, btype="highpass", output="sos")
        # State shape for sosfilt is (n_sections, 2). Initialize at rest:
        # mic audio is centered around 0, so previous-samples-were-0 is the
        # honest assumption. (scipy.signal.sosfilt_zi assumes input=1, which
        # produces a unit-magnitude startup transient — wrong for mic streams.)
        self._zero_state = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._state = self._zero_state.copy()

    def filter(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        out, self._state = sosfilt(self._sos, audio, zi=self._state)
        return out.astype(audio.dtype, copy=False)

    def reset(self) -> None:
        """Re-seed filter state to rest — call when recording starts/stops."""
        self._state = self._zero_state.copy()
