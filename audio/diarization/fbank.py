"""Pure-numpy Mel filterbank feature extraction (Kaldi-compatible).

Replaces torchaudio.compliance.kaldi.fbank() for use in the ONNX inference
path, where PyTorch is not available.  Parameters match the Kaldi defaults
used by 3D-Speaker / WeSpeaker pretrained models.

No external dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np


def compute_fbank(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    preemph_coeff: float = 0.97,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    window_type: str = "hamming",
    cmn: bool = True,
) -> np.ndarray:
    """Compute log-Mel filterbank features matching Kaldi conventions.

    Input:  1-D float32 audio (already scaled to [-1, 1] range).
            NOTE: Kaldi expects raw PCM amplitude — we multiply by 2**15
            internally, same as the torchaudio path.
    Output: (num_frames, num_mel_bins) float32 array.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()

    # Kaldi convention: scale normalised audio to 16-bit range
    audio = audio * (1 << 15)

    if high_freq <= 0:
        high_freq = sample_rate / 2.0

    # Pre-emphasis
    if preemph_coeff > 0:
        audio = np.concatenate([[audio[0]], audio[1:] - preemph_coeff * audio[:-1]])

    # Frame parameters (in samples)
    frame_length = int(round(frame_length_ms / 1000.0 * sample_rate))
    frame_shift = int(round(frame_shift_ms / 1000.0 * sample_rate))
    n_fft = _next_power_of_2(frame_length)

    # Number of frames
    if len(audio) < frame_length:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    num_frames = 1 + (len(audio) - frame_length) // frame_shift

    # Window function
    window = _get_window(window_type, frame_length)

    # Build mel filterbank
    mel_filters = _mel_filterbank(
        n_fft=n_fft,
        num_mel_bins=num_mel_bins,
        sample_rate=sample_rate,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    # Extract frames and compute features
    features = np.empty((num_frames, num_mel_bins), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_shift
        frame = audio[start : start + frame_length] * window

        # Power spectrum
        spectrum = np.fft.rfft(frame, n=n_fft)
        power = np.real(spectrum) ** 2 + np.imag(spectrum) ** 2

        # Apply mel filterbank
        mel_energies = mel_filters @ power
        mel_energies = np.maximum(mel_energies, 1e-10)
        features[i] = np.log(mel_energies)

    # Cepstral mean normalization.  Accumulate in float64 so the returned
    # float32 features remain numerically zero-mean within tight tolerances.
    if cmn and num_frames > 0:
        features = (features.astype(np.float64) - features.mean(axis=0, dtype=np.float64)).astype(
            np.float32
        )

    return features


# ── helpers ──────────────────────────────────────────────


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _get_window(window_type: str, length: int) -> np.ndarray:
    if window_type == "hamming":
        return np.hamming(length).astype(np.float32)
    elif window_type == "hanning":
        return np.hanning(length).astype(np.float32)
    elif window_type == "povey":
        # Kaldi's default: raised to power 0.85
        return (np.hanning(length).astype(np.float32)) ** 0.85
    else:
        return np.ones(length, dtype=np.float32)


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale (HTK formula)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel scale to frequency in Hz (HTK formula)."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_fft: int,
    num_mel_bins: int,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """Build a (num_mel_bins, n_fft//2 + 1) triangular mel filterbank matrix."""
    num_fft_bins = n_fft // 2 + 1

    low_mel = _hz_to_mel(low_freq)
    high_mel = _hz_to_mel(high_freq)

    # Linearly spaced in mel domain
    mel_points = np.linspace(low_mel, high_mel, num_mel_bins + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # Bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_mel_bins, num_fft_bins), dtype=np.float32)
    for m in range(num_mel_bins):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        # Rising slope
        for k in range(f_left, f_center):
            if k < num_fft_bins and f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        # Falling slope
        for k in range(f_center, f_right):
            if k < num_fft_bins and f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank
