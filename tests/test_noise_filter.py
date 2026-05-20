"""Smoke tests for the high-pass noise filter applied to mic input.

We don't try to assert exact filter coefficients — just verify the *shape*:
   - low-frequency content (60 Hz AC hum) gets attenuated
   - in-band speech content (1 kHz) passes through ~unchanged
   - the streaming API maintains state across chunk boundaries (no clicks)
   - degenerate inputs (empty, all-zero) don't crash
"""
import numpy as np
import pytest

from audio.noise_filter import HighPassFilter


SR = 16000


def _band_rms(x: np.ndarray, low: float, high: float) -> float:
    """Energy in a frequency band, via FFT magnitude integration."""
    if x.size == 0:
        return 0.0
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sqrt(np.mean(np.abs(spec[mask]) ** 2)))


def _seconds(t_sec: float) -> np.ndarray:
    return np.arange(0.0, t_sec, 1.0 / SR, dtype=np.float32)


def test_hpf_attenuates_60hz_ac_hum():
    t = _seconds(1.0)
    sig = 0.5 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    f = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    out = f.filter(sig)
    before = _band_rms(sig, 50, 70)
    after = _band_rms(out, 50, 70)
    # 2nd-order Butterworth at 60 Hz with 100 Hz cutoff should give ≥6 dB
    # attenuation. We use 4 dB as a safety margin to avoid CI flakes.
    assert after < before * 0.63, f"expected ≥4 dB attenuation, got {20*np.log10(after/before):.1f} dB"


def test_hpf_passes_1khz_voice_band():
    t = _seconds(1.0)
    sig = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    f = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    out = f.filter(sig)
    before = _band_rms(sig, 900, 1100)
    after = _band_rms(out, 900, 1100)
    # At 10x cutoff frequency the response is essentially 0 dB.
    assert after > before * 0.95, f"voice band attenuated unexpectedly: {20*np.log10(after/before):.1f} dB"


def test_hpf_streaming_state_continuous_across_chunks():
    """Filtering one big chunk vs many small chunks should yield the same output."""
    t = _seconds(0.5)
    sig = (0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    f1 = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    big = f1.filter(sig.copy())

    f2 = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    chunk_size = 512
    pieces = [f2.filter(sig[i:i + chunk_size].copy()) for i in range(0, sig.size, chunk_size)]
    small = np.concatenate(pieces)

    # The two filters were initialized identically, so outputs must match exactly.
    np.testing.assert_allclose(small, big, atol=1e-5, err_msg="streaming != bulk")


def test_hpf_empty_input_returns_empty():
    f = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    out = f.filter(np.array([], dtype=np.float32))
    assert out.size == 0


def test_hpf_zero_input_stays_zero_no_startup_transient():
    """Filter initialized at rest must produce zero output for zero input —
    no startup transient that could mask the first ~10 ms of mic audio."""
    f = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    zeros = np.zeros(1024, dtype=np.float32)
    out = f.filter(zeros)
    np.testing.assert_allclose(out, 0.0, atol=1e-6,
                               err_msg="filter produced output for zero input — startup transient bug")


def test_hpf_reset_clears_history():
    """After reset(), filtering the same signal twice should match the first run."""
    t = _seconds(0.2)
    sig = 0.3 * np.sin(2 * np.pi * 60 * t).astype(np.float32)

    f = HighPassFilter(cutoff_hz=100, sample_rate=SR, order=2)
    first = f.filter(sig.copy())
    f.reset()
    second = f.filter(sig.copy())
    np.testing.assert_allclose(first, second, atol=1e-6, err_msg="reset() did not restore initial state")


def test_hpf_invalid_cutoff_raises():
    # Cutoff at Nyquist is not a valid high-pass spec.
    with pytest.raises(ValueError):
        HighPassFilter(cutoff_hz=SR // 2, sample_rate=SR, order=2)
    with pytest.raises(ValueError):
        HighPassFilter(cutoff_hz=0.0, sample_rate=SR, order=2)
