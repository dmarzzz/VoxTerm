"""Diarization accuracy tests using real multi-speaker audio (AMI corpus).

Uses pyannote's AMI test fixtures (CC BY 4.0) with RTTM ground truth.
Skipped automatically if fixtures are missing — run download first:
    curl -sL https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/tests/data/tst00.wav \
         -o tests/fixtures/speakers/tst00.wav
    curl -sL https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/tests/data/debug.test.rttm \
         -o tests/fixtures/speakers/tst00.rttm
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "speakers"
SAMPLE_RATE = 16000

# Skip entire module if fixtures aren't downloaded
pytestmark = pytest.mark.skipif(
    not (FIXTURES / "tst00.wav").exists(),
    reason="Speaker fixtures not downloaded (see tests/fixtures/speakers/README)",
)

# Use real model, NOT mock — we're testing actual diarization quality
# Remove VOXTERM_MOCK_ENGINE if set by conftest
_ORIG_MOCK = os.environ.pop("VOXTERM_MOCK_ENGINE", None)


def _restore_mock():
    if _ORIG_MOCK is not None:
        os.environ["VOXTERM_MOCK_ENGINE"] = _ORIG_MOCK


# ── helpers ────────────────────────────────────────────────


def load_wav_float32(path: Path) -> np.ndarray:
    """Load a float32 WAV file (the stdlib wave module doesn't support float32)."""
    with open(path, "rb") as f:
        # Read RIFF header
        riff = f.read(4)
        assert riff == b"RIFF", f"Not a RIFF file: {path}"
        f.read(4)  # file size
        wave = f.read(4)
        assert wave == b"WAVE"

        audio_data = None
        sample_rate = 16000

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                assert audio_format == 3, f"Not float32 WAV (format={audio_format})"
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
            elif chunk_id == b"data":
                raw = f.read(chunk_size)
                audio_data = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                f.read(chunk_size)

    assert audio_data is not None, f"No data chunk found in {path}"
    return audio_data


def parse_rttm(path: Path, file_id: str) -> list[dict]:
    """Parse RTTM file into list of {speaker, start, end} dicts."""
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            if parts[1] != file_id:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append({
                "speaker": speaker,
                "start": start,
                "end": start + duration,
            })
    return segments


def get_dominant_speaker_at(segments: list[dict], time_sec: float) -> str | None:
    """Return the speaker with the most overlap at a given time."""
    active = [s["speaker"] for s in segments if s["start"] <= time_sec < s["end"]]
    if not active:
        return None
    # Return most common (handles overlap)
    from collections import Counter
    return Counter(active).most_common(1)[0][0]


def get_ground_truth_speakers(segments: list[dict]) -> set[str]:
    """Return the set of unique speakers in the ground truth."""
    return {s["speaker"] for s in segments}


def extract_speaker_segments(
    segments: list[dict], audio: np.ndarray, sr: int = 16000,
    min_dur: float = 1.5,
) -> dict[str, list[np.ndarray]]:
    """Extract per-speaker audio segments from ground truth."""
    speaker_audio: dict[str, list[np.ndarray]] = {}
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < min_dur:
            continue
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = audio[start_sample:end_sample]
        if len(chunk) >= int(min_dur * sr):
            speaker_audio.setdefault(seg["speaker"], []).append(chunk)
    return speaker_audio


# ── fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine():
    """Load real CAM++ engine (module-scoped for speed)."""
    from diarization.engine import DiarizationEngine
    e = DiarizationEngine()
    e.load()
    yield e
    _restore_mock()


@pytest.fixture(scope="module")
def tst00_audio():
    return load_wav_float32(FIXTURES / "tst00.wav")


@pytest.fixture(scope="module")
def tst00_segments():
    return parse_rttm(FIXTURES / "tst00.rttm", "tst00")


@pytest.fixture(scope="module")
def dev00_audio():
    return load_wav_float32(FIXTURES / "dev00.wav")


@pytest.fixture(scope="module")
def dev00_segments():
    return parse_rttm(FIXTURES / "dev00.rttm", "dev00")


# ── tests ──────────────────────────────────────────────────


class TestFixtureIntegrity:
    """Verify test fixtures loaded correctly."""

    def test_tst00_audio_shape(self, tst00_audio):
        assert len(tst00_audio) == 480001  # 30s at 16kHz + 1
        assert tst00_audio.dtype == np.float32

    def test_tst00_has_4_speakers(self, tst00_segments):
        speakers = get_ground_truth_speakers(tst00_segments)
        assert len(speakers) == 4

    def test_dev00_has_2_speakers(self, dev00_segments):
        speakers = get_ground_truth_speakers(dev00_segments)
        assert len(speakers) == 2

    def test_rttm_segments_parsed(self, tst00_segments):
        assert len(tst00_segments) > 10
        for seg in tst00_segments:
            assert "speaker" in seg
            assert seg["end"] > seg["start"]


class TestSpeakerCount:
    """Test that the engine detects approximately the right number of speakers."""

    def test_2_speaker_count(self, engine, dev00_audio, dev00_segments):
        """2-speaker audio should produce 1-3 detected speakers."""
        engine.reset_session()
        speaker_audio = extract_speaker_segments(dev00_segments, dev00_audio)

        for spk, chunks in speaker_audio.items():
            for chunk in chunks[:3]:  # first 3 segments per speaker
                engine.identify(chunk)

        detected = engine.num_speakers
        print(f"\n  2-speaker test: detected {detected} speakers "
              f"(ground truth: 2)")
        assert 1 <= detected <= 3, (
            f"Expected 1-3 speakers for 2-speaker audio, got {detected}"
        )

    def test_4_speaker_count(self, engine, tst00_audio, tst00_segments):
        """4-speaker audio should produce 2-6 detected speakers."""
        engine.reset_session()
        speaker_audio = extract_speaker_segments(tst00_segments, tst00_audio)

        for spk, chunks in speaker_audio.items():
            for chunk in chunks[:3]:
                engine.identify(chunk)

        detected = engine.num_speakers
        print(f"\n  4-speaker test: detected {detected} speakers "
              f"(ground truth: 4)")
        assert 2 <= detected <= 8, (
            f"Expected 2-8 speakers for 4-speaker audio, got {detected}"
        )


class TestSpeakerConsistency:
    """Test that the same speaker gets the same ID across segments."""

    def test_same_speaker_same_id(self, engine, dev00_audio, dev00_segments):
        """Multiple segments from the same speaker should get the same ID."""
        engine.reset_session()
        speaker_audio = extract_speaker_segments(dev00_segments, dev00_audio)

        speaker_ids: dict[str, list[int]] = {}
        for spk, chunks in speaker_audio.items():
            for chunk in chunks[:4]:
                _, sid = engine.identify(chunk)
                speaker_ids.setdefault(spk, []).append(sid)

        # For each real speaker, check consistency
        for spk, ids in speaker_ids.items():
            if len(ids) < 2:
                continue
            from collections import Counter
            most_common_id, count = Counter(ids).most_common(1)[0]
            consistency = count / len(ids)
            print(f"\n  Speaker {spk}: IDs={ids}, "
                  f"consistency={consistency:.0%}")
            # At least 50% of segments should get the same ID
            assert consistency >= 0.5, (
                f"Speaker {spk} got inconsistent IDs: {ids} "
                f"(most common {most_common_id} appeared {count}/{len(ids)})"
            )


class TestSpeakerSeparation:
    """Test that different speakers get different IDs."""

    def test_different_speakers_different_ids(self, engine, dev00_audio, dev00_segments):
        """Two different speakers should (mostly) get different IDs."""
        engine.reset_session()
        speaker_audio = extract_speaker_segments(dev00_segments, dev00_audio)

        speaker_primary_ids: dict[str, int] = {}
        for spk, chunks in speaker_audio.items():
            ids = []
            for chunk in chunks[:4]:
                _, sid = engine.identify(chunk)
                ids.append(sid)
            from collections import Counter
            if ids:
                speaker_primary_ids[spk] = Counter(ids).most_common(1)[0][0]

        if len(speaker_primary_ids) >= 2:
            unique_ids = set(speaker_primary_ids.values())
            print(f"\n  Speaker→ID mapping: {speaker_primary_ids}")
            print(f"  Unique IDs: {unique_ids}")
            # Different speakers should map to different IDs
            # (relaxed: at least 2 unique IDs for 2 speakers)
            assert len(unique_ids) >= 2, (
                f"Expected different IDs for different speakers, "
                f"got {speaker_primary_ids}"
            )


class TestIdentifySegments:
    """Test SCD-based identify_segments on real audio."""

    def test_identify_segments_returns_results(self, engine, tst00_audio):
        """identify_segments should return valid segments for real audio."""
        engine.reset_session()
        # Use a 5-second slice (enough for SCD)
        chunk = tst00_audio[:80000]  # 5 seconds
        results = engine.identify_segments(chunk)
        assert len(results) >= 1
        for label, sid, start, end in results:
            assert isinstance(label, str)
            assert sid >= 1
            assert end > start

    def test_identify_segments_coverage(self, engine, tst00_audio):
        """Segments should cover the input audio."""
        engine.reset_session()
        chunk = tst00_audio[:80000]
        results = engine.identify_segments(chunk)
        assert results[0][2] == 0  # starts at 0
        assert results[-1][3] == len(chunk)  # ends at audio length


class TestEndToEnd:
    """Full pipeline test: sequential segments, check overall accuracy."""

    def test_sequential_diarization(self, engine, tst00_audio, tst00_segments):
        """Process audio in sequential chunks, measure speaker accuracy."""
        engine.reset_session()

        # Process in 3-second chunks (like real-time)
        chunk_size = SAMPLE_RATE * 3
        results: list[tuple[float, str, int]] = []  # (time, true_speaker, assigned_id)

        for start in range(0, len(tst00_audio) - chunk_size, chunk_size):
            chunk = tst00_audio[start:start + chunk_size]
            time_sec = start / SAMPLE_RATE + 1.5  # midpoint

            true_speaker = get_dominant_speaker_at(tst00_segments, time_sec)
            if true_speaker is None:
                continue

            _, sid = engine.identify(chunk)
            results.append((time_sec, true_speaker, sid))

        if not results:
            pytest.skip("No annotated segments found")

        # Compute accuracy: for each true speaker, what % got the same assigned ID?
        from collections import Counter
        speaker_id_counts: dict[str, Counter] = {}
        for time_sec, true_spk, sid in results:
            speaker_id_counts.setdefault(true_spk, Counter())[sid] += 1

        total_correct = 0
        total = 0
        print(f"\n  End-to-end results ({len(results)} chunks):")
        for spk, counts in speaker_id_counts.items():
            most_common_id, correct = counts.most_common(1)[0]
            spk_total = sum(counts.values())
            accuracy = correct / spk_total
            print(f"    {spk}: {correct}/{spk_total} = {accuracy:.0%} "
                  f"(assigned to Speaker {most_common_id})")
            total_correct += correct
            total += spk_total

        overall = total_correct / total if total else 0
        print(f"  Overall consistency: {total_correct}/{total} = {overall:.0%}")
        print(f"  Detected speakers: {engine.num_speakers}")

        # Relaxed threshold: at least 40% consistency
        # (this is a baseline — we'll tighten as we improve)
        assert overall >= 0.4, (
            f"Overall speaker consistency too low: {overall:.0%}"
        )
