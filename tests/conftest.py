"""Shared fixtures for VoxTerm test suite."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enable mock engine for all tests — avoids PyTorch/SpeechBrain model loading
os.environ["VOXTERM_MOCK_ENGINE"] = "1"

EMBEDDING_DIM = 192
SAMPLE_RATE = 16000


@pytest.fixture
def random_embedding():
    """Generate a random L2-normalized 192-dim float32 embedding."""
    def _make(seed=None):
        rng = np.random.RandomState(seed)
        emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10
        return emb
    return _make


@pytest.fixture
def sample_audio():
    """Generate synthetic 16kHz mono audio (sine wave)."""
    def _make(duration_sec=2.0, freq=440.0):
        t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * freq * t)
    return _make


@pytest.fixture
def mock_engine():
    """DiarizationEngine without model load — state management only."""
    from diarization.engine import DiarizationEngine
    engine = DiarizationEngine()
    # Don't call load() — we test state management, not inference
    return engine


@pytest.fixture
def loaded_mock_engine():
    """DiarizationEngine with mock model loaded (for identify() tests)."""
    from diarization.engine import DiarizationEngine
    engine = DiarizationEngine()
    engine.load()  # Uses _MockEcapaModel via VOXTERM_MOCK_ENGINE env var
    return engine


@pytest.fixture
def in_memory_store(tmp_path):
    """SpeakerStore backed by a temp-file SQLite database."""
    from speakers.store import SpeakerStore
    db_path = tmp_path / "test_speakers.db"
    store = SpeakerStore(db_path=db_path)
    store.open()
    yield store
    store.close()


@pytest.fixture
def tmp_crash_dir():
    """Temporary directory for crash dump tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
