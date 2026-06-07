"""Tests for the optional sherpa-onnx streaming backend.

Skipped entirely when sherpa-onnx isn't installed (the default) — so CI / existing installs
are unaffected. When it IS installed, this asserts the config gating + factory dispatch are
wired. The actual decode is covered by the manual smoke test (downloads a 131 MB model).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_NO_SHERPA = importlib.util.find_spec("sherpa_onnx") is None


def test_gating_consistent_either_way():
    """Whether or not sherpa is installed, the gate is self-consistent and additive."""
    import config
    assert config._HAS_SHERPA == ("sherpa-stream-en" in config.SHERPA_MODELS)
    # the key is surfaced in AVAILABLE_MODELS iff the gate is on
    assert ("sherpa-stream-en" in config.AVAILABLE_MODELS) == config._HAS_SHERPA


@pytest.mark.skipif(_NO_SHERPA, reason="sherpa-onnx not installed (optional [streaming] extra)")
def test_factory_returns_sherpa_backend():
    import config
    from audio.transcriber import get_transcriber, SherpaStreamingTranscriber
    # both gated keys map to the streaming backend (UNLOADED — load() fetches the model)
    for key in ("sherpa-stream-en", "sherpa-nemotron-en"):
        assert key in config.SHERPA_MODELS
        tr = get_transcriber(key)
        assert isinstance(tr, SherpaStreamingTranscriber)
        assert not tr.is_loaded


@pytest.mark.skipif(_NO_SHERPA, reason="sherpa-onnx not installed (optional [streaming] extra)")
def test_silence_returns_empty_without_loading_network():
    # RMS gate short-circuits before any recognizer use → safe to call on a fresh instance.
    import numpy as np
    from audio.transcriber import SherpaStreamingTranscriber
    tr = SherpaStreamingTranscriber()
    assert tr.transcribe(np.zeros(16000, dtype=np.float32)) == {"text": "", "speaker": "", "speaker_id": 0}


def test_transcribe_before_load_raises_clear_error():
    # non-silent audio on an unloaded instance must fail loudly, not AttributeError on
    # self._rec (None). Needs no sherpa — the guard fires before any recognizer use.
    import numpy as np
    from audio.transcriber import SherpaStreamingTranscriber
    tr = SherpaStreamingTranscriber()
    loud = np.full(16000, 0.5, dtype=np.float32)  # RMS well above the 0.005 silence gate
    with pytest.raises(RuntimeError):
        tr.transcribe(loud)
