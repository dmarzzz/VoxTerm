import sys
import threading
import types

import numpy as np

from audio.transcriber import _MlxQwenWorker


def test_mlx_qwen_worker_uses_one_thread_for_load_and_transcribe(monkeypatch):
    calls = []

    class FakeResult:
        text = "hello from mlx"

    def load_model(model_id):
        calls.append(("load", model_id, threading.get_ident()))
        return {"model_id": model_id}, {}

    def transcribe(audio, *, model, language, verbose):
        calls.append(("transcribe", model["model_id"], language, threading.get_ident()))
        assert verbose is False
        assert len(audio) == 16000
        return FakeResult()

    fake_module = types.SimpleNamespace(load_model=load_model, transcribe=transcribe)
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", fake_module)

    worker = _MlxQwenWorker("demo/model")
    worker.load()
    text = worker.transcribe(np.zeros(16000, dtype=np.float32), "en")

    assert text == "hello from mlx"
    assert [call[0] for call in calls] == ["load", "transcribe"]
    assert calls[0][2] == calls[1][3]
