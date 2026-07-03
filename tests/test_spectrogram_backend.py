import base64
import json
import struct

import numpy as np

import config
import audio.transcriber as transcriber_mod
from audio.transcriber import (
    SpectrogramTranscriber,
    _audio_to_spectrogram_png_base64,
    get_transcriber,
)


def test_spectrogram_png_encoder_smoke():
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)

    raw = base64.b64decode(_audio_to_spectrogram_png_base64(audio))

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert raw[12:16] == b"IHDR"
    width, height = struct.unpack(">II", raw[16:24])
    assert width > 0
    assert height == 128


def test_spectrogram_transcriber_silence_short_circuits_without_network(monkeypatch):
    def fail_encoder(_audio):
        raise AssertionError("encoder should not run for silence")

    monkeypatch.setattr(transcriber_mod, "_audio_to_spectrogram_png_base64", fail_encoder)

    tr = SpectrogramTranscriber(server_url="http://vision.local", model="qwen2.5-vl")

    assert tr.transcribe(np.zeros(16000, dtype=np.float32)) == {
        "text": "",
        "speaker": "",
        "speaker_id": 0,
    }


def test_spectrogram_transcriber_posts_openai_compatible_payload(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, *_args):
            return b'{"choices":[{"message":{"content":" hello world "}}]}'

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        transcriber_mod,
        "_audio_to_spectrogram_png_base64",
        lambda _audio: "UE5H",
    )
    monkeypatch.setattr(transcriber_mod.urllib.request, "urlopen", fake_urlopen)

    tr = SpectrogramTranscriber(
        server_url="http://vision.local",
        model="qwen2.5-vl",
        language="en",
        request_timeout=7,
    )
    result = tr.transcribe(np.ones(16000, dtype=np.float32) * 0.01)

    assert result == {"text": "hello world", "speaker": "", "speaker_id": 0}
    req, timeout = calls[0]
    assert req.full_url == "http://vision.local/v1/chat/completions"
    assert timeout == 7
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "qwen2.5-vl"
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "English" in content[0]["text"]
    assert content[1]["image_url"]["url"] == "data:image/png;base64,UE5H"


def test_factory_returns_spectrogram_backend_when_configured(monkeypatch):
    monkeypatch.setitem(config.AVAILABLE_MODELS, "spec-test", "qwen2.5-vl")
    monkeypatch.setattr(config, "SPECTROGRAM_MODELS", {"spec-test"})
    monkeypatch.setattr(config, "SPECTROGRAM_SERVER_URL", "http://vision.local")

    tr = get_transcriber("spec-test", language="ja")

    assert isinstance(tr, SpectrogramTranscriber)
    assert tr.server_url == "http://vision.local"
    assert tr.model == "qwen2.5-vl"
    assert tr._language == "ja"
