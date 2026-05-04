"""Transcription engine — Qwen3-ASR (all platforms), mlx-whisper (macOS), faster-whisper (Linux fallback), llama server (remote)."""

from __future__ import annotations

import base64
import io
import json
import queue
import re
import struct
import threading
import urllib.request
import urllib.error

import numpy as np

from audio.platform import CURRENT_PLATFORM, Platform


class _DeduplicatorMixin:
    """Tracks recent outputs and suppresses consecutive duplicates."""

    def _init_dedup(self):
        self._recent: list[str] = []

    def _is_duplicate(self, text: str) -> bool:
        normalized = text.lower().strip().rstrip(".")
        if normalized in self._recent:
            return True
        self._recent.append(normalized)
        if len(self._recent) > 5:
            self._recent.pop(0)
        return False


def _is_hallucination(text: str, expected_language: str | None = "en") -> bool:
    """Detect common ASR hallucination patterns (shared by all transcribers)."""
    if not text:
        return False
    if len(text) < 2:
        return True

    # Reject non-Latin script when expecting a Latin-script language
    if expected_language and expected_language in (
        "en", "fr", "de", "es", "it", "pt", "nl", "tr",
    ):
        if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\u0400-\u04ff\u0600-\u06ff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text):
            return True

    words = text.lower().split()
    if len(words) > 80:
        return True

    if len(words) >= 8:
        from collections import Counter
        for n in range(2, min(11, len(words) // 2 + 1)):
            if len(words) < n * 2:
                continue
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)
            top_count = counts.most_common(1)[0][1]
            if top_count >= 3 and top_count > len(ngrams) * 0.25:
                return True

    hallucination_patterns = [
        r"^\.+$",
        r"^(thanks? (for )?watching)",
        r"^(subscribe)",
        r"^(please subscribe)",
        r"^(music|applause|\[music\])",
        r"^(you)$",
        r"^(so)$",
        r"^(oh)$",
        r"^(bye\.?)$",
        r"^(thank you\.?)$",
        r"^so,?\s+let'?s\s+go\.?$",
        r"^let'?s\s+go\.?$",
        r"^one,?\s+two,?\s+three,?\s+four\.?$",
        r"^i'?m\s+going\s+to\s+go\s+ahead",
    ]
    text_lower = text.lower().strip()
    for pattern in hallucination_patterns:
        if re.match(pattern, text_lower):
            return True
    return False


# qwen-asr (PyTorch) expects full language names, not ISO codes
_ISO_TO_LANG = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi", "it": "Italian",
    "tr": "Turkish", "nl": "Dutch", "id": "Indonesian", "th": "Thai",
    "vi": "Vietnamese", "ms": "Malay", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "fa": "Persian",
}


class _MlxQwenWorker:
    """Owns all MLX Qwen model work on one thread.

    MLX streams are thread-local. Loading a model on one Python thread and then
    sampling from it on a different Textual worker can raise
    ``RuntimeError: There is no Stream(gpu, N) in current thread``. Keeping load
    and inference on this worker avoids crossing those stream boundaries.
    """

    def __init__(self, model_id: str):
        self._model_id = model_id
        self._jobs: queue.Queue[tuple[str, tuple, dict, threading.Event, dict]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="mlx-qwen3-asr",
        )
        self._thread.start()

    def load(self) -> None:
        self._call("load")

    def transcribe(self, audio: np.ndarray, language: str | None) -> str:
        return self._call("transcribe", audio, language)

    def _call(self, op: str, *args):
        done = threading.Event()
        result: dict = {}
        self._jobs.put((op, args, {}, done, result))
        done.wait()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _run(self) -> None:
        model = None
        load_model = None
        transcribe = None
        while True:
            op, args, _kwargs, done, result = self._jobs.get()
            try:
                if load_model is None or transcribe is None:
                    from mlx_qwen3_asr import load_model as _load_model
                    from mlx_qwen3_asr import transcribe as _transcribe

                    load_model = _load_model
                    transcribe = _transcribe

                if op == "load":
                    if model is None:
                        model, _config = load_model(self._model_id)
                    result["value"] = None
                elif op == "transcribe":
                    if model is None:
                        model, _config = load_model(self._model_id)
                    audio, language = args
                    output = transcribe(
                        audio,
                        model=model,
                        language=language,
                        verbose=False,
                    )
                    result["value"] = str(output.text).strip() if hasattr(output, "text") else ""
                else:
                    raise RuntimeError(f"unknown MLX worker operation: {op}")
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()


class Qwen3Transcriber(_DeduplicatorMixin):
    """Qwen3-ASR transcriber — MLX on macOS, qwen-asr (PyTorch) on Linux."""

    def __init__(self, model: str = "Qwen/Qwen3-ASR-0.6B", language: str | None = "en"):
        self.model_id = model
        self._language = language
        self._model = None
        self._mlx_worker: _MlxQwenWorker | None = None
        self._loaded = False
        self._use_mlx = CURRENT_PLATFORM == Platform.MACOS
        self._init_dedup()

    def load(self):
        """Pre-load the model (downloads on first run)."""
        if self._use_mlx:
            self._mlx_worker = _MlxQwenWorker(self.model_id)
            self._mlx_worker.load()
        else:
            from qwen_asr import Qwen3ASRModel
            import torch
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=device,
                max_new_tokens=256,
            )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        """Transcribe audio array (float32, 16kHz mono).

        Returns:
            {"text": str, "speaker": str, "speaker_id": int}
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}

        if self._use_mlx:
            if self._mlx_worker is None:
                self._mlx_worker = _MlxQwenWorker(self.model_id)
            text = self._mlx_worker.transcribe(audio, self._language)
        else:
            lang = _ISO_TO_LANG.get(self._language, self._language) if self._language else None
            results = self._model.transcribe(
                audio=(audio, 16000),
                language=lang,
            )
            text = results[0].text.strip() if results else ""

        if _is_hallucination(text, self._language):
            return {"text": "", "speaker": "", "speaker_id": 0}

        if self._is_duplicate(text):
            return {"text": "", "speaker": "", "speaker_id": 0}

        return {"text": text, "speaker": "", "speaker_id": 0}

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class WhisperTranscriber(_DeduplicatorMixin):
    """Legacy mlx-whisper transcriber (fallback)."""

    def __init__(self, model: str = "mlx-community/whisper-small-mlx"):
        self.model = model
        self._loaded = False
        self._init_dedup()

    def load(self):
        import mlx_whisper
        silent = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(silent, path_or_hf_repo=self.model, verbose=False)
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        import mlx_whisper

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.0,
        )

        text = result.get("text", "").strip()
        if _is_hallucination(text):
            return {"text": "", "speaker": "", "speaker_id": 0}

        if self._is_duplicate(text):
            return {"text": "", "speaker": "", "speaker_id": 0}

        return {"text": text, "speaker": "", "speaker_id": 0}

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class FasterWhisperTranscriber(_DeduplicatorMixin):
    """Cross-platform transcriber using faster-whisper (CTranslate2 backend).

    Works on Linux (CPU/CUDA) and any platform with faster-whisper installed.
    """

    def __init__(self, model: str = "small", language: str | None = "en"):
        self.model_size = model
        self._language = language
        self._model = None
        self._loaded = False
        self._init_dedup()

    def load(self):
        """Pre-load the model (downloads on first run)."""
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.model_size, device="auto", compute_type="auto",
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}

        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,  # we already run Silero VAD upstream
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if _is_hallucination(text, self._language):
            return {"text": "", "speaker": "", "speaker_id": 0}

        if self._is_duplicate(text):
            return {"text": "", "speaker": "", "speaker_id": 0}

        return {"text": text, "speaker": "", "speaker_id": 0}

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _audio_to_wav_base64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Encode float32 audio array as base64 WAV string."""
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(audio_int16)
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def discover_llama_audio_models(server_url: str) -> list[str] | None:
    """Query a llama-swap, llama.cpp, or Ollama server for available models.

    Probes in order: /v1/models (llama-swap/OpenAI), /api/tags (Ollama),
    /health (plain llama.cpp).

    Returns:
        list[str] — model names (may be empty)
        None — if the server is unreachable
    """
    url = server_url.rstrip("/")

    # Try /v1/models first (llama-swap / OpenAI-compatible)
    try:
        req = urllib.request.Request(f"{url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        return [m.get("id") for m in models if m.get("id")]
    except Exception:
        pass

    # Try /api/tags (Ollama-compatible listing)
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:
        # Server unreachable — try /health as fallback
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                # Server is up but has no model listing endpoint
                return []
        except Exception:
            return None

    models = data.get("models", []) or data.get("data", [])
    audio_models = []
    for m in models[:20]:
        caps = m.get("capabilities", [])
        if "multimodal" in caps:
            name = m.get("id") or m.get("name", "")
            if name:
                audio_models.append(name)
            continue

        name = m.get("name", "")
        if not name:
            continue
        try:
            show_req = urllib.request.Request(
                f"{url}/api/show",
                data=json.dumps({"name": name}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(show_req, timeout=5) as resp:
                details = json.loads(resp.read())
            model_info = details.get("model_info", {})
            template = details.get("template", "")
            info_str = json.dumps(model_info).lower() + template.lower()
            if "audio" in info_str:
                audio_models.append(name)
        except Exception:
            continue

    return audio_models


class LlamaServerTranscriber(_DeduplicatorMixin):
    """Transcriber that delegates to a llama-swap/llama.cpp server via /v1/chat/completions.

    Requires a llama-swap (or compatible) server running with an audio-capable
    model (e.g. Qwen2.5-Omni). Uses the OpenAI-compatible input_audio content type.
    llama-swap auto-swaps the correct model based on the model field in the request.
    """

    def __init__(self, server_url: str = "http://localhost:8080",
                 model: str = "", language: str | None = "en"):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self._language = language
        self._loaded = False
        self._init_dedup()

    def load(self):
        """Verify the llama.cpp server is reachable via /health."""
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
        except Exception as e:
            raise ConnectionError(f"Cannot reach llama server at {self.server_url}: {e}")
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}

        wav_b64 = _audio_to_wav_base64(audio)

        lang_hint = ""
        if self._language:
            from config import AVAILABLE_LANGUAGES
            lang_name = AVAILABLE_LANGUAGES.get(self._language, self._language)
            lang_hint = f" The audio is in {lang_name}."

        prompt_text = f"Transcribe the following audio exactly as spoken. Output ONLY the transcription text, nothing else.{lang_hint}"

        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                ],
            }],
        }
        if self.model:
            payload["model"] = self.model
        endpoint = f"{self.server_url}/v1/chat/completions"

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Llama server request failed: {e}") from e

        text = ((result.get("choices") or [{}])[0]
                .get("message", {}).get("content", "")).strip()

        if _is_hallucination(text, self._language):
            return {"text": "", "speaker": "", "speaker_id": 0}

        if self._is_duplicate(text):
            return {"text": "", "speaker": "", "speaker_id": 0}

        return {"text": text, "speaker": "", "speaker_id": 0}

    @property
    def is_loaded(self) -> bool:
        return self._loaded
