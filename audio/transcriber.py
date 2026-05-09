"""Transcription engine — Qwen3-ASR (all platforms), mlx-whisper (macOS), faster-whisper (Linux fallback), llama server (remote)."""

from __future__ import annotations

import base64
import io
import json
import re
import struct
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


class Qwen3Transcriber(_DeduplicatorMixin):
    """Qwen3-ASR transcriber — MLX on macOS, qwen-asr (PyTorch) on Linux."""

    def __init__(self, model: str = "Qwen/Qwen3-ASR-0.6B", language: str | None = "en"):
        self.model_id = model
        self._language = language
        self._model = None
        self._loaded = False
        self._use_mlx = CURRENT_PLATFORM == Platform.MACOS
        self._init_dedup()

    def load(self):
        """Pre-load the model (downloads on first run)."""
        if self._use_mlx:
            from mlx_qwen3_asr import load_model
            model, _config = load_model(self.model_id)
            self._model = model
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
            from mlx_qwen3_asr import transcribe
            result = transcribe(
                audio,
                model=self._model if self._model else self.model_id,
                language=self._language,
                verbose=False,
            )
            text = str(result.text).strip() if hasattr(result, 'text') else ""
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


def _audio_to_spectrogram_base64(audio: np.ndarray, sample_rate: int = 16000,
                                   n_fft: int = 1024, hop_length: int = 256,
                                   n_mels: int = 128) -> str:
    """Convert float32 audio to a mel spectrogram PNG image, returned as base64.

    Uses pure numpy + scipy for the STFT and mel filterbank — no librosa needed.
    The resulting image is a grayscale PNG with time on the x-axis and frequency
    on the y-axis (low frequencies at the bottom).
    """
    # Pad audio to at least one full FFT frame
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    # ── STFT via scipy ──────────────────────────────────────────
    from scipy.signal import stft as scipy_stft
    _freqs, _times, Zxx = scipy_stft(
        audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length,
        boundary=None, padded=False,
    )
    magnitude = np.abs(Zxx)  # (n_fft//2+1, time_frames)

    # ── Mel filterbank (numpy) ──────────────────────────────────
    n_freqs = magnitude.shape[0]
    fmin, fmax = 0.0, sample_rate / 2.0

    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(fft_freqs):
            if lo <= f <= mid and mid > lo:
                filterbank[i, j] = (f - lo) / (mid - lo)
            elif mid < f <= hi and hi > mid:
                filterbank[i, j] = (hi - f) / (hi - mid)

    # ── Mel spectrogram in dB ───────────────────────────────────
    mel_spec = filterbank @ magnitude  # (n_mels, time_frames)
    mel_spec_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))

    # Normalize to 0–255 for image
    db_min, db_max = mel_spec_db.min(), mel_spec_db.max()
    if db_max - db_min > 0:
        mel_norm = (mel_spec_db - db_min) / (db_max - db_min)
    else:
        mel_norm = np.zeros_like(mel_spec_db)

    # Flip vertically so low frequencies are at the bottom
    img_array = np.flipud((mel_norm * 255).astype(np.uint8))

    # ── Encode as PNG ───────────────────────────────────────────
    import zlib

    height, width = img_array.shape

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    raw_rows = b""
    for row in img_array:
        raw_rows += b"\x00" + row.tobytes()

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw_rows))
    png += _png_chunk(b"IEND", b"")

    return base64.b64encode(png).decode("ascii")


class SpectrogramTranscriber(_DeduplicatorMixin):
    """Transcriber that converts audio to mel spectrogram images and sends them
    to a multimodal vision model (e.g. Qwen2.5-VL, Qwen3.5-VL) via a
    llama-swap/llama.cpp server's /v1/chat/completions endpoint.

    The model "reads" the spectrogram and produces a transcription.
    """

    def __init__(self, server_url: str = "http://localhost:8080",
                 model: str = "", language: str | None = "en"):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self._language = language
        self._loaded = False
        self._init_dedup()

    def load(self):
        """Verify the llama.cpp server is reachable."""
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

        spectrogram_b64 = _audio_to_spectrogram_base64(audio)

        lang_hint = ""
        if self._language:
            from config import AVAILABLE_LANGUAGES
            lang_name = AVAILABLE_LANGUAGES.get(self._language, self._language)
            lang_hint = f" The speech is in {lang_name}."

        prompt_text = (
            "This is a mel spectrogram of human speech. The x-axis is time and the "
            "y-axis is frequency (low at bottom, high at top). Transcribe the spoken "
            "words exactly as said. Output ONLY the transcription text, nothing else. "
            "Do not describe the image or add commentary."
            f"{lang_hint}"
        )

        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{spectrogram_b64}",
                        },
                    },
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 512,
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
