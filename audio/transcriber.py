"""Transcription engine — Qwen3-ASR (all platforms), mlx-whisper (macOS), faster-whisper (Linux fallback)."""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

from audio.platform import CURRENT_PLATFORM, Platform


def configure_mlx_memory() -> None:
    """Bound MLX's Metal allocator so the GPU buffer cache can't grow unbounded.

    MLX keeps freed buffers in a reuse cache that, by default, grows toward
    the device working-set size. With variable-length audio chunks every
    transcription caches new buffer sizes, so RSS climbs monotonically over
    a session. A cache ceiling caps that without measurably hurting latency
    (the model's live working set is well under the limit). Tunable via
    VOXTERM_MLX_CACHE_MB (0 disables the cap).

    Process-wide and idempotent — called from every entry point that loads
    an MLX transcriber (TUI and dictation) so both are capped.
    """
    if sys.platform != "darwin":
        return
    try:
        import mlx.core as mx
    except Exception:
        return
    try:
        cache_mb = int(os.environ.get("VOXTERM_MLX_CACHE_MB", "512"))
    except ValueError:
        cache_mb = 512
    if cache_mb <= 0:
        return
    try:
        mx.set_cache_limit(cache_mb * 1024 * 1024)
    except Exception:
        # Older/newer MLX without set_cache_limit — non-fatal.
        pass


_ASR_SR = 16000  # Qwen3-ASR fixed input rate


def _pad_to_shape_bucket(audio: np.ndarray) -> np.ndarray:
    """Snap an audio buffer's length up to a fixed grid by zero-padding.

    Real-time capture triggers transcription on silence *or* at the buffer
    cap, so the sample count differs on essentially every call (16k–48k).
    MLX allocates GPU tensors sized to the input, so a unique length each
    call means a unique allocation each call — the Metal allocator fragments
    and RSS climbs monotonically (~10 MB/transcription observed) even though
    the buffer cache is bounded and cleared. The ASR library documents the
    same failure mode for long audio; we reintroduce it *across* calls via
    variable chunk lengths.

    Rounding the length up to a coarse grid collapses thousands of distinct
    shapes down to a handful (e.g. {1s, 2s, 3s}), so MLX reuses the same
    buffer slabs every call and RSS stays flat. Trailing silence is safe:
    Qwen3-ASR tolerates it, and the caller's RMS energy gate runs on the
    original (unpadded) audio before this is called. Tunable via
    VOXTERM_ASR_PAD_SECONDS (0 disables).
    """
    try:
        grid_sec = float(os.environ.get("VOXTERM_ASR_PAD_SECONDS", "1.0"))
    except ValueError:
        grid_sec = 1.0
    # Reject non-positive and non-finite (nan/inf) tunables — otherwise
    # round(grid_sec * _ASR_SR) below raises mid-transcription.
    if not math.isfinite(grid_sec) or grid_sec <= 0:
        return audio
    n = len(audio)
    if n == 0:
        return audio
    grid = max(1, int(round(grid_sec * _ASR_SR)))
    target = ((n + grid - 1) // grid) * grid
    if target <= n:
        return audio
    return np.concatenate(
        [audio, np.zeros(target - n, dtype=audio.dtype)]
    )


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
            # Pad to a fixed shape bucket so MLX reuses GPU buffers across
            # calls instead of fragmenting the Metal heap (RMS gate above
            # already ran on the unpadded audio).
            audio = _pad_to_shape_bucket(audio)
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


# NEAR AI Cloud — OpenAI-compatible gateway, serves openai/whisper-large-v3.
NEAR_AI_BASE_URL = "https://cloud-api.near.ai/v1"


def _load_cloud_api_key() -> str | None:
    """API key from NEAR_AI_API_KEY env, else DATA_DIR/near_ai.key."""
    key = os.environ.get("NEAR_AI_API_KEY")
    if key:
        return key.strip()
    from config import DATA_DIR
    key_file = DATA_DIR / "near_ai.key"
    return key_file.read_text().strip() if key_file.exists() else None


class CloudTranscriber(_DeduplicatorMixin):
    """Remote transcription via an OpenAI-compatible /audio/transcriptions endpoint.

    Defaults to NEAR AI Cloud (openai/whisper-large-v3) but works with any
    OpenAI-compatible speech-to-text API. Each numpy chunk is encoded as an
    in-memory WAV and POSTed; VAD and diarization still run locally.
    """

    def __init__(self, model: str = "openai/whisper-large-v3", language: str | None = "en",
                 base_url: str | None = None, api_key: str | None = None):
        self.model_id = model
        self._language = language
        self._base_url = (base_url or os.environ.get("NEAR_AI_BASE_URL", NEAR_AI_BASE_URL)).rstrip("/")
        self._api_key = api_key or _load_cloud_api_key()
        self._loaded = False
        self._init_dedup()

    def load(self):
        if not self._api_key:
            from config import DATA_DIR
            raise RuntimeError(
                "No cloud API key — set NEAR_AI_API_KEY or write it to "
                f"{DATA_DIR / 'near_ai.key'}"
            )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}

        text = self._post_audio(audio)

        if _is_hallucination(text, self._language):
            return {"text": "", "speaker": "", "speaker_id": 0}
        if self._is_duplicate(text):
            return {"text": "", "speaker": "", "speaker_id": 0}
        return {"text": text, "speaker": "", "speaker_id": 0}

    def _post_audio(self, audio: np.ndarray) -> str:
        import io
        import json
        import uuid
        import urllib.request
        from scipy.io import wavfile
        from config import SAMPLE_RATE

        buf = io.BytesIO()
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(buf, SAMPLE_RATE, pcm)

        boundary = "----voxterm" + uuid.uuid4().hex

        def _field(name: str, value: str) -> bytes:
            return (f'--{boundary}\r\nContent-Disposition: form-data; '
                    f'name="{name}"\r\n\r\n{value}\r\n').encode()

        body = _field("model", self.model_id)
        if self._language:
            body += _field("language", self._language)
        body += _field("response_format", "json")
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
        body += buf.getvalue() + b"\r\n" + f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{self._base_url}/audio/transcriptions", data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("text", "").strip()

    @property
    def is_loaded(self) -> bool:
        return self._loaded
