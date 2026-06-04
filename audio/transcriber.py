"""Transcription engine — Qwen3-ASR (all platforms), mlx-whisper (macOS), faster-whisper (Linux fallback)."""

from __future__ import annotations

import math
import os
import platform
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


class ParakeetTranscriber(_DeduplicatorMixin):
    """NVIDIA Parakeet FastConformer transcriber — MLX on Apple Silicon.

    Runs NVIDIA's Parakeet family (FastConformer encoder + TDT/RNNT/CTC decoder)
    on Metal via the `parakeet-mlx` port. Wired models:
      - `mlx-community/parakeet-tdt-0.6b-v3`  (0.6B, multilingual)
      - `mlx-community/parakeet-tdt-1.1b`     (1.1B, higher accuracy)

    The models ship their own punctuation/capitalisation and infer language
    internally, so the `language` argument is accepted for interface parity
    (and the hallucination filter) but not passed to the model.

    Note on `nvidia/nemotron-speech-streaming-en-0.6b`: that cache-aware
    *streaming* model uses causal downsampling + batch-norm convs that
    parakeet-mlx 0.5.1 does not implement (its loader raises on
    `causal_downsampling`/subsampling), and NeMo itself is CUDA/Linux-only.
    So the streaming model can't run on this MLX stack today; the 0.6B TDT
    above is its supported, non-streaming sibling.
    """

    def __init__(self, model: str = "mlx-community/parakeet-tdt-1.1b", language: str | None = "en"):
        self.model_id = model
        self._language = language
        self._model = None
        self._loaded = False
        self._init_dedup()

    def load(self):
        """Pre-load the model (downloads on first run)."""
        # parakeet-mlx ships Apple-Silicon-only wheels; CURRENT_PLATFORM ==
        # MACOS is also true on Intel macs, so check the arch explicitly to
        # raise a clear error instead of an opaque ModuleNotFoundError.
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("Parakeet models require Apple Silicon (MLX).")
        try:
            from parakeet_mlx import from_pretrained
        except ImportError as e:
            raise RuntimeError(
                "parakeet-mlx is not installed; install it to use Parakeet "
                "models (pip install parakeet-mlx)."
            ) from e
        self._model = from_pretrained(self.model_id)
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        """Transcribe audio array (float32, 16kHz mono).

        Returns:
            {"text": str, "speaker": str, "speaker_id": int}
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}

        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        # Pad to a fixed shape bucket so MLX reuses Metal GPU buffers across
        # calls instead of fragmenting the heap (RMS gate above already ran on
        # the unpadded audio). Same rationale as Qwen3Transcriber.
        audio = _pad_to_shape_bucket(audio)

        mel = get_logmel(
            mx.array(audio, dtype=mx.float32)[None],
            self._model.preprocessor_config,
        )
        results = self._model.generate(mel)
        result = results[0] if isinstance(results, list) else results
        text = str(getattr(result, "text", "")).strip()

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


# --- optional cross-platform streaming backend (sherpa-onnx) -----------------

_SHERPA_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
# repo dir name -> download tarball. All are transducer models (encoder/decoder/joiner/tokens).
_SHERPA_MODEL_URLS = {
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17":
        f"{_SHERPA_RELEASE}/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
    "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25":
        f"{_SHERPA_RELEASE}/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
}


def _model_complete(d: "Path") -> bool:
    """True only when ALL four artifacts are present (so a half-extracted dir isn't trusted)."""
    return (d.is_dir() and any(d.glob("*encoder*.onnx")) and any(d.glob("*decoder*.onnx"))
            and any(d.glob("*joiner*.onnx")) and (d / "tokens.txt").exists())


def _ensure_sherpa_model(repo: str) -> "Path":
    """Return a local dir holding the streaming-zipformer ONNX files, downloading + extracting
    the published tarball on first use. Cached under ~/.cache/voxterm/sherpa/<repo>/. Both the
    download and the extraction are atomic, and a partial/corrupt cache self-heals."""
    from pathlib import Path
    import shutil
    import tarfile
    import urllib.request

    cache = Path.home() / ".cache" / "voxterm" / "sherpa"
    target = cache / repo
    if _model_complete(target):
        return target
    shutil.rmtree(target, ignore_errors=True)               # wipe any partial/corrupt extraction
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / (repo + ".tar.bz2")
    if not tarball.exists():
        tmp = tarball.with_suffix(".part")
        try:
            urllib.request.urlretrieve(_SHERPA_MODEL_URLS[repo], tmp)  # noqa: S310 (pinned github release URL)
        except Exception:
            tmp.unlink(missing_ok=True)                     # don't leak a partial download
            raise
        tmp.rename(tarball)
    # extract into a sibling staging dir, then atomically move into place — an interrupted
    # extraction never leaves a half-populated dir that _model_complete would accept.
    staging = cache / (repo + ".extracting")
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(tarball, "r:bz2") as tf:
        tf.extractall(staging, filter="data")               # produces staging/<repo>/ (safe filter)
    extracted = staging / repo
    if not _model_complete(extracted):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"sherpa model tarball for {repo} is incomplete after extraction")
    extracted.rename(target)
    shutil.rmtree(staging, ignore_errors=True)
    return target


class SherpaStreamingTranscriber(_DeduplicatorMixin):
    """Cross-platform CPU streaming ASR via sherpa-onnx (k2-fsa). Optional backend — only
    reachable when sherpa-onnx is installed (config gates the model key). Per-call
    create_stream makes it a drop-in for the existing chunked callers; the live loop can also
    drive it as a true streaming recognizer."""

    def __init__(self, model: str = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
                 language: str | None = "en"):
        self.model_id = model
        self._language = language
        self._rec = None
        self._loaded = False
        self._init_dedup()

    def load(self):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise RuntimeError(
                'sherpa-onnx is not installed; install it to use streaming models '
                '(pip install "voxterm[streaming]" or pip install sherpa-onnx).'
            ) from e
        d = _ensure_sherpa_model(self.model_id)

        def _pick(*globs):
            for g in globs:
                hits = sorted(d.glob(g))
                if hits:
                    return str(hits[0])
            raise RuntimeError(
                f"sherpa model dir {d} is missing a required file (looked for {globs!r}); "
                "delete it and re-run to re-download."
            )

        enc = _pick("*encoder*.int8.onnx", "*encoder*.onnx")
        dec = _pick("*decoder*.int8.onnx", "*decoder*.onnx")
        joi = _pick("*joiner*.int8.onnx", "*joiner*.onnx")
        tokens = _pick("tokens.txt")
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=enc, decoder=dec, joiner=joi,
            num_threads=2, provider="cpu", enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4, rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return {"text": "", "speaker": "", "speaker_id": 0}
        s = self._rec.create_stream()
        s.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
        while self._rec.is_ready(s):
            self._rec.decode_stream(s)
        s.input_finished()
        while self._rec.is_ready(s):
            self._rec.decode_stream(s)
        text = (self._rec.get_result(s) or "").strip()
        if text and text.isupper():
            # only sentence-case models that emit ALL-CAPS (zipformer); leave models with native
            # casing + punctuation (nemotron) untouched.
            text = text.capitalize()
        if not text or _is_hallucination(text, self._language) or self._is_duplicate(text):
            return {"text": "", "speaker": "", "speaker_id": 0}
        return {"text": text, "speaker": "", "speaker_id": 0}

    @property
    def is_loaded(self) -> bool:        # @property to match every other backend's contract
        return self._loaded


def get_transcriber(model_name: str, *, language: str | None = "en"):
    """Construct the transcriber backend for a model key.

    Centralizes the model_key -> backend selection that was hand-written
    identically in three places (tui/app.py x2, dictation/app.py). Returns an
    UNLOADED transcriber; the caller invokes ``.load()`` (typically on a dedicated
    MLX executor thread). Raises KeyError for an unknown model_name, same as the
    inline ``AVAILABLE_MODELS[...]`` lookups it replaces.
    """
    from config import (
        AVAILABLE_MODELS,
        FASTER_WHISPER_MODELS,
        PARAKEET_MODELS,
        QWEN3_MODELS,
        SHERPA_MODELS,
    )

    model_repo = AVAILABLE_MODELS[model_name]
    if model_name in QWEN3_MODELS:
        return Qwen3Transcriber(model=model_repo, language=language)
    if model_name in PARAKEET_MODELS:
        return ParakeetTranscriber(model=model_repo, language=language)
    if model_name in SHERPA_MODELS:
        return SherpaStreamingTranscriber(model=model_repo, language=language)
    if model_name in FASTER_WHISPER_MODELS:
        return FasterWhisperTranscriber(model=model_repo, language=language)
    return WhisperTranscriber(model=model_repo)
