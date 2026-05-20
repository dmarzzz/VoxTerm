"""Pluggable local-LLM summarizer.

Backends (selected by the ``summarization_model`` string):
  - ``mlx``    : on-device MLX chat model via ``mlx-lm`` (macOS default).
                 Used when the model name has no recognized backend prefix.
  - ``ollama`` : a local/remote Ollama server. Selected with an
                 ``ollama:`` prefix, e.g. ``ollama:qwen3:0.6b`` or
                 ``ollama:llama3.2@http://192.168.1.5:11434``. Works on any
                 platform (it's just HTTP), so it's also the escape hatch
                 where MLX is unavailable.

The factory ``get_summarizer()`` reads ConfigStore for the active model.
Backends are loaded lazily — importing this module does not load any LLM.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import urllib.error
import urllib.request
from typing import Protocol

from .prompts import Template


def _truncate_for_context(text: str, max_chars: int) -> str:
    """Bound transcript size so it can't overflow the model context window.

    Keeps the start and the most recent tail (where conclusions/action items
    usually land), eliding the middle with a visible marker so the model —
    and anyone reading the input — knows content was dropped.
    """
    if len(text) <= max_chars:
        return text
    marker = "\n\n[… transcript truncated for length — middle omitted …]\n\n"
    budget = max_chars - len(marker)
    if budget <= 0:
        return text[:max_chars]
    head = int(budget * 0.6)
    tail = budget - head
    return text[:head] + marker + text[-tail:]


class SummarizerError(RuntimeError):
    """Raised when summarization can't be performed (missing backend, load failure, etc.)."""


class Summarizer(Protocol):
    """A local-LLM summarizer."""

    def summarize(self, transcript: str, template: Template, custom_prompt: str = "") -> str:
        ...


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

class MLXSummarizer:
    """MLX chat-model summarizer for Apple Silicon.

    Loads ``mlx_lm`` lazily on first use. The model is cached on the instance,
    so repeated summarize() calls reuse the loaded weights.
    """

    DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    # ~4 chars/token heuristic. 48k chars ≈ 12k tokens of input, leaving
    # ample headroom for the prompt + response on a 32k-context model and
    # capping peak memory even on smaller-context models.
    DEFAULT_MAX_INPUT_CHARS = 48_000

    def __init__(
        self,
        model_name: str = "",
        max_tokens: int = 800,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._max_input_chars = max_input_chars
        self._model = None
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as e:
            raise SummarizerError(
                "mlx-lm not installed. Install with: pip install mlx-lm"
            ) from e
        try:
            self._model, self._tokenizer = load(self._model_name)
        except Exception as e:
            raise SummarizerError(
                f"Failed to load MLX model '{self._model_name}': {e}"
            ) from e

    def summarize(
        self, transcript: str, template: Template, custom_prompt: str = ""
    ) -> str:
        self._load()
        from mlx_lm import generate  # type: ignore

        transcript = _truncate_for_context(transcript, self._max_input_chars)
        user_msg = template.user.format(
            transcript=transcript, custom=custom_prompt or ""
        )
        messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": user_msg},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        try:
            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                verbose=False,
            )
        except Exception as e:
            raise SummarizerError(f"MLX generation failed: {e}") from e
        return text.strip()


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaSummarizer:
    """Summarizer backed by an Ollama server (``/api/chat``).

    No SDK dependency — talks to Ollama over its HTTP API with stdlib
    ``urllib``. The model is whatever you've ``ollama pull``-ed; the host
    defaults to ``$OLLAMA_HOST`` or ``http://localhost:11434``.

    Model-string forms accepted (the ``ollama:`` prefix is stripped by the
    factory before construction):
      - ``qwen3:0.6b``                      → default host
      - ``llama3.2@http://host:11434``      → explicit host after ``@``
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MAX_INPUT_CHARS = MLXSummarizer.DEFAULT_MAX_INPUT_CHARS

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 800,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        timeout: float = 300.0,
    ):
        model, _, host = model_name.partition("@")
        self._model = model.strip()
        self._host = (
            host.strip()
            or os.environ.get("OLLAMA_HOST")
            or self.DEFAULT_HOST
        ).rstrip("/")
        if "://" not in self._host:  # bare host:port from $OLLAMA_HOST
            self._host = "http://" + self._host
        self._max_tokens = max_tokens
        self._max_input_chars = max_input_chars
        self._timeout = timeout
        if not self._model:
            raise SummarizerError("Ollama model name is empty")

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}@{self._host}"

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._host}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise SummarizerError(
                f"Ollama HTTP {e.code} from {url}: {body}"
            ) from e
        except urllib.error.URLError as e:
            raise SummarizerError(
                f"Cannot reach Ollama at {self._host} ({e.reason}). "
                f"Is it running? Try: ollama serve"
            ) from e
        except (TimeoutError, json.JSONDecodeError) as e:
            raise SummarizerError(f"Ollama request failed: {e}") from e

    def summarize(
        self, transcript: str, template: Template, custom_prompt: str = ""
    ) -> str:
        transcript = _truncate_for_context(transcript, self._max_input_chars)
        user_msg = template.user.format(
            transcript=transcript, custom=custom_prompt or ""
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": template.system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": self._max_tokens},
        }
        result = self._post("/api/chat", payload)
        text = (result.get("message") or {}).get("content", "")
        if not text.strip():
            raise SummarizerError(
                f"Ollama returned an empty summary (model '{self._model}'). "
                f"Is the model pulled? Try: ollama pull {self._model}"
            )
        return text.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, Summarizer] = {}


OLLAMA_PREFIX = "ollama:"


def get_summarizer(model_name: str = "") -> Summarizer:
    """Return a (cached) Summarizer for the requested model.

    Instances are cached by model name so loaded weights / connections
    survive across repeated invocations — pressing the summarize key again
    reuses the already-loaded backend instead of rebuilding from scratch.

    Backend dispatch is by prefix on ``model_name``:
      - ``ollama:<model>[@host]`` → Ollama HTTP backend (any platform).
      - anything else             → MLX backend (Apple Silicon macOS only).

    An empty string selects the MLX default model on Apple Silicon macOS.
    """
    key = model_name or MLXSummarizer.DEFAULT_MODEL
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached

        if model_name.startswith(OLLAMA_PREFIX):
            backend: Summarizer = OllamaSummarizer(
                model_name=model_name[len(OLLAMA_PREFIX):]
            )
        elif sys.platform != "darwin" or platform.machine() != "arm64":
            raise SummarizerError(
                "MLX summarization is only supported on Apple Silicon macOS. "
                "Run a local Ollama server and set the "
                "summarization model to e.g. 'ollama:qwen3:0.6b'."
            )
        else:
            backend = MLXSummarizer(model_name=model_name)

        _cache[key] = backend
        return backend
