"""Pluggable local-LLM transcript redactor.

Design — identify, then replace (the LLM never rewrites the transcript):

  1. The transcript is split into chunks and each chunk is shown to a local
     LLM, which returns a JSON list of *verbatim* sensitive spans
     ({"text", "type"}). It does not produce redacted text.
  2. A deterministic regex pass adds high-confidence structured spans
     (emails, URLs, SSNs, phone-like and IP-like digit runs) that small
     models miss or mistype.
  3. The engine masks every span by exact string replacement over the full
     transcript — ``Alice`` → ``[NAME]``. A span the model invents that
     isn't found verbatim is simply skipped, so the rest of the transcript
     is preserved byte-for-byte and the model can't corrupt content.

Backends mirror ``summarizer`` (selected by the ``redaction_model`` string):
  - ``mlx``    : on-device MLX chat model via ``mlx-lm`` (macOS default).
  - ``ollama`` : a local/remote Ollama server, selected with an ``ollama:``
                 prefix — works on any platform and is the escape hatch
                 where MLX is unavailable.

Backends load lazily; importing this module loads no LLM.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .prompts import CATEGORIES, RedactionProfile

_VALID_TYPES = frozenset(CATEGORIES)


class RedactionError(RuntimeError):
    """Raised when redaction can't be performed (missing backend, load failure, etc.)."""


@dataclass(frozen=True)
class Finding:
    """One masked span: the original text, its category, how many times it hit."""

    text: str
    type: str
    count: int


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    findings: tuple[Finding, ...]
    counts: dict[str, int]  # category -> total replacements
    total: int  # total spans masked (sum of counts)


class Redactor(Protocol):
    """A local-LLM redactor."""

    def redact(
        self, transcript: str, profile: RedactionProfile, custom_instructions: str = ""
    ) -> RedactionResult:
        ...


# ---------------------------------------------------------------------------
# Pure helpers (no LLM) — fully unit-testable
# ---------------------------------------------------------------------------

# Deterministic, high-precision structured-PII patterns. These run regardless
# of the LLM: regex nails the formats small models fumble. Order is by
# specificity — URL/EMAIL before the looser digit-run patterns.
_REGEX_SPANS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("URL", re.compile(r"https?://[^\s)\]>\"']+")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # US SSN
    # phone-like: a run of digits and separators, >= 9 chars, no ':' so
    # transcript timestamps like 12:34:56 don't match.
    ("PHONE", re.compile(r"\+?\d[\d\-.\s()]{7,}\d")),
    ("OTHER", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),  # IPv4
)

# Spans shorter than this (after stripping) are dropped — masking 1-char
# fragments would shred the transcript for no privacy gain.
_MIN_SPAN_LEN = 2


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into <= max_chars chunks on line boundaries.

    Every line of the input lands in exactly one chunk (no truncation, no
    drops) — redaction must see the whole transcript. A single line longer
    than max_chars is hard-split.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:  # pathological single long line
            if cur:
                chunks.append("".join(cur))
                cur, cur_len = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if cur_len + len(line) > max_chars and cur:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def parse_spans(model_output: str) -> list[tuple[str, str]]:
    """Parse a model's JSON-array reply into [(text, type), ...].

    Defensive: tolerates leading/trailing prose or code fences by slicing
    to the outermost brackets, coerces unknown categories to OTHER, and
    returns [] on any parse failure rather than raising.
    """
    if not model_output:
        return []
    s = model_output.strip()
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    spans: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        typ = item.get("type")
        typ = typ.strip().upper() if isinstance(typ, str) else ""
        if typ not in _VALID_TYPES:
            typ = "OTHER"
        spans.append((text, typ))
    return spans


def regex_spans(text: str) -> list[tuple[str, str]]:
    """Find structured PII deterministically. Returns [(text, type), ...].

    Patterns are tried most-specific first (URL/EMAIL/SSN before the looser
    phone/IP digit runs) and each span is kept once with the first — i.e.
    most specific — type that matched it. An SSN like ``123-45-6789`` also
    satisfies the phone-run pattern, so this keeps it tagged ``ID``.
    """
    by_text: dict[str, str] = {}
    for typ, pattern in _REGEX_SPANS:
        for m in pattern.finditer(text):
            # Drop trailing sentence punctuation a greedy match may have
            # swallowed (e.g. a URL followed by a comma). The structured
            # spans here never legitimately end in these.
            span = m.group(0).strip().rstrip(".,;:!?")
            if len(span) >= _MIN_SPAN_LEN and span not in by_text:
                by_text[span] = typ
    return list(by_text.items())


def apply_redactions(
    text: str, spans: list[tuple[str, str]]
) -> RedactionResult:
    """Mask every verbatim occurrence of each span; return the result + tally.

    Spans are de-duplicated by exact text and applied longest-first so that a
    longer span (``Alice Smith``) is masked before a shorter overlapping one
    (``Alice``), avoiding double counting and leftover fragments. A span not
    found verbatim contributes nothing.
    """
    seen: dict[str, str] = {}  # text -> type (first wins)
    for span_text, typ in spans:
        st = span_text.strip()
        if len(st) >= _MIN_SPAN_LEN and st not in seen:
            seen[st] = typ

    redacted = text
    counts: dict[str, int] = {}
    findings: list[Finding] = []
    for span_text in sorted(seen, key=len, reverse=True):
        typ = seen[span_text]
        n = redacted.count(span_text)
        if n == 0:
            continue
        redacted = redacted.replace(span_text, f"[{typ}]")
        counts[typ] = counts.get(typ, 0) + n
        findings.append(Finding(text=span_text, type=typ, count=n))

    total = sum(counts.values())
    return RedactionResult(
        redacted_text=redacted,
        findings=tuple(findings),
        counts=counts,
        total=total,
    )


def overwrite_and_delete(path) -> None:
    """Best-effort shred: overwrite a file's bytes with random data, then
    unlink it.

    NOTE: on copy-on-write / flash filesystems (APFS, most SSDs) this does
    NOT guarantee the original bytes are unrecoverable — the overwrite may
    land on fresh blocks while the old ones linger until garbage-collected.
    It's a best-effort reduction of the on-disk plaintext, not a forensic
    wipe. We don't pretend otherwise in the UI.
    """
    p = os.fspath(path)
    try:
        if not os.path.exists(p):
            return
        size = os.path.getsize(p)
        if size:
            with open(p, "r+b", buffering=0) as f:
                f.write(os.urandom(size))
                f.flush()
                os.fsync(f.fileno())
    except OSError:
        pass  # fall through to unlink regardless
    try:
        os.remove(p)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _BaseRedactor:
    """Shared chunk → LLM → regex → replace pipeline.

    Subclasses implement ``_complete(system, user) -> str`` (one chat
    completion). Everything else — chunking, span union, masking — is the
    backend-independent logic above.
    """

    # ~4 chars/token. 6k chars ≈ 1.5k tokens/chunk: small enough for tight
    # context windows, large enough to keep the per-chunk call count sane.
    DEFAULT_MAX_CHUNK_CHARS = 6_000

    def __init__(self, max_tokens: int = 1024, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS):
        self._max_tokens = max_tokens
        self._max_chunk_chars = max_chunk_chars

    def _complete(self, system: str, user: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def redact(
        self, transcript: str, profile: RedactionProfile, custom_instructions: str = ""
    ) -> RedactionResult:
        spans: list[tuple[str, str]] = []
        for chunk in chunk_text(transcript, self._max_chunk_chars):
            user_msg = profile.user.format(
                transcript=chunk, custom=custom_instructions or ""
            )
            out = self._complete(profile.system, user_msg)
            spans.extend(parse_spans(out))
        # Deterministic backstop over the whole transcript.
        spans.extend(regex_spans(transcript))
        return apply_redactions(transcript, spans)


class MLXRedactor(_BaseRedactor):
    """MLX chat-model redactor for Apple Silicon. Loads ``mlx_lm`` lazily."""

    DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    def __init__(self, model_name: str = "", max_tokens: int = 1024, **kw):
        super().__init__(max_tokens=max_tokens, **kw)
        self._model_name = model_name or self.DEFAULT_MODEL
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
            raise RedactionError(
                "mlx-lm not installed. Install with: pip install mlx-lm"
            ) from e
        try:
            self._model, self._tokenizer = load(self._model_name)
        except Exception as e:
            raise RedactionError(
                f"Failed to load MLX model '{self._model_name}': {e}"
            ) from e

    def _complete(self, system: str, user: str) -> str:
        self._load()
        from mlx_lm import generate  # type: ignore

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        try:
            return generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                verbose=False,
            )
        except Exception as e:
            raise RedactionError(f"MLX generation failed: {e}") from e


class OllamaRedactor(_BaseRedactor):
    """Redactor backed by an Ollama server (``/api/chat``), stdlib urllib only.

    Model-string forms (the ``ollama:`` prefix is stripped by the factory):
      - ``qwen3:0.6b``                  → default host
      - ``llama3.2@http://host:11434``  → explicit host after ``@``
    """

    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        **kw,
    ):
        super().__init__(max_tokens=max_tokens, **kw)
        model, _, host = model_name.partition("@")
        self._model = model.strip()
        self._host = (
            host.strip() or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST
        ).rstrip("/")
        if "://" not in self._host:  # bare host:port from $OLLAMA_HOST
            self._host = "http://" + self._host
        self._timeout = timeout
        if not self._model:
            raise RedactionError("Ollama model name is empty")

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
            raise RedactionError(f"Ollama HTTP {e.code} from {url}: {body}") from e
        except urllib.error.URLError as e:
            raise RedactionError(
                f"Cannot reach Ollama at {self._host} ({e.reason}). "
                f"Is it running? Try: ollama serve"
            ) from e
        except (TimeoutError, json.JSONDecodeError) as e:
            raise RedactionError(f"Ollama request failed: {e}") from e

    def _complete(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": self._max_tokens},
        }
        result = self._post("/api/chat", payload)
        return (result.get("message") or {}).get("content", "") or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, Redactor] = {}

OLLAMA_PREFIX = "ollama:"
PRIVACY_FILTER_NAMES = frozenset({"privacy-filter", "openai/privacy-filter", "pf"})
PRIVACY_FILTER_PREFIXES = ("privacy-filter:", "pf:")


def get_redactor(model_name: str = "") -> Redactor:
    """Return a (cached) Redactor for the requested model.

    Cached by model name so loaded weights / connections survive across
    repeated invocations. Dispatch is by name/prefix:
      - ``privacy-filter[:<repo>]`` → OpenAI Privacy Filter via onnxruntime
        (any platform; identifiers + secrets only — pair with a chat backend
        for content-classes / proper nouns).
      - ``ollama:<model>[@host]``   → Ollama HTTP backend (any platform).
      - anything else               → MLX backend (Apple Silicon macOS only).

    An empty string selects the MLX default model on Apple Silicon macOS.
    """
    key = model_name or MLXRedactor.DEFAULT_MODEL
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached

        if model_name in PRIVACY_FILTER_NAMES or model_name.startswith(
            PRIVACY_FILTER_PREFIXES
        ):
            from .privacy_filter import PrivacyFilterRedactor

            repo = ""
            for pfx in PRIVACY_FILTER_PREFIXES:
                if model_name.startswith(pfx):
                    repo = model_name[len(pfx):].strip()
                    break
            backend: Redactor = PrivacyFilterRedactor(repo=repo)
        elif model_name.startswith(OLLAMA_PREFIX):
            backend = OllamaRedactor(model_name=model_name[len(OLLAMA_PREFIX):])
        elif sys.platform != "darwin" or platform.machine() != "arm64":
            raise RedactionError(
                "On-device MLX redaction is only supported on Apple Silicon "
                "macOS. Use the cross-platform 'privacy-filter' backend, or run "
                "a local Ollama server and set 'ollama:qwen3:0.6b'."
            )
        else:
            backend = MLXRedactor(model_name=model_name)

        _cache[key] = backend
        return backend
