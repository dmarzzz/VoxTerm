"""OpenAI Privacy Filter detection backend (ONNX, via onnxruntime).

`openai/privacy-filter` is a purpose-built *token-classification* model for PII
(gpt-oss derivative, 1.5B total / 50M active, Apache-2.0). Unlike the chat
backends, it returns labeled spans with character offsets directly — no JSON
to parse, no risk of the model "inventing" a span that isn't in the text.

OpenAI publish ONNX weights for it, so it runs through **onnxruntime** — which
VoxTerm already ships (VAD / diarization / LID). That keeps it torch-free and
gives non-Apple-Silicon users a fully-local detector without an Ollama server.
The heavy runtime (optimum + transformers, which drive onnxruntime) is an
OPTIONAL extra — imported lazily, with a clear install message if absent.

Scope: identifiers + secrets only (the model's 8 labels). It does NOT detect
the sensitivity content-classes (substances/health/legal/…) or proper nouns
(ORG/PROJECT) that the ROOM/WORLD tiers want — pair it with a chat pass for
those (the hybrid tracked in #150). The regex backstop still runs.
"""

from __future__ import annotations

from .engine import RedactionError, RedactionResult, apply_redactions, chunk_text, regex_spans

REPO = "openai/privacy-filter"

# Privacy Filter's labels → our category vocabulary (redaction/prompts.py).
_LABEL_MAP = {
    "private_person": "NAME",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "ADDRESS",
    "private_url": "URL",
    "account_number": "ID",
    "private_date": "DATE",
    "secret": "CREDENTIAL",
}


def _normalize_label(raw: str) -> str:
    """Strip any B-/I- prefix and lowercase a token-classification label."""
    label = (raw or "").strip()
    if len(label) > 2 and label[1] == "-" and label[0].lower() in ("b", "i"):
        label = label[2:]
    return label.lower()


def map_label(raw: str) -> str | None:
    """Map a model label to our category, or None if it's not a PII label."""
    return _LABEL_MAP.get(_normalize_label(raw))


def spans_from_entities(
    chunk: str, entities: list[dict], score_threshold: float
) -> list[tuple[str, str]]:
    """Turn a token-classifier's output for one chunk into (text, type) spans.

    Pure (no model) so it's unit-testable. Uses the char offsets to slice the
    ORIGINAL chunk — guaranteeing the span is verbatim and so masks cleanly —
    and falls back to the entity's own text when offsets are absent.
    """
    spans: list[tuple[str, str]] = []
    for ent in entities:
        cat = map_label(str(ent.get("entity_group") or ent.get("label") or ""))
        if cat is None:
            continue
        try:
            if float(ent.get("score", 1.0)) < score_threshold:
                continue
        except (TypeError, ValueError):
            pass
        start, end = ent.get("start"), ent.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(chunk):
            text = chunk[start:end].strip()
        else:
            text = str(ent.get("word") or ent.get("text") or "").strip()
        if text:
            spans.append((text, cat))
    return spans


class PrivacyFilterRedactor:
    """Redactor backed by OpenAI Privacy Filter via onnxruntime.

    Selected by the model string ``privacy-filter`` (optionally
    ``privacy-filter:<repo-or-onnx-variant>``). Loads lazily.
    """

    DEFAULT_MAX_CHUNK_CHARS = 6_000

    def __init__(
        self,
        repo: str = "",
        score_threshold: float = 0.5,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ):
        self._repo = repo or REPO
        self._threshold = score_threshold
        self._max_chunk_chars = max_chunk_chars
        self._pipe = None

    @property
    def model_name(self) -> str:
        return f"privacy-filter:{self._repo}"

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            # ORT* runs the ONNX weights on onnxruntime — no torch needed.
            from optimum.onnxruntime import ORTModelForTokenClassification
            from transformers import AutoTokenizer, pipeline
        except ImportError as e:
            raise RedactionError(
                "Privacy Filter needs the optional extra. Install with: "
                "pip install 'voxterm[privacy-filter]'  "
                "(optimum[onnxruntime] + transformers)."
            ) from e
        try:
            model = ORTModelForTokenClassification.from_pretrained(self._repo)
            tokenizer = AutoTokenizer.from_pretrained(self._repo)
            self._pipe = pipeline(
                task="token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        except Exception as e:
            raise RedactionError(
                f"Failed to load Privacy Filter '{self._repo}': {e}"
            ) from e

    def _classify(self, chunk: str) -> list[dict]:
        self._load()
        try:
            return list(self._pipe(chunk))
        except Exception as e:
            raise RedactionError(f"Privacy Filter inference failed: {e}") from e

    def detect(self, transcript: str) -> list[tuple[str, str]]:
        spans: list[tuple[str, str]] = []
        for chunk in chunk_text(transcript, self._max_chunk_chars):
            spans.extend(
                spans_from_entities(chunk, self._classify(chunk), self._threshold)
            )
        spans.extend(regex_spans(transcript))  # structured-PII backstop
        return spans

    def redact(self, transcript: str, profile=None, custom_instructions: str = "") -> RedactionResult:
        # profile/custom are ignored: this is a fixed classifier, not a prompt.
        return apply_redactions(transcript, self.detect(transcript))
