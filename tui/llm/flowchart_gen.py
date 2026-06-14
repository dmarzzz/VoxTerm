"""Local-LLM mermaid flowchart generator.

Reuses the project's local-LLM summarizer engine (MLX on Apple Silicon,
Ollama everywhere). The model is selected by ConfigStore's
``summarization_model`` key, the same setting the U-key summarizer uses,
so a user with Ollama already configured for summaries gets flowcharts
for free.
"""

from __future__ import annotations

import re

from summarizer import Template, get_summarizer
from summarizer.engine import SummarizerError


class FlowchartGenError(Exception):
    """Raised when the LLM can't produce a usable mermaid flowchart."""


# Tight, small-model-friendly prompt. Mirrors the summarizer's
# anti-preamble discipline; tiny models (qwen3:0.6b, Qwen2.5-3B) regress
# badly without explicit format gates.
_SYSTEM = (
    "You convert short conversation transcripts into a mermaid flowchart "
    "describing the topical/interactional structure (who said what about "
    "what, and what each turn responded to).\n\n"
    "Output rules — follow exactly:\n"
    "- Output ONLY mermaid source. No preamble, no commentary, no code "
    "fences, no explanation.\n"
    "- Start the very first line with `flowchart TD`.\n"
    "- Use real speaker labels where the transcript provides them "
    "(e.g. `A[Alice: greeting]`). Do not invent names.\n"
    "- Edge labels describe the relation between turns, "
    "e.g. `-->|asks about budget|`.\n"
    "- Keep node labels short (max ~6 words). Aim for 5-15 nodes total.\n"
    "- If the transcript is empty or too short, output exactly:\n"
    "  flowchart TD\n"
    "  A[no conversation yet]"
)

_USER = (
    "Build a mermaid flowchart for this conversation.\n\n"
    'The text between triple quotes is the transcript — do not repeat it.\n'
    '"""\n{transcript}\n"""'
)

_TEMPLATE = Template(
    id="flowchart",
    label="Flowchart",
    description="Mermaid flowchart of the conversation structure.",
    system=_SYSTEM,
    user=_USER,
)


_FENCE_RE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _clean(raw: str) -> str:
    cleaned = _strip_fences(raw)
    # Some small models prepend a sentence anyway; drop everything before
    # the first `flowchart` keyword if we find it.
    idx = cleaned.lower().find("flowchart")
    if idx > 0:
        cleaned = cleaned[idx:]
    if not cleaned.lower().startswith("flowchart"):
        cleaned = "flowchart TD\n" + cleaned
    return cleaned.strip()


def generate_flowchart(transcript: str, *, model_name: str = "") -> str:
    """Run the local LLM and return cleaned mermaid source.

    ``model_name`` follows the same convention as the summarizer:
      - ``""`` → MLX default on Apple Silicon, raises otherwise.
      - ``ollama:<model>[@host]`` → Ollama HTTP backend.
      - anything else → MLX backend (Apple Silicon only).
    """
    try:
        summarizer = get_summarizer(model_name=model_name)
        raw = summarizer.summarize(transcript, _TEMPLATE)
    except SummarizerError as e:
        raise FlowchartGenError(str(e)) from e
    except Exception as e:  # defensive: surface as our error type
        raise FlowchartGenError(f"unexpected: {e}") from e

    cleaned = _clean(raw)
    if not cleaned:
        raise FlowchartGenError("empty response")
    return cleaned
