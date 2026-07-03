"""Local-LLM mermaid flowchart generator for live transcript structure."""

from __future__ import annotations

import re

from summarizer import Template, get_summarizer
from summarizer.engine import SummarizerError


class FlowchartGenError(Exception):
    """Raised when a flowchart cannot be generated."""


_SYSTEM = (
    "You convert short conversation transcripts into mermaid flowchart source "
    "describing the topical and interaction structure.\n\n"
    "Output rules:\n"
    "- Output ONLY mermaid source. No preamble, commentary, code fences, or explanation.\n"
    "- Start the first line with `flowchart TD`.\n"
    "- Use speaker labels from the transcript. Do not invent names.\n"
    "- Keep node labels short, around six words or less.\n"
    "- Aim for 5-15 nodes total.\n"
    "- If the transcript is empty or too short, output exactly:\n"
    "  flowchart TD\n"
    "  A[no conversation yet]"
)

_USER = (
    "Build a mermaid flowchart for this conversation.\n\n"
    "The text between triple quotes is the transcript. Do not repeat it.\n"
    "\"\"\"\n{transcript}\n\"\"\""
)

_TEMPLATE = Template(
    id="flowchart",
    label="Flowchart",
    description="Mermaid flowchart of the conversation structure.",
    system=_SYSTEM,
    user=_USER,
)

_FENCE_RE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def clean_mermaid(raw: str) -> str:
    """Clean small-model output into mermaid source starting with flowchart."""
    cleaned = _strip_fences(raw)
    idx = cleaned.lower().find("flowchart")
    if idx > 0:
        cleaned = cleaned[idx:]
    if not cleaned.lower().startswith("flowchart"):
        cleaned = "flowchart TD\n" + cleaned
    return cleaned.strip()


def generate_flowchart(transcript: str, *, model_name: str = "") -> str:
    """Run the configured local LLM and return cleaned mermaid source."""
    try:
        summarizer = get_summarizer(model_name=model_name)
        raw = summarizer.summarize(transcript, _TEMPLATE)
    except SummarizerError as exc:
        raise FlowchartGenError(str(exc)) from exc
    except Exception as exc:
        raise FlowchartGenError(f"unexpected: {exc}") from exc

    cleaned = clean_mermaid(raw)
    if not cleaned:
        raise FlowchartGenError("empty response")
    return cleaned
