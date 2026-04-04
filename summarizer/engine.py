"""Local LLM summarizer — talks to llama-server on localhost.

Uses the OpenAI-compatible /v1/chat/completions endpoint exposed by
llama-server.  Zero network calls beyond 127.0.0.1.

Requires: a running llama-server (see `nix run .#llm-server`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Literal

Strength = Literal["low", "medium", "high"]

_SYSTEM_PROMPTS: dict[Strength, str] = {
    "low": (
        "You are a concise meeting summarizer. "
        "Given a transcript, produce 3-5 bullet points capturing the key topics discussed. "
        "No filler, no preamble. Just the bullets."
    ),
    "medium": (
        "You are a meeting summarizer. Given a transcript, produce a structured summary with:\n"
        "- **Topics discussed**: bullet list of main subjects\n"
        "- **Decisions made**: any conclusions or agreements reached\n"
        "- **Action items**: tasks assigned, with who is responsible if mentioned\n"
        "Be concise but thorough."
    ),
    "high": (
        "You are a detailed meeting minutes writer. Given a transcript, produce comprehensive minutes:\n"
        "- **Attendees**: list speakers identified in the transcript\n"
        "- **Agenda / Topics**: each major topic with a short paragraph of discussion context\n"
        "- **Decisions**: specific decisions made, with rationale if stated\n"
        "- **Action items**: each task, assignee, and deadline if mentioned\n"
        "- **Open questions**: unresolved items that need follow-up\n"
        "Write in clear, professional language."
    ),
}

DEFAULT_BASE_URL = "http://127.0.0.1:8081"
DEFAULT_TIMEOUT_SEC = 120


class SummarizerError(Exception):
    """Raised when the LLM server is unreachable or returns an error."""


def summarize(
    transcript: str,
    *,
    strength: Strength = "medium",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Summarize a transcript using the local LLM server.

    Args:
        transcript: The full transcript text (markdown or plain text).
        strength: Summary detail level — "low", "medium", or "high".
        base_url: llama-server base URL (default 127.0.0.1:8081).
        timeout: HTTP request timeout in seconds.

    Returns:
        The summary text.

    Raises:
        SummarizerError: If the server is unreachable or returns an error.
    """
    if not transcript.strip():
        return ""

    system_prompt = _SYSTEM_PROMPTS[strength]

    payload = json.dumps({
        "model": "local",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the transcript:\n\n{transcript}"},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        "stream": False,
    }).encode()

    url = f"{base_url}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise SummarizerError(
            f"Cannot reach LLM server at {base_url} — "
            f"is voxterm-llm-server running? ({exc})"
        ) from exc
    except TimeoutError as exc:
        raise SummarizerError(
            f"LLM server timed out after {timeout}s — "
            "transcript may be too long or model too slow"
        ) from exc

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise SummarizerError(
            f"Unexpected response from LLM server: {json.dumps(body)[:200]}"
        ) from exc


def is_server_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check if the llama-server is reachable."""
    url = f"{base_url}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False
