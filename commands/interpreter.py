"""Voice command interpreter — sends transcribed speech to llama-swap for intent parsing.

Classifies utterances into structured actions with parameters.
Talks to the local llama-swap instance on 127.0.0.1.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from summarizer.engine import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SEC, SummarizerError

_SYSTEM_PROMPT = """\
You are a voice command interpreter for a local-first desktop assistant.
The user speaks a command. You must parse it into a structured JSON action.

Respond with ONLY a JSON object, no markdown, no explanation. The schema:

{
  "action": "<action_name>",
  "params": { ... },
  "speech": "<short spoken confirmation to read back>"
}

Available actions:

- play_music: Play music files or a specific artist/album/song.
  params: { "query": "search terms" }  (empty query = resume playback)

- pause_music: Pause current playback.
  params: {}

- save_transcript: Save the current transcript to a file.
  params: { "filename": "optional name" }

- summarize: Summarize the current transcript.
  params: { "strength": "low|medium|high" }

- open_app: Open an application.
  params: { "app": "application name" }

- web_search: Search the web for something.
  params: { "query": "search terms" }

- set_timer: Set a timer or reminder.
  params: { "duration": "e.g. 5m, 1h", "label": "what for" }

- dictate: Start dictation mode (type what I say into the focused app).
  params: {}

- system: System command (volume, brightness, screenshot, lock).
  params: { "command": "volume_up|volume_down|mute|brightness_up|brightness_down|screenshot|lock" }

- unknown: If you cannot parse the command.
  params: { "original": "what the user said" }
  speech: "I didn't understand that."

Examples:
User: "play my music files"
{"action": "play_music", "params": {"query": ""}, "speech": "Playing music."}

User: "save this transcript as meeting notes"
{"action": "save_transcript", "params": {"filename": "meeting-notes"}, "speech": "Saved."}

User: "what's the weather in Zurich"
{"action": "web_search", "params": {"query": "weather in Zurich"}, "speech": "Searching for weather in Zurich."}

User: "turn the volume up"
{"action": "system", "params": {"command": "volume_up"}, "speech": "Volume up."}

User: "summarize what we just discussed"
{"action": "summarize", "params": {"strength": "medium"}, "speech": "Summarizing the transcript."}
"""


@dataclass
class Command:
    """A parsed voice command."""
    action: str
    params: dict = field(default_factory=dict)
    speech: str = ""
    raw_text: str = ""


def interpret(
    utterance: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> Command:
    """Parse a voice utterance into a structured Command.

    Args:
        utterance: Transcribed text from the user.
        base_url: llama-swap base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        A Command with action, params, and speech confirmation.

    Raises:
        SummarizerError: If the LLM server is unreachable.
    """
    if not utterance.strip():
        return Command(action="unknown", raw_text=utterance)

    payload = json.dumps({
        "model": "qwen2.5-3b",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
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
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SummarizerError(f"LLM server unreachable: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        parsed = json.loads(content)
        return Command(
            action=parsed.get("action", "unknown"),
            params=parsed.get("params", {}),
            speech=parsed.get("speech", ""),
            raw_text=utterance,
        )
    except (json.JSONDecodeError, KeyError, IndexError):
        return Command(
            action="unknown",
            params={"original": utterance},
            speech="I didn't understand that.",
            raw_text=utterance,
        )
