"""Transcript reaction helpers and CLI inbox writer."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from config import LIVE_DIR


REACTION_PRESETS = [
    ("clap", "👏"),
    ("heart", "❤️"),
    ("laugh", "😂"),
    ("question", "❓"),
    ("idea", "💡"),
    ("plus", "➕"),
]

REACTION_ALIASES = {name: emoji for name, emoji in REACTION_PRESETS}


def reaction_inbox_path() -> Path:
    """Path external input devices append to."""
    return Path(os.environ.get("VOXTERM_REACTION_INBOX") or (LIVE_DIR / "reactions.jsonl"))


def normalize_reaction(
    emoji: str,
    text: str = "",
    author: str = "you",
) -> dict[str, str]:
    """Return a clean reaction payload, raising ValueError for empty input."""
    emoji = REACTION_ALIASES.get(str(emoji).strip().lower(), str(emoji).strip())
    text = str(text or "").strip()
    author = str(author or "you").strip() or "you"
    if not emoji and not text:
        raise ValueError("reaction needs an emoji or text")
    return {"emoji": emoji, "text": text, "author": author}


def reaction_label(emoji: str, text: str = "") -> str:
    """Compact display text for transcript rows."""
    emoji = str(emoji or "").strip()
    text = str(text or "").strip()
    if emoji and text:
        return f"{emoji} {text}"
    return emoji or text


def parse_reaction_line(line: str) -> dict[str, str] | None:
    """Parse one external JSONL reaction event."""
    line = line.strip()
    if not line:
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("kind") not in (None, "reaction"):
        return None
    try:
        return normalize_reaction(
            str(raw.get("emoji") or raw.get("reaction") or ""),
            str(raw.get("text") or ""),
            str(raw.get("author") or "external"),
        )
    except ValueError:
        return None


def append_reaction_event(
    path: Path,
    emoji: str,
    text: str = "",
    author: str = "external",
) -> dict[str, str]:
    """Append one reaction JSON object for a running VoxTerm instance to ingest."""
    payload = normalize_reaction(emoji, text, author)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"t": round(time.time(), 4), "kind": "reaction", **payload}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send a non-verbal reaction into the running VoxTerm transcript.",
    )
    parser.add_argument(
        "emoji",
        help="Emoji or alias: " + ", ".join(f"{name}={emoji}" for name, emoji in REACTION_PRESETS),
    )
    parser.add_argument("text", nargs="*", help="Optional short text to include with the reaction")
    parser.add_argument("--author", default="external", help="Display name for the input source")
    parser.add_argument("--inbox", type=Path, default=reaction_inbox_path())
    args = parser.parse_args(argv)

    payload = append_reaction_event(args.inbox, args.emoji, " ".join(args.text), args.author)
    print(f"sent reaction: {reaction_label(payload['emoji'], payload['text'])} ({payload['author']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
