"""Transcript reaction helpers and CLI inbox writer."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Iterable, TextIO

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


def parse_reaction_command(line: str, author: str = "external") -> dict[str, str] | None:
    """Parse one stdin bridge line.

    Accepts either the JSONL inbox shape or a shell-like command form:
    ``clap``, ``question source?``, ``idea "save this"``.
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        return parse_reaction_line(line)
    try:
        parts = shlex.split(line)
    except ValueError:
        return None
    if not parts:
        return None
    try:
        return normalize_reaction(parts[0], " ".join(parts[1:]), author)
    except ValueError:
        return None


def append_reaction_payload(path: Path, payload: dict[str, str]) -> dict[str, str]:
    """Append an already-normalized reaction payload to the inbox."""
    clean = normalize_reaction(
        payload.get("emoji", ""),
        payload.get("text", ""),
        payload.get("author", "external"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"t": round(time.time(), 4), "kind": "reaction", **clean}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return clean


def append_reaction_event(
    path: Path,
    emoji: str,
    text: str = "",
    author: str = "external",
) -> dict[str, str]:
    """Append one reaction JSON object for a running VoxTerm instance to ingest."""
    return append_reaction_payload(path, normalize_reaction(emoji, text, author))


def append_reaction_stream(
    path: Path,
    lines: Iterable[str],
    author: str = "external",
    *,
    stderr: TextIO | None = None,
) -> int:
    """Append reactions from simple line commands or JSONL records."""
    count = 0
    for line_no, line in enumerate(lines, start=1):
        payload = parse_reaction_command(line, author=author)
        if payload is None:
            if line.strip() and stderr is not None:
                print(f"ignored malformed reaction line {line_no}", file=stderr)
            continue
        append_reaction_payload(path, payload)
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send a non-verbal reaction into the running VoxTerm transcript.",
    )
    parser.add_argument(
        "emoji",
        nargs="?",
        help="Emoji or alias: " + ", ".join(f"{name}={emoji}" for name, emoji in REACTION_PRESETS),
    )
    parser.add_argument("text", nargs="*", help="Optional short text to include with the reaction")
    parser.add_argument("--author", default="external", help="Display name for the input source")
    parser.add_argument("--inbox", type=Path, default=reaction_inbox_path())
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read newline-delimited reactions from stdin for a persistent device bridge",
    )
    args = parser.parse_args(argv)

    if args.stdin:
        count = append_reaction_stream(
            args.inbox,
            sys.stdin,
            author=args.author,
            stderr=sys.stderr,
        )
        print(f"sent {count} reactions to {args.inbox}")
        return 0

    if not args.emoji:
        parser.error("provide an emoji/alias or use --stdin")

    payload = append_reaction_event(args.inbox, args.emoji, " ".join(args.text), args.author)
    print(f"sent reaction: {reaction_label(payload['emoji'], payload['text'])} ({payload['author']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
