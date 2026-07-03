"""Small helpers for post-export feedback text."""

from __future__ import annotations

from pathlib import Path


def format_bytes(size: int) -> str:
    """Render a byte count compactly for status messages."""
    size = max(0, int(size))
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def format_file_saved(path: Path, entry_count: int, byte_count: int, *, summary: bool = False) -> str:
    """Status line shown after a transcript has been written to disk."""
    subject = f"{entry_count} entries"
    if summary:
        subject += " + summary"
    return (
        f"saved {subject} ({format_bytes(byte_count)}) to {path.expanduser()} "
        f"- press [E] to browse transcripts"
    )


def format_clipboard_saved(entry_count: int) -> str:
    """Status line shown after a transcript has been copied."""
    return f"copied {entry_count} entries to clipboard - transcript cleared for a new session"
