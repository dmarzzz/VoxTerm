"""Cross-platform clipboard command selection for TUI copy actions."""

from __future__ import annotations

import shutil
import sys


def clipboard_cmd() -> list[str] | None:
    """Return the clipboard copy command for this platform, or None."""
    if sys.platform == "darwin":
        return ["pbcopy"]
    if sys.platform == "win32":
        return ["clip.exe"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    if shutil.which("wl-copy"):
        return ["wl-copy"]
    return None
