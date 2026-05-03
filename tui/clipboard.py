"""Cross-platform clipboard helper.

Single source of truth for which clipboard command to invoke. Both
tui/app.py and the modal screens use this so Windows support and
fallback ordering stay in one place.
"""

from __future__ import annotations

import shutil
import sys


def clipboard_cmd() -> list[str] | None:
    """Return the clipboard copy command for this platform, or None if none available."""
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
