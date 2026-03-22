"""Platform-aware directory paths for VoxTerm.

macOS:
    Data:  ~/Library/Application Support/voxterm
    User:  ~/Documents/voxterm
Linux:
    Data:  $XDG_DATA_HOME/voxterm  (default: ~/.local/share/voxterm)
    User:  $XDG_DATA_HOME/voxterm
"""

import os
import sys
from pathlib import Path

_home = Path.home()

if sys.platform == "darwin":
    DATA_DIR = _home / "Library" / "Application Support" / "voxterm"
    USER_DIR = _home / "Documents" / "voxterm"
elif sys.platform.startswith("linux"):
    _xdg_data = os.environ.get("XDG_DATA_HOME", str(_home / ".local" / "share"))
    DATA_DIR = Path(_xdg_data) / "voxterm"
    USER_DIR = DATA_DIR
else:
    # Fallback (Windows or unknown)
    DATA_DIR = _home / ".voxterm"
    USER_DIR = DATA_DIR

LIVE_DIR = USER_DIR / ".live"
BIN_DIR = USER_DIR / ".bin"
CRASH_DIR = USER_DIR / ".crashes"
SESSIONS_DIR = USER_DIR
STATE_FILE = SESSIONS_DIR / ".state.json"

DB_DIR = DATA_DIR
DB_PATH = DB_DIR / ".speakers.db"
BACKUP_DIR = DB_DIR / ".backups"
