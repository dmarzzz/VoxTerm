"""Platform detection for audio capture backends."""

import sys
import shutil
from enum import Enum, auto


class Platform(Enum):
    MACOS = auto()
    WINDOWS = auto()
    LINUX = auto()
    UNKNOWN = auto()


def detect_platform() -> Platform:
    if sys.platform == "darwin":
        return Platform.MACOS
    elif sys.platform == "win32":
        return Platform.WINDOWS
    elif sys.platform.startswith("linux"):
        return Platform.LINUX
    return Platform.UNKNOWN


def has_swiftc() -> bool:
    return detect_platform() == Platform.MACOS and shutil.which("swiftc") is not None


CURRENT_PLATFORM = detect_platform()
