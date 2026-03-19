"""Platform detection for audio capture backends."""

import sys
import shutil
import subprocess
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


def get_output_device_info() -> dict:
    """Return info about the current default output audio device.

    Returns dict with at least {"name": str, "is_bluetooth": bool}.
    On failure, returns {"name": "unknown", "is_bluetooth": False}.
    """
    _BT_KEYWORDS = ("airpods", "bluetooth", " bt ", "beats pill", "jbl", "bose")
    fallback = {"name": "unknown", "is_bluetooth": False}

    if detect_platform() != Platform.MACOS:
        return fallback

    try:
        result = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "SPAudioDataType"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
    except Exception:
        return fallback

    # Parse SPAudioDataType — device names are section headers,
    # "Default Output Device: Yes" appears indented beneath them.
    # Also look for "Transport: Bluetooth" in the same section.
    device_name = ""
    transport = ""
    current_section = ""
    for line in output.splitlines():
        stripped = line.strip()

        # Section headers end with ":" and are indented exactly 8 spaces
        if line.startswith("        ") and stripped.endswith(":") and not stripped.startswith(("Default", "Transport", "Manufacturer", "Input", "Output", "Current")):
            current_section = stripped.rstrip(":")
            transport = ""

        if stripped.startswith("Transport:"):
            transport = stripped.split(":", 1)[1].strip().lower()

        if stripped == "Default Output Device: Yes":
            device_name = current_section
            break

    if not device_name:
        # Try sounddevice as secondary approach
        try:
            import sounddevice as sd
            dev = sd.query_devices(kind="output")
            if dev and isinstance(dev, dict):
                device_name = dev.get("name", "")
        except Exception:
            pass

    if not device_name:
        return fallback

    # Check Bluetooth via transport type from system_profiler
    is_bt = transport == "bluetooth"

    # Also check by device name keywords
    if not is_bt:
        name_lower = device_name.lower()
        is_bt = any(kw in name_lower for kw in _BT_KEYWORDS)

    return {"name": device_name, "is_bluetooth": is_bt}


CURRENT_PLATFORM = detect_platform()
