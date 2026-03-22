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
    _BT_KEYWORDS = ("airpods", "bluetooth", " bt ", "beats pill", "jbl", "bose", "bluez")
    fallback = {"name": "unknown", "is_bluetooth": False}
    platform = detect_platform()

    if platform == Platform.LINUX:
        return _get_output_device_info_linux(fallback, _BT_KEYWORDS)

    if platform != Platform.MACOS:
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


def _get_output_device_info_linux(fallback: dict, bt_keywords: tuple) -> dict:
    """Get output device info on Linux via pactl."""
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, timeout=5,
        )
        sink_name = result.stdout.strip()
        if not sink_name:
            return fallback
    except Exception:
        return fallback

    is_bt = any(kw in sink_name.lower() for kw in bt_keywords)

    # Check sink properties for bluetooth transport if name didn't match
    if not is_bt:
        try:
            result = subprocess.run(
                ["pactl", "list", "sinks", "short"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if sink_name in line and "bluez" in line.lower():
                    is_bt = True
                    break
        except Exception:
            pass

    return {"name": sink_name, "is_bluetooth": is_bt}


CURRENT_PLATFORM = detect_platform()
