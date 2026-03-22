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

    platform = detect_platform()
    if platform == Platform.MACOS:
        return _get_output_device_info_macos(_BT_KEYWORDS, fallback)
    elif platform == Platform.LINUX:
        return _get_output_device_info_linux(_BT_KEYWORDS, fallback)
    return fallback


def _get_output_device_info_macos(bt_keywords: tuple, fallback: dict) -> dict:
    """macOS: use system_profiler to detect output device and Bluetooth."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "SPAudioDataType"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
    except Exception:
        return fallback

    device_name = ""
    transport = ""
    current_section = ""
    for line in output.splitlines():
        stripped = line.strip()

        if line.startswith("        ") and stripped.endswith(":") and not stripped.startswith(("Default", "Transport", "Manufacturer", "Input", "Output", "Current")):
            current_section = stripped.rstrip(":")
            transport = ""

        if stripped.startswith("Transport:"):
            transport = stripped.split(":", 1)[1].strip().lower()

        if stripped == "Default Output Device: Yes":
            device_name = current_section
            break

    if not device_name:
        try:
            import sounddevice as sd
            dev = sd.query_devices(kind="output")
            if dev and isinstance(dev, dict):
                device_name = dev.get("name", "")
        except Exception:
            pass

    if not device_name:
        return fallback

    is_bt = transport == "bluetooth"
    if not is_bt:
        name_lower = device_name.lower()
        is_bt = any(kw in name_lower for kw in bt_keywords)

    return {"name": device_name, "is_bluetooth": is_bt}


def _get_output_device_info_linux(bt_keywords: tuple, fallback: dict) -> dict:
    """Linux: use pactl to detect output device and Bluetooth."""
    device_name = ""
    is_bt = False

    # Get default sink name
    try:
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith("Default Sink:"):
                device_name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    if not device_name:
        # Fallback to sounddevice
        try:
            import sounddevice as sd
            dev = sd.query_devices(kind="output")
            if dev and isinstance(dev, dict):
                device_name = dev.get("name", "")
        except Exception:
            pass

    if not device_name:
        return fallback

    # Check if sink is Bluetooth via pactl properties
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks"],
            capture_output=True, text=True, timeout=5,
        )
        in_target_sink = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:") and device_name in stripped:
                in_target_sink = True
            elif stripped.startswith("Name:"):
                in_target_sink = False
            if in_target_sink and "bluetooth" in stripped.lower():
                is_bt = True
                break
    except Exception:
        pass

    # Also check by device name keywords
    if not is_bt:
        name_lower = device_name.lower()
        is_bt = any(kw in name_lower for kw in bt_keywords)

    return {"name": device_name, "is_bluetooth": is_bt}


CURRENT_PLATFORM = detect_platform()
