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

    # Parse SPAudioDataType for the default output device name
    device_name = ""
    in_output_section = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Default Output Device:"):
            device_name = stripped.split(":", 1)[1].strip()
            break
        # Fallback: look for output section in detailed listing
        if "Output" in stripped and ":" in stripped and not device_name:
            in_output_section = True
        if in_output_section and stripped.startswith("Default Output Device:"):
            device_name = stripped.split(":", 1)[1].strip()
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

    # Check if the device name suggests Bluetooth
    name_lower = device_name.lower()
    is_bt = any(kw in name_lower for kw in _BT_KEYWORDS)

    # Also check if the device appears in the Bluetooth section of system_profiler output
    if not is_bt and "SPBluetoothDataType" in output:
        try:
            bt_section = output.split("Bluetooth:")[1] if "Bluetooth:" in output else ""
            if device_name in bt_section:
                is_bt = True
        except Exception:
            pass

    return {"name": device_name, "is_bluetooth": is_bt}


CURRENT_PLATFORM = detect_platform()
