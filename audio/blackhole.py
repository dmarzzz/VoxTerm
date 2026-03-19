"""BlackHole virtual audio device detection and setup guidance.

BlackHole is a macOS virtual audio loopback driver that allows applications
to capture system audio by routing it through a virtual device. When a user
has Bluetooth headphones (which block ScreenCaptureKit's audio tap), BlackHole
provides an alternative path: create a Multi-Output Device that sends audio
to both the headphones AND the BlackHole virtual device, then capture from
BlackHole.

This module provides detection (is it installed? is a Multi-Output Device
configured?) and user-facing setup instructions. It does NOT attempt to
programmatically create audio devices — that requires Audio MIDI Setup.

References:
    https://github.com/ExistentialAudio/BlackHole
    https://existential.audio/blackhole/
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from audio.platform import CURRENT_PLATFORM, Platform

# BlackHole installs its driver here
_DRIVER_DIRS = [
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"),
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver"),
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole64ch.driver"),
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole128ch.driver"),
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole256ch.driver"),
]

# Device names that BlackHole registers with CoreAudio
_BLACKHOLE_DEVICE_NAMES = (
    "BlackHole 2ch",
    "BlackHole 16ch",
    "BlackHole 64ch",
    "BlackHole 128ch",
    "BlackHole 256ch",
)


def is_blackhole_installed() -> bool:
    """Check whether any BlackHole driver variant is installed.

    Uses two strategies:
    1. Check for the HAL driver bundle on disk.
    2. Query sounddevice for a device whose name contains "BlackHole".

    Returns True if BlackHole is detected by either method.
    """
    if CURRENT_PLATFORM != Platform.MACOS:
        return False

    try:
        # Strategy 1: driver bundle on disk (fast, no subprocess)
        for driver_path in _DRIVER_DIRS:
            if driver_path.exists():
                return True

        # Strategy 2: query CoreAudio devices via sounddevice
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            if isinstance(devices, list):
                for dev in devices:
                    name = dev.get("name", "") if isinstance(dev, dict) else ""
                    if "blackhole" in name.lower():
                        return True
            elif isinstance(devices, dict):
                if "blackhole" in devices.get("name", "").lower():
                    return True
        except Exception:
            pass

        # Strategy 3: system_profiler as last resort
        try:
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if "BlackHole" in result.stdout:
                return True
        except Exception:
            pass

    except Exception:
        pass

    return False


def get_blackhole_device_name() -> str | None:
    """Return the name of the first detected BlackHole device, or None."""
    if CURRENT_PLATFORM != Platform.MACOS:
        return None

    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if isinstance(devices, list):
            for dev in devices:
                name = dev.get("name", "") if isinstance(dev, dict) else ""
                if "blackhole" in name.lower():
                    return name
        elif isinstance(devices, dict):
            name = devices.get("name", "")
            if "blackhole" in name.lower():
                return name
    except Exception:
        pass

    # Fallback: check driver bundles and infer name
    try:
        for driver_path, device_name in zip(_DRIVER_DIRS, _BLACKHOLE_DEVICE_NAMES):
            if driver_path.exists():
                return device_name
    except Exception:
        pass

    return None


def has_multi_output_device() -> bool:
    """Check if a Multi-Output Device exists in the current audio setup.

    Multi-Output Devices appear in system_profiler SPAudioDataType with the
    name "Multi-Output Device" (the macOS default name). We also check for
    any aggregate device that includes BlackHole.

    This is a best-effort heuristic — it cannot verify that the device is
    correctly configured with both BlackHole and the user's output device.
    """
    if CURRENT_PLATFORM != Platform.MACOS:
        return False

    try:
        # Check via sounddevice for any multi-output or aggregate device
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            if isinstance(devices, list):
                for dev in devices:
                    name = (dev.get("name", "") if isinstance(dev, dict) else "").lower()
                    if "multi-output" in name or "multi output" in name:
                        return True
        except Exception:
            pass

        # Check via system_profiler
        try:
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True, text=True, timeout=5,
            )
            output_lower = result.stdout.lower()
            if "multi-output" in output_lower or "multi output" in output_lower:
                return True
        except Exception:
            pass

    except Exception:
        pass

    return False


def get_setup_instructions(bt_device_name: str = "") -> str:
    """Return clear user-facing setup instructions for BlackHole.

    Args:
        bt_device_name: Name of the detected Bluetooth device (for context).

    Returns a multi-line string with setup steps.
    """
    device_context = ""
    if bt_device_name:
        device_context = (
            f"\nYour current output device ({bt_device_name}) uses Bluetooth,\n"
            "which prevents direct system audio capture. BlackHole solves this\n"
            "by routing audio through a virtual loopback device.\n"
        )

    has_bh = is_blackhole_installed()
    bh_name = get_blackhole_device_name()

    if not has_bh:
        install_section = (
            "STEP 1 — Install BlackHole\n"
            "  Option A (Homebrew):  brew install blackhole-2ch\n"
            "  Option B (download):  https://existential.audio/blackhole/\n"
            "\n"
        )
        step_offset = 2
    else:
        bh_label = f" ({bh_name})" if bh_name else ""
        install_section = (
            f"BlackHole is already installed{bh_label}.\n"
            "\n"
        )
        step_offset = 1

    multi_output_section = (
        f"STEP {step_offset} — Create a Multi-Output Device\n"
        "  1. Open Audio MIDI Setup (Cmd+Space, type 'Audio MIDI Setup')\n"
        "  2. Click the '+' button in the bottom-left corner\n"
        "  3. Select 'Create Multi-Output Device'\n"
        f"  4. Enable your output device ({bt_device_name or 'speakers/headphones'})\n"
        f"     — it MUST be the top (first) device in the list\n"
        f"  5. Enable '{bh_name or 'BlackHole 2ch'}' as the second device\n"
        "  6. Check 'Drift Correction' for BlackHole (NOT for the top device)\n"
        "\n"
        f"STEP {step_offset + 1} — Set the Multi-Output Device as system output\n"
        "  Right-click the Multi-Output Device in Audio MIDI Setup and\n"
        "  select 'Use This Device For Sound Output'.\n"
        "  (Or: System Settings > Sound > Output > Multi-Output Device)\n"
        "\n"
        "After this, VoxTerm's system audio capture will work because\n"
        "ScreenCaptureKit can tap the non-Bluetooth BlackHole leg of\n"
        "the Multi-Output Device."
    )

    return (
        "BlackHole Setup for System Audio Capture\n"
        "========================================\n"
        f"{device_context}\n"
        f"{install_section}"
        f"{multi_output_section}"
    )


def get_short_status(bt_device_name: str = "") -> str:
    """Return a one-line status string for the SystemCapture status message.

    This is designed to be shown in the TUI status bar.
    """
    has_bh = is_blackhole_installed()
    has_mod = has_multi_output_device()

    if has_bh and has_mod:
        return (
            f"Bluetooth output detected ({bt_device_name}). "
            "Multi-Output Device found — system audio capture should work"
        )
    elif has_bh and not has_mod:
        return (
            f"Bluetooth output detected ({bt_device_name}). "
            "BlackHole installed but no Multi-Output Device found — "
            "create one in Audio MIDI Setup for system audio capture"
        )
    else:
        return (
            f"system audio limited — Bluetooth output detected ({bt_device_name}). "
            "Install BlackHole (brew install blackhole-2ch) and create a "
            "Multi-Output Device for full system audio capture"
        )
