"""Keyboard text injection — types transcribed text into the focused application.

macOS: CoreGraphics CGEvent API via ctypes (no PyObjC dependency).
Linux/X11: xdotool subprocess.
Linux/Wayland: wtype subprocess, ydotool fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod

from audio.platform import CURRENT_PLATFORM, Platform


def _detect_display_server() -> str:
    """Return 'wayland', 'x11', or 'unknown'."""
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type in ("wayland", "x11"):
        return session_type
    return "unknown"


class KeyboardInjector(ABC):
    """Base class for platform-specific keyboard injection."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type text into the currently focused application."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the injection backend is functional."""


# ---------------------------------------------------------------------------
# macOS: CoreGraphics CGEvent API via ctypes
# ---------------------------------------------------------------------------

class _MacOSInjector(KeyboardInjector):
    """Injects keystrokes on macOS using CoreGraphics CGEvent API.

    Uses CGEventKeyboardSetUnicodeString to type Unicode text directly,
    bypassing keycode mapping.  Requires Accessibility permission.
    """

    _MAX_CHARS_PER_EVENT = 20  # CGEventKeyboardSetUnicodeString limit

    def __init__(self, inter_key_delay_ms: int = 1):
        self._delay = inter_key_delay_ms / 1000.0
        self._cg = None
        self._load_framework()

    def _load_framework(self) -> None:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("CoreGraphics")
        if path is None:
            raise RuntimeError("CoreGraphics framework not found")
        self._cg = ctypes.cdll.LoadLibrary(path)

        # CGEventCreateKeyboardEvent(source, virtualKey, keyDown) -> CGEventRef
        self._cg.CGEventCreateKeyboardEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool,
        ]
        self._cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p

        # CGEventKeyboardSetUnicodeString(event, length, unicodeString)
        self._cg.CGEventKeyboardSetUnicodeString.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p,
        ]
        self._cg.CGEventKeyboardSetUnicodeString.restype = None

        # CGEventPost(tap, event)
        self._cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        self._cg.CGEventPost.restype = None

        # CFRelease(event)
        cf_path = ctypes.util.find_library("CoreFoundation")
        self._cf = ctypes.cdll.LoadLibrary(cf_path)
        self._cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._cf.CFRelease.restype = None

        self._ctypes = ctypes

    def type_text(self, text: str) -> None:
        if not text or self._cg is None:
            return

        ctypes = self._ctypes
        kCGHIDEventTap = 0  # Post at HID level

        # Encode to UTF-16 and batch in chunks of _MAX_CHARS_PER_EVENT
        for i in range(0, len(text), self._MAX_CHARS_PER_EVENT):
            chunk = text[i:i + self._MAX_CHARS_PER_EVENT]
            utf16 = chunk.encode("utf-16-le")
            char_count = len(chunk)
            buf = ctypes.create_string_buffer(utf16)

            # Key down
            event_down = self._cg.CGEventCreateKeyboardEvent(None, 0, True)
            if event_down:
                self._cg.CGEventKeyboardSetUnicodeString(
                    event_down, char_count, buf,
                )
                self._cg.CGEventPost(kCGHIDEventTap, event_down)
                self._cf.CFRelease(event_down)

            # Key up
            event_up = self._cg.CGEventCreateKeyboardEvent(None, 0, False)
            if event_up:
                self._cg.CGEventKeyboardSetUnicodeString(
                    event_up, char_count, buf,
                )
                self._cg.CGEventPost(kCGHIDEventTap, event_up)
                self._cf.CFRelease(event_up)

            if self._delay > 0 and i + self._MAX_CHARS_PER_EVENT < len(text):
                time.sleep(self._delay)

    def is_available(self) -> bool:
        if self._cg is None:
            return False
        # Check Accessibility permission
        try:
            import ctypes.util
            path = self._ctypes.util.find_library("ApplicationServices")
            if path:
                app_svc = self._ctypes.cdll.LoadLibrary(path)
                app_svc.AXIsProcessTrusted.restype = self._ctypes.c_bool
                return bool(app_svc.AXIsProcessTrusted())
        except Exception:
            pass
        return True  # assume available if we can't check


# ---------------------------------------------------------------------------
# Linux/X11: xdotool
# ---------------------------------------------------------------------------

class _X11Injector(KeyboardInjector):
    """Injects keystrokes on Linux/X11 using xdotool."""

    def __init__(self):
        self._cmd = shutil.which("xdotool")

    def type_text(self, text: str) -> None:
        if not text or not self._cmd:
            return
        try:
            subprocess.run(
                [self._cmd, "type", "--clearmodifiers", "--delay", "0", "--", text],
                timeout=10,
                check=False,
            )
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._cmd is not None


# ---------------------------------------------------------------------------
# Linux/Wayland: wtype (primary), ydotool (fallback), clipboard paste (last resort)
# ---------------------------------------------------------------------------

class _WaylandInjector(KeyboardInjector):
    """Injects keystrokes on Linux/Wayland using wtype or ydotool."""

    def __init__(self):
        self._wtype = shutil.which("wtype")
        self._ydotool = shutil.which("ydotool")
        self._wl_copy = shutil.which("wl-copy")

    def type_text(self, text: str) -> None:
        if not text:
            return

        # Try wtype first (native Wayland)
        if self._wtype:
            try:
                subprocess.run(
                    [self._wtype, "--", text],
                    timeout=10,
                    check=False,
                )
                return
            except Exception:
                pass

        # Fallback: ydotool (requires ydotoold daemon)
        if self._ydotool:
            try:
                subprocess.run(
                    [self._ydotool, "type", "--", text],
                    timeout=10,
                    check=False,
                )
                return
            except Exception:
                pass

        # Last resort: clipboard paste via wl-copy + wtype Ctrl+V
        if self._wl_copy and self._wtype:
            try:
                subprocess.run(
                    [self._wl_copy, "--", text],
                    timeout=5,
                    check=False,
                )
                subprocess.run(
                    [self._wtype, "-M", "ctrl", "-k", "v"],
                    timeout=5,
                    check=False,
                )
            except Exception:
                pass

    def is_available(self) -> bool:
        return self._wtype is not None or self._ydotool is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Windows: Win32 SendInput via ctypes
# ---------------------------------------------------------------------------

class _WindowsInjector(KeyboardInjector):
    """Injects keystrokes on Windows using Win32 SendInput (ctypes).

    Uses KEYEVENTF_UNICODE to inject arbitrary Unicode text directly,
    bypassing keyboard layout — works for any language/script.
    No external tools or permissions required on Windows 10/11.
    """

    _INTER_CHAR_DELAY = 0.0  # seconds between chars; increase if apps miss chars

    def __init__(self):
        import ctypes
        import ctypes.wintypes as _wt

        self._ctypes = ctypes
        self._wt = _wt
        self._user32 = ctypes.windll.user32

        # Win32 constants
        self._INPUT_KEYBOARD = 1
        self._KEYEVENTF_UNICODE = 0x0004
        self._KEYEVENTF_KEYUP = 0x0002

        # Define KEYBDINPUT structure
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk",         _wt.WORD),
                ("wScan",       _wt.WORD),
                ("dwFlags",     _wt.DWORD),
                ("time",        _wt.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        # Define INPUT union/structure
        class _INPUT_UNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", _wt.DWORD),
                ("_input", _INPUT_UNION),
            ]

        self._KEYBDINPUT = KEYBDINPUT
        self._INPUT = INPUT
        self._INPUT_UNION = _INPUT_UNION

    def type_text(self, text: str) -> None:
        if not text:
            return

        ctypes = self._ctypes
        INPUT = self._INPUT
        KEYBDINPUT = self._KEYBDINPUT
        _INPUT_UNION = self._INPUT_UNION

        inputs = []
        for char in text:
            scan = ord(char)
            # Key down
            ki_down = KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=self._KEYEVENTF_UNICODE,
                time=0,
                dwExtraInfo=None,
            )
            inp_down = INPUT(type=self._INPUT_KEYBOARD, _input=_INPUT_UNION(ki=ki_down))
            inputs.append(inp_down)
            # Key up
            ki_up = KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=self._KEYEVENTF_UNICODE | self._KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=None,
            )
            inp_up = INPUT(type=self._INPUT_KEYBOARD, _input=_INPUT_UNION(ki=ki_up))
            inputs.append(inp_up)

        n = len(inputs)
        arr = (INPUT * n)(*inputs)
        self._user32.SendInput(n, arr, ctypes.sizeof(INPUT))

    def is_available(self) -> bool:
        # SendInput is available on all Windows versions we care about
        return True


def get_injector() -> KeyboardInjector:
    """Return the appropriate KeyboardInjector for the current platform."""
    if CURRENT_PLATFORM == Platform.MACOS:
        return _MacOSInjector()

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return _WindowsInjector()

    if CURRENT_PLATFORM == Platform.LINUX:
        ds = _detect_display_server()
        if ds == "wayland":
            inj = _WaylandInjector()
            if inj.is_available():
                return inj
            # Wayland session but no wtype/ydotool — try X11 tools (XWayland)
        inj = _X11Injector()
        if inj.is_available():
            return inj
        # Try Wayland tools as last resort even if display server detected as X11
        inj = _WaylandInjector()
        if inj.is_available():
            return inj

    raise RuntimeError(
        "No keyboard injection tool found.\n"
        "macOS: grant Accessibility permission in System Settings.\n"
        "Linux/X11: install xdotool (apt install xdotool / nix develop).\n"
        "Linux/Wayland: install wtype or ydotool."
    )
