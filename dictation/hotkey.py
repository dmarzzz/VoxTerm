"""Global hotkey listener — activates dictation from any application.

macOS: Carbon RegisterEventHotKey via ctypes.
Linux/X11: python-xlib XGrabKey.
Linux/Wayland: SIGUSR1 signal (user configures compositor keybind).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from abc import ABC, abstractmethod
from typing import Callable

from audio.platform import CURRENT_PLATFORM, Platform


class GlobalHotkey(ABC):
    """Base class for platform-specific global hotkey registration."""

    def __init__(self, callback: Callable[[], None]):
        self._callback = callback

    @abstractmethod
    def start(self) -> None:
        """Begin listening for the hotkey."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening."""


# ---------------------------------------------------------------------------
# macOS: Quartz CGEvent tap via ctypes
# ---------------------------------------------------------------------------

class _MacOSHotkey(GlobalHotkey):
    """Global hotkey on macOS using a Quartz event tap (background thread).

    Listens for Cmd+Shift+D by installing a CGEvent tap that monitors
    keyDown events.  Runs in a daemon thread with its own CFRunLoop.
    """

    # Virtual keycode for 'D' on macOS
    _KEY_D = 2
    # Modifier masks (CGEventFlags)
    _kCGEventFlagMaskCommand = 0x00100000
    _kCGEventFlagMaskShift = 0x00020000
    _REQUIRED_FLAGS = _kCGEventFlagMaskCommand | _kCGEventFlagMaskShift

    _DEBOUNCE_SEC = 0.4  # ignore repeated key events within this window

    def __init__(self, callback: Callable[[], None]):
        super().__init__(callback)
        self._thread: threading.Thread | None = None
        self._running = False
        self._run_loop_ref = None
        self._last_fire: float = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._run_loop_ref is not None:
            import ctypes
            import ctypes.util
            cf_path = ctypes.util.find_library("CoreFoundation")
            cf = ctypes.cdll.LoadLibrary(cf_path)
            cf.CFRunLoopStop(self._run_loop_ref)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        import ctypes
        import ctypes.util

        # Load frameworks
        cg_path = ctypes.util.find_library("CoreGraphics")
        cf_path = ctypes.util.find_library("CoreFoundation")
        cg = ctypes.cdll.LoadLibrary(cg_path)
        cf = ctypes.cdll.LoadLibrary(cf_path)

        # CGEventTapCreate callback type
        CGEventTapCallBack = ctypes.CFUNCTYPE(
            ctypes.c_void_p,  # return: CGEventRef (pass-through or NULL)
            ctypes.c_void_p,  # proxy
            ctypes.c_uint32,  # type (CGEventType)
            ctypes.c_void_p,  # event (CGEventRef)
            ctypes.c_void_p,  # userInfo
        )

        kCGEventKeyDown = 10
        kCGSessionEventTap = 1  # session-level tap
        kCGHeadInsertEventTap = 0
        kCGEventTapOptionListenOnly = 1

        callback_ref = self  # prevent GC

        @CGEventTapCallBack
        def _tap_callback(proxy, event_type, event, user_info):
            if event_type == kCGEventKeyDown:
                # Get keycode
                # kCGKeyboardEventKeycode = 9
                keycode = cg.CGEventGetIntegerValueField(event, 9)
                flags = cg.CGEventGetFlags(event)

                if (keycode == self._KEY_D and
                        (flags & self._REQUIRED_FLAGS) == self._REQUIRED_FLAGS):
                    import time as _time
                    now = _time.monotonic()
                    if now - callback_ref._last_fire >= callback_ref._DEBOUNCE_SEC:
                        callback_ref._last_fire = now
                        try:
                            callback_ref._callback()
                        except Exception:
                            pass
            return event

        # CGEventGetIntegerValueField
        cg.CGEventGetIntegerValueField.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cg.CGEventGetIntegerValueField.restype = ctypes.c_int64

        # CGEventGetFlags
        cg.CGEventGetFlags.argtypes = [ctypes.c_void_p]
        cg.CGEventGetFlags.restype = ctypes.c_uint64

        # CGEventTapCreate
        cg.CGEventTapCreate.argtypes = [
            ctypes.c_uint32,  # tap
            ctypes.c_uint32,  # place
            ctypes.c_uint32,  # options
            ctypes.c_uint64,  # eventsOfInterest
            CGEventTapCallBack,  # callback
            ctypes.c_void_p,  # userInfo
        ]
        cg.CGEventTapCreate.restype = ctypes.c_void_p

        # Create event tap for keyDown events
        event_mask = 1 << kCGEventKeyDown
        tap = cg.CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionListenOnly,
            event_mask,
            _tap_callback,
            None,
        )
        if not tap:
            print(
                "Failed to create event tap. "
                "Grant Accessibility permission in System Settings > Privacy > Accessibility.",
                file=sys.stderr,
            )
            return

        # Create a CFRunLoopSource from the tap and add it to a run loop
        cf.CFMachPortCreateRunLoopSource.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
        ]
        cf.CFMachPortCreateRunLoopSource.restype = ctypes.c_void_p

        source = cf.CFMachPortCreateRunLoopSource(None, tap, 0)

        cf.CFRunLoopGetCurrent.restype = ctypes.c_void_p
        run_loop = cf.CFRunLoopGetCurrent()
        self._run_loop_ref = run_loop

        # kCFRunLoopCommonModes
        cf.CFRunLoopAddSource.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        # Get kCFRunLoopCommonModes string constant
        cf.kCFRunLoopCommonModes = ctypes.c_void_p.in_dll(cf, "kCFRunLoopCommonModes")
        cf.CFRunLoopAddSource(run_loop, source, cf.kCFRunLoopCommonModes)

        # Enable the tap
        cg.CGEventTapEnable.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        cg.CGEventTapEnable(tap, True)

        # Run the loop (blocks until CFRunLoopStop)
        cf.CFRunLoopRun()

        # Cleanup
        self._run_loop_ref = None


# ---------------------------------------------------------------------------
# Linux/X11: python-xlib XGrabKey
# ---------------------------------------------------------------------------

class _X11Hotkey(GlobalHotkey):
    """Global hotkey on X11 using python-xlib XGrabKey.

    Grabs Super+Shift+D globally.  Runs in a daemon thread.
    """

    _DEBOUNCE_SEC = 0.4

    def __init__(self, callback: Callable[[], None]):
        super().__init__(callback)
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_fire: float = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        try:
            from Xlib import X, XK, display as xdisplay
            from Xlib.ext import record
        except ImportError:
            print(
                "python-xlib not installed. Install it: pip install python-xlib",
                file=sys.stderr,
            )
            return

        disp = xdisplay.Display()
        root = disp.screen().root

        # Get keysym for 'D' and convert to keycode
        keysym = XK.string_to_keysym("d")
        keycode = disp.keysym_to_keycode(keysym)

        # Super (Mod4) + Shift
        modifiers = X.Mod4Mask | X.ShiftMask

        # Grab the key — also grab with NumLock and CapsLock variations
        for extra_mod in (0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask):
            root.grab_key(
                keycode, modifiers | extra_mod,
                True, X.GrabModeAsync, X.GrabModeAsync,
            )
        disp.flush()

        try:
            while self._running:
                # Check for events with a short timeout
                if disp.pending_events():
                    event = disp.next_event()
                    if event.type == X.KeyPress and event.detail == keycode:
                        import time as _time
                        now = _time.monotonic()
                        if now - self._last_fire < self._DEBOUNCE_SEC:
                            continue
                        self._last_fire = now
                        try:
                            self._callback()
                        except Exception:
                            pass
                else:
                    import time
                    time.sleep(0.05)
        finally:
            for extra_mod in (0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask):
                root.ungrab_key(keycode, modifiers | extra_mod)
            disp.close()


# ---------------------------------------------------------------------------
# Linux/Wayland: SIGUSR1 signal handler
# ---------------------------------------------------------------------------

class _SignalHotkey(GlobalHotkey):
    """Hotkey via SIGUSR1 signal — user configures compositor to send signal.

    Writes PID to /tmp/voxterm-dictation.pid so compositor keybinds can
    target this process:
        kill -USR1 $(cat /tmp/voxterm-dictation.pid)
    """

    _PID_FILE = "/tmp/voxterm-dictation.pid"

    def __init__(self, callback: Callable[[], None]):
        super().__init__(callback)
        self._prev_handler = None

    def start(self) -> None:
        # Write PID file
        try:
            with open(self._PID_FILE, "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            pass

        self._prev_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handle_signal)

        print(
            f"Wayland: no global hotkey protocol. "
            f"Configure your compositor to send SIGUSR1:\n"
            f"  kill -USR1 $(cat {self._PID_FILE})\n"
            f"Example sway config:\n"
            f"  bindsym $mod+Shift+d exec kill -USR1 $(cat {self._PID_FILE})",
            file=sys.stderr,
        )

    def stop(self) -> None:
        if self._prev_handler is not None:
            signal.signal(signal.SIGUSR1, self._prev_handler)
        try:
            os.unlink(self._PID_FILE)
        except OSError:
            pass

    def _handle_signal(self, signum: int, frame) -> None:
        try:
            self._callback()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Windows: Win32 RegisterHotKey via ctypes
# ---------------------------------------------------------------------------

class _WindowsHotkey(GlobalHotkey):
    """Global hotkey on Windows using Win32 RegisterHotKey.

    Registers Ctrl+Shift+D system-wide. Runs a Win32 message pump in a
    daemon thread — no external libraries required.

    Hotkey: Ctrl+Shift+D  (avoids conflicts with common Win shortcuts)
    """

    _MOD_CONTROL = 0x0002
    _MOD_SHIFT   = 0x0004
    _MOD_NOREPEAT = 0x4000
    _HOTKEY_ID = 0xBEEF  # arbitrary unique ID
    _VK_D = 0x44         # Virtual key code for 'D'
    _WM_HOTKEY = 0x0312

    _DEBOUNCE_SEC = 0.4

    def __init__(self, callback: Callable[[], None]):
        super().__init__(callback)
        self._thread: threading.Thread | None = None
        self._running = False
        self._hwnd = None
        self._last_fire: float = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="win-hotkey")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Post a quit message to unblock the message loop
        try:
            import ctypes
            ctypes.windll.user32.PostQuitMessage(0)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        import ctypes
        import ctypes.wintypes as wt
        import time

        user32 = ctypes.windll.user32

        # Register the hotkey on this thread (Win32 hotkeys are thread-local)
        modifiers = self._MOD_CONTROL | self._MOD_SHIFT | self._MOD_NOREPEAT
        if not user32.RegisterHotKey(None, self._HOTKEY_ID, modifiers, self._VK_D):
            # Already registered or permission issue — log and bail
            import sys
            print(
                "VoxTerm: could not register hotkey Ctrl+Shift+D "
                "(may already be in use). Dictation hotkey disabled.",
                file=sys.stderr,
            )
            return

        try:
            msg = wt.MSG()
            while self._running:
                # PeekMessage with timeout so we can check _running
                result = user32.PeekMessageW(
                    ctypes.byref(msg), None, 0, 0, 1  # PM_REMOVE = 1
                )
                if result:
                    if msg.message == self._WM_HOTKEY and msg.wParam == self._HOTKEY_ID:
                        now = time.monotonic()
                        if now - self._last_fire >= self._DEBOUNCE_SEC:
                            self._last_fire = now
                            try:
                                self._callback()
                            except Exception:
                                pass
                    elif msg.message == 0x0012:  # WM_QUIT
                        break
                else:
                    time.sleep(0.02)
        finally:
            user32.UnregisterHotKey(None, self._HOTKEY_ID)


def get_hotkey(callback: Callable[[], None]) -> GlobalHotkey:
    """Return the appropriate GlobalHotkey for the current platform."""
    if CURRENT_PLATFORM == Platform.MACOS:
        return _MacOSHotkey(callback)

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return _WindowsHotkey(callback)

    if CURRENT_PLATFORM == Platform.LINUX:
        from dictation.injector import _detect_display_server
        ds = _detect_display_server()
        if ds == "x11":
            return _X11Hotkey(callback)
        # Wayland or unknown — use signal-based hotkey
        return _SignalHotkey(callback)

    raise RuntimeError(f"Unsupported platform for global hotkey: {CURRENT_PLATFORM}")
