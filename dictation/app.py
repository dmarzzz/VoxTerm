"""VoxTerm Dictation Mode — system-wide voice-to-text.

Speak into any app: global hotkey activates mic, transcribed text is typed
into the focused application via keyboard injection.

Usage:
    python3 -m dictation.app                   # default model + language
    python3 -m dictation.app -m qwen3-1.7b     # larger model
    python3 -m dictation.app -l ja             # Japanese
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
import threading

# Only needed when running as a script, not when installed as a package
if __package__ is None:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from audio.platform import CURRENT_PLATFORM, Platform
from config import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_MODELS,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    FASTER_WHISPER_MODELS,
    QWEN3_MODELS,
    SPECTROGRAM_MODELS,
    LLAMA_SERVER_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dictation")


def _check_accessibility_macos() -> bool:
    """Check and prompt for Accessibility permission on macOS."""
    import ctypes
    import ctypes.util
    import subprocess

    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        return True  # can't check, assume OK
    app_svc = ctypes.cdll.LoadLibrary(path)
    app_svc.AXIsProcessTrusted.restype = ctypes.c_bool
    if app_svc.AXIsProcessTrusted():
        return True

    print(
        "Accessibility permission required for keyboard injection.\n"
        "Opening System Settings...\n"
        "Grant permission to your terminal app, then restart dictation.",
        file=sys.stderr,
    )
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
        check=False,
    )
    return False


def _check_linux_tools() -> bool:
    """Check that keyboard injection tools are available on Linux."""
    import shutil
    tools = ["xdotool", "wtype", "ydotool"]
    if any(shutil.which(t) for t in tools):
        return True
    print(
        "No keyboard injection tool found.\n"
        "Install one of: xdotool (X11), wtype (Wayland), ydotool (both)\n"
        "  apt: sudo apt install xdotool\n"
        "  nix: nix develop  (tools included in flake)",
        file=sys.stderr,
    )
    return False


def _load_transcriber(model_name: str, model_repo: str, language: str):
    """Load the transcription model (same logic as app.py __main__)."""
    from audio.transcriber import (
        FasterWhisperTranscriber,
        Qwen3Transcriber,
        SpectrogramTranscriber,
        WhisperTranscriber,
    )

    print(f"VOXTERM DICTATION // loading model ({model_name}) lang={language}...")
    print("(first run downloads the model, please wait)\n")

    if model_name in SPECTROGRAM_MODELS:
        server_url = LLAMA_SERVER_URL or "http://localhost:8080"
        transcriber = SpectrogramTranscriber(
            server_url=server_url, model=model_repo, language=language,
        )
    elif model_name in QWEN3_MODELS:
        transcriber = Qwen3Transcriber(model=model_repo, language=language)
    elif model_name in FASTER_WHISPER_MODELS:
        transcriber = FasterWhisperTranscriber(model=model_repo, language=language)
    else:
        transcriber = WhisperTranscriber(model=model_repo)

    transcriber.load()
    print("Model ready.\n")
    return transcriber


def _write_pid_file() -> None:
    """Write PID file for signal-based hotkey on Wayland."""
    pid_file = "/tmp/voxterm-dictation.pid"
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        atexit.register(lambda: os.unlink(pid_file) if os.path.exists(pid_file) else None)
    except OSError:
        pass


def main() -> None:
    # ---- Resolve saved preferences ----
    try:
        from app import _get_config
        _cfg = _get_config()
        _saved_model = _cfg.get("last_model")
        _saved_lang = _cfg.get("last_language")
    except Exception:
        _saved_model = None
        _saved_lang = None

    _default_model = _saved_model if _saved_model and _saved_model in AVAILABLE_MODELS else DEFAULT_MODEL
    _default_lang = _saved_lang if _saved_lang and _saved_lang in AVAILABLE_LANGUAGES else DEFAULT_LANGUAGE

    # ---- Parse args ----
    parser = argparse.ArgumentParser(description="VOXTERM DICTATION — System-wide voice-to-text")
    parser.add_argument(
        "-m", "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default=_default_model,
        help=f"Transcription model (default: {_default_model})",
    )
    parser.add_argument(
        "-l", "--language",
        choices=list(AVAILABLE_LANGUAGES.keys()),
        default=_default_lang,
        help=f"Transcription language (default: {_default_lang})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, repo in AVAILABLE_MODELS.items():
            tag = " (default)" if name == _default_model else ""
            print(f"  {name:20s} -> {repo}{tag}")
        sys.exit(0)

    # ---- Platform checks ----
    if CURRENT_PLATFORM == Platform.MACOS:
        if not _check_accessibility_macos():
            sys.exit(1)
    elif CURRENT_PLATFORM == Platform.LINUX:
        if not _check_linux_tools():
            sys.exit(1)
    else:
        print(f"Unsupported platform: {CURRENT_PLATFORM}", file=sys.stderr)
        sys.exit(1)

    # ---- Load model ----
    model_repo = AVAILABLE_MODELS[args.model]
    transcriber = _load_transcriber(args.model, model_repo, args.language)

    # ---- Create components ----
    from dictation.injector import get_injector
    from dictation.indicator import get_indicator
    from dictation.hotkey import get_hotkey
    from dictation.loop import DictationLoop

    injector = get_injector()
    log.info("keyboard injector: %s", type(injector).__name__)

    indicator = get_indicator(
        model_name=args.model,
        language=args.language,
    )

    loop = DictationLoop(
        transcriber=transcriber,
        injector=injector,
        on_state_change=indicator.set_state,
    )

    def toggle_dictation():
        if loop.is_active:
            loop.stop()
        else:
            loop.start()

    hotkey = get_hotkey(toggle_dictation)

    # Wire quit
    def quit_all():
        loop.stop()
        hotkey.stop()
        indicator.stop()

    indicator._on_quit = quit_all

    # ---- Write PID file (for Wayland signal-based hotkey) ----
    _write_pid_file()

    # ---- Prevent C extension segfaults on shutdown ----
    atexit.register(os._exit, 0)

    # ---- Start ----
    hotkey.start()
    if CURRENT_PLATFORM == Platform.MACOS:
        log.info("hotkey: Cmd+Shift+D")
    else:
        log.info("hotkey: Super+Shift+D (X11) or SIGUSR1 (Wayland)")

    indicator.set_state("idle")
    print("VOXTERM DICTATION ready. Press hotkey to start/stop dictation.")

    try:
        indicator.run()  # blocks main thread
    except KeyboardInterrupt:
        pass
    finally:
        quit_all()


if __name__ == "__main__":
    main()
