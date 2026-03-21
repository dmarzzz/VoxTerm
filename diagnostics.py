"""Crash reporting, faulthandler setup, and signal handling.

Centralizes all diagnostic infrastructure so crash data is always
captured — even for C-level segfaults that bypass Python exception handling.
"""

from __future__ import annotations

import faulthandler
import gc
import json
import os
import resource
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

from config import CRASH_LOG_MAX_COUNT

# ── crash directory ───────────────────────────────────────────

CRASH_DIR = Path.home() / "Documents" / "voxterm" / ".crashes"


def _ensure_crash_dir() -> None:
    CRASH_DIR.mkdir(parents=True, exist_ok=True)


# ── faulthandler ──────────────────────────────────────────────

_fault_file = None


def setup_faulthandler() -> None:
    """Enable Python's faulthandler to write C-level tracebacks to a file.

    Must be called early — before the TUI takes over the terminal.
    Writes to CRASH_DIR/faulthandler.log (appended, not overwritten).
    """
    global _fault_file
    _ensure_crash_dir()
    _fault_file = open(CRASH_DIR / "faulthandler.log", "a")
    faulthandler.enable(file=_fault_file, all_threads=True)


# ── signal handlers ───────────────────────────────────────────

_saved_termios = None


def setup_signal_handlers() -> None:
    """Install a SIGSEGV handler that restores terminal settings before dying.

    Must be called after the terminal is configured but before app.run().
    """
    global _saved_termios

    try:
        import termios
        _saved_termios = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    signal.signal(signal.SIGSEGV, _segfault_handler)


def _segfault_handler(signum, frame):
    """Restore terminal, log crash marker, exit with SIGSEGV code."""
    # Restore terminal so the shell isn't stuck in raw mode
    if _saved_termios is not None:
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, _saved_termios)
        except Exception:
            pass

    # Write crash marker to faulthandler log
    # (faulthandler already printed the C traceback above this)
    try:
        ts = datetime.now().isoformat()
        with open(CRASH_DIR / "faulthandler.log", "a") as f:
            f.write(f"\n--- SIGSEGV caught at {ts} ---\n")
            f.write("Terminal restored. Check crash dumps for context.\n\n")
    except Exception:
        pass

    sys.stderr.write("\nVOXTERM: segfault caught — terminal restored\n")
    sys.stderr.write(f"Check {CRASH_DIR}/ for crash details\n")
    os._exit(139)


# ── exception hooks ───────────────────────────────────────────

_app_ref = None


def setup_exception_hooks(app) -> None:
    """Install sys.excepthook and threading.excepthook to capture crashes.

    Must be called after the App instance is created.
    """
    import threading

    global _app_ref
    _app_ref = app

    _orig_excepthook = sys.excepthook

    def _crash_excepthook(exc_type, exc_value, exc_tb):
        if _app_ref is not None:
            _write_app_crash_dump(_app_ref, f"uncaught:{exc_type.__name__}", exc_value)
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_excepthook

    _orig_thread_excepthook = getattr(threading, "excepthook", None)

    def _thread_crash_hook(args):
        if _app_ref is not None:
            thread_name = args.thread.name if args.thread else "unknown"
            _write_app_crash_dump(_app_ref, f"thread:{thread_name}", args.exc_value)
        if _orig_thread_excepthook:
            _orig_thread_excepthook(args)

    threading.excepthook = _thread_crash_hook


# ── crash dump writing ────────────────────────────────────────

def write_crash_dump(
    context: str,
    exc: BaseException | None = None,
    state: dict | None = None,
) -> None:
    """Write a crash dump to disk. Always runs, not gated by debug mode.

    Args:
        context: Where the crash occurred (e.g. "_transcribe_audio", "uncaught:ValueError")
        exc: The exception, if any
        state: Runtime state dict with keys like recording, model, rss_mb, etc.
    """
    try:
        _ensure_crash_dir()
        ts = datetime.now()
        base = ts.strftime("%Y-%m-%d_%H%M%S")
        state = state or {}

        # ── human-readable .log ───────────────────────
        lines = [
            "VOXTERM CRASH DUMP",
            "=" * 60,
            f"timestamp:        {ts.isoformat()}",
            f"uptime:           {state.get('uptime_sec', 0):.0f}s",
            f"context:          {context}",
            "",
            "-- error --",
            f"type:             {type(exc).__name__ if exc else 'N/A'}",
            f"message:          {exc}",
            "traceback:",
            traceback.format_exc() if exc else "  N/A",
            "",
            "-- runtime state --",
        ]
        for key in (
            "recording", "is_transcribing", "transcribe_count",
            "model", "model_loaded", "diarizer_loaded", "diarizer_mode",
            "language", "had_speech", "silence_chunks", "sys_capture",
        ):
            if key in state:
                lines.append(f"{key + ':':18s}{state[key]}")

        lines.append("")
        lines.append("-- memory --")
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_bytes / (1024 * 1024)
        lines.append(f"peak_rss_mb:      {rss_mb:.1f}")
        for key in (
            "audio_buf_dur", "style_cache", "transcript_entries",
            "speakers", "gc_counts",
        ):
            if key in state:
                lines.append(f"{key + ':':18s}{state[key]}")

        (CRASH_DIR / (base + ".log")).write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # ── machine-readable .json ────────────────────
        crash_json = {
            "timestamp": ts.isoformat(),
            "context": context,
            "error_type": type(exc).__name__ if exc else None,
            "error_message": str(exc) if exc else None,
            "peak_rss_mb": round(rss_mb, 1),
        }
        crash_json.update(state)
        # Remove non-serializable values
        for k, v in list(crash_json.items()):
            if not isinstance(v, (str, int, float, bool, type(None))):
                crash_json[k] = str(v)

        (CRASH_DIR / (base + ".json")).write_text(
            json.dumps(crash_json, indent=2), encoding="utf-8"
        )
    except Exception:
        pass  # crash dump must never itself crash the app


def _write_app_crash_dump(app, context: str, exc: BaseException | None = None):
    """Gather app state and write crash dump. Called from exception hooks."""
    try:
        from widgets.waveform import _make_style

        cache = _make_style.cache_info()

        try:
            from widgets.transcript import TranscriptPanel
            entry_count = len(app.query_one(TranscriptPanel).get_entries())
        except Exception:
            entry_count = -1

        state = {
            "uptime_sec": (datetime.now() - app._session_start).total_seconds(),
            "recording": app._recording,
            "is_transcribing": app._transcribing.is_set(),
            "transcribe_count": app._transcribe_count,
            "model": app._model_name,
            "model_loaded": app._model_loaded,
            "diarizer_loaded": app._diarizer_loaded,
            "language": app._language,
            "had_speech": app._had_speech,
            "silence_chunks": app._silence_chunks,
            "sys_capture": f"active={app.system_capture.is_active} msg={app.system_capture.status_message}",
            "audio_buf_dur": f"{app.audio_buffer.duration:.2f}s",
            "style_cache": f"hits={cache.hits} misses={cache.misses} size={cache.currsize}/{cache.maxsize}",
            "transcript_entries": entry_count,
            "speakers": app.diarizer.num_speakers if app._diarizer_loaded else 0,
            "gc_counts": str(gc.get_count()),
        }
        write_crash_dump(context, exc, state)
    except Exception:
        # Last resort — write minimal dump
        write_crash_dump(context, exc)


# ── log rotation ──────────────────────────────────────────────

def rotate_crash_logs() -> None:
    """Prune old crash logs, keeping the most recent CRASH_LOG_MAX_COUNT."""
    try:
        if not CRASH_DIR.exists():
            return
        logs = sorted(CRASH_DIR.glob("*.log"))
        jsons = sorted(CRASH_DIR.glob("*.json"))
        # Don't rotate faulthandler.log
        logs = [f for f in logs if f.name != "faulthandler.log"]
        for old in logs[:-CRASH_LOG_MAX_COUNT]:
            old.unlink()
        for old in jsons[:-CRASH_LOG_MAX_COUNT]:
            old.unlink()
        # Truncate faulthandler.log if > 1MB
        fh_log = CRASH_DIR / "faulthandler.log"
        if fh_log.exists() and fh_log.stat().st_size > 1_000_000:
            content = fh_log.read_text(encoding="utf-8", errors="replace")
            # Keep last 100KB
            fh_log.write_text(content[-100_000:], encoding="utf-8")
    except Exception:
        pass
