"""Action executor — runs parsed voice commands.

Each action is a function that takes params and returns a result string.
Actions that touch external systems (media, apps, files) use subprocess
with restricted capabilities.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# Registry of action handlers
_ACTIONS: dict[str, Callable] = {}


def _register(name: str):
    """Decorator to register an action handler."""
    def decorator(fn):
        _ACTIONS[name] = fn
        return fn
    return decorator


def execute(action: str, params: dict, context: dict | None = None) -> str:
    """Execute a parsed command.

    Args:
        action: Action name from the interpreter.
        params: Action parameters.
        context: Optional context (transcript text, session info, etc.)

    Returns:
        Human-readable result string.
    """
    handler = _ACTIONS.get(action)
    if handler is None:
        return f"Unknown action: {action}"
    try:
        return handler(params, context or {})
    except Exception as e:
        log.error("action %s failed: %s", action, e)
        return f"Failed: {e}"


@_register("play_music")
def _play_music(params: dict, _ctx: dict) -> str:
    query = params.get("query", "")
    playerctl = shutil.which("playerctl")
    if playerctl:
        if query:
            # Try to open in default music player via xdg-open
            subprocess.Popen(
                ["xdg-open", f"https://music.youtube.com/search?q={query}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Searching for: {query}"
        else:
            subprocess.run([playerctl, "play"], capture_output=True, timeout=5)
            return "Resumed playback."
    return "No media player found (install playerctl)."


@_register("pause_music")
def _pause_music(_params: dict, _ctx: dict) -> str:
    playerctl = shutil.which("playerctl")
    if playerctl:
        subprocess.run([playerctl, "pause"], capture_output=True, timeout=5)
        return "Paused."
    return "No media player found."


@_register("save_transcript")
def _save_transcript(params: dict, ctx: dict) -> str:
    transcript_md = ctx.get("transcript_markdown", "")
    if not transcript_md:
        return "No transcript to save."

    from paths import SESSIONS_DIR
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    filename = params.get("filename", "")
    if filename:
        # Sanitize
        filename = "".join(c for c in filename if c.isalnum() or c in "-_ ").strip()
        filename = filename.replace(" ", "-")
    if not filename:
        filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = SESSIONS_DIR / f"{filename}.md"

    filepath.write_text(transcript_md, encoding="utf-8")
    return f"Saved to {filepath}"


@_register("summarize")
def _summarize(params: dict, ctx: dict) -> str:
    # Delegate back to the summarizer — the TUI handles this
    return "__delegate_summarize__"


@_register("open_app")
def _open_app(params: dict, _ctx: dict) -> str:
    app = params.get("app", "")
    if not app:
        return "No app specified."
    binary = shutil.which(app) or shutil.which(app.lower())
    if binary:
        subprocess.Popen(
            [binary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Opened {app}."
    # Try xdg-open as fallback
    subprocess.Popen(
        ["xdg-open", app],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"Opening {app}."


@_register("web_search")
def _web_search(params: dict, _ctx: dict) -> str:
    query = params.get("query", "")
    if not query:
        return "No search query."
    subprocess.Popen(
        ["xdg-open", f"https://duckduckgo.com/?q={query}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"Searching: {query}"


@_register("system")
def _system_command(params: dict, _ctx: dict) -> str:
    cmd = params.get("command", "")
    commands = {
        "volume_up": ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+"],
        "volume_down": ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"],
        "mute": ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"],
        "brightness_up": ["brightnessctl", "set", "+10%"],
        "brightness_down": ["brightnessctl", "set", "10%-"],
        "screenshot": ["grim"],
        "lock": ["loginctl", "lock-session"],
    }
    argv = commands.get(cmd)
    if argv is None:
        return f"Unknown system command: {cmd}"
    binary = shutil.which(argv[0])
    if binary is None:
        return f"{argv[0]} not found."
    subprocess.run(argv, capture_output=True, timeout=5)
    return f"Done: {cmd}"


@_register("set_timer")
def _set_timer(params: dict, _ctx: dict) -> str:
    duration = params.get("duration", "5m")
    label = params.get("label", "timer")
    # Use notify-send after sleep via a background process
    notify = shutil.which("notify-send")
    if not notify:
        return "notify-send not found."
    # Parse simple durations
    seconds = 0
    dur = duration.strip().lower()
    if dur.endswith("h"):
        seconds = int(float(dur[:-1]) * 3600)
    elif dur.endswith("m"):
        seconds = int(float(dur[:-1]) * 60)
    elif dur.endswith("s"):
        seconds = int(float(dur[:-1]))
    else:
        try:
            seconds = int(dur)
        except ValueError:
            return f"Cannot parse duration: {duration}"
    subprocess.Popen(
        ["bash", "-c", f'sleep {seconds} && notify-send "Timer" "{label}"'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"Timer set: {label} in {duration}."


@_register("dictate")
def _dictate(_params: dict, _ctx: dict) -> str:
    return "__delegate_dictate__"


@_register("unknown")
def _unknown(params: dict, _ctx: dict) -> str:
    original = params.get("original", "")
    return f"Didn't understand: {original}" if original else "Didn't understand that."
