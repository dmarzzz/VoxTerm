"""Launch ShaderClaw's VoxTerm display from VoxTerm."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR, LIVE_DIR


DEFAULT_DISPLAY_PORT = 7778
SHADERCLAW_REPO_URL = "https://github.com/G3993/ShaderClaw3.git"


@dataclass(frozen=True)
class DisplayLaunchResult:
    ok: bool
    url: str
    message: str
    started: bool = False
    pid: int | None = None
    log_path: Path | None = None
    shaderclaw_dir: Path | None = None


def display_url(port: int = DEFAULT_DISPLAY_PORT) -> str:
    return f"http://localhost:{port}/voxterm"


def find_shaderclaw_dir(explicit: str | Path | None = None, repo_root: Path | None = None) -> Path | None:
    """Find a local ShaderClaw checkout that can serve VoxTerm display mode."""
    candidates: list[Path] = []

    def add(value: str | Path | None) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if path not in candidates:
            candidates.append(path)

    add(explicit)
    add(os.environ.get("VOXTERM_SHADERCLAW_DIR"))
    add(os.environ.get("SHADERCLAW_DIR"))

    root = repo_root or Path(__file__).resolve().parent
    home = Path.home()
    names = (
        "shader-claw3",
        "shaderclaw-3",
        "shaderclaw3",
        "shader-claw",
        "shaderclaw",
    )
    bases = (
        root.parent,
        home,
        home / "Code",
        home / "Developer",
        home / "Projects",
        home / "src",
    )
    for base in bases:
        for name in names:
            add(base / name)

    for candidate in candidates:
        if (candidate / "server.js").is_file() and (candidate / "package.json").is_file():
            return candidate
    return None


def _get_json(url: str, timeout: float = 0.4) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read(1024 * 128).decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def is_display_running(port: int = DEFAULT_DISPLAY_PORT) -> bool:
    status = _get_json(f"http://localhost:{port}/api/voxterm/status")
    return bool(status and status.get("ok") is True)


def open_display_browser(port: int = DEFAULT_DISPLAY_PORT) -> bool:
    url = display_url(port)
    if sys.platform == "darwin":
        command = ["open", url]
    elif sys.platform == "win32":
        command = ["cmd", "/c", "start", "", url]
    else:
        opener = shutil.which("xdg-open")
        if not opener:
            return False
        command = [opener, url]

    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def launch_display(
    *,
    port: int = DEFAULT_DISPLAY_PORT,
    shaderclaw_dir: str | Path | None = None,
    live_dir: str | Path = LIVE_DIR,
    open_browser: bool = True,
    wait_seconds: float = 6.0,
) -> DisplayLaunchResult:
    """Start or reveal ShaderClaw's VoxTerm display."""
    url = display_url(port)

    if is_display_running(port):
        if open_browser:
            open_display_browser(port)
        return DisplayLaunchResult(
            ok=True,
            url=url,
            message=f"display mode already running: {url}",
            started=False,
        )

    shaderclaw = find_shaderclaw_dir(shaderclaw_dir)
    if shaderclaw is None:
        return DisplayLaunchResult(
            ok=False,
            url=url,
            message=(
                "ShaderClaw checkout not found. Install it with "
                f"`git clone {SHADERCLAW_REPO_URL} ~/shader-claw3 && "
                "cd ~/shader-claw3 && npm install`, or set "
                "VOXTERM_SHADERCLAW_DIR to an existing shader-claw3 folder."
            ),
        )

    node = shutil.which("node")
    if not node:
        return DisplayLaunchResult(
            ok=False,
            url=url,
            message="node was not found on PATH; install Node.js to use display mode.",
            shaderclaw_dir=shaderclaw,
        )

    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "shaderclaw-display.log"

    env = os.environ.copy()
    env["SHADERCLAW_PORT"] = str(port)
    env["SHADERCLAW_VOXTERM_DISPLAY"] = "1"
    env["SHADERCLAW_NO_OPEN"] = "1"
    env["VOXTERM_LIVE_DIR"] = str(Path(live_dir).expanduser())

    log_file = open(log_path, "ab")
    try:
        process = subprocess.Popen(
            [node, "server.js", "--voxterm"],
            cwd=str(shaderclaw),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=(sys.platform != "win32"),
        )
    except OSError as exc:
        log_file.close()
        return DisplayLaunchResult(
            ok=False,
            url=url,
            message=f"could not start ShaderClaw display: {exc}",
            log_path=log_path,
            shaderclaw_dir=shaderclaw,
        )
    finally:
        if not log_file.closed:
            log_file.close()

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return DisplayLaunchResult(
                ok=False,
                url=url,
                message=f"ShaderClaw display exited early; see {log_path}",
                started=True,
                pid=process.pid,
                log_path=log_path,
                shaderclaw_dir=shaderclaw,
            )
        if is_display_running(port):
            if open_browser:
                open_display_browser(port)
            return DisplayLaunchResult(
                ok=True,
                url=url,
                message=f"display mode opened: {url}",
                started=True,
                pid=process.pid,
                log_path=log_path,
                shaderclaw_dir=shaderclaw,
            )
        time.sleep(0.15)

    if open_browser:
        open_display_browser(port)
    return DisplayLaunchResult(
        ok=True,
        url=url,
        message=f"display mode starting: {url}",
        started=True,
        pid=process.pid,
        log_path=log_path,
        shaderclaw_dir=shaderclaw,
    )
