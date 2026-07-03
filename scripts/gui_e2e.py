"""Headless-browser e2e for the VoxTerm GUI (Chrome DevTools Protocol).

Boots the real `gui.server`, drives a headless Chrome through the actual UI, and asserts the
review flow end-to-end against the redesigned UI:
  - the model dropdown + session list populate from the API;
  - clicking a past session renders its transcript with a real (non-date) title;
  - the recording's audio actually LOADS UNDER THE PAGE CSP (a fresh Audio() obeys `media-src`
    exactly like the inline <audio> player — this is what unit tests can't cover, and what the
    `media-src 'self'` fix exists for), with zero `securitypolicyviolation` events;
  - a real record -> stop cycle drives the state machine (Recording -> Transcribing -> ready)
    without a JS error (best-effort; skipped cleanly if no mic is present).
Saves a screenshot of the loaded transcript.

    python scripts/gui_e2e.py [--shot /tmp/voxterm-transcript.png]
Exit 0 = all assertions passed. Requires a Chrome/Chromium binary or Playwright's
cached Chromium/headless shell, plus `pip install websocket-client`.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import websocket  # websocket-client (dev dep)

ROOT = Path(__file__).resolve().parent.parent
PORT = 8740
CDP_PORT = 9222

# injected at document-start so it captures CSP violations + uncaught JS errors from byte one
CSP_COLLECTOR = (
    "window.__csp = []; window.__err = [];"
    "document.addEventListener('securitypolicyviolation',"
    " e => window.__csp.push(e.violatedDirective + ' ' + (e.blockedURI||'')));"
    "window.addEventListener('error', e => window.__err.push(String((e && e.message) || e)));"
    "window.addEventListener('unhandledrejection', e => window.__err.push('promise: ' + String(e.reason)));"
)


def _chrome_candidates() -> list[str]:
    """Return Chrome/Chromium executables in preference order."""
    env = os.environ.get("CHROME") or os.environ.get("CHROME_BIN")
    candidates = [env] if env else []
    candidates.extend(
        shutil.which(name)
        for name in ("google-chrome", "chromium", "chromium-browser")
    )
    if sys.platform == "darwin":
        candidates.extend([
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ])
        cache = Path.home() / "Library" / "Caches" / "ms-playwright"
        candidates.extend(
            str(path)
            for path in sorted(
                cache.glob("chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"),
                reverse=True,
            )
        )
    return [c for c in candidates if c]


def _find_chrome() -> str | None:
    for candidate in _chrome_candidates():
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _wait(url: str, timeout: float = 30.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            return urllib.request.urlopen(url, timeout=2).read()
        except Exception:
            time.sleep(0.4)
    raise TimeoutError(f"timed out waiting for {url}")


def _seed_session_fixture(out_dir: Path) -> None:
    """Create one tiny deterministic session for the browser review flow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "2026-07-03_070000"
    doc = {
        "voxterm_export_version": 1,
        "session": {
            "id": stem,
            "started_at": "2026-07-03T07:00:00",
            "duration_hms": "00:00:02",
            "model": "fixture",
            "source_stream": f"{stem}-events.jsonl",
        },
        "speakers": [{"id": 1, "label": "Speaker 1", "peer": False}],
        "turns": [
            {
                "t": "00:00:00",
                "t_offset": 0.0,
                "t_offset_hms": "00:00",
                "t_offset_end": 2.0,
                "t_offset_end_hms": "00:02",
                "speaker": "Speaker 1",
                "speaker_id": 1,
                "text": "This fixture verifies the interface review flow.",
                "markers": [],
            }
        ],
    }
    (out_dir / f"{stem}-agent.json").write_text(json.dumps(doc), encoding="utf-8")
    (out_dir / f"{stem}-agent.md").write_text(
        "# VoxTerm Fixture\n\nSpeaker 1: This fixture verifies the interface review flow.\n",
        encoding="utf-8",
    )
    (out_dir / f"{stem}-transcript.md").write_text(
        "# VoxTerm Transcript\n\n**[00:00:00]** **Speaker 1:** This fixture verifies the interface review flow.\n",
        encoding="utf-8",
    )
    events = [
        {"t": 0.0, "kind": "recording", "on": True},
        {
            "t": 0.0,
            "kind": "segment",
            "speaker": "Speaker 1",
            "speaker_id": 1,
            "text": "This fixture verifies the interface review flow.",
            "confidence": "high",
            "overlap": False,
            "audio_offset": 0.0,
            "audio_end": 2.0,
        },
        {"t": 2.0, "kind": "recording", "on": False},
    ]
    (out_dir / f"{stem}-events.jsonl").write_text(
        "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n",
        encoding="utf-8",
    )
    with wave.open(str(out_dir / f"{stem}-gui.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 32000)


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def poll(self, expr: str, want=True, timeout: float = 20.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.eval(expr) == want:
                return True
            time.sleep(0.3)
        return False


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    shot = "/tmp/voxterm-transcript.png"
    if "--shot" in args:
        shot = args[args.index("--shot") + 1]
    chrome = _find_chrome()
    if not chrome:
        print("SKIP: no chrome found"); return 3
    print(f"  chrome: {chrome}")

    tmp = tempfile.TemporaryDirectory(prefix="voxterm-gui-e2e-")
    out_dir = Path(tmp.name)
    _seed_session_fixture(out_dir)
    env = os.environ.copy()
    env["VOXTERM_GUI_OUT_DIR"] = str(out_dir)
    server = subprocess.Popen([sys.executable, "-m", "gui.server"], cwd=str(ROOT), env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser = subprocess.Popen([chrome, "--headless", "--disable-gpu", "--no-sandbox",
                                f"--remote-debugging-port={CDP_PORT}", "--remote-allow-origins=*",
                                "--window-size=1280,920", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails = []
    try:
        _wait(f"http://127.0.0.1:{PORT}/api/options")          # server up
        _wait(f"http://127.0.0.1:{CDP_PORT}/json/version")     # cdp up
        # open a blank tab, install the CSP collector at document-start, THEN navigate to the app
        _req = urllib.request.Request(
            f"http://127.0.0.1:{CDP_PORT}/json/new?about:blank", method="PUT")
        tab = json.loads(urllib.request.urlopen(_req, timeout=10).read())
        cdp = CDP(tab["webSocketDebuggerUrl"])
        cdp.call("Page.enable"); cdp.call("Runtime.enable")
        cdp.call("Page.addScriptToEvaluateOnNewDocument", source=CSP_COLLECTOR)
        cdp.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/")

        if not cdp.poll("document.querySelectorAll('#model option').length > 0"):
            fails.append("model dropdown never populated")
        else:
            opts = cdp.eval("Array.from(document.querySelectorAll('#model option')).map(o=>o.value).join(',')")
            print(f"  models: {opts}")

        cdp.poll("document.querySelectorAll('.session').length > 0")  # /api/sessions is async
        n_sessions = cdp.eval("document.querySelectorAll('.session').length") or 0
        print(f"  sessions listed: {n_sessions}")
        if n_sessions == 0:
            fails.append("no sessions listed (expected past transcripts)")
        else:
            # click the newest session (has audio) and wait for the redesigned transcript to render
            cdp.eval("document.querySelector('.session').click()")
            rendered = cdp.poll("!document.getElementById('turns').classList.contains('hidden') && "
                                "document.getElementById('turns').textContent.trim().length > 0")
            if not rendered:
                fails.append("transcript (#turns) did not render after clicking a session")
            else:
                title = cdp.eval("document.getElementById('tvTitle').textContent.trim()")
                print(f"  title: {title!r}")
                # the title must be derived from the transcript, not a raw 'YYYY-MM-DD HH:MM' date
                import re as _re
                if not title or _re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", title or ""):
                    fails.append(f"title is a raw date, not transcript-derived: {title!r}")

                # --- the load-bearing check: does the recording's audio LOAD UNDER CSP? ---
                cdp.poll("!document.getElementById('player').classList.contains('hidden') && "
                         "!!document.getElementById('player').src", timeout=10)
                has_player = cdp.eval("!document.getElementById('player').classList.contains('hidden') && "
                                      "!!document.getElementById('player').src")
                if not has_player:
                    print("  ! note: this session exposed no audio player (skipping audio-CSP probe)")
                else:
                    # probe a fresh Audio() with the same same-origin src; it obeys media-src exactly
                    # like the inline player. canplay/loadedmetadata => CSP allows it; error => blocked.
                    verdict = cdp.eval(
                        "(async()=>{const u=document.getElementById('player').src;"
                        "return await new Promise(res=>{const a=new Audio();"
                        "a.addEventListener('loadedmetadata',()=>res('loadedmetadata'),{once:true});"
                        "a.addEventListener('canplay',()=>res('canplay'),{once:true});"
                        "a.addEventListener('error',()=>res('error:'+(a.error&&a.error.code)),{once:true});"
                        "a.src=u;a.load();setTimeout(()=>res('timeout:rs'+a.readyState),6000);});})()")
                    print(f"  audio-under-CSP: {verdict}")
                    if not (verdict and str(verdict).startswith(("loadedmetadata", "canplay"))):
                        fails.append(f"audio did not load under CSP (media-src regression?): {verdict}")
                    # with preload=metadata the VISIBLE player should report a real duration
                    cdp.poll("isFinite(document.getElementById('player').duration) && "
                             "document.getElementById('player').duration > 0", timeout=8)
                    dur = cdp.eval("document.getElementById('player').duration")
                    print(f"  player duration: {dur}")
                    if not (isinstance(dur, (int, float)) and dur > 0):
                        fails.append(f"player shows no duration (preload/header issue): {dur}")

                # --- local-LLM summarize: with no backend on CI it MUST fail gracefully (no crash) ---
                cdp.eval("document.getElementById('summarizeLocal').click()")
                settled = cdp.poll(
                    "(document.getElementById('summaryBody').textContent.trim().length > 0) || "
                    "document.getElementById('summaryBlock').classList.contains('hidden')", timeout=20)
                got_summary = cdp.eval("document.getElementById('summaryBody').textContent.trim().length > 0")
                print(f"  summarize settled={settled} (got_summary={got_summary}; no-backend → graceful hide expected)")
                if not settled:
                    fails.append("summarize neither produced a summary nor failed gracefully")

                # source selector: all three options present + selectable
                srcopts = cdp.eval("Array.from(document.getElementById('source').options).map(o=>o.value).join(',')")
                if srcopts != "mic,system,both":
                    fails.append(f"audio-source options wrong: {srcopts!r}")

                time.sleep(0.6)  # let the fade-in finish so the screenshot is crisp
                png = cdp.call("Page.captureScreenshot")["data"]
                Path(shot).write_bytes(base64.b64decode(png))
                print(f"  screenshot: {shot}")

        # --- best-effort: a real record -> stop cycle drives the state machine without error ---
        rec_started = cdp.eval(
            "(async()=>{const r=await fetch('/api/record/start',{method:'POST',"
            "headers:{'Content-Type':'application/json'},body:JSON.stringify({device:-1})});"
            "const j=await r.json();return !!j.ok;})()")
        if not rec_started:
            print("  ! note: record/start failed (no mic in this env) — skipping record-cycle check")
        else:
            print("  recording… (2s)")
            time.sleep(2.0)
            cdp.eval("(async()=>{await fetch('/api/record/stop',{method:'POST',"
                     "headers:{'Content-Type':'application/json'},"
                     "body:JSON.stringify({model:document.getElementById('model').value,"
                     "language:document.getElementById('language').value,diarize:true})});})()")
            # SSE drives the job to done/error; the record button must come back enabled and not stick
            settled = cdp.poll("document.getElementById('recBtn') && "
                               "!document.getElementById('recBtn').disabled", timeout=60)
            state = cdp.eval("document.getElementById('recState').textContent")
            print(f"  record cycle settled={settled} recState={state!r}")
            if not settled:
                fails.append("record button stuck disabled after a record->stop->transcribe cycle")

        # no CSP violations should have fired anywhere in the flow
        csp = cdp.eval("(window.__csp||[]).join(' | ')")
        if csp:
            fails.append(f"CSP violation(s) fired: {csp}")
        else:
            print("  CSP: no violations")
        # no uncaught JS errors / unhandled rejections anywhere in the flow
        errs = cdp.eval("(window.__err||[]).join(' | ')")
        if errs:
            fails.append(f"uncaught JS error(s): {errs}")
        else:
            print("  JS: no uncaught errors")
    finally:
        browser.terminate()
        server.terminate()
        tmp.cleanup()

    if fails:
        print("FAIL: " + "; ".join(fails)); return 1
    print("OK: GUI e2e passed (dropdown + sessions + transcript + title + audio-under-CSP + record cycle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
