from pathlib import Path

import display_mode
from display_mode import find_shaderclaw_dir, launch_display


def _make_shaderclaw(path: Path) -> Path:
    path.mkdir()
    (path / "server.js").write_text("// server\n", encoding="utf-8")
    (path / "package.json").write_text("{}\n", encoding="utf-8")
    return path


def test_find_shaderclaw_dir_prefers_explicit_path(tmp_path):
    shaderclaw = _make_shaderclaw(tmp_path / "shader-claw3")

    assert find_shaderclaw_dir(shaderclaw) == shaderclaw


def test_find_shaderclaw_dir_detects_sibling_checkout(tmp_path):
    repo_root = tmp_path / "voxterm"
    repo_root.mkdir()
    shaderclaw = _make_shaderclaw(tmp_path / "shader-claw3")

    assert find_shaderclaw_dir(repo_root=repo_root) == shaderclaw


def test_launch_display_reuses_existing_server(monkeypatch):
    opened = []

    monkeypatch.setattr(display_mode, "is_display_running", lambda port: True)
    monkeypatch.setattr(
        display_mode, "open_display_browser", lambda port: opened.append(port) or True
    )

    result = launch_display(port=8899)

    assert result.ok is True
    assert result.started is False
    assert result.url == "http://localhost:8899/voxterm"
    assert opened == [8899]


def test_launch_display_starts_shaderclaw(monkeypatch, tmp_path):
    shaderclaw = _make_shaderclaw(tmp_path / "shader-claw3")
    calls = {"running": 0, "popen": None, "opened": []}

    class FakeProcess:
        pid = 4242

        def poll(self):
            return None

    def fake_is_running(port):
        calls["running"] += 1
        return calls["running"] >= 2

    def fake_popen(args, **kwargs):
        calls["popen"] = (args, kwargs)
        return FakeProcess()

    monkeypatch.setattr(display_mode, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(display_mode, "is_display_running", fake_is_running)
    monkeypatch.setattr(
        display_mode.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr(display_mode.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        display_mode,
        "open_display_browser",
        lambda port: calls["opened"].append(port) or True,
    )

    result = launch_display(port=8899, shaderclaw_dir=shaderclaw, live_dir=tmp_path / ".live")

    assert result.ok is True
    assert result.started is True
    assert result.pid == 4242
    assert calls["opened"] == [8899]
    args, kwargs = calls["popen"]
    assert args == ["/usr/bin/node", "server.js", "--voxterm"]
    assert kwargs["cwd"] == str(shaderclaw)
    assert kwargs["env"]["SHADERCLAW_PORT"] == "8899"
    assert kwargs["env"]["SHADERCLAW_VOXTERM_DISPLAY"] == "1"
    assert kwargs["env"]["VOXTERM_LIVE_DIR"] == str(tmp_path / ".live")


def test_launch_display_reports_missing_shaderclaw(monkeypatch, tmp_path):
    monkeypatch.setattr(display_mode, "is_display_running", lambda port: False)
    monkeypatch.setattr(display_mode, "find_shaderclaw_dir", lambda shaderclaw_dir: None)

    result = launch_display(shaderclaw_dir=tmp_path / "missing")

    assert result.ok is False
    assert "ShaderClaw checkout not found" in result.message
