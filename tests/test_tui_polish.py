from __future__ import annotations

import subprocess
from types import SimpleNamespace

from tui import clipboard
import tui.app as app_mod
from tui.app import VoxTerm
from tui.widgets.transcript import TranscriptPanel
from tui.widgets.transcript_explorer import TranscriptExplorerScreen
from tui.widgets.waveform import WaveformWidget


class _FakeTranscript:
    def __init__(self, entries=None):
        self._entries = list(entries or [])
        self.messages: list[tuple[str, str | None]] = []

    def get_entries(self):
        return list(self._entries)

    def get_plain_text(self):
        return "plain transcript"

    def system_message(self, message, category=None, highlights=None):
        self.messages.append((message, category))


class _FakeProc:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.input = b""

    def communicate(self, data):
        self.input = data
        return b"", b""


def _binding(action: str):
    return next(binding for binding in VoxTerm.BINDINGS if binding.action == action)


def test_tui_footer_bindings_surface_profiles_and_transcripts_label():
    assert _binding("manage_profiles").show is True
    assert _binding("manage_profiles").description == "Profiles"
    assert _binding("explore_transcripts").description == "Transcripts"
    assert _binding("clear_transcript").description == "Clear"
    assert _binding("toggle_debug").description == "Debug"


def test_clipboard_cmd_supports_windows(monkeypatch):
    monkeypatch.setattr(clipboard.sys, "platform", "win32")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: None)

    assert clipboard.clipboard_cmd() == ["clip.exe"]


def test_export_clipboard_reports_subprocess_exit(monkeypatch):
    transcript = _FakeTranscript(entries=[("12:00:00", "transcript", "hello", "Speaker 1", 1, "")])
    app = object.__new__(VoxTerm)
    monkeypatch.setattr(app, "query_one", lambda _selector: transcript)
    monkeypatch.setattr(app, "_start_new_session", lambda: transcript.messages.append(("cleared", None)))
    monkeypatch.setattr(app_mod, "clipboard_cmd", lambda: ["copy-tool"])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProc(returncode=17))

    app._export_to_clipboard()

    assert transcript.messages == [("clipboard copy failed (exit 17)", "rec")]


def test_clear_transcript_noops_when_empty(monkeypatch):
    transcript = _FakeTranscript(entries=[])
    app = object.__new__(VoxTerm)
    monkeypatch.setattr(app, "query_one", lambda _selector: transcript)
    monkeypatch.setattr(app, "push_screen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected dialog")))

    app.action_clear_transcript()


def test_clear_transcript_confirms_when_entries_exist(monkeypatch):
    transcript = _FakeTranscript(entries=[("12:00:00", "transcript", "hello", "Speaker 1", 1, "")])
    app = object.__new__(VoxTerm)
    pushed: list[object] = []
    monkeypatch.setattr(app, "query_one", lambda _selector: transcript)
    monkeypatch.setattr(app, "push_screen", lambda screen, callback: pushed.append((screen, callback)))

    app.action_clear_transcript()

    assert len(pushed) == 1


def test_loading_indicator_helper_toggles_widget(monkeypatch):
    app = object.__new__(VoxTerm)
    widget = SimpleNamespace(display=False)

    def fake_query(selector, cls=None):
        assert selector == "#model-loading-indicator"
        return widget

    monkeypatch.setattr(app, "query_one", fake_query)
    app._show_loading_indicator(True)
    assert widget.display is True
    app._show_loading_indicator(False)
    assert widget.display is False


def test_waveform_blank_render_lines_have_explicit_style():
    widget = WaveformWidget()
    widget._size = SimpleNamespace(width=12, height=3)

    strip = widget.render_line(0)

    assert all(segment.style is not None for segment in strip._segments)


def test_transcript_explorer_reports_clipboard_failure_detail(tmp_path, monkeypatch):
    transcript = tmp_path / "2026-01-02_030405.md"
    transcript.write_text("hello", encoding="utf-8")
    screen = TranscriptExplorerScreen(tmp_path)
    captured: list[object] = []
    monkeypatch.setattr(screen, "dismiss", lambda result=None: captured.append(result))
    monkeypatch.setattr("tui.widgets.transcript_explorer.clipboard_cmd", lambda: ["copy-tool"])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProc(returncode=9))

    event = SimpleNamespace(option=SimpleNamespace(id=str(transcript)))
    screen.on_option_list_option_selected(event)

    assert captured == [{"error": "clipboard copy failed (exit 9)"}]
