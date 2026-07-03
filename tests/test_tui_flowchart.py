from __future__ import annotations

from types import SimpleNamespace

import pytest

import tui.llm.flowchart_gen as flowchart_gen
from summarizer.engine import SummarizerError
from tui.app import VoxTerm
from tui.widgets.flowchart import FlowchartPanel
from tui.widgets.transcript import TranscriptPanel


class _FakeSummarizer:
    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def summarize(self, transcript, template, custom_prompt=""):
        self.calls.append((transcript, template, custom_prompt))
        return self.output


class _FakePanel:
    def __init__(self):
        self.status: list[str] = []
        self.errors: list[str] = []
        self.flowcharts: list[str] = []
        self.visible = False

    def set_visible(self, visible: bool):
        self.visible = visible

    def set_status(self, text: str):
        self.status.append(text)

    def update_flowchart(self, text: str):
        self.flowcharts.append(text)

    def show_error(self, text: str):
        self.errors.append(text)


class _FakeTranscript:
    def __init__(self, entries):
        self._entries = entries
        self.messages: list[str] = []

    def get_entries(self):
        return list(self._entries)

    def system_message(self, message, *args, **kwargs):
        self.messages.append(message)


def _binding(action: str):
    return next(binding for binding in VoxTerm.BINDINGS if binding.action == action)


def test_clean_mermaid_strips_fence_and_preamble():
    raw = "Here you go:\n```mermaid\nflowchart TD\nA[Alice] --> B[Bob]\n```"

    assert flowchart_gen.clean_mermaid(raw) == "flowchart TD\nA[Alice] --> B[Bob]"


def test_generate_flowchart_uses_summarizer_template(monkeypatch):
    fake = _FakeSummarizer("A[hello]")
    monkeypatch.setattr(flowchart_gen, "get_summarizer", lambda model_name="": fake)

    out = flowchart_gen.generate_flowchart("Speaker 1: hello", model_name="ollama:test")

    assert out == "flowchart TD\nA[hello]"
    assert fake.calls[0][1].id == "flowchart"


def test_generate_flowchart_wraps_summarizer_errors(monkeypatch):
    def raise_error(model_name=""):
        raise SummarizerError("no backend")

    monkeypatch.setattr(flowchart_gen, "get_summarizer", raise_error)

    with pytest.raises(flowchart_gen.FlowchartGenError, match="no backend"):
        flowchart_gen.generate_flowchart("text")


def test_flowchart_panel_escapes_mermaid_brackets():
    panel = FlowchartPanel()
    body = SimpleNamespace(values=[], update=lambda value: body.values.append(value))
    panel._body = body

    panel.update_flowchart("flowchart TD\nA[Alice]")

    assert body.values == ["flowchart TD\nA\\[Alice]"]

    panel.show_error("bad [model]")

    assert body.values[-1] == "[#ff6644]error:[/] [#cc8866]bad \\[model][/]"


def test_flowchart_binding_added_without_removing_gui_binding():
    assert _binding("toggle_flowchart").key == "f"
    assert _binding("toggle_flowchart").description == "Flowchart"
    assert _binding("launch_gui").key == "g"


def test_maybe_refresh_flowchart_waits_for_transcript_turns(monkeypatch):
    app = object.__new__(VoxTerm)
    panel = _FakePanel()
    transcript = _FakeTranscript([
        ("12:00:00", "system", "model loaded", "sys", 0, ""),
    ])

    def fake_query(selector):
        if selector is TranscriptPanel:
            return transcript
        if selector is FlowchartPanel:
            return panel
        raise AssertionError(selector)

    monkeypatch.setattr(app, "query_one", fake_query)
    monkeypatch.setattr(app, "_flowchart_model_name", lambda: "ollama:test")
    monkeypatch.setattr(
        app,
        "_run_flowchart_worker",
        lambda text, model: pytest.fail("should not start flowchart worker"),
    )
    app._flowchart_enabled = True
    app._flowchart_inflight = False
    app._flowchart_min_interval_sec = 0.0
    app._flowchart_last_run = 0.0
    app._flowchart_last_signature = ""

    app._maybe_refresh_flowchart()

    assert panel.status == ["[#607080]waiting for transcript[/]"]


def test_maybe_refresh_flowchart_uses_transcript_turns_only(monkeypatch):
    app = object.__new__(VoxTerm)
    panel = _FakePanel()
    transcript = _FakeTranscript([
        ("12:00:00", "system", "model loaded", "sys", 0, ""),
        ("12:00:01", "transcript", "hello", "Alice", 1, ""),
    ])
    captured = []

    def fake_query(selector):
        if selector is TranscriptPanel:
            return transcript
        if selector is FlowchartPanel:
            return panel
        raise AssertionError(selector)

    monkeypatch.setattr(app, "query_one", fake_query)
    monkeypatch.setattr(app, "_flowchart_model_name", lambda: "ollama:test")
    monkeypatch.setattr(app, "_run_flowchart_worker", lambda text, model: captured.append((text, model)))
    app._flowchart_enabled = True
    app._flowchart_inflight = False
    app._flowchart_min_interval_sec = 0.0
    app._flowchart_last_run = 0.0
    app._flowchart_last_signature = ""

    app._maybe_refresh_flowchart()
    app._flowchart_inflight = False
    app._maybe_refresh_flowchart()

    assert captured == [("[12:00:01] Alice: hello", "ollama:test")]
    assert panel.status == ["[#ffaa00]generating...[/]"]


def test_apply_flowchart_result_updates_panel_and_error_state(monkeypatch):
    app = object.__new__(VoxTerm)
    panel = _FakePanel()
    monkeypatch.setattr(app, "query_one", lambda selector: panel)
    app._flowchart_inflight = True

    app._apply_flowchart_result("flowchart TD\nA[Alice]", None)

    assert app._flowchart_inflight is False
    assert panel.flowcharts == ["flowchart TD\nA[Alice]"]
    assert panel.status[-1].startswith("[#607080]updated ")

    app._flowchart_inflight = True
    app._apply_flowchart_result(None, "no backend")

    assert app._flowchart_inflight is False
    assert panel.errors == ["no backend"]
    assert panel.status[-1] == "[#ff6644]error[/]"
