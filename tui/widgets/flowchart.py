from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


_PLACEHOLDER = (
    "[#607080]waiting for transcript…[/]\n"
    "[#445566]flowchart refreshes roughly every 15 seconds[/]"
)


class FlowchartPanel(VerticalScroll):
    """Side panel that displays an LLM-generated mermaid flowchart source.

    The LLM emits mermaid syntax (e.g. `A[Alice] -->|asks| B[Bob]`); we render
    the source as-is — it's structured enough to read in the terminal and can
    be copy-pasted into a real mermaid renderer for a graphical view.
    """

    DEFAULT_CSS = """
    FlowchartPanel {
        border: heavy #ff44aa;
        border-title-color: #ff88cc;
        border-title-style: bold;
        border-title-align: left;
        background: #0d1117;
        margin: 0 1 0 0;
        width: 42;
        scrollbar-size-vertical: 0;
        display: none;
    }
    FlowchartPanel.--visible {
        display: block;
    }
    FlowchartPanel > #flowchart-body {
        padding: 0 1;
        color: #ccddee;
    }
    FlowchartPanel > #flowchart-status {
        height: 1;
        padding: 0 1;
        color: #607080;
    }
    """

    def __init__(self):
        super().__init__()
        self.border_title = "FLOWCHART // LLM"
        self._body: Static | None = None
        self._status: Static | None = None

    def compose(self):
        self._status = Static("", id="flowchart-status", markup=True)
        self._body = Static(_PLACEHOLDER, id="flowchart-body", markup=True)
        yield self._status
        yield self._body

    def set_visible(self, visible: bool) -> None:
        if visible:
            self.add_class("--visible")
        else:
            self.remove_class("--visible")

    def is_visible(self) -> bool:
        return self.has_class("--visible")

    def update_flowchart(self, mermaid_source: str) -> None:
        if self._body is None:
            return
        # Escape Rich/Textual markup brackets — mermaid uses `[Label]` heavily.
        safe = mermaid_source.replace("[", r"\[")
        self._body.update(safe)

    def set_status(self, text: str) -> None:
        if self._status is None:
            return
        self._status.update(text)

    def show_error(self, message: str) -> None:
        if self._body is None:
            return
        self._body.update(f"[#ff6644]error:[/] [#cc8866]{message}[/]")
