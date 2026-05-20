"""Summary template picker modal — choose a prompt before save-with-summary."""

from __future__ import annotations

import shutil
import subprocess
import sys

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, OptionList, Input, Markdown
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen

from summarizer.prompts import TEMPLATES


def _clipboard_cmd() -> list[str] | None:
    if sys.platform == "darwin":
        return ["pbcopy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    if shutil.which("wl-copy"):
        return ["wl-copy"]
    return None


def _open_cmd() -> str | None:
    if sys.platform == "darwin":
        return "open"
    if sys.platform == "win32":
        return "start"
    if shutil.which("xdg-open"):
        return "xdg-open"
    return None


class SummaryScreen(ModalScreen):
    """Pick a summarization template and (optionally) supply a custom prompt.

    Dismisses with a dict or None:
        {"template_id": str, "custom_prompt": str, "summary_model": str}
            — proceed ("summary_model": blank = on-device MLX on Apple Silicon, or
              an "ollama:<model>[@host]" string)
        None — cancelled
    """

    DEFAULT_CSS = """
    SummaryScreen {
        align: center middle;
    }
    #summary-dialog {
        width: 64;
        height: auto;
        max-height: 24;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #summary-list {
        height: auto;
        max-height: 10;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #summary-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #summary-custom-container {
        height: 3;
        margin-top: 1;
        display: none;
    }
    #summary-custom-container.-visible {
        display: block;
    }
    #summary-custom {
        width: 100%;
        background: #111822;
        color: #00ffcc;
        border: tall #003344;
    }
    #summary-custom:focus {
        border: tall #00e5ff;
    }
    #summary-model-label {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    #summary-model {
        width: 100%;
        background: #111822;
        color: #00ffcc;
        border: tall #003344;
    }
    #summary-model:focus {
        border: tall #00e5ff;
    }
    #summary-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        default_template_id: str = "tldr",
        default_custom_prompt: str = "",
        default_summary_model: str = "",
    ):
        super().__init__()
        self._default_id = default_template_id
        self._default_custom = default_custom_prompt
        self._default_model = default_summary_model
        self._selected_id: str = default_template_id

    def compose(self) -> ComposeResult:
        with Vertical(id="summary-dialog") as dialog:
            dialog.border_title = "SAVE WITH SUMMARY"

            options = []
            initial_index = 0
            for idx, tmpl in enumerate(TEMPLATES):
                label = f"  {tmpl.label:18s}  [#607080]{tmpl.description}[/]"
                options.append(Option(label, id=tmpl.id))
                if tmpl.id == self._default_id:
                    initial_index = idx
            ol = OptionList(*options, id="summary-list")
            ol.highlighted = initial_index
            yield ol

            with Vertical(id="summary-custom-container"):
                yield Input(
                    placeholder="Your summarization instruction…",
                    id="summary-custom",
                    value=self._default_custom,
                )

            yield Static(
                "[#607080]summary model[/] "
                "[#405060](blank = Apple Silicon MLX, or "
                "ollama:model or ollama:model@host)[/]",
                id="summary-model-label",
                markup=True,
            )
            yield Input(
                placeholder="blank = Apple Silicon MLX  ·  e.g. "
                "ollama:qwen3.5:35b",
                id="summary-model",
                value=self._default_model,
            )

            yield Static(
                " [#607080]ENTER[/] summarize  [#607080]ESC[/] cancel",
                id="summary-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        self._sync_custom_visibility()

    def _sync_custom_visibility(self) -> None:
        container = self.query_one("#summary-custom-container")
        if self._selected_id == "custom":
            container.add_class("-visible")
            self.query_one("#summary-custom", Input).focus()
        else:
            container.remove_class("-visible")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option and event.option.id:
            self._selected_id = event.option.id
            self._sync_custom_visibility()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option and event.option.id:
            self._selected_id = event.option.id
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        custom = ""
        if self._selected_id == "custom":
            custom = self.query_one("#summary-custom", Input).value.strip()
            if not custom:
                return  # require a prompt for custom
        model = self.query_one("#summary-model", Input).value.strip()
        self.dismiss(
            {
                "template_id": self._selected_id,
                "custom_prompt": custom,
                "summary_model": model,
            }
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class SummaryResultScreen(ModalScreen):
    """Shows the generated summary with copy-to-clipboard and open-file actions.

    The transcript + summary have already been written to ``path``; this is a
    read-only result view so the summary is reachable without hunting through
    the transcripts folder. Self-contained (own clipboard/open helpers) to
    avoid importing from ``tui.app``.
    """

    DEFAULT_CSS = """
    SummaryResultScreen {
        align: center middle;
    }
    #summary-result-dialog {
        width: 84;
        height: auto;
        max-height: 30;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #summary-result-body {
        height: auto;
        max-height: 22;
        background: #0a0e14;
    }
    #summary-result-body Markdown {
        background: #0a0e14;
        color: #c0c0c0;
    }
    #summary-result-path {
        height: auto;
        color: #607080;
        margin-top: 1;
    }
    #summary-result-status {
        height: 1;
        color: #00ffcc;
        margin-top: 1;
    }
    #summary-result-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape,enter,q", "close", "Close"),
        Binding("c", "copy", "Copy"),
        Binding("o", "open_file", "Open file"),
    ]

    def __init__(self, summary: str, path: str = "", template_label: str = ""):
        super().__init__()
        self._summary = summary.strip()
        self._path = path
        self._label = template_label

    def compose(self) -> ComposeResult:
        with Vertical(id="summary-result-dialog") as dialog:
            dialog.border_title = (
                f"SUMMARY — {self._label}" if self._label else "SUMMARY"
            )
            with VerticalScroll(id="summary-result-body"):
                yield Markdown(self._summary)
            if self._path:
                yield Static(
                    f"saved → {self._path}",
                    id="summary-result-path",
                )
            yield Static("", id="summary-result-status", markup=True)
            yield Static(
                " [#607080]C[/] copy  [#607080]O[/] open file  "
                "[#607080]ESC[/] close",
                id="summary-result-hint",
                markup=True,
            )

    def _status(self, msg: str) -> None:
        self.query_one("#summary-result-status", Static).update(msg)

    def action_copy(self) -> None:
        cmd = _clipboard_cmd()
        if cmd is None:
            self._status(
                "[#ff5577]no clipboard tool found "
                "(install xclip, xsel, or wl-copy)[/]"
            )
            return
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            proc.communicate(self._summary.encode("utf-8"))
            self._status("[#00ffcc]✓ summary copied to clipboard[/]")
        except Exception:
            self._status("[#ff5577]clipboard copy failed[/]")

    def action_open_file(self) -> None:
        if not self._path:
            self._status("[#ff5577]no file path[/]")
            return
        opener = _open_cmd()
        if opener is None:
            self._status("[#ff5577]no file opener found[/]")
            return
        try:
            subprocess.Popen([opener, self._path])
            self._status(f"[#00ffcc]✓ opened {self._path}[/]")
        except Exception:
            self._status("[#ff5577]could not open file[/]")

    def action_close(self) -> None:
        self.dismiss(None)
