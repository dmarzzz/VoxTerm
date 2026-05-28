"""Redaction profile picker + result modals — choose what to mask, then
review the masked copy. Mirrors summary_screen.py but for PII redaction."""

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

from redaction.prompts import PROFILES


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


class RedactionScreen(ModalScreen):
    """Pick a redaction profile and (optionally) supply a custom instruction.

    Dismisses with a dict or None:
        {"profile_id": str, "custom_instructions": str, "redaction_model": str}
            — proceed ("redaction_model": blank = on-device MLX on Apple
              Silicon, or an "ollama:<model>[@host]" string)
        None — cancelled
    """

    DEFAULT_CSS = """
    RedactionScreen {
        align: center middle;
    }
    #redact-dialog {
        width: 64;
        height: auto;
        max-height: 24;
        border: heavy #ff8855;
        border-title-color: #ffb38a;
        border-title-style: bold;
        background: #140d0a;
        padding: 1 2;
    }
    #redact-list {
        height: auto;
        max-height: 10;
        background: #140d0a;
        color: #c0c0c0;
    }
    #redact-list > .option-list--option-highlighted {
        background: #3a1f1a;
        color: #ffb38a;
    }
    #redact-custom-container {
        height: 3;
        margin-top: 1;
        display: none;
    }
    #redact-custom-container.-visible {
        display: block;
    }
    #redact-custom {
        width: 100%;
        background: #221511;
        color: #ffb38a;
        border: tall #442211;
    }
    #redact-custom:focus {
        border: tall #ff8855;
    }
    #redact-model-label {
        height: 1;
        color: #807060;
        margin-top: 1;
    }
    #redact-model {
        width: 100%;
        background: #221511;
        color: #ffb38a;
        border: tall #442211;
    }
    #redact-model:focus {
        border: tall #ff8855;
    }
    #redact-hint {
        height: 1;
        color: #807060;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        default_profile_id: str = "standard",
        default_custom_instructions: str = "",
        default_redaction_model: str = "",
    ):
        super().__init__()
        self._default_id = default_profile_id
        self._default_custom = default_custom_instructions
        self._default_model = default_redaction_model
        self._selected_id: str = default_profile_id

    def compose(self) -> ComposeResult:
        with Vertical(id="redact-dialog") as dialog:
            dialog.border_title = "REDACT TRANSCRIPT"

            options = []
            initial_index = 0
            for idx, prof in enumerate(PROFILES):
                label = f"  {prof.label:20s}  [#807060]{prof.description}[/]"
                options.append(Option(label, id=prof.id))
                if prof.id == self._default_id:
                    initial_index = idx
            ol = OptionList(*options, id="redact-list")
            ol.highlighted = initial_index
            yield ol

            with Vertical(id="redact-custom-container"):
                yield Input(
                    placeholder="What should be redacted?…",
                    id="redact-custom",
                    value=self._default_custom,
                )

            yield Static(
                "[#807060]redaction model[/] "
                "[#605040](blank = Apple Silicon MLX, or "
                "ollama:model or ollama:model@host)[/]",
                id="redact-model-label",
                markup=True,
            )
            yield Input(
                placeholder="blank = Apple Silicon MLX  ·  e.g. "
                "ollama:qwen3:0.6b",
                id="redact-model",
                value=self._default_model,
            )

            yield Static(
                " [#807060]ENTER[/] redact  [#807060]ESC[/] cancel",
                id="redact-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        self._sync_custom_visibility()

    def _sync_custom_visibility(self) -> None:
        container = self.query_one("#redact-custom-container")
        if self._selected_id == "custom":
            container.add_class("-visible")
            self.query_one("#redact-custom", Input).focus()
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
            custom = self.query_one("#redact-custom", Input).value.strip()
            if not custom:
                return  # require an instruction for custom
        model = self.query_one("#redact-model", Input).value.strip()
        self.dismiss(
            {
                "profile_id": self._selected_id,
                "custom_instructions": custom,
                "redaction_model": model,
            }
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class RedactionResultScreen(ModalScreen):
    """Shows the redacted transcript + a per-category tally of what was masked.

    The redacted copy has already been written to ``path``; this is a
    read-only review with copy / open-file. Self-contained (own
    clipboard/open helpers) to avoid importing from ``tui.app``.
    """

    DEFAULT_CSS = """
    RedactionResultScreen {
        align: center middle;
    }
    #redact-result-dialog {
        width: 84;
        height: auto;
        max-height: 30;
        border: heavy #ff8855;
        border-title-color: #ffb38a;
        border-title-style: bold;
        background: #140d0a;
        padding: 1 2;
    }
    #redact-result-tally {
        height: auto;
        color: #ffb38a;
        margin-bottom: 1;
    }
    #redact-result-body {
        height: auto;
        max-height: 20;
        background: #140d0a;
    }
    #redact-result-body Markdown {
        background: #140d0a;
        color: #c0c0c0;
    }
    #redact-result-path {
        height: auto;
        color: #807060;
        margin-top: 1;
    }
    #redact-result-status {
        height: 1;
        color: #ffb38a;
        margin-top: 1;
    }
    #redact-result-hint {
        height: 1;
        color: #807060;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape,enter,q", "close", "Close"),
        Binding("c", "copy", "Copy"),
        Binding("o", "open_file", "Open file"),
    ]

    def __init__(
        self,
        redacted_text: str,
        counts: dict[str, int] | None = None,
        total: int = 0,
        path: str = "",
        profile_label: str = "",
    ):
        super().__init__()
        self._redacted = redacted_text
        self._counts = counts or {}
        self._total = total
        self._path = path
        self._label = profile_label

    def _tally_line(self) -> str:
        if self._total == 0:
            return "[#807060]no sensitive spans found[/]"
        parts = " · ".join(
            f"[#ffb38a]{n}[/] {cat}"
            for cat, n in sorted(
                self._counts.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        span_word = "span" if self._total == 1 else "spans"
        return f"[#ffb38a]{self._total}[/] {span_word} masked  —  {parts}"

    def compose(self) -> ComposeResult:
        with Vertical(id="redact-result-dialog") as dialog:
            dialog.border_title = (
                f"REDACTED — {self._label}" if self._label else "REDACTED"
            )
            yield Static(self._tally_line(), id="redact-result-tally", markup=True)
            with VerticalScroll(id="redact-result-body"):
                yield Markdown(self._redacted)
            if self._path:
                yield Static(f"saved → {self._path}", id="redact-result-path")
            yield Static("", id="redact-result-status", markup=True)
            yield Static(
                " [#807060]C[/] copy  [#807060]O[/] open file  "
                "[#807060]ESC[/] close",
                id="redact-result-hint",
                markup=True,
            )

    def _status(self, msg: str) -> None:
        self.query_one("#redact-result-status", Static).update(msg)

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
            proc.communicate(self._redacted.encode("utf-8"))
            self._status("[#ffb38a]✓ redacted transcript copied to clipboard[/]")
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
            self._status(f"[#ffb38a]✓ opened {self._path}[/]")
        except Exception:
            self._status("[#ff5577]could not open file[/]")

    def action_close(self) -> None:
        self.dismiss(None)
