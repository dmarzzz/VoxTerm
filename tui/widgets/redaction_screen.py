"""Redaction profile picker + result modals — choose what to mask, then
review the masked copy. Mirrors summary_screen.py but for PII redaction."""

from __future__ import annotations

import shutil
import subprocess
import sys

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Input,
    Markdown,
    OptionList,
    SelectionList,
    Static,
)
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection
from textual.binding import Binding
from textual.screen import ModalScreen

from redaction.engine import apply_redactions
from redaction.prompts import PROFILES

# Disposition of the unredacted live-autosave file once the redacted copy is
# written. Kept here so the screen and app agree on the string values.
DISPOSITIONS = (
    ("keep", "Keep original", "leave the unredacted live transcript on disk"),
    ("replace", "Replace original", "delete the unredacted live transcript"),
    ("shred", "Shred original", "overwrite then delete (best-effort, not a forensic wipe)"),
)


def _ellipsize(text: str, width: int = 56) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[: width - 1] + "…"


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
    #redact-result-orig {
        height: auto;
        color: #807060;
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
        original_note: str = "",
    ):
        super().__init__()
        self._redacted = redacted_text
        self._counts = counts or {}
        self._total = total
        self._path = path
        self._label = profile_label
        self._original_note = original_note

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
            if self._original_note:
                yield Static(f"original: {self._original_note}", id="redact-result-orig")
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


class RedactionReviewScreen(ModalScreen):
    """Review what will be masked BEFORE anything is written.

    Small models miss PII, and a miss written into a file named ``-redacted``
    is worse than no tool — it feels safe. This screen makes every miss
    catchable: the found spans are shown as a toggle list (uncheck false
    positives), you can type to add spans the model missed, a live preview
    reflects the current selection, and you choose what happens to the
    unredacted original. Nothing is written unless you confirm.

    Dismisses with a dict or None:
        {"spans": [(text, type), ...], "disposition": "keep"|"replace"|"shred"}
            — write with exactly these spans
        None — cancelled, write nothing
    """

    DEFAULT_CSS = """
    RedactionReviewScreen {
        align: center middle;
    }
    #review-dialog {
        width: 92;
        height: auto;
        max-height: 40;
        border: heavy #ff8855;
        border-title-color: #ffb38a;
        border-title-style: bold;
        background: #140d0a;
        padding: 1 2;
    }
    #review-instr { height: auto; color: #807060; margin-bottom: 1; }
    #review-spans {
        height: auto;
        max-height: 9;
        background: #140d0a;
        color: #c0c0c0;
    }
    #review-add {
        width: 100%;
        background: #221511;
        color: #ffb38a;
        border: tall #442211;
        margin-top: 1;
    }
    #review-add:focus { border: tall #ff8855; }
    #review-disp-label { height: 1; color: #807060; margin-top: 1; }
    #review-disposition {
        height: auto;
        max-height: 3;
        background: #140d0a;
        color: #c0c0c0;
    }
    #review-disposition > .option-list--option-highlighted {
        background: #3a1f1a;
        color: #ffb38a;
    }
    #review-preview-label { height: 1; color: #807060; margin-top: 1; }
    #review-preview {
        height: auto;
        max-height: 9;
        background: #0d0806;
        border: tall #442211;
    }
    #review-preview Static { color: #9aa0a6; }
    #review-buttons { height: auto; margin-top: 1; align-horizontal: right; }
    #review-buttons Button { margin-left: 2; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+w", "confirm", "Write"),
    ]

    def __init__(
        self,
        spans: list[tuple[str, str]],
        body: str,
        profile_label: str = "",
    ):
        super().__init__()
        # mutable working copy; add-span appends here
        self._spans: list[tuple[str, str]] = list(spans)
        self._body = body
        self._label = profile_label
        self._disposition = "keep"

    @staticmethod
    def _span_prompt(text: str, typ: str) -> str:
        return f"[#ffb38a]{typ:<10}[/] {_ellipsize(text)}"

    def compose(self) -> ComposeResult:
        with Vertical(id="review-dialog") as dialog:
            n = len(self._spans)
            dialog.border_title = (
                f"REVIEW REDACTION — {n} found"
                + (f" · {self._label}" if self._label else "")
            )
            yield Static(
                "Uncheck false positives · type below to add a missed span · "
                "models miss things, so scan the preview before writing.",
                id="review-instr",
            )

            sl: SelectionList[int] = SelectionList(
                *[
                    Selection(self._span_prompt(t, ty), i, True)
                    for i, (t, ty) in enumerate(self._spans)
                ],
                id="review-spans",
            )
            yield sl

            yield Input(
                placeholder="add a word/phrase the model missed, then Enter…",
                id="review-add",
            )

            yield Static("[#807060]original file[/]", id="review-disp-label", markup=True)
            disp = OptionList(
                *[
                    Option(f"  {label:<18s}  [#605040]{desc}[/]", id=did)
                    for did, label, desc in DISPOSITIONS
                ],
                id="review-disposition",
            )
            disp.highlighted = 0
            yield disp

            yield Static("[#807060]preview[/]", id="review-preview-label", markup=True)
            with VerticalScroll(id="review-preview"):
                yield Static("", id="review-preview-body")

            with Horizontal(id="review-buttons"):
                yield Button("Cancel", id="review-cancel")
                yield Button("Write redacted copy", id="review-confirm", variant="warning")

    def on_mount(self) -> None:
        self._refresh_preview()

    # --- preview ---------------------------------------------------------

    def _selected_spans(self) -> list[tuple[str, str]]:
        sl = self.query_one("#review-spans", SelectionList)
        chosen = set(sl.selected)
        return [self._spans[i] for i in range(len(self._spans)) if i in chosen]

    def _refresh_preview(self) -> None:
        result = apply_redactions(self._body, self._selected_spans())
        # show just the transcript portion (after the first --- divider)
        text = result.redacted_text
        marker = "\n---\n"
        idx = text.find(marker)
        shown = text[idx + len(marker):] if idx != -1 else text
        body = self.query_one("#review-preview-body", Static)
        if result.total == 0:
            body.update("[#807060](nothing will be masked)[/]\n\n" + shown)
        else:
            body.update(shown)

    def on_selection_list_selected_changed(self, event) -> None:
        self._refresh_preview()

    def on_option_list_option_highlighted(self, event) -> None:
        if event.option_list.id == "review-disposition" and event.option.id:
            self._disposition = event.option.id

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "review-add":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        idx = len(self._spans)
        self._spans.append((text, "OTHER"))
        self.query_one("#review-spans", SelectionList).add_option(
            Selection(self._span_prompt(text, "OTHER"), idx, True)
        )
        self._refresh_preview()

    # --- actions ---------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "review-confirm":
            self.action_confirm()
        elif event.button.id == "review-cancel":
            self.action_cancel()

    def action_confirm(self) -> None:
        self.dismiss(
            {"spans": self._selected_spans(), "disposition": self._disposition}
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
