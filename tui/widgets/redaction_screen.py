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
from redaction.tiers import TIERS, filter_spans, next_tier, resolve_tier, tier_masks

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


def _tier_meter(tier) -> str:
    """A 4-segment colored meter filled to the tier's rank (markup string)."""
    filled = "▰" * (tier.rank + 1)
    empty = "▱" * (len(TIERS) - tier.rank - 1)
    return f"[{tier.color}]{filled}[/][#3a332c]{empty}[/]"


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
    """Pick the starting disclosure tier and (optionally) the model.

    The tier is the *audience* you're redacting for — you can still dial it up
    or down on the review screen before writing. Detection always finds the
    full vocabulary; the tier decides what gets masked.

    Dismisses with a dict or None:
        {"tier_id": str, "redaction_model": str}
            — proceed ("redaction_model": blank = on-device MLX on Apple
              Silicon, or an "ollama:<model>[@host]" string)
        None — cancelled
    """

    DEFAULT_CSS = """
    RedactionScreen {
        align: center middle;
    }
    #redact-dialog {
        width: 70;
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
        max-height: 12;
        background: #140d0a;
        color: #c0c0c0;
    }
    #redact-list > .option-list--option-highlighted {
        background: #3a1f1a;
        color: #ffb38a;
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
        default_tier_id: str = "room",
        default_redaction_model: str = "",
    ):
        super().__init__()
        self._default_id = default_tier_id
        self._default_model = default_redaction_model
        self._selected_id: str = resolve_tier(default_tier_id).id

    def compose(self) -> ComposeResult:
        with Vertical(id="redact-dialog") as dialog:
            dialog.border_title = "REDACT — disclosure tier"

            options = []
            initial_index = 0
            for idx, tier in enumerate(TIERS):
                meter = _tier_meter(tier)
                label = (
                    f"  {meter}  [b {tier.color}]{tier.label:<6}[/] "
                    f"[#807060]{tier.audience} — {tier.description}[/]"
                )
                options.append(Option(label, id=tier.id))
                if tier.id == self._selected_id:
                    initial_index = idx
            ol = OptionList(*options, id="redact-list")
            ol.highlighted = initial_index
            yield ol

            yield Static(
                "[#807060]redaction model[/] "
                "[#605040](blank = Apple Silicon MLX · privacy-filter = "
                "OpenAI Privacy Filter (onnxruntime) · ollama:model[@host])[/]",
                id="redact-model-label",
                markup=True,
            )
            yield Input(
                placeholder="blank = MLX  ·  privacy-filter  ·  "
                "ollama:qwen3:0.6b",
                id="redact-model",
                value=self._default_model,
            )

            yield Static(
                " [#807060]ENTER[/] detect  ·  dial the tier on the next "
                "screen  ·  [#807060]ESC[/] cancel",
                id="redact-hint",
                markup=True,
            )

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option and event.option.id:
            self._selected_id = event.option.id

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option and event.option.id:
            self._selected_id = event.option.id
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        model = self.query_one("#redact-model", Input).value.strip()
        self.dismiss({"tier_id": self._selected_id, "redaction_model": model})

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
    #review-tier { height: auto; margin-bottom: 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+t", "cycle_tier", "Tier"),
        Binding("ctrl+w", "confirm", "Write"),
    ]

    def __init__(
        self,
        spans: list[tuple[str, str]],
        body: str,
        tier_id: str = "room",
    ):
        super().__init__()
        # mutable working copy; add-span appends here
        self._spans: list[tuple[str, str]] = list(spans)
        self._body = body
        self._tier = resolve_tier(tier_id)
        self._disposition = "keep"

    @staticmethod
    def _span_prompt(text: str, typ: str) -> str:
        return f"[#ffb38a]{typ:<12}[/] {_ellipsize(text)}"

    def _tier_banner(self) -> str:
        t = self._tier
        return (
            f"disclosure  {_tier_meter(t)}  [b {t.color}]{t.label}[/]  "
            f"[#807060]{t.audience} — {t.description}[/]   "
            f"[#605040](⌃T to dial)[/]"
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="review-dialog") as dialog:
            n = len(self._spans)
            dialog.border_title = f"REVIEW REDACTION — {n} found"

            yield Static(self._tier_banner(), id="review-tier", markup=True)
            yield Static(
                "The tier sets the baseline · uncheck false positives or check "
                "extras · type below to add a missed span · scan the preview.",
                id="review-instr",
            )

            # Initial check state follows the tier policy.
            sl: SelectionList[int] = SelectionList(
                *[
                    Selection(
                        self._span_prompt(t, ty), i, tier_masks(self._tier, ty)
                    )
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

    # --- tier dial -------------------------------------------------------

    def action_cycle_tier(self) -> None:
        self._tier = next_tier(self._tier)
        self.query_one("#review-tier", Static).update(self._tier_banner())
        # The dial resets the baseline selection to the new tier's policy;
        # any manual tweaks are intentionally cleared so the posture is honest.
        sl = self.query_one("#review-spans", SelectionList)
        for i, (_t, ty) in enumerate(self._spans):
            if tier_masks(self._tier, ty):
                sl.select(i)
            else:
                sl.deselect(i)
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
            {
                "spans": self._selected_spans(),
                "disposition": self._disposition,
                "tier_id": self._tier.id,
            }
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
