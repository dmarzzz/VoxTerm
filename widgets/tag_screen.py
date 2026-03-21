"""Speaker tagging modal — name session speakers via keyboard-driven UI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, OptionList, Input
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList


class MergeSpeakerScreen(ModalScreen):
    """Sub-modal for merging one session speaker into another."""

    DEFAULT_CSS = """
    MergeSpeakerScreen {
        align: center middle;
    }
    #merge-dialog {
        width: 52;
        height: auto;
        max-height: 16;
        border: heavy #ffaa00;
        border-title-color: #ffcc44;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #merge-list {
        height: auto;
        max-height: 8;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #merge-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #merge-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, source_id: int, source_name: str, targets: list[dict]):
        super().__init__()
        self._source_id = source_id
        self._source_name = source_name
        self._targets = targets

    def compose(self) -> ComposeResult:
        with Vertical(id="merge-dialog") as dialog:
            dialog.border_title = f"MERGE {self._source_name.upper()}"
            yield Static(
                f"  [#ffaa00]Merge[/] [bold #ffcc44]{self._source_name}[/] "
                f"[#ffaa00]into:[/]",
                markup=True,
            )
            options = []
            for t in self._targets:
                label = f"  {t['name']:16s}  {t['segments']:3d} segs"
                options.append(Option(label, id=str(t["id"])))
            yield OptionList(*options, id="merge-list")
            yield Static(
                " [#607080]ENTER[/] merge  [#607080]ESC[/] cancel",
                id="merge-hint",
                markup=True,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option and event.option.id:
            target_id = int(event.option.id)
            self.dismiss({
                "source_id": self._source_id,
                "target_id": target_id,
            })

    def action_cancel(self) -> None:
        self.dismiss(None)


class SpeakerTagScreen(ModalScreen):
    """Modal for naming/renaming speakers detected in the current session.

    Expects `speakers` as a list of dicts:
        [{"id": 1, "name": "Speaker 1", "color": "#00ffcc", "segments": 5, "tagged": False}, ...]

    `known_names` is a list of previously used names for autocomplete suggestions.

    Dismisses with a dict or None:
        {"speaker_id": int, "name": str}              — tag/rename
        {"merge_source": int, "merge_target": int}     — merge
        None                                            — cancelled
    """

    DEFAULT_CSS = """
    SpeakerTagScreen {
        align: center middle;
    }
    #tag-dialog {
        width: 56;
        height: auto;
        max-height: 22;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #tag-list {
        height: auto;
        max-height: 10;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #tag-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #tag-input-container {
        height: 3;
        margin-top: 1;
        padding: 0;
    }
    #tag-input {
        width: 100%;
        background: #111822;
        color: #00ffcc;
        border: tall #003344;
    }
    #tag-input:focus {
        border: tall #00e5ff;
    }
    #tag-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+m", "merge_speaker", "Merge", show=False),
    ]

    def __init__(
        self,
        speakers: list[dict],
        known_names: list[str] | None = None,
    ):
        super().__init__()
        self._speakers = speakers
        self._known_names = known_names or []
        self._selected_id: int | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="tag-dialog") as dialog:
            dialog.border_title = "TAG SPEAKERS"

            options = []
            for spk in self._speakers:
                sid = spk["id"]
                name = spk["name"]
                segs = spk["segments"]
                tagged = spk.get("tagged", False)
                status = "tagged" if tagged else "unknown"

                label = f"  {sid}  {name:16s}  {segs:3d} segs  {status}"
                options.append(Option(label, id=str(sid)))

            yield OptionList(*options, id="tag-list")

            with Vertical(id="tag-input-container"):
                suggester = SuggestFromList(self._known_names) if self._known_names else None
                yield Input(
                    placeholder="Type a name...",
                    id="tag-input",
                    suggester=suggester,
                )

            yield Static(
                " [#607080]ENTER[/] save  "
                "[#607080]^M[/] merge  "
                "[#607080]ESC[/] close",
                id="tag-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#tag-list", OptionList)
        # Highlight the first untagged speaker, or the first speaker
        for idx, spk in enumerate(self._speakers):
            if not spk.get("tagged", False):
                option_list.highlighted = idx
                break

        # Pre-fill input with current speaker's name
        if self._speakers:
            self._select_speaker(0)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option and event.option.id:
            idx = next(
                (i for i, s in enumerate(self._speakers) if str(s["id"]) == event.option.id),
                0,
            )
            self._select_speaker(idx)

    def _select_speaker(self, idx: int) -> None:
        if idx < len(self._speakers):
            spk = self._speakers[idx]
            self._selected_id = spk["id"]
            inp = self.query_one("#tag-input", Input)
            inp.value = spk["name"]
            inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """User pressed Enter in the name input."""
        name = event.value.strip()
        if name and self._selected_id is not None:
            self.dismiss({"speaker_id": self._selected_id, "name": name})

    def action_merge_speaker(self) -> None:
        """Open merge sub-modal for the selected speaker."""
        if self._selected_id is None or len(self._speakers) < 2:
            return

        source = next(
            (s for s in self._speakers if s["id"] == self._selected_id), None
        )
        if not source:
            return

        # Targets = all other speakers
        targets = [s for s in self._speakers if s["id"] != self._selected_id]

        def on_merge_result(result):
            if result:
                self.dismiss({
                    "merge_source": result["source_id"],
                    "merge_target": result["target_id"],
                })

        self.push_screen(
            MergeSpeakerScreen(self._selected_id, source["name"], targets),
            on_merge_result,
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
