"""Generic yes/no confirmation modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class ConfirmScreen(ModalScreen[bool]):
    """A reusable yes/no confirmation dialog.

    Returns True if confirmed, False otherwise (including ESC).
    """

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 56;
        height: auto;
        max-height: 14;
        border: heavy #ff6600;
        border-title-color: #ffaa00;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #confirm-message {
        color: #c0c0c0;
        margin-bottom: 1;
    }
    #confirm-list {
        height: auto;
        max-height: 4;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #confirm-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #confirm-hint {
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
        title: str,
        message: str,
        confirm_label: str = "Yes",
        cancel_label: str = "Cancel",
    ) -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog") as dialog:
            dialog.border_title = self._title
            yield Static(self._message, id="confirm-message", markup=True)
            yield OptionList(
                Option(f"  {self._confirm_label}", id="confirm"),
                Option(f"  {self._cancel_label}", id="cancel"),
                id="confirm-list",
            )
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="confirm-hint",
                markup=True,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)
