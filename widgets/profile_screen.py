"""Speaker profile management screen — view, edit, and delete voice profiles."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, OptionList, Input, Label
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen

from speakers.models import SpeakerMeta


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


def _format_date(iso_str: str) -> str:
    """Format ISO date to short display."""
    if not iso_str:
        return "—"
    return iso_str[:10]  # YYYY-MM-DD


class ProfileEditScreen(ModalScreen):
    """Sub-modal for editing a profile name."""

    DEFAULT_CSS = """
    ProfileEditScreen {
        align: center middle;
    }
    #edit-dialog {
        width: 44;
        height: auto;
        max-height: 10;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #edit-input {
        width: 100%;
        background: #111822;
        color: #00ffcc;
        border: tall #003344;
        margin-top: 1;
    }
    #edit-input:focus {
        border: tall #00e5ff;
    }
    #edit-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, profile_id: str, current_name: str):
        super().__init__()
        self._profile_id = profile_id
        self._current_name = current_name

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog") as dialog:
            dialog.border_title = "EDIT PROFILE"
            yield Label(f"  Rename speaker profile:", id="edit-label")
            yield Input(
                value=self._current_name,
                placeholder="Type new name...",
                id="edit-input",
            )
            yield Static(
                " [#607080]ENTER[/] save  [#607080]ESC[/] cancel",
                id="edit-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        self.query_one("#edit-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        if name:
            self.dismiss({"profile_id": self._profile_id, "name": name})

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProfileDeleteScreen(ModalScreen):
    """Confirmation dialog for deleting a profile."""

    DEFAULT_CSS = """
    ProfileDeleteScreen {
        align: center middle;
    }
    #delete-dialog {
        width: 50;
        height: auto;
        max-height: 10;
        border: heavy #ff4444;
        border-title-color: #ff6666;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #delete-options {
        height: auto;
        max-height: 4;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #delete-options > .option-list--option-highlighted {
        background: #331111;
        color: #ff4444;
    }
    #delete-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, profile_id: str, name: str):
        super().__init__()
        self._profile_id = profile_id
        self._name = name

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-dialog") as dialog:
            dialog.border_title = f"DELETE {self._name.upper()}?"
            yield Static(
                f"  [#ff6644]Permanently delete voice profile for[/]\n"
                f"  [bold #ff4444]{self._name}[/][#ff6644]?[/]\n"
                f"  [#607080]This cannot be undone.[/]",
                markup=True,
            )
            yield OptionList(
                Option("  Yes, delete", id="confirm"),
                Option("  Cancel", id="cancel"),
                id="delete-options",
            )
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="delete-hint",
                markup=True,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "confirm":
            self.dismiss(self._profile_id)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SpeakerProfileScreen(ModalScreen):
    """Modal for managing persistent speaker profiles.

    Shows all stored profiles with stats. Allows rename and delete.

    Dismisses with an action dict or None:
        {"action": "rename", "profile_id": str, "name": str}
        {"action": "delete", "profile_id": str}
        None (cancelled)
    """

    DEFAULT_CSS = """
    SpeakerProfileScreen {
        align: center middle;
    }
    #profile-dialog {
        width: 64;
        height: auto;
        max-height: 26;
        border: heavy #aa88ff;
        border-title-color: #cc99ff;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #profile-list {
        height: auto;
        max-height: 12;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #profile-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #profile-detail {
        height: auto;
        max-height: 5;
        margin-top: 1;
        padding: 0 1;
        color: #607080;
        border: round #1a2233;
        background: #0d1117;
    }
    #profile-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    #profile-empty {
        height: 3;
        color: #607080;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("enter", "edit_profile", "Edit"),
        Binding("ctrl+d", "delete_profile", "Delete"),
        Binding("ctrl+x", "delete_all_data", "Delete All", show=False),
    ]

    def __init__(self, profiles: list[SpeakerMeta]):
        super().__init__()
        self._profiles = profiles
        self._selected_idx = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="profile-dialog") as dialog:
            dialog.border_title = "SPEAKER PROFILES"

            if not self._profiles:
                yield Static(
                    "  [#607080]No speaker profiles yet.\n"
                    "  Tag speakers with [bold #00e5ff]T[/] during recording.[/]",
                    id="profile-empty",
                    markup=True,
                )
            else:
                options = []
                for p in self._profiles:
                    samples = p.confirmed_count + p.auto_assigned_count
                    dur = _format_duration(p.total_duration_sec)
                    seen = _format_date(p.last_seen_at)
                    label = f"  {p.name:18s}  {samples:4d} samples  {dur:>6s}  {seen}"
                    options.append(Option(label, id=p.id))

                yield OptionList(*options, id="profile-list")

                # Detail pane — updated on highlight
                yield Static("", id="profile-detail", markup=True)

            yield Static(
                " [#607080]ENTER[/] edit  "
                "[#607080]^D[/] delete  "
                "[#607080]^X[/] delete all  "
                "[#607080]ESC[/] close",
                id="profile-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        if self._profiles:
            self._update_detail(0)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option and event.option.id:
            idx = next(
                (i for i, p in enumerate(self._profiles) if p.id == event.option.id),
                0,
            )
            self._selected_idx = idx
            self._update_detail(idx)

    def _update_detail(self, idx: int) -> None:
        if idx >= len(self._profiles):
            return
        p = self._profiles[idx]
        samples = p.confirmed_count + p.auto_assigned_count
        dur = _format_duration(p.total_duration_sec)
        created = _format_date(p.created_at)
        seen = _format_date(p.last_seen_at)

        detail = (
            f"  [bold {p.color}]{p.name}[/]\n"
            f"  [#607080]Samples:[/] {samples} ({p.confirmed_count} confirmed)"
            f"    [#607080]Duration:[/] {dur}\n"
            f"  [#607080]Created:[/] {created}"
            f"    [#607080]Last seen:[/] {seen}"
            f"    [#607080]Quality:[/] {p.quality_score:.2f}"
        )
        try:
            self.query_one("#profile-detail", Static).update(detail)
        except Exception:
            pass

    def action_edit_profile(self) -> None:
        if not self._profiles:
            return
        p = self._profiles[self._selected_idx]

        def on_edit_result(result):
            if result:
                self.dismiss({"action": "rename", **result})

        self.push_screen(
            ProfileEditScreen(p.id, p.name),
            on_edit_result,
        )

    def action_delete_profile(self) -> None:
        if not self._profiles:
            return
        p = self._profiles[self._selected_idx]

        def on_delete_result(profile_id):
            if profile_id:
                self.dismiss({"action": "delete", "profile_id": profile_id})

        self.push_screen(
            ProfileDeleteScreen(p.id, p.name),
            on_delete_result,
        )

    def action_delete_all_data(self) -> None:
        """Confirm and delete ALL voice data."""
        def on_confirm(profile_id):
            if profile_id == "__all__":
                self.dismiss({"action": "delete_all"})

        self.push_screen(
            ProfileDeleteScreen("__all__", "ALL VOICE DATA"),
            on_confirm,
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
