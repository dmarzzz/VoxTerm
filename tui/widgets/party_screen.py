"""Party mode picker — host a new party or join one by code.

Auto-join was removed for security: the session code (which the AES-256-GCM
party key is derived from) is no longer broadcast over mDNS, so joining requires
the code shared out-of-band. This modal is the "explicit picker": host a new
party (and get a code to share), or type a code to join a discovered party.

Dismisses with one of:
    {"action": "host"}                                   — host a new party
    {"action": "join", "code": str, "party_id": str|None} — join by code
    None                                                  — cancelled
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from network.crypto import validate_session_code


class PartyScreen(ModalScreen):
    """Host-or-join picker. ``parties`` is a list of discovered-party dicts:
    ``[{"party_id", "display_name", "party_color", "peer_count"}, ...]`` (no
    codes — those are out-of-band)."""

    DEFAULT_CSS = """
    PartyScreen { align: center middle; }
    #party-dialog {
        width: 60; height: auto; max-height: 22;
        border: heavy #00e5ff; border-title-color: #00ffcc; border-title-style: bold;
        background: #0a0e14; padding: 1 2;
    }
    #party-list { height: auto; max-height: 8; background: #0a0e14; color: #c0c0c0; }
    #party-list > .option-list--option-highlighted { background: #1a1a3a; color: #00ffcc; }
    #party-code-container { height: 3; margin-top: 1; }
    #party-code { width: 100%; background: #111822; color: #00ffcc; border: tall #003344; }
    #party-code:focus { border: tall #00e5ff; }
    #party-error { height: auto; color: #f7768e; margin-top: 1; }
    #party-hint { height: auto; color: #607080; margin-top: 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+h", "host", "Host"),
    ]

    def __init__(self, parties: list[dict] | None = None, provider=None):
        super().__init__()
        self._parties = parties or []
        # optional callable returning the current discovered parties; when given, the
        # picker re-polls it so a host that appears AFTER opening shows up live (instead
        # of funnelling the code-holder into hosting a divergent party — the split-brain).
        self._provider = provider

    def _status_text(self) -> str:
        if self._parties:
            return "  [#00e5ff]Nearby parties[/] — pick one, then enter its code:"
        return ("  [#607080]Looking for nearby parties…[/]  "
                "[#00e5ff]^H[/] to host your own.")

    def _party_options(self) -> list:
        return [
            Option(
                f"  {p.get('display_name', '?'):20s}  {p.get('peer_count', 0)} peer(s)",
                id=p["party_id"],
            )
            for p in self._parties
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="party-dialog") as dialog:
            dialog.border_title = "PARTY MODE"
            # The list is always present (hidden when empty) so a re-poll can fill it
            # in place without losing the code the user is typing.
            yield Static(self._status_text(), id="party-status", markup=True)
            yield OptionList(*self._party_options(), id="party-list")
            with Vertical(id="party-code-container"):
                yield Input(
                    placeholder="session code to join (e.g. bacon-horse-galaxy)",
                    id="party-code",
                )
            yield Static("", id="party-error", markup=True)
            yield Static(
                " [#607080]ENTER[/] join   [#607080]^H[/] host new   [#607080]ESC[/] cancel\n"
                " [#607080]the host shares the code out-of-band — it is never sent over the network[/]",
                id="party-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        self._sync_list()
        self.query_one("#party-code", Input).focus()
        if self._provider:
            self.set_interval(1.5, self._refresh)   # live-refresh so a late host appears

    def _sync_list(self) -> None:
        """Reflect self._parties into the OptionList + status, preserving the typed code."""
        ol = self.query_one("#party-list", OptionList)
        ol.clear_options()
        for opt in self._party_options():
            ol.add_option(opt)
        ol.display = bool(self._parties)
        if self._parties and ol.highlighted is None:
            ol.highlighted = 0
        self.query_one("#party-status", Static).update(self._status_text())

    def _refresh(self) -> None:
        if not self._provider:
            return
        try:
            fresh = self._provider() or []
        except Exception:
            return
        # rebuild only when the set of party_ids actually changes (don't clobber selection)
        if [p.get("party_id") for p in fresh] != [p.get("party_id") for p in self._parties]:
            self._parties = fresh
            self._sync_list()

    def _selected_party_id(self) -> str | None:
        if not self._parties:
            return None
        try:
            ol = self.query_one("#party-list", OptionList)
        except Exception:
            return None
        idx = ol.highlighted if ol.highlighted is not None else 0
        if 0 <= idx < len(self._parties):
            return self._parties[idx]["party_id"]
        return None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        code = (event.value or "").strip().lower()
        err = self.query_one("#party-error", Static)
        if not code:
            return
        if not self._parties:
            err.update(" Still looking for the host — keep this open, or ^H to host your own.")
            return
        if validate_session_code(code) is None:
            err.update(" That isn't a valid code (three words, e.g. bacon-horse-galaxy).")
            return
        self.dismiss({
            "action": "join",
            "code": code,
            "party_id": self._selected_party_id(),
        })

    def action_host(self) -> None:
        self.dismiss({"action": "host"})

    def action_cancel(self) -> None:
        self.dismiss(None)


class PartyCodeScreen(ModalScreen):
    """Shows the host's session code so it can be shared out-of-band."""

    DEFAULT_CSS = """
    PartyCodeScreen { align: center middle; }
    #code-dialog {
        width: 56; height: auto; border: heavy #00ffcc; border-title-color: #00ffcc;
        border-title-style: bold; background: #0a0e14; padding: 1 2;
    }
    #the-code { color: #00ffcc; text-style: bold; text-align: center; padding: 1 0; }
    #code-hint { color: #607080; }
    """

    BINDINGS = [Binding("escape", "close", "Close"), Binding("enter", "close", "Close")]

    def __init__(self, code: str):
        super().__init__()
        self._code = code

    def compose(self) -> ComposeResult:
        with Vertical(id="code-dialog") as dialog:
            dialog.border_title = "YOU ARE HOSTING"
            yield Static(f"\n    {self._code}\n", id="the-code")
            yield Static(
                " Share this code with people you want in the party (say it aloud /\n"
                " paste it). They join with it. It is never broadcast over the network.\n\n"
                " [#607080]ENTER / ESC to continue[/]",
                id="code-hint",
                markup=True,
            )

    def action_close(self) -> None:
        self.dismiss(None)
