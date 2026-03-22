"""P2P session screens — create, join, and browse peers on the LAN.

These modals guide the user through the P2P setup with clear explanations
of what's happening at each step.  The goal is zero-confusion: the user
should understand the protocol just from reading the screens.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


# ── Session Create Screen ─────────────────────────────────────

class SessionCreateScreen(ModalScreen):
    """Modal to create a new P2P session."""

    DEFAULT_CSS = """
    SessionCreateScreen {
        align: center middle;
    }
    #session-create-dialog {
        width: 62;
        height: auto;
        max-height: 28;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #create-name-input {
        margin: 0 0 1 0;
    }
    #session-code-display {
        text-align: center;
        color: #00ffcc;
        text-style: bold;
        padding: 1 0;
        background: #0d1520;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, session_code: str) -> None:
        super().__init__()
        self._session_code = session_code

    def compose(self) -> ComposeResult:
        with Vertical(id="session-create-dialog") as dialog:
            dialog.border_title = "CREATE P2P SESSION"
            yield Static(
                "[#c0c0c0]Start a collaborative transcription session.\n"
                "Everyone in the room runs VoxTerm on their own laptop.\n"
                "Each device captures its closest speaker — the combined\n"
                "result is better than any single mic alone.[/]",
                markup=True,
            )
            yield Static("")
            yield Static("[b #00e5ff]Your name[/]  [dim](how others will see you)[/]", markup=True)
            yield Input(placeholder="e.g. halcyon", id="create-name-input")
            yield Static("[b #00e5ff]Session code[/]  [dim](read this aloud to others)[/]", markup=True)
            yield Static(
                f"\n    {self._session_code}\n",
                id="session-code-display",
            )
            yield Static(
                "[#c0c0c0]Others press [b #00e5ff]J[/b #00e5ff] and type this code to join.\n"
                "The code is the encryption key — only people who\n"
                "hear it in the room can connect. Nothing leaves\n"
                "your local network.[/]",
                markup=True,
            )
            yield Static("")
            yield Static(
                "[dim]ENTER to start  ·  ESC to cancel[/]",
                markup=True,
            )

    def on_mount(self) -> None:
        self.query_one("#create-name-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        if name:
            self.dismiss({
                "action": "create",
                "display_name": name,
                "session_code": self._session_code,
            })

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Session Join Screen ───────────────────────────────────────

class SessionJoinScreen(ModalScreen):
    """Modal to join an existing P2P session by entering a session code."""

    DEFAULT_CSS = """
    SessionJoinScreen {
        align: center middle;
    }
    #session-join-dialog {
        width: 62;
        height: auto;
        max-height: 26;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #join-name-input {
        margin: 0 0 1 0;
    }
    #join-code-input {
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="session-join-dialog") as dialog:
            dialog.border_title = "JOIN P2P SESSION"
            yield Static(
                "[#c0c0c0]Join a session that someone nearby has created.\n"
                "Ask them for the session code — it's on their screen.[/]",
                markup=True,
            )
            yield Static("")
            yield Static("[b #00e5ff]Your name[/]  [dim](how others will see you)[/]", markup=True)
            yield Input(placeholder="e.g. bob", id="join-name-input")
            yield Static("[b #00e5ff]Session code[/]  [dim](from the session creator)[/]", markup=True)
            yield Input(placeholder="e.g. VOXJ-7K3M", id="join-code-input")
            yield Static(
                "[#c0c0c0]Your device will scan the local network for the\n"
                "session creator. The code encrypts the connection —\n"
                "a wrong code simply won't connect.[/]",
                markup=True,
            )
            yield Static("")
            yield Static(
                "[dim]ENTER to join  ·  ESC to cancel[/]",
                markup=True,
            )

    def on_mount(self) -> None:
        self.query_one("#join-name-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "join-name-input":
            self.query_one("#join-code-input", Input).focus()
            return
        name = self.query_one("#join-name-input", Input).value.strip()
        code = self.query_one("#join-code-input", Input).value.strip().upper()
        if name and code:
            self.dismiss({
                "action": "join",
                "display_name": name,
                "session_code": code,
            })

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Peer Browser Screen ──────────────────────────────────────

class PeerBrowserScreen(ModalScreen):
    """Modal showing VoxTerm peers visible on the LAN."""

    DEFAULT_CSS = """
    PeerBrowserScreen {
        align: center middle;
    }
    #peer-browser-dialog {
        width: 62;
        height: auto;
        max-height: 24;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #peer-list {
        height: auto;
        max-height: 8;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #peer-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("n", "new_session", "New Session"),
        Binding("j", "join_session", "Join Session"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, peers: list | None = None) -> None:
        super().__init__()
        self._peers = peers or []

    def compose(self) -> ComposeResult:
        with Vertical(id="peer-browser-dialog") as dialog:
            dialog.border_title = "VOXTERM PEERS ON NETWORK"
            if self._peers:
                options = []
                for p in self._peers:
                    if p.in_session:
                        status = "[#00ff88]in session[/]"
                    else:
                        status = "[dim]idle[/]"
                    options.append(Option(
                        f"  {p.display_name:<16} {p.ip:<16} {status}",
                        id=p.node_id,
                    ))
                yield OptionList(*options, id="peer-list")
            else:
                yield Static(
                    "\n[dim]  No other VoxTerm instances found on this network.\n"
                    "  Make sure the other device is on the same WiFi.[/]\n",
                    markup=True,
                )
            yield Static("")
            yield Static(
                "[dim][N] New session  [J] Join  [R] Refresh  [ESC] Close[/]",
                markup=True,
            )

    def action_new_session(self) -> None:
        self.dismiss({"action": "new_session"})

    def action_join_session(self) -> None:
        self.dismiss({"action": "join_session"})

    def action_refresh(self) -> None:
        self.dismiss({"action": "refresh"})

    def action_cancel(self) -> None:
        self.dismiss(None)
