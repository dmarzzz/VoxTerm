"""Hivemind menu — discover transcript sinks on the LAN and opt-in to push.

Press `h` from the main TUI to open this. Discovery is always on in the
background; this screen is where the user actually says "yes, ship my
transcripts to this sink". The choice is persisted to ConfigStore so
subsequent launches re-enable automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from network.hivemind import HivemindClient, Sink


def _pubkey_short(pubkey: str) -> str:
    if not pubkey:
        return "(no pubkey)"
    if pubkey.startswith("ed25519:") and len(pubkey) > 20:
        return pubkey[:8] + ".." + pubkey[8 + 12:8 + 18]
    return pubkey[:14] + ".."


class HivemindScreen(ModalScreen):
    """Modal showing discovered sinks + per-sink push toggle."""

    DEFAULT_CSS = """
    HivemindScreen {
        align: center middle;
    }
    #hivemind-dialog {
        width: 78;
        height: auto;
        max-height: 24;
        border: heavy #5fa850;
        border-title-color: #b8e060;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #hivemind-status {
        height: auto;
        color: #b8e060;
        margin-bottom: 1;
    }
    #hivemind-sinks {
        height: auto;
        max-height: 12;
        background: #111822;
        border: tall #2a3a28;
    }
    #hivemind-sinks:focus {
        border: tall #5fa850;
    }
    #hivemind-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, hivemind_client: "HivemindClient | None"):
        super().__init__()
        self._client = hivemind_client

    # ── lifecycle ──────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Vertical(id="hivemind-dialog") as dialog:
            dialog.border_title = "HIVEMIND"
            yield Static(self._status_text(), id="hivemind-status", markup=True)
            yield OptionList(*self._sink_options(), id="hivemind-sinks")
            yield Static(
                " [#607080]ENTER[/] toggle push   "
                "[#607080]R[/] refresh   "
                "[#607080]ESC[/] close",
                id="hivemind-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        try:
            self.query_one("#hivemind-sinks", OptionList).focus()
        except Exception:
            pass

    # ── data ───────────────────────────────────────────────────────

    def _sinks(self) -> list["Sink"]:
        if self._client is None:
            return []
        try:
            return self._client.discovered_sinks()
        except Exception:
            return []

    def _status_text(self) -> str:
        if self._client is None:
            return "[#ff8866]hivemind disabled[/]  (relaunch without --hivemind off)"
        sinks = self._sinks()
        enabled = self._client.push_enabled
        pinned = self._client.pinned_sink_pubkey
        if not sinks:
            return (
                "[#b8e060]searching for sinks on the LAN...[/]\n"
                "[#607080]mDNS browser is running. New sinks will appear here.[/]"
            )
        if enabled:
            active = next((s for s in sinks if s.pubkey == pinned), None)
            if active is not None:
                return (
                    f"[#5fa850]pushing to:[/] "
                    f"[#b8e060]{active.transcripts_url}[/]\n"
                    f"[#607080]{len(sinks)} sink(s) discovered. "
                    f"ENTER on another to switch.[/]"
                )
            return (
                "[#ff8866]push enabled but pinned sink not present[/]  "
                f"[#607080]({_pubkey_short(pinned)})[/]\n"
                "[#607080]Pick a sink below to re-pin.[/]"
            )
        return (
            f"[#b8e060]{len(sinks)} sink(s) discovered.[/] "
            "[#607080]Push is off; ENTER on a sink to enable.[/]"
        )

    def _sink_options(self) -> list[Option]:
        if self._client is None:
            return [Option("(hivemind disabled — close this screen)", id="_disabled", disabled=True)]
        sinks = self._sinks()
        if not sinks:
            return [Option("(no sinks yet — press R to refresh)", id="_empty", disabled=True)]
        enabled = self._client.push_enabled
        pinned = self._client.pinned_sink_pubkey
        opts: list[Option] = []
        for sink in sinks:
            checked = enabled and sink.pubkey == pinned
            mark = "[#5fa850][x][/]" if checked else "[#607080][ ][/]"
            label = (
                f"{mark}  {sink.transcripts_url}  "
                f"[#607080]{_pubkey_short(sink.pubkey)} · "
                f"node={sink.node or '?'}[/]"
            )
            opts.append(Option(label, id=sink.pubkey or sink.transcripts_url))
        return opts

    # ── refresh path ───────────────────────────────────────────────

    def _refresh_view(self) -> None:
        try:
            self.query_one("#hivemind-status", Static).update(self._status_text())
            ol = self.query_one("#hivemind-sinks", OptionList)
            ol.clear_options()
            for opt in self._sink_options():
                ol.add_option(opt)
        except Exception:
            pass

    # ── actions ────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        self._refresh_view()

    def action_close(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._client is None:
            return
        opt_id = event.option.id
        if opt_id in (None, "_disabled", "_empty"):
            return
        # opt_id is the sink's pubkey (or its transcripts_url if no pubkey).
        sinks = self._sinks()
        target = next(
            (s for s in sinks if (s.pubkey or s.transcripts_url) == opt_id),
            None,
        )
        if target is None:
            return
        # Toggle: if this exact sink is the current pinned + push is on,
        # disable. Otherwise enable (and pin to this sink).
        if self._client.push_enabled and self._client.pinned_sink_pubkey == target.pubkey:
            self._client.disable_push()
            self._announce(f"hivemind: disconnected from {target.transcripts_url}")
        else:
            self._client.enable_push(sink_pubkey=target.pubkey)
            self._announce(f"hivemind: connected to {target.transcripts_url}")
        self._refresh_view()

    def _announce(self, msg: str) -> None:
        """Surface a hivemind state change in the TUI itself.

        Two channels:
          - ``self.app.notify`` for a toast that floats above whatever
            the user is looking at right now (visible even with the
            HivemindScreen modal still open).
          - A SYS system_message in the transcript panel so a user
            scrolling back later still sees the connection event.
        """
        try:
            self.app.notify(msg, severity="information", timeout=6)
        except Exception:
            pass
        try:
            from tui.widgets.transcript import Log, TranscriptPanel
            tp = self.app.query_one(TranscriptPanel)
            tp.system_message(msg, Log.SYS)
        except Exception:
            # Screen can be used in tests / without a TranscriptPanel
            # parent (e.g., the dictation app); best-effort.
            pass
