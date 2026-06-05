"""Headless runtime tests for the party picker UI (Textual Pilot).

Unit tests cover the PartyManager logic; these actually MOUNT the screens and
drive them, so a Textual API misuse (the screen importing fine but breaking at
runtime) is caught. Uses App.run_test() via asyncio.run inside sync tests, so no
pytest-asyncio config is needed.
"""

from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import Input, OptionList, Static

from tui.widgets.party_screen import PartyCodeScreen, PartyScreen

_PARTIES = [{"party_id": "pid-1", "display_name": "alice", "party_color": "#0fc", "peer_count": 2}]
_SENTINEL = "UNSET"


class _Harness(App):
    def __init__(self, screen):
        super().__init__()
        self._screen = screen
        self.result = _SENTINEL

    def on_mount(self) -> None:
        self.push_screen(self._screen, lambda r: setattr(self, "result", r))


def _drive(screen, steps) -> object:
    """Mount `screen`, run async `steps(app, pilot)`, return the dismiss result."""
    app = _Harness(screen)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await steps(app, pilot)
            await pilot.pause()
        return app.result

    return asyncio.run(go())


def test_join_dismisses_with_code_and_selected_party():
    async def steps(app, pilot):
        app.screen.query_one("#party-code", Input).value = "bacon-horse-galaxy"
        await pilot.press("enter")

    assert _drive(PartyScreen(_PARTIES), steps) == {
        "action": "join", "code": "bacon-horse-galaxy", "party_id": "pid-1",
    }


def test_host_binding_dismisses_with_host_action():
    async def steps(app, pilot):
        await pilot.press("ctrl+h")

    assert _drive(PartyScreen(_PARTIES), steps) == {"action": "host"}


def test_escape_cancels():
    async def steps(app, pilot):
        await pilot.press("escape")

    assert _drive(PartyScreen(_PARTIES), steps) is None


def test_invalid_code_is_rejected_not_dismissed():
    async def steps(app, pilot):
        app.screen.query_one("#party-code", Input).value = "not a real code"
        await pilot.press("enter")

    app_result = _drive(PartyScreen(_PARTIES), steps)
    assert app_result == _SENTINEL  # never dismissed — a dead key is not derived


def test_no_parties_join_is_refused():
    async def steps(app, pilot):
        app.screen.query_one("#party-code", Input).value = "bacon-horse-galaxy"
        await pilot.press("enter")

    # nothing to join -> not dismissed; the screen tells the user to host
    assert _drive(PartyScreen([]), steps) == _SENTINEL


def test_picker_repolls_so_a_late_host_becomes_joinable():
    # A host that comes up AFTER the picker opens must appear live (instead of the
    # old behaviour where an empty list funnelled the code-holder into hosting a
    # divergent party — the split-brain the security fix's review flagged).
    seq = [[], _PARTIES]

    def provider():
        return seq.pop(0) if seq else _PARTIES

    async def steps(app, pilot):
        app.screen._refresh()                       # tick 1: still nothing discovered
        await pilot.pause()
        assert app.screen.query_one("#party-list", OptionList).display is False
        app.screen._refresh()                       # tick 2: a host appears
        await pilot.pause()
        ol = app.screen.query_one("#party-list", OptionList)
        assert ol.display is True and ol.option_count == 1
        app.screen.query_one("#party-code", Input).value = "bacon-horse-galaxy"
        await pilot.press("enter")

    assert _drive(PartyScreen([], provider=provider), steps) == {
        "action": "join", "code": "bacon-horse-galaxy", "party_id": "pid-1",
    }


def test_code_display_screen_mounts_and_closes():
    async def steps(app, pilot):
        # the host's code-display screen mounts (so the host can read it) and
        # closes on enter
        assert app.screen.query_one("#the-code", Static) is not None
        await pilot.press("enter")

    assert _drive(PartyCodeScreen("bacon-horse-galaxy"), steps) is None
