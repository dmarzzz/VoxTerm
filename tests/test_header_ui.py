"""Headless runtime test for the pulsing recording indicator (#93, Textual Pilot)."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from tui.widgets.header import CyberHeader


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield CyberHeader()


def _run(steps) -> None:
    app = _Harness()

    async def go():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await steps(app, pilot)
            await pilot.pause()

    asyncio.run(go())


def test_recording_indicator_pulses_then_stops():
    async def steps(app, pilot):
        hdr = app.query_one(CyberHeader)
        assert hdr._blink_timer is None                  # idle: no pulse timer

        hdr.set_recording(True)
        await pilot.pause()
        assert hdr._recording is True
        assert hdr._blink_timer is not None              # pulse timer running while live
        assert "REC" in hdr.render_line(0).text          # renders the recording label, no crash

        hdr.set_recording(False)
        await pilot.pause()
        assert hdr._blink_timer is None                  # timer torn down when not recording
        assert "VOXTERM" in hdr.render_line(0).text      # back to the idle header
