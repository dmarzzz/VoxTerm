import time
from textual.widget import Widget
from textual.strip import Strip
from rich.text import Text
from rich.style import Style


class CyberHeader(Widget):
    """Header that transforms into a recording indicator when active."""

    DEFAULT_CSS = """
    CyberHeader {
        height: 1;
        background: #0a0e14;
        layer: base;
    }
    """

    def __init__(self):
        super().__init__()
        self._recording = False
        self._rec_start: float = 0.0
        self._blink_timer = None

    def set_recording(self, on: bool):
        self._recording = on
        if on:
            self._rec_start = time.time()
            if self._blink_timer is None:
                # Pulse ~2 Hz so the indicator can't be tuned out like a static red bar
                # (consent/awareness — testers kept forgetting the mic was live, #93).
                self._blink_timer = self.set_interval(0.5, self.refresh)
        elif self._blink_timer is not None:
            self._blink_timer.stop()
            self._blink_timer = None
        self.refresh()

    def on_unmount(self) -> None:
        # Defense-in-depth: tear the pulse timer down if the header is ever removed
        # mid-recording. It's long-lived today, but don't leave that an invariant.
        if self._blink_timer is not None:
            self._blink_timer.stop()
            self._blink_timer = None

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if y != 0:
            return Strip.blank(width)

        if self._recording:
            elapsed = int(time.time() - self._rec_start)
            mins, secs = divmod(elapsed, 60)
            ts = f"{mins:02d}:{secs:02d}"

            # Pulse between bright and dim red (~2 Hz) + blink the dot, so the
            # "mic is live" signal stays attention-grabbing rather than fading
            # into the chrome.
            bright = int(time.time() * 2) % 2 == 0
            rec_bg = "#ee0000" if bright else "#8a0000"
            dot = "●" if bright else "○"
            line = Text()
            rec_style = Style(color="#ffffff", bgcolor=rec_bg, bold=True)
            bar_style = Style(color=rec_bg, bgcolor=rec_bg)
            line.append(f"  {dot} REC {ts} ", rec_style)
            # Fill the rest with the (pulsing) red bar
            remaining = max(0, width - line.cell_len)
            line.append("━" * remaining, bar_style)
            line.truncate(width)
            return Strip(line.render(self.app.console))
        else:
            line = Text()
            line.append("  +++ ", Style(color="#00e5ff", bold=True))
            from config import VERSION
            line.append(f"VOXTERM v{VERSION}", Style(color="#00ffcc", bold=True))
            line.append(" // ", Style(color="#607080"))
            line.append("LOCAL VOICE TRANSCRIPTION ENGINE", Style(color="#00e5ff", bold=True))
            line.pad(width)
            return Strip(line.render(self.app.console))
