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
        self._hivemind_active = False
        self._hivemind_label = ""

    def set_recording(self, on: bool):
        self._recording = on
        if on:
            self._rec_start = time.time()
        self.refresh()

    def set_hivemind(self, active: bool, label: str = "") -> None:
        """Show whether off-device Hivemind transcript push is enabled."""
        self._hivemind_active = bool(active)
        self._hivemind_label = (label or "").strip()
        self.refresh()

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if y != 0:
            return Strip.blank(width)

        if self._recording:
            elapsed = int(time.time() - self._rec_start)
            mins, secs = divmod(elapsed, 60)
            ts = f"{mins:02d}:{secs:02d}"

            line = Text()
            rec_style = Style(color="#ffffff", bgcolor="#cc0000", bold=True)
            bar_style = Style(color="#cc0000", bgcolor="#cc0000")
            line.append(f"  ● REC {ts} ", rec_style)
            if self._hivemind_active:
                line.append(" ", bar_style)
                line.append(" HIVE ON ", Style(color="#0a0e14", bgcolor="#00ffcc", bold=True))
            # Fill the rest with the red bar
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
            if self._hivemind_active:
                line.append("  HIVE", Style(color="#00ff88", bold=True))
                if self._hivemind_label:
                    line.append(f" {self._hivemind_label}", Style(color="#607080"))
            line.pad(width)
            return Strip(line.render(self.app.console))
