import time
from textual.widget import Widget
from textual.strip import Strip
from rich.text import Text
from rich.style import Style

from redaction.tiers import TIERS, resolve_tier


def _tier_meter(tier) -> str:
    filled = "▰" * (tier.rank + 1)
    empty = "▱" * (len(TIERS) - tier.rank - 1)
    return filled + empty


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
        self._disclosure_tier_id = "room"

    def set_recording(self, on: bool):
        self._recording = on
        if on:
            self._rec_start = time.time()
        self.refresh()

    def set_disclosure_tier(self, tier_id: str):
        self._disclosure_tier_id = resolve_tier(tier_id).id
        self.refresh()

    def _disclosure_badge(self) -> Text:
        tier = resolve_tier(self._disclosure_tier_id)
        badge = Text()
        badge.append(" disclosure: ", Style(color="#607080"))
        badge.append(_tier_meter(tier), Style(color=tier.color, bold=True))
        badge.append(f" {tier.label} ", Style(color=tier.color, bold=True))
        return badge

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if y != 0:
            return Strip.blank(width, Style())

        badge = self._disclosure_badge()

        if self._recording:
            elapsed = int(time.time() - self._rec_start)
            mins, secs = divmod(elapsed, 60)
            ts = f"{mins:02d}:{secs:02d}"

            line = Text()
            rec_style = Style(color="#ffffff", bgcolor="#cc0000", bold=True)
            bar_style = Style(color="#cc0000", bgcolor="#cc0000")
            line.append(f"  ● REC {ts} ", rec_style)
            # Fill the rest with the red bar
            remaining = max(0, width - line.cell_len - badge.cell_len)
            line.append("━" * remaining, bar_style)
            line.append_text(badge)
            line.truncate(width)
            return Strip(line.render(self.app.console))
        else:
            line = Text()
            line.append("  +++ ", Style(color="#00e5ff", bold=True))
            from config import VERSION
            line.append(f"VOXTERM v{VERSION}", Style(color="#00ffcc", bold=True))
            line.append(" // ", Style(color="#607080"))
            line.append("LOCAL VOICE TRANSCRIPTION ENGINE", Style(color="#00e5ff", bold=True))
            remaining = max(0, width - line.cell_len - badge.cell_len)
            line.append(" " * remaining)
            line.append_text(badge)
            line.truncate(width)
            return Strip(line.render(self.app.console))
