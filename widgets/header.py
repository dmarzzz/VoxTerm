from textual.widget import Widget
from textual.strip import Strip
from rich.text import Text
from rich.style import Style


class CyberHeader(Widget):
    """Cyberpunk-styled header — brand line only."""

    DEFAULT_CSS = """
    CyberHeader {
        height: 1;
        background: #0a0e14;
    }
    """

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if y == 0:
            line = Text()
            line.append("  +++ ", Style(color="#00e5ff", bold=True))
            line.append("VOXTERM v1.0", Style(color="#00ffcc", bold=True))
            line.append(" // ", Style(color="#607080"))
            line.append("LOCAL VOICE TRANSCRIPTION ENGINE", Style(color="#00e5ff", bold=True))
            line.pad(width)
            return Strip(line.render(self.app.console))
        return Strip.blank(width)
