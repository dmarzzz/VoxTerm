from textual.widget import Widget
from textual.strip import Strip
from rich.text import Text
from rich.style import Style


class CyberHeader(Widget):
    """Cyberpunk-styled header with app title and status."""

    DEFAULT_CSS = """
    CyberHeader {
        height: 3;
        background: #0a0e14;
        border-bottom: heavy #00e5ff;
    }
    """

    def __init__(self):
        super().__init__()
        self._status = "IDLE"
        self._model = "whisper-small"
        self._audio = "inactive"

    def update_status(self, status: str = None, model: str = None, audio: str = None):
        if status is not None:
            self._status = status
        if model is not None:
            self._model = model
        if audio is not None:
            self._audio = audio
        self.refresh()

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        cyan = Style(color="#00e5ff", bold=True)
        accent = Style(color="#00ffcc")
        dim = Style(color="#607080")

        if y == 0:
            line = Text()
            line.append("  +++ ", cyan)
            line.append("VOXTERM v1.0", Style(color="#00ffcc", bold=True))
            line.append(" // ", dim)
            line.append("LOCAL VOICE TRANSCRIPTION ENGINE", cyan)
            line.pad(width)
            return Strip(line.render(self.app.console))
        elif y == 1:
            line = Text()
            line.append("  status: ", dim)
            status_color = "#00ff88" if self._status == "RECORDING" else "#ff6600" if self._status == "PAUSED" else "#607080"
            line.append(self._status, Style(color=status_color, bold=True))
            line.append(" // ", dim)
            line.append("model: ", dim)
            line.append(self._model, accent)
            line.append(" // ", dim)
            line.append("audio: ", dim)
            audio_color = "#00ff88" if self._audio == "active" else "#607080"
            line.append(self._audio, Style(color=audio_color))
            line.pad(width)
            return Strip(line.render(self.app.console))
        else:
            return Strip.blank(width)
