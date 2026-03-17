from datetime import datetime
from textual.widgets import RichLog
from rich.text import Text
from rich.style import Style


# Default speaker colour palette (matches DiarizationEngine)
_SPEAKER_COLORS = [
    "#00ffcc", "#ff44aa", "#44ff44", "#ffaa00",
    "#aa88ff", "#ff6644", "#44ddff", "#ffff44",
]


class TranscriptPanel(RichLog):
    """Cyberpunk-styled live transcription panel with speaker attribution."""

    DEFAULT_CSS = """
    TranscriptPanel {
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        border-title-align: left;
        background: #0d1117;
        margin: 0 1;
        min-height: 10;
        overflow-x: hidden;
        scrollbar-size-vertical: 0;
    }
    """

    def __init__(self):
        super().__init__(wrap=True, markup=True, auto_scroll=True)
        self.border_title = "TRANSCRIPT // LIVE"
        # (timestamp, type, content, speaker_label, speaker_id)
        self._entries: list[tuple[str, str, str, str, int]] = []

    def system_message(self, msg: str):
        """Add a system message."""
        text = Text()
        text.append("SYSTEM [SYS]  ", Style(color="#ff6600", bold=True))
        text.append(msg, Style(color="#607080"))
        self.write(text)

    def add_transcript(self, content: str, speaker: str = "", speaker_id: int = 0):
        """Add transcribed text with optional speaker attribution."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, "transcript", content, speaker, speaker_id))

        text = Text()
        text.append(f"[{timestamp}]  ", Style(color="#004455"))

        if speaker:
            color = _SPEAKER_COLORS[(speaker_id - 1) % len(_SPEAKER_COLORS)]
            text.append(f"{speaker}", Style(color=color, bold=True))
            text.append("  ", Style())
        else:
            text.append("> ", Style(color="#00e5ff", bold=True))

        text.append(content, Style(color="#c0c0c0"))
        self.write(text)

    def get_entries(self) -> list[tuple]:
        """Return all transcript entries."""
        return list(self._entries)

    def get_plain_text(self) -> str:
        """Return transcript as plain text."""
        lines = []
        for ts, _, content, speaker, *_ in self._entries:
            prefix = f"[{speaker}] " if speaker else ""
            lines.append(f"[{ts}] {prefix}{content}")
        return "\n".join(lines)

    def get_markdown(self, model_name: str = "unknown") -> str:
        """Return transcript as markdown."""
        now = datetime.now()
        lines = [
            f"# VOXTERM Transcript",
            f"",
            f"- **Date:** {now.strftime('%Y-%m-%d')}",
            f"- **Time:** {now.strftime('%H:%M:%S')}",
            f"- **Model:** whisper-{model_name}",
            f"",
            f"---",
            f"",
        ]
        for ts, _, content, speaker, *_ in self._entries:
            speaker_tag = f" **{speaker}:**" if speaker else ""
            lines.append(f"**[{ts}]**{speaker_tag} {content}")
            lines.append("")
        return "\n".join(lines)
