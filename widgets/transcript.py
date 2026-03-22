from __future__ import annotations

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
        # (timestamp, type, content, speaker_label, speaker_id, confidence)
        # confidence: "" = manual/default, "high" = auto-recognized,
        #             "medium" = suggested, "new" = unknown new speaker
        self._entries: list[tuple[str, str, str, str, int, str]] = []
        # Per-speaker color overrides (from persistent profiles)
        self._color_overrides: dict[int, str] = {}
        # Per-speaker confidence scores for display
        self._speaker_confidence: dict[int, tuple[str, float]] = {}  # sid → (tier, score)

    def system_message(self, msg: str):
        """Add a system message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, "system", msg, "", 0, ""))
        text = Text()
        text.append("SYSTEM [SYS]  ", Style(color="#ff6600", bold=True))
        text.append(msg, Style(color="#607080"))
        self.write(text)

    def add_transcript(
        self, content: str, speaker: str = "", speaker_id: int = 0,
        confidence: str = "",
    ):
        """Add transcribed text with optional speaker attribution.

        confidence: "" = default, "high" = auto-recognized,
                    "medium" = suggested, "new" = unknown new speaker
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, "transcript", content, speaker, speaker_id, confidence))

        text = self._render_entry(timestamp, content, speaker, speaker_id, confidence)
        self.write(text)

    def _render_entry(
        self, timestamp: str, content: str, speaker: str, speaker_id: int,
        confidence: str = "",
    ) -> Text:
        """Render a single transcript entry as Rich Text."""
        text = Text()
        text.append(f"[{timestamp}]  ", Style(color="#004455"))

        if speaker:
            color = self._color_overrides.get(
                speaker_id,
                _SPEAKER_COLORS[(speaker_id - 1) % len(_SPEAKER_COLORS)],
            )
            text.append(f"{speaker}", Style(color=color, bold=True))

            # Confidence indicator
            conf_info = self._speaker_confidence.get(speaker_id)
            if confidence == "medium" or (conf_info and conf_info[0] == "medium"):
                text.append("?", Style(color="#ffaa00"))
            elif confidence == "high" or (conf_info and conf_info[0] == "high"):
                score = conf_info[1] if conf_info else 0.0
                if score > 0:
                    pct = int(score * 100)
                    text.append(f" ~{pct}%", Style(color="#607080"))
            elif confidence == "new":
                text.append(" *", Style(color="#aa88ff"))

            text.append("  ", Style())
        else:
            text.append("> ", Style(color="#00e5ff", bold=True))

        text.append(content, Style(color="#c0c0c0"))
        return text

    def set_speaker_confidence(
        self, speaker_id: int, tier: str, score: float = 0.0
    ) -> None:
        """Set confidence info for a speaker (used for display indicators)."""
        self._speaker_confidence[speaker_id] = (tier, score)

    def rename_speaker(
        self, speaker_id: int, new_name: str, color: str | None = None
    ) -> None:
        """Rename all entries for a speaker and re-render the transcript."""
        if color:
            self._color_overrides[speaker_id] = color

        # Clear confidence indicator — manually tagged speakers show no indicator
        self._speaker_confidence.pop(speaker_id, None)

        updated = []
        for entry in self._entries:
            ts, typ, content, spk, sid = entry[0], entry[1], entry[2], entry[3], entry[4]
            conf = entry[5] if len(entry) > 5 else ""
            if sid == speaker_id:
                updated.append((ts, typ, content, new_name, sid, ""))
            else:
                updated.append((ts, typ, content, spk, sid, conf))
        self._entries = updated
        self._rerender()

    def _rerender(self) -> None:
        """Clear and re-render all transcript entries."""
        super().clear()
        for entry in self._entries:
            ts, typ, content, speaker = entry[0], entry[1], entry[2], entry[3]
            speaker_id = entry[4] if len(entry) > 4 else 0
            conf = entry[5] if len(entry) > 5 else ""
            if typ == "transcript":
                text = self._render_entry(ts, content, speaker, speaker_id, conf)
            else:
                # System message — re-render as-is
                text = Text()
                text.append("SYSTEM [SYS]  ", Style(color="#ff6600", bold=True))
                text.append(content, Style(color="#607080"))
            self.write(text)

    def clear(self):
        """Clear display and entries."""
        super().clear()
        self._entries.clear()
        self._color_overrides.clear()
        self._speaker_confidence.clear()

    def get_entries(self) -> list[tuple]:
        """Return all transcript entries."""
        return list(self._entries)

    def get_plain_text(self) -> str:
        """Return transcript as plain text."""
        lines = []
        for entry in self._entries:
            ts, _, content, speaker = entry[0], entry[1], entry[2], entry[3]
            prefix = f"[{speaker}] " if speaker else ""
            lines.append(f"[{ts}] {prefix}{content}")
        return "\n".join(lines)

    def get_markdown(
        self,
        model_name: str = "unknown",
        session_start: datetime | None = None,
        language: str = "",
    ) -> str:
        """Return transcript as markdown."""
        header_ts = session_start or datetime.now()
        lines = [
            f"# VOXTERM Transcript",
            f"",
            f"- **Date:** {header_ts.strftime('%Y-%m-%d')}",
            f"- **Time:** {header_ts.strftime('%H:%M:%S')}",
            f"- **Model:** {model_name}",
        ]
        if language:
            lines.append(f"- **Language:** {language}")
        lines.extend([
            f"",
            f"---",
            f"",
        ])
        for entry in self._entries:
            ts, _, content, speaker = entry[0], entry[1], entry[2], entry[3]
            speaker_tag = f" **{speaker}:**" if speaker else ""
            lines.append(f"**[{ts}]**{speaker_tag} {content}")
            lines.append("")
        return "\n".join(lines)
