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

# Peer source colours for merged view (distinct from speaker palette)
_PEER_COLORS = [
    "#00e5ff", "#ff44aa", "#44ff88", "#ffaa44",
    "#bb88ff", "#ff8844", "#44ddff", "#ccff44",
]

# ── Log categories ──────────────────────────────────────────
class Log:
    SYS   = "sys"     # system events, model loading, speaker recognition, errors
    PARTY = "party"   # P2P party mode, peer join/leave, networking
    REC   = "rec"     # recording start/stop, mic events, export/save

_LOG_CATEGORIES = {
    "sys":   {"tag": "SYS",   "tag_color": "#ff6600", "msg_color": "#cc8866"},
    "party": {"tag": "PARTY", "tag_color": "#00ffcc", "msg_color": "#80ccbb"},
    "rec":   {"tag": "REC",   "tag_color": "#ff4466", "msg_color": "#cc8899"},
}

_TIMESTAMP_COLOR = "#306070"

# Rainbow palette for special highlights (dim cyberpunk)
_RAINBOW = [
    "#993344", "#996644", "#998844", "#449966",
    "#448899", "#775599", "#993366",
]


def _append_rainbow(text: Text, s: str) -> None:
    """Append a string with per-character cycling rainbow colors."""
    for i, ch in enumerate(s):
        text.append(ch, Style(color=_RAINBOW[i % len(_RAINBOW)], bold=True))


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
        # Per-entry highlights: entry_index → {substring: color_or_"rainbow"}
        self._entry_highlights: dict[int, dict[str, str]] = {}
        # Merged view state
        self._merged_view = False
        self._peer_color_map: dict[str, str] = {}  # node_id → color
        self._peer_names: dict[str, str] = {}  # node_id → display_name
        # Redacted-view state: when on, the panel shows masked entries while
        # the originals are stashed for exact restore (non-destructive).
        self._redacted_view = False
        self._redacted_backup: list[tuple] | None = None

    def system_message(self, msg: str, category: str = "sys",
                       highlights: dict[str, str] | None = None):
        """Add a system message with optional log category.

        highlights: optional {substring: color} map. Use "rainbow" as the
        color value for per-character rainbow colouring.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, "system", msg, category, 0, ""))
        if highlights:
            self._entry_highlights[len(self._entries) - 1] = highlights
        if not self._merged_view:
            self._write_system(timestamp, msg, category, highlights)

    def _write_system(self, timestamp: str, msg: str, category: str,
                      highlights: dict[str, str] | None = None) -> None:
        """Render and write a system message line."""
        cat = _LOG_CATEGORIES.get(category, _LOG_CATEGORIES["sys"])
        text = Text()
        text.append(f"[{timestamp}]  ", Style(color=_TIMESTAMP_COLOR))
        text.append(f"{cat['tag']}  ", Style(color=cat["tag_color"], bold=True))
        if highlights:
            remaining = msg
            msg_style = Style(color=cat["msg_color"])
            for substr, color in highlights.items():
                if substr in remaining:
                    before, _, after = remaining.partition(substr)
                    if before:
                        text.append(before, msg_style)
                    if color == "rainbow":
                        _append_rainbow(text, substr)
                    else:
                        text.append(substr, Style(color=color, bold=True))
                    remaining = after
            if remaining:
                text.append(remaining, msg_style)
        else:
            text.append(msg, Style(color=cat["msg_color"]))
        self.write(text)

    def add_transcript(
        self, content: str, speaker: str = "", speaker_id: int = 0,
        confidence: str = "", overlap: bool = False,
    ):
        """Add transcribed text with optional speaker attribution.

        confidence: "" = default, "high" = auto-recognized,
                    "medium" = suggested, "new" = unknown new speaker
        overlap: True if overlapping speech detected in this segment
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, "transcript", content, speaker, speaker_id, confidence))

        if not self._merged_view:
            text = self._render_entry(timestamp, content, speaker, speaker_id, confidence, overlap)
            self.write(text)

    def _render_entry(
        self, timestamp: str, content: str, speaker: str, speaker_id: int,
        confidence: str = "", overlap: bool = False,
    ) -> Text:
        """Render a single transcript entry as Rich Text."""
        text = Text()
        text.append(f"[{timestamp}]  ", Style(color=_TIMESTAMP_COLOR))

        if speaker:
            color = self._color_overrides.get(
                speaker_id,
                _SPEAKER_COLORS[(speaker_id - 1) % len(_SPEAKER_COLORS)],
            )

            # Overlap indicator
            if overlap:
                text.append("[+] ", Style(color="#ff6600", bold=True))

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

    # ── merged view ──────────────────────────────────────────

    @property
    def merged_view(self) -> bool:
        return self._merged_view

    def set_merged_view(
        self, enabled: bool, assembler=None,
        local_name: str = "you", peer_names: dict[str, str] | None = None,
    ):
        """Toggle between local and merged transcript view.

        When enabling, pass the TranscriptAssembler to render the merged timeline.
        peer_names: optional mapping of node_id → display_name for peers.
        """
        self._merged_view = enabled
        if peer_names:
            self._peer_names = peer_names
        if enabled:
            self.border_title = "TRANSCRIPT // MERGED"
            if assembler:
                self._render_merged(assembler, local_name)
        else:
            self.border_title = "TRANSCRIPT // LIVE"
            self._rerender()

    def refresh_merged(
        self, assembler, local_name: str = "you",
        peer_names: dict[str, str] | None = None,
    ):
        """Re-render the merged timeline (call when new segments arrive while in merged view)."""
        if not self._merged_view:
            return
        if peer_names:
            self._peer_names = peer_names
        self._render_merged(assembler, local_name)

    def _get_peer_color(self, node_id: str) -> str:
        """Assign a stable color to a peer node_id."""
        if node_id not in self._peer_color_map:
            idx = len(self._peer_color_map) % len(_PEER_COLORS)
            self._peer_color_map[node_id] = _PEER_COLORS[idx]
        return self._peer_color_map[node_id]

    def _render_merged(self, assembler, local_name: str):
        """Render all segments from the assembler in time order."""
        from network.segments import LOCAL_NODE_ID

        super().clear()
        finals = assembler.get_finals()
        partials = assembler.get_partials()

        for seg in finals:
            text = self._render_merged_segment(seg, local_name)
            self.write(text)

        # Render in-progress partials dimmed at the bottom
        for seg in partials:
            text = self._render_merged_segment(seg, local_name, is_partial=True)
            self.write(text)

    def _render_merged_segment(self, seg, local_name: str, is_partial: bool = False) -> Text:
        """Render a single MergedSegment."""
        from network.segments import LOCAL_NODE_ID
        from datetime import datetime as dt

        text = Text()

        # Timestamp from adjusted_start_ts (monotonic → wall-clock approximation)
        try:
            import time
            wall = time.time() - time.monotonic() + seg.adjusted_start_ts
            ts_str = dt.fromtimestamp(wall).strftime("%H:%M:%S")
        except Exception:
            ts_str = "??:??:??"

        text.append(f"[{ts_str}]  ", Style(color="#004455"))

        # Source label
        is_local = seg.node_id == LOCAL_NODE_ID
        if is_local:
            source_color = "#00ffcc"
            source_label = local_name
        else:
            source_color = self._get_peer_color(seg.node_id)
            source_label = self._peer_names.get(seg.node_id, seg.node_id[:8])

        text.append(f"{source_label}", Style(color=source_color, bold=True))

        # Speaker name (if different from source)
        if seg.speaker_name and seg.speaker_name != source_label:
            text.append(f"/{seg.speaker_name}", Style(color="#607080"))

        # Dominant mic indicator for local segments (which mic contributed most)
        if is_local and seg.dominant_mic and seg.dominant_mic != LOCAL_NODE_ID:
            mic_name = self._peer_names.get(seg.dominant_mic, seg.dominant_mic[:6])
            text.append(f" via {mic_name}", Style(color="#607080", italic=True))

        text.append("  ", Style())

        # Content — dim if partial
        content_style = Style(color="#606060", italic=True) if (is_partial or seg.is_partial) else Style(color="#c0c0c0")
        text.append(seg.text, content_style)

        if is_partial or seg.is_partial:
            text.append(" …", Style(color="#404040"))

        return text

    # ── existing methods ─────────────────────────────────────

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
        if not self._merged_view:
            self._rerender()

    def _rerender(self) -> None:
        """Clear and re-render all transcript entries."""
        super().clear()
        for i, entry in enumerate(self._entries):
            ts, typ, content, speaker = entry[0], entry[1], entry[2], entry[3]
            speaker_id = entry[4] if len(entry) > 4 else 0
            conf = entry[5] if len(entry) > 5 else ""
            if typ == "transcript":
                text = self._render_entry(ts, content, speaker, speaker_id, conf)
                self.write(text)
            else:
                category = speaker if speaker in _LOG_CATEGORIES else "sys"
                self._write_system(ts, content, category,
                                   self._entry_highlights.get(i))


    def redact_view(self, spans: list[tuple[str, str]], label: str = "") -> int:
        """Mask the on-screen transcript with ``spans`` (content + speaker).

        Non-destructive: the original entries are stashed so ``restore_view``
        reverts exactly. Returns the number of entries changed. Imported here
        (not at module top) to avoid a TUI→engine import at load time.
        """
        from redaction.engine import apply_redactions

        if not spans:
            return 0
        if self._redacted_backup is None:
            self._redacted_backup = list(self._entries)
        changed = 0
        masked: list[tuple] = []
        for entry in self._entries:
            ts, typ, content, spk = entry[0], entry[1], entry[2], entry[3]
            sid = entry[4] if len(entry) > 4 else 0
            conf = entry[5] if len(entry) > 5 else ""
            if typ == "transcript":
                new_content = apply_redactions(content, spans).redacted_text
                new_spk = apply_redactions(spk, spans).redacted_text if spk else spk
                if new_content != content or new_spk != spk:
                    changed += 1
                masked.append((ts, typ, new_content, new_spk, sid, conf))
            else:
                masked.append(entry)
        self._entries = masked
        self._redacted_view = True
        if not self._merged_view:
            self._rerender()
        suffix = f" · {label}" if label else ""
        self.border_title = f"TRANSCRIPT // REDACTED{suffix}"
        return changed

    def restore_view(self) -> bool:
        """Revert a redacted view to the original entries. True if reverted."""
        if self._redacted_backup is None:
            return False
        self._entries = self._redacted_backup
        self._redacted_backup = None
        self._redacted_view = False
        if not self._merged_view:
            self._rerender()
        self.border_title = "TRANSCRIPT // LIVE"
        return True

    def is_redacted_view(self) -> bool:
        return self._redacted_view

    def clear(self):
        """Clear display and entries."""
        super().clear()
        self._entries.clear()
        self._entry_highlights.clear()
        self._color_overrides.clear()
        self._speaker_confidence.clear()
        self._peer_color_map.clear()
        self._peer_names.clear()
        self._merged_view = False
        self._redacted_view = False
        self._redacted_backup = None
        self.border_title = "TRANSCRIPT // LIVE"

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
