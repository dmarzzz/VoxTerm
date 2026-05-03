"""Transcript explorer modal — browse and copy saved transcripts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen

from tui.clipboard import clipboard_cmd


class TranscriptExplorerScreen(ModalScreen):
    """Modal listing saved transcripts. Select one to copy its content to clipboard."""

    DEFAULT_CSS = """
    TranscriptExplorerScreen {
        align: center middle;
    }
    #explorer-dialog {
        width: 62;
        height: auto;
        max-height: 24;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #explorer-list {
        height: auto;
        max-height: 16;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #explorer-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #explorer-empty {
        color: #607080;
        text-align: center;
        padding: 2 0;
    }
    #explorer-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, sessions_dir: Path):
        super().__init__()
        self._sessions_dir = sessions_dir
        self._files: list[Path] = []

    def compose(self) -> ComposeResult:
        # Gather .md transcript files (exclude hidden dirs)
        if self._sessions_dir.exists():
            self._files = sorted(
                self._sessions_dir.glob("*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

        with Vertical(id="explorer-dialog") as dialog:
            dialog.border_title = "TRANSCRIPTS"

            if not self._files:
                yield Static(
                    "no saved transcripts found",
                    id="explorer-empty",
                )
            else:
                options = []
                for f in self._files:
                    label = self._format_entry(f)
                    options.append(Option(label, id=str(f)))
                yield OptionList(*options, id="explorer-list")

            yield Static(
                " [#607080]ENTER[/] copy to clipboard  [#607080]ESC[/] close",
                id="explorer-hint",
                markup=True,
            )

    def _format_entry(self, path: Path) -> str:
        """Format a transcript filename into a readable label."""
        stem = path.stem  # e.g. "2026-04-04_160932"
        try:
            dt = datetime.strptime(stem, "%Y-%m-%d_%H%M%S")
            date_str = dt.strftime("%b %-d, %Y")
            time_str = dt.strftime("%-I:%M %p")
        except ValueError:
            date_str = stem
            time_str = ""

        # Parse first and last timestamps to get duration
        duration_str = ""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            timestamps = []
            for line in lines:
                if line.startswith("**[") and "]**" in line:
                    ts = line.split("**[")[1].split("]**")[0]
                    try:
                        timestamps.append(datetime.strptime(ts, "%H:%M:%S"))
                    except ValueError:
                        pass
            if len(timestamps) >= 2:
                delta = timestamps[-1] - timestamps[0]
                total_min = delta.total_seconds() / 60
                if total_min >= 60:
                    hrs = total_min / 60
                    duration_str = f"{hrs:.1f} hrs"
                elif total_min >= 1:
                    duration_str = f"{int(total_min)} min"
                else:
                    duration_str = "<1 min"
        except Exception:
            pass

        parts = [f"  {date_str}"]
        if time_str:
            parts.append(time_str)
        if duration_str:
            parts.append(f"({duration_str})")
        return "  ".join(parts)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option and event.option.id:
            path = Path(event.option.id)
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                self.dismiss({"error": "could not read transcript"})
                return

            cmd = clipboard_cmd()
            if cmd is None:
                self.dismiss({"error": "no clipboard tool found (install xclip, xsel, or wl-copy)"})
                return

            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                proc.communicate(content.encode("utf-8"))
                self.dismiss({"copied": path.stem})
            except Exception as e:
                self.dismiss({"error": f"clipboard copy failed: {e}"})

    def action_cancel(self) -> None:
        self.dismiss(None)
