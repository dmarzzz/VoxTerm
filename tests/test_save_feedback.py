from pathlib import Path

from tui.save_feedback import (
    format_bytes,
    format_clipboard_saved,
    format_file_saved,
)


def test_format_bytes_uses_readable_units():
    assert format_bytes(12) == "12 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(2 * 1024 * 1024) == "2.0 MB"


def test_file_saved_feedback_includes_count_size_path_and_next_action(tmp_path):
    path = tmp_path / "2026-07-03_070000-transcript.md"

    msg = format_file_saved(path, entry_count=4, byte_count=1536)

    assert "saved 4 entries" in msg
    assert "1.5 KB" in msg
    assert str(path) in msg
    assert "press [E] to browse transcripts" in msg


def test_summary_file_saved_feedback_names_summary(tmp_path):
    msg = format_file_saved(
        Path(tmp_path / "summary.md"),
        entry_count=2,
        byte_count=42,
        summary=True,
    )

    assert "2 entries + summary" in msg


def test_clipboard_feedback_says_session_was_cleared():
    assert (
        format_clipboard_saved(3)
        == "copied 3 entries to clipboard - transcript cleared for a new session"
    )
