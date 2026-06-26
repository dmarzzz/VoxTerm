"""Tests for TranscriptPanel relabel support used by the re-diarization pass."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tui.widgets.transcript import TranscriptPanel, _SPEAKER_COLORS


def _panel():
    # RichLog needs no running app for the data-layer methods we exercise; we
    # avoid .write() rendering paths by staying in non-merged data ops only.
    p = TranscriptPanel.__new__(TranscriptPanel)
    # minimal state init mirroring __init__ without RichLog/app machinery
    p._entries = []
    p._color_overrides = {}
    p._speaker_confidence = {}
    p._entry_highlights = {}
    p._merged_view = True  # suppress rendering (.write/_rerender are no-ops path)
    p._peer_color_map = {}
    p._peer_names = {}
    return p


def test_add_transcript_returns_index():
    p = _panel()
    i0 = p.add_transcript("hello", "Speaker 1", 1)
    i1 = p.add_transcript("world", "Speaker 2", 2)
    assert i0 == 0 and i1 == 1
    assert p._entries[0][3] == "Speaker 1" and p._entries[0][4] == 1


def test_apply_relabels_changes_entries():
    p = _panel()
    p.add_transcript("a", "Speaker 1", 1)
    p.add_transcript("b", "Speaker 1", 1)
    # relabel entry 1 to a fresh speaker 2 with a color
    changed = p.apply_relabels({1: (2, "Speaker 2", "#ff44aa")})
    assert changed == 1
    assert p._entries[1][3] == "Speaker 2" and p._entries[1][4] == 2
    assert p._color_overrides[2] == "#ff44aa"
    assert p._entries[0][4] == 1  # untouched


def test_apply_relabels_skips_system_and_out_of_range():
    p = _panel()
    p.system_message("sys line")          # entry 0, type "system"
    p.add_transcript("a", "Speaker 1", 1)  # entry 1
    changed = p.apply_relabels({0: (9, "X", "#000"), 1: (2, "Y", "#111"), 99: (3, "Z", "#222")})
    assert changed == 1                    # only the transcript entry 1
    assert p._entries[0][1] == "system"
    assert p._entries[1][3] == "Y" and p._entries[1][4] == 2


def test_apply_relabels_clears_confidence():
    p = _panel()
    p.add_transcript("a", "Speaker 1", 1, confidence="high")
    p.set_speaker_confidence(1, "high", 0.9)
    p.apply_relabels({0: (2, "Speaker 2", "#ff44aa")})
    assert 2 not in p._speaker_confidence
    assert p._entries[0][5] == ""  # confidence cleared on corrected entry
