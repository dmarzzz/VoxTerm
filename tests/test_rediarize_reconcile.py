"""Unit tests for the pure re-diarization reconciliation (no models/audio)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.diarization.rediarize import reconcile_labels

PALETTE = ["#00ffcc", "#ff44aa", "#44ff44", "#ffaa00"]


def _entry(idx, t0, t1, sid):
    return {"idx": idx, "t0": t0, "t1": t1, "sid": sid}


def test_empty_inputs():
    assert reconcile_labels([], [], {}, {}, PALETTE) == {}
    assert reconcile_labels([(0, 1, 0)], [], {}, {}, PALETTE) == {}
    assert reconcile_labels([], [_entry(0, 0, 1, 1)], {}, {}, PALETTE) == {}


def test_noop_when_clusters_match_online():
    # two speakers, online already correct -> no relabels
    segs = [(0.0, 5.0, 0), (5.0, 10.0, 1)]
    entries = [_entry(0, 0.0, 5.0, 1), _entry(1, 5.0, 10.0, 2)]
    out = reconcile_labels(segs, entries, {1: "A", 2: "B"},
                           {1: "#00ffcc", 2: "#ff44aa"}, PALETTE)
    # each cluster's dominant online sid is unique -> ids unchanged
    assert out == {}


def test_merge_oversegmented_speaker():
    # online split ONE real speaker into two ids (1 and 2); one cluster covers all
    segs = [(0.0, 10.0, 0)]
    entries = [_entry(0, 0.0, 6.0, 1), _entry(1, 6.0, 10.0, 2)]
    out = reconcile_labels(segs, entries, {1: "A", 2: "B"},
                           {1: "#00ffcc", 2: "#ff44aa"}, PALETTE)
    # cluster dominated by sid 1 (more duration) -> entry 1 (sid 2) merged into 1
    assert 1 in out and out[1][0] == 1
    assert 0 not in out  # entry 0 already sid 1


def test_split_merged_speakers():
    # online merged two real speakers into ONE id (1); two clusters split them
    segs = [(0.0, 5.0, 0), (5.0, 10.0, 1)]
    entries = [_entry(0, 0.0, 5.0, 1), _entry(1, 5.0, 10.0, 1)]
    out = reconcile_labels(segs, entries, {1: "A"}, {1: "#00ffcc"}, PALETTE)
    # one cluster keeps sid 1, the other gets a fresh id with a new label/color
    assert len(out) == 1
    (idx, (sid, label, color)), = out.items()
    assert sid != 1 and label.startswith("Speaker") and color in PALETTE


def test_preserves_cross_session_name_and_color():
    segs = [(0.0, 10.0, 7)]
    entries = [_entry(0, 0.0, 5.0, 3), _entry(1, 5.0, 10.0, 9)]
    # sid 3 dominates (tie broken by duration equal -> max picks one); give names
    names = {3: "Ron", 9: "Dana"}
    colors = {3: "#abcdef", 9: "#123456"}
    out = reconcile_labels(segs, entries, names, colors, PALETTE)
    # whichever sid is dominant, the merged entry inherits that sid's name+color
    for idx, (sid, label, color) in out.items():
        assert sid in (3, 9)
        assert label == names[sid] and color == colors[sid]


def test_entry_without_overlap_is_untouched():
    segs = [(0.0, 2.0, 0)]
    entries = [_entry(0, 0.0, 2.0, 1), _entry(1, 50.0, 52.0, 2)]  # entry 1 has no cluster
    out = reconcile_labels(segs, entries, {}, {}, PALETTE)
    assert 1 not in out  # no overlapping cluster -> left as-is


def test_new_ids_do_not_collide_with_existing():
    segs = [(0.0, 5.0, 0), (5.0, 10.0, 1)]
    entries = [_entry(0, 0.0, 5.0, 5), _entry(1, 5.0, 10.0, 5)]  # both online sid 5
    out = reconcile_labels(segs, entries, {5: "X"}, {5: "#000000"}, PALETTE)
    new_sids = {v[0] for v in out.values()}
    assert all(s > 5 for s in new_sids)  # fresh ids exceed the max existing sid
