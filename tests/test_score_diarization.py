from __future__ import annotations

import json
from pathlib import Path

from scripts.score_diarization import (
    ReferenceSet,
    Segment,
    load_predictions,
    main,
    score_manifest,
    score_session,
)


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_events(path: Path, turns: list[tuple[float, float, int, str]]) -> Path:
    rows = [{"kind": "session", "phase": "start", "t": 1000.0}]
    for idx, (start, end, sid, text) in enumerate(turns):
        rows.append({
            "kind": "text",
            "t": 1000.0 + start,
            "speaker": f"Speaker {sid}",
            "speaker_id": sid,
            "text": text,
            "audio_offset": start,
            "audio_end": end,
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _reference(path: Path, session_id: str = "jam", speakers: list[str] | None = None) -> Path:
    speakers = speakers or ["alice", "bob", "cara"]
    segments = [
        {"start": float(idx), "end": float(idx + 1), "speaker": speaker, "text": speaker[0]}
        for idx, speaker in enumerate(speakers)
    ]
    return _write_json(path, {
        "session_id": session_id,
        "speakers": speakers,
        "segments": segments,
    })


def test_score_session_maps_arbitrary_voxterm_cluster_ids():
    reference = ReferenceSet(
        session_id="jam",
        speakers=("alice", "bob", "cara"),
        segments=(
            Segment(0.0, 1.0, "alice", source_index=0),
            Segment(1.0, 2.0, "bob", source_index=1),
            Segment(2.0, 3.0, "alice", source_index=2),
            Segment(3.0, 4.0, "cara", source_index=3),
        ),
    )
    predictions = [
        Segment(0.0, 1.0, "sid:2"),
        Segment(1.0, 2.0, "sid:7"),
        Segment(2.0, 3.0, "sid:2"),
        Segment(3.0, 4.0, "sid:3"),
    ]

    result = score_session(
        session_id="jam",
        reference=reference,
        predictions=predictions,
        events_path=Path("jam-events.jsonl"),
        reference_path=Path("jam-reference.json"),
    )

    assert result["segment_accuracy"] == 1.0
    assert result["duration_accuracy"] == 1.0
    assert result["speaker_mapping"] == {"sid:2": "alice", "sid:3": "cara", "sid:7": "bob"}


def test_score_session_counts_misses_and_cluster_confusions():
    reference = ReferenceSet(
        session_id="jam",
        speakers=("alice", "bob", "cara"),
        segments=(
            Segment(0.0, 1.0, "alice", source_index=0),
            Segment(1.0, 2.0, "bob", source_index=1),
            Segment(2.0, 3.0, "cara", source_index=2),
        ),
    )
    predictions = [
        Segment(0.0, 1.0, "sid:1"),
        Segment(1.0, 2.0, "sid:1"),
    ]

    result = score_session(
        session_id="jam",
        reference=reference,
        predictions=predictions,
        events_path=Path("jam-events.jsonl"),
        reference_path=Path("jam-reference.json"),
    )

    assert result["correct_segments"] == 1
    assert result["confused_segments"] == 1
    assert result["missed_segments"] == 1
    assert result["segment_accuracy"] == 1 / 3


def test_load_predictions_prefers_audio_offsets_over_wall_clock(tmp_path: Path):
    events = tmp_path / "session-events.jsonl"
    events.write_text(
        "\n".join([
            json.dumps({"kind": "session", "phase": "start", "t": 1000.0}),
            json.dumps({
                "kind": "text",
                "t": 5000.0,
                "speaker_id": 4,
                "speaker": "Speaker 4",
                "text": "late wall clock",
                "audio_offset": 10.0,
                "audio_end": 11.5,
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions(events)

    assert predictions == [Segment(10.0, 11.5, "sid:4", "late wall clock", 0)]


def test_score_manifest_strict_reports_missing_corpus_shape(tmp_path: Path):
    events = _write_events(tmp_path / "one-events.jsonl", [
        (0.0, 1.0, 1, "a"),
        (1.0, 2.0, 2, "b"),
        (2.0, 3.0, 3, "c"),
    ])
    reference = _reference(tmp_path / "one-reference.json")
    manifest = _write_json(tmp_path / "manifest.json", {
        "sessions": [{"id": "one", "events": events.name, "reference": reference.name}],
    })

    report = score_manifest(manifest, strict=True)

    assert report["benchmark_passed"] is True
    assert report["passed"] is False
    assert report["strict_errors"] == [
        "strict mode requires at least 4 sessions; got 1",
        "strict mode requires at least one 5-speaker session",
    ]


def test_cli_json_passes_four_session_strict_manifest(tmp_path: Path, capsys):
    sessions = []
    for idx in range(4):
        speakers = ["alice", "bob", "cara", "dan", "eve"] if idx == 0 else ["alice", "bob", "cara"]
        events = _write_events(
            tmp_path / f"jam-{idx}-events.jsonl",
            [(float(turn), float(turn + 1), 20 - turn, speaker[0]) for turn, speaker in enumerate(speakers)],
        )
        reference = _reference(tmp_path / f"jam-{idx}-reference.json", session_id=f"jam-{idx}", speakers=speakers)
        sessions.append({"id": f"jam-{idx}", "events": events.name, "reference": reference.name})
    manifest = _write_json(tmp_path / "manifest.json", {
        "target_segment_accuracy": 0.9,
        "sessions": sessions,
    })

    rc = main([str(manifest), "--strict", "--json"])

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["passed"] is True
    assert out["aggregate"]["segment_accuracy"] == 1.0
    assert out["aggregate"]["reference_segments"] == 14
