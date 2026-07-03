#!/usr/bin/env python3
"""Score VoxTerm speaker attribution against labeled reference segments.

The scorer consumes a manifest that points at VoxTerm ``*-events.jsonl`` logs and
human-labeled reference JSON files. It does not run ASR or diarization; it makes
field recordings reproducibly scoreable after they have been processed once.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET = 0.90
DEFAULT_FALLBACK_TURN_SECONDS = 2.0
UNATTRIBUTED = "__unattributed__"


class BenchmarkError(Exception):
    """A manifest, reference, or event log cannot be scored honestly."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str = ""
    source_index: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ReferenceSet:
    session_id: str
    speakers: tuple[str, ...]
    segments: tuple[Segment, ...]


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise BenchmarkError(f"{field} must be a finite number, got boolean")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise BenchmarkError(f"{field} must be finite, got {value!r}")
    return out


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise BenchmarkError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc


def _coerce_speaker_id(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return int(stripped)
        except ValueError:
            return 0
    return 0


def _predicted_speaker_label(event: dict[str, Any]) -> str:
    sid = _coerce_speaker_id(event.get("speaker_id"))
    if sid > 0:
        return f"sid:{sid}"
    label = str(event.get("speaker") or "").strip()
    if label:
        return f"label:{label}"
    return UNATTRIBUTED


def _event_start(event: dict[str, Any], session_start: float) -> float:
    audio_offset = _finite_float_or_none(event.get("audio_offset"))
    if audio_offset is not None:
        return max(0.0, audio_offset)
    wall_time = _finite_float_or_none(event.get("t"))
    if wall_time is not None:
        return max(0.0, wall_time - session_start)
    return 0.0


def load_predictions(path: Path, *, fallback_turn_seconds: float = DEFAULT_FALLBACK_TURN_SECONDS) -> list[Segment]:
    """Load predicted speaker turns from a VoxTerm event log.

    ``audio_offset``/``audio_end`` are preferred because they are true recording
    offsets. Live logs without audio offsets fall back to wall-clock deltas from
    the session start, matching the exporter's behavior closely enough for field
    triage.
    """
    if fallback_turn_seconds <= 0 or not math.isfinite(fallback_turn_seconds):
        raise BenchmarkError("fallback_turn_seconds must be a positive finite number")

    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
                if isinstance(obj, dict):
                    events.append(obj)
    except FileNotFoundError as exc:
        raise BenchmarkError(f"{path} does not exist") from exc

    text_events = [
        event for event in events
        if event.get("kind") == "text" and str(event.get("text") or "").strip()
    ]
    if not text_events:
        return []

    session_starts = [
        _finite_float_or_none(event.get("t"))
        for event in events
        if event.get("kind") == "session" and event.get("phase") == "start"
    ]
    session_start = next((value for value in session_starts if value is not None), None)
    if session_start is None:
        session_start = _finite_float_or_none(text_events[0].get("t")) or 0.0

    starts = [_event_start(event, session_start) for event in text_events]
    predictions: list[Segment] = []
    for idx, event in enumerate(text_events):
        start = starts[idx]
        audio_end = _finite_float_or_none(event.get("audio_end"))
        if audio_end is not None and audio_end > start:
            end = audio_end
        elif idx + 1 < len(starts) and starts[idx + 1] > start:
            end = starts[idx + 1]
        else:
            end = start + fallback_turn_seconds

        predictions.append(Segment(
            start=start,
            end=end,
            speaker=_predicted_speaker_label(event),
            text=str(event.get("text") or "").strip(),
            source_index=idx,
        ))
    return predictions


def load_reference(path: Path) -> ReferenceSet:
    """Load human-labeled reference segments from JSON.

    Accepted shape:
        {"session_id": "jam-001", "speakers": ["alice"], "segments": [...]}

    ``speakers`` is optional; when present, every segment speaker must be listed.
    A top-level segment list is accepted for small fixtures.
    """
    data = _load_json(path)
    declared_speakers: set[str] | None = None
    session_id = path.stem
    if isinstance(data, list):
        raw_segments = data
    elif isinstance(data, dict):
        raw_segments = data.get("segments")
        session_id = str(data.get("session_id") or session_id)
        if "speakers" in data:
            if not isinstance(data["speakers"], list) or not data["speakers"]:
                raise BenchmarkError(f"{path}: speakers must be a non-empty list when provided")
            declared_speakers = {str(s).strip() for s in data["speakers"] if str(s).strip()}
            if len(declared_speakers) != len(data["speakers"]):
                raise BenchmarkError(f"{path}: speakers must be non-empty strings")
    else:
        raise BenchmarkError(f"{path}: reference must be an object or segment list")

    if not isinstance(raw_segments, list) or not raw_segments:
        raise BenchmarkError(f"{path}: reference requires a non-empty segments list")

    segments: list[Segment] = []
    for idx, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise BenchmarkError(f"{path}: segment {idx} must be an object")
        start = _finite_float(raw.get("start"), field=f"{path}: segment {idx} start")
        end = _finite_float(raw.get("end"), field=f"{path}: segment {idx} end")
        speaker = str(raw.get("speaker") or "").strip()
        if not speaker:
            raise BenchmarkError(f"{path}: segment {idx} speaker must be a non-empty string")
        if end <= start:
            raise BenchmarkError(f"{path}: segment {idx} end must be greater than start")
        if declared_speakers is not None and speaker not in declared_speakers:
            raise BenchmarkError(f"{path}: segment {idx} speaker {speaker!r} is not in speakers")
        segments.append(Segment(
            start=start,
            end=end,
            speaker=speaker,
            text=str(raw.get("text") or "").strip(),
            source_index=idx,
        ))

    inferred_speakers = tuple(sorted({segment.speaker for segment in segments}))
    return ReferenceSet(session_id=session_id, speakers=inferred_speakers,
                        segments=tuple(sorted(segments, key=lambda s: (s.start, s.end))))


def _overlap_seconds(left: Segment, right: Segment) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _merged_duration(intervals: list[tuple[float, float]]) -> float:
    clean = sorted((start, end) for start, end in intervals if end > start)
    if not clean:
        return 0.0
    merged: list[list[float]] = [[clean[0][0], clean[0][1]]]
    for start, end in clean[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def best_speaker_mapping(predictions: list[Segment], reference: list[Segment]) -> dict[str, str]:
    """Map arbitrary VoxTerm cluster labels to reference speaker names.

    The assignment maximizes overlapped reference duration and is one-to-one. This
    mirrors how diarization clusters are evaluated: ``sid:1`` has no intrinsic
    meaning until matched to a human label.
    """
    pred_speakers = sorted({segment.speaker for segment in predictions if segment.speaker != UNATTRIBUTED})
    ref_speakers = sorted({segment.speaker for segment in reference})
    if not pred_speakers or not ref_speakers:
        return {}

    matrix: dict[tuple[str, str], float] = {}
    for pred in predictions:
        if pred.speaker == UNATTRIBUTED:
            continue
        for ref in reference:
            overlap = _overlap_seconds(pred, ref)
            if overlap > 0:
                key = (pred.speaker, ref.speaker)
                matrix[key] = matrix.get(key, 0.0) + overlap

    # Strict issue #87 corpora are 3-5 speakers, so bitmask DP is small and exact.
    if len(ref_speakers) <= 12:
        states: dict[int, tuple[float, dict[str, str]]] = {0: (0.0, {})}
        for pred_speaker in pred_speakers:
            next_states: dict[int, tuple[float, dict[str, str]]] = {
                mask: (score, dict(mapping)) for mask, (score, mapping) in states.items()
            }
            for mask, (score, mapping) in states.items():
                for ref_idx, ref_speaker in enumerate(ref_speakers):
                    if mask & (1 << ref_idx):
                        continue
                    gain = matrix.get((pred_speaker, ref_speaker), 0.0)
                    if gain <= 0:
                        continue
                    new_mask = mask | (1 << ref_idx)
                    new_score = score + gain
                    current = next_states.get(new_mask)
                    if current is None or new_score > current[0]:
                        new_mapping = dict(mapping)
                        new_mapping[pred_speaker] = ref_speaker
                        next_states[new_mask] = (new_score, new_mapping)
            states = next_states
        return max(states.values(), key=lambda item: item[0])[1]

    # Fallback for unusually large non-strict corpora.
    mapping: dict[str, str] = {}
    used_refs: set[str] = set()
    for (pred_speaker, ref_speaker), _duration in sorted(matrix.items(), key=lambda item: item[1], reverse=True):
        if pred_speaker in mapping or ref_speaker in used_refs:
            continue
        mapping[pred_speaker] = ref_speaker
        used_refs.add(ref_speaker)
    return mapping


def score_session(
    *,
    session_id: str,
    reference: ReferenceSet,
    predictions: list[Segment],
    events_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    ref_segments = list(reference.segments)
    mapping = best_speaker_mapping(predictions, ref_segments)

    details: list[dict[str, Any]] = []
    correct_segments = 0
    covered_segments = 0
    missed_segments = 0
    confused_segments = 0

    covered_ref_seconds = 0.0
    correct_ref_seconds = 0.0
    for ref in ref_segments:
        overlaps = [
            (pred, _overlap_seconds(pred, ref))
            for pred in predictions
        ]
        overlaps = [(pred, overlap) for pred, overlap in overlaps if overlap > 0]
        best_pred: Segment | None = None
        best_overlap = 0.0
        if overlaps:
            best_pred, best_overlap = max(overlaps, key=lambda item: item[1])
            covered_segments += 1
            mapped_speaker = mapping.get(best_pred.speaker)
            is_correct = mapped_speaker == ref.speaker
            if is_correct:
                correct_segments += 1
            else:
                confused_segments += 1
        else:
            mapped_speaker = None
            is_correct = False
            missed_segments += 1

        covered_ref_seconds += _merged_duration([
            (max(pred.start, ref.start), min(pred.end, ref.end))
            for pred, _overlap in overlaps
        ])
        correct_ref_seconds += _merged_duration([
            (max(pred.start, ref.start), min(pred.end, ref.end))
            for pred, _overlap in overlaps
            if mapping.get(pred.speaker) == ref.speaker
        ])

        details.append({
            "index": ref.source_index,
            "start": ref.start,
            "end": ref.end,
            "speaker": ref.speaker,
            "best_prediction": best_pred.speaker if best_pred else None,
            "mapped_prediction": mapped_speaker,
            "overlap_seconds": round(best_overlap, 3),
            "correct": is_correct,
        })

    reference_seconds = sum(segment.duration for segment in ref_segments)
    predicted_seconds = sum(segment.duration for segment in predictions)
    pred_over_ref_seconds = 0.0
    for pred in predictions:
        pred_over_ref_seconds += _merged_duration([
            (max(pred.start, ref.start), min(pred.end, ref.end))
            for ref in ref_segments
            if _overlap_seconds(pred, ref) > 0
        ])
    false_alarm_seconds = max(0.0, predicted_seconds - pred_over_ref_seconds)

    total_segments = len(ref_segments)
    return {
        "session_id": session_id,
        "events": str(events_path),
        "reference": str(reference_path),
        "reference_speakers": len(reference.speakers),
        "reference_segments": total_segments,
        "predicted_segments": len(predictions),
        "correct_segments": correct_segments,
        "covered_segments": covered_segments,
        "missed_segments": missed_segments,
        "confused_segments": confused_segments,
        "segment_accuracy": correct_segments / total_segments if total_segments else 0.0,
        "segment_coverage": covered_segments / total_segments if total_segments else 0.0,
        "reference_seconds": round(reference_seconds, 3),
        "predicted_seconds": round(predicted_seconds, 3),
        "covered_ref_seconds": round(covered_ref_seconds, 3),
        "correct_ref_seconds": round(correct_ref_seconds, 3),
        "duration_accuracy": correct_ref_seconds / reference_seconds if reference_seconds else 0.0,
        "false_alarm_seconds": round(false_alarm_seconds, 3),
        "speaker_mapping": mapping,
        "details": details,
    }


def _resolve_manifest_path(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"manifest session {field} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _target_value(value: Any) -> float:
    target = DEFAULT_TARGET if value is None else _finite_float(value, field="target")
    if not 0 < target <= 1:
        raise BenchmarkError("target must be > 0 and <= 1")
    return target


def _aggregate(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["reference_segments"] for row in sessions)
    correct = sum(row["correct_segments"] for row in sessions)
    covered = sum(row["covered_segments"] for row in sessions)
    missed = sum(row["missed_segments"] for row in sessions)
    confused = sum(row["confused_segments"] for row in sessions)
    ref_seconds = sum(row["reference_seconds"] for row in sessions)
    correct_seconds = sum(row["correct_ref_seconds"] for row in sessions)
    covered_seconds = sum(row["covered_ref_seconds"] for row in sessions)
    false_alarm_seconds = sum(row["false_alarm_seconds"] for row in sessions)
    return {
        "sessions": len(sessions),
        "reference_segments": total,
        "correct_segments": correct,
        "covered_segments": covered,
        "missed_segments": missed,
        "confused_segments": confused,
        "segment_accuracy": correct / total if total else 0.0,
        "segment_coverage": covered / total if total else 0.0,
        "reference_seconds": round(ref_seconds, 3),
        "covered_ref_seconds": round(covered_seconds, 3),
        "correct_ref_seconds": round(correct_seconds, 3),
        "duration_accuracy": correct_seconds / ref_seconds if ref_seconds else 0.0,
        "false_alarm_seconds": round(false_alarm_seconds, 3),
    }


def _strict_errors(sessions: list[dict[str, Any]], *, target: float) -> list[str]:
    errors: list[str] = []
    if len(sessions) < 4:
        errors.append(f"strict mode requires at least 4 sessions; got {len(sessions)}")
    has_five_speaker_session = False
    for row in sessions:
        speakers = row["reference_speakers"]
        if speakers < 3 or speakers > 5:
            errors.append(
                f"{row['session_id']}: strict mode requires 3-5 reference speakers; got {speakers}"
            )
        if speakers == 5:
            has_five_speaker_session = True
            if row["segment_accuracy"] < target:
                errors.append(
                    f"{row['session_id']}: 5-speaker segment accuracy "
                    f"{_pct(row['segment_accuracy'])} is below target {_pct(target)}"
                )
    if sessions and not has_five_speaker_session:
        errors.append("strict mode requires at least one 5-speaker session")
    return errors


def score_manifest(
    manifest_path: Path,
    *,
    strict: bool = False,
    target: float | None = None,
    fallback_turn_seconds: float = DEFAULT_FALLBACK_TURN_SECONDS,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    data = _load_json(manifest_path)
    if not isinstance(data, dict):
        raise BenchmarkError("manifest must be a JSON object")
    raw_sessions = data.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise BenchmarkError("manifest requires a non-empty sessions list")

    target_value = _target_value(target if target is not None else data.get("target_segment_accuracy"))
    base = manifest_path.parent
    sessions: list[dict[str, Any]] = []
    for idx, raw_session in enumerate(raw_sessions):
        if not isinstance(raw_session, dict):
            raise BenchmarkError(f"manifest session {idx} must be an object")
        events_path = _resolve_manifest_path(base, raw_session.get("events"), field="events")
        reference_path = _resolve_manifest_path(
            base,
            raw_session.get("reference", raw_session.get("labels")),
            field="reference",
        )
        reference = load_reference(reference_path)
        predictions = load_predictions(events_path, fallback_turn_seconds=fallback_turn_seconds)
        session_id = str(raw_session.get("id") or reference.session_id or events_path.stem)
        sessions.append(score_session(
            session_id=session_id,
            reference=reference,
            predictions=predictions,
            events_path=events_path,
            reference_path=reference_path,
        ))

    aggregate = _aggregate(sessions)
    strict_errors = _strict_errors(sessions, target=target_value) if strict else []
    benchmark_passed = aggregate["segment_accuracy"] >= target_value
    return {
        "manifest": str(manifest_path),
        "target_segment_accuracy": target_value,
        "strict": strict,
        "strict_errors": strict_errors,
        "benchmark_passed": benchmark_passed,
        "passed": benchmark_passed and not strict_errors,
        "aggregate": aggregate,
        "sessions": sessions,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "VoxTerm diarization attribution benchmark",
        f"manifest: {report['manifest']}",
        f"target: segment_accuracy >= {_pct(report['target_segment_accuracy'])}",
        f"strict: {'on' if report['strict'] else 'off'}",
        "",
        "session                       spk   ref  pred  correct  seg_acc  coverage  dur_acc  false_alarm_s",
        "----------------------------  ----  ---  ----  -------  -------  --------  -------  -------------",
    ]
    for row in report["sessions"]:
        session = row["session_id"][:28]
        lines.append(
            f"{session:<28}  {row['reference_speakers']:>4}  "
            f"{row['reference_segments']:>3}  {row['predicted_segments']:>4}  "
            f"{row['correct_segments']:>7}  {_pct(row['segment_accuracy']):>7}  "
            f"{_pct(row['segment_coverage']):>8}  {_pct(row['duration_accuracy']):>7}  "
            f"{row['false_alarm_seconds']:>13.1f}"
        )

    aggregate = report["aggregate"]
    lines.extend([
        "",
        f"aggregate: {aggregate['correct_segments']}/{aggregate['reference_segments']} "
        f"segments correct = {_pct(aggregate['segment_accuracy'])}",
        f"coverage: {_pct(aggregate['segment_coverage'])} of reference segments, "
        f"{_pct(aggregate['duration_accuracy'])} of reference seconds correctly attributed",
        f"false_alarm_seconds: {aggregate['false_alarm_seconds']:.1f}",
    ])
    if report["strict_errors"]:
        lines.append("")
        lines.append("strict validation errors:")
        lines.extend(f"- {error}" for error in report["strict_errors"])
    lines.append("")
    lines.append("PASS" if report["passed"] else "FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score VoxTerm diarization speaker attribution")
    parser.add_argument("manifest", type=Path, help="benchmark manifest JSON")
    parser.add_argument("--strict", action="store_true",
                        help="require issue #87 corpus shape: >=4 sessions, each 3-5 speakers")
    parser.add_argument("--target", type=float, default=None,
                        help="segment accuracy target, default 0.90 or manifest target_segment_accuracy")
    parser.add_argument("--fallback-turn-seconds", type=float, default=DEFAULT_FALLBACK_TURN_SECONDS,
                        help="duration for final/no-end live turns, default 2.0")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        report = score_manifest(
            args.manifest,
            strict=args.strict,
            target=args.target,
            fallback_turn_seconds=args.fallback_turn_seconds,
        )
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))

    if report["passed"]:
        return 0
    return 2 if report["strict_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
