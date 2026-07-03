# Diarization attribution benchmark

Issue #87 is not considered proven until a labeled field corpus exists and passes
the same scorer every time. The scorer lives at `scripts/score_diarization.py` and
works from saved VoxTerm `*-events.jsonl` files plus human reference labels.

## Corpus requirements

Use `--strict` for the issue #87 acceptance target:

- at least 4 sessions
- each session has 3 to 5 distinct reference speakers
- at least one session should be the real 5-person jam-room shape
- any 5-speaker session must meet the same segment-accuracy target
- target: `segment_accuracy >= 0.90`

The current repo fixtures are useful for model smoke tests, but they are not enough
for this target: there are fewer than 4 committed sessions, and they are not field
recordings from the intended room.

## Reference labels

Create one reference JSON file per session:

```json
{
  "session_id": "jam-001",
  "speakers": ["alice", "bob", "cara", "dan", "eve"],
  "segments": [
    {"start": 0.00, "end": 2.30, "speaker": "alice", "text": "optional transcript"},
    {"start": 2.30, "end": 4.10, "speaker": "bob", "text": "optional transcript"}
  ]
}
```

`start` and `end` are seconds from the beginning of the recording. Speaker names are
human labels for the reference only; VoxTerm output can keep its normal `Speaker 1`
or `speaker_id` cluster labels.

## Manifest

Put the event logs and labels behind a manifest. Relative paths resolve from the
manifest location.

```json
{
  "target_segment_accuracy": 0.9,
  "sessions": [
    {
      "id": "jam-001",
      "events": "events/jam-001-events.jsonl",
      "reference": "labels/jam-001-reference.json"
    }
  ]
}
```

Run:

```bash
python scripts/score_diarization.py benchmarks/diarization/manifest.json --strict
python scripts/score_diarization.py benchmarks/diarization/manifest.json --strict --json > diarization-results.json
```

The process exits nonzero when the strict corpus shape is missing or when segment
accuracy is below the target.

## Methodology

The event log is treated as the prediction. Text events with `audio_offset` and
`audio_end` are preferred because those are true recording offsets from the headless
file transcriber. Live logs without audio offsets fall back to wall-clock deltas from
the session start and should be treated as weaker evidence.

VoxTerm diarization speaker IDs are arbitrary clusters, so `sid:1` is not expected
to equal any human name. The scorer builds a one-to-one mapping from VoxTerm clusters
to reference speakers that maximizes overlapped reference duration, then scores each
reference segment by the prediction with the most overlap. A segment is correct only
when the mapped prediction speaker matches the reference speaker.

The headline metric is segment accuracy:

```text
correct reference segments / total reference segments
```

The report also includes segment coverage, duration accuracy, and false-alarm
seconds so failures are easier to diagnose.

## Collecting field evidence

1. Record at least 4 consented sessions with 3 to 5 speakers.
2. Include one realistic 5-person jam session, not only clean turn-taking.
3. Process each recording through VoxTerm so every session has a `*-events.jsonl`
   file. For file-based runs, prefer the headless transcriber because it writes
   `audio_offset` and `audio_end` on each text event.
4. Label the same audio in seconds with human speaker names.
5. Run the strict manifest command and keep the JSON report with the benchmark
   evidence.

Do not commit private raw audio or labels unless the speakers explicitly agreed to
that. A PR can still include the manifest shape, scorer output, and redacted notes.

## Failure modes to document with every run

- overlapping speech: the dominant predicted segment may be correct while a quieter
  simultaneous speaker is missed
- short backchannels: sub-second "yeah" or "mm-hmm" turns are easy to merge into the
  surrounding speaker
- speaker movement: people walking around or turning away from the mic can split one
  person into multiple clusters
- similar voices: same-gender or same-timbre speakers can be confused without enough
  clean anchor speech
- room acoustics: fan noise, music, laptop speakers, and reverberation reduce both
  ASR and embedding quality
- live-log timing: logs without `audio_offset` rely on wall-clock deltas and are less
  precise for scoring than file-transcribed sessions
- cluster count drift: extra VoxTerm clusters are allowed only if the best mapping
  still attributes at least 90% of reference segments correctly
