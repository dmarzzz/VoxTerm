# Diarization Fix — Investigation & Changes

## Problem

April 7 recording (977 lines, ~3.5 hours) tagged 100% as "Speaker 1" despite two people speaking.

## Root Cause Chain

```
Model file missing or download fails
  → OnnxSpeakerEmbedder.load() raises FileNotFoundError
    → proxy.load() catches it, logs warning, tries fallback
      → Fallback also fails silently
        → self._loaded stays False
          → identify() returns ("Speaker 1", 1) on EVERY call
            → All speakers labeled "Speaker 1"
```

## Bugs Fixed

### 1. Silent fallback in identify() (CRITICAL)
**File:** `audio/diarization/engine.py:229-234`
- identify() returned ("Speaker 1", 1) with NO logging when model wasn't loaded
- Added explicit warning logs so the failure is visible

### 2. Silent fallback in identify_segments() (CRITICAL)
**File:** `audio/diarization/engine.py:600-610`
- Same silent fallback pattern
- Added warning log

### 3. Proxy fallback doesn't set _loaded (CRITICAL)
**File:** `audio/diarization/proxy.py:424-438`
- _fallback_to_inprocess() catches engine.load() failure but only sets _loaded=False
- No log message on final failure — user never knows diarization is broken
- Added error logging and a visible TUI warning

### 4. Party mode audio merging (HIGH)
**File:** `tui/app.py:944`
- When local_audio is empty, falls back to merged mono audio for diarization
- Merged audio destroys speaker identity — all embeddings similar → single cluster
- Added guard: skip diarization entirely if only merged audio available

## Threshold Notes (not changed)

Current values in engine.py:28-46 are reasonable for distinct speakers:
- MATCH_THRESHOLD = 0.55
- NEW_SPEAKER_THRESHOLD = 0.45
- SCD_CHANGE_THRESHOLD = 0.6

The real issue was the model not loading, not the thresholds.

## Testing

1. Check model exists: `ls ~/.cache/3dspeaker/eres2net_large/`
2. Run with debug mode (D key) to see diarization logs
3. Record with 2+ speakers and verify distinct labels
