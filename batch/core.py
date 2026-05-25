"""File → transcript core (issue #144).

Loads an audio file, segments it with VAD, and runs each speech region through
the same transcriber + diarization + cross-session speaker matching the live TUI
uses. Writes a session markdown identical in format to a live recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import (
    SAMPLE_RATE, SESSIONS_DIR, DEFAULT_MODEL, DEFAULT_LANGUAGE,
    AVAILABLE_MODELS, QWEN3_MODELS, FASTER_WHISPER_MODELS, AVAILABLE_LANGUAGES,
)
from audio.transcriber import (
    Qwen3Transcriber, FasterWhisperTranscriber, WhisperTranscriber,
)
from audio.diarization.proxy import DiarizationProxy
from audio.speakers.store import SpeakerStore
from audio.vad import SileroVAD

@dataclass
class Pipeline:
    transcriber: object
    diarizer: DiarizationProxy
    store: SpeakerStore
    vad: SileroVAD
    model_name: str
    language: str | None


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Read a WAV file → float32 mono at SAMPLE_RATE (16 kHz)."""
    rate, data = wavfile.read(str(path))
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, rate)
        data = resample_poly(data, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return data


def make_transcriber(model_name: str, language: str | None):
    repo = AVAILABLE_MODELS[model_name]
    if model_name in QWEN3_MODELS:
        return Qwen3Transcriber(model=repo, language=language)
    if model_name in FASTER_WHISPER_MODELS:
        return FasterWhisperTranscriber(model=repo, language=language)
    return WhisperTranscriber(model=repo)


def build_pipeline(model_name: str = DEFAULT_MODEL,
                   language: str | None = DEFAULT_LANGUAGE) -> Pipeline:
    """Load every engine once so multiple files can share them."""
    transcriber = make_transcriber(model_name, language)
    transcriber.load()
    diarizer = DiarizationProxy()
    diarizer.load()
    store = SpeakerStore()
    store.open()
    vad = SileroVAD()
    return Pipeline(transcriber, diarizer, store, vad, model_name, language)


def transcribe_file(
    path: str | Path,
    pipeline: Pipeline | None = None,
    model_name: str = DEFAULT_MODEL,
    language: str | None = DEFAULT_LANGUAGE,
    on_line: Callable[[float, str, str], None] | None = None,
) -> Path:
    """Transcribe one audio file and write a session .md. Returns its path."""
    path = Path(path)
    p = pipeline or build_pipeline(model_name, language)
    p.diarizer.reset_session()

    audio = load_audio_16k_mono(path)
    profile_map: dict[int, str] = {}   # session speaker id → DB profile id
    seg_counts: dict[int, int] = {}
    lines: list[tuple[float, str, str]] = []

    for start, end in p.vad.get_speech_segments(audio):
        chunk = audio[start:end]
        text = p.transcriber.transcribe(chunk).get("text", "").strip()
        if not text:
            continue
        _label, sid = p.diarizer.identify(chunk)
        _match_speaker(p.diarizer, p.store, sid, profile_map)
        label = p.diarizer.get_speaker_name(sid)
        seg_counts[sid] = seg_counts.get(sid, 0) + 1
        ts = start / SAMPLE_RATE
        lines.append((ts, label, text))
        if on_line:
            on_line(ts, label, text)

    out = _write_transcript(path, lines, p.model_name, p.language, len(audio) / SAMPLE_RATE)
    _record_session(p.store, path, profile_map, seg_counts)
    return out


def _match_speaker(diarizer, store, sid: int, profile_map: dict[int, str]) -> None:
    """Cross-session match a session speaker to a stored profile (high tier only)."""
    if sid <= 0 or diarizer.is_matched(sid):
        return
    centroid = diarizer.get_session_centroid(sid)
    if centroid is None:
        return
    match = store.classify_match(centroid)
    if match.tier == "high":
        diarizer.set_speaker_name(sid, match.name)
        diarizer.mark_matched(sid)
        profile_map[sid] = match.profile_id


def _fmt_ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _write_transcript(src: Path, lines, model_name, language, duration_sec) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SESSIONS_DIR / (src.stem + "-transcript.md")
    lang = AVAILABLE_LANGUAGES.get(language, language or "auto")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# VoxTerm Transcript\n\n")
        f.write(f"- **Source:** {src.name}\n")
        f.write(f"- **Duration:** {_fmt_ts(duration_sec)}\n")
        f.write(f"- **Model:** {model_name}\n")
        f.write(f"- **Language:** {lang}\n\n---\n\n")
        for ts, label, text in lines:
            f.write(f"**[{_fmt_ts(ts)}]** **{label}:** {text}\n\n")
    return out


def _record_session(store, src: Path, profile_map: dict[int, str], seg_counts) -> None:
    session_id = f"file:{src.stem}"
    for sid, pid in profile_map.items():
        store.record_session_speaker(session_id, pid, sid, seg_counts.get(sid, 0))
