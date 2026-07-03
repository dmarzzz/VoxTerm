"""Offline audio-file transcription using VoxTerm's headless GUI pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from config import SESSIONS_DIR
from gui import transcribe

Progress = Callable[[float, str], None]


def transcribe_file(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    model: str | None = None,
    language: str = "en",
    diarize: bool = True,
    progress: Progress | None = None,
) -> dict:
    """Transcribe one existing audio file and return artifact paths.

    The heavy lifting stays in ``gui.transcribe`` so the batch path reuses the
    same VAD, transcriber, diarizer, event log, and transcript writer as the
    reviewed GUI flow.
    """

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    if not src.is_file():
        raise IsADirectoryError(src)

    target_dir = Path(out_dir) if out_dir is not None else SESSIONS_DIR
    audio = transcribe.load_wav_16k_mono(src)
    result = transcribe.transcribe_audio(
        audio,
        target_dir,
        model=model,
        language=language,
        progress=progress,
        diarize=diarize,
    )
    result["source_path"] = str(src)
    result["diarize"] = diarize
    return result


def transcribe_files(
    paths: Iterable[str | Path],
    *,
    out_dir: str | Path | None = None,
    model: str | None = None,
    language: str = "en",
    diarize: bool = True,
) -> list[dict]:
    """Transcribe multiple files sequentially with the cached headless engines."""

    results = []
    for path in paths:
        results.append(
            transcribe_file(
                path,
                out_dir=out_dir,
                model=model,
                language=language,
                diarize=diarize,
            )
        )
    return results
