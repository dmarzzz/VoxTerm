"""CLI for offline file transcription and Rode imports."""

from __future__ import annotations

import argparse
from pathlib import Path

from gui.transcribe import gui_default_model
from .core import transcribe_file
from .rode import import_recordings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voxterm-batch",
        description="Transcribe existing audio files or import Rode recorder WAVs.",
    )
    parser.add_argument("files", nargs="*", help="audio files to transcribe")
    parser.add_argument("--rode", action="store_true", help="scan mounted Rode volumes")
    parser.add_argument(
        "--mount-root",
        action="append",
        default=[],
        help="extra mount root to scan for Rode WAVs; repeatable",
    )
    parser.add_argument("--no-transcribe", action="store_true", help="copy only with --rode")
    parser.add_argument("--out-dir", type=Path, default=None, help="transcript output directory")
    parser.add_argument("--import-dir", type=Path, default=None, help="Rode copy destination")
    parser.add_argument("--manifest", type=Path, default=None, help="Rode import manifest path")
    parser.add_argument("--model", default=gui_default_model())
    parser.add_argument("--language", default="en")
    parser.add_argument("--no-diarize", action="store_true")
    args = parser.parse_args(argv)

    if args.rode:
        results = import_recordings(
            roots=args.mount_root or None,
            import_dir=args.import_dir,
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            transcribe=not args.no_transcribe,
            model=args.model,
            language=args.language,
            diarize=not args.no_diarize,
        )
        if not results:
            print("No Rode recordings found.")
            return 0
        for result in results:
            line = f"[{result.action}] {result.source_path.name} -> {result.local_path}"
            if result.transcript_path:
                line += f" -> {result.transcript_path}"
            print(line)
        return 0

    if not args.files:
        parser.error("provide audio files or --rode")

    for file_path in args.files:
        print(f"transcribing {file_path}")

        def progress(frac: float, message: str) -> None:
            print(f"  {int(frac * 100):3d}% {message}", flush=True)

        result = transcribe_file(
            file_path,
            out_dir=args.out_dir,
            model=args.model,
            language=args.language,
            diarize=not args.no_diarize,
            progress=progress,
        )
        print(f"transcript -> {result['transcript_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
