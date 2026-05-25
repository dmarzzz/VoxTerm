"""CLI: `python -m batch FILE...`  or  `python -m batch --rode`."""

from __future__ import annotations

import argparse

from config import DEFAULT_MODEL, DEFAULT_LANGUAGE
from .core import transcribe_file, build_pipeline, _fmt_ts
from .rode import import_recordings


def main() -> None:
    ap = argparse.ArgumentParser(prog="batch", description="VoxTerm batch/file transcription")
    ap.add_argument("files", nargs="*", help="audio files to transcribe")
    ap.add_argument("--rode", action="store_true",
                    help="detect & import recordings from a plugged-in Rode mic")
    ap.add_argument("--no-transcribe", action="store_true",
                    help="(with --rode) copy/sync only, skip transcription")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-l", "--language", default=DEFAULT_LANGUAGE)
    args = ap.parse_args()

    if args.rode:
        results = import_recordings(
            transcribe=not args.no_transcribe, model_name=args.model, language=args.language,
        )
        if not results:
            print("No Rode recordings found. Is the mic plugged in and mounted?")
        for src, action, transcript in results:
            line = f"[{action}] {src.name}"
            if transcript:
                line += f" → {transcript}"
            print(line)
        return

    if not args.files:
        ap.error("provide audio files or --rode")

    pipeline = build_pipeline(args.model, args.language)
    for fpath in args.files:
        print(f"transcribing {fpath} …")
        out = transcribe_file(
            fpath, pipeline=pipeline,
            on_line=lambda ts, label, text: print(f"  [{_fmt_ts(ts)}] {label}: {text}"),
        )
        print(f"→ {out}")


if __name__ == "__main__":
    main()
