"""Batch / offline transcription of existing audio files.

Reuses the live pipeline (transcriber + diarization + speaker store) headlessly,
so a whole WAV file produces the same transcript+speaker output as live capture.
"""

from .core import transcribe_file, build_pipeline, load_audio_16k_mono

__all__ = ["transcribe_file", "build_pipeline", "load_audio_16k_mono"]
