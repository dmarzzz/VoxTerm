#!/usr/bin/env python3
"""VOXTERM — Cyberpunk TUI Voice Transcription Engine"""

from __future__ import annotations

import gc
import resource
import sys
import os
import subprocess
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Internal runtime defaults — prevent known framework conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen
from textual import work

from widgets.header import CyberHeader
from widgets.waveform import WaveformWidget, _make_style
from widgets.transcript import TranscriptPanel
from audio.capture import AudioCapture
from audio.buffer import AudioBuffer
from audio.system_capture import SystemCapture
from transcriber.engine import Qwen3Transcriber, WhisperTranscriber
from diarization.engine import DiarizationEngine
from config import (
    SAMPLE_RATE, CHUNK_SIZE, WAVEFORM_FPS,
    SILENCE_THRESHOLD, SILENCE_TRIGGER_SECONDS,
    MAX_BUFFER_SECONDS, MIN_BUFFER_SECONDS,
    DEFAULT_MODEL, AVAILABLE_MODELS, QWEN3_MODELS,
    DEFAULT_LANGUAGE, AVAILABLE_LANGUAGES,
    LIVE_DIR,
)

# Session save directory
SESSIONS_DIR = Path.home() / "Documents" / "voxterm"
# Persistent state file (remembers last-used model across launches)
STATE_FILE = SESSIONS_DIR / ".state.json"
# Crash logs — always written, not gated by debug mode
CRASH_DIR = SESSIONS_DIR / ".crashes"


def _load_state() -> dict:
    """Load persisted state from disk. Returns {} on any failure."""
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    """Write state dict to disk. Silently ignores errors."""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


class ModelSelectScreen(ModalScreen):
    """Modal for selecting a whisper model."""

    DEFAULT_CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model-dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: heavy #6644cc;
        border-title-color: #aa66ff;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #model-list {
        height: auto;
        max-height: 14;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #model-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #model-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_model: str):
        super().__init__()
        self._current = current_model

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog") as dialog:
            dialog.border_title = "SELECT MODEL"
            options = []
            for name, repo in AVAILABLE_MODELS.items():
                label = f"  {'▸ ' if name == self._current else '  '}{name:12s}  {repo}"
                options.append(Option(label, id=name))
            yield OptionList(*options, id="model-list")
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="model-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#model-list", OptionList)
        for idx, name in enumerate(AVAILABLE_MODELS):
            if name == self._current:
                option_list.highlighted = idx
                break

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self):
        self.dismiss(None)


class LanguageSelectScreen(ModalScreen):
    """Modal for selecting transcription language."""

    DEFAULT_CSS = """
    LanguageSelectScreen {
        align: center middle;
    }
    #lang-dialog {
        width: 50;
        height: auto;
        max-height: 22;
        border: heavy #cc6644;
        border-title-color: #ffaa66;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #lang-list {
        height: auto;
        max-height: 16;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #lang-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #lang-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_lang: str | None):
        super().__init__()
        self._current = current_lang or "en"

    def compose(self) -> ComposeResult:
        with Vertical(id="lang-dialog") as dialog:
            dialog.border_title = "SELECT LANGUAGE"
            options = []
            for code, name in AVAILABLE_LANGUAGES.items():
                marker = "▸ " if code == self._current else "  "
                label = f"  {marker}{code:5s}  {name}"
                options.append(Option(label, id=code))
            yield OptionList(*options, id="lang-list")
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="lang-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        option_list = self.query_one("#lang-list", OptionList)
        for idx, code in enumerate(AVAILABLE_LANGUAGES):
            if code == self._current:
                option_list.highlighted = idx
                break

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self):
        self.dismiss(None)


class ExportScreen(ModalScreen):
    """Modal for exporting transcript to a destination."""

    DEFAULT_CSS = """
    ExportScreen {
        align: center middle;
    }
    #export-dialog {
        width: 50;
        height: auto;
        max-height: 12;
        border: heavy #44cc66;
        border-title-color: #66ff88;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #export-list {
        height: auto;
        max-height: 6;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #export-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    #export-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="export-dialog") as dialog:
            dialog.border_title = "EXPORT TRANSCRIPT"
            yield OptionList(
                Option("  Save to file", id="file"),
                Option("  Copy to clipboard", id="clipboard"),
                id="export-list",
            )
            yield Static(
                " [#607080]ENTER[/] select  [#607080]ESC[/] cancel",
                id="export-hint",
                markup=True,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self):
        self.dismiss(None)


class VoxTerm(App):
    """Cyberpunk voice transcription TUI."""

    CSS_PATH = "cyberpunk.tcss"
    TITLE = "VOXTERM"

    BINDINGS = [
        Binding("r", "toggle_recording", "Record/Pause"),
        Binding("m", "switch_model", "Model"),
        Binding("l", "switch_language", "Language"),
        Binding("s", "export_transcript", "Export"),
        Binding("d", "toggle_debug", "Debug"),
        Binding("c", "clear_transcript", "Clear"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, transcriber=None, model_name="qwen3-0.6b", language="en"):
        super().__init__()
        self.audio_capture = AudioCapture()
        self.system_capture = SystemCapture()
        self.audio_buffer = AudioBuffer()
        self.transcriber = transcriber or Qwen3Transcriber()
        self.diarizer = DiarizationEngine()
        self._model_name = model_name
        self._language = language
        self._is_qwen3 = model_name in QWEN3_MODELS
        self._recording = False
        self._had_speech = False
        self._silence_chunks = 0
        self._transcribing = threading.Event()  # set = busy, clear = idle
        self._transcribe_started: float = 0.0
        self._debug = False
        self._last_dbg: float = 0.0
        self._transcribe_count = 0
        self._model_loaded = transcriber is not None and transcriber.is_loaded
        self._diarizer_loaded = False
        self._system_audio_notified = False
        self._last_saved_at: float | None = None
        self._session_start = datetime.now()
        self._live_file: Path | None = None
        self._live_header_written = False

    def compose(self) -> ComposeResult:
        yield CyberHeader()
        with Vertical(id="main-container"):
            yield WaveformWidget()
            yield TranscriptPanel()
            yield Static(
                "  [bold #607080]● IDLE[/]    [#00ffcc]loading...[/]",
                id="telemetry",
                markup=True,
            )
        yield Static(
            " [bold #00e5ff]\\[R][/][#607080] Record  [/]"
            "[bold #00e5ff]\\[M][/][#607080] Model  [/]"
            "[bold #00e5ff]\\[L][/][#607080] Lang  [/]"
            "[bold #00e5ff]\\[S][/][#607080] Export  [/]"
            "[bold #00e5ff]\\[C][/][#607080] Clear  [/]"
            "[bold #00e5ff]\\[Q][/][#607080] Quit[/]",
            id="footer-bar",
            markup=True,
        )

    def on_mount(self) -> None:
        if self._model_loaded:
            transcript = self.query_one(TranscriptPanel)
            transcript.system_message("VOXTERM engine online")
            transcript.system_message(f"model loaded: {self._model_name}")
            self._update_telemetry()
            self._start_audio_timer()
            self._load_diarizer()
        else:
            self.query_one(TranscriptPanel).system_message("initializing VOXTERM engine...")
            self._start_audio_timer()
            self._load_model()

    @property
    def _chunk_duration(self) -> float:
        return CHUNK_SIZE / SAMPLE_RATE

    def _update_telemetry(self):
        # Status dot
        if self._recording:
            status = "[bold #00ff88]● REC[/]"
        elif self._model_loaded:
            status = "[bold #607080]● IDLE[/]"
        else:
            status = "[bold #607080]● LOADING[/]"

        model_text = self._model_name if self._model_loaded else "loading..."
        lang_text = AVAILABLE_LANGUAGES.get(self._language, self._language) if self._language else "auto"

        spk_count = self.diarizer.num_speakers if self._diarizer_loaded else 0
        spk_text = f"    [#aa88ff]{spk_count} speakers[/]" if spk_count > 0 else ""

        # Auto-save indicator
        if self._last_saved_at is not None:
            ago = int(time.time() - self._last_saved_at)
            if ago < 60:
                saved_text = f"    [#00ff88]saved {ago}s ago[/]"
            elif ago < 3600:
                saved_text = f"    [#ffaa00]saved {ago // 60}m ago[/]"
            else:
                saved_text = f"    [#ff6600]saved {ago // 3600}h ago[/]"
        else:
            saved_text = ""

        self.query_one("#telemetry", Static).update(
            f"  {status}"
            f"    [#00ffcc]{model_text}[/]"
            f"    [#ffaa66]{lang_text}[/]"
            f"{spk_text}"
            f"{saved_text}"
        )

    def _start_audio_timer(self):
        self.set_interval(1.0 / WAVEFORM_FPS, self._process_audio, name="audio_timer")
        self.set_interval(5.0, self._refresh_telemetry, name="telemetry_timer")
        self.set_interval(60.0, self._periodic_gc, name="gc_timer")

    def _refresh_telemetry(self):
        """Periodic refresh so the 'saved: Xs ago' counter stays current."""
        if self._last_saved_at is not None:
            self._update_telemetry()

    def _periodic_gc(self):
        """Prevent memory fragmentation during long sessions."""
        gc.collect()

    def _write_crash_dump(self, context: str, exc: BaseException | None = None):
        """Write a diagnostic dump to disk. Always runs, not gated by debug."""
        try:
            CRASH_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now()
            filename = ts.strftime("%Y-%m-%d_%H%M%S") + ".log"
            uptime = (ts - self._session_start).total_seconds()

            # Memory usage (macOS: ru_maxrss is in bytes)
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss_bytes / (1024 * 1024)

            # Style cache stats
            cache = _make_style.cache_info()

            # Transcript entry count
            try:
                entry_count = len(self.query_one(TranscriptPanel).get_entries())
            except Exception:
                entry_count = -1

            # Diarizer state
            spk_count = self.diarizer.num_speakers if self._diarizer_loaded else 0

            lines = [
                f"VOXTERM CRASH DUMP",
                f"{'=' * 60}",
                f"timestamp:        {ts.isoformat()}",
                f"uptime:           {uptime:.0f}s ({uptime / 60:.1f}m)",
                f"context:          {context}",
                f"",
                f"-- error --",
                f"type:             {type(exc).__name__ if exc else 'N/A'}",
                f"message:          {exc}",
                f"traceback:",
                traceback.format_exc() if exc else "  N/A",
                f"",
                f"-- runtime state --",
                f"recording:        {self._recording}",
                f"is_transcribing:  {self._transcribing.is_set()}",
                f"transcribe_count: {self._transcribe_count}",
                f"model:            {self._model_name}",
                f"model_loaded:     {self._model_loaded}",
                f"diarizer_loaded:  {self._diarizer_loaded}",
                f"language:         {self._language}",
                f"had_speech:       {self._had_speech}",
                f"silence_chunks:   {self._silence_chunks}",
                f"sys_capture:      active={self.system_capture.is_active} msg={self.system_capture.status_message}",
                f"",
                f"-- memory --",
                f"peak_rss_mb:      {rss_mb:.1f}",
                f"audio_buf_dur:    {self.audio_buffer.duration:.2f}s",
                f"style_cache:      hits={cache.hits} misses={cache.misses} size={cache.currsize}/{cache.maxsize}",
                f"transcript_entries: {entry_count}",
                f"speakers:         {spk_count}",
                f"gc_counts:        {gc.get_count()}",
            ]

            (CRASH_DIR / filename).write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass  # crash dump must never itself crash the app

    def _process_audio(self):
        """Read audio chunks, update waveform, check transcription trigger."""
        try:
            self._process_audio_inner()
        except Exception as e:
            self._write_crash_dump("_process_audio", e)
            raise

    @staticmethod
    def _mix_chunks(mic: list[np.ndarray], sys: list[np.ndarray]) -> list[np.ndarray]:
        """Time-aligned addition of mic and system audio chunks."""
        mixed = []
        n = min(len(mic), len(sys))
        for i in range(n):
            mixed.append(np.clip(mic[i] + sys[i], -1.0, 1.0))
        mixed.extend(mic[n:])
        mixed.extend(sys[n:])
        return mixed

    def _process_audio_inner(self):
        waveform = self.query_one(WaveformWidget)

        if not self._recording:
            waveform.tick()
            return

        mic_chunks = self.audio_capture.drain()
        sys_chunks = self.system_capture.drain()
        chunks = self._mix_chunks(mic_chunks, sys_chunks) if sys_chunks else mic_chunks
        if not chunks:
            waveform.tick()
            return

        for chunk in chunks:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            waveform.push_samples(chunk)

            if rms < SILENCE_THRESHOLD:
                self._silence_chunks += 1
                # Only buffer silence if we're in an active speech segment
                if self._had_speech:
                    self.audio_buffer.append(chunk)
            else:
                self._silence_chunks = 0
                self._had_speech = True
                self.audio_buffer.append(chunk)

        waveform.tick()

        # Check transcription trigger
        silence_duration = self._silence_chunks * self._chunk_duration
        buffer_duration = self.audio_buffer.duration

        if self._transcribing.is_set():
            # Watchdog: force-reset if stuck for >10s
            elapsed = time.time() - self._transcribe_started if self._transcribe_started else 0
            if elapsed > 10:
                self._transcribing.clear()
                if self._debug:
                    self.query_one(TranscriptPanel).system_message(
                        f"[watchdog] reset after {elapsed:.0f}s"
                    )
            return

        if self._debug:
            now = time.time()
            if buffer_duration > 0.5 and now - self._last_dbg > 3:
                self._last_dbg = now
                self.query_one(TranscriptPanel).system_message(
                    f"[dbg] buf={buffer_duration:.1f}s sil={silence_duration:.1f}s "
                    f"speech={self._had_speech}"
                )

        if self._had_speech and silence_duration > SILENCE_TRIGGER_SECONDS and buffer_duration > MIN_BUFFER_SECONDS:
            self._trigger_transcription()
        elif buffer_duration >= MAX_BUFFER_SECONDS:
            self._trigger_transcription()

    def _trigger_transcription(self):
        """Send accumulated audio to transcription worker."""
        self._transcribing.set()
        self._transcribe_started = time.time()
        self._silence_chunks = 0
        audio = self.audio_buffer.get_and_clear()
        if len(audio) < int(SAMPLE_RATE * MIN_BUFFER_SECONDS):
            self._transcribing.clear()
            return

        self._had_speech = False
        self._transcribe_audio(audio)

    @work(thread=True, group="transcription")
    def _transcribe_audio(self, audio: np.ndarray):
        try:
            if self._debug:
                duration = len(audio) / SAMPLE_RATE
                self.call_from_thread(
                    self.query_one(TranscriptPanel).system_message,
                    f"[dbg] transcribing {duration:.1f}s audio..."
                )
            # 1. Transcribe (Qwen3: ~100ms, Whisper: ~2-4s)
            result = self.transcriber.transcribe(audio)

            # Periodic GC to prevent MLX/PyTorch memory buildup
            self._transcribe_count += 1
            if self._transcribe_count % 20 == 0:
                gc.collect()

            text = result.get("text", "")

            # 2. Speaker ID AFTER transcription (non-blocking)
            speaker_label, speaker_id = "", 0
            if text and self._diarizer_loaded:
                try:
                    speaker_label, speaker_id = self.diarizer.identify(
                        audio.copy()
                    )
                except Exception:
                    pass

            if text:
                self.call_from_thread(
                    self._on_transcription, text, speaker_label, speaker_id
                )
        except Exception as e:
            self._write_crash_dump("_transcribe_audio", e)
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"transcription error: {e}"
            )
        finally:
            # ALWAYS unblock — even if worker is cancelled or crashes
            self._transcribing.clear()

    def _on_transcription(self, text: str, speaker: str = "", speaker_id: int = 0):
        self.query_one(TranscriptPanel).add_transcript(text, speaker, speaker_id)
        self._append_live_transcript(text, speaker, speaker_id)
        self._update_telemetry()

    # ── background auto-save ───────────────────────────────────

    def _append_live_transcript(self, text: str, speaker: str, speaker_id: int):
        """Append a single transcript line to the live file on disk."""
        try:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            if self._live_file is None:
                fname = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
                self._live_file = LIVE_DIR / fname
                self._live_header_written = False

            with open(self._live_file, "a", encoding="utf-8") as f:
                if not self._live_header_written:
                    f.write(f"# VOXTERM Transcript\n\n")
                    f.write(f"- **Date:** {self._session_start.strftime('%Y-%m-%d')}\n")
                    f.write(f"- **Time:** {self._session_start.strftime('%H:%M:%S')}\n")
                    f.write(f"- **Model:** {self._model_name}\n\n---\n\n")
                    self._live_header_written = True

                ts = datetime.now().strftime("%H:%M:%S")
                if speaker:
                    f.write(f"**[{ts}]** **{speaker}:** {text}\n\n")
                else:
                    f.write(f"**[{ts}]** {text}\n\n")
            self._last_saved_at = time.time()
        except Exception:
            pass  # never block transcription on I/O failure

    @work(thread=True, group="model_loading")
    def _load_model(self):
        self.call_from_thread(
            self.query_one(TranscriptPanel).system_message,
            "loading whisper model (first run downloads ~461MB)..."
        )
        try:
            self.transcriber.load()
            self.call_from_thread(self._on_model_loaded)
        except Exception as e:
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"model load failed: {e}"
            )

    def _on_model_loaded(self):
        self._model_loaded = True
        transcript = self.query_one(TranscriptPanel)
        transcript.system_message(f"model loaded: {self._model_name}")
        if not self._diarizer_loaded:
            self._load_diarizer()
        else:
            transcript.system_message("press [R] to start recording")
        self._update_telemetry()

    @work(thread=True, group="diarizer_loading")
    def _load_diarizer(self):
        self.call_from_thread(
            self.query_one(TranscriptPanel).system_message,
            "loading speaker identification model..."
        )
        try:
            self.diarizer.load()
            self.call_from_thread(self._on_diarizer_loaded)
        except Exception as e:
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"speaker ID unavailable: {e}"
            )
            self.call_from_thread(self._on_diarizer_fallback)

    def _on_diarizer_loaded(self):
        self._diarizer_loaded = True
        self.query_one(TranscriptPanel).system_message("speaker identification online")
        self.query_one(TranscriptPanel).system_message("press [R] to start recording")
        self._update_telemetry()

    def _on_diarizer_fallback(self):
        self.query_one(TranscriptPanel).system_message("press [R] to start recording")

    # ── actions ─────────────────────────────────────────────────

    def action_toggle_recording(self):
        if not self._model_loaded:
            self.query_one(TranscriptPanel).system_message("model still loading, please wait...")
            return

        waveform = self.query_one(WaveformWidget)
        transcript = self.query_one(TranscriptPanel)
        if self._recording:
            self._recording = False
            self.audio_capture.stop()
            self.system_capture.stop()
            waveform.set_recording(False)
            transcript.system_message("recording paused")
        else:
            self._recording = True
            try:
                self.audio_capture.start()
                waveform.set_recording(True)
                transcript.system_message("recording started")
            except Exception as e:
                self._recording = False
                waveform.set_recording(False)
                transcript.system_message(
                    f"microphone error: {e} — grant Terminal mic access in System Settings > Privacy"
                )
                self._update_telemetry()
                return

            # Start system audio capture (non-fatal if unavailable)
            try:
                self.system_capture.start()
            except Exception:
                pass
        self._update_telemetry()

    def action_switch_model(self):
        if self._transcribing.is_set():
            self.query_one(TranscriptPanel).system_message("wait for transcription to finish...")
            return
        was_recording = self._recording
        if was_recording:
            self._recording = False
            self.audio_capture.stop()

        def on_model_selected(model_key):
            if model_key is None or model_key == self._model_name:
                if was_recording:
                    self.action_toggle_recording()
                return
            self._swap_model(model_key)

        self.push_screen(ModelSelectScreen(self._model_name), on_model_selected)

    def action_switch_language(self):
        def on_lang_selected(lang_code):
            if lang_code is None or lang_code == self._language:
                return
            self._language = lang_code
            lang_name = AVAILABLE_LANGUAGES.get(lang_code, lang_code)
            # Update transcriber language if it's Qwen3
            if self._is_qwen3 and hasattr(self.transcriber, '_language'):
                self.transcriber._language = lang_code
            _save_state({"last_model": self._model_name, "last_language": lang_code})
            self.query_one(TranscriptPanel).system_message(f"language set to {lang_name}")
            self._update_telemetry()

        self.push_screen(LanguageSelectScreen(self._language), on_lang_selected)

    def _swap_model(self, model_key: str):
        self._model_loaded = False
        self._model_name = model_key
        self._update_telemetry()
        # Free old model memory before loading the new one
        self.transcriber._model = None
        self.query_one(TranscriptPanel).system_message(
            f"switching to {model_key} (may take a minute)..."
        )
        self._do_swap(model_key)

    @work(thread=True, exclusive=True, group="model_loading")
    def _do_swap(self, model_key: str):
        repo = AVAILABLE_MODELS[model_key]
        try:
            if model_key in QWEN3_MODELS:
                new_transcriber = Qwen3Transcriber(model=repo, language=self._language)
            else:
                new_transcriber = WhisperTranscriber(model=repo)
            new_transcriber.load()
            self.call_from_thread(self._on_swap_done, new_transcriber, model_key)
        except Exception as e:
            self.call_from_thread(
                self._on_swap_error, f"model switch failed: {e}"
            )

    def _on_swap_done(self, transcriber, model_key):
        self.transcriber = transcriber
        self._model_name = model_key
        self._is_qwen3 = model_key in QWEN3_MODELS
        self._model_loaded = True
        _save_state({"last_model": model_key, "last_language": self._language})
        transcript = self.query_one(TranscriptPanel)
        transcript.system_message(f"model loaded: {model_key}")
        transcript.system_message("press [R] to start recording")
        self._update_telemetry()

    def _on_swap_error(self, msg: str):
        self.query_one(TranscriptPanel).system_message(msg)
        self._model_loaded = True
        self._update_telemetry()

    def action_export_transcript(self):
        """Open export modal to choose destination."""
        transcript = self.query_one(TranscriptPanel)
        if not transcript.get_entries():
            transcript.system_message("nothing to export")
            return

        def on_export_selected(destination):
            if destination is None:
                return
            if destination == "file":
                self._export_to_file()
            elif destination == "clipboard":
                self._export_to_clipboard()

        self.push_screen(ExportScreen(), on_export_selected)

    def _export_to_file(self):
        """Promote live file to final transcript."""
        transcript = self.query_one(TranscriptPanel)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        filename = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
        filepath = SESSIONS_DIR / filename

        # Write the full markdown (cleaner than the append-mode live file)
        md = transcript.get_markdown(self._model_name)
        filepath.write_text(md, encoding="utf-8")

        # Remove the live file since we promoted it
        if self._live_file and self._live_file.exists():
            self._live_file.unlink()

        entry_count = len(transcript.get_entries())
        self._start_new_session()
        transcript.system_message(f"exported {entry_count} entries → {filepath}")

    def _export_to_clipboard(self):
        """Copy transcript to clipboard."""
        transcript = self.query_one(TranscriptPanel)
        text = transcript.get_plain_text()
        entry_count = len(transcript.get_entries())
        try:
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            self._start_new_session()
            transcript.system_message(f"copied {entry_count} entries to clipboard")
        except Exception:
            transcript.system_message("clipboard copy failed")

    def _start_new_session(self):
        """Clear transcript and reset for a new session."""
        transcript = self.query_one(TranscriptPanel)
        transcript.clear()
        self.audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        if self._diarizer_loaded:
            self.diarizer.reset_session()
        # Start fresh live file
        self._session_start = datetime.now()
        self._live_file = None
        self._live_header_written = False

    def action_toggle_debug(self):
        self._debug = not self._debug
        state = "ON" if self._debug else "OFF"
        self.query_one(TranscriptPanel).system_message(f"debug mode {state}")

    def action_clear_transcript(self):
        """Clear display only — live file stays on disk as the record."""
        self.query_one(TranscriptPanel).clear()
        self.audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        if self._diarizer_loaded:
            self.diarizer.reset_session()

    def action_quit(self):
        # Live file already on disk — no extra save needed
        self.audio_capture.stop()
        self.system_capture.stop()

        # Let Textual restore the terminal, then hard-exit before
        # Python's GC triggers C extension segfaults
        threading.Timer(0.5, os._exit, args=[0]).start()
        self.exit()



if __name__ == "__main__":
    import argparse

    # Resolve defaults: saved preferences > config defaults
    _state = _load_state()
    _saved_model = _state.get("last_model")
    _saved_lang = _state.get("last_language")
    _default_model = _saved_model if _saved_model in AVAILABLE_MODELS else DEFAULT_MODEL
    _default_lang = _saved_lang if _saved_lang in AVAILABLE_LANGUAGES else DEFAULT_LANGUAGE

    parser = argparse.ArgumentParser(description="VOXTERM — Local Voice Transcription TUI")
    parser.add_argument(
        "-m", "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default=_default_model,
        help=f"Transcription model (default: {_default_model})",
    )
    parser.add_argument(
        "-l", "--language",
        choices=list(AVAILABLE_LANGUAGES.keys()),
        default=_default_lang,
        help=f"Transcription language (default: {_default_lang})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, repo in AVAILABLE_MODELS.items():
            tag = " (default)" if name == _default_model else ""
            qwen = " [qwen3-asr]" if name in QWEN3_MODELS else " [whisper]"
            print(f"  {name:12s} → {repo}{qwen}{tag}")
        sys.exit(0)

    model_repo = AVAILABLE_MODELS[args.model]
    model_name = args.model
    language = args.language

    print(f"VOXTERM // loading model ({model_name}) lang={language}...")
    print("(first run downloads the model, please wait)\n")
    if model_name in QWEN3_MODELS:
        transcriber = Qwen3Transcriber(model=model_repo, language=language)
    else:
        transcriber = WhisperTranscriber(model=model_repo)
    transcriber.load()
    print("Model ready. Launching TUI...\n")

    # Prevent segfault: PortAudio/PyTorch/SpeechBrain C threads crash
    # during Python's shutdown when native objects are GC'd in random order.
    # atexit fires before finalizers; the finally block catches SystemExit.
    import atexit
    atexit.register(os._exit, 0)

    app = VoxTerm(transcriber=transcriber, model_name=model_name, language=language)

    # Global exception hooks — dump diagnostics on any uncaught crash
    _orig_excepthook = sys.excepthook
    def _crash_excepthook(exc_type, exc_value, exc_tb):
        app._write_crash_dump(f"uncaught:{exc_type.__name__}", exc_value)
        _orig_excepthook(exc_type, exc_value, exc_tb)
    sys.excepthook = _crash_excepthook

    _orig_thread_excepthook = getattr(threading, "excepthook", None)
    def _thread_crash_hook(args):
        app._write_crash_dump(f"thread:{args.thread.name if args.thread else 'unknown'}", args.exc_value)
        if _orig_thread_excepthook:
            _orig_thread_excepthook(args)
    threading.excepthook = _thread_crash_hook

    try:
        app.run()
    finally:
        os._exit(0)
