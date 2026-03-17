#!/usr/bin/env python3
"""VOXTERM — Cyberpunk TUI Voice Transcription Engine"""

import sys
import os
import subprocess
import json
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
from widgets.waveform import WaveformWidget
from widgets.transcript import TranscriptPanel
from audio.capture import AudioCapture
from audio.buffer import AudioBuffer
from transcriber.engine import Qwen3Transcriber, WhisperTranscriber
from diarization.engine import DiarizationEngine
from config import (
    SAMPLE_RATE, CHUNK_SIZE, WAVEFORM_FPS,
    SILENCE_THRESHOLD, SILENCE_TRIGGER_SECONDS,
    MAX_BUFFER_SECONDS, MIN_BUFFER_SECONDS,
    DEFAULT_MODEL, AVAILABLE_MODELS, QWEN3_MODELS,
    DEFAULT_LANGUAGE, AVAILABLE_LANGUAGES,
)

# Session save directory
SESSIONS_DIR = Path.home() / "Documents" / "voxterm"
# Persistent state file (remembers last-used model across launches)
STATE_FILE = SESSIONS_DIR / ".state.json"


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


class VoxTerm(App):
    """Cyberpunk voice transcription TUI."""

    CSS_PATH = "cyberpunk.tcss"
    TITLE = "VOXTERM"

    BINDINGS = [
        Binding("r", "toggle_recording", "Record/Pause"),
        Binding("m", "switch_model", "Model"),
        Binding("l", "switch_language", "Language"),
        Binding("s", "save_transcript", "Save"),
        Binding("y", "copy_transcript", "Copy"),
        Binding("c", "clear_transcript", "Clear"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, transcriber=None, model_name="qwen3-0.6b", language="en"):
        super().__init__()
        self.audio_capture = AudioCapture()
        self.audio_buffer = AudioBuffer()
        self.transcriber = transcriber or Qwen3Transcriber()
        self.diarizer = DiarizationEngine()
        self._model_name = model_name
        self._language = language
        self._is_qwen3 = model_name in QWEN3_MODELS
        self._recording = False
        self._had_speech = False
        self._silence_chunks = 0
        self._is_transcribing = False
        self._model_loaded = transcriber is not None and transcriber.is_loaded
        self._diarizer_loaded = False
        self._session_start = datetime.now()

    def compose(self) -> ComposeResult:
        yield CyberHeader()
        with Vertical(id="main-container"):
            yield WaveformWidget()
            yield TranscriptPanel()
            telemetry = Static(
                "  status: [bold #607080]IDLE[/]    model: [#00ffcc]loading...[/]    audio: [#607080]inactive[/]",
                id="telemetry",
                markup=True,
            )
            telemetry.border_title = "TELEMETRY"
            yield telemetry
        yield Static(
            " [bold #00e5ff]\\[R][/][#607080] Record  [/]"
            "[bold #00e5ff]\\[M][/][#607080] Model  [/]"
            "[bold #00e5ff]\\[L][/][#607080] Lang  [/]"
            "[bold #00e5ff]\\[S][/][#607080] Save  [/]"
            "[bold #00e5ff]\\[Y][/][#607080] Copy  [/]"
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
        status = "RECORDING" if self._recording else "PAUSED" if self._model_loaded else "IDLE"
        status_color = "#00ff88" if self._recording else "#ff6600" if status == "PAUSED" else "#607080"
        model_text = self._model_name if self._model_loaded else "loading..."
        audio_text = "active" if self._recording else "inactive"
        audio_color = "#00ff88" if self._recording else "#607080"
        engine_text = "qwen3-asr" if self._is_qwen3 else "whisper"
        engine_color = "#aa88ff" if self._is_qwen3 else "#607080"
        lang_text = AVAILABLE_LANGUAGES.get(self._language, self._language) if self._language else "auto"
        spk_count = self.diarizer.num_speakers if self._diarizer_loaded else 0
        spk_text = f"    speakers: [#aa88ff]{spk_count}[/]" if spk_count > 0 else ""

        telemetry = self.query_one("#telemetry", Static)
        telemetry.update(
            f"  status: [bold {status_color}]{status}[/]"
            f"    model: [#00ffcc]{model_text}[/]"
            f"    audio: [{audio_color}]{audio_text}[/]"
            f"    engine: [{engine_color}]{engine_text}[/]"
            f"    lang: [#ffaa66]{lang_text}[/]"
            f"{spk_text}"
        )

        header = self.query_one(CyberHeader)
        header.update_status(status=status, audio=audio_text)

    def _start_audio_timer(self):
        self.set_interval(1.0 / WAVEFORM_FPS, self._process_audio, name="audio_timer")

    def _process_audio(self):
        """Read audio chunks, update waveform, check transcription trigger."""
        waveform = self.query_one(WaveformWidget)

        if not self._recording:
            waveform.tick()
            return

        chunks = self.audio_capture.drain()
        if not chunks:
            waveform.tick()
            return

        for chunk in chunks:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            self.audio_buffer.append(chunk)
            waveform.push_samples(chunk)

            if rms < SILENCE_THRESHOLD:
                self._silence_chunks += 1
            else:
                self._silence_chunks = 0
                self._had_speech = True

        waveform.tick()

        # Check transcription trigger
        silence_duration = self._silence_chunks * self._chunk_duration
        buffer_duration = self.audio_buffer.duration

        if self._is_transcribing:
            return

        if self._had_speech and silence_duration > SILENCE_TRIGGER_SECONDS and buffer_duration > MIN_BUFFER_SECONDS:
            self._trigger_transcription()
        elif buffer_duration >= MAX_BUFFER_SECONDS:
            self._trigger_transcription()

    def _trigger_transcription(self):
        """Send accumulated audio to transcription worker."""
        self._is_transcribing = True
        self._silence_chunks = 0
        audio = self.audio_buffer.get_and_clear()
        if len(audio) < int(SAMPLE_RATE * MIN_BUFFER_SECONDS):
            self._is_transcribing = False
            return
        # Reset _had_speech AFTER the min-length check so short
        # chunks don't silently eat the speech flag (Bug #4 fix)
        self._had_speech = False
        self._transcribe_audio(audio)

    @work(thread=True, group="transcription")
    def _transcribe_audio(self, audio: np.ndarray):
        try:
            # 1. Transcribe (Qwen3: ~100ms, Whisper: ~2-4s)
            result = self.transcriber.transcribe(audio)

            # 2. Unblock immediately so new chunks can queue
            self._is_transcribing = False

            text = result.get("text", "")

            # 3. Speaker ID AFTER transcription (non-blocking)
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
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"transcription error: {e}"
            )
            self._is_transcribing = False

    def _on_transcription(self, text: str, speaker: str = "", speaker_id: int = 0):
        self.query_one(TranscriptPanel).add_transcript(text, speaker, speaker_id)
        self._update_telemetry()

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
        if self._recording:
            self._recording = False
            self.audio_capture.stop()
            waveform.set_recording(False)
            self.query_one(TranscriptPanel).system_message("recording paused")
        else:
            self._recording = True
            try:
                self.audio_capture.start()
                waveform.set_recording(True)
                self.query_one(TranscriptPanel).system_message("recording started — speak into your microphone")
            except Exception as e:
                self._recording = False
                waveform.set_recording(False)
                self.query_one(TranscriptPanel).system_message(
                    f"microphone error: {e} — grant Terminal mic access in System Settings > Privacy"
                )
        self._update_telemetry()

    def action_switch_model(self):
        if self._is_transcribing:
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

    def action_save_transcript(self):
        """Save transcript to ~/Documents/voxterm/"""
        transcript = self.query_one(TranscriptPanel)
        entries = transcript.get_entries()
        if not entries:
            transcript.system_message("nothing to save")
            return

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        filename = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
        filepath = SESSIONS_DIR / filename
        md = transcript.get_markdown(self._model_name)
        filepath.write_text(md, encoding="utf-8")
        transcript.system_message(f"saved → {filepath}")

    def action_copy_transcript(self):
        """Copy transcript to system clipboard."""
        transcript = self.query_one(TranscriptPanel)
        entries = transcript.get_entries()
        if not entries:
            transcript.system_message("nothing to copy")
            return

        text = transcript.get_plain_text()
        try:
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            transcript.system_message(f"copied {len(entries)} entries to clipboard")
        except Exception:
            transcript.system_message("clipboard copy failed")

    def action_clear_transcript(self):
        self.query_one(TranscriptPanel).clear()
        self.query_one(TranscriptPanel)._entries.clear()
        self.audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        if self._diarizer_loaded:
            self.diarizer.reset_session()

    def action_quit(self):
        # Auto-save on quit if there are entries
        transcript = self.query_one(TranscriptPanel)
        if transcript.get_entries():
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            filename = self._session_start.strftime("%Y-%m-%d_%H%M%S") + ".md"
            filepath = SESSIONS_DIR / filename
            md = transcript.get_markdown(self._model_name)
            filepath.write_text(md, encoding="utf-8")

        # Explicit cleanup to avoid segfault from C extensions during GC
        self.audio_capture.stop()
        self.audio_buffer.clear()

        # Release native models before Python GC runs
        if hasattr(self, 'diarizer'):
            self.diarizer._model = None
        if hasattr(self, 'transcriber'):
            self.transcriber._model = None

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
    try:
        app.run()
    finally:
        os._exit(0)
