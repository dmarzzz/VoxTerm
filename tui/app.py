#!/usr/bin/env python3
"""VOXTERM — Cyberpunk TUI Voice Transcription Engine"""

from __future__ import annotations

import gc
import logging
import sys
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows: force UTF-8 output so Textual's Unicode box-drawing characters render correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Internal runtime defaults — prevent known framework conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Limit PyTorch internal threads to prevent C++ runtime conflicts with MLX
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Suppress leaked-semaphore warning from multiprocessing resource tracker
# on hard exit (os._exit). SpeechBrain/PyTorch create semaphores that aren't
# cleaned up before our forced exit — harmless, OS reclaims them immediately.
import diagnostics
diagnostics.setup_faulthandler()
diagnostics.rotate_crash_logs()

import warnings
warnings.filterwarnings("ignore", message="resource_tracker", category=UserWarning)

# Neuter the multiprocessing resource tracker subprocess.
# It prints leaked-semaphore warnings to its own stderr on shutdown,
# which we can't suppress from the main process. Since we use os._exit(),
# the OS reclaims all shared resources — the tracker is unnecessary.
# We let it spawn (libraries need register/unregister to function) but
# prevent it from printing warnings on stop.
import multiprocessing.resource_tracker as _rt
_rt._resource_tracker._stop = lambda **kwargs: None

from enum import Enum
import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.screen import ModalScreen
from textual import work

from tui.widgets.header import CyberHeader
from tui.widgets.waveform import WaveformWidget, _make_style
from tui.widgets.transcript import TranscriptPanel, Log
from tui.widgets.tag_screen import SpeakerTagScreen
from tui.widgets.profile_screen import SpeakerProfileScreen
from tui.widgets.transcript_explorer import TranscriptExplorerScreen
from tui.widgets.recording_pulse import RecordingPulse
from audio.capture import AudioCapture
from audio.buffer import AudioBuffer
from audio.system_capture import SystemCapture
from audio.transcriber import (
    Qwen3Transcriber, WhisperTranscriber, FasterWhisperTranscriber,
    configure_mlx_memory,
)
from audio.diarization.proxy import DiarizationProxy
from audio.speakers.store import SpeakerStore
from audio.vad import SileroVAD
from config import (
    SAMPLE_RATE, CHUNK_SIZE, WAVEFORM_FPS,
    SILENCE_THRESHOLD, SILENCE_TRIGGER_SECONDS,
    MAX_BUFFER_SECONDS, MIN_BUFFER_SECONDS,
    DEFAULT_MODEL, AVAILABLE_MODELS, QWEN3_MODELS, FASTER_WHISPER_MODELS,
    DEFAULT_LANGUAGE, AVAILABLE_LANGUAGES,
    LIVE_DIR,
)
from config import SESSIONS_DIR, STATE_FILE as _STATE_FILE


def _clipboard_cmd() -> list[str] | None:
    """Return the clipboard copy command for this platform."""
    if sys.platform == "darwin":
        return ["pbcopy"]
    if sys.platform == "win32":
        return ["clip.exe"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    if shutil.which("wl-copy"):
        return ["wl-copy"]
    return None


from network.party import PartyManager, PartyState, P2P_AVAILABLE as _P2P_AVAILABLE

from config import ConfigStore

def _setup_file_logging() -> None:
    """Best-effort file logging setup that never prevents module import."""
    try:
        log_path = Path.home() / "Documents" / "voxterm" / "voxterm.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root_log = logging.getLogger()
        # Default WARNING; set VOXTERM_LOG_LEVEL=DEBUG to see the quit/
        # shutdown breadcrumbs and other debug logs in voxterm.log.
        level_name = os.environ.get("VOXTERM_LOG_LEVEL", "WARNING").upper()
        root_log.setLevel(getattr(logging, level_name, logging.WARNING))
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(log_path)
            for h in root_log.handlers
        ):
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            root_log.addHandler(file_handler)
    except OSError as exc:
        sys.stderr.write(f"Warning: could not enable file logging: {exc}\n")


_setup_file_logging()

log = logging.getLogger(__name__)

# Module-level config store instance (lazy init for import safety)
_config: ConfigStore | None = None


def _get_config() -> ConfigStore:
    global _config
    if _config is None:
        _config = ConfigStore(path=_STATE_FILE)
    return _config


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
        all_names = list(AVAILABLE_MODELS.keys())
        for idx, name in enumerate(all_names):
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


class QuitConfirmScreen(ModalScreen):
    """Confirmation dialog when quitting during an active recording."""

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    #quit-dialog {
        width: 52;
        height: 12;
        border: heavy #ff6600;
        border-title-color: #ffaa00;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #quit-list {
        height: 4;
        background: #0a0e14;
        color: #c0c0c0;
    }
    #quit-list > .option-list--option-highlighted {
        background: #1a1a3a;
        color: #00ffcc;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "confirm_quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-dialog") as dialog:
            dialog.border_title = "QUIT?"
            yield Static(
                "[#ffaa00]Recording is active.[/]\n"
                "Transcript is auto-saved. Current segment will be lost.",
                markup=True,
            )
            yield Static("")
            yield OptionList(
                Option("  Quit anyway", id="quit"),
                Option("  Cancel", id="cancel"),
                id="quit-list",
            )
            yield Static("[dim]Q to quit  ·  ESC to cancel[/]", markup=True)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id == "quit")

    def action_confirm_quit(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class HelpScreen(ModalScreen):
    """Modal showing all keyboard shortcuts."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-dialog {
        width: 48;
        height: auto;
        max-height: 20;
        border: heavy #00e5ff;
        border-title-color: #00ffcc;
        border-title-style: bold;
        background: #0a0e14;
        padding: 1 2;
    }
    #help-content {
        height: auto;
        color: #c0c0c0;
    }
    #help-hint {
        height: 1;
        color: #607080;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("?", "dismiss", "Close", key_display="?"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog") as dialog:
            dialog.border_title = "KEYBOARD SHORTCUTS"
            yield Static(
                "[bold #00e5ff]R[/]       [#c0c0c0]Start / stop recording[/]\n"
                "[bold #00e5ff]T[/]       [#c0c0c0]Tag / name speakers[/]\n"
                "[bold #00e5ff]E[/]       [#c0c0c0]Browse saved transcripts[/]\n"
                "[bold #00e5ff]S[/]       [#c0c0c0]Save / copy transcript[/]\n"
                "[bold #00e5ff]M[/]       [#c0c0c0]Switch transcription model[/]\n"
                "[bold #00e5ff]L[/]       [#c0c0c0]Switch language[/]\n"
                "[bold #00e5ff]P[/]       [#c0c0c0]Party mode — join / leave[/]\n"
                "[bold #00e5ff]O[/]       [#c0c0c0]Speaker profiles[/]\n"
                "[bold #00e5ff]V[/]       [#c0c0c0]Toggle merged transcript view[/]\n"
                "[bold #00e5ff]C[/]       [#c0c0c0]Clear transcript[/]\n"
                "[bold #00e5ff]D[/]       [#c0c0c0]Toggle debug mode[/]\n"
                "[bold #00e5ff]Q[/]       [#c0c0c0]Quit[/]",
                id="help-content",
                markup=True,
            )
            yield Static(
                " [#607080]ESC[/] or [#607080]?[/] to close",
                id="help-hint",
                markup=True,
            )


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
                Option("  Discard transcript", id="discard"),
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

    CSS_PATH = "widgets/cyberpunk.tcss"
    TITLE = "VOXTERM"

    BINDINGS = [
        Binding("r", "toggle_recording", "Record/Pause"),
        Binding("t", "tag_speakers", "Tag"),
        Binding("o", "manage_profiles", "Profiles", show=False),
        Binding("m", "switch_model", "Model"),
        Binding("l", "switch_language", "Language"),
        Binding("ctrl+s", "export_transcript", "Save"),
        Binding("s", "export_transcript", "Export"),
        Binding("d", "toggle_debug", "Debug"),
        Binding("c", "clear_transcript", "Clear"),
        Binding("e", "explore_transcripts", "History"),
        Binding("p", "toggle_party", "Party"),
        Binding("v", "toggle_merged_view", "View"),
        Binding("h", "show_hivemind", "Hivemind"),
        Binding("?", "show_help", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", show=False),
    ]

    def __init__(self, transcriber=None, model_name="qwen3-0.6b", language="en",
                 p2p_name=None, p2p_create=False, p2p_join_code=None,
                 hivemind_client=None):
        super().__init__()
        self._p2p_auto_name = p2p_name
        self._p2p_auto_create = p2p_create
        self._p2p_auto_join_code = p2p_join_code
        # Hivemind transcript sink (None when --hivemind=off or no sink).
        # All segments produced in _on_transcription are mirrored to it.
        self._hivemind = hivemind_client
        self.audio_capture = AudioCapture()
        self.system_capture = SystemCapture()
        self.audio_buffer = AudioBuffer()
        self._local_audio_buffer = AudioBuffer()  # local-only audio for diarization
        self.vad = SileroVAD()
        self.transcriber = transcriber
        self.diarizer = DiarizationProxy()
        self.speaker_store = SpeakerStore()
        self._model_name = model_name
        self._language = language
        self._is_qwen3 = model_name in QWEN3_MODELS
        self._recording = False
        self._had_speech = False
        self._silence_chunks = 0
        # Held while a transcription worker is active. Acts as the busy gate:
        # the audio loop tries acquire(blocking=False) to decide whether to
        # spawn a worker, and the worker releases in finally. Replaces a prior
        # Event-based gate that the watchdog could clear from outside while the
        # worker was still mid-MLX-eval, allowing a second worker to start and
        # crash Metal command buffers via concurrent submission.
        self._transcribe_busy = threading.Lock()
        # Serializes Metal GPU access against concurrent submissions.
        self._transcribe_lock = threading.Lock()
        # Single-thread executor for all MLX work. MLX 0.31+ binds arrays to
        # per-thread streams — model load and inference must happen on the same
        # thread or `mx.argmax(...).item()` raises "no Stream(gpu, N) in current thread".
        self._mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._cache_clear_future = None  # tracks pending mx.clear_cache to avoid pile-up
        # Highest RSS already handled by the memory watchdog. ru_maxrss is a
        # high-water mark (never decreases), so without this the >6GB branch
        # would re-dump/flush/warn every timer tick forever; only re-act when
        # memory climbs meaningfully past what we already handled.
        self._mem_watchdog_handled_mb = 0.0
        self._transcribe_started: float = 0.0
        self._debug = False
        self._last_dbg: float = 0.0
        self._transcribe_count = 0
        self._model_loaded = transcriber is not None and getattr(transcriber, 'is_loaded', False)
        self._diarizer_loaded = False
        self._system_audio_notified = False
        # One-shot UI flags for audio device errors during a recording session
        # (reset on each start/stop so re-recording can surface a new error).
        self._mic_error_shown = False
        self._sck_death_shown = False
        self._last_saved_at: float | None = None
        self._save_confirmed = False  # True after first "autosaved" shown
        self._session_start = datetime.now()
        self._live_file: Path | None = None
        self._live_header_written = False
        # Maps session speaker_id → persistent profile_id (set on tagging)
        self._speaker_profile_map: dict[int, str] = {}
        # Active learning fatigue prevention
        self._prompt_times: list[float] = []        # timestamps of recent prompts
        self._prompt_confirmations: dict[int, int] = {}  # speaker_id → confirm count
        self._last_prompt_time: float = 0.0
        self._onboarding_shown = False
        # P2P party manager — owns all P2P state and logic
        self._party = PartyManager(self, _get_config())
        self._wire_party_callbacks()
        self._recording_pulse: RecordingPulse | None = None

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
            "[bold #00e5ff]\\[T][/][#607080] Tag  [/]"
            "[bold #00e5ff]\\[E][/][#607080] Transcripts  [/]"
            "[bold #00e5ff]\\[P][/][#607080] Party  [/]"
            "[bold #00e5ff]\\[V][/][#607080] Merged  [/]"
            "[bold #00e5ff]\\[?][/][#607080] Help[/]",
            id="footer-bar",
            markup=True,
        )

    # ── party manager callback wiring ─────────────────────────────

    def _wire_party_callbacks(self):
        """Wire PartyManager callbacks to VoxTerm UI handlers."""
        p = self._party

        _prev_state = [PartyState.SOLO]

        def _on_state_changed(state):
            prev = _prev_state[0]
            _prev_state[0] = state
            if state == PartyState.IN_PARTY and prev != PartyState.IN_PARTY:
                if self._party and self._party.is_host:
                    self.query_one(TranscriptPanel).system_message(
                        "no party found, you are the party now", Log.PARTY
                    )
                else:
                    self.query_one(TranscriptPanel).system_message("joined the party", Log.PARTY)
                self._trigger_fireworks(burst_count=10)
            self._update_telemetry()

        def _on_peer_joined(display_name):
            self.query_one(TranscriptPanel).system_message(
                f"{display_name} joined the party", Log.PARTY
            )
            self._update_telemetry()

        def _on_peer_left(node_id, display_name):
            self.query_one(TranscriptPanel).system_message(
                f"{display_name} left the party", Log.PARTY
            )
            self._update_telemetry()

        def _on_transcript_received(text, speaker, peer_display_name):
            tp = self.query_one(TranscriptPanel)
            tp.add_transcript(
                text, f"{peer_display_name}:{speaker}", 0, confidence="",
            )
            self._append_live_transcript(text, f"{peer_display_name}:{speaker}", 0)
            if tp.merged_view and p.assembler:
                tp.refresh_merged(
                    p.assembler,
                    local_name=p.display_name or "you",
                    peer_names=p.get_peer_names(),
                )

        def _on_partial_received():
            self._refresh_merged_if_active()

        def _on_debug(msg):
            self.query_one(TranscriptPanel).system_message(msg, Log.PARTY)

        def _on_party_color_changed(primary, light):
            self._apply_party_color(primary)

        def _on_party_colors_restored():
            self._restore_borders()
            self.query_one(TranscriptPanel).system_message("left the party", Log.PARTY)

        def _on_peer_bloom():
            self._peer_bloom()

        def _on_party_failed(error):
            self.query_one(TranscriptPanel).system_message(f"party failed: {error}", Log.PARTY)
            self._update_telemetry()

        p.on_state_changed = _on_state_changed
        p.on_peer_joined = _on_peer_joined
        p.on_peer_left = _on_peer_left
        p.on_transcript_received = _on_transcript_received
        p.on_partial_received = _on_partial_received
        p.on_debug = _on_debug
        p.on_party_color_changed = _on_party_color_changed
        p.on_party_colors_restored = _on_party_colors_restored
        p.on_peer_bloom = _on_peer_bloom
        p.on_party_failed = _on_party_failed

    # ── @work stubs for PartyManager ───────────────────────────

    @work(thread=True, group="p2p_discovery")
    def _party_start_passive_discovery_worker(self) -> None:
        """Worker thread: start passive mDNS discovery."""
        self._party.start_passive_discovery()

    @work(thread=True, group="p2p_setup")
    def _party_start_session_worker(self, code: str, is_creator: bool) -> None:
        """Worker thread: blocking party session setup."""
        self._party.start_session_blocking(code, is_creator)

    def on_mount(self) -> None:
        self._recording_pulse = RecordingPulse(self)

        # Open speaker profile store (fast — just SQLite + cache load)
        try:
            self.speaker_store.open()
            self.speaker_store.backup()
        except Exception:
            log.warning("speaker store init failed, running in ephemeral mode", exc_info=True)

        if self._model_loaded:
            transcript = self.query_one(TranscriptPanel)
            transcript.system_message(f"model loaded: {self._model_name}", Log.SYS, {self._model_name: "rainbow"})
            self._update_telemetry()
            self._start_audio_timer()
            self._load_diarizer()
        else:
            self.query_one(TranscriptPanel).system_message(f"loading model: {self._model_name}...", Log.SYS, {self._model_name: "rainbow"})
            self._start_audio_timer()
            self._load_model()

        # Start P2P peer discovery on launch (passive — just show who's nearby)
        if _P2P_AVAILABLE:
            self._party_start_passive_discovery_worker()

    def on_screen_resume(self) -> None:
        # Carry the recording border into modals that push on top of the main screen.
        if self._recording_pulse is not None:
            self._recording_pulse.reapply_to_active_screen()

    @property
    def _chunk_duration(self) -> float:
        return CHUNK_SIZE / SAMPLE_RATE

    def _update_telemetry(self):
        # Status dot
        if self._recording:
            status = "[bold #00ff88]● REC[/] [dim]\\[R][/]"
        elif self._model_loaded:
            status = "[bold #607080]● IDLE[/] [dim]\\[R][/]"
        else:
            status = "[bold #607080]● LOADING[/]"

        model_text = self._model_name if self._model_loaded else "loading..."
        lang_text = AVAILABLE_LANGUAGES.get(self._language, self._language) if self._language else "auto"

        # Party mode indicator (delegated to PartyManager)
        p2p_text = self._party.telemetry_text()

        self.query_one("#telemetry", Static).update(
            f"  {status}"
            f"    [#00ffcc]{model_text}[/] [dim]\\[M][/]"
            f"    [#ffaa66]{lang_text}[/] [dim]\\[L][/]"
            f"{p2p_text}"
            f"    [dim]\\[E] Transcripts  \\[S] Save  \\[Q] Quit[/]"
        )

        # Auto-save indicator in transcript border title
        # Show "✓ autosaved" once after first save, then go quiet.
        # Only resurface if idle >5 minutes (something might be off).
        tp = self.query_one(TranscriptPanel)
        base_title = "TRANSCRIPT // MERGED" if getattr(tp, 'merged_view', False) else "TRANSCRIPT // LIVE"
        if self._last_saved_at is not None:
            ago = time.time() - self._last_saved_at
            if not self._save_confirmed and ago < 8:
                tp.border_title = f"{base_title}  ·  ✓ autosaved"
            elif not self._save_confirmed and ago >= 8:
                self._save_confirmed = True
                tp.border_title = base_title
            elif ago > 300:  # 5 minutes idle
                tp.border_title = f"{base_title}  ·  last saved {int(ago // 60)}m ago"
            else:
                tp.border_title = base_title
        else:
            tp.border_title = base_title

    def _start_audio_timer(self):
        self.set_interval(1.0 / WAVEFORM_FPS, self._process_audio, name="audio_timer")
        self.set_interval(1.0, self._refresh_telemetry, name="telemetry_timer")
        self.set_interval(60.0, self._periodic_gc, name="gc_timer")

    def _refresh_telemetry(self):
        """Periodic refresh for saved counter and recording timer."""
        if self._recording:
            self.query_one(CyberHeader).refresh()
        if self._last_saved_at is not None:
            self._update_telemetry()

    def _periodic_gc(self):
        """Prevent memory fragmentation during long sessions + memory watchdog."""
        gc.collect()
        if sys.platform == "darwin":
            # Free MLX's GPU cache on the same thread that owns it. Fire-and-forget
            # — _periodic_gc runs on the Textual event loop, so we must not block.
            # Skip if a previous clear_cache is still pending (MLX thread is busy
            # with a transcription) to avoid queue pile-up.
            prev = self._cache_clear_future
            if prev is None or prev.done():
                try:
                    import mlx.core as mx
                    self._cache_clear_future = self._mlx_executor.submit(mx.clear_cache)
                except Exception:
                    pass

        # Memory watchdog: warn at 4GB, crash-dump at 6GB
        try:
            import resource as _resource
            rss_raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss_bytes = rss_raw
            else:
                # Linux reports ru_maxrss in KiB
                rss_bytes = rss_raw * 1024
            rss_mb = rss_bytes / (1024 * 1024)
        except ImportError:
            return  # Windows — can't check memory, skip watchdog
        # rss_mb is a peak/high-water value (ru_maxrss never decreases), so
        # only act the first time we cross 6GB and again only if it climbs
        # another 500MB — otherwise this fires every timer tick forever.
        if rss_mb > 6000 and rss_mb > self._mem_watchdog_handled_mb + 500:
            self._mem_watchdog_handled_mb = rss_mb
            self._write_crash_dump(f"memory_watchdog: {rss_mb:.0f}MB")
            # Actively reclaim instead of only warning.
            gc.collect()
            if sys.platform == "darwin":
                # Force a Metal cache flush on the MLX thread, ungated by the
                # usual skip-if-busy check (under this much pressure we want
                # it to run even if it has to queue behind a transcription).
                try:
                    import mlx.core as mx
                    self._cache_clear_future = self._mlx_executor.submit(mx.clear_cache)
                except Exception:
                    pass
                msg = (
                    f"high memory usage ({rss_mb:.0f}MB) — flushing GPU "
                    f"cache; save and restart if it persists"
                )
            else:
                # No GPU cache to flush off macOS — don't claim otherwise.
                msg = (
                    f"high memory usage ({rss_mb:.0f}MB) — consider saving "
                    f"and restarting"
                )
            self.query_one(TranscriptPanel).system_message(msg, Log.SYS)
        elif rss_mb > 4000 and self._debug:
            self.query_one(TranscriptPanel).system_message(
                f"[dbg] memory: {rss_mb:.0f}MB", Log.SYS
            )

    def _write_crash_dump(self, context: str, exc: BaseException | None = None):
        """Write a diagnostic dump to disk. Always runs, not gated by debug."""
        try:
            cache = _make_style.cache_info()
            try:
                entry_count = len(self.query_one(TranscriptPanel).get_entries())
            except Exception:
                entry_count = -1

            state = {
                "uptime_sec": (datetime.now() - self._session_start).total_seconds(),
                "recording": self._recording,
                "is_transcribing": self._transcribe_busy.locked(),
                "transcribe_count": self._transcribe_count,
                "model": self._model_name,
                "model_loaded": self._model_loaded,
                "diarizer_loaded": self._diarizer_loaded,
                "language": self._language,
                "had_speech": self._had_speech,
                "silence_chunks": self._silence_chunks,
                "sys_capture": f"active={self.system_capture.is_active} msg={self.system_capture.status_message}",
                "audio_buf_dur": f"{self.audio_buffer.duration:.2f}s",
                "style_cache": f"hits={cache.hits} misses={cache.misses} size={cache.currsize}/{cache.maxsize}",
                "transcript_entries": entry_count,
                "speakers": self.diarizer.num_speakers if self._diarizer_loaded else 0,
                "gc_counts": str(gc.get_count()),
            }
            diagnostics.write_crash_dump(context, exc, state)
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

        # Surface audio device errors (mic disconnect, permission revoke,
        # SCK helper death) once per recording session. Otherwise these
        # fail silently and the user just sees no transcripts.
        mic_err = self.audio_capture.device_error
        if mic_err and not self._mic_error_shown:
            self.query_one(TranscriptPanel).system_message(
                f"mic warning: {mic_err}", Log.SYS,
            )
            self._mic_error_shown = True
        if self.system_capture.died_during_recording and not self._sck_death_shown:
            self.query_one(TranscriptPanel).system_message(
                "system audio capture stopped unexpectedly", Log.SYS,
            )
            self._sck_death_shown = True

        # Send raw mic audio to peers via UDP (before mixing with system)
        if mic_chunks and self._party.audio_streamer and self._party.session_mgr:
            peer_addrs = self._party.get_peer_udp_addrs()
            if peer_addrs:
                now = time.monotonic()
                for chunk in mic_chunks:
                    self._party.audio_streamer.send_frame(
                        chunk, seq=self._party.audio_send_seq,
                        timestamp=now, peer_addrs=peer_addrs,
                    )
                    self._party.audio_send_seq += 1

        chunks = self._mix_chunks(mic_chunks, sys_chunks) if sys_chunks else mic_chunks
        if not chunks:
            waveform.tick()
            return

        # Build a mic-only stream aligned 1:1 with `chunks` so the
        # diarizer sees only the local microphone, never the system-audio
        # mix. _mix_chunks emits: n=min(len(mic),len(sys)) summed entries,
        # then any mic tail, then any sys tail. Pad sys-only tail with
        # silence so indices line up with `chunks`.
        if sys_chunks:
            n = min(len(mic_chunks), len(sys_chunks))
            mic_only_chunks: list[np.ndarray] = list(mic_chunks[:n])
            mic_only_chunks.extend(mic_chunks[n:])
            for sc in sys_chunks[n:]:
                mic_only_chunks.append(np.zeros_like(sc))
        else:
            mic_only_chunks = list(chunks)

        for idx, chunk in enumerate(chunks):
            mic_only_chunk = mic_only_chunks[idx]
            waveform.push_samples(chunk)

            # Speech detection: Silero VAD (neural) with RMS fallback
            # Runs on local audio immediately (no merge delay)
            if self.vad.is_loaded:
                is_speech = self.vad.is_speech(chunk)
            else:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                is_speech = rms >= SILENCE_THRESHOLD

            if is_speech:
                self._silence_chunks = 0
                self._had_speech = True
            else:
                self._silence_chunks += 1

            # Feed chunk through the merger (adds peer audio, applies jitter delay)
            _peer_mixer = self._party.peer_mixer
            if _peer_mixer is not None:
                merged_chunks = _peer_mixer.add_local_chunk(chunk, time.monotonic())
                for mc in merged_chunks:
                    if self._had_speech or is_speech:
                        self.audio_buffer.append(mc)
                # Diarizer sees raw mic only — never system audio or peer audio,
                # both of which acoustically blend speakers into one cluster.
                if self._had_speech or is_speech:
                    self._local_audio_buffer.append(mic_only_chunk)
            else:
                if is_speech:
                    self.audio_buffer.append(chunk)
                    self._local_audio_buffer.append(mic_only_chunk)
                elif self._had_speech:
                    self.audio_buffer.append(chunk)
                    self._local_audio_buffer.append(mic_only_chunk)

        waveform.tick()

        # Update waveform merge indicator (~every 0.5s, not every frame)
        mixer = self._party.peer_mixer
        if mixer and mixer.active_peers > 0 and self._recording:
            self._merge_display_counter = getattr(self, '_merge_display_counter', 0) + 1
            if self._merge_display_counter % 8 == 0:  # ~0.5s at 15fps
                stats = mixer.get_stats()
                weights = stats.get("live_weights", {})
                if weights:
                    names = self._party.get_peer_names()
                    parts = []
                    for nid, w in sorted(weights.items(), key=lambda x: -x[1]):
                        name = "you" if nid == "__local__" else names.get(nid, nid[:6])
                        parts.append(f"{name} {int(w*100)}%")
                    waveform.set_merge_status(" | ".join(parts))
                else:
                    waveform.set_merge_status("")
        elif not mixer or mixer.active_peers == 0:
            if getattr(self, '_merge_display_counter', 0) > 0:
                self._merge_display_counter = 0
                waveform.set_merge_status("")

        # Check transcription trigger
        merge_delay = self._party.peer_mixer.merge_delay if self._party.peer_mixer else 0.0
        silence_duration = self._silence_chunks * self._chunk_duration
        buffer_duration = self.audio_buffer.duration

        if self._transcribe_busy.locked():
            # Graduated watchdog: warn but do NOT force-clear the gate.
            # The lock is the gate; the worker releases it in finally.
            # Forcibly clearing it from outside (the prior Event-based design)
            # let a second worker start while the first was still mid-MLX-eval,
            # which is what crashed Metal command buffers in the first place.
            elapsed = time.time() - self._transcribe_started if self._transcribe_started else 0
            if elapsed > 30 and not getattr(self, '_watchdog_critical_warned', False):
                self._write_crash_dump(f"transcription_hung_{elapsed:.0f}s")
                self.query_one(TranscriptPanel).system_message(
                    f"transcription timed out ({elapsed:.0f}s) — try a smaller model [M]", Log.SYS
                )
                self._watchdog_critical_warned = True
            elif 15 < elapsed <= 30 and not getattr(self, '_watchdog_warned', False):
                self.query_one(TranscriptPanel).system_message(
                    f"[watchdog] transcription slow ({elapsed:.0f}s)", Log.SYS
                )
                self._watchdog_warned = True
            elif elapsed > 8 and self._debug:
                self.query_one(TranscriptPanel).system_message(
                    f"[dbg] transcription slow: {elapsed:.0f}s", Log.SYS
                )
            return

        if self._debug:
            now = time.time()
            if buffer_duration > 0.5 and now - self._last_dbg > 3:
                self._last_dbg = now
                dbg_parts = [
                    f"[dbg] buf={buffer_duration:.1f}s sil={silence_duration:.1f}s "
                    f"speech={self._had_speech}"
                ]
                _pm = self._party
                if _pm.session_mgr and _pm.session_mgr.is_in_session:
                    tp = self.query_one(TranscriptPanel)
                    mixer = _pm.peer_mixer
                    if _pm.assembler:
                        dbg_parts.append(
                            f"  asm={_pm.assembler.final_count}F/{_pm.assembler.partial_count}P "
                            f"view={'MERGED' if tp.merged_view else 'LOCAL'}"
                        )
                    if mixer and mixer.active_peers > 0:
                        stats = mixer.get_stats()
                        weights = stats.get("live_weights", {})
                        if weights:
                            names = _pm.get_peer_names()
                            parts = []
                            for nid, w in sorted(weights.items(), key=lambda x: -x[1]):
                                name = "you" if nid == "__local__" else names.get(nid, nid[:6])
                                parts.append(f"{name}={int(w*100)}%")
                            dbg_parts.append(f"  mix=[{' | '.join(parts)}]")
                self.query_one(TranscriptPanel).system_message(
                    "".join(dbg_parts), Log.SYS
                )

        effective_silence_trigger = SILENCE_TRIGGER_SECONDS + merge_delay
        if self._had_speech and silence_duration > effective_silence_trigger and buffer_duration > MIN_BUFFER_SECONDS:
            self._trigger_transcription()
        elif buffer_duration >= MAX_BUFFER_SECONDS:
            self._trigger_transcription()

    def _trigger_transcription(self):
        """Send accumulated audio to transcription worker."""
        if not self._transcribe_busy.acquire(blocking=False):
            return  # previous worker still running
        # Defer starting the watchdog timer until GPU work actually begins
        # (set inside _transcribe_lock in _transcribe_audio). 0 means "not
        # yet running"; the watchdog treats elapsed==0 as not-yet-started.
        self._transcribe_started = 0.0
        self._watchdog_warned = False
        self._watchdog_critical_warned = False
        self._silence_chunks = 0
        audio = self.audio_buffer.get_and_clear()
        local_audio = self._local_audio_buffer.get_and_clear()
        if len(audio) < int(SAMPLE_RATE * MIN_BUFFER_SECONDS):
            self._transcribe_busy.release()
            return

        self._had_speech = False
        self._transcribe_audio(audio, local_audio)

    @work(thread=True, group="transcription")
    def _transcribe_audio(self, audio: np.ndarray, local_audio: np.ndarray | None = None):
        # Use local-only audio for diarization to avoid peer audio corrupting embeddings.
        # If local_audio is unavailable (party mode with no local capture), we cannot
        # diarize reliably — merged mono audio makes all speakers look identical.
        has_local_audio = (
            local_audio is not None
            and len(local_audio) > 0
            and float(np.sqrt(np.mean(local_audio ** 2))) >= 0.003
        )
        diarize_audio = local_audio if has_local_audio else audio
        try:
            if self._debug:
                duration = len(audio) / SAMPLE_RATE
                self.call_from_thread(
                    self.query_one(TranscriptPanel).system_message,
                    f"[dbg] transcribing {duration:.1f}s audio..."
                )
            # 1. Transcribe (Qwen3: ~100ms, Whisper: ~2-4s)
            # MLX Metal command buffers are NOT thread-safe — concurrent GPU
            # submissions from multiple Textual workers (or transcribe vs.
            # model-load) cause segfaults. Serialize with a lock.
            with self._transcribe_lock:
                # Start the watchdog timer when GPU work actually begins so a
                # transcription queued behind a model swap isn't flagged hung
                # for the time it spent waiting on the lock or in the executor.
                self._transcribe_started = time.time()
                result = self._mlx_executor.submit(self.transcriber.transcribe, audio).result()

                # Reclaim the MLX buffer cache on the cadence of actual
                # transcription (the 60s _periodic_gc timer is skipped while
                # the executor is busy, so it can be starved under load).
                # Enqueue it while STILL holding _transcribe_lock so a model
                # swap (which also takes _transcribe_lock before queuing a
                # long load) can't slip a load between the transcribe and the
                # clear; and do NOT .result() it — blocking the worker here
                # would stall it behind that load while the hung-transcription
                # watchdog state is stale.
                self._transcribe_count += 1
                if self._transcribe_count % 20 == 0 and sys.platform == "darwin":
                    prev = self._cache_clear_future
                    if prev is None or prev.done():
                        try:
                            import mlx.core as mx
                            self._cache_clear_future = self._mlx_executor.submit(mx.clear_cache)
                        except Exception:
                            pass

            # Periodic Python GC (off the MLX lock — doesn't touch the GPU).
            if self._transcribe_count % 20 == 0:
                gc.collect()

            text = result.get("text", "")

            # 2. Speaker ID — use SCD for longer buffers, single identify for short
            speaker_label, speaker_id = "", 0
            confidence = ""
            is_overlap = False
            if text and self._diarizer_loaded:
                try:
                    # Skip diarization if we only have merged party audio —
                    # mono blend makes all speakers look identical
                    if not has_local_audio and local_audio is not None:
                        speaker_label, speaker_id = "Speaker 1", 1
                        segments = [("Speaker 1", 1, 0, len(diarize_audio))]
                        is_overlap = False
                    # Use speaker-change detection for buffers >= 3s
                    elif len(diarize_audio) >= 48000:
                        segments = self.diarizer.identify_segments(
                            diarize_audio.copy()
                        )
                    else:
                        lbl, sid = self.diarizer.identify(diarize_audio.copy())
                        segments = [(lbl, sid, 0, len(diarize_audio))]

                    # Check overlap metadata from last identify() call
                    # (only when diarization actually ran — otherwise meta is stale)
                    if has_local_audio or local_audio is None:
                        meta = self.diarizer.get_last_identify_meta()
                        is_overlap = meta.get("is_overlap", False)

                    # Use last segment as the "current" speaker for cross-session matching
                    speaker_label, speaker_id = segments[-1][0], segments[-1][1]

                    # Debug: show speaker assignment info
                    if self._debug:
                        dbg = getattr(self.diarizer, '_last_debug', {})
                        n_spk = dbg.get('debug_speakers', '?')
                        rms = dbg.get('debug_rms', '?')
                        best = dbg.get('debug_best_score', '?')
                        phase = dbg.get('debug_phase', '?')
                        thresh = dbg.get('debug_threshold', '?')
                        seg_info = f"segs={len(segments)}" if len(segments) > 1 else ""
                        overlap_info = " [OVERLAP]" if is_overlap else ""
                        self.call_from_thread(
                            self.query_one(TranscriptPanel).system_message,
                            f"[dbg] → {speaker_label} (id={speaker_id}) "
                            f"speakers={n_spk} sim={best}/{thresh} rms={rms} "
                            f"[{phase}]{seg_info}{overlap_info}",
                        )

                    # 3. Cross-session matching for each unique speaker in segments
                    seen_sids = set()
                    for seg_label, seg_sid, _, _ in segments:
                        if seg_sid in seen_sids or seg_sid <= 0:
                            continue
                        seen_sids.add(seg_sid)
                        self._try_cross_session_match(seg_sid)

                except Exception as _diar_err:
                    import traceback as _tb
                    log.error("Diarizer exception: %s\n%s", _diar_err,
                              "".join(_tb.format_exception(_diar_err)))
                    if self._debug:
                        self.call_from_thread(
                            self.query_one(TranscriptPanel).system_message,
                            f"[dbg] diarizer error: {_diar_err}",
                        )
                    segments = [("", 0, 0, len(audio))]

            else:
                segments = [("", 0, 0, len(audio))]

            if text:
                if len(segments) > 1:
                    # Split text across segments proportionally by duration
                    seg_texts = self._split_text_by_segments(text, segments)
                    for seg_text, seg_label, seg_sid in seg_texts:
                        if seg_text.strip():
                            seg_overlap = is_overlap and seg_sid == speaker_id
                            self.call_from_thread(
                                self._on_transcription, seg_text, seg_label,
                                seg_sid, confidence,
                                seg_overlap,
                            )
                else:
                    self.call_from_thread(
                        self._on_transcription, text, speaker_label,
                        speaker_id, confidence,
                        is_overlap,
                    )
        except Exception as e:
            self._write_crash_dump("_transcribe_audio", e)
            import traceback
            tb = traceback.format_exc()
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"transcription error: {e}\n{tb}"
            )
        finally:
            # The lock is the concurrency gate. Acquire happens exactly once
            # in _trigger_transcription, release happens exactly once here.
            # An unexpected RuntimeError means the gate is corrupt and
            # concurrent transcriptions could re-occur — surface it rather
            # than silently continuing.
            try:
                self._transcribe_busy.release()
            except RuntimeError as e:
                self._write_crash_dump(f"transcribe_busy_release_failed: {e}", e)

    def _try_cross_session_match(self, speaker_id: int) -> None:
        """Attempt cross-session matching for a speaker (worker thread)."""
        if not (
            self.speaker_store.is_open
            and not self.diarizer.is_matched(speaker_id)
            and speaker_id not in self._speaker_profile_map
            and self.diarizer.is_speaker_stable(speaker_id)
        ):
            return

        centroid = self.diarizer.get_session_centroid(speaker_id)
        if centroid is None:
            return

        match = self.speaker_store.classify_match(centroid)

        if match.tier == "high":
            self.diarizer.set_speaker_name(speaker_id, match.name)
            self.diarizer.mark_matched(speaker_id)
            self._speaker_profile_map[speaker_id] = match.profile_id

            if self.speaker_store.is_profile_mature(match.profile_id):
                seg_data = self.diarizer.get_segment_embeddings(speaker_id)
                for emb, dur in seg_data:
                    if dur >= 3.0:
                        self.speaker_store.update_profile_embedding(
                            match.profile_id, emb,
                            duration=dur, confirmed=False,
                        )

            self.call_from_thread(
                self._on_auto_recognition,
                speaker_id, match.name, match.color,
                "high", match.score,
            )

        elif match.tier == "medium":
            self.diarizer.mark_matched(speaker_id)
            self.call_from_thread(
                self._on_auto_recognition,
                speaker_id, f"{match.name}?", match.color,
                "medium", match.score,
            )

    @staticmethod
    def _split_text_by_segments(
        text: str,
        segments: list[tuple[str, int, int, int]],
    ) -> list[tuple[str, str, int]]:
        """Split transcribed text across segments proportionally by duration.

        Returns list of (text_portion, speaker_label, speaker_id).
        """
        words = text.split()
        if not words or not segments:
            return [(text, "", 0)]

        total_samples = sum(end - start for _, _, start, end in segments)
        if total_samples <= 0:
            return [(text, segments[0][0], segments[0][1])]

        result = []
        word_idx = 0
        for i, (label, sid, start, end) in enumerate(segments):
            duration_frac = (end - start) / total_samples
            if i == len(segments) - 1:
                # Last segment gets remaining words
                n_words = len(words) - word_idx
            else:
                n_words = max(1, round(len(words) * duration_frac))

            seg_words = words[word_idx:word_idx + n_words]
            word_idx += n_words
            result.append((" ".join(seg_words), label, sid))

        return result

    def _on_transcription(
        self, text: str, speaker: str = "", speaker_id: int = 0,
        confidence: str = "", overlap: bool = False,
    ):
        if not speaker:
            import traceback
            log.warning("Empty speaker label for text=%r, stack:\n%s",
                        text[:80], "".join(traceback.format_stack()[-4:]))
        self.query_one(TranscriptPanel).add_transcript(
            text, speaker, speaker_id, confidence=confidence,
            overlap=overlap,
        )
        self._append_live_transcript(text, speaker, speaker_id)

        # Mirror to hivemind sink (no-op if --hivemind=off or no sink).
        # Timestamp `t` is seconds since session start so the sink can
        # reconstruct ordering without needing wall-clock alignment.
        if self._hivemind is not None:
            try:
                t = (datetime.now() - self._session_start).total_seconds()
                self._hivemind.add_segment(t, speaker or "?", text)
            except Exception:
                # Never let hivemind errors break the dictation loop.
                log.warning("hivemind add_segment failed", exc_info=True)

        self._update_telemetry()

        # Broadcast to P2P peers if in a session (via bounded queue, not per-call threads)
        _pm = self._party
        if _pm.session_mgr and _pm.session_mgr.is_in_session and _pm.send_queue:
            _pm.transcript_seq += 1
            now = time.monotonic()
            _pm.enqueue_transcript(speaker, _pm.transcript_seq, text, now)

            # Track local segment in assembler for merged view
            if _pm.assembler:
                dom = ""
                mixer = _pm.peer_mixer
                if mixer and mixer.active_peers > 0:
                    dom = mixer.dominant_source
                _pm.track_local_segment(_pm.transcript_seq, speaker, text, now, dom)
                tp = self.query_one(TranscriptPanel)
                if tp.merged_view:
                    tp.refresh_merged(
                        _pm.assembler,
                        local_name=_pm.display_name or "you",
                        peer_names=_pm.get_peer_names(),
                    )

        # First-use onboarding tip
        if (
            not self._onboarding_shown
            and speaker_id > 0
            and not self.speaker_store.get_all_profiles()
            and not self.diarizer.get_speaker_names()
        ):
            self._onboarding_shown = True
            self.query_one(TranscriptPanel).system_message(
                "tip: press [T] to name speakers — VoxTerm will remember them", Log.SYS
            )

    def _on_auto_recognition(
        self, speaker_id: int, name: str, color: str,
        tier: str, score: float,
    ):
        """Called on main thread when cross-session matching identifies a speaker."""
        transcript = self.query_one(TranscriptPanel)
        transcript.set_speaker_confidence(speaker_id, tier, score)
        # Rename all prior entries for this speaker in the transcript
        transcript.rename_speaker(speaker_id, name, color)

        # For MEDIUM matches: show confirmation prompt (if within budget)
        if tier == "medium" and self._can_prompt():
            self._show_confirm_prompt(speaker_id, name.rstrip("?"), score)

        self._update_telemetry()

    # ── active learning / fatigue ──────────────────────────────

    def _can_prompt(self) -> bool:
        """Check if we're within the active learning prompt budget."""
        now = time.time()
        session_elapsed = now - self._session_start.timestamp()

        # Max 5 prompts per 10-minute window
        cutoff = now - 600
        recent = [t for t in self._prompt_times if t > cutoff]
        if len(recent) >= 5:
            return False

        # After first 5 min: max 1 prompt per 2 minutes
        if session_elapsed > 300 and (now - self._last_prompt_time) < 120:
            return False

        return True

    def _show_confirm_prompt(self, speaker_id: int, name: str, score: float):
        """Show a non-blocking confirmation for a MEDIUM-confidence match."""
        pct = int(score * 100)
        transcript = self.query_one(TranscriptPanel)
        transcript.system_message(
            f"is this {name}? (~{pct}%) "
            f"press [T] to confirm or rename",
            Log.SYS,
        )
        self._prompt_times.append(time.time())
        self._last_prompt_time = time.time()

    # ── background auto-save ───────────────────────────────────

    def _append_live_transcript(self, text: str, speaker: str, speaker_id: int):
        """Append a single transcript line to the live file on disk."""
        try:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            if self._live_file is None:
                fname = self._session_start.strftime("%Y-%m-%d_%H%M%S") + "-transcript.md"
                self._live_file = LIVE_DIR / fname
                self._live_header_written = False

            with open(self._live_file, "a", encoding="utf-8") as f:
                if not self._live_header_written:
                    lang = AVAILABLE_LANGUAGES.get(self._language, self._language) if self._language else "auto"
                    f.write(f"# VoxTerm Transcript\n\n")
                    f.write(f"- **Date:** {self._session_start.strftime('%A, %B %d, %Y')}\n")
                    f.write(f"- **Started:** {self._session_start.strftime('%I:%M %p')}\n")
                    f.write(f"- **Model:** {self._model_name}\n")
                    f.write(f"- **Language:** {lang}\n")
                    f.write(f"\n---\n\n")
                    self._live_header_written = True

                ts = datetime.now().strftime("%H:%M:%S")
                if speaker:
                    f.write(f"**[{ts}]** **{speaker}:** {text}\n\n")
                else:
                    f.write(f"**[{ts}]** {text}\n\n")
            self._last_saved_at = time.time()
        except Exception:
            pass  # never block transcription on I/O failure

    def _load_model(self):
        """Start model loading in a plain thread (not @work — avoids fd inheritance bugs)."""
        def _do_load():
            try:
                # Python 3.12 subprocess/multiprocessing fails with
                # "bad value(s) in fds_to_keep" when spawned from a thread
                # while Textual holds terminal FDs open.
                #
                # The previous patch on subprocess.Popen.__init__ didn't help
                # because multiprocessing.util.spawnv_passfds calls
                # _posixsubprocess.fork_exec directly, bypassing Popen.
                # Patch at the C-level entry point to sanitize fds_to_keep.
                if sys.platform != "win32":
                    import _posixsubprocess
                    if not getattr(_posixsubprocess, '_vt_patched', False):
                        _orig_fe = _posixsubprocess.fork_exec
                        def _safe_fork_exec(*args):
                            largs = list(args)
                            fds = largs[3]  # fds_to_keep (sorted tuple of ints)
                            if fds:
                                clean = tuple(
                                    fd for fd in fds
                                    if isinstance(fd, int) and 0 <= fd <= 2**31
                                )
                                largs[3] = clean
                            try:
                                return _orig_fe(*largs)
                            except ValueError:
                                largs[3] = ()
                                return _orig_fe(*largs)
                        _posixsubprocess.fork_exec = _safe_fork_exec
                        _posixsubprocess._vt_patched = True

                model_repo = AVAILABLE_MODELS[self._model_name]
                if self._model_name in QWEN3_MODELS:
                    self.transcriber = Qwen3Transcriber(model=model_repo, language=self._language)
                elif self._model_name in FASTER_WHISPER_MODELS:
                    self.transcriber = FasterWhisperTranscriber(model=model_repo, language=self._language)
                else:
                    self.transcriber = WhisperTranscriber(model=model_repo)
                self._mlx_executor.submit(self.transcriber.load).result()
                self.call_from_thread(self._on_model_loaded)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.call_from_thread(
                    self.query_one(TranscriptPanel).system_message,
                    f"model load failed: {e}\n{tb}"
                )
        threading.Thread(target=_do_load, daemon=True, name="model-loader").start()

    def _on_model_loaded(self):
        self._model_loaded = True
        self.query_one(TranscriptPanel).system_message(f"model loaded: {self._model_name}", Log.SYS, {self._model_name: "rainbow"})
        if not self._diarizer_loaded:
            self._load_diarizer()
        self._update_telemetry()

    @work(thread=True, group="diarizer_loading")
    def _load_diarizer(self):
        enc = ""
        if self.speaker_store.is_open and self.speaker_store.is_encrypted:
            enc = " (encrypted)"
        self.call_from_thread(
            lambda: self.query_one(TranscriptPanel).system_message(
                f"loading offline speaker recognition{enc}...", Log.SYS
            ),
        )
        try:
            # Set up crash/restart callbacks for subprocess mode
            self.diarizer.on_subprocess_crash = self._on_diarizer_crash
            self.diarizer.on_subprocess_ready = lambda: self.call_from_thread(
                lambda: self.query_one(TranscriptPanel).system_message(
                    "speaker identification restarted", Log.SYS
                ),
            )
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
        tp = self.query_one(TranscriptPanel)
        tp.system_message("speaker recognition ready — voices encrypted locally, no audio stored", Log.SYS)
        tp.system_message("press [R] to record", Log.SYS)
        self._update_telemetry()

    def _on_diarizer_crash(self, crash_count: int):
        """Called from worker thread when diarizer subprocess crashes."""
        self._write_crash_dump(f"diarizer_subprocess_crash #{crash_count}")
        if self.diarizer._mode == "inprocess":
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                "speaker ID subprocess failed — using in-process fallback"
            )
        else:
            self.call_from_thread(
                self.query_one(TranscriptPanel).system_message,
                f"speaker ID subprocess crashed — restarting ({crash_count}/{3})"
            )

    def _on_diarizer_fallback(self):
        self.query_one(TranscriptPanel).system_message("press [R] to record", Log.SYS)

    # ── actions ─────────────────────────────────────────────────

    def action_toggle_recording(self):
        if not self._model_loaded:
            self.query_one(TranscriptPanel).system_message("model still loading, please wait...", Log.SYS)
            return

        waveform = self.query_one(WaveformWidget)
        header = self.query_one(CyberHeader)
        transcript = self.query_one(TranscriptPanel)
        if self._recording:
            self._recording = False
            self.audio_capture.stop()
            self.system_capture.stop()
            waveform.set_recording(False)
            header.set_recording(False)
            if self._recording_pulse:
                self._recording_pulse.stop()
            transcript.system_message("recording stopped", Log.REC, {"stopped": "#ff4466"})
        else:
            self._recording = True
            self._mic_error_shown = False
            self._sck_death_shown = False
            self.vad.reset()
            try:
                self.audio_capture.start()
                waveform.set_recording(True)
                header.set_recording(True)
                if self._recording_pulse:
                    self._recording_pulse.start()
                mic_name = getattr(self.audio_capture, '_device_name', 'unknown')
                transcript.system_message("recording started", Log.REC, {"started": "#00ff88"})
                if self._debug:
                    transcript.system_message(f"mic: {mic_name}", Log.SYS)
            except Exception as e:
                self._recording = False
                waveform.set_recording(False)
                header.set_recording(False)
                if self._recording_pulse:
                    self._recording_pulse.stop()
                transcript.system_message(
                    f"microphone error: {e} — grant Terminal mic access in System Settings > Privacy",
                    Log.REC,
                )
                self._update_telemetry()
                return

            # Start system audio capture (non-fatal if unavailable)
            try:
                self.system_capture.start()
            except Exception:
                pass

            if self.system_capture.status_message:
                self._system_audio_notified = True
        self._update_telemetry()

    def action_switch_model(self):
        if self._transcribe_busy.locked():
            self.query_one(TranscriptPanel).system_message("wait for transcription to finish...", Log.REC)
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
            # Update transcriber language if it supports it (Qwen3, faster-whisper)
            if hasattr(self.transcriber, '_language'):
                self.transcriber._language = lang_code
            _get_config().update({"last_model": self._model_name, "last_language": lang_code})
            self.query_one(TranscriptPanel).system_message(f"language set to {lang_name}", Log.SYS)
            self._update_telemetry()

        self.push_screen(LanguageSelectScreen(self._language), on_lang_selected)

    def action_tag_speakers(self):
        """Open speaker tagging modal."""
        if not self._diarizer_loaded:
            self.query_one(TranscriptPanel).system_message(
                "speaker identification not loaded yet", Log.SYS
            )
            return

        session_speakers = self.diarizer.get_all_session_speakers()
        if not session_speakers:
            self.query_one(TranscriptPanel).system_message("no speakers detected yet", Log.SYS)
            return

        # Build speaker list for the modal
        speaker_names = self.diarizer.get_speaker_names()
        speakers = []
        for sid, seg_count in sorted(session_speakers.items()):
            name = speaker_names.get(sid, f"Speaker {sid}")
            color = self.diarizer.get_speaker_color(sid)
            tagged = sid in speaker_names
            speakers.append({
                "id": sid,
                "name": name,
                "color": color,
                "segments": seg_count,
                "tagged": tagged,
            })

        # Collect known names from persistent profiles for autocomplete
        known_names = list(self.speaker_store.get_profile_names().values())

        def on_tag_result(result):
            if result is None:
                return
            if "merge_source" in result:
                self._apply_speaker_merge(
                    result["merge_source"], result["merge_target"]
                )
            else:
                self._apply_speaker_tag(result["speaker_id"], result["name"])

        self.push_screen(
            SpeakerTagScreen(speakers, known_names=known_names),
            on_tag_result,
        )

    def _apply_speaker_tag(self, speaker_id: int, name: str):
        """Apply a speaker tag: update diarizer, transcript, and persistent store."""
        # 1. Update the in-session diarizer
        self.diarizer.set_speaker_name(speaker_id, name)

        # 2. Get the session embeddings for enrollment
        segment_data = self.diarizer.get_segment_embeddings(speaker_id)
        embeddings = [emb for emb, dur in segment_data if dur >= 3.0]
        durations = [dur for emb, dur in segment_data if dur >= 3.0]

        # Fall back to all embeddings if none pass the 3s filter
        if not embeddings and segment_data:
            embeddings = [emb for emb, dur in segment_data]
            durations = [dur for emb, dur in segment_data]

        # 3. Check if this name matches an existing profile
        existing_profiles = self.speaker_store.get_all_profiles()
        matched_profile = None
        for p in existing_profiles:
            if p.name.lower() == name.lower():
                matched_profile = p
                break

        if matched_profile:
            # Update existing profile with new embeddings
            profile_id = matched_profile.id
            color = matched_profile.color
            for emb, dur in zip(embeddings, durations):
                self.speaker_store.update_profile_embedding(
                    profile_id, emb, duration=dur, confirmed=True
                )
        elif embeddings:
            # Create a new profile
            color = self.diarizer.get_speaker_color(speaker_id)
            profile_id = self.speaker_store.create_profile(
                name=name,
                color=color,
                embeddings=embeddings,
                durations=durations,
            )
        else:
            # No embeddings available — just rename in-session
            color = None
            profile_id = None

        if profile_id:
            self._speaker_profile_map[speaker_id] = profile_id

        # 4. Update transcript display
        self.query_one(TranscriptPanel).rename_speaker(speaker_id, name, color)

        # 5. Feedback
        self.query_one(TranscriptPanel).system_message(f"tagged as {name}", Log.SYS)
        self._update_telemetry()

    def _apply_speaker_merge(self, source_id: int, target_id: int):
        """Merge source session speaker into target."""
        transcript = self.query_one(TranscriptPanel)
        target_name = self.diarizer.get_speaker_name(target_id)
        target_color = self.diarizer.get_speaker_color(target_id)

        # Merge in-session: rename source entries to target
        transcript.rename_speaker(source_id, target_name, target_color)

        # Merge in diarizer (centroids, embeddings, cleanup)
        self.diarizer.merge_speakers(source_id, target_id)

        # If both have persistent profiles, merge those too
        source_pid = self._speaker_profile_map.get(source_id)
        target_pid = self._speaker_profile_map.get(target_id)
        if source_pid and target_pid and source_pid != target_pid:
            self.speaker_store.merge_profiles(source_pid, target_pid)
            self._speaker_profile_map.pop(source_id, None)

        transcript.system_message(
            f"merged Speaker {source_id} into {target_name}", Log.SYS
        )
        self._update_telemetry()

    def action_manage_profiles(self):
        """Open speaker profiles management screen."""
        profiles = self.speaker_store.get_all_profiles()

        def on_profile_result(result):
            if result is None:
                return
            action = result.get("action")
            if action == "rename":
                pid = result["profile_id"]
                new_name = result["name"]
                self.speaker_store.rename_profile(pid, new_name)
                self.query_one(TranscriptPanel).system_message(
                    f"profile renamed to {new_name}", Log.SYS
                )
                # Update in-session labels if this profile is active
                for sid, mapped_pid in self._speaker_profile_map.items():
                    if mapped_pid == pid:
                        self.diarizer.set_speaker_name(sid, new_name)
                        self.query_one(TranscriptPanel).rename_speaker(
                            sid, new_name
                        )
            elif action == "delete":
                pid = result["profile_id"]
                meta = next(
                    (p for p in profiles if p.id == pid), None
                )
                name = meta.name if meta else "unknown"
                self.speaker_store.delete_profile(pid)
                # Remove from session mapping
                self._speaker_profile_map = {
                    sid: p for sid, p in self._speaker_profile_map.items()
                    if p != pid
                }
                self.query_one(TranscriptPanel).system_message(
                    f"deleted profile: {name}", Log.SYS
                )
            elif action == "delete_all":
                self.speaker_store.delete_all_data()
                self._speaker_profile_map.clear()
                self.query_one(TranscriptPanel).system_message(
                    "all voice data deleted", Log.SYS
                )

        self.push_screen(SpeakerProfileScreen(profiles), on_profile_result)

    def action_explore_transcripts(self):
        """Open transcript explorer to browse and copy saved transcripts."""
        def on_result(result):
            if result is None:
                return
            transcript = self.query_one(TranscriptPanel)
            if "error" in result:
                transcript.system_message(result["error"], Log.SYS)
            elif "copied" in result:
                transcript.system_message(
                    f"copied transcript {result['copied']} to clipboard", Log.SYS
                )

        self.push_screen(TranscriptExplorerScreen(SESSIONS_DIR), on_result)

    def _swap_model(self, model_key: str):
        self._model_loaded = False
        self._model_name = model_key
        self._update_telemetry()
        # Don't null the old model here — a transcription worker may still be
        # using it. _do_swap acquires _transcribe_lock for the load, which
        # waits for any active MLX work to finish before constructing the new
        # model. The old model is replaced atomically in _on_swap_done.
        self.query_one(TranscriptPanel).system_message(
            f"switching to {model_key} (may take a minute)...", Log.SYS, {model_key: "rainbow"}
        )
        self._do_swap(model_key)

    @work(thread=True, exclusive=True, group="model_loading")
    def _do_swap(self, model_key: str):
        repo = AVAILABLE_MODELS[model_key]
        try:
            if model_key in QWEN3_MODELS:
                new_transcriber = Qwen3Transcriber(model=repo, language=self._language)
            elif model_key in FASTER_WHISPER_MODELS:
                new_transcriber = FasterWhisperTranscriber(model=repo, language=self._language)
            else:
                new_transcriber = WhisperTranscriber(model=repo)
            # Serialize Metal access against active transcription: the
            # _transcribe_lock blocks transcribe submissions for the duration
            # of the load; the executor guarantees the load runs on the same
            # MLX thread so per-thread stream bindings stay consistent.
            with self._transcribe_lock:
                self._mlx_executor.submit(new_transcriber.load).result()
            self.call_from_thread(self._on_swap_done, new_transcriber, model_key)
        except Exception as e:
            self.call_from_thread(
                self._on_swap_error, f"model switch failed: {e}"
            )

    def _on_swap_done(self, transcriber, model_key):
        old_transcriber = self.transcriber
        self.transcriber = transcriber
        self._model_name = model_key
        self._is_qwen3 = model_key in QWEN3_MODELS
        self._model_loaded = True
        # Drop the previous model and flush Metal so a swap doesn't strand
        # a whole model (~0.6–1.5GB) in the GPU cache. Done as two serialized
        # tasks on the single MLX thread: _unload releases the old model
        # (passed positionally, so its only reference is gone once the task
        # returns), then _flush runs gc + clear_cache with NO live reference
        # to the old model so its buffers can actually be reclaimed.
        # Fire-and-forget so the event loop isn't blocked.
        if old_transcriber is not None and old_transcriber is not transcriber:
            def _unload(old):
                try:
                    unload = getattr(old, "unload", None)
                    if callable(unload):
                        unload()
                except Exception:
                    pass
                # `old` is this task's only reference; it is released when
                # the task returns, before _flush (queued next) runs.

            def _flush():
                gc.collect()
                if sys.platform == "darwin":
                    try:
                        import mlx.core as mx
                        mx.clear_cache()
                    except Exception:
                        pass
            try:
                self._mlx_executor.submit(_unload, old_transcriber)
                self._cache_clear_future = self._mlx_executor.submit(_flush)
            except Exception:
                pass
        # Drop this frame's reference too, so nothing here keeps the old
        # model alive until _flush runs on the MLX thread.
        del old_transcriber
        _get_config().update({"last_model": model_key, "last_language": self._language})
        self.query_one(TranscriptPanel).system_message(f"model loaded: {model_key}", Log.SYS, {model_key: "rainbow"})
        self._update_telemetry()

    def _on_swap_error(self, msg: str):
        self.query_one(TranscriptPanel).system_message(msg)
        self._model_loaded = True
        self._update_telemetry()

    def action_export_transcript(self):
        """Open export modal to choose destination."""
        transcript = self.query_one(TranscriptPanel)
        if not transcript.get_entries():
            transcript.system_message("nothing to export", Log.REC)
            return

        def on_export_selected(destination):
            if destination is None:
                return
            if destination == "file":
                self._export_to_file()
            elif destination == "clipboard":
                self._export_to_clipboard()
            elif destination == "discard":
                self._discard_transcript()

        self.push_screen(ExportScreen(), on_export_selected)

    def _export_to_file(self):
        """Promote live file to final transcript."""
        transcript = self.query_one(TranscriptPanel)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        filename = self._session_start.strftime("%Y-%m-%d_%H%M%S") + "-transcript.md"
        filepath = SESSIONS_DIR / filename

        # Write the full markdown (cleaner than the append-mode live file)
        md = transcript.get_markdown(self._model_name, session_start=self._session_start, language=self._language or "")
        filepath.write_text(md, encoding="utf-8")

        # Remove the live file since we promoted it
        if self._live_file and self._live_file.exists():
            self._live_file.unlink()

        entry_count = len(transcript.get_entries())
        self._start_new_session()
        transcript.system_message(f"exported {entry_count} entries → {filepath}", Log.REC)

    def _export_to_clipboard(self):
        """Copy transcript to clipboard."""
        transcript = self.query_one(TranscriptPanel)
        cmd = _clipboard_cmd()
        if cmd is None:
            transcript.system_message("no clipboard tool found (install xclip, xsel, or wl-copy)", Log.REC)
            return
        text = transcript.get_plain_text()
        entry_count = len(transcript.get_entries())
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            self._start_new_session()
            transcript.system_message(f"copied {entry_count} entries to clipboard", Log.REC)
        except Exception:
            transcript.system_message("clipboard copy failed", Log.REC)

    def _discard_transcript(self):
        """Discard transcript and delete the live file."""
        entry_count = len(self.query_one(TranscriptPanel).get_entries())
        # Delete the live auto-save file
        if self._live_file and self._live_file.exists():
            self._live_file.unlink()
        self._start_new_session()
        self.query_one(TranscriptPanel).system_message(
            f"discarded {entry_count} entries", Log.REC
        )

    def _start_new_session(self):
        """Clear transcript and reset for a new session."""
        # Record session-speaker mappings before clearing
        self._record_session_stats()

        transcript = self.query_one(TranscriptPanel)
        transcript.clear()
        self.audio_buffer.clear()
        self._local_audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        self.vad.reset()
        if self._diarizer_loaded:
            self.diarizer.reset_session()
        self._speaker_profile_map.clear()
        self._prompt_times.clear()
        self._prompt_confirmations.clear()
        self._last_prompt_time = 0.0
        # Start fresh live file
        self._session_start = datetime.now()
        self._live_file = None
        self._live_header_written = False

    def _record_session_stats(self):
        """Record speaker-session mappings to persistent store."""
        if not self.speaker_store.is_open or not self._speaker_profile_map:
            return
        session_id = self._session_start.strftime("%Y-%m-%d_%H%M%S")
        session_speakers = self.diarizer.get_all_session_speakers()
        for sid, profile_id in self._speaker_profile_map.items():
            seg_count = session_speakers.get(sid, 0)
            try:
                self.speaker_store.record_session_speaker(
                    session_id, profile_id, sid, seg_count
                )
            except Exception:
                pass

    # ── P2P party mode (thin delegation to PartyManager) ────────

    def action_toggle_party(self):
        """Toggle party mode: N to join the party, N again to leave."""
        self._party.toggle()

    def _apply_party_color(self, color: str | None = None) -> None:
        """Set panel borders to the party color. Stays until you leave."""
        try:
            c = color or self._party.color_pri
            tp = self.query_one(TranscriptPanel)
            wf = self.query_one(WaveformWidget)
            fb = self.query_one("#footer-bar")
            tp.styles.border = ("heavy", c)
            tp.styles.border_title_color = c
            wf.styles.border = ("heavy", c)
            wf.styles.border_title_color = c
            fb.styles.border_top = ("heavy", c)
        except Exception:
            pass

    def _restore_borders(self) -> None:
        """Restore default border colors when leaving the party."""
        try:
            tp = self.query_one(TranscriptPanel)
            wf = self.query_one(WaveformWidget)
            fb = self.query_one("#footer-bar")
            tp.styles.border = ("heavy", "#00e5ff")
            tp.styles.border_title_color = "#00ffcc"
            wf.styles.border = ("heavy", "#6644cc")
            wf.styles.border_title_color = "#aa66ff"
            fb.styles.border_top = ("heavy", "#003344")
        except Exception:
            pass

    def _trigger_fireworks(self, burst_count: int = 8) -> None:
        """Mount the bloom overlay for a colour-wash effect on party join."""
        try:
            from tui.widgets.fireworks import FireworkOverlay
            overlay = FireworkOverlay(burst_count=burst_count)
            self.mount(overlay)
        except Exception:
            pass

    def _peer_bloom(self) -> None:
        """Brief flash to lighter shade when a peer joins, then back to party color."""
        try:
            light = self._party.color_light
            tp = self.query_one(TranscriptPanel)
            wf = self.query_one(WaveformWidget)
            fb = self.query_one("#footer-bar")
            tp.styles.border = ("heavy", light)
            wf.styles.border = ("heavy", light)
            fb.styles.border_top = ("heavy", light)
            self.set_timer(1.0, self._apply_party_color)
        except Exception:
            pass

    def action_show_help(self):
        self.push_screen(HelpScreen())

    def action_show_hivemind(self):
        """Open the hivemind discovery + opt-in menu."""
        try:
            from tui.widgets.hivemind_screen import HivemindScreen
            self.push_screen(HivemindScreen(self._hivemind))
        except Exception:
            log.warning("could not open hivemind screen", exc_info=True)

    def action_toggle_debug(self):
        self._debug = not self._debug
        state = "ON" if self._debug else "OFF"
        tp = self.query_one(TranscriptPanel)
        tp.system_message(f"debug mode {state}", Log.SYS)
        if self._debug and self._party.is_in_party:
            debug_text = self._party.format_debug_text(tp.merged_view)
            if debug_text:
                tp.system_message(debug_text)

    def action_toggle_merged_view(self):
        """Toggle between local and merged transcript view (P2P only)."""
        _pm = self._party
        if not _pm.session_mgr or not _pm.session_mgr.is_in_session:
            self.query_one(TranscriptPanel).system_message(
                "merged view requires an active P2P session", Log.PARTY
            )
            return
        tp = self.query_one(TranscriptPanel)
        new_state = not tp.merged_view
        tp.set_merged_view(
            new_state,
            assembler=_pm.assembler,
            local_name=_pm.display_name or "you",
            peer_names=_pm.get_peer_names(),
        )
        if new_state:
            tp.system_message("merged view — all peers, time-ordered [V] to toggle back", Log.PARTY)
        self._update_telemetry()

    def _refresh_merged_if_active(self):
        """Refresh merged view if it's currently active."""
        _pm = self._party
        tp = self.query_one(TranscriptPanel)
        if tp.merged_view and _pm.assembler:
            tp.refresh_merged(
                _pm.assembler,
                local_name=_pm.display_name or "you",
                peer_names=_pm.get_peer_names(),
            )

    def action_clear_transcript(self):
        """Clear display only — live file stays on disk as the record."""
        self.query_one(TranscriptPanel).clear()
        self.audio_buffer.clear()
        self._local_audio_buffer.clear()
        self._had_speech = False
        self._silence_chunks = 0
        self.vad.reset()
        if self._diarizer_loaded:
            self.diarizer.reset_session()
        if self._party.assembler:
            self._party.assembler.clear()
        self._speaker_profile_map.clear()

    def action_quit(self):
        if self._recording:
            self.push_screen(QuitConfirmScreen(), self._on_quit_confirm)
            return
        self._do_quit()

    def _on_quit_confirm(self, confirmed: bool) -> None:
        if confirmed:
            self._do_quit()

    def _do_quit(self):
        # Record stats while we still can (fast, synchronous)
        self._record_session_stats()

        # Terminate child processes BEFORE arming the hard-exit timer.
        # os._exit() (timer/SIGALRM backstop below) skips atexit handlers
        # and finalizers, so any subprocess not signaled here is orphaned:
        #  - system_capture.stop() is the ONLY path that SIGTERMs the Swift
        #    sck-helper (so it stops the SCStream / releases the CoreAudio
        #    tap) and tears down the BlackHole multi-output device.
        #  - diarizer.shutdown() sends MSG_SHUTDOWN to the PyTorch worker;
        #    it early-returns in ONNX "direct"/"inprocess" mode, so it is
        #    always safe to call.
        log.debug("quit: stopping child processes")
        try:
            self.audio_capture.stop()
            log.debug("quit: audio_capture stopped")
        except Exception:
            log.warning("quit: audio_capture.stop() failed", exc_info=True)
        try:
            self.system_capture.stop()
            log.debug("quit: system_capture stopped")
        except Exception:
            log.warning("quit: system_capture.stop() failed", exc_info=True)
        try:
            self.diarizer.shutdown()
            log.debug("quit: diarizer shut down")
        except Exception:
            log.warning("quit: diarizer.shutdown() failed", exc_info=True)
        log.debug("quit: child processes stopped, proceeding to hard exit")

        try:
            self.speaker_store.close()
        except Exception:
            pass
        try:
            self._mlx_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        # Flush any pending hivemind batch on EOF (spec §4.3 cadence).
        if self._hivemind is not None:
            try:
                self._hivemind.close()
            except Exception:
                log.warning("hivemind close failed", exc_info=True)

        # Kill the resource tracker process if it somehow spawned —
        # prevents leaked-semaphore warning printed to terminal after exit.
        try:
            import multiprocessing.resource_tracker as _rt
            pid = getattr(_rt._resource_tracker, '_pid', None)
            if pid:
                import signal
                sig = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
                os.kill(pid, sig)
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
        # Hard-exit after Textual restores the terminal.
        def _hard_exit():
            os._exit(0)
        threading.Timer(0.5, _hard_exit).start()
        # Backstop: if a C extension is holding the GIL during teardown, the
        # Timer thread can't run. SIGALRM is delivered by the kernel and the
        # default disposition terminates the process — this fires only if the
        # Timer didn't (terminates with status 142, but we're already exiting).
        try:
            import signal as _signal
            _signal.alarm(2)
        except (OSError, AttributeError, ValueError):
            pass  # platform without alarm (e.g. Windows) — Timer is the only path
        self.exit()



def main():
    import argparse

    # Resolve defaults: saved preferences > config defaults
    _cfg = _get_config()
    _saved_model = _cfg.get("last_model")
    _saved_lang = _cfg.get("last_language")
    _default_model = _saved_model if _saved_model in AVAILABLE_MODELS else DEFAULT_MODEL
    _default_lang = _saved_lang if _saved_lang in AVAILABLE_LANGUAGES else DEFAULT_LANGUAGE

    parser = argparse.ArgumentParser(description="VOXTERM — Local Voice Transcription TUI")
    parser.add_argument(
        "-m", "--model",
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
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Display name for P2P sessions",
    )
    parser.add_argument(
        "--session-create",
        action="store_true",
        help="Create a P2P session on launch",
    )
    parser.add_argument(
        "--session-join",
        type=str,
        default=None,
        metavar="CODE",
        help="Join a P2P session on launch (e.g. --session-join bacon-horse-galaxy)",
    )
    parser.add_argument(
        "--hivemind",
        choices=["auto", "on", "off"],
        default=_cfg.get("hivemind_mode") or "auto",
        help=(
            "Hivemind transcript-sink mode (default: auto). "
            "auto = mDNS-discover; on = require a sink; off = never POST."
        ),
    )
    parser.add_argument(
        "--hivemind-sink-url",
        type=str,
        default=_cfg.get("hivemind_sink_url") or None,
        metavar="URL",
        help=(
            "Override mDNS discovery and POST directly to this swf-node "
            "hivemind sink (e.g. http://convent.local:7777)."
        ),
    )
    parser.add_argument(
        "--hivemind-location",
        type=str,
        default=_cfg.get("hivemind_location") or "",
        metavar="LABEL",
        help="Optional location tag attached to every transcript batch.",
    )
    args = parser.parse_args()

    # Validate model choice
    if args.model not in AVAILABLE_MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    if args.list_models:
        print("Available models:")
        for name, repo in AVAILABLE_MODELS.items():
            tag = " (default)" if name == _default_model else ""
            if name in QWEN3_MODELS:
                backend = " [qwen3-asr]"
            elif name in FASTER_WHISPER_MODELS:
                backend = " [faster-whisper]"
            else:
                backend = " [whisper]"
            print(f"  {name:20s} → {repo}{backend}{tag}")
        sys.exit(0)

    model_repo = AVAILABLE_MODELS[args.model]
    model_name = args.model
    language = args.language

    # Pre-TUI setup: install BlackHole if Bluetooth output detected (macOS only)
    # Must happen before TUI launches — brew needs the live terminal for sudo
    if sys.platform == "darwin":
        try:
            from audio.platform import get_output_device_info
            from audio.blackhole import is_blackhole_installed
            dev_info = get_output_device_info()
            if dev_info.get("is_bluetooth") and not is_blackhole_installed():
                print(f"VOXTERM // Bluetooth output detected ({dev_info['name']})")
                print("Installing BlackHole for system audio capture...\n")
                result = subprocess.run(["brew", "install", "blackhole-2ch"])
                if result.returncode == 0:
                    print("\nBlackHole installed. Restarting audio service...")
                    subprocess.run(
                        ["sudo", "killall", "coreaudiod"],
                        timeout=15,
                    )
                    import time
                    time.sleep(2)  # Give CoreAudio time to restart and detect BlackHole
                    print("Audio service restarted.\n")
                else:
                    print("\nBlackHole install failed — system audio capture will be limited.\n")
        except Exception:
            pass

    # Prevent segfault: PortAudio/PyTorch/SpeechBrain C threads crash
    # during Python's shutdown when native objects are GC'd in random order.
    # atexit fires before finalizers; the finally block catches SystemExit.
    import atexit
    atexit.register(os._exit, 0)

    # Restore terminal on segfault so the shell doesn't get stuck in raw mode
    diagnostics.setup_signal_handlers()

    # Configure hivemind transcript-sink (spec §4.3). Failures here
    # never prevent voxterm from starting unless mode=on and discovery
    # comes up empty — that's the contract.
    hivemind_client = None
    try:
        from network.hivemind import (
            HivemindMode,
            HIVEMIND_SERVICE_TYPE,
            configure as _hivemind_configure,
        )
        _hm_mode = HivemindMode.parse(args.hivemind)
        if _hm_mode != HivemindMode.OFF and not args.hivemind_sink_url:
            print(
                f"VOXTERM // hivemind: searching for sink on "
                f"{HIVEMIND_SERVICE_TYPE} (mDNS, mode={_hm_mode.value})..."
            )

        # Read the user's persisted opt-in state. configure() folds
        # this in so a returning user gets push enabled automatically
        # without having to re-toggle every launch.
        _persisted_push_enabled = bool(_cfg.get("hivemind_push_enabled") or False)
        _persisted_pinned_pubkey = str(_cfg.get("hivemind_pinned_sink_pubkey") or "")

        def _on_hivemind_state_change(enabled: bool, pubkey: str) -> None:
            """Save the user's `h`-menu toggle to ConfigStore."""
            try:
                _cfg.update({
                    "hivemind_push_enabled": bool(enabled),
                    "hivemind_pinned_sink_pubkey": str(pubkey or ""),
                })
            except Exception:
                log.warning("hivemind state persist failed", exc_info=True)

        hivemind_client = _hivemind_configure(
            mode=_hm_mode,
            sink_url=args.hivemind_sink_url,
            location=args.hivemind_location,
            push_enabled=_persisted_push_enabled,
            pinned_sink_pubkey=_persisted_pinned_pubkey,
            on_state_change=_on_hivemind_state_change,
        )
        if hivemind_client is not None:
            sink = hivemind_client.active_sink()
            push_on = hivemind_client.push_enabled
            push_state = (
                "[pushing]" if push_on else "[not pushing — press 'h' to enable]"
            )
            if sink is not None:
                print(
                    f"VOXTERM // hivemind sink: {sink.transcripts_url} "
                    f"{push_state}"
                )
            else:
                print(
                    "VOXTERM // hivemind: no sink yet (auto-discovery still "
                    "running; tail ~/Documents/voxterm/voxterm.log with "
                    "VOXTERM_LOG_LEVEL=INFO to see when one appears)"
                )
    except RuntimeError as exc:
        # mode=on with no sink discovered; fail loudly per task spec.
        print(f"VOXTERM // hivemind error: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception:
        log.warning("hivemind configure failed", exc_info=True)
        hivemind_client = None

    # Persist hivemind preferences (mode + URL) so a CLI flag becomes
    # the new default without forcing the user to retype it.
    try:
        _cfg.update({
            "hivemind_mode": args.hivemind,
            "hivemind_sink_url": args.hivemind_sink_url or "",
            "hivemind_location": args.hivemind_location or "",
        })
    except Exception:
        pass

    # Bound the MLX GPU cache before any model loads (root-cause fix for
    # unbounded RSS growth during long sessions).
    configure_mlx_memory()

    # Launch TUI immediately — model loads in the background
    app = VoxTerm(
        transcriber=None, model_name=model_name, language=language,
        p2p_name=args.name,
        p2p_create=args.session_create,
        p2p_join_code=args.session_join,
        hivemind_client=hivemind_client,
    )

    # Global exception hooks — dump diagnostics on any uncaught crash
    diagnostics.setup_exception_hooks(app)

    try:
        app.run()
    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()
