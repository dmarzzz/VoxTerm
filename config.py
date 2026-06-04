# VOXTERM Configuration

VERSION = "0.3.0"

import importlib.util
import sys
import platform

# Audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = "float32"

# Transcription — platform-aware model registry
if sys.platform == "darwin" and platform.machine() == "arm64":
    # macOS: Qwen3-ASR (primary, MLX) + mlx-whisper (fallback)
    DEFAULT_MODEL = "qwen3-0.6b"
    AVAILABLE_MODELS = {
        "qwen3-0.6b":  "Qwen/Qwen3-ASR-0.6B",
        "qwen3-1.7b":  "Qwen/Qwen3-ASR-1.7B",
        # NVIDIA Parakeet FastConformer (TDT) via parakeet-mlx. The 0.6B is the
        # supported, non-streaming sibling of nvidia/nemotron-speech-streaming
        # (whose causal cache-aware encoder parakeet-mlx can't load yet — see
        # ParakeetTranscriber docstring). The 1.1B is the higher-param variant.
        "parakeet-0.6b": "mlx-community/parakeet-tdt-0.6b-v3",
        "parakeet-1.1b": "mlx-community/parakeet-tdt-1.1b",
        "tiny":        "mlx-community/whisper-tiny",
        "small":       "mlx-community/whisper-small-mlx",
        "medium":      "mlx-community/whisper-medium-mlx",
        "large-v3":    "mlx-community/whisper-large-v3-mlx",
        "turbo":       "mlx-community/whisper-large-v3-turbo",
        "distil-v3":   "distil-whisper/distil-large-v3",
    }
    QWEN3_MODELS = {"qwen3-0.6b", "qwen3-1.7b"}
    # NVIDIA Parakeet FastConformer models via parakeet-mlx.
    PARAKEET_MODELS = {"parakeet-0.6b", "parakeet-1.1b"}
    WHISPER_MODEL = "mlx-community/whisper-small-mlx"
    FASTER_WHISPER_MODELS: set[str] = set()
elif sys.platform == "darwin":
    # macOS Intel: MLX has no x86_64 wheels, so use faster-whisper.
    AVAILABLE_MODELS = {
        "fw-tiny":           "tiny",
        "fw-base":           "base",
        "fw-small":          "small",
        "fw-medium":         "medium",
        "fw-large-v3":       "large-v3",
        "fw-distil-large-v3": "distil-large-v3",
    }
    DEFAULT_MODEL = "fw-small"
    QWEN3_MODELS = set()
    PARAKEET_MODELS: set[str] = set()
    WHISPER_MODEL = None
    FASTER_WHISPER_MODELS = set(AVAILABLE_MODELS)
elif sys.platform.startswith("linux"):
    # Linux: Qwen3-ASR (primary, via qwen-asr/PyTorch) + faster-whisper (fallback)
    _HAS_QWEN_ASR = __import__("importlib.util", fromlist=["find_spec"]).find_spec("qwen_asr") is not None
    AVAILABLE_MODELS = {
        "fw-tiny":           "tiny",
        "fw-base":           "base",
        "fw-small":          "small",
        "fw-medium":         "medium",
        "fw-large-v3":       "large-v3",
        "fw-distil-large-v3": "distil-large-v3",
    }
    FASTER_WHISPER_MODELS = set(AVAILABLE_MODELS)
    if _HAS_QWEN_ASR:
        AVAILABLE_MODELS["qwen3-0.6b"] = "Qwen/Qwen3-ASR-0.6B"
        AVAILABLE_MODELS["qwen3-1.7b"] = "Qwen/Qwen3-ASR-1.7B"
    QWEN3_MODELS = set(AVAILABLE_MODELS) - FASTER_WHISPER_MODELS
    PARAKEET_MODELS: set[str] = set()
    DEFAULT_MODEL = "qwen3-0.6b" if _HAS_QWEN_ASR else "fw-small"
    WHISPER_MODEL = None
elif sys.platform == "win32":
    # Windows: Qwen3-ASR (primary, via qwen-asr/PyTorch) + faster-whisper (fallback)
    DEFAULT_MODEL = "qwen3-0.6b"
    AVAILABLE_MODELS = {
        "qwen3-0.6b":  "Qwen/Qwen3-ASR-0.6B",
        "qwen3-1.7b":  "Qwen/Qwen3-ASR-1.7B",
        "fw-tiny":           "tiny",
        "fw-base":           "base",
        "fw-small":          "small",
        "fw-medium":         "medium",
        "fw-large-v3":       "large-v3",
        "fw-distil-large-v3": "distil-large-v3",
    }
    QWEN3_MODELS = {"qwen3-0.6b", "qwen3-1.7b"}
    PARAKEET_MODELS: set[str] = set()
    WHISPER_MODEL = None
    FASTER_WHISPER_MODELS = {"fw-tiny", "fw-base", "fw-small", "fw-medium", "fw-large-v3", "fw-distil-large-v3"}
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

# Optional cross-platform streaming backend (sherpa-onnx). Surfaced ONLY when the package is
# installed AND a wheel exists for this platform (there is no Intel-macOS wheel). 100% additive:
# if absent, SHERPA_MODELS stays empty, AVAILABLE_MODELS/DEFAULT_MODEL are byte-for-byte unchanged,
# and the transcriber's sherpa dispatch branch is unreachable. sherpa statically links its own
# ONNX Runtime, so it cannot collide with VoxTerm's pinned onnxruntime (Silero VAD / 3D-Speaker).
_HAS_SHERPA = (
    importlib.util.find_spec("sherpa_onnx") is not None
    and not (sys.platform == "darwin" and platform.machine() != "arm64")
)
if _HAS_SHERPA:
    AVAILABLE_MODELS["sherpa-stream-en"] = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    # nemotron-EN streaming (NeMo FastConformer-RNNT, 0.6B, exported for sherpa-onnx)
    AVAILABLE_MODELS["sherpa-nemotron-en"] = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
SHERPA_MODELS = {"sherpa-stream-en", "sherpa-nemotron-en"} if _HAS_SHERPA else set()

# Language forcing for Qwen3-ASR (None = auto-detect)
DEFAULT_LANGUAGE = "en"
AVAILABLE_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "tr": "Turkish",
    "nl": "Dutch",
}
MAX_BUFFER_SECONDS = 3.0
MIN_BUFFER_SECONDS = 1.0
SILENCE_THRESHOLD = 0.012
SILENCE_TRIGGER_SECONDS = 0.3
VAD_THRESHOLD = 0.5           # Silero VAD speech probability threshold

# ── Platform-aware paths ─────────────────────────────────────
# (merged from paths.py)
import os as _os
from pathlib import Path as _Path

_home = _Path.home()

if sys.platform == "darwin":
    # macOS paths
    SESSIONS_DIR = _home / "Documents" / "voxterm-transcripts"
    DATA_DIR = _home / "Library" / "Application Support" / "voxterm"
    LIVE_DIR = SESSIONS_DIR / ".live"
    BIN_DIR = DATA_DIR / ".bin"
    CRASH_DIR = DATA_DIR / ".crashes"
    STATE_FILE = DATA_DIR / ".state.json"
elif sys.platform.startswith("linux"):
    # Linux — XDG-compliant paths
    _xdg_data = _Path(_os.environ.get("XDG_DATA_HOME", _home / ".local" / "share"))
    _xdg_config = _Path(_os.environ.get("XDG_CONFIG_HOME", _home / ".config"))
    DATA_DIR = _xdg_data / "voxterm"
    CONFIG_DIR = _xdg_config / "voxterm"
    SESSIONS_DIR = DATA_DIR
    LIVE_DIR = DATA_DIR / ".live"
    BIN_DIR = DATA_DIR / ".bin"
    CRASH_DIR = DATA_DIR / ".crashes"
    STATE_FILE = CONFIG_DIR / "state.json"
elif sys.platform == "win32":
    # Windows — %LOCALAPPDATA%\voxterm
    _appdata = _Path(_os.environ.get("LOCALAPPDATA", _home / "AppData" / "Local"))
    DATA_DIR = _appdata / "voxterm"
    SESSIONS_DIR = _home / "Documents" / "voxterm"
    LIVE_DIR = SESSIONS_DIR / ".live"
    BIN_DIR = DATA_DIR / "bin"
    CRASH_DIR = DATA_DIR / "crashes"
    STATE_FILE = DATA_DIR / "state.json"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

# Speaker database
DB_DIR = DATA_DIR
DB_PATH = DB_DIR / ".speakers.db"
BACKUP_DIR = DB_DIR / ".backups"

# Diarizer subprocess
DIARIZER_TIMEOUT = 5.0        # seconds to wait for subprocess response
DIARIZER_STARTUP_TIMEOUT = 30.0  # seconds to wait for subprocess READY on startup
DIARIZER_MAX_RESTARTS = 3     # max restarts before falling back to in-process
DIARIZER_RESTART_WINDOW = 60  # seconds — restart counter resets after this

# Speaker embedding model (3D-Speaker)
# Backend: "onnx" (default, no subprocess needed) or "pytorch" (subprocess)
SPEAKER_MODEL_BACKEND = "onnx"
# ONNX model name: "eres2net_large" (512-dim, best accuracy),
#                   "eres2netv2" (192-dim), or "campplus" (512-dim)
SPEAKER_MODEL_NAME = "eres2net_large"
# Embedding dim is derived from the model registry to prevent desync.
# Mapping: eres2net_large=512, eres2netv2=192, campplus=512
_SPEAKER_DIM_REGISTRY = {"eres2net_large": 512, "eres2netv2": 192, "campplus": 512}
SPEAKER_EMBEDDING_DIM = _SPEAKER_DIM_REGISTRY[SPEAKER_MODEL_NAME]
SPEAKER_MODEL_ONNX_CACHE = __import__("pathlib").Path.home() / ".cache" / "3dspeaker"

# Clustering (3D-Speaker algorithms for periodic re-clustering)
CLUSTER_AHC_THRESHOLD = 0.50       # AHC cosine distance stop threshold
CLUSTER_SPECTRAL_PVAL_BETA = 1.0   # p-value pruning aggressiveness (higher = more pruning)
CLUSTER_AHC_MAX_SAMPLES = 40       # above this, switch from AHC to spectral


# Language identification (3D-Speaker LID)
LID_ENABLED = True
LID_MODEL_NAME = "campplus_lid"
LID_MIN_AUDIO_SEC = 3.0       # min audio duration for reliable detection
LID_AUTO_SWITCH = False        # auto-switch transcription language on detection

# Crash reporting
CRASH_LOG_MAX_COUNT = 50      # max crash logs to keep (rotated on startup)

# ── Background-noise high-pass filter ────────────────────────
# Applied to mic input before VAD/transcription. Attenuates AC hum, fan rumble,
# and HVAC noise (all below ~150 Hz) that otherwise inflate apparent RMS and
# hold SILENCE_THRESHOLD open. Speech intelligibility lives at 1-3 kHz, so a
# 100 Hz cutoff leaves voice essentially intact.
# Toggle with VOXTERM_NOISE_FILTER=0 if you're recording in a quiet space and
# need the lowest male fundamentals (~80 Hz) fully preserved.
NOISE_FILTER_ENABLED = _os.environ.get("VOXTERM_NOISE_FILTER", "1").lower() in ("1", "true", "yes")
NOISE_FILTER_CUTOFF_HZ = float(_os.environ.get("VOXTERM_NOISE_FILTER_CUTOFF", "100"))
NOISE_FILTER_ORDER = int(_os.environ.get("VOXTERM_NOISE_FILTER_ORDER", "2"))

# ── Event stream for out-of-process consumers ─────────────────
# When enabled, VoxTerm appends a JSONL event stream to LIVE_DIR so external
# tools (LED matrices, OBS overlays, dashboards) can subscribe via file-tail.
# Off by default — set VOXTERM_EVENTS=1 to enable. One {"t": ..., "kind":
# ..., ...fields} object per line; see tui/events.py for the documented kinds.
EVENTS_ENABLED = _os.environ.get("VOXTERM_EVENTS", "0").lower() in ("1", "true", "yes")

# Dictation mode
DICTATION_HOTKEY_MACOS = ("cmd", "shift", "d")
DICTATION_HOTKEY_LINUX = ("super", "shift", "d")
DICTATION_HOTKEY_WINDOWS = ("ctrl", "shift", "d")
DICTATION_INTER_KEY_DELAY_MS = 1

# Waveform
WAVEFORM_FPS = 15
WAVEFORM_HEIGHT = 11

# Online diarization thresholds
MATCH_THRESHOLD = 0.55             # cosine sim above this → assign to existing speaker
MATCH_THRESHOLD_DISCOVERY = 0.70   # stricter threshold during discovery phase
NEW_SPEAKER_THRESHOLD = 0.45      # must be below this vs ALL centroids to create new speaker
CONTINUITY_BONUS = 0.05           # small bias toward keeping the same speaker across short turns
DIARIZATION_CONFLICT_MARGIN = 0.05  # if top-2 within this → prefer more established speaker
MERGE_THRESHOLD = 0.65            # pairwise cosine sim above this → merge clusters
QUALITY_RMS_THRESHOLD = 0.003     # min RMS energy for quality-gated centroid update
MERGE_INTERVAL = 5                # check for cluster merges every N identify() calls
RECLUSTER_INTERVAL = 8            # spectral re-clustering every N identify() calls
RECLUSTER_MIN_SEGMENTS = 4        # min total segments before re-clustering kicks in
LOOP_PROB = 0.99                  # VBx-style HMM self-transition probability
WHITEN_MIN_SEGMENTS = 8           # min segments before PLDA-lite whitening kicks in
SCD_CHANGE_THRESHOLD = 0.6        # cosine distance above this → speaker change detected
SCD_WINDOW_SEC = 2.0              # sliding window duration for SCD embedding extraction
SCD_HOP_SEC = 0.5                 # hop between consecutive SCD windows
CENTROID_EMA_ALPHA = 0.3          # EMA weight for new embedding when updating centroids
CENTROID_UPDATE_MIN_SIM = 0.50    # min cosine sim to centroid before updating it
MAX_EMBEDDINGS_PER_SPEAKER = 20   # cap per-speaker embedding retention
MAX_SEGMENT_ORDER = 200           # cap temporal segment history
DISCOVERY_PHASE_CALLS = 30        # number of identify() calls for discovery phase

# Cross-session speaker matching thresholds
CROSS_SESSION_HIGH_BASE = 0.55    # base threshold for auto-assign
CROSS_SESSION_MEDIUM = 0.35       # below this → unknown
ADAPTIVE_BOOST = 0.15             # extra strictness for new profiles
ADAPTIVE_DECAY_RATE = 10          # how fast the boost decays with samples
CROSS_SESSION_CONFLICT_MARGIN = 0.05  # if top-2 profiles within this → treat as ambiguous
COLD_START_MIN_CONFIRMED = 10     # min confirmed before auto-updates allowed

# Colors
BG_COLOR = "#0a0e14"
BORDER_COLOR = "#00e5ff"
ACCENT_COLOR = "#00ffcc"
TEXT_COLOR = "#c0c0c0"
DIM_COLOR = "#004040"
BRIGHT_COLOR = "#00ffcc"
WARN_COLOR = "#ff6600"
ERROR_COLOR = "#ff0040"
ACTIVE_COLOR = "#00ff88"

# Block characters for waveform (high to low intensity)
WAVE_BLOCKS = ["█", "▓", "▒", "░", "·"]

# P2P networking
P2P_TCP_PORT = 9900
P2P_UDP_PORT = 9901
P2P_HEARTBEAT_INTERVAL = 1.0       # seconds between heartbeats
P2P_HEARTBEAT_TIMEOUT = 5.0        # seconds without heartbeat → peer dead
P2P_PROTO_VERSION = 1
P2P_MAX_PEERS = 20
P2P_AUDIO_FRAME_MS = 20            # milliseconds per UDP audio frame
P2P_AUDIO_MERGE_ENABLED = True     # merge peer audio before transcription
P2P_MERGE_DELAY_MS = 60            # jitter buffer delay for audio merging
P2P_AUDIO_QUALITY_GATE = 0.003     # min RMS to include a source in the mix
P2P_CLOCK_SYNC_WINDOW = 20         # sliding window of offset samples
P2P_SERVICE_TYPE = "_voxterm._tcp.local."


# ── ConfigStore (merged from config_store.py) ────────────────

import json
import threading
from typing import Any, Optional


# Schema: key → default value
_DEFAULTS: dict[str, Any] = {
    "last_model": "",
    "last_language": "",
    "audio_retention": False,
    "export_format": "markdown",
    "summarization_model": "",
    "summarization_strength": "medium",
    "summarization_template": "tldr",
    "summarization_custom_prompt": "",
    "p2p_display_name": "",
    # Hivemind transcript-sink (spec §4.3 of SHAPE-ROTATOR-OS-SPEC.md).
    # `hivemind_mode` is one of "auto" | "on" | "off"; when set on the
    # CLI it persists so the next launch picks the same default.
    "hivemind_mode": "auto",
    "hivemind_sink_url": "",
    "hivemind_location": "",
    # User opt-in to actually push transcripts to a discovered sink.
    # Default False: voxterm always discovers but never pushes until
    # the user enables it from the `h` hivemind menu. Once enabled,
    # this flips to True and persists so subsequent launches push
    # silently. `--hivemind on` overrides this (force push regardless).
    "hivemind_push_enabled": False,
    # When push is enabled, we pin the choice to a specific sink
    # pubkey so a different sink showing up on the LAN doesn't get
    # transcripts by accident. Empty string = "any discovered sink".
    "hivemind_pinned_sink_pubkey": "",
}

# Expected types per key (for validation)
_TYPES: dict[str, type] = {
    "last_model": str,
    "last_language": str,
    "audio_retention": bool,
    "export_format": str,
    "summarization_model": str,
    "summarization_strength": str,
    "summarization_template": str,
    "summarization_custom_prompt": str,
    "p2p_display_name": str,
    "hivemind_mode": str,
    "hivemind_sink_url": str,
    "hivemind_location": str,
    "hivemind_push_enabled": bool,
    "hivemind_pinned_sink_pubkey": str,
}


class ConfigStore:
    """Persistent configuration with typed schema, merge semantics, and atomic writes.

    Reads existing .state.json on init (backward compatible with bare 2-key files).
    Writes use tmp+rename for atomicity.
    """

    def __init__(self, path: Optional[_Path] = None) -> None:
        if path is None:
            path = _Path.home() / "Documents" / "voxterm" / ".state.json"
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        """Load from disk, merging with defaults. Unknown keys are preserved."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    # Validate type for known keys; keep unknown keys as-is
                    expected = _TYPES.get(key)
                    if expected is None or isinstance(value, expected):
                        self._data[key] = value
        except Exception:
            pass

    def _save(self) -> None:
        """Atomic write: write to .tmp then os.replace()."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            _os.replace(tmp, self._path)
        except Exception:
            pass

    def get(self, key: str) -> Any:
        """Get a config value. Returns the default if key is in schema, else None."""
        with self._lock:
            return self._data.get(key, _DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        """Set a config value and persist to disk (merge, not overwrite)."""
        with self._lock:
            expected = _TYPES.get(key)
            if expected is not None and not isinstance(value, expected):
                raise TypeError(
                    f"Invalid type for config key '{key}': "
                    f"expected {expected.__name__}, got {type(value).__name__}"
                )
            self._data[key] = value
            self._save()

    def update(self, values: dict[str, Any]) -> None:
        """Set multiple config values and persist once (single disk write)."""
        with self._lock:
            for key, value in values.items():
                expected = _TYPES.get(key)
                if expected is not None and not isinstance(value, expected):
                    raise TypeError(
                        f"Invalid type for config key '{key}': "
                        f"expected {expected.__name__}, got {type(value).__name__}"
                    )
                self._data[key] = value
            self._save()

    @property
    def data(self) -> dict[str, Any]:
        """Return a snapshot of all config data."""
        with self._lock:
            return dict(self._data)
