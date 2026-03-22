# VOXTERM Configuration

import sys

# Audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = "float32"

# Transcription — platform-aware model registry
if sys.platform == "darwin":
    # macOS: Qwen3-ASR (primary, MLX) + mlx-whisper (fallback)
    DEFAULT_MODEL = "qwen3-0.6b"
    AVAILABLE_MODELS = {
        "qwen3-0.6b":  "Qwen/Qwen3-ASR-0.6B",
        "qwen3-1.7b":  "Qwen/Qwen3-ASR-1.7B",
        "tiny":        "mlx-community/whisper-tiny",
        "small":       "mlx-community/whisper-small-mlx",
        "medium":      "mlx-community/whisper-medium-mlx",
        "large-v3":    "mlx-community/whisper-large-v3-mlx",
        "turbo":       "mlx-community/whisper-large-v3-turbo",
        "distil-v3":   "distil-whisper/distil-large-v3",
    }
    QWEN3_MODELS = {"qwen3-0.6b", "qwen3-1.7b"}
    WHISPER_MODEL = "mlx-community/whisper-small-mlx"
else:
    # Linux: faster-whisper (CTranslate2 backend)
    DEFAULT_MODEL = "fw-small"
    AVAILABLE_MODELS = {
        "fw-tiny":           "tiny",
        "fw-base":           "base",
        "fw-small":          "small",
        "fw-medium":         "medium",
        "fw-large-v3":       "large-v3",
        "fw-distil-large-v3": "distil-large-v3",
    }
    QWEN3_MODELS: set[str] = set()
    WHISPER_MODEL = ""

# Set of model keys that use faster-whisper backend
FASTER_WHISPER_MODELS = {"fw-tiny", "fw-base", "fw-small", "fw-medium", "fw-large-v3", "fw-distil-large-v3"}

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

# Session persistence & system audio capture paths
from paths import LIVE_DIR, BIN_DIR

# Diarizer subprocess
DIARIZER_TIMEOUT = 5.0        # seconds to wait for subprocess response
DIARIZER_MAX_RESTARTS = 3     # max restarts before falling back to in-process
DIARIZER_RESTART_WINDOW = 60  # seconds — restart counter resets after this

# Crash reporting
CRASH_LOG_MAX_COUNT = 50      # max crash logs to keep (rotated on startup)

# Waveform
WAVEFORM_FPS = 15
WAVEFORM_HEIGHT = 11

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
