# VOXTERM Configuration

# Audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = "float32"

# Transcription — Qwen3-ASR (primary) + legacy Whisper models
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
# Which model keys use Qwen3-ASR vs Whisper backend
QWEN3_MODELS = {"qwen3-0.6b", "qwen3-1.7b"}
WHISPER_MODEL = "mlx-community/whisper-small-mlx"  # legacy default

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
MAX_BUFFER_SECONDS = 5.0
MIN_BUFFER_SECONDS = 1.5
SILENCE_THRESHOLD = 0.015
SILENCE_TRIGGER_SECONDS = 1.0

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
