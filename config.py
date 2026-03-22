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
MAX_BUFFER_SECONDS = 3.0
MIN_BUFFER_SECONDS = 1.0
SILENCE_THRESHOLD = 0.012
SILENCE_TRIGGER_SECONDS = 0.3
VAD_THRESHOLD = 0.5           # Silero VAD speech probability threshold

# Session persistence
LIVE_DIR = __import__("pathlib").Path.home() / "Documents" / "voxterm" / ".live"

# System audio capture — compiled Swift helper cached here
BIN_DIR = __import__("pathlib").Path.home() / "Documents" / "voxterm" / ".bin"

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

# P2P networking
P2P_TCP_PORT = 9900
P2P_UDP_PORT = 9901
P2P_HEARTBEAT_INTERVAL = 1.0       # seconds between heartbeats
P2P_HEARTBEAT_TIMEOUT = 5.0        # seconds without heartbeat → peer dead
P2P_PROTO_VERSION = 1
P2P_MAX_PEERS = 20
P2P_AUDIO_FRAME_MS = 20            # milliseconds per UDP audio frame
P2P_CLOCK_SYNC_WINDOW = 20         # sliding window of offset samples
P2P_SERVICE_TYPE = "_voxterm._tcp.local."
