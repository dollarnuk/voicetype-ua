"""Application constants."""

import os
from pathlib import Path

# Application info
APP_NAME = "VoiceType UA"
APP_VERSION = "0.1.0"

# Paths
APP_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = APP_DIR / "config.json"
DEFAULT_CONFIG_FILE = APP_DIR / "config_default.json"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Audio settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "float32"

# Whisper model sizes
WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "uk": "Ukrainian",
    "en": "English",
    "auto": "Auto-detect",
}

# Default hotkeys
DEFAULT_HOTKEYS = {
    "push_to_talk": ["ctrl", "shift", "space"],
    "toggle_language": ["ctrl", "shift", "l"],
    "open_settings": ["ctrl", "shift", ","],
    "open_history": ["ctrl", "shift", "h"],
}

# Status states
class Status:
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"
