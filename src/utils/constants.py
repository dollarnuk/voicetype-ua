"""Application constants."""

import os
from pathlib import Path

# Application info
APP_NAME = "CORE"
APP_VERSION = "0.2.0"

# Paths
APP_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = APP_DIR / "config.json"
CONFIG_DEFAULT_FILE = APP_DIR / "config_default.json"
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"
RESOURCES_DIR = Path(__file__).parent.parent / "ui" / "resources"

# Ensure directories exist (may fail in PyInstaller environment)
try:
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
except Exception:
    pass

# Audio settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "float32"

# Deepgram models
DEEPGRAM_MODELS = {
    "nova-2-general": "nova-2-general",
    "nova-2-medical": "nova-2-medical",
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "uk": "Ukrainian",
    "en": "English",
    "auto": "Auto-detect",
}

# Default hotkeys
DEFAULT_HOTKEYS = {
    "push_to_talk": ["ctrl", "space"],
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
