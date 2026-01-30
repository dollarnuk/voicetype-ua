"""Configuration manager - handles user settings."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

from utils.constants import CONFIG_FILE, CONFIG_DEFAULT_FILE


class ConfigManager:
    """Manages application configuration.

    Loads and saves user settings to JSON file.
    Provides default values for missing settings.

    Attributes:
        config_file: Path to user config file
        config: Current configuration dictionary
    """

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to config file (uses default if None)
        """
        self.config_file = config_file or CONFIG_FILE
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}

        # Load defaults
        self._load_defaults()

        # Load user config
        self.load()

        logger.info(f"ConfigManager initialized: {self.config_file}")

    def _load_defaults(self):
        """Load default configuration."""
        try:
            if CONFIG_DEFAULT_FILE.exists():
                with open(CONFIG_DEFAULT_FILE, "r", encoding="utf-8") as f:
                    self._defaults = json.load(f)
            else:
                # Hardcoded defaults if file not found
                self._defaults = {
                    "general": {
                        "start_minimized": True,
                        "start_with_windows": False,
                        "language": "uk",
                    },
                    "hotkeys": {
                        "push_to_talk": "ctrl+space",
                        "toggle_language": "ctrl+shift+l",
                        "open_settings": "ctrl+shift+comma",
                        "open_history": "ctrl+shift+h",
                    },
                    "audio": {
                        "input_device": None,
                        "sample_rate": 16000,
                        "channels": 1,
                        "silence_threshold": 0.01,
                        "silence_duration_ms": 500,
                    },
                    "transcription": {
                        "model_size": "small",
                        "device": "cuda",
                        "compute_type": "int8",
                        "beam_size": 6,
                        "vad_filter": True,
                        "word_timestamps": False,
                        "initial_prompt": None,
                    },
                    "output": {
                        "insert_method": "clipboard",
                        "add_trailing_space": True,
                        "capitalize_sentences": True,
                        "preserve_clipboard": True,
                    },
                    "history": {
                        "enabled": True,
                        "max_entries": 1000,
                        "save_audio": False,
                    },
                    "ui": {
                        "theme": "system",
                        "show_status_indicator": True,
                        "notification_on_complete": False,
                    },
                }
        except Exception as e:
            logger.error(f"Failed to load defaults: {e}")
            self._defaults = {}

    def load(self) -> bool:
        """Load configuration from file.

        Returns:
            True if loaded successfully
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info("Configuration loaded")
            else:
                # Use defaults
                self._config = self._defaults.copy()
                logger.info("Using default configuration")

            # Merge with defaults (add missing keys)
            self._config = self._merge_with_defaults(self._config, self._defaults)

            return True

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config = self._defaults.copy()
            return False

    def _merge_with_defaults(
        self,
        config: Dict[str, Any],
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge config with defaults, keeping user values.

        Args:
            config: User configuration
            defaults: Default configuration

        Returns:
            Merged configuration
        """
        result = defaults.copy()

        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_with_defaults(value, result[key])
            else:
                result[key] = value

        return result

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if saved successfully
        """
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)

            logger.info("Configuration saved")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Key in format "section.setting" (e.g., "audio.sample_rate")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            parts = key.split(".")
            value = self._config

            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return default

                if value is None:
                    return default

            return value

        except Exception:
            return default

    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """Set configuration value by dot-notation key.

        Args:
            key: Key in format "section.setting"
            value: Value to set
            save: Save to file after setting

        Returns:
            True if set successfully
        """
        try:
            parts = key.split(".")
            config = self._config

            # Navigate to parent
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]

            # Set value
            config[parts[-1]] = value

            if save:
                return self.save()

            return True

        except Exception as e:
            logger.error(f"Failed to set config: {e}")
            return False

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., "audio", "hotkeys")

        Returns:
            Section dictionary or empty dict
        """
        return self._config.get(section, {})

    def reset_to_defaults(self, save: bool = True) -> bool:
        """Reset configuration to defaults.

        Args:
            save: Save to file after reset

        Returns:
            True if reset successfully
        """
        self._config = self._defaults.copy()

        if save:
            return self.save()

        return True

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config.copy()
