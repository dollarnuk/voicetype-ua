"""System tray application - PyQt6 GUI for CORE."""

import sys
from typing import Optional
from pathlib import Path

from loguru import logger

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QSystemTrayIcon,
        QMenu,
        QMessageBox,
    )
    from PyQt6.QtGui import QIcon, QAction
    from PyQt6.QtCore import QTimer, pyqtSignal, QObject
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    logger.warning("PyQt6 not installed. Install with: pip install PyQt6")

from utils.constants import APP_NAME, APP_VERSION, Status
from utils.logger import setup_logger
from utils.metrics import MetricsCollector
from engine.transcriber import Transcriber
from engine.text_processor import TextProcessor
from engine.dictionary import Dictionary
from input.audio_capture import AudioCapture
from input.hotkey_manager import HotkeyManager
from input.ptt_controller import PTTController
from input.toggle_controller import ToggleController
from output.text_inserter import TextInserter
from data.config_manager import ConfigManager
from data.history_storage import HistoryStorage
from ui.theme import ThemeManager
from ui.overlay_widget import RecordingOverlay
from ui.sound_player import SoundPlayer
from ui.settings_dialog import SettingsDialog
from ui.history_window import HistoryWindow
from ui.styles import MAIN_STYLE
from utils import os_utils


class SignalEmitter(QObject):
    """Qt signal emitter for thread-safe UI updates."""

    status_changed = pyqtSignal(str)
    transcription_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    audio_level_changed = pyqtSignal(float)
    deepgram_text_ready = pyqtSignal(str, bool)  # text, is_final


class TrayApp(QApplication):
    """System tray application for CORE.

    Runs in the background with system tray icon.
    Provides access to settings, history, and status.
    """

    def __init__(self, argv: list, debug: bool = False):
        """Initialize the tray application.

        Args:
            argv: Command line arguments
            debug: Enable debug mode
        """
        if not PYQT6_AVAILABLE:
            raise RuntimeError("PyQt6 not available")

        super().__init__(argv)

        # Setup logging
        setup_logger(debug=debug)
        logger.info(f"Starting {APP_NAME} v{APP_VERSION} (GUI mode)")

        # Don't quit when last window closes
        self.setQuitOnLastWindowClosed(False)

        # Signal emitter for thread-safe updates
        self.signals = SignalEmitter()
        self.signals.status_changed.connect(self._on_status_changed)
        self.signals.transcription_complete.connect(self._on_transcription_complete)
        self.signals.error_occurred.connect(self._on_error)

        # Load configuration
        self.config = ConfigManager()

        # Theme
        theme_name = self.config.get("ui.theme", "dark")
        self.theme = ThemeManager(theme_name)
        self.theme.apply(self)
        self.setStyleSheet(MAIN_STYLE)

        # State
        self._status = Status.IDLE
        self._current_language = self.config.get("general.language", "uk")

        # Tray icon pulse timer (for recording animation)
        self._tray_pulse_timer = None
        self._tray_pulse_state = False

        # Initialize components
        self._init_components()

        # Setup system tray
        self._init_tray()

        logger.info("Tray application initialized")

    def _init_components(self):
        """Initialize application components."""
        # Audio capture
        self.audio_capture = AudioCapture(
            sample_rate=self.config.get("audio.sample_rate", 16000),
            chunk_duration=self.config.get("transcription.chunk_duration", 1.5),
        )

        # Transcriber (API based)
        self.transcriber = Transcriber(
            api_key=self.config.get("transcription.deepgram_api_key", ""),
            model=self.config.get("transcription.deepgram_model", "nova-2-general"),
            language=self._current_language,
        )

        # Text inserter
        self.text_inserter = TextInserter(
            preserve_clipboard=self.config.get("output.preserve_clipboard", True),
        )

        # Hotkey manager
        self.hotkey_manager = HotkeyManager()

        # Input controller (PTT or Toggle based on config)
        input_mode = self.config.get("hotkeys.input_mode", "ptt")
        hotkey = self.config.get("hotkeys.push_to_talk", "ctrl+space")

        if input_mode == "toggle":
            self.controller = ToggleController(
                hotkey_manager=self.hotkey_manager,
                audio_capture=self.audio_capture,
                hotkey=hotkey,
            )
            logger.info("Using Toggle input mode")
        else:
            self.controller = PTTController(
                hotkey_manager=self.hotkey_manager,
                audio_capture=self.audio_capture,
                hotkey=hotkey,
            )
            logger.info("Using PTT input mode")

        # Set callbacks
        self.controller.set_callbacks(
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_transcription_ready=self._on_transcription_ready,
            on_streaming_text=self._on_streaming_text,
        )

        # Enable streaming mode if configured
        if self.config.get("transcription.streaming", True):
            self.controller.enable_streaming(True)
            logger.info("Streaming mode enabled")

        # Register language toggle
        self.hotkey_manager.register_hotkey(
            self.config.get("hotkeys.toggle_language", "ctrl+shift+l"),
            self._toggle_language,
        )

        # Connect audio level callback to signal
        self.audio_capture.set_audio_level_callback(
            lambda level: self.signals.audio_level_changed.emit(level)
        )

        # History storage
        self.history = HistoryStorage()

        # Dictionary + Text processor
        self.dictionary = Dictionary()
        self.text_processor = TextProcessor(
            capitalize=self.config.get("output.capitalize_sentences", True),
            auto_punctuation=self.config.get("output.auto_punctuation", True),
            dictionary=self.dictionary if self.config.get("output.dictionary_enabled", True) else None,
        )

        # Metrics collector
        self.metrics = MetricsCollector()

        # Потоковий текст від Deepgram
        self._current_streaming_text = ""

        # Settings/History windows (lazy)
        self._settings_dialog = None
        self._history_window = None

        # Overlay widget
        self._overlay = RecordingOverlay(self.theme.colors)
        self.signals.audio_level_changed.connect(self._overlay.update_audio_level)

        # Sound player
        self.sound_player = SoundPlayer(
            enabled=self.config.get("ui.play_sounds", True)
        )

    def _init_tray(self):
        """Initialize system tray icon and menu."""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)

        # Set icon using theme manager
        self.tray_icon.setIcon(self.theme.get_tray_icon(Status.IDLE))

        # Create menu
        menu = QMenu()

        # Status item (non-clickable)
        self.status_action = QAction(f"Статус: Готовий ({self._current_language.upper()})", self)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)

        menu.addSeparator()

        # Language toggle
        lang_action = QAction("Перемкнути мову (Ctrl+Shift+L)", self)
        lang_action.triggered.connect(self._toggle_language)
        menu.addAction(lang_action)

        menu.addSeparator()

        # Settings
        settings_action = QAction("Налаштування...", self)
        settings_action.triggered.connect(self._show_settings)
        menu.addAction(settings_action)

        # History
        history_action = QAction("Історія...", self)
        history_action.triggered.connect(self._show_history)
        menu.addAction(history_action)

        menu.addSeparator()

        # About
        about_action = QAction("Про програму", self)
        about_action.triggered.connect(self._show_about)
        menu.addAction(about_action)

        menu.addSeparator()

        # Exit
        exit_action = QAction("Вихід", self)
        exit_action.triggered.connect(self._quit_app)
        menu.addAction(exit_action)

        # Set menu
        self.tray_icon.setContextMenu(menu)

        # Set tooltip
        self.tray_icon.setToolTip(f"{APP_NAME}\nГотовий до запису")

        # Show tray icon
        self.tray_icon.show()

    def start(self):
        """Start the application."""
        logger.info("Starting application...")

        # Load model in background
        QTimer.singleShot(100, self._load_model)

        # Start PTT
        self.controller.start()
        
        # Ensure autostart registry matches config
        autostart_enabled = self.config.get("general.start_with_windows", False)
        os_utils.set_autostart(autostart_enabled)

        logger.info("Application started")

    def _load_model(self):
        """Load transcription model (called async)."""
        self._set_status(Status.PROCESSING)
        self.tray_icon.setToolTip(f"{APP_NAME}\nЗавантаження моделі...")

        if self.transcriber.load_model():
            self._set_status(Status.IDLE)
            self.tray_icon.setToolTip(f"{APP_NAME}\nГотовий до запису")
            logger.info("Model loaded successfully")
        else:
            self._set_status(Status.ERROR)
            self.signals.error_occurred.emit("Не вдалося завантажити модель")

    def _on_recording_start(self):
        """Called when recording starts."""

        self._set_status(Status.RECORDING)
        self.tray_icon.setToolTip(f"{APP_NAME}\nЗапис...")

        # Show overlay and play sound
        if self.config.get("ui.show_status_indicator", True):
            self._overlay.show_recording(self._current_language)
        self.sound_player.play("record_start")

        # Запуск WebSocket сесії Deepgram при старті запису
        if self.config.get("transcription.streaming", True):
            self._current_streaming_text = ""
            self.transcriber.start_streaming(
                callback=self._on_deepgram_message,
                language=self._current_language
            )

    def _on_recording_stop(self):
        """Called when recording stops."""
        self._set_status(Status.PROCESSING)
        self.tray_icon.setToolTip(f"{APP_NAME}\nОбробка...")

        # Update overlay to processing state
        if self.config.get("ui.show_status_indicator", True):
            self._overlay.show_processing()
        self.sound_player.play("record_stop")

    def _on_transcription_ready(self, audio_data):
        """Process transcription (called from worker thread)."""
        try:
            text, confidence = self.transcriber.transcribe(
                audio_data,
                language=self._current_language,
            )

            if text:
                # Apply text processing pipeline
                text = self.text_processor.process(text, language=self._current_language)

                # Wait for user to release all keys
                import time
                time.sleep(0.3)

                # Insert text
                self.text_inserter.insert_text(text)

                # Save to history
                duration_ms = int(len(audio_data) / 16000 * 1000)
                self.history.add_entry(
                    text=text,
                    language=self._current_language,
                    duration_ms=duration_ms,
                    confidence=confidence,
                )

                # Emit signal
                self.signals.transcription_complete.emit(text)
            else:
                logger.warning("No speech detected")

            self._set_status(Status.IDLE)
            self.tray_icon.setToolTip(f"{APP_NAME}\nГотовий до запису")
            self._overlay.hide_recording()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.signals.error_occurred.emit(str(e))
            self._set_status(Status.ERROR)
            self._overlay.hide_recording()
            self.sound_player.play("error")

    def _on_streaming_text(self, audio_chunk, is_final: bool = False):
        """Обробка чанку аудіо для Deepgram стрімінгу."""
        if not is_final:
            self.transcriber.send_audio_chunk(audio_chunk)
        else:
            # Завершуємо WebSocket сесію
            self.transcriber.stop_streaming()
            
            # Save full text to history if any
            full_text = self._current_streaming_text.strip()
            if full_text:
                self.history.add_entry(
                    text=full_text,
                    language=self._current_language,
                    duration_ms=0, # Could calculate from chunks
                    confidence=0.9,
                )
                self.signals.transcription_complete.emit(full_text)
                self._current_streaming_text = ""

            self._set_status(Status.IDLE)
            self.tray_icon.setToolTip(f"{APP_NAME}\nГотовий до запису")
            self._overlay.hide_recording()

    def _on_deepgram_message(self, text: str, is_final: bool):
        """Коллбек від Transcriber при отриманні тексту від Deepgram.
        
        УВАГА: Цей метод викликається з ФОНОВОГО потоку Deepgram WebSocket!
        Не можна напряму симулювати клавіші — Windows ігнорує SendInput з фонових потоків.
        Використовуємо QTimer.singleShot(0) для гарантованого виконання в головному потоці Qt.
        """
        if not text:
            return

        logger.info(f"Received from Deepgram (bg thread): '{text}' (final={is_final})")
        
        # QTimer.singleShot(0, fn) — гарантовано виконається в головному потоці Qt event loop
        from functools import partial
        QTimer.singleShot(0, partial(self._insert_text_on_main_thread, text, is_final))

    def _insert_text_on_main_thread(self, text: str, is_final: bool):
        """Вставка тексту в ГОЛОВНОМУ потоці Qt (викликається через QTimer.singleShot).
        
        Тут безпечно симулювати клавіші через pyautogui/pynput.
        """
        try:
            # Обробка тексту
            processed_text = self.text_processor.process_streaming(text, self._current_language)
            
            if is_final:
                use_space = bool(self._current_streaming_text)
                self.text_inserter.append_text(processed_text, use_space=use_space)
                self._current_streaming_text += (" " + processed_text if use_space else processed_text)
                logger.info(f"✅ Text inserted on MAIN thread: '{processed_text}'")
            else:
                logger.debug(f"Deepgram interim: '{processed_text}'")
        except Exception as e:
            logger.error(f"Error inserting text on main thread: {e}")


    def _get_new_text(self, previous: str, current: str) -> str:
        """Get only the new text that wasn't in previous transcription.

        Uses word-based comparison to handle slight transcription variations.

        Args:
            previous: Previously inserted text
            current: New full transcription

        Returns:
            Only the new portion of text
        """
        if not previous:
            return current

        # Normalize texts for comparison
        prev_clean = previous.strip().lower()
        curr_clean = current.strip().lower()

        # If current starts with previous, return the suffix
        if curr_clean.startswith(prev_clean):
            new_part = current[len(previous):].strip()
            return new_part

        # Word-based comparison for more flexibility
        prev_words = prev_clean.split()
        curr_words = current.strip().split()

        # Find where current diverges from previous
        match_len = 0
        for i, (p, c) in enumerate(zip(prev_words, [w.lower() for w in curr_words])):
            if p == c:
                match_len = i + 1
            else:
                break

        # Return words after the match
        if match_len < len(curr_words):
            new_words = curr_words[match_len:]
            return " ".join(new_words)

        return ""

    def _toggle_language(self):
        """Toggle between Ukrainian and English."""
        self._current_language = "en" if self._current_language == "uk" else "uk"
        self._update_status_text()

        # Show notification
        self.tray_icon.showMessage(
            APP_NAME,
            f"Мова: {'Українська' if self._current_language == 'uk' else 'English'}",
            QSystemTrayIcon.MessageIcon.Information,
            1500,
        )

        logger.info(f"Language switched to: {self._current_language}")

    def _set_status(self, status: str):
        """Set current status."""
        self._status = status
        self._update_tray_icon(status)
        self.signals.status_changed.emit(status)

    def _update_tray_icon(self, status: str):
        """Update tray icon based on status and manage pulse animation."""
        self.tray_icon.setIcon(self.theme.get_tray_icon(status))

        # Start/stop pulse animation for recording
        if status == Status.RECORDING:
            if self._tray_pulse_timer is None:
                self._tray_pulse_timer = QTimer(self)
                self._tray_pulse_timer.timeout.connect(self._pulse_tray_icon)
            self._tray_pulse_timer.start(600)
        else:
            if self._tray_pulse_timer is not None:
                self._tray_pulse_timer.stop()
            self._tray_pulse_state = False

    def _pulse_tray_icon(self):
        """Alternate tray icon for pulsing recording indicator."""
        self._tray_pulse_state = not self._tray_pulse_state
        if self._tray_pulse_state:
            self.tray_icon.setIcon(self.theme.get_recording_icon_alt())
        else:
            self.tray_icon.setIcon(self.theme.get_tray_icon(Status.RECORDING))

    def _on_status_changed(self, status: str):
        """Handle status change (UI thread)."""
        self._update_status_text()

    def _update_status_text(self):
        """Update status text in menu."""
        status_text = {
            Status.IDLE: "Готовий",
            Status.RECORDING: "Запис...",
            Status.PROCESSING: "Обробка...",
            Status.ERROR: "Помилка",
        }.get(self._status, "Невідомо")

        lang_text = "UK" if self._current_language == "uk" else "EN"
        self.status_action.setText(f"Статус: {status_text} ({lang_text})")

    def _on_transcription_complete(self, text: str):
        """Handle transcription complete (UI thread)."""
        if self.config.get("ui.notification_on_complete", False):
            self.tray_icon.showMessage(
                APP_NAME,
                f"Розпізнано: {text[:50]}...",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

    def _on_error(self, error: str):
        """Handle error (UI thread)."""
        self.tray_icon.showMessage(
            APP_NAME,
            f"Помилка: {error}",
            QSystemTrayIcon.MessageIcon.Warning,
            3000,
        )

    def _show_settings(self):
        """Show settings dialog."""
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(
                config_manager=self.config,
                audio_capture=self.audio_capture,
            )
            self._settings_dialog.settings_changed.connect(self._on_settings_changed)
        self._settings_dialog.exec()

    def _on_settings_changed(self):
        """Apply changed settings without full restart."""
        # Update sound player
        self.sound_player.enabled = self.config.get("ui.play_sounds", True)

        # Update text processor
        self.text_processor._capitalize = self.config.get("output.capitalize_sentences", True)
        self.text_processor._auto_punctuation = self.config.get("output.auto_punctuation", True)

        # Update autostart
        autostart_enabled = self.config.get("general.start_with_windows", False)
        os_utils.set_autostart(autostart_enabled)

        logger.info("Settings applied")

    def _show_history(self):
        """Show history window."""
        if self._history_window is None:
            self._history_window = HistoryWindow(history_storage=self.history)
        self._history_window.show()
        self._history_window.raise_()
        self._history_window.activateWindow()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            None,
            f"Про {APP_NAME}",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            "<p>Голосовий набір тексту для Windows</p>"
            "<p>Підтримка української та англійської мов</p>"
            "<p>Оптимізовано під Deepgram API для максимальної швидкості</p>"
            "<hr>"
            "<p><b>Гарячі клавіші:</b></p>"
            f"<p>Push-to-talk: {self.config.get('hotkeys.push_to_talk')}</p>"
            f"<p>Перемикання мови: {self.config.get('hotkeys.toggle_language')}</p>",
        )

    def _quit_app(self):
        """Quit the application."""
        logger.info("Quitting application...")

        # Stop PTT
        self.controller.stop()
        self.hotkey_manager.stop()

        # Unload model
        self.transcriber.unload_model()

        # Close history
        self.history.close()

        # Hide tray
        self.tray_icon.hide()

        # Quit
        self.quit()


def run_gui(debug: bool = False):
    """Run the GUI application."""
    if not PYQT6_AVAILABLE:
        logger.error("PyQt6 not available. Install with: pip install PyQt6")
        return 1

    app = TrayApp(sys.argv, debug=debug)
    app.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui(debug=True))
