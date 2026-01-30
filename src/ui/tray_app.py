"""System tray application - PyQt6 GUI for VoiceType UA."""

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
from core.transcriber import Transcriber
from core.hypothesis_buffer import HypothesisBuffer
from input.audio_capture import AudioCapture
from input.hotkey_manager import HotkeyManager
from input.ptt_controller import PTTController
from output.text_inserter import TextInserter
from data.config_manager import ConfigManager
from data.history_storage import HistoryStorage


class SignalEmitter(QObject):
    """Qt signal emitter for thread-safe UI updates."""

    status_changed = pyqtSignal(str)
    transcription_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)


class TrayApp(QApplication):
    """System tray application for VoiceType UA.

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

        # State
        self._status = Status.IDLE
        self._current_language = self.config.get("general.language", "uk")

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

        # Transcriber (lazy load model)
        self.transcriber = Transcriber(
            model_size=self.config.get("transcription.model_size", "base"),
            device=self.config.get("transcription.device", "auto"),
            compute_type=self.config.get("transcription.compute_type", "int8"),
        )

        # Text inserter
        self.text_inserter = TextInserter(
            preserve_clipboard=self.config.get("output.preserve_clipboard", True),
        )

        # Hotkey manager
        self.hotkey_manager = HotkeyManager()

        # PTT controller
        self.ptt_controller = PTTController(
            hotkey_manager=self.hotkey_manager,
            audio_capture=self.audio_capture,
            hotkey=self.config.get("hotkeys.push_to_talk", "ctrl+space"),
        )

        # Set callbacks
        self.ptt_controller.set_callbacks(
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_transcription_ready=self._on_transcription_ready,
            on_streaming_text=self._on_streaming_text,
        )

        # Enable streaming mode if configured
        streaming_enabled = self.config.get("transcription.streaming", True)
        if streaming_enabled:
            self.ptt_controller.enable_streaming(True)
            logger.info("Streaming mode enabled")

        # Register language toggle
        self.hotkey_manager.register_hotkey(
            self.config.get("hotkeys.toggle_language", "ctrl+shift+l"),
            self._toggle_language,
        )

        # History storage
        self.history = HistoryStorage()

        # LocalAgreement-2 buffer for streaming
        self._hypothesis_buffer = HypothesisBuffer()

    def _init_tray(self):
        """Initialize system tray icon and menu."""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)

        # Set icon (use default for now)
        icon_path = Path(__file__).parent / "resources" / "icons" / "tray_idle.png"
        if icon_path.exists():
            self.tray_icon.setIcon(QIcon(str(icon_path)))
        else:
            # Use application default icon
            self.tray_icon.setIcon(self.style().standardIcon(
                self.style().StandardPixmap.SP_MediaVolume
            ))

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
        self.ptt_controller.start()

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
        # Reset hypothesis buffer for new recording
        self._hypothesis_buffer.reset()

        self._set_status(Status.RECORDING)
        self.tray_icon.setToolTip(f"{APP_NAME}\nЗапис...")

    def _on_recording_stop(self):
        """Called when recording stops."""
        self._set_status(Status.PROCESSING)
        self.tray_icon.setToolTip(f"{APP_NAME}\nОбробка...")

    def _on_transcription_ready(self, audio_data):
        """Process transcription (called from worker thread)."""
        try:
            text, confidence = self.transcriber.transcribe(
                audio_data,
                language=self._current_language,
                beam_size=self.config.get("transcription.beam_size", 5),
            )

            if text:
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

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.signals.error_occurred.emit(str(e))
            self._set_status(Status.ERROR)

    def _on_streaming_text(self, audio_data, previous_text: str = "", is_final: bool = False):
        """Handle streaming audio chunk with LocalAgreement-2.

        Uses HypothesisBuffer to confirm words only when 2 consecutive
        transcriptions agree on them.

        Args:
            audio_data: Cumulative audio from start of recording
            previous_text: Previously confirmed and inserted text
            is_final: Whether this is the final chunk after PTT release
        """
        try:
            # Transcribe with word-level timestamps (required for LocalAgreement-2)
            words = self.transcriber.transcribe_chunk_with_timestamps(
                audio_data,
                language=self._current_language,
            )

            if words:
                # Insert words into hypothesis buffer (offset=0 since audio is cumulative)
                self._hypothesis_buffer.insert(words, offset=0)

                # Confirm words where buffer and new agree
                confirmed = self._hypothesis_buffer.flush()

                if confirmed:
                    # Insert only NEWLY confirmed words
                    new_text = " ".join([w[2] for w in confirmed])
                    use_space = bool(previous_text) and not new_text.startswith(" ")
                    self.text_inserter.append_text(new_text, use_space=use_space)

                    # Update streaming text with all confirmed text
                    self.ptt_controller._streaming_text = self._hypothesis_buffer.get_confirmed_text()

                    logger.debug(f"Streaming confirmed: '{new_text}' (total: {len(self._hypothesis_buffer)} words)")

            if is_final:
                # Finalize remaining words in buffer
                finalized = self._hypothesis_buffer.finalize()
                if finalized:
                    final_text = " ".join([w[2] for w in finalized])
                    use_space = bool(self.ptt_controller._streaming_text)
                    self.text_inserter.append_text(final_text, use_space=use_space)
                    self.ptt_controller._streaming_text += (" " + final_text if use_space else final_text)
                    logger.debug(f"Streaming finalized: '{final_text}'")

                # Save full text to history
                full_text_final = self.ptt_controller._streaming_text.strip()
                if full_text_final:
                    duration_ms = int(len(audio_data) / 16000 * 1000)
                    self.history.add_entry(
                        text=full_text_final,
                        language=self._current_language,
                        duration_ms=duration_ms,
                        confidence=0.8,  # Default confidence for streaming
                    )
                    self.signals.transcription_complete.emit(full_text_final)

                # Reset hypothesis buffer for next recording
                self._hypothesis_buffer.reset()

                self._set_status(Status.IDLE)
                self.tray_icon.setToolTip(f"{APP_NAME}\nГотовий до запису")

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            if is_final:
                self._hypothesis_buffer.reset()
                self._set_status(Status.IDLE)

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
        self.signals.status_changed.emit(status)

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
        # TODO: Implement settings dialog
        QMessageBox.information(
            None,
            "Налаштування",
            "Вікно налаштувань буде додано пізніше.\n\n"
            f"Push-to-talk: {self.config.get('hotkeys.push_to_talk')}\n"
            f"Перемикання мови: {self.config.get('hotkeys.toggle_language')}\n"
            f"Модель: {self.config.get('transcription.model_size')}",
        )

    def _show_history(self):
        """Show history window."""
        # TODO: Implement history window
        entries = self.history.get_entries(limit=10)
        if entries:
            text = "\n".join([f"- {e['text'][:50]}..." for e in entries])
        else:
            text = "Історія порожня"

        QMessageBox.information(
            None,
            "Історія",
            f"Останні записи:\n\n{text}",
        )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            None,
            f"Про {APP_NAME}",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            "<p>Голосовий набір тексту для Windows</p>"
            "<p>Підтримка української та англійської мов</p>"
            "<p>Використовує faster-whisper для локальної обробки</p>"
            "<hr>"
            "<p><b>Гарячі клавіші:</b></p>"
            f"<p>Push-to-talk: {self.config.get('hotkeys.push_to_talk')}</p>"
            f"<p>Перемикання мови: {self.config.get('hotkeys.toggle_language')}</p>",
        )

    def _quit_app(self):
        """Quit the application."""
        logger.info("Quitting application...")

        # Stop PTT
        self.ptt_controller.stop()
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
