"""VoiceType UA - Main entry point.

Voice-to-text application for Windows with Ukrainian language support.
Analog of SuperWhisper using faster-whisper for local transcription.
"""

import sys
import signal
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from utils.logger import setup_logger
from utils.constants import APP_NAME, APP_VERSION

from core.transcriber import Transcriber
from input.audio_capture import AudioCapture
from input.hotkey_manager import HotkeyManager
from input.ptt_controller import PTTController
from output.text_inserter import TextInserter
from data.config_manager import ConfigManager
from data.history_storage import HistoryStorage


class VoiceTypeApp:
    """Main application class.

    Coordinates all components for voice-to-text functionality.
    Can run in console mode (for testing) or GUI mode (with system tray).
    """

    def __init__(self, debug: bool = False):
        """Initialize the application.

        Args:
            debug: Enable debug mode with verbose logging
        """
        # Setup logging
        setup_logger(debug=debug)
        logger.info(f"Starting {APP_NAME} v{APP_VERSION}")

        # Load configuration
        self.config = ConfigManager()

        # Initialize components
        self._init_components()

        # State
        self._is_running = False
        self._current_language = self.config.get("general.language", "uk")

        logger.info("Application initialized")

    def _init_components(self):
        """Initialize all application components."""
        # Audio capture
        self.audio_capture = AudioCapture(
            sample_rate=self.config.get("audio.sample_rate", 16000),
            channels=self.config.get("audio.channels", 1),
        )

        # Transcriber
        self.transcriber = Transcriber(
            model_size=self.config.get("transcription.model_size", "base"),
            device=self.config.get("transcription.device", "auto"),
            compute_type=self.config.get("transcription.compute_type", "int8"),
        )

        # Text inserter
        self.text_inserter = TextInserter(
            preserve_clipboard=self.config.get("output.preserve_clipboard", True),
            add_trailing_space=self.config.get("output.add_trailing_space", True),
        )

        # Hotkey manager
        self.hotkey_manager = HotkeyManager()

        # PTT controller
        self.ptt_controller = PTTController(
            hotkey_manager=self.hotkey_manager,
            audio_capture=self.audio_capture,
            hotkey=self.config.get("hotkeys.push_to_talk", "ctrl+shift+space"),
        )

        # Set PTT callbacks
        self.ptt_controller.set_callbacks(
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_transcription_ready=self._on_transcription_ready,
        )

        # History storage
        self.history = HistoryStorage(
            max_entries=self.config.get("history.max_entries", 1000),
        )

        # Register additional hotkeys
        self._register_hotkeys()

    def _register_hotkeys(self):
        """Register additional hotkeys (language toggle, etc.)."""
        # Language toggle
        toggle_lang_key = self.config.get("hotkeys.toggle_language", "ctrl+shift+l")
        self.hotkey_manager.register_hotkey(toggle_lang_key, self._toggle_language)

    def _on_recording_start(self):
        """Called when recording starts."""
        logger.info("Recording started...")

    def _on_recording_stop(self):
        """Called when recording stops."""
        logger.info("Recording stopped, processing...")

    def _on_transcription_ready(self, audio_data):
        """Called when audio is ready for transcription.

        Args:
            audio_data: Recorded audio as numpy array
        """
        try:
            # Transcribe
            text, confidence = self.transcriber.transcribe(
                audio_data,
                language=self._current_language,
                beam_size=self.config.get("transcription.beam_size", 5),
                vad_filter=self.config.get("transcription.vad_filter", True),
            )

            if text:
                # Insert text
                self.text_inserter.insert_text(text)

                # Save to history
                if self.config.get("history.enabled", True):
                    duration_ms = int(len(audio_data) / 16000 * 1000)  # Assuming 16kHz
                    self.history.add_entry(
                        text=text,
                        language=self._current_language,
                        duration_ms=duration_ms,
                        confidence=confidence,
                    )

                logger.info(f"Transcribed ({self._current_language}): {text}")
            else:
                logger.warning("No speech detected")

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def _toggle_language(self):
        """Toggle between Ukrainian and English."""
        if self._current_language == "uk":
            self._current_language = "en"
        else:
            self._current_language = "uk"

        logger.info(f"Language switched to: {self._current_language}")

    def start(self):
        """Start the application."""
        if self._is_running:
            return

        logger.info("Starting application...")

        # Load model
        logger.info("Loading transcription model (this may take a moment)...")
        if not self.transcriber.load_model():
            logger.error("Failed to load transcription model")
            return

        # Start PTT controller
        self.ptt_controller.start()

        self._is_running = True
        logger.info(f"Application started. Press {self.config.get('hotkeys.push_to_talk')} to record.")

    def stop(self):
        """Stop the application."""
        if not self._is_running:
            return

        logger.info("Stopping application...")

        # Stop PTT
        self.ptt_controller.stop()

        # Stop hotkey manager
        self.hotkey_manager.stop()

        # Unload model
        self.transcriber.unload_model()

        # Close history
        self.history.close()

        self._is_running = False
        logger.info("Application stopped")

    def run_console(self):
        """Run in console mode (for testing)."""
        self.start()

        print(f"\n{APP_NAME} v{APP_VERSION}")
        print(f"Language: {self._current_language}")
        print(f"Push-to-talk: {self.config.get('hotkeys.push_to_talk')}")
        print(f"Toggle language: {self.config.get('hotkeys.toggle_language')}")
        print("\nPress Ctrl+C to exit.\n")

        # Handle Ctrl+C
        def signal_handler(sig, frame):
            print("\nExiting...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Keep running
        try:
            while self._is_running:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_NAME} - Voice to text")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--console", action="store_true", help="Run in console mode (no GUI)")
    parser.add_argument("--test", action="store_true", help="Run quick test (record 5 seconds)")

    args = parser.parse_args()

    if args.test:
        # Quick test mode
        run_test()
        return

    # Create and run application
    app = VoiceTypeApp(debug=args.debug)

    if args.console:
        app.run_console()
    else:
        # Run with GUI (system tray)
        try:
            from ui.tray_app import run_gui
            sys.exit(run_gui(debug=args.debug))
        except ImportError as e:
            logger.warning(f"GUI not available: {e}, falling back to console")
            app.run_console()


def run_test():
    """Run a quick test of the transcription pipeline."""
    setup_logger(debug=True)

    print(f"\n{APP_NAME} - Quick Test")
    print("=" * 40)

    # Test audio capture
    print("\n1. Testing audio capture...")
    try:
        audio_capture = AudioCapture()
        devices = audio_capture.get_input_devices()
        print(f"   Found {len(devices)} input device(s)")
        for d in devices[:3]:  # Show first 3
            print(f"   - {d['name']}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Test recording
    print("\n2. Recording 3 seconds of audio...")
    print("   Speak now!")
    try:
        audio_data = audio_capture.record_seconds(3.0)
        print(f"   Recorded {len(audio_data)} samples ({len(audio_data)/16000:.1f}s)")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Test transcription
    print("\n3. Transcribing (loading model)...")
    try:
        transcriber = Transcriber(model_size="base", device="auto")
        transcriber.load_model()

        text, confidence = transcriber.transcribe(audio_data, language="uk")
        print(f"   Result: \"{text}\"")
        print(f"   Confidence: {confidence:.2%}")

        transcriber.unload_model()
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    print("\n" + "=" * 40)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
