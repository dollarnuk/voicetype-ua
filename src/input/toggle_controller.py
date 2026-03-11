"""Toggle controller - Push-On/Push-Off mode for voice recording."""

from typing import Callable, Optional, Set
import threading
import queue
import time
from loguru import logger

try:
    from pynput.keyboard import Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

from .hotkey_manager import HotkeyManager
from .audio_capture import AudioCapture


class ToggleController:
    """Toggle controller for voice input.

    Handles the push-on/push-off recording mechanism:
    - First hotkey press: start recording
    - Second hotkey press: stop recording and transcribe

    Unlike PTTController (hold-to-record), this uses a toggle approach
    where each press flips the recording state.

    Attributes:
        hotkey: The toggle hotkey string
    """

    # Debounce window in milliseconds (higher than PTT to prevent double-toggle)
    DEBOUNCE_MS = 200

    # Key mapping for comparison
    MODIFIER_KEYS = {
        Key.ctrl_l, Key.ctrl_r,
        Key.shift_l, Key.shift_r,
        Key.alt_l, Key.alt_r,
    }

    def __init__(
        self,
        hotkey_manager: HotkeyManager,
        audio_capture: AudioCapture,
        hotkey: str = "ctrl+space",
    ):
        """Initialize Toggle controller.

        Args:
            hotkey_manager: HotkeyManager instance
            audio_capture: AudioCapture instance
            hotkey: Toggle hotkey string
        """
        self.hotkey_manager = hotkey_manager
        self.audio_capture = audio_capture
        self.hotkey = hotkey

        self._is_active = False
        self._is_recording = False
        self._ptt_keys: Set = set()
        self._current_keys: Set = set()
        self._lock = threading.Lock()

        # Debounce state to prevent rapid toggling
        self._last_toggle_time: float = 0

        # Callbacks
        self._on_recording_start: Optional[Callable] = None
        self._on_recording_stop: Optional[Callable] = None
        self._on_transcription_ready: Optional[Callable] = None
        self._on_streaming_text: Optional[Callable[[str], None]] = None

        # Transcription queue (fixes thread accumulation issue)
        self._transcription_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running = False

        # Streaming mode
        self._streaming_enabled: bool = False
        self._streaming_text: str = ""  # Accumulated text during streaming

        # Parse hotkey
        self._parse_ptt_hotkey()

        logger.info(f"ToggleController initialized with hotkey: {hotkey}")

    def _parse_ptt_hotkey(self):
        """Parse the toggle hotkey string into key set."""
        parts = self.hotkey.lower().replace(" ", "").split("+")

        for part in parts:
            if part in HotkeyManager.KEY_MAP:
                self._ptt_keys.add(HotkeyManager.KEY_MAP[part])
            elif len(part) == 1:
                self._ptt_keys.add(KeyCode.from_char(part))

        logger.debug(f"Toggle keys parsed: {self._ptt_keys}")

    def _normalize_key(self, key) -> Optional:
        """Normalize key for comparison.

        Maps left/right variants of modifier keys to a single canonical form.
        """
        if isinstance(key, Key):
            if key in (Key.ctrl_l, Key.ctrl_r):
                return Key.ctrl_l
            if key in (Key.shift_l, Key.shift_r):
                return Key.shift_l
            if key in (Key.alt_l, Key.alt_r):
                return Key.alt_l
            return key
        elif isinstance(key, KeyCode):
            if key.char:
                return KeyCode.from_char(key.char.lower())
            return key
        return None

    def _on_key_press(self, key):
        """Handle key press for toggle.

        On each valid hotkey combo press, flip the recording state:
        - If not recording -> start recording
        - If recording -> stop recording and transcribe
        """
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        self._current_keys.add(normalized)

        # Check if toggle combo is pressed
        if self._ptt_keys.issubset(self._current_keys):
            with self._lock:
                now = time.time()
                elapsed_ms = (now - self._last_toggle_time) * 1000

                # Debounce: ignore if too soon after last toggle
                if elapsed_ms < self.DEBOUNCE_MS:
                    return

                self._last_toggle_time = now

                if not self._is_recording:
                    # Toggle ON: start recording
                    self._is_recording = True
                    self._is_active = True
                    self._start_recording()
                else:
                    # Toggle OFF: stop recording and transcribe
                    self._is_recording = False
                    self._is_active = False
                    self._stop_recording()

    def _on_key_release(self, key):
        """Handle key release — only used for tracking current keys.

        Toggle mode does not use release to stop recording; it uses
        the next press instead.
        """
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        self._current_keys.discard(normalized)

    def _start_recording(self):
        """Start audio recording."""
        logger.info("Toggle: Recording started")

        try:
            # Clear streaming state
            self._streaming_text = ""

            # Set streaming chunk callback if streaming enabled
            if self._streaming_enabled:
                self.audio_capture.set_chunk_callback(self._on_audio_chunk)

            self.audio_capture.start_recording()

            if self._on_recording_start:
                self._on_recording_start()

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            # Reset state on failure
            self._is_recording = False
            self._is_active = False

    def _on_audio_chunk(self, audio_chunk):
        """Handle streaming audio chunk.

        Called by AudioCapture when a chunk is ready during recording.
        """
        if not self._streaming_enabled or not self._on_streaming_text:
            return

        # Put chunk in queue for processing
        self._transcription_queue.put(("chunk", audio_chunk))

    def _stop_recording(self):
        """Stop recording and trigger transcription."""
        logger.info("Toggle: Recording stopped")

        try:
            audio_data = self.audio_capture.stop_recording()

            # Clear chunk callback
            self.audio_capture.set_chunk_callback(None)

            if self._on_recording_stop:
                self._on_recording_stop()

            # Queue for transcription
            if self._streaming_enabled:
                # In streaming mode, transcribe any remaining audio
                if len(audio_data) > 0:
                    self._transcription_queue.put(("final", audio_data))
            else:
                # Normal mode - transcribe full audio
                if self._on_transcription_ready and len(audio_data) > 0:
                    self._transcription_queue.put(("full", audio_data))

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")

    def _cancel_recording(self):
        """Cancel recording without transcription."""
        logger.debug("Toggle: Recording cancelled")

        try:
            if self.audio_capture.is_recording:
                self.audio_capture.stop_recording()

            # Clear chunk callback
            self.audio_capture.set_chunk_callback(None)

            if self._on_recording_stop:
                self._on_recording_stop()

        except Exception as e:
            logger.error(f"Failed to cancel recording: {e}")

    def set_callbacks(
        self,
        on_recording_start: Optional[Callable] = None,
        on_recording_stop: Optional[Callable] = None,
        on_transcription_ready: Optional[Callable] = None,
        on_streaming_text: Optional[Callable[[str], None]] = None,
    ):
        """Set toggle event callbacks.

        Args:
            on_recording_start: Called when recording starts
            on_recording_stop: Called when recording stops
            on_transcription_ready: Called with audio data when ready to transcribe
            on_streaming_text: Called with partial text during streaming
        """
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop
        self._on_transcription_ready = on_transcription_ready
        self._on_streaming_text = on_streaming_text

    def enable_streaming(self, enabled: bool = True):
        """Enable or disable streaming transcription mode.

        When enabled, text is inserted in real-time as you speak.

        Args:
            enabled: Whether to enable streaming
        """
        self._streaming_enabled = enabled
        self.audio_capture.enable_streaming(enabled)
        logger.info(f"Streaming mode {'enabled' if enabled else 'disabled'}")

    def _transcription_worker(self):
        """Worker thread that processes transcription queue."""
        logger.info("Toggle transcription worker started")
        while self._worker_running:
            try:
                # Wait for audio data with timeout (allows checking _worker_running)
                item = self._transcription_queue.get(timeout=0.5)
                if item is None:
                    self._transcription_queue.task_done()
                    continue

                # Handle different item types
                if isinstance(item, tuple):
                    item_type, audio_data = item

                    if item_type == "chunk" and self._on_streaming_text:
                        # Streaming chunk - call streaming callback
                        try:
                            self._on_streaming_text(audio_data, self._streaming_text)
                        except Exception as e:
                            logger.error(f"Streaming callback error: {e}")

                    elif item_type == "final" and self._on_streaming_text:
                        # Final chunk after toggle off - signal completion
                        try:
                            self._on_streaming_text(audio_data, self._streaming_text, is_final=True)
                        except Exception as e:
                            logger.error(f"Final streaming callback error: {e}")
                        # Clear streaming text
                        self._streaming_text = ""

                    elif item_type == "full" and self._on_transcription_ready:
                        # Full audio (non-streaming mode)
                        try:
                            self._on_transcription_ready(audio_data)
                        except Exception as e:
                            logger.error(f"Transcription callback error: {e}")
                else:
                    # Legacy format - treat as full audio
                    if self._on_transcription_ready:
                        try:
                            self._on_transcription_ready(item)
                        except Exception as e:
                            logger.error(f"Transcription callback error: {e}")

                self._transcription_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription worker error: {e}")
        logger.info("Toggle transcription worker stopped")

    def start(self):
        """Start toggle controller."""
        # Start transcription worker thread
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="ToggleTranscriptionWorker",
        )
        self._worker_thread.start()

        # Set key callbacks on hotkey manager (need release too for key tracking)
        self.hotkey_manager.set_key_callbacks(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )

        if not self.hotkey_manager.is_running:
            self.hotkey_manager.start()

        logger.info("ToggleController started")

    def stop(self):
        """Stop toggle controller."""
        # Stop transcription worker
        self._worker_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        # Clear callbacks
        self.hotkey_manager.set_key_callbacks(None, None)

        # Cancel any in-progress recording
        if self._is_recording:
            self._cancel_recording()

        self._is_recording = False
        self._is_active = False
        self._current_keys.clear()

        logger.info("ToggleController stopped")

    def set_hotkey(self, hotkey: str):
        """Change the toggle hotkey.

        Args:
            hotkey: New hotkey string
        """
        self.hotkey = hotkey
        self._ptt_keys.clear()
        self._parse_ptt_hotkey()
        logger.info(f"Toggle hotkey changed to: {hotkey}")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def is_active(self) -> bool:
        """Check if toggle controller is active (same as is_recording)."""
        return self._is_active
