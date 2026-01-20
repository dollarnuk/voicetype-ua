"""Push-to-Talk controller - manages voice recording with hotkey."""

from typing import Callable, Optional, Set
import threading
import time
from loguru import logger

try:
    from pynput.keyboard import Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

from .hotkey_manager import HotkeyManager
from .audio_capture import AudioCapture


class PTTController:
    """Push-to-Talk controller for voice input.

    Handles the hold-to-record mechanism:
    - Press hotkey: start recording
    - Hold: continue recording
    - Release: stop recording and transcribe

    Attributes:
        hotkey: The push-to-talk hotkey string
        min_hold_time: Minimum hold time to register as PTT (not tap)
    """

    # Minimum time (seconds) to consider it a "hold" vs a "tap"
    MIN_HOLD_TIME = 0.15

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
        hotkey: str = "ctrl+shift+space",
    ):
        """Initialize PTT controller.

        Args:
            hotkey_manager: HotkeyManager instance
            audio_capture: AudioCapture instance
            hotkey: Push-to-talk hotkey string
        """
        self.hotkey_manager = hotkey_manager
        self.audio_capture = audio_capture
        self.hotkey = hotkey

        self._is_active = False
        self._press_time: Optional[float] = None
        self._ptt_keys: Set = set()
        self._current_keys: Set = set()
        self._lock = threading.Lock()

        # Callbacks
        self._on_recording_start: Optional[Callable] = None
        self._on_recording_stop: Optional[Callable] = None
        self._on_transcription_ready: Optional[Callable] = None

        # Parse hotkey
        self._parse_ptt_hotkey()

        logger.info(f"PTTController initialized with hotkey: {hotkey}")

    def _parse_ptt_hotkey(self):
        """Parse the PTT hotkey string into key set."""
        parts = self.hotkey.lower().replace(" ", "").split("+")

        for part in parts:
            if part in HotkeyManager.KEY_MAP:
                self._ptt_keys.add(HotkeyManager.KEY_MAP[part])
            elif len(part) == 1:
                self._ptt_keys.add(KeyCode.from_char(part))

        logger.debug(f"PTT keys parsed: {self._ptt_keys}")

    def _normalize_key(self, key) -> Optional:
        """Normalize key for comparison."""
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
        """Handle key press for PTT."""
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        self._current_keys.add(normalized)

        # Check if PTT combo is pressed
        if self._ptt_keys.issubset(self._current_keys):
            with self._lock:
                if not self._is_active:
                    self._is_active = True
                    self._press_time = time.time()
                    self._start_recording()

    def _on_key_release(self, key):
        """Handle key release for PTT."""
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        # Check if released key is part of PTT combo
        if normalized in self._ptt_keys:
            with self._lock:
                if self._is_active:
                    hold_time = time.time() - self._press_time if self._press_time else 0

                    if hold_time >= self.MIN_HOLD_TIME:
                        # Valid PTT - stop and transcribe
                        self._stop_recording()
                    else:
                        # Too short - cancel
                        logger.debug(f"PTT cancelled (too short: {hold_time:.3f}s)")
                        self._cancel_recording()

                    self._is_active = False
                    self._press_time = None

        self._current_keys.discard(normalized)

    def _start_recording(self):
        """Start audio recording."""
        logger.info("PTT: Recording started")

        try:
            self.audio_capture.start_recording()

            if self._on_recording_start:
                self._on_recording_start()

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")

    def _stop_recording(self):
        """Stop recording and trigger transcription."""
        logger.info("PTT: Recording stopped")

        try:
            audio_data = self.audio_capture.stop_recording()

            if self._on_recording_stop:
                self._on_recording_stop()

            # Trigger transcription callback
            if self._on_transcription_ready and len(audio_data) > 0:
                # Run in separate thread to not block
                threading.Thread(
                    target=self._on_transcription_ready,
                    args=(audio_data,),
                    daemon=True,
                ).start()

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")

    def _cancel_recording(self):
        """Cancel recording without transcription."""
        logger.debug("PTT: Recording cancelled")

        try:
            if self.audio_capture.is_recording:
                self.audio_capture.stop_recording()

            if self._on_recording_stop:
                self._on_recording_stop()

        except Exception as e:
            logger.error(f"Failed to cancel recording: {e}")

    def set_callbacks(
        self,
        on_recording_start: Optional[Callable] = None,
        on_recording_stop: Optional[Callable] = None,
        on_transcription_ready: Optional[Callable] = None,
    ):
        """Set PTT event callbacks.

        Args:
            on_recording_start: Called when recording starts
            on_recording_stop: Called when recording stops
            on_transcription_ready: Called with audio data when ready to transcribe
        """
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop
        self._on_transcription_ready = on_transcription_ready

    def start(self):
        """Start PTT controller."""
        # Set key callbacks on hotkey manager
        self.hotkey_manager.set_key_callbacks(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )

        if not self.hotkey_manager.is_running:
            self.hotkey_manager.start()

        logger.info("PTTController started")

    def stop(self):
        """Stop PTT controller."""
        # Clear callbacks
        self.hotkey_manager.set_key_callbacks(None, None)

        if self._is_active:
            self._cancel_recording()

        self._is_active = False
        self._current_keys.clear()

        logger.info("PTTController stopped")

    def set_hotkey(self, hotkey: str):
        """Change the PTT hotkey.

        Args:
            hotkey: New hotkey string
        """
        self.hotkey = hotkey
        self._ptt_keys.clear()
        self._parse_ptt_hotkey()
        logger.info(f"PTT hotkey changed to: {hotkey}")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_active

    @property
    def is_active(self) -> bool:
        """Check if PTT is active."""
        return self._is_active
