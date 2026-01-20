"""Hotkey manager module - global keyboard shortcuts using pynput."""

from typing import Callable, Dict, Set, Optional, List
import threading
from loguru import logger

try:
    from pynput import keyboard
    from pynput.keyboard import Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not installed. Install with: pip install pynput")


class HotkeyManager:
    """Global hotkey manager using pynput.

    Registers and handles global keyboard shortcuts.
    Supports modifier keys (ctrl, shift, alt) with regular keys.

    Example:
        manager = HotkeyManager()
        manager.register_hotkey("ctrl+shift+space", my_callback)
        manager.start()
    """

    # Key mapping for string to pynput key conversion
    KEY_MAP = {
        "ctrl": Key.ctrl_l,
        "ctrl_l": Key.ctrl_l,
        "ctrl_r": Key.ctrl_r,
        "shift": Key.shift_l,
        "shift_l": Key.shift_l,
        "shift_r": Key.shift_r,
        "alt": Key.alt_l,
        "alt_l": Key.alt_l,
        "alt_r": Key.alt_r,
        "space": Key.space,
        "enter": Key.enter,
        "tab": Key.tab,
        "escape": Key.esc,
        "esc": Key.esc,
        "backspace": Key.backspace,
        "delete": Key.delete,
        "home": Key.home,
        "end": Key.end,
        "pageup": Key.page_up,
        "pagedown": Key.page_down,
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "f1": Key.f1,
        "f2": Key.f2,
        "f3": Key.f3,
        "f4": Key.f4,
        "f5": Key.f5,
        "f6": Key.f6,
        "f7": Key.f7,
        "f8": Key.f8,
        "f9": Key.f9,
        "f10": Key.f10,
        "f11": Key.f11,
        "f12": Key.f12,
    }

    def __init__(self):
        """Initialize the hotkey manager."""
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput not available")

        self._hotkeys: Dict[frozenset, Dict] = {}
        self._current_keys: Set = set()
        self._listener: Optional[keyboard.Listener] = None
        self._is_running = False
        self._lock = threading.Lock()

        # Callbacks for key events (for PTT)
        self._on_key_press: Optional[Callable] = None
        self._on_key_release: Optional[Callable] = None

        logger.info("HotkeyManager initialized")

    def _parse_hotkey(self, hotkey_str: str) -> frozenset:
        """Parse hotkey string to set of keys.

        Args:
            hotkey_str: Hotkey string like "ctrl+shift+space"

        Returns:
            Frozenset of pynput key objects
        """
        keys = set()
        parts = hotkey_str.lower().replace(" ", "").split("+")

        for part in parts:
            if part in self.KEY_MAP:
                keys.add(self.KEY_MAP[part])
            elif len(part) == 1:
                # Single character key
                keys.add(KeyCode.from_char(part))
            else:
                logger.warning(f"Unknown key: {part}")

        return frozenset(keys)

    def register_hotkey(
        self,
        hotkey_str: str,
        callback: Callable,
        on_press: bool = True,
        on_release: bool = False,
    ):
        """Register a hotkey combination.

        Args:
            hotkey_str: Hotkey string (e.g., "ctrl+shift+space")
            callback: Function to call when hotkey is triggered
            on_press: Trigger on key press
            on_release: Trigger on key release
        """
        keys = self._parse_hotkey(hotkey_str)

        if not keys:
            logger.error(f"Invalid hotkey: {hotkey_str}")
            return

        with self._lock:
            self._hotkeys[keys] = {
                "callback": callback,
                "on_press": on_press,
                "on_release": on_release,
                "string": hotkey_str,
            }

        logger.info(f"Registered hotkey: {hotkey_str}")

    def unregister_hotkey(self, hotkey_str: str):
        """Unregister a hotkey.

        Args:
            hotkey_str: Hotkey string to unregister
        """
        keys = self._parse_hotkey(hotkey_str)

        with self._lock:
            if keys in self._hotkeys:
                del self._hotkeys[keys]
                logger.info(f"Unregistered hotkey: {hotkey_str}")

    def set_key_callbacks(
        self,
        on_press: Optional[Callable] = None,
        on_release: Optional[Callable] = None,
    ):
        """Set callbacks for all key events.

        Used for Push-to-Talk functionality.

        Args:
            on_press: Callback for key press events
            on_release: Callback for key release events
        """
        self._on_key_press = on_press
        self._on_key_release = on_release

    def _normalize_key(self, key) -> Optional:
        """Normalize key to comparable form."""
        if isinstance(key, Key):
            # Normalize left/right modifiers
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

    def _on_press(self, key):
        """Handle key press event."""
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        self._current_keys.add(normalized)

        # Check for hotkey match
        current_frozen = frozenset(self._current_keys)

        with self._lock:
            for hotkey_keys, hotkey_data in self._hotkeys.items():
                if hotkey_keys == current_frozen and hotkey_data["on_press"]:
                    try:
                        hotkey_data["callback"]()
                        logger.debug(f"Hotkey triggered: {hotkey_data['string']}")
                    except Exception as e:
                        logger.error(f"Hotkey callback error: {e}")

        # Global key press callback
        if self._on_key_press:
            try:
                self._on_key_press(key)
            except Exception as e:
                logger.error(f"Key press callback error: {e}")

    def _on_release(self, key):
        """Handle key release event."""
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        # Check for hotkey match before removing key
        current_frozen = frozenset(self._current_keys)

        with self._lock:
            for hotkey_keys, hotkey_data in self._hotkeys.items():
                if hotkey_keys == current_frozen and hotkey_data["on_release"]:
                    try:
                        hotkey_data["callback"]()
                        logger.debug(f"Hotkey released: {hotkey_data['string']}")
                    except Exception as e:
                        logger.error(f"Hotkey callback error: {e}")

        # Remove key from current set
        self._current_keys.discard(normalized)

        # Global key release callback
        if self._on_key_release:
            try:
                self._on_key_release(key)
            except Exception as e:
                logger.error(f"Key release callback error: {e}")

    def start(self):
        """Start listening for hotkeys."""
        if self._is_running:
            logger.warning("HotkeyManager already running")
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        self._is_running = True
        logger.info("HotkeyManager started")

    def stop(self):
        """Stop listening for hotkeys."""
        if not self._is_running:
            return

        if self._listener:
            self._listener.stop()
            self._listener = None

        self._is_running = False
        self._current_keys.clear()
        logger.info("HotkeyManager stopped")

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._is_running

    def get_registered_hotkeys(self) -> List[str]:
        """Get list of registered hotkeys.

        Returns:
            List of hotkey strings
        """
        with self._lock:
            return [data["string"] for data in self._hotkeys.values()]

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
