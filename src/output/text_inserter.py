"""Text inserter module - insert transcribed text into active window."""

import time
from typing import Optional
from loguru import logger

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False
    logger.warning("pyperclip not installed. Install with: pip install pyperclip")

try:
    from pynput.keyboard import Key, Controller
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not installed. Install with: pip install pynput")

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable fail-safe for automation
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class TextInserter:
    """Insert text into the currently active window.

    Uses clipboard + Ctrl+V method for reliable Unicode support.
    This is the most compatible method for Ukrainian text.

    Attributes:
        preserve_clipboard: Whether to restore original clipboard content
        paste_delay: Delay before pasting (seconds)
        add_trailing_space: Add space after inserted text
    """

    def __init__(
        self,
        preserve_clipboard: bool = True,
        paste_delay: float = 0.05,
        add_trailing_space: bool = True,
    ):
        """Initialize text inserter.

        Args:
            preserve_clipboard: Restore clipboard after insertion
            paste_delay: Delay between copy and paste (seconds)
            add_trailing_space: Add trailing space to text
        """
        if not PYPERCLIP_AVAILABLE:
            raise RuntimeError("pyperclip not available")
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput not available")

        self.preserve_clipboard = preserve_clipboard
        self.paste_delay = paste_delay
        self.add_trailing_space = add_trailing_space

        self._keyboard = Controller()

        logger.info("TextInserter initialized")

    def insert_text(self, text: str) -> bool:
        """Insert text into the active window.

        Uses clipboard + Ctrl+V method for best Unicode support.

        Args:
            text: Text to insert

        Returns:
            True if insertion was successful
        """
        if not text:
            logger.warning("Empty text, nothing to insert")
            return False

        try:
            # Add trailing space if configured
            if self.add_trailing_space and not text.endswith(" "):
                text = text + " "

            # Save original clipboard content
            original_clipboard = None
            if self.preserve_clipboard:
                try:
                    original_clipboard = pyperclip.paste()
                except Exception:
                    original_clipboard = None

            # Copy text to clipboard
            pyperclip.copy(text)

            # Small delay for clipboard to update
            time.sleep(self.paste_delay)

            # Release any held modifier keys first (virtual release)
            for key in [Key.ctrl, Key.ctrl_l, Key.ctrl_r,
                        Key.shift, Key.shift_l, Key.shift_r,
                        Key.alt, Key.alt_l, Key.alt_r]:
                try:
                    self._keyboard.release(key)
                except Exception:
                    pass

            # Also use pyautogui to release modifiers (more reliable)
            if PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.keyUp('ctrl')
                    pyautogui.keyUp('shift')
                    pyautogui.keyUp('alt')
                except Exception:
                    pass

            # Wait for user to physically release all keys
            time.sleep(0.4)

            # Simulate Ctrl+V - use pyautogui if available (more reliable)
            if PYAUTOGUI_AVAILABLE:
                pyautogui.hotkey('ctrl', 'v', interval=0.05)
            else:
                self._keyboard.press(Key.ctrl)
                self._keyboard.press("v")
                self._keyboard.release("v")
                self._keyboard.release(Key.ctrl)

            # Wait for paste to complete
            time.sleep(self.paste_delay)

            # Restore original clipboard
            if self.preserve_clipboard and original_clipboard is not None:
                time.sleep(0.1)  # Extra delay before restoring
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    pass

            logger.info(f"Inserted text: '{text[:30]}...' ({len(text)} chars)")
            return True

        except Exception as e:
            logger.error(f"Failed to insert text: {e}")
            return False

    def append_text(self, text: str, use_space: bool = True) -> bool:
        """Append text during streaming (faster, no key release wait).

        Optimized for real-time text insertion during PTT hold.
        Doesn't wait for modifier key release since user is still holding PTT.

        Args:
            text: Text to append
            use_space: Add space before text if needed

        Returns:
            True if successful
        """
        if not text:
            return False

        try:
            # Add leading space if needed
            if use_space and not text.startswith(" "):
                text = " " + text

            # Copy to clipboard
            pyperclip.copy(text)
            time.sleep(0.02)  # Minimal delay

            # Quick paste using pyautogui (doesn't interfere with held keys)
            if PYAUTOGUI_AVAILABLE:
                # Release only Ctrl temporarily for paste
                pyautogui.keyUp('ctrl')
                time.sleep(0.05)
                pyautogui.hotkey('ctrl', 'v', interval=0.02)
            else:
                # Fallback to pynput
                self._keyboard.release(Key.ctrl)
                self._keyboard.release(Key.ctrl_l)
                self._keyboard.release(Key.ctrl_r)
                time.sleep(0.05)
                self._keyboard.press(Key.ctrl)
                self._keyboard.press("v")
                self._keyboard.release("v")
                self._keyboard.release(Key.ctrl)

            logger.debug(f"Appended text: '{text[:20]}...'")
            return True

        except Exception as e:
            logger.error(f"Failed to append text: {e}")
            return False

    def type_text(self, text: str, delay: float = 0.02) -> bool:
        """Type text character by character.

        Alternative method that doesn't use clipboard.
        Slower but doesn't affect clipboard content.

        Note: May not work well with all characters (especially Ukrainian).
        Prefer insert_text() for Unicode support.

        Args:
            text: Text to type
            delay: Delay between keystrokes (seconds)

        Returns:
            True if typing was successful
        """
        if not text:
            return False

        try:
            for char in text:
                self._keyboard.type(char)
                time.sleep(delay)

            logger.info(f"Typed text: '{text[:30]}...'")
            return True

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False

    def insert_with_enter(self, text: str) -> bool:
        """Insert text and press Enter.

        Useful for chat applications or command lines.

        Args:
            text: Text to insert

        Returns:
            True if successful
        """
        if self.insert_text(text):
            time.sleep(self.paste_delay)
            self._keyboard.press(Key.enter)
            self._keyboard.release(Key.enter)
            return True
        return False

    def clear_input(self):
        """Clear current input field (Ctrl+A, Delete)."""
        try:
            self._keyboard.press(Key.ctrl)
            self._keyboard.press("a")
            self._keyboard.release("a")
            self._keyboard.release(Key.ctrl)
            time.sleep(0.02)
            self._keyboard.press(Key.delete)
            self._keyboard.release(Key.delete)
        except Exception as e:
            logger.error(f"Failed to clear input: {e}")

    @staticmethod
    def get_clipboard_text() -> Optional[str]:
        """Get current clipboard text.

        Returns:
            Clipboard text or None if empty/unavailable
        """
        if not PYPERCLIP_AVAILABLE:
            return None

        try:
            return pyperclip.paste()
        except Exception:
            return None

    @staticmethod
    def set_clipboard_text(text: str) -> bool:
        """Set clipboard text.

        Args:
            text: Text to copy to clipboard

        Returns:
            True if successful
        """
        if not PYPERCLIP_AVAILABLE:
            return False

        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False
