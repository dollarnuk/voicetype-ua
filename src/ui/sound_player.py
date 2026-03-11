"""Sound effects player - generates and plays notification tones programmatically."""

import threading
from typing import Dict, Optional
import numpy as np
from loguru import logger

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class SoundPlayer:
    """Plays notification sound effects using numpy-generated tones.

    Generates simple sine wave tones programmatically (no external files needed).
    Plays asynchronously on default output device using sounddevice.

    Sounds:
        record_start: Quick high-pitched blip (880 Hz, 100ms)
        record_stop: Lower confirmation tone (440 Hz, 150ms)
        error: Low warning tone (220 Hz, 300ms)
    """

    SAMPLE_RATE = 44100

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._sounds: Dict[str, np.ndarray] = {}
        self._generate_sounds()

        logger.info(f"SoundPlayer initialized (enabled={enabled})")

    def _generate_tone(
        self,
        frequency: float,
        duration: float,
        volume: float = 0.25,
        fade_ms: float = 10.0,
    ) -> np.ndarray:
        """Generate a sine wave tone with fade-in/fade-out envelope.

        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            volume: Volume (0.0-1.0)
            fade_ms: Fade in/out duration in milliseconds

        Returns:
            Audio samples as float32 numpy array
        """
        samples = int(self.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t) * volume

        # Apply fade-in/fade-out envelope for smooth sound
        fade_samples = int(fade_ms / 1000 * self.SAMPLE_RATE)
        if fade_samples > 0 and fade_samples < samples // 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        return tone.astype(np.float32)

    def _generate_sounds(self):
        """Generate all notification sounds."""
        # Record start: quick high-pitched blip
        self._sounds["record_start"] = self._generate_tone(880, 0.08, volume=0.2)

        # Record stop: lower confirmation tone with slight decay
        stop_tone = self._generate_tone(520, 0.12, volume=0.18)
        stop_tail = self._generate_tone(440, 0.06, volume=0.12, fade_ms=15)
        self._sounds["record_stop"] = np.concatenate([stop_tone, stop_tail])

        # Error: low warning double-beep
        beep1 = self._generate_tone(280, 0.12, volume=0.2)
        gap = np.zeros(int(0.06 * self.SAMPLE_RATE), dtype=np.float32)
        beep2 = self._generate_tone(220, 0.15, volume=0.2)
        self._sounds["error"] = np.concatenate([beep1, gap, beep2])

    def play(self, sound_name: str):
        """Play a sound effect asynchronously.

        Args:
            sound_name: Name of the sound ('record_start', 'record_stop', 'error')
        """
        if not self._enabled or not SOUNDDEVICE_AVAILABLE:
            return

        if sound_name not in self._sounds:
            logger.warning(f"Unknown sound: {sound_name}")
            return

        def _play():
            try:
                sd.play(self._sounds[sound_name], samplerate=self.SAMPLE_RATE, blocking=False)
            except Exception as e:
                logger.debug(f"Sound play failed: {e}")

        threading.Thread(target=_play, daemon=True).start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
