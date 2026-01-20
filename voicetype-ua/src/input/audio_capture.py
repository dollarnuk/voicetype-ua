"""Audio capture module - microphone input using sounddevice."""

from typing import Optional, List, Dict, Callable
import threading
import queue
import numpy as np
from loguru import logger

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not installed. Install with: pip install sounddevice")


class AudioCapture:
    """Real-time audio capture from microphone.

    Captures audio in a background thread and stores in a buffer.
    Optimized for speech recognition with 16kHz sample rate.

    Attributes:
        sample_rate: Audio sample rate (default 16000 for Whisper)
        channels: Number of audio channels (1 for mono)
        dtype: Data type for audio samples
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        device: Optional[int] = None,
        blocksize: int = 1024,
    ):
        """Initialize audio capture.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of channels (1 = mono)
            dtype: Data type ('float32', 'int16')
            device: Input device index (None = default)
            blocksize: Samples per callback block
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available")

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.blocksize = blocksize

        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._audio_buffer: List[np.ndarray] = []
        self._is_recording = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_audio_level: Optional[Callable[[float], None]] = None

        logger.info(f"AudioCapture initialized: {sample_rate}Hz, {channels}ch, {dtype}")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ):
        """Callback for audio stream.

        Called by sounddevice for each audio block.
        """
        if status:
            logger.warning(f"Audio status: {status}")

        if self._is_recording:
            # Copy data to avoid reference issues
            audio_chunk = indata.copy().flatten()
            self._audio_queue.put(audio_chunk)

            # Calculate audio level for visualization
            if self._on_audio_level:
                level = np.abs(audio_chunk).mean()
                self._on_audio_level(level)

    def start_recording(self):
        """Start capturing audio from microphone."""
        if self._is_recording:
            logger.warning("Already recording")
            return

        with self._lock:
            # Clear previous data
            self._audio_buffer = []
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Start stream
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    device=self.device,
                    blocksize=self.blocksize,
                    callback=self._audio_callback,
                    latency="low",
                )
                self._stream.start()
                self._is_recording = True
                logger.info("Recording started")

            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                raise

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return captured audio.

        Returns:
            Audio data as numpy array (float32, mono)
        """
        if not self._is_recording:
            logger.warning("Not recording")
            return np.array([], dtype=np.float32)

        with self._lock:
            self._is_recording = False

            # Stop stream
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Collect all queued audio
            while not self._audio_queue.empty():
                try:
                    chunk = self._audio_queue.get_nowait()
                    self._audio_buffer.append(chunk)
                except queue.Empty:
                    break

            # Concatenate all chunks
            if self._audio_buffer:
                audio_data = np.concatenate(self._audio_buffer)
            else:
                audio_data = np.array([], dtype=np.float32)

            duration = len(audio_data) / self.sample_rate
            logger.info(f"Recording stopped: {duration:.2f}s ({len(audio_data)} samples)")

            return audio_data

    def record_seconds(self, duration: float) -> np.ndarray:
        """Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as numpy array
        """
        logger.info(f"Recording for {duration}s...")

        frames = int(duration * self.sample_rate)

        try:
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
            )
            sd.wait()  # Wait for recording to complete

            return audio.flatten()

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.array([], dtype=np.float32)

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def set_audio_level_callback(self, callback: Callable[[float], None]):
        """Set callback for audio level updates.

        Args:
            callback: Function called with audio level (0.0 - 1.0)
        """
        self._on_audio_level = callback

    @staticmethod
    def get_input_devices() -> List[Dict]:
        """Get list of available input devices.

        Returns:
            List of device dictionaries with 'index', 'name', 'channels'
        """
        if not SOUNDDEVICE_AVAILABLE:
            return []

        devices = []
        try:
            for i, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    devices.append({
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    })
        except Exception as e:
            logger.error(f"Failed to query devices: {e}")

        return devices

    @staticmethod
    def get_default_device() -> Optional[int]:
        """Get default input device index.

        Returns:
            Device index or None
        """
        if not SOUNDDEVICE_AVAILABLE:
            return None

        try:
            return sd.default.device[0]
        except Exception:
            return None

    def test_microphone(self, duration: float = 1.0) -> bool:
        """Test if microphone is working.

        Args:
            duration: Test duration in seconds

        Returns:
            True if microphone captured audio with sufficient level
        """
        try:
            audio = self.record_seconds(duration)
            if len(audio) == 0:
                return False

            # Check if there's any audio signal
            level = np.abs(audio).mean()
            logger.debug(f"Microphone test level: {level:.4f}")

            return level > 0.001  # Minimum threshold

        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return False

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_recording:
            self.stop_recording()
