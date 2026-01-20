"""Transcriber module - faster-whisper wrapper for speech-to-text."""

from typing import Optional, Tuple, List
import numpy as np
from loguru import logger

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Install with: pip install faster-whisper")


class Transcriber:
    """Speech-to-text transcriber using faster-whisper.

    Provides high-quality transcription with support for Ukrainian and English.

    Attributes:
        model_size: Whisper model size (tiny, base, small, medium, large-v3, large-v3-turbo)
        device: Device to use (cuda, cpu, auto)
        compute_type: Computation type (int8, float16, float32)
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
    ):
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size
            device: Device to run on (cuda/cpu/auto)
            compute_type: Precision (int8/float16/float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self._is_loaded = False

        logger.info(f"Transcriber initialized: model={model_size}, device={device}, compute={compute_type}")

    def load_model(self) -> bool:
        """Load the Whisper model into memory.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper not available")
            return False

        if self._is_loaded:
            logger.debug("Model already loaded")
            return True

        try:
            logger.info(f"Loading Whisper model '{self.model_size}'...")

            # Determine device
            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            logger.info(f"Using device: {device}")

            # Load model
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=self.compute_type,
            )

            self._is_loaded = True
            logger.info(f"Model '{self.model_size}' loaded successfully on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded")

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: str = "uk",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Transcribe audio data to text.

        Args:
            audio_data: Audio samples as numpy array (float32, mono, 16kHz)
            language: Language code ('uk', 'en', or None for auto-detect)
            beam_size: Beam size for decoding (higher = more accurate, slower)
            vad_filter: Enable Voice Activity Detection filter
            initial_prompt: Optional prompt to guide transcription

        Returns:
            Tuple of (transcribed_text, confidence)
        """
        if not self._is_loaded:
            logger.warning("Model not loaded, loading now...")
            if not self.load_model():
                return "", 0.0

        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data")
            return "", 0.0

        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val

            logger.debug(f"Transcribing {len(audio_data)} samples, language={language}")

            # Ukrainian initial prompt for better accuracy
            if language == "uk" and initial_prompt is None:
                initial_prompt = "Привіт. Це транскрипція українською мовою."

            # Transcribe
            segments, info = self.model.transcribe(
                audio_data,
                language=language if language != "auto" else None,
                beam_size=beam_size,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
            )

            # Collect segments
            text_parts = []
            total_prob = 0.0
            segment_count = 0

            for segment in segments:
                text_parts.append(segment.text.strip())
                total_prob += segment.avg_logprob
                segment_count += 1

            text = " ".join(text_parts).strip()

            # Calculate average confidence
            confidence = 0.0
            if segment_count > 0:
                avg_logprob = total_prob / segment_count
                # Convert log probability to confidence (0-1)
                confidence = min(1.0, max(0.0, 1.0 + avg_logprob / 5.0))

            logger.info(f"Transcription complete: '{text[:50]}...' (confidence: {confidence:.2f})")
            return text, confidence

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", 0.0

    def transcribe_file(
        self,
        file_path: str,
        language: str = "uk",
        **kwargs,
    ) -> Tuple[str, float]:
        """Transcribe audio from file.

        Args:
            file_path: Path to audio file
            language: Language code
            **kwargs: Additional arguments for transcribe()

        Returns:
            Tuple of (transcribed_text, confidence)
        """
        if not self._is_loaded:
            if not self.load_model():
                return "", 0.0

        try:
            segments, info = self.model.transcribe(
                file_path,
                language=language if language != "auto" else None,
                **kwargs,
            )

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            text = " ".join(text_parts).strip()
            return text, 0.8  # Default confidence for file transcription

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return "", 0.0

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def get_available_models(self) -> List[str]:
        """Get list of available Whisper models."""
        return [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "large-v3-turbo",
        ]

    def __del__(self):
        """Cleanup on deletion."""
        self.unload_model()
