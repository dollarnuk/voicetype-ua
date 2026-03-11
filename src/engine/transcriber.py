"""Speech-to-text transcriber using Deepgram API (SDK v5+)."""

import io
import threading
from typing import Optional, Tuple, Callable
import numpy as np
from loguru import logger
import scipy.io.wavfile as wavfile

try:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
    from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logger.warning("deepgram-sdk not installed. Install with: pip install deepgram-sdk")


class Transcriber:
    """Speech-to-text transcriber using Deepgram API (v5+)."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "nova-2-general",
        language: str = "uk",
    ):
        """Initialize the transcriber.

        Args:
            api_key: API key for Deepgram
            model: Model name for Deepgram
            language: Default language
        """
        self.api_key = api_key
        self.model = model
        self.language = language

        self.dg_client: Optional[DeepgramClient] = None
        self._is_loaded = False

        # Streaming session
        self._dg_connection = None
        self._streaming_callback: Optional[Callable[[str, bool], None]] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_ctx = None  # context manager reference

        logger.info(f"Transcriber initialized with Deepgram (model={model}, lang={language})")

    def load_model(self) -> bool:
        """Initialize the Deepgram client.

        Returns:
            True if initialized successfully
        """
        if self._is_loaded:
            return True

        if not DEEPGRAM_AVAILABLE:
            logger.error("deepgram-sdk not available")
            return False

        if not self.api_key:
            logger.error("Deepgram API key is missing")
            return False

        try:
            self.dg_client = DeepgramClient(api_key=self.api_key)
            self._is_loaded = True
            logger.info("Deepgram client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to init Deepgram: {e}")
            return False

    def unload_model(self):
        """Clean up Deepgram client."""
        self.stop_streaming()
        self.dg_client = None
        self._is_loaded = False
        logger.info("Deepgram client unloaded")

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Transcribe full audio data using Prerecorded API (REST).

        Args:
            audio_data: Audio samples (float32, 16kHz)
            language: Optional language override

        Returns:
            Tuple of (text, confidence)
        """
        if not self.load_model():
            return "", 0.0

        try:
            # Normalize if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            lang = language or self.language
            if lang == "auto":
                lang = "uk"

            # Конвертація в WAV в пам'яті
            buffer = io.BytesIO()
            wavfile.write(buffer, 16000, audio_data)
            payload = buffer.getvalue()

            # SDK v5: client.listen.v1.media.transcribe_file(...)
            response = self.dg_client.listen.v1.media.transcribe_file(
                request=payload,
                model=self.model,
                smart_format=True,
                language=lang,
                punctuate=True,
            )

            if response and response.results and response.results.channels:
                alt = response.results.channels[0].alternatives[0]
                transcript = alt.transcript
                confidence = alt.confidence
                logger.debug(f"Deepgram transcript: '{transcript}' (conf: {confidence:.2f})")
                return transcript, confidence

            return "", 0.0

        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            return "", 0.0

    # --- Streaming Methods ---

    def start_streaming(self, callback: Callable[[str, bool], None], language: Optional[str] = None):
        """Start a WebSocket streaming session.

        SDK v5 uses a context manager for WebSocket connections.
        We run the connection and listener in a background thread.
        """
        # Session guard: prevent starting multiple sessions in parallel
        if self._dg_connection is not None:
            logger.warning("Deepgram streaming session already active, skipping start request")
            return

        if self._stream_thread and self._stream_thread.is_alive():
            logger.warning("Deepgram streaming thread still alive, skipping start request")
            return

        if not self.load_model():
            return

        self._streaming_callback = callback
        lang = language or self.language
        if lang == "auto":
            lang = "uk"

        def _run_stream():
            try:
                # SDK v5: context manager — всі параметри як рядки
                # endpointing="1000" (1s) дозволяє робити природні паузи
                # utterance_end_ms="1000" (1s) допомагає при довгій диктовці
                with self.dg_client.listen.v1.connect(
                    model=self.model,
                    language=lang,
                    smart_format="true",
                    interim_results="true",
                    encoding="linear16",
                    sample_rate="16000",
                    endpointing="1000",
                ) as ws:
                    self._dg_connection = ws

                    # Реєструємо обробник подій
                    def on_message(data):
                        if isinstance(data, ListenV1ResultsEvent):
                            alt = data.channel.alternatives[0]
                            sentence = alt.transcript
                            if not sentence:
                                return

                            is_final = bool(data.is_final)
                            logger.info(f"Received from Deepgram: '{sentence}' (final={is_final})")
                            if self._streaming_callback:
                                self._streaming_callback(sentence, is_final)

                    def on_error(error):
                        logger.error(f"Deepgram streaming error: {error}")

                    ws.on(EventType.MESSAGE, on_message)
                    ws.on(EventType.ERROR, on_error)

                    logger.info("Deepgram streaming session started")

                    # Блокуючий цикл слухання — тримає WebSocket відкритим
                    ws.start_listening()

            except Exception as e:
                logger.error(f"Deepgram streaming thread error: {e}")
            finally:
                self._dg_connection = None
                logger.info("Deepgram streaming session ended")

        self._stream_thread = threading.Thread(target=_run_stream, daemon=True)
        self._stream_thread.start()

    def send_audio_chunk(self, chunk: np.ndarray):
        """Send audio chunk to the active streaming session."""
        if self._dg_connection is None:
            return

        try:
            # Convert float32 to int16 for linear16 encoding
            if chunk.dtype == np.float32:
                chunk = (chunk * 32767).astype(np.int16)

            # logger.debug(f"Sending audio chunk ({len(chunk)} bytes)")
            self._dg_connection.send_media(chunk.tobytes())
        except Exception as e:
            logger.error(f"Error sending audio chunk to Deepgram: {e}")

    def stop_streaming(self):
        """Close the active streaming session."""
        ws = self._dg_connection
        if ws is not None:
            try:
                # SDK v5: use _send for CloseStream if finish() is missing
                # V1SocketClient inspection showed _send exists.
                import json
                ws._send(json.dumps({"type": "CloseStream"}))
                logger.info("Deepgram streaming session close requested")
            except Exception as e:
                logger.warning(f"Deepgram session close error: {e}")
            finally:
                self._dg_connection = None

        # Очікуємо завершення потоку
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        self._stream_thread = None

    def transcribe_chunk_with_timestamps(self, audio_data, language="uk"):
        """Deprecated for Deepgram-only mode. Use start_streaming instead."""
        logger.warning("transcribe_chunk_with_timestamps is deprecated. Use WebSocket streaming.")
        return []
