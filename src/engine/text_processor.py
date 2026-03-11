"""Text post-processing pipeline for transcribed text.

Applies formatting rules to raw Whisper output:
1. Dictionary/snippet replacements
2. Capitalize first letter of sentences
3. Basic punctuation fixes
4. Language-specific rules
"""

import re
from typing import Optional
from loguru import logger


class TextProcessor:
    """Post-processing pipeline for transcribed text.

    Transforms raw ASR output into properly formatted text.
    All processing is local — no network calls.
    """

    def __init__(
        self,
        capitalize: bool = True,
        auto_punctuation: bool = True,
        dictionary=None,
    ):
        self._capitalize = capitalize
        self._auto_punctuation = auto_punctuation
        self._dictionary = dictionary

        logger.info("TextProcessor initialized")

    def process(self, text: str, language: str = "uk") -> str:
        """Apply all processing steps to transcribed text.

        Args:
            text: Raw transcribed text
            language: Language code ('uk', 'en')

        Returns:
            Processed text
        """
        if not text or not text.strip():
            return text

        # 1. Dictionary replacements
        if self._dictionary is not None:
            text = self._dictionary.apply_replacements(text)

        # 2. Fix punctuation
        if self._auto_punctuation:
            text = self._fix_punctuation(text)

        # 3. Capitalize sentences
        if self._capitalize:
            text = self._capitalize_sentences(text)

        # 4. Language-specific fixes
        text = self._language_specific(text, language)

        return text

    def process_streaming(self, text: str, language: str = "uk") -> str:
        """Lightweight processing for streaming mode (lower latency).

        Only applies capitalization and dictionary — skips heavier rules.

        Args:
            text: Text chunk
            language: Language code

        Returns:
            Lightly processed text
        """
        if not text or not text.strip():
            return text

        if self._dictionary is not None:
            text = self._dictionary.apply_replacements(text)

        if self._capitalize:
            text = self._capitalize_first(text)

        return text

    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter after sentence-ending punctuation."""
        # Capitalize after . ! ? and at start of text
        result = re.sub(
            r'(^|[.!?]\s+)([a-zа-яіїєґ])',
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        # Always capitalize first character
        if result:
            result = result[0].upper() + result[1:]
        return result

    def _capitalize_first(self, text: str) -> str:
        """Capitalize only the first character of text."""
        if text:
            return text[0].upper() + text[1:]
        return text

    def _fix_punctuation(self, text: str) -> str:
        """Fix common punctuation issues from ASR output."""
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove space before punctuation marks
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Ensure space after punctuation (if followed by letter)
        text = re.sub(r'([.,!?;:])([a-zA-Zа-яА-ЯіІїЇєЄґҐ])', r'\1 \2', text)

        # Remove leading/trailing spaces
        text = text.strip()

        return text

    def _language_specific(self, text: str, language: str) -> str:
        """Apply language-specific text corrections.

        Args:
            text: Text to process
            language: Language code

        Returns:
            Corrected text
        """
        if language == "uk":
            # Normalize Ukrainian apostrophe: replace ASCII ' with proper ʼ
            # Common in ASR output: м'який -> м'який (should be м'який or мʼякий)
            # Keep ASCII apostrophe as it's more universally supported
            pass

        return text
