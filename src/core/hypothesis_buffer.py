"""HypothesisBuffer - LocalAgreement-2 algorithm for streaming transcription.

Based on UFAL whisper_streaming implementation.
https://github.com/ufal/whisper_streaming

The LocalAgreement-2 policy:
- If 2 consecutive transcriptions agree on a prefix, it is confirmed
- Only confirmed text is output to prevent duplication
"""

from typing import List, Tuple, Optional
from loguru import logger


class HypothesisBuffer:
    """Buffer implementing LocalAgreement-2 for streaming transcription.

    Maintains three states:
    - commited_in_buffer: Confirmed words that won't change
    - buffer: Previous transcription hypothesis
    - new: Current transcription hypothesis

    Words are confirmed only when consecutive transcriptions agree.
    """

    def __init__(self):
        """Initialize empty hypothesis buffer."""
        # Confirmed words: [(start_time, end_time, word), ...]
        self.commited_in_buffer: List[Tuple[float, float, str]] = []

        # Previous hypothesis (from last transcription)
        self.buffer: List[Tuple[float, float, str]] = []

        # Current hypothesis (from new transcription)
        self.new: List[Tuple[float, float, str]] = []

        # Tracking for timestamp-based filtering
        self.last_commited_time: float = 0
        self.last_commited_word: str = ""

    def insert(self, new_words: List[Tuple[float, float, str]], offset: float = 0):
        """Insert new transcription with optional timestamp offset.

        Args:
            new_words: List of (start_time, end_time, word) tuples
            offset: Timestamp offset to add (for cumulative audio)
        """
        # Apply offset to timestamps
        new = [(a + offset, b + offset, t) for a, b, t in new_words]

        # Filter: only keep words starting after last confirmed time (with 0.1s margin)
        # This prevents re-processing already confirmed words
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        logger.debug(f"HypothesisBuffer: inserted {len(self.new)} words (offset={offset:.2f}s)")

    def flush(self) -> List[Tuple[float, float, str]]:
        """Confirm words where buffer and new agree.

        Returns:
            List of newly confirmed (start_time, end_time, word) tuples
        """
        commit = []

        # Compare buffer (previous) with new (current)
        while self.new and self.buffer:
            na, nb, nt = self.new[0]
            ba, bb, bt = self.buffer[0]

            # Clean words for comparison (lowercase, strip)
            nt_clean = nt.strip().lower()
            bt_clean = bt.strip().lower()

            # Confirm only if words MATCH
            if nt_clean == bt_clean:
                # Use the new word's text (preserves original case)
                commit.append((na, nb, nt.strip()))
                self.last_commited_word = nt.strip()
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                # Words differ - stop comparing
                break

        # Store new hypothesis as buffer for next iteration
        self.buffer = self.new
        self.new = []

        # Add confirmed words to committed list
        self.commited_in_buffer.extend(commit)

        if commit:
            confirmed_text = " ".join([w[2] for w in commit])
            logger.debug(f"HypothesisBuffer: confirmed {len(commit)} words: '{confirmed_text}'")

        return commit

    def get_confirmed_text(self) -> str:
        """Get all confirmed text so far.

        Returns:
            Concatenated confirmed words
        """
        return " ".join([w[2] for w in self.commited_in_buffer])

    def get_unconfirmed_text(self) -> str:
        """Get current unconfirmed text (buffer).

        Returns:
            Concatenated unconfirmed words
        """
        return " ".join([w[2] for w in self.buffer])

    def finalize(self) -> List[Tuple[float, float, str]]:
        """Finalize remaining buffer as confirmed (call at end of recording).

        When PTT is released, any remaining words in buffer should be confirmed.

        Returns:
            List of finalized (start_time, end_time, word) tuples
        """
        finalized = self.buffer.copy()
        self.commited_in_buffer.extend(finalized)
        self.buffer = []

        if finalized:
            final_text = " ".join([w[2] for w in finalized])
            logger.debug(f"HypothesisBuffer: finalized {len(finalized)} words: '{final_text}'")

        return finalized

    def reset(self):
        """Reset buffer for new recording session."""
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = ""
        logger.debug("HypothesisBuffer: reset")

    def __len__(self) -> int:
        """Return total number of confirmed words."""
        return len(self.commited_in_buffer)

    def __repr__(self) -> str:
        return (
            f"HypothesisBuffer(confirmed={len(self.commited_in_buffer)}, "
            f"buffer={len(self.buffer)}, new={len(self.new)})"
        )
