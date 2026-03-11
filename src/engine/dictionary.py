"""Custom word/phrase replacement dictionary.

Supports simple word replacements and snippet triggers.
Stored as JSON file in the data/ directory.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List
from loguru import logger

from utils.constants import DATA_DIR


class Dictionary:
    """Custom word/phrase replacement dictionary.

    Supports:
    - Simple word replacements: "ШІ" -> "штучний інтелект"
    - Snippets with triggers: "/email" -> "user@example.com"
    - Case-insensitive matching for replacements

    Dictionary file format (JSON):
    {
        "replacements": {"trigger": "replacement", ...},
        "snippets": {"/trigger": "expanded text", ...}
    }
    """

    def __init__(self, dict_file: Optional[Path] = None):
        self._dict_file = dict_file or (DATA_DIR / "dictionary.json")
        self._replacements: Dict[str, str] = {}
        self._snippets: Dict[str, str] = {}
        self._load()

        logger.info(f"Dictionary initialized: {len(self._replacements)} replacements, {len(self._snippets)} snippets")

    def _load(self):
        """Load dictionary from JSON file."""
        if not self._dict_file.exists():
            # Create default dictionary file
            self._save_default()
            return

        try:
            with open(self._dict_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._replacements = data.get("replacements", {})
            self._snippets = data.get("snippets", {})

        except Exception as e:
            logger.error(f"Failed to load dictionary: {e}")
            self._replacements = {}
            self._snippets = {}

    def _save_default(self):
        """Create a default dictionary file with examples."""
        default_data = {
            "replacements": {},
            "snippets": {},
        }

        try:
            self._dict_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dict_file, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created default dictionary at {self._dict_file}")
        except Exception as e:
            logger.error(f"Failed to create default dictionary: {e}")

    def save(self):
        """Save current dictionary to file."""
        try:
            data = {
                "replacements": self._replacements,
                "snippets": self._snippets,
            }

            self._dict_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dict_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("Dictionary saved")
        except Exception as e:
            logger.error(f"Failed to save dictionary: {e}")

    def apply_replacements(self, text: str) -> str:
        """Apply all dictionary replacements and snippets to text.

        Snippets (starting with /) are applied first (exact match).
        Then word replacements are applied (case-insensitive).

        Args:
            text: Input text

        Returns:
            Text with replacements applied
        """
        if not text:
            return text

        # Apply snippets first (exact match, case-sensitive)
        for trigger, expansion in self._snippets.items():
            if trigger in text:
                text = text.replace(trigger, expansion)

        # Apply word replacements (case-insensitive, whole word)
        for trigger, replacement in self._replacements.items():
            pattern = re.compile(re.escape(trigger), re.IGNORECASE)
            text = pattern.sub(replacement, text)

        return text

    def add_replacement(self, trigger: str, replacement: str):
        """Add or update a word replacement.

        Args:
            trigger: Word/phrase to find
            replacement: Text to replace with
        """
        self._replacements[trigger] = replacement
        self.save()

    def remove_replacement(self, trigger: str):
        """Remove a word replacement."""
        if trigger in self._replacements:
            del self._replacements[trigger]
            self.save()

    def add_snippet(self, trigger: str, expansion: str):
        """Add or update a snippet.

        Args:
            trigger: Trigger string (e.g., "/email")
            expansion: Expanded text
        """
        self._snippets[trigger] = expansion
        self.save()

    def remove_snippet(self, trigger: str):
        """Remove a snippet."""
        if trigger in self._snippets:
            del self._snippets[trigger]
            self.save()

    def get_all_replacements(self) -> Dict[str, str]:
        return self._replacements.copy()

    def get_all_snippets(self) -> Dict[str, str]:
        return self._snippets.copy()

    def reload(self):
        """Reload dictionary from file."""
        self._load()
