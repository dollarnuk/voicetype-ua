"""History storage module - SQLite database for transcription history."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from utils.constants import DATA_DIR


class HistoryStorage:
    """SQLite storage for transcription history.

    Stores transcribed text with metadata for later reference.

    Attributes:
        db_path: Path to SQLite database file
        max_entries: Maximum number of entries to keep
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_entries: int = 1000,
    ):
        """Initialize history storage.

        Args:
            db_path: Path to database file (uses default if None)
            max_entries: Maximum entries to retain
        """
        self.db_path = db_path or (DATA_DIR / "history.db")
        self.max_entries = max_entries
        self._conn: Optional[sqlite3.Connection] = None

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"HistoryStorage initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        try:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            cursor = self._conn.cursor()

            # Create transcriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    language VARCHAR(5) NOT NULL DEFAULT 'uk',
                    duration_ms INTEGER,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    audio_path TEXT
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON transcriptions(created_at)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_language
                ON transcriptions(language)
            """)

            self._conn.commit()
            logger.debug("Database schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def add_entry(
        self,
        text: str,
        language: str = "uk",
        duration_ms: Optional[int] = None,
        confidence: Optional[float] = None,
        audio_path: Optional[str] = None,
    ) -> Optional[int]:
        """Add new transcription to history.

        Args:
            text: Transcribed text
            language: Language code
            duration_ms: Audio duration in milliseconds
            confidence: Transcription confidence (0-1)
            audio_path: Path to saved audio file (optional)

        Returns:
            Entry ID or None if failed
        """
        if not text:
            return None

        try:
            cursor = self._conn.cursor()

            cursor.execute(
                """
                INSERT INTO transcriptions
                (text, language, duration_ms, confidence, audio_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (text, language, duration_ms, confidence, audio_path),
            )

            self._conn.commit()
            entry_id = cursor.lastrowid

            # Cleanup old entries
            self._cleanup_old_entries()

            logger.debug(f"Added history entry: id={entry_id}")
            return entry_id

        except Exception as e:
            logger.error(f"Failed to add entry: {e}")
            return None

    def _cleanup_old_entries(self):
        """Remove oldest entries if exceeding max_entries."""
        try:
            cursor = self._conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM transcriptions")
            count = cursor.fetchone()[0]

            if count > self.max_entries:
                # Delete oldest entries
                delete_count = count - self.max_entries
                cursor.execute(
                    """
                    DELETE FROM transcriptions
                    WHERE id IN (
                        SELECT id FROM transcriptions
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                    """,
                    (delete_count,),
                )
                self._conn.commit()
                logger.debug(f"Cleaned up {delete_count} old entries")

        except Exception as e:
            logger.error(f"Failed to cleanup entries: {e}")

    def get_entries(
        self,
        limit: int = 50,
        offset: int = 0,
        language: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get history entries with optional filtering.

        Args:
            limit: Maximum entries to return
            offset: Number of entries to skip
            language: Filter by language code
            search_text: Filter by text content (case-insensitive)

        Returns:
            List of entry dictionaries
        """
        try:
            cursor = self._conn.cursor()

            query = "SELECT * FROM transcriptions WHERE 1=1"
            params = []

            if language:
                query += " AND language = ?"
                params.append(language)

            if search_text:
                query += " AND text LIKE ?"
                params.append(f"%{search_text}%")

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get entries: {e}")
            return []

    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """Get single entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Entry dictionary or None
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM transcriptions WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

            return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get entry: {e}")
            return None

    def delete_entry(self, entry_id: int) -> bool:
        """Delete entry by ID.

        Args:
            entry_id: Entry ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM transcriptions WHERE id = ?", (entry_id,))
            self._conn.commit()

            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete entry: {e}")
            return False

    def clear_history(self) -> bool:
        """Delete all history entries.

        Returns:
            True if cleared successfully
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM transcriptions")
            self._conn.commit()

            logger.info("History cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics.

        Returns:
            Dictionary with stats (total_entries, total_duration, etc.)
        """
        try:
            cursor = self._conn.cursor()

            stats = {}

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM transcriptions")
            stats["total_entries"] = cursor.fetchone()[0]

            # Total duration
            cursor.execute("SELECT SUM(duration_ms) FROM transcriptions")
            total_ms = cursor.fetchone()[0] or 0
            stats["total_duration_sec"] = total_ms / 1000

            # Entries by language
            cursor.execute(
                """
                SELECT language, COUNT(*) as count
                FROM transcriptions
                GROUP BY language
                """
            )
            stats["by_language"] = {row["language"]: row["count"] for row in cursor.fetchall()}

            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM transcriptions WHERE confidence IS NOT NULL")
            stats["avg_confidence"] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
