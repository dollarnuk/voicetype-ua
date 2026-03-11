"""History window — QMainWindow for viewing and managing transcription history."""

from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QComboBox,
    QPushButton,
    QLabel,
    QMenu,
    QMessageBox,
    QAbstractItemView,
    QStatusBar,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QClipboard, QGuiApplication
from loguru import logger

from data.history_storage import HistoryStorage


# Ukrainian month names (genitive case, abbreviated) for date formatting
_UK_MONTHS = {
    1: "Січ",
    2: "Лют",
    3: "Бер",
    4: "Кві",
    5: "Тра",
    6: "Чер",
    7: "Лип",
    8: "Сер",
    9: "Вер",
    10: "Жов",
    11: "Лис",
    12: "Гру",
}

_MAX_TEXT_LENGTH = 80
_LOAD_LIMIT = 200
_SEARCH_DEBOUNCE_MS = 300

_LANGUAGE_FILTERS = ["Всі", "UK", "EN"]


def _format_date_uk(iso_string: str) -> str:
    """Format ISO datetime string as 'DD Mon HH:MM' with Ukrainian month names.

    Args:
        iso_string: Date string from SQLite (e.g. '2026-01-30 14:02:15')

    Returns:
        Formatted string like '30 Січ 14:02'
    """
    try:
        dt = datetime.fromisoformat(iso_string)
        month_name = _UK_MONTHS.get(dt.month, str(dt.month))
        return f"{dt.day:02d} {month_name} {dt.hour:02d}:{dt.minute:02d}"
    except (ValueError, TypeError):
        return str(iso_string)


def _format_duration(duration_ms: Optional[int]) -> str:
    """Format duration in milliseconds to a human-readable string.

    Args:
        duration_ms: Duration in milliseconds, or None.

    Returns:
        Formatted string like '2.1с' or '—' if unavailable.
    """
    if duration_ms is None:
        return "—"
    seconds = duration_ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}с"
    minutes = seconds / 60.0
    return f"{minutes:.1f}хв"


def _truncate_text(text: str, max_length: int = _MAX_TEXT_LENGTH) -> str:
    """Truncate text to max_length characters, appending '...' if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


class HistoryWindow(QMainWindow):
    """Main window for viewing transcription history.

    Displays a searchable, filterable table of past transcriptions with
    statistics in the status bar. Supports copy, delete, and clear operations.

    Args:
        history_storage: HistoryStorage instance for data access.
        parent: Optional parent widget.
    """

    # Column indices
    COL_TEXT = 0
    COL_LANGUAGE = 1
    COL_DURATION = 2
    COL_DATE = 3

    def __init__(
        self,
        history_storage: HistoryStorage,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._storage = history_storage
        self._entries: list[dict] = []
        self._search_timer: Optional[int] = None

        self._init_ui()
        self._load_data()

    # ------------------------------------------------------------------ #
    #  UI setup
    # ------------------------------------------------------------------ #

    def _init_ui(self):
        """Build all UI elements and layout."""
        self.setWindowTitle("WispanTalk - Історія")
        self.resize(700, 500)
        self.setMinimumSize(550, 400)
        self.setStyleSheet(MAIN_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(6)

        # --- Toolbar row ---
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(8)

        search_label = QLabel("Пошук:")
        toolbar_layout.addWidget(search_label)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Введіть текст для пошуку...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(self._search_input, stretch=1)

        lang_label = QLabel("Мова:")
        toolbar_layout.addWidget(lang_label)

        self._lang_combo = QComboBox()
        self._lang_combo.addItems(_LANGUAGE_FILTERS)
        self._lang_combo.setFixedWidth(80)
        self._lang_combo.currentIndexChanged.connect(self._on_filter_changed)
        toolbar_layout.addWidget(self._lang_combo)

        self._refresh_btn = QPushButton("Оновити")
        self._refresh_btn.setFixedWidth(90)
        self._refresh_btn.clicked.connect(self._load_data)
        toolbar_layout.addWidget(self._refresh_btn)

        layout.addLayout(toolbar_layout)

        # --- Table ---
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Текст", "Мова", "Тривалість", "Дата"])
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        self._table.doubleClicked.connect(self._on_double_click)
        self._table.setSortingEnabled(False)
        
        # Modern Table Styling
        self._table.setShowGrid(False)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: #121212;
                border: none;
                gridline-color: transparent;
                alternate-background-color: #1A1A1A;
            }}
            QHeaderView::section {{
                background-color: #1E1E1E;
                color: #94A3B8;
                border: none;
                border-bottom: 2px solid #334155;
                padding: 10px;
                font-weight: bold;
            }}
            QTableWidget::item {{
                padding: 10px;
                border-bottom: 1px solid #252525;
            }}
            QTableWidget::item:selected {{
                background-color: #6366F1;
                color: white;
            }}
        """)

        # Column sizing
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self.COL_TEXT, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_LANGUAGE, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_DURATION, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_DATE, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self.COL_LANGUAGE, 60)
        self._table.setColumnWidth(self.COL_DURATION, 80)
        self._table.setColumnWidth(self.COL_DATE, 120)

        layout.addWidget(self._table)

        # --- Status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._stats_label = QLabel()
        self._status_bar.addPermanentWidget(self._stats_label, stretch=1)

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _get_current_filters(self) -> dict:
        """Return current search/language filter parameters."""
        search_text = self._search_input.text().strip() or None

        lang_index = self._lang_combo.currentIndex()
        language = None
        if lang_index > 0:
            lang_value = _LANGUAGE_FILTERS[lang_index]
            language = lang_value.lower()

        return {
            "search_text": search_text,
            "language": language,
        }

    def _load_data(self):
        """Fetch entries from storage and populate the table."""
        try:
            filters = self._get_current_filters()
            self._entries = self._storage.get_entries(
                limit=_LOAD_LIMIT,
                offset=0,
                search_text=filters["search_text"],
                language=filters["language"],
            )
            self._populate_table()
            self._update_statistics()
        except Exception as e:
            logger.error(f"Не вдалося завантажити історію: {e}")
            self._status_bar.showMessage(f"Помилка завантаження: {e}", 5000)

    def _populate_table(self):
        """Fill the table widget with current entries."""
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._entries))

        for row, entry in enumerate(self._entries):
            # Text (truncated)
            full_text = entry.get("text", "")
            text_item = QTableWidgetItem(_truncate_text(full_text))
            text_item.setToolTip(full_text)
            text_item.setData(Qt.ItemDataRole.UserRole, entry.get("id"))
            self._table.setItem(row, self.COL_TEXT, text_item)

            # Language
            lang = (entry.get("language") or "").upper()
            lang_item = QTableWidgetItem(lang)
            lang_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, self.COL_LANGUAGE, lang_item)

            # Duration
            duration_str = _format_duration(entry.get("duration_ms"))
            dur_item = QTableWidgetItem(duration_str)
            dur_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, self.COL_DURATION, dur_item)

            # Date
            date_str = _format_date_uk(entry.get("created_at", ""))
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, self.COL_DATE, date_item)

    def _update_statistics(self):
        """Fetch and display statistics in the status bar."""
        try:
            stats = self._storage.get_statistics()
            total = stats.get("total_entries", 0)
            duration_sec = stats.get("total_duration_sec", 0)
            avg_conf = stats.get("avg_confidence")

            duration_min = duration_sec / 60.0
            conf_str = f"{avg_conf * 100:.0f}%" if avg_conf is not None else "—"

            self._stats_label.setText(
                f"Всього: {total} записів  |  "
                f"Тривалість: {duration_min:.1f}хв  |  "
                f"Точність: {conf_str}"
            )
        except Exception as e:
            logger.error(f"Не вдалося отримати статистику: {e}")
            self._stats_label.setText("Статистика недоступна")

    # ------------------------------------------------------------------ #
    #  Search & filter
    # ------------------------------------------------------------------ #

    def _on_search_changed(self, _text: str):
        """Handle search input change with debounce."""
        QTimer.singleShot(_SEARCH_DEBOUNCE_MS, self._load_data)

    def _on_filter_changed(self, _index: int):
        """Handle language filter change — reload immediately."""
        self._load_data()

    # ------------------------------------------------------------------ #
    #  Context menu & actions
    # ------------------------------------------------------------------ #

    def _get_selected_row(self) -> Optional[int]:
        """Return the currently selected row index, or None."""
        selection = self._table.selectionModel().selectedRows()
        if selection:
            return selection[0].row()
        return None

    def _get_entry_id_at_row(self, row: int) -> Optional[int]:
        """Return the entry ID stored in the text column for a given row."""
        item = self._table.item(row, self.COL_TEXT)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _get_full_text_at_row(self, row: int) -> Optional[str]:
        """Return the full (non-truncated) text for a given row."""
        if 0 <= row < len(self._entries):
            return self._entries[row].get("text")
        return None

    def _show_context_menu(self, position):
        """Display right-click context menu at the given position."""
        row = self._table.rowAt(position.y())

        menu = QMenu(self)

        copy_action = QAction("Копіювати текст", self)
        copy_action.triggered.connect(lambda: self._copy_text(row))
        copy_action.setEnabled(row >= 0)
        menu.addAction(copy_action)

        delete_action = QAction("Видалити", self)
        delete_action.triggered.connect(lambda: self._delete_entry(row))
        delete_action.setEnabled(row >= 0)
        menu.addAction(delete_action)

        menu.addSeparator()

        clear_action = QAction("Очистити все", self)
        clear_action.triggered.connect(self._clear_all)
        menu.addAction(clear_action)

        menu.exec(self._table.viewport().mapToGlobal(position))

    def _on_double_click(self, index):
        """Handle double-click on a row — copy full text to clipboard."""
        row = index.row()
        self._copy_text(row)

    def _copy_text(self, row: int):
        """Copy the full transcription text of the given row to clipboard."""
        text = self._get_full_text_at_row(row)
        if text:
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
                self._status_bar.showMessage("Текст скопійовано в буфер обміну", 3000)
                logger.debug(f"Скопійовано текст запису (рядок {row})")

    def _delete_entry(self, row: int):
        """Delete a single history entry after confirmation."""
        entry_id = self._get_entry_id_at_row(row)
        if entry_id is None:
            return

        reply = QMessageBox.question(
            self,
            "Підтвердження",
            "Видалити цей запис з історії?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self._storage.delete_entry(entry_id)
            if success:
                self._load_data()
                self._status_bar.showMessage("Запис видалено", 3000)
                logger.info(f"Видалено запис id={entry_id}")
            else:
                self._status_bar.showMessage("Не вдалося видалити запис", 3000)

    def _clear_all(self):
        """Clear all history entries after confirmation."""
        if not self._entries:
            self._status_bar.showMessage("Історія вже порожня", 3000)
            return

        reply = QMessageBox.warning(
            self,
            "Очистити історію",
            "Ви впевнені, що хочете видалити ВСЮ історію транскрипцій?\n"
            "Цю дію неможливо скасувати.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self._storage.clear_history()
            if success:
                self._load_data()
                self._status_bar.showMessage("Історію очищено", 3000)
                logger.info("Історію транскрипцій повністю очищено")
            else:
                self._status_bar.showMessage("Не вдалося очистити історію", 3000)

    # ------------------------------------------------------------------ #
    #  Overrides
    # ------------------------------------------------------------------ #

    def showEvent(self, event):
        """Reload data every time the window is shown."""
        super().showEvent(event)
        self._load_data()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_F5:
            self._load_data()
        elif (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_F
        ):
            self._search_input.setFocus()
            self._search_input.selectAll()
        else:
            super().keyPressEvent(event)
