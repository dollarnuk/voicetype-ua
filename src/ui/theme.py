"""Theme manager - dark/light theme support with QSS styling."""

from typing import Dict, Optional
from loguru import logger

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QFont
    from PyQt6.QtCore import Qt, QSize
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False


class ThemeManager:
    """Manages application visual theme (dark/light).

    Generates QSS stylesheets and tray icons programmatically
    using QPainter (no external icon files required).
    """

    DARK = {
        "bg_primary": "#1a1a2e",
        "bg_secondary": "#16213e",
        "bg_tertiary": "#0f3460",
        "text_primary": "#e6e6e6",
        "text_secondary": "#a0a0a0",
        "text_disabled": "#606060",
        "accent": "#e94560",
        "accent_green": "#4ecca3",
        "accent_yellow": "#f0c040",
        "border": "#2a2a3e",
        "input_bg": "#1e1e32",
        "hover": "#253050",
        "selected": "#0f3460",
        "scrollbar": "#2a2a3e",
        "scrollbar_handle": "#3a3a5e",
    }

    LIGHT = {
        "bg_primary": "#f5f5f7",
        "bg_secondary": "#ffffff",
        "bg_tertiary": "#e8e8ec",
        "text_primary": "#1a1a2e",
        "text_secondary": "#666680",
        "text_disabled": "#a0a0b0",
        "accent": "#e94560",
        "accent_green": "#2ea87a",
        "accent_yellow": "#d4a520",
        "border": "#d0d0d8",
        "input_bg": "#ffffff",
        "hover": "#e0e0e8",
        "selected": "#d0d8f0",
        "scrollbar": "#e0e0e8",
        "scrollbar_handle": "#c0c0c8",
    }

    def __init__(self, theme_name: str = "dark"):
        self._theme_name = theme_name
        self._colors = self.DARK if theme_name == "dark" else self.LIGHT
        self._icon_cache: Dict[str, QIcon] = {}

    @property
    def colors(self) -> Dict[str, str]:
        return self._colors

    @property
    def theme_name(self) -> str:
        return self._theme_name

    def set_theme(self, theme_name: str):
        self._theme_name = theme_name
        self._colors = self.DARK if theme_name == "dark" else self.LIGHT
        self._icon_cache.clear()

    def apply(self, app: QApplication):
        """Apply QSS stylesheet to the entire application."""
        if not PYQT6_AVAILABLE:
            return

        qss = self._generate_qss()
        app.setStyleSheet(qss)
        logger.info(f"Applied {self._theme_name} theme")

    def _generate_qss(self) -> str:
        c = self._colors
        return f"""
/* === Global === */
QWidget {{
    background-color: {c['bg_primary']};
    color: {c['text_primary']};
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}}

/* === QMainWindow / QDialog === */
QMainWindow, QDialog {{
    background-color: {c['bg_primary']};
}}

/* === Labels === */
QLabel {{
    background-color: transparent;
    color: {c['text_primary']};
    padding: 2px;
}}

QLabel[class="secondary"] {{
    color: {c['text_secondary']};
}}

QLabel[class="heading"] {{
    font-size: 16px;
    font-weight: bold;
}}

/* === Buttons === */
QPushButton {{
    background-color: {c['bg_tertiary']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 6px 16px;
    min-height: 28px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {c['hover']};
    border-color: {c['accent']};
}}

QPushButton:pressed {{
    background-color: {c['accent']};
    color: white;
}}

QPushButton:disabled {{
    background-color: {c['bg_secondary']};
    color: {c['text_disabled']};
    border-color: {c['border']};
}}

QPushButton[class="primary"] {{
    background-color: {c['accent']};
    color: white;
    border: none;
}}

QPushButton[class="primary"]:hover {{
    background-color: #d03050;
}}

/* === Input Fields === */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {c['input_bg']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 28px;
    selection-background-color: {c['accent']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {c['accent']};
}}

/* === ComboBox === */
QComboBox {{
    background-color: {c['input_bg']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 28px;
}}

QComboBox:hover {{
    border-color: {c['accent']};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {c['bg_secondary']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    selection-background-color: {c['selected']};
    outline: none;
}}

/* === CheckBox === */
QCheckBox {{
    background-color: transparent;
    spacing: 8px;
    min-height: 24px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {c['border']};
    border-radius: 4px;
    background-color: {c['input_bg']};
}}

QCheckBox::indicator:checked {{
    background-color: {c['accent']};
    border-color: {c['accent']};
}}

QCheckBox::indicator:hover {{
    border-color: {c['accent']};
}}

/* === Slider === */
QSlider::groove:horizontal {{
    height: 6px;
    background-color: {c['border']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background-color: {c['accent']};
    border-radius: 8px;
}}

QSlider::sub-page:horizontal {{
    background-color: {c['accent']};
    border-radius: 3px;
}}

/* === Tab Widget === */
QTabWidget::pane {{
    background-color: {c['bg_secondary']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: {c['bg_primary']};
    color: {c['text_secondary']};
    border: 1px solid {c['border']};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 2px;
    min-width: 80px;
}}

QTabBar::tab:selected {{
    background-color: {c['bg_secondary']};
    color: {c['text_primary']};
    font-weight: bold;
}}

QTabBar::tab:hover:!selected {{
    background-color: {c['hover']};
    color: {c['text_primary']};
}}

/* === Table === */
QTableView, QTableWidget {{
    background-color: {c['bg_secondary']};
    alternate-background-color: {c['bg_primary']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    gridline-color: {c['border']};
    selection-background-color: {c['selected']};
    outline: none;
}}

QHeaderView::section {{
    background-color: {c['bg_primary']};
    color: {c['text_secondary']};
    border: none;
    border-bottom: 2px solid {c['border']};
    padding: 8px;
    font-weight: bold;
    font-size: 12px;
}}

/* === ScrollBar === */
QScrollBar:vertical {{
    background-color: {c['scrollbar']};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {c['scrollbar_handle']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {c['scrollbar']};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {c['scrollbar_handle']};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* === GroupBox === */
QGroupBox {{
    background-color: {c['bg_secondary']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 8px;
    color: {c['text_secondary']};
}}

/* === Menu (tray context menu) === */
QMenu {{
    background-color: {c['bg_secondary']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {c['hover']};
}}

QMenu::item:disabled {{
    color: {c['text_disabled']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {c['border']};
    margin: 4px 8px;
}}

/* === ToolTip === */
QToolTip {{
    background-color: {c['bg_secondary']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: 4px;
    padding: 4px 8px;
}}

/* === StatusBar === */
QStatusBar {{
    background-color: {c['bg_primary']};
    color: {c['text_secondary']};
    border-top: 1px solid {c['border']};
}}

/* === Frame separator === */
QFrame[class="separator"] {{
    background-color: {c['border']};
    max-height: 1px;
}}

/* === MessageBox === */
QMessageBox {{
    background-color: {c['bg_primary']};
}}

QMessageBox QLabel {{
    color: {c['text_primary']};
}}
"""

    def get_tray_icon(self, status: str) -> QIcon:
        """Generate tray icon for given status using QPainter.

        Args:
            status: One of 'idle', 'recording', 'processing', 'error'

        Returns:
            QIcon with colored circle indicator
        """
        if not PYQT6_AVAILABLE:
            return QIcon()

        if status in self._icon_cache:
            return self._icon_cache[status]

        size = 32
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colors per status
        colors = {
            "idle": QColor(self._colors["accent_green"]),
            "recording": QColor(self._colors["accent"]),
            "processing": QColor(self._colors["accent_yellow"]),
            "error": QColor("#808080"),
        }

        color = colors.get(status, QColor("#808080"))

        # Draw outer circle (subtle border)
        margin = 2
        painter.setPen(QPen(color.darker(130), 1.5))
        painter.setBrush(color)
        painter.drawEllipse(margin, margin, size - margin * 2, size - margin * 2)

        # Draw inner highlight (glossy effect)
        highlight = QColor(255, 255, 255, 60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(highlight)
        painter.drawEllipse(
            margin + 4, margin + 3,
            size - margin * 2 - 8, (size - margin * 2) // 2 - 2,
        )

        # Error: draw X mark
        if status == "error":
            pen = QPen(QColor(255, 255, 255, 200), 2.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            cx, cy = size // 2, size // 2
            d = 5
            painter.drawLine(cx - d, cy - d, cx + d, cy + d)
            painter.drawLine(cx + d, cy - d, cx - d, cy + d)

        painter.end()

        icon = QIcon(pixmap)
        self._icon_cache[status] = icon
        return icon

    def get_recording_icon_alt(self) -> QIcon:
        """Generate alternate recording icon for pulsing effect."""
        if not PYQT6_AVAILABLE:
            return QIcon()

        cache_key = "recording_alt"
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]

        size = 32
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(QColor(0, 0, 0, 0))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = QColor(self._colors["accent"])

        # Slightly smaller circle for pulse effect
        margin = 5
        painter.setPen(QPen(color.darker(130), 1.5))
        painter.setBrush(color.lighter(120))
        painter.drawEllipse(margin, margin, size - margin * 2, size - margin * 2)

        painter.end()

        icon = QIcon(pixmap)
        self._icon_cache[cache_key] = icon
        return icon
