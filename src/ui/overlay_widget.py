"""Floating recording overlay widget - shows recording status, audio level, language."""

import time
from loguru import logger

try:
    from PyQt6.QtWidgets import QWidget, QApplication
    from PyQt6.QtGui import (
        QPainter, QPainterPath, QColor, QLinearGradient,
        QFont, QPen, QBrush,
    )
    from PyQt6.QtCore import (
        Qt, QTimer, QPropertyAnimation, QEasingCurve,
        QPoint, QRectF, pyqtProperty, QSize,
    )
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

from .styles import BG_SIDEBAR, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, ERROR, SUCCESS


class RecordingOverlay(QWidget):
    """Floating semi-transparent recording indicator.

    Displays a sleek capsule-shaped overlay with:
    - Pulse dot (red when recording)
    - Status text
    - Minimalist audio level meter
    - Language badge
    - Elapsed time
    """

    PILL_WIDTH = 320
    PILL_HEIGHT = 44
    CORNER_RADIUS = 22

    def __init__(self, theme_colors: dict, parent=None):
        if not PYQT6_AVAILABLE:
            return

        super().__init__(parent)

        self._audio_level: float = 0.0
        self._smoothed_level: float = 0.0
        self._status: str = "idle"
        self._language: str = "UK"
        self._recording_start_time: float = 0
        self._elapsed_text: str = "0:00"
        self._dot_pulse: bool = False

        # Window flags
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.PILL_WIDTH, self.PILL_HEIGHT)

        # Drag state
        self._drag_pos = None

        # Timers
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._tick)

        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._toggle_dot)

        # Fade animation
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_anim.setDuration(250)

        # Position at top center
        self._center_on_screen()

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + (geom.width() - self.PILL_WIDTH) // 2
            y = geom.y() + 40
            self.move(x, y)

    def show_recording(self, language: str = "UK"):
        self._language = language.upper()
        self._status = "recording"
        self._recording_start_time = time.time()
        self._elapsed_text = "0:00"
        self._audio_level = 0.0
        self._smoothed_level = 0.0

        self._update_timer.start(50)
        self._dot_timer.start(600)

        self.setWindowOpacity(0.0)
        self.show()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(0.95)
        self._fade_anim.start()

    def show_processing(self):
        self._status = "processing"
        self._dot_timer.stop()
        self._dot_pulse = False
        self.update()

    def hide_recording(self):
        self._status = "idle"
        self._update_timer.stop()
        self._dot_timer.stop()

        self._fade_anim.stop()
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(0.0)
        try:
            self._fade_anim.finished.disconnect()
        except: pass
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()

    def update_audio_level(self, level: float):
        self._audio_level = min(1.0, level * 15.0)

    def _tick(self):
        if self._status == "recording":
            elapsed = time.time() - self._recording_start_time
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            self._elapsed_text = f"{mins}:{secs:02d}"

        target = self._audio_level
        self._smoothed_level += (target - self._smoothed_level) * 0.25
        self.update()

    def _toggle_dot(self):
        self._dot_pulse = not self._dot_pulse
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        
        # === Background capsule ===
        bg_color = QColor(BG_SIDEBAR)
        bg_color.setAlpha(230)
        
        path = QPainterPath()
        path.addRoundedRect(QRectF(1, 1, w-2, h-2), self.CORNER_RADIUS, self.CORNER_RADIUS)
        
        painter.fillPath(path, QBrush(bg_color))
        
        # Subtle border
        border_color = QColor(ACCENT)
        border_color.setAlpha(100)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawPath(path)

        # === Status dot ===
        dot_x = 22
        dot_y = h // 2
        
        dot_color = QColor(ERROR if self._status != "idle" else SUCCESS)
        if self._status == "recording" and self._dot_pulse:
            dot_color = dot_color.lighter(130)
            
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot_color)
        painter.drawEllipse(QPoint(dot_x, dot_y), 5, 5)
        
        # Dot Glow
        glow = QColor(dot_color)
        glow.setAlpha(60 if self._dot_pulse else 30)
        painter.setBrush(glow)
        painter.drawEllipse(QPoint(dot_x, dot_y), 8, 8)

        # === Status text ===
        painter.setPen(QColor(TEXT_PRIMARY))
        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        painter.setFont(font)
        status_msg = "RECORDING" if self._status == "recording" else "PROCESSING"
        painter.drawText(40, 0, 80, h, Qt.AlignmentFlag.AlignVCenter, status_msg)

        # === Audio level dots (Minimalist) ===
        bar_x = 125
        num_dots = 5
        dot_size = 4
        spacing = 6
        for i in range(num_dots):
            active = self._smoothed_level > (i / num_dots)
            color = QColor(ACCENT if active else "#334155")
            painter.setBrush(color)
            painter.drawEllipse(bar_x + i*(dot_size+spacing), h//2 - 2, dot_size, dot_size)

        # === Language badge ===
        badge_x = 185
        badge_w = 40
        badge_h = 22
        badge_y = (h - badge_h) // 2
        
        badge_path = QPainterPath()
        badge_path.addRoundedRect(QRectF(badge_x, badge_y, badge_w, badge_h), 6, 6)
        painter.setBrush(QColor("#334155"))
        painter.fillPath(badge_path, QColor("#334155"))
        
        painter.setPen(QColor(TEXT_PRIMARY))
        font_small = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_small)
        painter.drawText(badge_x, badge_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, self._language)

        # === Time ===
        painter.setPen(QColor(TEXT_SECONDARY))
        font_time = QFont("Segoe UI", 10)
        painter.setFont(font_time)
        painter.drawText(w - 70, 0, 50, h, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, self._elapsed_text)

        painter.end()

    # Dragging logic
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
