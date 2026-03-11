"""Central styling system for WispanTalk."""

from PyQt6.QtGui import QColor

# --- Color Palette (Modern Dark) ---
BG_MAIN = "#121212"       # Surface
BG_SIDEBAR = "#1E1E1E"    # Sidebar/Darker surface
BG_CARD = "#252525"       # Card/Element background
ACCENT = "#6366F1"        # Indigo-500
ACCENT_HOVER = "#818CF8"  # Indigo-400
TEXT_PRIMARY = "#F8FAFC"  # Slate-50
TEXT_SECONDARY = "#94A3B8" # Slate-400
BORDER = "#334155"        # Slate-700
SUCCESS = "#10B981"       # Emerald-500
ERROR = "#EF4444"         # Red-500

# --- Global Stylesheet ---
MAIN_STYLE = f"""
QWidget {{
    background-color: {BG_MAIN};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 10pt;
}}

/* Sidebar Label */
QLabel#sidebarLabel {{
    color: {TEXT_SECONDARY};
    font-weight: bold;
    font-size: 9pt;
    padding: 10px 5px;
    text-transform: uppercase;
}}

/* Sidebar Button */
QPushButton#sidebarBtn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-radius: 8px;
    padding: 10px 15px;
    text-align: left;
    font-weight: 500;
    font-size: 10pt;
}}

QPushButton#sidebarBtn:hover {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
}}

QPushButton#sidebarBtn[active="true"] {{
    background-color: {ACCENT};
    color: white;
}}

/* Main Content Area */
QFrame#contentArea {{
    background-color: {BG_MAIN};
    border-left: 1px solid {BORDER};
    padding: 20px;
}}

/* Forms and Inputs */
QLineEdit {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
}}

QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 150px;
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_CARD};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    image: url(check_mark.png); /* Fallback to style if needed */
}}

/* Buttons */
QPushButton {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {BORDER};
}}

QPushButton#primaryBtn {{
    background-color: {ACCENT};
    border: none;
    color: white;
}}

QPushButton#primaryBtn:hover {{
    background-color: {ACCENT_HOVER};
}}

/* Header Label */
QLabel#headerTitle {{
    font-size: 18pt;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}}

QLabel#headerSub {{
    font-size: 10pt;
    color: {TEXT_SECONDARY};
    margin-bottom: 15px;
}}
"""

# --- Overlay Styles ---
OVERLAY_STYLE = f"""
QFrame#overlayPill {{
    background-color: rgba(30, 30, 30, 220);
    border: 1px solid {ACCENT};
    border-radius: 20px;
}}

QLabel#overlayText {{
    color: white;
    font-weight: bold;
    font-size: 11pt;
    background: transparent;
}}

QLabel#overlayIcon {{
    background-color: {ERROR};
    border-radius: 6px;
}}
"""
