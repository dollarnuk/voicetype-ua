"""UI module - PyQt6 interface components."""

try:
    from .tray_app import TrayApp, run_gui
    __all__ = ["TrayApp", "run_gui"]
except ImportError:
    __all__ = []
