"""OS-specific utilities (Windows)."""

import os
import sys
import winreg
from pathlib import Path
from loguru import logger

from utils.constants import APP_NAME


def set_autostart(enabled: bool) -> bool:
    """Enable or disable application autostart with Windows.
    
    Args:
        enabled: True to enable, False to disable
        
    Returns:
        True if successful, False otherwise
    """
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = APP_NAME
    
    try:
        # Determine the command to run
        if getattr(sys, 'frozen', False):
            # Running as compiled .exe
            app_path = sys.executable
            command = f'"{app_path}"'
        else:
            # Running from source — use pythonw.exe directly (not .bat!)
            # .bat files are unreliable for Windows autostart (cd/start don't work)
            # __file__ = src/utils/os_utils.py → parent.parent.parent = project root
            project_root = Path(__file__).resolve().parent.parent.parent
            venv_pythonw = project_root / "venv" / "Scripts" / "pythonw.exe"
            core_py = project_root / "src" / "core.py"
            
            if venv_pythonw.exists():
                command = f'"{venv_pythonw}" "{core_py}" --debug'
            else:
                # Fallback: system pythonw
                pythonw = Path(sys.executable).parent / "pythonw.exe"
                command = f'"{pythonw}" "{core_py}" --debug'
        
        logger.debug(f"Autostart command: {command}")
        
        # Open the registry key
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            key_path,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
        )
        
        if enabled:
            winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, command)
            logger.info(f"Autostart enabled: added {app_name} to registry")
        else:
            try:
                winreg.DeleteValue(key, app_name)
                logger.info(f"Autostart disabled: removed {app_name} from registry")
            except FileNotFoundError:
                # Already disabled
                pass
                
        winreg.CloseKey(key)
        return True
        
    except Exception as e:
        logger.error(f"Failed to set autostart: {e}")
        return False


def is_autostart_enabled() -> bool:
    """Check if autostart is currently enabled in registry.
    
    Returns:
        True if enabled
    """
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = APP_NAME
    
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, app_name)
            enabled = True
        except FileNotFoundError:
            enabled = False
        winreg.CloseKey(key)
        return enabled
    except Exception:
        return False
