"""Build standalone .exe for VoiceType UA using PyInstaller."""

import subprocess
import sys
from pathlib import Path

def build():
    """Build the executable."""

    # Get project root
    project_dir = Path(__file__).parent
    src_dir = project_dir / "src"

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=VoiceTypeUA",
        "--onedir",  # onedir is faster to build and start than onefile
        "--windowed",  # No console window
        "--noconfirm",  # Overwrite without asking

        # Add src to Python path
        f"--paths={src_dir}",

        # Add data files
        f"--add-data={project_dir / 'config_default.json'};.",

        # Hidden imports (modules that PyInstaller might miss)
        "--hidden-import=faster_whisper",
        "--hidden-import=ctranslate2",
        "--hidden-import=sounddevice",
        "--hidden-import=numpy",
        "--hidden-import=pynput",
        "--hidden-import=pynput.keyboard",
        "--hidden-import=pynput.keyboard._win32",
        "--hidden-import=pyperclip",
        "--hidden-import=loguru",
        "--hidden-import=PyQt6",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=PyQt6.QtGui",
        "--hidden-import=PyQt6.QtCore",

        # Collect all files from these packages
        "--collect-all=faster_whisper",
        "--collect-all=ctranslate2",

        # Entry point
        str(src_dir / "main.py"),
    ]

    print("Building VoiceType UA...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run PyInstaller
    result = subprocess.run(cmd, cwd=str(project_dir))

    if result.returncode == 0:
        print()
        print("=" * 50)
        print("BUILD SUCCESSFUL!")
        print("=" * 50)
        print()
        print(f"Executable location: {project_dir / 'dist' / 'VoiceTypeUA' / 'VoiceTypeUA.exe'}")
        print()
        print("To run: double-click VoiceTypeUA.exe")
        print()
        print("NOTE: First run will download Whisper model (~150MB)")
    else:
        print("Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    build()
