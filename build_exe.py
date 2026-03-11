"""Build standalone .exe for CORE using PyInstaller."""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def build():
    """Build the executable."""
    project_dir = Path(__file__).parent
    src_dir = project_dir / "src"
    venv_site = project_dir / "venv" / "Lib" / "site-packages"

    # Use a temp build directory without cyrillic to avoid PyInstaller path issues
    build_base = Path(os.environ.get("TEMP", "C:\\Temp")) / "wispan_build"
    build_base.mkdir(parents=True, exist_ok=True)
    dist_dir = build_base / "dist"
    work_dir = build_base / "build"

    # Use global python (no cyrillic path) but add venv site-packages
    global_python = Path(sys.executable)
    # If running from venv, find the global python
    if "venv" in str(global_python).lower():
        global_python = Path("C:/Users/PAVLO/AppData/Local/Programs/Python/Python311/python.exe")

    # Build command: set PYTHONPATH to include venv packages + src
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{venv_site};{src_dir}"

    cmd = [
        str(global_python), "-c",
        "import sys; sys.argv = sys.argv[1:]; from PyInstaller.__main__ import run; run()",
        "--name=CORE",
        "--onedir",
        "--windowed",
        "--noconfirm",
        f"--paths={src_dir}",
        f"--paths={venv_site}",
        f"--distpath={dist_dir}",
        f"--workpath={work_dir}",
        f"--specpath={build_base}",
        f"--add-data={project_dir / 'config_default.json'};.",
        "--hidden-import=sounddevice",
        "--hidden-import=numpy",
        "--hidden-import=pynput",
        "--hidden-import=pynput.keyboard",
        "--hidden-import=pynput.keyboard._win32",
        "--hidden-import=pyperclip",
        "--hidden-import=pyautogui",
        "--hidden-import=loguru",
        "--hidden-import=PyQt6",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=PyQt6.QtGui",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=deepgram",
        "--exclude-module=faster_whisper",
        "--exclude-module=ctranslate2",
        "--exclude-module=onnxruntime",
        "--exclude-module=torch",
        "--exclude-module=PyQt5",
        "--exclude-module=tkinter",
        "--exclude-module=PIL",
        str(src_dir / "core.py"),
    ]

    print("Building CORE...")
    print(f"Build dir: {build_base}")
    print()

    result = subprocess.run(cmd, cwd=str(build_base), env=env)

    if result.returncode == 0:
        # Copy result back to project
        final_dist = project_dir / "dist" / "CORE"
        if final_dist.exists():
            shutil.rmtree(final_dist)
        final_dist.parent.mkdir(parents=True, exist_ok=True)
        # PyInstaller names the folder after the entry point file, not --name
        source = dist_dir / "CORE"
        if not source.exists():
            source = dist_dir / "main"
        if not source.exists():
            # List what's actually in dist_dir to help debug
            print()
            print("WARNING: Expected output folder not found!")
            print(f"Looked for: {dist_dir / 'CORE'}")
            print(f"        and: {dist_dir / 'main'}")
            if dist_dir.exists():
                contents = list(dist_dir.iterdir())
                if contents:
                    print(f"Contents of {dist_dir}:")
                    for item in contents:
                        print(f"  - {item.name}")
                    # Try to use the first directory found
                    dirs = [c for c in contents if c.is_dir()]
                    if dirs:
                        source = dirs[0]
                        print(f"Using: {source}")
                    else:
                        print("No directories found in dist output!")
                        sys.exit(1)
                else:
                    print(f"{dist_dir} is empty!")
                    sys.exit(1)
            else:
                print(f"{dist_dir} does not exist!")
                sys.exit(1)

        shutil.copytree(source, final_dist)

        print()
        print("=" * 50)
        print("BUILD SUCCESSFUL!")
        print("=" * 50)
        print()
        print(f"Executable: {final_dist / 'CORE.exe'}")
    else:
        print("Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    build()
