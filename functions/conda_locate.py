"""Find the conda PebbleMapper uses, without importing the detection stack.

Lives in ``functions`` (whose package import is light) rather than in
``detectors`` (whose package import loads TensorFlow), so the model install
worker can find conda cheaply. ``detectors.subprocess_runner`` re-exports it.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

# Written by install.ps1 in the app folder: the conda.bat the launcher uses.
_CONDA_LOCATION_FILE = Path(__file__).resolve().parents[1] / ".conda-location"


def _conda_from_location_file(path: Path = None) -> Optional[str]:
    """The conda.exe of the installation named in ``.conda-location``, or None.

    The file holds ``<root>\\condabin\\conda.bat``; the executable next to it is
    ``<root>\\Scripts\\conda.exe`` (``<root>/bin/conda`` elsewhere). Using it means
    the app runs every conda command with the conda the launcher uses (030
    FR-011), not whichever one a fixed search finds first.
    """
    path = path or _CONDA_LOCATION_FILE
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().strip('"')
    except OSError:
        return None
    if not text:
        return None
    root = Path(text).parent.parent
    for exe in (root / "Scripts" / "conda.exe", root / "bin" / "conda"):
        if exe.is_file():
            return str(exe)
    return None


def find_conda_exe() -> Optional[str]:
    """Locate a conda/mamba executable, or None.

    Honours ``PEBBLE_CONDA_EXE`` first (explicit override), then the conda the
    installer recorded in ``.conda-location``, then a handful of conventional
    install locations on Windows and POSIX, then the PATH.
    """
    env_override = os.environ.get("PEBBLE_CONDA_EXE")
    if env_override and Path(env_override).exists():
        return env_override
    recorded = _conda_from_location_file()
    if recorded:
        return recorded
    # A .bat/.cmd shim (what conda puts on PATH via condabin) does not launch
    # reliably through subprocess.run's list form on Windows; try real exes first.
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.expanduser("~/miniforge3/Scripts/conda.exe"),
        os.path.join(local, "miniforge3", "Scripts", "conda.exe") if local else "",
        "C:/ProgramData/miniforge3/Scripts/conda.exe",
        "C:/ProgramData/miniconda3/Scripts/conda.exe",
        "C:/ProgramData/Anaconda3/Scripts/conda.exe",
        os.path.expanduser("~/miniconda3/Scripts/conda.exe"),
        os.path.expanduser("~/anaconda3/Scripts/conda.exe"),
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        os.path.expanduser("~/miniforge3/bin/conda"),
        "/opt/conda/bin/conda",
    ]
    exe = next((c for c in candidates if c and Path(c).exists()), None)
    if exe:
        return exe
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found
    return None


