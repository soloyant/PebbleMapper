"""Where the installer keeps its own files, under the app's home folder."""
from __future__ import annotations

from pathlib import Path

from functions.layout import app_home


def models_state_dir() -> Path:
    """``~/.pebblemapper/models``: the registration file (030 FR-014)."""
    return app_home() / "models"


def installed_file() -> Path:
    return models_state_dir() / "installed.json"


def jobs_dir() -> Path:
    """``~/.pebblemapper/jobs``: the queue, one folder per job, the worker lock."""
    return app_home() / "jobs"


def list_cache_dir() -> Path:
    """``~/.pebblemapper/list``: the last list accepted from each source."""
    return app_home() / "list"


def write_json_atomic(path: Path, data) -> None:
    """Write JSON next to ``path`` and move it into place in one step, so a
    reader never sees half a file and a crash never leaves one."""
    import json
    import os
    import threading
    import time
    path.parent.mkdir(parents=True, exist_ok=True)
    # One temporary file per writer: two writers sharing a name collide on
    # Windows ("used by another process"), as two open pages did.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # A reader (or antivirus) holds the target for a moment.
            if delay is None:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(delay)
