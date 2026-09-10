"""A diagnostic bundle the user can attach to a bug report.

``build_bundle(project_root, note)`` zips, without any image or measurement:
versions of Python and the libraries that shape results, the machine and GPU,
the app configuration with user paths shortened, the crash breadcrumb and
crash traces, the app's own logs, the active project's detection logs and its
file inventory (names and sizes only), the persisted queues, and the user's
note. The zip lands in ``<app home>/feedback/``; nothing is sent anywhere.
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from functions import branding as _brand
from functions.layout import app_home, get_datasets_root

ISSUES_URL = "https://github.com/soloyant/pebblemapper/issues"
MAX_LOG_BYTES = 2_000_000       # per file
MAX_LOG_FILES = 40


def _versions() -> dict:
    out = {"python": sys.version, "platform": platform.platform(),
           "machine": platform.machine(), "processor": platform.processor()}
    try:
        from functions import __version__
        out[_brand.APP_NAME] = __version__
    except Exception:
        out[_brand.APP_NAME] = "unknown"
    for mod in ("numpy", "pandas", "scipy", "matplotlib", "tensorflow", "cv2",
                "skimage", "shapely", "nicegui", "reportlab", "PIL", "pillow_heif"):
        try:
            import importlib
            m = importlib.import_module(mod)
            out[mod] = str(getattr(m, "__version__", getattr(m, "VERSION", "?")))
        except Exception as ex:
            out[mod] = f"not importable ({type(ex).__name__})"
    try:
        from osgeo import gdal
        out["gdal"] = gdal.__version__
    except Exception:
        out["gdal"] = "not importable"
    # Whether an iPhone photograph opens here, and why not when it does not.
    try:
        from functions import images
        out["heif_available"] = bool(images.HEIF_AVAILABLE)
        out["heif_error"] = images.heif_import_error()
    except Exception as ex:
        out["heif_available"] = False
        out["heif_error"] = f"functions.images not importable ({type(ex).__name__}: {ex})"
    try:
        import tensorflow as tf
        out["gpu_devices"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        out["gpu_devices"] = "unknown"
    return out


def _redact(text: str) -> str:
    """Shorten user paths: C:\\Users\\name\\… becomes ~\\…, also inside JSON
    (doubled backslashes) and with forward slashes."""
    home = str(Path.home())
    for variant in (home, home.replace("\\", "\\\\"), home.replace("\\", "/")):
        text = text.replace(variant, "~")
    text = re.sub(r"[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[^\\/\"']+", "~", text)
    return text


def _add_text(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, _redact(text))


def _add_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) > MAX_LOG_BYTES:
        data = data[-MAX_LOG_BYTES:]
        arcname += ".tail"
    zf.writestr(arcname, _redact(data.decode("utf-8", errors="replace")))
    return True


def _inventory(root: Path) -> list:
    rows = []
    try:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                try:
                    rows.append(f"{p.stat().st_size:>12}  {p.relative_to(root)}")
                except OSError:
                    pass
            if len(rows) > 5000:
                rows.append("… (truncated)")
                break
    except OSError:
        pass
    return rows


def build_bundle(project_root: Optional[Path] = None, note: str = "",
                 out_dir: Optional[Path] = None) -> Path:
    """Write the zip and return its path."""
    home = app_home()
    out_dir = Path(out_dir) if out_dir else home / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = out_dir / f"{_brand.APP_SLUG}-feedback-{stamp}.zip"
    n_logs = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _add_text(zf, "README.txt",
                  f"{_brand.APP_NAME} diagnostic bundle, {stamp}.\n"
                  f"Attach this file to an issue at {ISSUES_URL}.\n"
                  "It holds versions, configuration, logs and file names only — "
                  "no images and no measurements. User paths are shortened to ~.\n")
        _add_text(zf, "note.txt", note or "(no note)")
        _add_text(zf, "versions.json", json.dumps(_versions(), indent=2))
        _add_text(zf, "environment.json", json.dumps(
            {k: v for k, v in os.environ.items()
             if k.startswith(("PEBBLEMAPPER", "CUDA", "TF_", "CONDA_DEFAULT_ENV"))},
            indent=2))
        cfg = home / "config.json"
        if cfg.is_file():
            _add_file(zf, cfg, "app/config.json")
        crumb = home / "last-session.json"
        if crumb.is_file():
            _add_file(zf, crumb, "app/last-session.json")
        for sub in ("logs", "queues"):
            d = home / sub
            if d.is_dir():
                for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:MAX_LOG_FILES]:
                    if p.is_file():
                        n_logs += _add_file(zf, p, f"app/{sub}/{p.name}")
        _add_text(zf, "datasets_root.txt", str(get_datasets_root()))
        if project_root and Path(project_root).is_dir():
            root = Path(project_root)
            _add_text(zf, "project/inventory.txt", "\n".join(_inventory(root)))
            logs = [p for p in root.rglob("*.txt") if p.name.endswith("_detection_log.txt")]
            logs += [p for p in root.rglob("crash_*.log")]
            logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for p in logs[:MAX_LOG_FILES]:
                n_logs += _add_file(zf, p, f"project/logs/{p.name}")
            for p in root.rglob("*.manifest.json"):
                # Its path inside the project, not its bare name: two runs in
                # two folders share a name, and a zip with the same entry
                # twice warns and keeps one.
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    rel = p.name
                _add_file(zf, p, f"project/manifests/{rel}")
        _add_text(zf, "summary.txt", f"{n_logs} log files included.")
    return out


__all__ = ["build_bundle", "ISSUES_URL"]
