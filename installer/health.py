"""Is each installed model still whole? (030 FR-043)

``quick_check`` runs after the first page has loaded, for every model the app
installed: folders present, the environment's package records present, the
plug-in package present, each recorded file present at its recorded size. It
reads no file's content, so it costs milliseconds. ``full_check`` (Repair)
also hashes every recorded file. A model that fails is marked
``needs_repair`` with the reason, shown on the Models tab -- never silently
dropped from the Detection model list.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from installer import registration


def quick_check(entry: dict) -> Optional[str]:
    """``None`` when whole, else the reason in plain words."""
    root = Path(entry.get("root", ""))
    if not root.is_dir():
        return f"its folder {root} is missing"
    if entry.get("runs_in", "own") != "app":
        env = Path(entry["env"])
        if not (env / "conda-meta").is_dir():
            return "its software (the conda environment) is missing or incomplete"
        if not ((env / "python.exe").is_file() or (env / "bin" / "python").is_file()):
            return "its software (Python) is missing"
    code = Path(entry["code"])
    for b in entry.get("backends", []):
        if not (code / b["package"] / "__init__.py").is_file():
            return f"its program files ({b['package']}) are missing"
    for f in entry.get("files", []):
        p = root / f["path"]
        if not p.is_file():
            return f"the file {f['path']} is missing"
        if p.stat().st_size != f["size"]:
            return f"the file {f['path']} has changed"
    return None


def full_check(entry: dict) -> List[str]:
    """Recorded files whose content no longer matches (Repair)."""
    from installer.fetch import sha256_of
    root = Path(entry["root"])
    bad = []
    for f in entry.get("files", []):
        p = root / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"] or sha256_of(p) != f["sha256"]:
            bad.append(f["path"])
    return bad


def check_all(path: Optional[Path] = None) -> List[dict]:
    """Quick-check every installed model and record the result; returns the
    entries that need repair."""
    broken = []
    for e in registration.read_installed(path):
        reason = quick_check(e)
        state = "needs_repair" if reason else "ok"
        if (e.get("state"), e.get("reason", "")) != (state, reason or ""):
            registration.set_state(e["id"], e["release"], e["variant"], state,
                                   reason or "", path)
        if reason:
            broken.append(dict(e, state=state, reason=reason))
    return broken
