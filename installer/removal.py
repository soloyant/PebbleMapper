"""Remove what the installer made, and nothing else (030 FR-040, FR-041).

Every deletion stays inside a folder the app created for one model (the
model's own folder under the models folder, or a job's working folder), and
never follows a link or a junction: a link found inside is unlinked, its
target left alone. The space actually freed is measured, not estimated.
"""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Optional

from installer import jobs, registration
from installer.archive import retry_io
from installer.errors import InstallError


def _is_link_or_junction(p: Path) -> bool:
    try:
        if p.is_symlink():
            return True
        st = os.lstat(p)
        # A Windows junction is a reparse point, not a symlink to Python 3.9.
        return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return False


def _size(folder: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if not _is_link_or_junction(Path(root) / d)]
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def safe_rmtree(folder: Path, inside: Path) -> int:
    """Delete ``folder`` (which must lie inside ``inside``) without following
    links; return the bytes freed."""
    folder, inside = Path(folder), Path(inside)
    try:
        Path(os.path.abspath(folder)).relative_to(Path(os.path.abspath(inside)))
    except ValueError:
        raise InstallError("unknown", detail=f"{folder} is outside {inside}; not removed")
    if Path(os.path.abspath(folder)) == Path(os.path.abspath(inside)):
        raise InstallError("unknown", detail=f"refusing to remove {inside} itself")
    if not folder.exists():
        return 0
    if _is_link_or_junction(folder):
        raise InstallError("unknown", detail=f"{folder} is a link; not removed")
    before = _size(folder)
    # Unlink every link/junction first, so rmtree can never walk through one.
    for root, dirs, files in os.walk(folder, topdown=True, followlinks=False):
        for d in list(dirs):
            p = Path(root) / d
            if _is_link_or_junction(p):
                os.rmdir(p) if os.name == "nt" else os.unlink(p)
                dirs.remove(d)

    def _onerror(fn, path, exc):
        os.chmod(path, stat.S_IWRITE)            # read-only files in envs
        fn(path)
    retry_io(shutil.rmtree, folder, onerror=_onerror)
    return before


def _models_dir_of(root: Path) -> Path:
    return Path(root).parent


def remove_job(job: dict) -> int:
    """Remove an interrupted, cancelled or failed job: its working folder and
    the model folder it was building (never a registered model's folder)."""
    req_file = jobs.job_dir(job["id"]) / "request.json"
    freed = 0
    try:
        import json
        req = json.loads(req_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        req = None
    if req:
        from installer.worker import model_root
        mdir = Path(req["models_dir"])
        root = model_root(mdir, req["model"], req["release"], req["variant"])
        registered = any(e.get("root") == str(root) for e in registration.read_installed())
        freed += safe_rmtree(mdir / ".work" / job["id"], mdir)
        if not registered:
            freed += safe_rmtree(root, mdir)
    jobs.forget(job["id"])
    return freed


def uninstall(model_id: str, release: str, variant: str) -> int:
    """Remove one installed release: unregister first (so it can never look
    installed while half deleted), then delete its folder. Returns bytes freed."""
    entry = next((e for e in registration.read_installed()
                  if (e["id"], e["release"], e["variant"]) == (model_id, release, variant)), None)
    if entry is None:
        raise InstallError("unknown", detail=f"{model_id} {release} is not installed")
    root = Path(entry["root"])
    registration.unregister(model_id, release, variant)
    freed = safe_rmtree(root, _models_dir_of(root))
    _forget_conda_env(root / "env")
    return freed


def _forget_conda_env(prefix: Path) -> None:
    """Drop the removed environment from conda's list of environments."""
    f = Path.home() / ".conda" / "environments.txt"
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    keep = [ln for ln in lines if os.path.normcase(ln.strip()) != os.path.normcase(str(prefix))]
    if keep != lines:
        f.write_text("\n".join(keep) + "\n", encoding="utf-8")


def run_uninstall_job(job: dict, req: dict, log) -> None:
    freed = uninstall(req["model"], req["release"], req["variant"])
    job["freed_bytes"] = freed
    jobs.write_job(job)
    log(f"removed {req['model']} {req['release']}: {freed} bytes freed")
