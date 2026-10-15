"""Build a model's environment from its release's lock files (030 R1, R2).

``conda create -p <env> --file conda-<variant>.lock.txt`` reads an
``@EXPLICIT`` list of package URLs with their SHA-256: no solver runs and
every package is checked. ``pip install --require-hashes --no-deps`` then adds
the pip packages, each pinned and hashed. Neither reads the user's own conda
or pip settings for what to install (FR-010): channels are overridden, the
pip configuration is ``os.devnull`` and ``--isolated`` drops ``PIP_*``. The
package caches go to the job's own folder (FR-012).

A partly built environment is never repaired: it is deleted and built again
(FR-032). Nothing here removes a working environment (FR-013) -- the caller
builds into a new folder.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from functions.conda_locate import find_conda_exe
from installer.errors import InstallError

MIN_CONDA = (24, 9)                       # explicit lists with --sha256
CONDA_TIMEOUT_S = 3600
PIP_TIMEOUT_S = 3600


def conda_version(conda_exe: str) -> tuple:
    out = subprocess.run([conda_exe, "--version"], capture_output=True,
                         text=True, timeout=60,
                         env=dict(os.environ, CONDA_NO_PLUGINS="true"))
    m = re.search(r"(\d+)\.(\d+)", out.stdout + out.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def conda_env_vars(work_dir: Path) -> dict:
    """Variables every conda call gets (research R2)."""
    return {
        "CONDA_PKGS_DIRS": str(Path(work_dir) / "pkgs"),
        "CONDA_CHANNEL_PRIORITY": "strict",
        "CONDA_NO_PLUGINS": "true",      # also keeps Anaconda's ToS gate out
        "CONDA_YES": "true",
        "CONDA_NUMBER_CHANNEL_NOTICES": "0",   # no extra fetch beyond the lock
        "PYTHONNOUSERSITE": "1",               # a user-level package never counts
        "TMP": str(Path(work_dir) / "tmp"),
        "TEMP": str(Path(work_dir) / "tmp"),
        "PYTHONIOENCODING": "utf-8",
    }


def pip_env_vars(work_dir: Path) -> dict:
    return {
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_CACHE_DIR": str(Path(work_dir) / "pip"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        # pip must not see ~\AppData\Roaming\Python: a package there would
        # count as installed and the lock would be left incomplete.
        "PYTHONNOUSERSITE": "1",
        "TMP": str(Path(work_dir) / "tmp"),
        "TEMP": str(Path(work_dir) / "tmp"),
        "PYTHONIOENCODING": "utf-8",
    }


def conda_create_argv(conda_exe: str, prefix: Path, conda_lock: Path) -> List[str]:
    return [conda_exe, "create", "-p", str(prefix), "--file", str(conda_lock),
            "--no-default-packages", "--override-channels", "-c", "conda-forge",
            "--yes"]


def env_python(prefix: Path) -> Path:
    win = Path(prefix) / "python.exe"
    return win if win.is_file() or os.name == "nt" else Path(prefix) / "bin" / "python"


def pip_install_argv(prefix: Path, pip_lock: Path,
                     find_links: Optional[Path] = None,
                     cache_dir: Optional[Path] = None) -> List[str]:
    """``find_links``: the folder of wheels the release ships (FR-008); pip
    still requires each one to match its hash in the lock. ``cache_dir`` is
    passed on the command line because ``--isolated`` makes pip ignore every
    ``PIP_*`` variable, PIP_CACHE_DIR included (FR-012)."""
    argv = [str(env_python(prefix)), "-m", "pip", "install", "--isolated",
            "--require-hashes", "--no-deps", "--only-binary", ":all:",
            "--no-warn-script-location", "-r", str(pip_lock)]
    if find_links:
        argv += ["--find-links", str(find_links)]
    argv += ["--cache-dir", str(cache_dir)] if cache_dir else ["--no-cache-dir"]
    return argv


def lock_has_packages(pip_lock: Optional[Path]) -> bool:
    """A pip lock with at least one requirement (index lines alone don't count)."""
    if not pip_lock or not Path(pip_lock).is_file():
        return False
    for line in Path(pip_lock).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith(("#", "-", "\\")):
            return True
    return False


def remove_env(prefix: Path) -> None:
    """Delete a partial or abandoned environment folder (never through a
    link: ``shutil.rmtree`` refuses a top-level link, and Python 3.8+ does
    not follow junctions inside the tree)."""
    prefix = Path(prefix)
    if prefix.is_symlink():
        raise InstallError("unknown", detail=f"{prefix} is a link; not removed")
    if prefix.exists():
        shutil.rmtree(prefix, ignore_errors=False)


def _run(argv, env_extra, step, cancel, log, timeout_s, drive="of the models folder"):
    """Run one child process, stream its output to ``log``, kill its whole
    tree on cancel or timeout."""
    from installer.fetch import Cancelled
    env = dict(os.environ, **env_extra)
    for k in [k for k in env if k.startswith("PIP_") and k not in env_extra]:
        del env[k]                       # a user's PIP_* never reaches pip
    log(f"$ {' '.join(argv)}")
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail: List[str] = []

    def _pump():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-40]
                log(line)
    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    started = time.monotonic()
    while proc.poll() is None:
        if cancel and cancel():
            _kill_tree(proc)
            raise Cancelled()
        if time.monotonic() - started > timeout_s:
            _kill_tree(proc)
            raise InstallError("timeout", step=step)
        time.sleep(0.5)
    t.join(timeout=5)
    if proc.returncode != 0:
        raise _classify_failure(tail, step, drive)


def _classify_failure(tail: List[str], step: str, drive: str = "of the models folder") -> InstallError:
    text = "\n".join(tail)
    low = text.lower()
    if "no space left" in low or "errno 28" in low or "not enough space" in low:
        return InstallError("disk_full", drive=drive, missing="several GB")
    if "certificate verify failed" in low or "ssl" in low and "certificate" in low:
        host = re.search(r"https?://([^/\s'\"]+)", text)
        return InstallError("clock", host=host.group(1) if host else "a package server")
    if ("connection" in low and ("error" in low or "refused" in low or "timed out" in low)) \
            or "could not resolve" in low or "name resolution" in low:
        host = re.search(r"https?://([^/\s'\"]+)", text)
        return InstallError("network", host=host.group(1) if host else "a package server")
    if "hashes are required" in low or "do not match the hashes" in low \
            or "checksum mismatch" in low or "sha256 mismatch" in low:
        return InstallError("published_changed", file=step)
    return InstallError("unknown", detail=f"{step} failed: {tail[-1] if tail else 'no output'}")


def _kill_tree(proc) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def create_env(prefix: Path, conda_lock: Path, pip_lock: Optional[Path],
               work_dir: Path, log: Callable[[str], None],
               cancel: Optional[Callable[[], bool]] = None,
               on_step: Optional[Callable[[str], None]] = None,
               conda_exe: Optional[str] = None,
               find_links: Optional[Path] = None) -> None:
    """Build the environment at ``prefix`` from the two lock files.

    A folder already at ``prefix`` is a partial build from an interrupted
    job: it is deleted first, never repaired.
    """
    conda_exe = conda_exe or find_conda_exe()
    if not conda_exe:
        raise InstallError("conda_too_old")
    if conda_version(conda_exe) < MIN_CONDA:
        raise InstallError("conda_too_old")
    remove_env(prefix)
    (Path(work_dir) / "tmp").mkdir(parents=True, exist_ok=True)
    if on_step:
        on_step("env-conda")
    try:
        _run(conda_create_argv(conda_exe, prefix, conda_lock),
             conda_env_vars(work_dir), "Installing the model's software",
             cancel, log, CONDA_TIMEOUT_S, _drive(prefix))
        if lock_has_packages(pip_lock):
            if on_step:
                on_step("env-pip")
            _run(pip_install_argv(prefix, pip_lock, find_links, Path(work_dir) / "pip"),
                 pip_env_vars(work_dir),
                 "Installing the model's Python packages", cancel, log, PIP_TIMEOUT_S,
                 _drive(prefix))
    except BaseException:
        # Leave no half-built environment that could look installed.
        try:
            remove_env(prefix)
        except Exception as ex:          # reported, not hidden (Principle I)
            log(f"Could not remove the partial build at {prefix}: {ex}")
        raise


def _drive(prefix: Path) -> str:
    return Path(prefix).drive or str(Path(prefix).anchor) or "of the models folder"
