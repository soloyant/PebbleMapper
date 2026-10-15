"""Prepare an update of PebbleMapper itself; the starter applies it (031 FR-010–FR-018).

Nothing of the running version changes here (FR-011). The job downloads and
verifies the release (030 fetch), unpacks it into ``<app>/.pm-update/stage``
(the app's own volume, so the starter's moves are renames), checks the staged
tree against the release's file list, builds the new environment from the
release's locks beside the current one in ``~/.pebblemapper/app-envs/<release>``,
runs the new version's own installation check in that environment against
the staged tree, and finally records the staged update in the starter's
pointer. The starter does the rest at the next start.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from installer import archive, envbuild, fetch, jobs
from installer.errors import InstallError, human_size

APP_DIR = Path(__file__).resolve().parents[1]
ENV_MARKER = ".pm-tested-locks"     # in an app env: the locks it was built from, once tested


def app_envs_dir() -> Path:
    from functions.layout import app_home
    return app_home() / "app-envs"


def _starter(app: Path):
    """The starter module of this installation, for its pointer helpers."""
    spec = importlib.util.spec_from_file_location("pm_starter", app / "pm_starter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Set by Restart now; gui.app exits with the starter's RESTART_CODE after
# the server stopped, and the starter applies the update and starts again.
RESTART = {"requested": False}
RESTART_CODE = 75


def in_place_refusal(app: Path = APP_DIR) -> Optional[str]:
    """Why this copy cannot update itself from the app, or ``None``."""
    if (app / ".git").exists():
        return "this is a development copy: update it with git"
    if not (app / "pm_starter.py").is_file() or not os.environ.get("PEBBLEMAPPER_STARTER"):
        return ("PebbleMapper was not started from its shortcut; start it from the "
                "PebbleMapper shortcut (or run the installer again) to update from here")
    return None


def staged_release(app: Path = APP_DIR) -> Optional[str]:
    """The release prepared for the next start, if any."""
    try:
        return (_starter(app).load_pointer(app).get("staged") or {}).get("release")
    except Exception:
        return None


def _name(f: dict) -> str:
    from urllib.parse import urlparse
    return Path(urlparse(f["url"]).path).name


class AppUpdateRunner:
    def __init__(self, job: dict, request: dict, app: Path = APP_DIR):
        self.job, self.req, self.app = job, request, Path(app)
        self.rel = request["release_entry"]
        self.variant = request["variant"]
        self.vinfo = self.rel["variants"][self.variant]
        self.upd = self.app / ".pm-update"
        self.work = self.upd / "work" / job["id"]
        self.stage = self.upd / "stage"
        self.env = app_envs_dir() / self.rel["release"]
        self.log = jobs.JobLog(Path(job["log"]))

    def _step(self, name, status, **extra):
        for s in self.job["steps"]:
            if s["name"] == name:
                s["status"] = status
                s.update(extra)
                if status == "running":
                    s["started"] = time.time()
        jobs.write_job(self.job)

    def cancelled(self) -> bool:
        return jobs.cancel_requested(self.job["id"])

    def run(self) -> None:
        st = _starter(self.app)
        if self.rel.get("min_starter", 1) > getattr(st, "STARTER_VERSION", 0):
            raise InstallError("starter_too_old")
        self.check_disk()
        files = self.fetch()
        self._step("verify", "done")
        new_list = self.unpack(files)
        self.build_env(files)
        self.self_test()
        (self.env / ENV_MARKER).write_text(self._locks_digest(files), encoding="utf-8")
        self._step("ready", "running")
        ptr = st.load_pointer(self.app)
        ptr["staged"] = {"release": self.rel["release"], "env": str(self.env),
                         "stage_dir": str(self.stage),
                         "file_list": str(self.upd / f"files-{self.rel['release']}.json")}
        st._write_json(self.app / st.POINTER, ptr)
        shutil.rmtree(self.work, ignore_errors=True)
        self._step("ready", "done")
        self.log(f"ready: {self.rel['release']} is applied at the next start "
                 f"({len(new_list['files'])} files)")

    def check_disk(self) -> None:
        need = int(self.vinfo.get("disk_bytes", 0))
        for where in {self.app.anchor, self.env.anchor}:
            free = shutil.disk_usage(where).free
            if need and free < need:
                raise InstallError("disk_full", drive=where.rstrip("\\"),
                                   missing=human_size(need - free))

    def fetch(self) -> dict:
        self._step("fetch", "running")
        out = {}
        total = sum(f["size"] for f in self.vinfo["files"])
        done = 0
        for f in self.vinfo["files"]:
            dest = self.work / "files" / _name(f)
            fetch.download(f["url"], dest, f["size"], f["sha256"], cancel=self.cancelled,
                           on_progress=lambda d, t, b=done: self._progress(b + d, total))
            done += f["size"]
            out[f["role"]] = dest
            self.log(f"verified {dest.name} ({human_size(f['size'])})")
        self._step("fetch", "done", bytes_done=total, bytes_total=total)
        return out

    def _progress(self, done, total):
        for s in self.job["steps"]:
            if s["name"] == "fetch":
                s["bytes_done"], s["bytes_total"] = done, total
        jobs.write_job(self.job)

    def unpack(self, files: dict) -> dict:
        self._step("stage", "running")
        if self.stage.exists():
            shutil.rmtree(self.stage)
        archive.safe_unzip(files["archive"], self.stage)
        new_list = json.loads(files["file_list"].read_text(encoding="utf-8"))
        st = _starter(self.app)
        try:
            bad = st.verify_stage(self.stage, new_list)
        except ValueError as ex:                       # a path leaving the app folder
            raise InstallError("published_changed", file=str(ex)) from ex
        if bad:
            raise InstallError("published_changed", file=f"{len(bad)} file(s) of the release")
        shutil.copyfile(files["file_list"], self.upd / f"files-{self.rel['release']}.json")
        self._step("stage", "done")
        return new_list

    def _locks_digest(self, files: dict) -> str:
        return "|".join(fetch.sha256_of(files[r]) for r in ("conda_lock", "pip_lock") if r in files)

    def build_env(self, files: dict) -> None:
        self._step("env-conda", "running")
        marker = self.env / ENV_MARKER
        try:
            if marker.read_text(encoding="utf-8") == self._locks_digest(files):
                # Built from these very locks and tested by an earlier job
                # (a stage discarded, an update prepared again).
                self.log(f"reusing the environment already built for {self.rel['release']}")
                self._step("env-conda", "done")
                self._step("env-pip", "done")
                return
        except OSError:
            pass
        if self.env.exists():
            marker.unlink(missing_ok=True)

        def on_step(name):
            if name == "env-pip":
                self._step("env-conda", "done")
                self._step("env-pip", "running")
        envbuild.create_env(self.env, files["conda_lock"], files.get("pip_lock"),
                            self.work, log=self.log, cancel=self.cancelled, on_step=on_step)
        self._step("env-conda", "done")
        self._step("env-pip", "done")

    def self_test(self) -> None:
        """The new version's own installation check, in its own environment,
        against the staged tree: the app-update analogue of a model's check run."""
        self._step("self-test", "running")
        py = envbuild.env_python(self.env)
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(self.stage))
        r = subprocess.run([str(py), str(self.stage / "tools" / "check_install.py")],
                           cwd=str(self.stage), env=env, capture_output=True, text=True,
                           timeout=900, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in (r.stdout + r.stderr).splitlines()[-30:]:
            self.log(line)
        if r.returncode != 0:
            raise InstallError("check_run_failed")
        self._step("self-test", "done")


def unseen_switch(app: Path = APP_DIR) -> Optional[dict]:
    """The starter's last switch, until the user has seen it on the first
    page (031 FR-014, FR-015): ``{"from", "to", "result", "kept_mine"}``."""
    try:
        ls = _starter(app).load_pointer(app).get("last_switch")
    except Exception:
        return None
    if not ls or ls.get("seen") or ls.get("result") == "applied":
        return None                     # "applied" = the provisional start is still on
    if ls.get("result") == "ok" and not ls.get("kept_mine"):
        return None                     # nothing to say; the version is in the header
    return ls


def mark_switch_seen(app: Path = APP_DIR) -> None:
    st = _starter(app)
    ptr = st.load_pointer(app)
    if ptr.get("last_switch"):
        ptr["last_switch"]["seen"] = True
        st._write_json(app / st.POINTER, ptr)


def request_for_app(data: dict, release: str, variant: str) -> dict:
    rel = next(r for r in data.get("app_releases", []) if r["release"] == release)
    return {"model": "pebblemapper", "release": release, "variant": variant,
            "display_name": "PebbleMapper", "release_entry": rel, "app_dir": str(APP_DIR)}


def previous_version(app: Path = APP_DIR) -> Optional[dict]:
    """``{"release", "return_requested"}`` while the version before the
    current one is still kept, else ``None``."""
    try:
        ptr = _starter(app).load_pointer(app)
    except Exception:
        return None
    if not ptr.get("previous") or ptr.get("provisional"):
        return None
    return {"release": ptr["previous"], "return_requested": bool(ptr.get("return_to_previous")),
            "last_result": (ptr.get("last_switch") or {}).get("result")}


def can_return(app: Path = APP_DIR) -> bool:
    """A return needs the journal of the last switch (after a return the
    journal is gone, and the version left is only removable)."""
    return previous_version(app) is not None and (app / ".pm-update" / "journal.json").is_file()


def request_restart_to_previous(app: Path = APP_DIR) -> None:
    """Return to version Y at the next start (031 FR-015)."""
    st = _starter(app)
    ptr = st.load_pointer(app)
    if not can_return(app):
        raise InstallError("unknown", detail="there is no previous version to return to")
    ptr["return_to_previous"] = True
    st._write_json(app / st.POINTER, ptr)


def cancel_return(app: Path = APP_DIR) -> None:
    st = _starter(app)
    ptr = st.load_pointer(app)
    ptr["return_to_previous"] = False
    st._write_json(app / st.POINTER, ptr)


def remove_previous(app: Path = APP_DIR) -> int:
    """Free the previous version's files and environment (US2 scenario 7)."""
    from installer.removal import safe_rmtree
    st = _starter(app)
    ptr = st.load_pointer(app)
    upd = app / ".pm-update"
    freed = safe_rmtree(upd / "previous", upd)
    if not ptr.get("staged"):
        freed += safe_rmtree(upd / "stage", upd)  # what a return to the previous put back
    for name in ("journal.json", "previous_files.json"):
        (upd / name).unlink(missing_ok=True)      # nothing left to reverse
    penv = ptr.get("previous_env")
    if penv and ("/" in penv or "\\" in penv) and penv != ptr.get("env"):
        freed += safe_rmtree(Path(penv), app_envs_dir())
    ptr.update(previous=None, previous_env=None, return_to_previous=False)
    st._write_json(app / st.POINTER, ptr)
    return freed
