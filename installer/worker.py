"""The install worker: ``python -m installer.worker``.

Started by the app when a job is queued and no worker runs (030 R10). It holds
the worker lock, runs queued jobs one at a time and exits when the queue is
empty. It never imports NiceGUI or ``gui``; a crash inside conda kills this
process, not the app, and the job is found ``interrupted`` at the next start.

Headless use, for the maintainer's rehearsals (quickstart A3)::

    python -m installer.worker --once --list <list.json or URL>
        --install orthosam [--release 1.0.0] --variant gpu --models E:\\pm-test-models
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

from installer import archive, envbuild, fetch, jobs, registration
from installer.errors import InstallError, human_size


# ------------------------------------------------------------------ helpers
def model_root(models_dir: Path, model_id: str, release: str, variant: str) -> Path:
    """Short folder names keep deep package paths under 259 characters (R7)."""
    return Path(models_dir) / f"{model_id}-{release.replace('.', '')}-{variant[0]}"


def _file_name(f: dict) -> str:
    return Path(urlparse(f["url"]).path).name


class JobRunner:
    """Runs one install or repair job, step by step, keeping ``job.json``
    current for the app to read."""

    def __init__(self, job: dict, request: dict):
        self.job = job
        self.req = request
        self.log = jobs.JobLog(Path(job["log"]))
        self.models_dir = Path(request["models_dir"])
        self.model = request["entry"]
        self.release = request["release_entry"]
        self.variant = request["variant"]
        self.vinfo = self.release["variants"][self.variant]
        self.root = model_root(self.models_dir, self.model["id"],
                               self.release["release"], self.variant)
        self.work = self.models_dir / ".work" / job["id"]
        self._last_write = 0.0

    # progress ------------------------------------------------------------
    def _step(self, name: str, status: str, **extra) -> None:
        for s in self.job["steps"]:
            if s["name"] == name:
                s["status"] = status
                s.update(extra)
                if status == "running":
                    s["started"] = time.time()
                elif status in ("done", "skipped"):
                    s["ended"] = time.time()
        self.job["current"] = name if status == "running" else self.job.get("current")
        jobs.write_job(self.job)

    def _progress(self, name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last_write < 1.0 and done < total:
            return
        self._last_write = now
        for s in self.job["steps"]:
            if s["name"] == name:
                s["bytes_done"], s["bytes_total"] = done, total
        jobs.write_job(self.job)

    def cancelled(self) -> bool:
        """Stop now: the user cancelled, or the battery fell below 15 %
        (checked at most every 30 s; FR-029). ``battery_low`` tells which."""
        if jobs.cancel_requested(self.job["id"]):
            return True
        now = time.monotonic()
        if now - getattr(self, "_power_checked", 0.0) > 30:
            self._power_checked = now
            from installer import sysinfo
            self.battery_low = sysinfo.battery_low(sysinfo.power())
        return bool(getattr(self, "battery_low", False))

    # the steps ----------------------------------------------------------
    def run(self) -> None:
        self.log(f"{self.job['kind']} {self.model['id']} {self.release['release']} "
                 f"({self.variant}) into {self.root}")
        self.check_disk()
        files = self.fetch()
        self._step("verify", "done")
        self.unpack(files)
        self.build_env(files)
        self.place_weights(files)
        entry = self.registration_entry(files)
        self.check_run(entry, files)
        self._step("register", "running")
        entry["size_bytes"] = _folder_bytes(self.root)
        registration.register(entry)
        self._step("register", "done")
        self._step("cleanup", "running")
        shutil.rmtree(self.work, ignore_errors=True)
        self._step("cleanup", "done")
        self.log(f"done: {human_size(entry['size_bytes'])} in {self.root}")

    def check_disk(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.models_dir).free
        need = int(self.vinfo.get("peak_bytes", 0))
        if need and free < need:
            raise InstallError("disk_full", drive=self.models_dir.drive or str(self.models_dir),
                               missing=human_size(need - free))

    def fetch(self) -> dict:
        self._step("fetch", "running")
        dl = self.work / "files"
        total = sum(f["size"] for f in self.vinfo["files"])
        done_before = 0
        out = {}
        for f in self.vinfo["files"]:
            name = _file_name(f)
            dest = dl / name

            def prog(d, t, base=done_before):
                self._progress("fetch", base + d, total)
            if f.get("dest") and not dest.exists():
                # Repair, or an update: a byte-identical copy already installed
                # (this release or another release of the model) is copied
                # rather than downloaded; download() verifies it once more.
                candidates = [self.root / f["dest"]] + [
                    Path(e["root"]) / f["dest"] for e in registration.read_installed()
                    if e.get("id") == self.model["id"]]
                for installed in candidates:
                    if fetch.verify_file(installed, f["size"], f["sha256"]):
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        archive.retry_io(shutil.copyfile, installed, dest)
                        self.log(f"reusing the installed {name}")
                        break
            fetch.download(f["url"], dest, f["size"], f["sha256"],
                           on_progress=prog, cancel=self.cancelled)
            done_before += f["size"]
            if name not in self.job["verified"]:
                self.job["verified"].append(name)
            out.setdefault(f["role"], []).append((f, dest))
            self.log(f"verified {name} ({human_size(f['size'])})")
        self._step("fetch", "done", bytes_done=total, bytes_total=total)
        return out

    def unpack(self, files: dict) -> None:
        self._step("unpack", "running")
        code = self.root / "code"
        if code.exists():
            shutil.rmtree(code)                  # a resumed job unpacks afresh
        for role in ("archive", "upstream"):
            for _, path in files.get(role, []):
                archive.safe_unzip(path, code)
        self._step("unpack", "done")

    def _env_whole(self) -> bool:
        """The installed environment holds its package records, Python and
        every module the plug-in requires (read from disk, as the start check
        does)."""
        env = self.root / "env"
        if not (env / "conda-meta").is_dir():
            return False
        if not ((env / "python.exe").is_file() or (env / "bin" / "python").is_file()):
            return False
        try:
            from installer import registration as _reg
            entry = next((e for e in _reg.read_installed()
                          if e["id"] == self.model["id"]
                          and e["release"] == self.release["release"]
                          and e["variant"] == self.variant), None)
            if entry is None:
                return False
            for _key, backend in _reg.load_entry(entry):
                if backend.missing_modules():
                    return False
        except Exception as ex:                 # cannot tell: rebuild
            self.log(f"repair: could not check the environment ({ex}); rebuilding it")
            return False
        return True

    def build_env(self, files: dict) -> None:
        if self.model.get("runs_in") == "app":
            self._step("env-conda", "skipped")
            self._step("env-pip", "skipped")
            return
        conda_lock = files["conda_lock"][0][1]
        pip_lock = files.get("pip_lock", [(None, None)])[0][1]
        status = {s["name"]: s.get("status") for s in self.job["steps"]}
        if (status.get("env-conda") == "done" and status.get("env-pip") in ("done", "skipped")
                and (self.root / "env" / "conda-meta").is_dir()):
            # Resumed after a later step failed: this environment finished
            # building; only a partial one is discarded (FR-032).
            self.log("the environment from the earlier attempt is complete; kept")
            return
        if self.job["kind"] == "repair" and self._env_whole():
            self.log("repair: the installed environment is whole; kept")
            self._step("env-conda", "skipped")
            self._step("env-pip", "skipped")
            return
        self._step("env-conda", "running")

        def on_step(name):
            if name == "env-pip":
                self._step("env-conda", "done")
                self._step("env-pip", "running")
        envbuild.create_env(self.root / "env", conda_lock, pip_lock, self.work,
                            log=self.log, cancel=self.cancelled, on_step=on_step,
                            find_links=(self.work / "files") if files.get("wheel") else None)
        self._step("env-conda", "done")
        self._step("env-pip", "done")

    def place_weights(self, files: dict) -> None:
        self._step("weights", "running")
        for f, path in files.get("weights", []):
            dest = self.root / f.get("dest", f"weights/{path.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            archive.retry_io(shutil.copyfile, path, dest)
        (self.root / "weights").mkdir(parents=True, exist_ok=True)
        self._step("weights", "done")

    def registration_entry(self, files: dict) -> dict:
        recorded = []
        for f, path in files.get("weights", []):
            rel = f.get("dest", f"weights/{path.name}")
            recorded.append({"path": rel, "size": f["size"], "sha256": f["sha256"]})
        return {
            "id": self.model["id"], "release": self.release["release"],
            "variant": self.variant, "root": str(self.root),
            "env": str(self.root / "env"), "code": str(self.root / "code"),
            "weights": str(self.root / "weights"),
            "backends": [{k: b[k] for k in ("name", "display_name", "package",
                                            "module", "factory") if k in b}
                         for b in self.model["backends"]],
            "files": recorded, "runs_in": self.model.get("runs_in", "own"),
            "citation": self.model.get("citation", ""),
            "installed": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": "app",
            "state": "ok"}

    def check_run(self, entry: dict, files: dict) -> None:
        """The plug-in's own run.py on the release's check photograph, through
        the real detection path (R13); at least one outline is required."""
        self._step("check-run", "running")
        images = files.get("check_image", [])
        if not images:
            raise InstallError("check_run_failed")
        out_dir = self.work / "check"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            pairs = registration.load_entry(entry)
        except registration.LoadError as ex:
            self.log(f"check run: the model could not be loaded: {ex}")
            raise InstallError("check_run_failed") from ex
        timeout = int(self.vinfo.get("check_timeout_s", 900))
        for key, backend in pairs:
            if not backend.is_available():
                self.log(f"check run: {key} reports itself unavailable after install")
                raise InstallError("check_run_failed")
            self.log(f"check run: {key} on {images[0][1].name}")
            try:
                frames = backend.detect_jobs(
                    "quadrat", [{"path": str(images[0][1])}], resolution=0.001,
                    output_dir=str(out_dir), log_fn=self.log, timeout=timeout,
                    stop_check=self.cancelled,
                    devicemode="CPU" if self.variant == "cpu" else "GPU")
            except Exception as ex:
                if self.cancelled():
                    raise fetch.Cancelled() from ex
                self.log(f"check run failed: {type(ex).__name__}: {ex}")
                raise InstallError("check_run_failed") from ex
            if self.cancelled():
                # A stopped detection returns empty, not an error: this was
                # the user's Cancel (or low battery), not a failed check.
                raise fetch.Cancelled()
            n = sum(len(f) for f in frames if f is not None)
            self.log(f"check run: {n} outline(s)")
            if n < 1:
                raise InstallError("check_run_failed")
        self._step("check-run", "done")


def _folder_bytes(folder: Path) -> int:
    total = 0
    for p in Path(folder).rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


# ------------------------------------------------------------------ the loop
def run_job(job_id: str) -> None:
    job = jobs.read_job(job_id)
    if job is None or job["state"] != "queued":
        return
    req = json.loads((jobs.job_dir(job_id) / "request.json").read_text(encoding="utf-8"))
    jobs.set_state(job, "running", started=time.time())
    log = jobs.JobLog(Path(job["log"]))
    try:
        if job["kind"] in ("install", "repair"):
            while True:
                runner = JobRunner(jobs.read_job(job_id) or job, req)
                try:
                    runner.run()
                    break
                except fetch.Cancelled:
                    if jobs.cancel_requested(job_id) or not getattr(runner, "battery_low", False):
                        raise
                    if not _pause_for_battery(job_id, log):
                        raise
        elif job["kind"] == "uninstall":
            from installer import removal
            removal.run_uninstall_job(job, req, log)
        elif job["kind"] == "app-update":
            from installer import appupdate
            appupdate.AppUpdateRunner(jobs.read_job(job_id) or job, req,
                                      Path(req.get("app_dir") or appupdate.APP_DIR)).run()
        else:
            raise InstallError("unknown", detail=f"job kind {job['kind']}")
        jobs.set_state(jobs.read_job(job_id) or job, "done", ended=time.time())
    except fetch.Cancelled:
        log("cancelled by the user")
        jobs.set_state(jobs.read_job(job_id) or job, "cancelled")
    except InstallError as ex:
        log(f"failed: {ex.cls}: {ex.sentence()}")
        jobs.set_state(jobs.read_job(job_id) or job, "failed", error=ex.as_dict())
    except Exception as ex:                       # reported, never swallowed
        log("failed: " + traceback.format_exc())
        err = InstallError("unknown", detail=f"{type(ex).__name__}: {ex}")
        jobs.set_state(jobs.read_job(job_id) or job, "failed", error=err.as_dict())


PAUSE_POLL_S = 30


def _pause_for_battery(job_id: str, log) -> bool:
    """Battery below 15 %: hold the job ``paused`` until the computer is
    plugged in (True: carry on) or the user cancels (False). What was already
    downloaded and checked is kept; a partial build is redone (FR-032)."""
    from installer import sysinfo
    log("paused: battery low; waiting for the computer to be plugged in")
    jobs.set_state(jobs.read_job(job_id), "paused")
    while True:
        if jobs.cancel_requested(job_id):
            jobs.set_state(jobs.read_job(job_id), "running")   # paused -> cancelled
            return False                                       # goes via running
        if not sysinfo.battery_low(sysinfo.power()):
            log("power is back; continuing")
            jobs.set_state(jobs.read_job(job_id), "running")
            return True
        time.sleep(PAUSE_POLL_S)


INSTALLING_LINE = "PebbleMapper is installing — closing this window will pause it"


def _mark_console(active: bool) -> None:
    """The launcher window's title and one line in it (US2): closing it
    interrupts the install, which can be resumed."""
    if os.name != "nt":
        return
    try:
        import ctypes
        if active:
            ctypes.windll.kernel32.SetConsoleTitleW(INSTALLING_LINE)
            print(f"\n*** {INSTALLING_LINE} ***\n", flush=True)
        else:
            ctypes.windll.kernel32.SetConsoleTitleW("PebbleMapper")
    except Exception:
        pass


def drain() -> int:
    """Run queued jobs until the queue is empty; 0 if this process did the
    work, 1 if another worker already holds the lock."""
    from installer.power import KeepAwake
    lock = jobs.WorkerLock()
    if not lock.acquire():
        return 1
    _mark_console(True)
    try:
        with KeepAwake("PebbleMapper is installing a model"):
            while True:
                jid = jobs.pop_next()
                if jid is None:
                    return 0
                run_job(jid)
    finally:
        _mark_console(False)
        lock.release()


def _load_list(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        from installer import catalog
        return catalog.fetch_list(src).data
    return json.loads(Path(src).read_text(encoding="utf-8"))


def request_for(list_data: dict, model_id: str, variant: str, models_dir: str,
                release: str = None) -> dict:
    model = next((m for m in list_data["models"] if m["id"] == model_id), None)
    if model is None:
        raise SystemExit(f"{model_id} is not in the list")
    rels = [r for r in model["releases"] if not r.get("withdrawn")
            and (release is None or r["release"] == release)]
    if not rels:
        raise SystemExit(f"no installable release of {model_id}")
    rel = sorted(rels, key=lambda r: tuple(int(x) for x in r["release"].split(".")))[-1]
    if variant not in rel["variants"]:
        raise SystemExit(f"{model_id} {rel['release']} has no {variant} version")
    return {"model": model_id, "release": rel["release"], "variant": variant,
            "display_name": model["backends"][0]["display_name"],
            "models_dir": str(models_dir), "entry": model, "release_entry": rel}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PebbleMapper model install worker")
    ap.add_argument("--once", action="store_true", help="drain the queue and exit")
    ap.add_argument("--list")
    ap.add_argument("--install")
    ap.add_argument("--release")
    ap.add_argument("--variant", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--models")
    args = ap.parse_args(argv)
    if args.install:
        if not (args.list and args.models):
            ap.error("--install needs --list and --models")
        req = request_for(_load_list(args.list), args.install, args.variant,
                          args.models, args.release)
        job = jobs.new_job("install", req)
        print(f"queued {job['id']}", flush=True)
    rc = drain()
    if args.install:
        job = jobs.read_job(job["id"])
        print(json.dumps({k: job.get(k) for k in ("id", "state", "error")}, indent=1))
        return 0 if job["state"] == "done" else 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
