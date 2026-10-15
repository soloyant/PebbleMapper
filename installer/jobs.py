"""The install queue: one folder per job under ``~/.pebblemapper/jobs``.

The app writes ``request.json`` and appends the job to ``queue.json``; the
worker (``installer.worker``) runs jobs one at a time and rewrites
``job.json`` at least every two seconds. Files, not sockets: a second copy of
the app reads progress, and a stopped app finds the state at its next start
(contracts/worker-protocol.md).

States (data-model §5)::

    queued -> running -> done
       |         |-> cancelled -> (Resume) queued / (Remove)
       |         |-> paused    -> running
       |         |-> failed    -> (Resume) queued / (Remove)
       |         '-> interrupted (worker gone) -> (Resume) queued / (Remove)
       '-> cancelled

Only those arrows are allowed; anything else is a programming error.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from installer.paths import jobs_dir, write_json_atomic

ALLOWED = {
    "queued": {"running", "cancelled"},
    "running": {"done", "cancelled", "paused", "failed", "interrupted"},
    "paused": {"running", "cancelled", "interrupted"},
    "failed": {"queued"},
    "cancelled": {"queued"},
    "interrupted": {"queued"},
    "done": set(),
}
ACTIVE = ("queued", "running", "paused")
RESUMABLE = ("failed", "cancelled", "interrupted")

INSTALL_STEPS = [
    ("fetch", "Downloading the model"),
    ("verify", "Checking the downloads"),
    ("unpack", "Unpacking"),
    ("env-conda", "Installing the model's software"),
    ("env-pip", "Installing the model's Python packages"),
    ("weights", "Placing the model's weights"),
    ("check-run", "Testing the model on a sample photograph"),
    ("register", "Adding it to the Detection model list"),
    ("cleanup", "Cleaning up"),
]
APP_UPDATE_STEPS = [
    ("fetch", "Downloading the update"),
    ("verify", "Checking the downloads"),
    ("stage", "Preparing the new version"),
    ("env-conda", "Installing its software"),
    ("env-pip", "Installing its Python packages"),
    ("self-test", "Testing the new version"),
    ("ready", "Ready for the next start"),
]


class StateError(Exception):
    """A transition the state machine does not allow."""


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def _job_file(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def read_job(job_id: str) -> Optional[dict]:
    try:
        return json.loads(_job_file(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_job(job: dict) -> None:
    job["heartbeat"] = time.time()
    write_json_atomic(_job_file(job["id"]), job)


def all_jobs() -> List[dict]:
    root = jobs_dir()
    if not root.is_dir():
        return []
    out = [read_job(p.name) for p in root.iterdir() if (p / "job.json").is_file()]
    return sorted([j for j in out if j], key=lambda j: j.get("created", ""))


def set_state(job: dict, new: str, **extra) -> dict:
    old = job["state"]
    if new not in ALLOWED.get(old, set()):
        raise StateError(f"job {job['id']}: {old} -> {new} is not allowed")
    job["state"] = new
    job.update(extra)
    if new == "queued":
        job.pop("error", None)
    write_job(job)
    return job


# ------------------------------------------------------------- the queue
def _queue_file() -> Path:
    return jobs_dir() / "queue.json"


def _with_queue_lock(fn):
    lock = jobs_dir() / "queue.lock"
    jobs_dir().mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 10
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            # A crashed writer can leave the lock behind; it never lives long.
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink()
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("the install queue is locked")
            time.sleep(0.05)
    try:
        return fn()
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass


def queue() -> List[str]:
    try:
        return list(json.loads(_queue_file().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def enqueue(job_id: str) -> None:
    def _add():
        q = [j for j in queue() if j != job_id]
        q.append(job_id)
        write_json_atomic(_queue_file(), q)
    _with_queue_lock(_add)


def pop_next() -> Optional[str]:
    """The next queued job, removed from the queue; ``None`` when empty."""
    def _pop():
        q = queue()
        while q:
            jid = q.pop(0)
            job = read_job(jid)
            if job and job["state"] == "queued":
                write_json_atomic(_queue_file(), q)
                return jid
        write_json_atomic(_queue_file(), q)
        return None
    return _with_queue_lock(_pop)


def new_job(kind: str, request: dict) -> dict:
    """Create a job (``install`` / ``repair`` / ``uninstall``) and queue it."""
    jid = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = job_dir(jid)
    d.mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / "request.json", dict(request, kind=kind))
    table = {"install": INSTALL_STEPS, "repair": INSTALL_STEPS,
             "app-update": APP_UPDATE_STEPS}.get(kind)
    steps = ([{"name": n, "label": lbl, "status": "pending"} for n, lbl in table]
             if table else [{"name": "remove", "label": "Removing", "status": "pending"}])
    job = {"id": jid, "kind": kind, "state": "queued",
           "model": request.get("model"), "release": request.get("release"),
           "variant": request.get("variant"), "display_name": request.get("display_name"),
           "created": datetime.now().isoformat(timespec="seconds"),
           "steps": steps, "verified": [], "log": str(d / "job.log")}
    write_job(job)
    enqueue(jid)
    return job


def resume(job_id: str) -> dict:
    job = read_job(job_id)
    if job is None:
        raise StateError(f"no job {job_id}")
    for p in (job_dir(job_id) / "cancel",):
        if p.exists():
            p.unlink()
    set_state(job, "queued")
    enqueue(job_id)
    return job


def request_cancel(job_id: str) -> None:
    """Ask the worker to stop this job. A queued job is cancelled at once."""
    job = read_job(job_id)
    if job is None:
        return
    if job["state"] == "queued":
        set_state(job, "cancelled")
        return
    (job_dir(job_id) / "cancel").write_text("1", encoding="utf-8")


def cancel_requested(job_id: str) -> bool:
    return (job_dir(job_id) / "cancel").exists()


def forget(job_id: str) -> None:
    """Delete a finished or abandoned job's own folder (never the model's)."""
    import shutil
    d = job_dir(job_id)
    if d.is_dir() and not d.is_symlink():
        shutil.rmtree(d)
    _with_queue_lock(lambda: write_json_atomic(
        _queue_file(), [j for j in queue() if j != job_id]))


# ------------------------------------------------------------ the worker lock
class WorkerLock:
    """Held by the one running worker (FR-034). Another process that cannot
    take it knows a worker is alive."""

    def __init__(self):
        self.path = jobs_dir() / "worker.lock"
        self._fh = None

    def acquire(self) -> bool:
        jobs_dir().mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            _lock_file(fh)
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} {time.time()}")
        fh.flush()
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh:
            try:
                _unlock_file(self._fh)
            finally:
                self._fh.close()
                self._fh = None


def _lock_file(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def worker_alive() -> bool:
    lock = WorkerLock()
    if lock.acquire():
        lock.release()
        return False
    return True


def detect_interrupted() -> List[dict]:
    """At app start: a job left running or paused with no live worker was
    interrupted (the app closed, the lid shut, power lost)."""
    if worker_alive():
        return []
    out = []
    for job in all_jobs():
        if job["state"] in ("running", "paused"):
            set_state(job, "interrupted")
            out.append(job)
    return out


# ---------------------------------------------------------------- the log
class JobLog:
    """The job's log, with the home folder and account name replaced by ``~``
    (FR-039)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, line: str) -> None:
        from functions.feedback import _redact
        stamp = datetime.now().strftime("%H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {_redact(str(line))}\n")

    def tail(self, n: int = 20) -> List[str]:
        try:
            return self.path.read_text(encoding="utf-8").splitlines()[-n:]
        except OSError:
            return []
