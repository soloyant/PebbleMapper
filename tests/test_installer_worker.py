"""030 T025-T026: the worker runs the install steps, registers only after
success, and resumes without re-downloading verified files."""
import hashlib
import http.server
import io
import json
import threading
import zipfile

import pytest

from installer import envbuild, fetch, jobs, registration, worker
from installer.errors import InstallError


def _zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


ARCHIVE = _zip_bytes({"pm_t_backend/__init__.py": "", "pm_t_backend/backend.py": "x = 1\n"})
FILES = {"/t-1.0.0.zip": ARCHIVE, "/conda-gpu.lock.txt": b"@EXPLICIT\n",
         "/pip-gpu.lock.txt": b"# no pip packages\n", "/check.jpg": b"jpg", "/w.pth": b"w" * 1000}


class _H(http.server.BaseHTTPRequestHandler):
    hits = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        body = FILES.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(fetch, "_ALLOW_LOCAL_HTTP", True)
    _H.hits = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    roles = {"/t-1.0.0.zip": "archive", "/conda-gpu.lock.txt": "conda_lock",
             "/pip-gpu.lock.txt": "pip_lock", "/check.jpg": "check_image",
             "/w.pth": "weights"}
    files = [dict({"role": r, "url": base + p, "size": len(FILES[p]),
                   "sha256": hashlib.sha256(FILES[p]).hexdigest()},
                  **({"dest": "weights/w.pth"} if r == "weights" else {}))
             for p, r in roles.items()]
    lst = {"format": 1, "list_version": 1, "published": "2026-10-13", "models": [{
        "id": "t", "runs_in": "own", "citation": "c",
        "backends": [{"name": "t", "display_name": "T", "package": "pm_t_backend",
                      "module": "backend", "factory": "make_backend"}],
        "releases": [{"release": "1.0.0", "commit": "0" * 40, "withdrawn": False,
                      "variants": {"gpu": {"files": files, "download_bytes": 1,
                                           "disk_bytes": 1, "peak_bytes": 1,
                                           "longest_path": 50, "minutes": [1, 2],
                                           "check_timeout_s": 60}}}]}]}
    yield tmp_path, lst
    srv.shutdown()


def _fake_env(prefix, conda_lock, pip_lock, work, log, cancel=None, on_step=None, **kw):
    (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)


def test_an_install_registers_only_after_every_step(env, monkeypatch):
    tmp_path, lst = env
    monkeypatch.setattr(envbuild, "create_env", _fake_env)
    seen = {}

    def fake_check(self, entry, files):
        seen["registered_before_check"] = bool(registration.read_installed())
        self._step("check-run", "done")
    monkeypatch.setattr(worker.JobRunner, "check_run", fake_check)
    req = worker.request_for(lst, "t", "gpu", tmp_path / "models")
    job = jobs.new_job("install", req)
    assert worker.drain() == 0
    job = jobs.read_job(job["id"])
    assert job["state"] == "done", job.get("error")
    assert seen["registered_before_check"] is False
    [entry] = registration.read_installed()
    root = tmp_path / "models" / "t-100-g"
    assert entry["root"] == str(root)
    assert (root / "code" / "pm_t_backend" / "backend.py").is_file()
    assert (root / "weights" / "w.pth").read_bytes() == FILES["/w.pth"]
    assert entry["files"][0]["path"] == "weights/w.pth"
    assert not (tmp_path / "models" / ".work" / job["id"]).exists(), "work folder cleaned"
    assert all(s["status"] in ("done", "skipped") for s in job["steps"])


def test_a_failed_job_is_not_registered_and_resume_reuses_downloads(env, monkeypatch):
    tmp_path, lst = env
    calls = []

    def flaky_env(prefix, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise InstallError("network", host="conda.anaconda.org")
        _fake_env(prefix, *a, **k)
    monkeypatch.setattr(envbuild, "create_env", flaky_env)
    monkeypatch.setattr(worker.JobRunner, "check_run",
                        lambda self, e, f: self._step("check-run", "done"))
    job = jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    j = jobs.read_job(job["id"])
    assert j["state"] == "failed"
    assert j["error"]["class"] == "network" and "conda.anaconda.org" in j["error"]["action"]
    assert registration.read_installed() == []
    hits_after_first = len(_H.hits)
    jobs.resume(job["id"])
    worker.drain()
    assert jobs.read_job(job["id"])["state"] == "done"
    assert len(_H.hits) == hits_after_first, "verified files are not downloaded again"


def test_cancel_during_the_build_leaves_nothing_registered(env, monkeypatch):
    tmp_path, lst = env

    def cancelled_env(*a, **k):
        raise fetch.Cancelled()
    monkeypatch.setattr(envbuild, "create_env", cancelled_env)
    job = jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    assert jobs.read_job(job["id"])["state"] == "cancelled"
    assert registration.read_installed() == []


def test_not_enough_disk_is_refused_before_downloading(env, monkeypatch):
    tmp_path, lst = env
    lst["models"][0]["releases"][0]["variants"]["gpu"]["peak_bytes"] = 10 ** 18
    job = jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    j = jobs.read_job(job["id"])
    assert j["error"]["class"] == "disk_full"
    assert _H.hits == []


def test_short_folder_names():
    from pathlib import Path
    assert worker.model_root(Path("D:/M"), "orthosam", "1.0.0", "gpu") == Path("D:/M/orthosam-100-g")
    assert worker.model_root(Path("D:/M"), "orthosam", "1.2.10", "cpu") == Path("D:/M/orthosam-1210-c")


def test_resume_keeps_a_finished_environment_and_rebuilds_a_partial_one(env, monkeypatch):
    tmp_path, lst = env
    builds = []

    def counting_env(prefix, *a, **k):
        builds.append(1)
        _fake_env(prefix, *a, **k) if not prefix.exists() else None
    monkeypatch.setattr(envbuild, "create_env", counting_env)
    checks = []

    def flaky_check(self, entry, files):
        checks.append(1)
        if len(checks) == 1:
            raise InstallError("check_run_failed")
        self._step("check-run", "done")
    monkeypatch.setattr(worker.JobRunner, "check_run", flaky_check)
    job = jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    assert jobs.read_job(job["id"])["state"] == "failed"
    jobs.resume(job["id"])
    worker.drain()
    assert jobs.read_job(job["id"])["state"] == "done"
    assert builds == [1], "a complete environment is not built twice"


def test_low_battery_pauses_then_continues_when_plugged_in(env, monkeypatch):
    from installer import sysinfo
    tmp_path, lst = env
    monkeypatch.setattr(worker, "PAUSE_POLL_S", 0)
    # Low on the first look (during the build), plugged in on the next.
    readings = iter([True, False, False, False, False])
    monkeypatch.setattr(sysinfo, "power", lambda: None)
    monkeypatch.setattr(sysinfo, "battery_low", lambda p: next(readings, False))
    builds = []

    def env_checks_battery(prefix, conda_lock, pip_lock, work, log, cancel=None, on_step=None, **kw):
        builds.append(1)
        if cancel and cancel():
            raise fetch.Cancelled()
        _fake_env(prefix, conda_lock, pip_lock, work, log)
    monkeypatch.setattr(envbuild, "create_env", env_checks_battery)
    monkeypatch.setattr(worker.JobRunner, "check_run",
                        lambda self, e, f: self._step("check-run", "done"))
    states = []
    real_set = jobs.set_state
    monkeypatch.setattr(jobs, "set_state",
                        lambda job, new, **kw: (states.append(new), real_set(job, new, **kw))[1])
    job = jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    assert jobs.read_job(job["id"])["state"] == "done"
    assert states == ["running", "paused", "running", "done"]
    assert len(builds) >= 1          # the low reading may land in the download or the build


def test_repair_reuses_intact_weights_and_replaces_damaged_ones(env, monkeypatch):
    tmp_path, lst = env
    monkeypatch.setattr(envbuild, "create_env", _fake_env)
    monkeypatch.setattr(worker.JobRunner, "check_run",
                        lambda self, e, f: self._step("check-run", "done"))
    req = worker.request_for(lst, "t", "gpu", tmp_path / "models")
    jobs.new_job("install", req)
    worker.drain()
    root = tmp_path / "models" / "t-100-g"
    # Intact: a repair downloads nothing it can take from the installed copy.
    _H.hits.clear()
    jobs.new_job("repair", req)
    worker.drain()
    assert "/w.pth" not in _H.hits
    # Damaged: the weights file is fetched again and restored.
    (root / "weights" / "w.pth").write_bytes(b"broken")
    _H.hits.clear()
    job = jobs.new_job("repair", req)
    worker.drain()
    assert jobs.read_job(job["id"])["state"] == "done", jobs.read_job(job["id"]).get("error")
    assert "/w.pth" in _H.hits
    assert (root / "weights" / "w.pth").read_bytes() == FILES["/w.pth"]
    assert len(registration.read_installed()) == 1


def test_cancel_during_the_check_run_is_a_cancel_not_a_failure(env, monkeypatch):
    tmp_path, lst = env
    monkeypatch.setattr(envbuild, "create_env", _fake_env)

    class _Stopped:
        def is_available(self):
            return True

        def detect_jobs(self, *a, **k):
            jobs.request_cancel(k["stop_check"].__self__.job["id"])
            return []                           # a stopped run returns empty
    monkeypatch.setattr(registration, "load_entry", lambda e: [("t@1.0.0", _Stopped())])
    job = jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    j = jobs.read_job(job["id"])
    assert j["state"] == "cancelled", j.get("error")
    assert registration.read_installed() == []


def test_an_update_reuses_identical_weights_from_the_installed_release(env, monkeypatch):
    tmp_path, lst = env
    monkeypatch.setattr(envbuild, "create_env", _fake_env)
    monkeypatch.setattr(worker.JobRunner, "check_run",
                        lambda self, e, f: self._step("check-run", "done"))
    jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models"))
    worker.drain()
    import copy as _copy
    newer = _copy.deepcopy(lst["models"][0]["releases"][0])
    newer["release"] = "1.0.1"
    lst["models"][0]["releases"].append(newer)
    _H.hits.clear()
    jobs.new_job("install", worker.request_for(lst, "t", "gpu", tmp_path / "models", "1.0.1"))
    worker.drain()
    assert "/w.pth" not in _H.hits, "the weights came from the installed 1.0.0"
    assert (tmp_path / "models" / "t-101-g" / "weights" / "w.pth").read_bytes() == FILES["/w.pth"]
    assert sorted(e["release"] for e in registration.read_installed()) == ["1.0.0", "1.0.1"]
