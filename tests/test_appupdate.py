"""031 T020: the app-update job prepares the new version without touching the
running one, and records it for the starter only once every check passed."""
import hashlib
import http.server
import io
import json
import shutil
import threading
import zipfile
from pathlib import Path

import pytest

from installer import appupdate, envbuild, fetch, jobs, worker
from installer.errors import InstallError

REPO = Path(__file__).resolve().parents[1]
NEW = {"gui/app.py": b"v2 app", "functions/__init__.py": b'__version__ = "1.1.0"\n',
       "tools/check_install.py": b"print('READY')\n"}


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _list(entries):
    return json.dumps({"release": "1.1.0", "files": [
        {"path": k, "size": len(v), "sha256": hashlib.sha256(v).hexdigest()}
        for k, v in entries.items()]}).encode()


class _H(http.server.BaseHTTPRequestHandler):
    files, hits = {}, []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(fetch, "_ALLOW_LOCAL_HTTP", True)
    app = tmp_path / "app"
    (app / "gui").mkdir(parents=True)
    (app / "gui" / "app.py").write_bytes(b"v1 app")
    shutil.copy(REPO / "pm_starter.py", app / "pm_starter.py")
    (app / "PebbleMapper.cmd").write_text("@echo off\n")
    _H.files = {"/pebblemapper-1.1.0.zip": _zip(NEW), "/files-1.1.0.json": _list(NEW),
                "/app-conda-gpu.lock.txt": b"@EXPLICIT\n", "/app-pip-gpu.lock.txt": b"# none\n"}
    _H.hits = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    roles = {"/pebblemapper-1.1.0.zip": "archive", "/files-1.1.0.json": "file_list",
             "/app-conda-gpu.lock.txt": "conda_lock", "/app-pip-gpu.lock.txt": "pip_lock"}

    def files():
        return [{"role": r, "url": base + p, "size": len(_H.files[p]),
                 "sha256": hashlib.sha256(_H.files[p]).hexdigest()} for p, r in roles.items()]
    data = {"app_releases": [{"release": "1.1.0", "commit": "0" * 40, "min_starter": 1,
                              "withdrawn": False, "variants": {"gpu": {
                                  "files": files(), "disk_bytes": 1}}}]}
    built = []

    def fake_env(prefix, *a, **k):
        built.append(prefix)
        (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(envbuild, "create_env", fake_env)
    monkeypatch.setattr(appupdate.AppUpdateRunner, "self_test",
                        lambda self: self._step("self-test", "done"))
    yield app, data, built
    srv.shutdown()


def _run(app, data):
    req = appupdate.request_for_app(data, "1.1.0", "gpu")
    req["app_dir"] = str(app)
    job = jobs.new_job("app-update", req)
    worker.drain()
    return jobs.read_job(job["id"])


def _pointer(app):
    p = app / ".pm-version.json"
    return json.loads(p.read_text()) if p.exists() else {}


def test_an_update_is_staged_and_the_running_version_is_untouched(setup):
    app, data, built = setup
    job = _run(app, data)
    assert job["state"] == "done", job.get("error")
    assert (app / "gui" / "app.py").read_bytes() == b"v1 app", "FR-011"
    stage = app / ".pm-update" / "stage"
    assert (stage / "gui" / "app.py").read_bytes() == b"v2 app"
    staged = _pointer(app)["staged"]
    assert staged["release"] == "1.1.0" and Path(staged["stage_dir"]) == stage
    assert Path(staged["env"]) == built[0] and built[0].parent.name == "app-envs"
    assert Path(staged["file_list"]).is_file()
    assert not (app / ".pm-update" / "work" / job["id"]).exists()
    assert all(s["status"] == "done" for s in job["steps"])


def test_a_release_that_does_not_match_its_file_list_is_not_staged(setup):
    app, data, built = setup
    _H.files["/files-1.1.0.json"] = _list(dict(NEW, **{"gui/app.py": b"something else"}))
    data["app_releases"][0]["variants"]["gpu"]["files"][1].update(
        size=len(_H.files["/files-1.1.0.json"]),
        sha256=hashlib.sha256(_H.files["/files-1.1.0.json"]).hexdigest())
    job = _run(app, data)
    assert job["state"] == "failed" and job["error"]["class"] == "published_changed"
    assert "staged" not in _pointer(app) and built == []


def test_a_path_leaving_the_app_folder_is_refused(setup):
    app, data, built = setup
    _H.files["/files-1.1.0.json"] = _list(dict(NEW, **{"../evil.py": b"x"}))
    data["app_releases"][0]["variants"]["gpu"]["files"][1].update(
        size=len(_H.files["/files-1.1.0.json"]),
        sha256=hashlib.sha256(_H.files["/files-1.1.0.json"]).hexdigest())
    job = _run(app, data)
    assert job["error"]["class"] == "published_changed"
    assert not (app.parent / "evil.py").exists()


def test_a_newer_starter_requirement_stops_before_downloading(setup):
    app, data, built = setup
    data["app_releases"][0]["min_starter"] = 99
    job = _run(app, data)
    assert job["error"]["class"] == "starter_too_old"
    assert _H.hits == []


def test_a_failed_self_test_leaves_nothing_staged(setup, monkeypatch):
    app, data, built = setup

    def failing(self):
        raise InstallError("check_run_failed")
    monkeypatch.setattr(appupdate.AppUpdateRunner, "self_test", failing)
    job = _run(app, data)
    assert job["state"] == "failed" and "staged" not in _pointer(app)


def _switch_to_staged(app):
    """What the starter does at the next start, and its healthy start."""
    st = appupdate._starter(app)
    ptr = st.apply_staged(app, st.load_pointer(app))
    ptr["provisional"] = False
    ptr["last_switch"]["result"] = "ok"
    st._write_json(app / st.POINTER, ptr)
    return st


def test_switch_then_return_then_remove(setup):
    app, data, built = setup
    (app / ".pm-files.json").write_text(json.dumps({"files": [
        {"path": "gui/app.py", "size": 6, "sha256": hashlib.sha256(b"v1 app").hexdigest()}]}))
    assert appupdate.previous_version(app) is None
    with pytest.raises(InstallError):
        appupdate.request_restart_to_previous(app)
    _run(app, data)
    st = _switch_to_staged(app)
    assert (app / "gui" / "app.py").read_bytes() == b"v2 app"
    assert appupdate.previous_version(app) == {"release": "unknown", "return_requested": False,
                                               "last_result": "ok"}
    assert appupdate.can_return(app)
    appupdate.request_restart_to_previous(app)
    appupdate.cancel_return(app)
    assert _pointer(app)["return_to_previous"] is False
    appupdate.request_restart_to_previous(app)
    # the starter applies the return at its next start:
    ptr = st.roll_back(app, st.load_pointer(app), "returned")
    st._write_json(app / st.POINTER, ptr)
    assert (app / "gui" / "app.py").read_bytes() == b"v1 app"
    assert appupdate.unseen_switch(app)["result"] == "returned"
    appupdate.mark_switch_seen(app)
    assert appupdate.unseen_switch(app) is None
    # After a return, the version left behind can be removed but not returned to.
    assert not appupdate.can_return(app) and appupdate.previous_version(app)
    (built[0] / "big").write_bytes(b"x" * 1000)
    freed = appupdate.remove_previous(app)
    assert freed >= 1000 and not built[0].exists(), "the 1.1.0 environment is freed"
    assert not (app / ".pm-update" / "stage").exists()
    assert appupdate.previous_version(app) is None
    assert (app / "gui" / "app.py").read_bytes() == b"v1 app"


def test_the_banner_shows_failures_and_kept_files_once(setup):
    app, data, built = setup
    st = appupdate._starter(app)
    for ls, shown in (({"result": "applied"}, False), ({"result": "ok"}, False),
                      ({"result": "ok", "kept_mine": ["a.csv.mine"]}, True),
                      ({"result": "rolled_back"}, True), ({"result": "damaged"}, True)):
        ptr = st.load_pointer(app)
        ptr["last_switch"] = dict(ls, to="1.1.0")
        st._write_json(app / st.POINTER, ptr)
        assert bool(appupdate.unseen_switch(app)) is shown, ls
        appupdate.mark_switch_seen(app)
        assert appupdate.unseen_switch(app) is None


def test_only_a_copy_started_by_the_starter_updates_itself(setup, monkeypatch):
    app, data, built = setup
    monkeypatch.delenv("PEBBLEMAPPER_STARTER", raising=False)
    assert "shortcut" in appupdate.in_place_refusal(app)
    monkeypatch.setenv("PEBBLEMAPPER_STARTER", "1")
    assert appupdate.in_place_refusal(app) is None
    (app / ".git").write_text("gitdir: elsewhere")
    assert "development copy" in appupdate.in_place_refusal(app)


def test_the_app_and_the_starter_agree_on_the_restart_code():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pm_starter_rc", REPO / "pm_starter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RESTART_CODE == appupdate.RESTART_CODE


def test_staged_release_reads_the_pointer(setup):
    app, data, built = setup
    assert appupdate.staged_release(app) is None
    _run(app, data)
    assert appupdate.staged_release(app) == "1.1.0"


def test_a_second_preparation_reuses_the_tested_environment(setup):
    app, data, built = setup
    _run(app, data)
    shutil.rmtree(app / ".pm-update" / "stage")        # e.g. discarded as damaged
    job = _run(app, data)
    assert job["state"] == "done" and len(built) == 1, "built once, reused once"
    assert (app / ".pm-update" / "stage" / "gui" / "app.py").read_bytes() == b"v2 app"
    # Different locks: built again.
    _H.files["/app-conda-gpu.lock.txt"] = b"@EXPLICIT\n# other\n"
    f = data["app_releases"][0]["variants"]["gpu"]["files"][2]
    f.update(size=len(_H.files["/app-conda-gpu.lock.txt"]),
             sha256=hashlib.sha256(_H.files["/app-conda-gpu.lock.txt"]).hexdigest())
    _run(app, data)
    assert len(built) == 2


def test_the_rollback_banner_names_the_failed_and_the_restored_version(setup):
    """The rehearsal showed them swapped: the starter records a rollback as a
    switch from the failed version to the one put back."""
    from gui.models_tab import _SWITCH_TEXT
    app, data, built = setup
    st = appupdate._starter(app)
    ptr = dict(st.fresh_pointer("1.1.1"), previous="1.1.0", previous_env="x", provisional=True)
    ls = st.roll_back(app, ptr, "rolled_back")["last_switch"]
    text = _SWITCH_TEXT["rolled_back"].format(detail="", **ls)
    assert text.startswith("PebbleMapper 1.1.1 did not start, so 1.1.0 was put back")
