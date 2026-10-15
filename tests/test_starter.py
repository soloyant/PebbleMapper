"""031 FR-012–FR-016, SC-007, SC-009: the starter switches only the files the
releases own, keeps what the user changed, and can always go back."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "pm_starter", Path(__file__).resolve().parents[1] / "pm_starter.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def _entry(rel, data):
    return {"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _tree(root, files):
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


OLD = {"gui/app.py": b"v1 app", "functions/__init__.py": b'__version__ = "1.0.0"\n',
       "datasets/example/a.csv": b"example v1", "docs/old.md": b"gone in v2"}
NEW = {"gui/app.py": b"v2 app", "functions/__init__.py": b'__version__ = "1.1.0"\n',
       "datasets/example/a.csv": b"example v2", "gui/new.py": b"new in v2"}
USER = {"datasets/MyBeach/vectors/x.csv": b"my results",
        "model_weights/mask_rcnn_clasts.h5": b"weights", "user_detectors.json": b"[]"}


@pytest.fixture
def app(tmp_path):
    app = tmp_path / "app"
    _tree(app, OLD)
    _tree(app, USER)
    stage = tmp_path / "stage"
    _tree(stage, NEW)
    old_list = {"files": [_entry(k, v) for k, v in OLD.items()]}
    new_list = {"files": [_entry(k, v) for k, v in NEW.items()]}
    return app, stage, old_list, new_list


def _snapshot(root):
    return {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes()
            for p in root.rglob("*") if p.is_file() and ".pm-update" not in p.parts
            and p.name != ".pm-files.json"}   # compared as JSON where it matters


def test_the_switch_moves_only_release_files(app):
    app_dir, stage, old_list, new_list = app
    res = st.switch(app_dir, stage, old_list, new_list)
    after = _snapshot(app_dir)
    for rel, data in NEW.items():
        assert after[rel] == data
    for rel, data in USER.items():
        assert after[rel] == data, f"{rel} must not change (SC-007)"
    assert "docs/old.md" not in after, "a file the new release dropped leaves"
    assert (app_dir / ".pm-update" / "previous" / "gui" / "app.py").read_bytes() == b"v1 app"
    assert res["kept_mine"] == []


def test_a_shipped_file_the_user_changed_is_kept_as_mine(app):
    app_dir, stage, old_list, new_list = app
    (app_dir / "datasets/example/a.csv").write_bytes(b"edited by the user")
    res = st.switch(app_dir, stage, old_list, new_list)
    assert (app_dir / "datasets/example/a.csv").read_bytes() == b"example v2"
    assert (app_dir / "datasets/example/a.csv.mine").read_bytes() == b"edited by the user"
    assert res["kept_mine"] == [str(Path("datasets/example/a.csv.mine"))]


def test_reverse_restores_the_previous_tree_exactly(app):
    app_dir, stage, old_list, new_list = app
    (app_dir / "datasets/example/a.csv").write_bytes(b"edited by the user")
    before = _snapshot(app_dir)
    res = st.switch(app_dir, stage, old_list, new_list)
    st.reverse(res["journal"])
    assert _snapshot(app_dir) == before


def test_a_failure_part_way_leaves_the_old_version_whole(app, monkeypatch):
    app_dir, stage, old_list, new_list = app
    before = _snapshot(app_dir)
    real = st._rename
    calls = {"n": 0}

    def flaky(src, dst, tries=6):
        calls["n"] += 1
        if calls["n"] == 5:
            raise PermissionError(13, "locked", str(src))
        real(src, dst, tries=1)
    monkeypatch.setattr(st, "_rename", flaky)
    with pytest.raises(PermissionError):
        st.switch(app_dir, stage, old_list, new_list)
    monkeypatch.setattr(st, "_rename", real)
    assert _snapshot(app_dir) == before


def test_a_damaged_stage_is_detected(app):
    app_dir, stage, old_list, new_list = app
    (stage / "gui" / "new.py").write_bytes(b"tampered")
    assert st.verify_stage(stage, new_list) == ["gui/new.py"]


def test_a_path_leaving_the_app_folder_is_refused(app):
    app_dir, stage, old_list, new_list = app
    new_list["files"].append({"path": "../outside.txt", "size": 1, "sha256": "0" * 64})
    with pytest.raises(ValueError):
        st.switch(app_dir, stage, old_list, new_list)


def test_apply_staged_then_roll_back(app, tmp_path):
    app_dir, stage, old_list, new_list = app
    (app_dir / st.FILES).write_text(json.dumps(old_list))
    nl = tmp_path / "files-1.1.0.json"
    nl.write_text(json.dumps(new_list))
    before = _snapshot(app_dir)
    ptr = st.load_pointer(app_dir)
    assert ptr["current"] == "1.0.0" and ptr["env"] == "maskrcnn"
    ptr["staged"] = {"release": "1.1.0", "env": "C:/envs/1.1.0", "stage_dir": str(stage),
                     "file_list": str(nl)}
    ptr = st.apply_staged(app_dir, ptr)
    assert ptr["current"] == "1.1.0" and ptr["provisional"] and ptr["previous"] == "1.0.0"
    assert (app_dir / "gui/app.py").read_bytes() == b"v2 app"
    ptr = st.roll_back(app_dir, ptr, "rolled_back")
    assert ptr["current"] == "1.0.0" and ptr["env"] == "maskrcnn"
    assert ptr["last_switch"]["result"] == "rolled_back"
    assert _snapshot(app_dir) == before
    assert json.loads((app_dir / st.FILES).read_text()) == old_list


def test_a_damaged_stage_starts_the_current_version(app, tmp_path):
    app_dir, stage, old_list, new_list = app
    (stage / "gui" / "app.py").write_bytes(b"tampered!")
    nl = tmp_path / "f.json"
    nl.write_text(json.dumps(new_list))
    before = _snapshot(app_dir)
    ptr = st.load_pointer(app_dir)
    ptr["staged"] = {"release": "1.1.0", "env": "x", "stage_dir": str(stage), "file_list": str(nl)}
    ptr = st.apply_staged(app_dir, ptr)
    assert ptr["last_switch"]["result"] == "damaged" and ptr["current"] == "1.0.0"
    assert _snapshot(app_dir) == before


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_a_junction_in_the_app_folder_is_not_followed(app, tmp_path):
    app_dir, stage, old_list, new_list = app
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("field data")
    subprocess.run(["cmd", "/c", "mklink", "/J", str(app_dir / "datasets" / "linked"),
                    str(outside)], check=True, capture_output=True)
    try:
        st.switch(app_dir, stage, old_list, new_list)
        assert (outside / "keep.txt").read_text() == "field data"
        assert list(outside.iterdir()) == [outside / "keep.txt"]
    finally:
        os.rmdir(app_dir / "datasets" / "linked")


def test_the_starter_imports_nothing_from_the_app():
    src = (Path(__file__).resolve().parents[1] / "pm_starter.py").read_text(encoding="utf-8")
    for pkg in ("functions", "gui", "detectors", "installer", "nicegui"):
        assert f"import {pkg}" not in src and f"from {pkg}" not in src


def test_restart_code_starts_again_without_a_second_browser_tab(monkeypatch):
    calls = []
    codes = iter([st.RESTART_CODE, st.RESTART_CODE, 0])
    monkeypatch.setattr(st, "start_once", lambda app, restarted: (calls.append(restarted),
                                                                  next(codes))[1])
    assert st.main([]) == 0
    assert calls == [False, True, True]
    assert "PEBBLEMAPPER_NO_BROWSER" not in st.child_env(False) or \
        os.environ.get("PEBBLEMAPPER_NO_BROWSER")
    assert st.child_env(True)["PEBBLEMAPPER_NO_BROWSER"] == "1"
    assert st.child_env(False)["PEBBLEMAPPER_STARTER"] == str(st.STARTER_VERSION)


# ------------------------------------------------- 031 T019: the installer
def _release_folder(tmp_path, files):
    src = tmp_path / "PebbleMapper-1.1.0"
    _tree(src, files)
    (src / st.FILES).write_text(json.dumps({"files": [_entry(k, v) for k, v in files.items()]}))
    (src / "installer").mkdir(exist_ok=True)
    (src / "installer" / "known_files.json").write_text(json.dumps({"format": 1, "versions": [
        {"ref": "v1.0.0", "files": [_entry(k, v) for k, v in OLD.items()]},
        {"ref": "fix", "files": [_entry(k, v) for k, v in dict(OLD, **{"gui/app.py": b"v1 fixed"}).items()]}]}))
    return src


def test_a_1_0_folder_is_updated_in_place_and_keeps_everything_of_the_user(app, tmp_path):
    old_dir = app[0]
    (old_dir / "gui" / "app.py").write_bytes(b"v1 fixed")          # the later 1.0 download
    (old_dir / "datasets/example/a.csv").write_bytes(b"edited by the user")
    user_before = {k: (old_dir / k).read_bytes() for k in USER}
    src = _release_folder(tmp_path, NEW)
    (src / "pm_starter.py").write_bytes(b"the new starter")
    (src / "PebbleMapper.cmd").write_bytes(b"@echo off\r\n")
    src_before = _snapshot(src)
    r = st.migrate_into(src, old_dir)
    assert (old_dir / "pm_starter.py").read_bytes() == b"the new starter"
    assert (old_dir / "PebbleMapper.cmd").read_bytes() == b"@echo off\r\n"
    assert (r["from"], r["to"]) == ("1.0.0", "1.1.0")
    for rel, data in NEW.items():
        assert (old_dir / rel).read_bytes() == data
    assert {k: (old_dir / k).read_bytes() for k in USER} == user_before
    assert r["kept_mine"] == [str(Path("datasets/example/a.csv.mine"))], \
        "only the file the user changed, not the one a later 1.0 download changed"
    assert not (old_dir / "docs/old.md").exists()
    assert _snapshot(src) == src_before, "the release folder is only read"
    ptr = st.load_pointer(old_dir)
    assert ptr["current"] == "1.1.0" and ptr["env"] == "maskrcnn" and ptr["previous"] == "1.0.0"
    assert json.loads((old_dir / st.FILES).read_text())["files"][0]["path"] in NEW
    # ... and the user can still go back to 1.0.0 from the app.
    ptr = st.roll_back(old_dir, ptr, "returned")
    assert (old_dir / "gui" / "app.py").read_bytes() == b"v1 fixed"


def test_migration_refuses_what_it_must_not_touch(app, tmp_path):
    src = _release_folder(tmp_path, NEW)
    with pytest.raises(ValueError):
        st.migrate_into(src, src)
    with pytest.raises(ValueError):
        st.migrate_into(src, tmp_path / "not-an-install")
    (app[0] / ".git").mkdir()
    with pytest.raises(ValueError):
        st.migrate_into(src, app[0])
    (src / st.FILES).unlink()
    with pytest.raises(ValueError):
        st.migrate_into(src, tmp_path / "PebbleMapper-1.1.0")


def test_record_install_writes_a_fresh_pointer_or_keeps_a_live_environment(app, tmp_path):
    a = app[0]
    assert st.record_install(a) is False, "no file list: cannot update itself"
    assert st.load_pointer(a)["env"] == "maskrcnn"
    (a / st.FILES).write_text("{}")
    env = tmp_path / "app-envs" / "1.0.0"
    env.mkdir(parents=True)
    ptr = st.load_pointer(a)
    ptr.update(env=str(env), previous="0.9")
    st._write_json(a / st.POINTER, ptr)
    assert st.record_install(a) is True
    assert st.load_pointer(a)["env"] == str(env), "a repair keeps the updated environment"
    env.rmdir()
    st.record_install(a)
    assert st.load_pointer(a)["env"] == "maskrcnn"
    (a / "functions/__init__.py").write_bytes(b'__version__ = "1.2.0"\n')
    st.record_install(a)
    assert st.load_pointer(a) == st.fresh_pointer("1.2.0")


def test_installer_commands_report_and_return_codes(app, tmp_path, monkeypatch, capsys):
    src = _release_folder(tmp_path, NEW)
    monkeypatch.setattr(st, "__file__", str(src / "pm_starter.py"))
    assert st.main(["--migrate-into", str(app[0])]) == 0
    assert "from 1.0.0 to 1.1.0" in capsys.readouterr().out
    assert st.main(["--migrate-into", str(tmp_path / "nothing")]) == 1
    assert st.main(["--bogus"]) == 2


# ------------------------------------- 031 SC-006: a crash during the switch
class _PowerCut(BaseException):
    """Not an Exception: nothing in the switch may catch it, as with a crash."""


@pytest.mark.parametrize("cut_at", [1, 4, 7])
def test_a_switch_cut_short_is_undone_at_the_next_start(app, monkeypatch, cut_at):
    app_dir, stage, old_list, new_list = app
    (app_dir / "datasets/example/a.csv").write_bytes(b"edited by the user")
    before = _snapshot(app_dir)
    real, calls = st._rename, {"n": 0}

    def cut(src, dst, tries=6):
        calls["n"] += 1
        if calls["n"] == cut_at:
            raise _PowerCut()
        real(src, dst, tries=1)
    monkeypatch.setattr(st, "_rename", cut)
    with pytest.raises(_PowerCut):
        st.switch(app_dir, stage, old_list, new_list)
    monkeypatch.setattr(st, "_rename", real)
    assert (app_dir / st.UPDATE_DIR / st.PARTIAL).exists()
    assert st.recover_interrupted(app_dir) is True
    assert _snapshot(app_dir) == before
    assert st.verify_stage(stage, new_list) == [], "the stage is whole again, ready to apply"
    assert st.recover_interrupted(app_dir) is False
    # ... and applying it again now succeeds.
    st.switch(app_dir, stage, old_list, new_list)
    assert (app_dir / "gui/app.py").read_bytes() == b"v2 app"


def test_a_torn_last_journal_line_is_skipped(app):
    app_dir = app[0]
    upd = app_dir / st.UPDATE_DIR
    upd.mkdir()
    moved = upd / "previous" / "gui" / "app.py"
    moved.parent.mkdir(parents=True)
    (app_dir / "gui" / "app.py").replace(moved)
    (upd / st.PARTIAL).write_text(
        json.dumps(["moved", str(app_dir / "gui" / "app.py"), str(moved)]) + '\n["placed", "C:/x', encoding="utf-8")
    assert st.recover_interrupted(app_dir)
    assert (app_dir / "gui" / "app.py").read_bytes() == b"v1 app"


def test_a_new_switch_drops_the_journal_of_the_version_it_replaces(app):
    app_dir, stage, old_list, new_list = app
    upd = app_dir / st.UPDATE_DIR
    upd.mkdir()
    (upd / "journal.json").write_text('{"journal": [["placed", "a", "b"]]}')
    (upd / "previous_files.json").write_text("{}")
    real = st._rename

    def fail(src, dst, tries=6):
        raise PermissionError(13, "locked", str(src))
    st._rename = fail
    try:
        with pytest.raises(PermissionError):
            st.switch(app_dir, stage, old_list, new_list)
    finally:
        st._rename = real
    assert not (upd / "journal.json").exists(), "a stale journal could be reversed over 1.0"


def test_the_switch_never_touches_the_starter_even_if_a_list_names_it(app):
    app_dir, stage, old_list, new_list = app
    (app_dir / "pm_starter.py").write_bytes(b"running starter")
    old_list["files"].append(_entry("pm_starter.py", b"running starter"))   # an early list
    st.switch(app_dir, stage, old_list, new_list)
    assert (app_dir / "pm_starter.py").read_bytes() == b"running starter"
