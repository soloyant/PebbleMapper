"""functions/queue_store.py — queue persistence & restore rules.

Exercises the real store against a disposable PEBBLEMAPPER_HOME, including
the atomicity property (a torn temp write must not damage the last good
version) and every restore rule.
"""
import json
from pathlib import Path

import pytest

import functions.queue_store as qs


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


class _FakeState:
    def __init__(self):
        for attr in qs.TABS.values():
            setattr(self, attr, [])
        self.current_project = "projX"
        self.current_date = ""


def test_round_trip_v1_schema(home):
    rows = [{"label": "#1 job", "status": "pending",
             "params": {"a": 1}, "out_path": "C:\\x\\y.tif"}]
    assert qs.save("zonal", rows, project="p1", date_layer=None,
                   app_version="1.0.0")
    doc = qs.load("zonal")
    assert doc["version"] == qs.FILE_VERSION
    assert doc["project"] == "p1"
    assert doc["rows"] == rows
    # Empty queue -> file removed, not an empty document.
    assert qs.save("zonal", [])
    assert qs.load("zonal") is None


def test_atomic_replace_survives_torn_tmp(home):
    rows = [{"label": "good", "status": "pending"}]
    qs.save("merge", rows)
    # Simulate a crash mid-write: a leftover temp beside the good file.
    (qs.queues_dir() / "merge.tmp-dead00").write_text("{ not json",
                                                      encoding="utf-8")
    doc = qs.load("merge")
    assert doc["rows"][0]["label"] == "good"


def test_unknown_version_refused_by_name(home):
    qs.queues_dir().mkdir(parents=True, exist_ok=True)
    (qs.queues_dir() / "map.json").write_text(
        json.dumps({"version": 99, "rows": [{"status": "pending"}]}),
        encoding="utf-8")
    doc = qs.load("map")
    assert "version_error" in doc and "99" in doc["version_error"]
    pend = qs.pending_restore()
    assert pend["map"]["version_error"]


def test_running_at_rest_becomes_interrupted(home):
    rows = [{"label": "a", "status": "running"},
            {"label": "b", "status": "pending"}]
    qs.revalidate_rows(rows)
    assert rows[0]["status"] == "interrupted"
    assert rows[1]["status"] == "pending"


def test_done_with_missing_outputs_demotes(home, tmp_path):
    kept = tmp_path / "kept.tif"
    kept.write_text("x", encoding="utf-8")
    rows = [{"status": "done", "out_path": str(kept)},
            {"status": "done", "out_path": str(tmp_path / "gone.tif")}]
    qs.revalidate_rows(rows)
    assert rows[0]["status"] == "done"
    assert rows[1]["status"] == "pending"
    assert "demoted" in rows[1]["error"]


def test_pending_restore_and_restore_into(home, tmp_path):
    out = tmp_path / "exists.csv"
    out.write_text("x", encoding="utf-8")
    qs.save("detection", [{"label": "j1", "status": "running"},
                          {"label": "j2", "status": "done",
                           "out_path": str(out)}])
    qs.save("rasterize", [{"label": "r1", "status": "done",
                           "out_path": str(out)}])  # all terminal: no offer
    pend = qs.pending_restore()
    assert "detection" in pend and pend["detection"]["rows"] == 1
    assert "rasterize" not in pend

    st = _FakeState()
    restored = qs.restore_into(st)
    assert restored["detection"] == 2
    assert st.det_jobs[0]["status"] == "interrupted"
    assert st.det_jobs[1]["status"] == "done"
    # Frozen content restored verbatim (labels intact).
    assert st.det_jobs[0]["label"] == "j1"


def test_decline_archives_and_does_not_reoffer(home):
    qs.save("zonal", [{"label": "z", "status": "pending"}])
    assert qs.pending_restore()
    n = qs.archive_declined()
    assert n == 1
    assert qs.pending_restore() == {}
    archived = list((qs.queues_dir() / "declined").glob("zonal-*.json"))
    assert len(archived) == 1


def test_watcher_writes_on_change_only(home):
    st = _FakeState()
    w = qs.QueueWatcher(st, app_version="t")
    w.flush_once()
    assert qs.load("zonal") is None          # empty stays absent
    st.zonal_jobs.append({"label": "j", "status": "pending"})
    w.flush_once()
    doc = qs.load("zonal")
    assert doc["rows"][0]["label"] == "j"
    first_saved_at = doc["saved_at"]
    w.flush_once()                            # unchanged -> no rewrite
    assert qs.load("zonal")["saved_at"] == first_saved_at
    st.zonal_jobs[0]["status"] = "running"
    w.flush_once()
    assert qs.load("zonal")["rows"][0]["status"] == "running"
    st.zonal_jobs.clear()
    w.flush_once()
    assert qs.load("zonal") is None
