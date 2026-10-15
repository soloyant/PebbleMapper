"""030 data-model §5, FR-030–FR-034: the queue and the job state machine."""
import itertools
import subprocess
import sys
import time

import pytest

from installer import jobs


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _job(**kw):
    return jobs.new_job("install", dict({"model": "orthosam", "release": "1.0.0",
                                         "variant": "gpu"}, **kw))


def test_new_jobs_are_queued_in_order():
    a, b = _job(), _job()
    assert jobs.queue() == [a["id"], b["id"]]
    assert jobs.read_job(a["id"])["state"] == "queued"
    assert [s["name"] for s in a["steps"]][0] == "fetch"
    assert jobs.pop_next() == a["id"]
    assert jobs.queue() == [b["id"]]


def test_only_the_documented_transitions_are_allowed():
    states = list(jobs.ALLOWED)
    for old, new in itertools.product(states, states):
        job = _job()
        job["state"] = old
        if new in jobs.ALLOWED[old]:
            jobs.set_state(job, new)
            assert jobs.read_job(job["id"])["state"] == new
        else:
            with pytest.raises(jobs.StateError):
                jobs.set_state(job, new)


def test_job_json_is_never_half_written(home):
    job = _job()
    jobs.write_job(job)
    assert not list(home.rglob("*.tmp"))
    assert jobs.read_job(job["id"])["heartbeat"] > 0


def test_cancel_a_queued_job_at_once_and_a_running_one_by_request():
    q = _job()
    jobs.request_cancel(q["id"])
    assert jobs.read_job(q["id"])["state"] == "cancelled"
    r = _job()
    jobs.set_state(r, "running")
    jobs.request_cancel(r["id"])
    assert jobs.cancel_requested(r["id"])
    assert jobs.read_job(r["id"])["state"] == "running"


def test_resume_requeues_and_clears_the_cancel_flag():
    r = _job()
    jobs.pop_next()
    jobs.set_state(r, "running")
    jobs.request_cancel(r["id"])
    jobs.set_state(jobs.read_job(r["id"]), "cancelled")
    jobs.resume(r["id"])
    assert jobs.read_job(r["id"])["state"] == "queued"
    assert not jobs.cancel_requested(r["id"])
    assert jobs.queue() == [r["id"]]


def test_a_running_job_without_a_worker_is_interrupted():
    r = _job()
    jobs.set_state(r, "running")
    done = _job()
    jobs.set_state(done, "running")
    jobs.set_state(jobs.read_job(done["id"]), "done")
    got = jobs.detect_interrupted()
    assert [j["id"] for j in got] == [r["id"]]
    assert jobs.read_job(r["id"])["state"] == "interrupted"
    assert jobs.read_job(done["id"])["state"] == "done"


def test_the_worker_lock_is_exclusive_across_processes(home):
    code = ("import os, time; os.environ['PEBBLEMAPPER_HOME']=%r; "
            "from installer import jobs; l=jobs.WorkerLock(); "
            "print(l.acquire(), flush=True); time.sleep(20)") % str(home)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "True"
        assert jobs.worker_alive() is True
        r = _job()
        jobs.set_state(r, "running")
        assert jobs.detect_interrupted() == [], "a live worker owns its running job"
    finally:
        proc.kill()
        proc.wait()
    time.sleep(0.2)
    assert jobs.worker_alive() is False


def test_the_log_hides_the_home_folder(tmp_path):
    from pathlib import Path
    log = jobs.JobLog(tmp_path / "job.log")
    log(f"writing {Path.home() / 'x'} and C:\\Users\\someone\\y")
    text = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert str(Path.home()) not in text and "someone" not in text
    assert "~" in text
