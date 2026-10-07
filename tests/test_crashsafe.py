"""functions/crashsafe.py — trace arming, breadcrumb, post-mortem.

Everything runs against a disposable PEBBLEMAPPER_HOME; a fixture restores
faulthandler to stderr afterwards so later tests keep pytest's own handler.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import functions.crashsafe as crashsafe


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"
    crashsafe.disarm()
    crashsafe._state["prev_crash"] = None
    crashsafe._state["degraded_reason"] = None


def _dead_pid():
    """A pid guaranteed to have exited (freshly spawned, already waited)."""
    proc = subprocess.run([sys.executable, "-c", "pass"], capture_output=True)
    assert proc.returncode == 0
    # We cannot read the child's pid from subprocess.run; spawn via Popen.
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# --------------------------------------------------------------------- #
#  Arming / re-pointing / pruning
# --------------------------------------------------------------------- #

def test_arm_creates_trace_and_repoint_switches(home, tmp_path):
    a = tmp_path / "projA" / "logs"
    b = tmp_path / "projB" / "logs"
    pa = crashsafe.arm(a)
    assert pa is not None and pa.parent == a and pa.exists()
    assert pa.read_text(encoding="utf-8").startswith("# PebbleMapper crash trace")
    assert crashsafe.current_trace_path() == pa

    pb = crashsafe.repoint(b)
    assert pb.parent == b and crashsafe.current_trace_path() == pb
    # Same target again: no new file (no-op).
    pb2 = crashsafe.repoint(b)
    assert pb2 == pb
    # None means the app-level fallback.
    pf = crashsafe.repoint(None)
    assert pf.parent == crashsafe.app_log_dir()


def test_arm_prunes_old_traces(home, tmp_path):
    d = tmp_path / "logs"
    d.mkdir(parents=True)
    for i in range(12):
        (d / f"crash_20200101T0000{i:02d}.log").write_text("old", encoding="utf-8")
    p = crashsafe.arm(d)
    remaining = sorted(d.glob("crash_*.log"))
    assert len(remaining) == crashsafe.TRACE_KEEP
    assert p in remaining  # the fresh trace survives its own prune


def test_arm_unwritable_location_degrades_not_raises(home, tmp_path):
    good = tmp_path / "good"
    pg = crashsafe.arm(good)
    assert pg is not None
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a directory is needed", encoding="utf-8")
    bad = crashsafe.arm(blocker / "logs")
    assert bad is None
    assert crashsafe.degraded_reason() is not None
    # Previous target still armed.
    assert crashsafe.current_trace_path() == pg


# --------------------------------------------------------------------- #
#  Breadcrumb lifecycle
# --------------------------------------------------------------------- #

def test_boot_writes_breadcrumb_and_clean_exit_flips(home):
    prev = crashsafe.boot()
    assert prev is None  # no earlier session in a fresh home
    bc = crashsafe.read_breadcrumb()
    assert bc["version"] == crashsafe.BREADCRUMB_VERSION
    assert bc["pid"] == os.getpid()
    assert bc["clean_exit"] is False
    assert bc["trace_path"].endswith(".log")
    crashsafe.mark_clean_exit()
    assert crashsafe.read_breadcrumb()["clean_exit"] is True


def test_repoint_updates_breadcrumb_trace_path(home, tmp_path):
    crashsafe.boot()
    newdir = tmp_path / "proj" / "logs"
    p = crashsafe.repoint(newdir)
    assert crashsafe.read_breadcrumb()["trace_path"] == str(p)


def test_previous_session_crashed_conditions(home):
    # Unclean + dead pid -> announced.
    crashsafe._write_breadcrumb({
        "version": 1, "started_at": "2026-09-04T10:00:00",
        "pid": _dead_pid(), "trace_path": "X:\\gone\\crash_x.log",
        "clean_exit": False,
    })
    got = crashsafe.previous_session_crashed()
    assert got == {"trace_path": "X:\\gone\\crash_x.log",
                   "started_at": "2026-09-04T10:00:00",
                   "evidence": "none"}      # no trace file: nothing to show
    # Clean exit -> silence.
    crashsafe._write_breadcrumb({
        "version": 1, "started_at": "t", "pid": _dead_pid(),
        "trace_path": "", "clean_exit": True,
    })
    assert crashsafe.previous_session_crashed() is None
    # Live pid (another running instance) -> silence.
    crashsafe._write_breadcrumb({
        "version": 1, "started_at": "t", "pid": os.getpid(),
        "trace_path": "", "clean_exit": False,
    })
    assert crashsafe.previous_session_crashed() is None


def test_boot_announcement_consumed_once(home):
    crashsafe._write_breadcrumb({
        "version": 1, "started_at": "2026-09-04T10:00:00",
        "pid": _dead_pid(), "trace_path": "T", "clean_exit": False,
    })
    prev = crashsafe.boot()
    assert prev is not None
    assert crashsafe.pending_announcement() == prev
    crashsafe.mark_announced()
    assert crashsafe.pending_announcement() is None
    # And this session's own breadcrumb replaced the crashed one.
    assert crashsafe.read_breadcrumb()["pid"] == os.getpid()


# --------------------------------------------------------------------- #
#  Worker post-mortem
# --------------------------------------------------------------------- #

def test_worker_postmortem_appends_to_current_trace(home, tmp_path):
    p = crashsafe.arm(tmp_path / "logs")
    out = crashsafe.write_worker_postmortem(
        0xC00000FD - (1 << 32), "zonal_polygons", "C:\\env.json",
        "2026-09-04T10:00:00", "2026-09-04T10:00:05")
    assert out == p
    text = p.read_text(encoding="utf-8")
    assert "worker post-mortem" in text
    assert "op          : zonal_polygons" in text
    assert "0xC00000FD" in text


def test_worker_postmortem_survives_disarmed_state(home):
    crashsafe.disarm()
    out = crashsafe.write_worker_postmortem(0xAD, "merge", "e.json", "a", "b")
    assert out is not None and out.parent == crashsafe.app_log_dir()
    assert "worker post-mortem" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  Fault injection ships inert
# --------------------------------------------------------------------- #

def test_induce_test_fault_refuses_without_env(home, monkeypatch):
    monkeypatch.delenv("PEBBLEMAPPER_ENABLE_TEST_FAULTS", raising=False)
    msg = crashsafe.induce_test_fault("segv")
    assert "Refused" in msg  # and, self-evidently, the process is alive


def test_induce_test_fault_kills_a_real_subprocess(home):
    """The armed kind really does end a process (proved on a child, not us)."""
    env = dict(os.environ)
    env["PEBBLEMAPPER_ENABLE_TEST_FAULTS"] = "1"
    repo = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "from functions.crashsafe import induce_test_fault; "
         "induce_test_fault('hard-exit')", repo],
        capture_output=True, env=env, timeout=60)
    assert proc.returncode == 0xAD


def test_only_the_proactors_connection_reset_report_is_dropped():
    """asyncio on Windows reports a peer that reset its connection as
    'Exception in callback _ProactorBasePipeTransport._call_connection_lost'
    with a traceback, at every probe of the address."""
    import asyncio
    from functions import crashsafe
    seen = []
    loop = asyncio.new_event_loop()
    try:
        loop.set_exception_handler(lambda lp, ctx: seen.append(ctx.get("message")))
        crashsafe.quiet_windows_connection_resets(loop)
        reset = {"message": "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
                 "exception": ConnectionResetError(10054, "reset"),
                 "handle": "<Handle _ProactorBasePipeTransport._call_connection_lost(None)>"}
        other = {"message": "Task exception was never retrieved",
                 "exception": RuntimeError("real")}
        loop.call_exception_handler(reset)
        loop.call_exception_handler(other)
        assert seen == ["Task exception was never retrieved"]
        assert crashsafe.is_windows_connection_reset(reset)
        assert not crashsafe.is_windows_connection_reset(other)
    finally:
        loop.close()


def test_a_run_that_did_not_crash_leaves_no_trace_behind(home, tmp_path):
    """Every clean exit left a header-only crash_*.log in the project's logs
    folder; seven accumulated in one day of testing."""
    d = tmp_path / "logs"
    d.mkdir(parents=True)
    header_only = d / "crash_20200101T000001.log"
    header_only.write_text("# PebbleMapper crash trace — armed ...\n# second line\n",
                           encoding="utf-8")
    real = d / "crash_20200101T000002.log"
    real.write_text("# header\nThread 0x1 (most recent call first):\n  File ...\n",
                    encoding="utf-8")
    assert crashsafe.is_empty_trace(header_only) and not crashsafe.is_empty_trace(real)
    fresh = crashsafe.arm(d)
    names = {p.name for p in d.glob("crash_*.log")}
    assert header_only.name not in names        # nothing to say, removed
    assert real.name in names                   # a real trace is kept
    assert fresh.name in names                  # and the one just armed
