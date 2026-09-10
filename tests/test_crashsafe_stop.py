"""functions/crashsafe.py: a closed window is a stop, not a crash.

The breadcrumb alone cannot tell a crash from a console window closed on a
healthy server, a killed process or a power cut. The trace file can: a real
crash leaves a faulthandler dump or a worker post-mortem after the armed
header, an abrupt stop leaves the header alone. Only the former is announced;
and closing the console window marks the exit clean in the first place.
"""
import os
import subprocess
import sys

import pytest

import functions.crashsafe as crashsafe

HEADER = ("# PebbleMapper crash trace - armed 2026-09-20T10:00:00 pid=1\n"
          "# If this file ends after this header, the process died without an\n"
          "# in-process trace.\n")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"
    crashsafe.disarm()
    crashsafe._state["prev_crash"] = None
    crashsafe._state["degraded_reason"] = None


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _unclean_breadcrumb(trace):
    crashsafe._write_breadcrumb({
        "version": 1, "started_at": "2026-09-20T10:00:00", "pid": _dead_pid(),
        "trace_path": str(trace), "clean_exit": False,
    })


def test_a_closed_window_leaves_no_evidence_but_a_dump_does(home, tmp_path):
    trace = tmp_path / "crash_20260920T100000.log"
    trace.write_text(HEADER, encoding="utf-8")
    _unclean_breadcrumb(trace)
    assert crashsafe.previous_session_crashed()["evidence"] == "none"
    # A faulthandler dump after the header is a crash.
    trace.write_text(HEADER + "Fatal Python error: Segmentation fault\n\n"
                     "Thread 0x00001 (most recent call first):\n", encoding="utf-8")
    assert crashsafe.previous_session_crashed()["evidence"] == "trace"
    # So is a worker post-mortem block.
    trace.write_text(HEADER + "\n---- worker post-mortem ----\nexit_code   : 3\n",
                     encoding="utf-8")
    assert crashsafe.previous_session_crashed()["evidence"] == "trace"
    # A trace that was never written (unwritable folder) is not evidence either.
    _unclean_breadcrumb("")
    assert crashsafe.previous_session_crashed()["evidence"] == "none"


def test_boot_only_calls_a_crash_a_crash(home, tmp_path, capsys):
    trace = tmp_path / "crash_a.log"
    trace.write_text(HEADER, encoding="utf-8")
    _unclean_breadcrumb(trace)
    prev = crashsafe.boot()
    out = capsys.readouterr().out
    assert prev["evidence"] == "none"
    assert "stopped without a clean shutdown" in out and "crashed" not in out
    crashsafe.disarm()
    trace.write_text(HEADER + "Fatal Python error: Aborted\n", encoding="utf-8")
    _unclean_breadcrumb(trace)
    prev = crashsafe.boot()
    out = capsys.readouterr().out
    assert prev["evidence"] == "trace" and "crashed" in out


def test_closing_the_console_marks_the_exit_clean(home):
    crashsafe.boot()
    assert crashsafe.read_breadcrumb()["clean_exit"] is False
    # Ctrl+C (0) and Ctrl+Break (1) belong to the server's own shutdown.
    assert crashsafe.console_close_marks_clean_exit(0) is False
    assert crashsafe.read_breadcrumb()["clean_exit"] is False
    # The window's close button (2), a log-off (5), a shutdown (6): a stop.
    assert crashsafe.console_close_marks_clean_exit(2) is True
    assert crashsafe.read_breadcrumb()["clean_exit"] is True
    assert crashsafe.previous_session_crashed() is None


def test_the_console_handler_is_installed_on_windows(home):
    crashsafe.boot()
    installed = crashsafe.install_console_close_handler()
    if os.name != "nt":
        assert installed is False
        return
    assert installed is True
    # The ctypes callback itself, as the console would call it.
    assert crashsafe._console_handler_ref(2) == 0
    assert crashsafe.read_breadcrumb()["clean_exit"] is True
