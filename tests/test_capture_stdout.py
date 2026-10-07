"""capture_stdout_to_log must never recurse into itself.

The crash it guards against: the browser
page is reloaded while a run streams stdout into a ui.log. NiceGUI then
warns through ``logging`` that the client is gone; the logging handler
writes to sys.stdout, which is the capturing writer, which pushes again,
which warns again -- until the stack overflows (0xC00000FD).

Two things are pinned here, with a fake log whose ``push`` misbehaves the
way the real one does:

* a push that writes to stdout (a warning, a logging handler) must not be
  pushed back into the log;
* once the log's client is marked deleted, lines go to the console and
  the widget is never touched again.
"""
from __future__ import annotations

import sys

import pytest

# The fallback writes to sys.__stdout__ (the real console), which only the
# file-descriptor capture sees.


@pytest.fixture
def capture():
    from gui.app import capture_stdout_to_log
    return capture_stdout_to_log


class _Client:
    def __init__(self):
        self._deleted = False


class _WarningLog:
    """A ui.log stand-in: every push warns to stdout, like NiceGUI's
    ``check_existence`` once the client is deleted."""

    def __init__(self):
        self.client = _Client()
        self.lines = []
        self.pushes = 0

    def push(self, line):
        self.pushes += 1
        self.lines.append(line)
        print("Client has been deleted but is still being used.")


def test_push_that_writes_to_stdout_does_not_recurse(capture, capfd):
    log = _WarningLog()
    with capture(log):
        print("tile 1/10")
        print("tile 2/10")
    sys.__stdout__.flush()
    out = capfd.readouterr().out
    # Two lines reached the log, and each warning went to the console.
    assert log.lines == ["tile 1/10", "tile 2/10"]
    assert log.pushes == 2
    assert out.count("Client has been deleted") == 2
    assert sys.stdout is not None


def test_deleted_client_stops_pushing(capture, capfd):
    log = _WarningLog()
    seen = []
    with capture(log, on_line=seen.append):
        print("before reload")
        log.client._deleted = True      # the page was reloaded
        print("after reload")
        print("still running")
    sys.__stdout__.flush()
    out = capfd.readouterr().out
    assert log.lines == ["before reload"]
    assert "after reload" in out and "still running" in out
    # on_line (the progress-bar hook) keeps seeing every line either way.
    assert seen == ["before reload", "after reload", "still running"]


def test_flush_and_stdout_restored(capture, capfd):
    log = _WarningLog()
    old = sys.stdout
    with capture(log):
        sys.stdout.write("no newline yet")
    assert sys.stdout is old
    assert log.lines == ["no newline yet"]


# --------------------------------------------------------------------------- #
#  A run outlives its page: the console re-attaches                           #
# --------------------------------------------------------------------------- #
def _lines(widget):
    return [c.text for c in widget.default_slot.children]


@pytest.fixture
def cols():
    from gui.app import _LIVE_LOGS, _LOG_HISTORY
    _LIVE_LOGS.pop("t", None)
    _LOG_HISTORY.pop("t", None)
    built = []
    yield built
    for c in built:
        try:
            c.delete()
        except Exception:
            pass
    _LIVE_LOGS.pop("t", None)
    _LOG_HISTORY.pop("t", None)


def test_a_console_built_later_replays_and_receives_the_run(cols):
    from nicegui import ui
    from gui.app import live_log
    with ui.column() as c1:
        w1 = live_log("t", ui.log())
    cols.append(c1)
    w1.push("tile 1/10")
    assert _lines(w1) == ["tile 1/10"]
    # The page is rebuilt (a reload): the new console shows what the run
    # printed so far, and the worker still holding w1 lands on it.
    with ui.column() as c2:
        w2 = live_log("t", ui.log())
    cols.append(c2)
    assert _lines(w2) == ["tile 1/10"]
    w1.push("tile 2/10")
    assert _lines(w2) == ["tile 1/10", "tile 2/10"]
    assert _lines(w1) == ["tile 1/10", "tile 2/10"], (
        "an older page that is still open shows the run too")


def test_capture_reaches_every_open_console(cols, capture):
    from nicegui import ui
    from gui.app import live_log
    with ui.column() as c1:
        w1 = live_log("t", ui.log())
    cols.append(c1)
    with ui.column() as c2:
        w2 = live_log("t", ui.log())
    cols.append(c2)
    with capture(w1):
        print("from the worker")
    assert _lines(w2) == ["from the worker"]
    assert _lines(w1) == ["from the worker"]


def test_a_console_whose_page_is_gone_is_forgotten(cols):
    """The first page of a session stayed silent because pushes went to
    the newest console alone, whatever built it; now they reach every
    console whose page is still there, and a dead one is dropped."""
    from nicegui import ui
    from gui.app import live_log, _LIVE_LOGS
    with ui.column() as c1:
        w1 = live_log("t", ui.log())
    cols.append(c1)

    class _Dead:
        client = _Client()
        default_slot = None
    dead = live_log("t", _Dead())
    dead.client._deleted = True          # that page was closed
    w1.push("tile 1/10")
    assert _lines(w1) == ["tile 1/10"]
    assert dead not in _LIVE_LOGS["t"] and _LIVE_LOGS["t"] == [w1]


def test_on_page_survives_a_pruned_slot_stack(monkeypatch):
    """NiceGUI gives every thread slot-stack key 0 and prunes it every
    10 s: a context held across a run finds nothing to pop at its end."""
    import threading
    from nicegui import context
    from nicegui.slot import Slot, get_task_id
    from gui.app import on_page
    client = context.client
    # Without a running loop this test's own thread has key 0 as well:
    # give the worker a stacks dict of its own.
    monkeypatch.setattr(Slot, "stacks", {})
    res = {}

    def worker():
        try:
            with client:
                Slot.stacks.pop(get_task_id(), None)   # the pruner ran
        except IndexError:
            res["plain"] = "IndexError"
        try:
            with on_page(client):
                Slot.stacks.pop(get_task_id(), None)
            res["on_page"] = "ok"
        except IndexError:
            res["on_page"] = "IndexError"
    t = threading.Thread(target=worker, name="batch")
    t.start()
    t.join()
    assert res == {"plain": "IndexError", "on_page": "ok"}, res


def test_clear_empties_the_history_too(cols):
    from nicegui import ui
    from gui.app import live_log
    with ui.column() as c1:
        w1 = live_log("t", ui.log())
    cols.append(c1)
    w1.push("old run")
    w1.clear()
    with ui.column() as c2:
        w2 = live_log("t", ui.log())
    cols.append(c2)
    assert _lines(w2) == []
