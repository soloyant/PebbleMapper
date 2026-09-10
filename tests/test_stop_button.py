"""The drawer's Stop PebbleMapper button: a confirmation, then a clean stop.

Built off-screen; the server's shutdown is stood in for. What is checked is
that the button asks first, that Cancel changes nothing, and that Stop marks
the session's breadcrumb clean before shutting the server down, so the next
start reports nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

import functions.crashsafe as crashsafe  # noqa: E402


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def _button(col, label):
    return next(e for e in _walk(col) if e.__class__.__name__ == "Button"
                and str(getattr(e, "text", "")) == label)


def _press(btn):
    from nicegui.events import ClickEventArguments, handle_event
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"
    crashsafe.disarm()


def test_stop_asks_first_and_then_stops_cleanly(home, monkeypatch):
    from nicegui import ui, app
    from gui.app import render_stop_button
    calls = []
    monkeypatch.setattr(app, "shutdown", lambda: calls.append("shutdown"))
    crashsafe.boot()
    assert crashsafe.read_breadcrumb()["clean_exit"] is False
    with ui.column() as col:
        render_stop_button()
    try:
        btn = _button(col, "Stop PebbleMapper")
        assert "pm-stop-button" in btn._classes
        dlg = next(e for e in _walk(col) if e.__class__.__name__ == "Dialog")
        assert dlg.value is False
        _press(btn)
        assert dlg.value is True, "the button opens the confirmation"
        _press(_button(col, "Cancel"))
        assert dlg.value is False and calls == []
        assert crashsafe.read_breadcrumb()["clean_exit"] is False
        _press(btn)
        _press(_button(col, "Stop"))
        assert calls == ["shutdown"]
        assert crashsafe.read_breadcrumb()["clean_exit"] is True
        assert crashsafe.previous_session_crashed() is None
    finally:
        col.delete()
