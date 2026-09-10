"""F1-3 — app-import smoke.

Importing ``gui.app`` must succeed without raising, so a broken import (bad
syntax, missing symbol, top-level exception) is caught in CI without launching
a browser or binding a port.

Port discipline: ``gui/app.py`` guards ``ui.run(...)`` behind
``if __name__ in ("__main__", "__mp_main__")`` (gui/app.py:14623), so a plain
``import`` starts no server and opens no browser. This test therefore does NOT
bind port 8080 — verified by asserting the module imported is not ``__main__``
and that ``ui.run`` was never invoked.
"""
from __future__ import annotations

import importlib

import pytest

# NiceGUI must be installed for the app to import at all; skip cleanly on a
# stripped env rather than reporting a hard error.
pytest.importorskip("nicegui")


def test_import_gui_app_does_not_raise():
    """``import gui.app`` succeeds end-to-end (catches broken imports)."""
    mod = importlib.import_module("gui.app")
    assert mod is not None
    # Sanity: it really is the app module, imported as a library (not run as a
    # script), so the ``__main__`` ui.run() block did not execute.
    assert mod.__name__ == "gui.app"
    assert getattr(mod, "ui", None) is not None, (
        "gui.app did not expose the NiceGUI 'ui' object after import")


def test_import_did_not_bind_port():
    """Re-importing must not have launched the server (no port bind).

    We monkeypatch ``nicegui.ui.run`` to fail loudly, then force a reimport of
    ``gui.app``. Because the ``ui.run`` call lives under the ``__main__`` guard,
    importing as a library must never reach it; if it did, the patched run would
    raise and fail this test.
    """
    from nicegui import ui

    called = {"hit": False}
    original_run = ui.run

    def _boom(*_args, **_kwargs):
        called["hit"] = True
        raise AssertionError("ui.run() was called during import — a port "
                             "would have been bound; the __main__ guard failed")

    ui.run = _boom
    try:
        import gui.app  # noqa: F401  (already cached; this just re-binds)
        importlib.reload(importlib.import_module("gui.app"))
    finally:
        ui.run = original_run

    assert called["hit"] is False


# --------------------------------------------------------------------------- #
#  The port is a default, not a hardcoding                                     #
# --------------------------------------------------------------------------- #
def test_the_default_port_leaves_8080_free():
    """8080 is the first port everything else claims, and a second NiceGUI
    application is often running beside this one. Sharing it is not possible;
    the loser used to fail with a bind error rather than anything that
    explained itself."""
    from gui.app import DEFAULT_PORT
    assert DEFAULT_PORT == 8081


def test_the_port_can_be_overridden(monkeypatch):
    from gui.app import _listen_port, DEFAULT_PORT
    monkeypatch.delenv("PEBBLEMAPPER_PORT", raising=False)
    monkeypatch.delenv("CSM_PORT", raising=False)
    assert _listen_port() == DEFAULT_PORT

    monkeypatch.setenv("PEBBLEMAPPER_PORT", "9000")
    assert _listen_port() == 9000

    monkeypatch.delenv("PEBBLEMAPPER_PORT")
    monkeypatch.setenv("CSM_PORT", "9100")
    assert _listen_port() == 9100


def test_a_nonsense_port_falls_back_rather_than_refusing_to_start(monkeypatch):
    """Refusing to boot because an environment variable is malformed helps
    nobody; the default is always a working answer."""
    from gui.app import _listen_port, DEFAULT_PORT
    monkeypatch.delenv("CSM_PORT", raising=False)
    for bad in ("", "   ", "abc", "0", "-1", "70000", "80.80"):
        monkeypatch.setenv("PEBBLEMAPPER_PORT", bad)
        assert _listen_port() == DEFAULT_PORT, f"{bad!r} was accepted"
