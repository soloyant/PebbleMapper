"""Rasters the browser fetches instead of being sent.

The alignment editor re-sends its whole SVG overlay
on every mouse move, and it used to embed the quadrat's PNG inline. Measured on
a real Bio_Station quadrat that was 192.7 KB a move against the site's own
2.42 mm ortho, and 1,056.8 KB a move on a 1 mm one -- 5.9 and 32.5 MB/s while
dragging, down the websocket the heartbeat shares.

These tests exist because this is GUI wiring, and GUI wiring is what this
suite has repeatedly failed to catch: defects that lived in closures inside
`gui/app.py` were only ever found by a browser. The route and
the cache are module-level, so they can be tested for real.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nicegui")
pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def gui():
    from gui import app as G
    return G


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from nicegui import app as ngapp
    return TestClient(ngapp)


def _png(seed=0, n=32):
    import cv2
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) * 255).astype("uint8")
    return cv2.imencode(".png", a)[1].tobytes()


def test_the_url_is_short_enough_to_send_every_mouse_move(gui):
    """The whole point. A data: URI of the same raster is six thousand times
    this, and the overlay carrying it is re-sent on every move."""
    url = gui._raster_url(_png())
    assert url.startswith("/pm-raster/")
    assert len(url) < 64


def test_the_route_serves_the_exact_bytes(gui, client):
    png = _png(1)
    r = client.get(gui._raster_url(png))
    assert r.status_code == 200
    assert r.content == png
    assert r.headers["content-type"] == "image/png"


def test_the_route_is_registered_at_import_not_on_first_use(client):
    """Starlette's router is mutable after startup, so a lazily registered
    route appears to work -- but the failure mode is a 404 in place of the
    quadrat, in a running server only, which no test would catch."""
    assert client.get("/pm-raster/deadbeefdeadbeef.png").status_code == 404


def test_the_name_is_the_content_so_a_stale_raster_is_impossible(gui):
    """The editor keeps a live transform over a cached image. If the URL did
    not change when the pixels did, a drag would show the previous quadrat
    while reading as correct."""
    png = _png(2)
    assert gui._raster_url(png) == gui._raster_url(png)
    assert gui._raster_url(png) != gui._raster_url(_png(3))


def test_it_is_cached_forever_because_the_name_pins_the_content(gui, client):
    r = client.get(gui._raster_url(_png(4)))
    cc = r.headers.get("cache-control", "")
    assert "immutable" in cc and "max-age=31536000" in cc


def test_the_cache_is_bounded_so_a_long_session_does_not_leak(gui):
    import cv2
    for i in range(gui._RASTER_KEEP * 2):
        gui._raster_url(cv2.imencode(".png", np.full((8, 8), i % 256,
                                                     "uint8"))[1].tobytes())
    assert len(gui._RASTER_CACHE) <= gui._RASTER_KEEP


def test_the_most_recent_raster_survives_eviction(gui, client):
    """Eviction is least-recently-used. Dropping the raster the editor is
    currently showing would blank the quadrat mid-drag."""
    import cv2
    live = _png(99)
    url = gui._raster_url(live)
    for i in range(gui._RASTER_KEEP - 1):
        gui._raster_url(cv2.imencode(".png",
                                     np.full((4, 4), i, "uint8"))[1].tobytes())
        gui._raster_url(live)          # as a redraw would
    assert client.get(url).content == live


def test_the_view_modes_do_not_collide_on_one_raster(gui, client):
    """The editor caches four rasters per quadrat -- base and quadrat, each in
    colour and high-pass -- and switching mode is a key lookup against them.
    If two variants produced the same URL the editor would show the wrong one
    while reading as correct, which is the failure mode content addressing
    exists to prevent.

    Driving the Quasar `View` select itself needs trusted events, which the
    test harness cannot synthesise, so the collision is what is asserted here.
    """
    import cv2
    rng = np.random.default_rng(7)
    a = (rng.random((64, 64)) * 255).astype("uint8")
    hp = cv2.normalize(a.astype("float32")
                       - cv2.GaussianBlur(a.astype("float32"), (0, 0), 3),
                       None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    urls = {name: gui._raster_url(cv2.imencode(".png", arr)[1].tobytes())
            for name, arr in (("colour", a), ("highpass", hp))}
    assert urls["colour"] != urls["highpass"]
    for u in urls.values():
        assert client.get(u).status_code == 200


def test_four_rasters_a_quadrat_fit_the_cache_several_times_over(gui):
    """base+quadrat x colour+high-pass = 4 per quadrat. The bound has to leave
    room to switch modes and step between quadrats without a refetch storm."""
    assert gui._RASTER_KEEP >= 4 * 4


# --------------------------------------------------------------------------- #
#  The heartbeat these bytes were competing with                               #
# --------------------------------------------------------------------------- #
def test_reconnect_timeout_is_above_the_floor_that_makes_it_a_no_op():
    """NiceGUI derives the socket heartbeat from reconnect_timeout through
    floors (nicegui/nicegui.py:124):

        ping_interval = max(reconnect_timeout * 0.8, 4)
        ping_timeout  = max(reconnect_timeout * 0.4, 2)

    so any value at or below 5.0 -- including the 3.0 default -- yields 4.0/2.0
    and changes nothing. A first draft of this fix set 5.0 and would have
    shipped a no-op.
    """
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "gui" / "app.py"
    text = src.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"reconnect_timeout\s*=\s*([0-9.]+)", text)
    assert m, "ui.run() no longer sets reconnect_timeout"
    rt = float(m.group(1))
    assert rt > 5.0, f"reconnect_timeout={rt} is floored out of effect"
    assert max(rt * 0.4, 2) > 2.0        # the gate a stall is measured against


def test_mode_value_is_a_float_from_scipys_moderesult():
    """scipy's mode() returns a ModeResult; the cell loop stored it as a
    float and died."""
    import numpy as np
    from functions.clasts_rasterize import mode_value
    assert mode_value(np.array([1.0, 2.0, 2.0, 3.0, np.nan])) == 2.0
    assert isinstance(mode_value([5, 5, 6]), float)
    assert np.isnan(mode_value([np.nan, np.nan]))
    assert np.isnan(mode_value([]))


def test_a_stopped_run_does_not_replace_a_complete_csv(tmp_path):
    """Stopping a re-run of a window that already finished used to overwrite
    its complete CSV with the partial one."""
    from functions.clasts_detection import _completed_before
    csv = tmp_path / "img_ws1m.csv"
    log = tmp_path / "img_ws1m_detection_log.txt"
    csv.write_text("clast_ID,x,y\n1,0,0\n", encoding="utf-8")
    assert not _completed_before(str(csv))          # no log yet
    log.write_text("[Run 1]\n  outcome : complete\n", encoding="utf-8")
    assert _completed_before(str(csv))
    # A later stopped run appends its own outcome: the file on disk is partial.
    log.write_text("[Run 1]\n  outcome : complete\n"
                   "[Run 2]\n  outcome : partial (stopped)\n", encoding="utf-8")
    assert not _completed_before(str(csv))
    # No CSV: nothing to protect.
    csv.unlink()
    assert not _completed_before(str(csv))
