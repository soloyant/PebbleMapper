"""Digitize's *No GSD? Scale from an object* option, built off-screen.

The section is present and closed until the user opens it; it holds the scaling-object
library, the two-click segment toggle, the assignment rows, the status line,
the result unit and the detection scale, and nothing of it shows elsewhere on
the tab. The toggle turns the canvas into the segment mode and back; two
clicks commit a bold, labelled segment that is not a clast record; the scale
in force is the GSD until a segment exists, then the object. Detect, with the
detector injected, loads the masks as editable proposals and writes nothing;
Figures writes the CSV, the sidecars and both figures from the records on the
canvas through functions.gauge in millimetres (no disclaimer) or in the
object's unit (with it); the truth CSV follows the unit. Select mode, Delete,
Undo, the clast table and the provenance sidecar are tested at the end.
Twelve tabs remain and none is Gauge.

What none of this proves -- the expansion animating, a drag, the real model
-- needs a visible browser.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from functions import gauge as Q  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"

_FIELDS = ("current_project", "current_date", "active_tab", "dig_src_path",
           "dig_image_dir", "dig_image_list", "dig_image_idx", "dig_records",
           "dig_out_path", "dig_resolution", "dig_per_image", "dig_scale_open",
           "dig_scale_draw", "dig_scale_library", "dig_scale_segments",
           "dig_scale_result_unit", "dig_detect_scale", "dig_detect_running",
           "dig_mask_detector", "dig_detect_threshold", "dig_mode",
           "dig_active_polygon", "dig_autosave", "dig_zoom", "devicemode",
           "det_model", "dig_detect_runs", "dig_figures_running",
           "dig_csv_loaded", "dig_csv_owned", "dig_label_set")


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


@pytest.fixture
def clean():
    from gui.app import state
    saved = {k: getattr(state, k) for k in _FIELDS}
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    state.current_project = ""
    state.current_date = ""
    state.dig_src_path = ""
    state.dig_image_dir = ""
    state.dig_image_list = []
    state.dig_image_idx = -1
    state.dig_records = []
    state.dig_out_path = ""
    state.dig_resolution = 0.001
    state.dig_per_image = {}
    state.dig_scale_open = False
    state.dig_scale_draw = False
    state.dig_scale_library = []
    state.dig_scale_segments = {}
    state.dig_scale_result_unit = ""
    state.dig_detect_scale = "1/2"
    state.dig_detect_running = False
    state.dig_mask_detector = None
    state.dig_detect_threshold = 0.70
    state.dig_mode = "polygon"
    state.dig_active_polygon = []
    state.dig_autosave = True
    state.dig_zoom = 1.0
    state.devicemode = "CPU"
    state.det_model = "maskrcnn"
    state.dig_detect_runs = {}
    state.dig_figures_running = False
    state.dig_csv_loaded = set()
    state.dig_csv_owned = set()
    state.dig_label_set = "truth"
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def _label(e):
    lab = str((getattr(e, "_props", {}) or {}).get("label", "") or "")
    if not lab and e.__class__.__name__ in ("Checkbox", "Button", "Toggle",
                                             "Switch", "Expansion"):
        lab = str(getattr(e, "text", "") or "")
    return lab


def _kind(e):
    return e.__class__.__name__


def _classes(e):
    return list(getattr(e, "_classes", []) or [])


def _texts(els):
    out = []
    for e in els:
        k = _kind(e)
        t = getattr(e, "content", None) if k == "Markdown" else (
            getattr(e, "text", None) if k == "Label" else None)
        if isinstance(t, str) and t.strip():
            out.append(t)
    return out


def _build(clean):
    """The tab in a column; walk it again after anything that rebuilds a
    part (a photograph switch, a refreshable, a result card)."""
    from nicegui import ui
    from gui.app import build_digitize_tab
    with ui.column() as col:
        build_digitize_tab()
    clean[1].append(col)
    return col


def _photo(path, w=60, h=40, seed=2):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))
    return path


@pytest.fixture
def project(tmp_path):
    """P/validation/images holds a GSD-tagged photograph (where Digitize
    opens by default); P/images a plain one and one with a GSD sidecar;
    P/detect a large plain one and P/detect_gsd a large tagged one."""
    import functions.layout as layout
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    root = tmp_path / "P"
    root.mkdir()
    tagged = layout.project_path("P", "validation") / "images"
    _photo(tagged / "q_GSD=0.002m.jpg")
    plain = layout.project_path("P", "images")
    _photo(plain / "a.jpg")
    _photo(plain / "s.jpg")
    (plain / "s.jpg.json").write_text(json.dumps({"PM_GSD": 0.0015}),
                                      encoding="utf-8")
    detect = root / "detect"
    _photo(detect / "e.jpg", w=400, h=300, seed=5)
    detect_gsd = root / "detect_gsd"
    _photo(detect_gsd / "d_GSD=0.002m.jpg", w=400, h=300, seed=6)
    try:
        yield {"root": root, "tagged": tagged, "plain": plain,
               "detect": detect, "detect_gsd": detect_gsd}
    finally:
        layout.set_datasets_root(original, persist=False)


def _expansion(col):
    return next(e for e in _walk(col) if _kind(e) == "Expansion"
                and _label(e) == "No GSD? Scale from an object")


def _mouse(col):
    img = next(e for e in _walk(col) if _kind(e) == "InteractiveImage")
    listener = [l for l in img._event_listeners.values() if l.type == "mouse"][0]

    def send(kind, x, y):
        from nicegui.events import GenericEventArguments, handle_event
        handle_event(listener.handler, GenericEventArguments(
            sender=img, client=img.client,
            args={"mouse_event_type": kind, "image_x": x, "image_y": y}))
    return img, send


def _svg(img):
    """Everything drawn on the canvas: the image's own content and its layers."""
    parts = [img.content or ""]
    for child in img.default_slot.children:
        parts.append(getattr(child, "content", "") or "")
    return "".join(parts)


def _press(btn):
    from nicegui.events import ClickEventArguments, handle_event
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))


def _one(col, pred):
    return next(e for e in _walk(col) if pred(e))


def _switch(col):
    return _one(col, lambda e: _kind(e) == "Switch" and _label(e) == "Draw scale segment")


def _status(col):
    return _one(col, lambda e: "pm-dig-scale-status" in _classes(e))


def _info(col):
    return _one(col, lambda e: "pm-dig-status" in _classes(e))


def _select_image(col, name):
    _one(col, lambda e: _label(e) == "Image" and _kind(e) == "Select").value = name


SECTION_CONTROLS = ("Name", "Known length", "Unit", "Add object",
                    "Save to project", "Draw scale segment", "Result unit",
                    "Clear segments", "Detection scale")


# --------------------------------------------------------------------------- #
#  Present, closed on a GSD, open without one; nothing of it elsewhere        #
# --------------------------------------------------------------------------- #
def test_the_section_is_closed_on_a_gsd_photograph_and_open_without(clean, project):
    from nicegui import binding
    state, _ = clean
    state.current_project = "P"
    col = _build(clean)
    els = _walk(col)
    assert Path(state.dig_image_dir) == project["tagged"]
    assert state.dig_image_list == ["q_GSD=0.002m.jpg"]
    exp = _expansion(col)
    binding._refresh_step()
    assert state.dig_scale_open is False and exp.value is False
    assert "pm-dig-scale" in _classes(exp)
    # The section holds exactly the object-scale controls, and the tab
    # shows none of them outside it.
    inside = _walk(exp)
    labels_inside = [_label(e) for e in inside]
    for want in SECTION_CONTROLS:
        assert want in labels_inside, f"{want!r} missing from the section"
    outside = [_label(e) for e in els if e not in inside]
    for c in SECTION_CONTROLS:
        assert c not in outside, f"{c!r} shows outside the section"
    caption = next(t for t in _texts(inside) if t.startswith("Mark an object"))
    assert caption == ("Mark an object of known length on the photograph; "
                       "sizes are then indicative (no perspective or lens "
                       "correction).")
    det = next(e for e in inside if _label(e) == "Detection scale")
    assert [o["label"] for o in det._props["options"]] == ["Auto", "1", "1/2", "1/3", "1/4"]
    assert det.value == state.dig_detect_scale
    assert any(t.startswith("Auto: half size for Mask R-CNN") for t in _texts(inside))
    # The default library, one object of unknown size, is on show.
    assert state.dig_scale_library == [{"name": "scale bar", "length": None,
                                        "unit": None}]
    name = next(e for e in inside if _label(e) == "Name")
    assert name.value == "scale bar" and name._props.get("placeholder") == "e.g. boot width"
    # The section sits with the inputs, before the sticky bar.
    i_exp = els.index(exp)
    i_res = next(i for i, e in enumerate(els) if _label(e) == "Resolution (m/pixel)")
    i_bar = next(i for i, e in enumerate(els) if "pm-sticky-toolbar" in _classes(e))
    assert i_res < i_exp < i_bar, (i_res, i_exp, i_bar)
    # The GSD is the scale in force, from the file name; the Resolution
    # field took it at once.
    assert "Scale: GSD 2.000 mm/px (file name)" in _info(col).text
    assert "not metres" not in _info(col).text
    assert state.dig_resolution == pytest.approx(0.002)

    # Closed by default, on a plain photograph too; opened, it stays open.
    inp = next(e for e in els if _label(e) == "Image directory")
    inp.value = str(project["plain"])
    assert state.dig_image_list == ["a.jpg", "s.jpg"]
    assert state.dig_src_path.endswith("a.jpg")
    binding._refresh_step()
    assert state.dig_scale_open is False and exp.value is False
    # The tagged photograph's 0.002 does not follow a plain one: an
    # automatic value is emptied, so the line says so.
    assert state.dig_resolution is None
    assert "Scale: no scale" in _info(col).text
    state.dig_scale_open = True
    _select_image(col, "s.jpg")
    assert state.dig_src_path.endswith("s.jpg")
    binding._refresh_step()
    assert state.dig_scale_open is True and exp.value is True
    assert "Scale: GSD 1.500 mm/px (sidecar)" in _info(col).text
    assert state.dig_resolution == pytest.approx(0.0015)
    _select_image(col, "a.jpg")
    binding._refresh_step()
    assert state.dig_scale_open is True and exp.value is True
    # Nothing was written by looking.
    assert not (project["root"] / "output_results" / "gauge").exists()
    assert not (project["root"] / "scaling_objects.json").exists()


# --------------------------------------------------------------------------- #
#  The toggle and the two-click segment                                        #
# --------------------------------------------------------------------------- #
def test_the_toggle_switches_the_canvas_mode_and_two_clicks_make_a_segment(clean, project):
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["plain"])
    col = _build(clean)
    assert state.dig_src_path.endswith("a.jpg")
    img, send = _mouse(col)
    sw = _switch(col)
    assert sw.value is False and state.dig_scale_draw is False
    assert _status(col).text.startswith("No scale segment yet")
    # Before the toggle, a click is a polygon vertex.
    send("click", 10.0, 10.0)
    assert state.dig_active_polygon == [[10, 10]]
    send("click", 20.0, 10.0)
    assert len(state.dig_active_polygon) == 2 and not state.dig_scale_segments.get("a.jpg")
    state.dig_active_polygon = []

    sw.value = True
    assert state.dig_scale_draw is True
    assert "Scale segment: click the first end" in _info(col).text
    assert state.dig_zoom == pytest.approx(1.0)
    send("click", 5.0, 20.0)
    assert state.dig_scale_segments.get("a.jpg", []) == []
    assert state.dig_active_polygon == [] and state.dig_records == []
    svg = _svg(img)
    # The first click: a 7 px filled circle with a white halo (zoom 1).
    assert 'r="7.0" fill="#ff3030" stroke="white" stroke-width="3.0"' in svg, svg
    assert "first end set" in _info(col).text
    # The band follows the cursor.
    send("mousemove", 30.0, 22.0)
    svg = _svg(img)
    assert 'points="5.0,20.0 30.0,22.0"' in svg and "stroke-dasharray" in svg, svg
    send("click", 45.0, 20.0)
    assert state.dig_scale_segments["a.jpg"] == [
        {"p0": [5, 20], "p1": [45, 20], "object": "scale bar"}]
    assert state.dig_records == [], "a segment is not a clast record"
    svg = _svg(img)
    assert "stroke-dasharray" not in svg, "the band goes with the commit"
    # The committed segment: a 4 px line over a 10 px white halo, filled
    # endpoints, the object's name in a white-backed box at the middle,
    # rotated by 0 about the midpoint (25, 20).
    assert 'stroke="white" stroke-width="10.0"' in svg, svg
    assert 'stroke="#ff8800" stroke-width="4.0"' in svg, svg
    assert svg.count('r="5.0" fill="#ff8800"') == 2, svg
    m = re.search(r'<g transform="rotate\((-?[\d.]+) ([\d.]+) ([\d.]+)\)">'
                  r'(<rect [^>]*/><text [^>]*>scale bar</text>)</g>', svg)
    assert m, svg
    assert [float(v) for v in m.groups()[:3]] == [0.0, 25.0, 20.0]
    # The status line, the assignment row, the unit, the info line.
    assert _status(col).text.startswith("1 segment: scale bar, 40 px"), _status(col).text
    assert "scale = 40.0 px per scale bar" in _status(col).text
    assert "Segment 1" in _texts(_walk(col))
    obj_sel = _one(col, lambda e: _label(e) == "Scaling object")
    assert obj_sel.value == "scale bar"
    assert state.dig_scale_result_unit == "scale bar"
    unit_sel = _one(col, lambda e: _label(e) == "Result unit")
    assert unit_sel.value == "scale bar"
    assert "Scale: object scale, 40.0 px per scale bar" in _info(col).text
    assert "the truth CSV is in scale bar units, not metres" in _info(col).text
    # A second, longer segment: the spread is reported and the tilt named.
    send("click", 5.0, 30.0)
    send("click", 50.0, 30.0)
    assert len(state.dig_scale_segments["a.jpg"]) == 2
    assert _status(col).text.startswith("2 segments: scale bar, 40 px; scale bar, 45 px")
    assert "segments disagree by 11.8 %" in _status(col).text
    assert "tilted" in _status(col).text and "text-warning" in _classes(_status(col))
    # Leaving the toggle returns the canvas to the clast mode it was in.
    sw.value = False
    assert state.dig_scale_draw is False
    send("click", 12.0, 12.0)
    assert state.dig_active_polygon == [[12, 12]]
    assert len(state.dig_scale_segments["a.jpg"]) == 2
    assert "polygon" in _info(col).text
    # The segments stay drawn while the clasts are edited.
    assert _svg(img).count('r="5.0" fill="#ff8800"') == 4
    # Clear segments: the GSD-less photograph has no scale again.
    _press(_one(col, lambda e: _label(e) == "Clear segments"))
    assert state.dig_scale_segments["a.jpg"] == []
    assert _status(col).text.startswith("No scale segment yet")
    # … and no boot default stands in for it.
    assert "Scale: no scale" in _info(col).text
    assert 'stroke="#ff8800"' not in _svg(img)
    # Still nothing on disk: no geometry, no run, no library.
    assert not list(project["plain"].glob("*.geojson"))
    assert not (project["root"] / "output_results" / "gauge").exists()
    assert not (project["root"] / "scaling_objects.json").exists()


def test_the_segments_are_kept_per_photograph(clean, project):
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["plain"])
    col = _build(clean)
    img, send = _mouse(col)
    _switch(col).value = True
    send("click", 5.0, 20.0)
    send("click", 45.0, 20.0)
    _select_image(col, "s.jpg")
    assert state.dig_src_path.endswith("s.jpg")
    img, send = _mouse(col)
    assert _status(col).text.startswith("No scale segment yet")
    assert 'stroke="#ff8800"' not in _svg(img)
    _select_image(col, "a.jpg")
    img, send = _mouse(col)
    assert _status(col).text.startswith("1 segment: scale bar, 40 px")
    assert 'stroke="#ff8800" stroke-width="4.0"' in _svg(img)


def test_escape_drops_the_first_end(clean, project):
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["plain"])
    col = _build(clean)
    img, send = _mouse(col)
    _switch(col).value = True
    send("click", 5.0, 20.0)
    assert 'fill="#ff3030"' in _svg(img)
    kb = _one(col, lambda e: _kind(e) == "Keyboard")
    from nicegui.events import GenericEventArguments, handle_event
    state.active_tab = "Digitize"
    for l in kb._event_listeners.values():
        if l.type == "key":
            handle_event(l.handler, GenericEventArguments(
                sender=kb, client=kb.client,
                args={"action": "keydown", "key": "Escape", "code": "Escape",
                      "repeat": False, "altKey": False, "ctrlKey": False,
                      "metaKey": False, "shiftKey": False, "location": 0}))
    assert 'fill="#ff3030"' not in _svg(img)
    assert state.dig_scale_segments.get("a.jpg", []) == []
    assert "click the first end" in _info(col).text


def test_the_bold_label_lies_along_the_segment_and_always_reads_left_to_right(
        clean, project):
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["plain"])
    col = _build(clean)
    img, send = _mouse(col)
    _switch(col).value = True

    def rotations(svg):
        return [(float(a), float(x), float(y)) for a, x, y in re.findall(
            r'<g transform="rotate\((-?[\d.]+) ([\d.]+) ([\d.]+)\)">'
            r'<rect [^>]*/><text [^>]*text-anchor="middle"[^>]*>scale bar</text></g>',
            svg)]

    # Down-right on screen (y grows downward): a 30 px by 30 px diagonal
    # is rotated +45 deg clockwise (SVG's positive sense) about its middle.
    send("click", 5.0, 5.0)
    send("click", 35.0, 35.0)
    assert rotations(_svg(img)) == [(45.0, 20.0, 20.0)], _svg(img)
    # The same diagonal drawn right to left would be upside down at 225
    # deg: it is flipped to the same readable 45 deg.
    send("click", 45.0, 35.0)
    send("click", 15.0, 5.0)
    assert rotations(_svg(img))[1] == (45.0, 30.0, 20.0), _svg(img)
    # Up-right on screen: -45 (counter-clockwise), also readable.
    send("click", 5.0, 38.0)
    send("click", 25.0, 18.0)
    assert rotations(_svg(img))[2] == (-45.0, 15.0, 28.0), _svg(img)


def test_the_library_loads_from_the_project_and_saves_back(clean, project):
    Q.save_library(project["root"], [Q.ScalingObject("boot width", 0.27, "m"),
                                     Q.ScalingObject("hammer length")])
    state, _ = clean
    state.current_project = "P"
    col = _build(clean)
    assert [d["name"] for d in state.dig_scale_library] == ["boot width", "hammer length"]
    names = [e for e in _walk(col) if _label(e) == "Name"]
    assert [n.value for n in names] == ["boot width", "hammer length"]
    _press(_one(col, lambda e: _label(e) == "Add object"))
    assert [d["name"] for d in state.dig_scale_library] == \
        ["boot width", "hammer length", "object"]
    _press(_one(col, lambda e: _label(e) == "Save to project"))
    back = Q.load_library(project["root"])
    assert [o.name for o in back] == ["boot width", "hammer length", "object"]
    assert back[0].metres == pytest.approx(0.27)
    # A metric-known object: the segment gives millimetres, the truth
    # stays in metres, the unit select offers the physical units.
    _one(col, lambda e: _label(e) == "Image directory").value = str(project["plain"])
    img, send = _mouse(col)
    _switch(col).value = True
    send("click", 5.0, 20.0)
    send("click", 45.0, 20.0)
    assert state.dig_scale_segments["a.jpg"][0]["object"] == "boot width"
    assert state.dig_scale_result_unit == "mm"
    assert "scale = 0.1 px per mm" in _status(col).text
    assert "1 px = 6.750 mm" in _status(col).text
    assert "Scale: object scale, 0.1 px per mm" in _info(col).text
    assert "not metres" not in _info(col).text
    unit_sel = _one(col, lambda e: _label(e) == "Result unit")
    assert [o["label"] for o in unit_sel._props["options"]] == ["mm", "cm", "m", "in", "ft"]


# --------------------------------------------------------------------------- #
#  Detect: the run through functions.gauge, in both scale cases               #
# --------------------------------------------------------------------------- #
def _disc_detector(seen):
    """Three discs across the middle of whatever image it is handed."""
    def detect(image, min_confidence, devicemode, devicenumber):
        seen.append((image.shape, min_confidence, devicemode))
        h, w = image.shape[:2]
        r = max(4, int(round(0.06 * min(h, w))))
        yy, xx = np.mgrid[0:h, 0:w]
        masks = [((xx - fx * w) ** 2 + (yy - 0.5 * h) ** 2) <= r * r
                 for fx in (0.25, 0.5, 0.75)]
        return np.dstack(masks), np.array([0.9, 0.8, 0.95])
    return detect


def _wait_detect(state, timeout=180.0):
    t0 = time.time()
    while state.dig_detect_running and time.time() - t0 < timeout:
        time.sleep(0.2)
    assert not state.dig_detect_running, "detection did not finish"


def _detect_button(col):
    return _one(col, lambda e: _kind(e) == "Button" and _label(e) == "Detect")


def _result_cards(col):
    results = _one(col, lambda e: "pm-dig-results" in _classes(e))
    return [e for e in _walk(results) if "pm-dig-result" in _classes(e)]


def _figures_button(col):
    return _one(col, lambda e: _kind(e) == "Button" and _label(e) == "Figures")


def _figures(col, state, timeout=180.0):
    """Press Figures and wait for the result card."""
    btn = _figures_button(col)
    assert btn.enabled, "Figures is disabled"
    _press(btn)
    t0 = time.time()
    while state.dig_figures_running and time.time() - t0 < timeout:
        time.sleep(0.2)
    assert not state.dig_figures_running, "the figures did not finish"


def test_detect_with_a_gsd_writes_millimetres_without_disclaimer_and_proposes(
        clean, project):
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["detect_gsd"])
    state.dig_detect_scale = "1"
    seen = []
    state.dig_mask_detector = _disc_detector(seen)
    col = _build(clean)
    assert state.dig_src_path.endswith("d_GSD=0.002m.jpg")
    assert "Scale: GSD 2.000 mm/px (file name)" in _info(col).text
    assert _figures_button(col).enabled is False, "Figures with no clast"
    btn = _detect_button(col)
    _press(btn)
    _wait_detect(state)
    assert btn.enabled is True
    assert seen == [((300, 400, 3), 0.7, "cpu")], seen
    out = project["root"] / "output_results" / "gauge"
    stem = "P__d_GSD=0.002m"
    csv, side = out / f"{stem}_gauge.csv", out / f"{stem}_gauge.json"
    # Detect loads proposals and writes no figure, no CSV, no card.
    assert not csv.exists() and not side.exists()
    assert not (out / f"{stem}_gauge_overlay.png").exists()
    assert _result_cards(col) == []
    # The proposals: three editable polygon records with their scores and
    # the model that made them.
    assert len(state.dig_records) == 3
    assert all(r["shape"] == "polygon" and r["mask"].shape == (300, 400)
               for r in state.dig_records)
    assert sorted(r["score"] for r in state.dig_records) == [0.8, 0.9, 0.95]
    assert {r["origin"] for r in state.dig_records} == {"maskrcnn"}
    assert _figures_button(col).enabled is True
    _figures(col, state)
    assert csv.exists() and side.exists()
    assert (out / f"{stem}_gauge_overlay.png").exists()
    assert (out / f"{stem}_gauge_distribution.png").exists()
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert meta["scale_source"] == "gsd:filename"
    assert meta["disclaimer"] is None
    assert meta["scale"]["mode"] == "metric" and meta["scale"]["unit"] == "mm"
    assert meta["scale"]["metres_per_px"] == pytest.approx(0.002)
    assert meta["scale"]["segments"] == []
    assert meta["models"][0]["detect_scale"] == 1.0
    assert meta["origins"] == {"maskrcnn": 3} and meta["edited"] == 0
    df = pd.read_csv(csv)
    assert len(df) == 3 and (df["unit"] == "mm").all()
    # Discs of radius 18 px: 36 px across, 2 mm per pixel.
    np.testing.assert_allclose(df["Clast_length"], 72.0, rtol=0.08)
    # The result card under the table: both figures, no disclaimer.
    cards = _result_cards(col)
    assert len(cards) == 1
    inside = _walk(cards[0])
    assert sum(1 for e in inside if _kind(e) == "Image") == 2
    texts = _texts(inside)
    assert any(t.startswith(f"{stem} — n = 3, D50 = ") and "mm" in t for t in texts)
    assert any(t.endswith(f"{stem}_gauge.csv") for t in texts)
    assert "Detections: maskrcnn, 3 kept" in texts
    assert Q.DISCLAIMER not in texts
    # The proposals are the model's own label set, autosaved beside the
    # truth, in metres as usual; Copy to truth makes them the truth.
    val = project["root"] / "validation"
    labels = pd.read_csv(val / "d_GSD=0.002m_labels=maskrcnn.csv")
    assert len(labels) == 3 and "unit" not in labels.columns
    np.testing.assert_allclose(labels["Clast_length"], 0.072, rtol=0.08)
    assert not (val / "d_GSD=0.002m_truth.csv").exists()
    _press(_button(col, "Copy to truth"))
    assert state.dig_label_set == "truth"
    truth = pd.read_csv(val / "d_GSD=0.002m_truth.csv")
    assert len(truth) == 3
    np.testing.assert_allclose(truth["Clast_length"], 0.072, rtol=0.08)


def test_detect_with_an_object_scale_writes_the_unit_and_the_disclaimer(
        clean, project):
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["detect"])
    state.dig_detect_scale = "1/2"
    seen = []
    state.dig_mask_detector = _disc_detector(seen)
    col = _build(clean)
    assert state.dig_src_path.endswith("e.jpg") and state.dig_scale_open is False
    img, send = _mouse(col)
    _switch(col).value = True
    # Clear of the three discs (row 150), so none is removed.
    send("click", 100.0, 40.0)
    send("click", 200.0, 40.0)           # 100 px = 1 scale bar
    assert _status(col).text.startswith("1 segment: scale bar, 100 px")
    _press(_detect_button(col))
    _wait_detect(state)
    # Detection ran on the half-size copy; the proposals come back in
    # original pixels, then Figures measures them in the unit.
    assert seen == [((150, 200, 3), 0.7, "cpu")], seen
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    stem = "P__e"
    csv, side = out / f"{stem}_gauge.csv", out / f"{stem}_gauge.json"
    assert csv.exists() and side.exists()
    assert (out / f"{stem}_gauge_overlay.png").exists()
    assert (out / f"{stem}_gauge_distribution.png").exists()
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert meta["scale_source"] == "object"
    assert meta["disclaimer"] == Q.DISCLAIMER
    assert meta["scale"]["mode"] == "custom" and meta["scale"]["unit_label"] == "scale bar"
    assert meta["scale"]["px_per_unit"] == pytest.approx(100.0)
    assert meta["scale"]["segments"] == [
        {"p0": [100.0, 40.0], "p1": [200.0, 40.0], "object": "scale bar",
         "px_length": 100.0}]
    assert meta["excluded_by_segments"] == 0
    assert meta["models"][0]["detect_scale"] == 0.5
    df = pd.read_csv(csv)
    assert len(df) == 3 and (df["unit"] == "scale bar").all()
    # Discs of radius 9 px on the half copy: 36 original px = 0.36 scale bar.
    np.testing.assert_allclose(df["Clast_length"], 0.36, rtol=0.12)
    # Proposals at full size, from the half-size masks.
    assert len(state.dig_records) == 3
    assert all(r["mask"].shape == (300, 400) for r in state.dig_records)
    assert all(r["mask"].sum() > 800 for r in state.dig_records)
    cards = _result_cards(col)
    assert len(cards) == 1
    texts = _texts(_walk(cards[0]))
    assert Q.DISCLAIMER in texts
    assert any(t.startswith(f"{stem} — n = 3, D50 = 0.3") and "scale bar" in t
               for t in texts)
    # The library was saved with the run; the set's CSV is in the unit.
    assert (project["root"] / "scaling_objects.json").exists()
    truth = pd.read_csv(project["root"] / "validation" / "e_labels=maskrcnn.csv")
    assert len(truth) == 3 and (truth["unit"] == "scale bar").all()
    np.testing.assert_allclose(truth["Clast_length"], 0.36, rtol=0.12)
    assert "the truth CSV is in scale bar units, not metres" in _info(col).text
    # Export writes the same table where it is told to.
    state.dig_out_path = str(project["root"] / "validation" / "e2_truth.csv")
    _press(_one(col, lambda e: _kind(e) == "Button" and _label(e) == "Export CSV"))
    back = pd.read_csv(state.dig_out_path)
    assert (back["unit"] == "scale bar").all() and len(back) == 3


# --------------------------------------------------------------------------- #
#  Twelve tabs, none of them Gauge                                             #
# --------------------------------------------------------------------------- #
def test_twelve_tabs_and_no_gauge_tab(src):
    strip = src[src.index('ui.tab("Overview", icon="home")'):]
    strip = strip[:strip.index("with ui.tab_panels")]
    tabs = re.findall(r'ui\.tab\("([A-Za-z]+)"', strip)
    assert tabs == ["Overview", "Express", "Orthorectify", "Detect", "Merge",
                    "Rasterize", "Map", "Zonal", "Georeference", "Digitize",
                    "Validate", "Report"]
    assert '"Gauge"' not in src and "build_gauge_tab" not in src
    assert "gauge_canvas" not in src and "gauge_library" not in src
    lines = src[src.index("_TAB_LINES = ["):]
    lines = lines[:lines.index("]\n")]
    assert lines.count("(\"") == 11 and "straighten" not in lines
    assert not (APP.parents[1] / "functions" / "quick.py").exists()
    body = src[src.index("def build_digitize_tab("):]
    body = body[:body.index("\ndef ")]
    assert 'ui.expansion("No GSD? Scale from an object"' in body
    assert "_gauge.run_gauge(" in body and "_gauge.write_gauge_figures(" in body
    assert "GaugeScale.from_gsd(" in body


# --------------------------------------------------------------------------- #
#  The model chosen in the drawer drives Digitize's Detect                     #
# --------------------------------------------------------------------------- #
def _fake_backend(name="fakeseg", display="Fake segmenter"):
    """An in-process backend: three discs as an instances file, measured by
    the shared step, CSV only (the wrapper persists the outlines)."""
    from detectors.base import BackendInfo, DetectorBackend, output_csv_path

    class Fake(DetectorBackend):
        info = BackendInfo(name=name, display_name=display,
                           framework="classical", license="MIT")
        calls = []

        def is_available(self):
            return True

        def detect_jobs(self, mode, jobs, **kw):
            from PIL import Image
            from detectors import measure, subprocess_runner as SR
            out = []
            for job in jobs:
                self.calls.append((job["path"], kw.get("resolution")))
                csv = output_csv_path(mode, job, kw)
                csv.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(job["path"]) as im:
                    w, h = im.size
                r = max(4, int(round(0.06 * min(h, w))))
                yy, xx = np.mgrid[0:h, 0:w]
                lab = np.zeros((h, w), np.int32)
                for k, fx in enumerate((0.25, 0.5, 0.75), start=1):
                    lab[((xx - fx * w) ** 2 + (yy - 0.5 * h) ** 2) <= r * r] = k
                npz = SR.instances_path_for(csv)
                np.savez(npz, labels=lab,
                         scores=np.array([0.9, 0.8, 0.95], np.float32),
                         shape=np.array([h, w]))
                df = measure.measure_instances(SR.read_instances_npz(npz),
                                               float(kw["resolution"]), height=h)
                df.to_csv(csv, index=False)
                out.append(df)
            return out
    return Fake()


def test_detect_runs_the_drawer_model_and_names_it(clean, project, monkeypatch):
    from detectors import registry
    from functions import clast_geometry as CG
    fake = _fake_backend()
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", fake)
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["detect_gsd"])
    state.dig_detect_scale = "1/2"
    state.det_model = "fakeseg"
    col = _build(clean)
    _press(_detect_button(col))
    _wait_detect(state)
    # The backend ran on the half-size copy, in pixel units.
    assert len(fake.calls) == 1 and fake.calls[0][1] == 2.0
    assert fake.calls[0][0].endswith("_gauge_detect.jpg")
    assert {r["origin"] for r in state.dig_records} == {"Fake segmenter"}
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    stem = "P__d_GSD=0.002m"
    meta = json.loads((out / f"{stem}_gauge.json").read_text(encoding="utf-8"))
    assert meta["models"][0]["backend"] == "fakeseg"
    assert meta["models"][0]["display_name"] == "Fake segmenter"
    assert meta["models"][0]["license"] == "MIT"
    assert meta["contours"] == f"{stem}_gauge.contours.json"
    df = pd.read_csv(out / f"{stem}_gauge.csv")
    assert len(df) == 3
    np.testing.assert_allclose(df["Clast_length"], 72.0, rtol=0.1)
    contours = CG.read_contours(out / f"{stem}_gauge.csv")
    assert len(contours) == 3
    # The backend's scratch folder is gone; only the result stays.
    assert not (out / "_detect_work").exists()
    # Proposals from its instance masks, at full size, agreeing with the
    # outlines written beside the CSV.
    assert len(state.dig_records) == 3
    from functions.digitize import polygon_to_mask
    for rec in state.dig_records:
        assert rec["mask"].shape == (300, 400) and rec["mask"].sum() > 800
    for cid, arr in contours.items():
        ring = [[int(round(x)), int(round(300 - y))] for x, y in arr]
        m = polygon_to_mask(ring, (300, 400)).astype(bool)
        best = max((m & r["mask"]).sum() / (m | r["mask"]).sum()
                   for r in state.dig_records)
        assert best > 0.85, (cid, best)
    cards = _result_cards(col)
    texts = _texts(_walk(cards[0]))
    assert "Model: Fake segmenter" in texts


def test_detect_with_mask_rcnn_names_it_and_writes_outlines(clean, project):
    from functions import clast_geometry as CG
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["detect_gsd"])
    state.dig_detect_scale = "1"
    seen = []
    state.dig_mask_detector = _disc_detector(seen)
    col = _build(clean)
    _press(_detect_button(col))
    _wait_detect(state)
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    stem = "P__d_GSD=0.002m"
    contours = CG.read_contours(out / f"{stem}_gauge.csv")
    assert contours is not None and len(contours) == 3
    texts = _texts(_walk(_result_cards(col)[0]))
    assert any(t.startswith("Model: ") for t in texts)


def test_the_drawer_offers_the_model_without_binding_and_hides_dev_backends(
        clean, monkeypatch):
    from nicegui import ui
    from detectors import registry
    from gui import app as A
    state, built = clean
    monkeypatch.delenv(registry.DEV_BACKENDS_ENV, raising=False)
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", _fake_backend())
    stub = registry.get_backend("stub")
    monkeypatch.setattr(type(stub), "is_available", lambda self: True)
    state.det_model = "stub"                  # hidden: falls back at build
    with ui.column() as col:
        A.render_detection_settings()
    built.append(col)
    sel = _one(col, lambda e: _kind(e) == "Select" and _label(e) == "Detection model")
    assert state.det_model == "maskrcnn" and sel.value == "maskrcnn"
    assert "stub" not in sel.options and "fakeseg" in sel.options
    assert not getattr(sel, "_props", {}).get("model-value-binding")
    sel.value = "fakeseg"
    assert state.det_model == "fakeseg"
    # A select on another page whose options lack the value cannot null it.
    state.det_model = "maskrcnn"
    assert sel.value == "fakeseg"
    monkeypatch.setenv(registry.DEV_BACKENDS_ENV, "1")
    with ui.column() as col2:
        A.render_detection_settings()
    built.append(col2)
    sel2 = _one(col2, lambda e: _kind(e) == "Select" and _label(e) == "Detection model")
    assert "stub" in sel2.options


# --------------------------------------------------------------------------- #
#  Reload model lives in the drawer, under the model select                    #
# --------------------------------------------------------------------------- #
def test_the_drawer_reload_button_clears_the_selected_backends_cache(
        clean, monkeypatch):
    from nicegui import ui
    from detectors import registry
    from gui import app as A
    state, built = clean
    fake = _fake_backend()
    cleared = []
    monkeypatch.setattr(type(fake), "clear_cache",
                        lambda self: cleared.append(self.info.name) or False,
                        raising=False)
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", fake)
    notes = []
    monkeypatch.setattr(A.ui, "notify", lambda msg, **kw: notes.append((msg, kw)))
    state.det_model = "fakeseg"
    with ui.column() as col:
        A.render_detection_settings()
    built.append(col)
    els = _walk(col)
    i_sel = next(i for i, e in enumerate(els)
                 if _kind(e) == "Select" and _label(e) == "Detection model")
    i_btn = next(i for i, e in enumerate(els)
                 if _kind(e) == "Button" and _label(e) == "Reload model")
    assert i_sel < i_btn
    btn = els[i_btn]
    assert btn._props.get("icon") == "refresh"
    assert "outline" in btn._props and "dense" in btn._props
    status = _one(col, lambda e: "pm-model-status" in _classes(e))
    assert status.text.startswith("Model: ")
    _press(btn)
    assert cleared == ["fakeseg"]
    assert notes[-1][0] == ("Fake segmenter loads its weights at every run; "
                            "nothing to reload.")
    # Mask R-CNN: its backend's clear_cache drops the cached model.
    mr = registry.get_backend("maskrcnn")
    hits = []
    import functions.clasts_detection as CD
    monkeypatch.setattr(CD, "clear_model_cache", lambda: hits.append(1))
    state.det_model = "maskrcnn"
    assert mr.clear_cache() is True and hits == [1]
    _press(btn)
    assert hits == [1, 1]
    assert notes[-1][0] == "Mask R-CNN weights will reload on the next run."
    assert notes[-1][1].get("type") == "positive"
    # The base contract: a backend that caches nothing reloads nothing.
    from detectors.base import DetectorBackend
    assert DetectorBackend.clear_cache(fake) is False


def test_detect_express_and_digitize_carry_no_reload_button(clean, project):
    from nicegui import ui
    from gui import app as A
    state, built = clean
    state.current_project = "P"
    for builder in (A.build_detection_tab, A.build_express_tab,
                    A.build_digitize_tab):
        with ui.column() as col:
            builder()
        built.append(col)
        els = _walk(col)
        assert not any("Reload" in _label(e) for e in els), builder.__name__
        cap = [e for e in els if "pm-model-caption" in _classes(e)]
        assert cap, f"{builder.__name__} lost its model caption"


# --------------------------------------------------------------------------- #
#  Detect drops the detections a scale segment crosses                         #
# --------------------------------------------------------------------------- #
def test_detect_drops_the_clasts_a_segment_crosses_from_outputs_and_proposals(
        clean, project):
    from functions import clast_geometry as CG
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["detect"])
    state.dig_detect_scale = "1"
    seen = []
    state.dig_mask_detector = _disc_detector(seen)
    col = _build(clean)
    img, send = _mouse(col)
    _switch(col).value = True
    # Discs of radius 18 at columns 100, 200, 300 on row 150: a vertical
    # segment through the middle one.
    send("click", 205.0, 120.0)
    send("click", 205.0, 185.0)
    _press(_detect_button(col))
    _wait_detect(state)
    # The proposals leave the removed disc out.
    assert len(state.dig_records) == 2
    cx = sorted(r["centroid_x"] for r in state.dig_records)
    assert cx[0] == pytest.approx(100, abs=2) and cx[1] == pytest.approx(300, abs=2)
    assert state.dig_detect_runs["e.jpg"][0]["excluded_by_segments"] == 1
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    stem = "P__e"
    meta = json.loads((out / f"{stem}_gauge.json").read_text(encoding="utf-8"))
    assert meta["excluded_by_segments"] == 1 and meta["n"] == 2
    df = pd.read_csv(out / f"{stem}_gauge.csv")
    assert len(df) == 2
    np.testing.assert_allclose(sorted(df["x"]), [100, 300], atol=2)
    assert len(CG.read_contours(out / f"{stem}_gauge.csv")) == 2
    texts = _texts(_walk(_result_cards(col)[0]))
    assert any(t.startswith(f"{stem} — n = 2, D50 = ")
               and t.endswith("; 1 crossing a scale segment removed")
               for t in texts), texts


def test_detect_with_a_backend_drops_crossed_instance_proposals(
        clean, project, monkeypatch):
    from detectors import registry
    fake = _fake_backend()
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", fake)
    state, _ = clean
    state.current_project = "P"
    state.dig_image_dir = str(project["detect"])
    state.dig_detect_scale = "1/2"
    state.det_model = "fakeseg"
    col = _build(clean)
    img, send = _mouse(col)
    _switch(col).value = True
    send("click", 95.0, 120.0)
    send("click", 95.0, 185.0)
    _press(_detect_button(col))
    _wait_detect(state)
    assert len(state.dig_records) == 2
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    meta = json.loads((out / "P__e_gauge.json").read_text(encoding="utf-8"))
    assert meta["excluded_by_segments"] == 1 and meta["n"] == 2
    assert min(r["centroid_x"] for r in state.dig_records) > 150


# --------------------------------------------------------------------------- #
#  Select, Delete, Undo; Figures on demand; the clast table; provenance;       #
#  the sample set                                                              #
# --------------------------------------------------------------------------- #
MRCNN = "Mask R-CNN (Soloy et al., 2020)"


def _disc_rec(cx, cy, r, shape=(300, 400), origin="hand", **kw):
    from functions.digitize import circle_to_mask
    rec = {"shape": "circle", "params": {"center": [cx, cy], "radius": r},
           "mask": circle_to_mask((cx, cy), r, shape).astype(bool),
           "centroid_x": float(cx), "centroid_y": float(cy), "label": "",
           "origin": origin, "edited": False}
    rec.update(kw)
    return rec


def _table(col):
    return _one(col, lambda e: "pm-dig-table" in _classes(e))


def _button(col, label):
    return _one(col, lambda e: _kind(e) == "Button" and _label(e) == label)


def _ck(folder, name):
    """The record cache's key: the photograph's full path (a same-named
    photograph of another folder must not share it)."""
    import os
    return os.path.normcase(os.path.abspath(os.path.join(str(folder), name)))


def _seed(state, project, recs, folder="detect_gsd", name="d_GSD=0.002m.jpg"):
    state.current_project = "P"
    state.dig_image_dir = str(project[folder])
    state.dig_per_image = {_ck(project[folder], name): {"records": list(recs), "out_path": ""}}


def _clear_all(col):
    """Clear all, confirmed in its dialog."""
    _press(_button(col, "Clear all"))
    _press(_button(col, "Clear canvas"))


def _key(col, name):
    """A keydown through the Delete / Backspace keyboard."""
    from nicegui.events import GenericEventArguments, handle_event
    kb = _one(col, lambda e: _kind(e) == "Keyboard"
              and e._props.get("ignore") == ["input", "select", "textarea"])
    for l in kb._event_listeners.values():
        if l.type == "key":
            handle_event(l.handler, GenericEventArguments(
                sender=kb, client=kb.client,
                args={"action": "keydown", "key": name, "code": name,
                      "repeat": False, "altKey": False, "ctrlKey": False,
                      "metaKey": False, "shiftKey": False, "location": 0}))
    return kb


def test_select_mode_picks_the_smallest_clast_under_the_click(clean, project):
    state, _ = clean
    big = _disc_rec(200, 150, 60)
    small = _disc_rec(210, 150, 15, origin=MRCNN, score=0.9)
    far = _disc_rec(60, 60, 20)
    _seed(state, project, [big, small, far])
    col = _build(clean)
    assert len(state.dig_records) == 3
    modes = [e for e in _walk(col) if "pm-dig-mode" in _classes(e)]
    assert [_label(e) for e in modes] == ["Select", "Polygon", "Circle", "Ellipse"]
    _press(_button(col, "Select"))
    assert state.dig_mode == "select"
    img, send = _mouse(col)
    table = _table(col)
    delete = _button(col, "Delete")
    assert delete.enabled is False, "Delete with no selection"
    # Inside both discs: the small one wins.
    send("click", 212.0, 152.0)
    assert [r["id"] for r in table.selected] == [2]
    assert _svg(img).count('class="pm-dig-selected"') == 1
    assert delete.enabled is True
    # Inside the big one only.
    send("click", 160.0, 150.0)
    assert [r["id"] for r in table.selected] == [1]
    # Bare ground clears; no record was added by clicking.
    send("click", 380.0, 280.0)
    assert table.selected == [] and "pm-dig-selected" not in _svg(img)
    assert delete.enabled is False and len(state.dig_records) == 3
    # Leaving Select mode drops the selection.
    send("click", 60.0, 60.0)
    assert [r["id"] for r in table.selected] == [3]
    _press(_button(col, "Polygon"))
    assert state.dig_mode == "polygon"
    assert table.selected == []


def test_a_table_row_selects_the_clast_on_the_canvas(clean, project):
    from nicegui.events import GenericEventArguments, handle_event
    state, _ = clean
    _seed(state, project, [_disc_rec(100, 100, 20), _disc_rec(300, 200, 25)])
    col = _build(clean)
    img, _send = _mouse(col)
    table = _table(col)
    for l in table._event_listeners.values():
        if l.type == "rowClick":
            handle_event(l.handler, GenericEventArguments(
                sender=table, client=table.client, args=[{}, {"id": 2}, 1]))
    assert [r["id"] for r in table.selected] == [2]
    assert 'class="pm-dig-selected"' in _svg(img)
    # The selection checkbox path does the same.
    for l in table._event_listeners.values():
        if l.type == "selection":
            handle_event(l.handler, GenericEventArguments(
                sender=table, client=table.client,
                args={"added": True, "rows": [table.rows[0]], "keys": [1]}))
    assert sorted(r["id"] for r in table.selected) == [1, 2]
    assert _button(col, "Delete").enabled is True
    label = _one(col, lambda e: "pm-dig-label" in _classes(e))
    label.value = "flat one"
    assert state.dig_records[0]["label"] == "flat one"


def test_delete_by_button_and_key_and_undo_restores(clean, project):
    state, _ = clean
    recs = [_disc_rec(100, 100, 20), _disc_rec(200, 100, 20, origin=MRCNN, score=0.8),
            _disc_rec(300, 100, 20)]
    _seed(state, project, recs)
    state.dig_mode = "select"
    col = _build(clean)
    img, send = _mouse(col)
    table = _table(col)
    truth = project["root"] / "validation" / "d_GSD=0.002m_truth.csv"
    send("click", 200.0, 100.0)
    _press(_button(col, "Delete"))
    assert [r["origin"] for r in state.dig_records] == ["hand", "hand"]
    assert len(table.rows) == 2 and table.selected == []
    assert len(pd.read_csv(truth)) == 2, "autosave after the deletion"
    # Undo puts it back where it was.
    _press(_button(col, "Undo"))
    assert [r["origin"] for r in state.dig_records] == ["hand", MRCNN, "hand"]
    assert len(table.rows) == 3 and len(pd.read_csv(truth)) == 3
    # The Delete key, on this tab, with a selection.
    state.active_tab = "Digitize"
    send("click", 300.0, 100.0)
    kb = _key(col, "Delete")
    assert len(state.dig_records) == 2
    # The browser side ignores keys typed into text fields.
    assert kb._props["ignore"] == ["input", "select", "textarea"]
    # Another tab, or no selection: the key does nothing.
    send("click", 100.0, 100.0)
    state.active_tab = "Detect"
    _key(col, "Backspace")
    assert len(state.dig_records) == 2
    state.active_tab = "Digitize"
    _key(col, "Backspace")
    assert len(state.dig_records) == 1
    _key(col, "Delete")
    assert len(state.dig_records) == 1
    # Undo twice restores both, in order.
    _press(_button(col, "Undo"))
    _press(_button(col, "Undo"))
    assert [r["centroid_x"] for r in state.dig_records] == [100.0, 200.0, 300.0]


def test_figures_build_from_the_records_after_a_deletion(clean, project, monkeypatch):
    state, _ = clean
    recs = [_disc_rec(100, 150, 18, origin=MRCNN, score=0.9),
            _disc_rec(200, 150, 18, origin=MRCNN, score=0.8),
            _disc_rec(300, 150, 18)]
    recs[0]["edited"] = True
    _seed(state, project, recs)
    state.dig_detect_runs = {"d_GSD=0.002m.jpg": [
        {"backend": "maskrcnn", "display_name": MRCNN, "detect_scale": 0.5,
         "min_confidence": 0.7, "excluded_by_segments": 0}]}
    seen = {}
    real = Q.write_gauge_figures

    def spy(image_path, df, scale, out_dir, stem, **kw):
        seen["n"] = len(df)
        seen["x"] = sorted(df["x"])
        seen["contours"] = len(kw.get("contours") or {})
        seen["note"] = kw.get("note")
        return real(image_path, df, scale, out_dir, stem, **kw)
    monkeypatch.setattr(Q, "write_gauge_figures", spy)
    state.dig_mode = "select"
    col = _build(clean)
    assert _figures_button(col).enabled is True
    img, send = _mouse(col)
    send("click", 200.0, 150.0)
    _press(_button(col, "Delete"))
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    stem = "P__d_GSD=0.002m"
    df = pd.read_csv(out / f"{stem}_gauge.csv")
    assert len(df) == 2
    np.testing.assert_allclose(sorted(df["x"]), [100, 300], atol=1)
    assert seen["n"] == 2 and seen["contours"] == 2
    np.testing.assert_allclose(seen["x"], [100, 300], atol=1)
    assert seen["note"] == f"Detections: {MRCNN}, 1 kept, 1 edited; 1 drawn by hand"
    meta = json.loads((out / f"{stem}_gauge.json").read_text(encoding="utf-8"))
    assert meta["origins"] == {"hand": 1, MRCNN: 1} and meta["edited"] == 1
    assert meta["provenance"] == seen["note"]
    prov = json.loads((out / f"{stem}_gauge.provenance.json").read_text(encoding="utf-8"))
    assert prov["clasts"] == {"1": {"origin": MRCNN, "edited": True},
                              "2": {"origin": "hand", "edited": False}}
    texts = _texts(_walk(_result_cards(col)[0]))
    assert f"Model: {MRCNN}" in texts and seen["note"] in texts
    # With every clast deleted, Figures is disabled.
    _clear_all(col)
    assert _figures_button(col).enabled is False


def test_the_table_lists_every_record_in_the_unit_in_force(clean, project):
    from functions.digitize import circle_to_mask
    state, _ = clean
    masks = {}
    recs = []
    for k in range(1500):
        cx, cy = 30 + (k % 12) * 30, 30 + (k // 12 % 8) * 30
        if (cx, cy) not in masks:
            masks[(cx, cy)] = circle_to_mask((cx, cy), 10, (300, 400)).astype(bool)
        recs.append({"shape": "circle", "params": {"center": [cx, cy], "radius": 10},
                     "mask": masks[(cx, cy)], "centroid_x": float(cx),
                     "centroid_y": float(cy), "label": "",
                     "origin": MRCNN if k % 3 else "hand", "edited": k == 4,
                     "score": 0.9})
    _seed(state, project, recs)
    state.dig_autosave = False
    col = _build(clean)
    table = _table(col)
    assert len(table.rows) == 1500
    assert [r["id"] for r in table.rows[:3]] == [1, 2, 3]
    assert table.rows[-1]["id"] == 1500
    labels = [c["label"] for c in table.columns]
    assert labels == ["ID", "Origin", "Clast length (mm)", "Clast width (mm)",
                      "Equivalent diameter (mm)", "Area (mm²)", "Orientation (°)",
                      "Score"]
    assert all(c.get("sortable") for c in table.columns)
    assert table._props["pagination"]["rowsPerPage"] == 50
    # Discs of radius 10 px at 2 mm per pixel.
    r0 = table.rows[0]
    assert r0["origin"] == "hand" and table.rows[1]["origin"] == MRCNN
    assert table.rows[4]["origin"] == f"{MRCNN} (edited)"
    assert r0["eqd"] == pytest.approx(40.0, rel=0.05)
    assert r0["area"] == pytest.approx(math.pi * 100 * 4, rel=0.08)
    count = _one(col, lambda e: "pm-dig-table-count" in _classes(e))
    assert count.text.startswith("1,500 clasts")
    # It follows a deletion at once, and stays quick at this size.
    t0 = time.time()
    _press(_button(col, "Undo"))
    assert time.time() - t0 < 3.0
    assert len(table.rows) == 1499


def test_origins_survive_a_save_and_a_reopen(clean, tmp_path, project):
    import functions.layout as layout
    state, _ = clean
    folder = layout.project_path("P", "images") / "pair"
    _photo(folder / "a_GSD=0.002m.jpg", w=400, h=300, seed=7)
    _photo(folder / "b_GSD=0.002m.jpg", w=400, h=300, seed=8)
    recs = [_disc_rec(100, 100, 20, origin=MRCNN, score=0.9),
            _disc_rec(250, 150, 30, origin=MRCNN, score=0.7, edited=True),
            _disc_rec(330, 220, 25)]
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    state.dig_per_image = {_ck(folder, "a_GSD=0.002m.jpg"): {"records": recs, "out_path": ""}}
    state.dig_detect_runs = {"a_GSD=0.002m.jpg": [
        {"backend": "maskrcnn", "display_name": MRCNN, "detect_scale": 0.5}]}
    col = _build(clean)
    assert state.dig_src_path.endswith("a_GSD=0.002m.jpg")
    _press(_button(col, "Export CSV"))
    truth = Path(state.dig_out_path)
    side = truth.with_name(truth.stem + ".provenance.json")
    doc = json.loads(side.read_text(encoding="utf-8"))
    assert doc["clasts"] == {"1": {"origin": MRCNN, "edited": False},
                             "2": {"origin": MRCNN, "edited": True},
                             "3": {"origin": "hand", "edited": False}}
    assert doc["models"][0]["display_name"] == MRCNN
    # Reopen from disk (a restart): away, forget the session, back, Load.
    _select_image(col, "b_GSD=0.002m.jpg")
    state.dig_per_image.pop(_ck(folder, "a_GSD=0.002m.jpg"), None)
    state.dig_csv_loaded.clear()
    state.dig_csv_owned.clear()
    state.dig_detect_runs.clear()
    _select_image(col, "a_GSD=0.002m.jpg")
    assert _button(col, "Load").enabled is False, "loaded on opening"
    assert [r["origin"] for r in state.dig_records] == [MRCNN, MRCNN, "hand"]
    assert [bool(r["edited"]) for r in state.dig_records] == [False, True, False]
    assert state.dig_detect_runs["a_GSD=0.002m.jpg"][0]["display_name"] == MRCNN
    assert [r["origin"] for r in _table(col).rows] == [MRCNN, f"{MRCNN} (edited)", "hand"]
    # A CSV with no sidecar: every clast is hand-drawn.
    side.unlink()
    _select_image(col, "b_GSD=0.002m.jpg")
    state.dig_per_image.pop(_ck(folder, "a_GSD=0.002m.jpg"), None)
    state.dig_csv_loaded.clear()
    _select_image(col, "a_GSD=0.002m.jpg")
    assert {r["origin"] for r in state.dig_records} == {"hand"}


def test_sample_set_pools_the_saved_photographs_of_the_folder(clean, project):
    import functions.layout as layout
    state, _ = clean
    folder = layout.project_path("P", "images") / "set"
    _photo(folder / "a_GSD=0.002m.jpg", w=400, h=300, seed=7)
    _photo(folder / "b_GSD=0.001m.jpg", w=400, h=300, seed=8)
    _photo(folder / "c_GSD=0.001m.jpg", w=400, h=300, seed=9)
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    state.dig_per_image = {
        _ck(folder, "a_GSD=0.002m.jpg"): {"records": [_disc_rec(100, 100, 20), _disc_rec(250, 150, 30)],
                                          "out_path": ""},
        _ck(folder, "b_GSD=0.001m.jpg"): {"records": [_disc_rec(100, 100, 10, origin=MRCNN),
                                                      _disc_rec(200, 100, 15, origin=MRCNN),
                                                      _disc_rec(300, 200, 25)], "out_path": ""}}
    state.dig_autosave = False
    col = _build(clean)
    btn = _button(col, "Sample set")
    assert btn._props.get("icon") == "stacked_bar_chart"
    assert btn.enabled is True, "the open photograph has clasts"
    _clear_all(col)
    assert btn.enabled is False, "no saved clasts anywhere"
    _select_image(col, "b_GSD=0.001m.jpg")
    _press(_button(col, "Export CSV"))
    assert btn.enabled is True
    state.dig_per_image[_ck(folder, "a_GSD=0.002m.jpg")] = {
        "records": [_disc_rec(100, 100, 20), _disc_rec(250, 150, 30)], "out_path": ""}
    _select_image(col, "a_GSD=0.002m.jpg")
    assert len(state.dig_records) == 2
    # a's records are not saved (autosave off): used from the canvas.
    _press(btn)
    out = project["root"] / "output_results" / "gauge"
    stem = "P__set"
    for name in (f"{stem}_sampleset.csv", f"{stem}_sampleset_summary.csv",
                 f"{stem}_sampleset_distribution.png"):
        assert (out / name).exists(), name
    pooled = pd.read_csv(out / f"{stem}_sampleset.csv")
    assert set(pooled["photo"]) == {"a_GSD=0.002m.jpg", "b_GSD=0.001m.jpg"}
    assert len(pooled) == 5 and (pooled["unit"] == "mm").all()
    summary = pd.read_csv(out / f"{stem}_sampleset_summary.csv").set_index("photo")
    assert summary.loc["(pooled)", "n"] == 5
    assert summary.loc["(pooled)", "D50"] == pytest.approx(pooled["Clast_length"].median())
    # b's disc of radius 25 px at 1 mm/px and a's of radius 30 at 2 mm/px.
    assert pooled["Equivalent_diameter"].max() == pytest.approx(120, rel=0.05)
    cards = [e for e in _walk(_one(col, lambda e: "pm-dig-results" in _classes(e)))
             if "pm-dig-sampleset" in _classes(e)]
    assert len(cards) == 1
    inside = _walk(cards[0])
    texts = _texts(inside)
    assert any(t.startswith("Sample set — set: 2 photographs, n = 5, D50 = ") for t in texts)
    assert any("autosave is off" in t for t in texts)
    assert any(t.startswith("1 photograph of the folder without clasts") for t in texts)
    assert sum(1 for e in inside if _kind(e) == "Image") == 1
    tbl = next(e for e in inside if "pm-dig-sampleset-summary" in _classes(e))
    assert [r["photo"] for r in tbl.rows] == ["a_GSD=0.002m.jpg", "b_GSD=0.001m.jpg", "(pooled)"]
    # The table of the open photograph sits above the result cards.
    els = _walk(col)
    i_table = next(i for i, e in enumerate(els) if "pm-dig-table" in _classes(e))
    i_results = next(i for i, e in enumerate(els) if "pm-dig-results" in _classes(e))
    assert i_table < i_results


def test_the_results_show_the_open_photograph_and_the_sample_set_only(clean, project):
    import functions.layout as layout
    state, _ = clean
    folder = layout.project_path("P", "images") / "set"
    _photo(folder / "a_GSD=0.002m.jpg", w=400, h=300, seed=7)
    _photo(folder / "b_GSD=0.001m.jpg", w=400, h=300, seed=8)
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    state.dig_per_image = {
        _ck(folder, "a_GSD=0.002m.jpg"): {"records": [_disc_rec(100, 100, 20), _disc_rec(250, 150, 30)],
                                          "out_path": ""},
        _ck(folder, "b_GSD=0.001m.jpg"): {"records": [_disc_rec(100, 100, 10)], "out_path": ""}}
    col = _build(clean)
    _select_image(col, "a_GSD=0.002m.jpg")

    def kinds():
        return ["sample" if "pm-dig-sampleset" in _classes(c) else "photo"
                for c in _result_cards(col)]
    _figures(col, state)
    _figures(col, state)
    assert kinds() == ["photo"], "a new Figures replaces the card"
    _press(_button(col, "Sample set"))
    _press(_button(col, "Sample set"))
    assert kinds() == ["photo", "sample"], "one sample set, under the photograph"
    _select_image(col, "b_GSD=0.001m.jpg")
    assert kinds() == ["sample"], "another photograph drops the previous figures"
    _figures(col, state)
    assert kinds() == ["photo", "sample"]
    texts = _texts(_walk(_result_cards(col)[0]))
    assert any(t.startswith("P__b_GSD=0.001m — n = 1") for t in texts), texts


# --------------------------------------------------------------------------- #
#  Saved clasts wait for Load; segments and outlines come back after a restart #
# --------------------------------------------------------------------------- #
def _restart(clean):
    """What a relaunch forgets, then the tab built again."""
    from gui.app import state
    state.dig_per_image = {}
    state.dig_records = []
    state.dig_csv_loaded = set()
    state.dig_csv_owned = set()
    state.dig_scale_segments = {}
    state.dig_scale_library = []
    state.dig_detect_runs = {}
    state.dig_image_idx = -1
    state.dig_scale_draw = False
    return _build(clean)


def _iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return (a & b).sum() / (a | b).sum()


def test_segments_objects_and_outlines_come_back_after_a_restart(clean, project):
    import functions.layout as layout
    from functions.digitize import circle_to_mask
    state, _ = clean
    folder = layout.project_path("P", "images") / "restart"
    _photo(folder / "r.jpg", w=400, h=300, seed=11)
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    col = _build(clean)
    assert state.dig_scale_open is False, "closed by default"
    img, send = _mouse(col)
    _switch(col).value = True
    send("click", 100.0, 40.0)
    send("click", 200.0, 40.0)            # 100 px = 30 mm
    _switch(col).value = False
    name = next(e for e in _walk(col) if _label(e) == "Name")
    name.value = "Boot"
    next(e for e in _walk(col) if _label(e) == "Known length").value = 30.0
    next(e for e in _walk(col) if _label(e) == "Unit").value = "mm"
    lib = json.loads((project["root"] / "scaling_objects.json").read_text(encoding="utf-8"))
    assert {"name": "Boot", "length": 30.0, "unit": "mm"} in lib, "saved without a click"
    state.dig_records = [_disc_rec(120, 150, 25), _disc_rec(300, 180, 40)]
    _press(_button(col, "Export CSV"))
    truth = Path(state.dig_out_path)
    scale_json = truth.with_name(truth.stem + ".scale.json")
    doc = json.loads(scale_json.read_text(encoding="utf-8"))
    assert doc["segments"] == [{"p0": [100.0, 40.0], "p1": [200.0, 40.0], "object": "Boot"}]
    assert doc["objects"] == [{"name": "Boot", "length": 30.0, "unit": "mm"}]
    assert truth.with_name(truth.stem + ".contours.json").exists()

    col = _restart(clean)
    assert state.dig_src_path.endswith("r.jpg")
    assert state.dig_scale_segments["r.jpg"] == doc["segments"]
    assert any(d["name"] == "Boot" for d in state.dig_scale_library)
    assert _status(col).text.startswith("1 segment: Boot, 100 px")
    assert _button(col, "Load").enabled is False, "loaded on opening"
    assert [r["shape"] for r in state.dig_records] == ["polygon", "polygon"]
    for rec, (cx, cy, r) in zip(state.dig_records, [(120, 150, 25), (300, 180, 40)]):
        assert _iou(rec["mask"], circle_to_mask((cx, cy), r, (300, 400))) > 0.9

    # Without the scale (the sidecar lost), the outlines still come back at
    # their true size: they are pixels, not the fitted ellipse in metres.
    scale_json.unlink()
    col = _restart(clean)
    assert "r.jpg" not in state.dig_scale_segments or not state.dig_scale_segments["r.jpg"]
    assert len(state.dig_records) == 2
    for rec, (cx, cy, r) in zip(state.dig_records, [(120, 150, 25), (300, 180, 40)]):
        assert _iou(rec["mask"], circle_to_mask((cx, cy), r, (300, 400))) > 0.9


def test_autosave_never_replaces_saved_clasts_that_are_not_loaded(clean, project):
    import functions.layout as layout
    state, _ = clean
    folder = layout.project_path("P", "images") / "guard"
    _photo(folder / "a_GSD=0.002m.jpg", w=400, h=300, seed=7)
    _photo(folder / "b_GSD=0.002m.jpg", w=400, h=300, seed=8)
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    state.dig_per_image = {_ck(folder, "a_GSD=0.002m.jpg"): {
        "records": [_disc_rec(100, 100, 20), _disc_rec(250, 150, 30)], "out_path": ""}}
    col = _build(clean)
    _press(_button(col, "Export CSV"))
    truth = Path(state.dig_out_path)
    assert len(pd.read_csv(truth)) == 2

    col = _restart(clean)
    assert len(state.dig_records) == 2, "the saved clasts load on opening"
    # Clear all asks, unloads, and nothing comes back by itself.
    _press(_button(col, "Clear all"))
    assert len(state.dig_records) == 2, "not before the confirmation"
    _press(_button(col, "Clear canvas"))
    assert state.dig_records == []
    assert len(pd.read_csv(truth)) == 2, "the file is untouched"
    _select_image(col, "b_GSD=0.002m.jpg")
    _select_image(col, "a_GSD=0.002m.jpg")
    assert state.dig_records == []
    assert _button(col, "Load").enabled is True
    # A clast drawn on the cleared canvas leaves the unloaded file alone.
    state.dig_mode = "circle"
    img, send = _mouse(col)
    send("click", 330.0, 220.0)
    send("click", 350.0, 220.0)
    assert len(state.dig_records) == 1, "a clast drawn by hand"
    assert len(pd.read_csv(truth)) == 2, "autosave left the saved clasts alone"
    _press(_button(col, "Load"))
    assert len(state.dig_records) == 3
    assert len(pd.read_csv(truth)) == 3, "loaded, autosave writes both"


# --------------------------------------------------------------------------- #
#  The quadrat frame: a dashed guide on the canvas, its band left out          #
# --------------------------------------------------------------------------- #
def _framed(folder, name, w=60, h=40, gsd=0.001, thickness=0.005, seed=9):
    """A rectified photograph whose record puts ``thickness`` m of frame
    along each edge (5 px + the 2 px margin by default)."""
    p = _photo(folder / name, w=w, h=h, seed=seed)
    (folder / (name + ".json")).write_text(json.dumps(
        {"PM_RECTIFIED": "yes", "PM_GSD": f"{gsd:.6f}",
         "PM_FRAME_THICKNESS_M": f"{thickness:.4f}"}), encoding="utf-8")
    return p


def _log_lines(log):
    lines = (getattr(log, "_props", {}) or {}).get("lines")
    if isinstance(lines, str):
        return lines.split("\n")
    if isinstance(lines, (list, tuple)):
        return [str(l) for l in lines]
    return [str(getattr(e, "text", "")) for e in _walk(log)]


def test_the_canvas_draws_the_inner_rectangle_of_a_framed_photograph(clean, project):
    state, _ = clean
    _framed(project["plain"], "f.jpg")
    state.current_project = "P"
    state.dig_image_dir = str(project["plain"])
    col = _build(clean)
    _select_image(col, "f.jpg")
    img, send = _mouse(col)
    send("mousemove", 10.0, 10.0)
    svg = _svg(img)
    assert 'class="pm-dig-frame"' in svg, svg
    assert '<rect x="7" y="7" width="46" height="26" fill="none"' in svg, svg
    assert "stroke-dasharray" in svg
    # A click still draws a polygon vertex: the guide is not a tool.
    send("click", 20.0, 20.0)
    assert state.dig_active_polygon == [[20, 20]]
    # A plain photograph has no frame, so no guide.
    _select_image(col, "a.jpg")
    img, send = _mouse(col)
    send("mousemove", 10.0, 10.0)
    assert "pm-dig-frame" not in _svg(img)


def test_detect_leaves_the_frame_band_out_for_a_plug_in_model(clean, project, monkeypatch):
    """The fake backend puts three discs across the middle (columns 100,
    200 and 300 of 400); a 0.25 m frame at 2 mm/px is 125 px (+ 12 px of
    margin), so only the middle disc is inside: run_gauge drops the other
    two, the proposals on the canvas agree, and the log says so."""
    from detectors import registry
    fake = _fake_backend()
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", fake)
    state, _ = clean
    (project["detect_gsd"] / "d_GSD=0.002m.jpg.json").write_text(json.dumps(
        {"PM_GSD": "0.002000", "PM_FRAME_THICKNESS_M": "0.2500"}),
        encoding="utf-8")
    state.current_project = "P"
    state.dig_image_dir = str(project["detect_gsd"])
    state.dig_detect_scale = "1/2"
    state.det_model = "fakeseg"
    col = _build(clean)
    img, send = _mouse(col)
    send("mousemove", 10.0, 10.0)
    assert '<rect x="137" y="137" width="126" height="26"' in _svg(img)
    _press(_detect_button(col))
    _wait_detect(state)
    assert len(fake.calls) == 1
    assert len(state.dig_records) == 1, "the two discs on the band are gone"
    from functions.digitize import mask_centroid
    cx, cy = mask_centroid(state.dig_records[0]["mask"])
    assert abs(cx - 200) < 3 and abs(cy - 150) < 3, (cx, cy)
    log = _one(col, lambda e: _kind(e) == "Log")
    assert any("2 on the quadrat frame removed" in l for l in _log_lines(log)), \
        _log_lines(log)[-5:]
    _figures(col, state)
    out = project["root"] / "output_results" / "gauge"
    df = pd.read_csv(out / "P__d_GSD=0.002m_gauge.csv")
    assert len(df) == 1


# --------------------------------------------------------------------------- #
#  Selection on a crowded canvas: handles, several clasts, the layers         #
# --------------------------------------------------------------------------- #
def _send_mods(img, kind, x, y, **mods):
    """A mouse event with modifier keys or buttons, as the browser sends it."""
    from nicegui.events import GenericEventArguments, handle_event
    listener = [l for l in img._event_listeners.values() if l.type == "mouse"][0]
    args = {"mouse_event_type": kind, "image_x": x, "image_y": y}
    args.update(mods)
    handle_event(listener.handler, GenericEventArguments(
        sender=img, client=img.client, args=args))


def _layers(img):
    return [c for c in img.default_slot.children if hasattr(c, "content")]


def _ids(table):
    return sorted(r["id"] for r in table.selected)


def test_select_is_apart_from_the_drawing_tools(clean, project):
    state, _ = clean
    _seed(state, project, [_disc_rec(100, 100, 20)])
    col = _build(clean)
    group = _one(col, lambda e: "pm-dig-draw-tools" in _classes(e))
    inside = [_label(e) for e in _walk(group) if _kind(e) == "Button"]
    assert inside == ["Polygon", "Circle", "Ellipse"], "Select is not a drawing tool"
    _press(_button(col, "Circle"))
    assert state.dig_mode == "circle"
    _press(_button(col, "Select"))
    assert state.dig_mode == "select"


def test_handles_show_only_on_the_one_selected_clast(clean, project):
    from functions.digitize import polygon_to_mask
    state, _ = clean
    sq = [[80, 80], [120, 80], [120, 120], [80, 120]]
    poly = {"shape": "polygon", "params": {"vertices": [list(v) for v in sq]},
            "mask": polygon_to_mask(sq, (300, 400)).astype(bool),
            "centroid_x": 100.0, "centroid_y": 100.0, "label": "",
            "origin": "hand", "edited": False}
    _seed(state, project, [poly, _disc_rec(300, 200, 25)])
    col = _build(clean)
    img, send = _mouse(col)
    table = _table(col)
    assert 'fill="orange"' not in _svg(img), "no handle with nothing selected"
    # In a drawing mode the vertices of a saved clast neither show nor drag.
    send("mousedown", 120.0, 120.0)
    send("mousemove", 150.0, 150.0)
    send("mouseup", 150.0, 150.0)
    assert state.dig_records[0]["params"]["vertices"][2] == [120, 120]
    _press(_button(col, "Select"))
    send("click", 100.0, 100.0)
    assert _svg(img).count('fill="orange"') == 4, "the polygon's four vertices"
    # A handle of the selected clast drags it.
    send("mousedown", 120.0, 120.0)
    send("mousemove", 140.0, 130.0)
    send("mouseup", 140.0, 130.0)
    send("click", 140.0, 130.0)       # the browser's click after the drag
    assert state.dig_records[0]["params"]["vertices"][2] == [140, 130]
    assert _ids(table) == [1], "the drag kept the selection"
    # Two selected: no handles.
    _send_mods(img, "click", 300.0, 200.0, shiftKey=True)
    assert _ids(table) == [1, 2]
    assert 'fill="orange"' not in _svg(img)


def test_a_box_selects_the_clasts_inside_and_one_undo_restores_a_deletion(
        clean, project):
    state, _ = clean
    recs = [_disc_rec(60, 60, 15), _disc_rec(120, 60, 15),
            _disc_rec(300, 200, 20), _disc_rec(340, 250, 15)]
    _seed(state, project, recs)
    col = _build(clean)
    img, send = _mouse(col)
    table = _table(col)
    _press(_button(col, "Select"))
    # A drag from bare ground draws a band and selects what is inside it.
    send("mousedown", 30.0, 30.0)
    _send_mods(img, "mousemove", 160.0, 100.0, buttons=1)
    assert "pm-dig-band" in _svg(img)
    _send_mods(img, "mouseup", 160.0, 100.0)
    send("click", 160.0, 100.0)       # the browser's click after the drag
    assert _ids(table) == [1, 2]
    assert "pm-dig-band" not in _svg(img)
    assert _svg(img).count('class="pm-dig-selected"') == 2
    # Shift keeps the selection and adds a second box.
    _send_mods(img, "mousedown", 275.0, 175.0, shiftKey=True)
    _send_mods(img, "mousemove", 370.0, 280.0, buttons=1, shiftKey=True)
    _send_mods(img, "mouseup", 370.0, 280.0, shiftKey=True)
    _send_mods(img, "click", 370.0, 280.0, shiftKey=True)
    assert _ids(table) == [1, 2, 3, 4]
    # Shift + click takes one out again.
    _send_mods(img, "click", 120.0, 60.0, shiftKey=True)
    assert _ids(table) == [1, 3, 4]
    # A plain click replaces the selection; bare ground clears it.
    send("click", 300.0, 200.0)
    assert _ids(table) == [3]
    send("click", 200.0, 150.0)
    assert _ids(table) == []
    # Delete removes every selected clast, and one Undo puts them all back.
    send("mousedown", 30.0, 30.0)
    _send_mods(img, "mousemove", 390.0, 290.0, buttons=1)
    _send_mods(img, "mouseup", 390.0, 290.0)
    send("click", 390.0, 290.0)
    _send_mods(img, "click", 120.0, 60.0, ctrlKey=True)
    assert _ids(table) == [1, 3, 4]
    _press(_button(col, "Delete"))
    assert [r["centroid_x"] for r in state.dig_records] == [120.0]
    assert table.selected == []
    _press(_button(col, "Undo"))
    assert [r["centroid_x"] for r in state.dig_records] == [60.0, 120.0, 300.0, 340.0]


def test_the_label_goes_to_every_selected_clast(clean, project):
    state, _ = clean
    _seed(state, project, [_disc_rec(60, 60, 15), _disc_rec(120, 60, 15),
                           _disc_rec(300, 200, 20)])
    col = _build(clean)
    img, send = _mouse(col)
    _press(_button(col, "Select"))
    send("click", 60.0, 60.0)
    _send_mods(img, "click", 120.0, 60.0, shiftKey=True)
    label = _one(col, lambda e: "pm-dig-label" in _classes(e))
    label.value = "frame"
    assert [r["label"] for r in state.dig_records] == ["frame", "frame", ""]


def test_a_selection_click_does_not_resend_the_outlines(clean, project):
    state, _ = clean
    _seed(state, project, [_disc_rec(60, 60, 15), _disc_rec(120, 60, 15),
                           _disc_rec(300, 200, 20)])
    col = _build(clean)
    img, send = _mouse(col)
    outline, marks = _layers(img)
    assert outline.content.count("<path") == 3
    assert "pm-dig-selected" not in outline.content
    sent = []
    real = outline.update
    outline.update = lambda: (sent.append(1), real())[1]
    _press(_button(col, "Select"))
    send("click", 60.0, 60.0)
    _send_mods(img, "click", 120.0, 60.0, shiftKey=True)
    send("click", 200.0, 150.0)
    assert not sent, "selecting redraws the marks layer only"
    assert "<path" not in (img.content or ""), "the image itself stays light"
    send("click", 300.0, 200.0)
    assert "pm-dig-selected" in marks.content
    _press(_button(col, "Delete"))
    assert sent and outline.content.count("<path") == 2


def test_the_frame_can_be_set_by_hand_and_its_clasts_selected(clean, project):
    state, _ = clean
    side = project["detect_gsd"] / "d_GSD=0.002m.jpg.json"
    _seed(state, project, [_disc_rec(200, 150, 20), _disc_rec(10, 150, 8),
                           _disc_rec(390, 20, 8)])
    col = _build(clean)
    field = _one(col, lambda e: "pm-dig-frame-field" in _classes(e))
    btn = _one(col, lambda e: "pm-dig-frame-select" in _classes(e))
    assert field.enabled is True, "the photograph has a GSD"
    assert field.value is None and btn.enabled is False
    field.value = 4.0          # 4 cm at 2 mm/px: 20 px of bar + 2 px margin
    assert json.loads(side.read_text(encoding="utf-8"))["PM_FRAME_THICKNESS_M"] == "0.0400"
    img, _send = _mouse(col)
    assert 'class="pm-dig-frame"' in _svg(img)
    assert btn.enabled is True and btn.text == "On the frame (2)"
    _press(btn)
    assert state.dig_mode == "select"
    assert _ids(_table(col)) == [2, 3]
    _press(_button(col, "Delete"))
    assert [r["centroid_x"] for r in state.dig_records] == [200.0]
    assert btn.enabled is False
    field.value = None
    assert "PM_FRAME_THICKNESS_M" not in json.loads(side.read_text(encoding="utf-8"))
    assert "pm-dig-frame" not in _svg(img)


def test_a_handle_drag_takes_the_nearest_vertex_of_a_traced_outline(clean, project):
    from functions.digitize import polygon_to_mask
    state, _ = clean
    # A traced outline: vertices 4 px apart, several inside the grab zone.
    ring = [[100 + 4 * k, 100] for k in range(10)] + [[136, 120], [100, 120]]
    poly = {"shape": "polygon", "params": {"vertices": [list(v) for v in ring]},
            "mask": polygon_to_mask(ring, (300, 400)).astype(bool),
            "centroid_x": 118.0, "centroid_y": 110.0, "label": "",
            "origin": "hand", "edited": False}
    _seed(state, project, [poly])
    col = _build(clean)
    img, send = _mouse(col)
    _press(_button(col, "Select"))
    send("click", 118.0, 110.0)
    send("mousedown", 117.0, 100.0)    # nearest: vertex 4, at (116, 100)
    send("mousemove", 117.0, 90.0)
    send("mouseup", 117.0, 90.0)
    verts = state.dig_records[0]["params"]["vertices"]
    assert verts[4] == [117, 90] and verts[0] == [100, 100]
    # Handles are no wider than the spacing, so they stay apart.
    import re
    radii = {float(r) for r in re.findall(r'<circle [^>]*r="([0-9.]+)" fill="orange"', _svg(img))}
    assert radii and max(radii) <= 2.0, radii


# --------------------------------------------------------------------------- #
#  Label sets: a photograph's truth and one set per model, each an entry      #
# --------------------------------------------------------------------------- #
def _nav(col, icon):
    return _one(col, lambda e: _kind(e) == "Button"
                and (getattr(e, "_props", {}) or {}).get("icon") == icon)


def _entries(col):
    sel = _one(col, lambda e: _label(e) == "Image" and _kind(e) == "Select")
    return list(sel.options.values()) if isinstance(sel.options, dict) else list(sel.options)


def test_each_model_fills_its_own_label_set_and_the_list_walks_through_them(
        clean, project, monkeypatch):
    from detectors import registry
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", _fake_backend())
    monkeypatch.setitem(registry._BACKENDS, "fakegrain",
                        _fake_backend("fakegrain", "Fake grains"))
    state, _ = clean
    folder, name = project["detect_gsd"], "d_GSD=0.002m.jpg"
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    state.dig_detect_scale = "1"
    state.dig_per_image = {_ck(folder, name): {
        "records": [_disc_rec(50, 50, 10)], "out_path": ""}}
    col = _build(clean)
    assert _entries(col) == [name], "one entry while there is only the truth"
    for model in ("fakeseg", "fakegrain"):
        state.det_model = model
        _press(_detect_button(col))
        _wait_detect(state)
        assert state.dig_label_set == model and len(state.dig_records) == 3
    val = project["root"] / "validation"
    assert len(pd.read_csv(val / "d_GSD=0.002m_labels=fakeseg.csv")) == 3
    assert len(pd.read_csv(val / "d_GSD=0.002m_labels=fakegrain.csv")) == 3
    assert _entries(col) == [f"{name} · truth", f"{name} · Fake grains",
                             f"{name} · Fake segmenter"]
    # Previous and Next walk the entries; the truth kept its hand clast.
    _press(_nav(col, "chevron_left"))
    assert state.dig_label_set == "truth"
    assert [r["origin"] for r in state.dig_records] == ["hand"]
    _press(_nav(col, "chevron_right"))
    _press(_nav(col, "chevron_right"))
    assert state.dig_label_set == "fakeseg"
    assert {r["origin"] for r in state.dig_records} == {"Fake segmenter"}
    # An edit stays in its set, and is saved to its own file.
    img, send = _mouse(col)
    _press(_button(col, "Select"))
    send("click", 100.0, 150.0)
    _press(_button(col, "Delete"))
    assert len(pd.read_csv(val / "d_GSD=0.002m_labels=fakeseg.csv")) == 2
    _press(_nav(col, "chevron_left"))
    assert state.dig_label_set == "fakegrain" and len(state.dig_records) == 3
    _press(_nav(col, "chevron_right"))
    assert len(state.dig_records) == 2
    # After a restart the sets come back from their files.
    col = _restart(clean)
    assert f"{name} · Fake segmenter" in _entries(col)
    _select_image(col, f"{name}\tfakeseg")
    assert len(state.dig_records) == 2
    assert {r["origin"] for r in state.dig_records} == {"Fake segmenter"}


def test_copy_to_truth_adds_or_replaces_and_never_drops_saved_truth(
        clean, project, monkeypatch):
    from detectors import registry
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", _fake_backend())
    state, _ = clean
    folder, name = project["detect_gsd"], "d_GSD=0.002m.jpg"
    val = project["root"] / "validation"
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    state.dig_detect_scale = "1"
    state.dig_per_image = {_ck(folder, name): {
        "records": [_disc_rec(50, 50, 10), _disc_rec(350, 50, 10)], "out_path": ""}}
    col = _build(clean)
    _press(_button(col, "Export CSV"))
    assert len(pd.read_csv(val / "d_GSD=0.002m_truth.csv")) == 2
    assert _button(col, "Copy to truth").enabled is False, "the truth is open"
    # A restart: the saved truth is not in this session's memory any more.
    col = _restart(clean)
    state.det_model = "fakeseg"
    _press(_detect_button(col))
    _wait_detect(state)
    assert _button(col, "Copy to truth").enabled is True
    _press(_button(col, "Copy to truth"))
    assert state.dig_label_set == "fakeseg", "asks first: the truth holds clasts"
    _press(_button(col, "Add to them"))
    assert state.dig_label_set == "truth"
    assert [r["origin"] for r in state.dig_records] == ["hand"] * 2 + ["Fake segmenter"] * 3
    assert len(pd.read_csv(val / "d_GSD=0.002m_truth.csv")) == 5
    # Replace: only the set's clasts. The set itself is unchanged.
    _select_image(col, f"{name}\tfakeseg")
    _press(_button(col, "Copy to truth"))
    _press(_button(col, "Replace them"))
    assert len(pd.read_csv(val / "d_GSD=0.002m_truth.csv")) == 3
    assert len(pd.read_csv(val / "d_GSD=0.002m_labels=fakeseg.csv")) == 3
    # The copies are independent: reshaping the truth leaves the set alone.
    first = state.dig_records[0]["params"]["vertices"][0]
    was = list(first)
    first[0] += 5
    _select_image(col, f"{name}\tfakeseg")
    assert state.dig_records[0]["params"]["vertices"][0] == was


def test_a_detect_tab_output_is_a_label_set_until_edited(clean, project):
    import functions.layout as layout
    from detectors.base import run_detect_jobs
    state, _ = clean
    folder, name = project["detect_gsd"], "d_GSD=0.002m.jpg"
    vec = layout.project_path("P", "vectors")
    vec.mkdir(parents=True, exist_ok=True)
    fake = _fake_backend()
    run_detect_jobs(fake, "quadrat", [{"path": str(folder / name),
                                       "out_stem": "P__d_GSD=0.002m"}],
                    resolution=0.002, output_dir=str(vec))
    outputs = list(vec.glob("*d_GSD=0.002m*.csv"))
    assert len(outputs) == 1
    Path(str(outputs[0]) + ".manifest.json").write_text(
        json.dumps({"model": "fakeseg"}), encoding="utf-8")
    before = outputs[0].read_bytes()
    state.current_project = "P"
    state.dig_image_dir = str(folder)
    from detectors import registry
    registry._BACKENDS["fakeseg"] = fake
    try:
        col = _build(clean)
        assert _entries(col) == [f"{name} · truth", f"{name} · Fake segmenter"]
        _select_image(col, f"{name}\tfakeseg")
        assert len(state.dig_records) == 3
        assert {r["origin"] for r in state.dig_records} == {"Fake segmenter"}
        img, send = _mouse(col)
        _press(_button(col, "Select"))
        send("click", 200.0, 150.0)
        _press(_button(col, "Delete"))
        labels = project["root"] / "validation" / "d_GSD=0.002m_labels=fakeseg.csv"
        assert len(pd.read_csv(labels)) == 2, "an edited output is saved as the set"
        assert outputs[0].read_bytes() == before, "the Detect output is untouched"
    finally:
        registry._BACKENDS.pop("fakeseg", None)
