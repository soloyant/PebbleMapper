"""The shared canvas conforms to the layout model once, for Zonal and Detection-ROI.

The image-feature canvas (`build_image_feature_canvas`) renders for both tabs, so
its toolbar is asserted on the component and the two hosts are asserted for what
each puts around it: Zonal's inputs above the canvas and its log below the run
row; Detection's ROI canvas as a band of the tab rather than content of the
*Advanced parameters* expander, with the file list as its only navigator.

Everything here is the rendered tree walked in DOM order. What it cannot see --
the bar engaging under the header while scrolling, a drag on a handle -- needs
a visible browser.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


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
    if not lab and e.__class__.__name__ in ("Checkbox", "Button", "Toggle"):
        lab = str(getattr(e, "text", "") or "")
    return lab


def _classes(e):
    return list(getattr(e, "_classes", []) or [])


def _kind(e):
    return e.__class__.__name__


def _first(els, pred):
    return next(i for i, e in enumerate(els) if pred(e))


_FIELDS = ("current_project", "zonal_view", "zonal_mode", "zonal_csv",
           "zonal_raster", "det_dir", "det_files", "det_mode")


@pytest.fixture
def clean():
    """Delete the garbage other test modules leave on the shared client
    (their strips stay bound to the state), and restore the state after."""
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
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def _build(clean, builder):
    from nicegui import ui
    with ui.column() as col:
        builder()
    clean[1].append(col)
    return _walk(col)


# --------------------------------------------------------------------------- #
#  The component                                                               #
# --------------------------------------------------------------------------- #
def test_the_component_has_one_sticky_toolbar_ending_with_save(clean):
    from gui.app import build_image_feature_canvas, CanvasContext
    ctx = CanvasContext()
    els = _build(clean, lambda: build_image_feature_canvas(ctx, title="t"))
    bars = [e for e in els if "pm-sticky-toolbar" in _classes(e)]
    assert len(bars) == 1, f"{len(bars)} toolbars"
    inside = _walk(bars[0])
    buttons = [_label(e) for e in inside if _kind(e) == "Button"]
    assert buttons and buttons[-1] == "Save GeoJSON", buttons
    for want in ("Fit", "Undo vertex", "Finalise", "Clear in-progress",
                 "Clear all"):
        assert want in buttons, f"{want!r} missing from {buttons}"
    toggles = [e for e in inside if _kind(e) == "Toggle"]
    assert len(toggles) == 2, "draw/modify and the shape toggle share the bar"
    # Order: input rows -> bar -> status -> box.
    i_src = _first(els, lambda e: _label(e) == "Source image")
    i_vec = _first(els, lambda e: _label(e).startswith("Vector layer"))
    i_bar = _first(els, lambda e: "pm-sticky-toolbar" in _classes(e))
    i_status = _first(els, lambda e: "pm-canvas-status" in _classes(e))
    i_box = _first(els, lambda e: "pm-canvas-box" in _classes(e))
    assert i_src < i_vec < i_bar < i_status < i_box
    box = els[i_box]
    assert "70vh" in str(getattr(box, "_style", {}) or {}).replace("'", ""), (
        "the canvas box is not capped at 70 % of the viewport")


def test_the_component_fits_the_image_to_the_box_not_to_700_px(src):
    body = src[src.index("def build_image_feature_canvas("):]
    body = body[:body.index("\ndef ")]
    fit = body[body.index("def _zoom_fit"):][:600]
    assert "700.0" not in fit and "ortho_viewport_h" in fit


# --------------------------------------------------------------------------- #
#  Zonal                                                                       #
# --------------------------------------------------------------------------- #
def test_zonal_inputs_precede_the_canvas_and_results_follow_the_run(clean):
    from gui.app import build_zonal_tab
    state, _ = clean
    state.zonal_view = "polygons"
    state.zonal_mode = "polygons"
    els = _build(clean, build_zonal_tab)
    i_csv = _first(els, lambda e: _label(e).startswith("Detection CSV"))
    i_use = _first(els, lambda e: _label(e) == "Use source from inputs")
    i_src = _first(els, lambda e: _label(e) == "Source image")
    i_bar = _first(els, lambda e: "pm-sticky-toolbar" in _classes(e))
    i_box = _first(els, lambda e: "pm-canvas-box" in _classes(e))
    i_run = _first(els, lambda e: _label(e) == "Run now")
    i_log = _first(els, lambda e: _kind(e) == "Log")
    assert i_csv < i_use < i_src < i_bar < i_box < i_run < i_log, (
        f"csv {i_csv} use {i_use} src {i_src} bar {i_bar} box {i_box} "
        f"run {i_run} log {i_log}")
    vectors = [e for e in els if _label(e).startswith("Vector layer")]
    assert len(vectors) == 1, "a second vector field is back"
    assert not any(_label(e).startswith("Output GeoJSON for Save")
                   for e in els), "the second output field is back"
    texts = [getattr(e, "content", "") if _kind(e) == "Markdown"
             else getattr(e, "text", "") for e in els
             if _kind(e) in ("Markdown", "Label")]
    assert not any(re.search(r"\bStep \d", str(t)) for t in texts)
    # Zonal's mode-dependent shape toggles live in the component's bar.
    bar = els[i_bar]
    toggles = [e for e in _walk(bar) if _kind(e) == "Toggle"]
    opts = [str(getattr(e, "options", "")) for e in toggles]
    assert any("transect" in o for o in opts), (
        "the transect toggle is not in the toolbar")


def test_zonal_status_reads_the_effective_shape(clean):
    """In transect mode the status must say transect."""
    from gui.app import build_zonal_tab
    state, _ = clean
    state.zonal_view = "transects"
    state.zonal_mode = "transects"
    els = _build(clean, build_zonal_tab)
    status = next(e for e in els if "pm-canvas-status" in _classes(e))
    assert "transect" in str(getattr(status, "text", "")).lower(), status.text


# --------------------------------------------------------------------------- #
#  Detection                                                                   #
# --------------------------------------------------------------------------- #
def test_detection_roi_canvas_is_a_band_of_the_tab(clean, src):
    from gui.app import build_detection_tab
    els = _build(clean, build_detection_tab)
    roi = next(e for e in els if "pm-det-roi" in _classes(e))
    p = roi.parent_slot.parent if roi.parent_slot else None
    while p is not None:
        assert _kind(p) != "Expansion", "the ROI canvas is inside an expander"
        p = p.parent_slot.parent if p.parent_slot else None
    i_adv = _first(els, lambda e: _kind(e) == "Expansion"
                   and _label(e).startswith("Advanced"))
    i_roi = _first(els, lambda e: "pm-det-roi" in _classes(e))
    i_add = _first(els, lambda e: _label(e) == "Add to queue")
    assert i_adv < i_roi < i_add, f"adv {i_adv} roi {i_roi} add {i_add}"
    assert "_roi_nav" not in src, "the second image navigator is back"
    # The model is chosen in the drawer: the tab has no select of its own,
    # only a caption naming it; Reload model and the status line are in the
    # drawer, under the select.
    assert not any(_kind(e) == "Select" and _label(e) == "Detection model"
                   for e in els), "the Detect tab has its own model select again"
    i_model = _first(els, lambda e: "pm-model-caption" in _classes(e))
    assert "chosen in the left panel" in els[i_model].text
    i_dir = _first(els, lambda e: _label(e) == "Image directory")
    assert i_model < i_dir, f"model {i_model} dir {i_dir}"
    assert not any("Reload" in _label(e) for e in els), \
        "the Detect tab has its own reload button again"
    assert not any(str(getattr(e, "text", "")).startswith("Model: ")
                   for e in els), "the Detect tab has its own status line again"


def test_a_typed_source_path_loads_the_canvas(clean, tmp_path):
    """Only Browse (and a file row's button) rebuilt the canvas; a typed or
    pasted path left it saying 'pick a source image above'. Leaving the field now loads it."""
    from PIL import Image
    from nicegui.events import ValueChangeEventArguments, handle_event
    from gui.app import build_image_feature_canvas, CanvasContext
    png = tmp_path / "small.png"
    Image.new("RGB", (40, 30), (120, 110, 100)).save(png)
    ctx = CanvasContext()
    els = _build(clean, lambda: build_image_feature_canvas(ctx, title="t"))
    inp = next(e for e in els if _kind(e) == "Input"
               and e._props.get("label") == "Source image")
    ctx.image = str(png)
    for l in inp._event_listeners.values():
        if l.type == "change":
            handle_event(l.handler, ValueChangeEventArguments(sender=inp, client=inp.client, value=str(png)))
    root = els[0]
    kinds = [_kind(e) for e in _walk(root)]
    assert "InteractiveImage" in kinds, kinds[:20]
