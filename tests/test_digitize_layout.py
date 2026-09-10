"""Digitize conforms to the layout model with its own toolbar.

Rendered-tree assertions (DOM order) plus the pre-fill behaviour. The
per-clast click loop itself, the drag of a vertex and the bar engaging under
the header need a visible browser.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

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


_FIELDS = ("current_project", "dig_src_path", "dig_image_dir", "dig_image_list",
           "dig_image_idx", "dig_records", "dig_out_path", "dig_resolution",
           "dig_label_set")


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
    state.dig_src_path = ""
    state.dig_image_dir = ""
    state.dig_image_list = []
    state.dig_image_idx = -1
    state.dig_label_set = "truth"
    state.dig_records = []
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def _build(clean):
    from nicegui import ui
    from gui.app import build_digitize_tab
    with ui.column() as col:
        build_digitize_tab()
    clean[1].append(col)
    return _walk(col)


def test_one_loader_one_sticky_bar_ending_with_detect(clean):
    els = _build(clean)
    labels = [_label(e) for e in els]
    assert "Source image" not in labels, "the single loader is back"
    assert labels.count("Image directory") == 1
    assert "Load image" not in labels, "selecting a photograph must load it"
    bars = [e for e in els if "pm-sticky-toolbar" in _classes(e)]
    assert len(bars) == 1
    inside = _walk(bars[0])
    caps = [_label(e) for e in inside
            if _kind(e) in ("Button", "Select", "Toggle") and _label(e)]
    for want in ("Image", "Fit", "Done", "Cancel", "Undo", "Clear all",
                 "Detect"):
        assert want in caps, f"{want!r} missing from the bar: {caps}"
    icons = [str((getattr(e, "_props", {}) or {}).get("icon", ""))
             for e in inside if _kind(e) == "Button"]
    assert "chevron_left" in icons and "chevron_right" in icons, icons
    buttons = [_label(e) for e in inside if _kind(e) == "Button"]
    assert buttons[-1] == "Detect", buttons
    # The bar is the first thing after the inputs; the photograph follows
    # the status; the results follow the photograph.
    i_dir = _first(els, lambda e: _label(e) == "Image directory")
    i_thr = _first(els, lambda e: _label(e).startswith("Detect score"))
    i_bar = _first(els, lambda e: "pm-sticky-toolbar" in _classes(e))
    i_status = _first(els, lambda e: "pm-dig-status" in _classes(e))
    i_surface = _first(els, lambda e: "pm-dig-surface" in _classes(e))
    i_log = _first(els, lambda e: _kind(e) == "Log")
    i_export = _first(els, lambda e: _label(e) == "Export CSV")
    assert i_dir < i_thr < i_bar < i_status < i_surface < i_log < i_export, (
        f"dir {i_dir} thr {i_thr} bar {i_bar} status {i_status} "
        f"surface {i_surface} log {i_log} export {i_export}")


def test_the_copy_is_one_line_per_group(clean):
    els = _build(clean)
    texts = [str(getattr(e, "content", "")) if _kind(e) == "Markdown"
             else str(getattr(e, "text", "")) for e in els
             if _kind(e) in ("Markdown", "Label")]
    long = [t for t in texts if len(t.split()) > 30]
    assert not long, [t[:60] for t in long]
    assert not any(re.search(r"\bStep \d", t) for t in texts)


def _photo(path, w=60, h=40):
    rng = np.random.default_rng(2)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))
    return path


@pytest.fixture
def project(tmp_path):
    import functions.layout as layout
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    (tmp_path / "P").mkdir()
    val = layout.project_path("P", "validation") / "images"
    _photo(val / "q1.jpg")
    try:
        yield {"root": tmp_path, "val": val}
    finally:
        layout.set_datasets_root(original, persist=False)


def test_the_folder_arrives_filled_from_validation_images(clean, project):
    state, _ = clean
    state.current_project = "P"
    els = _build(clean)
    assert Path(state.dig_image_dir) == project["val"]
    assert state.dig_image_list == ["q1.jpg"]
    inp = next(e for e in els if _label(e) == "Image directory")
    assert "from the active project" in str(inp._props.get("hint", ""))
    assert any(_kind(e) == "InteractiveImage" for e in els), (
        "the photograph is listed but not loaded")


def test_a_typed_folder_is_kept(clean, project, tmp_path):
    state, _ = clean
    mine = tmp_path / "mine"
    _photo(mine / "z.jpg")
    state.current_project = "P"
    state.dig_image_dir = str(mine)
    _build(clean)
    assert Path(state.dig_image_dir) == mine


def test_the_fit_uses_the_box_not_900_px(src):
    body = src[src.index("def build_digitize_tab("):]
    body = body[:body.index("\ndef ")]
    fit = body[body.index("def _dig_zoom_fit"):][:700]
    assert "900.0" not in fit and "ortho_viewport_h" in fit
    assert "pm-dig-box" in body and "70vh" in body[body.index("pm-dig-box"):][:200]


def test_the_folder_follows_a_canonical_project(clean, tmp_path):
    """Switching to a project laid out with validation/orthorectified (and
    validation/raw) left the folder on the previous project, because only
    validation/images and the legacy photographs folder were looked at
."""
    import functions.layout as layout
    from gui.app import state, _fire_project_hooks
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        rect = tmp_path / "P" / "validation" / "orthorectified"
        rect.mkdir(parents=True)
        _photo(rect / "IMG_1_rectified_GSD=0.0005m.jpg")
        state.current_project = "P"
        state.dig_image_dir = str(tmp_path / "elsewhere")
        _build(clean)
        _fire_project_hooks()
        assert Path(state.dig_image_dir) == rect
    finally:
        layout.set_datasets_root(original, persist=False)


def test_saving_the_library_refreshes_detects_file_list(clean, project, monkeypatch):
    """Detect's rows kept their old scale text after a known length or a
    segment changed in Digitize."""
    from gui import app as A
    state, _ = clean
    state.current_project = "P"
    els = _build(clean)
    calls = []
    monkeypatch.setitem(A._DETECT_FILES_REFRESH, "fn", lambda: calls.append(1))
    from nicegui.events import ClickEventArguments, handle_event
    btn = next(e for e in els if _kind(e) == "Button"
               and str(getattr(e, "text", "")) == "Save to project")
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))
    assert calls == [1]


def test_clear_all_gives_the_saved_csv_back(clean, tmp_path):
    """One clast drawn after Clear all overwrote the five saved ones: the
    cleared canvas still 'owned' the CSV, so autosave wrote it. Clear all must release the file so autosave pauses until
    Load puts the saved clasts back."""
    from gui import app as A
    state, _ = clean
    els = _build(clean)
    csv = tmp_path / "IMG_1_truth.csv"
    csv.write_text("clast_ID,x,y\n1,1,1\n", encoding="utf-8")
    state.dig_out_path = str(csv)
    state.dig_csv_owned.add(str(csv))
    state.dig_csv_loaded.add(str(csv))
    state.dig_records.append({"shape": "circle", "params": {"cx": 5, "cy": 5, "r": 2}})
    from nicegui.events import ClickEventArguments, handle_event
    btn = next(e for e in els if _kind(e) == "Button"
               and str(getattr(e, "text", "")) == "Clear canvas")
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))
    assert state.dig_records == []
    assert str(csv) not in state.dig_csv_owned
    assert str(csv) not in state.dig_csv_loaded
    assert csv.read_text(encoding="utf-8").startswith("clast_ID"), "nothing on disk is deleted"


def test_a_same_named_photograph_of_another_folder_opens_empty(clean, tmp_path):
    """The record cache was keyed by file name: a same-named photograph of
    another project opened with the first one's clasts on the canvas, one
    commit away from being autosaved into the wrong project."""
    state, _ = clean
    a = tmp_path / "A" / "images"
    b = tmp_path / "B" / "images"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _photo(a / "IMG_1.jpg")
    _photo(b / "IMG_1.jpg")
    els = _build(clean)
    inp = next(e for e in els if _label(e) == "Image directory")
    inp.value = str(a)
    assert state.dig_image_list == ["IMG_1.jpg"]
    state.dig_records.append({"shape": "circle", "params": {"cx": 5, "cy": 5, "r": 2}})
    inp.value = str(b)
    assert state.dig_image_list == ["IMG_1.jpg"]
    assert state.dig_records == [], "B's photograph must not show A's clasts"
    inp.value = str(a)
    assert len(state.dig_records) == 1, "A's clasts come back with A"


def test_a_drawn_ellipse_is_announced_in_the_table_s_convention():
    """The commit notice read "@ 22°" (the drawn axis from the image's x axis)
    while the clast table listed 112.1° for the same ellipse (the axial
    bearing the CSV records): two conventions on one tab."""
    from functions.digitize import axial_bearing
    assert axial_bearing(22.0) == pytest.approx(112.0)
    assert axial_bearing(0.0) == pytest.approx(90.0)
    # Always inside the 0-180 range the schema promises.
    for drawn in (-40.0, 95.0, 179.0, 200.0, 360.0):
        assert 0.0 <= axial_bearing(drawn) < 180.0
