"""The Orthorectify tab conforms to the three-band layout model.

Three kinds of assertion, deliberately:

* pure helpers (``fit_width``, ``loupe_marks``, ``batch_gate_message``) that
  define what the layout, the magnifier and the gate message must do, tested
  without NiceGUI;
* the rendered tab, built off-screen, walked in DOM order -- one table, one
  size field, no Load button, Rectify last in the toolbar, nothing of Band A
  below the button that consumes it;
* two behaviours: the folder arrives filled from the project and every
  corner click reaches the magnifier's overlay.

What none of this proves -- that the sticky bar engages under the header,
that a corner can be dragged, that the loupe pixels are drawn -- needs a
visible browser. A test that claimed those would be lying.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from functions import orthorectify as O          # noqa: E402

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


def _body(src: str, name: str) -> str:
    start = src.index(f"def {name}(")
    nxt = src.find("\ndef ", start + 1)
    return src[start:nxt if nxt != -1 else len(src)]


# --------------------------------------------------------------------------- #
#  Pure helpers                                                                #
# --------------------------------------------------------------------------- #
def test_fit_puts_the_whole_photograph_inside_the_box():
    """The surface is a 70 % box; Fit means all of it is visible."""
    assert O.fit_width(840, 840, box_h=630) == pytest.approx(630)
    # Wide: the column, not the box, is the limit.
    assert O.fit_width(4000, 2000, box_h=630) == pytest.approx(768)
    # Tall: the box height decides.
    assert O.fit_width(1000, 4000, box_h=630) == pytest.approx(157.5)
    # Never upscaled beyond the photograph's own pixels.
    assert O.fit_width(500, 400, box_h=630) == pytest.approx(500)


def test_fit_never_returns_a_degenerate_width():
    assert O.fit_width(0, 0, box_h=630) >= 1.0
    assert O.fit_width(800, 600, box_h=0) >= 1.0


def test_loupe_marks_map_placed_corners_into_loupe_pixels():
    """The magnifier must show the corners already placed. A 4x loupe of
    200 px shows a 50 px window of the photograph centred on the cursor."""
    corners = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    m = O.loupe_marks(corners, cx=110, cy=105, zoom=4, size=200, editing=0)
    # Corner 1 is 15 px left / 5 px up of the cursor -> 60 px left / 20 px up
    # of the loupe centre (100, 100).
    assert m["corners"] == [{"n": 1, "x": 60.0, "y": 80.0, "editing": True}]
    # Every edge is returned in loupe pixels; the canvas clips them.
    assert len(m["edges"]) == 4
    assert m["edges"][0] == [60.0, 80.0, 860.0, 80.0]


def test_loupe_marks_with_nothing_placed_is_empty():
    m = O.loupe_marks([], cx=10, cy=10)
    assert m == {"corners": [], "edges": []}


def test_loupe_marks_two_corners_is_one_open_edge():
    m = O.loupe_marks([[0, 0], [10, 0]], cx=5, cy=0, zoom=4, size=200)
    assert len(m["edges"]) == 1
    assert [c["n"] for c in m["corners"]] == [1, 2]


def test_the_batch_gate_names_the_field_and_its_band():
    """'Nothing is ready. The list above says why.' said what, not where."""
    msg = O.batch_gate_message([("a.jpg", "still at the default 1.000 m")])
    assert "Quadrat size" in msg and "Inputs, above" in msg
    msg = O.batch_gate_message([("a.jpg", "2/4 corners")])
    assert "corners" in msg and "Work surface, below" in msg
    msg = O.batch_gate_message([("a.jpg", "no segment length recorded")])
    assert "Record" in msg and "Inputs, above" in msg
    msg = O.batch_gate_message([("a.jpg", "already rectified")])
    assert "overwrite" in msg.lower()
    msg = O.batch_gate_message([])
    assert "Folder of photographs" in msg and "Inputs, above" in msg


# --------------------------------------------------------------------------- #
#  Rendering the tab off-screen                                                #
# --------------------------------------------------------------------------- #
_ORTHO_FIELDS = (
    "ortho_src_path", "ortho_corners", "ortho_click_idx", "ortho_seg_lengths",
    "ortho_seg_1", "ortho_seg_2", "ortho_seg_3", "ortho_seg_4",
    "ortho_frame_thickness_m", "ortho_zoom", "ortho_guide", "ortho_gsd_m",
    "ortho_out_path", "ortho_image_dir", "ortho_image_list", "ortho_image_idx",
    "ortho_selected_image", "ortho_per_image", "ortho_corner_armed",
    "ortho_batch_msg", "ortho_batch_rows", "current_project",
)


@pytest.fixture
def clean_state():
    """The GUI state is a process-wide singleton; leave it as it was found."""
    from gui.app import state
    saved = {k: getattr(state, k) for k in _ORTHO_FIELDS}
    # Other test modules build tabs off-screen and never delete them, so
    # their project strips stay bound to `state.current_project`. Setting
    # the project for THIS test then propagates through every one of them,
    # and their handlers re-seed the shared state before this build runs
    # (or recurse, the 023 binding family). Delete that garbage first.
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    state.ortho_src_path = ""
    state.ortho_corners = []
    state.ortho_click_idx = 0
    state.ortho_image_dir = ""
    state.ortho_image_list = []
    state.ortho_image_idx = -1
    state.ortho_selected_image = ""
    state.ortho_per_image = {}
    state.ortho_out_path = ""
    state.ortho_guide = None
    state.ortho_batch_msg = ""
    state.ortho_batch_rows = []
    state.current_project = ""
    yield state
    while _BUILT:
        try:
            _BUILT.pop().delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def _walk(root):
    """Every element under ``root`` in DOM order."""
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


_BUILT = []


def _build():
    """Build the tab off-screen. The tree is deleted by the ``clean_state``
    fixture: elements stay bound to the process-wide state until deleted, and
    a second build would otherwise fan every state change out through the
    first build's handlers (the binding-recursion family of 023)."""
    from nicegui import ui
    from gui.app import build_orthorectify_tab
    with ui.column() as col:
        build_orthorectify_tab()
    _BUILT.append(col)
    return _walk(col)


def _label(e):
    """A control's caption: the label prop for fields, selects, buttons and
    expanders; the text for checkboxes (NiceGUI keeps theirs in ``.text``)."""
    lab = str((getattr(e, "_props", {}) or {}).get("label", "") or "")
    if not lab and e.__class__.__name__ in ("Checkbox", "Button", "Toggle"):
        lab = str(getattr(e, "text", "") or "")
    return lab


def _classes(e):
    return list(getattr(e, "_classes", []) or [])


def _texts(els):
    """On-page prose: labels' and markdowns' text, tooltips excluded."""
    out = []
    for e in els:
        kind = e.__class__.__name__
        if kind == "Markdown":
            t = getattr(e, "content", None)
        elif kind == "Label":
            t = getattr(e, "text", None)
        else:
            continue          # buttons, checkboxes, tooltips are controls
        if isinstance(t, str) and t.strip():
            out.append(t)
    return out


def test_one_table_one_size_field_no_load_button(clean_state):
    els = _build()
    labels = [_label(e) for e in els]
    assert labels.count("Quadrat size (m)") == 1, labels
    assert not any(l.startswith("This quadrat frame is") for l in labels), (
        "the second frame-size entry is back")
    assert not any(l.startswith("Segment 1:") for l in labels), (
        "the segment-1 box is a second entry for the quadrat size")
    assert "Load image" not in labels, "selecting an image must load it"
    tables = [e for e in els if e.__class__.__name__ == "Table"]
    assert len(tables) == 1, f"{len(tables)} tables; the batch result table restates the folder table"


def test_nothing_of_band_a_renders_below_the_batch_button(clean_state):
    """The button consumes the size, its confirmation, the thickness."""
    els = _build()
    labels = [_label(e) for e in els]
    batch = labels.index("Rectify every ready file")
    for field in ("Folder of photographs", "Quadrat size (m)",
                  "Frame thickness (m)", "Segment lengths (non-square)",
                  "Lens distortion", "Output"):
        hits = [i for i, l in enumerate(labels) if l.startswith(field)]
        assert hits, f"missing {field}"
        assert hits[0] < batch, (
            f"{field!r} renders below the batch button that consumes it")
    confirm = next(i for i, l in enumerate(labels) if l.startswith("I confirm"))
    assert confirm < batch, "the size confirmation is below the batch button"
    # And the surface follows the inputs: Band A ends before the toolbar.
    assert batch < labels.index("Image") < labels.index("Rectify")


def test_the_toolbar_is_one_sticky_row_ending_with_rectify(clean_state):
    """Zoom, navigation, suggest, reset and the surface's own action in one
    bar attached to the surface, Rectify last."""
    els = _build()
    bar = next((e for e in els if "pm-sticky-toolbar" in _classes(e)), None)
    assert bar is not None, "no .pm-sticky-toolbar row"
    inside = [_label(e) for e in _walk(bar)
              if e.__class__.__name__ in ("Button", "Select", "Checkbox")]
    inside = [l for l in inside if l]
    for want in ("Image", "Previous", "Next", "Fit", "1:1",
                 "Suggest a position", "Reset corners", "Rectify"):
        assert want in inside, f"{want!r} not in the toolbar: {inside}"
    buttons = [_label(e) for e in _walk(bar) if e.__class__.__name__ == "Button"]
    assert buttons[-1] == "Rectify", f"Rectify is not the last item: {buttons}"
    assert inside.index("Reset corners") < inside.index("Rectify")


def test_results_render_below_the_surface(clean_state):
    """The log and the rectified preview are a Band C column after the
    surface container, never inside the parameters."""
    els = _build()
    cls = [_classes(e) for e in els]
    surface = next(i for i, c in enumerate(cls) if "pm-ortho-surface" in c)
    results = next(i for i, c in enumerate(cls) if "pm-ortho-results" in c)
    assert surface < results
    log = next(i for i, e in enumerate(els) if e.__class__.__name__ == "Log")
    assert log > results, "the run log is not in the results band"


def test_the_copy_is_one_line_per_group(clean_state):
    """No intro paragraph, no Step captions, no 44-word explainers."""
    els = _build()
    texts = _texts(els)
    assert not any(re.search(r"\bStep \d", t) for t in texts)
    long = [t for t in texts if len(t.split()) > 30]
    assert not long, f"on-page paragraphs remain: {[t[:60] for t in long]}"
    total = sum(len(t.split()) for t in texts)
    assert total <= 130, f"{total} words on page (baseline 178)"


def test_gate_messages_name_field_and_band(src):
    body = _body(src, "build_orthorectify_tab")
    for gone in ("The list above says why.", "Pick a source image file first.",
                 "Pick all 4 corners first", "Load a folder first.",
                 "Load a photograph first.", "Set 'Frame thickness (m)' first",
                 "Pick an output file path."):
        assert gone not in body, f"unnamed dependency still shown: {gone!r}"
    assert body.count("(Inputs, above)") >= 4
    assert "batch_gate_message(" in body


# --------------------------------------------------------------------------- #
#  The project fills the folder                                                #
# --------------------------------------------------------------------------- #
def _photo(path, w=60, h=40):
    rng = np.random.default_rng(1)
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
    images = layout.project_path("P", "images")
    _photo(images / "a.jpg")
    (tmp_path / "Empty").mkdir()
    layout.project_path("Empty", "images").mkdir(parents=True, exist_ok=True)
    try:
        yield {"root": tmp_path, "images": images}
    finally:
        layout.set_datasets_root(original, persist=False)


def test_the_folder_arrives_filled_and_the_first_photograph_loaded(
        clean_state, project):
    clean_state.current_project = "P"
    els = _build()
    assert Path(clean_state.ortho_image_dir) == project["images"]
    assert clean_state.ortho_image_list == ["a.jpg"]
    assert clean_state.ortho_selected_image == "a.jpg"
    inp = next(e for e in els if _label(e) == "Folder of photographs")
    assert Path(inp.value) == project["images"]
    assert "from the active project" in str(inp._props.get("hint", "")), (
        "a seeded default must be visibly a default")
    assert any(e.__class__.__name__ == "InteractiveImage" for e in els), (
        "the photograph is selected but not loaded: a Load step is back")


def test_a_typed_folder_is_not_clobbered_by_the_project(clean_state, project,
                                                         tmp_path):
    mine = tmp_path / "mine"
    _photo(mine / "z.jpg")
    clean_state.current_project = "P"
    clean_state.ortho_image_dir = str(mine)
    _build()
    assert Path(clean_state.ortho_image_dir) == mine


def test_a_project_without_photographs_seeds_nothing(clean_state, project):
    clean_state.current_project = "Empty"
    _build()
    assert clean_state.ortho_image_dir == "", (
        "an empty folder was invented for a project with no photographs")


def test_a_project_switch_reseeds_the_folder(src):
    body = _body(src, "build_orthorectify_tab")
    handler = body[body.index("def _on_proj_change"):][:600]
    assert "_seed_ortho_folder(force=True)" in handler


# --------------------------------------------------------------------------- #
#  Every corner reaches the magnifier                                          #
# --------------------------------------------------------------------------- #
def test_a_corner_click_is_pushed_to_the_loupe(clean_state, project,
                                                monkeypatch):
    import nicegui
    from nicegui.events import GenericEventArguments, handle_event
    pushed = []
    monkeypatch.setattr(nicegui.ui, "run_javascript",
                        lambda code, *a, **k: pushed.append(str(code)))
    clean_state.current_project = "P"
    els = _build()
    img = next(e for e in els if e.__class__.__name__ == "InteractiveImage")
    mouse = [l for l in img._event_listeners.values() if l.type == "mouse"]
    assert mouse, "the interactive image has no mouse handler"
    def mouse_ev(kind, x, y):
        handle_event(mouse[0].handler, GenericEventArguments(
            sender=img, client=img.client,
            args={"mouse_event_type": kind, "image_x": x, "image_y": y}))

    # A press-and-release that does not move is a click: it places a corner.
    mouse_ev("mousedown", 12.4, 9.0)
    mouse_ev("mouseup", 12.4, 9.0)
    assert clean_state.ortho_corners == [[12, 9]]
    hits = [c for c in pushed if "__ortho_corners" in c]
    assert hits, "the overlay redraw did not publish the corners to the client"
    assert re.search(r"\[\s*12(\.0)?\s*,\s*9(\.0)?\s*\]", hits[-1]), hits[-1][:200]

    # A press on a placed corner, a move, and a release is a drag: the corner
    # follows the cursor (corners are draggable).
    mouse_ev("mousedown", 12.0, 9.0)
    mouse_ev("mousemove", 16.0, 12.0)
    mouse_ev("mousemove", 21.0, 15.0)
    mouse_ev("mouseup", 21.0, 15.0)
    assert clean_state.ortho_corners == [[21, 15]], clean_state.ortho_corners
    assert re.search(r"\[\s*21(\.0)?\s*,\s*15(\.0)?\s*\]", pushed[-1]), pushed[-1][:200]
    # Moving is not placing: still one corner, the next click places #2.
    mouse_ev("mousedown", 40.0, 30.0)
    mouse_ev("mouseup", 40.0, 30.0)
    assert clean_state.ortho_corners == [[21, 15], [40, 30]]


def test_a_dropped_corner_stays_the_active_one(clean_state, project,
                                               monkeypatch):
    """With all four placed, dragging #2 and dropping it used to hand the
    selection back to #1 every time."""
    import nicegui
    from nicegui.events import GenericEventArguments, handle_event
    monkeypatch.setattr(nicegui.ui, "run_javascript", lambda *a, **k: None)
    clean_state.current_project = "P"
    els = _build()
    img = next(e for e in els if e.__class__.__name__ == "InteractiveImage")
    mouse = [l for l in img._event_listeners.values() if l.type == "mouse"]

    def mouse_ev(kind, x, y):
        handle_event(mouse[0].handler, GenericEventArguments(
            sender=img, client=img.client,
            args={"mouse_event_type": kind, "image_x": x, "image_y": y}))

    # Corners at the photograph's extremes (60 x 40 px), far beyond any hit
    # radius, so each click places rather than selects.
    for x, y in ((2, 2), (58, 2), (58, 38), (2, 38)):
        mouse_ev("mousedown", x, y)
        mouse_ev("mouseup", x, y)
    assert len(clean_state.ortho_corners) == 4, clean_state.ortho_corners
    mouse_ev("mousedown", 58, 2)
    mouse_ev("mousemove", 56, 4)
    mouse_ev("mousemove", 54, 6)
    mouse_ev("mouseup", 54, 6)
    assert clean_state.ortho_corners[1] == [54, 6]
    assert clean_state.ortho_click_idx == 1, "the selection fell back to #1"
    # A drop is not an arm: a stray click elsewhere moves nothing.
    mouse_ev("mousedown", 30, 20)
    mouse_ev("mouseup", 30, 20)
    assert clean_state.ortho_corners[1] == [54, 6]
    assert len(clean_state.ortho_corners) == 4


def test_a_suggestion_is_placed_not_proposed(src):
    """The suggested frame becomes four draggable
    corners recorded as suggested, cleared by Reset corners -- no dashed
    proposal, no Accept/Reject."""
    body = _body(src, "build_orthorectify_tab")
    assert "Accept frame" not in body and "_ortho_accept_suggestion" not in body
    fn = body[body.index("async def _ortho_suggest_corners"):]
    fn = fn[:fn.index("\n    def _reset_picks")]
    assert "state.ortho_corners = pts" in fn
    assert 'source="suggested"' in fn, "the provenance marker is gone"


def test_the_loupe_script_draws_from_the_published_corners(src):
    body = _body(src, "build_orthorectify_tab")
    draw = body[body.index("window.__loupe_draw = function"):][:4000]
    assert "__ortho_corners" in draw, "the magnifier still draws the raw image only"
    assert "loupe_marks" in draw, "the JS does not name the Python twin it mirrors"


def test_a_release_away_from_the_press_moves_the_corner_without_mousemoves(
        clean_state, project, monkeypatch):
    """The drag is drawn in the browser; the server sees only
    the press and the release, and the release position is the move."""
    import nicegui
    from nicegui.events import GenericEventArguments, handle_event
    monkeypatch.setattr(nicegui.ui, "run_javascript", lambda *a, **k: None)
    clean_state.current_project = "P"
    els = _build()
    img = next(e for e in els if e.__class__.__name__ == "InteractiveImage")
    assert "mousemove" not in img._props.get("events", []), (
        "every mousemove still goes to the server")
    mouse = [l for l in img._event_listeners.values() if l.type == "mouse"]

    def mouse_ev(kind, x, y):
        handle_event(mouse[0].handler, GenericEventArguments(
            sender=img, client=img.client,
            args={"mouse_event_type": kind, "image_x": x, "image_y": y}))

    for x, y in ((2, 2), (58, 2), (58, 38), (2, 38)):
        mouse_ev("mousedown", x, y)
        mouse_ev("mouseup", x, y)
    mouse_ev("mousedown", 58, 2)
    mouse_ev("mouseup", 50, 10)
    assert clean_state.ortho_corners[1] == [50, 10], clean_state.ortho_corners
    assert clean_state.ortho_click_idx == 1
    # A press and release on the spot is a click: it selects, nothing moves.
    mouse_ev("mousedown", 2, 38)
    mouse_ev("mouseup", 2, 38)
    assert clean_state.ortho_corners[3] == [2, 38]
    assert clean_state.ortho_click_idx == 3


def test_the_overlay_carries_the_ids_the_browser_drag_moves(src):
    body = _body(src, "build_orthorectify_tab")
    for anchor in ('id="pm-frame"', 'id="pm-corner-{i}"', 'id="pm-label-{i}"',
                   "getElementById('pm-corner-' + drag.i)",
                   "getElementById('pm-frame')"):
        assert anchor in body, anchor
