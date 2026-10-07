"""027 / User Story 6 -- Express, Georeference, Validate and the Zonal profile
arrive filled from the active project (Digitize and Orthorectify were done in
their own slices).

A temporary datasets root holds one project with one file of each kind; each
tab is built off-screen with that project active and its state read back.
Nothing here touches real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _photo(path, w=40, h=30):
    rng = np.random.default_rng(3)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix if path.suffix != ".tif" else ".png", img)
    assert ok
    buf.tofile(str(path))
    return path


def _text(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


_FIELDS = ("current_project", "express_ortho", "dig_georef_ortho", "dig_photo_dir",
           "val_truth_csv", "val_truth_dir", "val_detect_csv", "val_src_image",
           "val_uav_image", "zonal_overlay_csvs", "zonal_view", "zonal_csv",
           "zonal_raster")


@pytest.fixture
def clean():
    from gui.app import state
    # express_ortho is set by the Express builder, not declared on the state.
    saved = {k: getattr(state, k, "") for k in _FIELDS}
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    for k in _FIELDS:
        if k == "zonal_overlay_csvs":
            state.zonal_overlay_csvs = []
        elif k != "zonal_view":
            setattr(state, k, "")
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


@pytest.fixture
def project(tmp_path):
    import functions.layout as layout
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    (tmp_path / "P").mkdir()
    pp = lambda kind: layout.project_path("P", kind)
    files = {
        "ortho": _photo(pp("images") / "site_ortho.tif"),
        "truth": _text(pp("validation") / "q1_truth.csv", "x,y,Clast_length\n1,2,30\n"),
        "detect": _text(pp("vectors") / "site_ortho_window_size=2.5m_individual_clast_values.csv",
                        "x,y,Clast_length\n1,2,31\n"),
        "georect": _photo(pp("validation_georectified") / "q1_georeferenced.tif"),
        "valimg": _photo(pp("validation_images") / "q1.jpg"),
        "transect": _text(pp("zonal") / "site.transects.csv",
                          "transect_id,distance_m,raster_value\n0,0,0.05\n"),
    }
    try:
        yield files
    finally:
        layout.set_datasets_root(original, persist=False)


def _build(clean, builder):
    from nicegui import ui
    with ui.column() as col:
        builder()
    clean[1].append(col)
    return col


def test_express_arrives_with_the_newest_ortho(clean, project):
    from gui.app import build_express_tab
    state, _ = clean
    state.current_project = "P"
    _build(clean, build_express_tab)
    assert Path(state.express_ortho) == project["ortho"]


def test_georeference_arrives_with_ortho_and_folder(clean, project):
    from gui.app import build_georeference_tab
    state, _ = clean
    state.current_project = "P"
    _build(clean, build_georeference_tab)
    assert Path(state.dig_georef_ortho) == project["ortho"]
    assert Path(state.dig_photo_dir) == project["valimg"].parent


def test_validate_arrives_with_its_four_paths(clean, project):
    from gui.app import build_validate_tab
    state, _ = clean
    state.current_project = "P"
    _build(clean, build_validate_tab)
    assert Path(state.val_truth_csv) == project["truth"]
    assert Path(state.val_truth_dir) == project["truth"].parent
    assert Path(state.val_detect_csv) == project["detect"]
    assert Path(state.val_src_image) == project["georect"]
    assert Path(state.val_uav_image) == project["ortho"]


def _selects(col):
    from nicegui import ui
    out = []

    def walk(el):
        for c in el.default_slot.children:
            if isinstance(c, ui.select):
                out.append(c)
            walk(c)
    walk(col)
    return out


def test_express_shows_the_remembered_ortho_on_a_fresh_page(clean, project):
    """A page built while state.express_ortho already holds the project's
    ortho (a reload, a second window, the next launch) shows it in the
    select. The seed used to return early on "already chosen" and left the
    new page's select blank until the project was switched away and back."""
    from gui.app import build_express_tab
    state, _ = clean
    state.current_project = "P"
    state.express_ortho = str(project["ortho"])
    col = _build(clean, build_express_tab)
    sel = next(s for s in _selects(col) if s.props.get("label") == "UAV ortho-image")
    assert sel.value == str(project["ortho"])
    assert Path(state.express_ortho) == project["ortho"]


def test_profile_rows_arrive_from_the_transect_csvs(clean, project):
    from gui.app import build_zonal_tab
    state, _ = clean
    state.current_project = "P"
    state.zonal_view = "profile"
    _build(clean, build_zonal_tab)
    assert [Path(r["csv"]) for r in state.zonal_overlay_csvs] == [project["transect"]]


def test_a_typed_value_is_kept(clean, project, tmp_path):
    from gui.app import build_validate_tab
    state, _ = clean
    mine = _text(tmp_path / "mine.csv", "x,y\n")
    state.current_project = "P"
    state.val_truth_csv = str(mine)
    _build(clean, build_validate_tab)
    assert Path(state.val_truth_csv) == mine


def test_no_project_seeds_nothing(clean, project):
    from gui.app import build_validate_tab, build_express_tab
    state, _ = clean
    state.current_project = ""
    _build(clean, build_validate_tab)
    _build(clean, build_express_tab)
    assert state.val_truth_csv == "" and state.express_ortho == ""


# --------------------------------------------------------------------------- #
#  Gaps found in the rendered tabs                                           #
# --------------------------------------------------------------------------- #
def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def _find(col, cls, label):
    for e in _walk(col):
        if e.__class__.__name__ == cls and (e._props.get("label") == label):
            return e
    raise AssertionError(f"no {cls} labelled {label!r}")


def test_detection_directory_arrives_marked_as_a_default(clean, project):
    """Gap 1: the one seeded path without the 'from the active project' hint."""
    from gui import app as A
    saved = A.state.det_dir
    A.state.det_dir = ""
    try:
        A.state.current_project = "P"
        col = _build(clean, A.build_detection_tab)
        inp = _find(col, "Input", "Image directory")
        assert Path(inp.value) == project["ortho"].parent
        assert "from the active project" in str(inp._props.get("hint", ""))
        # A user edit is a user's value: the hint goes.
        inp.value = str(project["ortho"].parent / "elsewhere")
        assert "from the active project" not in str(inp._props.get("hint", ""))
    finally:
        A.state.det_dir = saved


def test_the_zonal_id_field_offers_no_blank_row(clean, project):
    """Gap 2: the FID fallback is a labelled entry."""
    from gui import app as A
    A.state.current_project = "P"
    col = _build(clean, A.build_zonal_tab)
    sel = _find(col, "Select", "ID field (label for each zone)")
    assert isinstance(sel.options, dict)
    assert sel.options[""].startswith("—") and "FID" in sel.options[""]
    assert "" not in list(sel.options.values())


def test_the_zonal_lists_refresh_from_the_csv_field_not_only_a_poll():
    """Gap 3: handler-driven, the timer is the fallback."""
    src = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    body = src[src.index("def build_zonal_tab("):]
    body = body[:body.index("\ndef ", 1)]
    assert '_zonal_field_els["csv"].on_value_change(' in body
    assert "_refresh_zonal_selectors())" in body


def test_a_path_from_another_project_is_dropped_so_the_seed_can_refill(project, tmp_path):
    """Pre-fills ran at page build and on a project change only, so a tab
    opened later still showed the previous project's ortho or the previous
    run's raster. Opening a tab now drops the
    paths that point outside the active project; the ones inside it stay."""
    from gui.app import drop_foreign_paths, state
    saved = (state.current_project, state.ras_csv, state.ras_tif)
    try:
        state.current_project = "P"
        mine = str(project["ortho"])
        theirs = str(tmp_path / "other_project" / "their_ortho.tif")
        state.ras_csv, state.ras_tif = mine, theirs
        assert drop_foreign_paths("ras_csv", "ras_tif") == 1
        assert state.ras_csv == mine          # typed for this project: kept
        assert state.ras_tif == ""            # another project's: dropped
        # No project selected: nothing to compare against, nothing dropped.
        state.current_project = ""
        state.ras_tif = theirs
        assert drop_foreign_paths("ras_tif") == 0 and state.ras_tif == theirs
    finally:
        state.current_project, state.ras_csv, state.ras_tif = saved


def test_opening_a_tab_runs_its_refresh_hook():
    """The hook the tab bar fires when a tab becomes active."""
    from gui.app import register_tab_refresh, _fire_tab_hooks
    seen = []
    register_tab_refresh("Rasterize", lambda: seen.append("ras"))
    _fire_tab_hooks("Map")
    assert seen == []                    # another tab: not this hook
    _fire_tab_hooks("Rasterize")
    assert seen == ["ras"]
    # A hook that raises never breaks the tab switch.
    register_tab_refresh("Map", lambda: 1 / 0)
    _fire_tab_hooks("Map")

