"""What the tab actually renders, and what the guards actually allow.

The layout assertions in `test_one_folder_workflow.py` read source text. Every
one of them can be satisfied while the tab is broken: a `bind_visibility_from`
on the survey panel leaves `value=True` in the source and renders it hidden;
guarding the corner-save with `if False:` keeps both greppable strings in
place. These tests build the real elements and read the props NiceGUI would
send to the browser, and they check that the safety guards can be neither
disarmed by the tool nor left impossible to satisfy.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from functions import orthorectify as O

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


def _body(src: str, name: str) -> str:
    start = src.index(f"def {name}(")
    nxt = src.find("\ndef ", start + 1)
    return src[start:nxt if nxt != -1 else len(src)]


def _built(builder):
    """Build a tab off-screen and return every element with a label prop."""
    from nicegui import ui
    found = []

    def walk(e):
        p = getattr(e, "_props", {}) or {}
        if p.get("label") is not None:
            found.append((e.__class__.__name__, str(p["label"]),
                          p.get("model-value"), p.get("disable")))
        for c in e.default_slot.children:
            walk(c)

    with ui.column() as col:
        builder()
    walk(col)
    return found


# --------------------------------------------------------------------------- #
#  Rendered, not merely written                                                #
# --------------------------------------------------------------------------- #
def test_the_rendered_georeference_panels_are_in_the_right_order():
    from gui.app import build_georeference_tab
    exps = [(lab, val) for kind, lab, val, _ in _built(build_georeference_tab)
            if kind == "Expansion"]
    names = [lab for lab, _ in exps]
    survey = next(i for i, n in enumerate(names)
                  if n.startswith("Check a whole survey"))
    locate = next(i for i, n in enumerate(names)
                  if n.startswith("Locate one quadrat"))
    pins = next(i for i, n in enumerate(names) if n.startswith("Pin quadrats"))
    assert survey < locate < pins, f"panel order is {names}"
    assert exps[survey][1] is True, "the survey panel does not render expanded"
    assert exps[locate][1] is not True, (
        "the single-quadrat editor renders expanded again")


def test_the_rendered_orthorectify_tab_leads_with_the_folder():
    from gui.app import build_orthorectify_tab
    labels = [lab for _, lab, _, _ in _built(build_orthorectify_tab)]
    assert labels.index("Folder of photographs") < \
        labels.index("Source photograph"), \
        "the single-photograph picker renders first"
    assert not any("Multi-file mode" in x for x in labels)


def test_the_survey_button_renders_disabled_with_its_reason_on_it():
    from gui.app import build_georeference_tab, state
    state.dig_georef_ortho = ""
    state.dig_photo_dir = ""
    hits = [(lab, dis) for _, lab, _, dis in _built(build_georeference_tab)
            if lab.startswith("Check all quadrats")]
    assert hits, "the survey button was not rendered"
    label, disabled = hits[0]
    assert disabled, "the survey button is pressable with nothing configured"
    assert "need" in label.lower(), "the button does not name what is missing"


# --------------------------------------------------------------------------- #
#  A guard must be neither self-disarming nor impossible to satisfy            #
# --------------------------------------------------------------------------- #
def test_the_rectify_path_never_sets_its_own_confirmation(src):
    """The defect was not the flag's scope, it was who wrote it: the tool set
    it on itself the instant it warned. Moving that to per-file storage and
    keeping the self-write would reproduce it exactly."""
    body = src[src.index("    def _do_rectify("):]
    body = body[:body.index("\n    ui.button")]
    assert "rectifying_at_the_untouched_default" in body
    pattern = r"""\[\s*['"]seg_confirmed['"]\s*\]\s*=|['"]seg_confirmed['"]\s*:\s*True"""
    assert not re.search(pattern, body), (
        "the rectify path writes its own confirmation flag; the scale guard "
        "is self-disabling again")


def test_a_user_can_actually_accept_the_default(src):
    """A guard nobody can satisfy is its own failure mode. Bio_Station's
    quadrat really is 1.000 m, and its user would be warned on all 58."""
    body = _body(src, "build_orthorectify_tab")
    assert "seg_confirm_checkbox" in body, "there is no way to accept"
    window = body[body.index("def _seg_confirm_changed"):][:400]
    assert '"seg_confirmed"' in window and "ortho_per_image" in window
    assert "state.ortho_src_path" in window, (
        "accepting is not scoped to the photograph on screen")


def test_a_click_off_the_corners_cannot_move_one(src):
    """Loading four saved corners wraps the click index to 0, so a click
    anywhere replaced corner 1 -- and corners now reach disk immediately,
    which made that unrecoverable."""
    handler = src[src.index("    def _on_mouse(e):"):]
    handler = handler[:handler.index("    def _load_image():")]
    assert "ortho_corner_armed" in handler
    assert "len(state.ortho_corners) >= 4 and not state.ortho_corner_armed" \
        in handler


# --------------------------------------------------------------------------- #
#  Naming, listing, and the folder this tool used to write to                  #
# --------------------------------------------------------------------------- #
def test_the_writer_asks_the_namer_rather_than_repeating_it(src):
    """Two implementations of one rule agreed only by having identical
    rounding code. Change one and the tab silently reports a finished folder
    as untouched, with every test green.

    The writer now lives in functions/orthorectify.rectify_one, so the
    assertion follows it there -- and the tab must not have grown its own
    copy back.
    """
    from pathlib import Path as _P
    mod = (_P(__file__).resolve().parents[1]
           / "functions" / "orthorectify.py").read_text(encoding="utf-8")
    writer = mod[mod.index("def rectify_one("):]
    writer = writer[:writer.index("\ndef is_ready_to_rectify")]
    assert "gsd_tagged_name(" in writer, "the writer no longer asks the namer"

    body = _body(src, "build_orthorectify_tab")
    assert 'rstrip("0")' not in body, "the tab builds the tag itself again"
    assert "cv2.imwrite" not in body, (
        "the tab writes the output itself again, beside the guards rather "
        "than behind them")


def test_a_survey_rectified_into_the_older_folder_still_counts(tmp_path):
    """Bio_Station's 27 outputs live in `validation/rectified/` under the
    older naming convention, and were reported as 0 of 58 done."""
    old = tmp_path / "rectified"
    old.mkdir()
    (old / "01_corrected_width=1m_height=1m_GSD=0.001m_per_px.jpg") \
        .write_bytes(b"x")
    path, exists = O.rectified_output_for(
        tmp_path / "orthorectified" / "01_rectified.jpg")
    assert exists and path.parent.name == "rectified"


def test_one_photograph_does_not_claim_anothers_output(tmp_path):
    """`S1_IMG_6738` must not be credited with `S10_IMG_6774`'s output."""
    out_dir = tmp_path / "orthorectified"
    out_dir.mkdir()
    (out_dir / "S10_IMG_6774_rectified_GSD=0.00078m.jpg").write_bytes(b"x")
    _, exists = O.rectified_output_for(out_dir / "S1_IMG_6738_rectified.jpg")
    assert not exists, "a photograph was credited with another's output"


def test_the_folder_is_listed_once_not_once_per_row(tmp_path):
    """Per-row listing is quadratic: measured 4 s for one pass over 400
    files, on the event loop, twice per image activation -- past the 8 s ping
    timeout this application already loses connections to."""
    out_dir = tmp_path / "orthorectified"
    out_dir.mkdir()
    (out_dir / "a_rectified_GSD=0.0005m.jpg").write_bytes(b"x")
    listing = O.output_listing(out_dir / "a_rectified.jpg")
    assert listing, "output_listing found nothing"

    calls = {"n": 0}
    real = Path.iterdir

    def counting(self):
        calls["n"] += 1
        return real(self)
    Path.iterdir = counting
    try:
        for name in ("a", "b", "c", "d"):
            O.folder_row(f"{name}.jpg", [], [1.185] * 4,
                         out_dir / f"{name}_rectified.jpg", None, listing)
    finally:
        Path.iterdir = real
    assert calls["n"] == 0, (
        f"{calls['n']} directory listings for 4 rows that were handed one")


# --------------------------------------------------------------------------- #
#  The sidecar format must not blind an older build                            #
# --------------------------------------------------------------------------- #
def test_an_older_reader_still_reads_every_corner(tmp_path):
    """The corners reader has no skip logic: any non-numeric line raises
    inside its loop and the blanket except returns []. An extra line in that
    file would make the previous build report ZERO corners for every
    photograph this one had touched, silently. So the segment lengths live in
    their own file.
    """
    from gui.app import _save_ortho_corners
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"x")
    corners = [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]
    _save_ortho_corners(str(photo), corners, [1.185] * 4)

    # The original reader, transcribed.
    pts = []
    for line in (tmp_path / "q_corners.txt").read_text(
            encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(";", ",").split(",")
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
    assert pts[:4] == corners, (
        "a build predating this one reads the wrong corners from a file this "
        "one wrote")


# --------------------------------------------------------------------------- #
#  Work already placed, wherever it was put                                    #
# --------------------------------------------------------------------------- #
def _mk(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_a_placement_in_the_write_target_is_found(tmp_path):
    from functions import georef as G
    photos = tmp_path / "orthorectified"
    photos.mkdir()
    _mk(photos / "georectified" / "DJI_0904_corrected_width=1.185m"
        "_GSD=0.0005m_per_px_georeferenced.tif")
    listing = G.georeferenced_listing(photos / "georectified",
                                      photos.parent / "georectified")
    hit, ok = G.georeferenced_for("DJI_0904_rectified_GSD=0.00076m.jpg",
                                  listing)
    assert ok and hit is not None


def test_a_placement_beside_the_photographs_is_found(tmp_path):
    """Bio_Station's outputs sit in `validation/georectified`, one level up
    from the photographs, not in the folder the setting points at."""
    from functions import georef as G
    photos = tmp_path / "raw"
    photos.mkdir()
    _mk(tmp_path / "georectified"
        / "01_corrected_width=1m_height=1m_GSD=0.001m_per_px_georeferenced.tif")
    listing = G.georeferenced_listing(photos / "georectified",
                                      tmp_path / "georectified")
    hit, ok = G.georeferenced_for("01.png", listing)
    assert ok and hit is not None


def test_a_placement_archived_in_a_subfolder_still_counts(tmp_path):
    """A survey's outputs were moved into `georectified/old/`.
    Work that was done is still done."""
    from functions import georef as G
    photos = tmp_path / "orthorectified"
    photos.mkdir()
    _mk(tmp_path / "georectified" / "old"
        / "DJI_0906_corrected_width=1.185m_GSD=0.0005m_per_px_georeferenced.tif")
    listing = G.georeferenced_listing(photos / "georectified",
                                      tmp_path / "georectified")
    hit, ok = G.georeferenced_for("DJI_0906_rectified_GSD=0.000687m.png",
                                  listing)
    assert ok and hit.parent.name == "old"


def test_one_quadrat_does_not_claim_anothers_placement(tmp_path):
    """`S1_IMG_6738` must not be credited with `S10_IMG_6774`'s output. A
    prefix match would; matching on the base name does not."""
    from functions import georef as G
    _mk(tmp_path / "georectified"
        / "S10_IMG_6774_corrected_GSD=0.0005m_per_px_georeferenced.tif")
    listing = G.georeferenced_listing(tmp_path / "georectified")
    _, ok = G.georeferenced_for("S1_IMG_6738_rectified_GSD=0.0007m.png",
                                listing)
    assert not ok, "a quadrat was credited with another's placement"


def test_sidecars_are_not_mistaken_for_placements(tmp_path):
    """One placement writes a .tif, a .pgw, an .aux.xml and a CSV. Only the
    raster is the placement, and a folder of sidecars alone is not one."""
    from functions import georef as G
    d = tmp_path / "georectified"
    for ext in (".pgw", ".aux.xml", ".csv", ".json"):
        _mk(d / f"DJI_0904_corrected_georeferenced{ext}")
    listing = G.georeferenced_listing(d)
    assert listing == [], f"sidecars counted as placements: {listing}"


def test_a_tif_and_its_png_twin_are_one_placement(tmp_path):
    """Bio_Station writes both. One placement, one row."""
    from functions import georef as G
    d = tmp_path / "georectified"
    stem = "01_corrected_width=1m_height=1m_GSD=0.001m_per_px_georeferenced"
    _mk(d / f"{stem}.tif")
    _mk(d / f"{stem}.png")
    assert len(G.georeferenced_listing(d)) == 1


def test_nothing_placed_reads_as_nothing_placed(tmp_path):
    from functions import georef as G
    listing = G.georeferenced_listing(tmp_path / "nope",
                                      tmp_path / "also-nope")
    assert listing == []
    _, ok = G.georeferenced_for("DJI_0904_rectified.jpg", listing)
    assert not ok


def test_the_coverage_table_looks_in_both_folders(src):
    body = src[src.index("    def _geo_preview_seeds():"):]
    body = body[:body.index("    def _geo_update_readiness(")]
    assert "georeferenced_listing(" in body
    assert "_geo_results_dir()" in body and 'folder.parent / "georectified"' \
        in body, ("the coverage table checks only one of the two folders a "
                  "survey's outputs can be in")
