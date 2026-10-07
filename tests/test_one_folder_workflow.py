"""The folder is the default and one file is the exception.

Two tabs each offered two ways to name the file to work on, with the
single-file way first. That order was inverted. Reordering
controls is not what these tests are mostly about, though: three defects were
found while specifying the change, and each of them made the file list -- the
whole point of a folder-first tab -- say something untrue.

Six tests are marked GUARD: they pin behaviour this change had to preserve
rather than behaviour it introduced.

The behavioural counterparts to the source-text assertions below -- rendered
panel order, a guard that cannot disarm itself, a guard a user can actually
satisfy -- live in `test_one_folder_rendered.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from functions import orthorectify as O
from functions import seeds as S

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


def _body(src: str, name: str) -> str:
    """The source of one top-level tab builder.

    Scoped, because several of these strings legitimately appear on OTHER
    tabs -- "Multi-file mode" is also on Digitize -- and a whole-file grep
    would report a failure that is not one.
    """
    start = src.index(f"def {name}(")
    nxt = src.find("\ndef ", start + 1)
    return src[start:nxt if nxt != -1 else len(src)]


# --------------------------------------------------------------------------- #
#  The scale guard must not own its own off switch                             #
# --------------------------------------------------------------------------- #
def test_a_file_nobody_confirmed_is_warned_about():
    """The guard added after the lost survey went quiet after one image.

    `ortho_seg_confirmed` was a single session-global bool that the tool set on
    ITSELF at the moment it warned -- "ask once, not per image". So rectifying
    image 1 at the untouched 1.000 m default warned, and images 2..20 were
    mis-scaled in silence. That is the lost survey again, minus nineteen of the
    twenty chances to notice it.
    """
    per_image = {"a.jpg": {"seg_confirmed": True}}
    lengths = [O.DEFAULT_SEG_M] * 4
    # b and c were never confirmed; confirming a must not cover them.
    for name in ("b.jpg", "c.jpg"):
        assert O.rectifying_at_the_untouched_default(
            lengths, O.seg_confirmed_for(per_image, name)), (
            f"{name} was rectified at the default without a warning")
    assert not O.rectifying_at_the_untouched_default(
        lengths, O.seg_confirmed_for(per_image, "a.jpg"))


def test_confirming_one_file_does_not_confirm_another():
    per_image = {"a.jpg": {"seg_confirmed": True}}
    assert O.seg_confirmed_for(per_image, "a.jpg") is True
    assert O.seg_confirmed_for(per_image, "b.jpg") is False
    assert not O.rectifying_at_the_untouched_default(
        [1.0] * 4, O.seg_confirmed_for(per_image, "a.jpg"))
    assert O.rectifying_at_the_untouched_default(
        [1.0] * 4, O.seg_confirmed_for(per_image, "b.jpg"))


def test_the_global_confirmation_flag_is_gone(src):
    assert "ortho_seg_confirmed: bool" not in src, (
        "a session-global confirmation flag is back; it disables the scale "
        "guard for every file after the first")


# GUARD: pins what must not change.
def test_a_real_length_needs_no_confirmation_at_all():
    """Typing 1.185 once must not produce sixteen warnings. The guard reads
    the VALUE, so a survey measured properly is silent whatever the flags."""
    assert not O.rectifying_at_the_untouched_default([1.185] * 4, False)


# --------------------------------------------------------------------------- #
#  The output-existence check must agree with the writer                       #
# --------------------------------------------------------------------------- #
def test_the_name_checked_for_is_the_name_written(tmp_path):
    """"0/30 files have a saved rectified output", with sixteen of them done.

    The check predicted `<stem>_rectified.jpg` while the writer produced
    `<stem>_rectified_GSD=0.00076m.jpg`, so it could never match. A tab that
    tells a user a morning's work did not happen is worse than one that says
    nothing.
    """
    out = tmp_path / "orthorectified" / "DJI_0904_rectified.jpg"
    out.parent.mkdir()
    written = O.gsd_tagged_name(out, 0.00076)
    assert written.name == "DJI_0904_rectified_GSD=0.00076m.jpg"

    written.write_bytes(b"x")
    path, exists = O.rectified_output_for(out, 0.00076)
    assert exists and path == written


def test_an_output_is_found_even_when_its_gsd_is_not_known(tmp_path):
    """The GSD is measured per photograph, so before opening a file there is
    nothing to predict from. The folder still knows."""
    out = tmp_path / "orthorectified" / "S2_IMG_6744_rectified.jpg"
    out.parent.mkdir()
    (out.parent / "S2_IMG_6744_rectified_GSD=0.000769m.png").write_bytes(b"x")
    path, exists = O.rectified_output_for(out, None)
    assert exists, "a rectified output on disk was reported as missing"
    assert "GSD=0.000769m" in path.name


def test_nothing_is_claimed_when_nothing_was_written(tmp_path):
    out = tmp_path / "orthorectified" / "DJI_0999_rectified.jpg"
    out.parent.mkdir()
    _, exists = O.rectified_output_for(out, 0.0005)
    assert not exists


def test_a_tagged_name_is_not_tagged_twice():
    once = O.gsd_tagged_name("out/a_rectified.jpg", 0.0005)
    assert O.gsd_tagged_name(once, 0.0005) == once


def test_no_gsd_yet_leaves_the_name_alone():
    for bad in (None, 0, 0.0, "", "abc"):
        assert O.gsd_tagged_name("out/a_rectified.jpg", bad).name == \
            "a_rectified.jpg"


# --------------------------------------------------------------------------- #
#  What a row says before anything is opened                                   #
# --------------------------------------------------------------------------- #
def test_a_row_states_corners_length_and_output(tmp_path):
    out = tmp_path / "orthorectified" / "q_rectified.jpg"
    out.parent.mkdir()
    (out.parent / "q_rectified_GSD=0.0008m.jpg").write_bytes(b"x")
    row = O.folder_row("q.JPG", [[1, 1], [2, 1], [2, 2], [1, 2]],
                       [1.185] * 4, out)
    assert row["corners"] == "4/4"
    assert row["length"] == "1.185 m"
    assert row["done"] is True
    assert "GSD=0.0008m" in row["output"]


def test_an_interrupted_file_reads_as_interrupted(tmp_path):
    row = O.folder_row("q.JPG", [[1, 1], [2, 1]], [1.185] * 4,
                       tmp_path / "orthorectified" / "q_rectified.jpg")
    assert row["corners"] == "2/4"
    assert row["done"] is False


def test_the_untouched_default_is_marked_in_the_row(tmp_path):
    """As a number, never as a 'confirmed' tick: confirmation is not a
    property of a photograph, and a tick asserts what a default cannot."""
    row = O.folder_row("q.JPG", [], [1.0] * 4,
                       tmp_path / "orthorectified" / "q_rectified.jpg")
    assert "1.000 m" in row["length"] and "default" in row["length"]
    assert "confirm" not in row["length"].lower()


# --------------------------------------------------------------------------- #
#  Corners are durable when they are picked, not when they are used            #
# --------------------------------------------------------------------------- #
def test_corners_are_saved_as_soon_as_the_fourth_is_placed(src):
    """They used to reach disk only inside `_do_rectify`. Picking corners on
    sixteen photographs and then losing the socket -- which this application
    does -- threw away sixty-four precision
    clicks and left the folder reading 0/4 everywhere."""
    handler = src[src.index("    def _on_mouse(e):"):]
    handler = handler[:handler.index("    def _load_image():")]
    assert "_save_ortho_corners" in handler, (
        "the corner-click handler does not persist a completed pick")
    assert "len(state.ortho_corners) == 4" in handler


# GUARD: pins what must not change.
def test_the_sidecar_round_trips(tmp_path):
    from gui.app import _save_ortho_corners, _load_ortho_corners
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"x")
    corners = [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]
    assert _save_ortho_corners(str(photo), corners)
    assert _load_ortho_corners(str(photo)) == corners


# --------------------------------------------------------------------------- #
#  A resolver refuses; it does not raise                                       #
# --------------------------------------------------------------------------- #
def test_resolve_seed_survives_a_missing_typed_coordinate(tmp_path):
    photo = tmp_path / "q.png"
    photo.write_bytes(b"x")
    r = S.resolve_seed(photo, None, None, None, "")
    assert r.ok is False and r.x is None


# GUARD: pins what must not change.
def test_a_typed_pair_still_works(tmp_path):
    r = S.resolve_seed(tmp_path / "q.png", None, None, (3.0, 49.0), "EPSG:4326")
    assert r.ok and (r.x, r.y) == (3.0, 49.0)


# --------------------------------------------------------------------------- #
#  Layout: the folder first, the single file kept as the exception             #
# --------------------------------------------------------------------------- #
def test_the_survey_panel_comes_first_and_is_the_one_left_open(src):
    body = _body(src, "build_georeference_tab")
    survey = body.index('ui.expansion("Check a whole survey')
    locate = body.index('ui.expansion("Locate one quadrat')
    pins = body.index('ui.expansion("Pin quadrats on the ortho')
    assert survey < locate < pins, "section 3 is back in the old order"
    assert "value=True" in body[survey:survey + 220]
    assert "value=False" in body[locate:locate + 260], (
        "the single-quadrat editor is expanded on load again")


def test_orthorectify_leads_with_the_folder(src):
    body = _body(src, "build_orthorectify_tab")
    folder = body.index('label="Folder of photographs"')
    single = body.index('"Source photograph"')
    assert folder < single, "the single-photograph picker is first again"
    assert "Multi-file mode" not in body, (
        "the folder is described as an optional mode again")


def test_both_single_file_inputs_survive_as_an_exception(src):
    """Demoted, not deleted. The single file is the exception, and Digitize
    hands a photograph to Georeference from an
    arbitrary directory -- deleting the box breaks that silently."""
    for tab, field in (("build_orthorectify_tab", '"ortho_src_path"'),
                       ("build_georeference_tab", '"geo_quadrat_path"')):
        body = _body(src, tab)
        assert field in body, f"{tab} lost its single-file input"
        idx = body.index(field)
        before = body[max(0, idx - 700):idx]
        assert "ui.expansion(" in before, (
            f"{tab}'s single-file input is not behind a disclosure")


# GUARD: pins what must not change.
def test_the_digitize_handoff_still_has_somewhere_to_land(src):
    body = src[src.index("def _dig_go_georeference"):]
    body = body[:body.index("\n    with ui.row")]
    assert "state.geo_quadrat_path = state.dig_src_path" in body


def test_the_survey_button_states_its_own_precondition(src):
    body = _body(src, "build_georeference_tab")
    idx = body.index('ui.button("Check all quadrats"')
    window = body[idx:idx + 900]
    assert 'bind_enabled_from' in window and '"geo_ready"' in window, (
        "the survey button can be pressed with nothing configured")
    assert '"geo_not_ready"' in window, (
        "the button does not say what it is waiting for")


def test_the_whole_ortho_sweep_is_not_the_first_thing_you_see(src):
    """It costs roughly an hour per unseeded quadrat, and it sat directly
    above a primary-coloured button as the panel's first control.

    Below the button, and priced in visible text rather than a tooltip. Not
    behind a second collapsible: this application deliberately has no
    expansion nested inside another (test_gui_ergonomics).
    """
    body = _body(src, "build_georeference_tab")
    button = body.index('ui.button("Check all quadrats"')
    sweep = body.index('"Also search the whole ortho')
    assert button < sweep, "the expensive sweep is above the run button again"
    cost = body[sweep:sweep + 900]
    assert "hour per unseeded quadrat" in cost, (
        "the cost is not stated where it can be read without hovering")
    assert "ui.expansion(" not in body[sweep - 400:sweep], (
        "the sweep was put behind a nested collapsible")


def test_a_results_row_opens_the_editor_it_points_at(src):
    body = src[src.index("def _geo_open_from_batch"):]
    body = body[:body.index("    def _sd_ortho_gsd")]
    assert "_locate_panel.value = True" in body, (
        "clicking a result leaves the collapsed editor shut, which reads as "
        "nothing having happened")


# --------------------------------------------------------------------------- #
#  One seed resolver, and only one                                             #
# --------------------------------------------------------------------------- #
# GUARD: pins what must not change.
def test_the_gui_does_not_carry_a_second_seed_resolver(src):
    """The split cost a survey: check_quadrats resolved one way and the
    single-quadrat panel another, and half the quadrats came back "no seed"."""
    assert len(re.findall(r"\bdef resolve_seed\b", src)) == 0
    assert "_sd.resolve_seed(" in src, "the GUI no longer calls the resolver"


# GUARD: pins what must not change.
def test_the_two_provenance_vocabularies_still_line_up(tmp_path):
    """The survey table and the coverage list describe the same fact in two
    different words -- `check_survey` records ``"list"`` where the GUI records
    ``"the seed list"``. That is fine while the mapping between them is
    known; it stops being fine silently, so it is asserted here.

    Calling `resolve_seed` twice and comparing would prove nothing: both
    paths already call it. What can actually drift is the vocabulary.
    """
    photo = tmp_path / "q_good.png"
    photo.write_bytes(b"x")
    csv = tmp_path / "seeds.csv"
    csv.write_text("photo,x,y,crs\nq_good,3.0,49.0,EPSG:4326\n",
                   encoding="utf-8")
    r = S.resolve_seed(photo, S.load_seed_table(csv), None, (None, None), "",
                       search_dirs=[])
    assert r.provenance == "the seed list"

    # And the survey path's own word for it, read off the survey path.
    survey_src = (Path(S.__file__)).read_text(encoding="utf-8")
    block = survey_src[survey_src.index("def check_survey"):]
    assert '"list"' in block, (
        "check_survey no longer records seed_source='list'; the coverage "
        "list and the results table can no longer be compared")


# --------------------------------------------------------------------------- #
#  A folder can say what size each photograph was measured at                  #
# --------------------------------------------------------------------------- #
def test_the_recorded_length_survives_a_reload(tmp_path):
    from gui.app import _save_ortho_corners, _load_ortho_segments
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"x")
    _save_ortho_corners(str(photo), [[1, 1], [2, 1], [2, 2], [1, 2]],
                        [1.185, 1.185, 1.185, 1.185])
    assert _load_ortho_segments(str(photo)) == [1.185] * 4


def test_a_length_line_is_not_read_as_a_corner(tmp_path):
    """The sidecar format is shared with files written before this existed,
    and an older one must not gain a fifth corner from a `seg=` line."""
    from gui.app import _save_ortho_corners, _load_ortho_corners
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"x")
    _save_ortho_corners(str(photo), [[1, 1], [2, 1], [2, 2], [1, 2]],
                        [1.185] * 4)
    assert _load_ortho_corners(str(photo)) == [[1, 1], [2, 1], [2, 2], [1, 2]]


def test_an_older_sidecar_reports_no_length_rather_than_a_default(tmp_path):
    """A file nobody has opened has no recorded length. Showing the session's
    current value in its place is the misinformation this record exists to
    prevent -- the survey went out at 1.000 m and every output agreed with
    itself."""
    from gui.app import _load_ortho_segments
    photo = tmp_path / "q.jpg"
    photo.write_bytes(b"x")
    (tmp_path / "q_corners.txt").write_text(
        "1,1\n2,1\n2,2\n1,2\n", encoding="utf-8")
    assert _load_ortho_segments(str(photo)) is None

    row = O.folder_row("q.jpg", [[1, 1]], None,
                       tmp_path / "orthorectified" / "q_rectified.jpg")
    assert row["length"] == "\u2014", (
        "a photograph with no recorded length claimed one")


def test_the_row_reads_the_files_own_length_not_the_sessions(src):
    body = src[src.index("    def _ortho_rows():"):]
    body = body[:body.index("    def _ortho_refresh_table(")]
    assert "_load_ortho_segments" in body
    assert "state.ortho_seg_lengths" not in body, (
        "the list falls back to the session's current length, so every "
        "unopened file claims a size nobody chose for it")


def test_one_function_names_the_rectified_output(src):
    """Three used to. The writer said JPEG in orthorectified/; the browse
    default and the autofill both said PNG beside images/; and the
    already-rectified check guessed a fourth name that matched none of them.
    """
    body = _body(src, "build_orthorectify_tab")
    assert "_rectified.png" not in body, (
        "a second output namer is back, and it disagrees with the writer")
