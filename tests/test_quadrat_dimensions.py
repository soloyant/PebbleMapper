"""The quadrat's size travels with the output it produced.

The Georeference tab crops the frame away before matching, because the frame
is equipment rather than ground and its straight high-contrast edges attract
features the ortho does not contain. Until now that thickness was typed from
memory into a box defaulting to zero, on a tab that has no way of knowing what
quadrat the photograph was taken with.

It is now written into the rectification record and read back. Three answers
have to stay apart, and the tests are mostly about that: a value the user
typed, a value the record carries, and nobody having said.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from functions import exif_seed as X
from functions import georef as G

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  What the record carries                                                     #
# --------------------------------------------------------------------------- #
def test_the_record_carries_the_quadrat_size():
    rec = X.rectification_record(
        source="DJI_0904.JPG", gsd="0.000760",
        segments_m="1.1850,1.1850,1.1850,1.1850",
        frame_thickness_m="0.0600")
    assert rec["PM_SEGMENTS_M"] == "1.1850,1.1850,1.1850,1.1850"
    assert rec["PM_FRAME_THICKNESS_M"] == "0.0600"


def test_an_output_from_before_this_says_nothing_rather_than_zero():
    """A photograph rectified last month recorded no thickness. That means
    "nobody said", and zero is a different claim -- it means the frame is not
    in the way. Writing zero for the first would quietly answer a question
    nobody asked."""
    rec = X.rectification_record(source="a.jpg", segments_m=None,
                                 frame_thickness_m=None)
    assert "PM_FRAME_THICKNESS_M" not in rec
    assert "PM_SEGMENTS_M" not in rec


def test_the_rectify_path_records_both():
    """The record is built inside functions/orthorectify.rectify_one now --
    the tab used to build it inline, which is why a folder run could not exist
    without a second copy of it. The guarantee is unchanged; the assertion
    follows it out of the closure."""
    from pathlib import Path as _P
    mod = (_P(__file__).resolve().parents[1]
           / "functions" / "orthorectify.py").read_text(encoding="utf-8")
    body = mod[mod.index("def rectify_one("):]
    body = body[:body.index("\ndef is_ready_to_rectify")]
    assert "segments_m=" in body and "frame_thickness_m=" in body
    assert "rectification_record(" in body


def test_the_tab_no_longer_builds_the_record_itself(src):
    body = src[src.index("    def _do_rectify("):]
    body = body[:body.index("\n    # ---------------- The whole folder")]
    assert "rectification_record(" not in body, (
        "the tab builds the record again, so a folder run would need a "
        "second copy of it")
    assert "_or.rectify_one(" in body


# --------------------------------------------------------------------------- #
#  Which value wins                                                            #
# --------------------------------------------------------------------------- #
def test_a_recorded_thickness_fills_an_untouched_box():
    v, note = G.frame_thickness_for({"PM_FRAME_THICKNESS_M": "0.0600"},
                                    0.0, None)
    assert v == pytest.approx(0.060)
    assert "rectification record" in note


def test_a_typed_thickness_is_never_overwritten():
    """Only the user knows whether this photograph is the exception. Replacing
    their number with a recorded one is the same class of mistake as
    rectifying at a default nobody confirmed."""
    v, note = G.frame_thickness_for({"PM_FRAME_THICKNESS_M": "0.0600"},
                                    0.020, None)
    assert v is None
    assert "yours" in note


def test_no_record_leaves_the_box_alone():
    v, note = G.frame_thickness_for({}, 0.0, None)
    assert v is None and note == ""


def test_the_next_photograph_can_still_refill_after_an_auto_fill():
    """The box is bound, so writing it programmatically fires the same event a
    keystroke does. A widget that read its own auto-fill as user input would
    refuse to update after the first photograph -- so "untouched" is decided
    by comparing values, not by listening."""
    v, _ = G.frame_thickness_for({"PM_FRAME_THICKNESS_M": "0.0550"},
                                 0.060, 0.060)
    assert v == pytest.approx(0.055)


def test_zero_is_an_answer_not_an_absence():
    v, note = G.frame_thickness_for({"PM_FRAME_THICKNESS_M": "0.0000"},
                                    0.0, None)
    assert v == pytest.approx(0.0)
    assert "rectification record" in note


def test_a_malformed_record_is_ignored_rather_than_believed():
    for bad in ("abc", "", "-0.05", None):
        v, _ = G.frame_thickness_for({"PM_FRAME_THICKNESS_M": bad}, 0.0, None)
        assert v is None, f"{bad!r} was accepted as a thickness"


def test_the_georeference_tab_asks_that_function(src):
    body = src[src.index("    def _geo_on_quadrat_change():"):]
    body = body[:body.index("    def _geo_fill_seed_from_sources")]
    assert "frame_thickness_for(" in body, (
        "the tab decides this itself again, where it cannot be tested")
    assert "geo_inset_auto" in body


def test_the_orthorectify_tab_offers_the_input(src):
    start = src.index("def build_orthorectify_tab(")
    body = src[start:src.find("\ndef ", start + 1)]
    assert 'label="Frame thickness (m) — optional"' in body
    assert "ortho_frame_thickness_m" in body


# --------------------------------------------------------------------------- #
#  Seeing the corners                                                          #
# --------------------------------------------------------------------------- #
def test_a_marker_is_the_same_size_on_screen_whatever_the_photograph():
    """Drawn at a fixed size in IMAGE pixels, a radius-8 marker on a 4000 px
    photograph fitted into a 768 px column is 1.5 screen pixels -- a dot too
    small to judge a corner by, on the one canvas whose job is judging
    corners. And the magnifier draws the raw image, not the overlay."""
    from functions import orthorectify as O
    big = O.marker_radius(4000, 768)
    small = O.marker_radius(800, 768)
    assert big > small * 4, "the marker does not scale with the photograph"
    # Same size on screen in both: radius / (image px per screen px).
    assert big / O.display_scale(4000, 768) == pytest.approx(
        small / O.display_scale(800, 768))


def test_zooming_in_does_not_change_the_marker_on_screen():
    from functions import orthorectify as O
    fit = O.marker_radius(4000, 768) / O.display_scale(4000, 768)
    one_to_one = O.marker_radius(4000, 4000) / O.display_scale(4000, 4000)
    assert fit == pytest.approx(one_to_one)


def test_the_hit_target_follows_the_marker():
    """4.6 screen pixels at fit-to-width is a coin flip, and a miss is what
    the armed-corner rule then refuses."""
    from functions import orthorectify as O
    assert O.hit_radius(4000, 768) > O.CORNER_HIT_PX
    assert O.hit_radius(768, 768) == pytest.approx(O.CORNER_HIT_PX)


def test_the_scale_is_one_when_nothing_is_resized():
    from functions import orthorectify as O
    assert O.display_scale(768, 768) == pytest.approx(1.0)
    for bad in ((0, 768), (768, 0), (None, 768), ("x", "y")):
        assert O.display_scale(*bad) == 1.0


def test_the_ortho_tab_draws_at_screen_scale(src):
    start = src.index("def build_orthorectify_tab(")
    body = src[start:src.find("\ndef ", start + 1)]
    assert "marker_radius(" in body and "hit_radius(" in body
    assert "_ortho_zoom_to(" in body, "the canvas has no zoom"


# --------------------------------------------------------------------------- #
#  Where the last one was                                                      #
# --------------------------------------------------------------------------- #
def test_the_guide_is_the_last_confirmed_quadrilateral():
    from functions import orthorectify as O
    q = [[1, 1], [2, 1], [2, 2], [1, 2]]
    rows = [("a.jpg", q), ("b.jpg", []), ("c.jpg", [])]
    name, got = O.guide_quadrilateral(rows, 2)
    assert name == "a.jpg"
    assert got == [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]


def test_a_skipped_photograph_does_not_break_the_chain():
    """Confirmed on #1, #2 never touched, opening #3 still shows #1's."""
    from functions import orthorectify as O
    rows = [("a.jpg", [[1, 1], [2, 1], [2, 2], [1, 2]]),
            ("b.jpg", None), ("c.jpg", [])]
    assert O.guide_quadrilateral(rows, 2)[0] == "a.jpg"


def test_the_first_photograph_gets_no_guide():
    from functions import orthorectify as O
    assert O.guide_quadrilateral([("a.jpg", [[1, 1]] * 4)], 0) is None
    assert O.guide_quadrilateral([], 3) is None


def test_a_half_picked_photograph_is_not_a_guide():
    from functions import orthorectify as O
    rows = [("a.jpg", [[1, 1], [2, 1]]), ("b.jpg", [])]
    assert O.guide_quadrilateral(rows, 1) is None


def test_the_guide_never_opens_the_photograph(monkeypatch):
    """It takes filenames and numbers. Nothing about the photograph's content
    is examined, which is what keeps it indifferent to whatever the quadrat is
    made of -- and it must stay that way."""
    import cv2
    import PIL.Image as PILImage
    from functions import orthorectify as O

    def boom(*a, **k):
        raise AssertionError("the guide opened a photograph")
    monkeypatch.setattr(cv2, "imread", boom)
    monkeypatch.setattr(cv2, "imdecode", boom)
    monkeypatch.setattr(PILImage, "open", boom)

    rows = [("a.jpg", [[1, 1], [2, 1], [2, 2], [1, 2]]),
            ("b.jpg", []), ("c.jpg", [])]
    assert O.guide_quadrilateral(rows, 2)[0] == "a.jpg"


def test_orthorectify_does_not_import_pillow():
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1]
           / "functions" / "orthorectify.py").read_text(encoding="utf-8")
    assert "PIL" not in src and "from PIL" not in src


def test_the_guide_is_drawn_only_when_nothing_is_placed(src):
    start = src.index("def build_orthorectify_tab(")
    body = src[start:src.find("\ndef ", start + 1)]
    idx = body.index("state.ortho_guide_shown and state.ortho_guide")
    assert "not state.ortho_corners" in body[idx:idx + 120], (
        "the guide would sit under the corners the user is placing")
