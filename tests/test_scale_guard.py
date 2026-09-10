"""Guards against a survey that is silently the wrong size.

Twenty Swimbeach quadrats were rectified with the segment lengths left at
their 1.0 m default. The frame is 1.185 m. Every output therefore declared a
ground sample distance 18.5% wrong, that figure went into every filename, the
Georeference tab read it back, and the matcher refused all twenty on scale --
`scale_tolerance` is 15%. The only symptom, a day later, was "0 of 20 located",
and the message blamed the fit rather than the scale.

Three defences, and it matters which one catches what:

* `size_disagreement` catches the tool NOT doing what it was told.
* `rectifying_at_the_untouched_default` catches being told the wrong thing --
  which is what actually happened, and which no property of the output can
  reveal, since 1.0 m in gives a perfectly self-consistent 1.0 m out.
* segment lengths carrying over between images means the right value is typed
  once for a survey rather than twenty times.
"""
from __future__ import annotations

import pytest

from functions import orthorectify as O


SWIM = 1.185          # the real Swimbeach quadrat frame


# --------------------------------------------------------------------------- #
#  What the output claims about itself                                         #
# --------------------------------------------------------------------------- #
def test_a_rectified_image_asserts_a_physical_size():
    assert O.declared_size_m((2370, 2370), 0.0005) == pytest.approx(1.185)


def test_junes_files_agree_with_the_real_frame():
    claimed, asked, wrong = O.size_disagreement((2370, 2370), 0.0005, [SWIM] * 4)
    assert claimed == pytest.approx(SWIM, abs=1e-3)
    assert wrong is False


def test_todays_files_are_caught_against_the_real_frame():
    """1518 px x 0.000659 m = 1.000 m, but the frame is 1.185 m."""
    claimed, asked, wrong = O.size_disagreement((1518, 1518), 0.000659,
                                                [SWIM] * 4)
    assert claimed == pytest.approx(1.000, abs=1e-3)
    assert wrong is True


def test_the_output_check_cannot_catch_a_wrong_instruction():
    """The honest limit of it. Enter 1.0 and get 1.0: the tool obeyed, the
    image is self-consistent, and nothing here can tell it is wrong. This is
    precisely the case that lost the survey, which is why the second guard
    exists."""
    claimed, asked, wrong = O.size_disagreement((1518, 1518), 0.000659,
                                                [1.0] * 4)
    assert wrong is False


def test_a_non_square_output_is_measured_on_its_long_side():
    assert O.declared_size_m((1000, 2000), 0.001) == pytest.approx(2.0)


@pytest.mark.parametrize("segs", [None, [], ["x"], [0.0]])
def test_a_missing_or_nonsense_segment_length_is_not_an_error(segs):
    claimed, asked, wrong = O.size_disagreement((100, 100), 0.01, segs)
    assert wrong is False


# --------------------------------------------------------------------------- #
#  The guard that would actually have caught it                                #
# --------------------------------------------------------------------------- #
def test_rectifying_at_the_untouched_default_is_flagged():
    assert O.rectifying_at_the_untouched_default([1.0, 1.0, 1.0, 1.0]) is True


def test_a_length_the_user_set_is_not_flagged():
    assert O.rectifying_at_the_untouched_default([SWIM] * 4) is False


def test_it_asks_once_not_per_image():
    """Twenty warnings for twenty photographs is a warning nobody reads."""
    assert O.rectifying_at_the_untouched_default([1.0] * 4,
                                                 confirmed=True) is False


def test_one_edited_side_counts_as_touched():
    """A rectangular quadrat: 1.0 x 1.4 is a deliberate choice, not a
    default."""
    assert O.rectifying_at_the_untouched_default([1.0, 1.4, 1.0, 1.4]) is False


@pytest.mark.parametrize("segs", [None, [], "junk"])
def test_nonsense_does_not_trigger_the_question(segs):
    assert O.rectifying_at_the_untouched_default(segs) is False


# --------------------------------------------------------------------------- #
#  The refusal names the likely cause                                          #
# --------------------------------------------------------------------------- #
def test_the_scale_refusal_names_the_declared_size():
    """'The fitted scale is 49% away' is true and useless. A survey refused on
    scale is almost never twenty coincidences -- it is one number entered
    once, and the message should say so."""
    import inspect
    from functions import georef as G
    src = inspect.getsource(G.match_quadrat)
    i = src.index("The fitted scale is")
    block = src[max(0, i - 800):i + 400]
    assert "m across" in block, "the refusal no longer states the declared size"
    assert "segment lengths" in block, \
        "the refusal no longer points at the likely cause"
    assert "quadrat_gray.shape" in block, \
        "the size is being computed from the rescaled array, not the quadrat"
