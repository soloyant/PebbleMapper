"""Picking a quadrat's corners, and picking one of them again.

The Orthorectify tab has told the user "All 4
corners picked. Click any corner to re-pick it" since it was written, while the
handler ignored where the click landed and cycled 1,2,3,4 instead. So the one
corner you wanted was the one you could not choose, which the user experiences
as "there is no way to modify a corner once it is placed".

Underneath that sat a worse one. `_load_image` set the pick index to
`len(saved_corners)`; a full set of four gave 4, and `4 < 4` is false, so the
next click appended a FIFTH corner and Rectify refused for good with "Pick all
4 corners first (5/4 done)". It was reachable on the first click of any image
with a saved `<stem>_corners.txt` -- every previously worked survey has those.

The rule lives in `functions/orthorectify` rather than the tab's closure
precisely because nothing could exercise a closure except a person clicking.
"""
from __future__ import annotations

import pytest

from functions import orthorectify as O


SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


# --------------------------------------------------------------------------- #
#  Clicking a corner selects THAT corner                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_a_click_on_a_corner_finds_that_corner(i):
    cx, cy = SQUARE[i]
    assert O.corner_under(SQUARE, cx, cy) == i


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_a_click_near_a_corner_still_finds_it(i):
    cx, cy = SQUARE[i]
    assert O.corner_under(SQUARE, cx + 10, cy - 10) == i


def test_a_click_on_open_ground_finds_nothing():
    """Otherwise a corner could never be placed once four exist."""
    assert O.corner_under(SQUARE, 200, 200) is None


def test_just_outside_the_radius_is_a_miss():
    r = O.CORNER_HIT_PX
    assert O.corner_under(SQUARE, 100 + r + 2, 100) is None
    assert O.corner_under(SQUARE, 100 + r - 2, 100) == 0


def test_the_nearest_corner_wins_not_the_first():
    """Two corners can sit closer together than the hit radius on a quadrat
    photographed at a sharp angle."""
    close = [[100, 100], [112, 100]]
    assert O.corner_under(close, 111, 100) == 1
    assert O.corner_under(close, 101, 100) == 0


def test_nothing_placed_yet_is_not_a_hit():
    assert O.corner_under([], 100, 100) is None
    assert O.corner_under(None, 100, 100) is None


def test_a_malformed_corner_does_not_raise():
    assert O.corner_under([[1, 2], None, "x", [100, 100]], 100, 100) == 3


# --------------------------------------------------------------------------- #
#  A click that missed places a corner — and never a fifth                     #
# --------------------------------------------------------------------------- #
def test_the_fifth_corner_is_impossible():
    """The bug: index 4 with four corners already placed. Every image with a
    saved corners file started in exactly this state."""
    out, nxt = O.place_corner(SQUARE, 4, 50, 50)
    assert len(out) == 4, "a fifth corner would deadlock Rectify"
    assert out[0] == [50, 50]
    assert nxt == 1


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5, 8])
def test_no_index_can_produce_more_than_four(idx):
    out, nxt = O.place_corner(SQUARE, idx, 7, 7)
    assert len(out) == 4
    assert 0 <= nxt <= 3


def test_placing_into_an_empty_set_builds_up_in_order():
    corners, idx = [], 0
    for k in range(4):
        corners, idx = O.place_corner(corners, idx, k * 10, k * 10)
    assert corners == [[0, 0], [10, 10], [20, 20], [30, 30]]
    assert idx == 0, "wraps back to the first corner for re-picking"


def test_replacing_keeps_the_other_three_untouched():
    out, _ = O.place_corner(SQUARE, 2, 999, 888)
    assert out[2] == [999, 888]
    assert out[0] == SQUARE[0] and out[1] == SQUARE[1] and out[3] == SQUARE[3]


def test_the_input_list_is_not_mutated():
    """The tab keeps a per-image cache of corners; mutating in place would
    edit the cached copy of a different image."""
    original = [list(c) for c in SQUARE]
    O.place_corner(SQUARE, 0, 1, 1)
    assert SQUARE == original


def test_a_saved_set_of_four_is_immediately_editable():
    """End to end on the real path: load four saved corners, click one, move
    it. Before the fix the click appended a fifth and Rectify was dead."""
    corners = [list(c) for c in SQUARE]
    idx = len(corners) % 4            # what _load_image now does
    assert idx == 0

    hit = O.corner_under(corners, 300, 300)   # the user clicks corner 3
    assert hit == 2
    idx = hit
    corners, idx = O.place_corner(corners, idx, 305, 295)
    assert len(corners) == 4
    assert corners[2] == [305, 295]
