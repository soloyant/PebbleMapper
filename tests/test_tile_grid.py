"""One tile grid fixed to the ortho, overlapping by a quadrat diagonal.

Two defects
in the per-seed windows this replaces:

**Nothing could be shared.** `plan_search` builds its windows around each
quadrat's seed, so two quadrats photographed a metre apart read overlapping
ground through rectangles that never coincide. Measured on a real 20-quadrat
survey: 1620 window reads, 1620 of them distinct, 1307 megapixels pulled off a
588-megapixel ortho -- 2.2x the whole image. On a fixed grid the same search
wants 256 distinct tiles and 207 megapixels.

**The overlap was too small for a rotated quadrat.** `plan_search` steps by
`size - quad_px`, which guarantees an AXIS-ALIGNED quadrat is whole in some
tile. Every quadrat in this project is rotated, and a square of side s turned
45 degrees spans s*sqrt(2). A quadrat could therefore be cut in every tile that
contained part of it, and the matcher would be asked to place something it had
never seen whole.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from functions import georef as G


SHAPE = (4000, 5000)        # rows, cols
GSD = 0.004                 # 4 mm/px, close to the Swimbeach ortho
EXTENT = 1.2                # a 1.2 m quadrat -> 300 px


def tiles(shape=SHAPE, gsd=GSD, extent=EXTENT, tile_px=None):
    return list(G.tile_grid(shape, gsd, extent, tile_px=tile_px))


# --------------------------------------------------------------------------- #
#  The grid covers the ortho                                                   #
# --------------------------------------------------------------------------- #
def test_every_pixel_of_the_ortho_is_in_some_tile():
    rows, cols = 900, 1400
    covered = np.zeros((rows, cols), bool)
    for r0, c0, r1, c1 in tiles((rows, cols)):
        covered[r0:r1, c0:c1] = True
    assert covered.all(), f"{(~covered).sum()} pixels in no tile"


def test_tiles_stay_inside_the_ortho():
    rows, cols = 900, 1400
    for r0, c0, r1, c1 in tiles((rows, cols)):
        assert 0 <= r0 < r1 <= rows
        assert 0 <= c0 < c1 <= cols


def test_a_tiny_ortho_is_one_tile():
    assert len(tiles((100, 120))) == 1


# --------------------------------------------------------------------------- #
#  The overlap is a DIAGONAL, which is the whole point                         #
# --------------------------------------------------------------------------- #
def test_the_overlap_is_at_least_the_quadrats_diagonal():
    """Not its extent. A square of side s rotated 45 degrees spans s*sqrt(2),
    and that is the case this project always has."""
    ts = tiles()
    rs = sorted({t[0] for t in ts})
    cs = sorted({t[1] for t in ts})
    size = G.MAX_WINDOW_PX
    diag_px = EXTENT * math.sqrt(2.0) / GSD
    for axis in (rs, cs):
        if len(axis) < 2:
            continue
        step = axis[1] - axis[0]
        assert size - step >= diag_px - 1, (
            f"overlap {size - step:.0f} px < diagonal {diag_px:.0f} px")


@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 75])
def test_a_rotated_quadrat_is_whole_in_at_least_one_tile(angle):
    """The guarantee the old overlap did not make. A quadrat is placed by
    matching its content; one cut in every tile that touches it can never be
    matched, and the failure looks like 'not in this ortho'."""
    rows, cols = 3000, 3000
    ts = tiles((rows, cols))
    side_px = EXTENT / GSD
    th = math.radians(angle)
    # half-extent of the rotated square's bounding box
    hw = side_px * (abs(math.cos(th)) + abs(math.sin(th))) / 2.0

    rng = np.random.default_rng(3)
    for _ in range(60):
        cy = rng.uniform(hw, rows - hw)
        cx = rng.uniform(hw, cols - hw)
        r0, r1 = cy - hw, cy + hw
        c0, c1 = cx - hw, cx + hw
        whole = any(t[0] <= r0 and t[1] <= c0 and t[2] >= r1 and t[3] >= c1
                    for t in ts)
        assert whole, (f"quadrat at ({cx:.0f},{cy:.0f}) rotated {angle} deg "
                       f"is cut in every tile")


# --------------------------------------------------------------------------- #
#  The grid is the SAME for every quadrat -- that is what allows sharing       #
# --------------------------------------------------------------------------- #
def test_two_seeds_ask_for_the_same_rectangles():
    """The per-seed windows this replaces produced 1620 distinct rectangles for
    1620 reads -- nothing to share. Neighbours must now agree."""
    a = G.tiles_for_seed(SHAPE, GSD, EXTENT, (1500, 2000), 10.0)
    b = G.tiles_for_seed(SHAPE, GSD, EXTENT, (1520, 2030), 10.0)
    assert set(a) & set(b), "neighbouring seeds share no tile at all"
    shared = len(set(a) & set(b)) / max(1, min(len(a), len(b)))
    assert shared > 0.5, f"only {shared:.0%} shared between adjacent seeds"


def test_the_grid_does_not_move_with_the_seed():
    for seed in ((1000, 1000), (1007, 1013), (2500, 400)):
        for t in G.tiles_for_seed(SHAPE, GSD, EXTENT, seed, 10.0):
            assert t in set(tiles()), "tile is off the fixed grid"


# --------------------------------------------------------------------------- #
#  Ordering and reach                                                          #
# --------------------------------------------------------------------------- #
def test_the_nearest_tile_comes_first_so_a_good_seed_stops_early():
    seed = (1500, 2000)
    ts = G.tiles_for_seed(SHAPE, GSD, EXTENT, seed, 10.0)
    assert ts
    d = [math.hypot((t[0] + t[2]) / 2 - seed[0], (t[1] + t[3]) / 2 - seed[1])
         for t in ts]
    # Non-decreasing to within rounding: equidistant tiles are common on a
    # regular grid, and the sort key is squared distance while this recomputes
    # with hypot, so exact equality fails on a 1-ULP tie rather than on order.
    for a, b in zip(d, d[1:]):
        assert b >= a - 1e-6, f"tiles are not nearest-first: {a} then {b}"
    first = ts[0]
    assert first[0] <= seed[0] < first[2] and first[1] <= seed[1] < first[3], \
        "the first tile does not even contain the seed"


def test_the_search_radius_is_covered():
    seed = (1500, 2000)
    radius_m = 8.0
    ts = G.tiles_for_seed(SHAPE, GSD, EXTENT, seed, radius_m)
    r_px = radius_m / GSD
    for ang in range(0, 360, 30):
        y = seed[0] + r_px * math.sin(math.radians(ang))
        x = seed[1] + r_px * math.cos(math.radians(ang))
        if not (0 <= y < SHAPE[0] and 0 <= x < SHAPE[1]):
            continue
        assert any(t[0] <= y < t[2] and t[1] <= x < t[3] for t in ts), \
            f"a point on the {radius_m} m radius is in no tile"


def test_a_bigger_radius_never_asks_for_fewer_tiles():
    seed = (1500, 2000)
    counts = [len(G.tiles_for_seed(SHAPE, GSD, EXTENT, seed, r))
              for r in (2.0, 5.0, 10.0, 20.0)]
    assert counts == sorted(counts)


# --------------------------------------------------------------------------- #
#  The measured reason MAX_WINDOW_PX is not raised                             #
# --------------------------------------------------------------------------- #
def test_the_tile_size_stays_near_twice_the_quadrat():
    """Measured on 41 quadrats with a known reference: at 900 px, 13/21
    Bio_Station and 9/20 Swimbeach placed; at 2000 px, 5/21 and 2/20. Accuracy
    barely moves, so a bigger tile costs only placements -- which is what
    spreading a capped keypoint budget over more ground predicts."""
    assert 600 <= G.MAX_WINDOW_PX <= 1000
