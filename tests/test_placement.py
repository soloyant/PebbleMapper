"""A placement, and every way a person may move it.

These run against the pure functions rather than a browser. That separation
is the point: without it every behavioural requirement would have lived in a
NiceGUI event handler no test could reach.

The agreement-score tests are synthetic. The reference is Bio_Station quadrat 14
(0.403 at the surveyed placement, 0.006 five centimetres east, -0.004 rotated
two degrees), but that is 13 GB of field data outside the repository and this
suite is deliberately self-contained. What is tested here is that the function
behaves the way those measurements describe: high where the imagery agrees,
collapsing under a few centimetres of translation or a couple of degrees of
rotation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from functions import placement as P


ORTHO_GSD = 0.0024          # a UAV ortho, mm-scale
QUAD_GSD = 0.001            # a rectified close-up
SHAPE = (1000, 1000)        # a 1 m quadrat at 1 mm/px
ORIGIN = (273000.0, 5307000.0)


def _p(rot=0.0, east=273010.0, north=5306990.0, scale=QUAD_GSD, shape=SHAPE):
    return P.Placement(east, north, rot, scale, shape)


# --------------------------------------------------------------------------- #
#  The transform, and reading one back                                         #
# --------------------------------------------------------------------------- #
def test_the_centre_is_where_the_placement_says_it_is():
    p = _p(rot=37.0)
    assert P.corners_world(p).mean(axis=0) == pytest.approx(
        [p.easting, p.northing], abs=1e-9)


def test_the_transform_reflects_because_rows_run_down_and_northing_runs_up():
    """A quadrat pixel frame is row-down; the world is north-up.

    So the transform is a reflection and its determinant is negative. Getting
    this wrong mirrors every placement while still looking like a square.
    """
    M = P.matrix_from_placement(_p(rot=20.0))
    assert np.linalg.det(M[:2, :2]) < 0


def test_rotation_is_the_bearing_of_the_col_axis_from_east():
    p = _p(rot=30.0)
    c = P.corners_world(p)
    along_col = c[1] - c[0]          # (0,0) -> (cols,0)
    assert math.degrees(math.atan2(along_col[1], along_col[0])) == \
        pytest.approx(30.0, abs=1e-6)


def test_a_fitted_matrix_round_trips_through_a_placement():
    """Requirement 15a: a fit keeps its own scale.

    A fitted scale is near but not equal to the GSD ratio -- accepted
    Bio_Station fits sit within 2% -- so snapping it here would make an
    untouched, merely-kept placement differ from the fit it came from, and the
    recorded edit delta would report a change nobody made.
    """
    fitted_scale = QUAD_GSD * 1.019            # 1.9% off, as a real fit is
    p = _p(rot=-166.0, scale=fitted_scale)
    M = P.matrix_from_placement(p)
    back = P.placement_from_matrix(M, SHAPE)
    assert back.scale_m_per_px == pytest.approx(fitted_scale, rel=1e-12)
    assert back.rotation_deg == pytest.approx(-166.0, abs=1e-9)
    assert back.easting == pytest.approx(p.easting, abs=1e-6)
    np.testing.assert_allclose(P.matrix_from_placement(back), M, atol=1e-9)


def test_a_seed_placement_is_north_up_at_the_quadrats_own_scale():
    p = P.placement_from_seed((273010.0, 5306990.0), SHAPE, QUAD_GSD, ORTHO_GSD)
    assert p.rotation_deg == 0.0
    assert p.scale_m_per_px == pytest.approx(QUAD_GSD)
    assert P.corners_world(p).mean(axis=0) == pytest.approx(
        [273010.0, 5306990.0], abs=1e-9)
    span = P.corners_world(p)[:, 0].ptp()
    assert span == pytest.approx(SHAPE[1] * QUAD_GSD, rel=1e-9)


# --------------------------------------------------------------------------- #
#  Display coordinates                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("disp_scale", [1.0, 2.5])
def test_display_coordinates_round_trip_to_world(disp_scale):
    p = _p(rot=20.0)
    d = P.corners_display(p, ORIGIN, ORTHO_GSD, disp_scale)
    back = np.array([P.world_from_display(x, y, ORIGIN, ORTHO_GSD, disp_scale)
                     for x, y in d])
    err = np.hypot(*(back - P.corners_world(p)).T).max()
    assert err < 0.5 * ORTHO_GSD


def test_a_rotated_placement_is_not_drawn_axis_aligned():
    """The defect the shipped overlay had, stated as a test.

    overlay_figure placed the quadrat by the axis-aligned bounding box of its
    rotated corners, so whatever the rotation the drawn shape came out square to
    the page -- about 21 degrees off the outline beside it. Comparing corners
    derived from the matrix against the matrix would not have caught it; this
    does.
    """
    d = P.corners_display(_p(rot=20.0), ORIGIN, ORTHO_GSD)
    xs, ys = d[:, 0], d[:, 1]
    side = np.hypot(*(d[1] - d[0]))
    # An axis-aligned square has two corners sharing each x and each y.
    assert xs.ptp() > side * 1.2
    assert ys.ptp() > side * 1.2
    assert min(abs(xs[0] - xs[1]), abs(ys[0] - ys[1])) > 0.05 * side


# --------------------------------------------------------------------------- #
#  Moving it                                                                   #
# --------------------------------------------------------------------------- #
def test_translate_moves_exactly_and_changes_nothing_else():
    p = _p(rot=12.0)
    q = P.translate(p, 0.25, -0.75)
    assert q.easting == pytest.approx(p.easting + 0.25, abs=1e-12)
    assert q.northing == pytest.approx(p.northing - 0.75, abs=1e-12)
    assert q.rotation_deg == p.rotation_deg
    assert q.scale_m_per_px == p.scale_m_per_px


def test_nudge_moves_exactly_one_ortho_pixel():
    p = _p()
    assert P.nudge(p, 1, 0, ORTHO_GSD).easting == \
        pytest.approx(p.easting + ORTHO_GSD, abs=1e-12)
    # A row is DOWN the image, which is south.
    assert P.nudge(p, 0, 1, ORTHO_GSD).northing == \
        pytest.approx(p.northing - ORTHO_GSD, abs=1e-12)


def test_setting_a_rotation_turns_about_an_unmoved_centre():
    p = _p(rot=0.0)
    q = P.set_rotation(p, 30.0)
    assert P.corners_world(q).mean(axis=0) == pytest.approx(
        P.corners_world(p).mean(axis=0), abs=1e-9)
    c0, c1 = P.corners_world(p), P.corners_world(q)
    centre = c0.mean(axis=0)
    a = math.radians(30.0)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    expected = (R @ (c0 - centre).T).T + centre
    assert np.abs(expected - c1).max() < 0.5 * ORTHO_GSD


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_dragging_a_corner_rotates_about_the_opposite_one(i):
    """Requirement 16: the corner follows the RAY, not the pointer.

    Scale is fixed and the opposite corner is pinned, so the dragged corner is
    confined to a circle of the diagonal's length. An arbitrary target off that
    circle is unreachable, and a test demanding the corner reach it would be
    asking for something geometrically impossible.
    """
    p = _p(rot=15.0)
    before = P.corners_world(p)
    pivot = before[(i + 2) % 4]
    target = pivot + np.array([1.7, 0.4])       # deliberately not on the circle

    q = P.drag_corner(p, i, target)
    after = P.corners_world(q)

    assert np.hypot(*(after[(i + 2) % 4] - pivot)) < ORTHO_GSD
    assert q.scale_m_per_px == pytest.approx(p.scale_m_per_px, rel=1e-12)
    want = math.degrees(math.atan2(*(target - pivot)[::-1]))
    got = math.degrees(math.atan2(*(after[i] - pivot)[::-1]))
    assert abs((got - want + 180) % 360 - 180) < 0.1
    assert np.hypot(*(after[i] - pivot)) == pytest.approx(
        np.hypot(*(before[i] - pivot)), rel=1e-9)


def test_dragging_onto_the_pivot_holds_the_last_rotation():
    p = _p(rot=15.0)
    pivot = P.corners_world(p)[2]
    assert P.drag_corner(p, 0, pivot) == p


def test_a_placement_has_no_way_to_change_scale():
    """Requirement 18. Scale is known from two GSDs and a square of stated size,
    so a control over it can only introduce error -- and a 15% scale mistake is
    a 15% error on every clast diameter in the quadrat."""
    import dataclasses
    p = _p()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.scale_m_per_px = 0.002

    # No editing operation may alter scale, whatever it does to the rest.
    edits = [lambda q: P.translate(q, 0.3, -0.2),
             lambda q: P.nudge(q, 2, -3, ORTHO_GSD),
             lambda q: P.set_rotation(q, 88.0),
             lambda q: P.drag_corner(q, 1, P.corners_world(q)[3] + [2.0, 1.0])]
    for edit in edits:
        assert edit(p).scale_m_per_px == pytest.approx(p.scale_m_per_px,
                                                       rel=1e-12)


# --------------------------------------------------------------------------- #
#  Reset and the keep gate                                                     #
# --------------------------------------------------------------------------- #
def test_reset_restores_the_start_after_any_number_of_edits():
    """Requirement 19: Reset returns to the start in ONE step.

    A frozen Placement makes this structural equality rather than an undo
    stack -- keep the starting value, hand it back. Five successive edits must
    not accumulate into something a single Reset cannot undo.
    """
    start = _p(rot=15.0)
    cur = start
    for k in range(5):
        cur = P.translate(cur, 0.11 * (k + 1), -0.07 * (k + 1))
        cur = P.set_rotation(cur, cur.rotation_deg + 9.0)
        cur = P.nudge(cur, 1, 1, ORTHO_GSD)
    assert cur != start
    assert start == start          # the Reset itself: hand the start back
    assert P.placement_delta(start, start)["translation_m"] == 0.0
    assert P.matrix_from_placement(start).tolist() == \
        P.matrix_from_placement(start).tolist()


def test_a_placement_that_has_not_moved_is_recognisably_unmoved():
    """Requirement 15's Keep gate, stated geometrically rather than as a latch.

    A one-way "has been edited" flag would survive a Reset and let the untouched
    seed placement be kept, which is what the gate exists to prevent.
    """
    start = P.placement_from_seed((273010.0, 5306990.0), SHAPE, QUAD_GSD)
    assert P.placement_delta(start, start)["translation_m"] == 0.0
    nudged = P.nudge(start, 1, 0, ORTHO_GSD)
    assert P.placement_delta(start, nudged)["translation_m"] > 0.5 * ORTHO_GSD
    assert P.placement_delta(start, start)["rotation_deg"] == 0.0


def test_the_delta_reports_translation_rotation_and_scale():
    a = _p(rot=10.0)
    b = P.set_rotation(P.translate(a, 0.10, 0.0), 12.0)
    d = P.placement_delta(a, b)
    assert d["translation_m"] == pytest.approx(0.10, abs=1e-9)
    assert d["rotation_deg"] == pytest.approx(2.0, abs=1e-9)
    assert d["scale_ratio"] == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------- #
#  Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _textured(h, w, seed=3):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(rng.random((h, w)), 2.0)
    for density, sigma, gain in ((0.010, 1.2, 3.0), (0.004, 2.4, 2.0)):
        blobs = (rng.random((h, w)) < density).astype(float)
        img = img + gaussian_filter(blobs, sigma) * gain
    img = img - img.min()
    return (255 * img / max(1e-9, img.max())).astype(np.uint8)


def test_the_blend_covers_the_inset_quadrilateral_and_nothing_else():
    """Three regions, no padding wedges anywhere.

    The shipped overlay rotated the raster with resize=True and cval=0, so 26%
    of what it drew was black corners; then it placed that by a bounding box.
    Warping by the placement's own affine has neither problem.
    """
    inset_px = 60
    quad = _textured(400, 400)
    p = P.Placement(273010.0, 5306990.0, 18.0, QUAD_GSD, (400, 400))
    win_shape = (700, 700)
    origin = (p.easting - 700 / 2 * ORTHO_GSD, p.northing + 700 / 2 * ORTHO_GSD)
    rgb, alpha = P.blend_layer(quad, p, win_shape, origin, ORTHO_GSD, inset_px)

    assert rgb.shape == win_shape and alpha.shape == win_shape
    assert set(np.unique(alpha)) <= {0.0, 1.0}

    # inside the inset quadrilateral -> covered
    M = P.matrix_from_placement(p)
    def to_win(col, row):
        wx, wy = M[:2, :2] @ np.array([col, row]) + M[:2, 2]
        return (int(round((wy - origin[1]) / -ORTHO_GSD)),
                int(round((wx - origin[0]) / ORTHO_GSD)))
    assert alpha[to_win(200, 200)] == 1.0                    # centre
    # the frame ring -> NOT covered, though it is inside the outer quadrilateral
    assert alpha[to_win(15, 200)] == 0.0
    assert alpha[to_win(200, 385)] == 0.0
    # far outside -> not covered
    assert alpha[5, 5] == 0.0
    # and the covered area is the inset square, not its bounding box
    side_m = (400 - 2 * inset_px) * QUAD_GSD
    expected_px = (side_m / ORTHO_GSD) ** 2
    assert alpha.sum() == pytest.approx(expected_px, rel=0.05)


def test_the_quadrat_is_downsampled_to_the_orthos_pixel_size_never_up():
    """Rendering a 1 mm/px quadrat against a 2.4 mm/px ortho at native
    resolution makes the seam obvious however good the alignment is."""
    quad = _textured(400, 400)
    p = P.Placement(273010.0, 5306990.0, 0.0, QUAD_GSD, (400, 400))
    prepared = P._prepare_quadrat(quad, p, ORTHO_GSD, 0)
    assert prepared.shape[0] < 400                      # downsampled
    assert prepared.shape[0] == pytest.approx(
        400 * QUAD_GSD / ORTHO_GSD, abs=1)


# --------------------------------------------------------------------------- #
#  The agreement score                                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scene():
    """An ortho window and a quadrat cut from it at a known placement."""
    import cv2
    ortho = _textured(900, 900, seed=11)
    p = P.Placement(273010.0, 5306990.0, 17.0, QUAD_GSD, (700, 700))
    origin = (p.easting - 450 * ORTHO_GSD, p.northing + 450 * ORTHO_GSD)
    # cut the quadrat OUT of the ortho at that placement, at the quadrat's scale
    M = P.matrix_from_placement(p)
    src = np.float32([[0, 0], [700, 0], [700, 700]])
    world = (M[:2, :2] @ src.T).T + M[:2, 2]
    dst = np.float32(np.column_stack([(world[:, 0] - origin[0]) / ORTHO_GSD,
                                      (origin[1] - world[:, 1]) / ORTHO_GSD]))
    A = cv2.getAffineTransform(dst, src)
    quad = cv2.warpAffine(ortho.astype(np.float32), A, (700, 700))
    return ortho, quad, p, origin


def test_the_score_is_high_where_the_imagery_agrees(scene):
    ortho, quad, p, origin = scene
    assert P.agreement_score(p, quad, ortho, origin, ORTHO_GSD) > 0.35


def test_the_score_collapses_a_few_centimetres_away(scene):
    ortho, quad, p, origin = scene
    for dx, dy in ((0.05, 0.0), (0.0, 0.05), (-0.05, 0.0)):
        moved = P.translate(p, dx, dy)
        assert P.agreement_score(moved, quad, ortho, origin, ORTHO_GSD) < 0.20


def test_the_score_collapses_under_a_couple_of_degrees(scene):
    """Half a degree displaces a corner by 6 mm while leaving the centre
    perfect, which is exactly what a centre-weighted blend hides."""
    ortho, quad, p, origin = scene
    turned = P.set_rotation(p, p.rotation_deg + 2.0)
    assert P.agreement_score(turned, quad, ortho, origin, ORTHO_GSD) < 0.20


def test_the_score_is_cheap_enough_to_show_live(scene):
    """Three scores per mouse move -- the placement and plus/minus
    5 cm -- must fit inside a frame."""
    import time
    ortho, quad, p, origin = scene
    P.agreement_score(p, quad, ortho, origin, ORTHO_GSD)      # warm
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        for d in (0.0, 0.05, -0.05):
            P.agreement_score(P.translate(p, d, 0.0), quad, ortho, origin,
                              ORTHO_GSD)
    ms = 1000 * (time.perf_counter() - t0) / reps
    assert ms < 200, f"three scores took {ms:.1f} ms"


def test_the_floor_admits_what_the_matcher_accepts(scene):
    """SCORE_FLOOR sits below the weakest placement the matcher itself accepted
    (0.27 on Bio_Station) and above a placement that is merely off the peak."""
    ortho, quad, p, origin = scene
    assert 0.0 < P.SCORE_FLOOR < 0.27
    assert P.agreement_score(p, quad, ortho, origin, ORTHO_GSD) > P.SCORE_FLOOR
    off = P.translate(p, 0.05, 0.05)
    assert P.agreement_score(off, quad, ortho, origin, ORTHO_GSD) < P.SCORE_FLOOR
