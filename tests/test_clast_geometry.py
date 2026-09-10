"""The orientation convention and the contour sidecar, pinned.

``Orientation`` is the bearing of a clast's long axis, clockwise from
image-up, axial on [0, 180): a long axis seen ``t`` degrees anticlockwise
from +x on screen has ``Orientation = 90 - t``. These tests build rotated
elliptical masks, measure them with the detector's own ``_measure_clast``
and check that ``functions.clast_geometry`` puts the major chord along the
true long axis in both frames a figure uses: image pixels (rows down) and
world coordinates through a north-up geotransform (y up). They also pin the
three consumers that used to mirror clasts across the diagonal: the merge
footprint, the report overlay and Digitize's rebuild of a saved truth CSV.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from functions import clast_geometry as CG

H, W = 160, 200
ANGLES = (0, 20, 45, 70, 90, 110, 135, 160)


def _axial_diff(a, b):
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


def ellipse_mask(t_deg, a=40.0, b=16.0, cx=100.0, cy=80.0, shape=(H, W)):
    """A filled ellipse whose long axis is ``t_deg`` anticlockwise from +x
    as seen on screen (rows count down)."""
    t = math.radians(t_deg)
    dx, dy = math.cos(t), -math.sin(t)          # screen angle in (col, row)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    u = (cc - cx) * dx + (rr - cy) * dy
    v = -(cc - cx) * dy + (rr - cy) * dx
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def _dilate(mask, r=2):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(mask, iterations=r)


def _inside(mask, col, row):
    c, r = int(round(col)), int(round(row))
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


@pytest.fixture(scope="module")
def measure():
    from functions.clasts_detection import _measure_clast
    return _measure_clast


# --------------------------------------------------------------------------- #
#  axis_direction                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("o, up, down", [
    (0.0, (0.0, 1.0), (0.0, -1.0)),                       # up-down
    (45.0, (math.sqrt(.5), math.sqrt(.5)), (math.sqrt(.5), -math.sqrt(.5))),
    (90.0, (1.0, 0.0), (1.0, 0.0)),                       # left-right
    (135.0, (math.sqrt(.5), -math.sqrt(.5)), (math.sqrt(.5), math.sqrt(.5))),
    (180.0, (0.0, 1.0), (0.0, -1.0)),                     # axial: 180 == 0
])
def test_axis_direction_is_a_bearing_clockwise_from_up(o, up, down):
    np.testing.assert_allclose(CG.axis_direction(o, y_down=False), up, atol=1e-12)
    np.testing.assert_allclose(CG.axis_direction(o, y_down=True), down, atol=1e-12)


def test_axis_angle_deg_matches_the_screen_angle():
    # A long axis at screen angle t: Orientation 90 - t; y up gives t back,
    # y down (cv2.ellipse, rows) gives -t.
    for t in ANGLES:
        o = (90.0 - t) % 180.0
        assert _axial_diff(CG.axis_angle_deg(o, y_down=False), t) < 1e-9
        assert _axial_diff(CG.axis_angle_deg(o, y_down=True), -t) < 1e-9


def test_chords_are_perpendicular_and_sized_by_length_and_width():
    (p0, p1), (q0, q1) = CG.axis_chords(10, 20, 8, 3, 30.0, y_down=False)
    maj = np.subtract(p1, p0)
    mnr = np.subtract(q1, q0)
    assert np.hypot(*maj) == pytest.approx(8)
    assert np.hypot(*mnr) == pytest.approx(3)
    assert abs(np.dot(maj, mnr)) < 1e-9
    np.testing.assert_allclose(np.add(p0, p1) / 2, (10, 20))


# --------------------------------------------------------------------------- #
#  The detector's measurement against the true long axis                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", ANGLES)
def test_measured_orientation_is_90_minus_the_screen_angle(measure, t):
    meas = measure(ellipse_mask(t), 1.0, 1.0)
    assert meas["Clast_length"] > meas["Clast_width"]
    assert _axial_diff(meas["Orientation"], 90.0 - t) < 2.0


@pytest.mark.parametrize("t", ANGLES)
def test_major_chord_follows_the_long_axis_in_pixels(measure, t):
    mask = ellipse_mask(t)
    meas = measure(mask, 1.0, 1.0)
    (p0, p1), (q0, q1) = CG.axis_chords(
        meas["center_x"], meas["center_y"], meas["Clast_length"],
        meas["Clast_width"], meas["Orientation"], y_down=True)
    ang = math.degrees(math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]))
    assert _axial_diff(ang, t) < 2.0
    grown = _dilate(mask)
    for pt in (p0, p1, q0, q1):
        assert _inside(grown, *pt), (t, pt)


@pytest.mark.parametrize("t", ANGLES)
def test_major_chord_follows_the_long_axis_in_world_coordinates(measure, t):
    g, x0, y0 = 0.004, 500000.0, 6000000.0
    to_world = CG.world_frame((x0, g, 0.0, y0, 0.0, -g))
    mask = ellipse_mask(t)
    meas = measure(mask, 1.0, g)
    wx, wy = to_world(meas["center_x"], meas["center_y"])
    (p0, p1), (q0, q1) = CG.axis_chords(
        float(wx), float(wy), meas["Clast_length"], meas["Clast_width"],
        meas["Orientation"], y_down=False)
    # The world vector is the screen vector (y up, north up).
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    assert _axial_diff(ang, t) < 2.0
    grown = _dilate(mask)
    for x, y in (p0, p1, q0, q1):
        assert _inside(grown, (x - x0) / g, (y0 - y) / g), (t, x, y)


@pytest.mark.parametrize("t", (20, 135))
def test_quadrat_csv_frame_draws_the_same_chord(measure, t):
    """x = col, y = H - row; drawn on rows through to_data(x, H - y)."""
    mask = ellipse_mask(t)
    meas = measure(mask, 1.0, 1.0)
    x, y = meas["center_x"], H - meas["center_y"]
    (p0, p1), _ = CG.axis_chords(x, H - y, meas["Clast_length"],
                                 meas["Clast_width"], meas["Orientation"],
                                 y_down=True)
    assert _inside(_dilate(mask), *p0) and _inside(_dilate(mask), *p1)


def test_ellipse_outline_lies_along_the_long_axis():
    pts = CG.ellipse_outline(0, 0, 10, 2, 90.0, y_down=False, n_points=8)
    assert pts[:, 0].max() == pytest.approx(5) and pts[:, 1].max() == pytest.approx(1)


# --------------------------------------------------------------------------- #
#  Contours                                                                    #
# --------------------------------------------------------------------------- #
def test_contour_from_measurement_hugs_the_mask_in_the_csv_frame(measure):
    mask = ellipse_mask(35)
    meas = measure(mask, 1.0, 1.0)
    outline = CG.contour_from_measurement(meas, to_frame=CG.quadrat_frame(H))
    arr = np.asarray(outline)
    assert 8 <= len(arr) < len(meas["_contour_x"])       # simplified
    from shapely.geometry import Polygon
    poly = Polygon(np.column_stack([arr[:, 0], H - arr[:, 1]]))
    assert poly.area == pytest.approx(mask.sum(), rel=0.06)
    # Two decimals in pixels.
    assert all(round(v, 2) == v for v in arr.ravel())


def test_sidecar_round_trip_and_stale_discard(tmp_path):
    csv = tmp_path / "site_ws1m.csv"
    csv.write_text("clast_ID,x\n1,0\n", encoding="utf-8")
    tri = [[0.0, 0.0], [1.5, 0.0], [0.0, 2.25]]
    out = CG.write_contours(csv, {1: tri, 7: tri, 9: [[0, 0], [1, 1]]},
                            frame="world")
    assert out == tmp_path / "site_ws1m.contours.json"
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["format"] == CG.CONTOURS_FORMAT and doc["frame"] == "world"
    assert doc["n"] == 2 and set(doc["contours"]) == {"1", "7"}
    back = CG.read_contours(csv)
    assert back.frame == "world" and set(back) == {1, 7}
    np.testing.assert_allclose(back[7], tri)
    assert CG.read_contours(tmp_path / "other.csv") is None
    import os
    os.utime(out, (1, 1))
    assert CG.discard_stale_contours(csv) is True
    assert CG.read_contours(csv) is None


def test_contours_for_frame_follows_clast_id_after_filtering():
    cs = {i: [[i, 0], [i + 1, 0], [i, 1]] for i in range(1, 6)}
    df = pd.DataFrame({"clast_ID": [1, 2, 3, 4, 5], "x": range(5)})
    sub = df[df["clast_ID"].isin([2, 5])]
    got = CG.contours_for_frame(sub, cs)
    assert set(got) == {0, 1}
    assert got[0][0, 0] == 2 and got[1][0, 0] == 5
    assert CG.contours_for_frame(sub.drop(columns=["clast_ID"]), cs) == {}


def test_contours_ride_on_a_dataframe_without_being_copied():
    df = pd.DataFrame({"clast_ID": [1, 2], "x": [0.0, 1.0]})
    CG.attach_contours(df, {1: [[0, 0], [1, 0], [0, 1]]}, frame="pixels")
    cs = CG.contours_of(df)
    kept = df[df["x"] > -1].copy()
    assert CG.contours_of(kept) is cs and cs.frame == "pixels"
    pd.concat([df, kept])                          # attrs compare cleanly


def test_draw_clasts_outlines_chords_and_note():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    df = pd.DataFrame({"clast_ID": [1, 2], "x": [10.0, 30.0], "y": [10.0, 30.0],
                       "Clast_length": [8.0, 6.0], "Clast_width": [4.0, 3.0],
                       "Orientation": [90.0, 0.0]})
    ax = Figure().add_subplot(111)
    got = CG.draw_clasts(ax, df, contours={1: [[6, 8], [14, 8], [10, 12]]},
                         y_down=False)
    assert got == {"n_outlines": 1, "n_chords": 2, "n_rows": 2}
    assert not ax.patches                           # no ellipse patch
    ax2 = Figure().add_subplot(111)
    CG.draw_clasts(ax2, df, contours=None, y_down=False)
    assert any(CG.NO_CONTOURS_NOTE in t.get_text() for t in ax2.texts)


# --------------------------------------------------------------------------- #
#  The consumers that used to mirror clasts across the diagonal                #
# --------------------------------------------------------------------------- #
def test_merge_footprint_is_oriented_by_the_bearing():
    from functions.clasts_merge import _ellipse_polygon
    poly = _ellipse_polygon(0.0, 0.0, 10.0, 2.0, 30.0, n_points=64)
    xs, ys = np.asarray(poly.exterior.coords).T
    k = int(np.argmax(np.hypot(xs, ys)))
    ang = math.degrees(math.atan2(xs[k], ys[k]))    # bearing from +y
    assert _axial_diff(ang, 30.0) < 3.0


def test_merge_iou_uses_correctly_oriented_footprints():
    """Two long clasts crossing at the same centroid: bearings 20 and 25
    overlap strongly; 20 and 70 barely. The old mirrored footprint rotated
    both by the same wrong amount, so the pair that must be deduplicated is
    built from clasts whose footprints were measured from real masks."""
    from functions.clasts_merge import dedup_clasts, _ellipse_polygon
    base = dict(Clast_width=0.02, Score=0.9)
    near = pd.DataFrame([dict(clast_ID=1, x=0.0, y=0.0, Clast_length=0.2,
                              Orientation=20.0, **base),
                         dict(clast_ID=2, x=0.03, y=0.08, Clast_length=0.2,
                              Orientation=20.0, **base)])
    # Displaced 0.085 along bearing 20 (dx = sin, dy = cos): the same stick,
    # slid along itself, overlaps; a mirrored footprint would lie along
    # bearing 70 and miss.
    a = _ellipse_polygon(0.0, 0.0, 0.2, 0.02, 20.0)
    b = _ellipse_polygon(0.03, 0.08, 0.2, 0.02, 20.0)
    iou = a.intersection(b).area / a.union(b).area
    assert iou > 0.3
    out = dedup_clasts(near, method="iou", overlap=0.3)
    assert len(out) == 1
    far = near.copy()
    far["x"] = [0.0, 0.08]
    far["y"] = [0.0, 0.03]            # slid along bearing 70 instead
    assert len(dedup_clasts(far, method="iou", overlap=0.3)) == 2


def test_report_overlay_chords_follow_the_long_axis(tmp_path, measure):
    """detection_overlay on a pixel photograph: the major chord it
    draws lies along the true long axis of the mask."""
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image
    from matplotlib.collections import LineCollection
    from functions import report_figures as RF
    t = 30
    mask = ellipse_mask(t)
    img = np.where(mask[..., None], 200, 40).astype(np.uint8).repeat(3, axis=2)
    path = tmp_path / "p.png"
    Image.fromarray(img).save(path)
    meas = measure(mask, 1.0, 0.001)
    df = pd.DataFrame([{"clast_ID": 1, "x": meas["center_x"],
                        "y": H - meas["center_y"],
                        "Clast_length": meas["Clast_length"],
                        "Clast_width": meas["Clast_width"],
                        "Orientation": meas["Orientation"]}])
    outline = CG.contour_from_measurement(meas, to_frame=CG.quadrat_frame(H))
    fig = RF.detection_overlay(path, df, gsd_m=0.001, window_m=None,
                               contours={1: outline})
    ax = fig.axes[0]
    assert fig.pm_meta["n_outlines"] == 1
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    # both chords are the same ink now; the major one is the wider line
    major = sorted(lines, key=lambda c: float(max(c.get_linewidths())))[-1:]
    (p0, p1) = major[0].get_segments()[0]
    ang = math.degrees(math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]))
    assert _axial_diff(ang, t) < 2.0
    assert _inside(_dilate(mask), *p0) and _inside(_dilate(mask), *p1)


def test_digitize_truth_round_trip_keeps_the_orientation(measure):
    """mask -> _measure_clast -> CSV row -> rebuilt ellipse -> re-measure."""
    from functions.digitize import ellipse_params_from_row, ellipse_to_mask
    res = 0.001
    for t in ANGLES:
        meas = measure(ellipse_mask(t, a=45, b=18), 1.0, res)
        row = {"x": meas["center_x"], "y": H - meas["center_y"],
               "Ellipse_major_axis": meas["Ellipse_major_axis"],
               "Ellipse_minor_axis": meas["Ellipse_minor_axis"],
               "Orientation": meas["Orientation"]}
        center, axes, angle = ellipse_params_from_row(row, H, res)
        rebuilt = ellipse_to_mask(center, axes, angle, (H, W)).astype(bool)
        again = measure(rebuilt, 1.0, res)
        assert _axial_diff(again["Orientation"], meas["Orientation"]) < 2.0, t
