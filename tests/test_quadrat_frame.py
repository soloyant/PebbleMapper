"""functions/quadrat_frame.py: where the frame is in a rectified photograph.

A rectified photograph's sidecar records the frame's thickness; from it and
the GSD the inset in pixels follows, and the inner rectangle is the ROI every
detector should measure inside. A photograph without the record has no frame
to exclude.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from functions import quadrat_frame as QF


def _rectified(tmp_path, w=1516, h=1516, gsd=0.000554, thickness=0.02, name="IMG_1_rectified_GSD=0.000554m.jpg"):
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    p = tmp_path / name
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(p))
    side = {"PM_RECTIFIED": "yes", "PM_GSD": f"{gsd:.6f}", "PM_SIZE": f"{w}x{h}",
            "PM_SEGMENTS_M": "0.8400,0.8400,0.8400,0.8400"}
    if thickness is not None:
        side[QF.SIDECAR_KEY] = f"{thickness:.4f}"
    (tmp_path / (name + ".json")).write_text(json.dumps(side), encoding="utf-8")
    return p


def test_the_inset_follows_the_sidecar_and_the_gsd(tmp_path):
    p = _rectified(tmp_path)
    fi = QF.frame_inset(p)
    assert fi is not None
    assert fi.frame_px == 36                     # 0.02 m / 0.000554 m per px
    assert fi.margin_px == 4 and fi.inset_px == 40   # + 10 %
    assert fi.inner == (40, 40, 1476, 1476)
    w_m, h_m = fi.inner_size_m
    assert w_m == pytest.approx(0.80, abs=0.005) and h_m == pytest.approx(0.80, abs=0.005)
    assert fi.contains(700, 700) and not fi.contains(10, 700) and not fi.contains(700, 1500)
    assert "2.0 cm = 36 px + 4 px margin" in QF.describe(fi) and "0.80 x 0.80 m" in QF.describe(fi)


def test_no_record_means_no_frame(tmp_path):
    assert QF.frame_inset(_rectified(tmp_path, thickness=None, name="a.jpg")) is None
    assert QF.frame_inset(_rectified(tmp_path, thickness=0.0, name="b.jpg")) is None
    plain = tmp_path / "plain.jpg"
    cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))[1].tofile(str(plain))
    assert QF.frame_inset(plain) is None
    assert QF.describe(None) == ""


def test_an_absurd_thickness_is_refused(tmp_path):
    # A frame thicker than half the photograph leaves nothing to measure.
    assert QF.frame_inset(_rectified(tmp_path, w=100, h=100, thickness=0.03)) is None
    # The margin is 2 px at least.
    fi = QF.frame_inset(_rectified(tmp_path, w=100, h=100, gsd=0.002, thickness=0.01, name="m.jpg"))
    assert fi.frame_px == 5 and fi.margin_px == 2 and fi.inset_px == 7


def test_the_roi_is_the_inner_rectangle_in_pixels(tmp_path):
    p = _rectified(tmp_path)
    roi = QF.write_inner_roi(p, tmp_path / "roi")
    doc = json.loads(roi.read_text(encoding="utf-8"))
    ring = doc["features"][0]["geometry"]["coordinates"][0]
    assert ring == [[40, 40], [1476, 40], [1476, 1476], [40, 1476], [40, 40]]
    assert doc["features"][0]["properties"]["inset_px"] == 40
    assert doc["features"][0]["properties"]["frame_px"] == 36
    # And the app's own ROI loader accepts it.
    from functions.clasts_detection import _load_roi_paths
    paths = _load_roi_paths(str(roi))
    assert len(paths) == 1
    assert paths[0].contains_point((700, 700)) and not paths[0].contains_point((10, 700))
    assert QF.write_inner_roi(_rectified(tmp_path, thickness=None, name="c.jpg"), tmp_path) is None
