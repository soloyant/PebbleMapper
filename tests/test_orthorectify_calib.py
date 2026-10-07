"""Orthorectify lens-distortion: calibration parsing + undistort warp path."""
import json

import numpy as np
import pytest

pytest.importorskip("cv2")


def test_parse_calibration_json(tmp_path):
    from functions.orthorectify import parse_calibration_file
    f = tmp_path / "calib.json"
    f.write_text(json.dumps({
        "camera_matrix": [[1000.0, 0, 960.0], [0, 1001.0, 540.0], [0, 0, 1]],
        "dist_coeff": [0.1, -0.05, 0.001, 0.002, 0.0],
    }), encoding="utf-8")
    res = parse_calibration_file(str(f))
    assert res is not None
    fx, fy, cx, cy, dist = res
    assert abs(fx - 1000.0) < 1e-6 and abs(fy - 1001.0) < 1e-6
    assert abs(cx - 960.0) < 1e-6 and abs(cy - 540.0) < 1e-6
    assert len(dist) == 5 and abs(dist[0] - 0.1) < 1e-9


def test_parse_calibration_npz(tmp_path):
    from functions.orthorectify import parse_calibration_file
    f = tmp_path / "calib.npz"
    np.savez(str(f),
             camera_matrix=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                                    dtype=float),
             dist=np.array([0.2, 0.0, 0.0, 0.0, 0.0]))
    res = parse_calibration_file(str(f))
    assert res is not None
    fx, fy, cx, cy, dist = res
    assert abs(fx - 800.0) < 1e-6 and abs(cy - 360.0) < 1e-6


def test_parse_calibration_bad(tmp_path):
    from functions.orthorectify import parse_calibration_file
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    assert parse_calibration_file(str(tmp_path / "bad.json")) is None
    assert parse_calibration_file(str(tmp_path / "missing.json")) is None


def test_orthorectify_with_distortion_runs():
    """With distortion params, the engine undistorts + warps without error
    (zero coefficients ⇒ undistort is ~identity)."""
    from functions.orthorectify import orthorectify
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    corners = [(50, 50), (150, 50), (150, 150), (50, 150)]
    seg = [1.0, 1.0, 1.0, 1.0]
    K = [[300.0, 0, 100.0], [0, 300.0, 100.0], [0, 0, 1]]
    dist = [0.0, 0.0, 0.0, 0.0, 0.0]
    rect, gsd = orthorectify(img, corners, seg, gsd_m=0.01,
                             camera_matrix=K, dist_coeffs=dist)
    assert rect.ndim == 3 and gsd > 0
