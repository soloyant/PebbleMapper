"""Per-clast measurement schema: the new shape fields + dropped ellipse axes."""
import math

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("shapely")


def test_clast_columns_schema():
    from functions.clasts_detection import _CLAST_COLUMNS
    for col in ("Perimeter", "Eccentricity", "Solidity", "Mean_intensity"):
        assert col in _CLAST_COLUMNS, f"{col} should be in the per-clast schema"
    # Ellipse axes are kept (second-moment a/b) for cross-method comparability.
    for col in ("Ellipse_major_axis", "Ellipse_minor_axis", "Orientation",
                "Clast_length", "Clast_width"):
        assert col in _CLAST_COLUMNS


def test_measure_clast_outputs_new_shape_fields():
    from functions.clasts_detection import _measure_clast
    # A filled disk (radius 20 px) in an 80×80 frame.
    yy, xx = np.ogrid[:80, :80]
    mask = ((xx - 40) ** 2 + (yy - 40) ** 2) <= 20 ** 2
    meas = _measure_clast(mask, score=0.9, resolution=0.01)
    assert meas is not None
    for k in ("Clast_length", "Clast_width", "Surface_area", "Perimeter",
              "Equivalent_diameter", "Eccentricity", "Solidity", "Orientation",
              "Ellipse_major_axis", "Ellipse_minor_axis"):
        assert k in meas
    # Second-moment axes of a disk ≈ its diameter (40 px × 0.01 m/px ≈ 0.4 m).
    assert 0.3 < meas["Ellipse_major_axis"] < 0.5
    # A disk is convex (solidity ≈ 1) and round (eccentricity ≈ 0).
    assert 0.9 <= meas["Solidity"] <= 1.0001
    assert 0.0 <= meas["Eccentricity"] < 0.4
    # Perimeter of an r=20 px circle at 0.01 m/px ≈ 2πr·res ≈ 1.26 m.
    assert 1.0 < meas["Perimeter"] < 1.5


def test_mask_mean_intensity():
    from functions.clasts_detection import _mask_mean_intensity
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:5, 2:5] = 200
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert abs(_mask_mean_intensity(img, mask) - 200.0) < 1e-6
    assert math.isnan(_mask_mean_intensity(img, np.zeros((10, 10), dtype=bool)))
