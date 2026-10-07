"""Regression tests for functions.digitize."""
import numpy as np
import pytest
from functions.digitize import digitize_records_to_dataframe


def _square_mask(size=32):
    m = np.zeros((size, size), dtype=bool)
    m[8:24, 8:24] = True
    return m


def test_equivalent_diameter_present_in_output():
    """Regression: Equivalent_diameter must appear with finite values."""
    records = [{"mask": _square_mask(), "score": 0.9, "label": ""}]
    df = digitize_records_to_dataframe(records, image_height=32, resolution=0.001)
    assert "Equivalent_diameter" in df.columns, (
        "Equivalent_diameter dropped from digitized records — regression of digitize")
    assert np.isfinite(df["Equivalent_diameter"]).all()
    assert (df["Equivalent_diameter"] > 0).all()


# --- Cropped masks: thousands of detections on a full-size photograph ------ #
from functions import digitize as D

_SHAPE = (2142, 2856)   # an iPhone HEIC, 5.8 MiB per full-frame bool mask


def _shapes():
    verts = [[100, 120], [190, 110], [230, 200], [140, 260], [90, 190]]
    return [
        (D.polygon_to_mask(verts, _SHAPE), D.polygon_to_cropped_mask(verts, _SHAPE)),
        (D.circle_to_mask((2850, 5), 30, _SHAPE),          # clipped by two edges
         D.circle_to_cropped_mask((2850, 5), 30, _SHAPE)),
        (D.ellipse_to_mask((1400, 1000), (60, 25), 37.0, _SHAPE),
         D.ellipse_to_cropped_mask((1400, 1000), (60, 25), 37.0, _SHAPE)),
    ]


@pytest.mark.parametrize("i", range(3))
def test_a_cropped_mask_is_the_full_mask(i):
    full, crop = _shapes()[i]
    assert isinstance(crop, D.CroppedMask) and crop.shape == _SHAPE
    np.testing.assert_array_equal(np.asarray(crop), full.astype(bool))
    assert crop.sum() == D.mask_area(crop) == int(full.sum())
    ys, xs = np.nonzero(full)
    assert D.mask_centroid(crop) == pytest.approx((xs.mean(), ys.mean()))
    assert crop[int(ys[0]), int(xs[0])] is True
    assert crop[0, 0] is bool(full[0, 0]) and crop[-1 + _SHAPE[0], 1] is False
    assert crop.crop.nbytes < full.nbytes / 100
    np.testing.assert_array_equal(D.CroppedMask.from_full(full).crop.sum(), full.sum())


@pytest.mark.parametrize("i", range(3))
def test_a_cropped_record_measures_like_the_full_one(i):
    full, crop = _shapes()[i]
    a = D.measure_record_px({"mask": full.astype(bool), "score": 1.0})
    b = D.measure_record_px({"mask": crop, "score": 1.0})
    if a is None:
        # A clast cut by the image edge is skipped either way.
        assert b is None
        return
    for k in ("center_x", "center_y", "Clast_length", "Clast_width",
              "Surface_area", "Perimeter", "Orientation"):
        # The ellipse fit's round-off depends on the window's offset
        # (measured: 0.04 px on a 128 px clast).
        assert b[k] == pytest.approx(a[k], rel=1e-3, abs=1e-2), k
    np.testing.assert_allclose(b["_contour_x"], a["_contour_x"])
    np.testing.assert_allclose(b["_contour_y"], a["_contour_y"])


def test_three_thousand_cropped_detections_fit_in_memory():
    import tracemalloc
    tracemalloc.start()
    recs = [{"mask": D.circle_to_cropped_mask((50 + (k % 90) * 30, 50 + (k // 90) * 30),
                                             12, _SHAPE), "score": 1.0}
            for k in range(3200)]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 60 * 2**20, peak      # full frames would be 18.6 GiB
    df = D.digitize_records_to_dataframe(recs, _SHAPE[0], 1.0)
    assert len(df) == 3200
