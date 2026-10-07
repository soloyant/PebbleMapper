"""Tests for functions/clasts_merge.py — dedup_clasts scenarios.

Covers:
  - Five identical ellipses → one survivor (IoU method)
  - Five non-overlapping ellipses → all five survive
  - Empty DataFrame → empty result
  - Single-row DataFrame → one row
  - priority_col: higher-priority row wins when two nearly-identical clasts conflict
  - centroid method: coincident duplicates collapsed
"""
import math
import numpy as np
import pandas as pd
import pytest

from functions.clasts_merge import dedup_clasts


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(clasts):
    """clasts = list of (cx, cy, length, width, orientation_deg, score)"""
    rows = [
        {
            "x": c[0],
            "y": c[1],
            "Clast_length": c[2],
            "Clast_width": c[3],
            "Orientation": c[4],
            "Score": c[5],
            "Ellipse_major_axis": c[2],
            "Ellipse_minor_axis": c[3],
            "Surface_area": c[2] * c[3] * math.pi / 4,
        }
        for c in clasts
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IoU method
# ---------------------------------------------------------------------------

def test_dedup_five_identical_ellipses():
    """Five clasts at exactly the same position → IoU=1 → only one survives."""
    df = _make_df([(0.0, 0.0, 0.1, 0.05, 0.0, 0.9)] * 5)
    result = dedup_clasts(df, method="iou", overlap=0.30)
    assert len(result) == 1


def test_dedup_five_non_overlapping_ellipses():
    """Five clasts spaced 1 m apart (length 0.05 m) → no pair overlaps → all survive."""
    df = _make_df([(float(i), 0.0, 0.05, 0.02, 0.0, 0.9) for i in range(5)])
    result = dedup_clasts(df, method="iou", overlap=0.30)
    assert len(result) == 5


def test_dedup_empty():
    """Empty input must return an empty DataFrame, not raise."""
    df = _make_df([])
    result = dedup_clasts(df, method="iou")
    assert len(result) == 0


def test_dedup_single_row():
    """Single clast — nothing to compare against; must survive."""
    df = _make_df([(0.0, 0.0, 0.1, 0.05, 0.0, 0.9)])
    result = dedup_clasts(df, method="iou")
    assert len(result) == 1


def test_priority_col_high_priority_wins():
    """Two nearly-identical clasts: the one with _priority=2 must survive."""
    df = _make_df(
        [
            (0.0,   0.0, 0.1, 0.05, 0.0, 0.5),   # low-priority
            (0.001, 0.0, 0.1, 0.05, 0.0, 0.8),   # high-priority
        ]
    )
    df["_priority"] = [1, 2]
    result = dedup_clasts(df, method="iou", overlap=0.30, priority_col="_priority")
    assert len(result) == 1
    assert result.iloc[0]["_priority"] == 2


def test_priority_col_absent_does_not_raise():
    """priority_col=None (default) must not raise even when Score varies."""
    df = _make_df(
        [
            (0.0, 0.0, 0.1, 0.05, 0.0, 0.3),
            (0.0, 0.0, 0.1, 0.05, 0.0, 0.9),   # same position — higher score wins
        ]
    )
    result = dedup_clasts(df, method="iou", overlap=0.30)
    assert len(result) == 1
    # The higher-score row should survive
    assert abs(result.iloc[0]["Score"] - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# Centroid method
# ---------------------------------------------------------------------------

def test_centroid_method_collapses_coincident():
    """Three clasts at the same position with centroid method → one survives."""
    df = _make_df([(0.0, 0.0, 0.1, 0.05, 0.0, 0.9)] * 3)
    result = dedup_clasts(df, method="centroid", overlap=0.5)
    assert len(result) == 1


def test_centroid_method_non_overlapping_survive():
    """Four clasts 1 m apart (length 0.05 m, overlap fraction 0.5) → all survive."""
    df = _make_df([(float(i), 0.0, 0.05, 0.02, 0.0, 0.9) for i in range(4)])
    result = dedup_clasts(df, method="centroid", overlap=0.5)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# return_kept_mask
# ---------------------------------------------------------------------------

def test_return_kept_mask_shape():
    """return_kept_mask=True must return (DataFrame, boolean array of same length)."""
    df = _make_df([(0.0, 0.0, 0.1, 0.05, 0.0, 0.9)] * 3)
    result, mask = dedup_clasts(df, method="iou", overlap=0.30, return_kept_mask=True)
    assert len(mask) == len(df)
    assert mask.sum() == len(result)


def test_equivalent_diameter_preserved_through_merge(tmp_path):
    """Regression: Equivalent_diameter survives the column reorder."""
    import pandas as pd
    from functions.clasts_merge import merge_csvs

    cols = ["clast_ID", "x", "y", "Ellipse_major_axis", "Ellipse_minor_axis",
            "Clast_length", "Clast_width", "Surface_area",
            "Equivalent_diameter", "Score", "Orientation"]

    df1 = pd.DataFrame([[1, 0.0, 0.0, 0.10, 0.05, 0.10, 0.05, 0.0039,
                         0.080, 0.95, 30.0]], columns=cols)
    df2 = pd.DataFrame([[1, 10.0, 10.0, 0.20, 0.10, 0.20, 0.10, 0.0157,
                         0.158, 0.92, 60.0]], columns=cols)
    p1, p2, pout = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out.csv"
    df1.to_csv(p1, index=False)
    df2.to_csv(p2, index=False)

    merge_csvs(str(p1), str(p2), str(pout))
    merged = pd.read_csv(pout)

    assert "Equivalent_diameter" in merged.columns, (
        "Equivalent_diameter dropped by clasts_merge._CLAST_COLUMNS reorder")
    assert merged["Equivalent_diameter"].notna().all()

    # Equivalent_diameter must sit in its canonical position
    # (between Surface_area and Score), mirroring clasts_detection._CLAST_COLUMNS.
    # Before the fix it was treated as an "extra" and shoved to the end of the
    # column list, which is what downstream rasters trip over.
    # Order follows the current canonical _CLAST_COLUMNS (filtered to the
    # columns present in the input); Equivalent_diameter stays between
    # Surface_area and Score rather than being shoved to the end.
    assert list(merged.columns) == [
        "clast_ID", "x", "y", "Clast_length", "Clast_width",
        "Ellipse_major_axis", "Ellipse_minor_axis", "Surface_area",
        "Equivalent_diameter", "Score", "Orientation",
    ], f"Column order drifted from canonical _CLAST_COLUMNS: {list(merged.columns)}"
