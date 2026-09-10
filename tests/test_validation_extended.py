"""Extended tests for functions/validation.py.

Supplements the existing test_a8_* tests in test_pure_helpers.py with:
  - pair_csvs: perfect match, empty detect, tolerance mechanics
  - _folk_sorting_phi: alignment with rasterize formula (regression)
  - detection_metrics: all-matched, none-matched
"""
import math
import numpy as np
import pandas as pd
import pytest

from functions.validation import (
    pair_csvs,
    detection_metrics,
    _folk_sorting_phi,
    _phi,
)


# ---------------------------------------------------------------------------
# pair_csvs
# ---------------------------------------------------------------------------

def _make_xy(coords):
    return pd.DataFrame(coords, columns=["x", "y"])


def test_pair_csvs_perfect_match():
    truth = _make_xy([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    detect = _make_xy([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    result = pair_csvs(truth, detect, tolerance=0.1)
    assert len(result["matched_pairs"]) == 3
    assert result["unmatched_truth"] == []
    assert result["unmatched_detect"] == []


def test_pair_csvs_empty_detect():
    truth = _make_xy([(1.0, 1.0), (2.0, 2.0)])
    detect = _make_xy([])
    result = pair_csvs(truth, detect)
    assert len(result["matched_pairs"]) == 0
    assert len(result["unmatched_truth"]) == 2
    assert len(result["unmatched_detect"]) == 0


def test_pair_csvs_empty_truth():
    truth = _make_xy([])
    detect = _make_xy([(1.0, 1.0)])
    result = pair_csvs(truth, detect)
    assert len(result["matched_pairs"]) == 0
    assert len(result["unmatched_detect"]) == 1


def test_pair_csvs_tolerance_rejects_distant():
    """Two points 1 m apart should not match with tolerance=0.1."""
    truth = _make_xy([(0.0, 0.0)])
    detect = _make_xy([(1.0, 0.0)])
    result = pair_csvs(truth, detect, tolerance=0.1)
    assert len(result["matched_pairs"]) == 0
    assert len(result["unmatched_truth"]) == 1
    assert len(result["unmatched_detect"]) == 1


def test_pair_csvs_tolerance_accepts_nearby():
    """Two points 0.01 m apart should match with tolerance=0.1."""
    truth = _make_xy([(0.0, 0.0)])
    detect = _make_xy([(0.01, 0.0)])
    result = pair_csvs(truth, detect, tolerance=0.1)
    assert len(result["matched_pairs"]) == 1


def test_pair_csvs_no_double_assignment():
    """One detection cannot match two truths (mutual best wins)."""
    truth = _make_xy([(0.0, 0.0), (0.0, 0.001)])
    detect = _make_xy([(0.0, 0.0)])
    result = pair_csvs(truth, detect, tolerance=0.5)
    # Only one match possible — one detect, two truths competing for it.
    assert len(result["matched_pairs"]) == 1
    assert len(result["unmatched_truth"]) == 1


# ---------------------------------------------------------------------------
# _folk_sorting_phi: regression against the rasterize formula
# ---------------------------------------------------------------------------

def test_folk_sorting_phi_matches_rasterize_formula():
    """Validate sorting must equal rasterize formula for same data."""
    sizes_m = np.array([0.030, 0.045, 0.060, 0.080, 0.100, 0.150, 0.200])
    phi = _phi(sizes_m)
    val_sort = _folk_sorting_phi(phi)
    p5, p16, p84, p95 = np.percentile(phi, [5, 16, 84, 95])
    rast_sort = (p84 - p16) / 4.0 + (p95 - p5) / 6.6
    assert abs(val_sort - rast_sort) < 1e-12


# ---------------------------------------------------------------------------
# detection_metrics
# ---------------------------------------------------------------------------

def test_detection_metrics_all_matched():
    result = {
        "matched_pairs": [(0, 0), (1, 1)],
        "unmatched_truth": [],
        "unmatched_detect": [],
    }
    m = detection_metrics(result)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert abs(m["f1"] - 1.0) < 1e-9


def test_detection_metrics_none_matched():
    result = {
        "matched_pairs": [],
        "unmatched_truth": [0, 1],
        "unmatched_detect": [0],
    }
    m = detection_metrics(result)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert math.isnan(m["f1"])


def test_detection_metrics_partial():
    result = {
        "matched_pairs": [(0, 0)],
        "unmatched_truth": [1],
        "unmatched_detect": [1, 2],
    }
    m = detection_metrics(result)
    # recall = 1/(1+1) = 0.5
    assert abs(m["recall"] - 0.5) < 1e-9
    # precision = 1/(1+2) ≈ 0.333
    assert abs(m["precision"] - 1 / 3) < 1e-9
    assert math.isfinite(m["f1"])
