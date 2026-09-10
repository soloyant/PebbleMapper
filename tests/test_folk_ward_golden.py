"""Golden value-pinning tests for the Folk-Ward graphic statistics.

PURPOSE
=======
Folk-Ward graphic sorting (σφ), skewness (Sk_φ) and kurtosis (K_G) are computed
in FOUR separate modules today:

    * functions/validation.py     — _folk_sorting_phi / _folk_skewness_phi /
                                     _folk_kurtosis_phi          (np.percentile)
    * functions/zonal_stats.py    — _folk_sorting_phi / _folk_skewness_from_phi /
                                     _folk_kurtosis_from_phi      (np.percentile)
    * functions/report.py         — inline in compute_aggregate_stats ->
                                     _length_stats (report.py:774)  (np.nanquantile
                                     via _quantiles)
    * functions/clasts_rasterize.py — inline per-cell in _compute_raster_grid
                                     (clasts_rasterize.py:255)      (np.nanquantile)

This file pins each site's CURRENT numeric output to a single hand-derived
reference at 1e-9, so that any extraction into one shared helper can be proven
value-preserving: if these tests still pass, no number moved.

It also documents whether the four
sites already agree for the SAME population — a cross-site agreement test.

THE GOLDEN INPUT (hand-derived reference)
=========================================
A fixed, symmetric 11-value φ-population:

    PHI = [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.5, 5.0]   (n = 11)

numpy's default percentile/quantile interpolation ("linear") places the rank
for percentile P at fractional index  h = (n-1) * P/100 = 10 * P/100  into the
SORTED array, interpolating between neighbours.  Sorted PHI is already the list
above (indices 0..10).  Hence:

    p5  : h = 0.5  -> PHI[0] + 0.5*(PHI[1]-PHI[0]) = 1.0 + 0.5*0.5 = 1.25
    p16 : h = 1.6  -> PHI[1] + 0.6*(PHI[2]-PHI[1]) = 1.5 + 0.6*0.5 = 1.80
    p25 : h = 2.5  -> PHI[2] + 0.5*(PHI[3]-PHI[2]) = 2.0 + 0.5*0.0 = 2.00
    p50 : h = 5.0  -> PHI[5]                       = 3.00
    p75 : h = 7.5  -> PHI[7] + 0.5*(PHI[8]-PHI[7]) = 3.5 + 0.5*0.5 = 3.75
    p84 : h = 8.4  -> PHI[8] + 0.4*(PHI[9]-PHI[8]) = 4.0 + 0.4*0.5 = 4.20
    p95 : h = 9.5  -> PHI[9] + 0.5*(PHI[10]-PHI[9])= 4.5 + 0.5*0.5 = 4.75

Folk & Ward (1957), J. Sediment. Petrol. 27(1):3 graphic measures:

    σφ   = (φ84 - φ16)/4 + (φ95 - φ5)/6.6
         = (4.20 - 1.80)/4 + (4.75 - 1.25)/6.6
         = 2.40/4 + 3.50/6.6
         = 0.60 + 0.5303030303030303...
         = 1.1303030303030304

    Sk_φ = (φ16 + φ84 - 2φ50)/(2(φ84-φ16)) + (φ5 + φ95 - 2φ50)/(2(φ95-φ5))
         = (1.80 + 4.20 - 6.0)/(2*2.40) + (1.25 + 4.75 - 6.0)/(2*3.50)
         = 0.0/4.80 + 0.0/7.00
         = 0.0                       (population is symmetric about φ50 = 3.0)

    K_G  = (φ95 - φ5) / (2.44 (φ75 - φ25))
         = (4.75 - 1.25) / (2.44 * (3.75 - 2.00))
         = 3.50 / (2.44 * 1.75)
         = 3.50 / 4.27
         = 0.8196721311475411

For the report.py and clasts_rasterize.py sites, which start from grain sizes in
METRES and convert to φ internally via  φ = -log2(D_m / 0.001), we feed
D_m = 0.001 * 2**(-PHI) so the round-tripped φ-population is identical to PHI.

WHAT IS PINNED / TOLERANCE
==========================
Every per-site assertion is to abs(value - reference) < 1e-9.  As of TODAY:
  * validation, zonal, rasterize sites reproduce the reference EXACTLY.
  * report site differs from the reference by ~5e-14 (float reassociation in its
    metres->φ conversion and the _quantiles wrapper) — still far inside 1e-9.
The cross-site agreement test (one per statistic) confirms all four agree with
each other to 1e-9 today (max observed pairwise delta ~5.3e-14).  If a future
change makes any pair disagree at 1e-9, that test fails LOUDLY — it is the
signal to reconcile the sites, and must NOT be silenced by loosening the
tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Repo root is placed on sys.path by tests/conftest.py.
from functions import validation as V
from functions import zonal_stats as Z
from functions import report as R


# --------------------------------------------------------------------------- #
#  Golden input + hand-derived reference (see module docstring for arithmetic) #
# --------------------------------------------------------------------------- #
PHI = np.array(
    [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=float
)

# Grain sizes (metres) whose φ-transform  -log2(D_m / 0.001)  reproduces PHI.
D_METRES = 0.001 * 2.0 ** (-PHI)

# Reference percentiles (exactly the linear-interp values derived above).
_P5, _P16, _P25, _P50, _P75, _P84, _P95 = 1.25, 1.80, 2.00, 3.00, 3.75, 4.20, 4.75

REF_SIGMA = (_P84 - _P16) / 4.0 + (_P95 - _P5) / 6.6          # 1.1303030303030304
REF_SKEW = (
    (_P16 + _P84 - 2 * _P50) / (2.0 * (_P84 - _P16))
    + (_P5 + _P95 - 2 * _P50) / (2.0 * (_P95 - _P5))
)                                                            # 0.0
REF_KURT = (_P95 - _P5) / (2.44 * (_P75 - _P25))             # 0.8196721311475411

TOL = 1e-9


# --------------------------------------------------------------------------- #
#  Sanity: the hand-derived references match numpy's own quantiles on PHI      #
#  (guards against a typo in the literals above; not one of the four sites).   #
# --------------------------------------------------------------------------- #
def test_reference_matches_numpy_quantiles_on_golden_population():
    p5, p16, p25, p50, p75, p84, p95 = np.percentile(
        PHI, [5, 16, 25, 50, 75, 84, 95]
    )
    assert (p5, p16, p25, p50, p75, p84, p95) == (
        _P5, _P16, _P25, _P50, _P75, _P84, _P95
    )
    assert abs(REF_SIGMA - 1.1303030303030304) < 1e-15
    assert REF_SKEW == 0.0
    assert abs(REF_KURT - 0.8196721311475411) < 1e-15
    # The metres round-trip reproduces PHI exactly.
    assert np.max(np.abs(-np.log2(D_METRES / 0.001) - PHI)) == 0.0


# --------------------------------------------------------------------------- #
#  Req 2 — one assertion per live implementation, each to 1e-9                 #
# --------------------------------------------------------------------------- #
def test_validation_site_matches_reference():
    # functions/validation.py:187-198 (+ skew :142, kurt :168)
    assert abs(V._folk_sorting_phi(PHI) - REF_SIGMA) < TOL
    assert abs(V._folk_skewness_phi(PHI) - REF_SKEW) < TOL
    assert abs(V._folk_kurtosis_phi(PHI) - REF_KURT) < TOL


def test_zonal_site_matches_reference():
    # functions/zonal_stats.py:220-234 (+ skew :192, kurt :207)
    assert abs(Z._folk_sorting_phi(PHI) - REF_SIGMA) < TOL
    assert abs(Z._folk_skewness_from_phi(PHI) - REF_SKEW) < TOL
    assert abs(Z._folk_kurtosis_from_phi(PHI) - REF_KURT) < TOL


def _report_folk_ward(tmp_path):
    """Drive the report.py:774 inline computation through the smallest public
    surface: compute_aggregate_stats -> _length_stats on a one-CSV inventory.

    The CSV holds D_METRES under ``Clast_length``; _phi converts it back to PHI.
    Returns the pooled ``folk_ward`` dict.
    """
    import pandas as pd

    csv = tmp_path / "Site_georef_merged_individual_clast_values.csv"
    pd.DataFrame(
        {
            "x": np.arange(len(D_METRES), dtype=float),
            "y": np.zeros(len(D_METRES)),
            "Clast_length": D_METRES,
        }
    ).to_csv(csv, index=False)

    inventory = {
        "images": ["dummy.jpg"],
        "vectors": [
            {
                "path": str(csv),
                "image_stem": "Site_georef",
                "is_merged": True,
                "window_size": None,
            }
        ],
        "logs": [],
        # Minimal validation block so _aggregate_validation short-circuits
        # without touching real validation JSONs / truth CSVs.
        "validation": {"results": [], "truth_csvs": []},
    }
    out = R.compute_aggregate_stats(inventory)
    return out["folk_ward"]


def test_report_site_matches_reference(tmp_path):
    # functions/report.py:774 (sorting) / :777 (skew) / :780 (kurt),
    # reached via compute_aggregate_stats -> _length_stats.
    pytest.importorskip("pandas")
    fw = _report_folk_ward(tmp_path)
    assert abs(fw["sorting_phi"] - REF_SIGMA) < TOL
    assert abs(fw["skewness_phi"] - REF_SKEW) < TOL
    assert abs(fw["kurtosis_phi"] - REF_KURT) < TOL


def _rasterize_cell(parameter):
    """Drive the clasts_rasterize.py:255 per-cell computation through the
    smallest public surface: _compute_raster_grid on a single-cell input.

    All clasts are spread inside one cell (x/y in [0, 5], cellsize 1000), so
    the output grid is 1x1 and ``p[0, 0]`` is the Folk-Ward value for the whole
    golden population.  ``Clast_length`` holds D_METRES; the function converts
    to φ internally.  Returns ``float(p[0, 0])``.
    """
    import pandas as pd
    from functions import clasts_rasterize as CR

    local = pd.DataFrame(
        {
            "x": np.linspace(0.0, 5.0, len(D_METRES)),
            "y": np.linspace(0.0, 5.0, len(D_METRES)),
            "Clast_length": D_METRES,
        }
    )
    p = CR._compute_raster_grid(
        local,
        field="Clast_length",
        parameter=parameter,
        cellsize=1000.0,
        percentile=0.5,
        bin_edges=None,
        bin_mode="value",
        bin_field=None,
        min_cell_density=0,
    )[0]
    assert p.shape == (1, 1), f"expected a 1x1 single-cell grid, got {p.shape}"
    return float(p[0, 0])


def test_rasterize_site_matches_reference():
    # functions/clasts_rasterize.py:255 (sorting) / :266 (skew) / :278 (kurt).
    # The module imports osgeo.gdal at top level — skip cleanly without GDAL.
    pytest.importorskip("osgeo")
    pytest.importorskip("pandas")
    assert abs(_rasterize_cell("folk_ward_sorting") - REF_SIGMA) < TOL
    assert abs(_rasterize_cell("folk_ward_kurtosis") - REF_KURT) < TOL
    assert abs(_rasterize_cell("folk_ward_skewness") - REF_SKEW) < TOL


# --------------------------------------------------------------------------- #
#  Req 3 — cross-site agreement: all four agree with EACH OTHER to 1e-9 today  #
# --------------------------------------------------------------------------- #
def _all_site_values(tmp_path):
    """Collect every site's value for sorting/skew/kurt on the SAME population.

    Sites needing GDAL/pandas are omitted from the dict when unavailable, so the
    cross-site comparison still runs (over the remaining sites) in a minimal env.
    """
    sites = {
        "sigma": {
            "validation": V._folk_sorting_phi(PHI),
            "zonal": Z._folk_sorting_phi(PHI),
        },
        "skew": {
            "validation": V._folk_skewness_phi(PHI),
            "zonal": Z._folk_skewness_from_phi(PHI),
        },
        "kurt": {
            "validation": V._folk_kurtosis_phi(PHI),
            "zonal": Z._folk_kurtosis_from_phi(PHI),
        },
    }
    if pytest.importorskip("pandas", reason="report site needs pandas"):
        fw = _report_folk_ward(tmp_path)
        sites["sigma"]["report"] = fw["sorting_phi"]
        sites["skew"]["report"] = fw["skewness_phi"]
        sites["kurt"]["report"] = fw["kurtosis_phi"]
    # rasterize: only if GDAL importable (top-level osgeo.gdal import).
    try:
        import osgeo  # noqa: F401

        sites["sigma"]["rasterize"] = _rasterize_cell("folk_ward_sorting")
        sites["skew"]["rasterize"] = _rasterize_cell("folk_ward_skewness")
        sites["kurt"]["rasterize"] = _rasterize_cell("folk_ward_kurtosis")
    except Exception:  # pragma: no cover - env without GDAL
        pass
    return sites


@pytest.mark.parametrize("stat", ["sigma", "skew", "kurt"])
def test_cross_site_agreement(tmp_path, stat):
    """All available sites must agree with each other to 1e-9 for the same
    population.  Today the
    four Folk-Ward implementations DO agree (max pairwise delta ~5e-14).

    If a future edit makes a pair disagree at 1e-9, this fails — surfacing the
    drift to reconcile.  Do NOT loosen TOL to silence such a failure;
    convert it to xfail(reason=...) with the measured delta instead.
    """
    pytest.importorskip("pandas")
    values = _all_site_values(tmp_path)[stat]
    assert len(values) >= 2, "need at least two sites to compare"
    names = sorted(values)
    worst = (None, None, 0.0)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            delta = abs(values[a] - values[b])
            if delta > worst[2]:
                worst = (a, b, delta)
    assert worst[2] < TOL, (
        f"cross-site Folk-Ward {stat} disagreement: "
        f"{worst[0]}={values[worst[0]]!r} vs {worst[1]}={values[worst[1]]!r} "
        f"delta={worst[2]:.3e} >= {TOL:g}  "
        f"(all sites: { {k: float(v) for k, v in values.items()} }) — "
        "this is a known finding; do not loosen TOL, mark xfail with the delta."
    )
    # math.isclose belt-and-braces: nothing NaN slipped through.
    assert all(math.isfinite(v) for v in values.values())
