"""Smoke tests for the pure functions of the units, report and validation
helpers.

These exercise the functions that don't pull matplotlib / GDAL / ReportLab,
so they can run cleanly even on Windows installs where matplotlib's
freetype binding is broken.

Run from the repo root::

    python -m pytest tests/test_pure_helpers.py -v

The tests do NOT generate a PDF — that requires matplotlib. Use
``functions.report.build_pdf`` for full-stack verification on a healthy
matplotlib install.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this file directly (`python test_pure_helpers.py`) or via
# pytest from any cwd. Add the repo root to sys.path.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


# --------------------------------------------------------------------------- #
#  shared functions/units.py
# --------------------------------------------------------------------------- #
def test_d1_field_unit_table_returns_size_units_in_mm():
    from functions.units import field_unit_and_factor
    assert field_unit_and_factor("Clast_length")  == ("mm", 1000.0)
    assert field_unit_and_factor("Clast_width")   == ("mm", 1000.0)
    assert field_unit_and_factor("Surface_area")  == ("mm²", 1_000_000.0)
    # Longest-match wins over substring 'axis'.
    assert field_unit_and_factor("Ellipse_major_axis") == ("mm", 1000.0)


def test_d1_orientation_is_degrees_not_metres():
    from functions.units import field_unit_and_factor
    assert field_unit_and_factor("Orientation") == ("°", 1.0)


def test_d1_unknown_field_defaults_to_metres_not_dimensionless():
    from functions.units import field_unit_and_factor
    assert field_unit_and_factor("Something_unknown") == ("m", 1.0)


def test_d1_size_field_detection_excludes_orientation_and_area():
    from functions.units import is_size_field_for_phi
    assert is_size_field_for_phi("Clast_length") is True
    assert is_size_field_for_phi("Equivalent_diameter") is True
    assert is_size_field_for_phi("Orientation") is False
    # Surface_area excluded — φ is defined on linear sizes only.
    assert is_size_field_for_phi("Surface_area") is False
    assert is_size_field_for_phi("") is False
    assert is_size_field_for_phi(None) is False


# --------------------------------------------------------------------------- #
#  parameter overrides field unit for compound stats
# --------------------------------------------------------------------------- #
def test_a2_sorting_parameter_uses_phi_not_field_unit():
    """sorting of Clast_length must render in φ (factor 1), not mm (×1000)."""
    from functions.units import resolve_display
    disp, unit, factor, cmap, diverging = resolve_display(
        "Clast_length", "sorting")
    assert unit == "φ"
    assert factor == 1.0
    assert cmap == "cividis"
    assert diverging is False


def test_a2_skewness_parameter_is_diverging_centred_at_0():
    """skewness of Clast_length: signed φ, RdBu_r diverging cmap."""
    from functions.units import resolve_display
    disp, unit, factor, cmap, diverging = resolve_display(
        "Clast_length", "skewness")
    assert unit == "φ"
    assert factor == 1.0
    assert cmap == "RdBu_r"
    assert diverging is True


def test_a2_percentile_parameter_keeps_field_unit():
    """D50 of Clast_length stays in mm (the field's unit). Parameter
    just modifies the display string, not the unit."""
    from functions.units import resolve_display
    disp, unit, factor, cmap, _ = resolve_display("Clast_length", "D50")
    assert unit == "mm"
    assert factor == 1000.0
    assert "Clast length" in disp
    assert "median" in disp.lower() or "D50" in disp


def test_a2_density_parameter_overrides_field_unit():
    """density of Clast_length is clasts/m² (parameter wins entirely)."""
    from functions.units import resolve_display
    disp, unit, factor, _, _ = resolve_display("Clast_length", "density")
    assert unit == "clasts/m²"
    assert factor == 1.0


# --------------------------------------------------------------------------- #
#  raster filename parser handles digit-containing parameters
# --------------------------------------------------------------------------- #
def test_a1_parser_handles_D_percentile_parameters():
    """D5 / D16 / D50 / D84 / D95 must parse — they used to silently fail
    because the old regex's [A-Za-z_]+ excluded digits."""
    from functions.report import _parse_raster_meta_from_name
    for p in ("D5", "D16", "D50", "D84", "D95"):
        meta = _parse_raster_meta_from_name(
            f"site_individual_clast_values_Clast_length_{p}_cellsize=1.0m.tif"
        )
        assert meta["parameter"] == p, f"failed for {p!r}: {meta}"
        assert meta["cellsize"] == 1.0
        assert meta["field"] and "Clast_length" in meta["field"]


def test_a1_parser_handles_folk_ward_parameters():
    """folk_ward_sorting / _skewness / _kurtosis pre-empt the bare
    'sorting' / 'skewness' / 'kurtosis' suffixes."""
    from functions.report import _parse_raster_meta_from_name
    meta = _parse_raster_meta_from_name(
        "x_individual_clast_values_Clast_length_folk_ward_sorting_cellsize=1.0m.tif"
    )
    assert meta["parameter"] == "folk_ward_sorting"
    assert "Clast_length" in (meta["field"] or "")


def test_a1_parser_handles_density_with_no_field():
    from functions.report import _parse_raster_meta_from_name
    meta = _parse_raster_meta_from_name(
        "x_individual_clast_values_density_cellsize=1.0m.tif"
    )
    assert meta["parameter"] == "density"


# --------------------------------------------------------------------------- #
#  compute_distribution_stats: φ-block only for size fields
# --------------------------------------------------------------------------- #
def test_a8_phi_block_skipped_for_orientation():
    import numpy as np
    from functions.validation import compute_distribution_stats
    truth = np.array([5, 10, 90, 95, 175, 178], dtype=float)
    detect = np.array([7, 11, 85, 93, 170, 179], dtype=float)
    out = compute_distribution_stats(truth, detect, field="Orientation")
    assert out["phi_meaningful"] is False
    # φ-block fields populated as NaN, not garbage values.
    assert not np.isfinite(out["truth_mean_phi"])
    assert not np.isfinite(out["truth_sorting_phi"])
    # K-S and D-percentiles still meaningful in degrees.
    assert 0 <= out["ks_statistic"] <= 1
    assert 80 <= out["truth_d50"] <= 100  # near 90°


def test_a8_phi_block_runs_for_size_field():
    import numpy as np
    from functions.validation import compute_distribution_stats
    truth = np.array([0.030, 0.045, 0.060, 0.080, 0.100, 0.150], dtype=float)
    detect = np.array([0.032, 0.043, 0.062, 0.082, 0.098, 0.155], dtype=float)
    out = compute_distribution_stats(truth, detect, field="Clast_length")
    assert out["phi_meaningful"] is True
    # σ_φ is positive and finite for a real size distribution.
    assert out["truth_sorting_phi"] > 0


def test_a8_legacy_no_field_call_still_works():
    """Older callers passed no `field` and got φ statistics back. The
    keyword is opt-in, so the legacy call must keep working."""
    import numpy as np
    from functions.validation import compute_distribution_stats
    sizes = np.array([0.030, 0.045, 0.060, 0.080, 0.100], dtype=float)
    out = compute_distribution_stats(sizes, sizes)
    assert out["phi_meaningful"] is True
    assert np.isfinite(out["truth_mean_phi"])


# --------------------------------------------------------------------------- #
#  canonical-vectors helper de-duplicates merged + per-window CSVs
# --------------------------------------------------------------------------- #
def test_a3_canonical_vectors_prefer_merged_per_stem():
    from functions.report import _canonical_vectors
    inv = {
        "vectors": [
            {"image_stem": "site_a", "is_merged": True,  "window_size": None,
             "path": Path("a_merged.csv"), "n_rows": 100},
            {"image_stem": "site_a", "is_merged": False, "window_size": 1.0,
             "path": Path("a_1m.csv"), "n_rows": 70},
            {"image_stem": "site_a", "is_merged": False, "window_size": 2.5,
             "path": Path("a_2.5m.csv"), "n_rows": 50},
            # Site B has no merged file — all three per-window rows
            # remain canonical.
            {"image_stem": "site_b", "is_merged": False, "window_size": 1.0,
             "path": Path("b_1m.csv"), "n_rows": 80},
            {"image_stem": "site_b", "is_merged": False, "window_size": 2.5,
             "path": Path("b_2.5m.csv"), "n_rows": 40},
        ]
    }
    canonical = _canonical_vectors(inv)
    # site_a contributes one canonical entry (the merged one); site_b
    # contributes two.
    assert len(canonical) == 3
    stems = [c["image_stem"] for c in canonical]
    assert stems.count("site_a") == 1
    assert stems.count("site_b") == 2
    site_a = [c for c in canonical if c["image_stem"] == "site_a"][0]
    assert site_a["is_merged"] is True


# --------------------------------------------------------------------------- #
#  validation aggregator counts unique pairings, not JSONs
# --------------------------------------------------------------------------- #
def test_a4_aggregate_validation_dedups_by_pairing():
    """Same (truth, detect) pair validated against four fields counts as
    1 pairing for the overall block, 4 field-runs total."""
    import json
    import tempfile
    from functions.report import _aggregate_validation
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        results_dir = td / "validation" / "results"
        results_dir.mkdir(parents=True)
        payloads = [
            {"field": "Clast_length",  "metrics": {"recall": 0.5,
                                                    "precision": 0.8,
                                                    "f1": 0.62,
                                                    "rmse": 0.004}},
            {"field": "Clast_width",   "metrics": {"recall": 0.5,
                                                    "precision": 0.8,
                                                    "f1": 0.62,
                                                    "rmse": 0.006}},
            {"field": "Surface_area",  "metrics": {"recall": 0.5,
                                                    "precision": 0.8,
                                                    "f1": 0.62,
                                                    "rmse": 0.0003}},
            {"field": "Orientation",   "metrics": {"recall": 0.5,
                                                    "precision": 0.8,
                                                    "f1": 0.62,
                                                    "rmse": 18.0}},
        ]
        # All four share the same pairing key: same truth/detect/tolerance.
        for i, p in enumerate(payloads):
            p["schema_version"] = 1
            p["truth_csv"]   = "/path/truth.csv"
            p["detect_csv"]  = "/path/detect.csv"
            p["tolerance_m"] = 0.025
            p["truth_gsd_m_per_px"]  = 0.001
            p["detect_gsd_m_per_px"] = 0.001
            (results_dir / f"r{i}.validation.json").write_text(
                json.dumps(p), encoding="utf-8")
        inventory = {
            "validation": {
                "truth_csvs": [],
                "images":     [],
                "results":    [
                    {"path": p} for p in results_dir.iterdir()],
            },
            "vectors": [],
        }
        va = _aggregate_validation(inventory)
        assert va is not None
        # 4 JSONs → 1 unique pairing, 4 field-runs.
        assert va["n_comparisons"] == 1
        assert va["n_field_runs"]  == 4
        # Per-field bucket carries its own n_comparisons (1 each).
        for f in ("Clast_length", "Clast_width",
                  "Surface_area", "Orientation"):
            assert va["by_field"][f]["n_comparisons"] == 1


# --------------------------------------------------------------------------- #
#  Quick run shim so this file is also runnable with plain Python
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Minimal pytest-less runner: walk module-level tests and call them.
    failures = []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            failures.append((name, type(ex).__name__, str(ex)))
            print(f"FAIL {name}: {type(ex).__name__}: {ex}")
        else:
            print(f"PASS {name}")
    print()
    if failures:
        print(f"❌ {len(failures)} failure(s)")
        raise SystemExit(1)
    print("✅ all smoke tests passed")
