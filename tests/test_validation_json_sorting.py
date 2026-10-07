"""schema_version<2 σφ standardisation in functions/report.py.

An older ``.validation.json`` (schema_version 1) stores ``sorting_phi`` as the
raw std of the φ-distribution, NOT the Folk-Ward graphic estimate. The raw φ
samples were never persisted, so :func:`functions.report._load_validation_json`
re-reads the source CSVs named in the payload and recomputes σφ with the SHARED
Folk-Ward helper (functions.units.folk_ward_sorting_phi, via
validation._folk_sorting_phi). This file pins:

  * the v1 σφ is replaced by the Folk-Ward value (≠ the v1 std(φ)), validated
    against a hand-derived figure to 1e-9;
  * a missing / unreadable source CSV falls back to the v1 value AND raises the
    caption flag ``_v1_sorting_unrecomputable``;
  * v2 payloads are left byte-identical (no recompute, no flags);
  * a non-size field is not "standardised" (φ-sorting is physically
    meaningless there) and is not flagged as a fallback.

GOLDEN POPULATION (shared with tests/test_folk_ward_golden.py)
==============================================================
PHI = [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.5, 5.0]  (n = 11)
D_metres = 1e-3 * 2**(-PHI) round-trips to PHI under φ = −log2(D/1e-3).

  Folk-Ward σφ = (φ84−φ16)/4 + (φ95−φ5)/6.6 = 1.1303030303030304   (hand-derived
                                                in the golden file docstring)
  v1 std(φ)    = np.std(PHI)                 = 1.2026142323020867   (≠ Folk-Ward)
"""
from __future__ import annotations

import json

import numpy as np
import pytest

# Repo root is placed on sys.path by tests/conftest.py.
from functions import report as R


PHI = np.array(
    [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=float
)
D_METRES = 1e-3 * 2.0 ** (-PHI)

# Hand-derived Folk-Ward σφ for this population (see golden file docstring).
REF_FOLK_WARD_SIGMA = 1.1303030303030304
# The (wrong) v1 estimator: raw std of the φ-distribution.
V1_STD_PHI = float(np.std(PHI))

TOL = 1e-9


def _write_csv(path, n_rows):
    """Write a minimal detection CSV holding ``Clast_length`` = D_METRES."""
    import pandas as pd

    pd.DataFrame(
        {
            "x": np.arange(n_rows, dtype=float),
            "y": np.zeros(n_rows),
            "Clast_length": D_METRES,
        }
    ).to_csv(path, index=False)


def _v1_payload(truth_csv, detect_csv, field="Clast_length"):
    """A schema_version=1 payload whose σφ is the stale std(φ) value."""
    return {
        "schema_version": 1,
        "field": field,
        "truth_csv": str(truth_csv),
        "detect_csv": str(detect_csv),
        "metrics": {"recall": 0.8, "precision": 0.9, "f1": 0.85},
        "distribution": {
            "truth_n": len(PHI),
            "detect_n": len(PHI),
            # The old std(φ) estimator — what the recompute must replace.
            "truth_sorting_phi": V1_STD_PHI,
            "detect_sorting_phi": V1_STD_PHI,
            "ks_p_value": 0.5,
        },
        "paired": {},
        "figures": {},
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Sanity: the two estimators really differ (guards a typo in the literals).   #
# --------------------------------------------------------------------------- #
def test_folk_ward_and_v1_estimators_differ():
    assert abs(REF_FOLK_WARD_SIGMA - V1_STD_PHI) > 0.05
    # And the Folk-Ward reference matches the live shared helper exactly.
    from functions import validation as V

    phi = V._phi(D_METRES)
    assert abs(V._folk_sorting_phi(phi) - REF_FOLK_WARD_SIGMA) < 1e-12


# --------------------------------------------------------------------------- #
#  Happy path: σφ recomputed to the Folk-Ward value from the source CSVs.      #
# --------------------------------------------------------------------------- #
def test_v1_sorting_recomputed_to_folk_ward(tmp_path):
    pytest.importorskip("pandas")
    truth = tmp_path / "truth.csv"
    detect = tmp_path / "detect.csv"
    _write_csv(truth, len(PHI))
    _write_csv(detect, len(PHI))

    jpath = tmp_path / "v1.validation.json"
    _write_json(jpath, _v1_payload(truth, detect))

    out = R._load_validation_json(jpath)
    assert out is not None
    assert out.get("_v1_sorting_recomputed") is True
    assert not out.get("_v1_sorting_unrecomputable")

    dist = out["distribution"]
    # Recomputed to the Folk-Ward value to 1e-9 (hand figure)…
    assert abs(dist["truth_sorting_phi"] - REF_FOLK_WARD_SIGMA) < TOL
    assert abs(dist["detect_sorting_phi"] - REF_FOLK_WARD_SIGMA) < TOL
    # …and demonstrably NOT the old std(φ) estimate.
    assert abs(dist["truth_sorting_phi"] - V1_STD_PHI) > 0.05
    assert abs(dist["detect_sorting_phi"] - V1_STD_PHI) > 0.05


# --------------------------------------------------------------------------- #
#  Fallback: a missing source CSV keeps the v1 value and flags the note.       #
# --------------------------------------------------------------------------- #
def test_v1_missing_csv_falls_back_and_flags(tmp_path):
    pytest.importorskip("pandas")
    # Only the truth CSV exists; detect CSV path points at a missing file.
    truth = tmp_path / "truth.csv"
    _write_csv(truth, len(PHI))
    missing_detect = tmp_path / "does_not_exist.csv"

    jpath = tmp_path / "v1_missing.validation.json"
    _write_json(jpath, _v1_payload(truth, missing_detect))

    out = R._load_validation_json(jpath)
    assert out is not None
    assert out.get("_v1_sorting_unrecomputable") is True
    assert not out.get("_v1_sorting_recomputed")

    # The stale v1 value is preserved untouched (no partial recompute).
    dist = out["distribution"]
    assert dist["truth_sorting_phi"] == V1_STD_PHI
    assert dist["detect_sorting_phi"] == V1_STD_PHI


def test_v1_unreadable_csv_falls_back_and_flags(tmp_path):
    pytest.importorskip("pandas")
    truth = tmp_path / "truth.csv"
    _write_csv(truth, len(PHI))
    # A CSV with no Clast_length column is "unreadable" for our purposes.
    bad = tmp_path / "no_field.csv"
    bad.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    jpath = tmp_path / "v1_badcol.validation.json"
    _write_json(jpath, _v1_payload(truth, bad))

    out = R._load_validation_json(jpath)
    assert out.get("_v1_sorting_unrecomputable") is True
    assert out["distribution"]["truth_sorting_phi"] == V1_STD_PHI


# --------------------------------------------------------------------------- #
#  No behaviour change for v2 files: untouched, no flags.                      #
# --------------------------------------------------------------------------- #
def test_v2_payload_untouched(tmp_path):
    pytest.importorskip("pandas")
    truth = tmp_path / "truth.csv"
    detect = tmp_path / "detect.csv"
    _write_csv(truth, len(PHI))
    _write_csv(detect, len(PHI))

    payload = _v1_payload(truth, detect)
    payload["schema_version"] = 2
    # A v2 file already carries the Folk-Ward value; pin a sentinel and
    # assert nothing rewrites it.
    payload["distribution"]["truth_sorting_phi"] = REF_FOLK_WARD_SIGMA
    payload["distribution"]["detect_sorting_phi"] = REF_FOLK_WARD_SIGMA

    jpath = tmp_path / "v2.validation.json"
    _write_json(jpath, payload)

    out = R._load_validation_json(jpath)
    assert out is not None
    assert not out.get("_v1_sorting_recomputed")
    assert not out.get("_v1_sorting_unrecomputable")
    assert out["distribution"]["truth_sorting_phi"] == REF_FOLK_WARD_SIGMA


# --------------------------------------------------------------------------- #
#  Non-size field: σφ is physically meaningless, so it is NOT standardised      #
#  and NOT flagged as a fallback.                                              #
# --------------------------------------------------------------------------- #
def test_v1_non_size_field_not_recomputed(tmp_path):
    pytest.importorskip("pandas")
    truth = tmp_path / "truth.csv"
    detect = tmp_path / "detect.csv"
    _write_csv(truth, len(PHI))
    _write_csv(detect, len(PHI))

    payload = _v1_payload(truth, detect, field="Orientation")
    jpath = tmp_path / "v1_orientation.validation.json"
    _write_json(jpath, payload)

    out = R._load_validation_json(jpath)
    assert out is not None
    # Not a size field → no Folk-Ward σφ to standardise, and not a
    # CSV-unavailable fallback either.
    assert not out.get("_v1_sorting_recomputed")
    assert not out.get("_v1_sorting_unrecomputable")
    # The stored value is left exactly as-is.
    assert out["distribution"]["truth_sorting_phi"] == V1_STD_PHI
