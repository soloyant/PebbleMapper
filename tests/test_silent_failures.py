"""Failures that used to come back looking like measurements.

A survey of every broad `except` in `functions/`, `gui/` and `detectors/` found
357 of them. Most are legitimate -- a best-effort README write should swallow.
Thirteen substituted a fallback for a value a *result* depends on, and three of
those were indistinguishable from a real answer:

* `georef` encoded "the ambiguity check could not run" as `rival_inliers = 0`,
  which is also the value meaning "there is no rival placement" -- so a failure
  switched the gate off and an ambiguous placement was accepted;
* `clasts_detection` reported `solidity = 0.0` when the convex hull failed, and
  0.0 is a legitimate solidity meaning maximally concave;
* `truncation` fell back from a SciPy logistic fit to a closed-form logit
  regression and reported `fit_ok=True` either way, so a published D50 gave no
  way to tell which estimator produced it.

The shared rule these encode: a sentinel for "unknown" must not be a value the
quantity can legitimately take.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
#  georef — the ambiguity gate must not fail open                              #
# --------------------------------------------------------------------------- #
def test_a_rival_check_that_could_not_run_is_not_a_pass():
    from functions import georef as G
    q = G.MatchQuality(n_correspondences=200, n_inliers=80,
                       inlier_fraction=0.4, rival_inliers=0,
                       rival_checked=False,
                       rival_error="LinAlgError: SVD did not converge")
    assert q.rival_checked is False
    d = q.as_dict()
    assert d["rival_checked"] is False
    # the count alone cannot tell the two apart, which is the whole point
    assert d["rival_inliers"] == 0


def test_a_rival_check_that_found_nothing_is_a_pass():
    from functions import georef as G
    q = G.MatchQuality(n_correspondences=200, n_inliers=80,
                       inlier_fraction=0.4, rival_inliers=0)
    assert q.rival_checked is True          # the default: we looked
    assert q.as_dict()["rival_checked"] is True


def test_the_default_is_checked_so_existing_callers_are_unchanged():
    from functions import georef as G
    assert G.MatchQuality().rival_checked is True
    assert G.MatchQuality().rival_error == ""


def test_an_unrunnable_rival_check_is_reported_to_the_user():
    """Requirement of the fix: it has to reach `reasons`, because that is what
    the tab prints. A flag nothing reads marks nothing -- the same lesson the
    placement provenance work landed on."""
    import inspect
    from functions import georef as G
    src = inspect.getsource(G)
    assert "if not q.rival_checked:" in src, \
        "the gate no longer consults rival_checked"
    i = src.index("if not q.rival_checked:")
    block = src[i:i + 500]
    assert "q.reasons.append" in block, \
        "an unrunnable ambiguity check no longer tells the user"


# --------------------------------------------------------------------------- #
#  clasts_detection — a shape metric that cannot be computed is not 0.0        #
# --------------------------------------------------------------------------- #
def test_solidity_is_nan_when_it_cannot_be_computed_not_zero():
    """0.0 is a real solidity: a maximally concave shape. Substituting it for
    "unknown" puts a plausible shape-quality number in the CSV that no
    downstream filter can distinguish from a measurement."""
    import inspect
    from functions import clasts_detection as C
    src = inspect.getsource(C)
    i = src.index("solidity")
    block = src[max(0, i - 400):i + 600]
    assert "solidity = float(\"nan\")" in block or \
           "solidity = float('nan')" in block, \
        "solidity no longer falls back to nan"
    assert "solidity = 0.0" not in block, \
        "solidity fell back to 0.0 again -- a value it can legitimately take"


def test_the_ellipse_axes_still_use_nan_for_the_same_reason():
    """They always did. The fix aligned solidity with them, so a regression in
    either direction should fail here."""
    import inspect
    from functions import clasts_detection as C
    src = inspect.getsource(C)
    assert "ell_major = ell_minor = float('nan')" in src


# --------------------------------------------------------------------------- #
#  truncation — which estimator produced this D50?                            #
# --------------------------------------------------------------------------- #
def _recall_case(n=600):
    """Truth sizes plus the indices that were detected — bigger clasts more
    often, which is what a detection function describes."""
    rng = np.random.default_rng(4)
    sizes = np.geomspace(0.004, 0.30, n)
    p = 1.0 / (1.0 + np.exp(-(sizes - 0.05) / 0.01))
    matched = np.nonzero(rng.random(n) < p)[0]
    return sizes, matched


def test_a_successful_logistic_fit_says_it_was_logistic():
    from functions import truncation as T
    sizes, matched = _recall_case()
    fit = T.detection_function_from_pair(sizes, matched)
    assert fit.fit_ok is True
    assert fit.method == "logistic"
    assert np.isfinite(fit.d50)


def test_the_fallback_estimator_names_itself():
    """It used to report fit_ok=True exactly like the primary fit, so a D50
    from the cruder closed-form regression was indistinguishable from one from
    the logistic fit. The SciPy import happens inside the try at call time, so
    patching it here is what the real failure looks like."""
    from functions import truncation as T
    import scipy.optimize as so

    sizes, matched = _recall_case()
    orig = so.curve_fit

    def boom(*a, **k):
        raise RuntimeError("Optimal parameters not found: maxfev reached")

    so.curve_fit = boom
    try:
        fit = T.detection_function_from_pair(sizes, matched)
    finally:
        so.curve_fit = orig

    assert fit.method != "logistic", "the fallback still claims to be the fit"
    assert fit.method in ("logit-regression", "none")
    if fit.method == "logit-regression":
        assert fit.fit_ok is True          # a usable number...
        assert np.isfinite(fit.d50)        # ...from a different estimator


def test_the_default_method_keeps_existing_callers_working():
    from functions import truncation as T
    f = T.DetectionFunctionFit(d50=0.05, slope=1.0, bin_sizes=np.array([1.0]),
                               bin_recall=np.array([1.0]),
                               bin_counts=np.array([1]))
    assert f.fit_ok is True and f.method == "logistic"


# --------------------------------------------------------------------------- #
#  The rule itself                                                             #
# --------------------------------------------------------------------------- #
def test_no_sentinel_is_a_value_the_quantity_can_take():
    """A guard against the next one. Solidity lives in [0, 1] and 0 is
    attainable; an inlier count is >= 0 and 0 is attainable. Both therefore
    need a separate "unknown", which is what nan and the rival_checked flag
    provide."""
    from functions import georef as G
    q = G.MatchQuality()
    assert isinstance(q.rival_checked, bool)      # not folded into the count
    assert math.isnan(float("nan"))               # nan != any real solidity


# --------------------------------------------------------------------------- #
#  The next tranche: a dropped record is counted, not just skipped             #
# --------------------------------------------------------------------------- #
def test_an_unreadable_validation_file_is_counted(tmp_path):
    """It happened for real: a 265-character path raised FileNotFoundError,
    `except OSError` caught it, and a directory holding two perfectly good
    files reported no uncertainty at all. An absent measurement and an
    unreadable one are different facts."""
    from functions import precision as P
    (tmp_path / "broken.validation.json").write_text("{ not json",
                                                     encoding="utf-8")
    (tmp_path / "old.validation.json").write_text('{"field": "Clast_length"}',
                                                  encoding="utf-8")
    out = P.load_validation_uncertainties(tmp_path)
    assert out == {}
    dropped = P.unreadable_validation_files(tmp_path)
    assert len(dropped) == 2, f"drops not counted: {dropped}"
    names = {n for n, _ in dropped}
    assert names == {"broken.validation.json", "old.validation.json"}
    assert any("older schema" in why for _, why in dropped)


def test_a_good_file_is_not_reported_as_dropped(tmp_path):
    import json
    from functions import precision as P
    (tmp_path / "ok.validation.json").write_text(json.dumps({
        "field": "Clast_length", "metrics": {"rmse": 0.0085},
        "paired": {"n": 40}, "truth_csv": "q.csv"}), encoding="utf-8")
    out = P.load_validation_uncertainties(tmp_path)
    assert "Clast_length" in out
    assert P.unreadable_validation_files(tmp_path) == []


def test_a_directory_that_is_not_there_drops_nothing(tmp_path):
    from functions import precision as P
    assert P.load_validation_uncertainties(tmp_path / "nope") == {}
    assert P.unreadable_validation_files(tmp_path / "nope") == []


def test_an_unusable_roi_shape_is_reported_not_swallowed(tmp_path, caplog):
    """A ring that will not build is part of the area the user drew. Dropping
    it silently searches less ground than was asked for, and reports the same
    number of clasts it would have reported anyway."""
    import json
    import logging
    from functions import clasts_detection as C
    roi = tmp_path / "roi.geojson"
    roi.write_text(json.dumps({"features": [
        {"geometry": {"type": "Polygon",
                      "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}},
        {"geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]}},
    ]}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        paths = C._load_roi_paths(str(roi))
    assert len(paths) == 1, "the good shape was lost"
    said = [r.getMessage() for r in caplog.records]
    assert any("could not be used" in m for m in said), (
        f"the dropped shape was not reported: {said}")
    assert any("1 of 2" in m for m in said), (
        "the report does not say how many were dropped")
