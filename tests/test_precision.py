"""Displayed precision derived from measured uncertainty.

The anchor case is real: this project's own
validation reports RMSE = 0.008521 m for Clast_length, so 0.0740348 m must
render as "74" mm — not "74.0348", which claims four digits the method cannot
support.
"""
from __future__ import annotations

import json
import math

import pytest

from functions import precision as P


# --------------------------------------------------------------------------- #
#  The rounding rule                                                           #
# --------------------------------------------------------------------------- #
def test_the_anchor_case_from_the_spec():
    """0.0740348 m ± 0.008521 m, in millimetres, is "74"."""
    value_mm = 0.0740348 * 1000.0
    u_mm = 0.008521108392415516 * 1000.0        # 8.52 mm -> 1 s.f. -> 9 mm
    assert P.decimals_for(u_mm, "mm") == 0
    assert P.format_value(value_mm, u_mm, "mm") == "74"


def test_precision_tracks_the_uncertainty():
    # 9 mm -> units place; 0.9 mm -> one decimal; 0.09 mm -> two.
    assert P.decimals_for(8.5, "mm") == 0
    assert P.decimals_for(0.85, "mm") == 1
    assert P.decimals_for(0.085, "mm") == 2
    # An uncertainty of tens of mm still shows whole millimetres, never fewer
    # digits than the units place — "70" would imply a precision claim of its
    # own that the rule does not make.
    assert P.decimals_for(85.0, "mm") == 0
    assert P.format_value(74.0348, 85.0, "mm") == "74"


def test_unknown_uncertainty_uses_the_stated_fallback():
    assert P.decimals_for(None, "mm") == P.DEFAULT_DISPLAY_DECIMALS["mm"]
    assert P.format_value(74.0348, None, "mm") == "74.0"
    # A unit with no default keeps four significant-ish digits.
    assert P.decimals_for(None, "m/s") == 4


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "x", None])
def test_degenerate_uncertainty_falls_back_rather_than_dividing_by_it(bad):
    assert P.decimals_for(bad, "mm") == P.DEFAULT_DISPLAY_DECIMALS["mm"]


def test_a_value_smaller_than_its_uncertainty_still_rounds_honestly():
    """Hiding a 0 behind extra digits would misrepresent how well it is known."""
    assert P.format_value(2.0, 9.0, "mm") == "2"
    assert P.format_value(0.4, 9.0, "mm") == "0"


def test_non_finite_values_render_as_a_dash():
    assert P.format_value(float("nan"), 8.5, "mm") == "—"
    assert P.format_value(float("inf"), 8.5, "mm") == "—"


# --------------------------------------------------------------------------- #
#  Reading the validation results                                              #
# --------------------------------------------------------------------------- #
def _write_validation(dirpath, field, rmse, n, quadrat, when="2026-01-01T00:00:00"):
    dirpath.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "generated_at": when,
        "truth_csv": quadrat,
        "field": field,
        "metrics": {"rmse": rmse, "bias": 0.001, "r2": 0.8},
        "paired": {"n": n, "rmse": rmse},
    }
    p = dirpath / f"{quadrat}.validation.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_reads_field_and_rmse_from_a_validation_json(tmp_path):
    res = tmp_path / "validation" / "results"
    _write_validation(res, "Clast_length", 0.008521108392415516, 17, "q1")
    u = P.uncertainty_for("Clast_length", res)
    assert u is not None
    assert u.value == pytest.approx(0.008521108392415516)
    assert u.n == 17
    assert u.measured is True


def test_mission_pooling_lies_between_its_quadrats(tmp_path):
    """A pooled mission uncertainty must sit inside its inputs' range."""
    res = tmp_path / "results"
    _write_validation(res, "Clast_length", 0.004, 10, "q1")
    _write_validation(res, "Clast_length", 0.012, 30, "q2")
    u = P.uncertainty_for("Clast_length", res)
    assert 0.004 <= u.value <= 0.012
    assert u.sources == 2
    assert u.n == 40
    assert u.scope == "mission"
    # Weighted towards the larger sample, so above the unweighted mean.
    assert u.value > math.sqrt((0.004 ** 2 + 0.012 ** 2) / 2) - 1e-9


def test_reruns_of_one_quadrat_do_not_double_its_weight(tmp_path):
    res = tmp_path / "results"
    _write_validation(res, "Clast_length", 0.004, 10, "q1", "2026-01-01T00:00:00")
    _write_validation(res, "Clast_length", 0.009, 10, "q1", "2026-06-01T00:00:00")
    u = P.uncertainty_for("Clast_length", res)
    assert u.sources == 1
    assert u.value == pytest.approx(0.009), "the most recent run should win"


def test_quadrats_with_too_few_pairs_are_excluded(tmp_path):
    res = tmp_path / "results"
    _write_validation(res, "Clast_length", 0.004, 1, "q1")
    assert P.uncertainty_for("Clast_length", res) is None


def test_a_corrupt_or_foreign_validation_file_is_skipped_not_fatal(tmp_path):
    res = tmp_path / "results"
    res.mkdir(parents=True)
    (res / "broken.validation.json").write_text("{not json", encoding="utf-8")
    (res / "empty.validation.json").write_text("{}", encoding="utf-8")
    (res / "nrmse.validation.json").write_text(
        json.dumps({"field": "Clast_length", "metrics": {"rmse": None}}),
        encoding="utf-8")
    _write_validation(res, "Clast_length", 0.005, 20, "good")
    u = P.uncertainty_for("Clast_length", res)
    assert u is not None and u.sources == 1


def test_absent_directory_is_unknown_not_an_error(tmp_path):
    assert P.uncertainty_for("Clast_length", tmp_path / "nope") is None
    assert P.load_validation_uncertainties(tmp_path / "nope") == {}


def test_a_field_without_validation_is_unknown(tmp_path):
    res = tmp_path / "results"
    _write_validation(res, "Clast_length", 0.005, 20, "q1")
    assert P.uncertainty_for("Orientation", res) is None


# --------------------------------------------------------------------------- #
#  Saying what the precision is                                                #
# --------------------------------------------------------------------------- #
def test_describe_states_the_uncertainty_scope_and_sample_size(tmp_path):
    res = tmp_path / "results"
    _write_validation(res, "Clast_length", 0.008521, 17, "q1")
    _write_validation(res, "Clast_length", 0.008, 23, "q2")
    u = P.uncertainty_for("Clast_length", res)
    text = P.describe(u, 1000.0, "mm")
    assert "±" in text and "mm" in text
    assert "40" in text, "must state the pooled sample size"
    assert "quadrat" in text.lower()


def test_describe_marks_the_last_resort_as_a_convention_not_a_measurement():
    """Superseded in part by the GSD work: with neither a validation result nor
    a ground sample distance there is nothing measured to report, and the
    document must not let a default read as one."""
    text = P.describe(None, 1000.0, "mm")
    assert "Neither a validation result" in text
    assert "convention, not a measured precision" in text


# --------------------------------------------------------------------------- #
#  Against the project's real file                                             #
# --------------------------------------------------------------------------- #
def test_against_the_real_swimbeach_validation_file():
    """The anchor case, on the actual validation file."""
    from pathlib import Path
    # Walk up rather than assuming a depth: a git worktree puts the tests four
    # levels below the checkout that actually holds datasets/.
    rel = Path("datasets") / "Swimbeach" / "2023-08-23" / "validation" / "results"
    res = None
    for parent in Path(__file__).resolve().parents:
        cand = parent / rel
        if cand.is_dir():
            res = cand
            break
    if res is None:
        pytest.skip("Swimbeach validation results not present in this checkout")
    by_field = P.load_validation_uncertainties(res)
    assert "Clast_length" in by_field, sorted(by_field)
    u = P.uncertainty_for("Clast_length", res)
    assert u is not None and u.measured
    # Millimetre-scale, as the method's ground sample distance implies.
    assert 0.0005 < u.value < 0.05, u.value
    assert P.decimals_for(u.in_display(1000.0), "mm") <= 1


# --------------------------------------------------------------------------- #
#  Uncertainty is GSD-dependent, not a constant                                #
# --------------------------------------------------------------------------- #
def test_precision_is_never_finer_than_one_pixel():
    """A regression can report an RMSE finer than the ground sample distance.
    Stating a length to less than a pixel claims a precision the imagery
    cannot carry, whatever the statistics say."""
    u = P.Uncertainty(field="Clast_length", value=0.0003, n=50, sources=2,
                      gsd_m=0.005)                      # 0.3 mm RMSE, 5 mm/px
    assert u.effective() == pytest.approx(0.005)
    assert P.decimals_for(u.effective() * 1000.0, "mm") == 0


def test_the_floor_does_not_shrink_a_larger_measured_error():
    u = P.Uncertainty(field="Clast_length", value=0.0085, n=17, gsd_m=0.005)
    assert u.effective() == pytest.approx(0.0085)


def test_a_caller_may_supply_the_gsd_of_the_data_being_shown():
    """The floor belongs to the imagery being displayed, which need not be the
    imagery the uncertainty was measured on."""
    u = P.Uncertainty(field="Clast_length", value=0.002, n=30, gsd_m=0.001)
    assert u.effective(gsd_m=0.010) == pytest.approx(0.010)


def test_without_validation_the_fallback_is_the_gsd_not_a_constant():
    """The whole objection to a fixed default: uncertainty scales with pixel
    size, so a constant would be wrong at any other resolution."""
    coarse = P.from_gsd(0.010, "Clast_length")          # 10 mm/px
    mid = P.from_gsd(0.001, "Clast_length")             # 1 mm/px
    fine = P.from_gsd(0.0001, "Clast_length")           # 0.1 mm/px
    assert coarse.measured is False and coarse.scope == "gsd"
    # A millimetre pixel supports whole millimetres, not tenths.
    assert P.decimals_for(coarse.in_display(1000.0), "mm") == 0
    assert P.decimals_for(mid.in_display(1000.0), "mm") == 0
    assert P.decimals_for(fine.in_display(1000.0), "mm") == 1
    # Same value, different imagery, different honest precision.
    assert P.format_value(74.0348, coarse.in_display(1000.0), "mm") == "74"
    assert P.format_value(74.0348, fine.in_display(1000.0), "mm") == "74.0"


def test_from_gsd_rejects_a_degenerate_gsd():
    for bad in (0.0, -1.0, float("nan"), None, "x"):
        assert P.from_gsd(bad, "Clast_length") is None


def test_describe_distinguishes_measured_gsd_and_last_resort():
    measured = P.Uncertainty(field="Clast_length", value=0.0085, n=29,
                             sources=2, gsd_m=0.005)
    assert "RMSE" in P.describe(measured, 1000.0, "mm")

    gsd_only = P.from_gsd(0.005, "Clast_length")
    text = P.describe(gsd_only, 1000.0, "mm")
    assert "ground sample distance" in text
    assert "not the detector" in text, "must not be read as a validated error"

    nothing = P.describe(None, 1000.0, "mm")
    assert "convention, not a measured precision" in nothing


def test_an_uncertainty_records_the_imagery_it_was_measured_on():
    """So that applying a mission's error to different imagery is detectable."""
    u = P.Uncertainty(field="Clast_length", value=0.0085, gsd_m=0.005)
    assert u.gsd_m == 0.005


# --------------------------------------------------------------------------- #
#  Wiring into the report                                                      #
# --------------------------------------------------------------------------- #
def test_report_resolves_uncertainty_from_validation_and_rounds_with_it(tmp_path):
    """The report's table formatter must use the mission's measured error.

    Note the example project cannot demonstrate this end to end: its polygon
    CSVs predate the `field` column, so no field can be resolved for them and
    the uncertainty has nothing to attach to. This exercises the same code path
    with a field present.
    """
    from functions import report as rpt

    res = tmp_path / "validation" / "results"
    _write_validation(res, "Clast_length", 0.008521, 17, "q1")
    _write_validation(res, "Clast_length", 0.011861, 12, "q2")

    inv = {"validation": {"results": [{"path": p} for p in res.iterdir()]},
           "images": []}
    uncerts = rpt._resolve_uncertainties(inv)
    assert "Clast_length" in uncerts

    u_disp, u = rpt._uncertainty_display(uncerts, "Clast_length", 1000.0)
    assert u is not None and u.measured
    # Pooled from the two real Swimbeach quadrat errors -> ~10 mm.
    assert 8.5 <= u_disp <= 11.9
    # …so the value that started this renders as whole millimetres.
    assert rpt._fmt_measured(0.0740348 * 1000.0, "mm", u_display=u_disp) == "74"


def test_report_falls_back_to_the_pixel_when_a_field_has_no_validation(tmp_path):
    """A field with no validation still gets a resolution-based precision."""
    from functions import report as rpt

    uncerts = {"__gsd_m__": 0.005}          # 5 mm/px, no validation at all
    u_disp, u = rpt._uncertainty_display(uncerts, "Clast_length", 1000.0)
    assert u is not None and u.measured is False
    assert u_disp == pytest.approx(5.0)
    assert rpt._fmt_measured(74.0348, "mm", u_display=u_disp) == "74"


def test_report_uncertainty_survives_a_project_with_nothing_to_go_on():
    from functions import report as rpt
    u_disp, u = rpt._uncertainty_display({}, "Clast_length", 1000.0)
    assert u_disp is None and u is None
    # …and the formatter keeps its old convention rather than crashing.
    assert rpt._fmt_measured(74.0348, "mm") == "74.0"


def test_the_pixel_fallback_applies_only_to_lengths():
    """A ground sample distance is metres per pixel. It bounds a length; it
    says nothing about a velocity, a shear stress or a dimensionless index,
    and using it there would be dimensionally meaningless."""
    from functions import report as rpt

    uncerts = {"__gsd_m__": 0.005}
    length_u, _ = rpt._uncertainty_display(uncerts, "Clast_length", 1000.0)
    assert length_u == pytest.approx(5.0), "a length is bounded by the pixel"

    for other in ("Hjulstrom_deposition_velocity", "Clast_circularity",
                  "Orientation", "Score"):
        u_disp, u = rpt._uncertainty_display(uncerts, other, 1.0)
        assert u_disp is None and u is None, other


def test_a_measured_error_is_not_floored_by_a_pixel_for_a_non_length():
    """The floor is a length too, so it must not touch a velocity's RMSE."""
    from functions import precision as P
    from functions import report as rpt

    uncerts = {"__gsd_m__": 5.0,        # absurdly large, to catch misuse
               "Hjulstrom_deposition_velocity": P.Uncertainty(
                   field="Hjulstrom_deposition_velocity", value=0.02, n=30)}
    u_disp, u = rpt._uncertainty_display(
        uncerts, "Hjulstrom_deposition_velocity", 1.0)
    assert u_disp == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
#  GUI readouts                                                                #
# --------------------------------------------------------------------------- #
def test_gui_formatter_rounds_to_the_uncertainty_when_given_one():
    """Requirement 12: a readout must not claim more digits than the
    comparison's own RMSE supports."""
    from functions.units import format_value_with_unit

    # 0.0740348 m of Clast_length, against this comparison's 8.5 mm RMSE.
    assert format_value_with_unit(0.0740348, "Clast_length", 8.521) == "74 mm"
    # Without an uncertainty the old per-unit convention is unchanged.
    assert format_value_with_unit(0.0740348, "Clast_length") == "74.03 mm"


def test_gui_formatter_still_handles_missing_values():
    from functions.units import format_value_with_unit
    assert format_value_with_unit(None, "Clast_length", 8.5) == "—"
    assert format_value_with_unit(float("nan"), "Clast_length", 8.5) == "—"


def test_gui_formatter_respects_the_fields_own_unit():
    """A velocity is not rounded by a millimetre-scale number."""
    from functions.units import format_value_with_unit
    out = format_value_with_unit(1.81234, "Hjulstrom_deposition_velocity", 0.02)
    assert out.endswith("m/s")
    assert "1.81" in out
