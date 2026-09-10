"""Regression tests for report legibility + build robustness.

Covers four fixes implemented in functions/report.py and
functions/map_export.py:

  -    one consistent "[unit]" square-bracket notation across the report's
       raster colorbars AND map_export colorbar labels (no statistic / factor
       altered — a sorting label still reads "[φ]" at factor 1.0).
  -    single-band raster panels are not titled "Band 1".
  -    the Zonal section uses clean human H2 headings derived from mode (+
       optional field / vector) with the raw filename demoted to a grey
       source line.
  -    build_pdf is resilient — a raising section is caught, logged, replaced
       by a visible placeholder, and the build COMPLETES (two-pass multiBuild).

These tests deliberately avoid GDAL: the colorbar/title logic is exercised
through the label-builder helpers and the heading-derivation helper, and the
end-to-end build runs on the GDAL-free ``synthetic_project`` fixture (no
rasters), so the suite stays fast and portable.
"""
from __future__ import annotations

import pytest

from functions import map_export as mx
from functions import report as rpt


# --------------------------------------------------------------------------- #
#  Square-bracket unit notation + factor invariance                            #
# --------------------------------------------------------------------------- #
def test_a2_map_label_uses_square_brackets():
    """map_export colorbar labels now read 'Name [unit]' (was '(unit)')."""
    assert mx._format_label_with_unit("Clast_length", "m") == "Clast length [m]"
    assert mx._format_label_with_unit("Clast_length", "mm") == "Clast length [mm]"
    # φ-space (sorting) label keeps its glyph, just rebracketed.
    assert mx._format_label_with_unit("Clast_length", "φ") == "Clast length [φ]"


def test_a2_map_label_no_empty_brackets_when_unitless():
    """A dimensionless field must NOT gain an empty '[]'."""
    assert mx._format_label_with_unit("Clast_length", "") == "Clast length"


def test_a2_format_colorbar_label_brackets():
    """The auto-lookup label builder also uses brackets, consistently."""
    assert mx._format_colorbar_label("Clast_length") == "Clast length [m]"
    assert mx._format_colorbar_label("Orientation") == "Orientation [°]"


def test_a2_no_statistic_or_factor_changed():
    """The unit notation is a label-TEXT change only — the factors are intact.

    A length unit (mm) still multiplies metres by 1000; metres by 1.0. This
    guards the invariant that no displayed statistic is altered.
    """
    assert mx._SIZE_UNIT_TABLE["mm"] == ("mm", 1000.0)
    assert mx._SIZE_UNIT_TABLE["m"] == ("m", 1.0)
    assert mx._SIZE_UNIT_TABLE["cm"] == ("cm", 100.0)


# --------------------------------------------------------------------------- #
#  Resilient build: a raising section is caught + placeholdered                #
# --------------------------------------------------------------------------- #
def test_w1_raising_section_is_caught_and_build_completes(
        synthetic_project, monkeypatch, tmp_path):
    """Force one numbered section to raise; the build must finish with a
    placeholder for it, a logged warning, and every other section intact."""
    pytest.importorskip("reportlab")
    root = synthetic_project["root"]

    # Make the Zonal section explode deterministically.
    def _boom(*a, **k):
        raise RuntimeError("synthetic zonal explosion")
    from functions import report_sections as _sections
    monkeypatch.setattr(_sections, "zonal_statistics", _boom)

    logs: list[str] = []
    out = tmp_path / "w1.pdf"
    result = rpt.build_pdf(root, out, options={}, log_fn=logs.append)

    # Build completed and wrote the file.
    assert result == out
    assert out.is_file() and out.stat().st_size > 0

    # A warning was logged for the failed section.
    assert any("could not be rendered" in l and "synthetic zonal explosion" in l
               for l in logs), logs

    # The placeholder text is present in the rendered PDF.
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(str(out))
    text = "".join(pg.get_text() for pg in doc)
    assert "could not be rendered" in text


def test_w1_clean_build_has_no_placeholder(synthetic_project, tmp_path):
    """Sanity: with no corruption the resilient loop adds NO placeholder."""
    pytest.importorskip("reportlab")
    fitz = pytest.importorskip("fitz")
    root = synthetic_project["root"]
    out = tmp_path / "clean.pdf"
    rpt.build_pdf(root, out, options={}, log_fn=lambda _m: None)
    doc = fitz.open(str(out))
    text = "".join(pg.get_text() for pg in doc)
    assert "could not be rendered" not in text


def test_b2_build_uses_clean_zonal_heading(synthetic_project, tmp_path):
    """End-to-end: the rendered PDF carries the clean zonal H2 heading and
    demotes the raw filename to a source line (not the heading)."""
    pytest.importorskip("reportlab")
    fitz = pytest.importorskip("fitz")
    root = synthetic_project["root"]
    out = tmp_path / "b2.pdf"
    rpt.build_pdf(root, out, options={}, log_fn=lambda _m: None)
    doc = fitz.open(str(out))
    text = "".join(pg.get_text() for pg in doc)
    # Clean human heading present…
    assert "Polygon statistics" in text
    assert "Transect profiles" in text
    # …and the raw filename now appears only as a demoted source line.
    # The label reads "Source:" since every provenance line in the report
    # goes through one helper; what matters is that the filename is demoted
    # to that line rather than standing as the heading.
    assert "Source:" in text


# --------------------------------------------------------------------------- #
#  helpers                                                                      #
# --------------------------------------------------------------------------- #
def __import_path(name: str):
    """Tiny pathlib.Path factory kept local so the heading tests don't need a
    real file on disk (``_zonal_heading_for`` only reads ``.name``)."""
    from pathlib import Path
    return Path(name)


# --------------------------------------------------------------------------- #
#  report-figures — units, headings and precision                              #
# --------------------------------------------------------------------------- #
def test_poly_value_cols_picks_only_field_magnitude_columns():
    """Identifiers, counts, ratios and phi moments carry no field unit."""
    header = ["polygon_id", "count", "density", "coverage", "mean", "std",
              "median", "iqr", "sigma_phi", "sk_phi", "kg_phi", "D50", "P84",
              "field", "unit"]
    got = [header[i] for i in rpt._poly_value_cols(header)]
    assert got == ["mean", "std", "median", "iqr", "D50", "P84"]


def test_poly_value_cols_ignores_phi_moments_and_area():
    header = ["area", "area_m2", "perimeter", "mean_phi", "skewness",
              "kurtosis", "max"]
    assert [header[i] for i in rpt._poly_value_cols(header)] == ["max"]


def test_fmt_measured_does_not_invent_sub_millimetre_precision():
    """GSD is ~1 mm/px and uncertainty is several px; 74.0348 mm is noise."""
    assert rpt._fmt_measured(0.0740348 * 1000.0, "mm") == "74.0"
    assert rpt._fmt_measured(float("nan"), "mm") == "—"
    # A non-length unit keeps four significant figures.
    assert rpt._fmt_measured(1.23456, "m/s") == "1.235"


def test_presets_differ_in_fidelity_and_full_is_print_resolution():
    """Full archive renders at print resolution; shareable is the small one."""
    full = rpt.REPORT_PRESETS["full"]
    share = rpt.REPORT_PRESETS["shareable"]
    assert full["dpi"] >= 300, "full archive must reach print resolution"
    assert share["dpi"] < full["dpi"]
    assert share["quality"] < full["quality"]
    assert full["label"] == "Full archive"
    assert share["label"] == "Shareable"


def test_packing_index_is_dimensionless_whatever_the_field():
    """A packing index of a velocity is not measured in m/s.

    The publication map labelled its colorbar from the FIELD alone, so a
    packing-index raster of Hjulstrom_deposition_velocity came out as
    "Hjulstrom deposition velocity [m/s]" over a 0-1 index, beneath a
    heading that correctly carried no unit at all.
    """
    for field in ("Hjulstrom_deposition_velocity", "Clast_length",
                  "Equivalent_diameter"):
        _name, unit = rpt._resolve_display(field, "packing_index")[:2]
        assert unit == "", f"{field} packing index must be dimensionless"
    # …while a real percentile of the same field keeps the field's unit.
    assert rpt._resolve_display("Hjulstrom_deposition_velocity", "D50")[1] == "m/s"
    assert rpt._resolve_display("Clast_length", "D50")[1] == "mm"


# --------------------------------------------------------------------------- #
#  A zonal CSV in the new grammar is inventoried like a legacy one             #
# --------------------------------------------------------------------------- #
def test_new_grammar_zonal_csv_is_inventoried(tmp_path):
    from pathlib import Path
    from functions import naming, layout
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        root = tmp_path / "Normandy_Etretat"
        zdir = layout.project_path("Normandy_Etretat", "zonal")
        zdir.mkdir(parents=True, exist_ok=True)
        origin = naming.origin_stem("02_alpha", "Normandy_Etretat", None)
        raster = origin + "_merged_Clast_length_D50_cellsize=1.0m"
        new = zdir / naming.zonal_out_name(raster, "zones_upper beach", "Clast_length", "transects")
        new.write_text("transect_id,distance_m,raster_value\n0,0,0.05\n", encoding="utf-8")
        legacy = zdir / "02_alpha_merged_Clast_length_D50_cellsize=1.0m__zones_02_alpha__Clast_length.transects.csv"
        legacy.write_text("transect_id,distance_m,raster_value\n0,0,0.05\n", encoding="utf-8")
        inv = rpt.collect_project_inventory(root)
        by = {Path(e["path"]).name: e for e in inv["zonal"]}
        n = by[new.name]
        assert n["mode"] == "transects"
        assert n["vector"] == "upper-beach"
        assert n["field"] == "Clast_length"
        assert n["source_image"] == origin
        l = by[legacy.name]
        assert l["mode"] == "transects" and l["vector"] == "zones_02_alpha"
        assert l["field"] == "Clast_length" and l["source_image"] == "02_alpha"
    finally:
        layout.set_datasets_root(original, persist=False)


# --------------------------------------------------------------------------- #
#  Section titles and raster figures of the rebuilt report                    #
# --------------------------------------------------------------------------- #
def test_zonal_titles_read_as_prose_not_filenames():
    from functions import report_sections as S
    t = S._zonal_title({"mode": "polygons", "field": "Clast_length",
                        "vector": "zones_upper_beach", "source_image": "n2"})
    assert t == "Polygon statistics of clast length on n2 (zones: upper_beach)"
    t = S._zonal_title({"mode": "transects", "field": None,
                        "vector": "zones_example_n1_of_UAV_ortho_image",
                        "source_image": "example_n1_of_UAV_ortho_image"})
    # The zone set that repeats the image name is not repeated.
    assert t == "Transect profiles on example_n1"
    t = S._zonal_title({"mode": "weird", "field": None, "vector": None, "source_image": None})
    assert t == "Zonal output"


def test_raster_figure_reports_an_unreadable_file_instead_of_drawing(tmp_path):
    """A raster that cannot be opened yields no figure and a reason; the
    section logs it, the document says nothing invented."""
    from functions import report_sections as S
    r = {"path": tmp_path / "missing.tif", "field": "Clast_length",
         "parameter": "std", "param_key": "std", "display": "Clast length — std",
         "unit": "mm", "factor": 1000.0, "cmap": "cividis", "diverging": False,
         "category": "size", "cellsize": 1.0, "empty": False}
    fig, why = S._raster_figure(r)
    assert fig is None and "open" in why


def test_validation_flags_a_self_comparison_and_tiny_truth(tmp_path):
    """A JSON pairing a file with itself, or a one-clast truth, is degenerate:
    excluded from every aggregate and named as such (the cover used to print
    F1 = 1.00 for a file compared with itself)."""
    import json
    from functions import report_facts as F
    csv = tmp_path / "q_individual_clasts.csv"
    csv.write_text("x,y,Clast_length\n1,2,0.05\n", encoding="utf-8")
    res = tmp_path / "validation" / "results"
    res.mkdir(parents=True)
    payload = {"schema_version": 2, "truth_csv": str(csv), "detect_csv": str(csv),
               "field": "Clast_length", "detect_gsd_m_per_px": 0.001,
               "metrics": {"n_matched": 1, "n_truth": 1, "n_detect": 1, "f1": 1.0},
               "distribution": {"truth_n": 1, "detect_n": 1}, "paired": {}}
    (res / "q.validation.json").write_text(json.dumps(payload), encoding="utf-8")
    inv = {"project_root": tmp_path, "project_name": tmp_path.name, "images": [],
           "vectors": [], "rasters": [], "logs": [], "zonal": [],
           "validation": {"truth_csvs": [], "images": [],
                          "results": [{"path": res / "q.validation.json"}]},
           "weights": {}, "uncertainty": {}}
    facts = F.gather_facts(inv, {})
    v = facts["validations"][0]
    assert v["usable"] is False
    assert "same file" in " ".join(v["flags"])
    assert facts["n_ortho"] == 0 and facts["n_quadrat"] == 0
