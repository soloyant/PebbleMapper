"""Tests for the profile figure and its tables.

The figure replaces a composer whose legend was six near-identical
110-character filenames, whose axis named a quantity the data was not, and
whose two rendering paths disagreed about units for the same numbers. These
tests pin the properties that made it unusable.
"""
import csv
import math

import pytest

from functions import zonal_stats as zs


def _write_transect(path, *, transect="T1", n=13, base=0.050, step=0.25,
                    elev0=None, field="Clast_length"):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["transect_id", "distance_m", "raster_value"]
    if elev0 is not None:
        header.append("elevation_m")
    if field is not None:
        header.append("field")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            row = [transect, i * step, base + i * 0.0012]
            if elev0 is not None:
                row.append(elev0 - i * 0.05)
            if field is not None:
                row.append(field)
            w.writerow(row)
    return path


def _dated(tmp_path, date, **kw):
    return _write_transect(tmp_path / date / "t.csv", **kw)


def _render_and_capture(series, out, monkeypatch, **kw):
    """Render, keeping the Figure alive so its axes can be inspected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close

    def spy(fig=None, *a, **k):
        if fig is not None and hasattr(fig, "axes"):
            captured["fig"] = fig
            return
        return real_close(fig, *a, **k) if fig is not None else None

    monkeypatch.setattr(plt, "close", spy)
    zs.plot_profile_figure(series, out, **kw)
    monkeypatch.undo()
    return captured["fig"], real_close


# --------------------------------------------------------------------------
# Labels: dates, never filenames
# --------------------------------------------------------------------------

def test_the_profile_figure_never_overwrites_its_own_output(tmp_path):
    """Blank means a fresh name on every render. The stamp used to be written
    back into the field, so the next render replaced the figure and its three
    tables; two renders inside one second collided too."""
    from datetime import datetime
    from functions.zonal_stats import profile_out_path
    when = datetime(2026, 9, 21, 12, 0, 0)
    first = profile_out_path("", tmp_path, when=when)
    assert first.name == "zonal_20260921_120000.png"
    # nothing written yet: the same second still gives the same name ...
    assert profile_out_path("", tmp_path, when=when) == first
    # ... but once the figure exists, the next render gets its own name.
    (tmp_path / "zonal_20260921_120000__profile.png").write_bytes(b"x")
    second = profile_out_path("", tmp_path, when=when)
    assert second != first and second.name == "zonal_20260921_120000-2.png"
    (tmp_path / "zonal_20260921_120000-2__profile.png").write_bytes(b"x")
    assert profile_out_path("", tmp_path, when=when).name == "zonal_20260921_120000-3.png"
    # A path the user typed is used as it is, overwriting or not as they chose.
    mine = tmp_path / "my_profile.png"
    assert profile_out_path(str(mine), tmp_path) == mine


def test_date_inferred_from_project_date_folder(tmp_path):
    assert zs.infer_profile_date(_dated(tmp_path, "2022-08-15")) == "2022-08-15"


def test_date_is_empty_for_an_undated_project(tmp_path):
    assert zs.infer_profile_date(_write_transect(tmp_path / "flat" / "t.csv")) == ""


def test_profile_label_shape():
    assert zs.profile_label("2022-08-15", "T1") == "2022-08-15 — T1"
    assert zs.profile_label("", "T1") == "T1"


def test_labels_never_contain_a_filename(tmp_path):
    a = _dated(tmp_path, "2022-08-15")
    b = _dated(tmp_path, "2023-07-02")
    series, _ = zs.load_profile_series([a, b])
    assert [s.label for s in series] == ["2022-08-15 — T1",
                                         "2023-07-02 — T1"]
    for s in series:
        assert ".csv" not in s.label and "__" not in s.label


def test_label_override_wins(tmp_path):
    a = _dated(tmp_path, "2022-08-15")
    series, _ = zs.load_profile_series([a], labels={str(a): "pre-storm"})
    assert series[0].label == "pre-storm"


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_same_csv_added_twice_yields_one_series(tmp_path):
    """The old overlay drew six legend entries for three series."""
    a = _dated(tmp_path, "2022-08-15")
    series, _ = zs.load_profile_series([a, a, a])
    assert len(series) == 1


def test_zero_sample_csv_is_named_and_excluded(tmp_path):
    good = _dated(tmp_path, "2022-08-15")
    empty = tmp_path / "2023-07-02" / "t.csv"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("transect_id,distance_m,raster_value,field\n",
                     encoding="utf-8")
    series, skipped = zs.load_profile_series([good, empty])
    assert len(series) == 1
    # The skip entry names the file AND the reason, so an
    # empty CSV can never be confused with an unreadable one.
    assert skipped == ["t.csv (carried no samples)"]


def test_mixed_fields_are_refused_and_both_named(tmp_path):
    a = _dated(tmp_path, "2022-08-15", field="Clast_length")
    b = _dated(tmp_path, "2023-07-02", field="Equivalent_diameter")
    with pytest.raises(ValueError) as ex:
        zs.load_profile_series([a, b])
    msg = str(ex.value)
    assert "Clast_length" in msg and "Equivalent_diameter" in msg


def test_csv_without_a_field_column_is_never_labelled_none(tmp_path):
    a = _dated(tmp_path, "2022-08-15", field=None)
    series, _ = zs.load_profile_series([a])
    label, factor, unit = zs._profile_display(series)
    assert "None" not in label
    assert label == "Value" and unit == "" and factor == 1.0


def test_dimensionless_field_gets_no_unit_bracket(tmp_path):
    a = _dated(tmp_path, "2022-08-15", field="Score", base=0.8)
    series, _ = zs.load_profile_series([a])
    label, _factor, unit = zs._profile_display(series)
    assert unit == ""
    assert "[" not in label and "nan" not in label.lower()


def test_transects_of_different_lengths_are_not_padded(tmp_path):
    a = _dated(tmp_path, "2022-08-15", n=13)
    b = _dated(tmp_path, "2023-07-02", n=5)
    series, _ = zs.load_profile_series([a, b])
    assert series[0].distance_m.size == 13
    assert series[1].distance_m.size == 5


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def test_summary_is_one_row_per_transect_per_date_in_display_units(tmp_path):
    a = _dated(tmp_path, "2022-08-15", base=0.050)
    b = _dated(tmp_path, "2023-07-02", base=0.058)
    series, _ = zs.load_profile_series([a, b])
    rows = zs.profile_summary_rows(series)
    assert len(rows) == 2
    for r in rows:
        assert r["unit"] == "mm"
        assert 10.0 < float(r["mean"]) < 100.0, "must be mm, not m"
        assert set(r) >= {"date", "transect_id", "field", "unit", "n_samples",
                          "length_m", "mean", "median", "min", "max",
                          "gradient_per_m"}


def test_change_difference_is_later_minus_earlier(tmp_path):
    a = _dated(tmp_path, "2022-08-15", base=0.050)
    b = _dated(tmp_path, "2023-07-02", base=0.058)
    series, _ = zs.load_profile_series([a, b])
    rows = zs.profile_change_rows(series)
    assert len(rows) == 1
    r = rows[0]
    assert r["date_from"] == "2022-08-15" and r["date_to"] == "2023-07-02"
    assert float(r["mean_difference"]) == pytest.approx(
        float(r["mean_to"]) - float(r["mean_from"]), rel=1e-6)
    assert float(r["median_difference"]) == pytest.approx(
        float(r["median_to"]) - float(r["median_from"]), rel=1e-6)
    assert float(r["mean_difference"]) > 0


def test_binned_row_count_is_ceil_length_over_bin(tmp_path):
    a = _dated(tmp_path, "2022-08-15", n=13, step=0.25)      # length 3.0 m
    series, _ = zs.load_profile_series([a])
    for bin_w in (1.0, 0.5, 2.0):
        rows = zs.profile_binned_rows(series, bin_w)
        assert len(rows) == math.ceil(3.0 / bin_w), bin_w


def test_bin_wider_than_the_transect_gives_one_bin(tmp_path):
    a = _dated(tmp_path, "2022-08-15", n=13, step=0.25)
    series, _ = zs.load_profile_series([a])
    assert len(zs.profile_binned_rows(series, 50.0)) == 1


def test_non_positive_bin_width_is_rejected(tmp_path):
    a = _dated(tmp_path, "2022-08-15")
    series, _ = zs.load_profile_series([a])
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            zs.profile_binned_rows(series, bad)


def test_orientation_tables_carry_the_degree_sign_as_utf8(tmp_path):
    a = _dated(tmp_path, "2022-08-15", field="Orientation", base=30.0)
    series, _ = zs.load_profile_series([a])
    out = tmp_path / "o__summary.csv"
    zs._write_rows_csv(zs.profile_summary_rows(series), out)
    raw = out.read_bytes()
    text = raw.decode("utf-8")              # raises if not valid UTF-8
    assert "°" in text
    assert b"\xc2\xb0" in raw


# --------------------------------------------------------------------------
# Figure anatomy
# --------------------------------------------------------------------------

def test_figure_is_two_panels_sharing_one_distance_axis(tmp_path, monkeypatch):
    a = _dated(tmp_path, "2022-08-15", elev0=3.0, base=0.050)
    b = _dated(tmp_path, "2023-07-02", elev0=2.6, base=0.058)
    series, _ = zs.load_profile_series([a, b])
    fig, close = _render_and_capture(series, tmp_path / "f.png", monkeypatch)
    try:
        assert len(fig.axes) == 2
        top, bottom = fig.axes
        assert top.get_xlim() == bottom.get_xlim(), "panels must share x"
        assert "[mm]" in top.get_ylabel(), top.get_ylabel()
        assert bottom.get_ylabel() == "Elevation [m]"
        assert "Distance along transect [m]" == bottom.get_xlabel()
        # Grain size plotted in display units, not metres.
        ydata = top.lines[0].get_ydata()
        assert 10.0 < float(min(ydata)) < 200.0
    finally:
        close(fig)


def test_figure_legend_names_dates_and_sits_outside_the_data(tmp_path,
                                                             monkeypatch):
    a = _dated(tmp_path, "2022-08-15", elev0=3.0)
    b = _dated(tmp_path, "2023-07-02", elev0=2.6)
    series, _ = zs.load_profile_series([a, b])
    fig, close = _render_and_capture(series, tmp_path / "f.png", monkeypatch)
    try:
        legend = fig.axes[0].get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == ["2022-08-15 — T1", "2023-07-02 — T1"]
        for lab in labels:
            assert ".csv" not in lab and "__" not in lab
        fig.canvas.draw()
        leg_bb = legend.get_window_extent()
        ax_bb = fig.axes[0].get_window_extent()
        assert leg_bb.x0 >= ax_bb.x1 - 1.0, "legend must not overlap the data"
    finally:
        close(fig)


def test_figure_is_single_panel_without_elevation(tmp_path, monkeypatch):
    a = _dated(tmp_path, "2022-08-15")          # no elevation column
    series, _ = zs.load_profile_series([a])
    assert all(s.elevation_m is None for s in series)
    fig, close = _render_and_capture(series, tmp_path / "f.png", monkeypatch)
    try:
        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == "Distance along transect [m]"
    finally:
        close(fig)


def test_elevation_present_for_only_some_dates(tmp_path, monkeypatch):
    a = _dated(tmp_path, "2022-08-15", elev0=3.0)
    b = _dated(tmp_path, "2023-07-02")          # no elevation
    series, _ = zs.load_profile_series([a, b])
    fig, close = _render_and_capture(series, tmp_path / "f.png", monkeypatch)
    try:
        top, bottom = fig.axes
        assert len(top.lines) == 2, "both dates on the grain-size panel"
        assert len(bottom.lines) == 1, "only the date that has elevation"
    finally:
        close(fig)


def test_caption_is_rendered_when_supplied(tmp_path, monkeypatch):
    a = _dated(tmp_path, "2022-08-15", elev0=3.0)
    series, _ = zs.load_profile_series([a])
    fig, close = _render_and_capture(series, tmp_path / "f.png", monkeypatch,
                                     caption="Coarsens landward.")
    try:
        texts = [t.get_text() for t in fig.texts]
        assert "Coarsens landward." in texts
    finally:
        close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def test_build_profile_outputs_writes_figure_and_three_tables(tmp_path):
    a = _dated(tmp_path, "2022-08-15", elev0=3.0, base=0.050)
    b = _dated(tmp_path, "2023-07-02", elev0=2.6, base=0.058)
    res = zs.build_profile_outputs([a, b], tmp_path / "out", "demo",
                                   bin_width_m=1.0, title="Upper beach",
                                   caption="Coarsening between dates.")
    assert res["png"].name == "demo__profile.png" and res["png"].exists()
    assert res["summary"].name == "demo__summary.csv" and res["summary"].exists()
    assert res["change"].name == "demo__change.csv" and res["change"].exists()
    assert res["binned"].name == "demo__binned.csv" and res["binned"].exists()
    assert res["skipped"] == []


def test_build_profile_outputs_refuses_when_nothing_plottable(tmp_path):
    empty = tmp_path / "2022-08-15" / "t.csv"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("transect_id,distance_m,raster_value\n", encoding="utf-8")
    with pytest.raises(ValueError) as ex:
        zs.build_profile_outputs([empty], tmp_path / "out", "demo")
    assert "t.csv" in str(ex.value)


# --------------------------------------------------------------------------
# Report integration
# --------------------------------------------------------------------------

def test_report_inventories_profile_figure_with_its_summary(tmp_path):
    from functions import report as rp

    zonal = tmp_path / "output_results" / "zonal"
    zonal.mkdir(parents=True)
    (zonal / "demo__profile.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    with open(zonal / "demo__summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "transect_id", "field", "unit", "mean"])
        w.writerow(["2022-08-15", "T1", "Clast_length", "mm", "57.2"])

    inv = rp.collect_project_inventory(tmp_path)
    figs = inv.get("zonal_profile_figures") or []
    assert len(figs) == 1
    assert figs[0]["path"].name == "demo__profile.png"
    assert figs[0]["summary"].name == "demo__summary.csv"

    # The profile tables must not also surface as unrecognised zonal CSVs.
    names = [e["path"].name for e in inv["zonal"]]
    assert "demo__summary.csv" not in names


def test_report_skips_profile_tables_in_the_generic_csv_scan(tmp_path):
    from functions import report as rp

    zonal = tmp_path / "output_results" / "zonal"
    zonal.mkdir(parents=True)
    for suffix in ("summary", "change", "binned"):
        (zonal / f"demo__{suffix}.csv").write_text("a,b\n1,2\n",
                                                   encoding="utf-8")
    inv = rp.collect_project_inventory(tmp_path)
    assert inv["zonal"] == []


# --------------------------------------------------------------------------
# A rejected run must leave nothing behind (review pass 1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 0.0, -1.0, None])
def test_bad_bin_width_writes_no_files(tmp_path, bad):
    """The rule is "Rejected with a message; no file written." The
    figure and two tables used to be written before the width was checked."""
    a = _dated(tmp_path, "2022-08-15", base=0.050)
    b = _dated(tmp_path, "2023-07-02", base=0.058)
    out = tmp_path / "out"
    with pytest.raises(ValueError) as ex:
        zs.build_profile_outputs([a, b], out, "z", bin_width_m=bad)
    assert "greater than zero" in str(ex.value)
    left = sorted(p.name for p in out.glob("*")) if out.exists() else []
    assert left == [], f"a rejected run left files behind: {left}"


def test_valid_bin_width_still_writes_everything(tmp_path):
    a = _dated(tmp_path, "2022-08-15", base=0.050)
    out = tmp_path / "out"
    res = zs.build_profile_outputs([a], out, "z", bin_width_m=0.5)
    assert sorted(p.name for p in out.glob("*")) == [
        "z__binned.csv", "z__change.csv", "z__profile.png", "z__summary.csv"]
    assert res["png"].exists()


def test_gui_does_not_coerce_a_typed_zero_bin_width():
    """`float(x or 1.0)` turned a typed 0 into 1 m bins, so the user never saw
    the rejection. Pin the call site."""
    import inspect
    import gui.app as app
    src = inspect.getsource(app)
    assert "state.zonal_profile_bin_m or 1.0" not in src, (
        "`or` re-introduces the silent 0 -> 1.0 coercion")
    # Anchor on the composer's own state field: other call sites legitimately
    # pass a literal bin width.
    i = src.find("state.zonal_profile_bin_m in (None")
    assert i > 0, "the composer must guard only against a blank field"
    assert "bin_width_m=" in src[max(0, i - 200):i + 200]


# --------------------------------------------------------------------------
# Report: profile outputs alone must still render (review pass 1)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# The replaced figure must be unreachable (review pass 2)
# --------------------------------------------------------------------------

def test_no_gui_path_still_builds_the_old_overlay():
    """The queue used to auto-generate `transects_overlay.png` with
    plot_transect_overlay, which labelled lines with truncated filenames, put
    the legend over the data, and forced a millimetre axis onto every field.
    Nothing may reach it any more."""
    from pathlib import Path as _P
    gui_dir = _P(__file__).resolve().parent.parent / "gui"
    offenders = []
    for py in gui_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for name in ("plot_transect_overlay",
                     "plot_transect_with_envelope_topo"):
            if name in text:
                offenders.append(f"{py.name}: {name}")
    assert offenders == [], offenders


def test_replaced_plot_functions_are_gone_from_zonal_stats():
    import functions.zonal_stats as _zs
    for name in ("plot_transect_overlay", "plot_transect_with_envelope_topo"):
        assert not hasattr(_zs, name), f"{name} should have been removed"
        assert name not in _zs.__all__


def test_queue_overlay_uses_the_profile_builder():
    """Pin the call site: the automatic post-queue figure must be the same one
    the composer produces."""
    import inspect
    import gui.app as app
    src = inspect.getsource(app)
    i = src.find("transects_overlay")
    assert i > 0, "the automatic overlay call site disappeared"
    window = src[max(0, i - 600):i + 600]
    assert "build_profile_outputs" in window, (
        "the queue overlay must go through build_profile_outputs")


def test_report_renders_profiles_when_no_other_zonal_csv_exists(tmp_path):
    """The zonal section renders a composed profile figure even when the
    project holds no polygon or transect CSV (a project of profiles only)."""
    from reportlab.platypus import Image as RImage, KeepTogether
    from PIL import Image
    from functions import report as rp, report_facts as F, report_sections as S

    zonal = tmp_path / "output_results" / "zonal"
    zonal.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(zonal / "demo__profile.png")
    with open(zonal / "demo__summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "transect_id", "field", "unit", "n_samples", "mean"])
        w.writerow(["2022-08-15", "T1", "Clast_length", "mm", "12", "57.2"])

    inv = rp.collect_project_inventory(tmp_path)
    assert inv["zonal"] == [], "fixture must have no polygon/transect CSVs"
    assert len(inv["zonal_profile_figures"]) == 1
    facts = F.gather_facts(inv, rp.compute_aggregate_stats(inv))
    story = []
    ok = S.zonal_statistics(story, rp._styles(), inv, facts, {}, lambda m: None, 4,
                            dpi=72, quality=60)
    assert ok and story, "the section must not be skipped when profiles exist"
    imgs = [f for k in story if isinstance(k, KeepTogether)
            for f in k._content if isinstance(f, RImage)]
    assert any("demo__profile.png" in str(getattr(f, "filename", ""))
               for f in imgs)


def test_report_skips_a_profile_figure_with_no_sample(tmp_path):
    """A composed profile whose summary holds n_samples = 0 everywhere is an
    empty frame; it is named as not shown, never embedded."""
    from PIL import Image
    from functions import report as rp, report_facts as F, report_sections as S
    zonal = tmp_path / "output_results" / "zonal"
    zonal.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(zonal / "demo__profile.png")
    with open(zonal / "demo__summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "transect_id", "field", "unit", "n_samples", "mean"])
        w.writerow(["2022-08-15", "T1", "Clast_length", "mm", "0", ""])
    inv = rp.collect_project_inventory(tmp_path)
    facts = F.gather_facts(inv, rp.compute_aggregate_stats(inv))
    story = []
    S.zonal_statistics(story, rp._styles(), inv, facts, {}, lambda m: None, 4,
                       dpi=72, quality=60)
    texts = " ".join(getattr(f, "text", "") for f in story)
    assert "Not shown" in texts and "no sample" in texts


def test_report_still_returns_early_with_nothing_zonal_at_all(tmp_path):
    from functions import report as rp, report_facts as F, report_sections as S
    (tmp_path / "output_results" / "zonal").mkdir(parents=True)
    inv = rp.collect_project_inventory(tmp_path)
    facts = F.gather_facts(inv, rp.compute_aggregate_stats(inv))
    story = []
    assert S.zonal_statistics(story, rp._styles(), inv, facts, {}, lambda m: None, 4,
                              dpi=72, quality=60) is False
    assert story == []
