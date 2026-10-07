"""Regression tests for the publication-map exporter (functions/map_export.py).

Each block traces one defect the figure review of the PDF report found in
the maps this module renders, and the option that fixes it:

  1. Frame — ``extent='raster'`` clips the frame to the valid-data bounding
     box plus a margin; the default ``'project'`` keeps the ortho union.
  2. Classes — ``classes=N`` draws a discrete, segmented colorbar with the
     class edges as ticks, rounded by ``uncertainty`` or a per-unit rule;
     the classification is stated in the metadata.
  3. Cyclic — an Orientation field always renders with ``twilight`` on a
     fixed 0–180° scale with ticks every 45°, whatever colormap was passed.
  4. Axes — no offset notation; attribution sits below the axes on every
     path (scatter and raster).
  5. Rainbow — jet/rainbow/turbo are replaced by viridis unless
     ``allow_rainbow=True``.
  6. Units — the ``functions.units`` registry is authoritative for a field
     it knows; the magnitude rule survives only for unknown fields.
  7. Metadata — a caption-ready dict on ``fig.map_metadata`` and, through
     ``save_publication_map``, a ``<png>.json`` sidecar.

Every map is rendered offline (``basemap=None``) on a tiny synthetic
project written with GDAL under ``tmp_path``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("osgeo")
pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

from functions import map_export as mx


# --------------------------------------------------------------------------- #
#  Synthetic project: 30 m × 30 m ortho, one size raster with a 6 × 6 valid
#  block in its top-left corner, one orientation raster, one per-clast CSV.
# --------------------------------------------------------------------------- #
EPSG = 32630
X0, Y1 = 500000.0, 4500030.0          # top-left corner (xmin, ymax)
SPAN = 30.0                            # metres on each side
VALID_ROWS = (2, 8)                    # valid raster rows   [2, 8)
VALID_COLS = (3, 9)                    # valid raster cols   [3, 9)


def _write_gtiff(path, arrays, px, nodata=None, dtype=None):
    from osgeo import gdal, osr
    arrays = [np.asarray(a) for a in arrays]
    ny, nx = arrays[0].shape
    if dtype is None:
        dtype = gdal.GDT_Byte if arrays[0].dtype == np.uint8 else gdal.GDT_Float32
    ds = gdal.GetDriverByName("GTiff").Create(str(path), nx, ny, len(arrays), dtype)
    ds.SetGeoTransform([X0, px, 0, Y1, 0, -px])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds.SetProjection(srs.ExportToWkt())
    for i, a in enumerate(arrays, start=1):
        band = ds.GetRasterBand(i)
        band.WriteArray(a)
        if nodata is not None:
            band.SetNoDataValue(nodata)
    ds = None
    return Path(path)


@pytest.fixture
def project(tmp_path):
    rng = np.random.default_rng(7)
    # Ortho: 60 × 60 px at 0.5 m, three uint8 bands, no saturated pixels.
    ortho = _write_gtiff(
        tmp_path / "site.tif",
        [rng.integers(30, 220, size=(60, 60)).astype(np.uint8) for _ in range(3)],
        px=0.5)

    # Size raster (metres): 30 × 30 cells at 1 m; valid only in a 6 × 6 block.
    size = np.full((30, 30), -9999.0, dtype=np.float32)
    r0, r1 = VALID_ROWS
    c0, c1 = VALID_COLS
    block = np.linspace(0.02, 0.12, (r1 - r0) * (c1 - c0)).reshape(r1 - r0, c1 - c0)
    size[r0:r1, c0:c1] = block
    size_ras = _write_gtiff(
        tmp_path / "site_individual_clast_values_Clast_length_D50_cellsize=1.0m.tif",
        [size], px=1.0, nodata=-9999.0)

    # Orientation raster (degrees, stored −90..90), fully valid.
    orient = rng.uniform(-90, 90, size=(30, 30)).astype(np.float32)
    orient_ras = _write_gtiff(
        tmp_path / "site_individual_clast_values_Orientation_average_cellsize=1.0m.tif",
        [orient], px=1.0, nodata=-9999.0)

    # Per-clast CSV inside the ortho.
    import pandas as pd
    n = 300
    df = pd.DataFrame({
        "clast_ID": np.arange(1, n + 1),
        "x": X0 + rng.uniform(5, 25, size=n),
        "y": Y1 - rng.uniform(5, 25, size=n),
        "Clast_length": rng.lognormal(np.log(0.05), 0.4, size=n),
        "Orientation": rng.uniform(-90, 90, size=n),
        "Foo_unknown": rng.uniform(0.2, 0.9, size=n),
    })
    csv = tmp_path / "site_merged_individual_clast_values.csv"
    df.to_csv(csv, index=False)
    return {"ortho": ortho, "size": size_ras, "orient": orient_ras,
            "csv": csv, "df": df, "root": tmp_path}


def _render(project, **kw):
    kw.setdefault("ortho_tif", str(project["ortho"]))
    kw.setdefault("basemap", None)
    if kw.get("layer", "vector") == "vector":
        kw.setdefault("clast_csv", str(project["csv"]))
    return mx.make_publication_map(**kw)


def _tick_labels(fig, cbar):
    fig.canvas.draw()
    return [t.get_text() for t in cbar.ax.get_yticklabels() if t.get_text()]


def _valid_bbox():
    r0, r1 = VALID_ROWS
    c0, c1 = VALID_COLS
    return (X0 + c0, X0 + c1, Y1 - r1, Y1 - r0)


# =========================================================================== #
#  1. Frame
# =========================================================================== #
def test_default_extent_keeps_project_union(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]))
    try:
        ax = fig.axes[0]
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # The whole ortho is inside the frame (today's Map-tab behaviour).
        assert xmin <= X0 and xmax >= X0 + SPAN
        assert ymin <= Y1 - SPAN and ymax >= Y1
        assert fig.map_metadata["extent_mode"] == "project"
    finally:
        plt.close(fig)


def test_extent_raster_clips_to_valid_data_plus_margin(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  extent="raster")
    try:
        ax = fig.axes[0]
        bx0, bx1, by0, by1 = _valid_bbox()
        m = 0.10 * (bx1 - bx0)
        assert ax.get_xlim() == pytest.approx((bx0 - m, bx1 + m), abs=1e-6)
        assert ax.get_ylim() == pytest.approx((by0 - m, by1 + m), abs=1e-6)
        meta = fig.map_metadata
        assert meta["extent_mode"] == "raster"
        assert meta["extent"] == pytest.approx(
            [bx0 - m, bx1 + m, by0 - m, by1 + m], abs=1e-6)
        # The raster now fills most of the frame instead of ~4 % of it.
        frame_area = (ax.get_xlim()[1] - ax.get_xlim()[0]) * \
                     (ax.get_ylim()[1] - ax.get_ylim()[0])
        assert (bx1 - bx0) * (by1 - by0) / frame_area > 0.6
    finally:
        plt.close(fig)


def test_extent_raster_margin_is_configurable(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  extent="raster", extent_margin=0.0)
    try:
        assert fig.axes[0].get_xlim() == pytest.approx(_valid_bbox()[:2])
    finally:
        plt.close(fig)


def test_extent_raster_on_vector_layer_frames_the_points(project):
    fig = _render(project, extent="raster")
    try:
        df = project["df"]
        sx = df.x.max() - df.x.min()
        assert fig.axes[0].get_xlim() == pytest.approx(
            (df.x.min() - 0.1 * sx, df.x.max() + 0.1 * sx))
    finally:
        plt.close(fig)


def test_extent_rejects_unknown_mode(project):
    with pytest.raises(ValueError):
        _render(project, extent="galaxy")


def test_valid_data_bbox_helper():
    arr = np.full((10, 10), np.nan)
    arr[2:5, 4:8] = 1.0
    ext = (0.0, 10.0, 0.0, 10.0)
    assert mx._valid_data_bbox(arr, ext) == (4.0, 8.0, 5.0, 8.0)
    # Nothing finite → the full extent, so a frame always exists.
    assert mx._valid_data_bbox(np.full((3, 3), np.nan), ext) == ext


# =========================================================================== #
#  2. Classes
# =========================================================================== #
def test_default_colorbar_is_continuous_with_three_decimals(project):
    """Without ``classes`` the Map tab's continuous 256-bin bar is unchanged."""
    from matplotlib.ticker import FormatStrFormatter
    fig = _render(project, layer="raster", raster_tif=str(project["size"]))
    try:
        cbar = fig._pm_colorbar
        assert isinstance(cbar.ax.yaxis.get_major_formatter(), FormatStrFormatter)
        assert fig._pm_mappable.norm.N == 257           # 256 quantile bins
        meta = fig.map_metadata
        assert meta["classes"] is None
        assert meta["classification"] == (
            "quantile, continuous (256 quantile bins), 2–98 % clip")
    finally:
        plt.close(fig)


def test_classes_draws_discrete_quantile_colorbar(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  classes=6)
    try:
        meta = fig.map_metadata
        assert meta["classes"] == 6
        assert len(meta["class_edges"]) == 7
        assert meta["classification"] == "quantile, 6 classes, 2–98 % clip"
        # The artist carries exactly six colours — a segmented bar, not a ramp.
        assert fig._pm_mappable.cmap.N == 6
        assert fig._pm_mappable.norm.N == 7
        cbar = fig._pm_colorbar
        assert list(cbar.get_ticks()) == pytest.approx(meta["class_edges"])
        # Edges are quantile-spaced inside the clipped range.
        edges = np.asarray(meta["class_edges"])
        assert edges[0] == pytest.approx(meta["value_min"])
        assert edges[-1] == pytest.approx(meta["value_max"])
        assert np.all(np.diff(edges) > 0)
        # A 20–120 mm scale is labelled to whole millimetres, not 73.365.
        labels = _tick_labels(fig, cbar)
        assert labels and all(re.fullmatch(r"\d+", s) for s in labels), labels
    finally:
        plt.close(fig)


def test_classes_linear_is_equal_interval(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  classes=4, color_scale="linear")
    try:
        meta = fig.map_metadata
        assert meta["classification"] == "equal interval, 4 classes, 5–95 % clip"
        edges = np.asarray(meta["class_edges"])
        assert np.allclose(np.diff(edges), np.diff(edges)[0])
    finally:
        plt.close(fig)


def test_classes_fixed_range_is_named(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  classes=5, vmin=20.0, vmax=100.0)
    try:
        meta = fig.map_metadata
        assert meta["classification"] == "quantile, 5 classes, fixed range"
        assert meta["class_edges"][0] == 20.0 and meta["class_edges"][-1] == 100.0
    finally:
        plt.close(fig)


def test_class_labels_round_to_uncertainty(project):
    from functions.precision import decimals_for
    # ± 5 mm → whole mm; ± 0.05 mm → two decimals (functions.precision rules).
    assert decimals_for(5.0, "mm") == 0 and decimals_for(0.05, "mm") == 2
    for u, pattern in ((5.0, r"\d+"), (0.05, r"\d+\.\d{2}")):
        fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                      classes=6, uncertainty=u)
        try:
            labels = _tick_labels(fig, fig._pm_colorbar)
            assert labels and all(re.fullmatch(pattern, s) for s in labels), (u, labels)
            assert fig.map_metadata["uncertainty"] == u
        finally:
            plt.close(fig)


def test_classes_on_vector_layer(project):
    fig = _render(project, classes=6)
    try:
        assert fig._pm_mappable.cmap.N == 6
        assert fig.map_metadata["classification"] == "quantile, 6 classes, 2–98 % clip"
    finally:
        plt.close(fig)


def test_classes_rejects_fewer_than_two(project):
    with pytest.raises(ValueError):
        _render(project, classes=1)


def test_default_decimals_rule():
    assert mx._default_decimals("mm", 73.4) == 0
    assert mx._default_decimals("mm", 8.2) == 1
    assert mx._default_decimals("°", 180) == 0
    assert mx._default_decimals("", 0.8) == 2
    assert mx._default_decimals("φ", 1.3) == 2
    # Velocities: two significant figures, not a fixed decimal count.
    fmt = mx._tick_formatter("m/s", None, 0.0421)
    assert fmt(0.0421) == "0.042"
    assert fmt(1.2345) == "1.2"
    assert mx._tick_formatter("mm", None, 50.0)(-0.0) == "0"
    assert mx._tick_formatter("mm", 0.05, 50.0)(12.345) == "12.35"


# =========================================================================== #
#  3. Cyclic (orientation)
# =========================================================================== #
@pytest.mark.parametrize("requested", ["jet", "viridis", "hsv", "twilight_shifted"])
def test_orientation_raster_always_twilight_0_180(project, requested):
    fig = _render(project, layer="raster", raster_tif=str(project["orient"]),
                  cmap=requested, classes=6, vmin=-90, vmax=90)
    try:
        im = fig._pm_mappable
        assert im.get_cmap().name == "twilight"
        assert im.get_clim() == (0.0, 180.0)
        cbar = fig._pm_colorbar
        assert list(cbar.get_ticks()) == [0, 45, 90, 135, 180]
        assert _tick_labels(fig, cbar) == ["0", "45", "90", "135", "180"]
        meta = fig.map_metadata
        assert meta["cyclic"] is True
        assert meta["unit"] == "°"
        assert meta["classes"] is None                  # classes ignored
        assert (meta["value_min"], meta["value_max"]) == (0.0, 180.0)
        assert meta["classification"] == "cyclic, fixed 0–180° scale"
        # −90..90 input is folded onto the axial 0–180 range.
        assert 0.0 <= meta["data_min"] and meta["data_max"] < 180.0
    finally:
        plt.close(fig)


def test_orientation_vector_always_twilight_0_180(project):
    fig = _render(project, field="Orientation", cmap="jet")
    try:
        sc = fig._pm_mappable
        assert sc.get_cmap().name == "twilight"
        assert sc.get_clim() == (0.0, 180.0)
        assert list(fig._pm_colorbar.get_ticks()) == [0, 45, 90, 135, 180]
        assert np.all(sc.get_array() >= 0) and np.all(sc.get_array() < 180)
    finally:
        plt.close(fig)


def test_is_cyclic_field_matches_registry_flag():
    assert mx._is_cyclic_field("Orientation")
    assert mx._is_cyclic_field("site_Orientation_average_cellsize=1.0m")
    assert not mx._is_cyclic_field("Clast_length")


# =========================================================================== #
#  4. Axes: no offset notation, attribution below the axes
# =========================================================================== #
def _attribution_annotations(ax):
    return [t for t in ax.texts
            if isinstance(t, Annotation) and "Basemap:" in t.get_text()]


@pytest.mark.parametrize("layer", ["vector", "raster"])
def test_no_offset_and_attribution_below_axes(project, layer, monkeypatch):
    monkeypatch.setattr(mx, "_add_basemap_safe", lambda *a, **k: True)
    kw = {"layer": layer, "basemap": "Test.Provider"}
    if layer == "raster":
        kw["raster_tif"] = str(project["size"])
    fig = _render(project, **kw)
    try:
        ax = fig.axes[0]
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            assert fmt.get_useOffset() is False
        fig.canvas.draw()
        # Full eastings on the ticks, no "+5e5" offset text.
        assert ax.xaxis.get_offset_text().get_text() == ""
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert any(s.replace(",", "").startswith("5000") for s in labels), labels
        notes = _attribution_annotations(ax)
        assert len(notes) == 1
        assert notes[0].xy[1] < 0                         # below the axes
        assert fig.map_metadata["basemap_added"] is True
    finally:
        plt.close(fig)


def test_no_attribution_without_basemap(project):
    fig = _render(project)
    try:
        assert _attribution_annotations(fig.axes[0]) == []
        assert fig.map_metadata["basemap_added"] is False
    finally:
        plt.close(fig)


# =========================================================================== #
#  5. Rainbow colormaps
# =========================================================================== #
@pytest.mark.parametrize("bad", ["jet", "rainbow", "turbo"])
def test_rainbow_refused_by_default(project, bad):
    messages = []
    fig = _render(project, cmap=bad, log_fn=messages.append)
    try:
        assert fig._pm_mappable.get_cmap().name == "viridis"
        assert fig.map_metadata["cmap"] == "viridis"
        assert any("rainbow" in m for m in messages)
    finally:
        plt.close(fig)


def test_rainbow_allowed_when_explicit(project):
    fig = _render(project, cmap="jet", allow_rainbow=True)
    try:
        assert fig._pm_mappable.get_cmap().name == "jet"
    finally:
        plt.close(fig)


def test_default_cmap_is_viridis(project):
    fig = _render(project)
    try:
        assert fig._pm_mappable.get_cmap().name == "viridis"
    finally:
        plt.close(fig)


# =========================================================================== #
#  6. Units: the registry is authoritative
# =========================================================================== #
def test_registry_unit_wins_over_magnitude_for_known_fields():
    vals = np.full(50, 0.05)                    # 5 cm: the magnitude rule says cm
    assert mx._pick_size_unit(vals) == ("cm", 100.0)
    assert mx._resolve_unit_for_field("Clast_length", vals, "metric", "auto") == ("mm", 1000.0)
    assert mx._resolve_unit_for_field("Clast_width", np.full(5, 2.0), "metric", "auto") == ("mm", 1000.0)
    # Known to the registry but absent from the module's own kind table.
    assert mx._resolve_unit_for_field("Perimeter", vals, "metric", "auto") == ("mm", 1000.0)
    assert mx._resolve_unit_for_field("Surface_area", vals, "metric", "auto") == ("mm²", 1_000_000.0)
    assert mx._resolve_unit_for_field("density", vals, "metric", "auto") == ("clasts/m²", 1.0)
    # A compound raster stem lands on the registry's longest match.
    stem = "site_individual_clast_values_Clast_length_D50_cellsize=1.0m"
    assert mx._resolve_unit_for_field(stem, vals, "metric", "auto") == ("mm", 1000.0)


def test_parameter_overrides_field_unit():
    vals = np.full(20, 1.2)
    assert mx._resolve_unit_for_field("Clast_length", vals, "metric", "auto",
                                      parameter="sorting") == ("φ", 1.0)
    assert mx._resolve_unit_for_field("Clast_length", vals, "metric", "auto",
                                      parameter="packing_index") == ("", 1.0)
    assert mx._resolve_unit_for_field("Clast_length", vals, "metric", "auto",
                                      parameter="D50") == ("mm", 1000.0)


def test_parameter_from_stem():
    f = mx._parameter_from_stem
    assert f("site_individual_clast_values_Clast_length_D50_cellsize=1.0m") == "D50"
    assert f("x_Clast_length_folk_ward_sorting_cellsize=1.0m") == "folk_ward_sorting"
    assert f("x_Clast_length_sorting_cellsize=2.5m") == "sorting"
    assert f("x_individual_clast_values_density_cellsize=1.0m") == "density"
    assert f("x_Clast_length_std_cellsize=1.0m") == "std"
    assert f("Clast_length") is None
    assert f(None) is None


def test_magnitude_rule_only_for_unknown_and_explicit_and_imperial():
    vals = np.full(20, 0.05)
    # Unknown field: unchanged fallback (no kind → dimensionless).
    assert mx._resolve_unit_for_field("Foo_unknown", vals, "metric", "auto") == ("", 1.0)
    # Explicit override still wins.
    assert mx._resolve_unit_for_field("Clast_length", vals, "metric", "cm") == ("cm", 100.0)
    assert mx._resolve_unit_for_field("Clast_length", vals, "metric", "m") == ("m", 1.0)
    # Imperial is not in the registry: magnitude rule.
    assert mx._resolve_unit_for_field("Clast_length", vals, "imperial", "auto") == ("in", 39.3701)


def test_rendered_size_raster_is_in_millimetres(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]))
    try:
        meta = fig.map_metadata
        assert meta["unit"] == "mm" and meta["unit_factor"] == 1000.0
        assert meta["colorbar_label"].endswith("[mm]")
        assert 20.0 <= meta["value_min"] < meta["value_max"] <= 120.0
    finally:
        plt.close(fig)


def test_rendered_sorting_raster_is_in_phi(project, tmp_path):
    arr = np.random.default_rng(3).uniform(0.3, 1.5, size=(30, 30)).astype(np.float32)
    ras = _write_gtiff(
        tmp_path / "site_individual_clast_values_Clast_length_sorting_cellsize=1.0m.tif",
        [arr], px=1.0, nodata=-9999.0)
    fig = _render(project, layer="raster", raster_tif=str(ras))
    try:
        meta = fig.map_metadata
        assert meta["parameter"] == "sorting"
        assert meta["unit"] == "φ" and meta["unit_factor"] == 1.0
        assert meta["value_max"] < 2.0                # not ×1000
    finally:
        plt.close(fig)


# =========================================================================== #
#  7. Metadata and sidecar
# =========================================================================== #
REQUIRED_KEYS = {
    "field", "parameter", "layer", "cell_size", "unit", "unit_factor", "cmap",
    "classification", "class_edges", "value_min", "value_max", "data_min",
    "data_max", "n_valid", "n_total", "crs", "crs_name", "extent_mode",
    "extent", "data_extent", "colorbar_label", "source", "ortho", "basemap",
    "basemap_added", "exporter_version", "exported_at",
}


def test_metadata_contents_raster(project):
    fig, meta = _render(project, layer="raster", raster_tif=str(project["size"]),
                        return_metadata=True)
    try:
        assert meta is fig.map_metadata
        assert REQUIRED_KEYS <= set(meta)
        assert meta["field"] == project["size"].stem
        assert meta["parameter"] == "D50"
        assert meta["layer"] == "raster"
        assert meta["cell_size"] == pytest.approx(1.0)
        assert meta["crs"] == f"EPSG:{EPSG}"
        assert meta["n_valid"] == 36 and meta["n_total"] == 900
        assert meta["data_min"] == pytest.approx(20.0) and meta["data_max"] == pytest.approx(120.0)
        assert meta["data_min"] <= meta["value_min"] < meta["value_max"] <= meta["data_max"]
        assert meta["exporter_version"] == mx.EXPORTER_VERSION
        stamp = datetime.fromisoformat(meta["exported_at"])
        assert stamp.tzinfo is not None
        assert meta["source"] == str(project["size"])
        assert meta["ortho"] == str(project["ortho"])
        assert meta["basemap"] is None
    finally:
        plt.close(fig)


def test_metadata_contents_vector(project):
    fig = _render(project, field="Clast_length")
    try:
        meta = fig.map_metadata
        assert REQUIRED_KEYS <= set(meta)
        assert meta["field"] == "Clast_length"
        assert meta["parameter"] is None
        assert meta["cell_size"] is None
        assert meta["n_valid"] == meta["n_total"] == len(project["df"])
        assert meta["unit"] == "mm"
    finally:
        plt.close(fig)


def test_explicit_parameter_is_recorded(project):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  parameter="median")
    try:
        assert fig.map_metadata["parameter"] == "median"
    finally:
        plt.close(fig)


def test_save_publication_map_writes_png_and_sidecar(project, tmp_path):
    fig = _render(project, layer="raster", raster_tif=str(project["size"]),
                  extent="raster", classes=6)
    out = tmp_path / "out" / "map.png"
    try:
        written = mx.save_publication_map(fig, out, dpi=72)
    finally:
        plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
    side = mx.metadata_sidecar_path(out)
    assert side == out.with_name("map.png.json") and side.exists()
    on_disk = json.loads(side.read_text(encoding="utf-8"))
    assert on_disk == mx.read_map_metadata(out)
    assert on_disk["output"] == str(out) and on_disk["dpi"] == 72
    assert on_disk["format"] == "png"
    assert on_disk["classification"] == "quantile, 6 classes, 2–98 % clip"
    assert on_disk["extent_mode"] == "raster"
    assert written["sidecar"] == str(side)
    # Plain JSON types only — the report reads this with the stdlib.
    assert all(isinstance(e, float) for e in on_disk["class_edges"])


def test_read_map_metadata_missing_returns_none(tmp_path):
    assert mx.read_map_metadata(tmp_path / "nothing.png") is None


def test_json_safe_converts_numpy_and_nan():
    out = mx._json_safe({"a": np.float32(1.5), "b": np.int64(3),
                         "c": np.array([1.0, np.nan]), "d": Path("x"),
                         "e": float("nan"), "f": np.bool_(True)})
    assert out == {"a": 1.5, "b": 3, "c": [1.0, None], "d": "x",
                   "e": None, "f": True}
    json.dumps(out)


# =========================================================================== #
#  Preserved behaviour
# =========================================================================== #
def test_return_type_is_figure_by_default(project):
    fig = _render(project)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)


def test_all_nodata_raster_still_renders(project, tmp_path):
    ras = _write_gtiff(
        tmp_path / "site_individual_clast_values_Clast_length_D50_cellsize=1.0m_empty.tif",
        [np.full((30, 30), -9999.0, dtype=np.float32)], px=1.0, nodata=-9999.0)
    fig = _render(project, layer="raster", raster_tif=str(ras), extent="raster",
                  classes=6)
    try:
        meta = fig.map_metadata
        assert meta["n_valid"] == 0 and meta["data_min"] is None
        # No valid cells → the frame falls back to the raster's full extent.
        assert meta["extent_mode"] == "raster"
    finally:
        plt.close(fig)


def test_the_scale_bar_follows_the_unit_system():
    """An imperial map used to carry a bar reading "0 - 2.5 - 5 m" beside a
    colorbar in inches."""
    from functions.map_export import _scale_bar_length
    # 40 m across: a 10 m bar in metric.
    length_m, label, unit = _scale_bar_length(40.0, "metric")
    assert (length_m, label, unit) == (10.0, 10.0, "m")
    # The same map in imperial: a round number of feet, drawn in metres.
    length_m, label, unit = _scale_bar_length(40.0, "imperial")
    assert unit == "ft" and label in (20, 20.0, 50, 50.0)
    assert length_m == pytest.approx(label / 3.28084)
    assert 0.1 * 40.0 < length_m < 0.5 * 40.0      # still about a fifth of the map
    # Kilometres across: miles rather than thousands of feet.
    length_m, label, unit = _scale_bar_length(20000.0, "imperial")
    assert unit == "mi" and length_m == pytest.approx(label * 5280 / 3.28084)


def test_a_raster_colorbar_is_labelled_by_its_quantity_not_its_filename():
    """The whole stem used to be prettified into the label: "Example 03
    Etretat etretat 20200610 ortho crop merged density cellsize=1.0m
    [clasts/m2]"."""
    from functions.map_export import _raster_label, _parameter_from_stem
    stem = "Project__ortho_crop_merged_Clast_length_D50_cellsize=1.0m"
    assert _raster_label(stem, _parameter_from_stem(stem), "mm") == "Clast length [mm]"
    # A parameter that replaces the unit names the quantity itself: the field
    # token in the stem must not be paired with clasts per m2.
    dens = "Project__ortho_crop_xprs_Clast_length_density_cellsize=1.0m"
    label = _raster_label(dens, _parameter_from_stem(dens), "clasts/m²")
    assert label.endswith("[clasts/m²]") and "cellsize" not in label
    assert "Clast length" not in label
    phi = "Project__ortho_merged_Clast_length_folk_ward_sorting_cellsize=1.0m"
    assert "φ" in _raster_label(phi, _parameter_from_stem(phi), "φ")


def test_a_multi_band_raster_says_which_band_is_drawn(tmp_path):
    """One band per size bin, drawn from band 1, was labelled as if it were
    the whole file."""
    from osgeo import gdal, osr
    from functions.map_export import _raster_band_count
    path = tmp_path / "bins.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, 3, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 4, 0, -1])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32630)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(1, 4):
        ds.GetRasterBand(b).WriteArray(np.full((4, 4), float(b), dtype="float32"))
    ds = None
    assert _raster_band_count(path) == 3
    single = tmp_path / "one.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(single), 4, 4, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 4, 0, -1]); ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.ones((4, 4), dtype="float32")); ds = None
    assert _raster_band_count(single) == 1
    assert _raster_band_count(tmp_path / "missing.tif") == 1
