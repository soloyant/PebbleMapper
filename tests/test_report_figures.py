"""functions/report_figures.py — the figure builders the PDF report embeds.

Every builder is exercised on synthetic data, rendered to a PNG under
``tmp_path`` (a crash inside matplotlib's draw path is only caught by an
actual render), and checked for its ``pm_meta``. The ValueError paths and
the axial-mean arithmetic are pinned numerically.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from functions import report_figures as RF

RNG_SEED = 20260910


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _lengths(rng, n, mean_m=0.04, sigma=0.45):
    return rng.lognormal(mean=np.log(mean_m), sigma=sigma, size=n)


def _clast_frame(rng, n, x0, y0, span_x, span_y, length_mean_m=0.04):
    L = _lengths(rng, n, length_mean_m)
    W = L * rng.uniform(0.5, 0.9, size=n)
    return pd.DataFrame({
        "clast_ID": np.arange(1, n + 1),
        "x": x0 + rng.uniform(0, span_x, size=n),
        "y": y0 + rng.uniform(0, span_y, size=n),
        "Clast_length": L,
        "Clast_width": W,
        "Orientation": rng.uniform(0, 180, size=n),
    })


def _render(fig: Figure, path):
    fig.savefig(str(path), dpi=80)
    assert path.stat().st_size > 1000


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def png_image(tmp_path):
    """A 64×64 RGB photograph-like PNG (pixel frame)."""
    from PIL import Image
    r = np.random.default_rng(1)
    arr = r.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    p = tmp_path / "photo.png"
    Image.fromarray(arr).save(p)
    return p


def _write_geotiff(path, arr, gt, epsg=32631):
    """Write ``arr`` (3, h, w) uint8 with a GDAL geotransform ``gt``.

    Returns the backend used, or None when neither GDAL nor rasterio imports.
    """
    try:
        from osgeo import gdal, osr
        h, w = arr.shape[1:]
        ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 3,
                                                  gdal.GDT_Byte)
        ds.SetGeoTransform(gt)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())
        for i in range(3):
            ds.GetRasterBand(i + 1).WriteArray(arr[i])
        ds.FlushCache()
        ds = None
        return "gdal"
    except ImportError:
        pass
    try:
        import rasterio
        from rasterio.transform import Affine
        h, w = arr.shape[1:]
        tr = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
        with rasterio.open(str(path), "w", driver="GTiff", width=w, height=h,
                           count=3, dtype="uint8", crs=f"EPSG:{epsg}",
                           transform=tr) as ds:
            ds.write(arr)
        return "rasterio"
    except ImportError:
        return None


# 1 cm pixels, 64 px → a 0.64 m square at (500000, 4500000).
GT = (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01)


@pytest.fixture
def geotiff(tmp_path):
    r = np.random.default_rng(2)
    arr = r.integers(60, 200, size=(3, 64, 64), dtype=np.uint8)
    p = tmp_path / "ortho.tif"
    backend = _write_geotiff(p, arr, GT)
    if backend is None:
        pytest.skip("neither GDAL nor rasterio is importable")
    return p


# --------------------------------------------------------------------------- #
#  8. axial_circular_mean                                                      #
# --------------------------------------------------------------------------- #
def test_axial_mean_straddles_zero():
    mean, R = RF.axial_circular_mean(np.array([5.0, 175.0]))
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert R == pytest.approx(math.cos(math.radians(10.0)), abs=1e-6)
    assert R == pytest.approx(0.985, abs=1e-3)


def test_axial_mean_of_uniform_is_zero_resultant():
    deg = np.arange(0, 180, 1.0)
    mean, R = RF.axial_circular_mean(deg)
    assert R == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= mean < 180.0


def test_axial_mean_aligned_and_wrapped():
    mean, R = RF.axial_circular_mean([40.0, 40.0, 220.0])   # 220 ≡ 40 axially
    assert mean == pytest.approx(40.0, abs=1e-9)
    assert R == pytest.approx(1.0, abs=1e-9)
    mean, _ = RF.axial_circular_mean([170.0, 10.0])
    assert mean == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        RF.axial_circular_mean([np.nan])


# --------------------------------------------------------------------------- #
#  1. detection_overlay                                                        #
# --------------------------------------------------------------------------- #
def test_detection_overlay_quadrat_pixels(png_image, tmp_path, rng):
    # 1 mm per pixel → the 64 px photo is 64 mm; clasts 3–6 mm long.
    df = _clast_frame(rng, 30, 5, 5, 54, 54, length_mean_m=0.004)
    fig = RF.detection_overlay(png_image, df, gsd_m=0.001, window_m=0.03)
    assert isinstance(fig, Figure)
    m = fig.pm_meta
    assert m["mode"] == "pixels"
    assert m["window_m"] == pytest.approx(0.03)
    assert 1 <= m["n_shown"] <= m["n_total"] == 30
    assert m["scale_bar_m"] == pytest.approx(0.1)
    _render(fig, tmp_path / "overlay_px.png")


def test_detection_overlay_whole_photo(png_image, tmp_path, rng):
    """``window_m=None`` shows the entire photograph — what the report
    passes for a quadrat photograph, which is one frame."""
    df = _clast_frame(rng, 25, 2, 2, 60, 60, length_mean_m=0.004)
    fig = RF.detection_overlay(png_image, df, gsd_m=0.001, window_m=None)
    m = fig.pm_meta
    assert m["window_m"] is None
    assert m["span_m"] == pytest.approx(0.064)
    assert m["n_shown"] == m["n_total"] == 25
    assert m["center_xy"] == pytest.approx((32.0, 32.0))
    _render(fig, tmp_path / "overlay_whole.png")
    with pytest.raises(ValueError, match="window_m"):
        RF.detection_overlay(png_image, df, gsd_m=0.001, window_m=-1.0)


def test_detection_overlay_georeferenced(geotiff, tmp_path, rng):
    df = _clast_frame(rng, 40, 500000.05, 4499999.4, 0.55, 0.55,
                      length_mean_m=0.03)
    # Transform read from the file itself.
    fig = RF.detection_overlay(geotiff, df, window_m=2.0)
    assert fig.pm_meta["mode"] == "georeferenced"
    assert fig.pm_meta["n_shown"] == 40         # window exceeds the raster
    _render(fig, tmp_path / "overlay_geo.png")
    # Transform passed explicitly as a GDAL geotransform, explicit centre.
    fig2 = RF.detection_overlay(geotiff, df, transform=GT,
                                center_xy=(500000.32, 4499999.68),
                                window_m=0.4, title="crop")
    assert fig2.pm_meta["center_xy"] == pytest.approx((500000.32,
                                                       4499999.68))
    assert 0 < fig2.pm_meta["n_shown"] <= 40
    _render(fig2, tmp_path / "overlay_geo2.png")


def test_detection_overlay_accepts_affine_object(geotiff, tmp_path, rng):
    affine = pytest.importorskip("affine")
    df = _clast_frame(rng, 10, 500000.1, 4499999.5, 0.4, 0.4,
                      length_mean_m=0.03)
    tr = affine.Affine(GT[1], GT[2], GT[0], GT[4], GT[5], GT[3])
    fig = RF.detection_overlay(geotiff, df, transform=tr, window_m=1.0)
    assert fig.pm_meta["mode"] == "georeferenced"
    _render(fig, tmp_path / "overlay_affine.png")


def test_detection_overlay_errors(png_image, rng):
    df = _clast_frame(rng, 5, 5, 5, 50, 50, length_mean_m=0.004)
    with pytest.raises(ValueError, match="gsd_m"):
        RF.detection_overlay(png_image, df)               # pixels, no GSD
    with pytest.raises(ValueError, match="no clasts"):
        RF.detection_overlay(png_image, df.iloc[0:0], gsd_m=0.001)
    with pytest.raises(ValueError, match="missing column"):
        RF.detection_overlay(png_image, df.drop(columns=["Orientation"]),
                             gsd_m=0.001)
    with pytest.raises(ValueError, match="outside"):
        RF.detection_overlay(png_image, df, gsd_m=0.001,
                             center_xy=(5000, 5000), window_m=0.01)
    with pytest.raises(ValueError, match="not found"):
        RF.detection_overlay(png_image.with_name("nope.png"), df,
                             gsd_m=0.001)


# --------------------------------------------------------------------------- #
#  2. cdf_overlay                                                              #
# --------------------------------------------------------------------------- #
def test_cdf_overlay_three_series(tmp_path, rng):
    series = {
        "1 m": _lengths(rng, 300, 0.03),
        "2.5 m": _lengths(rng, 800, 0.04),
        "5 m": _lengths(rng, 8300, 0.05),
    }
    fig = RF.cdf_overlay(series, dmin_mm={"1 m": 8.0, "2.5 m": 12.0,
                                          "5 m": 20.0})
    m = fig.pm_meta
    assert m["n"] == {"1 m": 300, "2.5 m": 800, "5 m": 8300}
    assert set(m["D50"]) == set(series) and set(m["D84"]) == set(series)
    for k in series:
        assert m["D50"][k] == pytest.approx(
            np.quantile(series[k], 0.5) * 1000.0)
        assert m["D84"][k] > m["D50"][k]
    assert m["dmin"]["5 m"] == pytest.approx(20.0)
    # The legend carries n with a thousands separator.
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "5 m (n = 8,300)" in texts
    _render(fig, tmp_path / "cdf.png")


def test_cdf_overlay_single_dmin_and_cm(tmp_path, rng):
    fig = RF.cdf_overlay({"a": _lengths(rng, 50), "b": _lengths(rng, 60)},
                         unit="cm", scale=100.0, marks=(50,), dmin_mm=15.0)
    assert fig.pm_meta["dmin"] == {"a": pytest.approx(1.5),
                                   "b": pytest.approx(1.5)}
    assert fig.pm_meta["marks"] == [50]
    _render(fig, tmp_path / "cdf_cm.png")


def test_cdf_overlay_errors(rng):
    with pytest.raises(ValueError, match="no series"):
        RF.cdf_overlay({})
    with pytest.raises(ValueError, match="n = 1"):
        RF.cdf_overlay({"ok": _lengths(rng, 20), "tiny": np.array([0.02])})
    with pytest.raises(ValueError, match="n = 0"):
        RF.cdf_overlay({"nan": np.array([np.nan, -1.0, 0.0])})


# --------------------------------------------------------------------------- #
#  3. histogram_adaptive                                                       #
# --------------------------------------------------------------------------- #
def test_histogram_adaptive(tmp_path, rng):
    v = _lengths(rng, 2000, 0.05, 0.4)
    fig = RF.histogram_adaptive(v, dmin_mm=10.0)
    m = fig.pm_meta
    assert m["n"] == 2000
    assert 8 <= m["bins"] <= 60
    assert m["D50"] < m["D84"] < m["D95"]
    assert m["D50"] == pytest.approx(np.quantile(v, 0.5) * 1000.0)
    assert m["fit"] is not None
    assert m["fit"]["sigma_ln"] == pytest.approx(0.4, abs=0.05)
    assert m["fit"]["median"] == pytest.approx(50.0, rel=0.1)
    assert 0.0 <= m["fit"]["ks_p"] <= 1.0
    assert m["mu_ln"] == m["fit"]["mu_ln"]        # flat copy for captions
    assert m["sigma_ln"] == m["fit"]["sigma_ln"]
    assert m["dmin"] == pytest.approx(10.0)
    _render(fig, tmp_path / "hist.png")


def test_histogram_bin_clamp_and_no_fit(tmp_path, rng):
    few = _lengths(rng, 12)
    fig = RF.histogram_adaptive(few, fit=False)
    assert fig.pm_meta["bins"] == 8
    assert fig.pm_meta["fit"] is None
    assert fig.pm_meta["mu_ln"] is None
    _render(fig, tmp_path / "hist_few.png")
    # A tight core with far outliers (tiny IQR, wide range) drives FD to
    # the top of the clamp.
    many = np.concatenate([_lengths(rng, 20000, 0.05, 0.05),
                           np.geomspace(0.001, 1.0, 20)])
    fig2 = RF.histogram_adaptive(many, fit=False)
    assert fig2.pm_meta["bins"] == 60
    _render(fig2, tmp_path / "hist_many.png")


def test_histogram_errors():
    with pytest.raises(ValueError, match="< 2"):
        RF.histogram_adaptive(np.array([0.03]))
    with pytest.raises(ValueError, match="identical"):
        RF.histogram_adaptive(np.full(10, 0.03))


# --------------------------------------------------------------------------- #
#  4. validation_diagnostics                                                   #
# --------------------------------------------------------------------------- #
def test_validation_diagnostics(tmp_path, rng):
    truth = _lengths(rng, 60, 0.05)
    detect = truth * rng.normal(1.05, 0.08, size=60)
    paired = pd.DataFrame({"truth": truth, "detect": detect})
    fig = RF.validation_diagnostics(paired)
    assert len(fig.axes) >= 3
    m = fig.pm_meta
    assert m["n"] == 60
    assert 0.5 < m["r2"] <= 1.0
    assert m["bias"] == pytest.approx(np.mean(detect - truth) * 1000.0)
    assert m["loa_low"] < m["bias"] < m["loa_high"]
    assert m["loa_high"] - m["bias"] == pytest.approx(1.96 * m["sd_diff"])
    assert m["rmse"] > 0
    assert m["D50_truth"] == pytest.approx(np.quantile(truth, 0.5) * 1000.0)
    _render(fig, tmp_path / "validation.png")


def test_validation_diagnostics_errors(rng):
    truth = _lengths(rng, 20)
    with pytest.raises(ValueError,
                       match="truth and detection are identical — no "
                             "diagnostic"):
        RF.validation_diagnostics(pd.DataFrame({"truth": truth,
                                                "detect": truth.copy()}))
    with pytest.raises(ValueError, match="< 5"):
        RF.validation_diagnostics(pd.DataFrame({"truth": truth[:4],
                                                "detect": truth[:4] * 1.1}))
    with pytest.raises(ValueError, match="missing column"):
        RF.validation_diagnostics(pd.DataFrame({"truth": truth}),
                                  detect_col="det")
    # NaNs are dropped before the n check.
    t = np.concatenate([truth[:3], [np.nan] * 5])
    d = np.concatenate([truth[:3] * 1.1, [0.02] * 5])
    with pytest.raises(ValueError, match="n = 3"):
        RF.validation_diagnostics(pd.DataFrame({"truth": t, "detect": d}))


# --------------------------------------------------------------------------- #
#  5. cell_count_map                                                           #
# --------------------------------------------------------------------------- #
def test_cell_count_map(tmp_path, rng):
    # 5 × 4 m of beach, dense on the left, sparse on the right.
    dense = _clast_frame(rng, 400, 500000.0, 4500000.0, 2.0, 4.0)
    sparse = _clast_frame(rng, 40, 500002.0, 4500000.0, 3.0, 4.0)
    df = pd.concat([dense, sparse], ignore_index=True)
    fig = RF.cell_count_map(df, cellsize_m=1.0, crs_label="EPSG:32631")
    m = fig.pm_meta
    assert m["n_cells_total"] == 20
    assert 1 <= m["n_cells"] <= 20
    assert m["n_min"] >= 1
    assert m["n_min"] <= m["n_median"] <= m["n_max"]
    assert 0.0 < m["frac_cells_lt30"] < 1.0
    assert m["extent"] == (500000.0, 500005.0, 4500000.0, 4500004.0)
    # Full coordinates on the axes: no offset text.
    fig.canvas.draw()
    assert fig.axes[0].xaxis.get_offset_text().get_text() == ""
    _render(fig, tmp_path / "cells.png")


def test_cell_count_map_explicit_extent_and_errors(tmp_path, rng):
    df = _clast_frame(rng, 50, 10.0, 20.0, 3.0, 3.0)
    fig = RF.cell_count_map(df, cellsize_m=0.5, extent=(10, 13, 20, 23))
    assert fig.pm_meta["n_cells_total"] == 36
    _render(fig, tmp_path / "cells_extent.png")
    with pytest.raises(ValueError, match="no clasts"):
        RF.cell_count_map(pd.DataFrame({"x": [np.nan], "y": [1.0]}))
    with pytest.raises(ValueError, match="inside the extent"):
        RF.cell_count_map(df, extent=(100, 101, 100, 101))
    with pytest.raises(ValueError, match="cellsize_m"):
        RF.cell_count_map(df, cellsize_m=0.0)
    with pytest.raises(ValueError, match="increase cellsize_m"):
        RF.cell_count_map(df, cellsize_m=0.001, extent=(0, 100, 0, 100))


# --------------------------------------------------------------------------- #
#  6. orientation_rose                                                         #
# --------------------------------------------------------------------------- #
def test_orientation_rose(tmp_path, rng):
    deg = np.mod(rng.normal(60.0, 15.0, size=500), 180.0)
    fig = RF.orientation_rose(deg, title="Orientation")
    m = fig.pm_meta
    assert m["n"] == 500
    assert m["mean_deg"] == pytest.approx(60.0, abs=4.0)
    assert 0.5 < m["R"] < 1.0
    ref_mean, ref_R = RF.axial_circular_mean(deg)
    assert (m["mean_deg"], m["R"]) == (ref_mean, ref_R)
    _render(fig, tmp_path / "rose.png")


def test_orientation_rose_errors():
    with pytest.raises(ValueError, match="< 2"):
        RF.orientation_rose(np.array([10.0, np.nan]))
    with pytest.raises(ValueError, match="bin_deg"):
        RF.orientation_rose(np.array([10.0, 20.0]), bin_deg=7.0)


# --------------------------------------------------------------------------- #
#  7. zone_location_map                                                        #
# --------------------------------------------------------------------------- #
def _zone_geojson(path, features):
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_zone_location_map_georeferenced(geotiff, tmp_path):
    x0, y0 = 500000.0, 4500000.0
    zones = _zone_geojson(tmp_path / "zones.geojson", [
        {"type": "Feature",
         "properties": {"id": "z1", "shape": "polygon", "name": "Zone A"},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[x0 + 0.1, y0 - 0.1],
                                       [x0 + 0.3, y0 - 0.1],
                                       [x0 + 0.3, y0 - 0.3],
                                       [x0 + 0.1, y0 - 0.3],
                                       [x0 + 0.1, y0 - 0.1]]]}},
        {"type": "Feature",
         "properties": {"id": "t1", "shape": "transect"},
         "geometry": {"type": "LineString",
                      "coordinates": [[x0 + 0.05, y0 - 0.5],
                                      [x0 + 0.6, y0 - 0.55]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [x0 + 0.5, y0 - 0.2]}},
    ])
    fig = RF.zone_location_map(geotiff, [zones])
    m = fig.pm_meta
    assert m["n_features"] == 3
    assert m["labels"] == ["Zone A", "t1", "3"]     # name → id → FID
    assert m["georeferenced"] is True
    assert m["crs"] and "32631" in m["crs"]
    _render(fig, tmp_path / "zones.png")


def test_zone_location_map_pixels_two_files(png_image, tmp_path):
    a = _zone_geojson(tmp_path / "a.geojson", [
        {"type": "Feature", "properties": {"name": "left"},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[5, 5], [30, 5], [30, 30], [5, 30],
                                       [5, 5]]]}}])
    b = _zone_geojson(tmp_path / "b.geojson", [
        {"type": "Feature", "properties": {"name": "right"},
         "geometry": {"type": "MultiPolygon",
                      "coordinates": [[[[35, 35], [60, 35], [60, 60],
                                        [35, 60], [35, 35]]]]}}])
    fig = RF.zone_location_map(png_image, [a, b], labels=False)
    assert fig.pm_meta["georeferenced"] is False
    assert fig.pm_meta["labels"] == []
    assert fig.pm_meta["n_features"] == 2
    assert fig.axes[0].get_legend() is not None       # one entry per file
    _render(fig, tmp_path / "zones_px.png")


def test_zone_location_map_errors(png_image, tmp_path):
    with pytest.raises(ValueError, match="no GeoJSON"):
        RF.zone_location_map(png_image, [])
    empty = _zone_geojson(tmp_path / "empty.geojson", [])
    with pytest.raises(ValueError, match="no features"):
        RF.zone_location_map(png_image, [empty])
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read GeoJSON"):
        RF.zone_location_map(png_image, [bad])


# --------------------------------------------------------------------------- #
#  Style contract                                                              #
# --------------------------------------------------------------------------- #
def test_figure_widths_and_no_titles_by_default(tmp_path, rng):
    v = _lengths(rng, 100)
    single = RF.histogram_adaptive(v)
    panel = RF.validation_diagnostics(pd.DataFrame(
        {"truth": v, "detect": v * 1.1}))
    assert single.get_size_inches()[0] == pytest.approx(6.5)
    assert panel.get_size_inches()[0] == pytest.approx(7.2)
    assert single.axes[0].get_title() == ""
    assert single.axes[0].get_xlabel() == "Clast length (mm)"
    assert single.get_constrained_layout()
    rose = RF.orientation_rose(rng.uniform(0, 180, 50))
    assert rose.axes[0].get_title() == ""


def test_transform_coercion():
    # GDAL 6-tuple, Affine 9-tuple and identity all coerce as documented.
    t = RF._coerce_transform(GT)
    assert t == (0.01, 0.0, 500000.0, 0.0, -0.01, 4500000.0)
    assert RF._coerce_transform((0.01, 0.0, 500000.0, 0.0, -0.01, 4500000.0,
                                 0.0, 0.0, 1.0)) == t
    assert RF._coerce_transform(None) is None
    assert RF._coerce_transform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) is None
    col, row = RF._world_to_pix(t, 500000.32, 4499999.68)
    assert (col, row) == pytest.approx((32.0, 32.0))
    assert RF._pix_to_world(t, 32, 32) == pytest.approx((500000.32,
                                                        4499999.68))
    with pytest.raises(ValueError):
        RF._coerce_transform((1.0, 2.0, 3.0))
