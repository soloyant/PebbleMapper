# tests/test_quadrat_validation.py
import numpy as np
import pandas as pd
import pytest

from functions import quadrat_validation as qv


def _make_tif(tmp_path, gt, nx=3352, ny=3352, proj_epsg=32630):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import osr
    p = str(tmp_path / "src.tif")
    ds = gdal.GetDriverByName("GTiff").Create(p, nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    if proj_epsg:
        sr = osr.SpatialReference(); sr.ImportFromEPSG(proj_epsg)
        ds.SetProjection(sr.ExportToWkt())
    ds.GetRasterBand(1).Fill(100)
    ds = None
    return p


# A north-up UTM transform mirroring the Swimbeach example: world origin
# at a UTM easting/northing, ~0.5 mm/px, negative gt[5] (north-up).
_GT = (707133.4308, 0.0005098, 0.0, 5328463.8146, 0.0, -0.0005098)


def test_pixel_truth_is_reprojected_to_world(tmp_path):
    """A truth CSV in the app's pixel frame (x = col, y = height - row) is
    forward-transformed onto the detection CSV's world coordinates."""
    src = _make_tif(tmp_path, _GT)
    H = 3352
    truth = pd.DataFrame({"x": [800.0, 1248.0, 2741.0],
                          "y": [1409.0, 1234.0, 538.0]})
    out, reprojected = qv.reproject_pixels_to_world(src, truth)
    assert reprojected is True
    # Each point matches the GDAL forward affine of (col, H - y) exactly.
    for i in range(len(truth)):
        col, row = truth["x"][i], H - truth["y"][i]
        wx = _GT[0] + col * _GT[1] + row * _GT[2]
        wy = _GT[3] + col * _GT[4] + row * _GT[5]
        assert out["x"][i] == pytest.approx(wx)
        assert out["y"][i] == pytest.approx(wy)
    # And the result now lands inside the raster's world bounds.
    assert out["x"].min() >= _GT[0] - 1e-6
    assert out["y"].max() <= _GT[3] + 1e-6


def test_world_coords_pass_through_untouched(tmp_path):
    """A detection CSV already in world coords is a no-op (flag False)."""
    src = _make_tif(tmp_path, _GT)
    detect = pd.DataFrame({"x": [707133.8, 707134.9],
                           "y": [5328462.4, 5328463.6]})
    out, reprojected = qv.reproject_pixels_to_world(src, detect)
    assert reprojected is False
    pd.testing.assert_frame_equal(out, detect)


def test_non_georeferenced_image_is_noop(tmp_path):
    """Identity GeoTransform → cannot reproject → return unchanged."""
    src = _make_tif(tmp_path, (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    nx=40, ny=30, proj_epsg=None)
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    out, reprojected = qv.reproject_pixels_to_world(src, df)
    assert reprojected is False
    pd.testing.assert_frame_equal(out, df)


def test_empty_or_missing_columns_is_noop(tmp_path):
    src = _make_tif(tmp_path, _GT)
    empty = pd.DataFrame({"x": [], "y": []})
    out, rep = qv.reproject_pixels_to_world(src, empty)
    assert rep is False and out.empty
    nocols = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out2, rep2 = qv.reproject_pixels_to_world(src, nocols)
    assert rep2 is False


def test_reprojected_truth_falls_inside_geotiff_quadrat(tmp_path):
    """Regression for the Swimbeach bug: pixel-space truth, once
    reprojected, survives the geotiff_aligned footprint clip instead of
    being dropped to zero."""
    src = _make_tif(tmp_path, _GT)
    truth = pd.DataFrame({"x": [800.0, 1248.0, 1893.0],
                          "y": [1409.0, 1234.0, 656.0]})
    # Before the fix: clipping raw pixels to the world footprint = empty.
    fp = qv.quadrat_from_geotiff(src)
    assert len(qv.clip_to_footprint(truth, fp)) == 0
    # After reprojection: all points land inside the quadrat.
    out, rep = qv.reproject_pixels_to_world(src, truth)
    assert rep is True
    assert len(qv.clip_to_footprint(out, fp)) == len(truth)


def test_reprojection_keeps_the_photograph_upright(tmp_path):
    """A clast near the TOP of the photograph (small row, large y in the
    app's y-up pixel frame) must land near the NORTH edge of a north-up
    raster, and one near the bottom near the south edge."""
    src = _make_tif(tmp_path, _GT)
    H = 3352
    df = pd.DataFrame({"x": [100.0, 100.0], "y": [H - 50.0, 50.0]})
    out, rep = qv.reproject_pixels_to_world(src, df)
    assert rep is True
    north, south = _GT[3], _GT[3] + H * _GT[5]
    assert abs(out["y"][0] - north) < abs(out["y"][0] - south)
    assert abs(out["y"][1] - south) < abs(out["y"][1] - north)


def test_row_indexed_csv_from_another_tool(tmp_path):
    """y_up=False reads y as a row index, for CSVs made outside the app."""
    src = _make_tif(tmp_path, _GT)
    df = pd.DataFrame({"x": [800.0], "y": [1409.0]})
    out, rep = qv.reproject_pixels_to_world(src, df, y_up=False)
    assert rep is True
    assert out["y"][0] == pytest.approx(_GT[3] + 1409.0 * _GT[5])


def _rotated_geotiff(path, size=100, angle_deg=40.0, gsd=0.0084):
    """A square quadrat raster placed on the ortho with a rotation, as the
    Georeference tab writes it."""
    import math
    from osgeo import gdal, osr
    ds = gdal.GetDriverByName("GTiff").Create(str(path), size, size, 1, gdal.GDT_Byte)
    a = math.radians(angle_deg)
    x0, y0 = 497976.84, 6960105.24
    # column axis rotated by `angle`, row axis perpendicular (y grows with rows)
    ds.SetGeoTransform((x0, -gsd * math.cos(a), -gsd * math.sin(a),
                        y0, -gsd * math.sin(a), gsd * math.cos(a)))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.full((size, size), 120, np.uint8))
    ds.FlushCache()
    ds = None
    return x0, y0, size * gsd, a


def test_a_rotated_quadrat_footprint_covers_all_four_corners(tmp_path):
    """The Georeference tab writes the placed quadrat as a rotated raster;
    a box from the origin and the far corner alone was a strip a tenth as
    high as the quadrat, and only a sliver of the clasts was compared
."""
    import math
    tif = tmp_path / "Q.tif"
    x0, y0, side, a = _rotated_geotiff(tif)
    fp = qv.quadrat_from_geotiff(str(tif))
    xmin, ymin, xmax, ymax = fp.bounds
    # the box of a square of side `side` rotated by `a`
    expect = side * (abs(math.cos(a)) + abs(math.sin(a)))
    assert xmax - xmin == pytest.approx(expect, rel=1e-6)
    assert ymax - ymin == pytest.approx(expect, rel=1e-6)
    # every corner of the raster lies on the box
    for c, r in ((0, 0), (100, 0), (0, 100), (100, 100)):
        gt = (x0, -0.0084 * math.cos(a), -0.0084 * math.sin(a), y0, -0.0084 * math.sin(a), 0.0084 * math.cos(a))
        x = gt[0] + c * gt[1] + r * gt[2]
        y = gt[3] + c * gt[4] + r * gt[5]
        assert xmin - 1e-9 <= x <= xmax + 1e-9 and ymin - 1e-9 <= y <= ymax + 1e-9


def test_pixel_clasts_of_a_rotated_quadrat_all_fall_inside_its_footprint(tmp_path):
    tif = tmp_path / "Q.tif"
    _rotated_geotiff(tif)
    df = pd.DataFrame({"x": [10.0, 50.0, 90.0, 20.0, 80.0],
                       "y": [10.0, 50.0, 90.0, 85.0, 15.0], "Clast_length": [0.05] * 5})
    world, changed = qv.reproject_pixels_to_world(str(tif), df)
    assert changed
    fp = qv.quadrat_from_geotiff(str(tif))
    inside = qv.clip_to_footprint(world, fp)
    assert len(inside) == 5


def test_a_whole_ortho_world_csv_is_not_taken_for_quadrat_pixels(tmp_path):
    """Most of an ortho lies outside one quadrat, so the 'majority inside
    the raster's bounds' test reprojected a merged world CSV as if it were
    pixels of the quadrat, and none of its 12,475 clasts fell in the
    footprint."""
    tif = tmp_path / "Q.tif"
    _rotated_geotiff(tif)
    # world coordinates over a 21 m ortho around the quadrat
    xs = np.linspace(497974.9, 497995.7, 50)
    ys = np.linspace(6960091.4, 6960112.1, 50)
    df = pd.DataFrame({"x": xs, "y": ys, "Clast_length": [0.05] * 50})
    out, changed = qv.reproject_pixels_to_world(str(tif), df)
    assert changed is False
    assert out["x"].tolist() == xs.tolist()
    # pixel coordinates of the raster itself are still reprojected
    px = pd.DataFrame({"x": [5.0, 50.0, 95.0], "y": [5.0, 50.0, 95.0], "Clast_length": [0.05] * 3})
    out2, changed2 = qv.reproject_pixels_to_world(str(tif), px)
    assert changed2 is True and out2["x"].iloc[0] > 497000
