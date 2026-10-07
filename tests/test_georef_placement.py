"""Digitize's 'is this photograph placed?' answer: the file's own
georeference, else a placement saved under validation/georectified."""
from __future__ import annotations

import numpy as np
import pytest

from functions import georef as G


def _plain_jpeg(path):
    from PIL import Image
    Image.fromarray(np.full((20, 20, 3), 120, np.uint8)).save(path, "JPEG")


def _geotiff(path):
    from osgeo import gdal, osr
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 20, 20, 3, gdal.GDT_Byte)
    ds.SetGeoTransform((497970.0, 0.0005, 0, 6960100.0, 0, -0.0005))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(1, 4):
        ds.GetRasterBand(b).WriteArray(np.full((20, 20), 120, np.uint8))
    ds.FlushCache()
    ds = None


def test_a_rectified_jpeg_with_no_saved_placement_still_needs_placing(tmp_path):
    photo = tmp_path / "orthorectified" / "IMG_0955_rectified_GSD=0.000567m.jpg"
    photo.parent.mkdir()
    _plain_jpeg(photo)
    info = G.describe_placement(str(photo), tmp_path / "georectified")
    assert info["georeferenced"] is False and info["placed_path"] is None
    assert "must be matched" in info["reason"]


def test_a_saved_placement_counts_as_placed(tmp_path):
    photo = tmp_path / "orthorectified" / "IMG_0955_rectified_GSD=0.000567m.jpg"
    photo.parent.mkdir()
    _plain_jpeg(photo)
    saved = tmp_path / "georectified" / "IMG_0955_rectified_GSD=0.000567m.tif"
    saved.parent.mkdir()
    _geotiff(saved)
    info = G.describe_placement(str(photo), None, tmp_path / "georectified")
    assert info["georeferenced"] is True
    assert info["placed_path"] == saved
    assert info["gsd_m"] == pytest.approx(0.0005)
    assert "georectified/IMG_0955_rectified_GSD=0.000567m.tif" in info["reason"]


def test_another_quadrats_placement_does_not_count(tmp_path):
    photo = tmp_path / "orthorectified" / "IMG_0957_rectified_GSD=0.000551m.jpg"
    photo.parent.mkdir()
    _plain_jpeg(photo)
    saved = tmp_path / "georectified" / "IMG_0955_rectified_GSD=0.000567m.tif"
    saved.parent.mkdir()
    _geotiff(saved)
    info = G.describe_placement(str(photo), tmp_path / "georectified")
    assert info["georeferenced"] is False


def test_a_georeferenced_file_answers_for_itself(tmp_path):
    tif = tmp_path / "IMG_0955.tif"
    _geotiff(tif)
    info = G.describe_placement(str(tif), tmp_path / "nowhere")
    assert info["georeferenced"] is True and info["placed_path"] is None


def test_the_clasts_csv_is_found_where_detect_writes_it(tmp_path):
    """The georeference save looked for <stem>_individual_clasts.csv beside
    the photograph only, so in a project no placement ever carried its
    world-coordinate CSV."""
    photo = tmp_path / "P" / "validation" / "orthorectified" / "IMG_0955_rectified_GSD=0.000567m.jpg"
    photo.parent.mkdir(parents=True)
    _plain_jpeg(photo)
    vectors = tmp_path / "P" / "output_results" / "vectors"
    vectors.mkdir(parents=True)
    assert G.find_clasts_csv(str(photo), vectors_dir=vectors, project="P") is None
    canon = vectors / "P__IMG_0955_rectified_GSD=0.000567m_individual_clasts.csv"
    canon.write_text("clast_ID,x,y\n1,1,1\n", encoding="utf-8")
    assert G.find_clasts_csv(str(photo), vectors_dir=vectors, project="P") == canon
    beside = photo.with_name("IMG_0955_rectified_GSD=0.000567m_individual_clasts.csv")
    beside.write_text("clast_ID,x,y\n1,1,1\n", encoding="utf-8")
    assert G.find_clasts_csv(str(photo), vectors_dir=vectors, project="P") == beside


def test_a_rotated_rasters_pixel_size_is_its_column_step(tmp_path):
    """A placed quadrat is a rotated raster; its x-scale term alone read
    0.43 mm for a 0.57 mm pixel."""
    import math
    from osgeo import gdal, osr
    tif = tmp_path / "rot.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(tif), 10, 10, 1, gdal.GDT_Byte)
    a = math.radians(40.0)
    ds.SetGeoTransform((497976.84, -0.000567 * math.cos(a), -0.000567 * math.sin(a),
                        6960105.24, -0.000567 * math.sin(a), 0.000567 * math.cos(a)))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.full((10, 10), 120, np.uint8))
    ds.FlushCache()
    ds = None
    info = G.describe_georeferencing(str(tif))
    assert info["georeferenced"] and info["gsd_m"] == pytest.approx(0.000567, rel=1e-6)
    from functions.quadrat_validation import detect_gsd_from_path
    assert detect_gsd_from_path(str(tif))["gsd"] == pytest.approx(0.000567, rel=1e-6)
