"""Reading the file a field GPS or a notebook already produces.

The file is written by hand, so the loader is
forgiving about spelling and unforgiving about guessing: a row it cannot parse
is reported with its line number rather than silently dropped or invented.
"""
from __future__ import annotations

import pytest

from functions import seeds as S


def _write(tmp_path, text, name="seeds.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
#  Reading it                                                                  #
# --------------------------------------------------------------------------- #
def test_a_plain_photo_lat_lon_file_loads(tmp_path):
    p = _write(tmp_path, "photo,lat,lon\nDJI_0904.JPG,49.30,-0.35\n")
    t = S.load_seed_table(p)
    assert len(t) == 1 and not t.problems
    s = t.get("DJI_0904.JPG")
    assert s is not None
    assert s.y == pytest.approx(49.30)
    assert s.x == pytest.approx(-0.35), "x must be longitude, not latitude"


@pytest.mark.parametrize("header,row", [
    ("image,longitude,latitude", "q1.jpg,-0.35,49.30"),
    ("filename,easting,northing", "q1.jpg,-0.35,49.30"),
    ("file,x,y", "q1.jpg,-0.35,49.30"),
    ("quadrat,E,N", "q1.jpg,-0.35,49.30"),
    ("name,long,lat", "q1.jpg,-0.35,49.30"),
    ("photograph,long,lat", "q1.jpg,-0.35,49.30"),
])
def test_header_spellings_are_recognised(tmp_path, header, row):
    """Nobody writes this file for our convenience."""
    t = S.load_seed_table(_write(tmp_path, f"{header}\n{row}\n"))
    assert len(t) == 1, t.problems
    s = t.get("q1.jpg")
    assert s.x == pytest.approx(-0.35) and s.y == pytest.approx(49.30)


def test_a_survey_export_is_keyed_by_its_photograph_not_its_label(tmp_path):
    """An RTK rover export names the quadrat (Q25) in 'name' and the file in
    'photograph'; the file is the identity the app looks up, so it must win
    over the label."""
    p = _write(tmp_path,
               "index,name,longitude,latitude,elevation,photograph\n"
               "25,Q25,0.2003103,49.7077467,1.66,IMG_0955.JPG\n"
               "26,Q26,0.2004192,49.7077869,1.83,IMG_0957.JPG\n")
    t = S.load_seed_table(p)
    assert len(t) == 2 and not t.problems
    assert t.columns[0] == "photograph"
    assert t.get("Q25") is None
    s = t.get("IMG_0955_rectified_GSD=0.000548m.jpg")
    assert s is not None and s.photo == "IMG_0955.JPG"
    assert s.x == pytest.approx(0.2003103) and s.y == pytest.approx(49.7077467)


def test_photos_match_with_or_without_extension_and_case(tmp_path):
    t = S.load_seed_table(_write(tmp_path, "photo,lat,lon\nDJI_0904,49.3,-0.35\n"))
    for probe in ("DJI_0904", "DJI_0904.JPG", "dji_0904.jpg",
                  "C:/somewhere/DJI_0904.tif"):
        assert t.get(probe) is not None, probe


def test_a_comma_decimal_separator_is_accepted(tmp_path):
    """A French-locale export writes 49,30."""
    t = S.load_seed_table(_write(tmp_path, "photo;lat;lon\n".replace(";", ",")
                                 + "q1,\"49,30\",\"-0,35\"\n"))
    s = t.get("q1")
    assert s is not None and s.y == pytest.approx(49.30)


def test_a_crs_column_is_used_and_a_bare_epsg_number_understood(tmp_path):
    t = S.load_seed_table(_write(
        tmp_path, "photo,x,y,crs\nq1,558250,6981600,2154\n"))
    assert t.get("q1").crs == "EPSG:2154"


def test_the_default_crs_applies_when_the_file_says_nothing(tmp_path):
    t = S.load_seed_table(_write(tmp_path, "photo,lat,lon\nq1,49.3,-0.35\n"),
                          default_crs="EPSG:4326")
    assert t.get("q1").crs == "EPSG:4326"


# --------------------------------------------------------------------------- #
#  Saying what is wrong with it                                                #
# --------------------------------------------------------------------------- #
def test_a_malformed_row_is_reported_by_line_and_the_rest_still_loads(tmp_path):
    p = _write(tmp_path,
               "photo,lat,lon\n"
               "good1,49.3,-0.35\n"
               "bad,not-a-number,-0.35\n"
               "good2,49.4,-0.36\n")
    t = S.load_seed_table(p)
    assert len(t) == 2, t.problems
    assert any("line 3" in m and "bad" in m for m in t.problems), t.problems


def test_a_missing_column_is_named(tmp_path):
    t = S.load_seed_table(_write(tmp_path, "photo,altitude\nq1,12\n"))
    assert len(t) == 0
    assert any("latitude" in m or "y/" in m for m in t.problems), t.problems


def test_a_duplicate_photo_reports_and_the_last_wins(tmp_path):
    t = S.load_seed_table(_write(
        tmp_path, "photo,lat,lon\nq1,49.3,-0.35\nq1,49.9,-0.99\n"))
    assert t.get("q1").y == pytest.approx(49.9)
    assert any("more than once" in m for m in t.problems), t.problems


def test_an_empty_or_unreadable_file_is_reported_not_raised(tmp_path):
    assert S.load_seed_table(_write(tmp_path, "")).problems
    assert S.load_seed_table(tmp_path / "absent.csv").problems


def test_a_row_with_no_photo_name_is_skipped(tmp_path):
    t = S.load_seed_table(_write(tmp_path, "photo,lat,lon\n,49.3,-0.35\n"))
    assert len(t) == 0
    assert any("no photo name" in m for m in t.problems)


# --------------------------------------------------------------------------- #
#  Pins beat the file                                                          #
# --------------------------------------------------------------------------- #
def test_a_pin_overrides_the_list_for_that_photo(tmp_path):
    """A pin is something the user just placed; a row was typed weeks ago."""
    t = S.load_seed_table(_write(tmp_path, "photo,lat,lon\nq1,49.3,-0.35\n"))
    pins = {"q1": (558250.0, 6981600.0, "EPSG:2154")}
    s = S.seed_for_photo(t, "q1.jpg", pins)
    assert s.x == pytest.approx(558250.0)
    assert s.crs == "EPSG:2154"
    # A photo with no pin still comes from the file.
    assert S.seed_for_photo(t, "q1.jpg", {}).x == pytest.approx(-0.35)


def test_a_photo_with_no_seed_anywhere_is_none(tmp_path):
    t = S.load_seed_table(_write(tmp_path, "photo,lat,lon\nq1,49.3,-0.35\n"))
    assert S.seed_for_photo(t, "q2.jpg", {}) is None
    assert S.seed_for_photo(None, "q2.jpg", {}) is None


# --------------------------------------------------------------------------- #
#  A seed that cannot be right                                                 #
# --------------------------------------------------------------------------- #
def test_a_seed_outside_the_ortho_is_detectable():
    """Latitude and longitude swapped cannot be detected in general, but the
    result lands somewhere impossible — and saying so beats "no match"."""
    bounds = (558250.0, 6981580.0, 558310.0, 6981610.0)
    assert S.outside_footprint(558270.0, 6981600.0, bounds) is False
    assert S.outside_footprint(-0.35, 49.30, bounds) is True
    # Degenerate bounds must not make everything "outside".
    assert S.outside_footprint(1.0, 2.0, None) is False


# --------------------------------------------------------------------------- #
#  Checking a survey                                                           #
# --------------------------------------------------------------------------- #
ORTHO_GSD = 0.005
ORIGIN = (558250.0, 6981600.0)


@pytest.fixture
def survey(tmp_path):
    """A real GeoTIFF ortho plus quadrat photos cut from it."""
    import numpy as np
    from osgeo import gdal, osr
    from PIL import Image
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_georef import _textured, _cut_quadrat

    ortho = _textured(1000, 1000, seed=7)
    tif = tmp_path / "ortho.tif"
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(tif), 1000, 1000, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([ORIGIN[0], ORTHO_GSD, 0.0, ORIGIN[1], 0.0, -ORTHO_GSD])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray((ortho * 255).astype("uint8"))
    ds.FlushCache()
    ds = None

    photos = tmp_path / "photos"
    photos.mkdir()
    # Representative of a real quadrat once resampled (~300 px), not the
    # 80 px toy that made OpenCV's ORB look broken.
    r0, c0, size = 260, 300, 300
    quad = _cut_quadrat(ortho, r0, c0, size)
    Image.fromarray((quad * 255).astype("uint8")).save(photos / "q_good.png")
    alien = _textured(400, 400, seed=4242)
    Image.fromarray((alien * 255).astype("uint8")).save(photos / "q_alien.png")
    seed = (ORIGIN[0] + (c0 + size / 2) * ORTHO_GSD,
            ORIGIN[1] - (r0 + size / 2) * ORTHO_GSD)
    return {"ortho": tif, "photos": photos, "seed": seed, "dir": tmp_path}


def test_a_batch_reports_every_photo_and_does_not_stop_at_a_failure(survey):
    """One quadrat in deep shadow must not hide the other eleven."""
    csvp = survey["dir"] / "seeds.csv"
    sx, sy = survey["seed"]
    csvp.write_text(
        "photo,x,y,crs\n"
        f"q_good,{sx},{sy},EPSG:2154\n"
        f"q_alien,{sx},{sy},EPSG:2154\n", encoding="utf-8")
    table = S.load_seed_table(csvp)
    photos = sorted(survey["photos"].glob("*.png"))
    res = S.check_quadrats(photos, survey["ortho"], table,
                           quadrat_gsd_m=0.001, search_radius_m=1.0)
    assert len(res) == len(photos), "every photo must be reported"
    by = {r.photo: r for r in res}
    assert by["q_good.png"].status == "located", by["q_good.png"].detail
    assert by["q_alien.png"].status == "not_located"
    assert by["q_alien.png"].detail, "a failure must say why"


def test_a_photo_with_no_seed_is_reported_as_such(survey):
    photos = sorted(survey["photos"].glob("*.png"))
    res = S.check_quadrats(photos, survey["ortho"], None, quadrat_gsd_m=0.001)
    assert {r.status for r in res} == {"no_seed"}
    assert all("nowhere to start looking" in r.detail for r in res)


def test_a_seed_outside_the_ortho_is_called_out_not_left_as_no_match(survey):
    """Lat/lon the wrong way round lands exactly like this."""
    csvp = survey["dir"] / "seeds.csv"
    csvp.write_text("photo,lat,lon\nq_good,49.30,-0.35\n", encoding="utf-8")
    table = S.load_seed_table(csvp, default_crs="EPSG:4326")
    res = S.check_quadrats([survey["photos"] / "q_good.png"], survey["ortho"],
                           table, quadrat_gsd_m=0.001)
    assert res[0].status == "not_located"
    assert "outside this ortho" in res[0].detail, res[0].detail


def test_a_batch_writes_nothing(survey):
    """Checking is safe to batch; writing stays deliberate."""
    before = sorted(p.name for p in survey["dir"].rglob("*"))
    csvp = survey["dir"] / "seeds.csv"
    sx, sy = survey["seed"]
    csvp.write_text(f"photo,x,y,crs\nq_good,{sx},{sy},EPSG:2154\n",
                    encoding="utf-8")
    table = S.load_seed_table(csvp)
    S.check_quadrats(sorted(survey["photos"].glob("*.png")), survey["ortho"],
                     table, quadrat_gsd_m=0.001, search_radius_m=1.0)
    after = sorted(p.name for p in survey["dir"].rglob("*"))
    assert after == sorted(before + ["seeds.csv"])


# --------------------------------------------------------------------------- #
#  A pin must land where it was clicked                                        #
# --------------------------------------------------------------------------- #
def _click_to_world(image_x, image_y, disp_scale, gt):
    """The conversion the pin canvas performs, isolated.

    Mirrors gui/app.py's _on_pin_click: displayed pixel -> original raster
    pixel -> world. A silent error here would place every pin at an offset
    and simply make matching fail for reasons nobody could see.
    """
    ox = float(image_x) * disp_scale
    oy = float(image_y) * disp_scale
    return (gt[0] + ox * gt[1] + oy * gt[2],
            gt[3] + ox * gt[4] + oy * gt[5])


def _world_to_click(wx, wy, disp_scale, gt):
    """And back, as the marker drawing does."""
    return ((wx - gt[0]) / gt[1] / disp_scale,
            (wy - gt[3]) / gt[5] / disp_scale)


@pytest.mark.parametrize("disp_scale", [1.0, 2.0, 6.5])
def test_a_pin_round_trips_through_the_display_scale(disp_scale):
    """A downsampled preview must not shift the pin."""
    gt = (558250.0, 0.005, 0.0, 6981600.0, 0.0, -0.005)
    for cx, cy in ((0.0, 0.0), (37.0, 12.0), (100.0, 250.0)):
        wx, wy = _click_to_world(cx, cy, disp_scale, gt)
        bx, by = _world_to_click(wx, wy, disp_scale, gt)
        assert bx == pytest.approx(cx)
        assert by == pytest.approx(cy)


def test_the_origin_click_lands_on_the_orthos_origin():
    gt = (558250.0, 0.005, 0.0, 6981600.0, 0.0, -0.005)
    wx, wy = _click_to_world(0, 0, 1.0, gt)
    assert (wx, wy) == pytest.approx((558250.0, 6981600.0))


def test_northing_decreases_as_the_click_moves_down():
    """A north-up raster has a negative row step; getting the sign wrong
    mirrors every pin about the top edge."""
    gt = (558250.0, 0.005, 0.0, 6981600.0, 0.0, -0.005)
    _, y_top = _click_to_world(0, 0, 1.0, gt)
    _, y_low = _click_to_world(0, 100, 1.0, gt)
    assert y_low < y_top


def test_a_pin_reaches_the_matcher_in_preference_to_the_list(tmp_path):
    """End of the chain: a pin placed on the ortho is what gets searched."""
    p = tmp_path / "seeds.csv"
    p.write_text("photo,x,y,crs\nq1,558999,6981999,EPSG:2154\n",
                 encoding="utf-8")
    table = S.load_seed_table(p)
    gt = (558250.0, 0.005, 0.0, 6981600.0, 0.0, -0.005)
    wx, wy = _click_to_world(120, 80, 2.0, gt)
    pins = {S.normalise_photo("q1.png"): (wx, wy, "EPSG:2154")}
    used = S.seed_for_photo(table, "q1.png", pins)
    assert used.x == pytest.approx(wx)
    assert used.y == pytest.approx(wy)
    assert used.x != pytest.approx(558999.0)


# --------------------------------------------------------------------------- #
#  The batch resolves seeds the way the single-quadrat path does               #
# --------------------------------------------------------------------------- #
#
# `check_quadrats` stopped at pin -> seed list, while the Georeference tab's
# single-quadrat path had already gained a fourth source: the photograph's own
# GPS fix. So a survey whose photographs carry a fix came back "no seed given"
# and was never searched. Measured on Swimbeach: ten of twenty quadrats, whose
# originals sat one directory across carrying a fix good to 4-7 m.
#
def _gps_original(directory, name, lat, lon):
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational
    import numpy as np
    p = directory / name
    Image.fromarray(np.zeros((16, 16, 3), "uint8")).save(str(p))

    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        s = (v - d - m / 60.0) * 3600.0
        return (IFDRational(d, 1), IFDRational(m, 1),
                IFDRational(int(round(s * 10000)), 10000))
    ex = Image.Exif()
    ex[34853] = {1: "N" if lat >= 0 else "S", 2: dms(lat),
                 3: "E" if lon >= 0 else "W", 4: dms(lon)}
    with Image.open(str(p)) as im:
        im.save(str(p), exif=ex.tobytes())
    return p


def test_the_batch_falls_back_to_the_photographs_own_fix(survey, tmp_path):
    """No seed list, no pins -- and it is still searched, because the original
    beside it says roughly where it was."""
    from osgeo import osr
    import functions.georef as G

    # the quadrat's true position, expressed as lon/lat for the original's EXIF
    sx, sy = survey["seed"]
    src = osr.SpatialReference(); src.ImportFromEPSG(2154)
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lon, lat, _ = osr.CoordinateTransformation(src, dst).TransformPoint(sx, sy)

    raw = survey["dir"] / "raw"          # a SIBLING of photos/, as a project keeps it
    raw.mkdir()
    _gps_original(raw, "q_good.png", lat, lon)

    good = survey["photos"] / "q_good.png"
    res = S.check_quadrats([good], survey["ortho"], None,   # table=None, pins=None
                           quadrat_gsd_m=0.001, search_radius_m=2.0)
    assert res[0].status != "no_seed", res[0].detail
    assert res[0].seed_source == "photo"


def test_a_seed_list_still_wins_over_the_photographs_fix(survey):
    """Precedence is unchanged: what a person supplied beats what a camera
    guessed, and the batch must agree with the single-quadrat path on that."""
    csvp = survey["dir"] / "seeds.csv"
    sx, sy = survey["seed"]
    csvp.write_text(f"photo,x,y,crs\nq_good,{sx},{sy},EPSG:2154\n",
                    encoding="utf-8")
    table = S.load_seed_table(csvp)
    res = S.check_quadrats([survey["photos"] / "q_good.png"], survey["ortho"],
                           table, quadrat_gsd_m=0.001, search_radius_m=1.0)
    assert res[0].seed_source == "list"


def test_no_seed_anywhere_still_says_so(survey):
    """The message has to name the third option now that there is one."""
    res = S.check_quadrats(sorted(survey["photos"].glob("*.png")),
                           survey["ortho"], None, quadrat_gsd_m=0.001)
    assert {r.status for r in res} == {"no_seed"}
    assert all("nowhere to start looking" in r.detail for r in res)
    assert any("raw/" in r.detail for r in res)


# --------------------------------------------------------------------------- #
#  A seed list survives its photographs being rectified again                  #
# --------------------------------------------------------------------------- #
def test_a_seed_row_still_matches_after_re_rectification(tmp_path):
    """This tool writes its own parameters into filenames, so re-rectifying at
    a corrected scale renames every file. A survey re-run at the right quadrat
    size lost its whole seed list that way, and the only visible symptom was
    seeds quietly arriving from somewhere else."""
    p = tmp_path / "s.csv"
    p.write_text(
        "photo,x,y\n"
        "DJI_0904_rectified_GSD=0.000641m.png,-114.219,48.0758\n",
        encoding="utf-8")
    t = S.load_seed_table(p, "EPSG:4326")
    # the same quadrat, rectified again at the corrected size
    assert t.get("DJI_0904_rectified_GSD=0.00076m.jpg") is not None
    # and the older convention, and the bare name
    assert t.get("DJI_0904_corrected_width=1.185m_GSD=0.0005m_per_px.jpg")
    assert t.get("DJI_0904.JPG") is not None


def test_an_exact_name_still_wins_over_a_base_match(tmp_path):
    """Two rectifications of one photograph can carry different coordinates if
    somebody edited the list; the exact row must not be shadowed."""
    p = tmp_path / "s.csv"
    p.write_text("photo,x,y\n"
                 "DJI_0904_rectified_GSD=0.001m.png,1.0,2.0\n"
                 "DJI_0904_rectified_GSD=0.002m.png,3.0,4.0\n",
                 encoding="utf-8")
    t = S.load_seed_table(p, "EPSG:4326")
    assert t.get("DJI_0904_rectified_GSD=0.002m.png").x == 3.0


def test_a_different_quadrat_is_not_matched_by_accident(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("photo,x,y\nDJI_0904_rectified_GSD=0.001m.png,1.0,2.0\n",
                 encoding="utf-8")
    t = S.load_seed_table(p, "EPSG:4326")
    assert t.get("DJI_0906_rectified_GSD=0.001m.png") is None
    assert t.get("S1_IMG_6738_rectified_GSD=0.001m.png") is None
