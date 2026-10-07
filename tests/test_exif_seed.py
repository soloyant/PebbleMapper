"""Reading the coordinate a photograph already carries.

These are self-contained: images are written here with Pillow rather than
read from the 13 GB of field data outside the repository. The real
measurement the feature rests on -- 21 Bio_Station raws, median 1.69 m from
the surveyed reference, 19 of 21 within 5 m -- is recorded in the module
docstring; what is tested here is that the function behaves the way that
measurement describes.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from functions import exif_seed as X


def _img(tmp, name="a.jpg", size=(32, 32)):
    p = tmp / name
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype("uint8")
                    ).save(str(p))
    return p


def _with_gps(tmp, lat, lon, name="g.jpg", lat_ref=None, lon_ref=None,
              hpe=None, alt=None):
    p = _img(tmp, name)
    with Image.open(str(p)) as im:
        im.load()
        exif = im.getexif()

        def dms(v):
            v = abs(float(v))
            d = int(v)
            m = int((v - d) * 60)
            s = (v - d - m / 60.0) * 3600.0
            return (IFDRational(d, 1), IFDRational(m, 1),
                    IFDRational(int(round(s * 10000)), 10000))
        gps = {1: lat_ref or ("N" if lat >= 0 else "S"), 2: dms(lat),
               3: lon_ref or ("E" if lon >= 0 else "W"), 4: dms(lon)}
        if hpe is not None:
            gps[31] = IFDRational(int(round(hpe * 100)), 100)
        if alt is not None:
            gps[5] = 0
            gps[6] = IFDRational(int(round(alt * 100)), 100)
        exif[X._GPS_IFD] = gps
        im.save(str(p), exif=exif.tobytes())
    return p


# --------------------------------------------------------------------------- #
#  Requirements 1-2 — reading a fix, with the right sign                       #
# --------------------------------------------------------------------------- #
def test_a_fix_round_trips(tmp_path):
    p = _with_gps(tmp_path, 47.876494, -114.034942)
    got = X.gps_from_image(p)
    assert got["lat"] == pytest.approx(47.876494, abs=1e-6)
    assert got["lon"] == pytest.approx(-114.034942, abs=1e-6)


def test_west_and_south_come_back_negative(tmp_path):
    """Getting the hemisphere wrong puts the search on the wrong continent
    while looking entirely reasonable."""
    w = X.gps_from_image(_with_gps(tmp_path, 48.0, -114.0, "w.jpg"))
    assert w["lon"] < 0 and w["lat"] > 0
    s = X.gps_from_image(_with_gps(tmp_path, -33.9, 18.4, "s.jpg"))
    assert s["lat"] < 0 and s["lon"] > 0


def test_the_tolerance_is_metre_scale_not_degree_scale(tmp_path):
    """0.001 degrees is ~111 m -- meaningless for a feature whose whole premise
    is a 1.7 m median."""
    p = _with_gps(tmp_path, 47.876494, -114.034942, "t.jpg")
    got = X.gps_from_image(p)
    assert abs(got["lat"] - 47.876494) < 1e-6
    assert abs(got["lon"] + 114.034942) < 1e-6


# --------------------------------------------------------------------------- #
#  Edge cases — every way a photograph declines to say where it was            #
# --------------------------------------------------------------------------- #
def test_no_exif_block_at_all(tmp_path):
    """Today's corrected images: rectification strips the block entirely."""
    assert X.gps_from_image(_img(tmp_path, "plain.jpg")) is None


def test_exif_present_but_no_gps(tmp_path):
    p = _img(tmp_path, "nogps.jpg")
    with Image.open(str(p)) as im:
        im.load()
        exif = im.getexif()
        exif[271] = "SomeCamera"
        im.save(str(p), exif=exif.tobytes())
    assert X.gps_from_image(p) is None


def test_null_island_is_a_missing_fix(tmp_path):
    """(0, 0) is a receiver that never locked, not a position in the Gulf of
    Guinea."""
    assert X.gps_from_image(_with_gps(tmp_path, 0.0, 0.0, "z.jpg")) is None


def test_a_missing_file_is_not_an_error(tmp_path):
    assert X.gps_from_image(tmp_path / "nope.jpg") is None
    assert X.gps_from_image("") is None


def test_altitude_is_read_but_never_required(tmp_path):
    got = X.gps_from_image(_with_gps(tmp_path, 48.0, -114.0, "alt.jpg",
                                     alt=829.3))
    assert got["alt"] == pytest.approx(829.3, abs=0.05)


# --------------------------------------------------------------------------- #
#  Requirement 10 — the reported accuracy is informative, not authoritative    #
# --------------------------------------------------------------------------- #
def test_a_wildly_pessimistic_accuracy_tag_refuses_the_fix(tmp_path):
    p = _with_gps(tmp_path, 48.0, -114.0, "bad.jpg", hpe=45.0)
    fix, why = X.seed_from_photo(p)
    assert fix is None and "45" in why


def test_an_ordinary_accuracy_tag_does_not(tmp_path):
    """The tag overstates about sixfold -- median 10 m reported against 1.7 m
    true -- so refusing anything worse than the search radius would discard
    most of the usable fixes."""
    p = _with_gps(tmp_path, 48.0, -114.0, "ok.jpg", hpe=20.0)
    fix, why = X.seed_from_photo(p)
    assert fix is not None and why == ""


def test_no_accuracy_tag_is_not_a_refusal(tmp_path):
    """The DJI files carry none at all."""
    fix, why = X.seed_from_photo(_with_gps(tmp_path, 48.0, -114.0, "none.jpg"))
    assert fix is not None and why == ""


# --------------------------------------------------------------------------- #
#  Requirement 8 — a fix that cannot be where the quadrat was                  #
# --------------------------------------------------------------------------- #
def test_a_fix_far_outside_the_ortho_is_refused(tmp_path):
    """One of the 21 Bio_Station fixes lands 31 m outside a 45 x 136 m ortho.
    Seeded there the search fails for reasons the user cannot see."""
    p = _with_gps(tmp_path, 48.0, -114.0, "far.jpg")
    fix, why = X.seed_from_photo(
        p, bounds=(0.0, 0.0, 100.0, 100.0),
        crs_transform=lambda lon, lat: (500.0, 500.0))
    assert fix is None and "outside this ortho-image" in why


def test_a_fix_inside_the_ortho_is_kept(tmp_path):
    p = _with_gps(tmp_path, 48.0, -114.0, "near.jpg")
    fix, why = X.seed_from_photo(
        p, bounds=(0.0, 0.0, 100.0, 100.0),
        crs_transform=lambda lon, lat: (50.0, 50.0))
    assert fix is not None and why == ""


def test_a_fix_just_outside_is_kept_because_a_poor_seed_is_still_a_seed(tmp_path):
    """A fix a little outside the footprint is a poor seed, not an impossible
    one; only a fix that cannot be in this ortho at all is refused here."""
    p = _with_gps(tmp_path, 48.0, -114.0, "edge.jpg")
    fix, _ = X.seed_from_photo(
        p, bounds=(0.0, 0.0, 100.0, 100.0),
        crs_transform=lambda lon, lat: (105.0, 50.0))     # 5 m out
    assert fix is not None


def test_the_bounds_refusal_is_reported_ahead_of_the_accuracy_one(tmp_path):
    """The worst Bio_Station fix trips both gates -- 65 m reported accuracy and
    31 m outside the ortho -- so whichever runs first is the only reason the
    user ever sees. "Check the photograph belongs to this survey" is the one
    they can act on."""
    p = _with_gps(tmp_path, 48.0, -114.0, "both.jpg", hpe=65.0)
    fix, why = X.seed_from_photo(
        p, bounds=(0.0, 0.0, 100.0, 100.0),
        crs_transform=lambda lon, lat: (500.0, 500.0))
    assert fix is None
    assert "outside this ortho-image" in why
    # and with no ortho to check against, the accuracy gate still refuses it
    assert X.seed_from_photo(p)[0] is None


# --------------------------------------------------------------------------- #
#  Requirement 3 — finding the raw photograph behind a rectified one           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rect_name", [
    "01_rectified.png",
    "01_corrected_width=1m_height=1m_GSD=0.001m_per_px.jpg",
])
def test_the_raw_sibling_is_found_through_the_tools_own_suffixes(tmp_path,
                                                                 rect_name):
    """A plain stem match does not work: three naming conventions are live and
    neither rectified form is the stem of the raw one."""
    raw = _with_gps(tmp_path, 48.0, -114.0, "01.png")
    rect = _img(tmp_path, rect_name)
    assert X.raw_sibling(rect) == raw
    fix, why = X.seed_from_photo(rect)
    assert fix is not None and fix["source"].endswith("01.png")


def test_a_rectified_image_with_no_raw_sibling_says_so(tmp_path):
    rect = _img(tmp_path, "99_rectified.png")
    fix, why = X.seed_from_photo(rect)
    assert fix is None and "No position" in why


# --------------------------------------------------------------------------- #
#  Requirements 11-14, 17 — carrying it forward                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["out.png", "out.jpg"])
def test_metadata_round_trips_for_every_format_this_tool_writes(tmp_path, name):
    """PNG especially: it is the Orthorectify tab's default output, and GDAL's
    PNG driver would have written a detachable .aux.xml instead of embedding."""
    p = _img(tmp_path, name)
    rec = X.rectification_record(source="01.png", gsd="0.001",
                                 corners=[(1.0, 2.0), (3.0, 4.0)])
    rec.update({"lat": 48.0, "lon": -114.0})
    assert X.write_placement_metadata(p, rec)

    back = X.read_rectification_record(p)
    assert back.get("PM_RECTIFIED") == "yes"
    assert back.get("PM_SOURCE") == "01.png"
    assert "PM_RECTIFIED_ON" in back

    fix = X.gps_from_image(p)
    assert fix is not None
    assert fix["lat"] == pytest.approx(48.0, abs=1e-5)
    assert fix["lon"] == pytest.approx(-114.0, abs=1e-5)


def test_no_geotransform_is_written(tmp_path):
    """Recording the GSD as a geotransform would make describe_georeferencing
    report the output as already placed, and the Georeference tab binds the
    whole 'Locate one quadrat' panel to that -- every newly rectified quadrat
    would silently lose the controls it needs."""
    from functions import georef as G
    p = _img(tmp_path, "geo.png")
    rec = X.rectification_record(source="01.png", gsd="0.001")
    rec.update({"lat": 48.0, "lon": -114.0})
    X.write_placement_metadata(p, rec)
    assert G.describe_georeferencing(str(p))["georeferenced"] is False


def test_a_stale_pam_sidecar_is_removed(tmp_path):
    """cv2.imwrite knows nothing about GDAL's sidecars, so re-running over an
    existing path left the PREVIOUS quadrat's position readable beside the new
    image -- a confidently wrong seed."""
    p = _img(tmp_path, "s.png")
    aux = p.with_name(p.name + ".aux.xml")
    aux.write_text("<PAMDataset><Metadata><MDI key='SOURCE_GPS_LON'>"
                   "-114.9</MDI></Metadata></PAMDataset>", encoding="utf-8")
    assert aux.is_file()
    X.write_placement_metadata(p, {"lat": 48.0, "lon": -114.0})
    assert not aux.exists()


def test_an_image_without_the_new_metadata_behaves_as_before(tmp_path):
    p = _img(tmp_path, "old.png")
    assert X.read_rectification_record(p) == {}
    assert X.gps_from_image(p) is None


# --------------------------------------------------------------------------- #
#  Requirements 4-6 — the fallback order, and what "not supplied" means        #
# --------------------------------------------------------------------------- #
#
# These call `seeds.resolve_seed` itself. An earlier draft re-implemented the
# rule here and asserted against the copy, which is the one arrangement that
# cannot catch the GUI and the rule disagreeing -- and they did disagree: the
# save path read the two coordinate boxes directly, so a quadrat seeded from a
# pin could be located and then never written.
#
def _pin_photo(tmp):
    return _with_gps(tmp, 48.0, -114.0, "01.png")


def _table(tmp, x=2.0, y=2.0):
    from functions import seeds as S
    p = tmp / "seeds.csv"
    p.write_text(f"photo,x,y\n01.png,{x},{y}\n", encoding="utf-8")
    return S.load_seed_table(p, "EPSG:32612")


def test_the_pin_wins_and_each_source_falls_through_in_order(tmp_path):
    from functions import seeds as S
    photo = _pin_photo(tmp_path)
    table = _table(tmp_path)
    pins = {S.normalise_photo("01.png"): (1.0, 1.0, "EPSG:32612")}

    r = S.resolve_seed(photo, table=table, pins=pins, typed=(3.0, 3.0))
    assert (r.x, r.y, r.provenance) == (1.0, 1.0, "a pin")

    r = S.resolve_seed(photo, table=table, pins=None, typed=(3.0, 3.0))
    assert (r.x, r.y, r.provenance) == (2.0, 2.0, "the seed list")

    r = S.resolve_seed(photo, table=None, pins=None, typed=(3.0, 3.0))
    assert (r.x, r.y, r.provenance) == (3.0, 3.0, "the coordinates you typed")

    r = S.resolve_seed(photo, table=None, pins=None, typed=(0.0, 0.0))
    assert r.provenance.startswith("the photograph's own GPS fix")
    assert (r.x, r.y, r.crs) == (-114.0, 48.0, "EPSG:4326")

    r = S.resolve_seed(_img(tmp_path, "nofix.png"), typed=(0.0, 0.0))
    assert r.x is None and r.ok is False


@pytest.mark.parametrize("typed", [(0.0, 0.0), (None, None)])
def test_an_unsupplied_typed_coordinate_does_not_beat_the_photograph(tmp_path,
                                                                     typed):
    """The two coordinate boxes default to 0.0, so read naively a typed
    coordinate is ALWAYS present and the EXIF branch can never run. A cleared
    NiceGUI number field is None, which is the second encoding of "empty"."""
    from functions import seeds as S
    r = S.resolve_seed(_pin_photo(tmp_path), typed=typed)
    assert r.provenance.startswith("the photograph's own GPS fix")
    assert (r.x, r.y) == (-114.0, 48.0)


def test_a_real_typed_coordinate_still_beats_the_photograph(tmp_path):
    from functions import seeds as S
    r = S.resolve_seed(_pin_photo(tmp_path), typed=(558260.0, 6981590.0),
                       typed_crs="EPSG:32612")
    assert (r.x, r.y) == (558260.0, 6981590.0)
    assert r.provenance == "the coordinates you typed"
    assert r.crs == "EPSG:32612"


def test_a_refused_seed_is_not_reported_as_one(tmp_path):
    """`ok` is what the tab keys its provenance line off. A refusal that came
    back looking supplied would print "Seed from None, None"."""
    from functions import seeds as S
    r = S.resolve_seed(_img(tmp_path, "blank.png"), typed=(None, None))
    assert r.ok is False and r.x is None
    assert "No position" in r.provenance


def test_the_search_radius_for_a_photograph_seed_covers_the_measured_spread():
    """The worst fix that survives HPE_LIMIT_M is 3.76 m out; the two worse
    than 5 m are refused before the radius is consulted."""
    assert X.EXIF_SEED_RADIUS_M >= 2 * 3.76


# --------------------------------------------------------------------------- #
#  Not destroying the image the metadata is attached to                        #
# --------------------------------------------------------------------------- #
#
# Every fixture below is written with cv2, as the Orthorectify tab writes it.
# The Pillow-written fixtures these replaced certified a write path that
# truncated a real .tif to a 229-byte husk: cv2 writes TIFF LZW-compressed,
# which routes Pillow's re-save through libtiff, which rejects the GPS IFD
# after `save` has already opened the file for truncation.
#
def _cv2_img(tmp, name, size=(48, 48)):
    import cv2
    p = tmp / name
    rng = np.random.default_rng(3)
    a = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
    assert cv2.imwrite(str(p), a)
    return p


@pytest.mark.parametrize("name", ["o.png", "o.jpg", "o.tif", "o.bmp", "o.webp"])
def test_the_pixels_are_never_changed_by_attaching_metadata(tmp_path, name):
    """Not "usually lossless" -- identical. A rectified quadrat is measurement
    data, and a re-compression here moves 97% of its pixels before a single
    clast is detected."""
    import cv2
    p = _cv2_img(tmp_path, name)
    before = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    n_before = p.stat().st_size
    ok = X.write_placement_metadata(p, {"lat": 48.0, "lon": -114.0})
    after = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    assert after is not None, f"{name} was destroyed by the metadata write"
    assert after.shape == before.shape
    np.testing.assert_array_equal(after, before)
    if not ok:
        # A format that cannot carry the block is left exactly as it was.
        assert p.stat().st_size == n_before


@pytest.mark.parametrize("name", ["o.png", "o.jpg"])
def test_the_formats_the_tab_writes_do_carry_the_fix(tmp_path, name):
    p = _cv2_img(tmp_path, name)
    assert X.write_placement_metadata(p, {"lat": 48.0, "lon": -114.0})
    fix = X.gps_from_image(p)
    assert fix is not None
    assert fix["lat"] == pytest.approx(48.0, abs=1e-5)


def test_a_format_that_cannot_carry_the_block_reports_failure(tmp_path):
    """Not silence, and not a false True. The tab tells the user the fix was
    carried forward; a BMP accepts the save and drops the block."""
    p = _cv2_img(tmp_path, "o.bmp")
    assert X.write_placement_metadata(p, {"lat": 48.0, "lon": -114.0}) is False
    assert X.gps_from_image(p) is None


def test_attaching_twice_does_not_accumulate_blocks(tmp_path):
    p = _cv2_img(tmp_path, "twice.png")
    X.write_placement_metadata(p, {"lat": 48.0, "lon": -114.0})
    n1 = p.stat().st_size
    X.write_placement_metadata(p, {"lat": 49.0, "lon": -113.0})
    assert p.stat().st_size == pytest.approx(n1, abs=64)
    assert X.gps_from_image(p)["lat"] == pytest.approx(49.0, abs=1e-5)


def test_the_stale_sidecar_goes_even_when_the_metadata_write_fails(tmp_path):
    """Requirement 15 exists for the case where the image was written and the
    block was not: the sidecar then describes the PREVIOUS quadrat, beside a
    brand-new image."""
    p = _cv2_img(tmp_path, "o.tif")
    aux = p.with_name(p.name + ".aux.xml")
    aux.write_text("<PAMDataset/>", encoding="utf-8")
    assert X.write_placement_metadata(p, {"lat": 48.0, "lon": -114.0}) is False
    assert not aux.exists()


def test_all_four_corners_survive_the_round_trip(tmp_path):
    """The record's fields are joined with ';', so corners joined the same way
    read back as one corner of four."""
    p = _cv2_img(tmp_path, "c.png")
    rec = X.rectification_record(
        source="01.png", gsd="0.001",
        corners=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)])
    assert X.write_placement_metadata(p, rec)
    got = X.read_rectification_record(p)["PM_CORNERS"]
    assert len(got.split("|")) == 4
    assert got.split("|")[3] == "7.0000,8.0000"


# --------------------------------------------------------------------------- #
#  Requirement 16 — never over a raw photograph                                #
# --------------------------------------------------------------------------- #
def test_a_windows_path_alias_does_not_defeat_the_source_guard(tmp_path):
    """`Path.resolve()` keeps a `\\\\?\\C:\\...` prefix verbatim, so comparing
    the two strings says they differ while the OS opens one file."""
    p = _cv2_img(tmp_path, "01.png")
    assert X.same_file(p, str(p))
    assert X.same_file("\\\\?\\" + str(p.resolve()), p)
    assert X.same_file(str(p).replace("\\", "/"), p)
    assert not X.same_file(p, _cv2_img(tmp_path, "02.png"))


def test_a_neighbouring_raw_photograph_is_recognised(tmp_path):
    """The guard protected only the source, and the output defaults into the
    directory where all 21 quadrats sit together -- so `01.png` rectified with
    the output typed `02.png` destroyed a quadrat that cannot be re-taken."""
    raw = _with_gps(tmp_path, 48.0, -114.0, "02.png")
    assert X.looks_like_raw_photograph(raw)

    ours = _cv2_img(tmp_path, "02_rectified.png")
    rec = X.rectification_record(source="02.png", gsd="0.001")
    rec.update({"lat": 48.0, "lon": -114.0})
    X.write_placement_metadata(ours, rec)
    # carries a fix too, but says where it came from
    assert X.looks_like_raw_photograph(ours) is False
    assert X.looks_like_raw_photograph(_cv2_img(tmp_path, "plain.png")) is False
    assert X.looks_like_raw_photograph(tmp_path / "nothing.png") is False


# --------------------------------------------------------------------------- #
#  Finding the original when it is one directory across                        #
# --------------------------------------------------------------------------- #
#
# A project keeps `validation/raw/` beside `validation/rectified/`. Searching
# only the folder the user selected, plus the file's own parent, therefore
# finds nothing -- and on the Swimbeach survey that was ten of twenty quadrats
# reporting "no seed given" while their raws, one directory across, carried a
# fix good to 4-7 m.
#
def _survey_layout(tmp_path, raw_dir_name="raw"):
    rect = tmp_path / "rectified"
    raw = tmp_path / raw_dir_name
    rect.mkdir()
    raw.mkdir()
    stem = "S1_IMG_6738"
    original = _with_gps(raw, 48.0757, -114.2190, f"{stem}.jpg")
    rectified = _img(
        rect, f"{stem}_corrected_width=1.185m_height=1.185m_GSD=0.0005m_per_px.jpg")
    return rectified, original


def test_the_original_is_found_in_a_sibling_raw_folder(tmp_path):
    rectified, original = _survey_layout(tmp_path)
    # the folder the user actually selects is the rectified one
    assert X.raw_sibling(rectified, [rectified.parent]) == original
    fix, why = X.seed_from_photo(rectified, search_dirs=[rectified.parent])
    assert fix is not None, why
    assert fix["lat"] == pytest.approx(48.0757, abs=1e-4)
    assert fix["source"].endswith(f"{original.name}")


@pytest.mark.parametrize("name", ["raw", "RAW", "originals", "source"])
def test_the_conventional_folder_names_are_all_recognised(tmp_path, name):
    rectified, original = _survey_layout(tmp_path, raw_dir_name=name)
    assert X.raw_sibling(rectified, [rectified.parent]) == original


def test_an_unrelated_sibling_folder_is_not_raided(tmp_path):
    """Only folders named for originals. A sibling `georectified/` holds this
    tool's own output, and matching a stem there would seed a quadrat from a
    file it already produced."""
    rectified, original = _survey_layout(tmp_path)
    other = tmp_path / "georectified"
    other.mkdir()
    decoy = _with_gps(other, 10.0, 10.0, original.name)
    got = X.raw_sibling(rectified, [rectified.parent])
    assert got == original, f"took the decoy in {decoy.parent.name}"


def test_nothing_is_found_when_there_is_no_original(tmp_path):
    rect = tmp_path / "rectified"
    rect.mkdir()
    lone = _img(rect, "99_corrected_width=1m_GSD=0.001m_per_px.jpg")
    assert X.raw_sibling(lone, [rect]) is None
    fix, why = X.seed_from_photo(lone, search_dirs=[rect])
    assert fix is None and "No position" in why
