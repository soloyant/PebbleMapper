"""functions.images: HEIC/HEIF photographs as first-class inputs, and the
one pixel frame shared by the canvases and every measurement.

A small HEIC is encoded with pillow-heif (skipped where it cannot encode);
the real iPhone photograph is checked when it is on this machine.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from functions import images as I

# A real camera HEIC to test against, when one is available: point
# PEBBLEMAPPER_TEST_HEIC at it. The tests that need it skip without it.
REAL_HEIC = Path(os.environ.get("PEBBLEMAPPER_TEST_HEIC", ""))


def _pattern(h=30, w=50, seed=1):
    """Asymmetric so a rotation or a transpose cannot go unnoticed."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    arr[:5, :, :] = 250          # a bright band along the top edge
    return arr


@pytest.fixture
def heic(tmp_path):
    pytest.importorskip("pillow_heif")
    assert I.register_heif_opener()
    p = tmp_path / "IMG_0001.heic"
    try:
        Image.fromarray(_pattern()).save(p, format="HEIF", quality=-1,
                                         chroma=444)
    except Exception as ex:                  # a build without an encoder
        pytest.skip(f"pillow-heif cannot encode here: {ex}")
    return p


# --------------------------------------------------------------------------- #
#  Opening and reading                                                         #
# --------------------------------------------------------------------------- #
def test_the_extension_lists_and_the_predicates():
    assert I.HEIF_EXTENSIONS == (".heic", ".heif")
    assert I.PHOTO_EXTENSIONS == (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                                  ".heic", ".heif")
    assert set(I.CAMERA_EXTENSIONS) == {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    assert I.is_heif("a/IMG_1.HEIC") and I.is_heif(Path("b.heif"))
    assert not I.is_heif("x.jpg")
    assert I.is_photo("x.JPG") and I.is_photo("y.heic") and not I.is_photo("z.csv")


def test_open_photo_returns_rgb_with_the_right_size(heic):
    with I.open_photo(heic) as im:
        assert im.mode == "RGB" and im.size == (50, 30)
    # The plain Pillow call works too: the opener is registered at import.
    with Image.open(heic) as im:
        assert im.size == (50, 30)


def test_read_rgb_matches_pil_and_is_lossless_enough(heic, tmp_path):
    arr = I.read_rgb(heic)
    assert arr.shape == (30, 50, 3) and arr.dtype == np.uint8
    with Image.open(heic) as im:
        assert np.array_equal(arr, np.asarray(im.convert("RGB")))
    # The encoder is not bit-exact on noise even at quality=-1 (chroma
    # subsampling); the flat band is exact and the rest is close, and
    # nothing is rotated or transposed.
    assert np.all(arr[:5] == 250)
    assert np.abs(arr.astype(int) - _pattern().astype(int)).mean() < 8
    # And a JPEG / PNG read the same way as through Pillow.
    png = tmp_path / "a.png"
    Image.fromarray(_pattern()).save(png)
    assert np.array_equal(I.read_rgb(png), _pattern())


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "I;16"])
def test_read_rgb_always_returns_three_uint8_channels(tmp_path, mode):
    """matplotlib returned 2-D for greyscale, 4 channels for RGBA and float
    for 16-bit; the detector wants H x W x 3 uint8 every time."""
    p = tmp_path / f"{mode.replace(';', '_')}.png"
    if mode == "I;16":
        Image.fromarray((np.full((8, 9), 128, dtype=np.uint16) * 257)).save(p)
    else:
        Image.fromarray(_pattern(8, 9)).convert(mode).save(p)
    arr = I.read_rgb(p)
    assert arr.shape == (8, 9, 3) and arr.dtype == np.uint8
    if mode == "I;16":
        assert int(arr[0, 0, 0]) == 128            # 16-bit scaled, not clipped white
    if mode == "RGBA":
        assert np.array_equal(arr, _pattern(8, 9))


def test_read_rgb_is_the_stored_frame_and_open_photo_can_be_upright(tmp_path):
    """The contract: read_rgb returns the frame the canvas shows (a rotated
    JPEG is transcoded unrotated), not the browser-upright one."""
    rot = tmp_path / "rot6.jpg"
    plain = tmp_path / "plain.jpg"
    im = Image.fromarray(_pattern(40, 100))
    ex = im.getexif()
    ex[274] = 6                                   # rotate 90 deg CW to display
    im.save(rot, exif=ex.tobytes(), quality=95)
    im.save(plain, quality=95)
    assert I.exif_orientation(rot) == 6 and I.exif_orientation(plain) == 1
    assert I.read_rgb(rot).shape == (40, 100, 3)
    assert np.array_equal(I.read_rgb(rot), I.read_rgb(plain))
    with I.open_photo(rot) as raw:
        assert raw.size == (100, 40)
    with I.open_photo(rot, upright=True) as up:
        assert up.size == (40, 100)
    # The Gauge working copy is for that case only; the copy is tagless.
    assert I.needs_working_copy(rot) and not I.needs_working_copy(plain)
    copy = I.prepare_photo(rot, tmp_path / "q")
    assert Path(copy) == tmp_path / "q" / "rot6.jpg"
    assert I.exif_orientation(copy) == 1
    assert I.prepare_photo(plain, tmp_path / "q") == str(plain)


def test_a_heic_needs_no_working_copy(heic, tmp_path):
    """pillow-heif hands Pillow the upright pixels with the tag reset to 1,
    so the canvas transcodes it and the detector reads the same frame."""
    assert I.exif_orientation(heic) == 1
    assert not I.needs_working_copy(heic)
    assert I.prepare_photo(heic, tmp_path / "q") == str(heic)
    assert not (tmp_path / "q").exists()


def test_the_canvas_transcodes_a_heic_to_the_same_frame(heic, tmp_path, monkeypatch):
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    import tempfile
    tempfile.tempdir = None
    from functions.zonal_canvas import prepare_browser_image, render_canvas_image
    png, scale = prepare_browser_image(str(heic), current_project="")
    assert png != str(heic) and png.endswith(".png") and scale == 1.0
    with Image.open(png) as c:
        assert np.array_equal(np.asarray(c.convert("RGB")), I.read_rgb(heic))
    info = render_canvas_image(str(heic), current_project="")
    assert (info["w_orig"], info["h_orig"]) == (50, 30)
    assert info["geotransform"] is None


def test_exif_tags_survive_on_a_heic(tmp_path):
    pytest.importorskip("pillow_heif")
    assert I.register_heif_opener()
    p = tmp_path / "tagged.heic"
    im = Image.fromarray(_pattern())
    ex = im.getexif()
    ex[271] = "Apple"
    ex[272] = "iPhone 16 Pro"
    im.save(p, format="HEIF", exif=ex.tobytes())
    with Image.open(p) as back:
        got = back.getexif()
    assert got.get(271) == "Apple" and got.get(272) == "iPhone 16 Pro"


@pytest.mark.skipif(not REAL_HEIC.is_file(), reason="the iPhone test photograph is not on this machine")
def test_the_real_iphone_photograph():
    """Portrait, upright out of pillow-heif (tag 1), GPS fix readable
    through the same getexif() path exif_seed uses."""
    assert I.exif_orientation(REAL_HEIC) == 1
    arr = I.read_rgb(REAL_HEIC)
    assert arr.shape == (2856, 2142, 3) and arr.dtype == np.uint8
    from functions.exif_seed import gps_from_image
    fix = gps_from_image(REAL_HEIC)
    assert fix and -90 <= fix["lat"] <= 90 and -180 <= fix["lon"] <= 180
    assert not I.needs_working_copy(REAL_HEIC)


# --------------------------------------------------------------------------- #
#  The clear error without pillow-heif                                         #
# --------------------------------------------------------------------------- #
def test_the_error_names_pillow_heif_when_it_is_missing(tmp_path, monkeypatch):
    fake = tmp_path / "IMG_9.heic"
    fake.write_bytes(b"\x00" * 16)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)   # import -> ImportError
    assert I.register_heif_opener() is False
    for fn in (I.read_rgb, I.open_photo):
        with pytest.raises(RuntimeError, match=r"pillow-heif.*IMG_9\.heic|IMG_9\.heic.*pillow-heif"):
            fn(fake)
    # A JPEG is unaffected by the missing plugin.
    jpg = tmp_path / "ok.jpg"
    Image.fromarray(_pattern()).save(jpg)
    assert I.read_rgb(jpg).shape == (30, 50, 3)


# --------------------------------------------------------------------------- #
#  Every listing includes it                                                   #
# --------------------------------------------------------------------------- #
def test_list_images_and_the_project_photo_resolver_include_heic(tmp_path, monkeypatch):
    from gui.app import list_images
    for n in ("b.HEIC", "a.jpg", "c.heif", "d.txt", "e.png", "f.tif"):
        (tmp_path / n).write_bytes(b"")
    assert list_images(str(tmp_path), list(I.PHOTO_EXTENSIONS)) == \
        ["a.jpg", "b.HEIC", "c.heif", "e.png", "f.tif"]
    assert list_images(str(tmp_path), list(I.CAMERA_EXTENSIONS)) == \
        ["a.jpg", "b.HEIC", "c.heif", "e.png"]

    import functions.layout as layout
    import functions.project_defaults as pdf
    monkeypatch.setattr(layout, "DATASETS_ROOT", tmp_path / "ds")
    layout.ensure_project_layout("P")
    d = layout.project_path("P", "images")
    for n in ("IMG_1.heic", "IMG_2.jpg", "ortho.tif"):
        (d / n).write_bytes(b"")
    # Newest first, and the two were written within the same instant or
    # not: the order is the file system's, the membership is the point.
    assert sorted(p.name for p in pdf.photos("P")) == ["IMG_1.heic", "IMG_2.jpg"]
    assert [p.name for p in pdf.orthos("P")] == ["ortho.tif"]


def test_layout_and_report_know_a_heic_is_a_photograph(tmp_path, monkeypatch):
    import functions.layout as layout
    hints = dict(layout._LEGACY_FILE_HINTS)
    assert hints[".heic"] == "images" and hints[".heif"] == "images"
    monkeypatch.setattr(layout, "DATASETS_ROOT", tmp_path)
    layout.ensure_project_layout("P")
    (tmp_path / "P" / "input_data" / "IMG_3.heic").write_bytes(b"x")
    layout.migrate_input_structure("P", dry_run=False)
    assert (tmp_path / "P" / "input_data" / "images" / "IMG_3.heic").exists()

    from functions import modes
    from functions.report import _identify_image_kind
    assert _identify_image_kind(Path("IMG_3.heic")) == modes.QUADRAT
    assert _identify_image_kind(Path("ortho.tif")) == modes.ORTHO


def test_orthorectify_lists_and_rectifies_a_heic(heic, tmp_path):
    from functions import orthorectify as O
    assert ".heic" in O._IMAGE_SUFFIXES and ".heif" in O._IMAGE_SUFFIXES
    # The folder listing a rectified output is matched against.
    listing = O.output_listing(tmp_path / "IMG_0001_rectified.jpg")
    assert heic in listing
    # And the source reads through Pillow: a HEIC rectifies to a JPEG.
    corners = [(5, 5), (45, 6), (44, 25), (6, 24)]
    r = O.rectify_one(heic, corners, [0.5] * 4,
                      tmp_path / "orthorectified" / "IMG_0001_rectified.jpg",
                      seg_confirmed=True)
    assert r.ok, r.refusal
    assert Path(r.out_path).suffix == ".jpg" and Path(r.out_path).is_file()
    assert r.record["PM_SOURCE"] == "IMG_0001.heic"


def test_rectify_one_reads_the_frame_the_corners_were_picked_in(tmp_path):
    """cv2.imdecode applies the EXIF orientation tag and the canvas does
    not; the source is read through Pillow so a rotated JPEG rectifies
    exactly like its tagless twin (same bytes, only the tag differs)."""
    from functions import orthorectify as O
    im = Image.fromarray(_pattern(300, 500, seed=7))
    ex = im.getexif()
    ex[274] = 6
    rot, plain = tmp_path / "rot.jpg", tmp_path / "plain.jpg"
    im.save(rot, exif=ex.tobytes(), quality=95)
    im.save(plain, quality=95)
    corners = [(50, 40), (450, 60), (440, 260), (60, 250)]
    a = O.rectify_one(rot, corners, [1.0] * 4, tmp_path / "o" / "rot_r.png",
                      seg_confirmed=True)
    b = O.rectify_one(plain, corners, [1.0] * 4, tmp_path / "o" / "plain_r.png",
                      seg_confirmed=True)
    assert a.ok and b.ok, (a.refusal, b.refusal)
    assert a.shape == b.shape
    assert np.array_equal(I.read_rgb(a.out_path), I.read_rgb(b.out_path))


def test_detect_quadrat_reads_a_heic_through_pillow(heic, tmp_path):
    """The reader path of ``_detect_quadrat`` (not the model): the array the
    model receives is the H x W x 3 uint8 stored frame of the HEIC."""
    pytest.importorskip("tensorflow")
    from functions.clasts_detection import _detect_quadrat
    seen = {}

    class _NoDetections:
        def detect(self, images, verbose=0):
            seen["image"] = images[0]
            h, w = images[0].shape[:2]
            return [{"masks": np.zeros((h, w, 0), dtype=bool),
                     "rois": np.zeros((0, 4)), "class_ids": np.zeros(0, int),
                     "scores": np.zeros(0)}]

    df = _detect_quadrat(_NoDetections(), str(heic), 0.001, False, False,
                         False, output_dir=str(tmp_path / "v"),
                         figures_dir=str(tmp_path / "f"))
    assert len(df) == 0
    assert seen["image"].shape == (30, 50, 3) and seen["image"].dtype == np.uint8
    assert np.array_equal(seen["image"], I.read_rgb(heic))


def test_the_heic_failure_hint_names_the_reason(tmp_path, monkeypatch):
    assert I.heif_failure_hint(tmp_path / "a.jpg") == ""
    # pillow-heif present: the file, not the package, is the problem.
    monkeypatch.setattr(I, "register_heif_opener", lambda: True)
    monkeypatch.setattr(I, "_import_error", "")
    hint = I.heif_failure_hint(tmp_path / "a.heic")
    assert hint.startswith(" — HEIC/HEIF needs the pillow-heif package") and "installed" in hint
    # pillow-heif missing or its DLL failing: the import error, verbatim.

    def _fail():
        I._import_error = "ImportError: No module named pillow_heif"
        return False
    monkeypatch.setattr(I, "register_heif_opener", _fail)
    assert I.heif_failure_hint("b.HEIF") == \
        " — HEIC support unavailable: ImportError: No module named pillow_heif"


def test_gauge_still_exports_the_helpers(heic):
    from functions import gauge as Q
    assert Q.PHOTO_EXTENSIONS is I.PHOTO_EXTENSIONS
    assert Q.register_heif_opener is I.register_heif_opener
    assert Q.prepare_photo is I.prepare_photo and Q.working_copy is I.working_copy
    with Q.open_photo(heic) as im:            # gauge's open_photo is upright
        assert im.size == (50, 30)
