"""A rotated photograph must not reach the browser untranscoded.

Browsers apply the EXIF orientation tag. PIL, which every reader in
``functions/`` uses, does not. If a browser-native photograph carrying a
rotation tag is served straight to the canvas, the user picks corners on a
rotated picture while the rectification reads the raw one, and every
coordinate is silently 90 or 180 degrees out.

Ten of the 77 hand-picked corner sets in the field archive ARE in a rotated
frame, and eight are predicted exactly by this tag -- but checked over the
whole corpus, none of them could have come through this code path, so their
origin is unexplained. The Orthorectify canvas did not go through this path
at all and served rotated JPEGs untouched until 2026-09-27; it now uses
``functions.images.display_source`` (tests at the end of this file).
"""
from pathlib import Path

import pytest
from PIL import Image

from gui.app import prepare_browser_image
from functions.zonal_canvas import exif_orientation as _exif_orientation


def _jpg(path, orientation=None, size=(600, 400)):
    im = Image.new("RGB", size, (120, 90, 60))
    for x in range(0, size[0], 40):          # something asymmetric to see
        for y in range(0, size[1] // 3):
            im.putpixel((x, y), (240, 240, 240))
    kw = {}
    if orientation is not None:
        ex = im.getexif()
        ex[274] = orientation                 # 274 == Orientation
        kw["exif"] = ex.tobytes()
    im.save(path, format="JPEG", **kw)
    return path


def test_an_upright_photograph_is_served_untouched(tmp_path):
    """The fast path must survive: transcoding every photo would be slow."""
    p = _jpg(tmp_path / "upright.jpg", orientation=1)
    out, scale = prepare_browser_image(str(p), current_project="")
    assert out == str(p), "an upright browser-native image should pass through"
    assert scale == 1.0


def test_a_photograph_with_no_exif_at_all_is_served_untouched(tmp_path):
    p = _jpg(tmp_path / "plain.jpg", orientation=None)
    out, _ = prepare_browser_image(str(p), current_project="")
    assert out == str(p)


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_a_rotated_photograph_is_transcoded_not_passed_through(tmp_path,
                                                               orientation):
    """The defect: the browser would rotate it and the pipeline would not."""
    p = _jpg(tmp_path / f"rot{orientation}.jpg", orientation=orientation)
    out, scale = prepare_browser_image(str(p), current_project="")
    assert out != str(p), (
        f"orientation {orientation} was served untranscoded; the browser will "
        "rotate it and every corner pick will be in the wrong frame")
    assert Path(out).suffix.lower() == ".png"
    assert _exif_orientation(out) == 1, "the cached copy must carry no tag"


def test_the_transcoded_copy_has_the_same_pixel_frame_as_the_source(tmp_path):
    """disp_scale is a pure scale, so the frames must not differ by a rotation."""
    p = _jpg(tmp_path / "rot6.jpg", orientation=6, size=(600, 400))
    out, scale = prepare_browser_image(str(p), current_project="")
    with Image.open(p) as a, Image.open(out) as b:
        assert (a.size[0] / a.size[1]) == pytest.approx(b.size[0] / b.size[1]), \
            "the cached copy was rotated; disp_scale cannot express that"
    assert scale == pytest.approx(1.0)


def test_an_unreadable_orientation_is_not_reported_as_upright(tmp_path):
    """Reporting 'no rotation' on failure is the reading that corrupts data."""
    bad = tmp_path / "not_an_image.jpg"
    bad.write_bytes(b"not an image")
    assert _exif_orientation(bad) != 1


# --------------------------------------------------------------------------- #
#  Orthorectify's canvas: functions.images.display_source                      #
# --------------------------------------------------------------------------- #
# The Orthorectify tab serves its photograph through display_source, not
# prepare_browser_image. It used to pass every JPEG through untouched, so an
# iPhone photograph stored landscape with orientation 6 (IMG_0806 of the
# paper's validation set) showed turned a quarter while the corners stayed
# in the stored frame.
def test_orthorectify_serves_an_upright_jpeg_untouched(tmp_path):
    from functions.images import display_source
    p = _jpg(tmp_path / "upright.jpg", orientation=1)
    assert display_source(p, tmp_path / "cache") == str(p)


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_orthorectify_serves_a_rotated_jpeg_as_its_stored_frame(tmp_path, orientation):
    import numpy as np
    from functions.images import display_source, read_rgb
    p = _jpg(tmp_path / f"rot{orientation}.jpg", orientation=orientation)
    out = display_source(p, tmp_path / "cache")
    assert out != str(p)
    assert _exif_orientation(out) == 1
    with Image.open(out) as im:
        assert im.size == (600, 400), "the copy must keep the stored frame"
    a, b = read_rgb(p).astype(int), read_rgb(out).astype(int)
    assert np.abs(a - b).mean() < 6, "the copy must show the same pixels"
