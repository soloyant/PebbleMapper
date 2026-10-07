"""Photograph reading, in one place: which files are photographs, how a
HEIC/HEIF (iPhone) opens, and the one pixel frame every measurement uses.

Formats
-------
JPEG, PNG and TIFF open through Pillow as they always did. HEIC/HEIF open
through Pillow too, once the ``pillow-heif`` plugin is registered
(:func:`register_heif_opener`, run at import of this module and again,
lazily, by :func:`open_photo` / :func:`read_rgb`). Every ``Image.open`` in
the application therefore reads an iPhone photograph without knowing it;
only the error when the plugin is missing needs this module.

The pixel frame (read this before touching orientation)
-------------------------------------------------------
The canvases (Orthorectify, Digitize, Zonal) and every array read
for measurement must share ONE pixel frame, because corner picks and
digitised polygons are stored in it. That frame is **the stored pixel
frame: what Pillow returns, with no EXIF-orientation transpose applied.**

* A browser applies the EXIF orientation tag; Pillow does not. So
  ``functions.zonal_canvas.prepare_browser_image`` serves a photograph
  verbatim only when its tag is plainly upright (1) and otherwise
  transcodes it, unrotated, to a tagless PNG: the browser shows the stored
  frame, and :func:`read_rgb` returns the stored frame.
* A HEIC needs no special case. libheif applies the container's own
  rotation (``irot``) while decoding and pillow-heif resets the EXIF
  orientation tag to 1, so Pillow's stored frame *is* the upright picture
  (verified on an iPhone 16 Pro photograph: 2142 x 2856, tag 1, the raw
  embedded tag was 8). The canvas transcodes it (not browser-native) from
  that same decode.
* OpenCV is the odd one out: ``cv2.imread`` / ``cv2.imdecode`` DO apply
  the EXIF tag, which is why nothing here reads through OpenCV.

:func:`open_photo` returns the stored frame by default; ``upright=True``
applies the tag, for previews and for the upright JPEG working copy the
:func:`prepare_photo` writes for an EXIF-rotated photograph (both the canvas and the
detector then read the copy, so the frame is still shared).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = [
    "HEIF_EXTENSIONS", "CAMERA_EXTENSIONS", "PHOTO_EXTENSIONS",
    "HEIF_AVAILABLE", "register_heif_opener", "heif_import_error", "heif_failure_hint",
    "is_heif", "is_photo",
    "exif_orientation", "open_photo", "read_rgb",
    "needs_working_copy", "working_copy", "prepare_photo", "display_source",
]

HEIF_EXTENSIONS: Tuple[str, ...] = (".heic", ".heif")
# What a camera writes. A TIFF inside a project is an ortho, so the lists
# that separate the two modes (Detect's Quadrat listing, the project's
# photo resolver) use this one.
CAMERA_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png") + HEIF_EXTENSIONS
# Everything a photograph folder may hold, TIFF included.
PHOTO_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".tif", ".tiff") + HEIF_EXTENSIONS

_HEIF_MISSING = (
    "cannot open {name}: HEIC/HEIF support needs the pillow-heif package "
    "(conda install -c conda-forge pillow-heif, or pip install pillow-heif) "
    "in the maskrcnn environment.")

_registered = False
_import_error = ""
_reported = False


def heif_import_error() -> str:
    """Why HEIC support is unavailable (empty when it is available)."""
    return _import_error


def register_heif_opener() -> bool:
    """Teach Pillow to open HEIC/HEIF through pillow-heif. True when the
    package is installed (registering once; later calls only re-check the
    import); False, and Pillow unchanged, when it is not. The reason is kept
    in :func:`heif_import_error` and printed once, so a missing package or a
    DLL that fails to load is visible in the console rather than silent."""
    global _registered, _import_error, _reported
    try:
        import pillow_heif
    except Exception as ex:          # ImportError, or an OSError from a DLL
        _import_error = f"{type(ex).__name__}: {ex}"
        if not _reported:
            _reported = True
            print(f"[images] HEIC/HEIF support unavailable ({_import_error}); "
                  "install pillow-heif in the maskrcnn environment.", flush=True)
        return False
    if not _registered:
        try:
            pillow_heif.register_heif_opener()
        except Exception as ex:
            _import_error = f"{type(ex).__name__}: {ex}"
            return False
        _registered = True
    _import_error = ""
    return True


# Registered once at import so every Pillow reader in the app (the canvas
# transcoder included) can open an iPhone photograph.
HEIF_AVAILABLE = register_heif_opener()


def is_heif(path) -> bool:
    """Whether the file name says HEIC/HEIF."""
    return Path(str(path)).suffix.lower() in HEIF_EXTENSIONS


def is_photo(path) -> bool:
    """Whether the file name is one of :data:`PHOTO_EXTENSIONS`."""
    return Path(str(path)).suffix.lower() in PHOTO_EXTENSIONS


def heif_failure_hint(path) -> str:
    """What to append to a "could not open" message when ``path`` is a
    HEIC/HEIF: the reason HEIC support is unavailable (the import error,
    re-checked so a package installed since start-up counts), or, when
    pillow-heif is present, that the file itself is the problem. Empty
    for any other file."""
    if not is_heif(path):
        return ""
    register_heif_opener()
    err = heif_import_error()
    if err:
        return f" — HEIC support unavailable: {err}"
    return (" — HEIC/HEIF needs the pillow-heif package; it is installed "
            "here, so the file itself could not be decoded")


def _require_heif(p: Path) -> None:
    if is_heif(p) and not register_heif_opener():
        raise RuntimeError(_HEIF_MISSING.format(name=p.name))


def exif_orientation(path) -> int:
    """The EXIF orientation tag, or 1 when there is none.

    A browser applies this tag and PIL (every reader in ``functions/``) does
    not, so it decides whether a file can be served untranscoded. A HEIC
    reports 1: pillow-heif has already applied the rotation to the pixels.
    """
    try:
        from PIL import Image as _PIL, ExifTags as _ExifTags
        tag = next(k for k, v in _ExifTags.TAGS.items() if v == "Orientation")
        with _PIL.open(path) as im:
            ex = im.getexif()
            return int(ex.get(tag, 1)) if ex else 1
    except Exception:
        # An unreadable tag must not read as "no rotation": 0 is outside the
        # valid range 1..8, so callers treat it as "not plainly upright".
        return 0


def open_photo(path, *, upright: bool = False):
    """The photograph as a loaded PIL image.

    By default in the stored pixel frame (the measurement frame, see the
    module docstring). With ``upright=True`` the EXIF orientation tag is
    applied (``ImageOps.exif_transpose``), so an EXIF-rotated portrait shows
    as the camera was held. Raises ``RuntimeError`` naming pillow-heif for
    a HEIC/HEIF file when that package is missing.
    """
    from PIL import Image, ImageOps
    p = Path(path)
    _require_heif(p)
    im = Image.open(p)
    if not upright:
        im.load()
        return im
    try:
        rotated = ImageOps.exif_transpose(im)
    except Exception:
        rotated = None
    if rotated is None:            # older Pillow returns None for "no tag"
        rotated = im
    rotated.load()
    if rotated is not im:
        im.close()
    return rotated


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """The same rules as ``clasts_detection._normalize_to_uint8`` for the
    non-8-bit modes Pillow will not convert sensibly (a 16-bit PNG converts
    to RGB by clipping at 255, which is white)."""
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmax(arr) <= 1.0:
            return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.dtype == np.uint16:
        return (arr // 257).astype(np.uint8)       # 257 = 65535/255 (exact)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi > lo:
        return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros(arr.shape, dtype=np.uint8)


def read_rgb(path) -> np.ndarray:
    """The photograph as an ``H x W x 3`` uint8 array in the stored pixel
    frame: what Pillow returns, **no EXIF-orientation transpose** (the frame
    the canvases show and the corner picks and polygons are stored in; see
    the module docstring). This replaces ``matplotlib.image.imread``, which
    read the same frame but returned float for some PNGs, 2-D for greyscale
    and 4 channels for RGBA.

    8-bit modes convert through Pillow (palette, greyscale, CMYK, alpha
    dropped); 16-bit and float modes are scaled to 0..255 the way the
    detector always normalised them. Raises ``RuntimeError`` naming
    pillow-heif for a HEIC/HEIF file when that package is missing, and
    whatever Pillow raises for a file it cannot identify.
    """
    from PIL import Image
    p = Path(path)
    _require_heif(p)
    with Image.open(p) as im:
        if im.mode in ("1", "L", "P", "LA", "PA", "RGB", "RGBA", "RGBX",
                       "CMYK", "YCbCr", "HSV", "LAB"):
            return np.asarray(im.convert("RGB"))
        arr = _to_uint8(np.asarray(im))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
#  The upright working copy                                                    #
# --------------------------------------------------------------------------- #
def needs_working_copy(path) -> bool:
    """True when the stored frame is not the upright picture: an EXIF
    orientation other than 1 (0, an unreadable tag, is left alone: the
    canvas transcodes it and the detector reads the same stored frame). A
    HEIC never needs one; pillow-heif delivers the upright pixels."""
    return exif_orientation(Path(path)) not in (0, 1)


def working_copy(path, out_dir) -> str:
    """Write ``<out_dir>/<stem>.jpg``, an upright RGB JPEG (quality 95) of
    the photograph, unless a copy newer than the source is already there;
    returns the copy's path. Detection and the canvas both use the copy."""
    p = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{p.stem}.jpg"
    try:
        fresh = dst.exists() and dst.stat().st_mtime >= p.stat().st_mtime
    except OSError:
        fresh = False
    if not fresh:
        im = open_photo(p, upright=True)
        try:
            im.convert("RGB").save(dst, format="JPEG", quality=95)
        finally:
            im.close()
    return str(dst)


def display_source(path, out_dir) -> str:
    """The file a browser canvas should show for ``path`` so that what it
    shows is the stored pixel frame: the photograph itself when a browser
    can decode it and its EXIF orientation is plainly upright (1), else
    ``<out_dir>/<stem>.jpg``, the stored frame re-encoded without any tag
    (a HEIC, which a browser cannot decode, or a rotated JPEG, which a
    browser would turn)."""
    p = Path(path)
    if not is_heif(p) and exif_orientation(p) == 1:
        return str(p)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{p.stem}.jpg"
    try:
        fresh = dst.exists() and dst.stat().st_mtime >= p.stat().st_mtime
    except OSError:
        fresh = False
    if not fresh:
        from PIL import Image
        Image.fromarray(read_rgb(p)).save(dst, format="JPEG", quality=92)
    return str(dst)


def prepare_photo(path, out_dir) -> str:
    """The file to draw on and detect on: the photograph itself when its
    stored frame is upright (every HEIC, most JPEGs), else its
    :func:`working_copy`."""
    return working_copy(path, out_dir) if needs_working_copy(path) else str(path)
