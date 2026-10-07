"""Browser-image preparation for the drawing canvases (Zonal, Detection ROI,
Digitize), runnable out of process by the worker.

GeoTIFFs and HEIC/HEIF photographs (which browsers cannot render) become
cached, optionally-downsampled PNGs; small browser-native files with a
plainly-upright EXIF orientation are served verbatim. The pixel-frame
contract (why a rotated JPEG is transcoded unrotated, why a HEIC needs no
special case) is documented in :mod:`functions.images`.
"""

from __future__ import annotations

from pathlib import Path

from functions.layout import project_path
from functions import branding as _brand
from functions import images as _images
# Importing functions.images registers the HEIF opener, so the transcoder
# below opens an iPhone photograph like any other. Re-exported: callers and
# tests import it from here.
from functions.images import exif_orientation  # noqa: F401


def prepare_browser_image(src_path: str, *,
                          current_project: str = "",
                          max_dim: int = 4000,
                          cache_subdir: str = "canvas_cache") -> tuple:
    """Return ``(png_path, disp_scale)`` suitable for ui.interactive_image.

    Browsers cannot show GeoTIFF, so the source is transcoded to a
    downsampled PNG and cached. ``disp_scale = original_W / displayed_W``
    converts displayed-pixel coords back to the GeoTransform's original-pixel
    frame; 1.0 when the source is served verbatim.
    """
    src_p = Path(src_path)
    if not src_p.exists():
        return str(src_p), 1.0
    ext = src_p.suffix.lower()

    from PIL import Image as _PIL
    try:
        _RES = _PIL.Resampling.LANCZOS  # Pillow 9.1+
    except AttributeError:
        _RES = _PIL.LANCZOS  # older Pillow

    try:
        with _PIL.open(src_p) as im:
            W0, H0 = im.size
    except Exception:
        return str(src_p), 1.0

    browser_native = ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")
    # A browser applies the EXIF orientation tag; PIL does not. Transcoding
    # writes unrotated pixels with no tag, so what the browser shows is what
    # the pipeline reads.
    if browser_native and max(W0, H0) <= max_dim \
            and exif_orientation(src_p) == 1:
        return str(src_p), 1.0

    import hashlib
    try:
        mtime = src_p.stat().st_mtime
    except OSError:
        mtime = 0
    sig = f"{src_p.resolve()}|{mtime}|{max_dim}"
    cache_id = hashlib.md5(sig.encode()).hexdigest()[:12]
    if current_project:
        cache_dir = (project_path(current_project, "zonal")
                     / f".{cache_subdir}")
    else:
        import tempfile
        cache_dir = (Path(tempfile.gettempdir())
                     / _brand.APP_PACKAGE / cache_subdir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{src_p.stem}_{cache_id}.png"
    if cache_path.exists():
        try:
            with _PIL.open(cache_path) as cim:
                W_disp, _h_disp = cim.size
            return str(cache_path), float(W0) / float(W_disp)
        except Exception:
            pass

    try:
        with _PIL.open(src_p) as im:
            if max(W0, H0) > max_dim:
                scale = float(max_dim) / float(max(W0, H0))
                new_W = max(1, int(round(W0 * scale)))
                new_H = max(1, int(round(H0 * scale)))
                im = im.convert("RGB").resize((new_W, new_H), _RES)
            else:
                im = im.convert("RGB")
                new_W, new_H = W0, H0
            im.save(cache_path, format="PNG", optimize=False)
        return str(cache_path), float(W0) / float(new_W)
    except Exception as ex:
        try:
            print(f"[canvas] PNG cache build failed: {ex}")
        except Exception:
            pass
        return str(src_p), 1.0


def render_canvas_image(src_path: str, *,
                        current_project: str = "",
                        max_dim: int = 4000,
                        cache_subdir: str = "canvas_cache") -> dict:
    """Probe + prepare a source for the canvas (worker op ``canvas_source``),
    returning everything the GUI needs to mount it. The heavy work (full
    decode, PNG transcode) happens here; the GUI then hits a hot cache.
    """
    src_p = Path(src_path)
    info: dict = {"src": str(src_p)}

    from PIL import Image as _PIL
    with _PIL.open(src_p) as im:          # unreadable source raises: loud
        info["w_orig"], info["h_orig"] = im.size

    info["geotransform"] = None
    info["img_size"] = None
    try:
        # GDAL has no HEIF driver in this environment and prints "ERROR 4"
        # on a file it does not recognise; a photograph has no geotransform.
        if _images.is_heif(src_p):
            raise ValueError("HEIF: not a GDAL raster")
        from osgeo import gdal as _gdal
        ds = _gdal.Open(str(src_p))
        if ds is not None:
            gt = ds.GetGeoTransform()
            info["img_size"] = [ds.RasterXSize, ds.RasterYSize]
            ds = None
            if gt and tuple(gt) != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                info["geotransform"] = list(gt)
    except Exception:
        pass  # a plain photo has no geotransform; that is not an error

    png, disp_scale = prepare_browser_image(
        str(src_p), current_project=current_project,
        max_dim=max_dim, cache_subdir=cache_subdir)
    info["png"] = str(png)
    info["disp_scale"] = float(disp_scale)
    try:
        with _PIL.open(png) as im2:
            info["w_disp"], info["h_disp"] = im2.size
    except Exception:
        info["w_disp"], info["h_disp"] = info["w_orig"], info["h_orig"]
    return info
