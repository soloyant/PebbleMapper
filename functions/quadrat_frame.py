"""The quadrat frame in a rectified photograph, and the rectangle inside it.

Orthorectify maps the frame's outer corners to the image's corners and records
the frame's thickness in the sidecar (``PM_FRAME_THICKNESS_M``), so a
rectified photograph carries the frame bars along its four edges — about
2 cm, 36 px at 0.55 mm/px. Georeference already insets them before matching;
detection did not, and every model measured pieces of the frame as clasts
(Mask R-CNN mostly learnt to ignore it, the plugged-in models did not).

This module is the one place that knows where the frame is: the inset in
pixels, the inner rectangle, and a GeoJSON ROI of it in the frame quadrat
detection already understands (image pixels, ``(col, row)``), so every
backend excludes the frame the same way.

The inset carries a margin of 10 % of the thickness (2 px at least): the
outer corners are picked by hand and the bars are not machined to the
millimetre, so on a real rectified photograph a sliver of bar sits inside
the nominal band (measured: 5 px of a 35 px bar on one edge of an Etretat
quadrat). The margin costs 0.5 % of the area and keeps the rule honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from functions.gsd import effective_gsd, sidecar_path

SIDECAR_KEY = "PM_FRAME_THICKNESS_M"
MARGIN_FRACTION = 0.10      # of the bar's width in pixels
MARGIN_MIN_PX = 2


@dataclass(frozen=True)
class FrameInset:
    thickness_m: float      # the frame bar's width, from the sidecar
    gsd_m: float            # metres per pixel of the rectified photograph
    frame_px: int           # the bar's width in pixels
    margin_px: int          # the safety margin added to it
    width: int              # image size, pixels
    height: int

    @property
    def inset_px(self) -> int:
        """Pixels left out along each edge: the bar plus the margin."""
        return self.frame_px + self.margin_px

    @property
    def inner(self) -> tuple:
        """``(x0, y0, x1, y1)`` of the rectangle inside the frame, pixels."""
        return (self.inset_px, self.inset_px,
                self.width - self.inset_px, self.height - self.inset_px)

    @property
    def inner_size_m(self) -> tuple:
        return ((self.width - 2 * self.inset_px) * self.gsd_m,
                (self.height - 2 * self.inset_px) * self.gsd_m)

    def contains(self, col: float, row: float) -> bool:
        x0, y0, x1, y1 = self.inner
        return x0 <= col < x1 and y0 <= row < y1


def frame_thickness_m(image_path) -> Optional[float]:
    """The frame thickness Orthorectify recorded for this photograph, or
    None when there is no sidecar, no record, or a value that is not a
    positive number."""
    sp = sidecar_path(image_path)
    try:
        raw = json.loads(Path(sp).read_text(encoding="utf-8")).get(SIDECAR_KEY)
    except Exception:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _image_size(image_path) -> Optional[tuple]:
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None


def set_frame_thickness(image_path, thickness_m: Optional[float]) -> Path:
    """Record the frame thickness of a photograph in its sidecar, where
    Orthorectify writes it, keeping every other key; None or 0 forgets it.
    An unreadable sidecar is left alone rather than overwritten."""
    sp = Path(sidecar_path(image_path))
    data = {}
    if sp.exists():
        data = json.loads(sp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{sp.name} is not a record")
    if thickness_m is not None and float(thickness_m) > 0:
        data[SIDECAR_KEY] = f"{float(thickness_m):.4f}"
    else:
        data.pop(SIDECAR_KEY, None)
    tmp = sp.with_name(sp.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(sp)
    return sp


def frame_inset(image_path, fallback_m: Optional[float] = None
                ) -> Optional[FrameInset]:
    """Where the frame is in this rectified photograph, or None when the
    photograph carries no frame record (a plain photo, an ortho, an older
    rectification) or no usable GSD -- then there is nothing to exclude.
    ``fallback_m`` is the thickness to assume when there is no record."""
    t = frame_thickness_m(image_path)
    if t is None and fallback_m is not None and float(fallback_m) > 0:
        t = float(fallback_m)
    if t is None:
        return None
    gsd = effective_gsd(image_path).gsd
    if not gsd or gsd <= 0:
        return None
    size = _image_size(image_path)
    if not size:
        return None
    w, h = int(size[0]), int(size[1])
    px = int(round(t / gsd))
    margin = max(MARGIN_MIN_PX, int(round(MARGIN_FRACTION * px)))
    if px <= 0 or 2 * (px + margin) >= min(w, h):
        return None
    return FrameInset(thickness_m=float(t), gsd_m=float(gsd), frame_px=px,
                      margin_px=margin, width=w, height=h)


def inner_rectangle_geojson(fi: FrameInset, source_name: str = "") -> dict:
    """A one-polygon FeatureCollection of the inner rectangle, in image pixels
    ``(col, row)`` -- the frame Quadrat-mode ROIs are stored in."""
    x0, y0, x1, y1 = fi.inner
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature",
                          "properties": {"name": "inside the quadrat frame",
                                         "source": source_name,
                                         "frame_thickness_m": fi.thickness_m,
                                         "frame_px": fi.frame_px,
                                         "margin_px": fi.margin_px,
                                         "inset_px": fi.inset_px},
                          "geometry": {"type": "Polygon", "coordinates": [ring]}}]}


def write_inner_roi(image_path, out_dir, fallback_m: Optional[float] = None
                    ) -> Optional[Path]:
    """Write the inner-rectangle ROI for ``image_path`` into ``out_dir`` and
    return its path, or None when the photograph carries no frame."""
    fi = frame_inset(image_path, fallback_m)
    if fi is None:
        return None
    out = Path(out_dir) / (Path(str(image_path)).stem + "_frame_roi.geojson")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inner_rectangle_geojson(fi, Path(str(image_path)).name)),
                   encoding="utf-8")
    return out


def describe(fi: Optional[FrameInset]) -> str:
    """One line for a log or a file list."""
    if fi is None:
        return ""
    w_m, h_m = fi.inner_size_m
    return (f"frame {fi.thickness_m * 100:.1f} cm = {fi.frame_px} px + "
            f"{fi.margin_px} px margin left out on each edge; "
            f"{w_m:.2f} x {h_m:.2f} m measured")
