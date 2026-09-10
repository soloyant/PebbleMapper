"""Gauge: clast sizes from one photograph, by its GSD or by an object.

The engine behind the Digitize tab's Detect button and its *No GSD? Scale
from an object* option. A photograph that carries a GSD is measured in
millimetres (:meth:`GaugeScale.from_gsd`). One that does not contains one
or more scaling objects, drawn as line segments on the image and each
assigned to an entry of the project's library (``<project
root>/scaling_objects.json``: a name and, when known, a length in a physical
unit). Either way the detector runs on the whole frame in pixel units and
every length is divided by the pixels-per-unit scale. When every object used
has a known length the results are in a physical unit of the user's choice
(metric mode); otherwise they are in multiples of one object of unknown size
("1 boot width", custom mode). With an object scale no perspective or lens
correction is attempted; the numbers are indicative and the figures carry
:data:`DISCLAIMER`.

Pure logic: no NiceGUI. ``gui.app.build_digitize_tab`` collects the inputs
and calls :func:`run_gauge`, then :func:`write_gauge_figures`.

Coordinates
-----------
``x`` and ``y`` stay in image pixels, as Quadrat mode writes them: ``x`` is
the column of the centroid and ``y`` is measured upward from the bottom edge
(``image height - row``). ``Orientation`` is in degrees, as in every other
clast CSV: the bearing of the long axis clockwise from image-up
(:mod:`functions.clast_geometry`). The mask outlines are kept beside the CSV
in ``<stem>_gauge.contours.json`` (same pixel frame) and the overlay draws
them with each clast's major and minor axes. Scale segments stay in image
rows (y down); :func:`drop_crossing_segments` flips them into the CSV frame
to remove the detections they cross.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from functions import images as _images
from functions import units as _units

__all__ = [
    "UNITS", "CUSTOM", "UNIT_CHOICES", "DISCLAIMER",
    "LENGTH_COLUMNS", "AREA_COLUMNS", "PIXEL_COLUMNS",
    "HEIF_EXTENSIONS", "PHOTO_EXTENSIONS", "HEIF_AVAILABLE",
    "DETECT_SCALES", "DETECT_SCALE_DEFAULT", "DETECT_SCALE_HINT",
    "DETECT_SCALE_AUTO", "DETECT_SCALE_OPTIONS", "DETECT_SCALE_TOOLTIP",
    "resolve_detect_scale",
    "DETECTOR_SUMMARY_RE",
    "is_detector_summary_line", "resample_for_detection",
    "register_heif_opener", "open_photo", "needs_working_copy",
    "working_copy", "prepare_photo",
    "LIBRARY_FILENAME", "METRIC_MODE", "CUSTOM_MODE",
    "ScalingObject", "GaugeSegment", "default_library", "library_path",
    "load_library", "save_library", "find_object",
    "scale_from_segments", "GaugeScale", "result_unit_options",
    "scale_segments_path_for", "write_scale_segments", "read_scale_segments",
    "scale_from_sidecar",
    "resolve_scale", "format_length",
    "convert_pixels_to_units", "gauge_stats", "run_gauge", "write_gauge_table",
    "SEGMENT_EXCLUSION_RULE", "drop_crossing_segments", "ring_crosses_segments",
    "gauge_overlay_figure", "gauge_distribution_figure",
    "write_gauge_figures", "gauge_output_names", "backend_detect_fn",
    "METRIC_GROUP", "SAMPLE_SET_UNIT", "sample_set_names", "pool_sample_set",
    "sample_set_figure", "write_sample_set",
]

# Metres per unit for the physical units the tab offers.
UNITS: Dict[str, float] = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048,
}
# The unit whose label is the scaling object's own name.
CUSTOM = "custom"
UNIT_CHOICES: Tuple[str, ...] = (CUSTOM,) + tuple(UNITS)

DISCLAIMER = (
    "No perspective or lens correction: with a tilted camera, lengths are "
    "typically within ±8 % and up to ±25 % off, areas up to twice that. "
    "Hold the camera vertical and put the scaling object near the middle "
    "of the frame."
)

# Which CSV columns are lengths (divided by px/unit), areas (divided by
# (px/unit)²) and pixel coordinates (left alone).
LENGTH_COLUMNS: Tuple[str, ...] = (
    "Clast_length", "Clast_width", "Ellipse_major_axis",
    "Ellipse_minor_axis", "Perimeter", "Equivalent_diameter",
)
AREA_COLUMNS: Tuple[str, ...] = ("Surface_area",)
PIXEL_COLUMNS: Tuple[str, ...] = ("x", "y")

# A segment shorter than this is a slip of the mouse, not a scaling object.
_MIN_SEGMENT_PX = 1.0

# Detection scale: the working copy is resampled by this factor before the
# detector sees it. Mask R-CNN, on a 2142 x 2856 phone photograph: 1 -> 155
# clasts in 97 s, 1/2 -> 147 in 14 s, 1/3 -> 129 in 8 s, 1/4 -> 100 in 4 s,
# D50 stable within 2 px throughout. A plug-in model sets its smallest grain
# in pixels, so a half-size copy drops every clast below four times that
# area: Segment Every Grain on a 1482 px quadrat, 1 -> 1822 clasts, D50
# 17.7 mm; 1/2 -> 388, D50 29.4 mm. Auto therefore means 1/2 for Mask R-CNN
# and 1 for any other model.
DETECT_SCALES: Dict[str, float] = {"1": 1.0, "1/2": 0.5, "1/3": 1.0 / 3.0,
                                   "1/4": 0.25}
DETECT_SCALE_AUTO = "Auto"
DETECT_SCALE_OPTIONS = (DETECT_SCALE_AUTO,) + tuple(DETECT_SCALES)
DETECT_SCALE_DEFAULT = DETECT_SCALE_AUTO
DETECT_SCALE_HINT = ("Auto: half size for Mask R-CNN, full size for a "
                     "plug-in model. 1/4 is a quick preview.")
DETECT_SCALE_TOOLTIP = ("Mask R-CNN keeps almost every clast on a half-size copy "
                        "and runs several times faster. A plug-in model sets its "
                        "smallest grain in pixels, so on a half-size copy it "
                        "misses the smaller clasts: it gets the full photograph.")


def resolve_detect_scale(label, model_name: str = "maskrcnn") -> float:
    """The resampling factor a Detection scale label stands for."""
    label = str(label or DETECT_SCALE_AUTO)
    if label in DETECT_SCALES:
        return DETECT_SCALES[label]
    return 0.5 if model_name == "maskrcnn" else 1.0

# The detector's own per-image summary ("  Clast Width D90 = 7786.25 cm"):
# centimetres of a pixel-unit run mean nothing, so the Gauge console drops
# them and prints its own summary in the result unit.
DETECTOR_SUMMARY_RE = re.compile(
    r"(Clast Length|Clast Width|Equivalent Diameter)\s+D\d+\s*=\s*[-\d.,]+\s*cm")


def is_detector_summary_line(line) -> bool:
    """True for the detector's centimetre summary lines (filtered from the
    Gauge console)."""
    return bool(DETECTOR_SUMMARY_RE.search(str(line)))

# --------------------------------------------------------------------------- #
#  Photographs: HEIC and EXIF orientation live in functions.images            #
# --------------------------------------------------------------------------- #
# Re-exported so ``from functions.gauge import open_photo`` keeps working;
# the single home of photo reading is :mod:`functions.images`.
HEIF_EXTENSIONS = _images.HEIF_EXTENSIONS
PHOTO_EXTENSIONS = _images.PHOTO_EXTENSIONS      # what the tab lists
HEIF_AVAILABLE = _images.HEIF_AVAILABLE
register_heif_opener = _images.register_heif_opener
needs_working_copy = _images.needs_working_copy
working_copy = _images.working_copy
prepare_photo = _images.prepare_photo


def open_photo(path):
    """The photograph as a PIL image, upright: the EXIF orientation tag is
    applied, so an EXIF-rotated portrait shows as the camera was held.
    ``functions.images.open_photo(path, upright=True)``; the stored
    (measurement) frame is ``upright=False`` there."""
    return _images.open_photo(path, upright=True)


def resample_for_detection(image_path, factor: float, out_dir, stem: str) -> Tuple[str, int, int]:
    """Write ``<out_dir>/<stem>_gauge_detect.jpg``, the photograph resampled
    by ``factor`` (LANCZOS, RGB, quality 95); returns ``(path, W, H)`` of
    the resampled image. ``factor`` must lie in (0, 1]."""
    from PIL import Image
    f = float(factor)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"detect_scale must be in (0, 1], got {factor!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{stem}_gauge_detect.jpg"
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:            # older Pillow
        resample = Image.LANCZOS
    with Image.open(image_path) as im:
        W, H = im.size
        w = max(1, int(round(W * f)))
        h = max(1, int(round(H * f)))
        im.convert("RGB").resize((w, h), resample).save(dst, format="JPEG",
                                                        quality=95)
    return str(dst), w, h


# --------------------------------------------------------------------------- #
#  The scaling-object library                                                  #
# --------------------------------------------------------------------------- #
LIBRARY_FILENAME = "scaling_objects.json"
METRIC_MODE = "metric"
CUSTOM_MODE = "custom"


@dataclass
class ScalingObject:
    """One object of the project's library: a name, and its length in a
    physical unit when known. ``length`` and ``unit`` may both be None (size
    unknown); such an object can only define a custom unit."""
    name: str = "scale bar"
    length: Optional[float] = None
    unit: Optional[str] = None

    @property
    def metres(self) -> Optional[float]:
        """The object's length in metres, or None when it is unknown."""
        if self.length is None or self.unit not in UNITS:
            return None
        try:
            v = float(self.length)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(v) and v > 0):
            return None
        return v * UNITS[self.unit]

    @property
    def is_metric(self) -> bool:
        return self.metres is not None

    def label(self) -> str:
        """``"boot width (0.27 m)"`` or ``"boot width (size unknown)"``."""
        if self.is_metric:
            return f"{self.name} ({format_length(self.length, self.unit)})"
        return f"{self.name} (size unknown)"

    def to_dict(self) -> dict:
        return {"name": self.name,
                "length": (float(self.length) if self.length not in (None, "")
                           else None),
                "unit": self.unit if self.unit in UNITS else None}

    @classmethod
    def from_dict(cls, d: dict) -> Optional["ScalingObject"]:
        """A library entry, or None when the dict has no usable name."""
        if not isinstance(d, dict):
            return None
        name = str(d.get("name") or "").strip()
        if not name:
            return None
        length = d.get("length")
        try:
            length = float(length) if length not in (None, "") else None
        except (TypeError, ValueError):
            length = None
        if length is not None and not (math.isfinite(length) and length > 0):
            length = None
        unit = d.get("unit")
        unit = unit if unit in UNITS else None
        return cls(name=name, length=length, unit=unit)


def default_library() -> List[ScalingObject]:
    """What a project starts with: one object of unknown size."""
    return [ScalingObject("scale bar", None, None)]


def library_path(project_root) -> Path:
    return Path(project_root) / LIBRARY_FILENAME


def load_library(project_root) -> List[ScalingObject]:
    """The project's scaling objects from ``<project root>/scaling_objects.json``;
    the default library when the file is absent, unreadable or empty.
    Malformed entries are skipped."""
    p = library_path(project_root)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default_library()
    if isinstance(raw, dict):
        raw = raw.get("objects", [])
    objs: List[ScalingObject] = []
    seen = set()
    for d in (raw if isinstance(raw, list) else []):
        o = ScalingObject.from_dict(d)
        if o is not None and o.name not in seen:
            seen.add(o.name)
            objs.append(o)
    return objs or default_library()


def save_library(project_root, objects: Sequence) -> Path:
    """Write the library (``ScalingObject`` instances or dicts) to
    ``<project root>/scaling_objects.json``; returns the path. Entries
    without a name are dropped; an empty library is written as the default."""
    objs: List[ScalingObject] = []
    seen = set()
    for o in objects or []:
        so = o if isinstance(o, ScalingObject) else ScalingObject.from_dict(o)
        if so is None or so.name in seen:
            continue
        seen.add(so.name)
        objs.append(so)
    if not objs:
        objs = default_library()
    p = library_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([o.to_dict() for o in objs], indent=2),
                 encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
#  A photograph's scale segments, kept beside its Digitize CSV                 #
# --------------------------------------------------------------------------- #
SEGMENTS_FORMAT = "pebblemapper-scale-segments"
SEGMENTS_VERSION = 1


def scale_segments_path_for(truth_csv) -> Path:
    """``<stem>.scale.json`` beside a Digitize CSV (``IMG_1_truth.csv`` ->
    ``IMG_1_truth.scale.json``)."""
    p = Path(str(truth_csv))
    stem = p.name[:-4] if p.name.lower().endswith(".csv") else p.name
    return p.with_name(stem + ".scale.json")


def scale_segments_document(image_name: str, segments: Sequence,
                            library: Sequence = ()) -> dict:
    """The sidecar body: the segments ``[{"p0", "p1", "object"}]`` in image
    pixels and, so the file stands alone without a project, the library
    entries they name."""
    segs = []
    for s in segments or []:
        a, b = s.get("p0"), s.get("p1")
        if not a or not b:
            continue
        segs.append({"p0": [float(a[0]), float(a[1])],
                     "p1": [float(b[0]), float(b[1])],
                     "object": str(s.get("object") or "")})
    used = {s["object"] for s in segs}
    objs, seen = [], set()
    for o in library or []:
        so = o if isinstance(o, ScalingObject) else ScalingObject.from_dict(o)
        if so is not None and so.name in used and so.name not in seen:
            seen.add(so.name)
            objs.append(so.to_dict())
    return {"format": SEGMENTS_FORMAT, "version": SEGMENTS_VERSION,
            "image": str(image_name), "segments": segs, "objects": objs}


def write_scale_segments(truth_csv, image_name: str, segments: Sequence,
                         library: Sequence = ()) -> Optional[Path]:
    """Write the sidecar for ``truth_csv``; with no segments, remove it.
    Returns the path written, or None."""
    p = scale_segments_path_for(truth_csv)
    doc = scale_segments_document(image_name, segments, library)
    if not doc["segments"]:
        try:
            p.unlink()
        except OSError:
            pass
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_scale_segments(truth_csv) -> Optional[dict]:
    """``{"segments": [...], "objects": [ScalingObject dicts]}`` from the
    sidecar beside ``truth_csv``, or None when there is none or it is
    unreadable. Malformed segments are skipped."""
    p = scale_segments_path_for(truth_csv)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    clean = scale_segments_document(doc.get("image", ""),
                                    doc.get("segments") if isinstance(doc.get("segments"), list) else [],
                                    [])
    objs = []
    for d in doc.get("objects") or []:
        so = ScalingObject.from_dict(d) if isinstance(d, dict) else None
        if so is not None:
            objs.append(so.to_dict())
    return {"segments": clean["segments"], "objects": objs}


def scale_from_sidecar(csv_candidates: Sequence, library: Sequence = (),
                       result_unit: Optional[str] = None):
    """The scale a photograph's saved segments give, for a tab that does not
    draw them (Detect).

    ``csv_candidates`` are the Digitize CSV paths the photograph could have
    (the project's ``validation/<stem>_truth.csv``, else the one beside the
    image); the first that has a ``.scale.json`` beside it wins. The
    project's ``library`` is the current word on an object's known length;
    the copy the sidecar carries fills in the objects the library does not
    have, so the file still stands alone. The library wins, so a length
    given in Digitize after the segment was drawn reaches Detect.
    Returns ``(scale, sidecar_path)``, or
    ``(None, None)`` when there is no sidecar, and ``(None, path)`` when the
    segments cannot give a scale.
    """
    for csv in csv_candidates or []:
        doc = read_scale_segments(csv)
        if not doc or not doc["segments"]:
            continue
        path = scale_segments_path_for(csv)
        segs = [GaugeSegment((s["p0"][0], s["p0"][1]), (s["p1"][0], s["p1"][1]),
                             s.get("object") or "")
                for s in doc["segments"]]
        lib = []
        for o in library or []:
            so = o if isinstance(o, ScalingObject) else ScalingObject.from_dict(o)
            if so is not None:
                lib.append(so)
        names = {o.name for o in lib}
        for d in doc["objects"]:
            so = ScalingObject.from_dict(d)
            if so is not None and so.name not in names:
                lib.append(so)
        try:
            return resolve_scale(segs, lib, result_unit), path
        except ValueError:
            return None, path
    return None, None


def find_object(library: Sequence[ScalingObject], name: str) -> Optional[ScalingObject]:
    for o in library:
        if o.name == name:
            return o
    return None


# --------------------------------------------------------------------------- #
#  Segments and the scale                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class GaugeSegment:
    """One drawn segment in image pixels and the library object it spans."""
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    object_name: str = ""

    @property
    def px_length(self) -> float:
        return math.hypot(float(self.p1[0]) - float(self.p0[0]),
                          float(self.p1[1]) - float(self.p0[1]))

    def to_dict(self) -> dict:
        return {"p0": [float(self.p0[0]), float(self.p0[1])],
                "p1": [float(self.p1[0]), float(self.p1[1])],
                "object": self.object_name, "px_length": self.px_length}


def scale_from_segments(segments_px: Sequence, object_length: float = 1.0) -> dict:
    """Pixels per unit from one or more drawn segments of ONE object.

    ``segments_px`` is ``[((x0, y0), (x1, y1)), ...]`` in image pixels, each
    spanning the object once; ``object_length`` is the object's length in
    the target unit (1.0 when the object *is* the unit).

    Returns ``{"px_per_unit", "per_segment", "spread", "n"}`` where
    ``px_per_unit`` is the mean over segments and ``spread`` is
    ``(max - min) / mean`` (0.0 for a single segment). A spread above a few
    per cent means the camera was tilted or the object was not drawn end to
    end. Raises ``ValueError`` with no segment, a non-positive length, or a
    degenerate (sub-pixel) segment.
    """
    segs = list(segments_px or [])
    if not segs:
        raise ValueError("no scaling segment: draw the scaling object on "
                         "the photograph first")
    try:
        length = float(object_length)
    except (TypeError, ValueError):
        raise ValueError(f"object length must be a number, got {object_length!r}")
    if not (math.isfinite(length) and length > 0):
        raise ValueError(f"object length must be > 0, got {object_length!r}")
    per = []
    for i, seg in enumerate(segs, start=1):
        try:
            (x0, y0), (x1, y1) = seg
            d = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
        except (TypeError, ValueError):
            raise ValueError(f"segment {i} is not a pair of (x, y) points: {seg!r}")
        if not (math.isfinite(d) and d >= _MIN_SEGMENT_PX):
            raise ValueError(f"segment {i} is degenerate ({d:.2f} px long)")
        per.append(d / length)
    mean = float(sum(per) / len(per))
    spread = float((max(per) - min(per)) / mean) if len(per) > 1 else 0.0
    return {"px_per_unit": mean, "per_segment": per, "spread": spread,
            "n": len(per)}


@dataclass
class GaugeScale:
    """The scale a run is expressed in.

    ``mode`` is ``"metric"`` (every object used on the photograph has a known
    physical length; ``unit`` is a key of :data:`UNITS`) or ``"custom"`` (the
    unit is one object of unknown size, named by ``object_name``).
    ``px_per_unit`` and ``spread`` come from the segments; ``segments`` keeps
    each drawn segment with its object and pixel length for the figure and
    the sidecar, ``segments_px`` the bare point pairs, ``ignored`` the names
    of the objects whose segments could not contribute (custom mode, other
    unknown objects) and ``note`` says so in words.
    """
    object_name: str = "scale bar"
    object_length: float = 1.0
    unit: str = CUSTOM
    px_per_unit: float = float("nan")
    spread: float = 0.0
    segments_px: list = field(default_factory=list)
    mode: str = CUSTOM_MODE
    segments: list = field(default_factory=list)     # [GaugeSegment]
    ignored: list = field(default_factory=list)      # object names
    note: str = ""

    @classmethod
    def from_segments(cls, segments_px: Sequence, *, object_name: str = "scale bar",
                      object_length: float = 1.0, unit: str = CUSTOM) -> "GaugeScale":
        """One object drawn one or more times; raises like
        :func:`scale_from_segments`. ``unit`` a key of ``UNITS`` means
        ``object_length`` is in that unit and the results are too."""
        if unit not in UNIT_CHOICES:
            raise ValueError(f"unit must be one of {UNIT_CHOICES}, got {unit!r}")
        res = scale_from_segments(segments_px, object_length)
        name = str(object_name or "").strip() or "scale bar"
        segs = [GaugeSegment((float(a[0]), float(a[1])),
                             (float(b[0]), float(b[1])), name)
                for a, b in segments_px]
        return cls(object_name=name, object_length=float(object_length),
                   unit=unit, px_per_unit=res["px_per_unit"],
                   spread=res["spread"],
                   segments_px=[(s.p0, s.p1) for s in segs],
                   mode=METRIC_MODE if unit in UNITS else CUSTOM_MODE,
                   segments=segs)

    @classmethod
    def from_gsd(cls, metres_per_px: float, unit: str = "mm") -> "GaugeScale":
        """The scale a photograph's own GSD gives: metric mode in ``unit``
        (a ``UNITS`` key), no segments, no spread. Raises ``ValueError``
        for a non-positive GSD or an unknown unit."""
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {tuple(UNITS)}, got {unit!r}")
        try:
            g = float(metres_per_px)
        except (TypeError, ValueError):
            raise ValueError(f"GSD must be a number, got {metres_per_px!r}")
        if not (math.isfinite(g) and g > 0):
            raise ValueError(f"GSD must be > 0, got {metres_per_px!r}")
        return cls(object_name="", object_length=1.0, unit=unit,
                   px_per_unit=UNITS[unit] / g, spread=0.0,
                   mode=METRIC_MODE)

    @property
    def unit_label(self) -> str:
        """The unit's name in labels: the ``UNITS`` key, or the object's
        name for the custom unit."""
        if self.unit in UNITS:
            return self.unit
        return (self.object_name or "").strip() or "unit"

    @property
    def is_metric(self) -> bool:
        """True when the unit is a physical length (a key of ``UNITS``,
        imperial included), so millimetres, φ and a scale bar can be
        derived; False for the custom unit."""
        return self.unit in UNITS

    @property
    def metres_per_unit(self) -> Optional[float]:
        return UNITS.get(self.unit)

    @property
    def metres_per_px(self) -> Optional[float]:
        """Metres per image pixel, or None for the custom unit or before a
        segment was drawn."""
        m = self.metres_per_unit
        if m is None or not (self.px_per_unit > 0):
            return None
        return m / float(self.px_per_unit)

    @property
    def valid(self) -> bool:
        try:
            return bool(self.px_per_unit > 0) and math.isfinite(self.px_per_unit)
        except TypeError:
            return False

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "object_name": self.object_name,
            "object_length": float(self.object_length),
            "unit": self.unit,
            "unit_label": self.unit_label,
            "px_per_unit": float(self.px_per_unit),
            "spread": float(self.spread),
            "metres_per_px": self.metres_per_px,
            "segments": [s.to_dict() for s in self.segments],
            "segments_px": [[[a[0], a[1]], [b[0], b[1]]]
                            for a, b in self.segments_px],
            "ignored": list(self.ignored),
            "note": self.note,
        }


def result_unit_options(segments: Sequence[GaugeSegment],
                        library: Sequence[ScalingObject]) -> dict:
    """What the *Result unit* select may offer for these segments.

    Returns ``{"mode", "options", "default", "unknown", "missing"}``:
    ``mode`` is ``"metric"`` when every object used has a known length (the
    options are then the ``UNITS`` keys, default ``"mm"``) and ``"custom"``
    otherwise (the options are the names of the unknown objects used, in
    first-use order, the first being the default). ``missing`` lists segment
    object names absent from the library. With no segment the default
    follows the library: metric (mm) when any object has a known length,
    else custom with the library's objects, the first being the default."""
    used: List[str] = []
    for s in segments:
        if s.object_name and s.object_name not in used:
            used.append(s.object_name)
    if not used:
        if any(o.is_metric for o in library):
            return {"mode": METRIC_MODE, "options": list(UNITS),
                    "default": "mm", "unknown": [], "missing": []}
        names = [o.name for o in library if o.name]
        return {"mode": CUSTOM_MODE, "options": names,
                "default": names[0] if names else "", "unknown": names,
                "missing": []}
    missing = [n for n in used if find_object(library, n) is None]
    unknown = [n for n in used
               if find_object(library, n) is not None
               and not find_object(library, n).is_metric]
    if used and (unknown or missing):
        opts = list(unknown) + [n for n in missing if n not in unknown]
        return {"mode": CUSTOM_MODE, "options": opts,
                "default": opts[0] if opts else "", "unknown": unknown,
                "missing": missing}
    return {"mode": METRIC_MODE, "options": list(UNITS), "default": "mm",
            "unknown": [], "missing": missing}


def resolve_scale(segments: Sequence[GaugeSegment],
                  library: Sequence[ScalingObject],
                  result_unit: Optional[str] = None) -> GaugeScale:
    """The scale rule.

    Metric mode (every object used has a known length): each segment gives
    pixels per metre from its own object, the mean is the scale, the spread
    between segments is the tilt warning, and ``result_unit`` (a ``UNITS``
    key, default mm) is the output unit.

    Custom mode (some object used has no known length): the output unit is
    one unknown object, ``result_unit`` when it names one of them, else the
    first one drawn. Only that object's segments set the scale; segments of
    other unknown objects are listed in ``ignored`` with a ``note``, since
    their ratio to the unit is unknown. Known objects drawn alongside are
    ignored too (their metres cannot be expressed in the custom unit).

    Raises ``ValueError`` with no segment, a degenerate segment, an object
    missing from the library, or an invalid ``result_unit``.
    """
    segs = list(segments or [])
    if not segs:
        raise ValueError("no scaling segment: draw the scaling object on "
                         "the photograph first")
    for i, s in enumerate(segs, start=1):
        if not (math.isfinite(s.px_length) and s.px_length >= _MIN_SEGMENT_PX):
            raise ValueError(f"segment {i} is degenerate ({s.px_length:.2f} px long)")
        if not s.object_name:
            raise ValueError(f"segment {i} has no scaling object")
        if find_object(library, s.object_name) is None:
            raise ValueError(f"segment {i}: {s.object_name!r} is not in the "
                             "scaling-object library")
    opts = result_unit_options(segs, library)
    pts = [(s.p0, s.p1) for s in segs]

    if opts["mode"] == METRIC_MODE:
        unit = result_unit if result_unit in UNITS else "mm"
        per_m = [s.px_length / find_object(library, s.object_name).metres
                 for s in segs]
        mean_m = float(sum(per_m) / len(per_m))
        spread = float((max(per_m) - min(per_m)) / mean_m) if len(per_m) > 1 else 0.0
        return GaugeScale(object_name="", object_length=1.0, unit=unit,
                          px_per_unit=mean_m * UNITS[unit], spread=spread,
                          segments_px=pts, mode=METRIC_MODE, segments=segs)

    unknown = opts["options"]
    if result_unit in unknown:
        chosen = result_unit
    elif result_unit in (None, ""):
        chosen = unknown[0]
    else:
        raise ValueError(f"result unit {result_unit!r} is not one of the "
                         f"unknown objects used: {unknown}")
    own = [s for s in segs if s.object_name == chosen]
    per = [s.px_length for s in own]
    mean = float(sum(per) / len(per))
    spread = float((max(per) - min(per)) / mean) if len(per) > 1 else 0.0
    ignored: List[str] = []
    for s in segs:
        if s.object_name != chosen and s.object_name not in ignored:
            ignored.append(s.object_name)
    note = ""
    if ignored:
        note = (f"Segments of {', '.join(ignored)} do not set the scale: "
                f"their ratio to 1 {chosen} is unknown.")
    return GaugeScale(object_name=chosen, object_length=1.0, unit=CUSTOM,
                      px_per_unit=mean, spread=spread, segments_px=pts,
                      mode=CUSTOM_MODE, segments=segs, ignored=ignored,
                      note=note)


# --------------------------------------------------------------------------- #
#  Formatting                                                                  #
# --------------------------------------------------------------------------- #
def format_length(value: float, unit_label: str = "", *, short: bool = False) -> str:
    """A length with sensible rounding: ``"63 mm"``, ``"0.71 boot width"``;
    ``short=True`` drops one decimal (``"0.7 boot width"``) for annotations."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    a = abs(v)
    if abs(v - round(v)) < 1e-9:
        dec = 0                       # "1 boot width", "30 cm"
    elif short:
        dec = 0 if a >= 10 else (1 if a >= 0.1 else (2 if a >= 0.01 else 3))
    else:
        dec = 0 if a >= 100 else (1 if a >= 10 else (2 if a >= 0.1 else 3))
    txt = f"{v:,.{dec}f}"
    return f"{txt} {unit_label}".strip()


# --------------------------------------------------------------------------- #
#  Conversion and statistics                                                   #
# --------------------------------------------------------------------------- #
def convert_pixels_to_units(df_px: pd.DataFrame, scale: GaugeScale) -> pd.DataFrame:
    """A copy of a pixel-unit clast table expressed in ``scale``'s unit:
    length columns divided by ``px_per_unit``, area columns by its square,
    ``x``/``y`` untouched, plus a ``unit`` column holding the unit label."""
    if not scale.valid:
        raise ValueError("the scale has no pixels-per-unit value")
    ppu = float(scale.px_per_unit)
    df = df_px.copy()
    for c in LENGTH_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / ppu
    for c in AREA_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / (ppu ** 2)
    df["unit"] = scale.unit_label
    return df


def gauge_stats(df: pd.DataFrame, scale: GaugeScale, column: str = "Clast_length") -> dict:
    """``n``, ``D16``, ``D50``, ``D84``, ``mean`` in the unit, plus
    ``sorting_phi`` (Folk–Ward graphic sorting) when the unit is a physical
    length and n ≥ 2; NaN otherwise."""
    nan = float("nan")
    out = {"n": 0, "D16": nan, "D50": nan, "D84": nan, "mean": nan,
           "sorting_phi": nan, "unit": scale.unit_label, "column": column}
    if column not in df.columns:
        return out
    v = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    out["n"] = int(len(v))
    if len(v) == 0:
        return out
    out["D16"], out["D50"], out["D84"] = (
        float(np.quantile(v, q)) for q in (0.16, 0.5, 0.84))
    out["mean"] = float(v.mean())
    if scale.is_metric and len(v) >= 2:
        phi = -np.log2(v * scale.metres_per_unit / _units.PHI_REF_M)
        p5, p16, p84, p95 = np.percentile(phi, [5, 16, 84, 95])
        out["sorting_phi"] = _units.folk_ward_sorting_phi(p5, p16, p84, p95)
    return out


# --------------------------------------------------------------------------- #
#  Detections crossing a scale segment                                         #
# --------------------------------------------------------------------------- #
SEGMENT_EXCLUSION_RULE = (
    "Every detected clast whose outline (its mask contour, else its fitted "
    "ellipse) intersects a scale segment drawn on the photograph is "
    "removed: it is the scaling object itself or a clast cut by it.")


def _segment_points(seg) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """``((x0, y0), (x1, y1))`` in image pixels from a :class:`GaugeSegment`,
    a ``{"p0", "p1"}`` dict or a pair of points; None when unusable."""
    try:
        if isinstance(seg, GaugeSegment):
            a, b = seg.p0, seg.p1
        elif isinstance(seg, dict):
            a, b = seg["p0"], seg["p1"]
        else:
            a, b = seg
        pts = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for p in pts for v in p):
        return None
    return pts


def _segment_geometry(p0, p1):
    from shapely.geometry import LineString, Point
    if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-9:
        return Point(p0)
    return LineString([p0, p1])


def _polygon(ring):
    """A valid shapely polygon from ``[[x, y], ...]``, or None."""
    from shapely.geometry import Polygon
    try:
        arr = np.asarray(ring, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3 \
                or not np.isfinite(arr).all():
            return None
        poly = Polygon(arr)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return None if poly.is_empty else poly
    except Exception:
        return None


def ring_crosses_segments(ring, segments) -> bool:
    """True when the closed outline ``ring`` (``[[x, y], ...]``) intersects
    any of ``segments``, both in the same frame (image columns and rows for
    an editable proposal)."""
    poly = _polygon(ring)
    if poly is None:
        return False
    for seg in segments or []:
        pts = _segment_points(seg)
        if pts is not None and _segment_geometry(*pts).intersects(poly):
            return True
    return False


def drop_crossing_segments(df: pd.DataFrame, contours, segments, *,
                           image_height: float) -> Tuple[pd.DataFrame, List[int]]:
    """The clasts of ``df`` that no scale segment crosses, and the
    ``clast_ID`` of those removed.

    ``df`` is a clast table in the gauge CSV frame (``x`` = column, ``y`` =
    ``image_height - row``), ``contours`` its outlines (``{clast_ID: [[x,
    y], ...]}`` in that same frame, or None) and ``segments`` the scale
    segments in image pixels (column, row), as drawn on the photograph. A
    clast is removed when its outline polygon intersects a segment
    (shapely ``intersects``: crossing, touching and containing all count);
    a clast without an outline is tested with the ellipse of its
    ``Ellipse_major_axis`` / ``Ellipse_minor_axis`` (``Clast_length`` /
    ``Clast_width`` when those are missing) and ``Orientation``. A row
    with neither is kept. Kept rows keep their ``clast_ID``; the returned
    frame has a fresh index and, when ``contours`` was given, carries the
    kept outlines in ``attrs["contours"]``. With no usable segment ``df``
    is returned unchanged with ``[]``."""
    from functions import clast_geometry as _CG
    H = float(image_height)
    geoms = []
    for seg in segments or []:
        pts = _segment_points(seg)
        if pts is None:
            continue
        (x0, r0), (x1, r1) = pts
        geoms.append(_segment_geometry((x0, H - r0), (x1, H - r1)))
    if not geoms or df is None or len(df) == 0:
        return df, []

    def _num(row, *names):
        for n in names:
            if n in row.index:
                try:
                    v = float(row[n])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    return v
        return float("nan")

    keep_mask = []
    removed: List[int] = []
    for _pos, row in df.iterrows():
        cid = None
        if "clast_ID" in df.columns:
            try:
                cid = int(row["clast_ID"])
            except (TypeError, ValueError):
                cid = None
        poly = None
        if contours is not None and cid is not None and cid in contours:
            poly = _polygon(contours[cid])
        if poly is None:
            x, y = _num(row, "x"), _num(row, "y")
            major = _num(row, "Ellipse_major_axis", "Clast_length")
            minor = _num(row, "Ellipse_minor_axis", "Clast_width")
            o = _num(row, "Orientation")
            if not math.isfinite(o):
                o = 0.0
            if all(math.isfinite(v) for v in (x, y, major, minor)) \
                    and major > 0 and minor > 0:
                poly = _polygon(_CG.ellipse_outline(x, y, major, minor, o,
                                                    y_down=False))
        crossed = poly is not None and any(g.intersects(poly) for g in geoms)
        keep_mask.append(not crossed)
        if crossed:
            removed.append(cid if cid is not None else int(_pos))
    kept = df.loc[np.asarray(keep_mask, dtype=bool)].reset_index(drop=True)
    kept.attrs = {k: v for k, v in df.attrs.items() if k != "contours"}
    if contours is not None:
        gone = set(removed)
        _CG.attach_contours(
            kept, _CG.ContourSet({int(k): v for k, v in contours.items()
                                  if int(k) not in gone},
                                 frame=getattr(contours, "frame", "pixels")),
            frame=getattr(contours, "frame", "pixels"))
    return kept, removed


FRAME_EXCLUSION_RULE = (
    "a detection whose centroid lies on the quadrat frame band (the "
    "PM_FRAME_THICKNESS_M of the rectification record, in pixels at the "
    "photograph's GSD) is not a clast of the bed and is removed")


def drop_outside_frame(df: pd.DataFrame, contours, frame_inset
                       ) -> Tuple[pd.DataFrame, List[int]]:
    """The clasts of ``df`` whose centroid lies inside the quadrat frame,
    and the ``clast_ID`` of those removed.

    ``df`` is a clast table in the gauge CSV frame (``x`` = column, ``y`` =
    ``image_height - row``); ``frame_inset`` a
    :class:`functions.quadrat_frame.FrameInset` of the same photograph. A
    row without a finite centroid is kept. Kept rows keep their
    ``clast_ID``; the frame returned has a fresh index and, when
    ``contours`` was given, carries the kept outlines."""
    from functions import clast_geometry as _CG
    if frame_inset is None or df is None or len(df) == 0:
        return df, []
    H = float(frame_inset.height)
    keep_mask, removed = [], []
    for _pos, row in df.iterrows():
        cid = None
        if "clast_ID" in df.columns:
            try:
                cid = int(row["clast_ID"])
            except (TypeError, ValueError):
                cid = None
        try:
            x, y = float(row["x"]), float(row["y"])
        except (KeyError, TypeError, ValueError):
            x = y = float("nan")
        outside = (math.isfinite(x) and math.isfinite(y)
                   and not frame_inset.contains(x, H - y))
        keep_mask.append(not outside)
        if outside:
            removed.append(cid if cid is not None else int(_pos))
    kept = df.loc[np.asarray(keep_mask, dtype=bool)].reset_index(drop=True)
    kept.attrs = {k: v for k, v in df.attrs.items() if k != "contours"}
    if contours is not None:
        gone = set(removed)
        _CG.attach_contours(
            kept, _CG.ContourSet({int(k): v for k, v in contours.items()
                                  if int(k) not in gone},
                                 frame=getattr(contours, "frame", "pixels")),
            frame=getattr(contours, "frame", "pixels"))
    return kept, removed


def frame_inset_for(image_path, source_image=None):
    """The quadrat frame of the photograph being measured, or None: read
    from the photograph the user picked first (its sidecar is beside it),
    then from the detection copy."""
    from functions import quadrat_frame as _qf
    for cand in (source_image, image_path):
        if cand is None:
            continue
        try:
            fi = _qf.frame_inset(cand)
        except Exception:
            fi = None
        if fi is not None:
            return fi
    return None


def _image_height(image_path) -> Optional[int]:
    """The stored frame's height in pixels (what the detector sees), or
    None when the photograph cannot be opened."""
    try:
        from PIL import Image
        _images.register_heif_opener()
        with Image.open(image_path) as im:      # the header only, no decode
            return int(im.size[1])
    except Exception:
        return None


def gauge_output_names(stem: str) -> dict:
    """The four files a run writes for ``stem``."""
    return {
        "csv": f"{stem}_gauge.csv",
        "json": f"{stem}_gauge.json",
        "overlay": f"{stem}_gauge_overlay.png",
        "distribution": f"{stem}_gauge_distribution.png",
        "contours": f"{stem}_gauge.contours.json",
    }


# --------------------------------------------------------------------------- #
#  The run                                                                     #
# --------------------------------------------------------------------------- #
def run_gauge(image_path, scale: GaugeScale, out_dir, *,
              min_confidence: Optional[float] = 0.7,
              devicemode: str = "gpu", devicenumber: int = 0,
              log_fn: Optional[Callable[[str], None]] = None,
              stop_check: Optional[Callable[[], bool]] = None,
              detect_fn: Optional[Callable] = None,
              out_stem: Optional[str] = None,
              model: str = "maskrcnn",
              source_image=None,
              detect_scale: float = 1.0,
              scale_source: Optional[str] = None,
              disclaimer: Optional[str] = DISCLAIMER,
              exclude_segments: bool = True,
              segments: Optional[Sequence] = None,
              exclude_frame: bool = True,
              write: bool = True) -> dict:
    """Detect on the whole photograph in pixel units, convert to the unit,
    write ``<stem>_gauge.csv`` and ``<stem>_gauge.json`` under ``out_dir``.

    ``scale_source`` names where the scale came from in the sidecar
    (``"object"`` for drawn segments, ``"gsd:filename"`` / ``"gsd:sidecar"``
    for the photograph's own GSD; by default ``"object"`` when the scale has
    segments, else ``"gsd"``). ``disclaimer`` is stored in the sidecar; pass
    None for a GSD-scaled run, which needs none.

    ``detect_fn`` defaults to ``functions.clasts_detection.clasts_detect_jobs``
    and is called with ``mode="quadrat"``, one job, ``resolution=1.0`` (pixel
    units), ``saveresults=False`` and ``plot=False``; tests inject a stand-in
    returning a pixel-unit DataFrame. ``out_stem`` overrides the image stem
    (pass ``naming.origin_stem(...)`` for site- and date-prefixed names).
    ``source_image`` is the photograph the user picked when ``image_path``
    is its :func:`working_copy`; the sidecar records both. ``detect_scale``
    in (0, 1] resamples the photograph by that factor before detection
    (:func:`resample_for_detection`) and runs at ``resolution =
    1 / detect_scale``, so every measurement comes back in original-image
    pixels; ``x`` and ``y`` are rescaled likewise. The resampled file is
    removed afterwards; the factor is recorded in the sidecar.

    The detector's mask outlines (``df.attrs["contours"]`` of the frame it
    returns, see :mod:`functions.clast_geometry`) are brought back to
    original pixels and written to ``<stem>_gauge.contours.json``; a
    detector that returns none leaves no sidecar (a stale one is removed).
    ``model`` is the backend name recorded in the sidecar.

    Scale segments: with ``exclude_segments`` (the default) every detected
    clast whose outline intersects a scale segment is dropped before the
    statistics, the CSV, the contours sidecar and the figures
    (:func:`drop_crossing_segments`), whatever scale is in force.
    ``segments`` (image pixels, as drawn) names them explicitly, e.g. the
    photograph's segments on a GSD-scaled run; by default they are
    ``scale.segments``. Kept clasts keep their ``clast_ID``; the sidecar
    records ``excluded_by_segments`` (the count), ``excluded_clast_IDs`` and
    ``exclusion_rule``.

    ``write=False`` writes nothing (Digitize's Detect: the proposals go on
    the canvas and *Figures* writes the table later, through
    :func:`write_gauge_table`); ``csv``, ``json`` and ``contours`` are then
    None and ``sidecar`` holds what the JSON would have carried.

    Returns ``{"csv", "json", "df", "stats", "stem", "out_dir", "contours",
    "model", "excluded_by_segments", "removed_ids", "sidecar"}``; the
    DataFrame is the converted table, carrying the outlines in its
    ``attrs``. Raises ``RuntimeError`` when the
    detector reports an error or the run was stopped before detection.
    """
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"photograph not found: {image_path}")
    if not scale.valid:
        raise ValueError("no scale: draw the scaling object on the "
                         "photograph first")
    out_dir = Path(out_dir)
    stem = str(out_stem or Path(image_path).stem)
    names = gauge_output_names(stem)
    log = log_fn or (lambda _s: None)

    if detect_fn is None:
        from functions.clasts_detection import clasts_detect_jobs as detect_fn

    outcome: Dict[str, str] = {}

    def _progress(_ji, event, payload):
        if event == "error":
            outcome["error"] = str((payload or {}).get("message")
                                   or "detection failed")
        elif event == "stopped":
            outcome["stopped"] = "stopped"

    log(f"[gauge] {Path(image_path).name}: {scale.px_per_unit:.2f} px per "
        f"{scale.unit_label} from {len(scale.segments_px)} segment(s), "
        f"{scale.mode} mode"
        + (f", segments disagree by {scale.spread * 100:.1f} %"
           if len(scale.segments_px) > 1 else ""))
    for k, seg in enumerate(scale.segments, start=1):
        log(f"[gauge]   segment {k}: {seg.px_length:.1f} px = {seg.object_name}")
    if scale.note:
        log(f"[gauge] {scale.note}")
    if scale.metres_per_px is not None:
        log(f"[gauge] 1 px = {scale.metres_per_px * 1000:.3f} mm")
    conf_txt = (f"{float(min_confidence):.2f}" if min_confidence is not None
                else "off")
    log(f"[gauge] detecting on the whole frame (min confidence {conf_txt}, "
        f"{devicemode} #{devicenumber})")

    factor = float(detect_scale)
    if not (0.0 < factor <= 1.0):
        raise ValueError(f"detect_scale must be in (0, 1], got {detect_scale!r}")
    detect_path, resampled = image_path, None
    if factor != 1.0:
        detect_path, w, h = resample_for_detection(image_path, factor, out_dir, stem)
        resampled = (w, h)
        log(f"[gauge] detection scale {factor:.3g}: resampled to {w}×{h} px, "
            f"measurements reported in original pixels")
    try:
        frames = detect_fn(
            mode="quadrat", jobs=[{"path": detect_path}],
            resolution=1.0 / factor,
            plot=False, saveplot=False, saveresults=False,
            devicemode=devicemode, devicenumber=devicenumber,
            min_confidence=min_confidence, stop_check=stop_check,
            progress_callback=_progress)
    finally:
        if resampled is not None:
            try:
                os.remove(detect_path)
            except OSError:
                pass
    if "error" in outcome:
        raise RuntimeError(outcome["error"])
    if "stopped" in outcome:
        raise RuntimeError("stopped before detection")
    df_px = frames[0] if frames else pd.DataFrame()
    if df_px is None:
        df_px = pd.DataFrame()
    from functions import clast_geometry as _CG
    contours = _CG.contours_of(df_px)
    if factor != 1.0 and len(df_px):
        # Lengths already carry ``resolution``; the centroid does not.
        df_px = df_px.copy()
        for c in PIXEL_COLUMNS:
            if c in df_px.columns:
                df_px[c] = pd.to_numeric(df_px[c], errors="coerce") / factor
        if contours is not None:
            contours = _CG.transform_contours(
                contours, lambda xs, ys: (np.round(xs / factor, 2),
                                          np.round(ys / factor, 2)))

    # Detections on a scaling object, or cut by its segment, are not clasts
    # of the bed: drop them before anything is counted or written.
    excl_segments = []
    if exclude_segments:
        src = scale.segments if segments is None else segments
        excl_segments = [p for p in (_segment_points(s) for s in (src or []))
                         if p is not None]
    removed_ids: List[int] = []
    if excl_segments and len(df_px):
        height = _image_height(image_path)
        if height is None:
            log("[gauge] could not read the photograph's height: detections "
                "crossing a scale segment were NOT removed")
        else:
            df_px, removed_ids = drop_crossing_segments(
                df_px, contours, excl_segments, image_height=height)
            if contours is not None:
                contours = _CG.contours_of(df_px)
            if removed_ids:
                log(f"[gauge] {len(removed_ids)} detection(s) crossing a scale "
                    f"segment removed (clast_ID "
                    f"{', '.join(str(i) for i in removed_ids)})")
            else:
                log("[gauge] no detection crosses a scale segment")

    # Nor are detections on the quadrat frame: a photograph Orthorectify
    # rectified carries the frame's thickness in its record, and whatever
    # sits on that band is a piece of the frame, not of the bed.
    frame_removed: List[int] = []
    frame = frame_inset_for(image_path, source_image) if exclude_frame else None
    if frame is not None and len(df_px):
        from functions import quadrat_frame as _qf
        df_px, frame_removed = drop_outside_frame(df_px, contours, frame)
        if contours is not None:
            contours = _CG.contours_of(df_px)
        log(f"[gauge] {_qf.describe(frame)}; {len(frame_removed)} detection(s) "
            f"on the frame removed"
            + (f" (clast_ID {', '.join(str(i) for i in frame_removed)})"
               if frame_removed else ""))

    df = convert_pixels_to_units(df_px, scale)
    if contours is not None:
        _CG.attach_contours(df, contours, frame="pixels")
    stats = gauge_stats(df, scale)

    sidecar = {
        "image": os.path.abspath(str(source_image or image_path)),
        "image_name": os.path.basename(str(source_image or image_path)),
        "detection_image": os.path.abspath(image_path),
        "scale_source": scale_source or ("object" if scale.segments else "gsd"),
        "model": {"backend": model, "min_confidence": min_confidence,
                  "devicemode": devicemode, "devicenumber": int(devicenumber),
                  "detect_scale": factor,
                  "detection_size_px": list(resampled) if resampled else None},
        "excluded_by_segments": len(removed_ids),
        "excluded_clast_IDs": [int(i) for i in removed_ids],
        "exclusion_rule": SEGMENT_EXCLUSION_RULE if exclude_segments else None,
        "exclusion_segments": [[[a[0], a[1]], [b[0], b[1]]]
                               for a, b in excl_segments],
        "excluded_by_frame": len(frame_removed),
        "excluded_by_frame_clast_IDs": [int(i) for i in frame_removed],
        "frame_rule": FRAME_EXCLUSION_RULE if frame is not None else None,
        "frame": ({"thickness_m": frame.thickness_m, "inset_px": frame.inset_px,
                   "measured_size_m": [round(v, 4) for v in frame.inner_size_m]}
                  if frame is not None else None),
        "disclaimer": disclaimer or None,
    }
    ul = scale.unit_label
    log(f"[gauge] {stats['n']} clast(s); D50 = "
        f"{format_length(stats['D50'], ul)}, D84 = "
        f"{format_length(stats['D84'], ul)}")
    out = {"csv": None, "json": None, "df": df, "stats": stats, "stem": stem,
           "out_dir": str(out_dir), "contours": None, "model": model,
           "excluded_by_segments": len(removed_ids),
           "removed_ids": [int(i) for i in removed_ids],
           "excluded_by_frame": len(frame_removed),
           "frame_removed_ids": [int(i) for i in frame_removed],
           "frame": frame, "sidecar": sidecar}
    if not write:
        return out
    written = write_gauge_table(df, scale, out_dir, stem, contours=contours,
                                stats=stats, sidecar=sidecar,
                                contours_extra={"model": model})
    log(f"[gauge] wrote {Path(written['csv']).name} and "
        f"{Path(written['json']).name}")
    out.update(written)
    return out


def write_gauge_table(df: pd.DataFrame, scale: GaugeScale, out_dir, stem: str,
                      *, contours=None, stats: Optional[dict] = None,
                      sidecar: Optional[dict] = None,
                      contours_extra: Optional[dict] = None) -> dict:
    """Write ``<stem>_gauge.csv`` (the table in ``scale``'s unit), its
    ``<stem>_gauge.contours.json`` (``contours`` in the CSV's pixel frame;
    a stale one is removed when there are none) and ``<stem>_gauge.json``:
    the tool, timestamp, file names, scale, columns, n and statistics,
    merged with ``sidecar`` (image, model, exclusions, disclaimer,
    provenance counts...). Used by :func:`run_gauge` and by Digitize's
    *Figures*, which writes the records on the canvas. Returns ``{"csv",
    "json", "contours"}`` (paths as strings, contours None when none)."""
    from functions import clast_geometry as _CG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = gauge_output_names(stem)
    stats = stats or gauge_stats(df, scale)
    csv_path = out_dir / names["csv"]
    json_path = out_dir / names["json"]
    df.to_csv(csv_path, index=False, float_format="%.6g")
    contours_path = None
    if contours is not None:
        contours_path = _CG.write_contours(csv_path, contours, frame="pixels",
                                           extra=contours_extra)
    else:
        try:
            _CG.contours_path_for(csv_path).unlink()
        except OSError:
            pass
    doc = {
        "tool": "PebbleMapper Digitize",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "csv": names["csv"],
        "contours": contours_path.name if contours_path else None,
        "scale": scale.to_dict(),
        "columns": {"lengths": list(LENGTH_COLUMNS),
                    "areas": list(AREA_COLUMNS),
                    "pixels": list(PIXEL_COLUMNS)},
        "n": stats["n"],
        "stats": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in stats.items()},
    }
    for k, v in (sidecar or {}).items():
        doc.setdefault(k, v)
    json_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path),
            "contours": str(contours_path) if contours_path else None}


def backend_detect_fn(backend, work_dir, *, log_fn=None, keep=None,
                      timeout=None):
    """A ``detect_fn`` for :func:`run_gauge` that runs any registered
    detection backend in Quadrat mode on the (possibly resampled)
    photograph, through :func:`detectors.base.run_detect_jobs`, writing its
    CSV into ``work_dir``.

    The frame it returns carries the outlines (``attrs["contours"]``): the
    ones the backend attached, else its ``.contours.json`` sidecar. When the
    backend left an instances file (the subprocess protocol) the cropped
    instance masks are read back into ``keep["instances"]`` with the image
    shape in ``keep["shape"]``; otherwise only the outlines are there
    (``keep["contours"]``, pixel frame y up, detection-image pixels). The
    caller turns either into editable proposals."""
    from detectors.base import output_csv_path, run_detect_jobs
    from detectors import subprocess_runner as _sr
    from functions import clast_geometry as _CG
    keep = keep if keep is not None else {}

    def detect(**kw):
        jobs = [dict(j) for j in kw.pop("jobs")]
        mode = kw.pop("mode", "quadrat")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        for j in jobs:
            j.setdefault("out_stem", Path(j["path"]).stem)
        kw.update(output_dir=str(work), saveresults=True, plot=False,
                  saveplot=False, exclude_frame=False)
        if log_fn is not None:
            kw["log_fn"] = log_fn
        if timeout is not None:
            kw["timeout"] = timeout
        frames = run_detect_jobs(backend, mode, jobs, **kw) or []
        out = []
        for ji, job in enumerate(jobs):
            df = frames[ji] if ji < len(frames) else None
            if df is None:
                df = pd.DataFrame()
            csv = output_csv_path(mode, job, kw)
            if (df is None or not len(df)) and csv is not None and csv.exists():
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    pass
            cs = _CG.contours_of(df)
            if cs is None and csv is not None:
                cs = _CG.read_contours(csv)
                if cs is not None:
                    _CG.attach_contours(df, cs, frame="pixels")
            if ji == 0:
                keep["contours"] = cs
                keep["instances"] = None
                keep["shape"] = None
                npz = _sr.instances_path_for(csv) if csv is not None else None
                if npz is not None and npz.exists():
                    try:
                        keep["instances"] = _sr.read_instances_npz(npz)
                        keep["shape"] = _sr.read_instances_shape(npz)
                    except Exception:
                        keep["instances"] = None
            out.append(df)
        return out

    return detect


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #
def _figure_imports():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm, colors, rc_context
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.patheffects import withStroke
    return (cm, colors, rc_context, PatchCollection, Figure, Line2D,
            Ellipse, withStroke)


def _read_image_rgb(image_path, max_px: int = 2400):
    """The photograph as an RGB uint8 array for drawing (downsampled when
    wider than ``max_px``) and its original ``(W, H)``."""
    from PIL import Image
    with Image.open(image_path) as im:
        W, H = im.size
        im = im.convert("RGB")
        if max(W, H) > max_px:
            s = max_px / float(max(W, H))
            im = im.resize((max(1, int(round(W * s))),
                            max(1, int(round(H * s)))), Image.BILINEAR)
        arr = np.asarray(im)
    return arr, W, H


def _clean_lengths(df: pd.DataFrame, column: str = "Clast_length") -> np.ndarray:
    if column not in df.columns:
        return np.empty(0)
    v = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v) & (v > 0)]


def _pick_label_rows(df: pd.DataFrame, n_labels: int) -> List[int]:
    """Row positions of ~``n_labels`` clasts spread across the size range
    (quantiles 5 … 95 of Clast_length), each used once."""
    v = pd.to_numeric(df["Clast_length"], errors="coerce").to_numpy(dtype=float)
    ok = np.where(np.isfinite(v) & (v > 0))[0]
    if len(ok) == 0 or n_labels <= 0:
        return []
    n = min(int(n_labels), len(ok))
    targets = np.quantile(v[ok], np.linspace(0.05, 0.95, n))
    chosen: List[int] = []
    for t in targets:
        order = ok[np.argsort(np.abs(v[ok] - t))]
        for i in order:
            if int(i) not in chosen:
                chosen.append(int(i))
                break
    return chosen


def _segment_label(seg: "GaugeSegment", scale: GaugeScale,
                   library: Optional[Sequence[ScalingObject]]) -> str:
    """``"boot width: 0.27 m"`` for a known object, ``"1 boot width"`` for
    the object defining a custom unit, ``"<name> (not used)"`` for an
    ignored one, ``"<length> <unit>"`` for the single-object path."""
    name = seg.object_name or scale.object_name
    if scale.ignored and name in scale.ignored:
        return f"{name} (not used)"
    obj = find_object(library or [], name) if name else None
    if obj is not None and obj.is_metric:
        return f"{name}: {format_length(obj.length, obj.unit)}"
    if scale.mode == CUSTOM_MODE:
        # The object is the unit: "1 boot width".
        return f"{format_length(scale.object_length, short=True)} {name}".strip()
    # Metric mode without a library entry (the single-object path):
    # "0.27 m (boot width)".
    txt = format_length(scale.object_length, scale.unit_label, short=True)
    return f"{txt} ({name})" if name else txt


def segment_label_angle(p0, p1) -> float:
    """The angle, in degrees counter-clockwise on screen, that lays a text
    along the segment ``p0 -> p1`` given in image pixels (y growing
    downward). Kept within (-90, 90] so the text always reads left to
    right: a segment drawn right to left gets the same angle as the same
    segment drawn left to right. Matplotlib's ``rotation`` takes it as is;
    SVG's ``rotate()`` is clockwise-positive, so it takes the negative."""
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    deg = math.degrees(math.atan2(-dy, dx))
    while deg > 90.0:
        deg -= 180.0
    while deg <= -90.0:
        deg += 180.0
    return deg


def _footer_text(disclaimer: Optional[str]) -> str:
    """The disclaimer wrapped for a figure footer ('' when there is none)."""
    import textwrap
    if not disclaimer:
        return ""
    return textwrap.fill(str(disclaimer), 118)


def gauge_overlay_figure(image_path, df: pd.DataFrame, scale: GaugeScale, *,
                         n_labels: int = 8, cmap: str = "viridis",
                         stats: Optional[dict] = None,
                         library: Optional[Sequence[ScalingObject]] = None,
                         disclaimer: Optional[str] = DISCLAIMER,
                         contours=None, note: Optional[str] = None):
    """The whole photograph, in darkened greyscale, with every clast as its
    mask outline filled by its half-phi size class (legend under the frame),
    its length (solid) and width (dotted) chords and its centroid, the
    scaling segment(s) in orange with the object's length written beside
    them, ~``n_labels`` clasts across the size range annotated with their
    length, a box with n / D50 / D84, a scale bar when the unit is a
    physical length, and ``disclaimer`` written under the frame (none when
    it is None: a GSD-scaled run), after ``note`` when given (Digitize's
    model line: which detections, how many kept, edited, drawn by hand).
    No axes ticks, no ellipses.

    ``contours`` (``{clast_ID: [[x, y], ...]}`` in the CSV's pixel frame)
    defaults to the outlines attached to ``df``; without any, the axes are
    drawn alone with a note.

    Returns a :class:`matplotlib.figure.Figure` with ``fig.pm_meta``.
    """
    from functions import clast_geometry as _CG
    if contours is None:
        contours = _CG.contours_of(df)
    (cm, mcolors, rc_context, PatchCollection, Figure, Line2D,
     Ellipse, withStroke) = _figure_imports()
    from functions.report_figures import (_RC, FONT_PT, _add_scale_bar,
                                          _length_label, _nice_length)

    if not scale.valid:
        raise ValueError("the scale has no pixels-per-unit value")
    ppu = float(scale.px_per_unit)
    ul = scale.unit_label
    img, W, H = _read_image_rgb(image_path)
    stats = stats or gauge_stats(df, scale)

    need = ["x", "y", "Clast_length", "Clast_width", "Orientation"]
    have = all(c in df.columns for c in need) and len(df) > 0
    if have:
        sub = df[need + [c for c in ("clast_ID",) if c in df.columns]] \
            .apply(pd.to_numeric, errors="coerce")
        sub = sub.dropna(subset=need)
        sub = sub[(sub["Clast_length"] > 0) & (sub["Clast_width"] > 0)]
    else:
        sub = pd.DataFrame(columns=need)

    lengths = sub["Clast_length"].to_numpy(dtype=float) if len(sub) else np.empty(0)

    width_in = 11.0
    height_in = float(min(11.0, max(4.0, width_in * H / float(max(W, 1)))))
    labelled = []
    with rc_context(_RC):
        fig = Figure(figsize=(width_in, height_in), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.imshow(_CG.muted_background(img), extent=(0, W, H, 0),
                  interpolation="bilinear", zorder=1)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        # Every clast as its outline, filled by its half-phi size class, with
        # its length (solid) and width (dotted) chords and its centroid.
        drawn = {"n_outlines": 0, "n_chords": 0}
        if len(sub):
            classes = _CG.size_classes(lengths, unit=ul)
            drawn = _CG.draw_clasts(
                ax, sub, contours=contours, y_down=True,
                to_data=lambda xs, ys: (xs, H - np.asarray(ys, dtype=float)),
                units_per_m=ppu, edge_colors=classes["colours"])
            _CG.add_size_legend(ax, classes, fontsize=FONT_PT + 1, ncol=3,
                                outside_figure=True)

        # The scaling segment(s): orange over a white halo, each labelled
        # with its object (and its known length, or "1 <unit>" for the
        # object that defines a custom unit).
        diag = math.hypot(W, H)
        seg_items = list(scale.segments) or [
            GaugeSegment(a, b, scale.object_name) for a, b in scale.segments_px]
        for seg in seg_items:
            a, b = seg.p0, seg.p1
            ignored = seg.object_name in (scale.ignored or [])
            colour, edge = ("#9a9a9a", "#777") if ignored else ("#ff7f00", "#b34700")
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=3.2,
                    solid_capstyle="round", zorder=20,
                    path_effects=[withStroke(linewidth=6.5, foreground="white")])
            ax.plot([a[0], b[0]], [a[1], b[1]], ls="none", marker="o", ms=6,
                    mfc=colour, mec="white", mew=1.2, zorder=21)
            # The label lies along the segment, centred on its midpoint and
            # lifted a little to the side the text's top faces so the line
            # stays visible. The axes are image pixels with y inverted, so
            # the on-screen "up" of a text rotated by ``rot`` (CCW) is
            # (-sin, -cos) in data coordinates; the readability flip keeps
            # ``rot`` within (-90, 90], hence cos >= 0 and the offset never
            # crosses the line.
            mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            rot = segment_label_angle(a, b)
            r = math.radians(rot)
            off = 0.006 * diag
            ax.text(mx - math.sin(r) * off, my - math.cos(r) * off,
                    _segment_label(seg, scale, library),
                    rotation=rot, rotation_mode="anchor",
                    color=edge, fontsize=FONT_PT + 3, fontweight="bold",
                    ha="center", va="bottom", zorder=22,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=colour, alpha=0.9))

        # A handful of clasts across the size range, their length written
        # at the end of their major axis (drawn above with every clast's).
        for i in _pick_label_rows(sub, n_labels) if len(sub) else []:
            row = sub.iloc[i]
            Lpx = float(row["Clast_length"]) * ppu
            cx, cy = float(row["x"]), H - float(row["y"])
            hx, hy = _CG.axis_direction(float(row["Orientation"]), y_down=True)
            hx, hy = hx * Lpx / 2.0, hy * Lpx / 2.0
            txt = format_length(float(row["Clast_length"]), ul, short=True)
            ax.annotate(txt, xy=(cx + hx, cy + hy), xytext=(6, -6),
                        textcoords="offset points", color="white",
                        fontsize=FONT_PT + 1, ha="left", va="top", zorder=16,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="black", edgecolor="none",
                                  alpha=0.65))
            labelled.append({"x": cx, "y": cy, "Clast_length": float(row["Clast_length"])})

        # n and the percentiles, top left.
        box = (f"n = {stats['n']:,}\n"
               f"D50 = {format_length(stats['D50'], ul)}\n"
               f"D84 = {format_length(stats['D84'], ul)}")
        ax.text(0.012, 0.985, box, transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT_PT + 2, zorder=30, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", alpha=0.9))

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        bar_m = None
        if scale.is_metric:
            px_per_m = ppu / float(scale.metres_per_unit)
            bar_m = _nice_length(W / px_per_m)
            _add_scale_bar(ax, bar_m * px_per_m, _length_label(bar_m),
                           loc="lower right")
        footer = _footer_text(disclaimer)
        if note:
            footer = _footer_text(note) + ("\n" + footer if footer else "")
        if footer:
            ax.set_xlabel(footer, fontsize=FONT_PT, color="#666", labelpad=4)

    fig.pm_meta = {
        "n": int(stats["n"]), "D50": stats["D50"], "D84": stats["D84"],
        "unit": ul, "labelled": labelled, "scale_bar_m": bar_m,
        "n_segments": len(scale.segments_px),
        "n_outlines": int(drawn["n_outlines"]), "n_chords": int(drawn["n_chords"]),
    }
    return fig


def gauge_distribution_figure(df: pd.DataFrame, scale: GaugeScale, *,
                              column: str = "Clast_length",
                              disclaimer: Optional[str] = DISCLAIMER):
    """Two panels: a log-x histogram with a KDE and D16 / D50 / D84 marked
    and labelled in the unit, and the empirical CDF with D16 / D50 / D84
    markers. Axis labels carry the unit label; no φ axis; ``disclaimer``
    as a footer (none when it is None). Raises ``ValueError`` with fewer
    than two positive values."""
    (cm, mcolors, rc_context, PatchCollection, Figure, Line2D,
     Ellipse, withStroke) = _figure_imports()
    from functions.report_figures import (_RC, FONT_PT, _apply_log_axis,
                                          _fd_bins_log, _fmt)

    v = np.sort(_clean_lengths(df, column))
    if len(v) < 2:
        raise ValueError(f"n = {len(v)} positive {column} value(s) (< 2)")
    ul = scale.unit_label
    lo, hi = float(v.min()), float(v.max())
    if hi > lo:
        nb = _fd_bins_log(v)
        edges = np.geomspace(lo, hi, nb + 1)
    else:
        nb = 1
        edges = np.array([lo * 0.9, hi * 1.1])
    d16, d50, d84 = (float(np.quantile(v, q)) for q in (0.16, 0.5, 0.84))
    xlo, xhi = lo * 0.8, hi * 1.25
    if not xhi > xlo:
        xlo, xhi = lo * 0.5, hi * 2.0
    field_label = f"Clast length ({ul})"

    with rc_context(_RC):
        fig = Figure(figsize=(11.0, 4.3), constrained_layout=True)
        ax1, ax2 = fig.subplots(1, 2)

        # Histogram + KDE.
        ax1.hist(v, bins=edges, density=True, color=cm.viridis(0.45),
                 edgecolor="white", linewidth=0.3, zorder=3,
                 label=f"Histogram ({nb} bins)")
        if hi > lo:
            try:
                from scipy.stats import gaussian_kde
                xs = np.geomspace(lo, hi, 400)
                ax1.plot(xs, gaussian_kde(v)(xs), color=cm.viridis(0.0),
                         lw=1.6, zorder=5, label="KDE")
            except Exception:
                pass
        _apply_log_axis(ax1)
        ax1.set_xlim(xlo, xhi)
        ax1.set_xlabel(field_label)
        ax1.set_ylabel("Probability density")
        ax1.legend(loc="upper right", framealpha=0.85, edgecolor="none")
        for name, val, row in (("D16", d16, 0), ("D50", d50, 1), ("D84", d84, 0)):
            ax1.axvline(val, color="#333", lw=0.8, ls=":", zorder=7)
            ax1.annotate(f"{name} = {_fmt(val)}", xy=(val, 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(0, 3 + 12 * row), textcoords="offset points",
                         ha="center", va="bottom", fontsize=FONT_PT - 1,
                         color="#222", annotation_clip=False)

        # Empirical CDF.
        n = len(v)
        yy = np.arange(1, n + 1) / n
        col = cm.viridis(0.25)
        ax2.plot(v, yy, color=col, lw=1.8, drawstyle="steps-post", zorder=4,
                 label=f"n = {n:,}")
        handles = []
        for q, val, mk in ((16, d16, "v"), (50, d50, "o"), (84, d84, "s")):
            ax2.axhline(q / 100.0, color="grey", lw=0.5, ls=":", alpha=0.6, zorder=2)
            ax2.plot([val], [q / 100.0], marker=mk, ms=6, color=col, mec="white",
                     mew=0.8, ls="none", zorder=10)
            handles.append(Line2D([], [], marker=mk, color="#444", mec="white",
                                  ls="none", ms=6,
                                  label=f"D{q} = {format_length(val, ul)}"))
        _apply_log_axis(ax2)
        ax2.set_xlim(xlo, xhi)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xlabel(field_label)
        ax2.set_ylabel("Cumulative fraction")
        ax2.legend(handles=[Line2D([], [], color=col, lw=1.8, label=f"n = {n:,}")]
                   + handles, loc="lower right", framealpha=0.85, edgecolor="none")
        footer = _footer_text(disclaimer)
        if footer:
            fig.supxlabel(footer, fontsize=FONT_PT - 1, color="#666")

    fig.pm_meta = {"n": int(n), "bins": int(nb), "D16": d16, "D50": d50,
                   "D84": d84, "unit": ul}
    return fig


def write_gauge_figures(image_path, df: pd.DataFrame, scale: GaugeScale,
                        out_dir, stem: str, *, dpi: int = 200,
                        n_labels: int = 8, stats: Optional[dict] = None,
                        library: Optional[Sequence[ScalingObject]] = None,
                        log_fn: Optional[Callable[[str], None]] = None,
                        disclaimer: Optional[str] = DISCLAIMER,
                        contours=None, note: Optional[str] = None) -> dict:
    """Save ``<stem>_gauge_overlay.png`` and ``<stem>_gauge_distribution.png``
    under ``out_dir`` at ``dpi``, both carrying ``disclaimer`` as a footer
    (none when it is None); ``note`` (a model line) heads the overlay's
    footer. ``contours`` defaults to the outlines attached to ``df`` (the
    frame :func:`run_gauge` returns). Returns ``{"overlay": path, "distribution":
    path-or-None}``; the distribution is skipped (None) when fewer than two
    clasts were measured."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = gauge_output_names(stem)
    log = log_fn or (lambda _s: None)
    stats = stats or gauge_stats(df, scale)

    fig = gauge_overlay_figure(image_path, df, scale, n_labels=n_labels,
                               stats=stats, library=library,
                               disclaimer=disclaimer, contours=contours,
                               note=note)
    overlay = out_dir / names["overlay"]
    try:
        fig.savefig(overlay, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        try:
            fig.clear()
        except Exception:
            pass
    log(f"[gauge] wrote {overlay.name}")

    distribution: Optional[Path] = None
    try:
        fig2 = gauge_distribution_figure(df, scale, disclaimer=disclaimer)
    except ValueError as ex:
        log(f"[gauge] no distribution figure: {ex}")
    else:
        distribution = out_dir / names["distribution"]
        try:
            fig2.savefig(distribution, dpi=dpi, bbox_inches="tight",
                         facecolor="white")
        finally:
            try:
                fig2.clear()
            except Exception:
                pass
        log(f"[gauge] wrote {distribution.name}")
    return {"overlay": str(overlay),
            "distribution": str(distribution) if distribution else None}


# --------------------------------------------------------------------------- #
#  Sample set: every clast of every photograph of a folder, pooled             #
# --------------------------------------------------------------------------- #
METRIC_GROUP = "metric"
SAMPLE_SET_UNIT = "mm"


def sample_set_names(stem: str) -> dict:
    """The three files a sample set writes for ``stem`` (the folder's origin)."""
    return {"csv": f"{stem}_sampleset.csv",
            "summary": f"{stem}_sampleset_summary.csv",
            "distribution": f"{stem}_sampleset_distribution.png"}


def _entry_unit(entry: Mapping) -> Optional[str]:
    """The custom unit of a Digitize truth table, None when it is in metres:
    the ``unit`` entry, else the table's ``unit`` column."""
    u = entry.get("unit")
    if u is None:
        df = entry.get("df")
        if df is not None and "unit" in getattr(df, "columns", []) and len(df):
            vals = [str(v) for v in df["unit"].dropna().unique() if str(v)]
            u = vals[0] if vals else None
    u = str(u).strip() if u is not None else ""
    return u or None


def _origin_text(provenance: Optional[Mapping]) -> Tuple[str, str]:
    """``(models, origins)`` for a summary row from a provenance sidecar."""
    if not provenance:
        return "", ""
    from functions import provenance as _prov
    clasts = provenance.get("clasts") or {}
    models = _prov.model_names(clasts, provenance.get("models") or [])
    counts = _prov.origin_counts(clasts)
    return ("; ".join(models),
            "; ".join(f"{k}: {v}" for k, v in counts.items()))


def pool_sample_set(entries: Sequence[Mapping]) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """Pool the Digitize tables of a folder's photographs.

    ``entries``: one mapping per photograph, ``{"photo": name, "df": the
    truth table (lengths in metres, areas in m², or in a custom unit named
    by its ``unit`` column), "unit": optional override of that unit,
    "provenance": optional sidecar dict (functions.provenance)}``. Tables
    in metres pool together and are shown in millimetres; a custom unit
    pools only with tables in the same unit name. When the folder mixes
    groups the one with the most photographs is kept (metric on a tie) and
    the others are listed in ``skipped``, as are empty or unreadable
    tables. Scale-segment exclusions are already applied to each table.

    Returns ``(pooled, summary, skipped)``: ``pooled`` has a ``photo``
    column first then the table's columns in the pooled unit and a ``unit``
    column; ``summary`` one row per pooled photograph and a last
    ``(pooled)`` row with ``photo, n, D16, D50, D84, mean, sorting_phi,
    unit, models, origins`` (sorting only in a physical unit); ``skipped``
    ``[{"photo", "reason"}]``. ``pooled.attrs`` and ``summary.attrs`` carry
    ``unit`` and ``metric``."""
    groups: Dict[str, List[Mapping]] = {}
    skipped: List[dict] = []
    for e in entries or []:
        photo = str(e.get("photo") or "")
        df = e.get("df")
        if df is None or not len(df) or "Clast_length" not in df.columns:
            skipped.append({"photo": photo, "reason": "no saved clasts"})
            continue
        unit = _entry_unit(e)
        key = METRIC_GROUP if unit is None else f"custom:{unit}"
        groups.setdefault(key, []).append(e)
    if not groups:
        empty = pd.DataFrame(columns=["photo"])
        return empty, pd.DataFrame(columns=["photo", "n"]), skipped
    best = max(groups, key=lambda k: (len(groups[k]), k == METRIC_GROUP))
    metric = best == METRIC_GROUP
    unit_label = SAMPLE_SET_UNIT if metric else best.split(":", 1)[1]
    for key, es in groups.items():
        if key == best:
            continue
        other = "metres" if key == METRIC_GROUP else f"'{key.split(':', 1)[1]}' units"
        for e in es:
            skipped.append({"photo": str(e.get("photo") or ""),
                            "reason": f"in {other}, not pooled with the "
                                      f"{'millimetre' if metric else repr(unit_label)} set"})
    if metric:
        scale = GaugeScale(object_name="", unit=SAMPLE_SET_UNIT, px_per_unit=1.0,
                           mode=METRIC_MODE)
        lf, af = 1000.0, 1.0e6
    else:
        scale = GaugeScale(object_name=unit_label, unit=CUSTOM, px_per_unit=1.0,
                           mode=CUSTOM_MODE)
        lf, af = 1.0, 1.0
    frames, rows = [], []
    for e in groups[best]:
        photo = str(e.get("photo") or "")
        df = e["df"].copy()
        for c in LENGTH_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * lf
        for c in AREA_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * af
        df["unit"] = unit_label
        df.insert(0, "photo", photo)
        frames.append(df)
        st = gauge_stats(df, scale)
        models, origins = _origin_text(e.get("provenance"))
        rows.append({"photo": photo, "n": st["n"], "D16": st["D16"],
                     "D50": st["D50"], "D84": st["D84"], "mean": st["mean"],
                     "sorting_phi": st["sorting_phi"], "unit": unit_label,
                     "models": models, "origins": origins})
    pooled = pd.concat(frames, ignore_index=True)
    st = gauge_stats(pooled, scale)
    all_models = []
    for r in rows:
        for m in (r["models"] or "").split("; "):
            if m and m not in all_models:
                all_models.append(m)
    counts: Dict[str, int] = {}
    for e in groups[best]:
        prov = e.get("provenance") or {}
        from functions import provenance as _prov
        for k, v in _prov.origin_counts(prov.get("clasts") or {}).items():
            counts[k] = counts.get(k, 0) + v
    rows.append({"photo": "(pooled)", "n": st["n"], "D16": st["D16"],
                 "D50": st["D50"], "D84": st["D84"], "mean": st["mean"],
                 "sorting_phi": st["sorting_phi"], "unit": unit_label,
                 "models": "; ".join(all_models),
                 "origins": "; ".join(f"{k}: {v}" for k, v in counts.items())})
    summary = pd.DataFrame(rows, columns=["photo", "n", "D16", "D50", "D84", "mean",
                                          "sorting_phi", "unit", "models", "origins"])
    for d in (pooled, summary):
        d.attrs["unit"] = unit_label
        d.attrs["metric"] = metric
    return pooled, summary, skipped


def sample_set_figure(pooled: pd.DataFrame, *, unit: Optional[str] = None,
                      column: str = "Clast_length",
                      disclaimer: Optional[str] = None):
    """Two panels for a pooled sample set: a log-x histogram with a KDE and
    D16 / D50 / D84 marked; the pooled empirical CDF in a thick line over
    each photograph's CDF in thin lines (a legend with n per photograph up
    to ten photographs, thin grey lines and a count note above). The
    ``disclaimer`` is the footer (none when None). Raises ``ValueError``
    with fewer than two positive values."""
    (cm, mcolors, rc_context, PatchCollection, Figure, Line2D,
     Ellipse, withStroke) = _figure_imports()
    from functions.report_figures import (_RC, FONT_PT, _apply_log_axis,
                                          _fd_bins_log, _fmt)
    ul = unit or pooled.attrs.get("unit") or ""
    v = np.sort(_clean_lengths(pooled, column))
    if len(v) < 2:
        raise ValueError(f"n = {len(v)} positive {column} value(s) (< 2)")
    lo, hi = float(v.min()), float(v.max())
    if hi > lo:
        nb = _fd_bins_log(v)
        edges = np.geomspace(lo, hi, nb + 1)
    else:
        nb = 1
        edges = np.array([lo * 0.9, hi * 1.1])
    d16, d50, d84 = (float(np.quantile(v, q)) for q in (0.16, 0.5, 0.84))
    xlo, xhi = lo * 0.8, hi * 1.25
    photos = [p for p in pd.unique(pooled["photo"])] if "photo" in pooled.columns else []
    field_label = f"Clast length ({ul})"
    with rc_context(_RC):
        fig = Figure(figsize=(11.0, 4.6), constrained_layout=True)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.hist(v, bins=edges, density=True, color=cm.viridis(0.45),
                 edgecolor="white", linewidth=0.3, zorder=3,
                 label=f"Pooled, n = {len(v):,} ({nb} bins)")
        if hi > lo:
            try:
                from scipy.stats import gaussian_kde
                xs = np.geomspace(lo, hi, 400)
                ax1.plot(xs, gaussian_kde(v)(xs), color=cm.viridis(0.0),
                         lw=1.6, zorder=5, label="KDE")
            except Exception:
                pass
        _apply_log_axis(ax1)
        ax1.set_xlim(xlo, xhi)
        ax1.set_xlabel(field_label)
        ax1.set_ylabel("Probability density")
        ax1.legend(loc="upper right", framealpha=0.85, edgecolor="none")
        for name, val, row in (("D16", d16, 0), ("D50", d50, 1), ("D84", d84, 0)):
            ax1.axvline(val, color="#333", lw=0.8, ls=":", zorder=7)
            ax1.annotate(f"{name} = {_fmt(val)}", xy=(val, 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(0, 3 + 12 * row), textcoords="offset points",
                         ha="center", va="bottom", fontsize=FONT_PT - 1,
                         color="#222", annotation_clip=False)

        many = len(photos) > 10
        colours = ([cm.viridis(t) for t in np.linspace(0.0, 0.85, max(1, len(photos)))]
                   if not many else None)
        for k, p in enumerate(photos):
            pv = np.sort(_clean_lengths(pooled[pooled["photo"] == p], column))
            if not len(pv):
                continue
            yy = np.arange(1, len(pv) + 1) / len(pv)
            if many:
                ax2.plot(pv, yy, color="#9a9a9a", lw=0.7, alpha=0.8,
                         drawstyle="steps-post", zorder=3)
            else:
                ax2.plot(pv, yy, color=colours[k], lw=0.9, alpha=0.9,
                         drawstyle="steps-post", zorder=3,
                         label=f"{p} (n = {len(pv):,})")
        yy = np.arange(1, len(v) + 1) / len(v)
        ax2.plot(v, yy, color="#111", lw=2.6, drawstyle="steps-post", zorder=6,
                 label=f"Pooled (n = {len(v):,})")
        for q, val, mk in ((16, d16, "v"), (50, d50, "o"), (84, d84, "s")):
            ax2.axhline(q / 100.0, color="grey", lw=0.5, ls=":", alpha=0.6, zorder=2)
            ax2.plot([val], [q / 100.0], marker=mk, ms=6, color="#111",
                     mec="white", mew=0.8, ls="none", zorder=10)
        _apply_log_axis(ax2)
        ax2.set_xlim(xlo, xhi)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xlabel(field_label)
        ax2.set_ylabel("Cumulative fraction")
        if many:
            ax2.legend(loc="lower right", framealpha=0.85, edgecolor="none")
            ns = [int((pooled["photo"] == p).sum()) for p in photos]
            ax2.text(0.02, 0.98, f"{len(photos)} photographs (grey), "
                     f"n = {min(ns):,} to {max(ns):,} each",
                     transform=ax2.transAxes, ha="left", va="top",
                     fontsize=FONT_PT - 1, color="#444",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="#ccc", alpha=0.9))
        else:
            ax2.legend(loc="lower right", framealpha=0.85, edgecolor="none",
                       fontsize=FONT_PT - 1)
        footer = _footer_text(disclaimer)
        if footer:
            fig.supxlabel(footer, fontsize=FONT_PT - 1, color="#666")
    fig.pm_meta = {"n": int(len(v)), "photos": len(photos), "bins": int(nb),
                   "D16": d16, "D50": d50, "D84": d84, "unit": ul,
                   "per_photo_legend": not many}
    return fig


def write_sample_set(entries: Sequence[Mapping], out_dir, stem: str, *,
                     disclaimer: Optional[str] = None, dpi: int = 200,
                     log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """Pool ``entries`` (:func:`pool_sample_set`) and write
    ``<stem>_sampleset.csv``, ``<stem>_sampleset_summary.csv`` and
    ``<stem>_sampleset_distribution.png`` under ``out_dir``. Returns
    ``{"csv", "summary", "distribution", "pooled", "summary_df",
    "skipped", "unit", "metric"}``; ``distribution`` is None when fewer
    than two clasts were pooled."""
    log = log_fn or (lambda _s: None)
    pooled, summary, skipped = pool_sample_set(entries)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sample_set_names(stem)
    csv_path = out_dir / names["csv"]
    sum_path = out_dir / names["summary"]
    pooled.to_csv(csv_path, index=False, float_format="%.6g")
    summary.to_csv(sum_path, index=False, float_format="%.6g")
    dist = None
    try:
        fig = sample_set_figure(pooled, disclaimer=disclaimer)
    except (ValueError, KeyError) as ex:
        log(f"[sample set] no distribution figure: {ex}")
    else:
        dist = out_dir / names["distribution"]
        try:
            fig.savefig(dist, dpi=dpi, bbox_inches="tight", facecolor="white")
        finally:
            try:
                fig.clear()
            except Exception:
                pass
    log(f"[sample set] wrote {csv_path.name}, {sum_path.name}"
        + (f" and {dist.name}" if dist else ""))
    return {"csv": str(csv_path), "summary": str(sum_path),
            "distribution": str(dist) if dist else None, "pooled": pooled,
            "summary_df": summary, "skipped": skipped,
            "unit": pooled.attrs.get("unit"), "metric": pooled.attrs.get("metric")}
