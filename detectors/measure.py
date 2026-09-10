"""Shared, model-agnostic instance measurement.

Every backend that emits instance geometry turns a clast mask into the same
measurements (ellipse axes, length/width, surface area, equivalent diameter,
orientation, centroid) so grain sizes are comparable across models. This
re-exports the detector module's measurement rather than re-implementing it.

``measure_mask(mask, score, resolution)`` -> dict with keys:
    center_x, center_y, Ellipse_major_axis, Ellipse_minor_axis,
    Clast_length, Clast_width, Surface_area, Perimeter,
    Equivalent_diameter, Eccentricity, Solidity, Score, Orientation
(or ``None`` when the instance should be skipped).

``measure_instances(...)`` applies it to a whole instances file (see
``detectors.subprocess_runner.read_instances_npz``) and returns the canonical
per-clast table, with the centroid mapped to world coordinates the same way
Mask R-CNN maps it: ``y = height - center_y`` for a quadrat photograph, the
GeoTIFF geotransform for an ortho. Each instance's mask outline goes the same
way into ``df.attrs["contours"]`` (``functions.clast_geometry``), and into
``<csv stem>.contours.json`` when ``csv_path`` is given; a backend that writes
the CSV itself without ``csv_path`` still gets the sidecar from
``detectors.base.run_detect_jobs``, which persists the attached outlines.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from functions.clasts_detection import _measure_clast as measure_mask
from functions.clasts_detection import _load_roi_paths as load_roi_paths
from functions.clasts_detection import _mask_mean_intensity as mask_mean_intensity

__all__ = ["measure_mask", "measure_instances", "load_roi_paths",
           "mask_mean_intensity", "world_xy"]


def world_xy(cx: float, cy: float, *, height: Optional[int] = None,
             geotransform=None):
    """Map an image-pixel centroid to the canonical ``x``/``y``.

    With a ``geotransform`` (GDAL's 6-tuple) the result is in the ortho's
    CRS; otherwise ``(cx, height - cy)``, the quadrat photograph convention.
    """
    if geotransform is not None:
        gt = geotransform
        return (gt[0] + cx * gt[1] + cy * gt[2],
                gt[3] + cx * gt[4] + cy * gt[5])
    if height is None:
        raise ValueError("world_xy needs height (quadrat) or geotransform (ortho)")
    return cx, float(height) - cy


def measure_instances(instances: Iterable, resolution: float, *,
                      height: Optional[int] = None, geotransform=None,
                      image=None, roi_paths: Optional[List] = None,
                      log_fn=None, csv_path=None):
    """Measure ``Instance`` records into the canonical per-clast table.

    ``instances``: from ``read_instances_npz`` (mask cropped to its bounding
    box at ``y0``/``x0``). ``resolution``: metres per pixel. ``height``: image
    height in pixels (quadrat mode) or ``geotransform`` (ortho mode) for the
    world mapping. ``image`` (HxWx3, optional) fills ``Mean_intensity``;
    ``roi_paths`` (from ``load_roi_paths``, image-pixel coordinates) drops
    instances whose centroid falls outside every polygon. Instances that
    ``measure_mask`` rejects are skipped and counted in the log line.
    Returns a ``pandas.DataFrame`` with ``detectors.CANONICAL_COLUMNS`` whose
    ``attrs["contours"]`` holds every outline under its ``clast_ID``, in the
    same frame as ``x``/``y``. ``csv_path`` (optional) also writes the
    outlines beside that CSV; the caller still writes the CSV.
    """
    import math
    import pandas as pd
    from detectors.base import CANONICAL_COLUMNS
    from functions import clast_geometry as CG

    if geotransform is not None:
        to_frame = CG.world_frame(geotransform)
        frame, decimals = "world", CG.world_decimals(geotransform[1])
    elif height is not None:
        to_frame = CG.quadrat_frame(height)
        frame, decimals = "pixels", 2
    else:
        to_frame, frame, decimals = None, "pixels", 2
    contours = {}

    records = []
    n_in = n_skipped = n_roi = 0
    for inst in instances:
        n_in += 1
        meas = measure_mask(inst.mask, float(inst.score), float(resolution))
        if meas is None:
            n_skipped += 1
            continue
        cx = float(meas["center_x"]) + inst.x0
        cy = float(meas["center_y"]) + inst.y0
        if roi_paths and not any(rp.contains_point((cx, cy)) for rp in roi_paths):
            n_roi += 1
            continue
        wx, wy = world_xy(cx, cy, height=height, geotransform=geotransform)
        if to_frame is not None:
            try:
                outline = CG.contour_from_measurement(
                    meas, to_frame=to_frame, offset=(inst.x0, inst.y0),
                    decimals=decimals)
                if outline:
                    contours[len(records) + 1] = outline
            except Exception:
                pass    # an outline never costs a measurement
        if image is not None:
            h, w = inst.mask.shape
            intensity = mask_mean_intensity(
                image[inst.y0:inst.y0 + h, inst.x0:inst.x0 + w], inst.mask)
        else:
            intensity = math.nan
        records.append({
            "clast_ID": len(records) + 1, "x": wx, "y": wy,
            "Clast_length": meas["Clast_length"],
            "Clast_width": meas["Clast_width"],
            "Ellipse_major_axis": meas["Ellipse_major_axis"],
            "Ellipse_minor_axis": meas["Ellipse_minor_axis"],
            "Surface_area": meas["Surface_area"],
            "Perimeter": meas["Perimeter"],
            "Equivalent_diameter": meas["Equivalent_diameter"],
            "Eccentricity": meas["Eccentricity"],
            "Solidity": meas["Solidity"],
            "Mean_intensity": intensity,
            "Score": meas["Score"],
            "Orientation": meas["Orientation"],
        })
    if log_fn and (n_skipped or n_roi):
        log_fn(f"[measure] {n_in} instance(s): {len(records)} measured, "
               f"{n_skipped} unmeasurable skipped, {n_roi} outside the ROI")
    df = pd.DataFrame(records, columns=list(CANONICAL_COLUMNS))
    CG.attach_contours(df, contours, frame=frame)
    if csv_path is not None:
        CG.write_contours(csv_path, contours, frame=frame)
    return df
