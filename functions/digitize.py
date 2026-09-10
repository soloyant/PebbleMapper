"""Manual digitization: binary masks from user clicks (polygon, circle,
ellipse), measured with the detection pipeline's _measure_clast so the output
schema matches detection CSVs."""
import numpy as np


class CroppedMask:
    """A record's binary mask stored as the window that holds it.

    A full-frame boolean mask costs H x W bytes (5.8 MiB on an iPhone
    photograph), so thousands of detections cannot each keep one. This
    keeps ``crop`` (bool) and its top-left ``(r0, c0)`` in a frame of
    ``shape`` (H, W). It reads as the full-frame array wherever numpy needs
    one (``np.asarray``, ``m & other``), and answers ``shape``, a pixel
    lookup ``m[row, col]``, ``sum()``, ``any()`` and :func:`mask_area`
    without building it."""

    ndim = 2
    dtype = np.dtype(bool)

    def __init__(self, crop, r0, c0, shape):
        self.crop = np.ascontiguousarray(crop, dtype=bool)
        self.r0, self.c0 = int(r0), int(c0)
        self.shape = (int(shape[0]), int(shape[1]))

    @classmethod
    def from_full(cls, mask):
        """The tight window of a full-frame mask (an empty mask keeps a
        0 x 0 window)."""
        if isinstance(mask, cls):
            return mask
        m = np.asarray(mask).astype(bool, copy=False)
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        if rows.size == 0:
            return cls(np.zeros((0, 0), bool), 0, 0, m.shape)
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return cls(m[r0:r1, c0:c1], r0, c0, m.shape)

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def __array__(self, dtype=None, copy=None):
        full = np.zeros(self.shape, dtype=bool)
        h, w = self.crop.shape
        full[self.r0:self.r0 + h, self.c0:self.c0 + w] = self.crop
        return full if dtype is None else full.astype(dtype)

    def __getitem__(self, key):
        if (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, (int, np.integer)) for k in key)):
            r, c = int(key[0]) - self.r0, int(key[1]) - self.c0
            h, w = self.crop.shape
            return bool(0 <= r < h and 0 <= c < w and self.crop[r, c])
        return np.asarray(self)[key]

    def __and__(self, other):
        return np.asarray(self) & np.asarray(other)

    def __or__(self, other):
        return np.asarray(self) | np.asarray(other)

    __rand__, __ror__ = __and__, __or__

    def sum(self):
        return int(np.count_nonzero(self.crop))

    def any(self):
        return bool(self.crop.any())

    def astype(self, dtype, copy=True):
        return np.asarray(self).astype(dtype)


def axial_bearing(drawn_angle_deg: float) -> float:
    """The clast table's ``Orientation`` for an axis drawn at
    ``drawn_angle_deg`` from the image's x axis.

    The canvas measures the angle it was given; the CSV records the axial
    bearing of the major axis, 0-180 degrees, from image-up. One shape, one
    convention: the tab announces what the table will list.
    """
    return float(drawn_angle_deg + 90.0) % 180.0


def mask_area(mask) -> int:
    """Pixels set in a full-frame or cropped mask."""
    if isinstance(mask, CroppedMask):
        return mask.sum()
    return int(np.count_nonzero(mask))


def _window(box, shape):
    """``(r0, r1, c0, c1)`` of a float box ``(ymin, ymax, xmin, xmax)``,
    padded by 2 px and clipped to ``shape``; None when nothing is left."""
    H, W = int(shape[0]), int(shape[1])
    r0 = max(0, int(np.floor(box[0])) - 2)
    r1 = min(H, int(np.ceil(box[1])) + 3)
    c0 = max(0, int(np.floor(box[2])) - 2)
    c1 = min(W, int(np.ceil(box[3])) + 3)
    return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None


def polygon_to_cropped_mask(vertices, image_shape):
    """:func:`polygon_to_mask` drawn only in the polygon's window."""
    import cv2
    shape = image_shape[:2]
    if len(vertices) < 3:
        return CroppedMask(np.zeros((0, 0), bool), 0, 0, shape)
    v = np.asarray(vertices, dtype=np.int32).reshape(-1, 2)
    box = _window((v[:, 1].min(), v[:, 1].max(), v[:, 0].min(), v[:, 0].max()),
                  shape)
    if box is None:
        return CroppedMask(np.zeros((0, 0), bool), 0, 0, shape)
    r0, r1, c0, c1 = box
    crop = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    cv2.fillPoly(crop, [v.reshape(-1, 1, 2)], 1, offset=(-c0, -r0))
    return CroppedMask(crop, r0, c0, shape)


def circle_to_cropped_mask(center, radius, image_shape):
    """:func:`circle_to_mask` drawn only in the circle's window."""
    import cv2
    shape = image_shape[:2]
    cx, cy, r = int(round(center[0])), int(round(center[1])), int(round(radius))
    box = _window((cy - r, cy + r, cx - r, cx + r), shape)
    if box is None:
        return CroppedMask(np.zeros((0, 0), bool), 0, 0, shape)
    r0, r1, c0, c1 = box
    crop = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    cv2.circle(crop, (cx - c0, cy - r0), r, 1, thickness=-1)
    return CroppedMask(crop, r0, c0, shape)


def ellipse_to_cropped_mask(center, axes, angle_deg, image_shape):
    """:func:`ellipse_to_mask` drawn only in the ellipse's window."""
    import cv2
    shape = image_shape[:2]
    a, b = int(round(axes[0])), int(round(axes[1]))
    cx, cy = int(round(center[0])), int(round(center[1]))
    empty = CroppedMask(np.zeros((0, 0), bool), 0, 0, shape)
    if a <= 0 or b <= 0:
        return empty
    rr = max(a, b)
    box = _window((cy - rr, cy + rr, cx - rr, cx + rr), shape)
    if box is None:
        return empty
    r0, r1, c0, c1 = box
    crop = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    cv2.ellipse(crop, (cx - c0, cy - r0), (a, b), float(angle_deg), 0, 360, 1,
                thickness=-1)
    return CroppedMask(crop, r0, c0, shape)


def mask_centroid(mask):
    """``(x, y)`` mean pixel of a full-frame or cropped mask, None if empty."""
    if isinstance(mask, CroppedMask):
        ys, xs = np.nonzero(mask.crop)
        if not xs.size:
            return None
        return float(xs.mean()) + mask.c0, float(ys.mean()) + mask.r0
    ys, xs = np.nonzero(np.asarray(mask))
    if not xs.size:
        return None
    return float(xs.mean()), float(ys.mean())


def polygon_to_mask(vertices, image_shape):
    """Build a filled-polygon binary mask.

    vertices    : list of (x, y) pixel coords (image-space, top-origin)
    image_shape : (H, W) target mask shape

    Returns uint8 mask with values 0/1 of shape (H, W).
    """
    import cv2
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(vertices) < 3:
        return mask
    pts = np.asarray(vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def circle_to_mask(center, radius, image_shape):
    """Build a filled-circle binary mask.

    center      : (x, y) pixel coords
    radius      : in pixels (float ok, rounded internally)
    image_shape : (H, W)
    """
    import cv2
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (int(round(center[0])), int(round(center[1]))),
               int(round(radius)), 1, thickness=-1)
    return mask


def ellipse_to_mask(center, axes, angle_deg, image_shape):
    """Build a filled rotated-ellipse binary mask.

    center      : (x, y) pixel coords of the ellipse center
    axes        : (a, b) semi-major and semi-minor axes in pixels
    angle_deg   : rotation of the major axis from +col, in degrees, in the
                  image frame (rows down: positive turns clockwise on
                  screen), as cv2.ellipse takes it. Convert a CSV
                  ``Orientation`` with ``clast_geometry.axis_angle_deg(o,
                  y_down=True)``, never by negating it.
    image_shape : (H, W)
    """
    import cv2
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    a = int(round(axes[0]))
    b = int(round(axes[1]))
    if a <= 0 or b <= 0:
        return mask
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    cv2.ellipse(mask, (cx, cy), (a, b), float(angle_deg), 0, 360, 1, thickness=-1)
    return mask


def ellipse_params_from_row(row, image_height, resolution):
    """The drawable ellipse of one saved truth-CSV row, in image pixels.

    ``row`` carries ``x``, ``y`` (pixels, y up from the bottom edge),
    ``Ellipse_major_axis``/``Ellipse_minor_axis`` (full lengths in metres,
    or in the CSV's unit) and ``Orientation`` (bearing of the long axis,
    clockwise from image-up; see ``functions.clast_geometry``);
    ``resolution`` is metres (or units) per pixel. Returns ``(center,
    (a, b), angle_deg)`` for :func:`ellipse_to_mask`: the centre in
    (col, row), the semi-axes in pixels and the rotation in the image frame
    (y down) that ``cv2.ellipse`` takes. Raises ``ValueError`` for a row
    that cannot be drawn."""
    import math
    from functions.clast_geometry import axis_angle_deg
    cx = float(row["x"])
    cy = float(image_height) - float(row["y"])
    a = float(row["Ellipse_major_axis"]) / float(resolution) / 2.0
    b = float(row["Ellipse_minor_axis"]) / float(resolution) / 2.0
    o = float(row["Orientation"])
    if not all(math.isfinite(v) for v in (cx, cy, a, b, o)) or a <= 0 or b <= 0:
        raise ValueError("row has no drawable ellipse")
    return (cx, cy), (a, b), axis_angle_deg(o, y_down=True)


# Lengths scale with the resolution, areas with its square; the rest does not.
_LINEAR_KEYS = ("Clast_length", "Clast_width", "Ellipse_major_axis",
                "Ellipse_minor_axis", "Perimeter", "Equivalent_diameter")
_AREA_KEYS = ("Surface_area",)


def record_bbox(rec, shape, pad: int = 3):
    """``(r0, r1, c0, c1)``: a window of the full-size mask that holds the
    whole record, from its parameters (polygon vertices, circle, ellipse)
    with ``pad`` pixels of margin, clipped to ``shape``; the mask's own
    extent when the parameters say nothing. None for an empty mask."""
    H, W = int(shape[0]), int(shape[1])
    params = rec.get("params") or {}
    shp = rec.get("shape")
    box = None
    try:
        if shp == "polygon" and params.get("vertices"):
            v = np.asarray(params["vertices"], dtype=float)
            box = (v[:, 1].min(), v[:, 1].max(), v[:, 0].min(), v[:, 0].max())
        elif shp == "circle":
            (cx, cy), r = params["center"], float(params["radius"])
            box = (cy - r, cy + r, cx - r, cx + r)
        elif shp == "ellipse":
            (cx, cy) = params["center"]
            r = float(max(params["axes"]))
            box = (cy - r, cy + r, cx - r, cx + r)
    except (KeyError, TypeError, ValueError, IndexError):
        box = None
    if box is not None:
        r0 = max(0, int(np.floor(box[0])) - pad)
        r1 = min(H, int(np.ceil(box[1])) + pad + 1)
        c0 = max(0, int(np.floor(box[2])) - pad)
        c1 = min(W, int(np.ceil(box[3])) + pad + 1)
        if r1 > r0 and c1 > c0:
            return r0, r1, c0, c1
    mask = rec.get("mask")
    if mask is None:
        return None
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return (max(0, rows[0] - pad), min(H, rows[-1] + pad + 1),
            max(0, cols[0] - pad), min(W, cols[-1] + pad + 1))


def measure_record_px(rec):
    """``_measure_clast`` of one record in pixel units (resolution 1), on a
    window of its mask, with the centre and the contour moved back to
    full-image pixels (column, row). Cached on the record under
    ``"_meas_px"`` for as long as ``rec["mask"]`` is the same array (an
    edit replaces the mask, which invalidates it). None when the mask is
    empty or cannot be measured as one clast."""
    from functions.clasts_detection import _measure_clast
    mask = rec.get("mask")
    if mask is None:
        return None
    cached = rec.get("_meas_px")
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] is mask:
        return cached[1]
    meas = None
    if isinstance(mask, CroppedMask):
        # Already a window; pad it so the contour closes, but never past
        # the frame, so a clast cut by the image edge is judged as in the
        # full frame.
        h, w = mask.crop.shape
        H, W = mask.shape
        top, left = min(3, mask.r0), min(3, mask.c0)
        bottom = max(0, min(3, H - (mask.r0 + h)))
        right = max(0, min(3, W - (mask.c0 + w)))
        crop = np.pad(mask.crop, ((top, bottom), (left, right)))
        if crop.any():
            meas = _measure_clast(crop, rec.get("score", 1.0), 1.0)
            if meas is not None:
                c0, r0 = mask.c0 - left, mask.r0 - top
                meas = dict(meas)
                meas["center_x"] = float(meas["center_x"]) + c0
                meas["center_y"] = float(meas["center_y"]) + r0
                meas["_contour_x"] = np.asarray(meas["_contour_x"], float) + c0
                meas["_contour_y"] = np.asarray(meas["_contour_y"], float) + r0
        rec["_meas_px"] = (mask, meas)
        return meas
    box = record_bbox(rec, mask.shape)
    if box is not None:
        r0, r1, c0, c1 = box
        crop = mask[r0:r1, c0:c1]
        # A window that misses part of the mask would measure a fragment.
        if np.any(crop) and int(np.count_nonzero(crop)) == int(np.count_nonzero(mask)):
            meas = _measure_clast(crop, rec.get("score", 1.0), 1.0)
            if meas is not None:
                meas = dict(meas)
                meas["center_x"] = float(meas["center_x"]) + c0
                meas["center_y"] = float(meas["center_y"]) + r0
                meas["_contour_x"] = np.asarray(meas["_contour_x"], float) + c0
                meas["_contour_y"] = np.asarray(meas["_contour_y"], float) + r0
        elif np.any(mask):
            meas = _measure_clast(mask, rec.get("score", 1.0), 1.0)
    rec["_meas_px"] = (mask, meas)
    return meas


def measure_records(records, image_height, resolution, *, with_contours=False):
    """The records as a clast table in the detection schema, with the row
    each came from.

    Returns ``(df, positions)``, or ``(df, positions, contours)`` with
    ``with_contours``: ``positions[k]`` is the index in ``records`` of row
    ``k`` (``clast_ID`` = k + 1), and ``contours`` maps ``clast_ID`` to the
    mask outline in the CSV's pixel frame (column, ``image_height`` - row),
    unscaled. Rows are measured in pixels once (:func:`measure_record_px`)
    and scaled by ``resolution`` (lengths) and its square (areas).
    Records whose mask is empty or unmeasurable are skipped, so a later
    record takes the next ID. A ``Label`` column follows ``clast_ID`` when
    any record carries a label."""
    import pandas as pd
    from functions.clasts_detection import _CLAST_COLUMNS
    from functions import clast_geometry as _CG

    res = float(resolution)
    out, labels, positions = [], [], []
    contours = _CG.ContourSet(frame="pixels")
    has_any_label = False
    to_frame = _CG.quadrat_frame(image_height)
    for pos, rec in enumerate(records):
        meas = measure_record_px(rec)
        if meas is None:
            continue
        lbl = (rec.get("label") or "").strip()
        if lbl:
            has_any_label = True
        labels.append(lbl)
        positions.append(pos)
        cid = len(out) + 1
        row = {
            "clast_ID": cid,
            "x": meas["center_x"],
            "y": image_height - meas["center_y"],
            "Eccentricity": meas["Eccentricity"],
            "Solidity": meas["Solidity"],
            # Manual digitisation has no image pixels here → no brightness.
            "Mean_intensity": float("nan"),
            "Score": rec.get("score", 1.0),
            "Orientation": meas["Orientation"],
        }
        for k in _LINEAR_KEYS:
            row[k] = float(meas[k]) * res
        for k in _AREA_KEYS:
            row[k] = float(meas[k]) * res * res
        out.append(row)
        if with_contours:
            try:
                ring = _CG.contour_from_measurement(meas, to_frame=to_frame)
                if ring:
                    contours[cid] = ring
            except Exception:
                pass
    df = pd.DataFrame(out, columns=_CLAST_COLUMNS)
    # Label column only when at least one record carries a label.
    if has_any_label and labels:
        df.insert(1, "Label", labels)
    if with_contours:
        return df, positions, contours
    return df, positions


def digitize_records_to_dataframe(records, image_height, resolution):
    """Convert digitized-clast records into a DataFrame with the detection
    schema (_CLAST_COLUMNS).

    records      : dicts with a 'mask' key and optionally 'score' (default 1.0)
    image_height : source image pixel height; y is flipped (image_height -
                   center_y) so the y-axis points up, as in quadrat detection
    resolution   : m/pixel

    Rows where _measure_clast returns None are skipped
    (:func:`measure_records`, which also says which record each row is).
    """
    return measure_records(records, image_height, resolution)[0]
