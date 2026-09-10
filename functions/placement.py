"""Where a quadrat sits in the world, and every way a person may move it.

A value object for a placement, and pure functions for every operation the
editor performs on one; nothing here touches the GUI.

Two conventions, both load-bearing.

**The matrix maps quadrat pixels to world coordinates.** Quadrat pixels are
(col, row) with row increasing downward; world is (easting, northing) with
northing increasing upward. So the transform contains a reflection and its
determinant is negative. A placement rotated by ``theta`` sends the +col axis to
``s * (cos, sin)`` and the +row axis to ``s * (sin, -cos)``.

**Rotation is the bearing of the +col axis, anticlockwise from east**, i.e.
``atan2(m[1, 0], m[0, 0])``, the same convention ``functions.georef`` reports.

**Scale is not editable.** It is known (two measured ground sample distances
and a square of stated size), so a control over it can only introduce error.
A placement built from a fitted matrix keeps that matrix's scale rather than
snapping to the ratio, so merely keeping an untouched fit does not alter it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

__all__ = [
    "Placement",
    "placement_from_matrix",
    "matrix_from_placement",
    "placement_from_seed",
    "corners_world",
    "corners_display",
    "world_from_display",
    "translate",
    "drag_corner",
    "nudge",
    "set_rotation",
    "placement_delta",
    "agreement_score",
    "blend_layer",
    "SCORE_FLOOR",
]

# A saved placement scoring below this is refused. A correct placement reads
# 0.40-0.45 and collapses to ~0.01 within 5 cm or 2 degrees; the weakest fit
# the matcher accepts reads ~0.27. The score is recorded in the sidecar.
SCORE_FLOOR = 0.20


@dataclass(frozen=True)
class Placement:
    """Where the quadrat sits, as the editor thinks of it.

    ``easting``/``northing`` are the quadrat's CENTRE, which is the natural
    centre of rotation and keeps ``set_rotation`` from also translating.
    ``rotation_deg`` is the bearing of the +col axis anticlockwise from east.
    ``scale_m_per_px`` is metres of ground per quadrat pixel. ``quad_shape`` is
    ``(rows, cols)`` of the quadrat image, needed to turn the centre back into a
    transform.

    Frozen, so equality is structural and ``Reset`` can be tested with ``==``.
    """

    easting: float
    northing: float
    rotation_deg: float
    scale_m_per_px: float
    quad_shape: Tuple[int, int]


def matrix_from_placement(p: Placement) -> np.ndarray:
    """The 3x3 transform taking quadrat pixels (col, row) to world (E, N)."""
    s = float(p.scale_m_per_px)
    a = math.radians(float(p.rotation_deg))
    c, sn = math.cos(a), math.sin(a)
    # +col -> s*(c, sn); +row -> s*(sn, -c). The second row's sign is the
    # row-down / north-up reflection, not a mistake: det = -s^2.
    R = np.array([[s * c, s * sn], [s * sn, -s * c]], dtype=float)
    rows, cols = p.quad_shape
    centre_px = np.array([cols / 2.0, rows / 2.0], dtype=float)
    t = np.array([p.easting, p.northing], dtype=float) - R @ centre_px
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = t
    return M


def placement_from_matrix(matrix, quad_shape) -> Placement:
    """Read a fitted transform back into a Placement.

    Keeps the matrix's own scale (a fitted scale is near but not equal to the
    GSD ratio), so an untouched placement does not differ from its fit.
    """
    M = np.asarray(matrix, dtype=float)
    scale = float(math.hypot(M[0, 0], M[1, 0]))
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    rows, cols = int(quad_shape[0]), int(quad_shape[1])
    centre = M[:2, :2] @ np.array([cols / 2.0, rows / 2.0]) + M[:2, 2]
    return Placement(float(centre[0]), float(centre[1]), rot, scale,
                     (rows, cols))


def placement_from_seed(seed_xy, quad_shape, quadrat_gsd_m,
                        ortho_gsd_m=None) -> Placement:
    """A starting placement when the matcher refused and there is no fit:
    centred on the seed, north-up, at the quadrat's own GSD.

    ``ortho_gsd_m`` is accepted and ignored: the scale in world metres per
    quadrat pixel is the quadrat's own GSD, not a ratio.
    """
    rows, cols = int(quad_shape[0]), int(quad_shape[1])
    return Placement(float(seed_xy[0]), float(seed_xy[1]), 0.0,
                     float(quadrat_gsd_m), (rows, cols))


def corners_world(p: Placement) -> np.ndarray:
    """The quadrat's four corners in world coordinates, (4, 2).

    Order follows the image: (0,0), (cols,0), (cols,rows), (0,rows) — so corner
    0 is the pixel origin and corner 2 is diagonally opposite it.
    """
    rows, cols = p.quad_shape
    q = np.array([[0.0, 0.0], [cols, 0.0], [cols, rows], [0.0, rows]])
    M = matrix_from_placement(p)
    return (M[:2, :2] @ q.T).T + M[:2, 2]


def corners_display(p: Placement, window_origin_xy, ortho_gsd_m,
                    disp_scale: float = 1.0) -> np.ndarray:
    """The corners in the coordinates the browser reports, (4, 2).

    ``window_origin_xy`` is the world coordinate of the window's top-left pixel;
    ``disp_scale`` is original raster pixels per displayed pixel, as
    ``prepare_browser_image`` returns.
    """
    w = corners_world(p)
    gsd = float(ortho_gsd_m)
    x = (w[:, 0] - float(window_origin_xy[0])) / gsd / float(disp_scale)
    y = (float(window_origin_xy[1]) - w[:, 1]) / gsd / float(disp_scale)
    return np.column_stack([x, y])


def world_from_display(x, y, window_origin_xy, ortho_gsd_m,
                       disp_scale: float = 1.0) -> Tuple[float, float]:
    """The inverse of :func:`corners_display`, for one point."""
    gsd = float(ortho_gsd_m)
    return (float(window_origin_xy[0]) + float(x) * float(disp_scale) * gsd,
            float(window_origin_xy[1]) - float(y) * float(disp_scale) * gsd)


def translate(p: Placement, dx_m: float, dy_m: float) -> Placement:
    """Move the placement, changing nothing else."""
    return replace(p, easting=p.easting + float(dx_m),
                   northing=p.northing + float(dy_m))


def nudge(p: Placement, dcol: int, drow: int, ortho_gsd_m: float) -> Placement:
    """Move by whole ORTHO pixels, the finest step a placement is worth
    (a drag cannot express less than one screen pixel)."""
    g = float(ortho_gsd_m)
    return translate(p, float(dcol) * g, -float(drow) * g)


def set_rotation(p: Placement, rotation_deg: float) -> Placement:
    """Set the bearing, about the unmoved centre."""
    return replace(p, rotation_deg=float(rotation_deg))


def drag_corner(p: Placement, corner_index: int, target_xy) -> Placement:
    """Rotate so a corner swings towards the pointer, about the opposite corner.

    Scale is fixed, so the dragged corner is confined to a circle: it follows
    the ray from the opposite corner towards the pointer at its own distance.
    """
    i = int(corner_index) % 4
    opp = (i + 2) % 4
    cw = corners_world(p)
    pivot = cw[opp]
    v_now = cw[i] - pivot
    v_want = np.asarray(target_xy, dtype=float) - pivot
    if not np.hypot(*v_want) > 0 or not np.hypot(*v_now) > 0:
        return p            # pointer on the pivot: bearing undefined, hold
    # World is right-handed with northing up; the placement's rotation is
    # measured the same way, so the world-frame angle change applies directly.
    d = math.degrees(math.atan2(v_want[1], v_want[0])
                     - math.atan2(v_now[1], v_now[0]))
    rotated = replace(p, rotation_deg=p.rotation_deg + d)
    # Rotating about the centre moved the pivot; put it back.
    moved = corners_world(rotated)[opp]
    return translate(rotated, float(pivot[0] - moved[0]),
                     float(pivot[1] - moved[1]))


def placement_delta(fitted: Placement, saved: Placement) -> dict:
    """How far a placement was moved from the one it started as."""
    d = math.hypot(saved.easting - fitted.easting,
                   saved.northing - fitted.northing)
    dr = (saved.rotation_deg - fitted.rotation_deg + 180.0) % 360.0 - 180.0
    ratio = (saved.scale_m_per_px / fitted.scale_m_per_px
             if fitted.scale_m_per_px else float("nan"))
    return {"translation_m": float(d), "rotation_deg": float(dr),
            "scale_ratio": float(ratio)}


# --- Rendering and measuring ---
def _highpass(a, sigma: float = 3.0):
    import cv2
    f = np.asarray(a, dtype=np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), float(sigma))


def _prepare_quadrat(quad, p: Placement, ortho_gsd_m: float,
                     frame_inset_px: int = 0):
    """The quadrat cropped of its frame and dropped to the ortho's pixel size.

    Downsampling the quadrat rather than upsampling the ortho: a finer
    quadrat drawn at native resolution makes the seam obvious however good
    the alignment is.
    """
    import cv2
    a = np.asarray(quad)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    n = int(frame_inset_px)
    if n > 0 and 2 * n < min(a.shape[:2]):
        a = a[n:a.shape[0] - n, n:a.shape[1] - n]
    pre = float(p.scale_m_per_px) / float(ortho_gsd_m)
    if pre <= 0:
        return a.astype(np.float32)
    # Never upward: an ortho pixel is the finest real detail either side has.
    interp = cv2.INTER_AREA if pre < 1.0 else cv2.INTER_LINEAR
    out = cv2.resize(a.astype(np.float32), None, fx=pre, fy=pre,
                     interpolation=interp)
    return out


def blend_layer(quad, p: Placement, window_shape, window_origin_xy,
                ortho_gsd_m: float, frame_inset_px: int = 0):
    """The quadrat painted into the ortho window's frame, plus its coverage.

    Returns ``(rgb, alpha)`` both shaped like the window. ``alpha`` is 1 inside
    the inset quadrilateral and 0 everywhere else, including the frame ring,
    which is equipment absent from the ortho. The quadrat is warped by the
    placement's own affine; nothing is rotated into a bounding box.
    """
    import cv2
    h, w = int(window_shape[0]), int(window_shape[1])
    gsd = float(ortho_gsd_m)
    n = int(frame_inset_px)
    src = _prepare_quadrat(quad, p, gsd, n)
    sh, sw = src.shape[:2]

    rows, cols = p.quad_shape
    inset = np.array([[n, n], [cols - n, n], [cols - n, rows - n],
                      [n, rows - n]], dtype=float)
    M = matrix_from_placement(p)
    world = (M[:2, :2] @ inset.T).T + M[:2, 2]
    dst = np.column_stack([
        (world[:, 0] - float(window_origin_xy[0])) / gsd,
        (float(window_origin_xy[1]) - world[:, 1]) / gsd])
    A = cv2.getAffineTransform(
        np.float32([[0, 0], [sw, 0], [sw, sh]]), np.float32(dst[:3]))
    rgb = cv2.warpAffine(src, A, (w, h), flags=cv2.INTER_LINEAR,
                         borderValue=0.0)
    alpha = cv2.warpAffine(np.ones((sh, sw), np.float32), A, (w, h),
                           flags=cv2.INTER_NEAREST, borderValue=0.0)
    return rgb, (alpha > 0.5).astype(np.float32)


def agreement_score(p: Placement, quad, ortho_window, window_origin_xy,
                    ortho_gsd_m: float, frame_inset_px: int = 0) -> float:
    """How well the ortho under this placement agrees with the quadrat.

    High-pass normalised cross-correlation, frame excluded: the one check a
    hand-dragged placement cannot bypass. As sharp in rotation as in
    translation (half a degree displaces a corner by 6 mm while leaving the
    centre perfect). About 1 ms once the window is filtered, so it is
    affordable live if the caller filters the window once and reuses it.
    """
    import cv2
    win = np.asarray(ortho_window, dtype=np.float32)
    if win.ndim == 3:
        win = win[..., :3].mean(axis=2)
    wh = _highpass(win)

    src = _prepare_quadrat(quad, p, ortho_gsd_m, int(frame_inset_px))
    qh = _highpass(src)
    sh, sw = qh.shape[:2]
    if sh < 8 or sw < 8:
        return float("nan")

    n = int(frame_inset_px)
    rows, cols = p.quad_shape
    inset = np.array([[n, n], [cols - n, n], [cols - n, rows - n]], dtype=float)
    M = matrix_from_placement(p)
    world = (M[:2, :2] @ inset.T).T + M[:2, 2]
    gsd = float(ortho_gsd_m)
    dst = np.column_stack([
        (world[:, 0] - float(window_origin_xy[0])) / gsd,
        (float(window_origin_xy[1]) - world[:, 1]) / gsd])
    # Sample the ortho INTO the quadrat's frame, so the comparison happens at
    # the quadrat's resolution and the window is never resampled up.
    A = cv2.getAffineTransform(np.float32(dst),
                               np.float32([[0, 0], [sw, 0], [sw, sh]]))
    samp = cv2.warpAffine(wh, A, (sw, sh), flags=cv2.INTER_LINEAR,
                          borderValue=0.0)
    a = qh.ravel().astype(np.float64)
    b = samp.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denom if denom > 0 else 0.0
