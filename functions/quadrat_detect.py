"""Propose where a quadrat's four corners are, and refuse when unsure.

This does not place corners: the best method measured reaches about 3% of a
side, and a ground sample distance needs 1%. The output is a starting
position for the user's own four clicks, offered only on the photographs
where it can be trusted, and never written anywhere.

One photograph the user already rectified becomes an exemplar: rectified
into a square, the quadrat's rail is a ring of known width, and everything
else (the interior gravel and the ground outside) is masked away, since it
is a different patch of beach in every shot. That ring is matched into the
new photograph and the pose is then fitted continuously.

The gate is the point of the module. Ungated, the detector will confidently
propose a quadrilateral nowhere near the quadrat, and a single matcher's own
score cannot tell a failure from a success. What works is agreement between
two semi-independent estimates, the same logic as the rival check in
``functions/georef.py``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as _dc_field
from typing import Optional

import numpy as np

__all__ = [
    "DEFAULTS",
    "refine_corners",
    "Exemplar",
    "CornerProposal",
    "build_exemplar",
    "propose",
]

DEFAULTS = {
    # Long side of the image this module works at. The gate threshold was
    # calibrated here and is not resolution-independent, so both exemplar and
    # target are fitted to this first and corners mapped back on the way out.
    "work_px": 1000,
    # Maximum disagreement between the ring placement and the corner-derived
    # placement, as a fraction of one quadrat side. Calibrated against this
    # module as it ships; a looser threshold offers more but starts to offer
    # wrong positions.
    "max_disagreement": 0.020,
    # Rectified exemplar size; larger costs time and buys nothing.
    "rect_px": 520,
    # Ground kept outside the outline, so the template carries the rail's outer
    # edge rather than ending on it.
    "pad_frac": 0.035,
    # Pose search. Rotation between consecutive photographs is under 11 degrees
    # and framing varies by under 12%. The grid is coarse on purpose: the
    # continuous fit that follows supplies the precision.
    "rot_deg": 12,
    "rot_step": 6,
    "scale_span": 0.12,
    "scale_steps": 3,
    # Corner arm length for the second, semi-independent estimate.
    "corner_arm": 0.34,
    "corner_candidates": 6,
    # Local contrast normalisation window.
    "norm_px": 41,
    "refine_iters": 45,
    "refine": True,
    # --- Snapping the offered quadrilateral onto the rails ---
    # The template is the exemplar's own rail cross-profile: the intensity
    # section across the rail with the user's pick at its origin, so it already
    # contains the shadow, both rail edges and where the pick sits among them.
    # Sampled in fractions of a side, never in pixels: a scale change between
    # photographs smears a pixel-spaced template.
    "refine_profile": 0.10,      # profile half-width, fraction of a side
    "refine_search": 0.06,       # search half-width, fraction of a side
    "refine_step": 0.0004,       # sampling pitch, fraction of a side
    "refine_stations": 61,       # profiles sampled along each edge
    "refine_min_stations": 12,   # below this an edge has no usable support
}


@dataclass
class CornerProposal:
    """Four proposed corners, and whether they may be shown."""
    corners: Optional[np.ndarray] = None      # 4x2, ORIGINAL image pixels
    disagreement: float = float("nan")        # fraction of a side
    score: float = float("nan")               # masked correlation of the fit
    accepted: bool = False
    # `refined` says whether the corners were snapped onto the rails;
    # `refine_score` is the median per-edge correlation. A refinement refusal
    # is recorded, not swallowed: the unrefined proposal still stands.
    refined: bool = False
    refine_score: float = float("nan")
    reasons: list = _dc_field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "corners": (None if self.corners is None
                        else [[float(x), float(y)] for x, y in self.corners]),
            "disagreement": float(self.disagreement),
            "score": float(self.score),
            "accepted": bool(self.accepted),
            "refined": bool(self.refined),
            "refine_score": float(self.refine_score),
            "reasons": list(self.reasons),
        }


@dataclass
class Exemplar:
    """Everything carried forward from the one photograph a person rectified."""
    ring: np.ndarray                 # rectified, contrast-normalised
    ring_mask: np.ndarray            # the rail only
    corners_rect: np.ndarray         # 4x2 in rectified coordinates
    side_px: float                   # the quadrat's side in the SOURCE photograph
    corner_templates: list           # (patch, mask, corner offset) per corner
    rail_frac: float
    # Rail cross-profile per edge, in fractions of a side; None when it could
    # not be built, in which case refinement is skipped.
    edge_profiles: Optional[list] = None


def _cv2():
    import cv2
    return cv2


def _gray(img) -> np.ndarray:
    cv2 = _cv2()
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return a.astype(np.uint8)


def _local_norm(x: np.ndarray, k: int) -> np.ndarray:
    """Flatten brightness and exposure differences between photographs, so
    the match is not dominated by how bright the day was."""
    cv2 = _cv2()
    k = int(k) | 1
    f = x.astype(np.float32)
    mu = cv2.blur(f, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - mu * mu, 1.0))
    return np.clip((f - mu) / sd * 40.0 + 128.0, 0, 255).astype(np.uint8)


def _fit_long(image, long_px):
    """Scale an image so its long side is ``long_px``. Returns (image, scale)."""
    cv2 = _cv2()
    a = np.asarray(image)
    h, w = a.shape[:2]
    m = max(h, w)
    if m <= 0:
        return a, 1.0
    k = float(long_px) / float(m)
    if abs(k - 1.0) < 1e-3:
        return a, 1.0
    return cv2.resize(a, (max(1, int(round(w * k))), max(1, int(round(h * k)))),
                      interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR), k


def _side_of(q) -> float:
    q = np.asarray(q, dtype=float)
    return float(np.median([np.hypot(*(q[(i + 1) % 4] - q[i])) for i in range(4)]))


def _masked_ncc(warped, warped_mask, target) -> float:
    """Zero-mean correlation over the mask only.

    OpenCV's masked matchTemplate has no zero-mean variant and its score
    saturates near 0.96 whether the fit is right or wrong; this does not.
    """
    m = warped_mask > 127
    if int(m.sum()) < 500:
        return -1.0
    a = warped[m].astype(np.float32)
    b = target[m].astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    da, db = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    if da < 1e-6 or db < 1e-6:
        return -1.0
    return float((a * b).sum() / (da * db))


def build_exemplar(image, corners, frame_thickness_m, quadrat_side_m,
                   settings: Optional[dict] = None) -> Exemplar:
    """Carry one hand-rectified photograph forward as a template.

    ``corners`` are the four the user placed, in image pixels, in order.
    ``frame_thickness_m`` is the rail's width and is required: the rail
    cannot be masked without it.
    """
    cv2 = _cv2()
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    if not (frame_thickness_m and frame_thickness_m > 0):
        raise ValueError(
            "The quadrat's frame thickness is needed to know which part of the "
            "picture is the frame. Set 'Frame thickness (m)' and try again.")
    if not (quadrat_side_m and quadrat_side_m > 0):
        raise ValueError(
            "The quadrat's side length is needed to convert the frame "
            "thickness into pixels. Set the segment lengths and try again.")
    q = np.asarray(corners, dtype=np.float32)
    if q.shape != (4, 2):
        raise ValueError("Four corners are required, in order.")

    small, k = _fit_long(image, cfg["work_px"])
    g = _local_norm(_gray(small), cfg["norm_px"])
    q = q * float(k)
    rect_px = int(cfg["rect_px"])
    pad = float(cfg["pad_frac"]) * rect_px
    canvas = int(round(rect_px + 2 * pad))
    dst = np.array([[pad, pad], [pad + rect_px, pad],
                    [pad + rect_px, pad + rect_px], [pad, pad + rect_px]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    ring = cv2.warpPerspective(g, M, (canvas, canvas))

    rail_frac = float(frame_thickness_m) / float(quadrat_side_m)
    w = rail_frac * rect_px
    outer = np.zeros((canvas, canvas), np.uint8)
    cv2.rectangle(outer, (0, 0), (canvas - 1, canvas - 1), 255, -1)
    hole = np.zeros((canvas, canvas), np.uint8)
    cv2.rectangle(hole, (int(pad + w), int(pad + w)),
                  (int(pad + rect_px - w), int(pad + rect_px - w)), 255, -1)
    ring_mask = cv2.bitwise_and(outer, cv2.bitwise_not(hole))

    arm = float(cfg["corner_arm"]) * rect_px
    templates = []
    for k in range(4):
        cx, cy = dst[k]
        sx = 1.0 if k in (0, 3) else -1.0
        sy = 1.0 if k in (0, 1) else -1.0
        x0, x1 = sorted([cx - sx * pad, cx + sx * arm])
        y0, y1 = sorted([cy - sy * pad, cy + sy * arm])
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(canvas, x1)), int(min(canvas, y1))
        crop = ring[y0:y1, x0:x1]
        m = np.zeros(crop.shape, np.uint8)
        lx, ly = cx - x0, cy - y0
        ax0, ax1 = sorted([lx - sx * pad, lx + sx * arm])
        ay0, ay1 = sorted([ly - sy * pad, ly + sy * w])
        m[int(max(0, ay0)):int(ay1), int(max(0, ax0)):int(ax1)] = 255
        bx0, bx1 = sorted([lx - sx * pad, lx + sx * w])
        by0, by1 = sorted([ly - sy * pad, ly + sy * arm])
        m[int(max(0, by0)):int(by1), int(max(0, bx0)):int(bx1)] = 255
        templates.append((crop, m, np.array([float(lx), float(ly)])))

    ex = Exemplar(ring=ring, ring_mask=ring_mask,
                  corners_rect=dst.astype(float), side_px=_side_of(q),
                  corner_templates=templates, rail_frac=rail_frac)
    # The rail cross-profiles come from the SOURCE photograph at the user's own
    # picks, so no resampling sits between the template and the pixels it is
    # matched against. Deliberately not wrapped in try/except: `_edge_profile`
    # returns None for the ordinary data reasons; anything else is a bug.
    gsrc = _local_norm(_gray(small), cfg["norm_px"]).astype(np.float32)
    side_src = _side_of(q)
    profs = [_edge_profile(gsrc, q[i], q[(i + 1) % 4], side_src, cfg)
             for i in range(4)]
    ex.edge_profiles = None if any(pr is None for pr in profs) else profs
    return ex


def _warp_similarity(theta_deg, scale, tx, ty, centre):
    th = math.radians(theta_deg)
    a, b = scale * math.cos(th), scale * math.sin(th)
    M = np.array([[a, -b, 0.0], [b, a, 0.0]])
    o = np.asarray(centre, dtype=float)
    M[:, 2] = o - M[:, :2] @ o + np.array([tx, ty], dtype=float)
    return M


def _grid_pose(target, ex: Exemplar, base_scale, cfg):
    """Coarse exhaustive pose. Precision comes from the fit that follows."""
    cv2 = _cv2()
    best = (-2.0, None)
    rots = range(-int(cfg["rot_deg"]), int(cfg["rot_deg"]) + 1, int(cfg["rot_step"]))
    span, n = float(cfg["scale_span"]), int(cfg["scale_steps"])
    scales = np.linspace(1.0 - span, 1.0 + span, n)
    for deg, s in itertools.product(rots, scales):
        sc = base_scale * float(s)
        t = cv2.resize(ex.ring, None, fx=sc, fy=sc)
        m = cv2.resize(ex.ring_mask, None, fx=sc, fy=sc,
                       interpolation=cv2.INTER_NEAREST)
        pts = ex.corners_rect * sc
        if deg:
            h, w = t.shape
            c = np.array([w / 2.0, h / 2.0])
            Mr = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
            t = cv2.warpAffine(t, Mr, (w, h), borderMode=cv2.BORDER_REFLECT)
            m = cv2.warpAffine(m, Mr, (w, h), flags=cv2.INTER_NEAREST)
            th = math.radians(-deg)
            R = np.array([[math.cos(th), -math.sin(th)],
                          [math.sin(th), math.cos(th)]])
            pts = (R @ (pts - c).T).T + c
        if min(t.shape) < 20 or t.shape[0] >= target.shape[0] \
                or t.shape[1] >= target.shape[1] or m.sum() < 255 * 200:
            continue
        try:
            res = cv2.matchTemplate(target, t, cv2.TM_CCORR_NORMED, mask=m)
        except Exception:
            continue
        res[~np.isfinite(res)] = -2.0
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), (pts + np.array(loc, dtype=float), deg, float(s)))
    return best[1]


def _refine(target, ex: Exemplar, M0, cfg):
    """Continuous similarity from a grid start.

    Richer models (affine, four-point homography) score worse here: more
    constraint on more data beats more freedom on fewer points.
    """
    cv2 = _cv2()
    from scipy.optimize import minimize
    H, W = target.shape
    c = np.array([ex.ring.shape[1] / 2.0, ex.ring.shape[0] / 2.0])
    a, b = M0[0, 0], M0[1, 0]
    s0 = float(np.hypot(a, b))
    th0 = float(math.degrees(math.atan2(b, a)))
    o = np.array([c[0], c[1]])
    t0 = M0[:, 2] - (o - M0[:, :2] @ o)

    def neg(p):
        th, s, tx, ty = p
        if not (0.5 < s < 2.0) or abs(th - th0) > float(cfg["rot_deg"]):
            return 1.0
        M = _warp_similarity(th, s, tx, ty, c)
        tw = cv2.warpAffine(ex.ring, M, (W, H), flags=cv2.INTER_LINEAR)
        mw = cv2.warpAffine(ex.ring_mask, M, (W, H), flags=cv2.INTER_NEAREST)
        return -_masked_ncc(tw, mw, target)

    p0 = np.array([th0, s0, t0[0], t0[1]], dtype=float)
    best, base = p0, neg(p0)
    try:
        res = minimize(neg, p0, method="Powell",
                       options=dict(xtol=0.02, ftol=1e-4,
                                    maxiter=int(cfg["refine_iters"]), disp=False))
        if np.isfinite(res.fun) and res.fun < base:
            best, base = res.x, float(res.fun)
    except Exception:
        pass
    M = _warp_similarity(best[0], best[1], best[2], best[3], c)
    return (M[:, :2] @ ex.corners_rect.T).T + M[:, 2], float(-base)


def _corner_estimate(target, ex: Exemplar, base_scale, cfg):
    """A second, semi-independent placement, from the four corners alone.

    Its own accuracy is poor; it exists so the two estimates can be compared.
    """
    cv2 = _cv2()
    K = int(cfg["corner_candidates"])
    per_corner = []
    for (patch, mask, off) in ex.corner_templates:
        found = []
        for deg in range(-int(cfg["rot_deg"]), int(cfg["rot_deg"]) + 1, 6):
            t, m, o = patch, mask, off.copy()
            sc = base_scale
            t = cv2.resize(t, None, fx=sc, fy=sc)
            m = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            o = o * sc
            if deg:
                h, w = t.shape
                cc = np.array([w / 2.0, h / 2.0])
                Mr = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
                t = cv2.warpAffine(t, Mr, (w, h), borderMode=cv2.BORDER_REFLECT)
                m = cv2.warpAffine(m, Mr, (w, h), flags=cv2.INTER_NEAREST)
                th = math.radians(-deg)
                R = np.array([[math.cos(th), -math.sin(th)],
                              [math.sin(th), math.cos(th)]])
                o = R @ (o - cc) + cc
            if min(t.shape) < 12 or t.shape[0] >= target.shape[0] \
                    or t.shape[1] >= target.shape[1] or m.sum() < 255 * 40:
                continue
            try:
                res = cv2.matchTemplate(target, t, cv2.TM_CCORR_NORMED, mask=m)
            except Exception:
                continue
            res[~np.isfinite(res)] = -2.0
            _, mx, _, loc = cv2.minMaxLoc(res)
            found.append((float(mx), np.array([loc[0] + o[0], loc[1] + o[1]])))
        found.sort(key=lambda z: -z[0])
        if not found:
            return None
        per_corner.append(found[:K])
    quad = np.array([c[0][1] for c in per_corner], dtype=np.float32)
    A = cv2.estimateAffinePartial2D(
        ex.corners_rect.astype(np.float32), quad, method=cv2.LMEDS)[0]
    if A is None:
        return None
    return (A[:, :2] @ ex.corners_rect.T).T + A[:, 2]


def propose(image, exemplar: Exemplar,
            settings: Optional[dict] = None) -> CornerProposal:
    """Where the quadrat probably is, or a refusal that says why.

    Never raises on ordinary input: a photograph the detector cannot place is
    a normal outcome. Nothing here writes anything.

    The returned corners are the right quadrilateral in an order that is only
    fixed up to a quarter-turn: the ring template is four-fold symmetric. A
    caller that cares which corner is which must establish that itself.
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    p = CornerProposal()
    try:
        small, k = _fit_long(image, cfg["work_px"])
        g = _local_norm(_gray(small), cfg["norm_px"])
    except Exception as ex_:
        p.reasons.append(
            "This photograph could not be read for corner detection "
            f"({type(ex_).__name__}). Place the corners by hand.")
        return p
    if min(g.shape[:2]) < 64:
        p.reasons.append("This photograph is too small to search.")
        return p

    base_scale = float(exemplar.side_px) / float(cfg["rect_px"])
    grid = _grid_pose(g, exemplar, base_scale, cfg)
    if grid is None:
        p.reasons.append(
            "No position for the quadrat frame was found in this photograph. "
            "Place the corners by hand.")
        return p
    pts, deg, s = grid
    cv2 = _cv2()
    c = np.array([exemplar.ring.shape[1] / 2.0, exemplar.ring.shape[0] / 2.0])
    M0 = _warp_similarity(-deg, base_scale * s, 0.0, 0.0, c)
    now = (M0[:, :2] @ exemplar.corners_rect.T).T + M0[:, 2]
    M0[:, 2] += pts.mean(0) - now.mean(0)
    corners, score = _refine(g, exemplar, M0, cfg)
    p.score = float(score)

    other = _corner_estimate(g, exemplar, base_scale, cfg)
    side = _side_of(corners)
    if other is None or side <= 0:
        p.reasons.append(
            "The second, independent check could not be run, so this position "
            "cannot be called trustworthy. Place the corners by hand.")
        return p
    p.disagreement = float(
        max(np.hypot(*(other[k] - corners[k])) for k in range(4)) / side)

    # The gate: a refusal, not a warning.
    if p.disagreement > float(cfg["max_disagreement"]):
        p.reasons.append(
            "Two independent estimates of where the quadrat is disagree by "
            f"{100 * p.disagreement:.0f}% of a side, more than the "
            f"{100 * float(cfg['max_disagreement']):.0f}% allowed, so no "
            "position is offered. Place the corners by hand.")
        return p

    # Back into the caller's full-size pixels.
    p.corners = np.asarray(corners, dtype=float) / float(k if k else 1.0)
    p.accepted = True

    # Snap onto the rails only AFTER the gate: the gate was calibrated on
    # unrefined proposals. If refinement finds no support the unrefined
    # proposal stands.
    if cfg.get("refine", True) and exemplar.edge_profiles is not None:
        fine, fsc = refine_corners(image, exemplar, p.corners, settings=cfg)
        if fine is not None:
            moved = max(float(np.hypot(*(fine[i] - p.corners[i])))
                        for i in range(4)) / max(_side_of(p.corners), 1e-6)
            # A refinement that leaps beyond the gate's own tolerance is the fit
            # running away, not correcting.
            if moved <= 4.0 * float(cfg["max_disagreement"]):
                p.corners = fine
                p.refined = True
                p.refine_score = fsc
            else:
                p.reasons.append(
                    f"The rail fit moved the position by {100 * moved:.0f}% of "
                    "a side, too far to be a correction, so the unrefined "
                    "position is offered instead.")
        else:
            # Refined offers sit at a fraction of a per cent, unrefined ones at a
            # few per cent; the user is told which they were given.
            p.reasons.append(
                "The rails could not be followed on this photograph, so the "
                "unrefined position is offered instead — expect it to be a "
                "few per cent of a side out rather than a fraction of one.")
    return p


# --- Refine a proposed quadrilateral onto the rails ---
def _sample(img, xy):
    """Bilinear sample at float coordinates; NaN outside the image."""
    x, y = xy[:, 0], xy[:, 1]
    h, w = img.shape
    ok = (x >= 0) & (x <= w - 2) & (y >= 0) & (y <= h - 2)
    out = np.full(len(x), np.nan, dtype=np.float32)
    if not ok.any():
        return out
    xi, yi = np.floor(x[ok]).astype(int), np.floor(y[ok]).astype(int)
    fx, fy = x[ok] - xi, y[ok] - yi
    out[ok] = (img[yi, xi] * (1 - fx) * (1 - fy)
               + img[yi, xi + 1] * fx * (1 - fy)
               + img[yi + 1, xi] * (1 - fx) * fy
               + img[yi + 1, xi + 1] * fx * fy)
    return out


def _profile_at(img, base, n, offs):
    """Intensity along the normal at ``base``, at the given offsets in px."""
    p = _sample(img, base[None, :] + offs[:, None] * n[None, :])
    if np.isnan(p).mean() > 0.1:
        return None
    m = float(np.nanmean(p))
    return np.nan_to_num(p, nan=(m if np.isfinite(m) else 0.0))


def _ncc1(a, b):
    a = a - a.mean()
    b = b - b.mean()
    da, db = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    return float((a * b).sum() / (da * db)) if da > 1e-6 and db > 1e-6 else -1.0


def _edge_profile(img, a, b, side, cfg):
    """The exemplar's cross-profile for one edge; the pick sits at offset 0."""
    d = np.asarray(b, float) - np.asarray(a, float)
    L = float(np.hypot(*d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    offs = np.arange(-cfg["refine_profile"], cfg["refine_profile"] + 1e-9,
                     cfg["refine_step"]) * side
    acc = []
    for t in np.linspace(0.12, 0.88, int(cfg["refine_stations"])):
        pr = _profile_at(img, np.asarray(a, float) + t * L * u, n, offs)
        if pr is not None:
            acc.append(pr)
    if len(acc) < int(cfg["refine_stations"]) // 3:
        return None
    return np.median(np.stack(acc), axis=0)


def _refine_edge(img, a, b, side, tmpl, cfg):
    """Fit the rail's line for one edge. Returns ((point, direction), score)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    L = float(np.hypot(*d))
    if L < 1e-6:
        return None, -1.0
    u = d / L
    n = np.array([-u[1], u[0]])
    step = cfg["refine_step"]
    half = int(round(cfg["refine_search"] / step))
    wide = np.arange(-(cfg["refine_profile"] + cfg["refine_search"]),
                     cfg["refine_profile"] + cfg["refine_search"] + 1e-9,
                     step) * side
    pts, scores = [], []
    for t in np.linspace(0.10, 0.90, int(cfg["refine_stations"])):
        base = np.asarray(a, float) + t * L * u
        pr = _profile_at(img, base, n, wide)
        if pr is None:
            continue
        best, bi = -2.0, None
        for sft in range(0, 2 * half + 1):
            seg = pr[sft:sft + len(tmpl)]
            if len(seg) != len(tmpl):
                continue
            v = _ncc1(tmpl, seg)
            if v > best:
                best, bi = v, sft
        if bi is None or bi <= 0 or bi >= 2 * half:
            continue
        y0 = _ncc1(tmpl, pr[bi - 1:bi - 1 + len(tmpl)])
        y2 = _ncc1(tmpl, pr[bi + 1:bi + 1 + len(tmpl)])
        den = y0 - 2 * best + y2
        dlt = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-12 else 0.0
        pts.append(base + ((bi + dlt) - half) * step * side * n)
        scores.append(best)
    if len(pts) < int(cfg["refine_min_stations"]):
        return None, -1.0
    P = np.array(pts)
    c = P.mean(0)
    dd = P - c
    _, vec = np.linalg.eigh(dd.T @ dd)
    v = vec[:, -1] / np.linalg.norm(vec[:, -1])
    for _ in range(3):
        # Robust: a buried or shadowed stretch of rail must not drag the line.
        r = np.abs((P - c) @ np.array([-v[1], v[0]]))
        keep = r < max(1.0, 2.5 * float(np.median(r)))
        if keep.sum() < int(cfg["refine_min_stations"]):
            break
        c = P[keep].mean(0)
        dd = P[keep] - c
        _, vec = np.linalg.eigh(dd.T @ dd)
        v = vec[:, -1] / np.linalg.norm(vec[:, -1])
    return (c, v), float(np.median(scores))


def refine_corners(image, exemplar: "Exemplar", seed,
                   settings: Optional[dict] = None):
    """Snap a seeded quadrilateral onto the rails. Returns (corners, score).

    Returns ``(None, -1.0)`` when an edge has no usable support, an ordinary
    outcome; the caller keeps its unrefined seed. Seeded with the reference
    position it stays there, which is the property that makes it a refiner.
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    if exemplar.edge_profiles is None:
        return None, -1.0
    try:
        small, k = _fit_long(image, cfg["work_px"])
        img = _local_norm(_gray(small), cfg["norm_px"]).astype(np.float32)
    except Exception:
        return None, -1.0
    q = np.asarray(seed, dtype=float) * float(k)
    side = _side_of(q)
    if side <= 0:
        return None, -1.0
    lines, scores = [], []
    for i in range(4):
        ln, sc = _refine_edge(img, q[i], q[(i + 1) % 4], side,
                              exemplar.edge_profiles[i], cfg)
        if ln is None:
            return None, -1.0
        lines.append(ln)
        scores.append(sc)
    out = []
    for i in range(4):
        (c1, v1), (c2, v2) = lines[(i - 1) % 4], lines[i]
        A = np.array([v1, -v2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return None, -1.0
        t = np.linalg.solve(A, c2 - c1)
        out.append(c1 + t[0] * v1)
    return np.asarray(out, dtype=float) / float(k if k else 1.0),         float(np.median(scores))
