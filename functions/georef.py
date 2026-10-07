"""Place a rectified quadrat photograph into a UAV ortho's coordinate frame.

The search is local (the user gives a rough seed) and scale is known on both
sides (quadrat GSD from Orthorectify, ortho GSD from its GeoTransform), so
only rotation and a metre-scale translation are unknown. The failure to
design against is a confident wrong match, which corrupts every validation
statistic downstream: every result carries an inlier count, a residual and a
fitted scale checked against the GSD ratio, a match failing any gate is
rejected rather than warned about, and nothing is written for a rejection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MatchQuality",
    "QuadratMatch",
    "DEFAULTS",
    "match_quadrat",
    "world_from_pixels",
]

# Thresholds, deliberately conservative: a rejected match costs a retry, an
# accepted wrong one costs the validation numbers.
DEFAULTS = {
    "search_radius_m": 10.0,
    # Metres of quadrat frame to exclude from matching, inward from each
    # edge: the frame is high-contrast equipment, not ground, and its
    # keypoints are noise. Set it to the frame's thickness.
    "frame_inset_m": 0.0,
    "min_inliers": 12,          # correspondences supporting the model
    # With a real search window most correspondences are spurious by
    # construction, so the fraction is a weak signal; the absolute inlier
    # count and the residual are the real gates.
    "min_inlier_fraction": 0.03,
    "max_residual_m": 0.05,     # RMS reprojection error
    "scale_tolerance": 0.15,    # fitted scale vs the GSD ratio, fractional
    "ambiguity_ratio": 0.75,    # 2nd-best model this close ⇒ ambiguous
    # Unseeded RANSAC gives a different coverage on every re-run; None
    # restores random sampling.
    "ransac_seed": 20260805,
    # SIFT + Lowe's ratio gives better coverage, accuracy and speed than
    # ORB + crossCheck; descriptor="orb", match_ratio=0 recovers the latter.
    "descriptor": "sift",
    "match_ratio": 0.75,
    # Keypoints per image. Matching needs a keypoint DENSITY (~50,000 per
    # megapixel), so the ortho budget scales with the searched area, capped;
    # past the cap, tile rather than enlarge the window.
    "orb_keypoints": 800,
    "orb_keypoints_per_megapixel": 50000,
    "orb_keypoints_max": 20000,
}


@dataclass
class MatchQuality:
    """How well the quadrat matched, and whether that is good enough."""
    n_correspondences: int = 0
    n_inliers: int = 0
    inlier_fraction: float = 0.0
    residual_m: float = float("nan")
    scale: float = float("nan")
    scale_expected: float = float("nan")
    rotation_deg: float = float("nan")
    accepted: bool = False
    frame_inset_px: int = 0
    rival_inliers: int = 0      # best competing placement, 0 = none found
    # Whether the rival search ran; "could not look" must not read as a pass.
    rival_checked: bool = True
    rival_error: str = ""
    # Placements that could not be chosen between, best first, so a person
    # can resolve the ambiguity by looking.
    candidates: list = _dc_field(default_factory=list)
    reasons: list = _dc_field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_correspondences": self.n_correspondences,
            "n_inliers": self.n_inliers,
            "inlier_fraction": round(float(self.inlier_fraction), 4),
            "residual_m": (None if not np.isfinite(self.residual_m)
                           else round(float(self.residual_m), 5)),
            "scale": (None if not np.isfinite(self.scale)
                      else round(float(self.scale), 6)),
            "scale_expected": (None if not np.isfinite(self.scale_expected)
                               else round(float(self.scale_expected), 6)),
            "rotation_deg": (None if not np.isfinite(self.rotation_deg)
                             else round(float(self.rotation_deg), 3)),
            "accepted": bool(self.accepted),
            "frame_inset_px": int(self.frame_inset_px),
            "rival_inliers": int(self.rival_inliers),
            "rival_checked": bool(self.rival_checked),
            "candidates": list(self.candidates),
            "reasons": list(self.reasons),
        }


@dataclass
class QuadratMatch:
    """The fitted placement of a quadrat in an ortho.

    ``matrix`` is the 3x3 similarity transform taking quadrat PIXEL
    coordinates (col, row) to ortho WORLD coordinates. ``quality`` says
    whether it may be trusted; callers must not write anything when
    ``quality.accepted`` is False.
    """
    matrix: Optional[np.ndarray]
    quality: MatchQuality
    seed_xy: Optional[Tuple[float, float]] = None
    quadrat_gsd_m: Optional[float] = None
    ortho_gsd_m: Optional[float] = None

    @property
    def accepted(self) -> bool:
        return bool(self.quality.accepted)


def _finite_positive(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def world_from_pixels(matrix, cols, rows) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a fitted transform to pixel coordinates (col, row) -> world (x, y)."""
    m = np.asarray(matrix, dtype=float)
    c = np.asarray(cols, dtype=float).ravel()
    r = np.asarray(rows, dtype=float).ravel()
    pts = np.vstack([c, r, np.ones_like(c)])
    out = m @ pts
    return out[0], out[1]


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Single-band float image in 0..1, whatever the input looked like."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2) if a.shape[2] >= 3 else a[..., 0]
    a = a.astype(np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.float64)
    a = np.where(finite, a, 0.0)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.float64)
    return (a - lo) / (hi - lo)


def _decompose(matrix: np.ndarray) -> Tuple[float, float]:
    """(scale, rotation in degrees) of a similarity matrix."""
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = math.hypot(a, b)
    rot = math.degrees(math.atan2(b, a))
    return scale, rot


def match_quadrat(
    quadrat_gray: np.ndarray,
    ortho_window_gray: np.ndarray,
    *,
    quadrat_gsd_m: float,
    ortho_gsd_m: float,
    window_origin_xy: Tuple[float, float],
    seed_xy: Optional[Tuple[float, float]] = None,
    search_radius_m: Optional[float] = None,
    frame_inset_m: Optional[float] = None,
    settings: Optional[dict] = None,
) -> QuadratMatch:
    """Fit a similarity transform placing the quadrat inside an ortho window
    whose top-left pixel is at ``window_origin_xy`` (world) with pixel size
    ``ortho_gsd_m``. Returns a :class:`QuadratMatch`; never raises, since a
    quadrat that cannot be matched is an ordinary outcome."""
    from skimage.measure import ransac
    from skimage.transform import SimilarityTransform

    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    q = MatchQuality()
    q.scale_expected = (float(quadrat_gsd_m) / float(ortho_gsd_m)
                        if _finite_positive(quadrat_gsd_m)
                        and _finite_positive(ortho_gsd_m) else float("nan"))

    if not _finite_positive(quadrat_gsd_m) or not _finite_positive(ortho_gsd_m):
        q.reasons.append(
            "The ground sample distance of the quadrat or of the ortho is "
            "unknown, so scale cannot be constrained. Set both and retry.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)

    a_full = _to_gray(quadrat_gray)
    b = _to_gray(ortho_window_gray)
    if a_full.size == 0 or b.size == 0:
        q.reasons.append("The quadrat or the ortho window is empty.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)

    # Match inside the frame only; the crop offset is composed back in at the
    # end so the returned transform still maps ORIGINAL quadrat pixels.
    inset_m = float(frame_inset_m if frame_inset_m is not None
                    else cfg.get("frame_inset_m", 0.0) or 0.0)
    inset_px = int(round(max(0.0, inset_m) / float(quadrat_gsd_m)))
    if inset_px > 0:
        h_f, w_f = a_full.shape[:2]
        if 2 * inset_px >= min(h_f, w_f):
            q.reasons.append(
                f"A frame inset of {inset_m:.3f} m removes the whole quadrat "
                f"({inset_px} px from each edge of a {w_f}x{h_f} px image). "
                "Check the frame thickness.")
            return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)
        a = a_full[inset_px:h_f - inset_px, inset_px:w_f - inset_px]
    else:
        a = a_full
    q.frame_inset_px = inset_px

    # Resample the quadrat to the ortho's pixel size first; applying the scale
    # prior makes the match far more reliable.
    from skimage.transform import rescale
    pre_scale = float(quadrat_gsd_m) / float(ortho_gsd_m)
    if not (0.05 < pre_scale < 20.0):
        q.reasons.append(
            f"The two ground sample distances differ by {pre_scale:.3g}×, "
            "which is too far apart to be the same scene.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)
    a_rs = rescale(a, pre_scale, anti_aliasing=True, preserve_range=True)
    if min(a_rs.shape) < 16:
        q.reasons.append(
            "At the ortho's resolution the quadrat is only "
            f"{min(a_rs.shape)} pixels across — too small to match.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)

    n_kp = int(cfg.get("orb_keypoints", 800))
    # Ortho budget proportional to its area, so a wider search does not thin
    # out the keypoints on the target.
    per_mp = float(cfg.get("orb_keypoints_per_megapixel", 0) or 0)
    n_kp_b = n_kp
    if per_mp > 0:
        n_kp_b = max(n_kp, int(per_mp * b.size / 1e6))
        n_kp_b = min(n_kp_b, int(cfg.get("orb_keypoints_max", 20000)))
    src_all, dst_all = _detect_and_match(a_rs, b, n_kp, n_kp_b, settings=cfg)
    if src_all is None:
        q.reasons.append(
            "No describable features were found in the quadrat or in this "
            "part of the ortho. A uniform or deeply shadowed quadrat cannot "
            "be located by image content.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)
    q.n_correspondences = int(len(src_all))
    if q.n_correspondences < max(4, int(cfg["min_inliers"]) // 2):
        q.reasons.append(
            f"Only {q.n_correspondences} correspondences were found between "
            "the quadrat and the ortho; the quadrat may not be in this image.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)

    src, dst = src_all, dst_all                       # already (col, row)
    seed = cfg.get("ransac_seed", None)
    try:
        model, inliers = ransac(
            (src, dst), SimilarityTransform, min_samples=3,
            residual_threshold=2.0, max_trials=2000, rng=seed)
    except TypeError:
        # Older scikit-image has no rng argument; unseeded beats no result.
        try:
            model, inliers = ransac(
                (src, dst), SimilarityTransform, min_samples=3,
                residual_threshold=2.0, max_trials=2000)
        except Exception:
            model, inliers = None, None
    except Exception:
        model, inliers = None, None
    if model is None or inliers is None or not inliers.any():
        q.reasons.append(
            "No consistent alignment was found among the correspondences.")
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, ortho_gsd_m)

    q.n_inliers = int(inliers.sum())
    q.inlier_fraction = float(q.n_inliers) / float(max(1, len(src)))
    resid_px = float(np.sqrt(np.mean(
        np.sum((model(src[inliers]) - dst[inliers]) ** 2, axis=1))))
    q.residual_m = resid_px * float(ortho_gsd_m)

    # Compose: original quadrat px -> cropped px -> resampled px -> window px -> world.
    to_rs = np.array([[pre_scale, 0.0, -pre_scale * inset_px],
                      [0.0, pre_scale, -pre_scale * inset_px],
                      [0.0, 0.0, 1.0]])
    win_to_world = np.array(
        [[float(ortho_gsd_m), 0.0, float(window_origin_xy[0])],
         [0.0, -float(ortho_gsd_m), float(window_origin_xy[1])],
         [0.0, 0.0, 1.0]])
    matrix = win_to_world @ np.asarray(model.params, dtype=float) @ to_rs
    q.scale, q.rotation_deg = _decompose(
        np.asarray(model.params, dtype=float) @ to_rs)

    # Rival check: a beach is full of look-alike patches, so the best fit
    # means nothing unless it clearly beats what the remaining
    # correspondences can support.
    rest = ~inliers
    if int(rest.sum()) >= 3:
        try:
            # "No inliers found" is the expected, welcome answer here.
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                try:
                    rival, rival_in = ransac(
                        (src[rest], dst[rest]), SimilarityTransform,
                        min_samples=3, residual_threshold=2.0,
                        max_trials=1000, rng=seed)
                except TypeError:
                    rival, rival_in = ransac(
                        (src[rest], dst[rest]), SimilarityTransform,
                        min_samples=3, residual_threshold=2.0, max_trials=1000)
            if rival is not None and rival_in is not None:
                q.rival_inliers = int(rival_in.sum())
            q.rival_checked = True
        except Exception as ex:
            # Not rival_inliers = 0: zero means "no second placement" and the
            # gate below reads it as a pass.
            q.rival_checked = False
            q.rival_error = f"{type(ex).__name__}: {ex}"

    # --- gates: each is a rejection, not a warning ---
    if not q.rival_checked:
        q.reasons.append(
            "The check for a second, equally good placement could not be run"
            + (f" ({q.rival_error})" if q.rival_error else "")
            + ", so this placement cannot be called unambiguous. Re-run, or "
            "place it by hand in the alignment editor and judge it there.")
    if (q.n_inliers > 0
            and q.rival_inliers >= float(cfg["ambiguity_ratio"]) * q.n_inliers):
        q.reasons.append(
            f"A second, unrelated placement is almost as good "
            f"({q.rival_inliers} correspondences against {q.n_inliers}), so "
            "the quadrat cannot be located unambiguously. Narrow the search "
            "with a seed position, or use a quadrat with more distinctive "
            "ground in it.")
    if q.n_inliers < int(cfg["min_inliers"]):
        q.reasons.append(
            f"Only {q.n_inliers} of {q.n_correspondences} correspondences "
            f"agreed on a placement (at least {int(cfg['min_inliers'])} are "
            "required). The quadrat is probably not in this ortho.")
    if q.inlier_fraction < float(cfg["min_inlier_fraction"]):
        q.reasons.append(
            f"Only {100 * q.inlier_fraction:.0f}% of correspondences agreed, "
            "which is the signature of a repetitive or ambiguous scene.")
    if not np.isfinite(q.residual_m) or q.residual_m > float(cfg["max_residual_m"]):
        q.reasons.append(
            f"The alignment is off by {q.residual_m * 100:.1f} cm on average, "
            f"more than the {float(cfg['max_residual_m']) * 100:.0f} cm "
            "allowed.")
    if np.isfinite(q.scale_expected) and q.scale_expected > 0:
        rel = abs(q.scale - q.scale_expected) / q.scale_expected
        if rel > float(cfg["scale_tolerance"]):
            # Name the likely cause: a quadrat rectified with the wrong segment
            # lengths declares a wrong GSD and fails this gate every time.
            # quadrat_gray, not a_rs: a_rs is already on the ortho's grid.
            side_m = (max(quadrat_gray.shape) * float(quadrat_gsd_m)
                      if _finite_positive(quadrat_gsd_m) else None)
            hint = ""
            if side_m:
                hint = (f" This image says it is {side_m:.3f} m across — if "
                        "that is not the size of the quadrat, it was "
                        "rectified with the wrong segment lengths and needs "
                        "doing again.")
            q.reasons.append(
                f"The fitted scale is {rel * 100:.0f}% away from the one the "
                "two ground sample distances imply, so the match has locked "
                "onto the wrong thing." + hint)
    if seed_xy is not None:
        radius = float(search_radius_m if search_radius_m is not None
                       else cfg["search_radius_m"])
        cx, cy = _centre_world(matrix, a_full.shape)
        d = math.hypot(cx - float(seed_xy[0]), cy - float(seed_xy[1]))
        if d > radius:
            q.reasons.append(
                f"The best placement is {d:.1f} m from the seed position, "
                f"beyond the {radius:.0f} m search radius. Check the seed.")

    q.accepted = not q.reasons
    return QuadratMatch(matrix if q.accepted else None, q, seed_xy,
                        float(quadrat_gsd_m), float(ortho_gsd_m))


def _centre_world(matrix: np.ndarray, shape) -> Tuple[float, float]:
    """World coordinate of the quadrat's centre under a fitted transform."""
    rows, cols = shape[0], shape[1]
    x, y = world_from_pixels(matrix, [cols / 2.0], [rows / 2.0])
    return float(x[0]), float(y[0])


# --- Writing the result ---
def write_georeferenced(
    match: "QuadratMatch",
    quadrat_image,
    out_dir,
    stem: str,
    *,
    crs_wkt: str = "",
    clasts=None,
    x_col: str = "x",
    y_col: str = "y",
    source_files: Optional[dict] = None,
    hand_placed: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """Write the GeoTIFF, the world-coordinate CSV and the audit sidecar;
    returns the paths written. Refuses when the match was not accepted.

    ``hand_placed`` is the only way past that refusal: ``{"matrix": 3x3,
    "provenance": "hand-edited"|"manual", "agreement_score": float, "delta",
    "operator", "view"}``. It is what gets written; the match then supplies
    only the quality report of the FIT, and the sidecar says so.

    ``overwrite`` must be set explicitly to replace an existing stem, since
    ``validation/georectified/`` holds hand-made reference placements.
    """
    import json

    from functions.placement import SCORE_FLOOR

    hand = dict(hand_placed) if hand_placed else None
    if hand is None:
        if match is None or not match.accepted or match.matrix is None:
            raise ValueError(
                "Refusing to write outputs for a match that was not accepted: "
                + ("; ".join(match.quality.reasons) if match is not None
                   else "no match"))
        m = np.asarray(match.matrix, dtype=float)
        provenance = "fitted"
    else:
        m = np.asarray(hand["matrix"], dtype=float)
        provenance = str(hand.get("provenance") or "hand-edited")
        score = hand.get("agreement_score")
        if score is not None and float(score) < SCORE_FLOOR:
            raise ValueError(
                f"Refusing to write this placement: it agrees with the ortho "
                f"at {float(score):.3f}, below the {SCORE_FLOOR:.2f} floor. "
                "A correct placement reads about 0.4 and falls to near zero "
                "within five centimetres, so this is off the peak.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        clash = [p for p in (out / f"{stem}.tif",
                             out / f"{stem}_individual_clasts.csv",
                             out / f"{stem}.georef.json") if p.exists()]
        if clash:
            import datetime as _dt
            first = clash[0]
            was = ""
            side = out / f"{stem}.georef.json"
            if side.exists():
                try:
                    was = json.loads(side.read_text(encoding="utf-8")).get(
                        "provenance", "")
                except Exception:
                    was = ""
            when = _dt.datetime.fromtimestamp(
                first.stat().st_mtime).isoformat(timespec="seconds")
            raise FileExistsError(
                f"{first.name} already exists"
                + (f", placed by {was}" if was else "")
                + f", written {when}. This directory holds the reference "
                "placements this method is validated against. Pass "
                "overwrite=True only if you mean to replace it.")
    written: dict = {}

    # --- GeoTIFF: the GeoTransform is this affine in GDAL's order ---
    try:
        from osgeo import gdal, osr
        arr = np.asarray(quadrat_image)
        if arr.ndim == 3:
            arr = arr[..., :3]
            bands = arr.shape[2]
        else:
            bands = 1
        h, w = arr.shape[0], arr.shape[1]
        tif = out / f"{stem}.tif"
        drv = gdal.GetDriverByName("GTiff")
        dtype = gdal.GDT_Float32 if arr.dtype.kind == "f" else gdal.GDT_Byte
        ds = drv.Create(str(tif), w, h, bands, dtype)
        ds.SetGeoTransform([m[0, 2], m[0, 0], m[0, 1],
                            m[1, 2], m[1, 0], m[1, 1]])
        if crs_wkt:
            srs = osr.SpatialReference()
            if srs.SetFromUserInput(str(crs_wkt)) == 0:
                ds.SetProjection(srs.ExportToWkt())
        # Provenance travels with the raster, which is what the Validate tab reads.
        meta = {"PM_PLACEMENT": provenance}
        if hand and hand.get("agreement_score") is not None:
            meta["PM_AGREEMENT_SCORE"] = f"{float(hand['agreement_score']):.4f}"
        ds.SetMetadata(meta)
        for b in range(bands):
            band_arr = arr if bands == 1 else arr[..., b]
            ds.GetRasterBand(b + 1).WriteArray(np.asarray(band_arr))
        ds.FlushCache()
        ds = None
        written["geotiff"] = tif
    except Exception as ex:                       # GDAL missing or read-only
        written["geotiff_error"] = f"{type(ex).__name__}: {ex}"

    # --- clast CSV in world coordinates, same transform as the GeoTIFF ---
    if clasts is not None:
        df = clasts.copy()
        if x_col in df.columns and y_col in df.columns:
            xs, ys = world_from_pixels(m, df[x_col].to_numpy(),
                                       df[y_col].to_numpy())
            df[x_col] = xs
            df[y_col] = ys
            df["placement"] = provenance
            csv = out / f"{stem}_individual_clasts.csv"
            df.to_csv(csv, index=False)
            written["csv"] = csv
        else:
            written["csv_error"] = (
                f"clast table has no '{x_col}'/'{y_col}' columns")

    # --- audit sidecar ---
    side = out / f"{stem}.georef.json"
    payload = {
        "transform": [list(map(float, row)) for row in m],
        "provenance": provenance,
        "quality": (match.quality.as_dict() if match is not None else None),
        "seed_xy": (list(match.seed_xy)
                    if match is not None and match.seed_xy else None),
        "quadrat_gsd_m": (match.quadrat_gsd_m if match is not None else None),
        "ortho_gsd_m": (match.ortho_gsd_m if match is not None else None),
        "crs": str(crs_wkt or ""),
        "sources": {k: str(v) for k, v in (source_files or {}).items()},
        "ransac_seed": DEFAULTS.get("ransac_seed"),
    }
    if hand:
        # The quality block describes the FIT, not the placement written.
        payload["quality_describes"] = "the fitted placement, not this one"
        payload["fitted_transform"] = (
            [list(map(float, r)) for r in np.asarray(match.matrix, float)]
            if match is not None and match.matrix is not None else None)
        for k in ("delta", "agreement_score", "operator", "timestamp", "view"):
            if hand.get(k) is not None:
                payload[k] = hand[k]
        if payload["fitted_transform"] is None:
            # No fit, so no number rather than a zero that reads as "measured".
            payload["quality"] = None
            payload["quality_describes"] = "no fit was made; placed by hand"
    side.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written["sidecar"] = side
    return written


def _detect_and_match(img_a, img_b, n_kp_a, n_kp_b, settings=None):
    """Matched keypoints between two images, as (col, row) arrays.

    OpenCV does the work: scikit-image's ``match_descriptors`` is a pure-NumPy
    O(n*m) loop that takes minutes on a realistic window where cv2's brute
    force takes milliseconds (FLANN measured slower at this scale); scikit-
    image remains the fallback. SIFT + Lowe's ratio finds far more correct
    correspondences than ORB + crossCheck and, with fewer spurious ones
    handed to RANSAC, is faster overall. ``cv2.SIFT_create`` is in the plain
    opencv-python wheel.

    Returns ``(src, dst)`` or ``(None, None)`` when either side describes
    nothing, an ordinary outcome for a flat or tiny tile.
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    descriptor = str(cfg.get("descriptor", "sift")).lower()
    ratio = float(cfg.get("match_ratio", 0.75) or 0.0)
    try:
        import cv2 as _cv

        def _u8(x):
            x = np.asarray(x, dtype=np.float64)
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.zeros(np.shape(x), dtype=np.uint8)
            return (255.0 * (x - lo) / (hi - lo)).astype(np.uint8)

        ga, gb = _u8(img_a), _u8(img_b)
        if min(ga.shape[:2]) < 16 or min(gb.shape[:2]) < 16:
            return None, None
        if descriptor == "sift" and hasattr(_cv, "SIFT_create"):
            def _mk(n):
                return _cv.SIFT_create(nfeatures=int(n))
            norm = _cv.NORM_L2
        else:
            def _mk(n):
                return _cv.ORB_create(nfeatures=int(n))
            norm = _cv.NORM_HAMMING
        ka, da = _mk(n_kp_a).detectAndCompute(ga, None)
        kb, db = _mk(n_kp_b).detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return None, None
        if ratio > 0:
            knn = _cv.BFMatcher(norm, crossCheck=False).knnMatch(da, db, k=2)
            pairs = [(p[0].queryIdx, p[0].trainIdx) for p in knn
                     if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        else:
            raw = _cv.BFMatcher(norm, crossCheck=True).match(da, db)
            pairs = [(m.queryIdx, m.trainIdx) for m in raw]
        if not pairs:
            return None, None
        src = np.array([ka[i].pt for i, _ in pairs], dtype=float)
        dst = np.array([kb[j].pt for _, j in pairs], dtype=float)
        return src, dst
    except Exception:
        pass

    from skimage.feature import ORB as _ORB, match_descriptors as _md
    oa, ob = _ORB(n_keypoints=int(n_kp_a)), _ORB(n_keypoints=int(n_kp_b))
    try:
        oa.detect_and_extract(np.asarray(img_a))
        ob.detect_and_extract(np.asarray(img_b))
        da = oa.descriptors
        db = ob.descriptors
        if da is None or db is None or len(da) == 0 or len(db) == 0:
            return None, None
        idx = _md(da, db, cross_check=True)
    except Exception:
        return None, None
    if idx is None or len(idx) == 0:
        return None, None
    return (oa.keypoints[idx[:, 0]][:, ::-1],
            ob.keypoints[idx[:, 1]][:, ::-1])


def _tiles_over(region, size, step, centre):
    """Windows of at most ``size`` covering ``region``, nearest ``centre``
    first, so a seeded search still stops early when the seed is good."""
    r0, c0, r1, c1 = region
    if (r1 - r0) <= size and (c1 - c0) <= size:
        return [region]
    out = []
    rr = r0
    while True:
        cc = c0
        while True:
            out.append((rr, cc, min(r1, rr + size), min(c1, cc + size)))
            if cc + size >= c1:
                break
            cc += step
        if rr + size >= r1:
            break
        rr += step
    cy, cx = centre
    out.sort(key=lambda w: ((w[0] + w[2]) / 2 - cy) ** 2
             + ((w[1] + w[3]) / 2 - cx) ** 2)
    return out


# Largest window worth handing to the matcher, per side. Keypoint density has
# to stay high to match at all, so past this the answer is more windows, not
# a bigger one; measured, larger tiles lose up to half the placements.
MAX_WINDOW_PX = 900


def tile_grid(ortho_shape, ortho_gsd_m, quadrat_extent_m, *, tile_px=None):
    """Tiles on one grid fixed to the ortho, overlapping by a quadrat DIAGONAL.

    Unlike :func:`plan_search`'s per-seed windows, a fixed grid lets nearby
    quadrats share tile reads across a survey. The overlap is the diagonal,
    not the extent, because a rotated square of side s spans s*sqrt(2) and
    would otherwise be cut in two tiles.

    Yields ``(row0, col0, row1, col1)``. Reads nothing.
    """
    rows, cols = int(ortho_shape[0]), int(ortho_shape[1])
    gsd = float(ortho_gsd_m)
    size = int(tile_px or MAX_WINDOW_PX)
    diag_px = max(8, int(math.ceil(float(quadrat_extent_m) * math.sqrt(2.0) / gsd)))
    step = max(1, size - diag_px)
    for r in range(0, max(1, rows), step):
        for c in range(0, max(1, cols), step):
            yield (r, c, min(rows, r + size), min(cols, c + size))
            if c + size >= cols:
                break
        if r + size >= rows:
            break


def tiles_for_seed(ortho_shape, ortho_gsd_m, quadrat_extent_m, seed_px,
                   search_radius_m, *, tile_px=None, max_radius_m=None):
    """Grid tiles that could hold a quadrat seeded at ``seed_px``, nearest
    first, so a good seed still stops at the first tile while nearby quadrats
    share the same rectangles."""
    gsd = float(ortho_gsd_m)
    rr, cc = int(seed_px[0]), int(seed_px[1])
    reach = float(search_radius_m) + float(quadrat_extent_m)
    if max_radius_m:
        reach = min(reach, float(max_radius_m) + float(quadrat_extent_m))
    reach_px = max(1, int(math.ceil(reach / gsd)))
    out = []
    for t in tile_grid(ortho_shape, gsd, quadrat_extent_m, tile_px=tile_px):
        r0, c0, r1, c1 = t
        # nearest point of the tile to the seed
        dr = max(r0 - rr, 0, rr - r1)
        dc = max(c0 - cc, 0, cc - c1)
        if math.hypot(dr, dc) > reach_px:
            continue
        out.append(t)
    out.sort(key=lambda t: ((t[0] + t[2]) / 2.0 - rr) ** 2
             + ((t[1] + t[3]) / 2.0 - cc) ** 2)
    return out


def plan_search(ortho_shape, ortho_gsd_m, quadrat_extent_m, *,
                seed_px=None, search_radius_m=None, max_radius_m=None):
    """Windows to try, nearest-first: widening rings around a seed, or
    overlapping tiles over the whole ortho (overlap at least the quadrat's
    extent so a straddling quadrat is whole in one tile). Yields
    ``(row0, col0, row1, col1)`` in pixels; reads nothing."""
    rows, cols = int(ortho_shape[0]), int(ortho_shape[1])
    gsd = float(ortho_gsd_m)
    quad_px = max(8, int(round(float(quadrat_extent_m) / gsd)))

    def _clip(r0, c0, r1, c1):
        return (max(0, r0), max(0, c0), min(rows, r1), min(cols, c1))

    if seed_px is not None:
        rr, cc = int(seed_px[0]), int(seed_px[1])
        r_m = float(search_radius_m if search_radius_m is not None
                    else DEFAULTS["search_radius_m"])
        cap = float(max_radius_m) if max_radius_m else None
        seen = set()
        # Windows never exceed the cap (keypoint density); a larger radius is
        # tiled across the region, nearest the seed first, never skipped.
        step = max(1, MAX_WINDOW_PX - quad_px)
        while True:
            half = quad_px + int(round(r_m / gsd))
            region = _clip(rr - half, cc - half, rr + half, cc + half)
            for win in _tiles_over(region, MAX_WINDOW_PX, step, (rr, cc)):
                if win not in seen:
                    seen.add(win)
                    yield win
            covers_all = (region[0] == 0 and region[1] == 0
                          and region[2] == rows and region[3] == cols)
            if covers_all or (cap is not None and r_m >= cap):
                return
            r_m *= 2.0                      # widen and try again
        return

    # No seed: tile, with overlap >= the quadrat's own extent.
    size = min(2 * quad_px, MAX_WINDOW_PX)
    step = max(1, min(quad_px, size // 2))  # overlap >= the quadrat extent
    for r0 in range(0, max(1, rows - 1), step):
        for c0 in range(0, max(1, cols - 1), step):
            yield _clip(r0, c0, r0 + size, c0 + size)


def locate_quadrat(
    quadrat_gray: np.ndarray,
    read_window,
    ortho_shape,
    *,
    quadrat_gsd_m: float,
    ortho_gsd_m: float,
    seed_xy: Optional[Tuple[float, float]] = None,
    ortho_origin_xy: Tuple[float, float] = (0.0, 0.0),
    search_radius_m: Optional[float] = None,
    max_radius_m: Optional[float] = None,
    frame_inset_m: Optional[float] = None,
    settings: Optional[dict] = None,
    log_fn=None,
):
    """Find the quadrat, searching outward from a seed or over the whole
    ortho. ``read_window(r0, c0, r1, c1)`` returns that window's pixels, so a
    survey is never loaded whole. Two distant placements that both pass the
    gates are reported as ambiguity rather than one being picked."""
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    gsd = float(ortho_gsd_m)
    quad_extent_m = max(quadrat_gray.shape[:2]) * float(quadrat_gsd_m)
    ox, oy = float(ortho_origin_xy[0]), float(ortho_origin_xy[1])

    seed_px = None
    if seed_xy is not None:
        seed_px = (int(round((oy - float(seed_xy[1])) / gsd)),
                   int(round((float(seed_xy[0]) - ox) / gsd)))

    accepted: list = []
    last: Optional[QuadratMatch] = None
    for (r0, c0, r1, c1) in plan_search(
            ortho_shape, gsd, quad_extent_m, seed_px=seed_px,
            search_radius_m=search_radius_m, max_radius_m=max_radius_m):
        if r1 <= r0 or c1 <= c0:
            continue
        try:
            win = read_window(r0, c0, r1, c1)
        except Exception as ex:
            if log_fn:
                log_fn(f"could not read window {(r0, c0, r1, c1)}: {ex}")
            continue
        if win is None or np.asarray(win).size == 0:
            continue
        m = match_quadrat(
            quadrat_gray, win, quadrat_gsd_m=quadrat_gsd_m,
            ortho_gsd_m=gsd,
            window_origin_xy=(ox + c0 * gsd, oy - r0 * gsd),
            seed_xy=seed_xy if seed_xy is not None else None,
            search_radius_m=(search_radius_m if seed_xy is not None else None),
            frame_inset_m=frame_inset_m, settings=cfg)
        last = m
        if m.accepted:
            accepted.append(m)
            if seed_xy is not None:
                # Seeded: the first ring that works is the answer.
                return m
    if not accepted:
        return last if last is not None else QuadratMatch(
            None, MatchQuality(reasons=["No window produced a usable match."]))

    # Prefer the placement nearest the seed over the one with most inliers: a
    # richly textured patch elsewhere can out-score the true placement, and
    # the seed is independent evidence.
    if seed_xy is not None:
        def _near(m):
            cx, cy = _centre_world(m.matrix, quadrat_gray.shape)
            return math.hypot(cx - float(seed_xy[0]), cy - float(seed_xy[1]))
        best = min(accepted, key=_near)
    else:
        best = max(accepted, key=lambda m: m.quality.n_inliers)
    bx, by = _centre_world(best.matrix, quadrat_gray.shape)
    rivals = []
    rival_matches = []
    for m in accepted:
        if m is best:
            continue
        cx, cy = _centre_world(m.matrix, quadrat_gray.shape)
        if math.hypot(cx - bx, cy - by) > quad_extent_m:
            rivals.append(m.quality.n_inliers)
            rival_matches.append(m)
    if rivals and max(rivals) >= float(cfg["ambiguity_ratio"]) * best.quality.n_inliers:
        q = best.quality
        q.accepted = False
        q.rival_inliers = int(max(rivals))
        q.reasons.append(
            f"The quadrat matched {len(accepted)} separate places in this "
            f"ortho about equally well (best {q.n_inliers} correspondences, "
            f"runner-up {max(rivals)} somewhere else). Give a seed position "
            "to say which one you mean, or look at the candidates and pick "
            "one.")
        # Hand back the candidates: a person looking at them on the ortho can
        # usually tell which is right.
        cands = []
        for m in [best] + rival_matches:
            cx, cy = _centre_world(m.matrix, quadrat_gray.shape)
            cands.append({
                "matrix": [list(map(float, row)) for row in m.matrix],
                "n_inliers": int(m.quality.n_inliers),
                "residual_m": float(m.quality.residual_m),
                "centre": (float(cx), float(cy)),
                "from_seed_m": (
                    float(math.hypot(cx - float(seed_xy[0]),
                                     cy - float(seed_xy[1])))
                    if seed_xy is not None else float("nan")),
            })
        cands.sort(key=lambda c: -c["n_inliers"])
        q.candidates = cands
        return QuadratMatch(None, q, seed_xy, quadrat_gsd_m, gsd)
    return best


def describe_georeferencing(image_path) -> dict:
    """Does this image already carry a georeference? Returns
    ``{georeferenced, crs, gsd_m, reason}``; never raises."""
    out = {"georeferenced": False, "crs": "", "gsd_m": None, "reason": ""}
    try:
        from osgeo import gdal
    except Exception:
        out["reason"] = "GDAL is unavailable, so this cannot be determined."
        return out
    p = str(image_path or "")
    if not p:
        out["reason"] = "No image selected."
        return out
    # GDAL has no HEIF driver here and prints "ERROR 4 ... not recognized"
    # on every probe; an iPhone photograph carries no geotransform anyway.
    from functions import images as _images
    if _images.is_heif(p):
        out["reason"] = ("This image carries no georeference, so it must be "
                         "matched to the ortho to be placed.")
        return out
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        ds = gdal.Open(p)
    except Exception:
        ds = None
    finally:
        gdal.PopErrorHandler()
    if ds is None:
        out["reason"] = "The image could not be opened by GDAL."
        return out
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection() or ""
    ds = None
    # GDAL returns the identity transform for a plain image.
    identity = (not gt) or tuple(gt) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if identity and not proj:
        out["reason"] = ("This image carries no georeference, so it must be "
                         "matched to the ortho to be placed.")
        return out
    if identity and proj:
        out["crs"] = proj
        out["reason"] = ("This image names a CRS but has no geotransform, so "
                         "it still needs to be placed.")
        return out
    out["georeferenced"] = True
    out["crs"] = proj
    try:
        # The column step's length: a placed quadrat is a rotated raster,
        # whose x-scale term alone understates the pixel.
        out["gsd_m"] = float(math.hypot(float(gt[1]), float(gt[2])))
    except Exception:
        out["gsd_m"] = None
    out["reason"] = ("This image is already georeferenced, so there is "
                     "nothing to match — digitised clasts can be written "
                     "straight out in world coordinates.")
    return out


DEFAULT_SEED_CRS = "EPSG:4326"     # what a field GPS gives: WGS84 lat/lon


def ortho_crs(ortho_path) -> str:
    """The ortho's CRS as WKT, or '' when it declares none."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(ortho_path))
        if ds is None:
            return ""
        wkt = ds.GetProjection() or ""
        ds = None
        return wkt
    except Exception:
        return ""


def transform_seed(x, y, source_crs=DEFAULT_SEED_CRS, target_crs=""):
    """Convert a seed coordinate into the ortho's CRS.

    ``x``/``y`` are in traditional GIS order (lon/lat or easting/northing).
    That order must be forced: GDAL 3 honours the authority's axis order, so
    EPSG:4326 would otherwise come back latitude-first and silently transpose
    every seed.

    Returns ``(x, y, note)``; ``note`` is empty when the conversion is exact.
    The coordinate is returned unchanged when no conversion can be made.
    """
    xf, yf = float(x), float(y)
    if not target_crs:
        return xf, yf, ("The ortho declares no CRS, so the seed is used as "
                        "given. If it is latitude/longitude the search will "
                        "look in the wrong place.")
    try:
        from osgeo import osr
    except Exception:
        return xf, yf, "GDAL is unavailable, so the seed was not converted."

    src = osr.SpatialReference()
    dst = osr.SpatialReference()
    if src.SetFromUserInput(str(source_crs or DEFAULT_SEED_CRS)) != 0:
        return xf, yf, f"Unrecognised seed CRS {source_crs!r}; used as given."
    if dst.SetFromUserInput(str(target_crs)) != 0:
        return xf, yf, "The ortho's CRS could not be read; seed used as given."
    if src.IsSame(dst):
        return xf, yf, ""
    try:
        # Traditional order on BOTH sides, so x is always east/longitude.
        for srs in (src, dst):
            try:
                srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            except Exception:
                pass
        ct = osr.CoordinateTransformation(src, dst)
        ox, oy, *_ = ct.TransformPoint(xf, yf)
    except Exception as ex:
        return xf, yf, (f"Could not convert the seed from {source_crs} to the "
                        f"ortho's CRS ({type(ex).__name__}); used as given.")
    if not (math.isfinite(ox) and math.isfinite(oy)):
        return xf, yf, (f"Converting the seed from {source_crs} produced no "
                        "finite coordinate; check it is the right CRS.")
    name = dst.GetAttrValue("AUTHORITY", 1) or dst.GetName() or "the ortho's CRS"
    return ox, oy, f"Seed converted from {source_crs} to EPSG:{name}."


def survey_overlay_figure(placements, ortho_path, *, max_px=1600,
                          quadrat_shapes=None):
    """Every placement drawn on the ortho, the cheapest review of a survey.
    ``placements`` is ``{photo_name: matrix}``. Returns a matplotlib Figure."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from osgeo import gdal
    gdal.UseExceptions()

    ds = gdal.Open(str(ortho_path))
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    k = max(1, int(math.ceil(max(W, H) / float(max_px))))
    arr = ds.ReadAsArray(buf_xsize=max(1, W // k), buf_ysize=max(1, H // k))
    ds = None
    arr = np.moveaxis(arr, 0, -1) if getattr(arr, "ndim", 2) == 3 else arr
    rgb = (arr[:, :, :3] if getattr(arr, "ndim", 2) == 3
           else np.dstack([arr] * 3)).astype(np.float32)
    lit = rgb[rgb > 0]
    lo, hi = (np.percentile(lit, 2), np.percentile(lit, 98)) if lit.size \
        else (0.0, 1.0)
    rgb = np.clip((rgb - lo) / max(1e-6, hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=120)
    ax.imshow(rgb, extent=[gt[0], gt[0] + W * gt[1],
                           gt[3] + H * gt[5], gt[3]], origin="upper")
    n = 0
    for name, matrix in (placements or {}).items():
        if matrix is None:
            continue
        M = np.asarray(matrix, dtype=float)
        shape = (quadrat_shapes or {}).get(name, (1000, 1000))
        rows, cols = int(shape[0]), int(shape[1])
        corners = [(0, 0), (cols, 0), (cols, rows), (0, rows)]
        world = [(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                  M[1, 0] * x + M[1, 1] * y + M[1, 2]) for x, y in corners]
        ax.add_patch(Polygon(world, closed=True, fill=False,
                             edgecolor="#ffd000", linewidth=1.8))
        cx = sum(p[0] for p in world) / 4.0
        cy = sum(p[1] for p in world) / 4.0
        ax.annotate(str(name).split("_rect")[0].split("_corrected")[0],
                    (cx, cy), fontsize=6, color="white",
                    ha="center", va="center")
        n += 1
    ax.set_title(f"{n} placement(s) on {Path(str(ortho_path)).name}",
                 fontsize=11)
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# --- Which quadrats have already been placed ---
_PLACED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def georeferenced_listing(*dirs):
    """Every georeferenced raster under these folders, one level deep (an
    archive subfolder still counts; a full tree walk does not pay). Several
    folders because outputs may sit beside the photos or under the project's
    ``validation/georectified``. Sidecars are ignored."""
    out = []
    seen = set()
    for d in dirs:
        try:
            folder = Path(str(d or ""))
            if not folder.is_dir():
                continue
            for entry in sorted(folder.iterdir()):
                if entry.is_file():
                    if entry.suffix.lower() in _PLACED_SUFFIXES:
                        out.append(entry)
                elif entry.is_dir():
                    for sub in sorted(entry.iterdir()):
                        if (sub.is_file()
                                and sub.suffix.lower() in _PLACED_SUFFIXES):
                            out.append(sub)
        except OSError:
            continue
    # A .png may sit beside every .tif; one placement, one row.
    deduped = []
    for p in out:
        if p.stem in seen:
            continue
        seen.add(p.stem)
        deduped.append(p)
    return deduped


def georeferenced_for(photo, listing):
    """``(path, exists)`` for one photograph, matched on the base name that
    :func:`functions.seeds._base_photo` extracts, since only that survives
    the rectified and georeferenced naming (and a prefix match would confuse
    ``S1_`` with ``S10_``)."""
    from functions.seeds import _base_photo
    want = _base_photo(photo)
    if not want:
        return None, False
    for cand in listing or ():
        if _base_photo(cand.name) == want:
            return cand, True
    return None, False


def find_clasts_csv(photo_path, vectors_dir=None, project=None, date=None):
    """The detection CSV of a photograph, to be rewritten in world
    coordinates beside its placement: ``<stem>_individual_clasts.csv``
    beside the photograph (the layout before projects), else
    ``<vectors_dir>/<origin stem>_individual_clasts.csv`` where Detect
    writes it in a project. None when neither exists. Only the first
    location was searched, so no placement ever carried its CSV in the
    canonical layout."""
    photo = Path(str(photo_path or ""))
    if not photo.name:
        return None
    stem = photo.stem
    candidates = [photo.with_name(f"{stem}_individual_clasts.csv")]
    if vectors_dir:
        try:
            from functions.naming import origin_stem
            origin = origin_stem(stem, project or None, date or None)
        except Exception:
            origin = stem
        candidates.append(Path(str(vectors_dir)) / f"{origin}_individual_clasts.csv")
        if origin != stem:
            candidates.append(Path(str(vectors_dir)) / f"{stem}_individual_clasts.csv")
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def describe_placement(image_path, *search_dirs) -> dict:
    """Is this photograph placed in the ortho — either because the file
    itself carries a georeference, or because a placement of it was saved
    (``<stem>.tif`` under one of ``search_dirs``, the project's
    ``validation/georectified`` and the folder beside the photographs)?
    Returns :func:`describe_georeferencing`'s dict plus ``placed_path``
    (the saved raster, or None). Never raises. Digitize asks this before
    the drawing tools: a rectified JPEG never carries a georeference itself,
    so without the saved placement it kept sending the user back to the
    Georeference tab."""
    out = dict(describe_georeferencing(image_path))
    out["placed_path"] = None
    if out["georeferenced"] or not image_path:
        return out
    try:
        listing = georeferenced_listing(*[d for d in search_dirs if d])
        hit, ok = georeferenced_for(Path(str(image_path)).name, listing)
    except Exception:
        return out
    if not ok:
        return out
    saved = describe_georeferencing(str(hit))
    if not saved.get("georeferenced"):
        return out
    out.update({"georeferenced": True, "crs": saved.get("crs", ""),
                "gsd_m": saved.get("gsd_m"), "placed_path": hit,
                "reason": (f"Placed in the ortho already: its placement "
                           f"is saved as {hit.parent.name}/{hit.name}.")})
    return out


def frame_thickness_for(record, current, auto_last):
    """Which frame thickness to show, and where it came from. Returns
    ``(value, note)``; ``value`` is ``None`` to leave the box as it is.

    A value the user typed always wins; a value in the record fills an
    untouched box; no value in the record leaves the box alone (it means
    "nobody said", not zero). "Untouched" is decided by comparing values,
    because a bound widget fires the same change event for a programmatic
    write as for a keystroke.
    """
    try:
        cur = float(current or 0.0)
    except (TypeError, ValueError):
        cur = 0.0
    touched = not (abs(cur) < 1e-9
                   or (auto_last is not None
                       and abs(cur - float(auto_last)) < 1e-9))
    if touched:
        return None, f"Frame thickness {cur:.3f} m \u2014 yours, kept."

    raw = (record or {}).get("PM_FRAME_THICKNESS_M")
    if raw in (None, ""):
        return None, ""
    try:
        thick = float(raw)
    except (TypeError, ValueError):
        return None, ""
    if not (thick >= 0.0):
        return None, ""
    return thick, (f"Frame thickness {thick:.3f} m, read from this "
                   "photograph's rectification record.")
