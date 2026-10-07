"""Quadrat photograph orthorectification.

Given a photograph containing a quadrat (rectangle) of known dimensions plus
the pixel coordinates of its 4 corners and the lengths of the 4 segments
between them, computes the homography that maps the quadrat to a rectified
rectangle of known ground sample distance (GSD) and returns the cropped,
rectified image. Pure functions, testable without the GUI.

Inputs:
    image           : numpy uint8 array (H, W, 3), the source photograph
    corners_px      : 4 (x, y) tuples in image pixel coords, in order
    seg_lengths_m   : 4 floats, the metric length of segments
                      (corner_0 -> corner_1, 1 -> 2, 2 -> 3, 3 -> 0);
                      for a rectangle these come in pairs (s1, s2, s1, s2)
    gsd_m           : desired ground sample distance in metres/pixel;
                      None or 'auto' picks it from the source pixel density

Returns:
    rectified       : numpy uint8 array, shape (out_h, out_w, 3)
    gsd_actual      : float, the GSD actually used (m/pixel)

Raises ValueError if the inputs are malformed.
"""
from dataclasses import dataclass, field as _dc_field
import re
from pathlib import Path
from typing import Optional

import numpy as np

from functions import images as _images


def compute_homography_target(corners_px, seg_lengths_m):
    """Compute the target rectangle dimensions and source/target point arrays.

    Returns:
        src_pts: (4, 2) float32 array of source pixel coords
        dst_pts: (4, 2) float32 array of target meter coords (0,0) to (W, H)
        target_w_m, target_h_m: target rectangle dimensions in meters
    """
    if len(corners_px) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners_px)}")
    if len(seg_lengths_m) != 4:
        raise ValueError(f"Expected 4 segment lengths, got {len(seg_lengths_m)}")
    for i, L in enumerate(seg_lengths_m):
        if not (L > 0):
            raise ValueError(f"Segment {i} length must be > 0, got {L}")

    # Opposite sides are averaged for robustness against small measurement errors.
    s1 = (seg_lengths_m[0] + seg_lengths_m[2]) / 2.0   # "horizontal" sides
    s2 = (seg_lengths_m[1] + seg_lengths_m[3]) / 2.0   # "vertical" sides

    src_pts = np.asarray(corners_px, dtype=np.float32)
    dst_pts = np.asarray([
        [0.0,  0.0],
        [s1,   0.0],
        [s1,   s2],
        [0.0,  s2],
    ], dtype=np.float32)
    return src_pts, dst_pts, s1, s2


def auto_gsd(corners_px, target_w_m, target_h_m):
    """Pick a GSD (m/pixel) so the finest source axis keeps its pixel count in
    the rectified output (no aliasing, no over-stretching)."""
    src_pts = np.asarray(corners_px, dtype=np.float32)
    px_lens = []
    for i in range(4):
        a = src_pts[i]
        b = src_pts[(i + 1) % 4]
        px_lens.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    src_h = (px_lens[0] + px_lens[2]) / 2.0
    src_v = (px_lens[1] + px_lens[3]) / 2.0
    gsd_h = target_w_m / max(src_h, 1.0)   # m / px in the horizontal direction
    gsd_v = target_h_m / max(src_v, 1.0)   # m / px in the vertical direction
    return min(gsd_h, gsd_v)


def orthorectify(image, corners_px, seg_lengths_m, gsd_m=None,
                 camera_matrix=None, dist_coeffs=None):
    """Apply the homography and return the rectified, cropped image.

    Lazy-imports cv2 so the module imports without OpenCV installed (only
    homography-math helpers are usable then).

    ``camera_matrix`` (3×3) + ``dist_coeffs`` (k1,k2,p1,p2[,k3…]) are optional.
    When both are given, the image is undistorted and the picked corner points
    are mapped to the undistorted frame *before* the homography, so lens
    distortion is removed as part of the crop + rectify.
    """
    import cv2

    if camera_matrix is not None and dist_coeffs is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        image = cv2.undistort(image, K, D)
        _src = np.asarray(corners_px, dtype=np.float32).reshape(-1, 1, 2)
        _und = cv2.undistortPoints(_src, K, D, P=K).reshape(-1, 2)
        corners_px = [(float(px), float(py)) for px, py in _und]

    src_pts, dst_pts, target_w_m, target_h_m = compute_homography_target(
        corners_px, seg_lengths_m)

    if gsd_m is None or gsd_m == 'auto':
        gsd_actual = auto_gsd(corners_px, target_w_m, target_h_m)
    else:
        gsd_actual = float(gsd_m)
    if gsd_actual <= 0:
        raise ValueError(f"Computed GSD must be > 0, got {gsd_actual}")

    out_w = max(1, int(round(target_w_m / gsd_actual)))
    out_h = max(1, int(round(target_h_m / gsd_actual)))

    dst_pts_px = dst_pts / gsd_actual
    H = cv2.getPerspectiveTransform(src_pts, dst_pts_px)
    rectified = cv2.warpPerspective(
        image, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rectified, gsd_actual


def _coerce_matrix(v):
    """Coerce a camera-matrix value (list-of-lists, flat 9-list, or an
    OpenCV {'data':[...],'rows':r,'cols':c} dict) into a 3×3 array, or None."""
    if v is None:
        return None
    if isinstance(v, dict) and "data" in v:
        v = v["data"]
    a = np.asarray(v, dtype=float).reshape(-1)
    return a[:9].reshape(3, 3) if a.size >= 9 else None


def _coerce_vector(v):
    """Coerce a distortion-coeffs value into a flat float array, or None."""
    if v is None:
        return None
    if isinstance(v, dict) and "data" in v:
        v = v["data"]
    a = np.asarray(v, dtype=float).reshape(-1)
    return a if a.size >= 4 else None


def parse_calibration_file(path):
    """Parse camera intrinsics + distortion from a calibration file.

    Supports OpenCV-style **JSON**, **YAML** (.yml/.yaml via cv2.FileStorage),
    and NumPy **.npz**. Recognises the common key spellings for the camera
    matrix (camera_matrix / cameraMatrix / mtx / K / M) and the distortion
    coefficients (dist / dist_coeff(s) / distortion_coefficients / D).

    Returns ``(fx, fy, cx, cy, [dist coeffs])`` or ``None`` on failure.
    """
    from pathlib import Path as _P
    p = _P(path)
    suf = p.suffix.lower()
    _K_KEYS = ("camera_matrix", "cameraMatrix", "mtx", "K", "M")
    _D_KEYS = ("distortion_coefficients", "dist_coeffs", "dist_coeff",
               "dist", "D")
    try:
        K = D = None
        if suf == ".json":
            import json as _json
            data = _json.loads(p.read_text(encoding="utf-8"))
            K = _coerce_matrix(next((data[k] for k in _K_KEYS if k in data),
                                    None))
            D = _coerce_vector(next((data[k] for k in _D_KEYS if k in data),
                                    None))
        elif suf in (".yml", ".yaml"):
            import cv2
            fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
            for k in _K_KEYS:
                node = fs.getNode(k)
                if not node.empty():
                    K = node.mat()
                    break
            for k in _D_KEYS:
                node = fs.getNode(k)
                if not node.empty():
                    D = node.mat()
                    break
            fs.release()
            K = _coerce_matrix(K)
            D = _coerce_vector(D)
        elif suf == ".npz":
            npz = np.load(str(p))
            K = _coerce_matrix(next((npz[k] for k in _K_KEYS if k in npz),
                                    None))
            D = _coerce_vector(next((npz[k] for k in _D_KEYS if k in npz),
                                    None))
        if K is None or D is None:
            return None
        return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]),
                float(K[1, 2]), [float(v) for v in D])
    except Exception:
        return None


# --- Picking corners, and picking them again ---
# Kept here rather than in the tab's closure so the rule can be tested
# without a browser.

CORNER_HIT_PX = 24.0
"""How near a click counts as "that corner", in SOURCE image pixels: three
marker-widths (the marker is drawn at r=8 in the same space)."""


def corner_under(corners, x, y, radius=CORNER_HIT_PX):
    """Index of the placed corner under ``(x, y)``, or ``None``. Nearest wins."""
    best, best_d = None, float("inf")
    for i, c in enumerate(corners or ()):
        try:
            cx, cy = float(c[0]), float(c[1])
        except (TypeError, ValueError, IndexError):
            continue
        d = ((cx - float(x)) ** 2 + (cy - float(y)) ** 2) ** 0.5
        if d <= float(radius) and d < best_d:
            best, best_d = i, d
    return best


def place_corner(corners, idx, x, y):
    """Where a click goes when it did not land on an existing corner.

    Returns ``(corners, next_idx)``. Never returns more than four, whatever
    index a loader set.
    """
    out = [list(c) for c in (corners or [])]
    i = int(idx) % 4
    if i < len(out):
        out[i] = [x, y]
    elif len(out) < 4:
        while len(out) < i:
            out.append([0, 0])
        out.append([x, y])
    else:
        out[i] = [x, y]
    return out, (i + 1) % 4


DEFAULT_SEG_M = 1.0
"""The segment length a fresh form carries. Not a measurement of anything."""


def rectifying_at_the_untouched_default(seg_lengths_m, confirmed=False):
    """Is a quadrat about to be rectified at a length nobody has confirmed?

    A default is not a measurement, and no property of the output can reveal
    a wrong instruction: the output is only as true as the number typed in.
    """
    if confirmed:
        return False
    try:
        vals = [float(x) for x in seg_lengths_m]
    except (TypeError, ValueError):
        return False
    return bool(vals) and all(abs(v - DEFAULT_SEG_M) < 1e-9 for v in vals)


def declared_size_m(shape, gsd_m):
    """The physical size a rectified image asserts, from its own pixels.

    A rectified image plus its GSD is a claim about how big the thing in it
    was; a wrong segment length writes a wrong GSD into the filename and the
    matcher later refuses every placement.
    """
    side = max(int(shape[0]), int(shape[1]))
    return side * float(gsd_m)


def size_disagreement(shape, gsd_m, seg_lengths_m, tol=0.02):
    """``(claimed_m, asked_m, is_wrong)`` for a just-rectified image.

    ``tol`` is fractional; two percent is well inside the matcher's 15% scale
    gate, so this catches a mis-scale at creation rather than a survey later.
    """
    claimed = declared_size_m(shape, gsd_m)
    try:
        asked = max(float(x) for x in seg_lengths_m)
    except (TypeError, ValueError):
        return claimed, None, False
    if not asked > 0:
        return claimed, None, False
    return claimed, asked, abs(claimed - asked) / asked > float(tol)


# --- Naming a rectified output, once, for everyone ---
def gsd_tagged_name(out_path, gsd_m):
    """The name a rectified output actually ends up with.

    The measured GSD is written into the filename so it can be read back when
    populating Detection's resolution field. Both the writer and the "already
    rectified?" check must ask this one function, or the check cannot match.

    ``gsd_m`` of None or zero leaves the name alone (no measured GSD yet).
    An existing ``_GSD=<value>m`` tag in the stem is replaced by the fresh
    one: the Output field keeps the last saved name, and a second run at
    another size must not write a 2.06 mm/px image under a 2.45 mm/px name.
    """
    p = Path(str(out_path or ""))
    if "GSD=" in p.stem:
        stripped = re.sub(r"_GSD=[0-9.]+m$", "", p.stem)
        if stripped == p.stem:
            return p          # a foreign GSD= token: leave the name alone
        p = p.with_name(stripped + p.suffix)
    try:
        g = float(gsd_m)
    except (TypeError, ValueError):
        return p
    if not (g > 0.0):
        return p
    # Six decimals of a metre never rounds a real pixel size to zero; trailing
    # zeros are trimmed so the name stays readable.
    gsd_str = f"{g:.6f}".rstrip("0").rstrip(".")
    return p.with_name(f"{p.stem}_GSD={gsd_str}m{p.suffix}")


def output_listing(out_path):
    """List the folders a rectified output could be in, once per run rather
    than once per row (per-row listing is quadratic)."""
    base = Path(str(out_path or ""))
    names = []
    for folder in (base.parent, base.parent.with_name("rectified")):
        try:
            if not folder.is_dir():
                continue
            names.extend(c for c in folder.iterdir()
                         if c.is_file()
                         and c.suffix.lower() in _IMAGE_SUFFIXES)
        except OSError:
            continue
    return sorted(names)


def _match_in(listing, stem, tagged):
    source_stem = stem[:-len("_rectified")] if stem.endswith("_rectified") \
        else stem
    for cand in listing:
        if cand.stem == stem or cand.stem.startswith(stem + "_GSD="):
            return cand, True
        # The older convention kept the stem and appended its own tokens
        # (`01_rectified…`, `01_corrected_width=…`), so require one of those
        # tokens after the stem: a bare `IMG_0957_` prefix would claim
        # `IMG_0957_copy_rectified…`, another photograph's output, and the
        # folder run would then skip IMG_0957 as already rectified.
        if source_stem and (cand.stem.startswith(source_stem + "_rectified")
                            or cand.stem.startswith(source_stem + "_corrected")):
            return cand, True
    return tagged, False


def rectified_output_for(out_path, gsd_m=None, listing=None):
    """Whether a rectified output already exists, and where.

    Returns ``(path, exists)``. Callers that know the GSD get the tagged name;
    callers that do not fall back to scanning the folder, because the GSD is
    measured per photograph. Pass ``listing`` from :func:`output_listing` to
    scan once for a whole folder.
    """
    tagged = gsd_tagged_name(out_path, gsd_m)
    if tagged.exists():
        return tagged, True
    base = Path(str(out_path or ""))
    stem = base.stem
    if not stem:
        return tagged, False
    if listing is not None:
        return _match_in(listing, stem, tagged)
    # Also accept `<stem>.png` and the older `rectified/` folder: a survey
    # rectified under an earlier convention is still rectified.
    return _match_in(output_listing(out_path), stem, tagged)


# A folder of photographs: the camera formats (HEIC included), plus TIFF.
_IMAGE_SUFFIXES = _images.PHOTO_EXTENSIONS


def seg_confirmed_for(per_image, fname) -> bool:
    """Has the user accepted this file's segment lengths?

    Confirmation is a fact about one photograph, not about a session: a
    sentinel that switches off a safety check must not be reachable by the
    thing it guards.
    """
    try:
        entry = (per_image or {}).get(str(fname))
    except (AttributeError, TypeError):
        return False
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("seg_confirmed"))


def folder_row(name, corners, seg_lengths, out_path, gsd_m=None,
               listing=None):
    """The three facts about one photograph, before anything is opened: how
    much of the picking is done, what size it is being measured at, and
    whether it has already been rectified.

    The segment length is reported as a number, marked when it is still the
    untouched default, never as a "confirmed" tick: a row reading ``1.000 m``
    beside fifteen reading ``1.185 m`` explains itself.
    """
    path, exists = rectified_output_for(out_path, gsd_m, listing)
    try:
        seg1 = float(list(seg_lengths or [])[0])
    except (TypeError, ValueError, IndexError):
        seg1 = None
    if seg1 is None:
        length = "\u2014"
    elif abs(seg1 - DEFAULT_SEG_M) < 1e-9:
        length = f"{seg1:.3f} m \u26a0 default"
    else:
        length = f"{seg1:.3f} m"
    return {
        "photo": str(name),
        "corners": f"{len(corners or [])}/4",
        "length": length,
        "done": bool(exists),
        "output": Path(path).name if exists else "\u2014",
    }


# --- Seeing the corners, and where the last ones were ---
MARKER_SCREEN_PX = 9.0
"""How big a corner marker should look, in SCREEN pixels, at any zoom.

Fixed in image pixels it would shrink to a dot on a large photograph fitted
into the column.
"""


def display_scale(natural_px, display_px):
    """Image pixels per screen pixel (1.0 when nothing is being resized); the
    overlay draws in image coordinates."""
    try:
        n, d = float(natural_px), float(display_px)
    except (TypeError, ValueError):
        return 1.0
    if not (n > 0 and d > 0):
        return 1.0
    return n / d


def marker_radius(natural_px, display_px, screen_px=MARKER_SCREEN_PX):
    """Radius to draw a corner marker at, in image pixels."""
    return max(1.0, float(screen_px) * display_scale(natural_px, display_px))


def hit_radius(natural_px, display_px, screen_px=CORNER_HIT_PX):
    """How near a click counts as "that corner", in image pixels, scaled like
    the marker so the target stays the same size under the cursor."""
    return max(4.0, float(screen_px) * display_scale(natural_px, display_px))


def guide_quadrilateral(rows, idx):
    """The last confirmed quadrilateral before ``idx``, or ``None``.

    A guide, not a placement: it shows where the previous quadrat was and
    which physical corner is number 1 and which way the winding runs, the one
    decision a carried position genuinely helps with (getting it wrong puts
    segment 1's declared length on the wrong side of a non-square quadrat).

    ``rows`` is a list of ``(name, corners)`` in folder order, where ``corners``
    is a list of four ``[x, y]`` or empty. Filenames and numbers only; this
    must never need the pixels.
    """
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return None
    if i <= 0 or not rows:
        return None
    # Walk back to the most recent one that has four, so a skipped or
    # half-picked photograph does not break the chain.
    for k in range(min(i, len(rows)) - 1, -1, -1):
        try:
            name, corners = rows[k]
        except (TypeError, ValueError):
            continue
        if corners and len(corners) == 4:
            return name, [[float(a), float(b)] for a, b in corners]
    return None


# --- One rectification, without a browser ---
@dataclass
class RectifyOutcome:
    """What happened, as data rather than as log lines: the tab prints it and a test asserts it."""
    ok: bool
    out_path: Optional[str] = None
    gsd_m: float = 0.0
    shape: Optional[tuple] = None
    refusal: str = ""
    notes: list = _dc_field(default_factory=list)
    warnings: list = _dc_field(default_factory=list)
    record: dict = _dc_field(default_factory=dict)


def rectify_one(src_path, corners, seg_lengths_m, out_path, *,
                gsd_m=None, camera_matrix=None, dist_coeffs=None,
                frame_thickness_m=None, seg_confirmed=False,
                overwrite=True, tool_name="PebbleMapper"):
    """Rectify one photograph and write it, or refuse and write nothing.

    The guards travel with the write so no second caller can overwrite a
    photograph that cannot be taken again. A refusal returns ``ok=False``
    having written nothing. A wrong scale is a warning rather than a refusal,
    because the number came from the user; it is returned rather than logged
    so a batch can show it against the file it belongs to.
    """
    import cv2
    from functions import exif_seed as _xs

    out = RectifyOutcome(ok=False)
    src = Path(str(src_path or ""))
    if not src.is_file():
        out.refusal = f"{src.name or 'that photograph'} is not there."
        return out
    if not corners or len(corners) != 4:
        out.refusal = f"Pick all 4 corners first ({len(corners or [])}/4 done)."
        return out
    if not out_path:
        out.refusal = "No output path."
        return out
    try:
        segs = [float(v) for v in (seg_lengths_m or [])][:4]
    except (TypeError, ValueError):
        segs = []
    if len(segs) != 4 or min(segs) <= 0:
        out.refusal = "The segment lengths are not four positive numbers."
        return out

    # A default is not a measurement. Warned, not refused: a batch decides
    # whether to offer the file at all.
    if rectifying_at_the_untouched_default(segs, seg_confirmed):
        out.warnings.append(
            f"Rectifying at the default {DEFAULT_SEG_M:.3f} m segment length. "
            "If the quadrat frame is a different size, every output will be "
            "scaled wrong and no quadrat will place in an ortho.")

    # Pillow, in the stored pixel frame: the frame the canvas showed when the
    # corners were picked. (cv2.imdecode applies the EXIF orientation tag,
    # which the canvas deliberately does not, so it would rectify a rotated
    # JPEG against corners picked on the unrotated one.) A HEIC reads directly.
    try:
        img = _images.read_rgb(src)
    except Exception as ex:
        out.refusal = f"{src.name} could not be read ({ex})."
        return out

    rectified, gsd_used = orthorectify(
        img, [tuple(c) for c in corners], segs,
        gsd_m=(gsd_m if (gsd_m or 0) > 0 else None),
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    out.gsd_m = float(gsd_used)
    out.shape = tuple(rectified.shape)

    # The pixel count and the GSD together are a physical claim; check it
    # against what was asked for.
    claimed = max(rectified.shape[0], rectified.shape[1]) * float(gsd_used)
    asked = max(segs)
    out.notes.append(f"This output says it is {claimed:.3f} m across.")
    if asked > 0 and abs(claimed - asked) / asked > 0.02:
        out.warnings.append(
            f"The rectified image is {claimed:.3f} m across but the segment "
            f"lengths say {asked:.3f} m.")

    target = gsd_tagged_name(out_path, gsd_used)

    # The guards, before anything is written, against BOTH the name the user
    # gave and the name the GSD tag turns it into: checking only the tagged
    # name lets a derived image land in the folder of irreplaceable photographs.
    for candidate in (Path(str(out_path)), target):
        if _xs.same_file(str(candidate), str(src)):
            out.refusal = "the output path is the source photograph itself"
            return out
        if _xs.looks_like_raw_photograph(str(candidate)):
            out.refusal = (f"{candidate.name} is already a photograph "
                           "straight from a camera")
            return out
    if target.exists() and not overwrite:
        out.refusal = f"{target.name} already exists"
        return out

    target.parent.mkdir(parents=True, exist_ok=True)
    params = ([int(cv2.IMWRITE_JPEG_QUALITY), 100]
              if target.suffix.lower() in (".jpg", ".jpeg") else [])
    ok, buf = cv2.imencode(target.suffix,
                           cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR), params)
    if not ok:
        out.refusal = f"{target.suffix} could not be encoded."
        return out
    buf.tofile(str(target))
    # cv2.imwrite's return value is checked: a write that produced nothing
    # must not be reported as success.
    if not target.is_file() or target.stat().st_size <= 0:
        out.refusal = f"{target.name} was not written."
        return out
    out.out_path = str(target)
    out.notes.append(f"Wrote {target.name}")

    rec = _xs.rectification_record(
        source=src.name,
        gsd=f"{gsd_used:.6f}",
        size=f"{rectified.shape[1]}x{rectified.shape[0]}",
        corners=[(float(a), float(b)) for a, b in corners] or None,
        segments_m=",".join(f"{v:.4f}" for v in segs),
        frame_thickness_m=(f"{float(frame_thickness_m):.4f}"
                           if frame_thickness_m not in (None, "") else None),
        tool=tool_name)
    fix = _xs.gps_from_image(str(src))
    if fix:
        rec["lat"] = f"{fix['lat']:.8f}"
        rec["lon"] = f"{fix['lon']:.8f}"
    out.record = dict(rec)

    # A sidecar as well as the embedded block: the embedded one travels with
    # the file but is unreadable in most tools.
    try:
        import json as _json
        side = target.with_suffix(target.suffix + ".json")
        side.write_text(_json.dumps(rec, indent=1), encoding="utf-8")
        out.notes.append(f"Wrote {side.name}")
    except OSError as ex:
        out.warnings.append(f"Could not write the sidecar: {ex}")

    if _xs.write_placement_metadata(str(target), rec):
        out.notes.append(
            "Recorded the rectification in the output's metadata"
            + (f", carrying the source GPS fix ({fix['lat']:.6f}, "
               f"{fix['lon']:.6f})" if fix else " (no GPS fix in the source)"))
    else:
        # The image is intact (the metadata write never re-encodes pixels) but
        # the coordinate is not in it, so say so.
        out.warnings.append(
            "Could not attach metadata to this format. The image is written "
            "and unchanged, but it does not carry the source GPS fix.")

    out.ok = True
    return out


def is_ready_to_rectify(corners, seg_lengths_m, seg_confirmed=False):
    """Whether a photograph can be rectified without anybody guessing: four
    recorded corners, and a size somebody actually chose (a file still at the
    untouched default is not ready however many corners it has).

    Returns ``(ready, reason)``; ``reason`` is empty when ready.
    """
    n = len(corners or [])
    if n != 4:
        return False, f"{n}/4 corners"
    try:
        segs = [float(v) for v in (seg_lengths_m or [])][:4]
    except (TypeError, ValueError):
        segs = []
    if len(segs) != 4 or min(segs) <= 0:
        return False, "no segment length recorded"
    if rectifying_at_the_untouched_default(segs, seg_confirmed):
        return False, f"still at the default {DEFAULT_SEG_M:.3f} m"
    return True, ""


# --- A folder of them ---
@dataclass
class BatchItem:
    """One photograph's outcome in a folder run."""
    photo: str
    status: str          # rectified | skipped | failed
    reason: str = ""
    out_path: Optional[str] = None
    gsd_m: float = 0.0
    warnings: list = _dc_field(default_factory=list)


def plan_rectify_folder(rows):
    """Split a folder into what will be rectified and what will not, and why.

    ``rows`` is ``[(name, corners, seg_lengths_m, seg_confirmed, already_done)]``
    (filenames and numbers, so a plan can be shown before anything is opened).

    Returns ``(ready, skipped)`` where ``skipped`` is ``[(name, reason)]``.
    """
    ready, skipped = [], []
    for row in rows or ():
        name, corners, segs, confirmed, done = (list(row) + [None] * 5)[:5]
        # Readiness is decided before doneness: unreadiness is the fact the
        # user can act on.
        ok, why = is_ready_to_rectify(corners, segs, bool(confirmed))
        if not ok:
            skipped.append((name, why))
        elif done:
            skipped.append((name, "already rectified"))
        else:
            ready.append(name)
    return ready, skipped


def lengths_in_play(rows):
    """Every distinct segment length a run would use, with its count, shown
    before the run: `is_ready_to_rectify` blocks the untouched default but not
    a typo, and the length carries forward to every photograph after it."""
    counts = {}
    for row in rows or ():
        segs = (list(row) + [None] * 5)[2]
        try:
            v = float(list(segs)[0])
        except (TypeError, ValueError, IndexError):
            continue
        counts[round(v, 6)] = counts.get(round(v, 6), 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def colliding_outputs(jobs):
    """Jobs that would write to the same place, keyed by that place.

    ``01.jpg`` and ``01.tif`` side by side both resolve to
    ``orthorectified/01_rectified.jpg``, and the GSD tag does not separate
    them. Detected before the run, because by the time the second write lands
    the first rectification is already gone.
    """
    seen = {}
    for job in jobs or ():
        key = str(Path(str(job.get("out") or "")).resolve()).lower()
        if not key:
            continue
        seen.setdefault(key, []).append(
            job.get("photo") or Path(str(job.get("src") or "")).name)
    return {k: v for k, v in seen.items() if len(v) > 1}


def rectify_folder(jobs, *, overwrite=False, progress_fn=None,
                   should_stop=None, tool_name="PebbleMapper"):
    """Rectify every job, reporting as it goes and stopping when asked.

    ``jobs`` is ``[dict(photo=, src=, corners=, segs=, out=, gsd=, thickness=,
    confirmed=)]``, already filtered by :func:`plan_rectify_folder`.

    Every job is independent: one that fails is recorded and the rest continue.
    A stop keeps what completed. Nothing here touches a widget; the caller
    runs it off the event loop, since forty synchronous JPEG writes on
    NiceGUI's loop would exceed its ping window and drop the client.
    """
    out = []
    # Nobody is rectified if two of them would land on one file: skipping only
    # the second would leave the first silently replaced.
    clash = colliding_outputs(jobs)
    clashing = {n for names in clash.values() for n in names}

    total = len(jobs or ())
    for i, job in enumerate(jobs or ()):
        name = job.get("photo") or Path(str(job.get("src") or "")).name
        if name in clashing:
            others = [n for k, v in clash.items() if name in v
                      for n in v if n != name]
            out.append(BatchItem(
                name, "skipped",
                "would write to the same file as "
                + ", ".join(sorted(set(others)))))
            if progress_fn is not None:
                progress_fn(i + 1, total, name)
            continue
        if should_stop is not None and should_stop():
            out.append(BatchItem(name, "skipped", "stopped before this one"))
            continue
        try:
            r = rectify_one(
                job.get("src"), job.get("corners"), job.get("segs"),
                job.get("out"),
                gsd_m=job.get("gsd"),
                camera_matrix=job.get("camera_matrix"),
                dist_coeffs=job.get("dist_coeffs"),
                frame_thickness_m=job.get("thickness"),
                seg_confirmed=bool(job.get("confirmed")),
                overwrite=overwrite, tool_name=tool_name)
        except Exception as ex:                      # noqa: BLE001
            # One unreadable photograph must not cost the other thirty-nine.
            out.append(BatchItem(name, "failed",
                                 f"{type(ex).__name__}: {ex}"))
        else:
            if r.ok:
                out.append(BatchItem(name, "rectified", "", r.out_path,
                                     r.gsd_m, list(r.warnings)))
            else:
                out.append(BatchItem(name, "skipped", r.refusal,
                                     warnings=list(r.warnings)))
        if progress_fn is not None:
            progress_fn(i + 1, total, out[-1].photo)
    return out


def batch_manifest(items, when=""):
    """What the run did, as text to write beside the outputs, on disk because
    ``state`` does not survive a reconnect."""
    done = [i for i in items or () if i.status == "rectified"]
    skipped = [i for i in items or () if i.status == "skipped"]
    failed = [i for i in items or () if i.status == "failed"]
    lines = ["Rectification run" + (f" — {when}" if when else ""),
             f"{len(done)} rectified, {len(skipped)} skipped, "
             f"{len(failed)} failed", ""]
    for i in done:
        line = f"  rectified  {i.photo}  ->  {Path(i.out_path).name}"
        lines.append(line if not i.warnings
                     else line + "   [" + "; ".join(i.warnings) + "]")
    for i in skipped:
        lines.append(f"  skipped    {i.photo}  ({i.reason})")
    for i in failed:
        lines.append(f"  FAILED     {i.photo}  ({i.reason})")
    return "\n".join(lines) + "\n"


# --- The layout model, in numbers the tab can be tested against ---
FIT_COLUMN_PX = 768.0
"""How wide the photograph's column is on screen, in CSS pixels."""


def fit_width(natural_w, natural_h, box_h, column_w=FIT_COLUMN_PX):
    """The drawn width, in screen px, at which the whole photograph is inside
    its box (the surface is capped at 70 % of the viewport and pans/zooms
    inside that box; "Fit" means all of it is visible).

    Never wider than the column, never wider than the photograph's own pixels,
    never taller than the box, never below one pixel (a degenerate width would
    make every screen-pixel constant divide by zero downstream).
    """
    try:
        W = float(natural_w or 0.0)
        H = float(natural_h or 0.0)
        box = float(box_h or 0.0)
        col = float(column_w or 0.0)
    except (TypeError, ValueError):
        return 1.0
    w = col if col > 0 else 1.0
    if W > 0:
        w = min(w, W)
        if H > 0 and box > 0:
            w = min(w, W * box / H)
    return max(1.0, w)


def loupe_marks(corners, cx, cy, zoom=4, size=200, editing=None):
    """Where the placed corners fall inside the magnifier.

    The loupe shows a ``size / zoom`` px window of the photograph centred on
    ``(cx, cy)`` at ``zoom`` x. A corner at image ``(x, y)`` lands at
    ``((x - (cx - w/2)) * zoom, (y - (cy - w/2)) * zoom)`` in loupe pixels.
    Corners outside the window are dropped; edges are returned whole (the
    canvas clips them) so a side crossing the window is still drawn.

    The JavaScript in the tab is a transcription of this function; this is
    the one that is tested.
    """
    try:
        z = float(zoom)
        s = float(size)
    except (TypeError, ValueError):
        return {"corners": [], "edges": []}
    if z <= 0 or s <= 0:
        return {"corners": [], "edges": []}
    win = s / z
    ox, oy = float(cx) - win / 2.0, float(cy) - win / 2.0

    def to_loupe(pt):
        return (float(pt[0]) - ox) * z, (float(pt[1]) - oy) * z

    pts = [to_loupe(c) for c in (corners or []) if c is not None
           and len(c) >= 2]
    out = []
    for i, (lx, ly) in enumerate(pts):
        if 0.0 <= lx <= s and 0.0 <= ly <= s:
            out.append({"n": i + 1, "x": lx, "y": ly,
                        "editing": (editing is not None and i == editing)})
    edges = []
    n = len(pts)
    if n >= 2:
        pairs = [(i, i + 1) for i in range(n - 1)]
        if n == 4:
            pairs.append((3, 0))
        for a, b in pairs:
            edges.append([pts[a][0], pts[a][1], pts[b][0], pts[b][1]])
    return {"corners": out, "edges": edges}


def batch_gate_message(skipped):
    """Why the folder run did nothing, said as what to do and where (every
    gate names the field and its band).

    ``skipped`` is the ``[(name, reason)]`` list of :func:`plan_rectify_folder`;
    each reason maps to the control that clears it.
    """
    reasons = [str(r or "") for _, r in (skipped or ())]
    if not reasons:
        return ("Nothing to rectify: set *Folder of photographs* "
                "(Inputs, above) to a folder with photographs.")
    parts = []
    if any("default" in r for r in reasons):
        parts.append("Confirm *Quadrat size* or type the frame's size, then "
                     "press *Record for every photograph* (Inputs, above) "
                     "— the folder run only rectifies photographs whose size "
                     "somebody chose")
    if any("corners" in r for r in reasons):
        parts.append("Place the 4 corners on the image "
                     "(Work surface, below) for the photographs still "
                     "missing them")
    if any("no segment length" in r for r in reasons):
        parts.append("Press *Record for every photograph* beside "
                     "*Quadrat size* (Inputs, above) so each photograph "
                     "carries its size")
    if any("already rectified" in r for r in reasons):
        parts.append("Tick *overwrite existing outputs* (Inputs, above) to "
                     "redo photographs that already have an output")
    if not parts:
        parts.append("The photo table (Inputs, above) says why each one "
                     "was skipped")
    return "Nothing is ready to rectify. " + ". ".join(parts) + "."
