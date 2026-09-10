import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import measure
from osgeo import gdal
import pandas as pd
from shapely.geometry import Polygon,LineString
from datetime import datetime 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from mrcnn import visualize
import mrcnn.model as modellib
from functions import clasts_config
from functions import images as _images
from functions import naming
from functions import modes
config = clasts_config.clastsConfig()
from functions import operations as op
from functions._logging import get_logger

_log = get_logger(__name__)


# Standard column order; keep in sync with clasts_merge._CLAST_COLUMNS.
# Equivalent_diameter is 2·√(A/π) in metres.
_CLAST_COLUMNS = ['clast_ID', 'x', 'y',
                  'Clast_length', 'Clast_width',
                  'Ellipse_major_axis', 'Ellipse_minor_axis',
                  'Surface_area', 'Perimeter', 'Equivalent_diameter',
                  'Eccentricity', 'Solidity', 'Mean_intensity',
                  'Score', 'Orientation']

# .run.csv column order; _tile_idx lets resume derive kstart from the file
# alone and is stripped before final output.
_STREAMING_COLUMNS = _CLAST_COLUMNS + ['_tile_idx']


def _run_state_path_for(output_dir, image_path, image_stem, metric_cropsize):
    """Path of the streaming run-state file for one (image, window size),
    next to the final CSV. Deleting the ``.run.csv`` forces a fresh run."""
    fname = naming.detection_csv_name(image_stem, metric_cropsize, run=True)
    target_dir = output_dir if output_dir else os.path.dirname(image_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, fname)
    return fname


def _run_contours_path_for(streaming_path):
    """``<stem>.run.contours.jsonl`` beside the ``.run.csv``: one JSON line
    per tile, ``{"tile": k, "contours": {clast_ID: [[x, y], ...]}}``, appended
    and flushed after the tile's CSV rows so a resumed run gets back every
    outline of the tiles it does not redo."""
    sp = str(streaming_path)
    base = sp[:-len('.run.csv')] if sp.endswith('.run.csv') else sp
    return base + '.run.contours.jsonl'


def _run_grid_path_for(streaming_path):
    """``<stem>.run.grid.json`` beside the ``.run.csv``: the tile grid the
    checkpoint was laid on (overlap, tile and stride in pixels, tile
    counts). A checkpoint is keyed by image and window size only; a job
    with another overlap lays a different grid, and the checkpoint's tile
    indices then mean other places."""
    sp = str(streaming_path)
    base = sp[:-len('.run.csv')] if sp.endswith('.run.csv') else sp
    return base + '.run.grid.json'


def write_run_grid(streaming_path, *, overlap, cropsize_px, stride_px,
                   n_tiles_x, n_tiles_y) -> dict:
    """Record the grid of a checkpoint that is being started. Never raises."""
    import json as _json
    payload = {"overlap": float(overlap), "cropsize_px": int(cropsize_px),
               "stride_px": int(stride_px), "n_tiles_x": int(n_tiles_x),
               "n_tiles_y": int(n_tiles_y)}
    try:
        with open(_run_grid_path_for(streaming_path), 'w',
                  encoding='utf-8') as fh:
            _json.dump(payload, fh)
    except OSError:
        pass
    return payload


def read_run_grid(streaming_path):
    """The grid a checkpoint was laid on, or None when it left no record
    (a checkpoint from before this record existed)."""
    import json as _json
    try:
        with open(_run_grid_path_for(streaming_path),
                  encoding='utf-8') as fh:
            grid = _json.load(fh)
        return grid if isinstance(grid, dict) else None
    except (OSError, ValueError):
        return None


def run_grid_mismatch(streaming_path, *, overlap, cropsize_px=None,
                      stride_px=None):
    """Why the checkpoint cannot be resumed on the grid asked for, in
    words, or None when it can (or left no record)."""
    grid = read_run_grid(streaming_path)
    if grid is None:
        return None
    try:
        if abs(float(grid.get("overlap", overlap)) - float(overlap)) > 1e-6:
            return (f"it was laid with overlap {float(grid['overlap']):.2f}, "
                    f"this job asks for {float(overlap):.2f}")
        if (cropsize_px is not None and "cropsize_px" in grid
                and int(grid["cropsize_px"]) != int(cropsize_px)):
            return (f"its tiles are {int(grid['cropsize_px'])} px, "
                    f"this job's {int(cropsize_px)} px")
        if (stride_px is not None and "stride_px" in grid
                and int(grid["stride_px"]) != int(stride_px)):
            return (f"its stride is {int(grid['stride_px'])} px, "
                    f"this job's {int(stride_px)} px")
    except (TypeError, ValueError):
        return "its grid record is unreadable"
    return None


def discard_mismatched_checkpoint(streaming_path, contours_run_path, *,
                                  overlap, cropsize_px, stride_px):
    """Remove a checkpoint (and its outlines and grid record) that was laid
    on another grid than the one asked for. Returns the reason when one was
    removed, else None."""
    if not os.path.exists(streaming_path):
        return None
    why = run_grid_mismatch(streaming_path, overlap=overlap,
                            cropsize_px=cropsize_px, stride_px=stride_px)
    if not why:
        return None
    for p in (streaming_path, contours_run_path,
              _run_grid_path_for(streaming_path)):
        try:
            os.remove(p)
        except OSError:
            pass
    return why


def _read_run_contours(path, last_tile):
    """``{clast_ID: outline}`` from the tiles ``<= last_tile`` of a run's
    JSON-lines file; a truncated last line (a crash mid-write) is skipped.
    Never raises."""
    import json as _json
    out = {}
    if last_tile is None or not os.path.exists(path):
        return out
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if not line.endswith('\n'):
                    break
                try:
                    doc = _json.loads(line)
                except ValueError:
                    continue
                if int(doc.get('tile', -1)) > int(last_tile):
                    continue
                for k, v in (doc.get('contours') or {}).items():
                    out[int(k)] = v
    except Exception as ex:
        _log.warning("Could not read the run's outlines %s (%s); the "
                     "resumed tiles will have none.", path, ex)
    return out


def inspect_run_state(output_dir, image_path, metric_cropsize, out_stem=None,
                      overlap=None):
    """Report progress on an interrupted run as a dict: exists, path,
    last_tile, n_clasts (excluding empty-tile markers), resume_k (the kstart
    to continue from, last_tile + 1, or 0) and, when the job's ``overlap``
    is given, grid_mismatch: why the checkpoint cannot be resumed by that
    job (None when it can)."""
    image_stem = out_stem or os.path.splitext(os.path.basename(image_path))[0]
    path = _run_state_path_for(output_dir, image_path, image_stem, metric_cropsize)
    if not os.path.exists(path):
        return {"exists": False, "path": None, "last_tile": None,
                "n_clasts": 0, "resume_k": 0, "grid_mismatch": None}
    mismatch = (run_grid_mismatch(path, overlap=overlap)
                if overlap is not None else None)
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"exists": True, "path": path, "last_tile": None,
                "n_clasts": 0, "resume_k": 0, "grid_mismatch": mismatch}
    last_tile = int(df['_tile_idx'].max()) if '_tile_idx' in df.columns and len(df) > 0 else None
    n_clasts = int(df['clast_ID'].notna().sum()) if 'clast_ID' in df.columns else 0
    return {"exists": True, "path": path, "last_tile": last_tile,
            "n_clasts": n_clasts,
            "resume_k": (last_tile + 1) if last_tile is not None else 0,
            "grid_mismatch": mismatch}


class _StreamingResultsWriter:
    """Append-only ``.run.csv`` writer for crash-resilient detection runs:
    each tile's rows are flushed (and fsynced) as they are measured, so a
    crash between tiles loses nothing already completed."""
    def __init__(self, path, columns):
        self.path = path
        # An existing .run.csv keeps ITS header so appended rows line up on
        # resume across schema changes; columns missing from the old header
        # are dropped, unknown ones become empty cells.
        self._preferred_columns = list(columns)
        self._file = None
        self._is_new = not os.path.exists(path)
        if self._is_new:
            self.columns = list(columns)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    first = fh.readline()
                self.columns = [c.strip() for c in first.split(',')
                                if c.strip()]
                if not self.columns:
                    self.columns = list(columns)
                    self._is_new = True
            except OSError:
                self.columns = list(columns)
                self._is_new = True

    def __enter__(self):
        # Append mode so a reopen after a crash never truncates; line-buffered.
        self._file = open(self.path, 'a', encoding='utf-8',
                          newline='', buffering=1)
        if self._is_new:
            self._file.write(','.join(self.columns) + '\n')
            self._file.flush()
        return self

    def __exit__(self, *exc):
        if self._file is not None:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except (OSError, AttributeError):
                pass
            self._file.close()
            self._file = None

    def write_tile(self, tile_idx, rows, *, fsync=True):
        """Append one tile's rows (missing keys become empty cells), flush,
        and fsync unless ``fsync=False``. Skip the fsync for empty-tile
        markers: losing one on a hard crash only re-processes that tile,
        and fsyncing every empty tile is the main I/O cost."""
        if not rows:
            return
        for row in rows:
            row = dict(row)
            row['_tile_idx'] = tile_idx
            line = ','.join(_csv_value(row.get(c, '')) for c in self.columns)
            self._file.write(line + '\n')
        self._file.flush()
        if fsync:
            try:
                os.fsync(self._file.fileno())
            except (OSError, AttributeError):
                pass


def _csv_value(v):
    """Format one CSV cell: floats as %.5f, strings quoted if needed."""
    if isinstance(v, float):
        return f'{v:.5f}'
    if isinstance(v, (int, bool)):
        return str(v)
    if v is None:
        return ''
    s = str(v)
    if ',' in s or '"' in s or '\n' in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _normalize_to_uint8(image):
    """Convert any incoming image array to 3-channel uint8 in 0-255.

    The model expects uint8 RGB (MEAN_PIXEL ≈ [111, 111, 107]); readers
    return float 0-1 for PNG, uint16 for some orthos, and an alpha channel for
    RGBA. Fed unnormalised, a PNG yields silent garbage detections.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if image.dtype == np.uint16:
        return (image / 257).astype(np.uint8)   # 257 = 65535/255 (exact)
    # Other dtypes: min/max stretch to 0-255
    img_min, img_max = float(image.min()), float(image.max())
    if img_max > img_min:
        return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return np.zeros(image.shape, dtype=np.uint8)


def get_ax(rows=1, cols=1, size=16):

    """Matplotlib Axes array sized ``size`` inches per panel."""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Single-entry cache for the normalised full-ortho array (a multi-window run
# detects on the same ortho back-to-back; re-reading a large ortho costs
# gigabytes and seconds). Keyed by (abspath, mtime) so an edited image is
# never served stale; only one ortho is ever held in RAM.
_ORTHO_READ_CACHE = {"key": None, "image": None, "gt": None}


def _read_ortho_for_detection(imgpath):
    """(normalised uint8 RGB image, geotransform) for a UAV ortho, via the
    one-entry cache."""
    try:
        key = (os.path.abspath(imgpath), os.path.getmtime(imgpath))
    except OSError:
        key = None
    if (key is not None and _ORTHO_READ_CACHE["key"] == key
            and _ORTHO_READ_CACHE["image"] is not None):
        return _ORTHO_READ_CACHE["image"], _ORTHO_READ_CACHE["gt"]
    raster = gdal.Open(imgpath, gdal.GA_ReadOnly)
    geotransform = raster.GetGeoTransform()
    image = np.dstack([raster.GetRasterBand(1).ReadAsArray(),
                       raster.GetRasterBand(2).ReadAsArray(),
                       raster.GetRasterBand(3).ReadAsArray()])
    raster = None
    image = _normalize_to_uint8(image)
    if key is not None:
        _ORTHO_READ_CACHE["key"] = key
        _ORTHO_READ_CACHE["image"] = image
        _ORTHO_READ_CACHE["gt"] = geotransform
    return image, geotransform

# Single-entry model cache keyed by (devicemode, devicenumber, min_confidence),
# all of which affect graph construction. build_args lets the tile loop
# rebuild the model if the TF session is reset mid-run.
_MODEL_CACHE = {
    "key": None,
    "model": None,
    "loaded_at": None,
    "build_args": None,
    "session": None,     # the TF session the model was built in
}


def _keras_backend():
    """Keras's backend module (the one Mask R-CNN uses), or None."""
    try:
        from keras import backend as _K
        return _K
    except Exception:
        try:
            return tf.compat.v1.keras.backend
        except Exception:
            return None


def _remember_session():
    """Record the session the model was just built in."""
    K = _keras_backend()
    try:
        _MODEL_CACHE["session"] = K.get_session() if K is not None else None
    except Exception:
        _MODEL_CACHE["session"] = None


def _pin_cached_session() -> bool:
    """Make the cached model's graph and session the calling thread's.

    TensorFlow's default graph is a property of the thread, and Keras keeps
    its session per thread. Every Detect batch runs in a fresh worker
    thread: the second batch started with an empty default graph, Keras saw
    that the cached session did not belong to that graph and opened a new
    session on it, the first predict failed with "Tensor input_image:0 ...
    was not found in the Graph", and the rebuild that followed ran out of
    GPU memory because the first session still held its share. Installing the model's graph as the thread's default
    graph the way reset_default_graph() itself does (the thread's context
    stack stays empty, so a later clear_session() is still allowed) and
    pinning its session makes the cached model usable from any thread.
    Returns True when both were pinned."""
    sess = _MODEL_CACHE.get("session")
    K = _keras_backend()
    if sess is None or K is None:
        return False
    try:
        from tensorflow.python.framework import ops as _ops
        if _ops.get_default_graph() is not sess.graph:
            stack = _ops._default_graph_stack
            if stack.stack:
                # Inside somebody's `with graph.as_default()`: not ours.
                return False
            stack._global_default_graph = sess.graph
        K.set_session(sess)
        return True
    except Exception:
        return False


def _close_cached_session():
    """Close the cached model's session so its device memory can be reused
    by the next build (clear_session alone leaves it allocated)."""
    sess = _MODEL_CACHE.get("session")
    _MODEL_CACHE["session"] = None
    if sess is not None:
        try:
            sess.close()
        except Exception:
            pass


def get_model_status():
    """Snapshot of the model cache: loaded, key, loaded_at (epoch s), age_s."""
    import time as _t
    key = _MODEL_CACHE["key"]
    loaded_at = _MODEL_CACHE["loaded_at"]
    return {
        "loaded": _MODEL_CACHE["model"] is not None,
        "key": key,
        "loaded_at": loaded_at,
        "age_s": (_t.time() - loaded_at) if loaded_at else None,
    }


def clear_model_cache():
    """Drop the cached model (freeing GPU memory) and the cached ortho array;
    the next detection call rebuilds."""
    if _MODEL_CACHE["model"] is not None:
        _log.info("Clearing cached model and TF session.")
        _MODEL_CACHE["model"] = None
        _MODEL_CACHE["key"] = None
        _MODEL_CACHE["loaded_at"] = None
        _MODEL_CACHE["build_args"] = None
        _close_cached_session()
        tf.keras.backend.clear_session()
    _ORTHO_READ_CACHE["key"] = None
    _ORTHO_READ_CACHE["image"] = None
    _ORTHO_READ_CACHE["gt"] = None


def _build_model(devicemode, devicenumber, min_confidence, use_cache=True):
    """Construct a Mask R-CNN inference model and load its weights, reusing
    the cached model when the (devicemode, devicenumber, min_confidence) key
    matches and ``use_cache`` is set."""
    import time as _t
    cache_key = (str.lower(devicemode), int(devicenumber),
                 float(min_confidence) if min_confidence is not None else None)

    if use_cache and _MODEL_CACHE["key"] == cache_key and _MODEL_CACHE["model"] is not None:
        age = _t.time() - _MODEL_CACHE["loaded_at"]
        _log.info("Reusing cached model (key=%s, age=%.0fs).", cache_key, age)
        _MODEL_CACHE["build_args"] = (devicemode, devicenumber, min_confidence)
        _pin_cached_session()
        return _MODEL_CACHE["model"]

    if _MODEL_CACHE["model"] is not None:
        if _MODEL_CACHE["key"] == cache_key:
            _log.info("Rebuilding the model on the same key %s — the previous "
                      "TF session was invalidated (an error, or a cache "
                      "clear), not a parameter change.", cache_key)
        else:
            _log.info("Model cache key changed (was %s, now %s). Rebuilding.",
                      _MODEL_CACHE['key'], cache_key)

    DEVICE = "/" + str.lower(devicemode) + ":" + str(devicenumber)  # /cpu:0 or /gpu:0
    model_dir = os.path.join(ROOT_DIR, "model_weights")

    # Null the cache BEFORE clear_session() so a concurrent reader never sees
    # a model whose TF graph is about to be destroyed.
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["key"] = None
    _MODEL_CACHE["loaded_at"] = None
    _MODEL_CACHE["build_args"] = None

    # Release the previous graph first, or a second build in the same process OOMs.
    _close_cached_session()
    tf.keras.backend.clear_session()

    # DETECTION_MIN_CONFIDENCE is baked into the graph at build time, so the
    # override must precede MaskRCNN.__init__.
    if min_confidence is not None:
        config.DETECTION_MIN_CONFIDENCE = float(min_confidence)
        _log.info("DETECTION_MIN_CONFIDENCE = %s", config.DETECTION_MIN_CONFIDENCE)

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)

    # Per-backend weights layout first, legacy location as fallback.
    _wcands = [os.path.join(ROOT_DIR, "models", "maskrcnn", "mask_rcnn_clasts.h5"),
               os.path.join(model_dir, "mask_rcnn_clasts.h5")]
    weights_path = next((p for p in _wcands if os.path.exists(p)), _wcands[-1])
    _log.info("Loading weights %s", weights_path)
    model.load_weights(weights_path, by_name=True)

    if use_cache:
        _MODEL_CACHE["key"] = cache_key
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["loaded_at"] = _t.time()
        _MODEL_CACHE["build_args"] = (devicemode, devicenumber, min_confidence)
        _remember_session()
        _log.info("Model cached with key=%s.", cache_key)
    return model


# --- GPU out-of-memory recovery ---
# Per-tile inference can OOM allocating the mask-head tensor
# [DETECTION_MAX_INSTANCES, 256, 28, 28] (~0.8 GB at 1000 instances). On OOM
# the per-tile caps are halved and the model rebuilt, down to the stock floor;
# dense tiles may then under-count, but a capped result beats a crash.
_OOM_FLOOR_INSTANCES = 100      # mask_rcnn stock DETECTION_MAX_INSTANCES
_OOM_FLOOR_ROIS = 1000          # mask_rcnn stock POST_NMS_ROIS_INFERENCE


def _is_oom_error(ex) -> bool:
    """True when ``ex`` looks like a GPU out-of-memory error."""
    name = type(ex).__name__
    msg = str(ex)
    return ("ResourceExhaustedError" in name
            or "OOM" in msg
            or "out of memory" in msg.lower())


def _rebuild_with_reduced_caps():
    """Halve the per-tile instance / ROI caps (down to the stock floor) and
    rebuild the model. Returns the rebuilt model, or ``None`` when the caps
    are at the floor or the build arguments are unavailable. Mutates the
    shared ``config`` so the reduced caps persist for the rest of the run."""
    cur_inst = int(getattr(config, "DETECTION_MAX_INSTANCES",
                           _OOM_FLOOR_INSTANCES))
    cur_rois = int(getattr(config, "POST_NMS_ROIS_INFERENCE",
                           _OOM_FLOOR_ROIS))
    build_args = _MODEL_CACHE.get("build_args")
    if cur_inst <= _OOM_FLOOR_INSTANCES or build_args is None:
        return None
    new_inst = max(_OOM_FLOOR_INSTANCES, cur_inst // 2)
    new_rois = max(_OOM_FLOOR_ROIS, cur_rois // 2)
    _log.warning(
        "GPU OOM recovery: reducing per-tile caps DETECTION_MAX_INSTANCES "
        "%d -> %d, POST_NMS_ROIS_INFERENCE %d -> %d, and rebuilding the "
        "model. Very dense tiles may now under-count (max %d clasts/tile).",
        cur_inst, new_inst, cur_rois, new_rois, new_inst)
    clear_model_cache()
    config.DETECTION_MAX_INSTANCES = new_inst
    config.POST_NMS_ROIS_INFERENCE = new_rois
    return _build_model(*build_args, use_cache=True)


def _mask_mean_intensity(image, mask):
    """Mean brightness (0–255) of the image pixels under a boolean mask, across
    all channels. NaN for an empty mask. Best-effort: never raises."""
    try:
        m = np.asarray(mask).astype(bool)
        if not m.any():
            return float('nan')
        return float(np.mean(image[m]))
    except Exception:
        return float('nan')


def _measure_clast(mask, score, resolution):
    """Measure one per-clast binary mask. Returns a dict (metres, m², degrees)
    without world coordinates, which the caller derives differently for
    quadrat (pixel space) and ortho (geotransform + tile offset) modes, or
    None if the clast should be skipped.

    Keys: center_x/center_y (pixels in the tile), Clast_length/Clast_width
    (polygon-axis intersection), Ellipse_major/minor_axis (second-moment),
    Surface_area, Perimeter, Equivalent_diameter, Eccentricity (0 circle → 1
    elongated), Solidity (area / convex-hull area), Score, Orientation.
    """
    contours = measure.find_contours(mask.astype(int), 0.99999)

    # A disconnected mask cannot be measured reliably as one clast.
    if len(contours) != 1:
        return None

    x = contours[0][:, 1]
    y = contours[0][:, 0]
    ell = op.fit_ellipse(x, y)
    center = op.ellipse_center(ell)
    phi = op.ellipse_angle_of_rotation(ell)
    axes = op.ellipse_axis_length(ell)  # results = radii !
    a, b = axes

    # Axis lines extend 10x past the ellipse so they clip the polygon at both ends.
    xx_ax1 = np.linspace(-a*10, a*10, 100)
    yy_ax1 = np.zeros_like(xx_ax1)
    xx_ax1_rot = center[0] + xx_ax1 * np.cos(phi) + yy_ax1 * np.sin(phi)
    yy_ax1_rot = center[1] + xx_ax1 * np.sin(phi) + yy_ax1 * np.cos(phi)

    xx_ax2 = np.linspace(-b*10, b*10, 100)
    yy_ax2 = np.zeros_like(xx_ax2)
    xx_ax2_rot = center[0] + xx_ax2 * np.cos(phi - np.pi/2) + yy_ax2 * np.sin(phi - np.pi/2)
    yy_ax2_rot = center[1] + xx_ax2 * np.sin(phi - np.pi/2) + yy_ax2 * np.cos(phi - np.pi/2)

    # Keep find_contours' natural vertex order; sorting by polar angle breaks
    # non-star-convex shapes.
    polygon = list(zip(x, y))
    p1 = Polygon(polygon).buffer(0)

    line_ax1 = LineString([(xx_ax1_rot[0], yy_ax1_rot[0]),
                           (xx_ax1_rot[-1], yy_ax1_rot[-1])])
    line_ax2 = LineString([(xx_ax2_rot[0], yy_ax2_rot[0]),
                           (xx_ax2_rot[-1], yy_ax2_rot[-1])])

    if p1.is_valid and line_ax1.is_valid and line_ax2.is_valid:
        length = p1.intersection(line_ax1).length
        width = p1.intersection(line_ax2).length
    else:
        length = math.nan
        width = math.nan

    clast_length = length * resolution
    clast_width = width * resolution
    # NaN comparisons are False, so this also drops unmeasurable clasts.
    if not (clast_length > 0 and clast_width > 0):
        return None

    surface_area = float(np.sum(mask)) * resolution**2

    # Diameter of the disc with the same area as the mask.
    eq_diameter = 2.0 * math.sqrt(surface_area / math.pi) \
                  if surface_area > 0 else math.nan

    _cx = np.append(x, x[0])
    _cy = np.append(y, y[0])
    perimeter = float(np.sum(np.hypot(np.diff(_cx), np.diff(_cy)))) * resolution
    _amax, _amin = max(a, b), min(a, b)
    eccentricity = (float(math.sqrt(1.0 - (_amin / _amax) ** 2))
                    if _amax > 0 else 0.0)
    try:
        _hull_area = p1.convex_hull.area
        # nan, not 0.0: zero is a legitimate (maximally concave) solidity.
        solidity = (float(p1.area / _hull_area) if _hull_area > 0
                    else float("nan"))
    except Exception:
        solidity = float("nan")

    # Second-moment ellipse axes (full lengths, m), the regionprops a/b other
    # tools report; kept for cross-method comparability.
    try:
        _rp = measure.regionprops((mask > 0).astype(int))
        if _rp:
            ell_major = float(_rp[0].axis_major_length) * resolution
            ell_minor = float(_rp[0].axis_minor_length) * resolution
        else:
            ell_major = ell_minor = float('nan')
    except Exception:
        ell_major = ell_minor = float('nan')

    return {
        'center_x': abs(center[0]),
        'center_y': abs(center[1]),
        'Clast_length': clast_length,
        'Clast_width': clast_width,
        'Ellipse_major_axis': ell_major,
        'Ellipse_minor_axis': ell_minor,
        'Surface_area': surface_area,
        'Perimeter': perimeter,
        'Equivalent_diameter': eq_diameter,
        'Eccentricity': eccentricity,
        'Solidity': solidity,
        'Score': score,
        'Orientation': np.rad2deg(phi) + 90,
        # extras for plotting (quadrat mode only)
        '_contour_x': x,
        '_contour_y': y,
        '_ellipse_phi': phi,
        '_ellipse_a': a,
        '_ellipse_b': b,
        '_ellipse_center': center,
    }


def _completed_before(csv_path) -> bool:
    """True when the CSV beside this run's log was written by a run that
    finished. The log is append-mode, so the last ``outcome`` line is the
    state of the file on disk."""
    stem = csv_path[:-len('.csv')] if str(csv_path).endswith('.csv') else str(csv_path)
    log_path = stem + '_detection_log.txt'
    if not (os.path.exists(csv_path) and os.path.exists(log_path)):
        return False
    outcome = None
    try:
        with open(log_path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if 'outcome' in line and ':' in line:
                    outcome = line.split(':', 1)[1].strip()
    except OSError:
        return False
    return outcome == 'complete'


def quadrat_overlay_name(img_stem):
    """``<stem>_overlay.png``: the Quadrat overlay (was ``_ellipses.png``)."""
    return img_stem + '_overlay.png'


def _plot_quadrat_overlay(ax, image, clasts, contours, resolution):
    """The photograph, in darkened greyscale, with every clast as its mask
    outline filled by its half-phi size class, its length chord (solid) and
    width chord (dotted), and its centroid; a legend of the classes under it.
    Axes in image pixels, rows down. ``clasts`` is in the CSV frame (y up),
    lengths in metres; ``resolution`` metres per pixel."""
    from functions import clast_geometry as CG
    H, W = image.shape[:2]
    ax.imshow(CG.muted_background(image), extent=(0, W, H, 0), interpolation='bilinear')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    if len(clasts):
        classes = CG.size_classes(clasts['Clast_length'] * 1000.0, unit="mm")
        CG.draw_clasts(ax, clasts, contours=contours, y_down=True,
                       to_data=lambda xs, ys: (xs, H - np.asarray(ys, float)),
                       units_per_m=1.0 / float(resolution),
                       edge_colors=classes["colours"])
        CG.add_size_legend(ax, classes)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{len(clasts)} clasts by size class, with their length "
                 f"(solid) and width (dotted) chords")


def _print_size_stats(clasts, n_detected_total):
    """Log the D10/D50/D90 grain-size statistics block for one image."""
    _log.info("Detected clasts: %d", n_detected_total)
    for label, col in [('Clast Length',       'Clast_length'),
                       ('Clast Width',        'Clast_width'),
                       ('Equivalent Diameter', 'Equivalent_diameter')]:
        for q, qlabel in [(0.1, 'D10'), (0.5, 'D50'), (0.9, 'D90')]:
            v = round(np.quantile(clasts[col], q) * 100, 2)
            _log.info("  %s %s = %.2f cm", label, qlabel, v)


def _load_roi_paths(roi_path):
    """Load ROI polygons from a GeoJSON into matplotlib Paths. Coordinates are
    taken as stored: world coords for ortho ROIs, image-pixel (col, row) for
    quadrat photographs. Returns [] on any problem; ROI filtering must never
    crash a detection run."""
    roi_paths = []
    if not roi_path:
        return roi_paths
    try:
        import json as _json
        with open(roi_path, "r", encoding="utf-8") as _fh:
            _gj = _json.load(_fh)
        from matplotlib.path import Path as _MplPath
        _features = _gj.get("features", [])
        if len(_features) > 1000:
            _log.warning(
                "ROI file %s has %d features; capping at 1000 to avoid lag.",
                os.path.basename(roi_path), len(_features))
            _features = _features[:1000]
        # A ring that will not build shrinks the searched area; count and report it.
        _dropped = 0
        for _feat in _features:
            _geom = _feat.get("geometry") or {}
            _gt = _geom.get("type")
            if _gt == "Polygon":
                _rings = _geom.get("coordinates", [])
                if _rings and isinstance(_rings[0], list) and len(_rings[0]) >= 3:
                    try:
                        roi_paths.append(_MplPath(_rings[0]))
                    except Exception:
                        _dropped += 1
                else:
                    _dropped += 1
            elif _gt == "MultiPolygon":
                for _poly in _geom.get("coordinates", []):
                    if (_poly and isinstance(_poly[0], list)
                            and len(_poly[0]) >= 3):
                        try:
                            roi_paths.append(_MplPath(_poly[0]))
                        except Exception:
                            _dropped += 1
                    else:
                        _dropped += 1
        if _dropped:
            _log.warning(
                "ROI %s: %d of %d shape(s) could not be used and are NOT "
                "masked. The area actually searched is smaller than the one "
                "drawn.", os.path.basename(roi_path), _dropped,
                _dropped + len(roi_paths))
    except Exception as _ex:
        _log.warning("Could not load ROI %s: %s; no ROI filtering.",
                     roi_path, _ex)
        return []
    return roi_paths


def _detect_quadrat(model, imgpath, resolution, plot, saveplot, saveresults,
                    output_dir=None, figures_dir=None, roi_path=None,
                    out_stem=None):
    """Detect and measure clasts in a single scaled photograph. ``output_dir``
    (CSV) and ``figures_dir`` (PNGs) default to the image's own directory."""
    class_names = ['BG', 'Clast']
    # Pillow, in the stored pixel frame (the frame the ROI canvas shows and
    # the quadrat ROI polygons are saved in); a HEIC reads directly.
    image = _normalize_to_uint8(_images.read_rgb(imgpath))
    try:
        results = model.detect([image], verbose=0)
    except Exception as _det_ex:
        _ex_type = type(_det_ex).__name__
        if ("InvalidArgumentError" in _ex_type
                or "input_image" in str(_det_ex)):
            _build_args = _MODEL_CACHE.get("build_args")
            if _build_args is not None:
                _log.warning(
                    "TF session invalidated for %s (%s). "
                    "Rebuilding model and retrying.",
                    os.path.basename(imgpath), _ex_type)
                model = _build_model(*_build_args, use_cache=False)
                results = model.detect([image], verbose=0)
            else:
                raise
        else:
            raise
    r = results[0]
    n_detected = np.shape(r['masks'])[2]
    image_height = np.shape(image)[0]

    # The output stem carries the origin (site, date, image) when supplied.
    img_stem = out_stem or os.path.splitext(os.path.basename(imgpath))[0]
    csv_dir = output_dir if output_dir else os.path.dirname(imgpath) or "."
    fig_dir = figures_dir if figures_dir else os.path.dirname(imgpath) or "."
    if output_dir:
        os.makedirs(csv_dir, exist_ok=True)
    if figures_dir:
        os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, naming.quadrat_csv_name(img_stem))
    overlay_png = os.path.join(fig_dir, quadrat_overlay_name(img_stem))
    histogram_png = os.path.join(fig_dir, img_stem + '_histogram.png')
    from functions import clast_geometry as _CG
    to_frame = _CG.quadrat_frame(image_height)
    contours = {}

    if n_detected == 0:
        clasts = pd.DataFrame([], columns=_CLAST_COLUMNS)
        _CG.attach_contours(clasts, {}, frame="pixels")
        if saveresults:
            clasts.to_csv(csv_path, index=False, float_format='%.5f')
            _CG.write_contours(csv_path, {}, frame="pixels")
        _log.info("No detections for %s.", os.path.basename(imgpath))
        return clasts

    # per-image ROI(s). Quadrat photographs are not tiled like orthos, so we
    # filter in pixel space: a clast is kept only if its centroid
    # (center_x, center_y, in image-pixel coords) falls inside an ROI polygon.
    # The canvas saves quadrat ROIs in the same (col, row) pixel frame.
    roi_paths = _load_roi_paths(roi_path)
    if roi_paths:
        _log.info("ROI: loaded %d polygon(s) from %s; clasts outside every "
                  "polygon will be dropped.",
                  len(roi_paths), os.path.basename(roi_path))
    elif roi_path:
        _log.warning("ROI file %s yielded no usable polygons; no clasts "
                     "filtered.", os.path.basename(roi_path))
    n_roi_skipped = 0

    records = []
    for i in range(n_detected):
        meas = _measure_clast(r['masks'][:, :, i], r['scores'][i], resolution)
        if meas is None:
            continue
        if roi_paths and not any(
                rp.contains_point((meas['center_x'], meas['center_y']))
                for rp in roi_paths):
            n_roi_skipped += 1
            continue
        try:
            outline = _CG.contour_from_measurement(meas, to_frame=to_frame)
            if outline:
                contours[len(records) + 1] = outline
        except Exception:
            pass    # an outline is a nicety; the measurement is the product
        records.append({
            'clast_ID': len(records) + 1,                       # audit fix #1
            'x': meas['center_x'],
            'y': image_height - meas['center_y'],               # audit fix #2 (image HEIGHT)
            'Clast_length': meas['Clast_length'],
            'Clast_width': meas['Clast_width'],
            'Ellipse_major_axis': meas['Ellipse_major_axis'],
            'Ellipse_minor_axis': meas['Ellipse_minor_axis'],
            'Surface_area': meas['Surface_area'],
            'Perimeter': meas['Perimeter'],
            'Equivalent_diameter': meas['Equivalent_diameter'],
            'Eccentricity': meas['Eccentricity'],
            'Solidity': meas['Solidity'],
            'Mean_intensity': _mask_mean_intensity(image, r['masks'][:, :, i]),
            'Score': meas['Score'],
            'Orientation': meas['Orientation'],
        })

    clasts = pd.DataFrame(records, columns=_CLAST_COLUMNS)
    _CG.attach_contours(clasts, contours, frame="pixels")
    if roi_paths and n_roi_skipped:
        _log.info("ROI: dropped %d clast(s) outside the region(s) of interest "
                  "in %s.", n_roi_skipped, os.path.basename(imgpath))

    if plot or saveplot:
        ax = get_ax(1)
        _plot_quadrat_overlay(ax, image, clasts, _CG.contours_of(clasts),
                              resolution)
        fig = plt.gcf()
        fig.set_size_inches(16, 16)
        if saveplot:
            fig.savefig(overlay_png, dpi=100, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close(fig)

    if len(clasts) > 0:
        if plot or saveplot:
            fig = plt.figure(figsize=(10, 10))
            plt.hist(clasts['Clast_length'] * 100, 20)
            plt.xlabel('Grain Size (cm)')
            plt.ylabel('Number of clasts')
            if saveplot:
                fig.savefig(histogram_png, dpi=100, bbox_inches='tight')
            if plot:
                plt.show()
            else:
                plt.close(fig)
        _print_size_stats(clasts, n_detected)
    else:
        _log.info("No valid clasts measured for %s.", os.path.basename(imgpath))

    if saveresults:
        clasts.to_csv(csv_path, index=False, float_format='%.5f')
        _CG.write_contours(csv_path, contours, frame="pixels")
    return clasts


def _tile_figure(tile, r, class_names, title, tile_png, show):
    """Draw one tile's masks on its own figure, save it to ``tile_png`` when
    given and show it when ``show``. The figure is ours, so the helper's
    own ``plt.show()`` never runs: in the app (a headless backend) it
    printed a Matplotlib warning into the console for every tile, and an
    empty tile printed '*** No instances to display ***'. An empty tile
    is drawn as the tile alone."""
    fig, ax = plt.subplots(1, figsize=(16, 16))
    try:
        if r['rois'].shape[0]:
            visualize.display_instances(tile, r['rois'], r['masks'],
                                        r['class_ids'], class_names,
                                        r.get('scores'), title=title, ax=ax)
        else:
            ax.imshow(tile.astype(np.uint8))
            ax.set_title(f"{title} — no detection")
            ax.axis('off')
        if tile_png:
            fig.savefig(tile_png, dpi=80, bbox_inches='tight')
        if show:
            plt.show()
    finally:
        plt.close(fig)


def _dedup_on_completion(clasts, contours, dedup_method, dedup_overlap,
                         tile_overlap):
    """Merge the duplicates that overlapping tiles produce, once a run is
    complete. A threshold of 0 (or None) disables the step, as the manual
    says: with "IoU at or above 0" every neighbouring pair would be a
    conflict. Returns (clasts, contours)."""
    try:
        threshold = float(dedup_overlap) if dedup_overlap is not None else 0.0
    except (TypeError, ValueError):
        threshold = 0.0
    if threshold <= 0.0:
        if tile_overlap and float(tile_overlap) > 0 and len(clasts) > 1:
            _log.info("Tile dedup disabled (threshold 0): a clast seen in "
                      "two overlapping tiles is counted twice.")
        return clasts, contours
    if len(clasts) <= 1:
        return clasts, contours
    from functions.clasts_merge import dedup_clasts
    pre_n = len(clasts)
    clasts = dedup_clasts(clasts, method=dedup_method, overlap=threshold)
    # The outlines follow the kept rows to their new IDs.
    _old_ids = clasts['clast_ID'].tolist()
    contours = {new: contours[int(float(old))]
                for new, old in enumerate(_old_ids, start=1)
                if old == old and int(float(old)) in contours}
    clasts['clast_ID'] = range(1, len(clasts) + 1)
    _log.info(
        "Tile dedup (%s at or above %.2f): %d → %d clasts (%d duplicates removed).",
        dedup_method, threshold, pre_n, len(clasts), pre_n - len(clasts))
    return clasts, contours


def _tile_rejection(tile, dark_thr, bright_thr, nodata_max_frac):
    """Why a tile is left out before detection: 'uniform' (fewer than three
    distinct values), 'dark' or 'bright' (more than ``nodata_max_frac`` of
    its pixels beyond the thresholds; the larger share names it), else
    None. The run's log then says how many tiles were left out and why, so
    a filter that drops every tile does not finish in silence."""
    if np.shape(np.unique(tile))[0] < 3:
        return 'uniform'
    if dark_thr is not None or bright_thr is not None:
        mean_rgb = tile.mean(axis=2)
        dark = (mean_rgb < dark_thr) if dark_thr is not None \
            else np.zeros(mean_rgb.shape, dtype=bool)
        bright = (mean_rgb > bright_thr) if bright_thr is not None \
            else np.zeros(mean_rgb.shape, dtype=bool)
        if float((dark | bright).mean()) > float(nodata_max_frac):
            return 'dark' if int(dark.sum()) >= int(bright.sum()) else 'bright'
    return None


def _detect_ortho(model, imgpath, metric_cropsize, plot, saveplot, saveresults,
                kstart, ksaveint, overlap, dedup_method, dedup_overlap,
                stop_check=None,
                output_dir=None,
                figures_dir=None,
                out_stem=None,
                # Brightness / nodata tile filter, applied before the model.
                # Pixels with mean RGB below dark_threshold (None/0 disables)
                # or above bright_threshold (None/255 disables) are nodata; a
                # tile whose nodata fraction exceeds nodata_max_frac is dropped.
                dark_threshold=None,
                bright_threshold=None,
                nodata_max_frac=0.95,
                # GeoJSON of Polygon/MultiPolygon ROIs in world coordinates;
                # tiles whose centre falls outside every polygon are skipped.
                roi_path=None,
                # Deprecated, ignored (ksaveint too): the .run.csv is the
                # single resume source.
                save_checkpoints=None,
                checkpoint_dir=None,
                clean_checkpoints_on_completion=None):
    """Detect and measure clasts in a single georeferenced UAV ortho-image.

    A streaming ``.run.csv`` next to the final CSV records each tile as it is
    processed; a stopped or crashed run resumes from ``last_tile + 1`` and a
    clean completion removes the file. ``stop_check`` is polled between
    tiles; ``output_dir`` defaults to the image's own directory.
    """
    class_names = ['BG', 'Clast']
    start_time = datetime.now()
    image, geotransform = _read_ortho_for_detection(imgpath)

    shpimg = np.shape(image)
    resolution = np.abs(geotransform[1])
    image_x_corner = geotransform[0]
    image_y_corner = geotransform[3]

    cropsize = int(metric_cropsize / resolution)
    if not (0.0 <= overlap < 0.95):
        raise ValueError(f"overlap must be in [0, 0.95), got {overlap}")
    stride = max(1, int(cropsize * (1.0 - overlap)))
    n_tiles_y = max(1, int(np.ceil((shpimg[0] - cropsize) / stride)) + 1)
    n_tiles_x = max(1, int(np.ceil((shpimg[1] - cropsize) / stride)) + 1)
    n_crops = n_tiles_y * n_tiles_x

    img_stem = out_stem or os.path.splitext(os.path.basename(imgpath))[0]
    streaming_path = _run_state_path_for(
        output_dir, imgpath, img_stem, metric_cropsize)

    from functions import clast_geometry as _CG
    contours_run_path = _run_contours_path_for(streaming_path)
    to_world = _CG.world_frame((image_x_corner, resolution, 0.0,
                                image_y_corner, 0.0, -resolution))
    world_dec = _CG.world_decimals(resolution)
    contours = {}

    # Resume from the streaming .run.csv.
    records = []
    effective_kstart = kstart
    if not os.path.exists(streaming_path) and os.path.exists(contours_run_path):
        # Outlines of a run whose CSV is gone belong to nobody.
        try:
            os.remove(contours_run_path)
        except OSError:
            pass
    _why = discard_mismatched_checkpoint(
        streaming_path, contours_run_path, overlap=overlap,
        cropsize_px=cropsize, stride_px=stride)
    if _why:
        _log.warning(
            "Checkpoint %s ignored: %s. Its rows would have landed on other "
            "places; it was removed and the run starts from tile %d.",
            os.path.basename(streaming_path), _why, kstart)
    if os.path.exists(streaming_path):
        try:
            prev = pd.read_csv(streaming_path)
            if '_tile_idx' in prev.columns and len(prev) > 0:
                last_tile = int(prev['_tile_idx'].max())
                # Empty-tile markers have NaN clast_ID.
                core = prev[prev['_tile_idx'] <= last_tile]
                core = core[core['clast_ID'].notna()]
                core = core.drop(columns=['_tile_idx'])
                records = core.to_dict('records')
                effective_kstart = max(kstart, last_tile + 1)
                _log.info(
                    "Resuming: loaded %d clasts from %s "
                    "(last tile=%d), starting at k=%d.",
                    len(records), os.path.basename(streaming_path),
                    last_tile, effective_kstart)
        except Exception as e:
            _log.warning(
                "Resume file %s could not be parsed (%s). "
                "Attempting line-by-line salvage before starting from scratch.",
                streaming_path, e)
            # Salvage: only fully-written lines (ending with \n), so a
            # truncated last row is excluded.
            try:
                import io as _io
                with open(streaming_path, "r",
                          encoding="utf-8", errors="replace") as _fh:
                    _lines = [ln for ln in _fh if ln.endswith("\n")]
                if len(_lines) > 1:
                    _salvage_df = pd.read_csv(_io.StringIO("".join(_lines)))
                    if ('_tile_idx' in _salvage_df.columns
                            and len(_salvage_df) > 0):
                        last_tile = int(_salvage_df['_tile_idx'].max())
                        _core = (_salvage_df[_salvage_df['clast_ID'].notna()]
                                 .drop(columns=['_tile_idx']))
                        records = _core.to_dict('records')
                        effective_kstart = max(kstart, last_tile + 1)
                        _log.info(
                            "Salvaged %d clasts from corrupted resume file. "
                            "Resuming at k=%d.",
                            len(records), effective_kstart)
            except Exception as _salvage_ex:
                _log.warning(
                    "Salvage also failed (%s). Starting from scratch.",
                    _salvage_ex)
                records = []
                effective_kstart = kstart

    # The outlines of the tiles already done come back with their rows;
    # anything written for a later tile (a crash between the two files) is
    # dropped with the file's rewrite below.
    if records and effective_kstart > 0:
        contours = _read_run_contours(contours_run_path, effective_kstart - 1)
        _kept = {int(float(r['clast_ID'])) for r in records
                 if r.get('clast_ID') == r.get('clast_ID')}
        contours = {k: v for k, v in contours.items() if k in _kept}
        if len(contours) < len(records):
            _log.info("Resume: %d of %d earlier clasts have an outline.",
                      len(contours), len(records))
    try:
        import json as _json
        # Compacted to one line, through a temporary file: a crash here
        # leaves the previous file whole.
        _tmp = contours_run_path + '.tmp'
        with open(_tmp, 'w', encoding='utf-8') as _fh:
            if contours:
                _fh.write(_json.dumps({"tile": effective_kstart - 1,
                                       "contours": {str(k): v for k, v
                                                    in contours.items()}},
                                      separators=(',', ':')) + '\n')
        os.replace(_tmp, contours_run_path)
        contours_run_ok = True
    except OSError as _cex:
        _log.warning("Outlines will not be kept for this run (%s).", _cex)
        contours_run_ok = False

    # Tile selection: skip uniform tiles, nodata-dominated tiles (too dark:
    # vignette, transparent edges; too bright: sky) and tiles outside the ROI.
    dark_thr = (None if dark_threshold in (None, 0)
                else float(dark_threshold))
    bright_thr = (None if bright_threshold in (None, 255)
                  else float(bright_threshold))

    roi_paths = _load_roi_paths(roi_path)
    if roi_paths:
        _log.info(
            "ROI: loaded %d polygon(s) from %s; tiles outside every polygon "
            "will be skipped.", len(roi_paths), os.path.basename(roi_path))
    elif roi_path:
        _log.warning(
            "ROI file %s yielded no usable polygons; no tiles filtered.",
            os.path.basename(roi_path))

    work_tiles_mat = np.zeros([n_tiles_y, n_tiles_x])
    left_out = {'uniform': 0, 'dark': 0, 'bright': 0, 'roi': 0}
    for n in range(n_tiles_y):
        for m in range(n_tiles_x):
            tile = image[n*stride:n*stride+cropsize, m*stride:m*stride+cropsize, :]
            why = _tile_rejection(tile, dark_thr, bright_thr, nodata_max_frac)
            if why:
                left_out[why] += 1
                continue
            # ROI check on the tile's world-space centre (full affine, so
            # rotated geotransforms are handled too).
            if roi_paths:
                cx_px = m * stride + cropsize / 2.0
                cy_px = n * stride + cropsize / 2.0
                wx = (geotransform[0]
                      + cx_px * geotransform[1]
                      + cy_px * geotransform[2])
                wy = (geotransform[3]
                      + cx_px * geotransform[4]
                      + cy_px * geotransform[5])
                inside = any(rp.contains_point((wx, wy))
                             for rp in roi_paths)
                if not inside:
                    left_out['roi'] += 1
                    continue
            work_tiles_mat[n, m] = 1
    work_tiles_nm = op.matrix2xyz(work_tiles_mat)
    work_tiles_nm = work_tiles_nm[work_tiles_nm[:, 2] == 1, :]

    n_work = np.shape(work_tiles_nm)[0]
    if any(left_out.values()):
        (_log.warning if n_work == 0 else _log.info)(
            "Tiles: %d of %d to process; left out: %d uniform, %d dark, "
            "%d bright, %d outside the ROI%s.",
            n_work, n_crops, left_out['uniform'], left_out['dark'],
            left_out['bright'], left_out['roi'],
            " — nothing to detect" if n_work == 0 else "")
    if overlap > 0:
        _log.info(
            "Tile overlap: %.0f%% (stride=%dpx, cropsize=%dpx, "
            "%dx%d=%d tiles total).",
            overlap * 100, stride, cropsize, n_tiles_y, n_tiles_x, n_crops)

    stopped_at = None     # set to k if user requested stop
    raised_error = None   # set to the exception if the loop crashes
    run_start = datetime.now()
    # Explicit __enter__/__exit__ avoids re-indenting the loop; the save block
    # at the bottom runs unconditionally, even on stop or exception.
    streaming_writer = _StreamingResultsWriter(streaming_path, _STREAMING_COLUMNS)
    if streaming_writer._is_new:
        write_run_grid(streaming_path, overlap=overlap, cropsize_px=cropsize,
                       stride_px=stride, n_tiles_x=n_tiles_x,
                       n_tiles_y=n_tiles_y)
    streaming_writer.__enter__()
    try:
     for k in range(effective_kstart, n_work):
        if stop_check is not None and stop_check():
            _log.info(
                "STOP at tile k=%d; %d clasts so far. "
                "Re-run this image to resume from k=%d.",
                k, len(records), k)
            stopped_at = k
            break

        n = int(work_tiles_nm[k, 0])
        m = int(work_tiles_nm[k, 1])

        croppedimage = image[n*stride:n*stride+cropsize, m*stride:m*stride+cropsize, :]

        # A TF session reset mid-run (e.g. "Reload model" clicked) invalidates
        # the graph; rebuild once and retry the tile instead of dying.
        try:
            results = model.detect([croppedimage], verbose=0)
        except Exception as _tile_ex:
            _ex_type = type(_tile_ex).__name__
            if _is_oom_error(_tile_ex):
                _log.error(
                    "GPU out-of-memory at tile k=%d (%s). Attempting "
                    "automatic recovery by reducing the model's per-tile "
                    "detection caps.", k, _ex_type)
                results = None
                while True:
                    _smaller = _rebuild_with_reduced_caps()
                    if _smaller is None:
                        raise RuntimeError(
                            "GPU ran out of memory during detection and "
                            "could not recover even at the smallest model "
                            "size. Options: (1) switch the device to CPU in "
                            "the left sidebar (slower, but not limited by "
                            "video memory); (2) use a SMALLER window size — "
                            "larger windows tile the ortho into bigger, more "
                            "memory-hungry crops; (3) close other GPU "
                            f"applications and re-run. Tile k={k}."
                        ) from _tile_ex
                    model = _smaller
                    try:
                        results = model.detect([croppedimage], verbose=0)
                        _log.info(
                            "Recovered from GPU OOM at tile k=%d "
                            "(now %d instances/tile).",
                            k, int(config.DETECTION_MAX_INSTANCES))
                        break
                    except Exception as _again:
                        if _is_oom_error(_again):
                            continue  # shrink again
                        raise
            elif ("InvalidArgumentError" in _ex_type
                    or "input_image" in str(_tile_ex)):
                _build_args = _MODEL_CACHE.get("build_args")
                if _build_args is not None:
                    _log.warning(
                        "TF session invalidated at tile k=%d (%s). "
                        "Rebuilding model and retrying tile.",
                        k, _ex_type)
                    model = _build_model(*_build_args, use_cache=False)
                    results = model.detect([croppedimage], verbose=0)
                else:
                    raise RuntimeError(
                        "TF session was reset while detection was running "
                        "(did you click 'Reload model' during the job?). "
                        f"Tile k={k}. Re-run this job to resume from k={k}."
                    ) from _tile_ex
            else:
                raise

        r = results[0]

        if plot or saveplot:
            tile_png = None
            if saveplot:
                # Tile PNGs go in a tiles/ subfolder of figures_dir.
                if figures_dir:
                    tiles_dir = os.path.join(figures_dir, "tiles")
                    os.makedirs(tiles_dir, exist_ok=True)
                    tile_png = os.path.join(
                        tiles_dir,
                        f"{img_stem}_tile_k={k}_n={n}_m={m}.png")
                else:
                    tile_png = imgpath[0:-4] + f'_tile_k={k}_n={n}_m={m}.png'
            _tile_figure(croppedimage, r, class_names,
                         f"Tile k={k} (row={n}, col={m})", tile_png, plot)

        n_detected = np.shape(r['masks'])[2]
        pct = 100.0 * (k + 1) / max(n_work, 1)
        remaining = n_work - (k + 1)
        _log.info(
            "Tile %d/%d (%.1f%%, %d remaining) row=%d col=%d: "
            "%d detections, %d total clasts.",
            k + 1, n_work, pct, remaining, int(n), int(m),
            n_detected, len(records))

        if n_detected > 0:
            crop_x = m * stride
            crop_y = -n * stride
            n_records_before_tile = len(records)
            tile_contours = {}
            for i in range(n_detected):
                meas = _measure_clast(r['masks'][:, :, i], r['scores'][i], resolution)
                if meas is None:
                    continue
                try:
                    outline = _CG.contour_from_measurement(
                        meas, to_frame=to_world, offset=(crop_x, n * stride),
                        decimals=world_dec)
                    if outline:
                        tile_contours[len(records) + 1] = outline
                except Exception:
                    pass
                records.append({
                    'clast_ID': len(records) + 1,
                    'x': image_x_corner + (crop_x + meas['center_x']) * resolution,
                    'y': image_y_corner + (crop_y - meas['center_y']) * resolution,
                    'Clast_length': meas['Clast_length'],
                    'Clast_width': meas['Clast_width'],
                    'Ellipse_major_axis': meas['Ellipse_major_axis'],
                    'Ellipse_minor_axis': meas['Ellipse_minor_axis'],
                    'Surface_area': meas['Surface_area'],
                    'Perimeter': meas['Perimeter'],
                    'Equivalent_diameter': meas['Equivalent_diameter'],
                    'Eccentricity': meas['Eccentricity'],
                    'Solidity': meas['Solidity'],
                    'Mean_intensity': _mask_mean_intensity(
                        croppedimage, r['masks'][:, :, i]),
                    'Score': meas['Score'],
                    'Orientation': meas['Orientation'],
                })
            new_in_tile = records[n_records_before_tile:]
            if new_in_tile:
                streaming_writer.write_tile(k, new_in_tile)
            if tile_contours:
                contours.update(tile_contours)
                if contours_run_ok:
                    try:
                        import json as _json
                        with open(contours_run_path, 'a',
                                  encoding='utf-8') as _fh:
                            _fh.write(_json.dumps(
                                {"tile": k, "contours": {
                                    str(kk): vv for kk, vv
                                    in tile_contours.items()}},
                                separators=(',', ':')) + '\n')
                    except OSError as _cex:
                        _log.warning("Outlines of tile %d not kept (%s).",
                                     k, _cex)
        else:
            # Empty tile: a marker row with blank clast_ID, or resume would
            # re-process it. fsync=False: losing the marker is harmless.
            streaming_writer.write_tile(k, [{
                '_tile_idx': k,
                'clast_ID': '',
            }], fsync=False)
    except Exception as _ex:
        raised_error = _ex
        _log.error("EXCEPTION in tile loop: %s", _ex, exc_info=True)
    finally:
        try:
            streaming_writer.__exit__(None, None, None)
        except Exception:
            pass

    run_end = datetime.now()
    this_elapsed_s = max(0.0, (run_end - run_start).total_seconds())
    is_partial = (stopped_at is not None) or (raised_error is not None)

    # Clean completion removes the streaming file; partial runs keep it to resume.
    if not is_partial and os.path.exists(streaming_path):
        try:
            os.remove(streaming_path)
            try:
                os.remove(_run_grid_path_for(streaming_path))
            except OSError:
                pass
        except OSError as e:
            _log.warning("Failed to remove resume file %s: %s.", streaming_path, e)
    if not is_partial and os.path.exists(contours_run_path):
        try:
            os.remove(contours_run_path)
        except OSError:
            pass

    clasts = pd.DataFrame(records, columns=_CLAST_COLUMNS)

    # Dedup only on a clean completion; a partial dedup would drop clasts that
    # later tiles have not met yet.
    if not is_partial:
        clasts, contours = _dedup_on_completion(
            clasts, contours, dedup_method, dedup_overlap, overlap)

    _CG.attach_contours(clasts, contours, frame="world")

    # Always write the final CSV + log, partial or complete. The log is
    # append-mode so successive resumes build a history.
    if saveresults:
        out_csv_dir = output_dir if output_dir else os.path.dirname(imgpath) or "."
        if output_dir:
            os.makedirs(out_csv_dir, exist_ok=True)
        csv_name = naming.detection_csv_name(img_stem, metric_cropsize)
        out_path = os.path.join(out_csv_dir, csv_name)
        # A stopped run must not replace a complete one. It keeps its clasts
        # under a .partial.csv of its own and says so; the complete file, and
        # the checkpoint a resume reads, are left alone.
        if is_partial and _completed_before(out_path):
            partial_path = out_path[:-len('.csv')] + '.partial.csv'
            clasts.to_csv(partial_path, index=False, float_format='%.5f')
            _log.warning(
                "A complete %s is already here, so this stopped run was "
                "written to %s instead. Delete the complete file, or press "
                "the row's reset, to replace it.",
                os.path.basename(out_path), os.path.basename(partial_path))
            out_path = partial_path
        else:
            clasts.to_csv(out_path, index=False, float_format='%.5f')
        _CG.write_contours(out_path, contours, frame="world")

        log_stem = csv_name[:-len('.csv')] if csv_name.endswith('.csv') else csv_name
        log_path = os.path.join(out_csv_dir, log_stem + '_detection_log.txt')
        outcome = ('partial (stopped)' if stopped_at is not None
                   else 'partial (error)' if raised_error is not None
                   else 'complete')

        # Cumulative elapsed across resumes lives at the tail of the log.
        prev_cumulative_s = 0.0
        if os.path.exists(log_path):
            try:
                with open(log_path) as fh:
                    for line in fh:
                        if line.startswith('Cumulative elapsed (s):'):
                            try:
                                prev_cumulative_s = float(line.split(':', 1)[1].strip())
                            except ValueError:
                                pass
            except OSError:
                pass
        cumulative_s = prev_cumulative_s + this_elapsed_s

        with open(log_path, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"[Run {run_end.isoformat(timespec='seconds')}]\n")
            f.write(f"  outcome             : {outcome}\n")
            f.write(f"  this run start      : {run_start.isoformat(timespec='seconds')}\n")
            f.write(f"  this run elapsed (s): {this_elapsed_s:.2f}\n")
            f.write(f"  kstart this run     : {kstart}\n")
            if stopped_at is not None:
                f.write(f"  stopped at tile k   : {stopped_at}\n")
            if raised_error is not None:
                f.write(f"  error               : {type(raised_error).__name__}: {raised_error}\n")
            f.write(f"\n[Inputs]\n")
            f.write(f"  image path          : {imgpath}\n")
            f.write(f"  image dimensions    : {shpimg[0]} x {shpimg[1]} x {shpimg[2]}\n")
            f.write(f"  GeoTransform pixel  : {resolution} m/px\n")
            f.write(f"  image_x_corner      : {image_x_corner}\n")
            f.write(f"  image_y_corner      : {image_y_corner}\n")
            f.write(f"\n[Parameters]\n")
            f.write(f"  metric_cropsize     : {metric_cropsize} m\n")
            f.write(f"  cropsize            : {cropsize} px\n")
            f.write(f"  overlap             : {overlap}\n")
            f.write(f"  stride              : {stride} px\n")
            f.write(f"  tile grid           : {n_tiles_y} rows x {n_tiles_x} cols = {n_crops} tiles\n")
            f.write(f"  tiles to process    : {n_work} (after uniform-tile filter)\n")
            f.write(f"  dedup_method        : {dedup_method}\n")
            f.write(f"  dedup_overlap       : {dedup_overlap}\n")
            f.write(f"  output_dir          : {output_dir}\n")
            f.write(f"  figures_dir         : {figures_dir}\n")
            f.write(f"  run_state_file      : {streaming_path}\n")
            f.write(f"\n[Outputs]\n")
            f.write(f"  output CSV          : {out_path}\n")
            f.write(f"  records in CSV      : {len(clasts)}\n")
            f.write(f"  saveplot            : {saveplot}\n")
            f.write(f"\nCumulative elapsed (s): {cumulative_s:.2f}\n")
        _log.info("Detection complete: %d clasts written to %s.", len(clasts), out_path)
    # Machine-readable crash manifest, best-effort; must not suppress the
    # re-raise below.
    if raised_error is not None:
        try:
            import json as _json
            try:
                from functions import __version__ as _ver
            except Exception:
                _ver = "dev"
            # `k` may be unbound if the loop never ran.
            _last_k = locals().get('k', effective_kstart - 1)
            _manifest = {
                "tool_version":          str(_ver),
                "image":                 os.path.basename(imgpath),
                "cropsize_m":            float(metric_cropsize),
                "total_tiles":           int(n_work),
                "last_attempted_tile":   (int(_last_k)
                                          if _last_k is not None else None),
                "error_type":            type(raised_error).__name__,
                "error_message":         str(raised_error),
                "oom":                   bool(_is_oom_error(raised_error)
                                              or "ran out of memory"
                                              in str(raised_error).lower()),
                "timestamp":             datetime.now().isoformat(),
                "clasts_recovered":      len(records),
                "resume_csv":            os.path.basename(streaming_path),
            }
            # Strip only the trailing ".run.csv"; str.replace could hit a parent dir.
            _sp = streaming_path
            _manifest_path = (
                _sp[: -len(".run.csv")] + ".crash.json"
                if _sp.endswith(".run.csv")
                else _sp + ".crash.json"
            )
            with open(_manifest_path, "w", encoding="utf-8") as _mf:
                _json.dump(_manifest, _mf, indent=2)
            _log.error("Crash manifest written to %s", _manifest_path)
        except OSError as _mf_err:
            _log.warning("Could not write crash manifest: %s", _mf_err)

    # Re-raise so the caller can route to 'error'; the CSV + log are on disk.
    if raised_error is not None:
        raise raised_error
    return clasts


def clasts_detect_jobs(mode, jobs, resolution=0.001, metric_cropsize=1,
                       plot=True, saveplot=False, saveresults=False,
                       devicemode="gpu", devicenumber=0, ksaveint=1,
                       min_confidence=None, overlap=0.0, dedup_method='iou',
                       dedup_overlap=None, stop_check=None,
                       output_dir=None, figures_dir=None,
                       progress_callback=None,
                       # Tile filter and ROI, ortho path only (see _detect_ortho).
                       dark_threshold=None,
                       bright_threshold=None,
                       nodata_max_frac=0.95,
                       roi_path=None,
                       # Deprecated, ignored.
                       save_checkpoints=None, checkpoint_dir=None,
                       clean_checkpoints_on_completion=None):
    """Detect clasts on a list of ``{"path", "kstart", ...}`` jobs, each
    resuming from its own kstart. ``mode`` is ``modes.ORTHO`` or
    ``modes.QUADRAT``; the older spellings (any case) are accepted and
    normalised at entry;
    ``progress_callback(job_index, event, payload)`` fires "start", "done",
    "stopped" or "error" at job boundaries. Returns one DataFrame per job in
    input order (empty for failed jobs)."""
    mode = modes.normalise_mode(mode)
    model = _build_model(devicemode, devicenumber, min_confidence)

    results_per_image = []
    for ji, job in enumerate(jobs):
        if stop_check is not None and stop_check():
            _log.info("STOP: halting before job %d/%d.", ji + 1, len(jobs))
            if progress_callback:
                try:
                    progress_callback(ji, "stopped", {})
                except Exception:
                    pass
            results_per_image.append(pd.DataFrame())
            continue

        path = job["path"]
        kstart = int(job.get("kstart", 0))
        if progress_callback:
            try:
                progress_callback(ji, "start", {"path": path})
            except Exception:
                pass

        try:
            if mode == modes.QUADRAT:
                clasts = _detect_quadrat(model, path, resolution,
                                         plot, saveplot, saveresults,
                                         output_dir=output_dir,
                                         figures_dir=figures_dir,
                                         roi_path=job.get("roi_path",
                                                          roi_path),
                                         out_stem=job.get("out_stem"))
            elif mode == modes.ORTHO:
                job_roi = job.get("roi_path", roi_path)
                clasts = _detect_ortho(model, path, metric_cropsize,
                                     plot, saveplot, saveresults,
                                     kstart, ksaveint, overlap,
                                     dedup_method, dedup_overlap,
                                     stop_check=stop_check,
                                     output_dir=output_dir,
                                     figures_dir=figures_dir,
                                     out_stem=job.get("out_stem"),
                                     dark_threshold=dark_threshold,
                                     bright_threshold=bright_threshold,
                                     nodata_max_frac=nodata_max_frac,
                                     roi_path=job_roi)
            else:
                raise ValueError(f"mode must be {modes.ORTHO!r} or {modes.QUADRAT!r}, got {mode!r}")

            results_per_image.append(clasts)

            # A stop mid-job is reported as "stopped", not "done" — for a
            # tiled ortho, whose CSV is then partial. A quadrat photograph
            # is one detection: a stop asked during it takes effect after
            # it, and its complete result is "done".
            if (mode == modes.ORTHO and stop_check is not None
                    and stop_check()):
                if progress_callback:
                    try:
                        progress_callback(ji, "stopped", {})
                    except Exception:
                        pass
            else:
                if progress_callback:
                    try:
                        progress_callback(ji, "done",
                                          {"n_clasts": len(clasts) if clasts is not None else 0})
                    except Exception:
                        pass
        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            _log.error("Failed to process %r: %s", path, e, exc_info=True)
            # One actionable sentence for the queue row; the full text is in the log.
            msg = str(e)
            if _is_oom_error(e):
                msg = ("GPU ran out of memory — switch the sidebar to CPU "
                       "and re-run, or restart the app to free GPU memory.")
            if progress_callback:
                try:
                    progress_callback(ji, "error",
                                      {"message": msg, "traceback": tb})
                except Exception:
                    pass
            results_per_image.append(pd.DataFrame())

    return results_per_image
