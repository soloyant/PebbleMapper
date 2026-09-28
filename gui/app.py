"""NiceGUI front-end for the PebbleMapper (clast grain-size mapping) package.

Run from the repo root with `python -m gui.app` (or launch_gui.bat /
launch_gui.sh). Wraps the pipeline functions (detection, rasterize, merge)
plus tabs for orthorectification, manual digitisation, publication maps and
validation against ground truth. State lives in a single module-level
AppState dataclass; heavy TF/Mask R-CNN imports stay inside button handlers
so the GUI boots quickly.
"""
from __future__ import annotations
import os
import re
import sys
import threading
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Make `from functions import ...` work from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the native numeric chain on the main thread: on Windows, importing
# it first from a worker thread can bind a conflicting DLL set (0xc06d007f).
import numpy  # noqa: F401
import scipy  # noqa: F401
import skimage.color  # noqa: F401
import skimage.io  # noqa: F401

from functions import naming
from functions.modes import ORTHO, QUADRAT, MODE_LABELS, normalise_mode

# Lift PIL's ~178 MP decompression-bomb cap: UAV orthos routinely exceed it,
# and the inputs are local files the user picked.
import warnings as _warnings
from PIL import Image as _PILImage_base
_PILImage_base.MAX_IMAGE_PIXELS = None
try:
    _warnings.simplefilter(
        "ignore", _PILImage_base.DecompressionBombWarning)
except Exception:
    pass
# Registers the HEIC/HEIF opener with Pillow at app start, so every
# Image.open below reads an iPhone photograph; also the extension lists.
from functions import images as _images  # noqa: E402


def _job_mode(job: dict) -> str:
    """Canonical mode of a queued job. Queues persisted by an older session
    may carry an older spelling; every reader goes through here."""
    return normalise_mode(job.get("mode"), default=ORTHO)


# Project layout registry (datasets/<project>/...). The active project seeds
# the starting folder of every Browse… dialog.
from functions.layout import (  # noqa: E402
    project_path,
    list_projects,
    ensure_project_layout,
    migrate_validation_structure,
    migrate_input_structure,
    get_datasets_root,
    set_datasets_root,
    list_dates,
    set_active_date,
)
from functions.layout import default_starting_dir as _layout_default_starting_dir  # noqa: E402
from functions import branding as _brand  # noqa: E402
from functions import crashsafe as _crashsafe  # noqa: E402
from functions import zonal_canvas as _zonal_canvas  # noqa: E402
from functions import worker as _worker_mod  # noqa: E402
from functions import queue_store as _queue_store  # noqa: E402

# Queue-restore plumbing. Filled by the __main__ boot block; index() turns a
# staged offer into the restore dialog. Plain imports see it resolved and inert.
_qrestore = {"pending": {}, "resolved": True, "watcher": None}


def _ensure_queue_watcher():
    """Start the write-through persistence watcher exactly once."""
    if _qrestore["watcher"] is None:
        try:
            from detectors.base import TOOL_VERSION as _tv
        except Exception:
            _tv = ""
        w = _queue_store.QueueWatcher(state, app_version=str(_tv))
        w.start()
        _qrestore["watcher"] = w
        print("[queue] persistence watcher started "
              f"(dir: {_queue_store.queues_dir()})", flush=True)


def _project_addons() -> list:
    """The custom-equation add-ons in force, lowest precedence first: the
    shipped hydraulic indices, the active project's addons.json, then the
    installation's user_addons.json."""
    try:
        from functions.addons import load_addons as _load_addons
        paths = [REPO_ROOT / "hydraulic_addons.json"]
        if state.current_project:
            paths.append(project_path(state.current_project) / "addons.json")
        paths.append(REPO_ROOT / "user_addons.json")
        return _load_addons(*paths)
    except Exception as ex:
        print(f"[addons] load skipped: {ex}")
        return []


def parse_gsd_from_filename(filename: str):
    """Delegate to ``functions.gsd`` (the sidecar-aware form is ``effective_gsd``)."""
    from functions.gsd import parse_gsd_from_filename as _p
    return _p(filename)


def default_starting_dir(kind: str) -> str:
    """Seed directory for Browse... dialogs, from the active project.

    ``kind`` is any ``functions.layout.PATH_KINDS`` value or '' (project root).
    """
    proj = state.current_project if hasattr(state, "current_project") else ""
    return _layout_default_starting_dir(kind, proj)


def _seed_default(el, value, *, force=False):
    """Fill a path field from the active project and mark it as a default.

    ``force=True`` (project/date switch) overwrites; otherwise a filled field
    is left alone so a user edit is never clobbered by a rescan. The hint
    clears as soon as the user changes the value.
    """
    if value is None:
        return
    cur = (getattr(el, "value", "") or "").strip()
    if cur and not force:
        return
    sval = str(value)
    el.value = sval
    el._seed_val = sval
    try:
        el.props('hint="from the active project — edit or Browse to change"')
    except Exception:
        pass
    if not getattr(el, "_seed_wired", False):
        el._seed_wired = True

        def _maybe_clear(_=None):
            if str(getattr(el, "value", "") or "") != getattr(
                    el, "_seed_val", ""):
                try:
                    el.props(remove="hint")
                except Exception:
                    pass
        el.on_value_change(_maybe_clear)


def zonal_auto_out_name(base_stem: str, vec_stem: str, field_name: str,
                        mode: str, id_field: str = "") -> str:
    """Auto-generated filename for a Zonal-tab result CSV.

    The field must appear in the name: in polygon mode ``base_stem`` is the
    same clast CSV for every field, so two fields would otherwise share a
    path. So must the ID field, which chooses how the zones are labelled.
    """
    return naming.zonal_out_name(base_stem, vec_stem, field_name, mode, id_field)


def field_from_raster(raster_path) -> str:
    """Recover the quantity a statistic raster represents.

    Prefers the sidecar JSON written by the rasterize step, then a known field
    name inside the filename, then the bare stem so the answer is never empty.
    """
    p = Path(raster_path)
    try:
        import json as _json
        side = Path(str(p) + ".json")
        if side.exists():
            meta = _json.loads(side.read_text(encoding="utf-8"))
            f = str(meta.get("field") or "").strip()
            if f:
                return f
    except Exception:
        pass
    stem = re.sub(r"_cellsize=[0-9.]+m.*$", "", p.name, flags=re.IGNORECASE)
    stem = re.sub(r"\.(tif|tiff)$", "", stem, flags=re.IGNORECASE)
    try:
        from functions.units import FIELD_UNIT_TABLE
        low = stem.lower()
        best = ""
        for key, _val in FIELD_UNIT_TABLE:
            if key and key in low and len(key) > len(best):
                best = key
        if best:
            i = low.rfind(best)
            return stem[i:i + len(best)]
    except Exception:
        pass
    return stem or p.stem


def zonal_shape_for_mode(mode: str, current_shape: str) -> str:
    """Draw-shape implied by an analysis mode.

    Transect mode only draws transects; leaving it must drop a stale
    "transect" shape or the next commit is rejected for having < 3 points.
    """
    if mode == "transects":
        return "transect"
    if current_shape == "transect":
        return "polygon"
    return current_shape


DRAW_SHAPES = ("polygon", "rectangle", "circle", "transect")


def normalise_draw_shape(shape) -> str:
    """Coerce a draw-shape to one the canvas can commit.

    A Quasar toggle handed a value it has no option for emits null back, which
    sets the bound shape to None; that must never reach the ``min_pts`` lookup.
    """
    return shape if shape in DRAW_SHAPES else "polygon"


def zonal_target_out_path(last_written: str, set_name: str) -> str:
    """The file the zones canvas last wrote counts as the target only while
    the set name still names it. After another layer is imported or the
    name is cleared or changed, the name alone decides (empty: the
    component prompts), so a drawing cannot land in the previous target or
    in an imported layer."""
    if not last_written:
        return ""
    name = sanitise_zone_set_name(set_name)
    if not name or Path(last_written).stem != name:
        return ""
    return last_written


def zonal_set_save_path(explicit_out: str, set_name: str, project: str,
                        source_image: str):
    """Path a zone set is written to, or None when it has no name yet.

    An explicit output path wins; otherwise the set name under the project's
    ``geometries`` kind; with no project, next to the source image. The caller
    creates the directory.
    """
    if explicit_out:
        return Path(explicit_out)
    name = sanitise_zone_set_name(set_name)
    if not name:
        return None
    if project:
        out_dir = project_path(project, "geometries")
    else:
        out_dir = Path(source_image or "").parent or Path.cwd()
    return Path(out_dir) / f"{name}.geojson"


# Summary-CSV columns expressed in the summarised field's own unit; every
# other column (count, area_m2, cv, *_phi, dem_*, ...) has a unit of its own.
_ZONAL_FIELD_UNIT_COLUMNS = frozenset({
    "mean", "std", "median", "iqr", "min", "max", "range"})
_ZONAL_PERCENTILE_RE = re.compile(r"^[DP][0-9]{1,3}$")

# Provenance columns, excluded from plot-axis choices (an empty `unit` column
# reads back from pandas as float64/NaN and would otherwise look numeric).
ZONAL_PROVENANCE_COLUMNS = ("field", "unit", "coverage")

# Values that mean "no unit" (an all-empty column reads back as "nan").
_ZONAL_NON_UNITS = frozenset({"", "nan", "none"})


def zonal_column_unit(column: str, csv_unit) -> str:
    """Unit for a zonal summary column, or '' when it carries none.

    ``csv_unit`` is the CSV's own `unit` column; '' or NaN leaves the axis
    unlabelled rather than reading "mean [nan]".
    """
    unit = "" if csv_unit is None else str(csv_unit).strip()
    if unit.lower() in _ZONAL_NON_UNITS:
        return ""
    if (column in _ZONAL_FIELD_UNIT_COLUMNS
            or _ZONAL_PERCENTILE_RE.match(column or "")):
        return unit
    return ""


def zonal_axis_label(column: str, unit: str) -> str:
    """Axis label for a zonal plot: bare column name when there is no unit."""
    return f"{column} [{unit}]" if unit else str(column)


_ZONE_SET_NAME_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z ._-]*$")


def sanitise_zone_set_name(name: str) -> str:
    """Return ``name`` if it is usable as a zone-set filename stem, else ''.

    Rejects rather than repairs; the stem is joined onto the project
    directory, so separators and parent references must never survive.
    """
    n = (name or "").strip()
    if not n:
        return ""
    if any(c in n for c in ("/", "\\", "\x00")):
        return ""
    if ".." in n:
        return ""
    if not _ZONE_SET_NAME_RE.match(n):
        return ""
    return n


def features_world_bbox(features) -> "tuple | None":
    """(xmin, xmax, ymin, ymax) of staged features in world coordinates.

    Reads pt['x'] / pt['y'] (the values persisted to GeoJSON). None when
    nothing carries world coordinates.
    """
    xs, ys = [], []
    for feat in features or []:
        for pt in feat.get("pts", []):
            if "x" in pt and "y" in pt:
                try:
                    xs.append(float(pt["x"]))
                    ys.append(float(pt["y"]))
                except (TypeError, ValueError):
                    continue
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def raster_world_bbox(gt, width, height) -> "tuple | None":
    """(xmin, xmax, ymin, ymax) covered by a raster with GeoTransform ``gt``."""
    if not gt or not width or not height:
        return None
    try:
        x0, x1 = gt[0], gt[0] + float(width) * gt[1]
        y0, y1 = gt[3] + float(height) * gt[5], gt[3]
    except (TypeError, IndexError, ValueError):
        return None
    return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def features_match_image(feat_bbox, img_bbox) -> bool:
    """True when staged geometry plausibly belongs to the source image.

    Permissive: an unknown box on either side returns True; only two known,
    non-overlapping boxes count as a mismatch.
    """
    if feat_bbox is None or img_bbox is None:
        return True
    fx0, fx1, fy0, fy1 = feat_bbox
    ix0, ix1, iy0, iy1 = img_bbox
    return fx0 <= ix1 and fx1 >= ix0 and fy0 <= iy1 and fy1 >= iy0


def _job_error_text(outcome) -> str:
    """One actionable line from a failed worker job."""
    msg = outcome.error or "the job failed"
    if getattr(outcome, "died", False) and getattr(outcome, "postmortem_path", None):
        msg += f" Post-mortem recorded: {outcome.postmortem_path}"
    elif getattr(outcome, "log_path", None):
        msg += f" Full detail: {outcome.log_path}"
    return msg


def prepare_browser_image(src_path: str, *,
                           current_project: str = "",
                           max_dim: int = 4000,
                           cache_subdir: str = "canvas_cache") -> tuple[str, float]:
    """Return ``(png_path, disp_scale)`` suitable for ui.interactive_image.

    Delegates to :mod:`functions.zonal_canvas` so the transcode can also run
    in a worker subprocess.
    """
    return _zonal_canvas.prepare_browser_image(
        src_path, current_project=current_project,
        max_dim=max_dim, cache_subdir=cache_subdir)

from nicegui import ui, app, context  # noqa: E402

from gui.components.job_queue import render_queue  # noqa: E402
from gui.components.log_console import build_log_console  # noqa: E402,F401
from gui.components.progress import build_progress  # noqa: E402,F401
from gui.components.styles import (  # noqa: E402,F401
    PANEL_BG, PANEL_BORDER, PANEL_TEXT, PANEL_MUTED, PANEL_MUTED_2)


# --- Affine helpers shared by every canvas ---
def _ctx_image_to_world(disp_scale: float, gt,
                        px: float, py: float) -> tuple[float, float]:
    """Display pixel → world coordinate (GDAL GeoTransform convention).

    Accounts for the cached PNG's downsample factor. With ``gt`` None the
    result is the original-image pixel coordinate."""
    orig_px = float(px) * float(disp_scale)
    orig_py = float(py) * float(disp_scale)
    if gt is None:
        return orig_px, orig_py
    x = gt[0] + orig_px * gt[1] + orig_py * gt[2]
    y = gt[3] + orig_px * gt[4] + orig_py * gt[5]
    return float(x), float(y)


def _ctx_world_to_image(disp_scale: float, gt,
                        wx: float, wy: float) -> tuple[float, float]:
    """Inverse of ``_ctx_image_to_world``: world coordinate → display pixel."""
    s = float(disp_scale) or 1.0
    if gt is None:
        return float(wx) / s, float(wy) / s
    g0, g1, g2, g3, g4, g5 = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
    det = g1 * g5 - g2 * g4
    if det == 0:
        return 0.0, 0.0
    dx = wx - g0
    dy = wy - g3
    orig_col = (g5 * dx - g2 * dy) / det
    orig_row = (-g4 * dx + g1 * dy) / det
    return float(orig_col) / s, float(orig_row) / s


# --- Per-canvas state ---
@dataclass
class CanvasContext:
    """State for one image-feature canvas instance (each tab owns its own)."""
    image: str = ""
    # Committed features: {"shape", "pts": [{px, py, x, y}, ...], "id",
    # "source": "drawn" | "imported"}. World coords "x"/"y" are authoritative;
    # display pixels "px"/"py" are recomputed on every canvas rebuild.
    features: list = field(default_factory=list)
    # In-progress (un-finalised) click vertices.
    pts: list = field(default_factory=list)
    shape: str = "polygon"
    canvas_mode: str = "draw"  # "draw" | "modify"
    # CSS width-based zoom; 1.0 = one cached-PNG pixel per display pixel.
    zoom: float = 1.0
    selected_idx: int = -1  # -1 = no selection
    # Autosave target; empty = derive from the image path and save_dir_kind.
    out_path: str = ""
    # The saved GeoJSON path downstream consumers read back.
    vector: str = ""
    # Numeric Modify-panel staging values.
    xform_rotate_deg: float = 0.0
    xform_dx: float = 0.0
    xform_dy: float = 0.0
    xform_scale: float = 1.0


# --- Shared image-feature canvas builder ---
def build_image_feature_canvas(
    ctx: CanvasContext,
    *,
    title: str = "Define features",
    description: str = "",
    allowed_shapes: tuple = ("polygon", "rectangle",
                              "circle", "transect"),
    current_project_getter=lambda: "",
    image_picker_default_kind: str = "images",
    vector_picker_default_kind: str = "vectors",
    save_dir_resolver=None,
    save_stem_template: str = "features_{image_stem}.geojson",
    on_save=None,
    tab_active_check=lambda: True,
    save_path_resolver=None,
    on_unnamed_save=None,
    write_provenance: bool = False,
    disjoint_policy: str = "block",
    confirm_overwrite: bool = False,
    show_shape_toggle: bool = True,
    overlay_select_recolor: bool = False,
    import_dedup: bool = False,
    effective_shape_getter=None,
    toolbar_extra=None,
    show_vector_row: bool = True,
    show_image_row: bool = True,
    two_click_transect: bool = False,
    feature_style: str = "plain",
    feature_label=None,
    status_text=None,
) -> dict:
    """Render an image+features canvas card and return widget handles.

    Pick an image, import / draw / modify / delete features, save as GeoJSON,
    autosave on every mutation. Shared by the Zonal, Detection-ROI and
    Digitize workflows.

    Parameters
    ----------
    ctx
        Per-instance CanvasContext.
    allowed_shapes
        Subset of {polygon, rectangle, circle, transect} to expose.
    image_picker_default_kind, vector_picker_default_kind
        ``functions.layout.PATH_KINDS`` values seeding the file pickers.
    save_dir_resolver, save_stem_template
        Where the autosaved GeoJSON lands (default: next to the source
        image) and its filename; ``{image_stem}`` is the only placeholder.
    on_save
        ``(saved_path: Path) -> None`` called after every write.
    tab_active_check
        True when the host tab is active; scopes the keyboard shortcuts.
    save_path_resolver, on_unnamed_save, write_provenance,
    disjoint_policy, confirm_overwrite
        Save-policy hooks. ``save_path_resolver`` returning None fires
        ``on_unnamed_save``; ``write_provenance`` adds per-feature names and
        a source-extent block; ``disjoint_policy`` is "block" or "warn".
    show_shape_toggle, overlay_select_recolor, import_dedup
        ``show_shape_toggle=False`` lets the host drive ``ctx.shape``;
        ``import_dedup`` makes auto-import idempotent and recovers names.
    show_vector_row, show_image_row, two_click_transect
        ``show_vector_row=False`` hides the vector-layer import row (a host
        whose features never come from a file); ``show_image_row=False``
        hides the source-image path row (a host with its own image
        picker); ``two_click_transect=True``
        commits a transect on its second click, like rectangle and circle
        (a segment tool, no Finalise).
    feature_style, feature_label, status_text
        ``feature_style="bold"`` draws for a photograph at fit zoom: the
        in-progress vertex as a 7 px filled circle with a white halo, a
        rubber band from it to the cursor, and every committed transect as
        a 4 px line over a white halo with filled endpoints and a
        white-backed label at its midpoint (sizes in screen pixels; the
        default ``"plain"`` is the thin overlay Zonal, Detect and Digitize
        use). ``feature_label(feature, index) -> str`` names the label;
        ``status_text() -> str | None`` replaces the status line under
        the toolbar when it returns text.

    Returns
    -------
    dict
        ``{"rebuild_canvas", "refresh_overlay", "refresh_features_table",
        "refresh_status", "zoom_fit"}`` callables.
    """
    _state: dict = {
        "interactive": None,
        "gt": None,
        "W": 0, "H": 0,
        "W_orig": 0, "H_orig": 0,
        "disp_scale": 1.0,
        "zoom_wrap": None,
        "drag": None,
        "last_imported_path": "",
        # Cursor position (cached-PNG px) for the bold rubber band.
        "cursor": None, "cursor_t": 0.0,
    }

    with ui.card().classes("w-full"):
        if title:
            ui.label(title).classes("text-subtitle2 text-grey-8")
        if description:
            ui.label(description).classes("text-xs text-grey-7")

        with ui.row().classes("w-full items-end gap-2") as _image_row:
            # A typed or pasted path loads the canvas when the field is
            # left, as Browse does; before, only Browse did and the canvas
            # said "pick a source image above" forever.
            ui.input("Source image",
                      placeholder="path to ortho / raster preview") \
.bind_value(ctx, "image") \
.on("change", lambda e: _rebuild_canvas()) \
.classes("flex-grow")

            def _pick_image():
                p = native_file_picker(
                    title="Select image",
                    filetypes=[
                        ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.heic *.heif"),
                        ("All", "*.*")],
                    initialdir=default_starting_dir(
                        image_picker_default_kind),
                )
                if p:
                    # A new image gets a fresh feature set and its own
                    # expected sidecar, which _rebuild_canvas auto-imports.
                    if p != ctx.image:
                        ctx.features.clear()
                        ctx.pts.clear()
                        ctx.selected_idx = -1
                        _state["last_imported_path"] = ""
                        ctx.image = p
                        try:
                            ctx.vector = str(_resolve_save_path())
                        except Exception:
                            ctx.vector = ""
                    else:
                        ctx.image = p
                    _rebuild_canvas()

            ui.button("Browse…", icon="folder_open",
                       on_click=_pick_image).props("outline")
        if not show_image_row:
            _image_row.set_visibility(False)

        with ui.row().classes("w-full items-end gap-2") as _vector_row:
            ui.input("Vector layer (loaded into canvas)",
                      placeholder="optional shapefile or GeoJSON") \
.bind_value(ctx, "vector") \
.classes("flex-grow") \
.tooltip("Picking a file loads its polygons / "
                         "LineStrings into the canvas. Saving the "
                         "canvas writes back to this field.")

            def _pick_vector():
                p = native_file_picker(
                    title="Select vector layer",
                    filetypes=[
                        ("Vector", "*.shp *.geojson *.json *.gpkg"),
                        ("All", "*.*")],
                    initialdir=default_starting_dir(
                        vector_picker_default_kind),
                )
                if p:
                    ctx.vector = p
                    _auto_import()

            ui.button("Browse…", icon="folder_open",
                       on_click=_pick_vector).props("outline")
            ui.button("Re-import", icon="refresh",
                       on_click=lambda: _auto_import(replace=True)) \
.props("outline") \
.tooltip("Reload from disk; replaces imported "
                         "features, keeps drawn ones.")
        if not show_vector_row:
            _vector_row.set_visibility(False)

        shape_label_map = {
            "polygon":   "Polygon",
            "rectangle": "Rectangle",
            "circle":    "Circle",
            "transect":  "Transect",
        }

        def _zoom_in():
            ctx.zoom = min(8.0, ctx.zoom * 1.5)
            _apply_zoom()

        def _zoom_out():
            ctx.zoom = max(0.125, ctx.zoom / 1.5)
            _apply_zoom()

        def _zoom_reset():
            ctx.zoom = 1.0
            _apply_zoom()

        def _zoom_fit():
            # Whole image inside the 70 % box; the client-reported viewport
            # height (pm_viewport listener), 900 px until reported.
            H = _state.get("H") or 1000
            vh = float(getattr(state, "ortho_viewport_h", 900) or 900)
            target = min(1.0, (0.70 * vh - 8.0) / max(H, 1))
            ctx.zoom = max(0.05, target)
            _apply_zoom()

        # One toolbar (mode, shape, zoom, edit, save) attached to the canvas
        # and sticky while it is in view; `toolbar_extra` lets a host add its
        # own controls to the same bar.
        with ui.column().classes("w-full gap-1 pm-canvas-work"):
            with ui.row().classes("w-full items-center gap-1 flex-wrap "
                                  "pm-sticky-toolbar"):
                if toolbar_extra is not None:
                    toolbar_extra()
                ui.toggle({"draw": "Draw", "modify": "Modify"}) \
                    .bind_value(ctx, "canvas_mode") \
                    .props("dense no-caps") \
                    .tooltip("Draw places new vertices on click. Modify "
                             "lets you select a feature and drag its "
                             "bounding-box handles to translate, scale, "
                             "or rotate.")
                if show_shape_toggle:
                    ui.toggle({s: shape_label_map[s]
                               for s in allowed_shapes
                               if s in shape_label_map}) \
                        .bind_value(ctx, "shape") \
                        .props("dense no-caps") \
                        .tooltip("polygon: click vertices, then Finalise. "
                                 "rectangle: 2 clicks (opposite corners). "
                                 "circle: 2 clicks (centre, rim). "
                                 "transect: click waypoints, then Finalise.")
                ui.separator().props("vertical")
                ui.button(icon="zoom_out", on_click=_zoom_out) \
                    .props("dense flat").tooltip("Zoom out")
                ui.button(icon="home", on_click=_zoom_reset) \
                    .props("dense flat").tooltip("Zoom 1.0×")
                ui.button("Fit", icon="fit_screen", on_click=_zoom_fit) \
                    .props("dense flat no-caps") \
                    .tooltip("The whole image inside the box.")
                ui.button(icon="zoom_in", on_click=_zoom_in) \
                    .props("dense flat").tooltip("Zoom in")
                ui.label().bind_text_from(
                    ctx, "zoom", lambda v: f"{v:.2f}×"
                ).classes("text-xs text-grey-7 min-w-[3rem]")
                ui.separator().props("vertical")
                ui.button("Undo vertex", icon="undo",
                          on_click=lambda: _undo_last_vertex()) \
                    .props("dense outline no-caps")
                ui.button("Finalise", icon="check",
                          on_click=lambda: _finalise_feature()) \
                    .props("dense outline no-caps color=primary") \
                    .tooltip("Commit the in-progress polygon or "
                             "transect. Rectangle and circle auto-"
                             "commit after 2 clicks.")
                ui.button("Clear in-progress", icon="backspace",
                          on_click=lambda: _clear_in_progress()) \
                    .props("dense outline no-caps")
                ui.button("Clear all", icon="delete_sweep",
                          on_click=lambda: _clear_all_features()) \
                    .props("dense outline no-caps color=negative") \
                    .tooltip("Remove every drawn and imported feature "
                             "from the canvas (the file on disk is kept).")
                ui.separator().props("vertical")
                ui.button("Save GeoJSON", icon="save",
                          on_click=lambda: _save_geojson()) \
                    .props("color=primary dense no-caps") \
                    .tooltip("Write every feature (drawn + imported) to "
                             "a GeoJSON file.")

            status_label = ui.label("").classes(
                "text-xs text-grey-7 pm-canvas-status")

            # Canvas box capped at 70 % of the viewport; pan/zoom inside.
            canvas_scroll = ui.element("div").classes("pm-canvas-box") \
                .style("overflow:auto; max-width:100%; max-height:70vh; "
                       "border:1px solid #ccc; background:#fafafa; "
                       "display:inline-block;")
            with canvas_scroll:
                zoom_wrap = ui.element("div").style(
                    "display: inline-block;")
            _state["zoom_wrap"] = zoom_wrap

            features_table = ui.column().classes("w-full mt-2")

        def _image_to_world(px, py):
            return _ctx_image_to_world(
                _state["disp_scale"], _state["gt"], px, py)

        def _world_to_image(wx, wy):
            return _ctx_world_to_image(
                _state["disp_scale"], _state["gt"], wx, wy)

        def _ogr_ring_to_pts(geom_or_ring):
            n = geom_or_ring.GetPointCount()
            out = []
            for i in range(n):
                x, y, *_ = geom_or_ring.GetPoint(i)
                px, py = _world_to_image(float(x), float(y))
                out.append({"px": px, "py": py,
                            "x": float(x), "y": float(y)})
            return out

        def _reproject_features_to_display():
            for feat in ctx.features:
                for pt in feat.get("pts", []):
                    if "x" in pt and "y" in pt:
                        px, py = _world_to_image(pt["x"], pt["y"])
                        pt["px"] = px
                        pt["py"] = py
            for pt in ctx.pts:
                if "x" in pt and "y" in pt:
                    px, py = _world_to_image(pt["x"], pt["y"])
                    pt["px"] = px
                    pt["py"] = py

        def _auto_import(replace=False):
            p = ctx.vector
            if not p or not Path(p).exists():
                return
            if (not replace
                and _state.get("last_imported_path") == p):
                return
            if replace:
                ctx.features[:] = [
                    f for f in ctx.features
                    if f.get("source") != "imported"]
            try:
                from osgeo import ogr
                vds = ogr.Open(p, 0)
                if vds is None:
                    ui.notify(f"OGR could not open {p} — choose another vector file (Import, bar above).",
                               type="negative")
                    return
                lyr = vds.GetLayer(0)
                n_added = 0

                def _read_feat_name(ogr_feat, fallback):
                    # The layer's own name, whether or not this import
                    # de-duplicates: without it every imported zone read
                    # "polygon_1 · imported" and the four quadrats of a zone
                    # set could not be told apart.
                    for key in ("name", "id", "Name", "ID", "NAME"):
                        try:
                            fidx = ogr_feat.GetFieldIndex(key)
                        except Exception:
                            fidx = -1
                        if fidx is not None and fidx >= 0:
                            try:
                                val = ogr_feat.GetField(fidx)
                            except Exception:
                                val = None
                            if val not in (None, ""):
                                return str(val)
                    return fallback

                def _geom_sig(shape, pts):
                    return (shape, tuple(
                        (round(float(pp["x"]), 4), round(float(pp["y"]), 4))
                        for pp in pts if "x" in pp and "y" in pp))

                _existing_sigs = ({
                    _geom_sig(f.get("shape", "polygon"), f.get("pts", []))
                    for f in ctx.features} if import_dedup else set())

                def _stage(shape, pts, fid):
                    if import_dedup:
                        sig = _geom_sig(shape, pts)
                        if sig in _existing_sigs:
                            return False
                        _existing_sigs.add(sig)
                    ctx.features.append({
                        "shape": shape, "pts": pts, "id": fid,
                        "source": "imported"})
                    return True

                for feat in lyr:
                    geom = feat.GetGeometryRef()
                    if geom is None:
                        continue
                    gtype = geom.GetGeometryType()
                    if gtype in (ogr.wkbPolygon, ogr.wkbPolygon25D):
                        ring = geom.GetGeometryRef(0)
                        if ring is None:
                            continue
                        pts = _ogr_ring_to_pts(ring)
                        if len(pts) < 3:
                            continue
                        if _stage("polygon", pts, _read_feat_name(
                                feat, f"imported_{len(ctx.features) + 1}")):
                            n_added += 1
                    elif gtype in (ogr.wkbMultiPolygon,
                                    ogr.wkbMultiPolygon25D):
                        for k in range(geom.GetGeometryCount()):
                            sub = geom.GetGeometryRef(k)
                            if sub is None:
                                continue
                            ring = sub.GetGeometryRef(0)
                            if ring is None:
                                continue
                            pts = _ogr_ring_to_pts(ring)
                            if len(pts) >= 3:
                                if _stage("polygon", pts, _read_feat_name(
                                        feat,
                                        f"imported_{len(ctx.features) + 1}")):
                                    n_added += 1
                    elif gtype in (ogr.wkbLineString,
                                    ogr.wkbLineString25D):
                        pts = _ogr_ring_to_pts(geom)
                        if len(pts) >= 2:
                            if _stage("transect", pts, _read_feat_name(
                                    feat,
                                    f"imported_{len(ctx.features) + 1}")):
                                n_added += 1
                vds = None
                _state["last_imported_path"] = p
                _refresh_overlay()
                _refresh_features_table()
                _refresh_status()
                if n_added:
                    ui.notify(
                        f"Imported {n_added} feature(s) from "
                        f"{Path(p).name}", type="positive")
            except Exception as ex:
                ui.notify(f"Import failed: {ex}. Check the file, then Import again (bar above).",
                           type="negative")

        def _expand_for_overlay(feat):
            shape = feat.get("shape", "polygon")
            pts = feat.get("pts", [])
            if shape == "rectangle" and len(pts) >= 2:
                p0, p1 = pts[0], pts[1]
                x0, y0 = p0["px"], p0["py"]
                x1, y1 = p1["px"], p1["py"]
                return [{"px": x0, "py": y0}, {"px": x1, "py": y0},
                        {"px": x1, "py": y1}, {"px": x0, "py": y1}]
            if shape == "circle" and len(pts) >= 2:
                cx, cy = pts[0]["px"], pts[0]["py"]
                rx, ry = pts[1]["px"], pts[1]["py"]
                r = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
                import math
                return [
                    {"px": cx + r * math.cos(2 * math.pi * i / 64),
                     "py": cy + r * math.sin(2 * math.pi * i / 64)}
                    for i in range(64)
                ]
            return list(pts)

        def _feature_bbox_pixels(feat):
            verts = _expand_for_overlay(feat)
            xs = [v["px"] for v in verts]
            ys = [v["py"] for v in verts]
            if not xs:
                return (0.0, 0.0, 0.0, 0.0)
            return (min(xs), min(ys), max(xs), max(ys))

        def _handle_positions_pixels(feat):
            xmin, ymin, xmax, ymax = _feature_bbox_pixels(feat)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            zoom = max(0.01, float(ctx.zoom))
            W = _state.get("W") or 1
            H = _state.get("H") or 1
            offset = max(12.0,
                          (W * W + H * H) ** 0.5 * 0.005) / zoom
            return {
                "nw":  (xmin, ymin), "n": (cx, ymin),
                "ne":  (xmax, ymin), "e": (xmax, cy),
                "se":  (xmax, ymax), "s": (cx, ymax),
                "sw":  (xmin, ymax), "w": (xmin, cy),
                "rot": (cx, ymin - offset),
            }

        _ANCHOR = {
            "nw": ("xmax", "ymax"), "n": (None, "ymax"),
            "ne": ("xmin", "ymax"), "e": ("xmin", None),
            "se": ("xmin", "ymin"), "s": (None, "ymin"),
            "sw": ("xmax", "ymin"), "w": ("xmax", None),
        }

        def _hit_test_handle(ix, iy, feat):
            zoom = max(0.01, float(ctx.zoom))
            W = _state.get("W") or 1
            H = _state.get("H") or 1
            hit_tol = max(10.0,
                           (W * W + H * H) ** 0.5 * 0.006) / zoom
            # Rotation handle first: on a small bbox the "n" scale handle
            # would otherwise shadow it.
            positions = _handle_positions_pixels(feat)
            rot_xy = positions.get("rot")
            if rot_xy is not None:
                if (((ix - rot_xy[0]) ** 2
                     + (iy - rot_xy[1]) ** 2) ** 0.5 < hit_tol):
                    return "rot"
            for name, (hx, hy) in positions.items():
                if name == "rot":
                    continue
                if ((ix - hx) ** 2 + (iy - hy) ** 2) ** 0.5 < hit_tol:
                    return name
            return None

        def _seg_distance(px, py, ax, ay, bx, by):
            """Shortest distance from point (px,py) to segment a->b."""
            dx, dy = bx - ax, by - ay
            if dx == 0.0 and dy == 0.0:
                return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * dx, ay + t * dy
            return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

        def _point_near_polyline(ix, iy, feat, tol):
            """True when (ix,iy) is within ``tol`` px of any segment; lets
            transects (degenerate bbox when axis-aligned) be selected."""
            verts = _expand_for_overlay(feat)
            for j in range(len(verts) - 1):
                if _seg_distance(ix, iy,
                                 verts[j]["px"], verts[j]["py"],
                                 verts[j + 1]["px"],
                                 verts[j + 1]["py"]) <= tol:
                    return True
            return False

        def _select_tolerance():
            zoom = max(0.01, float(ctx.zoom))
            W = _state.get("W") or 1
            H = _state.get("H") or 1
            return max(10.0, (W * W + H * H) ** 0.5 * 0.006) / zoom

        def _hit_test_body(ix, iy, feat):
            # Transects have a degenerate bbox when axis-aligned, so select
            # them by proximity to the drawn line instead of bbox containment.
            if feat.get("shape") == "transect":
                return _point_near_polyline(ix, iy, feat, _select_tolerance())
            xmin, ymin, xmax, ymax = _feature_bbox_pixels(feat)
            return (xmin <= ix <= xmax and ymin <= iy <= ymax)

        def _hit_test_any_feature(ix, iy):
            tol = _select_tolerance()
            for i in range(len(ctx.features) - 1, -1, -1):
                feat = ctx.features[i]
                if feat.get("shape") == "transect":
                    if _point_near_polyline(ix, iy, feat, tol):
                        return i
                    continue
                xmin, ymin, xmax, ymax = _feature_bbox_pixels(feat)
                if xmin <= ix <= xmax and ymin <= iy <= ymax:
                    return i
            return -1

        def _ensure_polygon_repr(feat):
            shape = feat.get("shape", "polygon")
            if shape in ("polygon", "transect"):
                return
            expanded = _expand_for_overlay(feat)
            new_pts = []
            for v in expanded:
                wx, wy = _image_to_world(v["px"], v["py"])
                new_pts.append({"px": v["px"], "py": v["py"],
                                "x": wx, "y": wy})
            feat["pts"] = new_pts
            feat["shape"] = "polygon"

        def _start_drag(handle, ix, iy, idx):
            feat = ctx.features[idx]
            if handle in ("rot",) or (handle in _ANCHOR):
                _ensure_polygon_repr(feat)
            start_pts = [
                {"x": float(p["x"]), "y": float(p["y"])}
                for p in feat["pts"]
            ]
            xs = [p["x"] for p in start_pts]
            ys = [p["y"] for p in start_pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            mx, my = _image_to_world(ix, iy)
            _state["drag"] = {
                "handle": handle, "feat_idx": idx,
                "start_pts": start_pts, "start_bbox": bbox,
                "start_centroid": (cx, cy),
                "start_mouse": (mx, my),
                "start_ixy": (float(ix), float(iy)),
                "has_moved": False,
            }

        def _apply_drag(ix, iy):
            d = _state.get("drag")
            if d is None:
                return
            # Ignore mousemove jitter under 3 display px so a plain click
            # never deforms the shape.
            if not d.get("has_moved", False):
                six, siy = d.get("start_ixy", (ix, iy))
                if (((ix - six) ** 2
                     + (iy - siy) ** 2) ** 0.5 < 3.0):
                    return
                d["has_moved"] = True
            idx = d["feat_idx"]
            if idx < 0 or idx >= len(ctx.features):
                return
            feat = ctx.features[idx]
            mx, my = _image_to_world(ix, iy)
            handle = d["handle"]
            start_pts = d["start_pts"]
            cx, cy = d["start_centroid"]
            smx, smy = d["start_mouse"]
            bbox = d["start_bbox"]
            new_pts = [dict(p) for p in feat["pts"]]
            if handle == "body":
                dx = mx - smx
                dy = my - smy
                for j, sp in enumerate(start_pts):
                    new_pts[j]["x"] = sp["x"] + dx
                    new_pts[j]["y"] = sp["y"] + dy
            elif handle == "rot":
                import math
                a0 = math.atan2(smy - cy, smx - cx)
                a1 = math.atan2(my - cy, mx - cx)
                theta = a1 - a0
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                for j, sp in enumerate(start_pts):
                    rx = sp["x"] - cx
                    ry = sp["y"] - cy
                    new_pts[j]["x"] = cx + cos_t * rx - sin_t * ry
                    new_pts[j]["y"] = cy + sin_t * rx + cos_t * ry
            elif handle in _ANCHOR:
                # _ANCHOR uses screen-y semantics ("n" = top). A north-up
                # georeferenced image has world ymin at the BOTTOM of the
                # screen, so swap the y-side meaning when gt[5] < 0.
                gt = _state.get("gt")
                y_flipped = (gt is not None
                             and len(gt) >= 6
                             and gt[5] < 0)
                ax_side, ay_side = _ANCHOR[handle]
                ax = (bbox[0] if ax_side == "xmin"
                      else bbox[2] if ax_side == "xmax" else cx)
                if y_flipped:
                    ay = (bbox[3] if ay_side == "ymin"
                          else bbox[1] if ay_side == "ymax" else cy)
                else:
                    ay = (bbox[1] if ay_side == "ymin"
                          else bbox[3] if ay_side == "ymax" else cy)
                orig_hx = (bbox[2] if handle in
                           ("ne", "e", "se") else
                           bbox[0] if handle in
                           ("nw", "w", "sw") else cx)
                if y_flipped:
                    orig_hy = (bbox[1] if handle in
                               ("sw", "s", "se") else
                               bbox[3] if handle in
                               ("nw", "n", "ne") else cy)
                else:
                    orig_hy = (bbox[3] if handle in
                               ("sw", "s", "se") else
                               bbox[1] if handle in
                               ("nw", "n", "ne") else cy)
                if ax_side and orig_hx != ax:
                    sx = 1.0 + (mx - smx) / (orig_hx - ax)
                else:
                    sx = 1.0
                if ay_side and orig_hy != ay:
                    sy = 1.0 + (my - smy) / (orig_hy - ay)
                else:
                    sy = 1.0
                # Floor the scale so a near-zero drag cannot collapse the
                # shape into an ungrabbable sliver.
                sx = max(0.05, sx) if sx > 0 else (
                    min(-0.05, sx) if sx < 0 else 1.0)
                sy = max(0.05, sy) if sy > 0 else (
                    min(-0.05, sy) if sy < 0 else 1.0)
                for j, sp in enumerate(start_pts):
                    new_pts[j]["x"] = ax + sx * (sp["x"] - ax)
                    new_pts[j]["y"] = ay + sy * (sp["y"] - ay)
            for j, np_ in enumerate(new_pts):
                feat["pts"][j]["x"] = np_["x"]
                feat["pts"][j]["y"] = np_["y"]
                px, py = _world_to_image(np_["x"], np_["y"])
                feat["pts"][j]["px"] = px
                feat["pts"][j]["py"] = py
            _refresh_overlay()

        def _end_drag():
            d = _state.get("drag")
            _state["drag"] = None
            if d is not None:
                _autosave()
                _refresh_features_table()

        def _draw_shape():
            """The shape currently being drawn: ``effective_shape_getter``
            when the host injects one, else ``ctx.shape``, normalised."""
            s = (effective_shape_getter() if effective_shape_getter is not None
                 else ctx.shape)
            return normalise_draw_shape(s)

        def _build_overlay_svg():
            parts = []
            W = _state.get("W") or 1
            H = _state.get("H") or 1
            diag = (W * W + H * H) ** 0.5
            zoom = max(0.01, float(ctx.zoom))
            base_mr = min(8.0, max(2.0, diag * 0.0010))
            base_sw = min(3.0, max(1.0, diag * 0.0004))
            mr = base_mr / zoom
            sw = base_sw / zoom
            # Bold style: screen-pixel sizes divided by the zoom, since the
            # SVG is in cached-PNG pixels scaled by CSS width.
            bold = (feature_style == "bold")
            b_vr = 7.0 / zoom       # in-progress vertex radius
            b_lw = 4.0 / zoom       # committed line
            b_halo = 10.0 / zoom    # white line underneath (3 px each side)
            b_er = 5.0 / zoom       # endpoint circles
            b_fs = 12.0 / zoom      # label font size
            b_thin = max(1.0, 1.5 / zoom)

            def _label_box(x, y, text, colour):
                # A white-backed label above (x, y); the width is estimated
                # from the character count (no text metrics server-side).
                import html as _html
                w = b_fs * (0.62 * len(text) + 1.2)
                h = b_fs * 1.5
                x0, y0 = x - w / 2.0, y - h
                return (
                    f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" '
                    f'height="{h:.1f}" rx="{3.0 / zoom:.1f}" '
                    f'fill="white" fill-opacity="0.92" stroke="{colour}" '
                    f'stroke-width="{b_thin:.1f}"/>'
                    f'<text x="{x:.1f}" y="{y - h * 0.36:.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="{b_fs:.1f}" font-weight="600" fill="#222">'
                    f'{_html.escape(text)}</text>')

            for i, feat in enumerate(ctx.features):
                shape = feat.get("shape", "polygon")
                verts = _expand_for_overlay(feat)
                if not verts:
                    continue
                pts_str = " ".join(f"{v['px']:.1f},{v['py']:.1f}"
                                   for v in verts)
                is_sel = (overlay_select_recolor
                          and i == ctx.selected_idx)
                if is_sel:
                    colour = "#22aaff"
                    fill = "rgba(34,170,255,0.20)"
                    stroke_w = max(sw * 1.8, sw + 0.6 / zoom)
                else:
                    colour = ("#33aa66" if feat.get("source") == "imported"
                              else "#ff8800")
                    fill = ("rgba(50,170,100,0.18)"
                            if feat.get("source") == "imported"
                            else "rgba(255,180,30,0.22)")
                    stroke_w = sw
                if shape == "transect" and bold:
                    lw = b_lw * (1.3 if is_sel else 1.0)
                    parts.append(
                        f'<polyline points="{pts_str}" fill="none" '
                        f'stroke="white" stroke-width="{b_halo:.1f}" '
                        f'stroke-linecap="round" stroke-linejoin="round"/>')
                    parts.append(
                        f'<polyline points="{pts_str}" fill="none" '
                        f'stroke="{colour}" stroke-width="{lw:.1f}" '
                        f'stroke-linecap="round" stroke-linejoin="round"/>')
                    for v in (verts[0], verts[-1]):
                        parts.append(
                            f'<circle cx="{v["px"]:.1f}" cy="{v["py"]:.1f}" '
                            f'r="{b_er:.1f}" fill="{colour}" stroke="white" '
                            f'stroke-width="{b_thin:.1f}"/>')
                    try:
                        label = (feature_label(feat, i) if feature_label
                                 else str(feat.get("id") or ""))
                    except Exception:
                        label = str(feat.get("id") or "")
                    if label:
                        # The label lies along the segment: box and text
                        # are drawn a few screen px above the midpoint,
                        # then the group is rotated about the midpoint by
                        # the segment's on-screen angle (SVG rotate() is
                        # clockwise-positive, hence the sign; the angle
                        # is flipped to stay readable). The SVG is in
                        # image px scaled by CSS, so the angle survives
                        # the zoom unchanged.
                        from functions.gauge import segment_label_angle
                        p_a, p_b = verts[0], verts[-1]
                        mx = (p_a["px"] + p_b["px"]) / 2.0
                        my = (p_a["py"] + p_b["py"]) / 2.0
                        ang = -segment_label_angle(
                            (p_a["px"], p_a["py"]), (p_b["px"], p_b["py"]))
                        parts.append(
                            f'<g transform="rotate({ang:.1f} {mx:.1f} '
                            f'{my:.1f})">'
                            + _label_box(mx, my - b_er - 4.0 / zoom,
                                         label, colour)
                            + '</g>')
                elif shape == "transect":
                    parts.append(
                        f'<polyline points="{pts_str}" fill="none" '
                        f'stroke="{colour}" stroke-width="{stroke_w:.1f}"/>')
                else:
                    parts.append(
                        f'<polygon points="{pts_str}" fill="{fill}" '
                        f'stroke="{colour}" stroke-width="{stroke_w:.1f}"/>')

            pts = ctx.pts
            shape = _draw_shape()
            if pts:
                pts_str = " ".join(f"{p['px']:.1f},{p['py']:.1f}"
                                   for p in pts)
                if shape == "transect" and bold:
                    # Rubber band from the first vertex to the cursor.
                    cur = _state.get("cursor")
                    if len(pts) == 1 and cur is not None:
                        band = (f'{pts[0]["px"]:.1f},{pts[0]["py"]:.1f} '
                                f'{cur[0]:.1f},{cur[1]:.1f}')
                        parts.append(
                            f'<polyline points="{band}" fill="none" '
                            f'stroke="white" stroke-width="{b_halo * 0.8:.1f}" '
                            f'stroke-linecap="round" stroke-opacity="0.8"/>')
                        parts.append(
                            f'<polyline points="{band}" fill="none" '
                            f'stroke="#22aaff" stroke-width="{b_lw * 0.75:.1f}" '
                            f'stroke-linecap="round" '
                            f'stroke-dasharray="{8.0 / zoom:.1f},{5.0 / zoom:.1f}"/>')
                    elif len(pts) >= 2:
                        parts.append(
                            f'<polyline points="{pts_str}" fill="none" '
                            f'stroke="white" stroke-width="{b_halo:.1f}" '
                            f'stroke-linecap="round"/>')
                        parts.append(
                            f'<polyline points="{pts_str}" fill="none" '
                            f'stroke="#22aaff" stroke-width="{b_lw:.1f}" '
                            f'stroke-linecap="round" '
                            f'stroke-dasharray="{8.0 / zoom:.1f},{5.0 / zoom:.1f}"/>')
                elif shape == "transect" and len(pts) >= 2:
                    parts.append(
                        f'<polyline points="{pts_str}" fill="none" '
                        f'stroke="#22aaff" stroke-width="{sw:.1f}" '
                        f'stroke-dasharray="6,3"/>')
                elif shape == "polygon" and len(pts) >= 2:
                    parts.append(
                        f'<polyline points="{pts_str}" fill="none" '
                        f'stroke="#ffaa00" stroke-width="{sw:.1f}" '
                        f'stroke-dasharray="6,3"/>')
                elif shape == "rectangle" and len(pts) >= 2:
                    x0, y0 = pts[0]["px"], pts[0]["py"]
                    x1, y1 = pts[1]["px"], pts[1]["py"]
                    parts.append(
                        f'<polygon points="{x0:.1f},{y0:.1f} '
                        f'{x1:.1f},{y0:.1f} {x1:.1f},{y1:.1f} '
                        f'{x0:.1f},{y1:.1f}" '
                        f'fill="rgba(255,180,30,0.22)" '
                        f'stroke="#ff8800" stroke-width="{sw:.1f}" '
                        f'stroke-dasharray="6,3"/>')
                elif shape == "circle" and len(pts) >= 2:
                    cx, cy = pts[0]["px"], pts[0]["py"]
                    rx, ry = pts[1]["px"], pts[1]["py"]
                    r = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
                    parts.append(
                        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" '
                        f'r="{r:.1f}" fill="rgba(255,180,30,0.22)" '
                        f'stroke="#ff8800" stroke-width="{sw:.1f}" '
                        f'stroke-dasharray="6,3"/>')
                ready_to_close = (
                    shape == "polygon" and len(pts) >= 3)
                for i, p in enumerate(pts):
                    if i == 0 and ready_to_close:
                        parts.append(
                            f'<circle cx="{p["px"]:.1f}" '
                            f'cy="{p["py"]:.1f}" '
                            f'r="{mr * 1.8:.1f}" fill="none" '
                            f'stroke="#33aa66" '
                            f'stroke-width="{max(2.0, sw):.1f}"/>')
                    if bold:
                        parts.append(
                            f'<circle cx="{p["px"]:.1f}" cy="{p["py"]:.1f}" '
                            f'r="{b_vr:.1f}" fill="#ff3030" stroke="white" '
                            f'stroke-width="{3.0 / zoom:.1f}"/>')
                        continue
                    parts.append(
                        f'<circle cx="{p["px"]:.1f}" cy="{p["py"]:.1f}" '
                        f'r="{mr:.1f}" fill="#ff5050" stroke="white" '
                        f'stroke-width="{max(1.5, sw / 2):.1f}"/>')

            if ctx.canvas_mode == "modify":
                sel_idx = ctx.selected_idx
                if 0 <= sel_idx < len(ctx.features):
                    sel = ctx.features[sel_idx]
                    xmin, ymin, xmax, ymax = _feature_bbox_pixels(sel)
                    parts.append(
                        f'<rect x="{xmin:.1f}" y="{ymin:.1f}" '
                        f'width="{xmax-xmin:.1f}" '
                        f'height="{ymax-ymin:.1f}" '
                        f'fill="none" stroke="#22aaff" '
                        f'stroke-width="{sw:.1f}" '
                        f'stroke-dasharray="6,4"/>')
                    hsize = max(6.0, mr * 1.4)
                    handles = _handle_positions_pixels(sel)
                    rot_xy = handles.pop("rot", None)
                    cx_top = (xmin + xmax) / 2.0
                    if rot_xy is not None:
                        parts.append(
                            f'<line x1="{cx_top:.1f}" y1="{ymin:.1f}" '
                            f'x2="{rot_xy[0]:.1f}" '
                            f'y2="{rot_xy[1]:.1f}" '
                            f'stroke="#33aa66" '
                            f'stroke-width="{sw:.1f}"/>')
                        parts.append(
                            f'<circle cx="{rot_xy[0]:.1f}" '
                            f'cy="{rot_xy[1]:.1f}" '
                            f'r="{hsize:.1f}" fill="#33aa66" '
                            f'stroke="white" '
                            f'stroke-width="{max(1.0, sw / 2):.1f}"/>')
                    for name, (hx, hy) in handles.items():
                        parts.append(
                            f'<rect x="{hx - hsize:.1f}" '
                            f'y="{hy - hsize:.1f}" '
                            f'width="{hsize * 2:.1f}" '
                            f'height="{hsize * 2:.1f}" '
                            f'fill="#22aaff" stroke="white" '
                            f'stroke-width="{max(1.0, sw / 2):.1f}"/>')
            return "<g>" + "".join(parts) + "</g>"

        def _refresh_overlay():
            iimg = _state.get("interactive")
            if iimg is None:
                return
            try:
                iimg.content = _build_overlay_svg()
                iimg.update()
            except Exception:
                pass

        def _refresh_status():
            if status_text is not None:
                try:
                    custom = status_text()
                except Exception:
                    custom = None
                if custom:
                    status_label.set_text(str(custom))
                    return
            n = len(ctx.pts)
            n_imported = sum(1 for f in ctx.features
                             if f.get("source") == "imported")
            n_drawn = sum(1 for f in ctx.features
                          if f.get("source") != "imported")
            shape = _draw_shape()
            tips = {
                "polygon":
                    f"Click vertices, then Finalise (need ≥ 3, "
                    f"have {n}).",
                "rectangle":
                    "Click two opposite corners — auto-commits."
                    if n < 2 else "Rectangle ready.",
                "circle":
                    "Click centre, then click rim — auto-commits."
                    if n < 2 else "Circle ready.",
                "transect":
                    ("Click both ends of the object — auto-commits."
                     if two_click_transect else
                     f"Click waypoints, then Finalise (need ≥ 2, "
                     f"have {n})."),
            }
            tip = tips.get(shape, "")
            status_label.set_text(
                f"Shape: {shape}   |   In-progress: {n}   |   "
                f"Imported: {n_imported}   |   Drawn: {n_drawn}   "
                f"|   Zoom: {ctx.zoom:.2f}×   |   {tip}")

        def _refresh_features_table():
            features_table.clear()
            with features_table:
                if not ctx.features:
                    ui.label(
                        "(no features yet — pick a Vector layer "
                        "above OR draw with the toolbar)") \
.classes("text-grey-6 italic")
                    return
                ui.label(
                    f"Features ({len(ctx.features)})"
                ).classes("text-sm font-bold")
                for i, feat in enumerate(ctx.features):
                    with ui.row().classes(
                            "items-center gap-2 w-full"):
                        is_sel = (ctx.selected_idx == i)

                        def _select(idx=i):
                            ctx.selected_idx = idx
                            _refresh_features_table()
                            _refresh_overlay()
                            _refresh_status()

                        ui.button(
                            icon=("radio_button_checked" if is_sel
                                  else "radio_button_unchecked"),
                            on_click=_select,
                        ).props("flat dense color=primary") \
.tooltip("Select for the Modify-mode "
                                  "handles.")
                        ui.label(f"{i+1}.").classes("text-grey-7")
                        # Editable name; persisted as GeoJSON properties.id.
                        name_input = ui.input(
                            value=str(feat.get("id", "")),
                            placeholder="layer name",
                        ).classes("text-sm font-mono w-48") \
.props("dense outlined")

                        def _on_name_change(e, idx=i):
                            new = (e.value or "").strip() or feat.get(
                                "id", f"feature_{idx + 1}")
                            ctx.features[idx]["id"] = new
                            _autosave()

                        name_input.on_value_change(_on_name_change)
                        ui.label(
                            f"{feat.get('shape')}  ·  "
                            f"{len(feat['pts'])} pts  ·  "
                            f"{feat.get('source', 'drawn')}"
                        ).classes("flex-grow text-sm text-grey-7")

                        def _remove(idx=i):
                            ctx.features.pop(idx)
                            if ctx.selected_idx >= idx:
                                ctx.selected_idx = -1
                            _refresh_overlay()
                            _refresh_features_table()
                            _refresh_status()
                            _autosave()

                        ui.button(icon="delete",
                                   on_click=_remove) \
.props("flat dense color=negative")

        def _on_canvas_mouse(e):
            etype = getattr(e, "type", "")
            ix = float(getattr(e, "image_x", 0))
            iy = float(getattr(e, "image_y", 0))
            mode = ctx.canvas_mode
            if mode == "modify":
                if etype == "mousedown":
                    idx = ctx.selected_idx
                    if 0 <= idx < len(ctx.features):
                        feat = ctx.features[idx]
                        handle = _hit_test_handle(ix, iy, feat)
                        if handle is not None:
                            _start_drag(handle, ix, iy, idx)
                            return
                        if _hit_test_body(ix, iy, feat):
                            _start_drag("body", ix, iy, idx)
                            return
                    new_idx = _hit_test_any_feature(ix, iy)
                    ctx.selected_idx = new_idx if new_idx >= 0 else -1
                    _refresh_overlay()
                    _refresh_features_table()
                    _refresh_status()
                    return
                if etype == "mousemove":
                    if _state.get("drag") is not None:
                        _apply_drag(ix, iy)
                    return
                if etype == "mouseup":
                    if _state.get("drag") is not None:
                        _end_drag()
                    return
                return
            # Draw mode
            if etype == "mousemove":
                _track_cursor(ix, iy)
                return
            if etype != "click":
                return
            shape = _draw_shape()
            if (shape == "polygon" and len(ctx.pts) >= 3):
                # A click within 12 display px of the first vertex closes
                # the polygon (Enter closes it too).
                _SNAP_PX = 12
                first = ctx.pts[0]
                if ((ix - first["px"]) ** 2
                    + (iy - first["py"]) ** 2) ** 0.5 < _SNAP_PX:
                    _commit_current()
                    return
            wx, wy = _image_to_world(ix, iy)
            pt = {"px": ix, "py": iy, "x": wx, "y": wy}
            ctx.pts.append(pt)
            if (shape in ("rectangle", "circle")
                    or (shape == "transect" and two_click_transect)) \
                    and len(ctx.pts) >= 2:
                _commit_current()
            _refresh_overlay()
            _refresh_status()

        def _track_cursor(ix, iy):
            """The bold rubber band: mouse moves already reach the server
            (the widget subscribes to them for Modify drags), so this only
            re-renders the overlay, and only with one transect vertex down,
            after a 2-screen-px move, at most every 40 ms."""
            if feature_style != "bold" or len(ctx.pts) != 1 \
                    or _draw_shape() != "transect":
                return
            import time as _time
            last = _state.get("cursor")
            zoom = max(0.01, float(ctx.zoom))
            if last is not None and (((ix - last[0]) ** 2
                                      + (iy - last[1]) ** 2) ** 0.5) * zoom < 2.0:
                return
            now = _time.monotonic()
            if now - float(_state.get("cursor_t") or 0.0) < 0.04:
                return
            _state["cursor"] = (float(ix), float(iy))
            _state["cursor_t"] = now
            _refresh_overlay()

        def _commit_current():
            shape = _draw_shape()
            pts = ctx.pts
            min_pts = {"polygon": 3, "transect": 2,
                       "rectangle": 2, "circle": 2}[shape]
            if len(pts) < min_pts:
                ui.notify(
                    f"{shape} needs >={min_pts} clicks on the canvas (Work surface), "
                    f"have {len(pts)}.", type="warning")
                return
            new_id = f"{shape}_{len(ctx.features) + 1}"
            n_pts = len(pts)          # read before the clear below empties it
            ctx.features.append({
                "shape": shape, "pts": list(pts),
                "id": new_id, "source": "drawn",
            })
            ctx.pts.clear()
            _state["cursor"] = None
            _refresh_overlay()
            _refresh_features_table()
            _refresh_status()
            _autosave()
            try:
                ui.notify(
                    f"Committed {new_id} ({n_pts} vertices); "
                    f"autosaved to {Path(ctx.vector).name}"
                    if ctx.vector else
                    f"Committed {new_id} ({n_pts} vertices)",
                    type="positive")
            except Exception:
                pass

        def _undo_last_vertex():
            if ctx.pts:
                ctx.pts.pop()
                _refresh_overlay()
                _refresh_status()

        def _finalise_feature():
            _commit_current()

        def _clear_in_progress():
            ctx.pts.clear()
            _state["cursor"] = None
            _refresh_overlay()
            _refresh_status()

        def _clear_all_features():
            ctx.features.clear()
            ctx.pts.clear()
            _state["cursor"] = None
            _state["last_imported_path"] = ""
            _refresh_overlay()
            _refresh_features_table()
            _refresh_status()
            _autosave()

        def _apply_zoom():
            zoom = max(0.01, float(ctx.zoom))
            W_disp = int(_state.get("W") or 0)
            iimg = _state.get("interactive")
            if iimg is not None and W_disp > 0:
                try:
                    iimg.style(
                        f"width:{int(round(W_disp * zoom))}px; "
                        f"height:auto; display:block;")
                except Exception:
                    pass
            _refresh_overlay()
            _refresh_status()

        def _rebuild_canvas():
            zw = _state.get("zoom_wrap")
            if zw is None:
                _refresh_status()
                return
            zw.clear()
            _state["interactive"] = None
            _state["gt"] = None
            _state["W"] = 0
            _state["H"] = 0
            _state["W_orig"] = 0
            _state["H_orig"] = 0
            _state["disp_scale"] = 1.0

            p = ctx.image
            if not p or not Path(p).exists():
                with zw:
                    ui.label(
                        "(pick a source image above to enable the "
                        "drawing canvas)") \
.classes("text-grey-6 italic p-4")
                _refresh_status()
                return

            try:
                from PIL import Image as _PILImage
                with _PILImage.open(p) as _pim:
                    W_orig, H_orig = _pim.size
                _state["W_orig"] = W_orig
                _state["H_orig"] = H_orig
            except Exception as ex:
                with zw:
                    ui.label(f"Could not read image: {ex}"
                             f"{_images.heif_failure_hint(p)}") \
.classes("text-negative")
                _refresh_status()
                return

            try:
                # GDAL has no HEIF driver here and prints "ERROR 4" on a
                # file it does not recognise; a photograph has no geotransform.
                if _images.is_heif(p):
                    raise ValueError("HEIF: not a GDAL raster")
                from osgeo import gdal as _gdal
                _ds = _gdal.Open(p)
                if _ds is not None:
                    _gt = _ds.GetGeoTransform()
                    _ds = None
                    if _gt and _gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                        _state["gt"] = _gt
            except Exception:
                pass

            try:
                serve_path_str, disp_scale = prepare_browser_image(
                    p, current_project=current_project_getter(),
                    max_dim=4000)
                serve_path = Path(serve_path_str)
                _state["disp_scale"] = float(disp_scale)
            except Exception as ex:
                with zw:
                    ui.label(
                        f"Could not prepare canvas image: {ex}"
                        f"{_images.heif_failure_hint(p)}") \
.classes("text-negative")
                _refresh_status()
                return

            try:
                from PIL import Image as _PILImage
                with _PILImage.open(serve_path) as _pim2:
                    W_disp, H_disp = _pim2.size
            except Exception:
                W_disp, H_disp = W_orig, H_orig
            _state["W"], _state["H"] = W_disp, H_disp

            from nicegui import app as _app
            import hashlib
            src_dir = str(serve_path.parent)
            mount = ("canvas_"
                     + hashlib.md5(src_dir.encode()).hexdigest()[:8])
            try:
                _app.add_static_files(f"/{mount}", src_dir)
            except Exception:
                pass
            url = f"/{mount}/{serve_path.name}"

            with zw:
                iimg = ui.interactive_image(
                    source=url,
                    events=["click", "mousedown",
                            "mousemove", "mouseup"],
                    cross=True,
                    on_mouse=_on_canvas_mouse,
                ).style("display:block;")
                _state["interactive"] = iimg
                if disp_scale != 1.0:
                    ui.label(
                        f"Display: {W_disp}×{H_disp} px "
                        f"(cached PNG, {disp_scale:.2f}× downsample "
                        f"of {W_orig}×{H_orig} px original)") \
.classes("text-xs text-grey-7")
                else:
                    ui.label(
                        f"Display: {W_disp}×{H_disp} px (native)") \
.classes("text-xs text-grey-7")
            if H_disp > 1500:
                ctx.zoom = max(0.05, 700.0 / H_disp)
            _apply_zoom()
            _reproject_features_to_display()
            # Auto-import only when the vector path changed, or every
            # rebuild would append the same sidecar again.
            if (ctx.vector
                and Path(ctx.vector).exists()
                and _state.get("last_imported_path") != ctx.vector):
                _auto_import(replace=True)
            _refresh_overlay()
            _refresh_status()

        def _resolve_save_path():
            # A host resolver may return None (target not named yet).
            if save_path_resolver is not None:
                return save_path_resolver()
            if ctx.out_path:
                return Path(ctx.out_path)
            if save_dir_resolver is not None:
                base = Path(save_dir_resolver())
            else:
                base = Path(ctx.image or "").parent or Path.cwd()
            base.mkdir(parents=True, exist_ok=True)
            image_stem = (Path(ctx.image).stem
                          if ctx.image else "features")
            return base / save_stem_template.format(
                image_stem=image_stem)

        def _source_image_bbox():
            """World extent of the loaded source image, or None."""
            return raster_world_bbox(
                _state.get("gt"), _state.get("W_orig"),
                _state.get("H_orig"))

        def _write_features_geojson(path):
            import json
            features = []
            for feat in ctx.features:
                shape = feat.get("shape", "polygon")
                expanded = _expand_for_overlay(feat)
                coords = []
                for v in expanded:
                    if ("x" in v and "y" in v
                        and feat.get("shape")
                            not in ("rectangle", "circle")):
                        coords.append([v["x"], v["y"]])
                    else:
                        wx, wy = _image_to_world(
                            v["px"], v["py"])
                        coords.append([wx, wy])
                if shape == "transect":
                    geom = {"type": "LineString",
                            "coordinates": coords}
                else:
                    if coords and coords[0] != coords[-1]:
                        coords.append(coords[0])
                    geom = {"type": "Polygon",
                            "coordinates": [coords]}
                props = {
                    "id": feat["id"],
                    "shape": shape,
                    "source": feat.get("source", "drawn"),
                }
                if write_provenance:
                    props["name"] = feat["id"]
                features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": geom,
                })
            payload = {"type": "FeatureCollection",
                       "features": features}
            if write_provenance:
                # Source image and extents, under a non-standard member that
                # GeoJSON readers ignore.
                src_img = ctx.image or ""
                if src_img:
                    payload["pebblemapper"] = {
                        "source_image": Path(src_img).name,
                        "source_image_extent": _source_image_bbox(),
                        "features_extent": features_world_bbox(ctx.features),
                    }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            return len(features)

        def _autosave():
            try:
                path = _resolve_save_path()
                if path is None:
                    if on_unnamed_save is not None:
                        try:
                            on_unnamed_save(False)
                        except Exception:
                            pass
                    return
                _write_features_geojson(path)
                ctx.out_path = str(path)
                ctx.vector = str(path)
                _state["last_imported_path"] = str(path)
                if on_save is not None:
                    try:
                        on_save(path)
                    except Exception:
                        pass
                if disjoint_policy == "warn":
                    # A named set may span many orthos: notify once, never block.
                    try:
                        if not features_match_image(
                                features_world_bbox(ctx.features),
                                _source_image_bbox()):
                            if not _state.get("warned_extent_mismatch"):
                                _state["warned_extent_mismatch"] = True
                                ui.notify(
                                    "Saved. Note these features lie outside "
                                    f"{Path(ctx.image).name}, so the canvas "
                                    "won't show them over this image.",
                                    type="info", timeout=8000,
                                    multi_line=True, close_button=True)
                    except Exception:
                        pass
            except Exception as ex:
                try:
                    print(f"[canvas autosave] failed: {ex}")
                except Exception:
                    pass

        def _save_geojson():
            if not ctx.features:
                ui.notify("No features staged — draw or import one on the canvas (Work surface, above).", type="warning")
                return
            path = _resolve_save_path()
            if path is None:
                if on_unnamed_save is not None:
                    try:
                        on_unnamed_save(True)
                    except Exception:
                        pass
                return
            # "block": the save target is image-derived, so geometry disjoint
            # from the loaded image would be filed under the wrong image.
            # Refuse once; a second Save overrides.
            if disjoint_policy == "block":
                try:
                    img_bbox = raster_world_bbox(
                        _state.get("gt"), _state.get("W_orig"),
                        _state.get("H_orig"))
                    feat_bbox = features_world_bbox(ctx.features)
                    if not features_match_image(feat_bbox, img_bbox):
                        key = str(path)
                        if _state.get("mismatch_armed") != key:
                            _state["mismatch_armed"] = key
                            ui.notify(
                                "The staged geometry does not overlap the "
                                "loaded image at all — saving now would write "
                                f"{Path(key).name}, named after an image it "
                                "was not drawn on. Nothing has been written. "
                                "Press Save GeoJSON (bar above) again to save it anyway, or load "
                                "the image these features belong to.",
                                type="warning", multi_line=True,
                                timeout=12000, close_button=True)
                            return
                except Exception:
                    pass  # the guard must never block a legitimate save
                _state["mismatch_armed"] = None
            # First Save onto an existing name arms; a second Save replaces it.
            if confirm_overwrite:
                if (path.exists() and not ctx.out_path
                        and _state.get("overwrite_armed") != str(path)):
                    _state["overwrite_armed"] = str(path)
                    ui.notify(
                        f"{path.name} already exists. Nothing has been "
                        "written. Press Save GeoJSON (bar above) again to replace it, or change "
                        "the name.",
                        type="warning", multi_line=True, timeout=12000,
                        close_button=True)
                    return
                _state["overwrite_armed"] = None
            try:
                n = _write_features_geojson(path)
                ctx.out_path = str(path)
                ctx.vector = str(path)
                _state["last_imported_path"] = str(path)
                if on_save is not None:
                    try:
                        on_save(path)
                    except Exception:
                        pass
                ui.notify(f"Saved {n} feature(s) -> {path}",
                           type="positive")
            except Exception as ex:
                ui.notify(f"Save failed: {ex}. Check the output name (Inputs, above), then Save GeoJSON again.",
                           type="negative")

        def _kb(e):
            try:
                if not tab_active_check():
                    return
                if not (e.action and e.action.keydown):
                    return
                if e.key.enter:
                    _finalise_feature()
                elif e.key.escape:
                    _clear_in_progress()
                elif e.key.backspace:
                    _undo_last_vertex()
            except Exception:
                pass

        try:
            ui.keyboard(on_key=_kb)
        except Exception:
            pass

        with ui.row().classes("items-center gap-2 mt-1") \
.style("font-size:0.8em;color:#5a6878;"):
            ui.label("Shortcuts (when tab is active):") \
.classes("font-bold")
            ui.label(
                "Enter — finalise   ·   Esc — clear in-progress   "
                "·   Backspace — undo vertex   ·   Click first "
                "vertex (green halo) — close polygon")

        _rebuild_canvas()
        _refresh_features_table()
        _refresh_status()

        return {
            "rebuild_canvas": _rebuild_canvas,
            "refresh_overlay": _refresh_overlay,
            "refresh_features_table": _refresh_features_table,
            "refresh_status": _refresh_status,
            "zoom_fit": _zoom_fit,
        }


# --- Application state ---
@dataclass
class AppState:
    """All cross-tab GUI state. One instance per running GUI."""

    devicemode: str = "GPU"
    devicenumber: int = 0
    # Bound to ui.tab_panels so the selection survives reactive re-syncs.
    active_tab: str = "Overview"

    # Active project (a subdirectory of datasets/); '' = none selected.
    current_project: str = ""

    # Active date of a multi-temporal project; '' for a flat project.
    current_date: str = ""

    # ---- Orthorectify tab state ----
    ortho_src_path: str = ""
    # Picked corner pixel coords, [x, y] lists (lists serialise cleanly
    # through NiceGUI's binding layer; tuples do not).
    ortho_corners: list = field(default_factory=list)
    # Next corner to set (0..3); wraps so any corner can be refined.
    ortho_click_idx: int = 0
    # Segment lengths in metres; opposite sides default to equal.
    ortho_seg_lengths: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # Quadrat frame thickness in metres, a property of the equipment. None =
    # not stated; 0.0 = the frame is not in the way. Keep them distinct.
    ortho_frame_thickness_m: Optional[float] = None
    # Zoom on the picking canvas, 1.0 = fitted to the column.
    ortho_zoom: float = 1.0
    # Client viewport height (px), read once per page load, so "Fit" can put
    # the whole photograph inside the 70 % box; 900 until reported.
    ortho_viewport_h: int = 900
    # The previous quadrat's outline, drawn as a backdrop; never a placement
    # (see functions.orthorectify.guide_quadrilateral).
    ortho_guide: Optional[tuple] = None
    ortho_guide_shown: bool = True
    # True while the guide came from the detector rather than from a
    # photograph whose corners a person placed.
    ortho_guide_is_suggestion: bool = False
    # Folder-run progress, held here so the widget timer can read it.
    ortho_batch_busy: str = ""
    ortho_batch_done: int = 0
    ortho_batch_total: int = 0
    ortho_batch_now: str = ""
    ortho_batch_stop: bool = False
    ortho_batch_msg: str = ""
    ortho_batch_overwrite: bool = False
    ortho_batch_rows: list = field(default_factory=list)
    # The user deliberately chose a placed corner to move; with four corners
    # loaded the click index wraps, so a stray click must not replace one.
    ortho_corner_armed: bool = False
    # Segment-length confirmation is per file, in ortho_per_image under
    # "seg_confirmed" (functions.orthorectify.seg_confirmed_for).
    # Per-segment scalars bound to the four inputs; kept in sync with
    # ortho_seg_lengths (NiceGUI binds to attribute names, not list indices).
    ortho_seg_1: float = 1.0
    ortho_seg_2: float = 1.0
    ortho_seg_3: float = 1.0
    ortho_seg_4: float = 1.0
    # Which physical edge the picks follow; labels only, no effect on the math.
    ortho_edge_mode: str = "outer"     # 'inner' or 'outer'
    # Output GSD (m/pixel); 0.0 = auto from the source pixel density.
    ortho_gsd_m: float = 0.0
    ortho_out_path: str = ""
    # Folder workflow: a directory of photos plus per-image cached state.
    ortho_image_dir: str = ""
    ortho_image_list: list = field(default_factory=list)
    ortho_image_idx: int = -1
    # The Image select binds its value here (a bound field resolves on first
    # paint where set_options(value=) on an empty select does not).
    ortho_selected_image: str = ""
    ortho_per_image: dict = field(default_factory=dict)
    # Optional lens-distortion correction (OpenCV intrinsics): the image is
    # undistorted and the corners mapped to that frame before the homography.
    ortho_use_distortion: bool = False
    ortho_fx: float = 0.0
    ortho_fy: float = 0.0
    ortho_cx: float = 0.0
    ortho_cy: float = 0.0
    ortho_k1: float = 0.0
    ortho_k2: float = 0.0
    ortho_p1: float = 0.0
    ortho_p2: float = 0.0
    ortho_k3: float = 0.0

    # ---- Georeference tab state ----
    geo_quadrat_path: str = ""
    geo_quadrat_note: str = ""
    # A quadrat GeoTIFF already knows where it is; only a plain photograph
    # needs matching. Never offer to replace a surveyed georeference.
    geo_quadrat_needs_georef: bool = True
    geo_resolution: float = 0.001
    # "ortho" uses whatever the ortho declares; "specify" uses geo_seed_epsg
    # (a field GPS gives WGS84 lat/lon).
    geo_seed_crs_mode: str = "specify"
    geo_seed_epsg: str = "EPSG:4326"
    # Seed-list CSV dialect.
    geo_csv_separator: str = "comma"
    geo_csv_has_header: bool = True
    geo_csv_decimal: str = "auto"
    geo_csv_preview: list = field(default_factory=list)
    geo_csv_preview_msg: str = ""
    # Long-running action in flight, "" when idle; drives the spinner.
    geo_busy: str = ""
    # ---- alignment editor ----
    # High-pass is the default on purpose: in plain colour the eye judges
    # tone and blur rather than position.
    geo_editor_on: bool = False
    geo_view_mode: str = "highpass"
    geo_blend_alpha: float = 0.55
    geo_edit_east: float = 0.0
    geo_edit_north: float = 0.0
    geo_edit_rot: float = 0.0
    geo_score_msg: str = ""
    geo_can_keep: bool = False
    geo_edit_score: float = 0.0
    # Display magnification; at 1:1 one screen pixel is one ortho pixel.
    geo_zoom: int = 1
    geo_progress: str = ""
    geo_warn_msg: str = ""
    geo_seed_note: str = ""
    # Survey-check progress and stop flag.
    geo_batch_done: int = 0
    geo_batch_total: int = 0
    geo_batch_now: str = ""
    geo_batch_stop: bool = False
    # Sweep the whole ortho for quadrats with no seed (every tile instead
    # of ~80).
    geo_full_grid: bool = False
    # Replace existing outputs when saving a whole survey. Off:
    # validation/georectified/ holds hand-made reference placements.
    geo_save_overwrite: bool = False
    # Rival placements the matcher could not choose between.
    geo_candidates: list = field(default_factory=list)
    # Full survey-check results (matrices and residuals), needed to save a
    # whole survey without re-running the matcher.
    geo_batch_results: list = field(default_factory=list)
    geo_overlay_note: str = ""
    # Frame-thickness provenance. A recorded value fills an untouched box and
    # must never overwrite a typed one; "untouched" is decided by comparing
    # values, since a bound box fires the same change event for a
    # programmatic write as for a keystroke.
    geo_inset_note: str = ""
    geo_inset_auto: Optional[float] = None
    geo_preflight: str = ""
    geo_ready: bool = False
    geo_not_ready: str = "pick an ortho and a folder first"
    # photo name -> {matrix, seed_world}, so a row opens without re-matching.
    geo_batch_placements: dict = field(default_factory=dict)
    geo_kept: bool = False
    # The placement being edited and the one it started from
    # (functions.placement.Placement, frozen; Reset is equality).
    geo_placement: object = None
    geo_start_placement: object = None
    # photo key -> "kept" | "discarded", so a season can be resumed.
    geo_decisions: dict = field(default_factory=dict)

    # ---- Digitize tab state ----
    dig_src_path: str = ""
    # Quadrat georeferencing inputs (controls live in the Georeference tab;
    # the matcher, pin canvas and batch check all read these).
    dig_georef_ortho: str = ""
    dig_georef_seed_x: float = 0.0
    dig_georef_seed_y: float = 0.0
    dig_georef_radius: float = 10.0
    # Whole-survey check: a photo->coordinate list plus a folder of quadrats.
    dig_seed_csv: str = ""
    dig_photo_dir: str = ""
    dig_batch_rows: list = field(default_factory=list)
    dig_batch_msg: str = ""
    dig_batch_busy: bool = False
    # Pins: photo key -> (x, y, crs) in the ortho's CRS. A pin beats a row
    # in the seed list.
    dig_pins: dict = field(default_factory=dict)
    dig_pin_photo: str = ""
    dig_pin_msg: str = ""
    dig_georef_inset: float = 0.0
    dig_georef_ok: bool = False
    dig_georef_msg: str = ""
    dig_georef_overlay: str = ""      # data URI of the alignment preview
    # A quadrat GeoTIFF already knows where it is; only a plain photograph
    # needs matching.
    dig_src_needs_georef: bool = True
    dig_src_georef_note: str = ""
    dig_georef_match: object = None
    # Source-image GSD in m/pixel (auto-filled from a _GSD=...m filename).
    dig_resolution: float = 0.001
    dig_mode: str = "polygon"  # "polygon" | "circle" | "ellipse" | "select"
    # Completed clast records ({shape, params, mask, centroid_x/y, score,
    # label, origin: "hand" or the model's display name, edited: bool}).
    dig_records: list = field(default_factory=list)
    # In-progress shape clicks.
    dig_active_polygon: list = field(default_factory=list)
    dig_active_circle_center: list = field(default_factory=list)
    # Ellipse: clicks 1-2 are the major-axis endpoints (any rotation),
    # click 3 sets the minor axis by perpendicular distance.
    dig_active_ellipse_clicks: list = field(default_factory=list)
    # Drag-to-edit target: {"source": "active"} or {"source": "record",
    # "record_idx": N}. Cleared on commit, cancel, or mode switch.
    dig_drag_target: dict = field(default_factory=dict)
    dig_drag_active: bool = False
    # Set by mouseup after a real drag; the next click consumes it instead
    # of placing a vertex.
    dig_drag_consumed: bool = False
    dig_out_path: str = ""
    # Multi-file workflow; dig_per_image caches each image's records and
    # out_path, dig_autosave writes the CSV after every committed shape.
    dig_image_dir: str = ""
    dig_image_list: list = field(default_factory=list)
    dig_image_idx: int = -1
    dig_per_image: dict = field(default_factory=dict)
    dig_autosave: bool = True
    # Saved clasts are never put on the canvas without Load. A CSV path is
    # "loaded" while its saved clasts are on the canvas (Clear all unloads
    # it) and "owned" once this session loaded or wrote it: autosave only
    # writes over a saved CSV it owns, so an unloaded file is never replaced
    # by whatever happens to be on the canvas.
    dig_csv_loaded: set = field(default_factory=set)
    dig_csv_owned: set = field(default_factory=set)
    # CSS-scale zoom of the interactive image (1.0 = 1 image px per screen px).
    dig_zoom: float = 1.0
    # Marker / hit-zone scale, 1.0 = 0.6 % of the image diagonal; 0.2..3.0.
    dig_size_scale: float = 1.0
    # Confidence threshold used by the 'Detect' button.
    dig_detect_threshold: float = 0.70
    # Quadrat / Object scale: leave the quadrat frame out of a rectified
    # photograph (its thickness is in the rectification record).
    det_exclude_frame: bool = True
    # Frame thickness (cm) for a photograph whose record gives none.
    det_frame_fallback_cm: Optional[float] = None
    # ---- Digitize: "No GSD? Scale from an object" (engine: functions.gauge)
    # The section starts closed; the user opens it.
    dig_scale_open: bool = False
    # The canvas draws two-click scale segments instead of clasts.
    dig_scale_draw: bool = False
    # The scaling-object library as plain dicts {"name", "length", "unit"},
    # loaded from <project>/scaling_objects.json and edited in place.
    dig_scale_library: list = field(default_factory=list)
    # Scale segments per photograph: image name -> [{"p0", "p1", "object"}]
    # in image pixels. Not clast records; kept in <truth stem>.scale.json
    # beside the photograph's Digitize CSV and read back when it is opened.
    dig_scale_segments: dict = field(default_factory=dict)
    # The output unit: a functions.gauge.UNITS key in metric mode, the name
    # of an unknown object in custom mode ('' = the default).
    dig_scale_result_unit: str = ""
    # Detection scale label, a functions.gauge.DETECT_SCALES key.
    dig_detect_scale: str = "Auto"
    # The label set on the canvas: "truth", or a model's own set.
    dig_label_set: str = "truth"
    dig_detect_running: bool = False
    # (image, min_confidence, devicemode, devicenumber) -> (masks, scores);
    # None runs Mask R-CNN. Tests inject a stand-in.
    dig_mask_detector: object = None
    # The model runs whose proposals are on each photograph: image name ->
    # [functions.provenance.model_entry dicts], one per backend (the latest
    # run). Written to the provenance sidecar and the figure sidecar.
    dig_detect_runs: dict = field(default_factory=dict)
    dig_figures_running: bool = False

    # ---- Validate tab state ----
    val_detect_csv: str = ""
    val_truth_csv: str = ""
    # Quadrat navigator (mirrors Digitize's multi-file mode).
    val_truth_dir: str = ""
    val_truth_files: list = field(default_factory=list)
    val_truth_index: int = 0
    # Field to compare (a column present in both CSVs).
    val_field: str = "Clast_length"
    # Spatial matching tolerance in metres; 0 = auto (half median NN dist).
    val_tolerance: float = 0.0
    # Also reject pairs whose sizes differ by more than 50 %.
    val_size_filter: bool = False
    # GSD of each CSV's coordinate frame (m/pixel if x/y are in pixels, 1.0
    # if already metric), so two GSDs still pair in a common metric frame.
    val_truth_gsd: float = 0.001
    val_detect_gsd: float = 0.001
    # Whether the size field values are in pixels and need GSD scaling.
    val_truth_size_in_px: bool = False
    val_detect_size_in_px: bool = False
    # Optional source image (usually the quadrat truth GeoTIFF) to draw
    # the validation overlay on.
    val_src_image: str = ""
    # Optional UAV ortho drawn beneath it, the truth on top at val_truth_alpha.
    val_uav_image: str = ""
    val_truth_alpha: float = 0.55

    # ---- detection-limit truncation (Soloy et al. 2020) ----
    # The UAV detector cannot resolve grains below k·GSD on the long axis
    # (4 cm at 5 mm/pixel, k = 8, in Soloy et al., Remote Sensing 12(21):3659).
    # Distribution comparisons run on the conditional P(D | D >= D_min) on
    # both sides.
    val_dmin_apply: bool = True
    val_dmin_auto: bool = True
    val_dmin_k_pixels: int = 8     # from Soloy et al. 2020
    val_dmin_uav_gsd: float = 0.005  # 5 mm/pixel default (Soloy 2020 case)
    val_dmin_m: float = 0.040      # 40 mm = 8 × 5 mm
    # When True, D_min = max(k·GSD, smallest observed UAV detection).
    val_dmin_floor_to_observed: bool = False
    # Queued validation jobs: a snapshot of every val_* field plus
    # status: 'pending'|'running'|'done'|'error', metrics, error.
    val_jobs: list = field(default_factory=list)
    # Skip spatial pairing and run only the distribution comparison (for
    # truth and detection CSVs from different scenes).
    val_statistical_only: bool = False

    # ---- quadrat-aware validation ----
    # Footprint both CSVs are clipped to before comparison:
    #   geotiff_aligned : the truth GeoTIFF's extent is the quadrat.
    #   centroid_dims   : centroid (val_quad_cx, val_quad_cy) + dims.
    #   point_corner    : recorded point + its compass corner + dims; the
    #                     footprint is a disc of radius half-diagonal.
    val_quad_mode: str = "geotiff_aligned"
    val_quad_cx: float = 0.0
    val_quad_cy: float = 0.0
    val_quad_px: float = 0.0
    val_quad_py: float = 0.0
    val_quad_width: float = 1.0
    val_quad_height: float = 1.0
    val_quad_corner: str = "NW"   # one of N/NE/E/SE/S/SW/W/NW
    # Where each GSD came from (GeoTransform / filename / manual).
    val_truth_gsd_source: str = "manual"
    val_detect_gsd_source: str = "manual"

    # ---- Quadrat extraction ----
    # Crop a UAV ortho and its detection CSV to a field quadrat, given as a
    # corner pair or centroid+dimensions, in the ortho's native CRS units.
    quad_ortho: str = ""
    quad_csv: str = ""
    quad_mode: str = "corners"          # 'corners' or 'centroid'
    quad_x0: float = 0.0
    quad_y0: float = 0.0
    quad_x1: float = 1.0
    quad_y1: float = 1.0
    quad_cx: float = 0.5
    quad_cy: float = 0.5
    quad_width: float = 1.0
    quad_height: float = 1.0
    quad_label: str = "1"
    quad_out_dir: str = ""

    # ---- Report tab state ----
    # Cover-page metadata, persisted to a per-project JSON next to the PDF.
    report_author: str = ""
    report_affiliation: str = ""
    report_description: str = ""
    # Empty = auto (first publication map, else first figure, else first
    # source image).
    report_cover_image: str = ""
    # Items: {"path", "title", "description"}.
    report_illustrations: list = field(default_factory=list)
    report_out_path: str = ""
    report_sections: dict = field(default_factory=lambda: {
        "cover":                    True,
        "toc":                      True,
        "spatial_maps":             True,
        "zonal_statistics":         True,
        "validation":               True,
        "appendix_logs":            True,
        "appendix_samples":         True,
        "appendix_field_reference": True,
    })

    # Detect tab
    det_mode: str = ORTHO
    det_dir: str = ""
    det_files: List[str] = field(default_factory=list)
    det_files_checked: dict = field(default_factory=dict)
    det_model: str = "maskrcnn"   # selected detection backend (detectors.registry)
    det_resolution: float = 0.001
    det_metric_cropsize: float = 1.0
    det_min_conf_enabled: bool = True
    det_min_confidence: float = 0.7
    det_overlap: float = 0.20
    det_dedup_overlap: float = 0.30
    # Brightness / nodata tile filter (Ortho mode only); 0 (dark) and 255
    # (bright) mean "disabled".
    det_brightness_filter: bool = False   # off by default (manual, tooltip)
    det_dark_threshold: int = 15        # 0 = disabled
    det_bright_threshold: int = 245     # 255 = disabled
    det_nodata_max_frac: float = 0.95
    # Default follows the mode: Quadrat seeds True (the ellipse overlay
    # is that workflow's product), Ortho seeds False.
    det_saveplot: bool = False
    det_saveresults: bool = True
    det_kstart: int = 0
    # Most recent *_overlay.png produced (Quadrat only; *_ellipses.png before the rename).
    det_last_preview: str = ""
    # det_stop_all halts the queue; det_stop_current skips to the next job.
    det_stop_all: bool = False
    det_stop_current: bool = False
    # Queued detection jobs, an (image, params) snapshot each:
    #   {image_path, image_name, mode, resolution, metric_cropsize,
    #    min_confidence, overlap, dedup_overlap, saveplot, kstart,
    #    status: 'pending'|'running'|'done'|'error'|'stopped',
    #    n_clasts: int|None, error: str|None}
    det_jobs: list = field(default_factory=list)
    # Per-image ROI canvas (both modes; the field keeps its historical
    # name). Saves <image_stem>_roi.geojson next
    # to the image; the path is captured into each job's snapshot.
    det_uav_roi_canvas: CanvasContext = field(
        default_factory=CanvasContext)
    # Detect's scale-segment canvas (Quadrat): one photograph at a time,
    # its segments saved to the same <stem>_truth.scale.json Digitize writes.
    det_scale_canvas: CanvasContext = field(default_factory=CanvasContext)
    # The scaling object the batch pass assigns to the segments it draws.
    det_scale_object: str = ""

    # Zonal tab source canvas (zones / transects); the saved zone-set path is
    # zonal_zones_canvas.vector.
    zonal_zones_canvas: CanvasContext = field(
        default_factory=CanvasContext)

    # Rasterize tab
    ras_csv: str = ""
    ras_tif: str = ""
    ras_field: str = "Clast_length"
    ras_parameter: str = "quantile"
    ras_cellsize: float = 1.0
    # Cells with fewer clasts than this are NaN'd; 0 keeps every non-empty cell.
    ras_min_cell_density: int = 0
    # Empty = the active project's results/rasters/ folder.
    ras_output_dir: str = ""
    ras_percentile: float = 0.5
    ras_T: float = 10.0
    ras_rho_water: float = 1025.0
    ras_out: str = ""
    # Job queue: 'Add to queue' stages the cartesian product of the checked
    # fields and parameters; entries are {field, parameter, percentile, T}.
    ras_jobs: list = field(default_factory=list)
    ras_fields_checked: dict = field(default_factory=dict)
    ras_parameters_checked: dict = field(default_factory=dict)
    # Size bins for packing_index / packing_clustering: 'fixed' (mm) or
    # 'percentile' (0-1 quantiles of bin_field); edges parsed at Add-time.
    ras_bin_mode: str = "fixed"
    # Wentworth doublings in the pebble range (very fine pebble .. small cobble).
    ras_bin_edges_text: str = "8, 16, 32, 64, 128"
    ras_bin_field: str = "Clast_length"

    # Merge tab
    # N >= 2 CSVs, {path, window_size}; window size is read from the
    # '_ws<size>m' filename token when possible. The merge runs from the
    # largest window down, so those detections win on conflicts.
    merge_csvs_list: list = field(default_factory=lambda: [
        {"path": "", "window_size": None},
        {"path": "", "window_size": None},
    ])
    merge_out: str = ""
    merge_method: str = "iou"
    merge_overlap: float = 0.30
    # Snapshots of the form (csvs, out_path, method, overlap, n_points) + status.
    merge_jobs: list = field(default_factory=list)
    merge_n_points: int = 32

    # Map tab
    map_ortho: str = ""
    map_csv: str = ""
    map_raster: str = ""
    map_layer: str = "vector"          # 'vector' or 'raster'
    map_field: str = "Clast_length"
    map_cmap: str = "viridis"   # perceptually-uniform default (turbo still selectable)
    map_basemap: str = "ESRI World Imagery (satellite)"
    map_show_ortho: bool = True
    map_ortho_alpha: float = 1.0   # source ortho, 1.0 = opaque
    map_raster_alpha: float = 0.7  # data raster (raster mode only)
    map_point_size: float = 8.0    # vector mode only
    # 'linear' (5th/95th percentile) or 'quantile' (BoundaryNorm).
    map_color_scale: str = "quantile"
    map_vmin_auto: bool = True
    map_vmax_auto: bool = True
    map_vmin: float = 0.0
    map_vmax: float = 1.0
    # Display units only ('metric' | 'imperial'); the size unit is auto-picked
    # from the data magnitude unless map_size_unit overrides it.
    map_unit_system: str = "metric"
    map_size_unit: str = "auto"            # 'auto', 'm', 'cm', 'mm', 'ft', 'in'
    map_show_grid: bool = True
    map_show_legend: bool = True
    map_show_crs: bool = True
    map_show_north: bool = True
    map_show_scale: bool = True
    map_show_zebra: bool = True
    # Colorbar extend='both' caps (meaningless for cyclic colormaps).
    map_show_colorbar_extends: bool = True
    map_title: str = ""
    map_dpi: int = 300
    map_out: str = ""

    # Deep-copied form snapshots staged for one batch.
    map_jobs: list = field(default_factory=list)

    # Zonal-stats tab. Polygon mode summarises the per-clast CSV directly
    # (count, density, moments, Folk-Ward phi statistics, percentiles);
    # transect mode samples a raster. zonal_dem is sampled in both modes.
    zonal_csv: str = ""
    zonal_raster: str = ""
    zonal_dem: str = ""
    zonal_field: str = "Clast_length"
    zonal_mode: str = "polygons"       # 'polygons' or 'transects'
    zonal_view: str = "polygons"       # 'polygons' | 'transects' | 'profile'
    # A zone/transect set is applied to several orthos and dates, so it is
    # identified by this name and not by whichever image is loaded.
    zonal_set_name: str = ""
    zonal_band: int = 1
    zonal_id_field: str = ""
    # The full Folk-Ward percentile family by default.
    zonal_percentiles_text: str = "5, 16, 25, 50, 75, 84, 95"
    zonal_step_m: float = 0.5
    zonal_interpolation: str = "bilinear"
    zonal_out_csv: str = ""
    # Empty = the project's results/zonal/ folder.
    zonal_output_dir: str = ""
    zonal_jobs: list = field(default_factory=list)
    # Generic combined-plot composer state, kept for persisted layouts; the
    # recipe-based composers use the fields below.
    zonal_plot_layers: list = field(default_factory=list)
    zonal_plot_title: str = ""
    zonal_plot_primary_label: str = ""
    zonal_plot_secondary_label: str = ""
    zonal_plot_out_png: str = ""
    # Profile figure: bin width of the binned table and the caption.
    zonal_profile_bin_m: float = 1.0
    zonal_profile_caption: str = ""
    # Overlay recipe: [{csv: str}, ...] so the row UI can mutate in place.
    zonal_overlay_csvs: list = field(default_factory=list)
    # Envelope+topo recipe.
    zonal_etopo_csv: str = ""
    zonal_etopo_distance_field: str = "distance_m"
    zonal_etopo_value_field: str = "raster_value"
    zonal_etopo_elevation_field: str = "elevation_m"
    zonal_etopo_envelope_lower: str = ""
    zonal_etopo_envelope_upper: str = ""
    zonal_plot_field_name: str = "Clast_length"
    # Most recent rendered plot, embedded below the Render button.
    zonal_plot_last_png: str = ""
    # Most recent zonal run, shown inline: {mode, out_csv, pngs (transect
    # quick-looks), overlay_png, error}.
    zonal_last_result: dict = field(default_factory=dict)
    # Scatter / violin axis selections, persisted across refreshable re-renders.
    zonal_scatter_x: str = ""
    zonal_scatter_y: str = ""
    zonal_scatter_type: str = "scatter"
    # Base-64 PNG of the last scatter/violin, kept in state so a refreshable
    # re-render never holds a stale element reference.
    zonal_scatter_img: str = ""


state = AppState()


# --- Path helpers and file-picker dialogs ---
def native_dir_picker(title: str = "Select directory",
                      initialdir: str = "") -> str:
    """Open an OS-native folder picker via tkinter. Returns '' on cancel."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title, parent=root,
                                       initialdir=initialdir or None)
        root.destroy()
        return path or ""
    except Exception as e:
        ui.notify(f"Directory picker unavailable: {e} — type the folder path into the field instead.", type="warning")
        return ""


def native_file_picker(title: str = "Select file", filetypes=None,
                       initialdir: str = "") -> str:
    """Open an OS-native file picker via tkinter. Returns '' on cancel."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title, parent=root,
            filetypes=filetypes or [("All files", "*.*")],
            initialdir=initialdir or None,
        )
        root.destroy()
        return path or ""
    except Exception as e:
        ui.notify(f"File picker unavailable: {e} — type the file path into the field instead.", type="warning")
        return ""


def native_save_file_picker(title: str = "Save file as",
                            filetypes=None,
                            initialfile: str = "",
                            defaultextension: str = "",
                            initialdir: str = "") -> str:
    """Open an OS-native save-as dialog via tkinter. Returns '' on cancel."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title=title, parent=root,
            filetypes=filetypes or [("All files", "*.*")],
            initialfile=initialfile,
            defaultextension=defaultextension,
            initialdir=initialdir or None,
        )
        root.destroy()
        return path or ""
    except Exception as e:
        ui.notify(f"Save dialog unavailable: {e} — type the output path into the field instead.", type="warning")
        return ""


def list_images(directory: str, extensions) -> list:
    """Sorted list of files in `directory` matching any of `extensions`."""
    if not directory or not os.path.isdir(directory):
        return []
    exts = tuple(e.lower() for e in extensions)
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(exts))


# --- Content-addressed raster route ---
# The alignment editor re-sends its SVG overlay on every mouse move; embedding
# the quadrat PNG as a data: URI pushed hundreds of KB per move down the
# websocket the heartbeat uses and could drop the client. Served by URL, the
# browser fetches each raster once; the name is a content hash, so the URL
# changes exactly when the pixels do and the response can be immutable.
_RASTER_ROUTE = "/pm-raster"
_RASTER_KEEP = 24            # bounded cache, a handful of quadrats' worth
from collections import OrderedDict as _OrderedDict     # noqa: E402
_RASTER_CACHE = _OrderedDict()


# Registered at import: a route added to the router after startup is not a
# supported contract.
@app.get(_RASTER_ROUTE + "/{name}")
def _serve_raster(name: str):                  # noqa: ANN202  (FastAPI route)
    from fastapi import Response
    data = _RASTER_CACHE.get(name)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control":
                             "public, max-age=31536000, immutable"})


def _raster_url(png_bytes: bytes) -> str:
    """Publish PNG bytes at a content-addressed URL and return it."""
    import hashlib
    name = hashlib.sha1(png_bytes).hexdigest()[:16] + ".png"
    _RASTER_CACHE[name] = png_bytes
    _RASTER_CACHE.move_to_end(name)
    while len(_RASTER_CACHE) > _RASTER_KEEP:
        _RASTER_CACHE.popitem(last=False)
    return f"{_RASTER_ROUTE}/{name}"


def _fig_to_image_url(fig, dpi: int = 100) -> str:
    """Render a matplotlib Figure to a base64 PNG data URL for ui.image().

    Safe to call from worker threads, unlike ui.matplotlib().
    """
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Per-clast field -> (display unit, multiplier) lives in functions/units.py,
# shared with the report builder.
from functions.units import (
    field_unit_and_factor as _field_display_unit,
    format_value_with_unit as _format_value_with_unit,
)


def _safe_relerr(detect, truth):
    """Signed relative error (detect - truth) / truth, or None when undefined."""
    try:
        if detect is None or truth is None:
            return None
        d, t = float(detect), float(truth)
        if not (t == t) or t == 0:  # NaN check via self-equality
            return None
        return (d - t) / t
    except (TypeError, ValueError):
        return None


def _build_validation_figures(metrics, dist, paired, pair_res,
                               truth_df, detect_df, field,
                               src_image=None, truth_gsd=1.0,
                               uav_image=None, truth_alpha=0.55,
                               d_min_m=None):
    """Build every validation diagnostic figure once.

    Returns {short_name: matplotlib.Figure}; the caller closes them. Both the
    on-screen renderer and the disk-saving routine use this, so the report
    matches the live view.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figs = {}

    unit, factor = _field_display_unit(field)

    def _fld(value=None, label=None):
        """Axis label, e.g. _fld(label='Truth') -> 'Truth (mm)'."""
        return f"{label or field} ({unit})"

    t_v_raw_all = truth_df[field].to_numpy()
    d_v_raw_all = detect_df[field].to_numpy()
    # The full arrays feed the CCDF (which shows the divergence below
    # D_min on purpose); histogram / CDF / QQ use the D_min-filtered ones.
    t_v_raw = t_v_raw_all
    d_v_raw = d_v_raw_all

    from functions.units import is_size_field_for_phi as _is_size_for_dmin
    apply_dmin = (d_min_m is not None and d_min_m > 0
                  and _is_size_for_dmin(field))
    t_below = d_below = 0
    if apply_dmin:
        t_mask = np.isfinite(t_v_raw_all) & (t_v_raw_all >= d_min_m)
        d_mask = np.isfinite(d_v_raw_all) & (d_v_raw_all >= d_min_m)
        t_below = int((np.isfinite(t_v_raw_all)
                       & (t_v_raw_all < d_min_m)).sum())
        d_below = int((np.isfinite(d_v_raw_all)
                       & (d_v_raw_all < d_min_m)).sum())
        t_v_raw = t_v_raw_all[t_mask]
        d_v_raw = d_v_raw_all[d_mask]

    t_v = t_v_raw * factor
    d_v = d_v_raw * factor
    all_v = np.concatenate([t_v, d_v])
    # Orientation may be negative ([-90, 90]); sizes must be positive.
    is_size = unit in ("mm", "mm²", "m", "m/s")
    if is_size:
        all_v = all_v[np.isfinite(all_v) & (all_v > 0)]
    else:
        all_v = all_v[np.isfinite(all_v)]

    if apply_dmin:
        dmin_mm = d_min_m * 1000.0
        title_suffix = (f"  (filtered: D ≥ {dmin_mm:.2f} mm; "
                        f"{t_below} truth + {d_below} detect points "
                        f"below threshold excluded)")
    else:
        title_suffix = ""

    # --- Histogram ---
    # With D_min active the excluded tail is drawn in faded grey behind the
    # kept bins; bin edges span the full range so the two layers align.
    fig, ax = plt.subplots(figsize=(5, 4))
    if apply_dmin:
        t_full_arr = (dist.get("_truth_df_full")[field].to_numpy() * factor
                      if (dist.get("_truth_df_full") is not None) else t_v)
        d_full_arr = (dist.get("_detect_df_full")[field].to_numpy() * factor
                      if (dist.get("_detect_df_full") is not None) else d_v)
        bin_v = np.concatenate([t_full_arr, d_full_arr])
        if is_size:
            bin_v = bin_v[np.isfinite(bin_v) & (bin_v > 0)]
        else:
            bin_v = bin_v[np.isfinite(bin_v)]
    else:
        bin_v = all_v
    if len(bin_v) > 0:
        bins = np.linspace(bin_v.min(), bin_v.max(), 30)
        if apply_dmin:
            t_excl = t_full_arr[t_full_arr < d_min_m * factor] \
                if d_min_m else np.array([])
            d_excl = d_full_arr[d_full_arr < d_min_m * factor] \
                if d_min_m else np.array([])
            if len(t_excl) > 0:
                ax.hist(t_excl, bins=bins, alpha=0.35,
                        color="#888888", density=True,
                        label=f"Truth < D_min ({len(t_excl)}, excluded)")
            if len(d_excl) > 0:
                ax.hist(d_excl, bins=bins, alpha=0.35,
                        color="#aaaaaa", density=True,
                        label=f"Detect < D_min ({len(d_excl)}, excluded)")
            ax.axvline(d_min_m * factor, color="grey",
                       linestyle="--", linewidth=1.0)
        ax.hist(t_v, bins=bins, alpha=0.5,
                label=f"Truth ({len(t_v)})",
                color="#2a7fff", density=True)
        ax.hist(d_v, bins=bins, alpha=0.5,
                label=f"Detection ({len(d_v)})",
                color="#ff7f2a", density=True)
        ax.set_xlabel(_fld())
        ax.set_ylabel("Density")
        ax.set_title(f"{field} — histogram{title_suffix}",
                     fontsize=10)
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["hist"] = fig

    # --- CDF ---
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(t_v) > 0 and len(d_v) > 0:
        if is_size:
            t_sorted = np.sort(t_v[np.isfinite(t_v) & (t_v > 0)])
            d_sorted = np.sort(d_v[np.isfinite(d_v) & (d_v > 0)])
        else:
            t_sorted = np.sort(t_v[np.isfinite(t_v)])
            d_sorted = np.sort(d_v[np.isfinite(d_v)])
        ax.plot(t_sorted, np.linspace(0, 1, len(t_sorted)),
                color="#2a7fff", label="Truth", lw=2)
        ax.plot(d_sorted, np.linspace(0, 1, len(d_sorted)),
                color="#ff7f2a", label="Detection", lw=2)
        ax.set_xlabel(_fld())
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"{field} — CDF{title_suffix}", fontsize=10)
        ax.legend()
        ax.axhline(0.5, color="grey", lw=0.5, ls="--")
        ax.axhline(0.84, color="grey", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["cdf"] = fig

    # --- Q-Q plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    qt = dist.get("qq_truth_quantiles")
    qd = dist.get("qq_detect_quantiles")
    if qt is not None and qd is not None and len(qt) > 0 and len(qd) > 0:
        qt_disp = np.asarray(qt) * factor
        qd_disp = np.asarray(qd) * factor
        lo = min(qt_disp.min(), qd_disp.min())
        hi = max(qt_disp.max(), qd_disp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label="1:1")
        ax.scatter(qt_disp, qd_disp, color="#5a6878", s=12)
        ax.set_xlabel(_fld(label="Truth quantile"))
        ax.set_ylabel(_fld(label="Detection quantile"))
        ax.set_title(f"{field} — Q-Q plot{title_suffix}",
                      fontsize=10)
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["qq"] = fig

    # --- Paired scatter + Bland-Altman (only if paired stats exist) ---
    if paired and paired.get("n", 0) >= 2 and pair_res["matched_pairs"]:
        t_paired_raw = truth_df[field].iloc[
            [p[0] for p in pair_res["matched_pairs"]]].to_numpy()
        d_paired_raw = detect_df[field].iloc[
            [p[1] for p in pair_res["matched_pairs"]]].to_numpy()
        # Same angle-wrap as `_compute_validation`, so the points match
        # the RMSE / bias in the title.
        if "orientation" in field.lower():
            t_paired_raw = np.mod(t_paired_raw, 180.0)
            d_paired_raw = np.mod(d_paired_raw, 180.0)
            diff = d_paired_raw - t_paired_raw
            d_paired_raw = np.where(diff >  90.0, d_paired_raw - 180.0, d_paired_raw)
            d_paired_raw = np.where(diff < -90.0, d_paired_raw + 180.0, d_paired_raw)
        t_paired_arr = t_paired_raw * factor
        d_paired_arr = d_paired_raw * factor

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(t_paired_arr, d_paired_arr,
                   color="#5a6878", s=15, alpha=0.6)
        lo = min(t_paired_arr.min(), d_paired_arr.min())
        hi = max(t_paired_arr.max(), d_paired_arr.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label="1:1")
        xs = np.linspace(lo, hi, 100)
        # Slope is dimensionless; intercept needs the same display factor.
        ys = paired["slope"] * xs + paired["intercept"] * factor
        ax.plot(xs, ys, color="#ff7f2a",
                label=f"y = {paired['slope']:.2f}x + "
                      f"{paired['intercept']*factor:.3f} {unit}")
        ax.set_xlabel(_fld(label="Truth"))
        ax.set_ylabel(_fld(label="Detection"))
        ax.set_title(
            f"{field} — paired (R² = {paired['r_squared']:.3f}, "
            f"RMSE = {paired['rmse']*factor:.2f} {unit}, "
            f"MAE = {paired.get('mae', 0)*factor:.2f} {unit})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["paired_scatter"] = fig

        # Bland-Altman in the field's display unit.
        ba_mean = paired["bland_altman_mean"] * factor
        ba_diff = paired["bland_altman_diff"] * factor
        bias_d = paired["bias"] * factor
        loa_hi_d = paired["bland_altman_loa_hi"] * factor
        loa_lo_d = paired["bland_altman_loa_lo"] * factor
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(ba_mean, ba_diff, color="#5a6878", s=15, alpha=0.6)
        ax.axhline(bias_d, color="#ff7f2a",
                   label=f"bias = {bias_d:+.2f} {unit}")
        ax.axhline(loa_hi_d, color="#ff7f2a", linestyle="--",
                   label=f"+1.96σ = {loa_hi_d:+.2f} {unit}")
        ax.axhline(loa_lo_d, color="#ff7f2a", linestyle="--",
                   label=f"-1.96σ = {loa_lo_d:+.2f} {unit}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel(_fld(label="Mean of pair"))
        ax.set_ylabel(f"Detection − Truth ({unit})")
        ax.set_title(f"{field} — Bland-Altman")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["bland_altman"] = fig

    # --- CCDF (log-log) + detection function ---
    # The CCDF shows the divergence below D_min without bin artefacts; the
    # detection function is the empirical UAV recall per size bin.
    try:
        from functions import truncation as _trunc
        t_vals = truth_df[field].to_numpy(dtype=float)
        d_vals = detect_df[field].to_numpy(dtype=float)
        # A log axis needs positive linear sizes.
        from functions.units import is_size_field_for_phi
        if is_size_field_for_phi(field):
            fig_ccdf = _trunc.plot_ccdf_log_log(
                t_vals, d_vals, field_name=field, d_min=d_min_m)
            figs["ccdf"] = fig_ccdf
        if pair_res.get("matched_pairs"):
            matched_truth = [int(p[0]) for p in pair_res["matched_pairs"]]
            fit = _trunc.detection_function_from_pair(
                t_vals, matched_truth, n_bins=10)
            if fit.bin_sizes.size:
                fig_det = _trunc.plot_detection_function(
                    fit, d_min=d_min_m, field_name=field)
                figs["detection_function"] = fig_det
    except Exception as _ex:
        try:
            print(f"[truncation plots] skipped: {_ex}")
        except Exception:
            pass

    # --- Spatial pattern of matches & misses ---
    fig, ax = plt.subplots(figsize=(10, 8))
    if pair_res["unmatched_truth"]:
        u_t = truth_df.iloc[pair_res["unmatched_truth"]]
        ax.scatter(u_t["x"], u_t["y"], marker='x', color='red', s=60,
                   label=f"FN: missed truth ({len(u_t)})", linewidths=2)
    if pair_res["unmatched_detect"]:
        u_d = detect_df.iloc[pair_res["unmatched_detect"]]
        ax.scatter(u_d["x"], u_d["y"], marker='^', color='#ff7f2a', s=40,
                   label=f"FP: spurious detect ({len(u_d)})",
                   edgecolors='black', linewidths=0.5)
    if pair_res["matched_pairs"]:
        m_t = truth_df.iloc[[p[0] for p in pair_res["matched_pairs"]]]
        ax.scatter(m_t["x"], m_t["y"], marker='o', color='#2eb872',
                   s=20, alpha=0.6, label=f"Matched ({len(m_t)})")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Validation map — {field}")
    fig.tight_layout()
    figs["spatial"] = fig

    # --- Image overlay ---
    # The UAV ortho (if any) is drawn first at full opacity, the truth image
    # on top at `truth_alpha`. Each image's extent comes from its own
    # GeoTransform, else from pixels x GSD; after the GSD scaling in
    # `_compute_validation` the CSV x/y are in the same world units.

    def _load_image_with_extent(path, gsd_fallback):
        """Return (rgb_array, extent, origin, is_world_coords, affine);
        origin follows the y-axis convention of the extent source. The
        extent of a georeferenced raster is the box of its four corners;
        a rotated one (every placed quadrat) also carries the affine that
        draws it in place — from the origin and the far corner alone the
        box of a 40° quadrat was a strip a tenth as high, and the image
        was stretched into it unrotated."""
        from PIL import Image as _PILImage
        with _PILImage.open(path) as _pim:
            arr = np.asarray(_pim.convert("RGB"))
        H_, W_ = arr.shape[:2]
        world, affine = None, None
        try:
            if Path(path).suffix.lower() in (".tif", ".tiff"):
                from osgeo import gdal as _gdal
                _ds = _gdal.Open(str(path))
                if _ds is not None:
                    _gt = _ds.GetGeoTransform()
                    _ds = None
                    if (_gt is not None and
                        _gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
                        from functions.quadrat_validation import raster_world_bounds
                        xmin, ymin, xmax, ymax = raster_world_bounds(_gt, W_, H_)
                        world = (xmin, xmax, ymin, ymax)
                        if abs(_gt[2]) > 1e-12 or abs(_gt[4]) > 1e-12:
                            from matplotlib.transforms import Affine2D
                            # pixel (col, row) -> world, GDAL's convention
                            affine = Affine2D.from_values(
                                _gt[1], _gt[4], _gt[2], _gt[5], _gt[0], _gt[3])
        except Exception:
            world, affine = None, None
        if world is not None:
            return arr, world, "upper", True, affine
        ext = (0.0, W_ * gsd_fallback,
               0.0, H_ * gsd_fallback)
        return arr, ext, "lower", False, None

    def _draw_layer(ax, layer, **kw):
        """imshow a layer; a rotated raster is drawn in pixel space and
        mapped to the world through its affine."""
        arr, ext, origin, _wc, affine = layer
        if affine is None:
            ax.imshow(arr, extent=ext, origin=origin, aspect="equal", **kw)
            return
        H_, W_ = arr.shape[:2]
        im = ax.imshow(arr, extent=(0, W_, H_, 0), origin="upper",
                       aspect="equal", **kw)
        im.set_transform(affine + ax.transData)

    if (src_image and Path(src_image).exists()) or \
            (uav_image and Path(uav_image).exists()):
        try:
            gsd = truth_gsd or 1.0
            fig_o, ax_o = plt.subplots(figsize=(12, 9))

            uav_layer = None
            truth_layer = None
            if uav_image and Path(uav_image).exists():
                uav_layer = _load_image_with_extent(uav_image, gsd)
            if src_image and Path(src_image).exists():
                truth_layer = _load_image_with_extent(src_image, gsd)

            if uav_layer is not None:
                _draw_layer(ax_o, uav_layer, zorder=1)
            if truth_layer is not None:
                # Alpha only when the UAV layer is under it; never invisible.
                alpha = (float(truth_alpha)
                         if uav_layer is not None else 1.0)
                alpha = max(0.05, min(1.0, alpha))
                _draw_layer(ax_o, truth_layer, alpha=alpha, zorder=2)

            ref_layer = truth_layer or uav_layer
            ref_ext = ref_layer[1]
            world_coords = ref_layer[3]
            ax_o.set_xlim(ref_ext[0], ref_ext[1])
            ax_o.set_ylim(ref_ext[2], ref_ext[3])
            axes_label_units = (
                "(world coords)" if world_coords
                else f"(m, GSD={gsd*1000:.3f} mm/px)")

            # Mask outlines, when the CSVs carry a contour file: truth in
            # green, detections in orange, under the markers. Outlines only:
            # the frames' lengths are not in these axes' units.
            try:
                from functions import clast_geometry as _CG
                for _df, _col in ((truth_df, "#2eb872"), (detect_df, "#ff7f2a")):
                    _cs = _CG.contours_of(_df)
                    if _cs:
                        _CG.draw_clasts(ax_o, _df, contours=_cs,
                                        y_down=False, chords=False,
                                        centres=False, note=False,
                                        edge_colors=_col, face_alpha=0.45,
                                        outline_width=0.8, zorder=3)
            except Exception as _oex:
                print(f"[validation overlay] outlines skipped: {_oex}")

            if pair_res["matched_pairs"]:
                for ti, di in pair_res["matched_pairs"]:
                    tx_, ty_ = truth_df.iloc[ti][["x", "y"]]
                    dx_, dy_ = detect_df.iloc[di][["x", "y"]]
                    ax_o.plot([tx_, dx_], [ty_, dy_],
                              color='yellow', lw=0.5, alpha=0.6,
                              zorder=3)
                m_t = truth_df.iloc[
                    [p[0] for p in pair_res["matched_pairs"]]]
                m_d = detect_df.iloc[
                    [p[1] for p in pair_res["matched_pairs"]]]
                ax_o.scatter(m_t["x"], m_t["y"], marker='o', s=40,
                             facecolor='#2eb872', edgecolor='white',
                             linewidths=1.5, zorder=4,
                             label=f"Matched truth ({len(m_t)})")
                ax_o.scatter(m_d["x"], m_d["y"], marker='o', s=15,
                             facecolor='#a8e0c2', edgecolor='black',
                             linewidths=0.5, zorder=4)
            if pair_res["unmatched_truth"]:
                u_t = truth_df.iloc[pair_res["unmatched_truth"]]
                ax_o.scatter(u_t["x"], u_t["y"], marker='x',
                             color='red', s=80, linewidths=3, zorder=5,
                             label=f"FN: missed truth ({len(u_t)})")
            if pair_res["unmatched_detect"]:
                u_d = detect_df.iloc[pair_res["unmatched_detect"]]
                ax_o.scatter(u_d["x"], u_d["y"], marker='^', s=60,
                             facecolor='#ff7f2a', edgecolor='white',
                             linewidths=1.5, zorder=5,
                             label=f"FP: spurious detect ({len(u_d)})")
            ax_o.set_xlabel(f"x {axes_label_units}")
            ax_o.set_ylabel(f"y {axes_label_units}")
            ax_o.legend(loc='best', framealpha=0.85)
            if uav_layer and truth_layer:
                title = (
                    f"Image overlay — ortho {Path(uav_image).name} (bg) "
                    f"+ truth {Path(src_image).name} (α={truth_alpha:.2f})"
                )
            elif uav_layer:
                title = f"Image overlay — ortho {Path(uav_image).name}"
            else:
                title = f"Image overlay — {Path(src_image).name}"
            if world_coords:
                title += " · world coords"
            else:
                title += f" · GSD={gsd*1000:.3f} mm/px"
            ax_o.set_title(title)
            fig_o.tight_layout()
            figs["overlay"] = fig_o
        except Exception as _ex:
            try:
                print(f"[validation overlay] skipped: {_ex}")
            except Exception:
                pass

    return figs


# Windows MAX_PATH (260) still binds matplotlib and Pillow; check a few
# characters under it so the user gets a clear message before savefig fails.
_WIN_MAX_PATH = 250


def _save_validation_figures(figs, out_dir, stem, log=None):
    """Save each fig in ``figs`` to ``<out_dir>/<stem>_<name>.png``.

    Returns ``{name: Path}``. Errors are logged, never raised. On Windows an
    over-long path shortens the stem (and warns) before savefig.
    """
    import os as _os
    out_paths = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    win = _os.name == "nt"
    for name, fig in figs.items():
        p = out_dir / f"{stem}_{name}.png"
        if win and len(str(p)) > _WIN_MAX_PATH:
            extra = len(str(p)) - _WIN_MAX_PATH
            short_stem = stem[: max(0, len(stem) - extra - 1)]
            if not short_stem:
                if log:
                    log(f"  [warn] skipping figure {name}: target path "
                        f"({len(str(p))} chars) exceeds Windows "
                        f"MAX_PATH ({_WIN_MAX_PATH}) and the output "
                        f"directory alone already approaches the limit.")
                continue
            new_p = out_dir / f"{short_stem}_{name}.png"
            if log:
                log(f"  [warn] shortening figure stem to fit Windows "
                    f"MAX_PATH: '{stem}' -> '{short_stem}' "
                    f"({len(str(p))}/{ _WIN_MAX_PATH} chars).")
            p = new_p
        try:
            fig.savefig(p, dpi=140, bbox_inches="tight")
            out_paths[name] = p
        except Exception as ex:
            if log:
                log(f"  [warn] could not save {p.name}: {ex}")
    return out_paths


def _persist_validation_result(job, summary, dist, paired, pair_res,
                                figure_paths=None):
    """Write a validation comparison to <project>/validation/results/ as JSON.

    The Report tab's `compute_aggregate_stats` reads this folder.
    """
    import json as _json
    import time as _time
    from functions.report import _VALIDATION_SCHEMA_VERSION

    truth_p = Path(job["truth"])
    detect_p = Path(job["detect"])
    # Find the project's validation/ folder as a sibling of an ancestor of
    # the truth CSV, or as an ancestor itself. Refuse to save otherwise, so
    # results never scatter outside a project.
    out_dir = None
    for parent in truth_p.resolve().parents:
        cand = parent / "validation" / "results"
        if cand.parent.exists():
            out_dir = cand
            break
        if parent.name == "validation":
            out_dir = parent / "results"
            break
    if out_dir is None:
        import os as _os
        if _os.environ.get("CLAST_VALIDATION_ALLOW_LOOSE_SAVE") == "1":
            out_dir = truth_p.parent
        else:
            try:
                ui.notify(
                    f"Validation result not saved: truth CSV "
                    f"{truth_p.name} is not inside a project (no "
                    f"'validation/' folder found in any ancestor). "
                    f"Move the truth CSV into "
                    f"<project>/validation/ and rerun, or set "
                    f"CLAST_VALIDATION_ALLOW_LOOSE_SAVE=1 to drop the "
                    f"JSON next to the truth file.",
                    type="warning", multi_line=True, timeout=8000)
            except Exception:
                pass
            raise FileNotFoundError(
                f"Cannot identify project for truth CSV {truth_p!s}: "
                f"no 'validation/' folder in any ancestor directory.")
    # One subfolder per detection CSV (validation/results/<detect_stem>/),
    # capped at 60 chars for Windows MAX_PATH.
    detect_subfolder = naming.image_stem(detect_p.name)
    if len(detect_subfolder) > 60:
        detect_subfolder = detect_subfolder[:60].rstrip("_")
    out_dir = out_dir / detect_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as _np

    def _jsonable(v, max_array_len=8):
        """JSON-safe view: numpy scalars to Python, NaN to None, bulk
        per-pair arrays dropped."""
        if v is None:
            return None
        if isinstance(v, (_np.floating, _np.integer, _np.bool_)):
            v = v.item()
        if isinstance(v, _np.ndarray):
            if v.size > max_array_len:
                return None
            return [_jsonable(x, max_array_len) for x in v.tolist()]
        if isinstance(v, float):
            return v if v == v else None  # filter NaN
        if isinstance(v, dict):
            return {k: _jsonable(x, max_array_len) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x, max_array_len) for x in v]
        return v

    # Per-pair plotting arrays and underscore-prefixed in-memory plumbing
    # (the unfiltered DataFrames under dist) are not part of the schema.
    paired_lite = {k: v for k, v in (paired or {}).items()
                   if k not in ("bland_altman_mean", "bland_altman_diff")}
    dist_lite = {k: v for k, v in (dist or {}).items()
                 if k not in ("qq_truth_quantiles",
                              "qq_detect_quantiles")
                 and not k.startswith("_")}

    payload = {
        "schema_version": _VALIDATION_SCHEMA_VERSION,
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "truth_csv": str(truth_p),
        "detect_csv": str(detect_p),
        "field": job.get("field"),
        "truth_gsd_m_per_px": job.get("truth_gsd"),
        "detect_gsd_m_per_px": job.get("detect_gsd"),
        "tolerance_m": pair_res.get("used_tolerance") if pair_res else None,
        "metrics": _jsonable(summary),
        "distribution": _jsonable(dist_lite),
        "paired": _jsonable(paired_lite),
        "figures": _jsonable(figure_paths or {}),
    }
    # Compact name for Windows MAX_PATH: short stem prefixes plus a hash of
    # the pair and the job parameters, so different parameters never overwrite.
    import hashlib as _hashlib
    hash_input = (f"{truth_p}|{detect_p}|{job.get('field')}|"
                  f"{job.get('truth_gsd')}|{job.get('detect_gsd')}|"
                  f"{job.get('tolerance')}")
    pair_hash = _hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]

    def _short(stem: str, n: int = 40) -> str:
        return stem if len(stem) <= n else stem[:n].rstrip("_") + "_"

    out_name = (f"{_short(truth_p.stem)}__vs__"
                f"{_short(detect_p.stem)}__{pair_hash}.validation.json")
    out_path = out_dir / out_name
    # Keep the .validation.json suffix: the Report tab's scanner filters on it.
    if len(str(out_path)) > 240:
        out_path = out_dir / f"validation_{pair_hash}.validation.json"
    out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# path -> data: URL, so checklist re-renders do not re-decode JPGs.
_thumbnail_cache: dict = {}


def _quadrat_thumbnail(image_path: str, size: int = 80) -> str:
    """Thumbnail data URL for a photo, '' on failure; cached by path."""
    if image_path in _thumbnail_cache:
        return _thumbnail_cache[image_path]
    try:
        from PIL import Image
        import base64, io
        with Image.open(image_path) as im:
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        _thumbnail_cache[image_path] = data_url
        return data_url
    except Exception:
        _thumbnail_cache[image_path] = ""
        return ""


# The drawer's Reload model button, one per page: the Detect run locks it
# while a queue runs (clearing the model cache mid-run would crash the
# worker) and unlocks it after. Registered when the drawer is drawn.
_RELOAD_MODEL_BTN: dict = {"btn": None}


def _num(value, default):
    """A number field's value, or ``default`` once the field was emptied
    (NiceGUI then holds None): Add to queue died on int(None) and
    float(None). The default's type is kept."""
    if value is None or value == "":
        return default
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return default


# Detect's file list, one per page: Digitize calls it after a scaling
# object or a scale segment changed, so the rows say what a queued job
# would use.
_DETECT_FILES_REFRESH: dict = {"fn": None}


def _refresh_detect_files():
    fn = _DETECT_FILES_REFRESH.get("fn")
    if fn is None:
        return
    try:
        fn()
    except Exception:
        pass


def _reload_model_button():
    """The drawer's Reload model button of this page, or None."""
    return _RELOAD_MODEL_BTN.get("btn")


def _disable(el, why: str):
    """Disable a control and say why. The reason rides as the native
    ``title`` because a Quasar tooltip does not open over a disabled button."""
    el.set_enabled(False)
    el.props(f'title="{why}"')


def _enable(el):
    el.set_enabled(True)
    el.props(remove="title")


# A run outlives the page that started it (reload, closed tab). Each tab's
# console is registered under a key; pushes through any console of that key
# land on the newest one, and a console built later replays the history.
_LIVE_LOGS: dict = {}       # key -> the consoles built for that key, newest last
_LOG_HISTORY: dict = {}     # key -> deque of lines pushed so far
_LOG_HISTORY_LINES = 2000


def _log_client_alive(widget) -> bool:
    """False once NiceGUI has deleted the page the console belongs to."""
    try:
        client = widget.client
        if getattr(client, "_deleted", False):
            return False
        cid = getattr(client, "id", None)
        if cid is None:
            return True
        from nicegui import Client as _Client
        return cid in _Client.instances
    except Exception:
        return False


def _live_targets(key, widget) -> list:
    """Every console registered under `key` whose page is still there,
    newest last; the consoles of pages that are gone are forgotten."""
    consoles = _LIVE_LOGS.get(key)
    if consoles is None:
        return [widget] if _log_client_alive(widget) else []
    alive = [w for w in consoles if _log_client_alive(w)]
    if len(alive) != len(consoles):
        consoles[:] = alive
    return alive


def _live_target(widget):
    """The newest console a push through `widget` reaches, else `widget`."""
    key = getattr(widget, "_pm_live_key", None)
    if key is None:
        return widget
    targets = _live_targets(key, widget)
    return targets[-1] if targets else widget


def live_log(key: str, widget):
    """Register `widget` as a console for `key` and return it.

    The widget's ``push`` and ``clear`` are redirected to every console of
    the key whose page is still open: a worker holding an older console
    keeps printing into the one on screen, and two open pages show the
    same lines. Routing to the newest console alone left the first page
    of a session silent whenever anything built a page after it (a probe
    of the address, a second tab, a browser prerender).
    History is replayed into a newly built console."""
    import collections
    from nicegui.elements.log import Log as _Log
    hist = _LOG_HISTORY.setdefault(
        key, collections.deque(maxlen=_LOG_HISTORY_LINES))
    widget._pm_live_key = key
    _LIVE_LOGS.setdefault(key, []).append(widget)
    for line in list(hist):
        try:
            _Log.push(widget, line)
        except Exception:
            break

    def _push(line, **kw):
        for part in str(line).splitlines() or [""]:
            hist.append(part)
        pushed = False
        for target in _live_targets(key, widget):
            try:
                _Log.push(target, line, **kw)
                pushed = True
            except Exception:
                pass
        if pushed:
            return
        try:
            _orig = getattr(sys, "__stdout__", None)
            if _orig is not None:
                _orig.write(str(line) + "\n")
        except Exception:
            pass

    def _clear():
        hist.clear()
        for target in _live_targets(key, widget):
            try:
                _Log.clear(target)
            except Exception:
                pass

    widget.push = _push
    widget.clear = _clear
    return widget


@contextlib.contextmanager
def on_page(client):
    """Touch a page's UI from a worker thread.

    NiceGUI keys its slot stacks by asyncio task, gives every thread the
    same key and prunes the stacks that belong to no task every 10 s. A
    ``with client:`` held around a whole run is therefore torn down under
    the worker: a later ui.notify dies with "the slot stack for this task
    is empty" and the exit pops from an empty list. Enter
    this for each UI touch, briefly; a stack pruned meanwhile is not an
    error."""
    client.__enter__()
    try:
        yield client
    finally:
        try:
            client.__exit__(None, None, None)
        except IndexError:
            pass


@contextlib.contextmanager
def capture_stdout_to_log(log_widget, on_line=None):
    """Redirect prints to a NiceGUI ui.log() widget for live display.

    ``on_line`` is called with each line after it is pushed (drives
    progress bars). A worker may outlast the browser tab that created
    ``log_widget``; push errors are swallowed so the worker keeps going.
    """
    def _safe_push(target, line):
        try:
            target.push(line)
        except RuntimeError as ex:
            if "client this element belongs to" in str(ex):
                return
            try:
                _orig = getattr(sys, "__stdout__", None)
                if _orig is not None:
                    _orig.write(line + "\n")
            except Exception:
                pass
        except Exception:
            pass

    def _client_gone(log) -> bool:
        # NiceGUI >= 2 does not raise on a deleted client; it WARNS through
        # logging, which lands in this captured writer, which pushes again:
        # unbounded recursion. Stop pushing as soon as the client is deleted.
        try:
            return bool(getattr(_live_target(log).client, "_deleted", False))
        except Exception:
            return False

    class _LineWriter:
        def __init__(self, log):
            self.log = log
            self.buf = ""
            self.dead = False      # client gone: lines go to the console
            self._inside = False   # reentrancy: write() called from push()

        def _fallback(self, text):
            try:
                _orig = getattr(sys, "__stdout__", None)
                if _orig is not None:
                    _orig.write(text if text.endswith("\n") else text + "\n")
            except Exception:
                pass

        def close(self):
            # A logging handler created while this writer was sys.stdout
            # (absl, at TensorFlow's import) closes its stream at exit;
            # there is nothing here to close.
            pass

        def _emit(self, line):
            if self._inside:
                # Something inside push() (a logging handler, a warning)
                # wrote to stdout. Pushing it would recurse; print it.
                self._fallback(line)
                return
            if self.dead or _client_gone(self.log):
                self.dead = True
                self._fallback(line)
            else:
                self._inside = True
                try:
                    _safe_push(self.log, line)
                finally:
                    self._inside = False
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    pass  # never let UI updates crash the worker

        def write(self, s):
            self.buf += s
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                if line:
                    self._emit(line)
        def flush(self):
            if self.buf:
                self._emit(self.buf)
                self.buf = ""

    writer = _LineWriter(log_widget)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = writer
    sys.stderr = writer

    # csm.* log records also reach on_line (progress parsing). The handler
    # calls on_line directly, not writer.write(): the NiceGUI log handler
    # already pushes the formatted line, so writing it would double-display.
    import logging as _logging
    _csm_intercept = None
    if on_line is not None:
        class _InterceptHandler(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                try:
                    on_line(self.format(record))
                except Exception:
                    pass
        _csm_intercept = _InterceptHandler()
        _csm_intercept.setLevel(_logging.INFO)
        _csm_intercept.setFormatter(_logging.Formatter("%(message)s"))
        _logging.getLogger("csm").addHandler(_csm_intercept)

    try:
        yield
    finally:
        writer.flush()
        sys.stdout = old_out
        sys.stderr = old_err
        if _csm_intercept is not None:
            _logging.getLogger("csm").removeHandler(_csm_intercept)
            _csm_intercept.close()


# --- Reusable UI components ---
def path_input_with_browse(label: str, state_attr: str, kind: str = "dir",
                           filetypes=None, on_change=None,
                           default_kind: str = ""):
    """A row with a path text input bound to ``state.<state_attr>`` and a
    'Browse...' button. ``on_change`` fires after a pick; ``default_kind``
    (a PATH_KINDS value or '') seeds the dialog with the project subfolder."""
    with ui.row().classes("w-full items-center gap-2") as row:
        inp = ui.input(label=label).classes("flex-grow").bind_value(state, state_attr)
        def _browse():
            current = getattr(state, state_attr, "")
            if current:
                if kind == "dir":
                    initialdir = current
                else:
                    initialdir = str(Path(current).parent) if Path(current).parent else ""
            else:
                initialdir = default_starting_dir(default_kind) if default_kind else ""

            if kind == "dir":
                picked = native_dir_picker(f"Pick: {label}", initialdir=initialdir)
            else:
                picked = native_file_picker(f"Pick: {label}", filetypes,
                                            initialdir=initialdir)
            if picked:
                setattr(state, state_attr, picked)
                if on_change:
                    on_change()
        ui.button("Browse…", icon="folder_open",
                  on_click=_browse).props("outline") \
.tooltip("Open a native file/folder picker seeded with the "
                     "current path or the canonical default for this field.")
    row._path_input = inp   # callers seed it via _seed_default(...)
    return row


def _sync_active_date():
    """Reconcile layout's active date with the current project/date state.

    A dated project defaults to its latest date; a flat project clears it.
    Idempotent."""
    proj = state.current_project
    dates = list_dates(proj) if proj else []
    if dates:
        if state.current_date not in dates:
            state.current_date = dates[-1]
    else:
        state.current_date = ""
    set_active_date(state.current_date or None, proj or None)
    # Crash traces land in the logs/ of the project active at the fault.
    try:
        _crashsafe.repoint(project_path(proj, "logs") if proj else None)
    except Exception:
        pass  # the recorder must never break a project switch


# The project picker lives in the left drawer, once per page. A tab registers
# what it wants run when the project or date changes; the picker fires the
# hooks of its own page.
_PROJECT_HOOKS: dict = {}     # client id -> [callable]


def _project_hooks_key():
    try:
        from nicegui import context
        return context.client.id
    except Exception:
        return None


def render_project_strip(on_change=None):
    """Register `on_change` to run when the active project or date changes
    (the picker itself is drawn by `render_project_picker`)."""
    if on_change is None:
        return
    key = _project_hooks_key()
    hooks = _PROJECT_HOOKS.setdefault(key, [])
    if not hooks:
        try:
            from nicegui import context
            context.client.on_delete(lambda: _PROJECT_HOOKS.pop(key, None))
        except Exception:
            pass
    hooks.append(on_change)


_TAB_HOOKS: dict = {}


def register_tab_refresh(tab_name: str, fn):
    """Run ``fn`` when ``tab_name`` becomes the active tab.

    Pre-fills are computed at page build and on a project change, so without
    this a tab opened later still shows the previous project's files. Hooks are per client and go with it.
    """
    key = _project_hooks_key()
    per_tab = _TAB_HOOKS.setdefault(key, {})
    if not per_tab:
        try:
            from nicegui import context
            context.client.on_delete(lambda: _TAB_HOOKS.pop(key, None))
        except Exception:
            pass
    per_tab.setdefault(str(tab_name), []).append(fn)


def _fire_tab_hooks(tab_name):
    for hook in list(_TAB_HOOKS.get(_project_hooks_key(), {})
                     .get(str(tab_name), [])):
        try:
            hook()
        except Exception as ex:
            print(f"[tab refresh] {tab_name}: {ex}")


def drop_foreign_paths(*attrs) -> int:
    """Empty the state paths that point outside the active project.

    A path the user typed for this project survives; one left over from
    another project does not, so the seeds can fill the field again. Returns
    how many were dropped. No project, nothing to compare against: no-op.
    """
    proj = state.current_project
    if not proj:
        return 0
    try:
        root = Path(project_path(proj)).resolve()
    except Exception:
        return 0
    dropped = 0
    for attr in attrs:
        value = (getattr(state, attr, "") or "").strip()
        if not value:
            continue
        try:
            inside = Path(value).resolve().is_relative_to(root)
        except (OSError, ValueError):
            inside = False
        if not inside:
            setattr(state, attr, "")
            dropped += 1
    return dropped


def _fire_project_hooks():
    for hook in list(_PROJECT_HOOKS.get(_project_hooks_key(), [])):
        try:
            hook()
        except Exception as ex:
            from functions._logging import get_logger as _gl
            _gl("gui.app").warning("project-change hook failed: %s", ex)


_PROJECT_PICKERS: list = []   # (client, refresher) per open page's pickers


def _client_alive(client) -> bool:
    """False once NiceGUI has deleted the page's client."""
    try:
        from nicegui import Client as _Client
        return (not getattr(client, "_deleted", False)
                and client.id in _Client.instances)
    except Exception:
        return False


def element_alive(element) -> bool:
    """False once the page that holds ``element`` is gone.

    A queue worker outlives a reload, and every element it touched then
    logged "Client has been deleted but is still being used". The run continues; the drawing is what has nowhere to go.
    """
    try:
        return _client_alive(getattr(element, "client", None))
    except Exception:
        return False


def refresh_project_pickers() -> None:
    """Rebuild every open page's project pickers (the drawer's, the Overview
    tab's) from disk and from the state: after a project or a date was
    created or picked anywhere. A page that is gone is dropped first:
    NiceGUI 2 does not raise on a deleted client, it warns "Client has been
    deleted but is still being used", and every reload left one more such
    entry behind."""
    for entry in list(_PROJECT_PICKERS):
        client, r = entry
        if not _client_alive(client):
            try:
                _PROJECT_PICKERS.remove(entry)
            except ValueError:
                pass
            continue
        try:
            with client:
                if hasattr(r, "refresh"):
                    r.refresh()
                else:
                    r()
        except Exception:
            try:
                _PROJECT_PICKERS.remove(entry)
            except ValueError:
                pass


def render_project_picker():
    """The project + date selector in the left drawer; every tab's registered
    hook fires after a pick.

    The selects carry ``value=`` and an ``on_value_change`` handler instead
    of a two-way binding: a select bound to ``state.current_project`` whose
    options lack a value (a project created a moment ago, or a stale page)
    emits None and unsets the project for everyone. The picker is a
    refreshable rebuilt from disk by :func:`refresh_project_pickers`."""
    with ui.column().classes("w-full gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("folder").style("color: #5a6878;")
            ui.label("Project").classes("text-h6")
            ui.space()
            ui.button(icon="refresh",
                      on_click=lambda: ui.navigate.reload()) \
                .props("flat dense round") \
                .tooltip("Reload the page to pick up newly-created "
                         "projects/dates.")
        picker_box = ui.column().classes("w-full gap-1")

        @ui.refreshable
        def _render_picker():
            picker_box.clear()
            with picker_box:
                existing = list_projects()
                current = (state.current_project
                           if state.current_project in existing else "")
                # A labelled placeholder, so no-project never reads as a
                # blank row.
                sel = ui.select(
                    options={"": "— no project —",
                             **{p: p for p in existing}},
                    value=current,
                ).classes("w-full").props("dense") \
                    .tooltip("Switch the active project. All Browse... "
                             "defaults follow it.")
                sel.on_value_change(_on_project)
                dates = list_dates(state.current_project) \
                    if state.current_project else []
                if dates:
                    dsel = ui.select(
                        options=dates, label="Date",
                        value=(state.current_date if state.current_date in dates
                               else dates[-1]),
                    ).classes("w-full").props("dense") \
                        .tooltip("Active acquisition date for this "
                                 "multi-temporal project.")
                    dsel.on_value_change(_on_date)
                ui.label(
                    f"→ {project_path(state.current_project)}"
                    if state.current_project
                    else "→ (no project — using datasets/)"
                ).classes("text-xs text-grey-7") \
                    .style("word-break: break-all;")

        def _on_project(e):
            state.current_project = str(getattr(e, "value", "") or "")
            _sync_active_date()
            refresh_project_pickers()
            _fire_project_hooks()

        def _on_date(e):
            state.current_date = str(getattr(e, "value", "") or "")
            _sync_active_date()
            refresh_project_pickers()
            _fire_project_hooks()

        _sync_active_date()
        _PROJECT_PICKERS.append((context.client, _render_picker))
        _render_picker()


def detection_model_options() -> dict:
    """``{backend name: display name}`` the drawer offers (developer
    backends hidden unless PEBBLEMAPPER_SHOW_DEV_BACKENDS=1)."""
    try:
        from detectors import selector_options
        opts = selector_options()
    except Exception:
        opts = {}
    return opts or {"maskrcnn": "Mask R-CNN (Soloy et al., 2020)"}


def current_detection_model() -> tuple:
    """``(name, display name)`` of the model in force: ``state.det_model``
    normalised against the offered options (an unknown or hidden model falls
    back to Mask R-CNN)."""
    opts = detection_model_options()
    name = state.det_model if state.det_model in opts else next(iter(opts))
    if state.det_model != name:
        state.det_model = name
    return name, opts[name]


def _model_hint(name: str) -> str:
    if name == "maskrcnn":
        return ("Used by Detect, Express and Digitize's Detect.")
    return ("Used by Detect, Express and Digitize's Detect (on the "
            "photograph, at its Detection scale). Score threshold is a "
            "Mask R-CNN setting.")


def _short_model_name(display_name: str) -> str:
    """``"Mask R-CNN (Soloy et al., 2020)"`` -> ``"Mask R-CNN"``."""
    s = str(display_name or "").split(" (", 1)[0].strip()
    return s or str(display_name or "the model")


def _model_unavailable_text(backend) -> str:
    """Why a detection model cannot run, and what to do: for the built-in
    Mask R-CNN, the weights notice of detectors.download_weights (how to get
    the weights while they are not public); for a plug-in, its environment."""
    name = getattr(getattr(backend, "info", None), "name", "") or ""
    if name == "maskrcnn":
        from detectors.download_weights import weights_missing_message
        return weights_missing_message()
    label = getattr(getattr(backend, "info", None), "display_name", name) or "this model"
    return (f"Detection model '{label}' is not available: its environment or "
            "weights are missing. Install them, then Reload model (left panel, Settings).")


def model_status_text(name: str = None) -> str:
    """The drawer's status line for the selected backend: Mask R-CNN's
    cache (``Model: loads on first run`` / ``Model: loaded (…)``), a
    subprocess backend that loads at every run, or an unavailable one."""
    if name is None:
        name = current_detection_model()[0]
    backend = None
    try:
        from detectors import get_backend as _gb
        backend = _gb(name)
    except Exception:
        backend = None
    if backend is not None:
        try:
            if not backend.is_available():
                return ("Model: Mask R-CNN weights not installed (see Detect)"
                        if name == "maskrcnn" else
                        "Model: selected backend unavailable — weights or "
                        "environment missing")
        except Exception:
            pass
    if name != "maskrcnn":
        in_process = bool(getattr(getattr(backend, "info", None),
                                  "in_process", False))
        return ("Model: loads on first run" if in_process
                else "Model: loads its weights at every run")
    try:
        from functions.clasts_detection import get_model_status
        s = get_model_status()
    except Exception as e:
        return f"Model: (status err: {e})"
    if not s["loaded"]:
        return "Model: loads on first run"
    key = s["key"] or ()
    age = s.get("age_s") or 0
    if age < 90:
        age_str = f"{age:.0f}s ago"
    elif age < 60 * 90:
        age_str = f"{age / 60:.0f}m ago"
    else:
        age_str = f"{age / 3600:.1f}h ago"
    if len(key) == 3:
        dm, dn, mc = key
        mc_str = f"min_conf={mc}" if mc is not None else "default conf"
        return f"Model: loaded ({dm}:{dn}, {mc_str}, {age_str})"
    return f"Model: loaded ({age_str})"


def reload_detection_model(name: str = None) -> tuple:
    """Drop the selected backend's cached weights (``clear_cache``) so the
    next run loads them fresh. Returns ``(message, notify type)``."""
    if name is None:
        name = current_detection_model()[0]
    label = detection_model_options().get(name, name)
    short = _short_model_name(label)
    try:
        from detectors import get_backend as _gb
        backend = _gb(name)
    except Exception:
        backend = None
    try:
        if backend is None:
            if name != "maskrcnn":
                return (f"{short} is not registered; nothing to reload.",
                        "warning")
            from functions.clasts_detection import clear_model_cache
            clear_model_cache()
            cleared = True
        else:
            cleared = bool(backend.clear_cache())
    except Exception as ex:
        return (f"Reload failed: {ex}. Check the weights, then Reload model "
                "again.", "negative")
    if cleared:
        return (f"{short} weights will reload on the next run.", "positive")
    return (f"{short} loads its weights at every run; nothing to reload.",
            "info")


def render_stop_button():
    """The drawer's way to stop the server, so nobody has to hunt for the
    console window: a confirmation, then a clean shutdown. The breadcrumb is
    marked clean first, so the next start has nothing to report."""
    with ui.dialog() as dlg, ui.card().classes("max-w-md"):
        ui.label("Stop PebbleMapper?").classes("text-h6")
        ui.label("The server stops and this page goes blank. A detection that "
                 "is running is interrupted; queued jobs are saved and offered "
                 "back at the next start.").classes("text-sm text-grey-7")

        def _stop():
            from nicegui import app as _ng_app
            dlg.close()
            _crashsafe.mark_clean_exit()
            _ng_app.shutdown()
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
            ui.button("Stop", icon="power_settings_new", on_click=_stop)                 .props("color=negative no-caps").classes("pm-stop-confirm")
    ui.button("Stop PebbleMapper", icon="power_settings_new", on_click=dlg.open)         .props("outline dense no-caps color=negative").classes("pm-stop-button")         .tooltip("Shut the server down cleanly. Closing the console window or "
                 "Ctrl+C there does the same; none of them counts as a crash.")


def render_detection_settings():
    """The drawer's Settings: GPU/CPU, Device #, the Detection model, its
    status line and Reload model.

    The model select is built with ``value=`` and writes ``state.det_model``
    in its on_change, never two-way bound: a select bound to shared state
    whose options lack the value nulls it from every other open page."""
    ui.toggle(["GPU", "CPU"]).bind_value(state, "devicemode")
    ui.number(label="Device #", min=0, step=1).bind_value(state, "devicenumber")
    opts = detection_model_options()
    name, _label = current_detection_model()
    hint = ui.label(_model_hint(name)) \
        .classes("text-caption text-grey-7 pm-model-hint")

    def _on_model(e):
        if e.value and e.value in opts:
            state.det_model = e.value
            hint.set_text(_model_hint(e.value))
            _refresh_status()
    sel = ui.select(opts, label="Detection model", value=name,
                    on_change=_on_model) \
        .classes("w-full pm-det-model-select").props("dense") \
        .tooltip("The detection model every run uses: Detect, Express and "
                 "Digitize's Detect. Mask R-CNN is built in; more models are "
                 "added as plug-ins (docs/developer/adding-a-detection-model.md).")
    hint.move(target_index=sel.parent_slot.children.index(sel) + 1)

    def _refresh_status():
        try:
            status.set_text(model_status_text())
        except Exception as ex:
            status.set_text(f"Model: (status err: {ex})")

    def _reload():
        msg, kind = reload_detection_model()
        ui.notify(msg, type=kind)
        _refresh_status()
    _RELOAD_MODEL_BTN["btn"] = ui.button("Reload model", icon="refresh", on_click=_reload) \
        .classes("pm-reload-model").props("outline dense") \
        .tooltip("Drop the selected model's cached weights (and free GPU "
                 "memory) so the next run loads them fresh from disk: after "
                 "updating a weights file, or when a run looks stuck or "
                 "fails with a model error.")
    status = ui.label("").classes("text-xs text-grey-7 font-mono pm-model-status")
    _refresh_status()
    # get_model_status reads a dict: a 2 s poll costs nothing.
    ui.timer(2.0, _refresh_status)
    return sel


# --- Tab builders ---

def build_express_tab():
    """One-click pipeline: pick a UAV ortho + a preset, Run, get a report."""
    import threading
    from functions import express as _express

    render_project_strip(on_change=lambda: (_refresh_orthos(),
                                            _seed_express(force=True)))
    ui.markdown(f"### {_brand.APP_NAME} Express")
    ui.label("The whole pipeline in one run: detection, merge, rasters, maps "
             "and the PDF report, for one ortho and one to three detection "
             "windows.").classes("text-sm text-grey-7")
    # fast/medium/slow are presented as one/two/three-window combinations
    # with editable sizes.
    _MODE_LABELS = {"fast": "Single window",
                    "medium": "Two-window combination",
                    "slow": "Three-window combination"}
    _MODE_DEFAULT_WINDOWS = {"fast": [2.5], "medium": [2.5, 5.0],
                             "slow": [1.0, 2.5, 5.0]}
    state.express_preset = getattr(state, "express_preset", "fast")
    if state.express_preset not in _MODE_LABELS:
        state.express_preset = "fast"
    state.express_ortho = getattr(state, "express_ortho", "")
    if not getattr(state, "express_windows", None):
        state.express_windows = list(_MODE_DEFAULT_WINDOWS[state.express_preset])
    state.express_stop = False

    ortho_select = ui.select([], label="UAV ortho-image").classes("w-96")
    info_label = ui.label("").classes("text-sm")

    def _refresh_orthos():
        proj = state.current_project
        items = []
        if proj:
            d = project_path(proj, "images")
            if d.is_dir():
                items = [str(p) for p in sorted(d.iterdir())
                         if p.suffix.lower() in (".tif", ".tiff")]
        ortho_select.options = items
        ortho_select.update()
    _refresh_orthos()

    def _update_preview():
        """Refresh the extent + window preview (follows both the ortho and
        the preset)."""
        if not state.express_ortho:
            info_label.set_text("")
            return
        try:
            ext = _express.read_ortho_extent(state.express_ortho)
            if not ext["is_georeferenced"]:
                info_label.set_text("Not georeferenced — Express needs a georeferenced "
                                    "ortho-image. Use the Detect tab in Quadrat "
                                    "mode for scaled photographs.")
                return
            ws = list(state.express_windows)
            info_label.set_text(
                f"{ext['width_m']:.0f}x{ext['height_m']:.0f} m, GSD "
                f"{ext['gsd_m'] * 100:.2f} cm/px — window(s): "
                + ", ".join(f"{w:g}" for w in ws) + " m"
                + ("  (results merged)" if len(ws) >= 2 else ""))
        except Exception as ex:
            info_label.set_text(f"Could not read ortho: {ex}")

    def _on_ortho(e):
        state.express_ortho = e.value or ""
        _update_preview()
    ortho_select.on_value_change(_on_ortho)

    def _on_preset(e):
        state.express_preset = e.value
        state.express_windows = list(_MODE_DEFAULT_WINDOWS[e.value])
        _render_windows.refresh()
        _update_preview()
    ui.radio(_MODE_LABELS, value=state.express_preset,
             on_change=_on_preset) \
        .bind_value(state, "express_preset").props("inline")

    # The model is the drawer's (Settings), shared with Detect and Digitize.
    ui.label("").classes("text-sm text-grey-7 pm-model-caption") \
        .bind_text_from(state, "det_model", backward=lambda _m: (
            f"Detection model: {current_detection_model()[1]} "
            "(chosen in the left panel, Settings)"))

    ui.label("Detection window size(s) — editable; results are merged when "
             "more than one window is used:").classes("text-sm text-grey-7 mt-1")
    windows_box = ui.row().classes("items-end gap-2")

    @ui.refreshable
    def _render_windows():
        windows_box.clear()
        with windows_box:
            for i in range(len(state.express_windows)):
                def _bind(idx):
                    def _set_w(e):
                        try:
                            state.express_windows[idx] = min(
                                50.0, max(0.25, float(e.value or 0)))
                        except (TypeError, ValueError):
                            pass
                        _update_preview()
                    ui.number(f"Window {idx + 1} (m)", min=0.25, max=50.0,
                              step=0.5, format="%.2f",
                              value=state.express_windows[idx]) \
                        .classes("w-32").on_value_change(_set_w)
                _bind(i)
    _render_windows()

    # The stages live in the state, not in this builder: a page reloaded
    # mid-run rebuilds them, and the worker updates the shared list.
    _STAGE_NAMES = ("Detection", "Merge", "Rasterize", "Map", "Report")
    if not getattr(state, "express_stages", None):
        state.express_stages = [{"name": n, "status": "pending"}
                                for n in _STAGE_NAMES]
    stage_jobs = state.express_stages
    queue_box = ui.column().classes("w-full")

    def _render_stages():
        render_queue(queue_box, stage_jobs, title="Pipeline stages",
                     primary_text=lambda j, i: j["name"])
    _render_stages()

    _STAGE_ORDER = ["Detection", "Merge", "Rasterize", "Map", "Report"]
    def _progress_from_stages():
        """``(text, value, running)`` read from the stages themselves.

        One source of truth: a page rebuilt during a run finds the stage
        list in the state and can say where the run is, without a second
        copy of the same fact to go stale.
        """
        names = [j["name"] for j in stage_jobs]
        statuses = [j["status"] for j in stage_jobs]
        if "running" in statuses:
            i = statuses.index("running")
            return f"{names[i]}…", i / len(stage_jobs), True
        if all(s == "done" for s in statuses):
            return "Done.", 1.0, False
        if "error" in statuses:
            return "Failed.", statuses.index("error") / len(stage_jobs), False
        if "stopped" in statuses:
            return "Stopped.", statuses.index("stopped") / len(stage_jobs), False
        done = sum(1 for s in statuses if s == "done")
        return ("", done / len(stage_jobs), False) if done else ("", 0.0, False)

    _p_text, _p_value, _p_running = _progress_from_stages()
    progress_label = ui.label(_p_text).classes("text-sm text-grey-7")
    progress_bar = ui.linear_progress(value=_p_value, show_value=False) \
        .classes("w-full")
    progress_bar.visible = _p_running

    def _follow_run():
        """A page rebuilt mid-run gets no callback from the worker that is
        still going, so it watches the shared stages instead."""
        text, value, running = _progress_from_stages()
        snapshot = tuple(j["status"] for j in stage_jobs)
        if snapshot != _follow_run.last:
            _follow_run.last = snapshot
            _render_stages()
            progress_label.set_text(text)
            progress_bar.value = value
        progress_bar.visible = running
    _follow_run.last = tuple(j["status"] for j in stage_jobs)
    ui.timer(1.0, _follow_run)

    def _on_stage(name, status):
        try:
            idx = _STAGE_ORDER.index(name)
        except ValueError:
            return
        progress_bar.value = (idx / len(_STAGE_ORDER) if status == "running"
                              else (idx + 1) / len(_STAGE_ORDER))
        if status == "running":
            progress_label.set_text(f"{name}…")
        for j in stage_jobs:
            if j["name"] == name:
                j["status"] = status
        _render_stages()

    log_widget = live_log("express", build_log_console(max_lines=2000, height="h-64"))

    def _run():
        for _j in stage_jobs:
            _j["status"] = "pending"
        _render_stages()
        if not state.current_project or not state.express_ortho:
            ui.notify("Select a project (Project, left panel) and a *UAV ortho-image* (Inputs, above) first.", type="warning")
            return
        # Pre-flight: the selected backend must report itself available.
        try:
            from detectors import get_backend as _get_backend
            _b = _get_backend(current_detection_model()[0])
            if _b is not None and not _b.is_available():
                ui.notify(_model_unavailable_text(_b), type="negative",
                          multi_line=True, timeout=0, close_button=True)
                return
        except Exception:
            weights = REPO_ROOT / "model_weights" / "mask_rcnn_clasts.h5"
            if not weights.exists():
                from detectors.download_weights import weights_missing_message
                ui.notify(weights_missing_message(), type="negative",
                          multi_line=True, timeout=0, close_button=True)
                return
        state.express_stop = False
        for j in stage_jobs:
            j["status"] = "pending"
        _render_stages()
        progress_bar.value = 0.0
        progress_bar.visible = True
        progress_label.set_text("Starting…")
        # The worker runs in a thread: a notification needs the page's
        # client, or it dies with "the slot stack for this task is empty".
        _page = context.client

        def _say(message: str, kind: str):
            try:
                with on_page(_page):
                    ui.notify(message, type=kind)
            except Exception:
                pass

        def _worker():
            # Route the detection engine's INFO progress into the log too;
            # stdout alone stays silent through the ortho read and tiling.
            from functions._logging import (
                NiceGUILogHandler, attach_gui_handler, detach_gui_handler)
            _gui_log_handler = NiceGUILogHandler(log_widget)
            attach_gui_handler(_gui_log_handler)
            try:
                with capture_stdout_to_log(log_widget):
                    res = _express.run_express_pipeline(
                        state.current_project, state.express_ortho,
                        state.express_preset,
                        windows=list(state.express_windows),
                        model=current_detection_model()[0],
                        log_fn=log_widget.push,
                        on_stage=_on_stage,
                        stop_check=lambda: state.express_stop,
                        devicemode=state.devicemode.lower(),
                        devicenumber=int(state.devicenumber))
                for s in res.stages:
                    for j in stage_jobs:
                        if j["name"] == s.name:
                            j["status"] = s.status
                _render_stages()
                progress_bar.value = 1.0
                progress_label.set_text("Done.")
                if res.report_pdf:
                    log_widget.push(f"[express] report: {res.report_pdf}")
                    _say("Express run complete.", "positive")
            except Exception as ex:
                log_widget.push(f"[express] failed: {ex}")
                for j in stage_jobs:
                    if j["status"] in ("pending", "running"):
                        j["status"] = "error"
                        break
                _render_stages()
                progress_label.set_text(f"Failed: {ex}")
                _say(f"Express failed: {str(ex).rstrip('.')}. See the log "
                     "(below Run), fix the input, then Run again.", "negative")
            finally:
                progress_bar.visible = False
                detach_gui_handler(_gui_log_handler)
        threading.Thread(target=_worker, daemon=True).start()

    with ui.row().classes("gap-2 items-center") as _ex_run_row:
        ui.button("Run", icon="play_arrow", on_click=_run).props("color=primary")
        ui.button("Stop", icon="stop",
                  on_click=lambda: setattr(state, "express_stop", True)).props("flat")

    # Stages, progress and log sit below Run so the button keeps its place.
    _ex_root = _ex_run_row.parent_slot.parent
    for _el in (queue_box, progress_label, progress_bar, log_widget):
        _el.move(_ex_root)

    def _seed_express(force=False):
        """Seed the project's newest ortho when none is chosen (or the
        project changed); never a value the list does not offer."""
        from functions import project_defaults as _pdf
        proj = state.current_project
        if not proj:
            return
        opts = [str(o) for o in (ortho_select.options or [])]
        if state.express_ortho in opts and not force:
            # A remembered choice on a new page (a reload, a second window,
            # the next launch): the select is new and must show it too.
            pick = state.express_ortho
        else:
            pick = next((str(p) for p in _pdf.orthos(proj) if str(p) in opts),
                        None)
            if pick is None:
                return
            state.express_ortho = pick
        if ortho_select.value != pick:
            ortho_select.value = pick
        try:
            ortho_select.props('hint="from the active project — pick '
                               'another to change"')
        except Exception:
            pass
        _update_preview()
    _seed_express()

    def _refresh_express_on_open():
        if drop_foreign_paths("express_ortho"):
            _refresh_orthos()
        _seed_express()
    register_tab_refresh("Express", _refresh_express_on_open)


def build_overview_tab():
    """Landing page: datasets root, active project, tab directory."""
    with ui.row().classes("items-center gap-3"):
        icon_path = Path(__file__).parent / "icon.svg"
        if icon_path.exists():
            ui.image("/static_icon/icon.svg").style("width: 56px; height: 56px;")
        ui.label(_brand.APP_NAME).classes("text-h4")
    ui.markdown(f"**{_brand.APP_TAGLINE}.**").classes("text-subtitle1")
    ui.markdown(
        "*Model: Soloy et al. (2020). Licence and citation: see the README.*"
    ).classes("text-caption text-grey-7")

    ui.separator()
    ui.markdown("## Datasets")
    ui.label("One datasets root holds every project.").classes("text-sm text-grey-7")
    with ui.row().classes("w-full items-center gap-3 mt-1"):
        ui.icon("storage").style("color: #5a6878;")
        root_label = ui.label(f"Datasets root: {get_datasets_root()}") \
            .classes("text-sm")

        def _change_root():
            new = native_dir_picker("Select the datasets root folder",
                                    initialdir=str(get_datasets_root()))
            if not new:
                return
            set_datasets_root(new)
            root_label.set_text(f"Datasets root: {get_datasets_root()}")
            ui.notify(f"Datasets root set to {new}. Reloading…",
                      type="positive")
            ui.navigate.reload()
        ui.button("Change…", icon="folder_open", on_click=_change_root) \
            .props("outline dense")

    ui.separator()
    ui.markdown("## Active project")
    ui.label("Every tab works on this project.").classes("text-sm text-grey-7")

    project_picker_row = ui.row().classes("w-full items-center gap-3 mt-1")
    date_mgmt_row = ui.row().classes("w-full items-center gap-3 mt-1")

    def _refresh_dates():
        date_mgmt_row.clear()
        proj = state.current_project
        if not proj:
            return
        with date_mgmt_row:
            dates = list_dates(proj)
            if dates:
                ui.icon("event").style("color: #5a6878;")
                dsel = ui.select(
                    options=dates, label="Active date",
                    value=(state.current_date if state.current_date in dates
                           else dates[-1]),
                ).classes("min-w-[10rem]") \
.tooltip("Active acquisition date — all paths for this project resolve "
                         "under <project>/<date>/.")

                def _on_active_date(e):
                    state.current_date = str(getattr(e, "value", "") or "")
                    _sync_active_date()
                    refresh_project_pickers()
                    _fire_project_hooks()
                dsel.on_value_change(_on_active_date)
            else:
                ui.label("Single-acquisition project. Add a date to make it "
                         "multi-temporal:").classes("text-grey-7 text-sm")
            new_date = ui.input(label="Add date", placeholder="YYYY-MM-DD") \
.classes("min-w-[10rem]") \
.tooltip("Create an acquisition date folder under this project.")

            def _add_date():
                d = (new_date.value or "").strip()
                if not d:
                    ui.notify("Enter a date (e.g. 2024-06-01) in the date field (Active project, above) first.",
                              type="warning")
                    return
                try:
                    ensure_project_layout(proj, date=d)
                except ValueError as e:
                    ui.notify(f"{e} — fix the date (Active project, above), then Add date again.",
                          type="negative")
                    return
                state.current_date = d
                _sync_active_date()
                _refresh_dates()
                refresh_project_pickers()
                ui.notify(f"Added date '{d}' to project '{proj}'.",
                          type="positive")
            ui.button("Add date", icon="event_available",
                      on_click=_add_date).props("outline dense")

    def _refresh_project_picker():
        project_picker_row.clear()
        with project_picker_row:
            existing = list_projects()
            if existing:
                psel = ui.select(
                    options={"": "— no project —",
                             **{p: p for p in existing}},
                    label="Existing project",
                    value=(state.current_project
                           if state.current_project in existing else ""),
                ).classes("min-w-[14rem]") \
.tooltip("Set the active project. File dialogs across "
                             "the application default to its canonical "
                             "subfolders.")

                def _on_proj(e):
                    state.current_project = str(getattr(e, "value", "") or "")
                    _sync_active_date()
                    _refresh_dates()
                    refresh_project_pickers()
                    _fire_project_hooks()
                psel.on_value_change(_on_proj)
            else:
                ui.label("No projects yet. Create one below.") \
.classes("text-grey-7 italic")

            new_name_input = ui.input(label="New project") \
.classes("min-w-[12rem]") \
.tooltip("Create a new project under the datasets root. The "
                         "canonical subfolder layout is initialised "
                         "automatically.")
            new_date_input = ui.input(label="Initial date (optional)",
                                      placeholder="YYYY-MM-DD") \
.classes("min-w-[11rem]") \
.tooltip("Leave blank for a single-acquisition (flat) project, or set a "
                         "date to start a multi-temporal project.")

            def _create_project():
                name = (new_name_input.value or "").strip()
                if not name:
                    ui.notify("Enter a project name (Active project, above) first.", type="warning")
                    return
                if any(c in name for c in ("/", "\\", "..", "\0")):
                    ui.notify("Project names may not contain the characters "
                              "/ \\ .. or a null byte. Change the name (Active project, above).",
                              type="negative")
                    return
                date = (new_date_input.value or "").strip()
                try:
                    ensure_project_layout(name, date=date or None)
                except ValueError as e:
                    ui.notify(f"{e} — fix the name (Active project, above), then Create again.",
                          type="negative")
                    return
                state.current_project = name
                state.current_date = date or ""
                _sync_active_date()
                _refresh_dates()
                refresh_project_pickers()
                _fire_project_hooks()
                ui.notify(
                    f"Project '{name}' created"
                    + (f" with date '{date}'." if date
                       else " with the canonical subfolder layout."),
                    type="positive")

            ui.button("Create", icon="add",
                      on_click=_create_project) \
.props("color=primary outline")
        _refresh_dates()

    _PROJECT_PICKERS.append((context.client, _refresh_project_picker))
    _refresh_project_picker()

    project_paths_label = ui.markdown("").classes("mt-1")

    def _refresh_project_paths():
        proj = state.current_project
        if proj:
            base = project_path(proj)  # date-aware: <project> or <project>/<date>
            base_disp = (base.relative_to(REPO_ROOT)
                         if base.is_relative_to(REPO_ROOT) else base)
            date_note = (f" &nbsp;·&nbsp; date: `{state.current_date}`"
                         if state.current_date else "")
            project_paths_label.set_content(
                f"**Active project root:** `{base_disp}`{date_note}\n\n"
                f"inputs: `input_data/{{images,dem,geometries}}/` "
                f"&nbsp;·&nbsp; outputs: "
                f"`output_results/{{vectors,rasters,figures,maps,zonal,reports,logs}}/` "
                f"&nbsp;·&nbsp; validation: "
                f"`validation/{{raw,orthorectified,georectified,gps,results}}/`"
            )
        else:
            project_paths_label.set_content(
                "*No project selected.*"
            )

    _refresh_project_paths()
    ui.timer(0.5, _refresh_project_paths)

    def _reorganise_files():
        proj = state.current_project
        if not proj:
            ui.notify("Select a project (Project, left panel) first.", type="warning")
            return
        in_actions = migrate_input_structure(proj, dry_run=True)
        val_actions = migrate_validation_structure(proj, dry_run=True)
        if not in_actions and not val_actions:
            ui.notify("Nothing to reorganise — this project already matches "
                      "the current layout.", type="info")
            return
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Reorganise files in '{proj}'").classes("text-lg font-bold")
            ui.label("Imagery → input_data/images, DEMs → input_data/dem, ROIs → "
                     "input_data/geometries, georeferenced GeoTIFFs → "
                     "validation/georectified. Ambiguous files are left in "
                     "place.").classes("text-sm text-grey-7")
            with ui.scroll_area().classes("w-[36rem] h-48 border rounded p-1"):
                for a in (in_actions + val_actions):
                    ui.label(a).classes("text-xs")

            def _do_move():
                moved = (migrate_input_structure(proj, dry_run=False)
                         + migrate_validation_structure(proj, dry_run=False))
                n = sum(1 for m in moved if m.startswith("moved"))
                ui.notify(f"Moved {n} file(s) into the current layout.",
                          type="positive")
                dlg.close()
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Move files", icon="drive_file_move",
                          on_click=_do_move).props("color=primary")
        dlg.open()

    with ui.row().classes("items-center gap-2 mt-1"):
        ui.button("Reorganise project files…", icon="folder_special",
                  on_click=_reorganise_files).props("outline dense") \
.tooltip("Best-effort move of an older project's inputs + validation files "
                 "into the current input_data/ and validation/ structure. "
                 "Shows a preview before moving anything.")

    ui.separator()
    ui.markdown("## Tabs")

    def _jump_to(tab_name: str):
        state.active_tab = tab_name

    _TAB_LINES = [
        ("Express", "bolt", "Whole Ortho pipeline from one ortho-image."),
        ("Orthorectify", "crop_rotate",
         "Photographs → flat, scaled images."),
        ("Detect", "photo_camera",
         "Per-clast measurements, optional ROI."),
        ("Merge", "merge", "One CSV from several window sizes."),
        ("Rasterize", "map", "Grain-size rasters from a clast CSV."),
        ("Map", "picture_as_pdf", "Figure to PNG, PDF or SVG."),
        ("Zonal", "layers", "Polygon statistics, transects, profile figures."),
        ("Georeference", "my_location",
         "Quadrat photograph placed in the ortho."),
        ("Digitize", "edit", "Hand-outlined clasts → ground-truth CSV."),
        ("Validate", "fact_check", "Detection CSV against ground truth."),
        ("Report", "description", "One PDF for the project."),
    ]
    with ui.column().classes("w-full gap-0"):
        for _name, _icon, _line in _TAB_LINES:
            with ui.row().classes("items-center gap-2 w-full flex-nowrap"):
                ui.button(_name, icon=_icon,
                          on_click=lambda n=_name: _jump_to(n)) \
                    .props("flat dense no-caps") \
                    .classes("min-w-[10rem] justify-start")
                ui.label(_line).classes("text-sm text-grey-8")
    ui.label("Guide: docs/user-manual.md.").classes("text-xs text-grey-7 mt-1")

    ui.separator()
    ui.separator()
    ui.markdown(
        f"<small>Repo root: <code>{REPO_ROOT}</code></small>"
    )


# Detect's third choice. It is a Quadrat run whose scale comes from an object
# drawn on each photograph instead of a GSD, so it never leaves this tab:
# jobs, CSV names, the detector call and the report all see QUADRAT.
OBJECT_SCALE = "object_scale"
DET_MODE_LABELS = dict(MODE_LABELS)
DET_MODE_LABELS[OBJECT_SCALE] = "Object scale"


def det_photo_mode(mode) -> bool:
    """True for the two modes that list photographs."""
    return mode in (QUADRAT, OBJECT_SCALE)


def det_run_mode(mode) -> str:
    """What the detector and the outputs are told: Object scale runs as
    Quadrat, only its scale comes from elsewhere."""
    return QUADRAT if mode == OBJECT_SCALE else normalise_mode(mode,
                                                               default=ORTHO)


def _det_frame_fallback_m():
    """Detect's frame thickness for unrecorded photographs, in metres, or None."""
    try:
        v = float(state.det_frame_fallback_cm or 0)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if v > 0 else None


def build_detection_tab():
    # refresh_files is defined later; the hook is wired at the end.
    _project_change_hook = {"fn": None}
    render_project_strip(on_change=lambda: _project_change_hook["fn"]() if _project_change_hook["fn"] else None)
    ui.markdown("### Detect")
    ui.label('Per-clast measurements from a folder of images: Ortho for georeferenced ortho-images, Quadrat for scaled photographs.').classes("text-sm text-grey-7")

    # The model is chosen once, in the drawer (Settings); this line names it.
    with ui.row().classes("items-center gap-2") as _det_model_row:
        _det_model_caption = ui.label(
            f"Detection model: {current_detection_model()[1]} "
            "(chosen in the left panel, Settings)") \
            .classes("text-sm text-grey-8 pm-model-caption")

    _det_dir_input = {"el": None}

    def _det_mode_changed():
        # The mode carries its own defaults: Save plots follows it, and a
        # still-seeded directory is re-seeded.
        state.det_saveplot = det_photo_mode(state.det_mode)
        seeded_defaults = ("", default_starting_dir("images"),
                           getattr(_det_dir_input.get("el"), "_seed_val", None))
        if state.det_dir in seeded_defaults:
            _seed_det_dir(force=True)
        refresh_files()

    # A Quasar toggle fed a value it has no option for pushes null back
    # into the bound state: normalise older spellings BEFORE binding.
    if state.det_mode != OBJECT_SCALE:
        state.det_mode = normalise_mode(state.det_mode, default=ORTHO)
    with ui.row().classes("w-full"):
        ui.toggle(dict(DET_MODE_LABELS)).bind_value(state, "det_mode").on(
            "update:model-value",
            lambda _: _det_mode_changed())

    def _det_dir_default_kind():
        return "images"

    def _det_scale_candidates(image_path):
        """Where the photograph's Digitize CSV could be, so its scale
        sidecar can be found: the project's validation folder, then beside
        the image itself."""
        stem = Path(image_path).stem
        out = []
        if state.current_project:
            out.append(str(project_path(state.current_project, "validation")
                           / f"{stem}_truth.csv"))
        out.append(str(Path(image_path).with_name(f"{stem}_truth.csv")))
        return out

    def _det_object_scale(image_path):
        """The scale the photograph's saved segments give (drawn in
        Digitize, *No GSD? Scale from an object*), or None. It is another
        way of measuring the GSD, so it stands in for one here."""
        from functions import gauge as _g
        try:
            lib = (_g.load_library(project_path(state.current_project))
                   if state.current_project else _g.default_library())
            scale, _path = _g.scale_from_sidecar(
                _det_scale_candidates(image_path), lib)
            return scale
        except (OSError, ValueError):
            return None

    def _det_scale_line(scale):
        """How the file list and the queue name an object scale."""
        if scale is None:
            return ""
        if scale.is_metric:
            return (f"Object scale {scale.metres_per_px * 1000:.3f} mm/px "
                    f"({scale.unit_label} from its segments)")
        return (f"Object scale {scale.px_per_unit:.1f} px per "
                f"{scale.unit_label} (no known length: not metres)")

    def _default_det_dir() -> str:
        """The photo modes prefer the project's rectified photographs
        (validation/orthorectified, else input_data/images/orthorectified)
        when there are any; otherwise the images folder."""
        base = default_starting_dir("images")
        if det_photo_mode(state.det_mode) and state.current_project:
            from functions import project_defaults as _pdf
            try:
                rect = _pdf.rectified_photos(state.current_project)
            except Exception:
                rect = []
            if rect:
                return str(rect[0].parent)
        return base
    with ui.row().classes("w-full items-center gap-2"):
        inp = ui.input(label="Image directory").classes("flex-grow") \
.bind_value(state, "det_dir")
        inp.on_value_change(lambda _: refresh_files())
        _det_dir_input["el"] = inp

        def _browse_det_dir():
            current = state.det_dir
            initialdir = current if current else default_starting_dir(_det_dir_default_kind())
            picked = native_dir_picker("Pick: Image directory", initialdir=initialdir)
            if picked:
                state.det_dir = picked
                refresh_files()
        ui.button("Browse…", icon="folder_open", on_click=_browse_det_dir).props("outline")

    def _seed_det_dir(force=False):
        """Seed the directory like every other seeded path; `force` on a
        project or mode switch."""
        _seed_default(inp, _default_det_dir() or None, force=force)

    file_status_label = ui.label("")
    with ui.row().classes("w-full gap-2 items-center") as file_controls_row:
        ui.button("Select all", icon="done_all",
                  on_click=lambda: _set_all_checked(True)).props("dense outline")
        ui.button("Unselect all", icon="remove_done",
                  on_click=lambda: _set_all_checked(False)).props("dense outline")
        file_count_label = ui.label("").classes("ml-2 text-grey-7")
    file_list_card = ui.card().classes("w-full") \
.style("max-height: 220px; overflow-y: auto;")

    def _set_all_checked(checked: bool):
        for fname in state.det_files:
            state.det_files_checked[fname] = checked

    def _refresh_count():
        n = sum(1 for f in state.det_files if state.det_files_checked.get(f))
        file_count_label.set_text(f"({n} of {len(state.det_files)} selected)")

    register_tab_refresh("Detect", lambda: refresh_files())

    def refresh_files():
        if state.det_mode == ORTHO:
            state.det_files = list_images(state.det_dir, [".tif", ".tiff"])
        else:
            state.det_files = list_images(state.det_dir,
                                          list(_images.CAMERA_EXTENSIONS))

        # Mutate det_files_checked in place: the checkboxes bind to keys of
        # this dict object, so reassigning it would orphan them.
        for f in list(state.det_files_checked.keys()):
            if f not in state.det_files:
                del state.det_files_checked[f]
        for f in state.det_files:
            state.det_files_checked.setdefault(f, True)

        if state.det_dir and not state.det_files:
            ext_msg = "expects .tif/.tiff" if state.det_mode == ORTHO else "expects .jpg/.jpeg/.png"
            file_status_label.set_text(f"No matching files in this folder ({DET_MODE_LABELS.get(state.det_mode, state.det_mode)} mode {ext_msg}).")
            file_controls_row.set_visibility(False)
            file_list_card.set_visibility(False)
        elif state.det_files:
            file_status_label.set_text(f"Found {len(state.det_files)} image(s) in this folder:")
            file_controls_row.set_visibility(True)
            file_list_card.set_visibility(True)
            file_list_card.clear()
            with file_list_card:
                # Thumbnails in Quadrat mode only; multi-GB orthos are
                # too slow to decode live.
                n_own_gsd = 0
                for fname in state.det_files:
                    with ui.row().classes("items-center gap-3 w-full"):
                        if det_photo_mode(state.det_mode):
                            thumb_data_url = _quadrat_thumbnail(
                                os.path.join(state.det_dir, fname))
                            if thumb_data_url:
                                ui.image(thumb_data_url) \
.style("width: 80px; height: 80px; object-fit: cover; border-radius: 4px;")
                        cb = ui.checkbox(fname)
                        ui.button(icon="crop_free",
                                  on_click=lambda f=fname: _roi_load_path(
                                      os.path.join(state.det_dir, f))) \
                            .props("flat dense round size=sm") \
                            .tooltip("Draw this image's ROI on the canvas "
                                     "below.")
                        if det_photo_mode(state.det_mode):
                            ui.button(icon="straighten",
                                      on_click=lambda f=fname: _scale_load(
                                          os.path.join(state.det_dir, f))) \
                                .props("flat dense round size=sm") \
                                .tooltip("Draw this photograph's scale "
                                         "segment on the canvas below.")
                        if det_photo_mode(state.det_mode):
                            from functions.gsd import effective_gsd as _egsd
                            _fpath = os.path.join(state.det_dir, fname)
                            _obj = _det_object_scale(_fpath)
                            _info = _egsd(_fpath) if _obj is None else None
                            if _obj is not None:
                                n_own_gsd += 1
                                ui.label(_det_scale_line(_obj)) \
                                    .classes("text-xs text-primary "
                                             "pm-det-object-scale") \
                                    .tooltip("Drawn on this photograph (the "
                                             "ruler button, or Digitize) and "
                                             "saved beside its CSV; it wins "
                                             "over the GSD and over the "
                                             "Resolution field.")
                            elif state.det_mode == OBJECT_SCALE:
                                ui.label("No scale yet — draw the object on it "
                                         "(ruler button)") \
                                    .classes("text-xs text-negative "
                                             "pm-det-no-scale")
                            elif _info.gsd:
                                n_own_gsd += 1
                                ui.label(f"GSD {_info.gsd:.6g} m/px · "
                                         f"{_info.source}") \
                                    .classes("text-xs text-primary")
                            elif _info.warning:
                                ui.label(_info.warning) \
                                    .classes("text-xs text-negative") \
                                    .tooltip("This file's own GSD record could "
                                             "not be used; the global "
                                             "Resolution field applies.")
                            from functions import quadrat_frame as _qf
                            _fi = _qf.frame_inset(_fpath, _det_frame_fallback_m())
                            if _fi is not None and state.det_exclude_frame:
                                ui.label(_qf.describe(_fi)) \
                                    .classes("text-xs text-grey-7 pm-det-frame")
                        cb.bind_value(state.det_files_checked, fname)
                        cb.on_value_change(lambda _: _refresh_count())
            _refresh_count()
            # With sidecar-bearing files listed, the global Resolution field
            # says it is only a fallback.
            try:
                if state.det_mode == QUADRAT and n_own_gsd:
                    n_plain = len(state.det_files) - n_own_gsd
                    res_gsd_hint.set_text(
                        f"{n_own_gsd} of {len(state.det_files)} listed file(s) "
                        "carry their own GSD (sidecar/filename) and will use "
                        "it — this field applies only to the "
                        f"{n_plain} other(s)." if n_plain else
                        f"All {n_own_gsd} listed file(s) carry their own GSD "
                        "— this field is not used for them.")
                else:
                    res_gsd_hint.set_text("")
            except NameError:
                pass  # field not built yet (first render ordering)
        else:
            file_status_label.set_text("")
            file_controls_row.set_visibility(False)
            file_list_card.set_visibility(False)
            try:
                res_gsd_hint.set_text("")
            except NameError:
                pass

    file_controls_row.set_visibility(False)
    file_list_card.set_visibility(False)

    ui.separator()
    ui.markdown("**Parameters**")

    with ui.row().classes("w-full gap-8"):
        with ui.column().classes("min-w-[18rem]"):
            ui.label("Scale").classes("text-sm text-grey-7")
            ui.number(label="Resolution (m/pixel)", min=0.0001, max=1.0,
                      step=0.0001, format="%.4f") \
.bind_value(state, "det_resolution") \
.bind_visibility_from(state, "det_mode", value=QUADRAT) \
.classes("w-64") \
.tooltip("Image scale in metres per pixel, used for files that do not "
                         "carry their own GSD (rectified images bring theirs "
                         "in a sidecar or the filename).")
            res_gsd_hint = ui.label("") \
                .classes("text-xs text-primary w-64") \
                .bind_visibility_from(state, "det_mode", value=QUADRAT)
            ui.checkbox("Exclude the quadrat frame",
                        on_change=lambda _e: refresh_files()) \
                .bind_value(state, "det_exclude_frame") \
                .bind_visibility_from(state, "det_mode", backward=det_photo_mode) \
                .classes("pm-det-exclude-frame") \
                .tooltip("A photograph rectified by Orthorectify carries the "
                         "frame's bars along its edges and its thickness in its "
                         "record; the band is left out of the measurements, "
                         "for every detection model. A photograph without the "
                         "record uses the thickness below, or is measured "
                         "whole.")
            ui.number(label="Frame thickness (cm), if not recorded",
                      min=0.0, max=50.0, step=0.1, format="%.1f",
                      on_change=lambda _e: refresh_files()) \
                .bind_value(state, "det_frame_fallback_cm") \
                .bind_visibility_from(state, "det_exclude_frame") \
                .classes("w-64 pm-det-frame-fallback") \
                .tooltip("The width of the frame's bars, for the listed "
                         "photographs whose record does not give it. It needs "
                         "the photograph's GSD. Empty: such a photograph is "
                         "measured whole.")
            ui.number(label="Tile size (m)", min=0.5, max=20.0, step=0.5,
                      format="%.1f") \
.bind_value(state, "det_metric_cropsize") \
.bind_visibility_from(state, "det_mode", value=ORTHO) \
.classes("w-64") \
.tooltip("Side length of each processed tile, in metres. "
                         "Larger tiles are faster but may miss smaller clasts. "
                         "Typical values: 1 m for pebbles, 2.5 m for cobbles.")

        with ui.column().classes("min-w-[18rem]"):
            ui.label("Detection filter").classes("text-sm text-grey-7")
            ui.checkbox("Filter by confidence") \
.bind_value(state, "det_min_conf_enabled") \
.tooltip("If enabled, drops detections with classifier score below the threshold. "
                         "Strongly recommended.")
            with ui.row().classes("items-center gap-2"):
                ui.label("Min confidence:")
                ui.slider(min=0.0, max=1.0, step=0.05) \
.bind_value(state, "det_min_confidence") \
.bind_enabled_from(state, "det_min_conf_enabled") \
.classes("w-40") \
.tooltip("Mask R-CNN papers use 0.7. Lower keeps more (potentially spurious) detections.")
                ui.label().bind_text_from(state, "det_min_confidence", lambda v: f"{v:.2f}")

        with ui.column().classes("min-w-[18rem]") \
.bind_visibility_from(state, "det_mode", value=ORTHO):
            ui.label("Tile overlap (Ortho)").classes("text-sm text-grey-7 font-bold")
            with ui.expansion("Tile overlap explained", icon="info").classes("w-full"):
                ui.label('Overlap 0: a clast on a tile edge is split and usually missed; 0.20–0.25 shows it whole in one tile at ~1.8× the tiles. See the user manual.').classes("text-sm text-grey-7")
            with ui.row().classes("items-center gap-2"):
                ui.label("Overlap:")
                ui.slider(min=0.0, max=0.5, step=0.05) \
.bind_value(state, "det_overlap") \
.classes("w-40") \
.tooltip("0.20–0.25 catches boundary-crossing clasts. "
                             "Tile count grows as 1/(1-overlap)².")
                ui.label().bind_text_from(state, "det_overlap", lambda v: f"{v:.2f}")

            with ui.expansion("Deduplication IoU explained", icon="info").classes("w-full"):
                ui.label('With overlap > 0 a clast can appear in 2–4 tiles; detections with IoU ≥ this are merged, keeping the surer one. 0.30 by default, 0 disables.').classes("text-sm text-grey-7")
            with ui.row().classes("items-center gap-2"):
                ui.label("Dedup IoU:")
                ui.slider(min=0.0, max=1.0, step=0.05) \
.bind_value(state, "det_dedup_overlap") \
.classes("w-40") \
.tooltip("Only used when overlap > 0. 0.30 is a sensible default.")
                ui.label().bind_text_from(state, "det_dedup_overlap", lambda v: f"{v:.2f}")

    with ui.expansion("Advanced parameters", icon="settings") \
            .classes("w-full mt-2") as _det_advanced:
        with ui.row().classes("gap-8 items-center"):
            ui.checkbox("Save plots").bind_value(state, "det_saveplot") \
.tooltip("Save PNG figures alongside each output CSV. "
                         "Quadrat mode: an ellipses-overlay PNG + histogram PNG. "
                         "Ortho mode: a per-tile PNG with mask overlays in the tiles/ "
                         "subfolder of the checkpoint directory.")
            ui.checkbox("Save CSV").bind_value(state, "det_saveresults") \
.tooltip("Write the per-image clast list to disk. "
                         "Disable only when testing parameters and the files are not needed.")

        with ui.column().classes("w-full mt-2") \
.bind_visibility_from(state, "det_mode", value=ORTHO):
            with ui.row().classes("gap-4 items-end"):
                ui.number(label="Default kstart for fresh jobs",
                          min=0, step=1) \
.bind_value(state, "det_kstart") \
.classes("w-48") \
.tooltip("Fallback kstart used when a queued job has no "
                             ".run.csv next to its output. Jobs that have an "
                             "interrupted run resume from their own last "
                             "tile + 1 regardless of this value.")

        with ui.column().classes("w-full mt-2") \
.bind_visibility_from(state, "det_mode", value=ORTHO):
            ui.checkbox("Drop dark / bright tiles before detection") \
.bind_value(state, "det_brightness_filter") \
.tooltip("Discard tiles whose pixels are dominated by "
                         "over-dark (vignette, ortho-edge nodata) or "
                         "over-bright (sky, blown highlights) values. "
                         "Disabled by default — old behaviour processes "
                         "every non-uniform tile.")
            with ui.row().classes("gap-4 items-end") \
.bind_visibility_from(state, "det_brightness_filter",
                                            value=True):
                ui.number(label="Dark threshold (0–254, 0 = disabled)",
                          min=0, max=254, step=1) \
.bind_value(state, "det_dark_threshold") \
.classes("w-56") \
.tooltip("Pixels with mean RGB strictly below this "
                             "count as dark nodata. Typical UAV ortho "
                             "edges sit at 0–10; vignetted corners "
                             "around 15–40.")
                ui.number(label="Bright threshold (1–255, 255 = disabled)",
                          min=1, max=255, step=1) \
.bind_value(state, "det_bright_threshold") \
.classes("w-56") \
.tooltip("Pixels with mean RGB strictly above this "
                             "count as bright nodata. Clear-sky pixels "
                             "typically sit at 240–255.")
                ui.number(label="Max bad-pixel fraction (0.0–1.0)",
                          min=0.0, max=1.0, step=0.05, format="%.2f") \
.bind_value(state, "det_nodata_max_frac") \
.classes("w-56") \
.tooltip("A tile is dropped if the fraction of dark "
                             "or bright pixels (together) exceeds this. "
                             "0.95 = only drop tiles that are almost "
                             "entirely vignette / sky / nodata.")

        # Per-image ROI canvas (Ortho and Quadrat; the state field keeps
        # its det_uav_ name). Autosaves <image_stem>_roi.geojson, which
        # _probe_roi_for() stamps onto each job at Add-to-queue time. Built
        # here, then moved out after the expander.
        with ui.column().classes("w-full mt-3 pm-det-roi") as _det_roi_section:
            with ui.card().classes("w-full bg-grey-1"):
                ui.label("ROI per image (optional)") \
                    .classes("text-subtitle2 text-grey-8") \
                    .tooltip("Ortho: tiles whose centre falls outside every ROI "
                             "polygon are skipped. Quadrat: clasts whose "
                             "centroid falls outside are dropped. No "
                             "<stem>_roi.geojson beside an image = the whole "
                             "image is processed. Saved per image; the file "
                             "list's crop button opens an image here.")

                _roi_canvas_refs: dict = {}

                def _roi_load_path(path):
                    if not path or not os.path.isfile(path):
                        ui.notify("That file is no longer in *Image "
                                  "directory* (Inputs, above).",
                                  type="warning")
                        return
                    state.det_uav_roi_canvas.image = path
                    rebuild = _roi_canvas_refs.get("rebuild_canvas")
                    if rebuild:
                        rebuild()

                def _roi_save_dir():
                    # input_data/geometries/, or next to the image with no
                    # project.
                    proj = state.current_project
                    if proj:
                        return str(project_path(proj, "geometries"))
                    img = state.det_uav_roi_canvas.image
                    return (str(Path(img).parent) if img
                            else str(default_starting_dir("geometries")))

                def _roi_on_save(path: Path):
                    # The queue snapshot reads ctx.vector at Add-to-queue time.
                    pass

                _roi_canvas_refs.update(build_image_feature_canvas(
                    state.det_uav_roi_canvas,
                    title="",
                    description="",
                    allowed_shapes=("polygon", "rectangle", "circle"),
                    current_project_getter=lambda:
                        state.current_project,
                    image_picker_default_kind="images",
                    vector_picker_default_kind="images",
                    save_dir_resolver=_roi_save_dir,
                    save_stem_template="{image_stem}_roi.geojson",
                    on_save=_roi_on_save,
                    tab_active_check=lambda:
                        state.active_tab == "Detect",
                ))

    # --- Scale from an object, photograph by photograph (Quadrat) ------
    # The segments are the ones Digitize draws and saves; this is the same
    # pass over a whole folder, so a batch can be scaled before Run.
    with ui.column().classes("w-full mt-3 pm-det-scale") as _det_scale_section:
        with ui.card().classes("w-full bg-grey-1"):
            ui.label("Scale from an object, per photograph (optional)") \
                .classes("text-subtitle2 text-grey-8") \
                .tooltip("For photographs with no GSD. Pick the object, then "
                         "click its two ends on each photograph: the segment "
                         "is saved beside that photograph's Digitize CSV and "
                         "Detect runs the file with it. Go down the list with "
                         "Next; Digitize draws the same segments and can mix "
                         "several objects on one photograph.")
            _scale_refs: dict = {}
            _scale_state = {"loading": False}

            def _scale_lib():
                from functions import gauge as _g
                return (_g.load_library(project_path(state.current_project))
                        if state.current_project else _g.default_library())

            def _scale_names():
                return [o.name for o in _scale_lib()]

            def _scale_object():
                names = _scale_names()
                if state.det_scale_object in names:
                    return state.det_scale_object
                return names[0] if names else ""

            def _scale_csv_for(image_path):
                """Where this photograph's segments live: the file Digitize
                reads (the project's validation/, else beside the image)."""
                return _det_scale_candidates(image_path)[0]

            def _scale_segments_of_features():
                segs = []
                obj = _scale_object()
                for f in state.det_scale_canvas.features:
                    pts = f.get("pts") or []
                    if f.get("shape") != "transect" or len(pts) < 2:
                        continue
                    segs.append({"p0": [float(pts[0]["x"]), float(pts[0]["y"])],
                                 "p1": [float(pts[-1]["x"]), float(pts[-1]["y"])],
                                 "object": obj})
                return segs

            def _scale_write(_explicit=False):
                """Every canvas change writes the photograph's sidecar."""
                from functions import gauge as _g
                img = state.det_scale_canvas.image
                if not img or _scale_state["loading"]:
                    return
                try:
                    _g.write_scale_segments(_scale_csv_for(img),
                                            Path(img).name,
                                            _scale_segments_of_features(),
                                            _scale_lib())
                except OSError as ex:
                    ui.notify(f"Could not save the scale segments: {ex}",
                              type="negative")
                    return
                _refresh_scale_status()
                refresh_files()

            def _scale_load(path):
                """Open a photograph with its saved segments on the canvas."""
                if not path or not os.path.isfile(path):
                    ui.notify("That file is no longer in *Image directory* "
                              "(Inputs, above).", type="warning")
                    return
                from functions import gauge as _g
                ctx = state.det_scale_canvas
                _scale_state["loading"] = True
                try:
                    ctx.image = path
                    ctx.shape = "transect"
                    ctx.pts = []
                    ctx.features = []
                    doc = _g.read_scale_segments(_scale_csv_for(path))
                    for i, s in enumerate((doc or {}).get("segments", []), 1):
                        ctx.features.append({
                            "shape": "transect", "id": f"transect_{i}",
                            "source": "imported",
                            "pts": [{"px": p[0], "py": p[1],
                                     "x": float(p[0]), "y": float(p[1])}
                                    for p in (s["p0"], s["p1"])]})
                        if s.get("object") in _scale_names():
                            state.det_scale_object = s["object"]
                    rebuild = _scale_refs.get("rebuild_canvas")
                    if rebuild:
                        rebuild()
                finally:
                    _scale_state["loading"] = False
                try:
                    _scale_object_select.refresh()
                except Exception:
                    pass
                _refresh_scale_status()

            def _scale_step(delta):
                files = list(state.det_files)
                if not files:
                    return
                cur = Path(state.det_scale_canvas.image).name
                i = files.index(cur) if cur in files else -delta
                _scale_load(os.path.join(state.det_dir,
                                         files[(i + delta) % len(files)]))

            def _scale_next_without(_e=None):
                """The next photograph of the folder that has no scale yet."""
                files = list(state.det_files)
                if not files:
                    return
                cur = Path(state.det_scale_canvas.image).name
                start = files.index(cur) + 1 if cur in files else 0
                for f in files[start:] + files[:start]:
                    if _det_object_scale(os.path.join(state.det_dir, f)) is None:
                        _scale_load(os.path.join(state.det_dir, f))
                        return
                ui.notify("Every photograph of this folder has a scale.",
                          type="positive")

            def _refresh_scale_status():
                img = state.det_scale_canvas.image
                if not img:
                    scale_status.set_text(
                        "Pick a photograph: Next, or the ruler button in the "
                        "file list.")
                    return
                scale = _det_object_scale(img)
                n = len(_scale_segments_of_features())
                name = Path(img).name
                if scale is not None:
                    scale_status.set_text(f"{name}: {_det_scale_line(scale)}")
                elif n:
                    scale_status.set_text(
                        f"{name}: {n} segment(s), no scale yet — give the "
                        "object a known length in Digitize, or pick another.")
                else:
                    scale_status.set_text(
                        f"{name}: no segment yet — click both ends of the "
                        "object on the photograph.")

            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                ui.button(icon="chevron_left",
                          on_click=lambda: _scale_step(-1)) \
                    .props("outline dense").tooltip("Previous photograph")
                ui.button(icon="chevron_right",
                          on_click=lambda: _scale_step(+1)) \
                    .props("outline dense").tooltip("Next photograph")
                ui.button("Next without a scale", icon="skip_next",
                          on_click=_scale_next_without) \
                    .props("outline dense no-caps") \
                    .tooltip("Jump to the next photograph of the folder that "
                             "carries no scale yet.")

                @ui.refreshable
                def _scale_object_select():
                    # Options and value together, never bound: a select whose
                    # options lack its value nulls the state.
                    names = _scale_names()
                    cur = _scale_object()

                    def _pick(e):
                        if not e.value:
                            return
                        state.det_scale_object = e.value
                        _scale_write()
                    ui.select({n: n for n in names}, label="Scaling object",
                              value=cur or None, on_change=_pick) \
                        .props("dense options-dense").classes("w-56") \
                        .tooltip("From the project's library, which Digitize "
                                 "edits (No GSD? Scale from an object). Every "
                                 "segment drawn here spans this object.")
                _scale_object_select()

                def _scale_clear():
                    state.det_scale_canvas.features.clear()
                    _scale_write()
                    rebuild = _scale_refs.get("rebuild_canvas")
                    if rebuild:
                        rebuild()
                ui.button("Clear segments", icon="delete_sweep",
                          on_click=_scale_clear) \
                    .props("outline dense no-caps") \
                    .tooltip("Remove this photograph's segments, here and in "
                             "its saved file.")
            scale_status = ui.label("").classes("text-sm text-grey-7 "
                                                "pm-det-scale-status")

            _scale_refs.update(build_image_feature_canvas(
                state.det_scale_canvas,
                title="",
                description="",
                allowed_shapes=("transect",),
                current_project_getter=lambda: state.current_project,
                image_picker_default_kind="images",
                save_dir_resolver=lambda: "",
                # Nothing is written as GeoJSON: every change goes to the
                # photograph's scale sidecar instead.
                save_path_resolver=lambda: None,
                on_unnamed_save=_scale_write,
                show_vector_row=False,
                show_image_row=False,
                show_shape_toggle=False,
                two_click_transect=True,
                feature_style="bold",
                feature_label=lambda _f, _i: _scale_object(),
                tab_active_check=lambda: state.active_tab == "Detect",
            ))
            _refresh_scale_status()
    _det_scale_section.bind_visibility_from(state, "det_mode",
                                            backward=det_photo_mode)

    _det_root = _det_advanced.parent_slot.parent
    _det_roi_section.move(
        _det_root,
        target_index=_det_root.default_slot.children.index(_det_advanced) + 1)
    _det_scale_section.move(
        _det_root,
        target_index=_det_root.default_slot.children.index(_det_roi_section) + 1)

    ui.separator()

    with ui.row().classes("items-center gap-2"):
        add_btn = ui.button("Add to queue", icon="add") \
.props("color=primary") \
.tooltip("Snapshot the current parameters once per checked file. "
                     "To add the same image with different params, change "
                     "the form and click again.")
        run_btn = ui.button("Run all queued", icon="playlist_play") \
.props("color=primary outline") \
.tooltip("Execute every queued job in order. Jobs with an "
                     "existing .run.csv resume; jobs whose final CSV already "
                     "exists are skipped.")
        stop_current_btn = ui.button("Stop", icon="skip_next") \
.props("color=warning outline") \
.tooltip("Finish the current tile, flush the .run.csv, and move "
                     "on to the next queued job. The current job resumes "
                     "from this point on a future run.")
        stop_all_btn = ui.button("Stop all", icon="stop") \
.props("color=negative outline") \
.tooltip("Halt the entire queue after the current tile.")
        def _clear_det_queue():
            state.det_jobs[:] = [j for j in state.det_jobs
                                 if j.get("status") == "running"]
            _render_queue.refresh()
            ui.notify("Detection queue cleared.", type="info")
        def _reset_det_queue():
            n_reset = 0
            for j in state.det_jobs:
                if j.get("status") in ("done", "error", "stopped"):
                    j["status"] = "pending"
                    j["error"] = None
                    n_reset += 1
            _render_queue.refresh()
            ui.notify(
                f"Reset {n_reset} job(s) to pending.",
                type="info" if n_reset else "warning")
        ui.button("Reset all", icon="restart_alt",
                  on_click=_reset_det_queue) \
.props("flat color=primary") \
.tooltip("Mark every done / errored / stopped job as "
                     "pending again so 'Run all queued' will re-run "
                     "them with the same parameters. Running jobs "
                     "are kept. Output CSVs on disk are not deleted "
                     "— use the per-row ↺ button to reset a single "
                     "job's run.csv / final CSV if you want a clean "
                     "re-run.")
        ui.button("Clear queue", icon="delete_sweep",
                  on_click=_clear_det_queue) \
.props("flat color=grey") \
.tooltip("Remove every pending / done / error job from the queue. "
                     "Running jobs are kept. Output CSVs stay on disk.")
        queue_count_label = ui.label("").classes("text-sm text-grey-7 ml-2")
        stop_current_btn.set_visibility(False)
        stop_all_btn.set_visibility(False)

    queue_container = ui.column().classes("w-full mt-1")

    _DETECT_FILES_REFRESH["fn"] = refresh_files

    def _refresh_model_caption():
        # The status line and Reload model live in the drawer (Settings).
        try:
            _det_model_caption.set_text(
                f"Detection model: {current_detection_model()[1]} "
                "(chosen in the left panel, Settings)")
        except Exception:
            pass
    ui.timer(2.0, _refresh_model_caption)
    progress_bar = ui.linear_progress(value=0.0, show_value=False) \
.classes("w-full mt-2")
    progress_label = ui.label("").classes("text-sm text-grey-7")
    progress_bar.set_visibility(False)
    progress_label.set_visibility(False)
    log_widget = live_log("detection", build_log_console(max_lines=2000, height="h-64"))

    # Most recent Quadrat *_overlay.png after a run with saveplot.
    preview_label = ui.label("").classes("text-sm text-grey-7 mt-2")
    preview_label.set_visibility(False)
    preview_container = ui.column().classes("w-full")

    import re
    import time as _time
    # Progress line from _detect_uav: "Tile 100/14625 (0.7%, ...)".
    _progress_re = re.compile(r"Tile\s+(\d+)\s*/\s*(\d+)\s+\((\d+(?:\.\d+)?)%")
    _stop_at_tile_re = re.compile(r"STOP requested by user at tile k=(\d+)")

    # ETA from the average per-tile duration since the first tile seen.
    _eta_state = {
        "first_tile_time": None,
        "first_tile_index": None,
        "last_seen_total": None,
    }

    def _format_eta(seconds: float) -> str:
        """Render seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
        if seconds <= 0 or not (seconds < float("inf")):
            return "—"
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _on_log_line(line: str):
        m = _progress_re.search(line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            pct = float(m.group(3)) / 100.0
            progress_bar.value = pct
            now = _time.time()
            if _eta_state["first_tile_time"] is None:
                _eta_state["first_tile_time"] = now
                _eta_state["first_tile_index"] = current
                _eta_state["last_seen_total"] = total
                progress_label.set_text(
                    f"Tile {current} of {total} ({float(m.group(3)):.1f}%) — calculating ETA…"
                )
            else:
                tiles_done = max(1, current - _eta_state["first_tile_index"])
                elapsed = now - _eta_state["first_tile_time"]
                rate = elapsed / tiles_done   # seconds per tile
                remaining = (total - current) * rate
                progress_label.set_text(
                    f"Tile {current} of {total} ({float(m.group(3)):.1f}%) — "
                    f"~{_format_eta(remaining)} remaining"
                )
            return

        m2 = _stop_at_tile_re.search(line)
        if m2:
            log_widget.push(
                f"   ↪ Stop checkpoint written at tile {m2.group(1)}. "
                f"This file's resume tile will be picked up automatically."
            )

    def _resolve_dirs(to_validation: bool = False):
        """(out_dir, fig_dir) for the active project, or (None, None).
        ``to_validation`` routes both under the project's validation/ tree."""
        if state.current_project:
            if to_validation:
                base = project_path(state.current_project, "validation")
                out_dir = base
                fig_dir = base / "figures"
            else:
                out_dir = project_path(
                    state.current_project, "vectors")
                fig_dir = project_path(
                    state.current_project, "figures")
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_dir.mkdir(parents=True, exist_ok=True)
            return str(out_dir), str(fig_dir)
        return None, None

    def _expected_output_paths(image_path, image_name, metric_cropsize):
        """(run_csv, final_csv, out_dir) for this (image, cropsize) pair."""
        out_dir_str, _ = _resolve_dirs()
        final_dir = out_dir_str or os.path.dirname(image_path)
        stem = _det_out_stem(image_name)
        run_path = os.path.join(
            final_dir, naming.detection_csv_name(stem, metric_cropsize, run=True))
        final_csv = os.path.join(
            final_dir, naming.detection_csv_name(stem, metric_cropsize))
        return run_path, final_csv, out_dir_str

    def _probe_resume(image_path, image_name, metric_cropsize):
        """Inspect disk for a previous run of this (image, cropsize) pair.

        Returns (resume_state, kstart, note, n_clasts); resume_state is
        'fresh' / 'resume' / 'done'.
        """
        from functions.clasts_detection import inspect_run_state
        run_path, final_csv, out_dir_str = _expected_output_paths(
            image_path, image_name, metric_cropsize)
        # The checkpoint carries the origin stem like every other output:
        # without it the probe never found a project image's checkpoint and
        # offered the partial final CSV as "done".
        info = inspect_run_state(out_dir_str, image_path, metric_cropsize,
                                 out_stem=_det_out_stem(image_name),
                                 overlap=float(state.det_overlap or 0.0))
        if info["exists"] and info.get("grid_mismatch"):
            # A checkpoint of another tile grid: the engine discards it and
            # starts fresh; say so here rather than offer a resume.
            return ("fresh", int(state.det_kstart or 0),
                    f"A checkpoint of another tile grid is on disk "
                    f"({info['grid_mismatch']}); it will be discarded and "
                    f"the run starts fresh.", None)
        if info["exists"]:
            return ("resume", int(info["resume_k"]),
                    f"Will resume from tile {info['resume_k']} "
                    f"({info['n_clasts']} clasts already saved). "
                    f"Click ↺ to restart fresh instead.",
                    None)
        if os.path.exists(final_csv):
            n = None
            try:
                import pandas as pd
                n = int(len(pd.read_csv(final_csv)))
            except Exception:
                pass
            return ("done", 0,
                    "Final CSV already exists — this job is marked done "
                    "and skipped on Run. Click ↺ to redo from scratch.",
                    n)
        # An emptied field holds None: Add to queue died on int(None) with
        # no feedback.
        return ("fresh", int(state.det_kstart or 0), "Fresh run.", None)

    def _det_out_stem(image_name: str) -> str:
        """Origin stem every output of this image is written under: site,
        date (dated projects) and image stem, plus the detection model's
        token when it is not Mask R-CNN."""
        from functions.layout import get_active_date as _gad
        return naming.with_model(
            naming.origin_stem(os.path.splitext(image_name)[0],
                               state.current_project, _gad()),
            _det_run_model())

    def _det_run_model() -> str:
        """The model a run will use: the one in force, else Mask R-CNN."""
        from detectors import get_backend as _gbm
        name = current_detection_model()[0]
        return name if _gbm(name) is not None else "maskrcnn"

    def _snapshot_form_as_job(image_name: str) -> dict:
        """Job dict from the current form state and one image, probing for
        an existing .run.csv / final CSV so the queue can say what Run will do."""
        image_path = os.path.join(state.det_dir, image_name)
        if state.det_mode == ORTHO:
            resume_state, kstart, note, n_clasts = _probe_resume(
                image_path, image_name, _num(state.det_metric_cropsize, 1.0))
        else:
            resume_state, kstart, note, n_clasts = "fresh", 0, "Fresh run.", None
        # A quadrat photograph's own GSD (sidecar, else filename) is
        # authoritative; the global field covers only files without one.
        resolution = _num(state.det_resolution, 0.001)
        gsd_source = "global"
        unit_label = None
        if det_photo_mode(state.det_mode):
            from functions.gsd import effective_gsd
            info = effective_gsd(image_path)
            if info.warning and state.det_mode == QUADRAT:
                ui.notify(f"{info.warning} Check *Resolution* (Parameters, above).",
                          type="warning", multi_line=True,
                          timeout=10000)
            if info.gsd:
                resolution, gsd_source = float(info.gsd), info.source
            # A scale drawn on the photograph was drawn on purpose, so it
            # wins: metric in metres per pixel like any GSD, otherwise in
            # the object's own unit (the CSV then carries a unit column).
            obj = _det_object_scale(image_path)
            if obj is not None:
                gsd_source = f"object: {obj.unit_label}"
                if obj.is_metric:
                    resolution = float(obj.metres_per_px)
                else:
                    resolution = 1.0 / float(obj.px_per_unit)
                    unit_label = obj.unit_label
        return {
            "image_path": image_path,
            "image_name": image_name,
            "out_stem": _det_out_stem(image_name),
            "mode": det_run_mode(state.det_mode),
            "resolution": resolution,
            "gsd_source": gsd_source,
            # Set only for an object of unknown length: the run is in that
            # unit, not in metres.
            "unit_label": unit_label,
            "exclude_frame": bool(state.det_exclude_frame),
            "frame_fallback_m": _det_frame_fallback_m(),
            "metric_cropsize": _num(state.det_metric_cropsize, 1.0),
            "min_confidence": (_num(state.det_min_confidence, 0.7)
                               if state.det_min_conf_enabled else None),
            "overlap": _num(state.det_overlap, 0.0),
            # None (shown as "—") without tile overlap, and at 0: the
            # manual says 0 disables the step.
            "dedup_overlap": (_num(state.det_dedup_overlap, 0.3)
                              if _num(state.det_overlap, 0.0) > 0
                              and _num(state.det_dedup_overlap, 0.0) > 0
                              else None),
            "saveplot": bool(state.det_saveplot),
            "saveresults": bool(state.det_saveresults),
            # True routes this job's outputs under the project's validation/.
            "save_to_validation": False,
            "kstart": int(kstart),
            "status": ("done" if resume_state == "done" else "pending"),
            "n_clasts": n_clasts,
            "error": None,
            "resume_state": resume_state,
            "resume_note": note,
            "dark_threshold": (_num(state.det_dark_threshold, 15)
                               if state.det_brightness_filter else None),
            "bright_threshold": (_num(state.det_bright_threshold, 245)
                                 if state.det_brightness_filter else None),
            "nodata_max_frac": _num(state.det_nodata_max_frac, 0.95),
            "roi_path": _probe_roi_for(image_path),
        }

    def _probe_roi_for(image_path: str) -> Optional[str]:
        """Per-image ROI GeoJSON path if it exists, else None: the project's
        ``geometries/`` folder first, then next to the image."""
        try:
            p = Path(image_path)
            name = f"{p.stem}_roi.geojson"
            proj = state.current_project
            if proj:
                cand = project_path(proj, "geometries") / name
                if cand.exists():
                    return str(cand)
            legacy = p.with_name(name)
            return str(legacy) if legacy.exists() else None
        except Exception:
            return None

    def _reset_job_files(idx: int):
        """Delete a queued job's .run.csv and final CSV; reset kstart and
        status so it restarts from scratch."""
        if idx < 0 or idx >= len(state.det_jobs):
            return
        job = state.det_jobs[idx]
        run_path, final_csv, _ = _expected_output_paths(
            job["image_path"], job["image_name"], job["metric_cropsize"])
        removed = []
        base = run_path[:-len(".run.csv")] if run_path.endswith(".run.csv") else run_path
        for p in (run_path, final_csv, base + ".run.grid.json",
                  base + ".run.contours.jsonl"):
            if os.path.exists(p):
                try:
                    os.remove(p)
                    removed.append(os.path.basename(p))
                except OSError as ex:
                    ui.notify(f"Could not remove {os.path.basename(p)}: {ex}. Close whatever holds it, "
                              "then Reset the job (queue, below) again.",
                              type="negative")
        job["kstart"] = 0
        job["status"] = "pending"
        job["n_clasts"] = None
        job["error"] = None
        job["resume_state"] = "fresh"
        job["resume_note"] = (
            f"Restart fresh ({len(removed)} previous file(s) removed)."
        )
        # Notify BEFORE refresh(): the refresh clears the slot ui.notify()
        # needs to resolve the client.
        if removed:
            ui.notify(f"Job #{idx+1} reset — removed {len(removed)} "
                      f"existing file(s). Will start from k=0.",
                      type="positive")
        else:
            ui.notify(f"Job #{idx+1} reset to k=0.", type="info")
        _render_queue.refresh()

    def _set_all_save_val(val: bool):
        for j in state.det_jobs:
            j["save_to_validation"] = bool(val)
        _render_queue.refresh()

    def _det_header_extra():
        ui.button("Save all to validation", icon="done_all",
                   on_click=lambda: _set_all_save_val(True)) \
.props("flat dense color=primary") \
.tooltip("Enable the ‘save to validation’ "
                 "checkbox on every queued job.")
        ui.button("Exclude all from validation", icon="remove_done",
                   on_click=lambda: _set_all_save_val(False)) \
.props("flat dense") \
.tooltip("Disable the ‘save to validation’ "
                 "checkbox on every queued job; "
                 "outputs revert to the canonical "
                 "results/ tree.")

    def _det_primary(job, idx):
        return job["image_name"]

    def _det_params(job, idx):
        if _job_mode(job) == ORTHO:
            params = (f"Ortho cropsize={job['metric_cropsize']}m"
                      f" overlap={job['overlap']:.2f}"
                      f" dedup={job['dedup_overlap'] if job['dedup_overlap'] is not None else '—'}")
        else:
            unit = job.get("unit_label")
            params = (f"Quadrat res={job['resolution']:.5g} per px, in "
                      f"{unit} (not metres)" if unit
                      else f"Quadrat res={job['resolution']:.6g} m/px")
            src = job.get("gsd_source")
            if src and src != "global":
                params += f" ({src})"
        if job["min_confidence"] is not None:
            params += f" conf≥{job['min_confidence']:.2f}"
        return params

    def _stamp_unit(job, out_dir, log):
        """A run scaled on an object of unknown length is not in metres: the
        CSV gets a ``unit`` column (as Digitize's Figures writes) and the
        log carries the disclaimer."""
        unit = job.get("unit_label")
        if not unit:
            return
        stem = job.get("out_stem") or os.path.splitext(job["image_name"])[0]
        csv = Path(out_dir or os.path.dirname(job["image_path"]))             / naming.quadrat_csv_name(stem)
        try:
            import pandas as _pd
            df = _pd.read_csv(csv)
            df["unit"] = unit
            df.to_csv(csv, index=False, float_format="%.5f")
        except Exception as ex:
            log(f"  [warn] could not write the unit column in {csv.name}: {ex}")
            return
        from functions import gauge as _g
        log(f"  Lengths are in {unit}, not metres (a unit column says so). "
            f"{_g.DISCLAIMER}")

    def _det_extra(job, idx):
        status = job.get("status", "pending")
        if _job_mode(job) == ORTHO:
            # A checkpoint resumes from its own last tile: the engine takes
            # max(this number, last tile + 1), so raising it skips tiles and
            # lowering it changes nothing. The tooltip says so, and ↺ is the
            # way to start over.
            kstart_inp = ui.number(value=job["kstart"], min=0,
                                   step=1).classes("w-24") \
.tooltip("Resume from this tile. Auto-filled "
                     "from the .run.csv at add time. The "
                     "backend reconciles this against the "
                     ".run.csv on Run; use ↺ to truly "
                     "start from scratch.")
            def _on_k(e, _idx=idx):
                try:
                    state.det_jobs[_idx]["kstart"] = int(e.value or 0)
                except Exception:
                    pass
            kstart_inp.on_value_change(_on_k)
            kstart_inp.set_enabled(status != "running")
        if job.get("n_clasts") is not None:
            ui.label(f"→ {job['n_clasts']} clasts") \
.classes("text-xs text-green-7")
        save_val_cb = ui.checkbox(
            "→ validation",
            value=bool(job.get("save_to_validation", False)),
        ).props("dense") \
.tooltip("When ticked, outputs (CSV, log, "
                  "figures) land under "
                  "<project>/validation/ instead of "
                  "results/. Use when running the "
                  "detector against truth quadrats.")

        def _on_save_val(e, _idx=idx):
            state.det_jobs[_idx]["save_to_validation"] = bool(e.value)

        save_val_cb.on_value_change(_on_save_val)

    def _det_delete(job, idx):
        # Per-job stop while running; reorder + reset + delete otherwise.
        status = job.get("status", "pending")
        if status == "running":
            def _stop_this(_e):
                state.det_stop_current = True
            ui.button(icon="skip_next", on_click=_stop_this) \
.props("flat dense color=warning") \
.tooltip("Finish the current tile, save "
                     "progress, and skip to the next "
                     "queued job.")
        else:
            if status == "pending":
                def _move_up(_e, _idx=idx):
                    if _idx > 0:
                        state.det_jobs[_idx-1], state.det_jobs[_idx] = (
                            state.det_jobs[_idx], state.det_jobs[_idx-1])
                        _render_queue.refresh()
                def _move_dn(_e, _idx=idx):
                    if _idx < len(state.det_jobs) - 1:
                        state.det_jobs[_idx], state.det_jobs[_idx+1] = (
                            state.det_jobs[_idx+1], state.det_jobs[_idx])
                        _render_queue.refresh()
                ui.button(icon="arrow_upward", on_click=_move_up) \
.props("flat dense") \
.set_enabled(idx > 0)
                ui.button(icon="arrow_downward", on_click=_move_dn) \
.props("flat dense") \
.set_enabled(idx < len(state.det_jobs) - 1)

            def _reset(_e, _idx=idx):
                _reset_job_files(_idx)
            reset_visible = (job.get("resume_state")
                             in ("resume", "done")
                             or status in ("stopped", "error", "done"))
            if reset_visible:
                ui.button(icon="restart_alt", on_click=_reset) \
.props("flat dense color=primary") \
.tooltip("Delete this (image, cropsize) "
                         "pair's .run.csv and final CSV, "
                         "reset kstart to 0. Use this "
                         "when you want a clean re-run "
                         "instead of resuming.")
            def _del(_e, _idx=idx):
                state.det_jobs.pop(_idx)
                _render_queue.refresh()
            ui.button(icon="delete", on_click=_del) \
.props("flat dense")

    def _det_footer(job, idx):
        note = job.get("resume_note")
        if note:
            note_color = {
                "resume": "text-orange-8",
                "done": "text-green-8",
                "fresh": "text-grey-7",
            }.get(job.get("resume_state", "fresh"), "text-grey-7")
            ui.label(note) \
.classes(f"text-xs {note_color} ml-12 mb-1")

    @ui.refreshable
    def _render_queue():
        render_queue(
            queue_container, state.det_jobs,
            title="Queued jobs",
            count_label=queue_count_label,
            primary_text=_det_primary,
            params_text=_det_params,
            render_extra=_det_extra,
            render_header_extra=_det_header_extra,
            render_row_footer=_det_footer,
            render_delete=_det_delete,
        )

    _render_queue()

    def _resume_state_notice(job):
        """(notify_type, message) for a job's resume_state."""
        rs = job.get("resume_state", "fresh")
        if rs == "resume":
            return ("warning",
                    f"{job['image_name']} (cropsize={job['metric_cropsize']}m): "
                    f"resuming from tile {job['kstart']}. Click ↺ on the "
                    f"row to start fresh instead.")
        if rs == "done":
            n = job.get("n_clasts")
            return ("info",
                    f"{job['image_name']} (cropsize={job['metric_cropsize']}m): "
                    f"final CSV already exists"
                    + (f" ({n} clasts)" if n is not None else "")
                    + ". The job is marked done; click ↺ to redo.")
        return ("positive",
                f"Added {job['image_name']} "
                f"(cropsize={job['metric_cropsize']}m) — fresh run.")

    def _add_to_queue():
        """Snapshot the current form once per checked file."""
        if not state.det_dir or not state.det_files:
            ui.notify("Set *Image directory* (Inputs, above) first.", type="warning")
            return
        targets = [f for f in state.det_files if state.det_files_checked.get(f)]
        if not targets:
            ui.notify("No files ticked in the file list (Inputs, above) — use *Select all*, or tick "
                      "files to process.", type="warning")
            return
        if state.det_mode == OBJECT_SCALE:
            # Nothing measures a photograph with no scale here: the
            # Resolution field is a metre fallback, which this mode has not.
            unscaled = [f for f in targets
                        if _det_object_scale(os.path.join(state.det_dir, f))
                        is None]
            targets = [f for f in targets if f not in unscaled]
            if unscaled:
                ui.notify(
                    f"{len(unscaled)} photograph(s) carry no scale yet and "
                    "were left out: draw the object on them (the ruler "
                    "button, or Next without a scale) — "
                    + ", ".join(unscaled[:3])
                    + ("…" if len(unscaled) > 3 else ""),
                    type="warning", multi_line=True, timeout=10000)
            if not targets:
                return
        n_resume = n_done = n_fresh = 0
        for f in targets:
            job = _snapshot_form_as_job(f)
            state.det_jobs.append(job)
            rs = job.get("resume_state", "fresh")
            if rs == "resume": n_resume += 1
            elif rs == "done": n_done += 1
            else: n_fresh += 1
        _render_queue.refresh()
        if len(targets) == 1:
            kind, msg = _resume_state_notice(state.det_jobs[-1])
            ui.notify(msg, type=kind, timeout=6000)
        else:
            parts = []
            if n_fresh: parts.append(f"{n_fresh} fresh")
            if n_resume: parts.append(f"{n_resume} to resume")
            if n_done: parts.append(f"{n_done} already done")
            ui.notify(
                f"Added {len(targets)} job(s) to the queue "
                f"({', '.join(parts)}). Use the ↺ button on a row to restart "
                f"it from scratch.",
                type="positive", timeout=6000)

    add_btn.on_click(_add_to_queue)

    def do_run():
        if not state.det_jobs:
            ui.notify("The queue is empty — press *Add to queue* (above) first.",
                      type="warning")
            return
        try:
            from detectors import get_backend as _gb_pre
            _be_pre = _gb_pre(_det_run_model())
            if _be_pre is not None and not _be_pre.is_available():
                _msg = _model_unavailable_text(_be_pre)
                log_widget.push("[detect] " + _msg)
                ui.notify(_msg, type="negative", multi_line=True, timeout=0,
                          close_button=True)
                return
        except Exception:
            pass    # the check must never stop a good run

        state.det_stop_all = False
        state.det_stop_current = False
        _eta_state["first_tile_time"] = None
        _eta_state["first_tile_index"] = None
        _eta_state["last_seen_total"] = None

        run_btn.props("loading")
        _disable(run_btn, "Queue running — Stop all (right) halts it")
        _disable(add_btn, "Queue running — add files once it has finished")
        for b in (stop_current_btn, stop_all_btn):
            b.set_visibility(True)
            _enable(b)
        # Clearing the model cache mid-run would crash the worker.
        if _reload_model_button() is not None:
            _disable(_reload_model_button(),
                     "Queue running — reload once it has finished")
        log_widget.clear()

        from datetime import datetime as _dt
        log_widget.push("=" * 60)
        log_widget.push(f"  Batch started: {_dt.now().isoformat(timespec='seconds')} "
                        f"({len(state.det_jobs)} job(s))")
        log_widget.push("=" * 60)
        log_widget.push(f"  active project    = {state.current_project or '(none)'}")
        log_widget.push(f"  device            = {state.devicemode} #{state.devicenumber}")
        if state.current_project:
            log_widget.push(f"  output (CSV) dir  = {project_path(state.current_project, 'vectors')}")
            log_widget.push(f"  figures dir       = {project_path(state.current_project, 'figures')}")
        else:
            log_widget.push("  output dir        = (no project — alongside input image)")
        log_widget.push("")
        for i, job in enumerate(state.det_jobs):
            mode_bits = (f"cropsize={job['metric_cropsize']}m "
                         f"overlap={job['overlap']:.2f} "
                         f"dedup={job['dedup_overlap']}"
                         if _job_mode(job) == ORTHO
                         else ((f"res={job['resolution']:.5g} per px in "
                                f"{job['unit_label']} "
                                if job.get("unit_label")
                                else f"res={job['resolution']:.6g} m/px ")
                               + f"[{job.get('gsd_source', 'global')}]"))
            conf_bit = (f" conf≥{job['min_confidence']:.2f}"
                        if job['min_confidence'] is not None
                        else " conf=off")
            log_widget.push(f"  [#{i+1}] {job['image_name']} — "
                            f"{MODE_LABELS[_job_mode(job)]} {mode_bits}{conf_bit} "
                            f"(kstart={job['kstart']})")
        log_widget.push("=" * 60)
        log_widget.push("")

        if any(_job_mode(j) == ORTHO for j in state.det_jobs):
            progress_bar.value = 0.0
            progress_label.set_text("Initializing…")
            progress_bar.set_visibility(True)
            progress_label.set_visibility(True)

        # The worker runs in a thread; its notifications (resume, done,
        # stopped, failed) need the page's client context or they die with
        # "the slot stack for this task is empty" and take the run's
        # terminal handling with them.
        _page = context.client

        def _worker():
            with on_page(_page):
                _worker_body()

        def _worker_body():
            run_start_wall = _time.time()
            _gui_log_handler = None
            try:
                from functions._logging import (
                    NiceGUILogHandler,
                    attach_gui_handler, detach_gui_handler,
                    attach_file_handler, detach_file_handler,
                )
                _gui_log_handler = NiceGUILogHandler(log_widget)
                attach_gui_handler(_gui_log_handler)

                out_dir_str, fig_dir_str = _resolve_dirs()

                pending_idx = [i for i, j in enumerate(state.det_jobs)
                               if j.get("status") in ("pending", "stopped", "error", "interrupted")]
                if not pending_idx:
                    log_widget.push("Nothing pending in the queue.")
                    return

                def _stop_check():
                    return state.det_stop_all or state.det_stop_current

                for ji, job_idx in enumerate(pending_idx, start=1):
                    if state.det_stop_all:
                        log_widget.push("⏹  Halt requested; stopping the queue.")
                        break
                    job = state.det_jobs[job_idx]
                    state.det_stop_current = False
                    job["status"] = "running"
                    _render_queue.refresh()
                    _eta_state["first_tile_time"] = None
                    _eta_state["first_tile_index"] = None
                    log_widget.push(
                        f"\n[{ji}/{len(pending_idx)}] {job['image_name']} — "
                        f"{MODE_LABELS[_job_mode(job)]} cropsize={job['metric_cropsize']}m "
                        f"kstart={job['kstart']}"
                    )

                    # Each entry runs as a single-image call so per-job params
                    # and kstart are honoured. Validation outputs land in a
                    # per-image subfolder.
                    image_path = job["image_path"]
                    if job.get("save_to_validation"):
                        _base_val, _base_fig = _resolve_dirs(
                            to_validation=True)
                        _val_stem = os.path.splitext(job["image_name"])[0]
                        if _base_val is not None:
                            import pathlib as _pl
                            _sub = _pl.Path(_base_val) / _val_stem
                            _sub.mkdir(parents=True, exist_ok=True)
                            _fig_sub = _pl.Path(_base_fig) / _val_stem
                            _fig_sub.mkdir(parents=True, exist_ok=True)
                            out_dir_str = str(_sub)
                            fig_dir_str = str(_fig_sub)
                        else:
                            out_dir_str, fig_dir_str = _base_val, _base_fig
                    else:
                        out_dir_str, fig_dir_str = _resolve_dirs()
                    # The outputs carry the model that runs now, whatever
                    # was in force when the job was queued.
                    job["out_stem"] = naming.with_model(
                        job.get("out_stem")
                        or os.path.splitext(job["image_name"])[0],
                        _det_run_model())
                    # Where this job's overlay lands, for the preview below.
                    job["fig_dir"] = fig_dir_str
                    if _job_mode(job) == ORTHO:
                        from functions.clasts_detection import inspect_run_state
                        info = inspect_run_state(out_dir_str, image_path,
                                                 job["metric_cropsize"],
                                                 out_stem=job.get("out_stem"))
                        stem = job.get("out_stem") or os.path.splitext(job["image_name"])[0]
                        final_name = naming.detection_csv_name(
                            stem, job['metric_cropsize'])
                        final_dir = out_dir_str or os.path.dirname(image_path)
                        final_csv = os.path.join(final_dir, final_name)
                        if not info["exists"] and os.path.exists(final_csv):
                            job["status"] = "done"
                            log_widget.push(
                                f"  ⏭  Final CSV already exists — skipping. "
                                f"Delete it (or its .run.csv) to re-run."
                            )
                            _render_queue.refresh()
                            continue

                        if info["exists"] and info["n_clasts"] > 0:
                            with on_page(_page):
                                ui.notify(
                                    f"Resuming from tile {info['resume_k']} "
                                    f"({info['n_clasts']} clasts already detected).",
                                    type="info",
                                )
                            log_widget.push(
                                f"  ↩  Resuming from tile {info['resume_k']} "
                                f"({info['n_clasts']} clasts from previous run)."
                            )

                    def _job_progress(_ji, event, payload, job=job, log=log_widget):
                        def _notify(*a, **k):
                            with on_page(_page):
                                ui.notify(*a, **k)
                        if event == "done":
                            job["n_clasts"] = int(payload.get("n_clasts", 0))
                            job["status"] = "done"
                            log.push(f"  ✓ {job['n_clasts']} clasts.")
                            _notify(
                                f"Detection complete — {job['n_clasts']} clasts "
                                f"({job.get('image_name', '')})",
                                type="positive",
                            )
                        elif event == "stopped":
                            job["status"] = "stopped"
                            stopped_tile = payload.get("stopped_tile")
                            log.push("  ⏹  Stopped mid-run.")
                            _notify(
                                f"Detection stopped by user"
                                + (f" at tile {stopped_tile}" if stopped_tile is not None else "")
                                + f" ({job.get('image_name', '')}). Run all queued (above) resumes it.",
                                type="warning",
                            )
                        elif event == "error":
                            job["status"] = "error"
                            job["error"] = payload.get("message", "?")
                            log.push(f"  ✗ {job['error']}")
                            _notify(
                                f"Detection failed: {job['error']} "
                                f"— see the log (below); fix the input, then Run all queued again.",
                                type="negative",
                                timeout=0,
                            )
                        _render_queue.refresh()

                    # Per-job log file: <project>/output_results/logs/
                    # <stem>_detection_<ts>.log (or <image_dir>/logs/).
                    _img_stem_log = os.path.splitext(job["image_name"])[0]
                    _log_ts = _dt.now().strftime("%Y%m%dT%H%M%S")
                    if state.current_project:
                        _log_dir = project_path(state.current_project, "logs")
                    else:
                        _log_dir = Path(image_path).parent / "logs"
                    _job_log_path = (
                        Path(_log_dir) / f"{_img_stem_log}_detection_{_log_ts}.log"
                    )
                    _fh = None
                    try:
                        _fh = attach_file_handler(_job_log_path)
                        log_widget.push(
                            f"  📄 Log file: {_job_log_path}"
                        )
                    except OSError as _fh_err:
                        log_widget.push(
                            f"  [warn] could not create log file "
                            f"{_job_log_path}: {_fh_err}"
                        )
                    try:
                        with capture_stdout_to_log(log_widget, on_line=_on_log_line):
                            from detectors import get_backend as _get_backend
                            from detectors import run_detect_jobs as _run_jobs
                            _det_backend = _get_backend(_det_run_model())
                            # The wrapper settles the row from the return
                            # value when a backend never reports "done".
                            _run_jobs(
                                _det_backend,
                                mode=_job_mode(job),
                                jobs=[{"path": image_path, "kstart": job["kstart"],
                                       "out_stem": job.get("out_stem")}],
                                resolution=job["resolution"],
                                metric_cropsize=job["metric_cropsize"],
                                plot=False,
                                # The overlay's scale bar and the histogram
                                # are labelled in metres, which a run in an
                                # object's own unit is not.
                                saveplot=(job["saveplot"]
                                          and not job.get("unit_label")),
                                # Save CSV (Advanced): the switch was not in
                                # the job and every run wrote.
                                saveresults=job.get("saveresults", True),
                                devicemode=state.devicemode.lower(),
                                devicenumber=int(state.devicenumber),
                                min_confidence=job["min_confidence"],
                                overlap=job["overlap"],
                                dedup_method="iou",
                                dedup_overlap=job["dedup_overlap"],
                                stop_check=_stop_check,
                                output_dir=out_dir_str,
                                figures_dir=fig_dir_str,
                                progress_callback=_job_progress,
                                log_fn=log_widget.push,
                                # .get: jobs restored from an older session
                                # may lack these keys.
                                dark_threshold=job.get("dark_threshold"),
                                bright_threshold=job.get("bright_threshold"),
                                nodata_max_frac=job.get("nodata_max_frac",
                                                          0.95),
                                roi_path=job.get("roi_path"),
                                exclude_frame=job.get("exclude_frame", True),
                                frame_fallback_m=job.get("frame_fallback_m"),
                            )
                            if job.get("unit_label"):
                                if job["saveplot"]:
                                    log_widget.push(
                                        "  Plots skipped: they are labelled "
                                        "in metres and this run is not.")
                                _stamp_unit(job, out_dir_str, log_widget.push)
                    except Exception as ex:
                        import traceback
                        job["status"] = "error"
                        job["error"] = f"{type(ex).__name__}: {ex}"
                        log_widget.push(f"  ✗ {job['error']}")
                        log_widget.push(traceback.format_exc())
                        _render_queue.refresh()
                    finally:
                        if _fh is not None:
                            detach_file_handler(_fh)

                run_elapsed = _time.time() - run_start_wall
                done_n = sum(1 for j in state.det_jobs if j.get("status") == "done")
                stopped_n = sum(1 for j in state.det_jobs if j.get("status") == "stopped")
                err_n = sum(1 for j in state.det_jobs if j.get("status") == "error")
                if state.det_stop_all:
                    log_widget.push(f"\n⏹  Queue halted after {_format_eta(run_elapsed)}.")
                else:
                    log_widget.push(f"\n✅  Queue finished in {_format_eta(run_elapsed)}.")
                log_widget.push(
                    f"    done={done_n}, stopped={stopped_n}, errors={err_n}"
                )

                quadrat_jobs = [j for j in state.det_jobs
                                if _job_mode(j) == QUADRAT and j["saveplot"]]
                if quadrat_jobs:
                    candidate_pngs = []
                    for j in quadrat_jobs:
                        # The overlay is <stem>_overlay.png in the job's
                        # figures folder; results from before the rename
                        # are still previewed as <stem>_ellipses.png.
                        stems = [s for s in (j.get("out_stem"),
                                             os.path.splitext(j["image_name"])[0])
                                 if s]
                        dirs = [d for d in (j.get("fig_dir"), state.det_dir) if d]
                        for d in dirs:
                            for s in stems:
                                for suffix in ("_overlay.png", "_ellipses.png"):
                                    png_path = os.path.join(d, s + suffix)
                                    if (os.path.exists(png_path)
                                            and png_path not in candidate_pngs):
                                        candidate_pngs.append(png_path)
                    if candidate_pngs:
                        latest = max(candidate_pngs, key=os.path.getmtime)
                        state.det_last_preview = latest
                        try:
                            import base64
                            with open(latest, "rb") as f:
                                data_url = ("data:image/png;base64,"
                                            + base64.b64encode(f.read()).decode())
                            preview_container.clear()
                            with preview_container:
                                ui.image(data_url).classes("w-full") \
.style("max-width: 900px;")
                            preview_label.set_text(
                                f"Last detection preview: {os.path.basename(latest)}")
                            preview_label.set_visibility(True)
                        except Exception as e:
                            log_widget.push(f"   (could not load preview: {e})")
            except Exception as e:
                import traceback
                log_widget.push(f"❌  {type(e).__name__}: {e}")
                log_widget.push(traceback.format_exc())
            finally:
                if _gui_log_handler is not None:
                    detach_gui_handler(_gui_log_handler)
                run_btn.props(remove="loading")
                _enable(run_btn)
                _enable(add_btn)
                for b in (stop_current_btn, stop_all_btn):
                    b.set_visibility(False)
                if _reload_model_button() is not None:
                    _enable(_reload_model_button())
                progress_bar.set_visibility(False)
                progress_label.set_visibility(False)
                progress_bar.value = 0.0
                _render_queue.refresh()

        threading.Thread(target=_worker, daemon=True).start()

    def do_stop_current():
        state.det_stop_current = True
        log_widget.push("⏭  Skipping current file. Will move on after the current tile.")
        _disable(stop_current_btn, "Skip requested — waits for the current tile")

    def do_stop_all():
        state.det_stop_all = True
        log_widget.push("⏹  Halt requested. Will exit after the current tile.")
        _disable(stop_current_btn, "Halt requested — waits for the current tile")
        _disable(stop_all_btn, "Halt requested — waits for the current tile")

    run_btn.on_click(do_run)
    stop_current_btn.on_click(do_stop_current)
    stop_all_btn.on_click(do_stop_all)

    def _on_project_change():
        _seed_det_dir(force=True)
        refresh_files()
    _project_change_hook["fn"] = _on_project_change
    _seed_det_dir(force=False)


def build_rasterize_tab():
    if not state.ras_output_dir:
        state.ras_output_dir = default_starting_dir("rasters")

    def _seed_rasterize(force=False):
        # Newest clast CSV (merge preferred), its stem-matched ortho, and
        # the project's rasters/ as output dir.
        from functions import project_defaults as _pdf
        proj = state.current_project
        csv = _pdf.best_clast_csv(proj)
        _seed_default(_ras_csv_row._path_input, csv, force=force)
        if csv is not None:
            _seed_default(_ras_tif_row._path_input,
                          _pdf.source_ortho_for(csv.name, proj), force=force)
        state.ras_output_dir = default_starting_dir("rasters")

    def _refresh_rasterize_on_open():
        # Another project's CSV or ortho is stale; this project's stays.
        if drop_foreign_paths("ras_csv", "ras_tif"):
            _seed_rasterize(force=True)
        else:
            _seed_rasterize()
        state.ras_output_dir = default_starting_dir("rasters")
    register_tab_refresh("Rasterize", _refresh_rasterize_on_open)

    def _on_proj_change():
        _seed_rasterize(force=True)
        # The project's addons.json names extra fields.
        try:
            _field_checkboxes.refresh()
        except Exception:
            pass
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Rasterize")
    ui.label("Grain-size rasters from a clast CSV: pick fields × statistics, "
             "add them to the queue (each job keeps its own percentile), run "
             "the queue.").classes("text-sm text-grey-7")

    _ras_csv_row = path_input_with_browse(
        "Clast list CSV", "ras_csv", kind="file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        default_kind="vectors",
    )
    _ras_tif_row = path_input_with_browse(
        "Source ortho-image (.tif)", "ras_tif", kind="file",
        filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")],
        default_kind="images",
    )
    _seed_rasterize()

    ui.separator()
    ui.markdown("**Selection** — choose the fields and statistics to add to the queue.")

    _ras_addons = _project_addons()

    with ui.expansion(
        "Custom equations (user add-ons)",
        icon="science",
    ).classes("w-full"):
        ui.label("Per-clast formulas from `<project>/addons.json` or "
                 "`<repo>/user_addons.json` (name, inputs, expression, "
                 "units); see Custom equations in the user manual.") \
            .classes("text-sm text-grey-7") \
            .tooltip("Registered fields appear in the field grid below, "
                     "the Map tab's field selector and the report. "
                     "Expressions are sandboxed: arithmetic, comparisons, "
                     "ternaries and sqrt/log/exp/sin/cos/atan2/abs/min/"
                     "max/pow only; anything else is rejected at load.")
        if _ras_addons:
            with ui.row().classes("gap-2 mt-1 flex-wrap"):
                ui.label(
                    f"Registered: {len(_ras_addons)} addon(s)"
                ).classes("text-sm font-bold text-positive")
                for ad in _ras_addons:
                    ui.label(
                        f"{ad.name} [{ad.units or 'dimensionless'}]"
                    ).classes("text-xs font-mono px-2 "
                              "bg-grey-2 rounded") \
.tooltip(ad.description
                              or ad.expression)
        else:
            ui.label(
                "(no addons loaded — drop an addons.json file at "
                "one of the paths above)"
            ).classes("text-sm text-grey-6 italic")

        # In-page addon creator; writes <project>/addons.json. A restart is
        # needed to activate the field because load_addons runs at startup.
        ui.separator().classes("mt-3 mb-1")
        ui.label("Create a new custom field").classes(
            "text-sm font-bold mt-1 text-grey-8")
        with ui.card().classes("w-full mt-1").props("flat bordered"):
            with ui.column().classes("w-full gap-2 p-2"):
                with ui.row().classes("gap-2 w-full flex-wrap"):
                    addon_name_inp = ui.input(
                        label="Field name",
                        placeholder="e.g. My_ratio",
                    ).classes("flex-1 min-w-[12rem]") \
.props("dense outlined") \
.tooltip("Column name added to the per-clast CSV. "
                         "Use underscores, no spaces.")
                    addon_units_inp = ui.input(
                        label="Units",
                        placeholder="e.g. mm, N/m², phi",
                    ).classes("w-32") \
.props("dense outlined")
                addon_inputs_inp = ui.input(
                    label="Input columns (comma-separated)",
                    placeholder="e.g. Clast_length, Clast_width",
                ).classes("w-full") \
.props("dense outlined") \
.tooltip("Column names the expression reads. "
                     "Separate multiple names with commas.")
                addon_expr_inp = ui.input(
                    label="Expression",
                    placeholder="e.g. Clast_width / Clast_length",
                ).classes("w-full") \
.props("dense outlined") \
.tooltip("Math expression evaluated per-clast. "
                     "Safe functions: sqrt, log, exp, sin, cos, "
                     "atan2, abs, min, max, pow. "
                     "No attribute access or arbitrary calls.")
                addon_desc_inp = ui.input(
                    label="Description (optional)",
                    placeholder="e.g. Width-to-length ratio (0–1)",
                ).classes("w-full") \
.props("dense outlined")
                with ui.row().classes("items-center gap-2"):
                    addon_factor_inp = ui.number(
                        label="Display scale factor",
                        value=1.0, min=1e-12,
                        format="%.6g",
                    ).classes("w-36") \
.props("dense outlined") \
.tooltip("Multiply the raw expression result by this factor "
                         "before storing. Use 1000 to convert metres → mm, "
                         "etc. Stored as the 'factor' field in the JSON.")
                    ui.label(
                        "× raw value   (1.0 = no conversion)"
                    ).classes("text-xs text-grey-6")
                addon_status = ui.label("").classes(
                    "text-sm italic text-grey-6")

                def _validate_addon_expr():
                    from functions.addons import Addon as _AddonCls
                    expr = (addon_expr_inp.value or "").strip()
                    name = (addon_name_inp.value or "").strip()
                    if not name:
                        addon_status.set_text(
                            "⚠ Enter a field name first.")
                        addon_status.classes(
                            remove="text-positive text-negative",
                            add="text-warning")
                        return False
                    if not expr:
                        addon_status.set_text(
                            "⚠ Enter an expression first.")
                        addon_status.classes(
                            remove="text-positive text-negative",
                            add="text-warning")
                        return False
                    _inputs = [
                        x.strip()
                        for x in (addon_inputs_inp.value or "").split(",")
                        if x.strip()
                    ]
                    ad = _AddonCls(
                        name=name,
                        inputs=_inputs,
                        expression=expr,
                        units=(addon_units_inp.value or "").strip(),
                        description=(addon_desc_inp.value or "").strip(),
                        factor=float(addon_factor_inp.value or 1.0),
                    )
                    if ad.validate():
                        addon_status.set_text(
                            "✓ Expression is valid.")
                        addon_status.classes(
                            remove="text-warning text-negative italic",
                            add="text-positive")
                        return True
                    else:
                        addon_status.set_text(
                            f"✗ {ad._error}")
                        addon_status.classes(
                            remove="text-warning text-positive italic",
                            add="text-negative")
                        return False

                def _save_addon():
                    import json as _json
                    if not _validate_addon_expr():
                        return
                    name = (addon_name_inp.value or "").strip()
                    _inputs = [
                        x.strip()
                        for x in (addon_inputs_inp.value or "").split(",")
                        if x.strip()
                    ]
                    entry = {
                        "name": name,
                        "inputs": _inputs,
                        "expression": (addon_expr_inp.value or "").strip(),
                        "units": (addon_units_inp.value or "").strip(),
                        "description": (addon_desc_inp.value or "").strip(),
                        "factor": float(addon_factor_inp.value or 1.0),
                    }
                    proj = state.current_project
                    if not proj:
                        ui.notify("Select a project (Project, left panel) first.",
                                  type="warning")
                        return
                    addon_path = project_path(proj) / "addons.json"
                    existing = []
                    if addon_path.exists():
                        try:
                            raw = _json.loads(
                                addon_path.read_text(encoding="utf-8"))
                            existing = [
                                e for e in
                                (raw if isinstance(raw, list) else [raw])
                                if isinstance(e, dict)
                                and not e.get("_comment")
                            ]
                        except Exception:
                            pass
                    existing = [
                        e for e in existing
                        if e.get("name") != name
                    ]
                    existing.append(entry)
                    addon_path.parent.mkdir(parents=True, exist_ok=True)
                    addon_path.write_text(
                        _json.dumps(existing, indent=2),
                        encoding="utf-8")
                    ui.notify(
                        f"Saved '{name}' to {addon_path.name}. "
                        "Restart the app to activate the new field.",
                        type="positive", timeout=6000)
                    addon_status.set_text(
                        f"✓ Saved to {addon_path}. "
                        f"Restart to activate.")
                    addon_status.classes(
                        remove="text-negative text-warning italic",
                        add="text-positive")

                def _export_addon_json():
                    import json as _json
                    if not _validate_addon_expr():
                        return
                    name = (addon_name_inp.value or "").strip()
                    _inputs = [
                        x.strip()
                        for x in (addon_inputs_inp.value or "").split(",")
                        if x.strip()
                    ]
                    entry = {
                        "name": name,
                        "inputs": _inputs,
                        "expression": (addon_expr_inp.value or "").strip(),
                        "units": (addon_units_inp.value or "").strip(),
                        "description": (addon_desc_inp.value or "").strip(),
                        "factor": float(addon_factor_inp.value or 1.0),
                    }
                    init_dir = ""
                    if state.current_project:
                        init_dir = str(project_path(state.current_project))
                    save_path = native_save_file_picker(
                        "Export add-on JSON",
                        filetypes=[("JSON file", "*.json"),
                                   ("All files", "*.*")],
                        initialfile=f"{name}_addon.json",
                        defaultextension=".json",
                        initialdir=init_dir,
                    )
                    if not save_path:
                        return
                    Path(save_path).write_text(
                        _json.dumps([entry], indent=2),
                        encoding="utf-8")
                    ui.notify(
                        f"Exported '{name}' to {Path(save_path).name}.",
                        type="positive")

                with ui.row().classes("gap-2 mt-1"):
                    ui.button("Validate", icon="check_circle",
                              on_click=_validate_addon_expr) \
.props("flat color=primary") \
.tooltip("Check the expression for syntax errors "
                         "and disallowed constructs.")
                    ui.button("Add to project addons.json",
                              icon="save",
                              on_click=_save_addon) \
.props("color=primary outline") \
.tooltip("Append this entry to <project>/addons.json "
                         "(creating it if absent). "
                         "Restart the app to activate.")
                    ui.button("Export JSON…",
                              icon="download",
                              on_click=_export_addon_json) \
.props("flat color=grey") \
.tooltip("Save this add-on as a standalone JSON file that can be "
                         "copied to any project as addons.json.")

    # Fields that need an Equivalent_diameter column; ticking one on a CSV
    # without it warns instead of producing an all-NaN raster.
    _TRANSPORT_FIELDS_LOWER = {
        "van_rijn_dimensionless_diameter",
        "soulsby_critical_shields",
        "shields_critical_shear_stress",
        "shields_critical_shear_velocity",
        "shields_critical_grain_reynolds_number",
        "hjulstrom_deposition_velocity",
        "hjulstrom_erosion_velocity",
        "leroux_wave_orbital_velocity",
    }

    def _check_equiv_diam_warning(fname: str, checked: bool) -> None:
        """Warn if a transport field is ticked but Equivalent_diameter is absent."""
        if not checked or fname.lower() not in _TRANSPORT_FIELDS_LOWER:
            return
        csv_path = state.ras_csv
        if not csv_path:
            return
        try:
            import pandas as _pd
            df = _pd.read_csv(csv_path, nrows=10)
            if ("Equivalent_diameter" not in df.columns
                    or df["Equivalent_diameter"].isna().all()):
                ui.notify(
                    f"'{fname}' requires Equivalent_diameter, which is absent "
                    "or all-NaN in the selected CSV. Re-run the Detect tab, then "
                    "pick the CSV again (Inputs, above).",
                    type="warning",
                    timeout=0,
                )
        except Exception:
            pass

    field_options = [
        # sizes
        "Clast_length", "Clast_width", "Ellipse_major_axis", "Ellipse_minor_axis",
        "Equivalent_diameter", "Perimeter", "Surface_area",
        # shape + appearance
        "Eccentricity", "Solidity", "Mean_intensity",
        "Clast_elongation", "Clast_circularity",
        # detection + orientation
        "Score", "Orientation",
        # sediment-transport thresholds
        "Van_Rijn_dimensionless_diameter",
        "Soulsby_critical_shields",
        "Shields_critical_shear_stress", "Shields_critical_shear_velocity",
        "Shields_critical_grain_reynolds_number",
        "Hjulstrom_deposition_velocity", "Hjulstrom_erosion_velocity",
        "Leroux_wave_orbital_velocity",
    ]
    _base_field_options = list(field_options)

    def _field_options():
        """The shipped fields plus the active project's add-on fields (its
        addons.json), read when the list is drawn: a project picked after
        the page was built, or an add-on saved from the creator below,
        shows its fields without a restart."""
        opts = list(_base_field_options)
        try:
            for ad in _project_addons():
                if ad.name not in opts:
                    opts.append(ad.name)
        except Exception:
            pass
        for f in opts:
            state.ras_fields_checked.setdefault(f, False)
        return opts

    field_options = _field_options()
    parameter_options = ["quantile", "average", "std", "cv", "kurtosis",
                         "skewness", "mode", "sorting", "distribution",
                         "d_percentiles",
                         "folk_ward_sorting", "folk_ward_skewness",
                         "folk_ward_kurtosis",
                         "density",
                         "packing_index", "packing_clustering"]

    # Coherence groups enforced at Add-to-queue: FIELD_INDEPENDENT indices
    # take no field (one raster each); SIZE_ONLY statistics are phi-based;
    # INVALID_ON_ORIENTATION statistics are undefined for axial data.
    _PARAMS_FIELD_INDEPENDENT = {"density", "packing_index",
                                 "packing_clustering"}
    _PARAMS_SIZE_ONLY = {"folk_ward_sorting", "folk_ward_skewness",
                         "folk_ward_kurtosis"}
    _PARAMS_INVALID_ON_ORIENTATION = {"std", "cv", "skewness", "kurtosis",
                                      "mode", "folk_ward_sorting",
                                      "folk_ward_skewness",
                                      "folk_ward_kurtosis"}
    from functions.units import is_size_field_for_phi as _is_size_field

    def _is_orientation_field(fname):
        return "orientation" in (fname or "").lower()

    for f in field_options:
        state.ras_fields_checked.setdefault(f, False)
    for p in parameter_options:
        state.ras_parameters_checked.setdefault(p, False)

    selection_summary = ui.label("").classes("text-sm text-grey-7")

    # Per-job parameter widgets whose visibility follows the selection.
    _per_job_widgets = {"percentile_row": None, "T_input": None,
                        "rho_water_input": None, "bin_row": None}

    def _refresh_selection_summary():
        n_fields = sum(1 for v in state.ras_fields_checked.values() if v)
        _checked_params = [p for p, v in state.ras_parameters_checked.items()
                           if v]
        n_dep = sum(1 for p in _checked_params
                    if p not in _PARAMS_FIELD_INDEPENDENT)
        n_ind = sum(1 for p in _checked_params
                    if p in _PARAMS_FIELD_INDEPENDENT)
        selection_summary.set_text(
            f"Currently selected: {n_fields} field(s) × {n_dep} per-field "
            f"statistic(s) + {n_ind} field-independent index(es) = up to "
            f"{n_fields * n_dep + n_ind} job(s) "
            "(incoherent field × statistic combinations are skipped)"
        )
        if _per_job_widgets["percentile_row"] is not None:
            _per_job_widgets["percentile_row"].set_visibility(
                bool(state.ras_parameters_checked.get("quantile", False)))
        if _per_job_widgets["T_input"] is not None:
            _per_job_widgets["T_input"].set_visibility(
                bool(state.ras_fields_checked.get("Leroux_wave_orbital_velocity", False)))
        if _per_job_widgets["rho_water_input"] is not None:
            _any_transport = any(
                state.ras_fields_checked.get(f, False)
                for f in state.ras_fields_checked
                if f.lower() in _TRANSPORT_FIELDS_LOWER
            )
            _per_job_widgets["rho_water_input"].set_visibility(_any_transport)
        if _per_job_widgets["bin_row"] is not None:
            _per_job_widgets["bin_row"].set_visibility(
                bool(state.ras_parameters_checked.get("packing_index", False))
                or bool(state.ras_parameters_checked.get("packing_clustering", False))
            )

    def _set_all(d, value):
        for k in d:
            d[k] = value
        _refresh_selection_summary()

    with ui.row().classes("w-full gap-8 pm-ras-cards") as _ras_cards_row:
        with ui.column().classes("min-w-[20rem]"):
            with ui.row().classes("items-center"):
                ui.label("Fields").classes("text-base font-bold")
                ui.button("All", on_click=lambda: _set_all(state.ras_fields_checked, True)) \
.props("dense outline").tooltip("Check every field")
                ui.button("None", on_click=lambda: _set_all(state.ras_fields_checked, False)) \
.props("dense outline").tooltip("Uncheck every field")
            with ui.card().classes("w-full") \
.style("max-height: 240px; overflow-y: auto;"):
                _FIELD_DESC = {
                    "Clast_length":  "Long axis of the fitted-ellipse approximation, in metres.",
                    "Clast_width":   "Short axis of the fitted-ellipse approximation, in metres.",
                    "Ellipse_major_axis": "Major axis of the segmentation-mask's best-fit ellipse, in metres.",
                    "Ellipse_minor_axis": "Minor axis of the segmentation-mask's best-fit ellipse, in metres.",
                    "Equivalent_diameter": "Diameter of a circle with the same area as the mask: 2·√(A/π), in metres.",
                    "Surface_area":  "Mask surface area in m². Used by `packing_index`.",
                    "Score":         "Mask R-CNN classifier confidence (0–1).",
                    "Orientation":   "Major-axis orientation, axial (0–180°). Statistical operators "
                                     "handle the axial nature automatically.",
                    "Clast_elongation":   "Clast_width / Clast_length (0–1, 1 = round).",
                    "Ellipse_elongation": "Ellipse_minor / Ellipse_major (0–1).",
                    "Clast_circularity":  "Equivalent_diameter / Clast_length (0–1, 1 = round).",
                    "Ellipse_circularity":"Equivalent_diameter / Ellipse_major_axis.",
                    "Van_Rijn_dimensionless_diameter":
                                     "D* = D·[g(s−1)/ν²]^(1/3). Van Rijn (1984).",
                    "Soulsby_critical_shields":
                                     "θ_cr from Soulsby & Whitehouse (1997) single-equation fit.",
                    "Shields_critical_shear_stress":
                                     "τ_cr (Pa) = θ_cr·(ρ_s−ρ_w)·g·D.",
                    "Shields_critical_shear_velocity":
                                     "u* (m/s) = √(τ_cr / ρ_w).",
                    "Shields_critical_grain_reynolds_number":
                                     "Re* = u*·D/ν (dimensionless).",
                    "Hjulstrom_deposition_velocity":
                                     "Sundborg/Hjulström deposition velocity (m/s).",
                    "Hjulstrom_erosion_velocity":
                                     "Hjulström erosion velocity (m/s).",
                    "Leroux_wave_orbital_velocity":
                                     "Le Roux (2007) wave orbital velocity at wave period T "
                                     "(set in Per-job parameters). Use for relative comparison; "
                                     "absolute values are not fully verified.",
                }
                @ui.refreshable
                def _field_checkboxes():
                    for fname in _field_options():
                        cb = ui.checkbox(fname)
                        cb.bind_value(state.ras_fields_checked, fname)
                        cb.on_value_change(
                            lambda e, f=fname: (
                                _refresh_selection_summary(),
                                _check_equiv_diam_warning(f, e.value),
                            )
                        )
                        desc = _FIELD_DESC.get(fname)
                        if desc:
                            cb.tooltip(desc)
                _field_checkboxes()

        with ui.column().classes("min-w-[16rem]"):
            with ui.row().classes("items-center"):
                ui.label("Statistics").classes("text-base font-bold")
                ui.button("All", on_click=lambda: _set_all(state.ras_parameters_checked, True)) \
.props("dense outline")
                ui.button("None", on_click=lambda: _set_all(state.ras_parameters_checked, False)) \
.props("dense outline")
            with ui.card().classes("w-full") \
.style("max-height: 240px; overflow-y: auto;"):
                _PARAM_DESC = {
                    "quantile": "Quantile of `field` per cell at the percentile set below "
                                "(e.g. percentile=0.5 → D50). One band.",
                    "density":  "Clast count per cell area. Field-agnostic; only one density job "
                                "is produced per Add click regardless of which fields are checked.",
                    "average":  "Arithmetic mean of `field` per cell.",
                    "std":      "Sample standard deviation of `field` per cell.",
                    "cv":       "Coefficient of variation of `field` per cell "
                                "(std / mean). Dimensionless dispersion; higher "
                                "= more size variability. Not valid on Orientation.",
                    "kurtosis": "Fisher kurtosis of `field` per cell (population shape).",
                    "skewness": "Skewness of `field` per cell.",
                    "mode":     "Mode of `field` per cell (most-common value).",
                    "sorting":  "Legacy alias for folk_ward_sorting — operates on the raw "
                                "field values (not φ). Use folk_ward_sorting for new projects; "
                                "it converts to the φ scale first and matches the published "
                                "Folk & Ward (1957) formula exactly.",
                    "distribution": "Per-cell empirical CDF written as a 99-band raster (q01..q99 of `field`).",
                    "d_percentiles": "Five-band raster of D5, D16, D50, D84, D95 in one pass.",
                    "folk_ward_sorting":  "σ_φ on the phi scale (φ=-log2(D/1mm)). "
                                          "Standard Folk-Ward sorting in φ.",
                    "folk_ward_skewness": "Sk_φ on the phi scale. Folk-Ward skewness.",
                    "folk_ward_kurtosis": "K_G on the phi scale. Folk-Ward kurtosis.",
                    "packing_index": "Σ(Surface_area) / cell_area, clamped to [0, 1]. "
                                     "Relative visible-clast coverage; counts only detected, "
                                     "fully-visible clasts and so under-represents true cover. "
                                     "When size bins are configured below the output becomes "
                                     "multi-band, one band per bin.",
                    "packing_clustering":
                                  "Shannon evenness of the size-bin packing fractions in each "
                                  "cell: H' = -Σ p_i·ln(p_i) / ln(n_bins). 1 = clasts evenly "
                                  "spread across all bins (well-mixed sizes), 0 = one bin "
                                  "dominates (size-clustered). Requires bin edges.",
                }
                for pname in parameter_options:
                    cb = ui.checkbox(pname)
                    cb.bind_value(state.ras_parameters_checked, pname)
                    cb.on_value_change(lambda _: _refresh_selection_summary())
                    desc = _PARAM_DESC.get(pname)
                    if desc:
                        cb.tooltip(desc)

        with ui.column().classes("min-w-[18rem]"):
            ui.label("Per-job parameters (baked in at Add time)").classes("text-base font-bold")
            # Visibility follows dict state, which bind_visibility_from
            # cannot express; _refresh_selection_summary toggles it.
            percentile_row = ui.row().classes("items-center gap-2")
            with percentile_row:
                ui.label("Percentile (for 'quantile'):")
                ui.slider(min=0.0, max=1.0, step=0.05) \
.bind_value(state, "ras_percentile") \
.classes("w-40") \
.tooltip("Each ‘quantile’ job added to the queue records this value at add-time. "
                             "Add ‘Clast_length × quantile’ three times with percentile 0.05, 0.50, 0.95 "
                             "to get D5, D50, D95 maps in one run.")
                ui.label().bind_text_from(state, "ras_percentile", lambda v: f"{v:.2f}")
            T_input = ui.number(label="Wave period T (s)", min=1.0, max=30.0, step=0.5) \
.bind_value(state, "ras_T") \
.classes("w-64") \
.tooltip("Wave period for the Le Roux formula. "
                         "Used only when 'Leroux_wave_orbital_velocity' is among the checked fields.")
            rho_water_input = ui.number(
                label="Water density (kg/m³)",
                value=1025.0, min=900, max=1100, step=1.0,
            ) \
.bind_value(state, "ras_rho_water") \
.classes("w-64") \
.tooltip(
    "Water density used for sediment-transport threshold fields "
    "(Van Rijn, Soulsby/Shields, Hjulström-erosion, Le Roux). "
    "Default 1025.0 = seawater at ~10 °C. Use 1000.0 for fresh water. "
    "Hidden when no transport-threshold field is selected."
)
            # Size bins for packing_index / packing_clustering.
            bin_row = ui.column().classes("gap-1")
            with bin_row:
                ui.label("Size bins (packing_index / packing_clustering)") \
.classes("text-sm font-bold")
                with ui.row().classes("items-end gap-3"):
                    ui.toggle({"fixed": "Fixed (mm)",
                               "percentile": "Percentile of CSV"}) \
.bind_value(state, "ras_bin_mode") \
.tooltip("How to interpret the edges below. "
                                 "'Fixed' = absolute millimetre values. "
                                 "'Percentile' = quantiles of the bin field (e.g. "
                                 "0.25, 0.5, 0.75 → D25/D50/D75).")
                    ui.input(label="Bin edges (comma-separated)") \
.bind_value(state, "ras_bin_edges_text") \
.classes("w-64") \
.tooltip("Fixed mode: edges in mm (e.g. 16, 64, 256 → "
                                 "<16, 16-64, 64-256, 256+ mm bins). "
                                 "Percentile mode: quantiles in [0, 1] (e.g. "
                                 "0.25, 0.5, 0.75). Leave empty in fixed mode "
                                 "to fall back to single-band packing_index.")
                    ui.select(["Clast_length", "Clast_width",
                               "Ellipse_major_axis", "Ellipse_minor_axis",
                               "Equivalent_diameter"],
                              label="Bin field") \
.bind_value(state, "ras_bin_field") \
.classes("w-48") \
.tooltip("Which clast dimension defines the bin "
                                 "assignment. Surface area is always what's "
                                 "summed per bin.")
            _per_job_widgets["percentile_row"] = percentile_row
            _per_job_widgets["T_input"] = T_input
            _per_job_widgets["rho_water_input"] = rho_water_input
            _per_job_widgets["bin_row"] = bin_row

    # Per-statistic description, equation, references and reading notes,
    # rendered in info_container for every checked parameter.
    _PARAM_WIKI = {
        "quantile": {
            "summary": "Sample quantile of the chosen field within each cell. "
                       "Setting the percentile to 0.5, 0.84, or 0.95 yields the "
                       "standard D₅₀, D₈₄, and D₉₅ percentiles used in "
                       "sedimentology; staging several quantile jobs with "
                       "different percentile values produces a percentile family "
                       "in a single batch.",
            "equation": "value(cell) = quantile_q(field for clasts inside cell)",
            "refs": "Folk (1974), *Petrology of Sedimentary Rocks*; standard "
                    "sample quantile (Hyndman & Fan 1996).",
            "notes": "One raster band per job, one job per percentile. The "
                     "percentile value is captured at Add time and survives "
                     "later UI changes.",
        },
        "density": {
            "summary": "Clast count per unit cell area. Field-agnostic — a "
                       "single density map is produced regardless of which "
                       "fields are checked at queue time.",
            "equation": "value(cell) = N_clasts_in_cell / cell_area  (clasts / m²)",
            "refs": "—",
            "notes": "Useful for assessing whether differences in other "
                     "statistics reflect compositional change or simply "
                     "thinning of detected clasts.",
        },
        "average": {
            "summary": "Arithmetic mean of the chosen field within each cell. "
                       "Non-robust under heavy-tailed grain-size distributions: "
                       "for size fields the median (`quantile` at 0.5) or the "
                       "Folk–Ward graphic mean is generally preferable.",
            "equation": "μ(cell) = (1/N) · Σ field_i",
            "refs": "Folk & Ward (1957), *J. Sedimentary Petrology* 27.",
            "notes": "Paired with `std` to give a Gaussian first/second-moment "
                     "summary of the cell.",
        },
        "std": {
            "summary": "Sample (Bessel-corrected) standard deviation of the "
                       "chosen field within each cell. Reported in the field's "
                       "own units and complements the per-cell mean; for "
                       "size-field dispersion in φ units use "
                       "`folk_ward_sorting` instead.",
            "equation": "σ(cell) = sqrt( (1/(N − 1)) · Σ (field_i − μ)² )",
            "refs": "—",
            "notes": "Highlights locally heterogeneous patches but is sensitive "
                     "to a handful of large clasts when N is small.",
        },
        "cv": {
            "summary": "Coefficient of variation of the chosen field within "
                       "each cell — the per-cell standard deviation divided by "
                       "the per-cell mean. Dimensionless, so it compares "
                       "size variability across cells and fields on the same "
                       "footing. Not defined for circular Orientation.",
            "equation": "CV(cell) = σ(cell) / μ(cell)",
            "refs": "—",
            "notes": "NaN where the cell mean is zero. Higher CV = more "
                     "internally variable (poorly sorted) cell; complements "
                     "`folk_ward_sorting`, which expresses the same idea on "
                     "the φ scale.",
        },
        "skewness": {
            "summary": "Standardised third central moment of the chosen field "
                       "within each cell. Reports the asymmetry of the local "
                       "distribution about its mean.",
            "equation": "g₁(cell) = μ₃ / μ₂^(3/2)",
            "refs": "Joanes & Gill (1998), *J. Royal Stat. Soc. D* 47.",
            "notes": "Numerically unstable when N < ~10. For grain-size work "
                     "use `folk_ward_skewness`, which is defined on the φ scale "
                     "and bounded to [−1, 1].",
        },
        "kurtosis": {
            "summary": "Fisher (excess) kurtosis of the chosen field within "
                       "each cell — fourth standardised moment minus 3, so 0 "
                       "corresponds to a Gaussian tail.",
            "equation": "g₂(cell) = μ₄ / μ₂² − 3",
            "refs": "Joanes & Gill (1998).",
            "notes": "Same small-N caveat as `skewness`. The φ-scale "
                     "Folk–Ward analogue (`folk_ward_kurtosis`) is the "
                     "convention in grain-size studies.",
        },
        "mode": {
            "summary": "Most-frequent value of the chosen field within each "
                       "cell. On continuous variables the mode is essentially "
                       "the most-common rounded value and is only meaningful "
                       "once the field has been pre-binned.",
            "equation": "value(cell) = argmax_v P̂(field = v in cell)",
            "refs": "—",
            "notes": "Prefer `quantile` (0.5) for a robust central-tendency "
                     "statistic on size fields.",
        },
        "sorting": {
            "summary": "Folk–Ward inclusive graphic standard deviation computed "
                       "directly on the chosen field — i.e. the textbook "
                       "Folk–Ward σ formula applied without the φ transform. "
                       "Reported in the field's native units.",
            "equation": "σ = (q₈₄ − q₁₆)/4 + (q₉₅ − q₅)/6.6",
            "refs": "Folk & Ward (1957).",
            "notes": "For grain-size sorting use `folk_ward_sorting`, which "
                     "applies the φ transform first and matches the classical "
                     "sedimentological convention.",
        },
        "distribution": {
            "summary": "Per-cell empirical CDF stored as a 99-band raster "
                       "(quantiles q₀₁ … q₉₉ of the chosen field). Allows "
                       "reconstruction of arbitrary downstream percentile "
                       "statistics without re-running the rasterisation.",
            "equation": "band k = quantile_{k/100}(field in cell),  k = 1 … 99",
            "refs": "—",
            "notes": "Storage cost is ~99× a single-band map. If only the "
                     "five conventional grain-size percentiles are required, "
                     "use `d_percentiles` instead.",
        },
        "d_percentiles": {
            "summary": "Five-band raster reporting the canonical "
                       "sedimentological percentiles D₅, D₁₆, D₅₀, D₈₄, D₉₅ in "
                       "a single pass. Band names are written into the "
                       "GeoTIFF metadata.",
            "equation": "band_i = quantile_{p_i}(field in cell), "
                        "p_i ∈ {0.05, 0.16, 0.50, 0.84, 0.95}",
            "refs": "Soulsby (1997), *Dynamics of Marine Sands*; "
                    "Folk & Ward (1957).",
            "notes": "Equivalent to running five `quantile` jobs but stored in "
                     "one file, which keeps related percentiles co-located.",
        },
        "folk_ward_sorting": {
            "summary": "Folk–Ward inclusive graphic standard deviation on the "
                       "Krumbein φ scale (φ = −log₂(D / 1 mm)). σ_φ is the "
                       "conventional grain-size sorting parameter used in "
                       "sedimentological work since the 1950s.",
            "equation": "σ_φ = (φ₈₄ − φ₁₆)/4 + (φ₉₅ − φ₅)/6.6",
            "refs": "Folk & Ward (1957), *J. Sedimentary Petrology* 27.",
            "notes": "Verbal sorting classes (Folk 1974): σ_φ < 0.35 very "
                     "well sorted, 0.35–0.50 well, 0.50–0.70 moderately well, "
                     "0.70–1.00 moderately, 1.00–2.00 poorly, >2.00 very "
                     "poorly sorted.",
        },
        "folk_ward_skewness": {
            "summary": "Folk–Ward inclusive graphic skewness in φ — bounded "
                       "asymmetry parameter of the local grain-size "
                       "distribution. Positive values indicate a fines-biased "
                       "tail, negative values a coarse-biased tail.",
            "equation": "Sk_φ = (φ₁₆ + φ₈₄ − 2 φ₅₀) / (2 (φ₈₄ − φ₁₆)) + "
                        "(φ₅ + φ₉₅ − 2 φ₅₀) / (2 (φ₉₅ − φ₅))",
            "refs": "Folk & Ward (1957).",
            "notes": "Domain is bounded to [−1, 1] by construction. "
                     "Verbal classes (Folk 1974): |Sk_φ| < 0.10 near-"
                     "symmetrical, 0.10–0.30 fine/coarse skewed, "
                     ">0.30 strongly skewed.",
        },
        "folk_ward_kurtosis": {
            "summary": "Folk–Ward inclusive graphic kurtosis on the φ scale — "
                       "ratio of the tail spread (q₅–q₉₅) to the central "
                       "spread (q₂₅–q₇₅). Diagnostic of how peaked the "
                       "central portion of the distribution is relative to "
                       "its tails.",
            "equation": "K_G = (φ₉₅ − φ₅) / (2.44 (φ₇₅ − φ₂₅))",
            "refs": "Folk & Ward (1957).",
            "notes": "Verbal classes (Folk 1974): K_G < 0.67 very platykurtic, "
                     "0.67–0.90 platykurtic, 0.90–1.11 mesokurtic, "
                     "1.11–1.50 leptokurtic, >1.50 very leptokurtic.",
        },
        "packing_index": {
            "summary": "Relative areal coverage of detected, fully-visible "
                       "clasts within each cell. The estimator is biased "
                       "downward by Mask R-CNN's incomplete recall and its "
                       "preference for fully-exposed grains, so absolute "
                       "values systematically under-represent true clast "
                       "cover. Interpret the field as a *relative* index "
                       "comparing cells within a single survey, not as an "
                       "absolute cover fraction.",
            "equation": "value(cell) = clip(Σ A_i / A_cell, 0, 1)\n"
                        "With size bins: one band per bin, summing only "
                        "clasts in that bin.",
            "refs": "Bespoke metric of this package; conceptually related to "
                    "the packing density of Allen (1985), *Principles of "
                    "Physical Sedimentology*.",
            "notes": "Output is multi-band (one band per bin) when bin edges "
                     "are configured under Per-job parameters → Size bins, "
                     "otherwise single-band summing all clasts.",
        },
        "packing_clustering": {
            "summary": "Pielou evenness (J′) of the size-bin packing fractions "
                       "within each cell — a Shannon-entropy-based index of "
                       "how mixed the local grain-size distribution is. "
                       "Requires bin edges so the multi-band packing index is "
                       "available as input.",
            "equation": "p_i = packing_in_bin_i / Σ packing_in_cell\n"
                        "J′  = − Σ p_i · ln(p_i) / ln(n_bins)",
            "refs": "Shannon (1948), *Bell System Tech. J.* 27; "
                    "Pielou (1966), *J. Theor. Biol.* 13; "
                    "Magurran (2004), *Measuring Biological Diversity*.",
            "notes": "Bounded to [0, 1]: 1 indicates packing is distributed "
                     "evenly across all size bins (locally well-mixed); 0 "
                     "indicates one bin dominates (size-clustered, locally "
                     "well-sorted). Undefined (NaN) when fewer than two bins "
                     "are populated in the cell.",
        },
    }

    info_container = ui.column().classes("w-full mt-2 mb-1")

    @ui.refreshable
    def _refresh_param_info():
        info_container.clear()
        checked = [p for p in parameter_options
                   if state.ras_parameters_checked.get(p, False)]
        if not checked:
            return
        with info_container:
            with ui.expansion(
                f"About the selected statistic"
                f"{'s' if len(checked) > 1 else ''} ({len(checked)})",
                value=False,
                icon="info",
            ).classes("w-full"):
                for pname in checked:
                    entry = _PARAM_WIKI.get(pname)
                    if not entry:
                        continue
                    with ui.card().classes("w-full mb-1") \
.style("background:#f7faff; border:1px solid #d6e4ff;"):
                        ui.label(pname).classes("text-base font-bold")
                        ui.markdown(entry["summary"])
                        with ui.expansion("Equation").classes("w-full"):
                            ui.html(f"<pre style='white-space:pre-wrap; margin:0;'>"
                                    f"{entry['equation']}</pre>")
                        with ui.expansion("References").classes("w-full"):
                            ui.markdown(entry["refs"])
                        with ui.expansion("Notes / interpretation").classes("w-full"):
                            ui.markdown(entry["notes"])

    # A 0.5 s timer rebuilds the info panel only when the checked set changed.
    _last_param_sig = {"v": None}
    def _maybe_refresh_param_info():
        sig = tuple(sorted(p for p in parameter_options
                           if state.ras_parameters_checked.get(p, False)))
        if sig != _last_param_sig["v"]:
            _last_param_sig["v"] = sig
            _refresh_param_info.refresh()
    ui.timer(0.5, _maybe_refresh_param_info)
    _refresh_selection_summary()
    _refresh_param_info()

    ui.separator()

    job_list_container = ui.column().classes("w-full") \
.style("max-height: 280px; overflow-y: auto;")

    # Output controls are baked in at Add-to-queue time, so they sit above
    # the Add button.
    with ui.row().classes("w-full gap-4 items-end"):
        ui.number(label="Cell size (in ortho units)", min=0.05, max=100.0, step=0.5) \
.bind_value(state, "ras_cellsize") \
.classes("w-32") \
.tooltip("Output raster cell size, applied to every job in the queue.")
        ui.number(label="Min cell density (clasts/cell)",
                  min=0, max=1000, step=1, format="%d") \
.bind_value(state, "ras_min_cell_density") \
.classes("w-48") \
.tooltip("Cells containing fewer than this many clasts are "
                     "set to NaN in the output raster. Useful for filtering "
                     "out spurious low-density cells that do not carry "
                     "statistically meaningful aggregates. Set to 0 to "
                     "keep every non-empty cell (legacy behaviour).")
        ui.input(
            label="Output directory",
        ).bind_value(state, "ras_output_dir") \
.classes("flex-grow") \
.tooltip("All output GeoTIFFs are written here. "
                     "Defaults to results/rasters/ under the active project.")

    with ui.row().classes("items-center gap-2 mt-2") as _ras_action_row:
        add_btn = ui.button("Add to queue", icon="add") \
.props("color=primary") \
.tooltip("Snapshot the checked fields × statistics into the queue. "
                     "The form stays editable to stage further batches.")
        run_btn = ui.button("Run all queued", icon="playlist_play") \
.props("color=primary outline")
        ui.button("Reset all", icon="restart_alt",
                  on_click=lambda: _reset_ras_queue()) \
.props("flat color=primary") \
.tooltip("Mark every done / errored job pending again "
                     "so 'Run all queued' will re-run them.")
        ui.button("Clear queue", icon="delete_sweep",
                  on_click=lambda: _clear_ras_queue()) \
.props("flat color=grey") \
.tooltip("Empty the queue. Already-written rasters stay on disk.")
        job_count_label = ui.label("").classes("text-sm text-grey-7 ml-2")

    def _refresh_job_count():
        n = len(state.ras_jobs)
        job_count_label.set_text(
            "" if n == 0 else
            f"{n} job{'s' if n != 1 else ''} queued"
        )

    def _ras_primary(job, idx):
        if job['parameter'] == 'density':
            desc = "density (field-agnostic)"
        else:
            desc = f"{job['field']} × {job['parameter']}"
            if job['parameter'] == 'quantile':
                desc += f" (p={job['percentile']:.2f})"
            if job['field'] == 'Leroux_wave_orbital_velocity':
                desc += f"  T={job['T']}s"
        return desc

    def _ras_extra(job, idx):
        # Source-image chip: which CSV/ortho the job was queued for.
        src_stem = ""
        if job.get("csv_stem"):
            src_stem = job["csv_stem"]
        elif job.get("tif"):
            src_stem = Path(job["tif"]).stem
        if src_stem:
            short = (src_stem if len(src_stem) <= 32
                     else "…" + src_stem[-31:])
            ui.label(short) \
.classes("text-xs text-grey-6 font-mono") \
.tooltip(f"Source: {src_stem}")

    def _ras_delete_btn(job, idx):
        def _remove(_e=None, _idx=idx):
            state.ras_jobs.pop(_idx)
            _render_ras_queue()
        ui.button(icon="delete", on_click=_remove) \
.props("flat dense color=negative").tooltip("Remove this job")

    def _render_ras_queue():
        """Re-populate the queue container from state.ras_jobs."""
        render_queue(
            job_list_container, state.ras_jobs,
            title="Queued jobs",
            count_label=None,  # bespoke pluralized count below
            empty_text="The queue is empty. Configure a selection above and click Add to queue.",
            primary_text=_ras_primary,
            render_extra=_ras_extra,
            render_delete=_ras_delete_btn,
        )
        _refresh_job_count()

    _render_ras_queue()

    def _add_jobs():
        fields = [f for f, v in state.ras_fields_checked.items() if v]
        params = [p for p, v in state.ras_parameters_checked.items() if v]
        if not fields or not params:
            ui.notify("Tick at least one field and one statistic (Selection, above), then Add to queue.",
                      type="warning")
            return
        # CSV / TIF / output dir / cellsize are snapshotted into each job, so
        # jobs for different source images can share the queue.
        if not (state.ras_csv and state.ras_tif):
            ui.notify(
                "Set *Clast list CSV* and *Source ortho-image* (Inputs, above) first — those paths "
                "are stored in each queued job so jobs for different "
                "source images do not collide when run.",
                type="warning", multi_line=True, timeout=8000)
            return
        snap_csv = state.ras_csv
        snap_tif = state.ras_tif
        snap_out_dir = (state.ras_output_dir
                        or default_starting_dir("rasters"))
        snap_cellsize = float(state.ras_cellsize)
        snap_csv_stem = Path(snap_csv).stem
        snap_min_density = max(0, int(state.ras_min_cell_density or 0))
        # Round away slider float noise; the dedup check is dict equality.
        pct_rounded = round(float(state.ras_percentile), 2)
        T_rounded = round(float(state.ras_T), 2)
        rho_water_snap = round(float(state.ras_rho_water or 1025.0), 1)
        added = 0
        skipped = 0

        # Field-independent indices produce one raster per Add click.
        params_independent = [p for p in params
                              if p in _PARAMS_FIELD_INDEPENDENT]
        params_field_dependent = [p for p in params
                                  if p not in _PARAMS_FIELD_INDEPENDENT]

        # Empty / invalid bin text means "no bins": packing_index runs
        # single-band, packing_clustering reports an error at run time.
        def _parse_bins():
            txt = (state.ras_bin_edges_text or "").strip()
            if not txt:
                return None
            try:
                vals = [float(p.strip()) for p in txt.split(",") if p.strip()]
            except ValueError:
                return None
            if not vals:
                return None
            # Fixed edges are in mm; the backend wants metres. Percentile
            # edges pass through.
            if state.ras_bin_mode == "fixed":
                return [v / 1000.0 for v in vals]
            return vals

        bin_edges_for_jobs = _parse_bins()
        bin_mode_for_jobs = state.ras_bin_mode
        bin_field_for_jobs = state.ras_bin_field

        def _baked_bin_info(parameter_name):
            """Bake the current bin config into the job ONLY if it actually
            applies. Keeps the dedup-by-equality check stable for non-packing
            jobs (which never read bin config)."""
            if parameter_name in ("packing_index", "packing_clustering"):
                return {
                    "bin_edges": (tuple(bin_edges_for_jobs)
                                  if bin_edges_for_jobs is not None else None),
                    "bin_mode": bin_mode_for_jobs,
                    "bin_field": bin_field_for_jobs,
                }
            return {}

        per_image = {
            "csv":             snap_csv,
            "tif":             snap_tif,
            "out_dir":         snap_out_dir,
            "cellsize":        snap_cellsize,
            "csv_stem":        snap_csv_stem,
            "min_cell_density": snap_min_density,
        }
        incoherent = 0
        incoherent_reasons = set()
        for f in fields:
            for p in params_field_dependent:
                if p in _PARAMS_SIZE_ONLY and not _is_size_field(f):
                    incoherent += 1
                    incoherent_reasons.add(
                        f"{p} needs a grain-size field (not {f})")
                    continue
                if (p in _PARAMS_INVALID_ON_ORIENTATION
                        and _is_orientation_field(f)):
                    incoherent += 1
                    incoherent_reasons.add(
                        f"{p} is undefined for circular Orientation")
                    continue
                new_job = {
                    "field": f,
                    "parameter": p,
                    "percentile": pct_rounded,
                    "T": T_rounded,
                    "rho_water": rho_water_snap,
                    **_baked_bin_info(p),
                    **per_image,
                }
                if new_job in state.ras_jobs:
                    skipped += 1
                    continue
                state.ras_jobs.append(new_job)
                added += 1

        # Shannon evenness across size bins needs bins: without them the
        # job would sit in the queue and fail at run time.
        if ("packing_clustering" in params_independent
                and not _parse_bins()):
            params_independent = [p for p in params_independent
                                  if p != "packing_clustering"]
            ui.notify("packing_clustering needs size bins: type them in "
                      "*Bin edges* (Selection, above) and add the job again.",
                      type="warning", multi_line=True, timeout=9000)
        for p in params_independent:
            ind_job = {
                "field": "(any)",
                "parameter": p,
                "percentile": pct_rounded,
                "T": T_rounded,
                "rho_water": rho_water_snap,
                **_baked_bin_info(p),
                **per_image,
            }
            if ind_job in state.ras_jobs:
                skipped += 1
            else:
                state.ras_jobs.append(ind_job)
                added += 1

        msg = f"Added {added} job{'s' if added != 1 else ''} to the queue."
        if skipped:
            msg += f" ({skipped} duplicate{'s' if skipped != 1 else ''} skipped.)"
        if incoherent:
            msg += (f" ({incoherent} incoherent combination"
                    f"{'s' if incoherent != 1 else ''} skipped: "
                    + "; ".join(sorted(incoherent_reasons)) + ".)")
        ui.notify(msg, type=("warning" if incoherent and not added
                             else "positive"), multi_line=bool(incoherent),
                  timeout=(9000 if incoherent else 4000))
        _render_ras_queue()

    add_btn.on_click(_add_jobs)

    ui.separator()

    def _reset_ras_queue():
        n = 0
        for j in state.ras_jobs:
            if j.get("status") in ("done", "error"):
                j["status"] = "pending"
                j["error"] = None
                n += 1
        _render_ras_queue()
        ui.notify(
            f"Reset {n} job(s) to pending.",
            type="info" if n else "warning")

    def _clear_ras_queue():
        state.ras_jobs.clear()
        _render_ras_queue()
        ui.notify("Rasterize queue cleared.", type="info")

    progress_bar = ui.linear_progress(value=0.0, show_value=False).classes("w-full mt-2")
    progress_label = ui.label("").classes("text-sm text-grey-7")
    progress_bar.set_visibility(False)
    progress_label.set_visibility(False)

    plot_container = ui.column().classes("w-full")
    log_widget = live_log("merge", build_log_console(max_lines=2000, height="h-40"))

    def _do_run_queue():
        if not state.ras_jobs:
            ui.notify("The queue is empty — tick fields × statistics (Selection, above), then *Add to queue*.",
                      type="warning")
            return

        # Snapshot the queue so edits mid-run do not surprise the worker.
        jobs = list(state.ras_jobs)
        run_btn.props("loading")
        plot_container.clear()
        log_widget.clear()
        progress_bar.value = 0.0
        progress_bar.set_visibility(True)
        progress_label.set_visibility(True)
        log_widget.push(f"Running {len(jobs)} queued job(s)…")
        _page = context.client

        def _mark(job, status, error=None):
            # The rows stayed "pending" after a run:
            # each job now takes done or error, and the queue is redrawn.
            job["status"] = status
            job["error"] = error
            try:
                with on_page(_page):
                    _render_ras_queue()
            except Exception:
                pass

        def _worker():
            try:
                with capture_stdout_to_log(log_widget):
                    for i, job in enumerate(jobs):
                        field = job["field"]
                        parameter = job["parameter"]
                        percentile = job["percentile"]
                        T_value = job["T"]
                        # Per-job paths; state is the fallback for a job
                        # dict without them.
                        job_csv = job.get("csv") or state.ras_csv
                        job_tif = job.get("tif") or state.ras_tif
                        out_dir = (job.get("out_dir")
                                   or state.ras_output_dir
                                   or default_starting_dir("rasters"))
                        os.makedirs(out_dir, exist_ok=True)
                        cellsize = job.get("cellsize") or state.ras_cellsize
                        csv_stem = (job.get("csv_stem")
                                    or (Path(job_csv).stem
                                        if job_csv else "output"))
                        min_cell_density = int(
                            job.get("min_cell_density",
                                    state.ras_min_cell_density or 0))

                        is_density = (parameter == "density")
                        if is_density:
                            display_label = "density (field-agnostic)"
                            paramname = "density"
                            # The field arg must name a real column; density
                            # does not read it.
                            effective_field = "Clast_length"
                        else:
                            display_label = f"{field} × {parameter}"
                            if parameter == "quantile":
                                paramname = f"D{int(percentile * 100)}"
                            else:
                                paramname = parameter
                            effective_field = field

                        progress_label.set_text(
                            f"[{i+1}/{len(jobs)}]  {display_label}")
                        progress_bar.value = i / len(jobs)
                        log_widget.push(
                            f"--- Job #{i+1}/{len(jobs)}: {display_label}" +
                            (f" (p={percentile:.2f})" if parameter == "quantile" else "") +
                            " ---"
                        )
                        bin_edges = job.get("bin_edges")
                        if bin_edges is not None:
                            bin_edges = list(bin_edges)
                        bin_mode = job.get("bin_mode", "fixed")
                        bin_field = job.get("bin_field", "Clast_length")
                        # The bins and the minimum density are in the name:
                        # they change the raster, and two runs that differ
                        # only by them must not share a file.
                        out_path = os.path.join(out_dir, naming.raster_name(
                            csv_stem,
                            "density" if is_density else field,
                            "" if is_density else paramname,
                            cellsize, bin_edges=bin_edges, bin_mode=bin_mode,
                            min_density=min_cell_density))
                        # Rasterization runs in a worker subprocess; the
                        # thumbnail comes back as a PNG file.
                        import tempfile as _tf
                        import uuid as _uuid_mod
                        thumb_path = os.path.join(
                            _tf.gettempdir(),
                            f"pm_thumb_{_uuid_mod.uuid4().hex[:10]}.png")
                        outcome = _worker_mod.run_job(
                            "rasterize",
                            {"ClastImageFilePath": job_tif,
                             "ClastSizeListCSVFilePath": job_csv,
                             "RasterFileWritingPath": out_path,
                             "field": effective_field,
                             "parameter": parameter,
                             "cellsize": cellsize,
                             "percentile": percentile,
                             "plot": False,
                             "figuresize": [10, 8],
                             "T": T_value,
                             "bin_edges": bin_edges,
                             "bin_mode": bin_mode,
                             "bin_field": bin_field,
                             "min_cell_density": min_cell_density,
                             "rho_water": float(job.get("rho_water", 1025.0)),
                             "thumbnail_png": thumb_path},
                            log_cb=log_widget.push)
                        if not outcome.ok:
                            log_widget.push(f"   ⚠️  Failed: "
                                            f"{_job_error_text(outcome)}")
                            _mark(job, "error", _job_error_text(outcome))
                            continue
                        _mark(job, "done")
                        with plot_container:
                            with ui.column().classes("inline-block m-2"):
                                ui.label(out_path).classes("text-xs text-grey-6")
                                if os.path.exists(thumb_path):
                                    ui.image(thumb_path) \
                                        .style("max-width: 480px;")
                        progress_bar.value = (i + 1) / len(jobs)
                log_widget.push(f"✅  Done. {len(jobs)} job(s) processed.")
            except Exception as e:
                import traceback
                log_widget.push(f"❌  {type(e).__name__}: {e}")
                log_widget.push(traceback.format_exc())
            finally:
                run_btn.props(remove="loading")
                progress_bar.set_visibility(False)
                progress_label.set_visibility(False)

        threading.Thread(target=_worker, daemon=True).start()

    run_btn.on_click(_do_run_queue)

    # The queue renders below the action bar; the selection summary follows
    # the cards it counts.
    _ras_root = _ras_action_row.parent_slot.parent
    job_list_container.classes("pm-ras-queue")
    job_list_container.move(
        _ras_root,
        target_index=_ras_root.default_slot.children.index(_ras_action_row) + 1)
    selection_summary.classes("pm-ras-summary")
    selection_summary.move(
        _ras_root,
        target_index=_ras_root.default_slot.children.index(_ras_cards_row) + 1)


def build_merge_tab():
    def _on_proj_change():
        for entry in state.merge_csvs_list:
            entry["path"] = ""
        state.merge_out = ""
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Merge")
    ui.label("One CSV from detection runs at several window sizes; on conflicts the larger window's rows win.").classes("text-sm text-grey-7")

    def _auto_window_size(path: str):
        """Auto-detect window size from filename. Returns float or None."""
        if not path:
            return None
        return naming.parse_window_size(Path(path).name)

    csv_rows_container = ui.column().classes("w-full")

    @ui.refreshable
    def _build_csv_rows():
        for i, entry in enumerate(state.merge_csvs_list):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(f"#{i+1}").classes("w-8 text-grey-7")
                path_inp = ui.input(label=f"CSV path",
                                    value=entry["path"]).classes("flex-grow") \
.tooltip("CSV file produced by detection. The window size is auto-detected from the filename.")
                def _on_path_change(e, idx=i, ip=None):
                    state.merge_csvs_list[idx]["path"] = e.value
                    # The window follows the path: its _ws token, else
                    # empty, to be typed, never the previous CSV's window.
                    state.merge_csvs_list[idx]["window_size"] = _auto_window_size(e.value)
                    _build_csv_rows.refresh()  # rebuild so the auto-detected size shows
                path_inp.on_value_change(lambda e, idx=i: _on_path_change(e, idx))

                def _browse(idx=i):
                    current = state.merge_csvs_list[idx]["path"]
                    initialdir = (str(Path(current).parent) if current
                                  else default_starting_dir("vectors"))
                    picked = native_file_picker(
                        f"Pick CSV #{idx+1}",
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                        initialdir=initialdir)
                    if picked:
                        state.merge_csvs_list[idx]["path"] = picked
                        state.merge_csvs_list[idx]["window_size"] = _auto_window_size(picked)
                        _build_csv_rows.refresh()
                ui.button("Browse…", icon="folder_open", on_click=_browse).props("outline dense")

                ws_inp = ui.number(label="Window (m)", min=0.1, max=100.0, step=0.1,
                                    format="%.2f",
                                    value=entry["window_size"]) \
.classes("w-32") \
.tooltip("Window size in metres used to produce this CSV. "
                             "Auto-detected from the filename if it contains "
                             "a '_ws<size>m' token. Larger values get priority on "
                             "merge conflicts.")
                def _on_ws_change(e, idx=i):
                    state.merge_csvs_list[idx]["window_size"] = e.value
                ws_inp.on_value_change(lambda e, idx=i: _on_ws_change(e, idx))

                def _remove(idx=i):
                    if len(state.merge_csvs_list) <= 2:
                        ui.notify("At least two CSVs are needed — add a row (Inputs, above) before removing this one.", type="warning")
                        return
                    state.merge_csvs_list.pop(idx)
                    _build_csv_rows.refresh()
                ui.button(icon="delete", on_click=_remove).props("flat dense color=negative") \
.tooltip("Remove this CSV from the merge")

    with csv_rows_container:
        _build_csv_rows()

    def _add_csv():
        state.merge_csvs_list.append({"path": "", "window_size": None})
        _build_csv_rows.refresh()

    def _fill_from_stem(stem: str):
        # Assemble the merge from the project's per-window CSVs for one
        # image stem; windows parse from the filenames.
        from functions import project_defaults as _pdf
        rows = _pdf.window_csvs_by_stem(state.current_project).get(stem, [])
        if len(rows) < 2:
            ui.notify(f"{stem}: fewer than two window CSVs — nothing to merge. Pick another image (Inputs, above).",
                      type="warning")
            return
        state.merge_csvs_list = [{"path": p, "window_size": w}
                                 for p, w in rows]      # already window-desc
        out_dir = (project_path(state.current_project, "vectors")
                   if state.current_project else Path(rows[0][0]).parent)
        # The merge keeps the origin its inputs were written under.
        state.merge_out = str(Path(out_dir) / naming.merged_csv_name(
            naming.run_stem(Path(rows[0][0]).name)))
        _merge_out_last_auto["value"] = state.merge_out
        _build_csv_rows.refresh()
        ui.notify(f"Filled {len(rows)} window CSV(s) for '{stem}' "
                  f"({', '.join(f'{w:g}m' for _, w in rows)}). "
                  "Review and Run — or edit any row.", type="positive",
                  multi_line=True)

    with ui.row().classes("items-center gap-2"):
        ui.button("Add another CSV", icon="add", on_click=_add_csv) \
            .props("outline") \
            .tooltip("Add another CSV to the merge. Three or more CSVs are "
                     "merged iteratively, largest-window first.")

        def _open_from_project():
            from functions import project_defaults as _pdf
            stems = _pdf.mergeable_stems(state.current_project)
            if not stems:
                ui.notify("No image in this project has two or more "
                          "per-window detection CSVs yet — run the Detect tab at "
                          "more than one window size first.", type="warning",
                          multi_line=True)
                return
            if len(stems) == 1:
                _fill_from_stem(stems[0])
                return
            with ui.menu() as _m:
                for s in stems:
                    _rows = _pdf.window_csvs_by_stem(
                        state.current_project).get(s, [])
                    ui.menu_item(
                        f"{s}  ({len(_rows)} windows: "
                        f"{', '.join(f'{w:g}m' for _, w in _rows)})",
                        on_click=lambda s=s: (_fill_from_stem(s), _m.close()))
            _m.open()
        ui.button("From project…", icon="auto_awesome_motion",
                  on_click=_open_from_project).props("outline color=primary") \
            .tooltip("Fill the rows from this project's per-window "
                     "detection CSVs, grouped by image — one click instead of "
                     "browsing each file and typing each window size.")

    ui.separator()

    # Default output from the largest-window CSV, refreshed while the field
    # still holds the last auto-filled value (i.e. not hand-edited).
    _merge_out_last_auto = {"value": ""}

    def update_default_out():
        sized = [e for e in state.merge_csvs_list
                 if e["path"] and e["window_size"] is not None]
        if not sized:
            return
        largest = max(sized, key=lambda e: e["window_size"])
        base = Path(largest["path"]).parent
        img_stem = naming.run_stem(Path(largest["path"]).name)
        candidate = str(base / naming.merged_csv_name(img_stem))
        if (not state.merge_out
                or state.merge_out == _merge_out_last_auto["value"]):
            state.merge_out = candidate
            _merge_out_last_auto["value"] = candidate
    ui.timer(1.0, update_default_out)

    ui.input(label="Output CSV path").classes("w-full") \
.bind_value(state, "merge_out") \
.tooltip("Where the merged CSV is written. Auto-derived from the largest-window CSV name "
                 "when left empty; otherwise the value you type is kept.")

    ui.separator()
    ui.markdown("**Merge parameters**")

    with ui.row().classes("w-full gap-8 items-center"):
        ui.select(["iou", "centroid"], label="Method") \
.bind_value(state, "merge_method") \
.classes("w-48") \
.tooltip("Which dedup method to use when comparing two clasts. "
                     "'iou' (precise) computes intersection-over-union of the ellipse "
                     "polygons. 'centroid' (fast) just compares centroid distances "
                     "scaled by clast size.")
        with ui.row().classes("items-center gap-2"):
            ui.label("Overlap threshold:")
            ui.slider(min=0.0, max=1.0, step=0.05) \
.bind_value(state, "merge_overlap") \
.classes("w-40") \
.tooltip("Above this value, two clasts count as duplicates and the "
                         "lower-priority one is dropped. 0.30 is a sensible default for IoU.")
            ui.label().bind_text_from(state, "merge_overlap", lambda v: f"{v:.2f}")
        ui.number(label="Polygon vertices (IoU)", min=8, max=128, step=4) \
.bind_value(state, "merge_n_points") \
.bind_enabled_from(state, "merge_method", backward=lambda v: v == "iou") \
.classes("w-48") \
.tooltip("Number of vertices used to approximate each ellipse for IoU. "
                     "32 is a good accuracy/speed tradeoff. Used only when method='iou'.")

    ui.separator()

    with ui.row().classes("items-center gap-2"):
        save_job_btn = ui.button("Add to queue", icon="add") \
.props("color=primary") \
.tooltip("Snapshot the current configuration into the queue. "
                     "The form stays editable so you can stage another "
                     "job. Click Run all queued when ready.")
        run_queue_btn = ui.button("Run all queued", icon="playlist_play") \
.props("color=primary outline") \
.tooltip("Execute every queued job in order.")
        def _reset_merge_queue():
            n = 0
            for j in state.merge_jobs:
                if j.get("status") in ("done", "error"):
                    j["status"] = "pending"
                    j["error"] = None
                    n += 1
            _render_queue.refresh()
            ui.notify(f"Reset {n} job(s) to pending.",
                       type="info" if n else "warning")
        ui.button("Reset all", icon="restart_alt",
                  on_click=_reset_merge_queue) \
.props("flat color=primary") \
.tooltip("Mark every done / errored job pending again.")
        def _clear_merge_queue():
            state.merge_jobs.clear()
            _render_queue.refresh()
            ui.notify("Merge queue cleared.", type="info")
        ui.button("Clear queue", icon="delete_sweep",
                  on_click=_clear_merge_queue) \
.props("flat color=grey") \
.tooltip("Empty the queue. Already-written CSVs stay on disk.")
        queue_count_label = ui.label("").classes("text-sm text-grey-7 ml-2")

    queue_container = ui.column().classes("w-full mt-2 mb-2")

    def _merge_primary(job, idx):
        return f"out: {Path(job['out_path']).name}"

    def _merge_params(job, idx):
        n_inputs = len([c for c in job["csvs"] if c["path"]])
        return (f"({n_inputs} inputs, method={job['method']}, "
                f"overlap={job['overlap']:.2f})")

    def _merge_extra(job, idx):
        if job.get("status") == "done" and job.get("n_rows") is not None:
            ui.label(f"-> {job['n_rows']} rows") \
.classes("text-xs text-green-7")

    def _merge_delete(idx):
        state.merge_jobs.pop(idx)
        _render_queue.refresh()

    @ui.refreshable
    def _render_queue():
        render_queue(
            queue_container, state.merge_jobs,
            title="Queued jobs",
            count_label=queue_count_label,
            primary_text=_merge_primary,
            params_text=_merge_params,
            render_extra=_merge_extra,
            on_delete=_merge_delete,
        )

    _render_queue()

    result_container = ui.column().classes("w-full")
    log_widget = live_log("rasterize", build_log_console(height="h-32"))

    def _snapshot_current_config():
        """Build a deep-copied job dict from the current form state."""
        import copy
        return {
            "csvs": copy.deepcopy([dict(e) for e in state.merge_csvs_list
                                    if e.get("path")]),
            "out_path": state.merge_out,
            "method": state.merge_method,
            "overlap": state.merge_overlap,
            "n_points": int(state.merge_n_points),
            "status": "pending",
            "n_rows": None,
            "error": None,
        }

    def _validate_config(csvs, out_path):
        """Return (ready_csvs_sorted_by_window_desc, error_msg) tuple.
        error_msg is None when valid."""
        ready = []
        for i, entry in enumerate(csvs):
            if not entry.get("path"):
                continue
            if entry.get("window_size") is None:
                return None, (f"CSV #{i+1} is missing its window size. "
                              f"Set it manually or rename the file to "
                              f"include a '_ws<size>m' token.")
            ready.append(entry)
        if len(ready) < 2:
            return None, "Fill in at least 2 CSV paths (Inputs, above) to merge."
        if not out_path:
            return None, "Set the output path (Inputs, above)."
        return sorted(ready, key=lambda e: e["window_size"], reverse=True), None

    def _on_save_job():
        snap = _snapshot_current_config()
        ordered, err = _validate_config(snap["csvs"], snap["out_path"])
        if err:
            ui.notify(err, type="warning")
            return
        state.merge_jobs.append(snap)
        _render_queue.refresh()
        ui.notify(
            f"Job #{len(state.merge_jobs)} saved to queue. "
            f"Configure another or click Run all queued.",
            type="positive",
        )
    save_job_btn.on_click(_on_save_job)

    def _execute_merge(ordered, out_path, method, overlap, n_points):
        """Run the iterative pairwise merge in a worker subprocess (op
        ``merge``) and return the merged row count; raises RuntimeError with
        the worker's one-line error on failure."""
        outcome = _worker_mod.run_job(
            "merge",
            {"ordered": [{"path": e["path"],
                          "window_size": e["window_size"]} for e in ordered],
             "out_path": out_path, "method": method,
             "overlap": float(overlap), "n_points": int(n_points)},
            log_cb=log_widget.push)
        if not outcome.ok:
            raise RuntimeError(_job_error_text(outcome))
        n_rows = int(outcome.summary.get("n_rows", 0))
        log_widget.push(f"\n✅  Wrote {out_path} ({n_rows} rows).")
        return n_rows

    def do_merge_queue():
        pending_idx = [i for i, j in enumerate(state.merge_jobs)
                       if j.get("status") in ("pending", "error", "interrupted")]
        if not pending_idx:
            ui.notify("No pending jobs — press *Add to queue* (above) first.",
                      type="info")
            return

        run_queue_btn.props("loading")
        _disable(save_job_btn, "Queue running — add jobs once it has finished")
        result_container.clear()
        log_widget.clear()

        def _worker():
            try:
                with capture_stdout_to_log(log_widget):
                    log_widget.push(
                        f"\n=== Running {len(pending_idx)} queued merge jobs ===\n")
                    for ji_pos, ji in enumerate(pending_idx, start=1):
                        job = state.merge_jobs[ji]
                        log_widget.push(
                            f"\n>>> Job {ji_pos}/{len(pending_idx)} "
                            f"(queue index {ji+1}): "
                            f"{Path(job['out_path']).name}\n")
                        job["status"] = "running"
                        _render_queue.refresh()
                        try:
                            ordered, err = _validate_config(
                                job["csvs"], job["out_path"])
                            if err:
                                raise ValueError(err)
                            n_rows = _execute_merge(
                                ordered, job["out_path"],
                                job["method"], job["overlap"],
                                int(job["n_points"]),
                            )
                            job["status"] = "done"
                            job["n_rows"] = n_rows
                            job["error"] = None
                        except Exception as e:
                            import traceback
                            job["status"] = "error"
                            job["error"] = f"{type(e).__name__}: {e}"
                            log_widget.push(f"❌  {job['error']}")
                            log_widget.push(traceback.format_exc())
                        _render_queue.refresh()
                    log_widget.push(
                        f"\n=== Queue done. {sum(1 for i in pending_idx if state.merge_jobs[i]['status'] == 'done')} succeeded, "
                        f"{sum(1 for i in pending_idx if state.merge_jobs[i]['status'] == 'error')} failed. ==="
                    )
            except Exception as e:
                import traceback
                log_widget.push(f"❌  Queue runner crashed: {type(e).__name__}: {e}")
                log_widget.push(traceback.format_exc())
            finally:
                run_queue_btn.props(remove="loading")
                _enable(save_job_btn)

        threading.Thread(target=_worker, daemon=True).start()

    run_queue_btn.on_click(do_merge_queue)


def build_map_tab():
    def _on_proj_change():
        # The queue is dropped: its jobs point at the old project's files.
        state.map_out = ""   # _default_map_out timer will refill
        state.map_jobs.clear()
        try:
            _seed_map(force=True)
        except NameError:
            pass  # rows not built yet on the very first strip render
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Map")
    ui.label("A publication figure — basemap, ortho, data layer, legend, "
             "grid, north arrow, scale bar — to PNG, PDF or SVG. "
             "Georeferenced data in a projected CRS only.") \
        .classes("text-sm text-grey-7") \
        .tooltip("Quadrat photographs and their CSVs are in pixel "
                 "coordinates with no CRS: view those through the "
                 "ellipse-overlay PNGs of the Detect tab.")

    _map_ortho_row = path_input_with_browse(
        "Source ortho-image (.tif)", "map_ortho", kind="file",
        filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")],
        default_kind="images",
    )

    with ui.row().classes("w-full"):
        ui.toggle(["vector", "raster"]).bind_value(state, "map_layer")

    _map_csv_row = path_input_with_browse(
        "Clast list CSV (vector)", "map_csv", kind="file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        default_kind="vectors",
    )
    _map_csv_row.bind_visibility_from(state, "map_layer", value="vector")

    _map_raster_row = path_input_with_browse(
        "Rasterized GeoTIFF (raster)", "map_raster", kind="file",
        filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")],
        default_kind="rasters",
    )
    _map_raster_row.bind_visibility_from(state, "map_layer", value="raster")

    def _seed_map(force=False):
        # Newest raster, its stem-matched ortho, and (vector mode) the newest
        # clast CSV.
        from functions import project_defaults as _pdf
        proj = state.current_project
        r = (_pdf.rasters(proj) or [None])[0]
        _seed_default(_map_raster_row._path_input, r, force=force)
        csv = _pdf.best_clast_csv(proj)
        _seed_default(_map_csv_row._path_input, csv, force=force)
        ref = r if r is not None else csv
        if ref is not None:
            _seed_default(_map_ortho_row._path_input,
                          _pdf.source_ortho_for(ref.name, proj), force=force)
    _seed_map()

    def _refresh_map_on_open():
        if drop_foreign_paths("map_raster", "map_csv", "map_ortho"):
            _seed_map(force=True)
        else:
            _seed_map()
    register_tab_refresh("Map", _refresh_map_on_open)

    _map_content_sep = ui.separator()
    ui.markdown("**Map content**")

    # Lazy import: keeps contextily off the GUI startup path.
    from functions.map_export import DEFAULT_BASEMAPS, DEFAULT_COLORMAPS

    field_options = [
        "Clast_length", "Clast_width", "Ellipse_major_axis", "Ellipse_minor_axis",
        "Equivalent_diameter", "Perimeter", "Surface_area",
        "Eccentricity", "Solidity", "Mean_intensity", "Score", "Orientation",
    ]
    for _ad in _project_addons():
        if _ad.name not in field_options:
            field_options.append(_ad.name)

    with ui.row().classes("w-full gap-8"):
        with ui.column().classes("min-w-[16rem]"):
            ui.select(list(DEFAULT_BASEMAPS.keys()), label="Basemap") \
.bind_value(state, "map_basemap")
            ui.select(field_options, label="Color by (vector mode)") \
.bind_value(state, "map_field") \
.bind_visibility_from(state, "map_layer", value="vector")
            ui.select(DEFAULT_COLORMAPS, label="Colormap") \
.bind_value(state, "map_cmap")
            with ui.row().classes("items-center gap-2") \
.bind_visibility_from(state, "map_layer", value="vector"):
                ui.label("Point size:")
                ui.slider(min=1.0, max=40.0, step=1.0) \
.bind_value(state, "map_point_size").classes("w-40") \
.tooltip("Marker size for the colored points (vector mode).")
                ui.label().bind_text_from(state, "map_point_size", lambda v: f"{v:.0f}")
        with ui.column().classes("min-w-[18rem]"):
            ui.checkbox("Show source ortho overlay").bind_value(state, "map_show_ortho")
            # The UI shows "transparency" (1 - alpha); state stores alpha.
            with ui.row().classes("items-center gap-2") \
.bind_visibility_from(state, "map_layer", value="raster"):
                ui.label("Background transparency:")
                ui.slider(min=0.0, max=1.0, step=0.05) \
.bind_value(state, "map_ortho_alpha",
                                forward=lambda v: 1.0 - v,
                                backward=lambda v: 1.0 - v) \
.bind_enabled_from(state, "map_show_ortho") \
.classes("w-40") \
.tooltip("How see-through the source ortho-image is, revealing the basemap below. "
                             "0 = fully opaque (basemap hidden), 1 = invisible (only basemap shows). "
                             "Raster mode only.")
                ui.label().bind_text_from(state, "map_ortho_alpha", lambda v: f"{1 - v:.2f}")
            with ui.row().classes("items-center gap-2") \
.bind_visibility_from(state, "map_layer", value="raster"):
                ui.label("Foreground transparency:")
                ui.slider(min=0.0, max=1.0, step=0.05) \
.bind_value(state, "map_raster_alpha",
                                forward=lambda v: 1.0 - v,
                                backward=lambda v: 1.0 - v) \
.classes("w-40") \
.tooltip("How see-through the colored data raster is, revealing the ortho/basemap below. "
                             "0 = fully opaque (only data visible), 1 = invisible.")
                ui.label().bind_text_from(state, "map_raster_alpha", lambda v: f"{1 - v:.2f}")
            ui.checkbox("Coordinate grid").bind_value(state, "map_show_grid")
            ui.checkbox("Colorbar legend").bind_value(state, "map_show_legend")
            ui.checkbox("Pointy colorbar tips").bind_value(state, "map_show_colorbar_extends") \
.bind_visibility_from(state, "map_show_legend") \
.tooltip("Triangular caps at both ends of the colorbar, indicating "
                         "values outside vmin/vmax are clipped to the endpoint colors. "
                         "Disable for cyclic colormaps (twilight, hsv) where there's no "
                         "'beyond the range' direction.")
        with ui.column().classes("min-w-[16rem]"):
            ui.checkbox("CRS info box").bind_value(state, "map_show_crs")
            ui.checkbox("North arrow").bind_value(state, "map_show_north") \
.tooltip("Compass-style north arrow in the upper-right corner.")
            ui.checkbox("Scale bar").bind_value(state, "map_show_scale") \
.tooltip("Cartographic scale bar with one black/white subdivision.")
            ui.checkbox("Zebra border").bind_value(state, "map_show_zebra") \
.tooltip("Cartographic zebra border (alternating black/white segments) "
                         "around the map edges.")
            ui.input(label="Title (optional)").bind_value(state, "map_title")

    ui.separator()
    ui.markdown("**Color scale**")
    with ui.row().classes("items-center gap-4"):
        ui.toggle(["quantile", "linear"]).bind_value(state, "map_color_scale") \
.tooltip("'quantile' uses BoundaryNorm with 256 quantile-spaced edges, "
                     "giving uniform color distribution across the empirical data. "
                     "'linear' uses 5th-95th percentiles by default for vmin/vmax "
                     "(robust to outliers; set vmin/vmax manually for true min/max).")
        with ui.row().classes("items-center gap-2"):
            ui.checkbox("Auto vmin").bind_value(state, "map_vmin_auto")
            ui.number(label="vmin", step=0.001, format="%.3f") \
.bind_value(state, "map_vmin") \
.bind_enabled_from(state, "map_vmin_auto", backward=lambda v: not v) \
.classes("w-32")
        with ui.row().classes("items-center gap-2"):
            ui.checkbox("Auto vmax").bind_value(state, "map_vmax_auto")
            ui.number(label="vmax", step=0.001, format="%.3f") \
.bind_value(state, "map_vmax") \
.bind_enabled_from(state, "map_vmax_auto", backward=lambda v: not v) \
.classes("w-32")

    ui.separator()
    ui.markdown("**Display units** — affects the colorbar and labels only; "
                "the underlying data is never modified.")
    with ui.row().classes("items-center gap-4"):
        ui.toggle(["metric", "imperial"]).bind_value(state, "map_unit_system") \
.tooltip("Metric (m, cm, mm; m/s; Pa) or imperial (ft, in; ft/s; psi).")
        ui.select(
            options=["auto", "m", "cm", "mm", "ft", "in"],
            label="Size unit (override)",
        ).bind_value(state, "map_size_unit") \
.classes("w-56") \
.tooltip("'auto' picks m/cm/mm or ft/in based on data magnitude. "
                     "Override to force a specific unit. Affects only fields that "
                     "represent sizes (lengths, diameters, areas).")

    _map_exp_sep = ui.separator()
    _map_exp_title = ui.markdown("**Export**")

    with ui.row().classes("w-full gap-4 items-end") as _map_exp_row:
        ui.number(label="DPI", min=72, max=600, step=50).bind_value(state, "map_dpi") \
.classes("w-32") \
.tooltip("Resolution in dots per inch. Ignored for vector formats (.pdf, .svg). "
                     "300 is publication-quality for PNG.")
        ui.input(label="Output file (.png, .pdf, .svg)").bind_value(state, "map_out") \
.classes("flex-grow") \
.tooltip("Path where the exported figure is saved. PNG is the default; "
                     "PDF and SVG are vector formats that scale without quality loss.")
        def _browse_save():
            current = state.map_out or ""
            if current:
                initial = Path(current).name
                initial_dir = str(Path(current).parent)
            elif state.map_layer == "vector" and state.map_csv:
                initial = f"{Path(state.map_csv).stem}_map.png"
                initial_dir = default_starting_dir("maps")
            elif state.map_layer == "raster" and state.map_raster:
                initial = f"{Path(state.map_raster).stem}_map.png"
                initial_dir = default_starting_dir("maps")
            else:
                initial = "map.png"
                initial_dir = default_starting_dir("maps")
            picked = native_save_file_picker(
                title="Save figure as",
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF document", "*.pdf"),
                    ("SVG vector", "*.svg"),
                    ("All files", "*.*"),
                ],
                initialfile=initial,
                defaultextension=".png",
                initialdir=initial_dir,
            )
            if picked:
                state.map_out = picked
        ui.button("Browse…", icon="folder_open", on_click=_browse_save).props("outline") \
.tooltip("Choose where to save the exported figure.")

    # Last auto-filled candidate and its project: state.map_out equal to the
    # candidate means the user has not customised it.
    _map_out_last = {"name": "", "candidate": ""}

    def _default_map_out():
        """Auto-derive a default output path under the active project's maps/,
        refreshed on every input change unless the user edited it.
        """
        figures_dir = Path(default_starting_dir("maps"))
        proj_changed = (_map_out_last["name"] != state.current_project)
        candidate = ""
        if state.map_layer == "vector" and state.map_csv:
            candidate = str(
                figures_dir / f"{Path(state.map_csv).stem}_map.png")
        elif state.map_layer == "raster" and state.map_raster:
            candidate = str(
                figures_dir / f"{Path(state.map_raster).stem}_map.png")

        if not candidate:
            return

        was_user_edited = (state.map_out
                            and state.map_out
                            != _map_out_last["candidate"])

        if not state.map_out:
            state.map_out = candidate
        elif proj_changed:
            state.map_out = candidate
        elif not was_user_edited and state.map_out != candidate:
            state.map_out = candidate
        _map_out_last["name"] = state.current_project
        _map_out_last["candidate"] = candidate
    ui.timer(1.0, _default_map_out)

    ui.separator()

    with ui.row().classes("items-center gap-2") as _map_action_row:
        preview_btn = ui.button("Preview", icon="visibility") \
.props("color=primary outline") \
.tooltip("Render the current configuration inline without "
                     "writing a file. Useful to dial in colors / scale "
                     "before committing.")
        # Copies the PNG bytes stashed in _last_preview_state; no re-render.
        save_preview_btn = ui.button("Save preview",
                                       icon="save") \
.props("outline") \
.tooltip("Save the most-recently-rendered preview to a "
                     "file. Skips the re-render so dialling in the "
                     "exact same parameters again is free.")
        add_job_btn = ui.button("Add to queue", icon="add") \
.props("color=primary") \
.tooltip("Snapshot the current configuration into the queue. "
                     "The form stays editable so you can stage another "
                     "map (different field, basemap, etc.).")
        # One job per .tif under results/rasters/, sharing the layout
        # settings; per-raster settings (range, units, title) reset to auto.
        add_all_rasters_btn = ui.button("Add all rasters",
                                         icon="dynamic_feed") \
.props("flat color=primary") \
.tooltip("Queue one raster map per .tif in this project "
                     "that belongs to the selected source ortho. "
                     "Rasters from other ortho-images sit in different "
                     "geographic extents and are skipped to avoid "
                     "misalignment. Layout settings (basemap, ortho "
                     "overlay, transparency, grid, legend, north "
                     "arrow, scale bar, zebra border, DPI) are "
                     "inherited from the current form. Per-raster "
                     "settings (output filename, vmin/vmax, unit, "
                     "colorbar range) are reset to auto.")
        run_queue_btn = ui.button("Run all queued", icon="playlist_play") \
.props("color=primary outline") \
.tooltip("Export every queued map in order.")
        def _reset_map_queue():
            n = 0
            for j in state.map_jobs:
                if j.get("status") in ("done", "error"):
                    j["status"] = "pending"
                    j["error"] = None
                    n += 1
            _render_map_jobs.refresh()
            ui.notify(f"Reset {n} job(s) to pending.",
                       type="info" if n else "warning")
        ui.button("Reset all", icon="restart_alt",
                  on_click=_reset_map_queue) \
.props("flat color=primary") \
.tooltip("Mark every done / errored job pending again "
                     "so 'Run all queued' will re-render them.")
        def _clear_map_queue():
            state.map_jobs.clear()
            _render_map_jobs.refresh()
            ui.notify("Map queue cleared.", type="info")
        ui.button("Clear queue", icon="delete_sweep",
                  on_click=_clear_map_queue) \
.props("flat color=grey") \
.tooltip("Empty the queue. Already-written maps stay on disk.")
        map_queue_count_label = ui.label("").classes("text-sm text-grey-7 ml-2")

    map_queue_container = ui.column().classes("w-full mt-2 mb-2")

    plot_container = ui.column().classes("w-full pm-map-preview")
    log_widget = live_log("map", build_log_console(height="h-24"))

    # The preview comes right after the action bar, then the queue and the
    # log; the Export group moves above Map content so the styling groups
    # are the last inputs before the bar.
    _map_root = _map_action_row.parent_slot.parent

    def _map_kids():
        return _map_root.default_slot.children
    plot_container.move(_map_root,
                        target_index=_map_kids().index(_map_action_row) + 1)
    _at = _map_kids().index(_map_content_sep)
    for _el in (_map_exp_sep, _map_exp_title, _map_exp_row):
        _el.move(_map_root, target_index=_at)
        _at += 1

    def _snapshot_map_config():
        """Deep-copy the current form into a queueable job dict."""
        return {
            "layer":          state.map_layer,
            "ortho":          state.map_ortho,
            "csv":            state.map_csv,
            "raster":         state.map_raster,
            "field":          state.map_field,
            "cmap":           state.map_cmap,
            "basemap":        state.map_basemap,
            "show_ortho":     state.map_show_ortho,
            "ortho_alpha":    state.map_ortho_alpha,
            "raster_alpha":   state.map_raster_alpha,
            "show_grid":      state.map_show_grid,
            "show_legend":    state.map_show_legend,
            "show_crs":       state.map_show_crs,
            "show_north":     state.map_show_north,
            "show_scale":     state.map_show_scale,
            "show_zebra":     state.map_show_zebra,
            "show_colorbar_extends": state.map_show_colorbar_extends,
            "title":          state.map_title,
            "point_size":     state.map_point_size,
            "color_scale":    state.map_color_scale,
            "vmin_auto":      state.map_vmin_auto,
            "vmax_auto":      state.map_vmax_auto,
            "vmin":           state.map_vmin,
            "vmax":           state.map_vmax,
            "unit_system":    state.map_unit_system,
            "size_unit":      state.map_size_unit,
            "dpi":            int(state.map_dpi),
            "out_path":       state.map_out,
            "status":         "pending",
            "error":          None,
        }

    def _validate_map_config(job):
        """Cheap pre-flight: paths set and output filename provided."""
        if job["layer"] == "vector" and not job["csv"]:
            return "Set the clast CSV (Map content, above) for vector mode."
        if job["layer"] == "raster" and not job["raster"]:
            return "Set the raster (Map content, above) for raster mode."
        if not job["out_path"]:
            return "Set the output filename (Export, above)."
        return None

    def _map_primary(job, idx):
        return "out:"

    def _map_params(job, idx):
        src_name = (Path(job["csv"]).name
                    if job["layer"] == "vector" and job["csv"]
                    else Path(job["raster"]).name
                    if job["raster"] else "—")
        return (f"({job['layer']}, {src_name}, field={job['field']}, "
                f"cmap={job['cmap']}, dpi={job['dpi']})")

    def _map_extra(job, idx):
        # The output path stays editable after queueing.
        def _update_out_path(e, _idx=idx):
            new_val = e.value or ""
            if new_val:
                state.map_jobs[_idx]["out_path"] = new_val
        out_field = ui.input(
            value=job["out_path"],
            on_change=_update_out_path) \
.props("dense outlined") \
.classes("text-xs w-96") \
.tooltip(
                "Output file path for this job. Edit directly to override "
                "the auto-derived default; the change applies on the next "
                "‘Run all queued’. The file is saved exactly here — verify "
                "before running so jobs do not overwrite each other.")
        if job.get("status") == "running":
            _disable(out_field, "Running — the path is fixed until the job ends")

    def _map_delete(idx):
        state.map_jobs.pop(idx)
        _render_map_jobs.refresh()

    @ui.refreshable
    def _render_map_jobs():
        render_queue(
            map_queue_container, state.map_jobs,
            title="Queued maps",
            count_label=map_queue_count_label,
            primary_text=_map_primary,
            params_text=_map_params,
            render_extra=_map_extra,
            on_delete=_map_delete,
        )
    _render_map_jobs()

    def _on_add_map_job():
        snap = _snapshot_map_config()
        err = _validate_map_config(snap)
        if err:
            ui.notify(err, type="warning")
            return
        state.map_jobs.append(snap)
        _render_map_jobs.refresh()
        ui.notify(
            f"Map #{len(state.map_jobs)} saved to queue. "
            f"Configure another or click Run all queued.",
            type="positive")
    add_job_btn.on_click(_on_add_map_job)

    def _on_add_all_rasters():
        """Queue one map job per raster derived from the selected ortho
        (filename starts with the ortho's stem; other rasters sit in other
        extents). Jobs share the form's layout settings; the output name and
        vmin / vmax are auto-derived per raster.
        """
        if not state.current_project:
            ui.notify("Pick a project (Project, left panel) first.", type="warning")
            return
        if not state.map_ortho:
            ui.notify(
                "Set *Source ortho-image* (Inputs, above) first. The button queues "
                "every raster that belongs to that ortho — without an "
                "ortho selected we cannot tell which rasters are "
                "geographically aligned.",
                type="warning", multi_line=True, timeout=8000)
            return
        ortho_stem = Path(state.map_ortho).stem
        rasters_dir = project_path(state.current_project, "rasters")
        if not rasters_dir.is_dir():
            ui.notify(
                f"No rasters folder at {rasters_dir}. Run the Rasterize tab for this ortho first.", type="warning")
            return
        all_tifs = [
            p for p in rasters_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".tif"
            and not p.stem.startswith(("_smoke_", "_verify_"))]
        # Stem-prefix match keeps the merged and the per-window rasters.
        tifs = sorted(p for p in all_tifs
                       if p.stem.startswith(ortho_stem)
                       or naming.same_image(p.name, ortho_stem))
        if not tifs:
            others = len(all_tifs)
            extra = (f" ({others} raster(s) belong to OTHER source "
                     f"image(s) and were skipped to avoid "
                     f"misalignment.)" if others else "")
            ui.notify(
                f"No .tif files for {ortho_stem} in "
                f"{rasters_dir.name}/.{extra} Run the Rasterize tab "
                f"against this ortho first.",
                type="warning", multi_line=True, timeout=8000)
            return

        figures_dir = project_path(state.current_project, "maps")
        figures_dir.mkdir(parents=True, exist_ok=True)
        shared = _snapshot_map_config()
        shared["layer"] = "raster"
        shared["csv"] = ""              # vector mode irrelevant
        shared["field"] = ""            # ditto
        shared["title"] = ""            # let map_export derive from name
        shared["vmin_auto"] = True
        shared["vmax_auto"] = True
        shared["vmin"] = 0.0
        shared["vmax"] = 0.0

        n_added = 0
        for tif in tifs:
            cfg = dict(shared)
            cfg["raster"] = str(tif)
            cfg["out_path"] = str(figures_dir / f"{tif.stem}_map.png")
            cfg["status"] = "pending"
            cfg["error"] = None
            err = _validate_map_config(cfg)
            if err:
                log_widget.push(
                    f"  [warn] skipping {tif.name}: {err}")
                continue
            state.map_jobs.append(cfg)
            n_added += 1

        _render_map_jobs.refresh()
        skipped = len(all_tifs) - len(tifs)
        skipped_note = (f" ({skipped} raster(s) from other source "
                        f"images were skipped to avoid misalignment.)"
                        if skipped else "")
        ui.notify(
            f"Queued {n_added} raster map(s) for {ortho_stem}."
            f"{skipped_note} Click Run all queued to export.",
            type="positive", timeout=8000, multi_line=True)

    add_all_rasters_btn.on_click(_on_add_all_rasters)

    def _check_ortho_data_overlap(ortho_path, csv_path, raster_path):
        """Bounding-box check that the ortho intersects the data layer;
        a warning string when it does not, else None."""
        try:
            from osgeo import gdal
            ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
            if ds is None:
                return None
            gt = ds.GetGeoTransform()
            xs = [gt[0], gt[0] + ds.RasterXSize * gt[1]]
            ys = [gt[3], gt[3] + ds.RasterYSize * gt[5]]
            o_xmin, o_xmax = min(xs), max(xs)
            o_ymin, o_ymax = min(ys), max(ys)
            ds = None
        except Exception:
            return None

        if csv_path:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, usecols=["x", "y"])
                d_xmin, d_xmax = float(df["x"].min()), float(df["x"].max())
                d_ymin, d_ymax = float(df["y"].min()), float(df["y"].max())
            except Exception:
                return None
        elif raster_path:
            try:
                from osgeo import gdal
                rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
                if rds is None:
                    return None
                rgt = rds.GetGeoTransform()
                rxs = [rgt[0], rgt[0] + rds.RasterXSize * rgt[1]]
                rys = [rgt[3], rgt[3] + rds.RasterYSize * rgt[5]]
                d_xmin, d_xmax = min(rxs), max(rxs)
                d_ymin, d_ymax = min(rys), max(rys)
                rds = None
            except Exception:
                return None
        else:
            return None

        if d_xmax < o_xmin or d_xmin > o_xmax or d_ymax < o_ymin or d_ymin > o_ymax:
            return (f"The chosen ortho's footprint ({o_xmin:.0f}, {o_ymin:.0f}) – "
                    f"({o_xmax:.0f}, {o_ymax:.0f}) does not overlap the data layer "
                    f"({d_xmin:.0f}, {d_ymin:.0f}) – ({d_xmax:.0f}, {d_ymax:.0f}). "
                    f"You may have selected the wrong file, or the CRS does not match. "
                    f"The map will still render, but the ortho will not show under the data.")
        return None

    def _routed_log(msg: str):
        """Push to the log widget AND surface basemap failures via toast."""
        log_widget.push(msg)
        if "basemap fetch failed" in msg or "basemap unavailable" in msg:
            ui.notify(
                f"The basemap could not load — see the log for details. "
                f"Try a different provider (Basemap, above) — OpenStreetMap usually works.",
                type="warning", timeout=8000)

    def _render_map_job(job, *, preview_png=None, save_png=None,
                        out_path=None, dpi=None):
        """Render one map job in a worker subprocess and save the requested
        artefacts there; raises RuntimeError with the worker's one-line error
        on failure."""
        outcome = _worker_mod.run_job(
            "publication_map",
            {"job": job, "preview_png": preview_png, "save_png": save_png,
             "out_path": out_path, "dpi": int(dpi or job.get("dpi") or 300)},
            log_cb=_routed_log)
        if not outcome.ok:
            raise RuntimeError(_job_error_text(outcome))
        return outcome

    def _do_preview():
        """Render the current form inline — no file written."""
        snap = _snapshot_map_config()
        if snap["layer"] == "vector" and not snap["csv"]:
            ui.notify("Set the clast CSV (Map content, above) for vector mode.", type="warning")
            return
        if snap["layer"] == "raster" and not snap["raster"]:
            ui.notify("Set the raster (Map content, above) for raster mode.", type="warning")
            return

        if snap["ortho"]:
            warn = _check_ortho_data_overlap(
                snap["ortho"],
                snap["csv"] if snap["layer"] == "vector" else None,
                snap["raster"] if snap["layer"] == "raster" else None)
            if warn:
                ui.notify(f"{warn} Check the ortho and layer (Map content, above).",
                          type="warning", timeout=10000)
        plot_container.clear()
        log_widget.clear()
        preview_btn.props("loading")

        def _worker():
            try:
                # Both renders happen in the worker subprocess: the 120 dpi
                # preview and the full-dpi PNG cached for Save preview.
                import tempfile as _tf
                import uuid as _uuid_mod
                stem = os.path.join(_tf.gettempdir(),
                                    f"pm_map_{_uuid_mod.uuid4().hex[:10]}")
                preview_png = stem + "_preview.png"
                save_png = stem + "_full.png"
                with capture_stdout_to_log(log_widget):
                    _render_map_job(snap, preview_png=preview_png,
                                    save_png=save_png,
                                    dpi=int(snap.get("dpi") or 300))
                with plot_container:
                    ui.image(preview_png).classes("w-full")
                try:
                    with open(save_png, "rb") as fh:
                        _last_preview_state["png_bytes"] = fh.read()
                    _last_preview_state["snap"] = snap
                    log_widget.push(
                        f"  (preview cached; "
                        f"💾 Save preview is ready)")
                except Exception as ex:
                    log_widget.push(
                        f"  [preview cache] skipped: {ex}")
            except Exception as e:
                log_widget.push(f"❌  {type(e).__name__}: {e}")
            finally:
                preview_btn.props(remove="loading")
        threading.Thread(target=_worker, daemon=True).start()

    _last_preview_state: dict = {"png_bytes": None, "snap": None}

    def _do_save_preview():
        png = _last_preview_state.get("png_bytes")
        if not png:
            ui.notify("No preview to save. Press *Preview* (the row above) first.", type="warning")
            return
        snap = _last_preview_state.get("snap") or {}
        default = (snap.get("out_path")
                   or default_starting_dir("maps"))
        p = native_save_file_picker(
            title="Save preview as map file…",
            initialdir=str(Path(default).parent
                            if default else
                            default_starting_dir("maps")),
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if not p:
            return
        try:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(png)
            ui.notify(f"Preview saved -> {p}", type="positive")
        except Exception as ex:
            ui.notify(f"Save failed: {ex}. Choose another location, then Save preview again.", type="negative")

    save_preview_btn.on_click(_do_save_preview)

    def _do_run_map_queue():
        """Iterate over queued jobs, render each, save to its out_path."""
        pending_idx = [i for i, j in enumerate(state.map_jobs)
                       if j.get("status") in ("pending", "error", "interrupted")]
        if not pending_idx:
            ui.notify("No pending maps — press *Add to queue* (above) first.",
                      type="info")
            return
        run_queue_btn.props("loading")
        _disable(add_job_btn, "Queue running — add maps once it has finished")
        plot_container.clear()
        log_widget.clear()

        def _worker():
            log_widget.push(
                f"============================================================\n"
                f"  Map batch started — {len(pending_idx)} job(s)\n"
                f"============================================================")
            for n, idx in enumerate(pending_idx, start=1):
                job = state.map_jobs[idx]
                job["status"] = "running"
                _render_map_jobs.refresh()
                src = (Path(job["csv"]).name
                       if job["layer"] == "vector" and job["csv"]
                       else Path(job["raster"]).name if job["raster"] else "—")
                log_widget.push(
                    f"\n[{n}/{len(pending_idx)}] {src} → {Path(job['out_path']).name}")
                try:
                    import tempfile as _tf
                    import uuid as _uuid_mod
                    preview_png = os.path.join(
                        _tf.gettempdir(),
                        f"pm_map_{_uuid_mod.uuid4().hex[:10]}_preview.png")
                    with capture_stdout_to_log(log_widget):
                        _render_map_job(job, preview_png=preview_png,
                                        out_path=job["out_path"],
                                        dpi=int(job["dpi"]))
                    log_widget.push(
                        f"  ✅  Wrote {job['out_path']} at {job['dpi']} dpi.")
                    # The map is written whatever happened to the page;
                    # only its preview needs somewhere to go.
                    if element_alive(plot_container):
                        plot_container.clear()
                        with plot_container:
                            ui.label(f"Preview of #{idx+1}: "
                                     f"{Path(job['out_path']).name}") \
.classes("text-sm text-grey-7")
                            ui.image(preview_png).classes("w-full")
                    job["status"] = "done"
                    job["error"] = None
                except Exception as ex:
                    import traceback
                    job["status"] = "error"
                    job["error"] = f"{type(ex).__name__}: {ex}"
                    log_widget.push(f"  ❌  {job['error']}")
                    log_widget.push(traceback.format_exc())
                _render_map_jobs.refresh()
            log_widget.push(
                f"\n✅  Map batch finished. "
                f"done={sum(1 for j in state.map_jobs if j['status']=='done')}, "
                f"errors={sum(1 for j in state.map_jobs if j['status']=='error')}")
            run_queue_btn.props(remove="loading")
            _enable(add_job_btn)
        threading.Thread(target=_worker, daemon=True).start()

    preview_btn.on_click(_do_preview)
    run_queue_btn.on_click(_do_run_map_queue)


# --- Orthorectify tab helpers ---
DEFAULT_PORT = 8081
"""The port the GUI listens on unless told otherwise.

Override with ``PEBBLEMAPPER_PORT`` (``CSM_PORT`` is honoured too).
"""


def _listen_port():
    """The port to bind, from the environment or :data:`DEFAULT_PORT`.
    A malformed value is ignored rather than raised."""
    for name in ("PEBBLEMAPPER_PORT", "CSM_PORT"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            port = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return DEFAULT_PORT


def _ortho_corners_path(src_path):
    """Path of the ``<stem>_corners.txt`` sidecar next to a source photo."""
    p = Path(src_path)
    return p.with_name(f"{p.stem}_corners.txt")


# Sidecar texts by path, validated by (mtime, size). The folder table, the
# guide and the batch plan read every photograph's sidecars on the event
# loop; on a slow disk the re-reads could stall the socket past its ping.
_SIDECAR_CACHE = {}


def _sidecar_text(path):
    """Text of a sidecar file, or ``None`` if absent or unreadable.

    Cached by ``(mtime_ns, size)``; writers drop their entry so a rewrite
    within the same timestamp tick is never served stale.
    """
    try:
        p = Path(path)
        st = p.stat()
    except OSError:
        return None
    key = str(p)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _SIDECAR_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    _SIDECAR_CACHE[key] = (stamp, text)
    return text


def _forget_sidecar(path):
    _SIDECAR_CACHE.pop(str(Path(path)), None)


def _load_ortho_corners(src_path):
    """Corner pixel coords from ``<stem>_corners.txt`` (one ``x,y`` per
    line) as ``[x, y]`` floats, or ``[]`` if absent/unreadable."""
    try:
        f = _ortho_corners_path(src_path)
        text = _sidecar_text(f)
        if text is None:
            return []
        pts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("seg="):
                continue
            parts = line.replace(";", ",").split(",")
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
        return pts[:4]
    except Exception:
        return []


def _ortho_segments_path(src_path):
    """Path of the ``<stem>_segments.txt`` sidecar next to a source photo.

    A separate file: the corners format predates this application and is
    read by other tools, so nothing may be added to it.
    """
    p = Path(src_path)
    return p.with_name(f"{p.stem}_segments.txt")


def _load_ortho_segments(src_path):
    """The segment lengths recorded for one photograph, or ``None``.

    ``None`` means "never measured" and must never be replaced by a default.
    """
    try:
        f = _ortho_segments_path(src_path)
        raw = _sidecar_text(f)
        if raw is None:
            return None
        # Format: one line of numbers, then `key=value` annotations.
        text = "\n".join(
            ln for ln in raw.splitlines()
            if "=" not in ln)
        vals = [float(v) for v in text.replace(";", ",").split(",")
                if v.strip()]
        if not vals:
            return None
        while len(vals) < 4:
            vals.append(vals[-1])
        return vals[:4]
    except Exception:
        return None


def _load_ortho_frame_thickness(src_path):
    """The frame thickness recorded for one photograph, or ``None``."""
    try:
        f = _ortho_segments_path(src_path)
        raw = _sidecar_text(f)
        if raw is None:
            return None
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln.lower().startswith("frame="):
                v = float(ln[6:])
                return v if v >= 0.0 else None
        return None
    except Exception:
        return None


def _load_ortho_corner_source(src_path):
    """Provenance of this photograph's corners: 'suggested' when somebody
    accepted an automatically detected frame, None when a person clicked."""
    try:
        f = _ortho_segments_path(src_path)
        raw = _sidecar_text(f)
        if raw is None:
            return None
        for ln in raw.splitlines():
            if ln.strip().lower().startswith("source="):
                return ln.split("=", 1)[1].strip() or None
    except Exception:
        return None
    return None


def _save_ortho_segments(src_path, seg_lengths, frame_thickness_m=None,
                         source=None):
    """Record the size a photograph is being measured at, beside it."""
    try:
        vals = [float(v) for v in (seg_lengths or [])][:4]
        if not vals:
            return None
        body = ",".join(f"{v:.6f}" for v in vals) + "\n"
        if frame_thickness_m not in (None, ""):
            body += f"frame={float(frame_thickness_m):.6f}\n"
        if source:
            body += f"source={str(source).strip()}\n"
        f = _ortho_segments_path(src_path)
        f.write_text(body, encoding="utf-8")
        _forget_sidecar(f)
        return str(f)
    except Exception:
        return None


def _save_ortho_corners(src_path, corners, seg_lengths=None,
                        frame_thickness_m=None, source=None):
    """Write corners to ``<stem>_corners.txt`` (one ``x,y`` per line).
    Best-effort; returns the path or None. Segment lengths and ``source``
    (None or "suggested") go to the segments sidecar, never the corner file.
    """
    try:
        f = _ortho_corners_path(src_path)
        f.write_text("\n".join(f"{float(x)},{float(y)}" for x, y in corners)
                     + "\n", encoding="utf-8")
        _forget_sidecar(f)
        if seg_lengths:
            _save_ortho_segments(src_path, seg_lengths,
                                 frame_thickness_m, source=source)
        return str(f)
    except Exception:
        return None


_PM_LAYOUT_CSS = """
<style id="pm-layout">
/* Sticky toolbars need the page as scroll root: Quasar's tab-panel clip
   (`.q-panel-parent{overflow:hidden}`) only exists for the slide between
   tabs. */
.q-panel-parent, .q-tab-panels .q-panel { overflow: visible !important; }
.pm-ortho-status { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pm-sticky-toolbar { position: sticky; top: var(--pm-header-h, 68px);
  z-index: 5; background: #fff; border-bottom: 1px solid #e0e4e8;
  padding: 0 4px; }
.pm-sticky-toolbar .q-select .q-field__native { display: block; overflow: hidden;
  white-space: nowrap; text-overflow: ellipsis; }
</style>
<script>
(function () {
  function setHeader() {
    var h = document.querySelector('.q-header');
    if (h) document.documentElement.style.setProperty('--pm-header-h',
        Math.round(h.getBoundingClientRect().height) + 'px');
  }
  // Report the viewport size for "Fit". An event, not an awaited request:
  // it never times out.
  function tell() {
    setHeader();
    if (window.emitEvent) {
      try { window.emitEvent('pm_viewport', { h: window.innerHeight,
                                              w: window.innerWidth }); }
      catch (e) {}
    }
  }
  // Several tries: emitEvent exists before the socket is connected, and a
  // slow client (or a throttled background tab) connects late.
  [300, 1500, 4000, 8000, 15000].forEach(function (t) { setTimeout(tell, t); });
  var _rs = null;
  window.addEventListener('resize', function () {
    clearTimeout(_rs); _rs = setTimeout(tell, 250); });
})();
</script>
"""


# Token of the newest Orthorectify build. Elements of an earlier build (a
# page whose client is not yet deleted, an off-screen test build) stay bound
# to the state singleton and would answer the same events; only the live
# build acts.
_ORTHO_LIVE_BUILD = {"token": None}


def _ensure_layout_css():
    """Inject the layout CSS once per page (client)."""
    try:
        client = context.client
    except Exception:
        return
    if getattr(client, "_pm_layout_css", False):
        return
    client._pm_layout_css = True
    try:
        ui.add_head_html(_PM_LAYOUT_CSS)
    except Exception:
        pass


def _run_js(code: str) -> None:
    """Fire-and-forget JavaScript for a canvas's client-side aids.

    NiceGUI raises without a connected client; a drawing aid must never
    break the corner pick or the image load it decorates.
    """
    try:
        ui.run_javascript(code)
    except Exception:
        pass


def build_orthorectify_tab():
    from functions import orthorectify as _or_mod
    """Quadrat photographs in, flat scaled images out.

    Three bands in reading order: Inputs (folder, photo table, frame
    geometry, folder run) -> Work surface (sticky toolbar, status line,
    the photograph in a box capped at 70 % of the viewport) -> Results.
    """
    _ensure_layout_css()
    _build_token = object()
    _ORTHO_LIVE_BUILD["token"] = _build_token
    try:
        _build_client = context.client
    except Exception:
        _build_client = None

    def _live():
        """Does this build still have a page? Per client, not per build:
        two windows are both live until NiceGUI deletes one's client. The
        shared auto-index client is never deleted, so it falls back to
        the newest-build token."""
        from nicegui import Client as _Client
        c = _build_client
        if c is None:
            return True
        if c.is_auto_index_client:
            return _ORTHO_LIVE_BUILD["token"] is _build_token
        return _Client.instances.get(c.id) is c

    def _on_proj_change():
        if not _live():
            return
        state.ortho_src_path = ""
        state.ortho_out_path = ""
        _seed_ortho_folder(force=True)
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Orthorectify")
    ui.label("Mark the four corners of the quadrat, give its size, get a "
             "flat scaled image for Quadrat-mode detection. Corners and "
             "sizes are saved beside each photograph.") \
        .classes("text-sm text-grey-7")

    # ---- The three bands ------------------------------------------------ #
    ortho_band_a = ui.column().classes("w-full gap-2 pm-ortho-inputs")
    # B and C share one container: a sticky bar stays within its parent's
    # bounds, and it must still be there while the result is looked at.
    with ui.column().classes("w-full gap-1 pm-ortho-work"):
        ortho_band_b = ui.column().classes("w-full gap-1 pm-ortho-surface")
        ortho_band_c = ui.column().classes("w-full gap-1 pm-ortho-results")

    # Set while the code is driving the image dropdown rather than the user,
    # so its on_change does not bounce back into the navigation that set it.
    _ortho_select_sync = {"quiet": False}
    # The picker select is rebuilt in a refreshable and may fire before
    # `_switch_to_ortho_image` exists.
    _ortho_picker_handlers = {"switch": lambda _v: None}
    # The folder the surface currently shows (see _refresh_ortho_dir).
    _ortho_dir_seen = {"dir": state.ortho_image_dir}

    # --- Band A: Inputs --------------------------------------------------- #
    with ortho_band_a:
        # --- the folder, and what it already knows ---
        with ui.card().classes("w-full") \
                .style("background-color: #f4f6f9; border: 1px solid #ddd;"):
            with ui.row().classes("w-full items-center gap-2"):
                ortho_dir_inp = ui.input(label="Folder of photographs") \
                    .classes("flex-grow") \
                    .bind_value(state, "ortho_image_dir") \
                    .tooltip("The photographs to rectify. Pick one in the "
                             "table or the toolbar below to work on it; "
                             "corners and sizes are kept per photograph.")

                def _browse_ortho_dir():
                    current = state.ortho_image_dir
                    initialdir = current if current \
                        else default_starting_dir("images")
                    picked = native_dir_picker("Pick image directory",
                                               initialdir=initialdir)
                    if picked:
                        state.ortho_image_dir = picked
                        _refresh_ortho_dir()
                ui.button("Browse…", icon="folder_open",
                          on_click=_browse_ortho_dir).props("outline")
            ortho_dir_inp.on_value_change(lambda _: _refresh_ortho_dir())

            ortho_file_table = ui.table(
                columns=[
                    {"name": "photo", "label": "Photograph", "field": "photo",
                     "align": "left"},
                    {"name": "corners", "label": "Corners", "field": "corners",
                     "align": "left"},
                    {"name": "length", "label": "Size", "field": "length",
                     "align": "left"},
                    {"name": "output", "label": "Rectified output",
                     "field": "output", "align": "left"},
                    {"name": "drop", "label": "", "field": "drop",
                     "align": "right"},
                ], rows=[], row_key="photo").classes("w-full") \
                .props('dense flat :rows-per-page-options="[0]" hide-bottom') \
                .style("max-height: 22rem; overflow: auto")
            ortho_file_table.add_slot("body-cell-drop", r"""
                <q-td :props="props" auto-width>
                  <q-btn dense flat round icon="close" size="sm" color="grey-7"
                         @click.stop="() => $parent.$emit('dropRow', props.row)">
                    <q-tooltip>Take this photograph out of the list</q-tooltip>
                  </q-btn>
                </q-td>
            """)
            ortho_file_table.on("dropRow",
                                lambda e: _ortho_drop_image(
                                    (e.args or {}).get("photo")))
            ortho_file_table.on(
                "rowClick",
                lambda e: _switch_to_ortho_image(
                    (e.args[1] or {}).get("photo") if len(e.args or []) > 1
                    else None))
            ortho_file_table.bind_visibility_from(
                state, "ortho_image_list", backward=lambda v: bool(v))

        # --- The exception: one photograph from outside that folder ---
        with ui.expansion("Work on one photograph from outside this folder",
                          icon="insert_photo").classes("w-full"):
            ui.label("For a single photograph that lives elsewhere; the "
                     "Digitize tab hands images over this way too.") \
                .classes("text-sm text-grey-7 q-mb-xs")
            _outside_row = path_input_with_browse(
                "Source photograph",
                "ortho_src_path",
                kind="file",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.heic *.heif"),
                           ("All files", "*.*")],
                default_kind="images",
                on_change=lambda: _load_outside_photograph(),
            )
            _outside_row._path_input.on_value_change(
                lambda _: _load_outside_photograph())

        # --- Frame geometry: one size, its confirmation, the thickness ---
        _seg_state = {"override_seg2": False, "override_seg34": False}

        def _effective_segments():
            """The 4 segment lengths the user actually wants, applying the
            implicit equalities when the override checkboxes are off."""
            s1 = float(state.ortho_seg_1 or 0.001)
            s2 = float(state.ortho_seg_2 or 0.001) \
                if _seg_state["override_seg2"] else s1
            if _seg_state["override_seg34"]:
                s3 = float(state.ortho_seg_3 or 0.001)
                s4 = float(state.ortho_seg_4 or 0.001)
            else:
                s3, s4 = s1, s2
            return [s1, s2, s3, s4]

        def _sync_seg_list_all():
            state.ortho_seg_lengths = _effective_segments()

        with ui.card().classes("w-full") \
                .style("background-color: #f4f6f9; border: 1px solid #ddd;"):
            ui.label("Frame geometry").classes("text-sm font-bold")
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                # Segment 1 (corner 1 to 2); the other sides follow it
                # unless the non-square expander says otherwise.
                seg1_inp = ui.number(
                    label="Quadrat size (m)",
                    value=1.0, step=0.001, min=0.001, format="%.3f",
                ).bind_value(state, "ortho_seg_1") \
                    .classes("w-40") \
                    .tooltip("Side length in metres, corner 1 to corner 2. "
                             "The other sides follow it unless *Segment "
                             "lengths (non-square)* says otherwise. Used by "
                             "Rectify and by the folder run alike.")
                seg1_inp.on_value_change(lambda _: _sync_seg_list_all())

                # Without this the scale guard warns on every rectify at
                # the default size and can never be satisfied.
                seg_confirm_checkbox = ui.checkbox(
                    f"I confirm this size ({1.0:.3f} m)") \
                    .tooltip("Shown only while the size is still the untouched "
                             "default. Applies to THIS photograph, not to the "
                             "survey: a default accepted for one frame is not "
                             "a measurement of the next.")

                def _seg_confirm_changed(e):
                    name = Path(state.ortho_src_path or "").name
                    if not name:
                        return
                    entry = state.ortho_per_image.setdefault(name, {})
                    entry["seg_confirmed"] = bool(e.value)
                seg_confirm_checkbox.on_value_change(_seg_confirm_changed)
                seg_confirm_checkbox.bind_visibility_from(
                    state, "ortho_seg_lengths",
                    backward=lambda v: bool(v) and all(
                        abs(float(x) - 1.0) < 1e-9 for x in v))

                def _record_size_for_folder():
                    if not state.ortho_image_list:
                        ui.notify("Set *Folder of photographs* (Inputs, "
                                  "above) first.", type="warning")
                        return
                    segs = _effective_segments()
                    written, differing = 0, []
                    for f in state.ortho_image_list:
                        full = os.path.join(state.ortho_image_dir, f)
                        have = _load_ortho_segments(full)
                        if have and abs(float(have[0])
                                        - float(segs[0])) > 1e-9:
                            # A different recorded length is a measurement:
                            # list it, never replace it.
                            differing.append(f)
                            continue
                        if _save_ortho_segments(full, segs,
                                                state.ortho_frame_thickness_m):
                            written += 1
                    msg = (f"Recorded {segs[0]:.3f} m for {written} "
                           f"photograph(s).")
                    if differing:
                        msg += (f" {len(differing)} already record a "
                                f"different size and were left alone: "
                                + ", ".join(differing[:4])
                                + (" …" if len(differing) > 4 else ""))
                    state.ortho_batch_msg = msg
                    log_widget.push(msg)
                    ui.notify(msg, type="positive", timeout=12000)
                    _ortho_update_nav_summary()
                    _ortho_batch_preflight()

                ui.button("Record for every photograph", icon="straighten",
                          on_click=_record_size_for_folder) \
                    .props("outline dense no-caps") \
                    .tooltip("Writes this size beside each photograph in the "
                             "folder, which is what makes them ready for the "
                             "folder run. A photograph that already records "
                             "a DIFFERENT size is listed and left alone.")

                ui.number(label="Frame thickness (m) — optional",
                          step=0.005, min=0.0, format="%.3f") \
                    .bind_value(state, "ortho_frame_thickness_m") \
                    .classes("w-48") \
                    .tooltip("Width of the quadrat's frame. Needed by "
                             "*Suggest a position*; recorded with the output "
                             "so Georeference can crop the frame before "
                             "matching. Empty and 0.000 are different "
                             "answers: 0.000 means the frame is not in the "
                             "way.")
                ortho_edge_toggle = ui.toggle(["outer", "inner"]) \
                    .bind_value(state, "ortho_edge_mode") \
                    .props("dense") \
                    .tooltip("Which edge of the frame the corners are marked "
                             "against. Use the same edge for all four.")

            with ui.expansion("Segment lengths (non-square)",
                              icon="straighten").classes("w-full"):
                seg2_checkbox = ui.checkbox(
                    "Different value for segment 2 (non-square quadrat)") \
                    .tooltip("Enable if width differs from height.")
                seg2_block = ui.column().classes("gap-2")
                with seg2_block:
                    seg2_inp = ui.number(
                        label="Segment 2: corner 2 → corner 3",
                        value=1.0, step=0.001, min=0.001, format="%.3f",
                    ).bind_value(state, "ortho_seg_2").classes("w-48") \
                        .tooltip("Metres, corner 2 to corner 3.")
                    seg2_inp.on_value_change(lambda _: _sync_seg_list_all())
                    seg34_checkbox = ui.checkbox(
                        "Different values for segments 3 and 4 "
                        "(opposite sides not equal)") \
                        .tooltip("By default segment 3 = segment 1 and "
                                 "segment 4 = segment 2.")
                seg34_block = ui.row().classes("gap-3")
                with seg34_block:
                    seg3_inp = ui.number(
                        label="Segment 3: corner 3 → corner 4",
                        value=1.0, step=0.001, min=0.001, format="%.3f",
                    ).bind_value(state, "ortho_seg_3").classes("w-48") \
                        .tooltip("Metres, corner 3 to corner 4.")
                    seg3_inp.on_value_change(lambda _: _sync_seg_list_all())
                    seg4_inp = ui.number(
                        label="Segment 4: corner 4 → corner 1",
                        value=1.0, step=0.001, min=0.001, format="%.3f",
                    ).bind_value(state, "ortho_seg_4").classes("w-48") \
                        .tooltip("Metres, corner 4 to corner 1.")
                    seg4_inp.on_value_change(lambda _: _sync_seg_list_all())
                seg2_block.set_visibility(False)
                seg34_block.set_visibility(False)

                def _on_seg2_override(e):
                    _seg_state["override_seg2"] = bool(e.value)
                    seg2_block.set_visibility(_seg_state["override_seg2"])
                    if not _seg_state["override_seg2"]:
                        _seg_state["override_seg34"] = False
                        seg34_checkbox.value = False
                        seg34_block.set_visibility(False)
                    _sync_seg_list_all()
                seg2_checkbox.on_value_change(_on_seg2_override)

                def _on_seg34_override(e):
                    _seg_state["override_seg34"] = bool(e.value)
                    seg34_block.set_visibility(_seg_state["override_seg34"])
                    _sync_seg_list_all()
                seg34_checkbox.on_value_change(_on_seg34_override)
            _sync_seg_list_all()

            # --- Optional lens distortion correction ---
            with ui.expansion("Lens distortion", icon="lens") \
                    .classes("w-full"):
                def _load_calib():
                    p = native_file_picker(
                        "Camera calibration file",
                        filetypes=[("Calibration", "*.json *.yml *.yaml *.npz"),
                                   ("All files", "*.*")],
                        initialdir=default_starting_dir("images"))
                    if not p:
                        return
                    from functions.orthorectify import parse_calibration_file
                    parsed = parse_calibration_file(p)
                    if not parsed:
                        ui.notify("Could not read a camera matrix + distortion "
                                  "coefficients from that file. Pick an OpenCV "
                                  ".json/.yml/.npz (Inputs, *Lens distortion* "
                                  "above).", type="negative")
                        return
                    fx, fy, cx, cy, dist = parsed
                    state.ortho_fx, state.ortho_fy = fx, fy
                    state.ortho_cx, state.ortho_cy = cx, cy
                    d = list(dist) + [0.0] * 5
                    (state.ortho_k1, state.ortho_k2, state.ortho_p1,
                     state.ortho_p2, state.ortho_k3) = d[0], d[1], d[2], d[3], d[4]
                    state.ortho_use_distortion = True
                    ui.notify(f"Loaded calibration from {Path(p).name}.",
                              type="positive")

                with ui.row().classes("items-center gap-3"):
                    ui.button("Load calibration file…", icon="upload_file",
                              on_click=_load_calib).props("outline dense") \
                        .tooltip("OpenCV calibration (.json, .yml/.yaml, .npz): "
                                 "intrinsics + distortion coefficients. The "
                                 "photograph is undistorted and the corners "
                                 "corrected as part of the rectification.")
                    ui.checkbox("Apply distortion correction") \
                        .bind_value(state, "ortho_use_distortion")
                with ui.row().classes("items-end gap-2 flex-wrap"):
                    for lbl, attr in [("fx", "ortho_fx"), ("fy", "ortho_fy"),
                                      ("cx", "ortho_cx"), ("cy", "ortho_cy")]:
                        ui.number(label=lbl, format="%.3f") \
                            .bind_value(state, attr).classes("w-28")
                with ui.row().classes("items-end gap-2 flex-wrap"):
                    for lbl, attr in [("k1", "ortho_k1"), ("k2", "ortho_k2"),
                                      ("p1", "ortho_p1"), ("p2", "ortho_p2"),
                                      ("k3", "ortho_k3")]:
                        ui.number(label=lbl, format="%.5f") \
                            .bind_value(state, attr).classes("w-28")

            # --- Output: GSD and file, derived unless overridden ---
            with ui.expansion("Output", icon="save").classes("w-full"):
                with ui.row().classes("w-full items-end gap-3"):
                    ui.number(label="Output GSD (m/pixel)",
                              value=0.0, step=0.0001, min=0.0, format="%.5f") \
                        .bind_value(state, "ortho_gsd_m") \
                        .classes("w-48") \
                        .tooltip("Ground sample distance of the rectified "
                                 "output. 0 = auto (finest source-pixel "
                                 "resolution).")
                    ui.input(label="Output file (.png/.jpg)") \
                        .bind_value(state, "ortho_out_path") \
                        .classes("flex-grow") \
                        .props('hint="A _GSD=<value>m suffix is added at save; '
                               'after rectifying, this shows the path written."') \
                        .tooltip("Where to save the rectified image. The "
                                 "writer appends the computed GSD to the name; "
                                 "the field updates to the real path once "
                                 "written.")

                    def _browse_save():
                        current = state.ortho_out_path
                        if current:
                            initial = Path(current).name
                            initial_dir = str(Path(current).parent)
                        elif state.ortho_src_path:
                            _d = Path(_ortho_default_out_for(
                                state.ortho_src_path))
                            initial, initial_dir = _d.name, str(_d.parent)
                        else:
                            initial = "rectified.jpg"
                            initial_dir = default_starting_dir("images")
                        picked = native_save_file_picker(
                            title="Save rectified image as",
                            filetypes=[("PNG image", "*.png"),
                                       ("JPEG image", "*.jpg *.jpeg"),
                                       ("All files", "*.*")],
                            initialfile=initial,
                            defaultextension=".png",
                            initialdir=initial_dir,
                        )
                        if picked:
                            state.ortho_out_path = picked
                    ui.button("Browse…", icon="folder_open",
                              on_click=_browse_save).props("outline")

            ui.label("Sizes in m. The GSD and the output name derive from "
                     "the size; corners and sizes are saved beside each "
                     "photograph.").classes("text-xs text-grey-7")

    # --- Auto-fill the output path when the source changes ---
    def _autofill_output():
        if (not state.ortho_out_path) and state.ortho_src_path:
            state.ortho_out_path = _ortho_default_out_for(state.ortho_src_path)
    ui.timer(1.0, _autofill_output)

    # --- Band B: the work surface (toolbar, status, photograph) ----------- #
    img_widgets = {
        "interactive": None,
        "loupe": None,
        "info_label": None,
        "natural_size": (0, 0),   # (W, H) in image pixels
        "src_url": None,
        "box": None,
    }

    def _ortho_box_px():
        """The surface box is 70 % of the viewport, in pixels."""
        return 0.70 * float(state.ortho_viewport_h or 900)

    def _ortho_fit_px():
        """Drawn width at zoom 1: the whole photograph inside the box."""
        W, H = img_widgets["natural_size"]
        return _or_mod.fit_width(W or 0, H or 0, _ortho_box_px())

    def _ortho_display_px():
        """How wide the photograph is being drawn, in screen pixels."""
        return max(1.0, _ortho_fit_px() * float(state.ortho_zoom or 1.0))

    def _push_corners_to_loupe():
        """The client-side magnifier draws the placed corners; publish
        them whenever they change."""
        import json as _json
        _run_js(
            f"window.__ortho_corners = "
            f"{_json.dumps([[c[0], c[1]] for c in state.ortho_corners])}; "
            f"window.__ortho_editing = {int(state.ortho_click_idx)};")

    def _update_overlay():
        """Rebuild the SVG overlay (corners + polygon) over the image.

        Drawn in image coordinates, so marker sizes are scaled to stay
        constant on screen.
        """
        _push_corners_to_loupe()
        if img_widgets["interactive"] is None:
            return
        from functions import orthorectify as _og
        W = img_widgets["natural_size"][0] or 1
        disp = _ortho_display_px()
        k = _og.display_scale(W, disp)          # image px per screen px
        r_on = _og.marker_radius(W, disp)
        parts = []
        # Guide: where the last confirmed quadrat was (dashed, no handles).
        if state.ortho_guide_shown and state.ortho_guide and \
                not state.ortho_corners:
            _gname, _gq = state.ortho_guide
            _d = "M " + " L ".join(f"{a} {b}" for a, b in _gq) + " Z"
            parts.append(
                f'<path d="{_d}" fill="none" stroke="#3b82f6" '
                f'stroke-width="{2.0 * k:.1f}" '
                f'stroke-dasharray="{10 * k:.0f},{8 * k:.0f}" '
                f'opacity="0.85"/>')
            for _i, (_a, _b) in enumerate(_gq):
                parts.append(
                    f'<text x="{_a + 6 * k:.0f}" y="{_b - 6 * k:.0f}" '
                    f'font-size="{13 * k:.0f}" fill="#1d4ed8" stroke="white" '
                    f'stroke-width="{3 * k:.0f}" paint-order="stroke">'
                    f'{_i + 1}</text>')
        if len(state.ortho_corners) >= 2:
            d = "M " + " L ".join(f"{c[0]} {c[1]}" for c in state.ortho_corners)
            if len(state.ortho_corners) == 4:
                d += " Z"
            parts.append(
                f'<path id="pm-frame" d="{d}" fill="rgba(255,200,0,0.15)" '
                f'stroke="orange" stroke-width="3"/>'
            )
        # The ids (pm-corner-i, pm-label-i, pm-frame) are what the
        # client-side drag moves between press and release.
        for i, (cx, cy) in enumerate(state.ortho_corners):
            editing = (i == state.ortho_click_idx)
            parts.append(
                f'<circle id="pm-corner-{i}" cx="{cx}" cy="{cy}" '
                f'r="{r_on * (1.35 if editing else 1.0):.1f}" '
                f'fill="{"#ffd000" if editing else "orange"}" '
                f'stroke="{"#0057ff" if editing else "black"}" '
                f'stroke-width="{(4 if editing else 2) * k:.1f}"/>'
            )
            parts.append(
                f'<text id="pm-label-{i}" x="{cx + 14 * k:.0f}" '
                f'y="{cy + 6 * k:.0f}" '
                f'font-size="{20 * k:.0f}" fill="black" stroke="white" '
                f'stroke-width="{3 * k:.0f}" paint-order="stroke">'
                f'{i + 1}</text>'
            )
        img_widgets["interactive"].set_content("".join(parts))

    def _update_info():
        n = len(state.ortho_corners)
        next_idx = state.ortho_click_idx
        if n < 4:
            msg = (f"{n} of 4 corners — click corner #{next_idx + 1}; "
                   f"the magnifier follows the cursor")
        else:
            msg = (f"4 of 4 corners — click one to move it (#{next_idx + 1} "
                   f"active), then Rectify")
        if img_widgets["info_label"] is not None:
            img_widgets["info_label"].set_text(msg)

    # Drag in progress: corner index, moved flag, last redraw (~30/s cap).
    _drag = {"idx": None, "moved": False, "last": 0.0}

    def _end_drag():
        dragged = _drag["idx"]
        _drag["idx"] = None
        _drag["moved"] = False
        state.ortho_corner_armed = False
        n = len(state.ortho_corners)
        if n < 4:
            # Moving is not placing: the next click places the next corner.
            state.ortho_click_idx = n
        elif dragged is not None:
            # All four placed: the corner just moved stays the active one.
            state.ortho_click_idx = dragged
        # A moved corner is a changed pick: it reaches disk now.
        if len(state.ortho_corners) == 4:
            _save_ortho_corners(state.ortho_src_path, state.ortho_corners,
                                _effective_segments(),
                                state.ortho_frame_thickness_m)
        _ortho_update_nav_summary()
        _update_overlay()
        _update_info()

    def _on_mouse(e):
        """Corners are placed by a click and moved by a drag.

        mousedown on a corner arms a drag; mouseup ends it and saves. A
        press-and-release that did not move is a click: on a corner it
        selects it, elsewhere it places the next corner.
        """
        from functions import orthorectify as _or
        import time as _time
        t = e.type
        if t == "mousemove":
            if _drag["idx"] is None:
                return
            i = _drag["idx"]
            if 0 <= i < len(state.ortho_corners):
                state.ortho_corners[i] = [round(e.image_x), round(e.image_y)]
                _drag["moved"] = True
                now = _time.monotonic()
                if now - _drag["last"] > 0.03:
                    _drag["last"] = now
                    _update_overlay()
            return
        if t == "mouseleave":
            if _drag["idx"] is not None and _drag["moved"]:
                _end_drag()
            else:
                _drag["idx"] = None
            return
        if t not in ("mousedown", "mouseup"):
            return
        x = round(e.image_x)
        y = round(e.image_y)
        if t == "mousedown":
            # Hit radius in screen pixels, whatever the photograph's size.
            hit = _or.corner_under(
                state.ortho_corners, x, y,
                _or.hit_radius(img_widgets["natural_size"][0] or 1,
                               _ortho_display_px()))
            _drag["idx"] = hit
            _drag["moved"] = False
            _drag["down"] = (x, y)
            if hit is not None:
                state.ortho_click_idx = hit
                state.ortho_corner_armed = True
                _update_overlay()
                _update_info()
            return

        # mouseup. The drag is client-side (a round trip per mousemove
        # lags): the browser sends one mouseup with the final position.
        # A release away from the press is a move; on the spot, a click.
        if _drag["idx"] is not None:
            i = _drag["idx"]
            dx, dy = _drag.get("down") or (x, y)
            if (_drag["moved"] or abs(x - dx) > 1 or abs(y - dy) > 1) \
                    and 0 <= i < len(state.ortho_corners):
                state.ortho_corners[i] = [x, y]
                _drag["moved"] = True
                _end_drag()
            else:
                _drag["idx"] = None      # a click on a corner: selected only
            return

        # All four placed and none chosen: a stray click must not replace
        # corner 1 (corners reach disk as soon as they change).
        if len(state.ortho_corners) >= 4 and not state.ortho_corner_armed:
            ui.notify("All four corners are placed. Drag the one you want "
                      "to move, or click it first (on the image, Work "
                      "surface).", type="info")
            return

        state.ortho_corners, state.ortho_click_idx = _or.place_corner(
            state.ortho_corners, state.ortho_click_idx, x, y)
        # Four corners is finished manual work: persist now, not at Rectify.
        state.ortho_corner_armed = False
        if len(state.ortho_corners) == 4:
            _save_ortho_corners(state.ortho_src_path,
                                state.ortho_corners,
                                _effective_segments(),
                                state.ortho_frame_thickness_m)
        _ortho_update_nav_summary()
        _update_overlay()
        _update_info()

    def _load_image():
        """Read the source image, set up the interactive widget + loupe."""
        if not state.ortho_src_path or not Path(state.ortho_src_path).exists():
            ui.notify("Set *Folder of photographs* (Inputs, above), or a "
                      "*Source photograph* in the expander below it.",
                      type="warning")
            return
        try:
            from PIL import Image
            with Image.open(state.ortho_src_path) as pim:
                W, H = pim.size
        except Exception as e:
            ui.notify(f"Could not read image: {e}"
                      f"{_images.heif_failure_hint(state.ortho_src_path)}. "
                      "Pick another photograph (Image, bar above).",
                      type="negative")
            return

        img_widgets["natural_size"] = (W, H)
        _saved_corners = _load_ortho_corners(state.ortho_src_path)
        state.ortho_corners = _saved_corners
        # % 4: a full set must not leave click_idx outside 0..3.
        state.ortho_click_idx = len(_saved_corners) % 4
        state.ortho_corner_armed = False
        # Guide from the recorded corners only; no photograph is opened.
        try:
            from functions import orthorectify as _og2
            _rows = [(nm, (state.ortho_per_image.get(nm) or {}).get("corners")
                      or _load_ortho_corners(
                          os.path.join(state.ortho_image_dir, nm)))
                     for nm in state.ortho_image_list]
            state.ortho_guide = _og2.guide_quadrilateral(
                _rows, state.ortho_image_idx)
        except Exception:
            state.ortho_guide = None
        state.ortho_guide_is_suggestion = False
        _saved_thick = _load_ortho_frame_thickness(state.ortho_src_path)
        if _saved_thick is not None:
            state.ortho_frame_thickness_m = _saved_thick
        _saved_seg = _load_ortho_segments(state.ortho_src_path)
        if _saved_seg:
            (state.ortho_seg_1, state.ortho_seg_2,
             state.ortho_seg_3, state.ortho_seg_4) = _saved_seg
            state.ortho_seg_lengths = list(_saved_seg)
        if _saved_corners:
            ui.notify(
                f"Loaded {len(_saved_corners)} saved corner(s) from "
                f"{_ortho_corners_path(state.ortho_src_path).name} — "
                f"nudge them and Rectify to re-save.", type="info")

        # Static-serve the folder so the loupe can fetch the full-res image.
        from nicegui import app
        import hashlib
        served = Path(state.ortho_src_path)
        # Show the stored pixel frame, the frame the corners are picked in
        # and the rectifier reads: a HEIC (not browser-decodable) or a JPEG
        # with an EXIF rotation tag (a browser would turn it) is served as an
        # untagged copy of that frame.
        try:
            from functions.layout import app_home as _app_home
            served = Path(_images.display_source(
                served, Path(_app_home()) / "cache" / "ortho_display"))
        except Exception as _ex:
            ui.notify(f"Could not prepare {served.name} for display: "
                      f"{_ex}", type="negative", timeout=10000)
        src_dir = str(served.parent)
        mount_name = "ortho_" + hashlib.md5(src_dir.encode()).hexdigest()[:8]
        try:
            app.add_static_files(f"/{mount_name}", src_dir)
        except Exception:
            pass  # already mounted, fine
        url = f"/{mount_name}/{served.name}"
        img_widgets["src_url"] = url

        # The photograph in a scroller capped at 70 % of the viewport.
        interactive_container.clear()
        with interactive_container:
            scroller = ui.element("div").classes("w-full pm-ortho-box") \
                .style("overflow:auto; max-height:70vh; "
                       "border:1px solid #dfe3e8;")
            with scroller:
                # No mousemove to the server: the drag is drawn client-side.
                interactive = ui.interactive_image(
                    source=url,
                    events=["mousedown", "mouseup", "mouseleave"],
                    cross=True,
                    on_mouse=_on_mouse,
                ).classes("ortho-image")
            img_widgets["interactive"] = interactive
            img_widgets["box"] = scroller
            _apply_ortho_zoom()
            ui.label(f"Source: {W} × {H} px") \
                .classes("text-xs text-grey-7 mt-1")
            img_widgets["loupe"] = ui.html(
                f'<div id="ortho-loupe-wrapper" '
                f'style="position:fixed; pointer-events:none; z-index:1000; '
                f'display:none; left:0; top:0;">'
                f'<canvas id="ortho-loupe-canvas" width="200" height="200" '
                f'style="display:block; border:2px solid #5a6878; '
                f'background:#222; image-rendering:pixelated; '
                f'box-shadow:0 2px 8px rgba(0,0,0,0.4);"></canvas>'
                f'<div style="background:rgba(0,0,0,0.7); color:white; '
                f'font-size:10px; text-align:center; padding:2px;">'
                f'Magnifier (4×)</div>'
                f'</div>'
                f'<img id="ortho-loupe-source-img" src="{url}" '
                f'style="display:none;" crossorigin="anonymous" />'
            )
            _run_js(
                f"""
                (function() {{
                    var WRAP_W = 204, WRAP_H = 218;
                    // k = image px per screen px as drawn; the loupe magnifies
                    // what is on screen 4x (a 50-screen-px window at any zoom).
                    window.__loupe_draw = function(cx, cy, k) {{
                        var c = document.getElementById('ortho-loupe-canvas');
                        var img = document.getElementById('ortho-loupe-source-img');
                        if (!c || !img || !img.complete || img.naturalWidth === 0) return;
                        var ctx = c.getContext('2d');
                        k = (k && k > 0) ? k : 1;
                        ctx.imageSmoothingEnabled = (k > 1);
                        ctx.fillStyle = '#222';
                        ctx.fillRect(0, 0, 200, 200);
                        var ZOOM = 4 / k;
                        var srcW = 200 / ZOOM, srcH = 200 / ZOOM;
                        ctx.drawImage(img, cx - srcW/2, cy - srcH/2, srcW, srcH, 0, 0, 200, 200);
                        // Placed corners and their polygon; mirrors
                        // functions.orthorectify.loupe_marks.
                        var pts = window.__ortho_corners || [];
                        var ed = window.__ortho_editing;
                        var ox = cx - srcW/2, oy = cy - srcH/2;
                        var L = pts.map(function(p) {{
                            return [(p[0] - ox) * ZOOM, (p[1] - oy) * ZOOM]; }});
                        if (L.length >= 2) {{
                            ctx.strokeStyle = 'orange'; ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            for (var i = 0; i < L.length - 1; i++) {{
                                ctx.moveTo(L[i][0], L[i][1]); ctx.lineTo(L[i+1][0], L[i+1][1]); }}
                            if (L.length === 4) {{ ctx.moveTo(L[3][0], L[3][1]); ctx.lineTo(L[0][0], L[0][1]); }}
                            ctx.stroke();
                        }}
                        L.forEach(function(q, i) {{
                            if (q[0] < 0 || q[0] > 200 || q[1] < 0 || q[1] > 200) return;
                            var e = (i === ed);
                            ctx.beginPath(); ctx.arc(q[0], q[1], e ? 7 : 5, 0, 2 * Math.PI);
                            ctx.fillStyle = e ? '#ffd000' : 'orange'; ctx.fill();
                            ctx.strokeStyle = e ? '#0057ff' : 'black'; ctx.lineWidth = 2; ctx.stroke();
                            ctx.font = 'bold 12px sans-serif'; ctx.lineWidth = 3;
                            ctx.strokeStyle = 'white'; ctx.strokeText(String(i + 1), q[0] + 8, q[1] - 6);
                            ctx.fillStyle = 'black'; ctx.fillText(String(i + 1), q[0] + 8, q[1] - 6);
                        }});
                        ctx.strokeStyle = 'red';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(100, 90); ctx.lineTo(100, 110);
                        ctx.moveTo(90, 100); ctx.lineTo(110, 100);
                        ctx.stroke();
                    }};
                    // position:fixed is viewport-relative, so use clientX/Y.
                    window.__loupe_position = function(clientX, clientY) {{
                        var wrap = document.getElementById('ortho-loupe-wrapper');
                        if (!wrap) return;
                        var OFFSET = 16;
                        var x = clientX + OFFSET, y = clientY + OFFSET;
                        if (x + WRAP_W > window.innerWidth)  x = clientX - WRAP_W - OFFSET;
                        if (y + WRAP_H > window.innerHeight) y = clientY - WRAP_H - OFFSET;
                        if (x < 4) x = 4;
                        if (y < 4) y = 4;
                        wrap.style.left = x + 'px';
                        wrap.style.top  = y + 'px';
                    }};
                    // Move the wrapper to document.body so ancestor
                    // transforms can't re-anchor position:fixed.
                    function ensureOrthoWrapper() {{
                        var w = document.getElementById('ortho-loupe-wrapper');
                        if (w && w.parentElement !== document.body) {{
                            document.body.appendChild(w);
                        }}
                    }}
                    ensureOrthoWrapper();
                    function hookOrthoProbe() {{
                        var probe = document.querySelector('.ortho-image img')
                                    || document.querySelector('img[src*="/ortho_"]');
                        if (!probe) return false;
                        if (probe.__ortho_hooked) return true;
                        probe.__ortho_hooked = true;
                        var wrap = document.getElementById('ortho-loupe-wrapper');
                        probe.addEventListener('mouseenter', function() {{
                            ensureOrthoWrapper();
                            if (wrap) wrap.style.display = 'block';
                        }});
                        probe.addEventListener('mouseleave', function() {{
                            if (wrap) wrap.style.display = 'none';
                        }});
                        probe.addEventListener('mousemove', function(e) {{
                            var rect = probe.getBoundingClientRect();
                            var img_x = (e.clientX - rect.left) * probe.naturalWidth  / rect.width;
                            var img_y = (e.clientY - rect.top)  * probe.naturalHeight / rect.height;
                            window.__loupe_draw(img_x, img_y, probe.naturalWidth / rect.width);
                            window.__loupe_position(e.clientX, e.clientY);
                        }});
                        // Client-side drag: marker, label and frame follow the
                        // cursor; the server redraws from the one mouseup.
                        // Leaving the photograph mid-drag releases in place.
                        var drag = null;
                        function imgXY(e) {{
                            var rect = probe.getBoundingClientRect();
                            return [(e.clientX - rect.left) * probe.naturalWidth / rect.width,
                                    (e.clientY - rect.top) * probe.naturalHeight / rect.height,
                                    probe.naturalWidth / rect.width];
                        }}
                        probe.addEventListener('mousedown', function(e) {{
                            if (e.button !== 0) return;
                            var pts = window.__ortho_corners || [];
                            var p = imgXY(e), k = p[2];
                            var R = Math.max(4, 24 * k);   // functions.orthorectify.hit_radius
                            var best = -1, bd = 1e12;
                            pts.forEach(function(q, i) {{
                                var d = Math.hypot(q[0] - p[0], q[1] - p[1]);
                                if (d <= R && d < bd) {{ best = i; bd = d; }}
                            }});
                            drag = best >= 0 ? {{ i: best, k: k, x: e.clientX, y: e.clientY }} : null;
                        }});
                        probe.addEventListener('mousemove', function(e) {{
                            if (!drag) return;
                            var p = imgXY(e), x = Math.round(p[0]), y = Math.round(p[1]);
                            var pts = window.__ortho_corners || [];
                            if (drag.i >= pts.length) return;
                            pts[drag.i] = [x, y];
                            drag.x = e.clientX; drag.y = e.clientY;
                            var c = document.getElementById('pm-corner-' + drag.i);
                            if (c) {{ c.setAttribute('cx', x); c.setAttribute('cy', y); }}
                            var t = document.getElementById('pm-label-' + drag.i);
                            if (t) {{ t.setAttribute('x', Math.round(x + 14 * drag.k));
                                     t.setAttribute('y', Math.round(y + 6 * drag.k)); }}
                            var f = document.getElementById('pm-frame');
                            if (f && pts.length >= 2) {{
                                var d = 'M ' + pts.map(function(q) {{ return q[0] + ' ' + q[1]; }}).join(' L ');
                                if (pts.length === 4) d += ' Z';
                                f.setAttribute('d', d);
                            }}
                        }});
                        probe.addEventListener('mouseup', function() {{ drag = null; }});
                        probe.addEventListener('mouseleave', function() {{
                            if (!drag) return;
                            var x = drag.x, y = drag.y;
                            drag = null;
                            probe.dispatchEvent(new MouseEvent('mouseup', {{
                                bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }}));
                        }});
                        return true;
                    }}
                    if (!hookOrthoProbe()) {{
                        setTimeout(hookOrthoProbe, 100);
                        setTimeout(hookOrthoProbe, 500);
                    }}
                    var src_img = document.getElementById('ortho-loupe-source-img');
                    var center = function() {{ window.__loupe_draw({W//2}, {H//2}); }};
                    if (src_img && src_img.complete && src_img.naturalWidth > 0) center();
                    else if (src_img) src_img.addEventListener('load', center);
                }})();
                """
            )
        _update_overlay()
        _update_info()

    def _load_outside_photograph():
        """A path typed or browsed into *Source photograph* loads at once."""
        p = state.ortho_src_path
        if not p or not Path(p).is_file():
            return
        if state.ortho_image_list and 0 <= state.ortho_image_idx < \
                len(state.ortho_image_list) and \
                Path(p) == Path(os.path.join(
                    state.ortho_image_dir,
                    state.ortho_image_list[state.ortho_image_idx])):
            return  # it is the folder's own current photograph
        state.ortho_out_path = ""
        _load_image()
        _ortho_update_nav_summary()
        ortho_nav_summary.set_text(f"Outside the folder — {Path(p).name}")

    def _apply_ortho_zoom():
        """Set the drawn width, then redraw the overlay at the new scale."""
        el = img_widgets.get("interactive")
        if el is None:
            return
        try:
            el.style(f"width:{_ortho_display_px():.0f}px; max-width:none;")
        except Exception:
            return
        _update_overlay()

    def _ortho_zoom_to(z):
        state.ortho_zoom = max(0.25, min(8.0, float(z)))
        _apply_ortho_zoom()

    def _ortho_zoom_1to1():
        W = img_widgets["natural_size"][0] or 0
        fit = _ortho_fit_px()
        _ortho_zoom_to(max(1.0, W / fit) if W and fit else 1.0)

    # Viewport height, emitted by the _PM_LAYOUT_CSS head script at load and
    # on resize. An event, not an awaited request: a throttled client delays
    # it instead of losing it; the 900 px default stands until it arrives.
    def _on_viewport(e):
        try:
            h = int((e.args or {}).get("h") or 0)
        except (TypeError, ValueError, AttributeError):
            return
        if h > 200 and h != state.ortho_viewport_h:
            state.ortho_viewport_h = h
            _apply_ortho_zoom()
    try:
        ui.on("pm_viewport", _on_viewport)
    except Exception:
        pass  # no client context (an off-screen build): the default stands

    async def _ortho_suggest_corners():
        """Offer a starting position for the corners, or say why it will not.

        The detection runs off the event loop (``run.io_bound``, a few
        seconds) so corners can still be placed while it thinks.
        """
        if not state.ortho_src_path:
            ui.notify("Select a photograph in the toolbar (Work surface), or "
                      "set *Folder of photographs* (Inputs, above).",
                      type="warning")
            return
        thick = state.ortho_frame_thickness_m
        if not thick or thick <= 0:
            ui.notify("Set *Frame thickness* (Inputs, above) first — the "
                      "frame cannot be told from the ground without it.",
                      type="warning")
            return
        # Exemplar: the nearest earlier photograph with hand-placed corners.
        ex_name, ex_corners = None, None
        try:
            here = state.ortho_image_list.index(
                os.path.basename(state.ortho_src_path))
        except Exception:
            here = len(state.ortho_image_list)
        for nm in reversed(state.ortho_image_list[:here]):
            got = (state.ortho_per_image.get(nm) or {}).get("corners") \
                or _load_ortho_corners(
                    os.path.join(state.ortho_image_dir, nm))
            if got and len(got) == 4:
                ex_name, ex_corners = nm, got
                break
        if ex_corners is None:
            ui.notify("No earlier photograph in this folder (photo table, "
                      "Inputs, above) has corners placed yet. Rectify one by "
                      "hand first, then this can carry it forward.",
                      type="info")
            return
        # Everything the worker needs is read here, on the event loop.
        from nicegui import run as _run
        _ex_path = os.path.join(state.ortho_image_dir, ex_name)
        _tg_path = state.ortho_src_path
        _side_m = float(max(state.ortho_seg_lengths or [1.0]) or 1.0)
        _thick = float(thick)

        def _work():
            import numpy as _np
            from PIL import Image as _Im
            from functions import quadrat_detect as _qd
            with _Im.open(_ex_path) as _e:
                ex_img = _np.asarray(_e.convert("RGB"))
            with _Im.open(_tg_path) as _t:
                tg_img = _np.asarray(_t.convert("RGB"))
            exm = _qd.build_exemplar(
                ex_img, _np.asarray(ex_corners, dtype=float),
                frame_thickness_m=_thick, quadrat_side_m=_side_m)
            return _qd.propose(tg_img, exm)

        ui.notify("Looking for the frame — a few seconds. Carry on placing "
                  "corners; this will not interrupt you.", type="info",
                  timeout=4000)
        try:
            prop = await _run.io_bound(_work)
        except Exception as exc:                    # noqa: BLE001
            ui.notify(f"Could not suggest a position "
                      f"({type(exc).__name__}: {exc}). Place the corners "
                      f"by hand — click them on the image (Work surface).", type="negative")
            return
        # A result for a photograph since left, or since hand-placed, is dropped.
        if state.ortho_src_path != _tg_path:
            ui.notify("You moved to another photograph, so the suggestion "
                      "was discarded. Press it again here if you want one.",
                      type="info")
            return
        if len(state.ortho_corners or []) == 4:
            ui.notify("You placed the four corners while it was looking, "
                      "so the suggestion was discarded — yours are better.",
                      type="info")
            return
        if not prop.accepted:
            state.ortho_guide = None
            state.ortho_guide_is_suggestion = False
            _update_overlay()
            ui.notify(prop.reasons[0] if prop.reasons
                      else "No trustworthy position was found. Click the "
                           "corners on the photograph (Work surface).",
                      type="warning", timeout=9000)
            return
        # The suggestion is placed as four draggable corners, recorded with
        # source=suggested; a corner dragged afterwards is saved as the user's.
        pts = [[float(x), float(y)] for x, y in prop.corners][:4]
        if len(pts) != 4:
            ui.notify("The suggestion is incomplete and was not placed. "
                      "Click the four corners on the image yourself (Work surface).", type="negative")
            return
        state.ortho_corners = pts
        state.ortho_click_idx = 0
        state.ortho_corner_armed = False
        state.ortho_guide = None
        state.ortho_guide_is_suggestion = False
        _save_ortho_corners(state.ortho_src_path, state.ortho_corners,
                            _effective_segments(),
                            state.ortho_frame_thickness_m,
                            source="suggested")
        _update_overlay()
        _update_info()
        _ortho_update_nav_summary()
        ui.notify("Corners placed from the suggestion and recorded as "
                  "suggested (good to a few per cent of a side). Drag any "
                  "corner to refine it; Reset corners clears them.",
                  type="positive", timeout=9000)

    def _reset_picks():
        state.ortho_corners = []
        state.ortho_click_idx = 0
        # The saved picks go too: with the file kept, reopening the
        # photograph brought the four corners back.
        src = state.ortho_src_path
        if src:
            try:
                f = _ortho_corners_path(src)
                if f.exists():
                    f.unlink()
                _forget_sidecar(f)
            except Exception:
                pass
            entry = state.ortho_per_image.get(Path(src).name)
            if isinstance(entry, dict):
                entry.pop("corners", None)
        _update_overlay()
        _update_info()
        try:
            _ortho_refresh_table()
        except Exception:
            pass

    # --- The toolbar, attached to the surface and sticky ---
    with ortho_band_b:
        ortho_toolbar = ui.row() \
            .classes("w-full items-center gap-1 flex-nowrap pm-sticky-toolbar")
        with ortho_toolbar:
            @ui.refreshable
            def _ortho_image_picker():
                # Rebuilt with options and value together, and not bound to
                # the state: a select whose options lack the value sets
                # itself to None, and a binding would write that None back.
                files = list(state.ortho_image_list)
                current = state.ortho_selected_image
                ui.select(
                    {f: f for f in files},
                    label="Image",
                    value=current if current in files else None,
                    on_change=lambda e: _ortho_picker_handlers["switch"](e.value),
                ).props("dense options-dense")                     .classes("min-w-[18rem] max-w-[30rem]")
            _ortho_image_picker()
            ortho_prev_btn = ui.button("Previous", icon="chevron_left",
                                       on_click=lambda: _step_ortho_image(-1)) \
                .props("outline dense")
            ortho_next_btn = ui.button("Next",
                                       on_click=lambda: _step_ortho_image(+1)) \
                .props("outline dense icon-right=chevron_right")
            ui.separator().props("vertical").classes("q-mx-xs")
            ui.button("Fit", on_click=lambda: _ortho_zoom_to(1.0)) \
                .props("flat dense no-caps") \
                .tooltip("The whole photograph inside the box.")
            ui.button("1:1", on_click=_ortho_zoom_1to1) \
                .props("flat dense no-caps") \
                .tooltip("One image pixel per screen pixel — the only view "
                         "in which a corner on a 4000 px photograph can be "
                         "judged.")
            ui.button("", icon="zoom_in",
                      on_click=lambda: _ortho_zoom_to(state.ortho_zoom * 1.6)) \
                .props("flat dense")
            ui.button("", icon="zoom_out",
                      on_click=lambda: _ortho_zoom_to(state.ortho_zoom / 1.6)) \
                .props("flat dense")
            ui.separator().props("vertical").classes("q-mx-xs")
            ui.checkbox("Guide") \
                .bind_value(state, "ortho_guide_shown") \
                .on_value_change(lambda _: _update_overlay()) \
                .bind_visibility_from(state, "ortho_guide",
                                      backward=lambda g: bool(g)) \
                .props("dense") \
                .tooltip("The previous quadrat's outline, dashed, with its "
                         "corner numbers — a guide only, never placed or "
                         "saved. It tells you which physical corner is #1 "
                         "and which way the numbering runs.")
            ui.button("Suggest a position", icon="auto_fix_high",
                      on_click=_ortho_suggest_corners) \
                .props("flat dense no-caps") \
                .tooltip("Looks for this quadrat's frame using a photograph "
                         "already rectified in this folder and places the "
                         "four corners where it thinks the frame is, "
                         "recorded as suggested. Silent unless two "
                         "independent checks agree; accurate to a few per "
                         "cent of a side, so drag the corners onto the frame "
                         "for the GSD. Reset corners clears them.")
            ui.button("Reset corners", icon="restart_alt",
                      on_click=_reset_picks) \
                .props("outline dense no-caps")
        # Rectify is created below, after _do_rectify, and moved in last.

        # Status line: which photograph, how far the pick is, the guide.
        with ui.row().classes("w-full items-center gap-3 text-xs flex-nowrap "
                              "pm-ortho-status"):
            ortho_nav_summary = ui.label("").classes(
                "text-xs text-grey-7 pm-ortho-status")
            img_widgets["info_label"] = ui.label("").classes(
                "text-xs font-mono px-2 pm-ortho-status") \
                .style("background-color: #fffbe6;")
            ui.label("").bind_text_from(
                state, "ortho_guide",
                backward=lambda g: (f"guide from {g[0]}" if g else "")) \
                .classes("text-xs text-grey-7 pm-ortho-status")
        interactive_container = ui.column().classes("w-full")

    # --- Band C: results, appended below the surface --------------------- #
    with ortho_band_c:
        log_widget = live_log("orthorectify", build_log_console(height="h-32"))
        preview_label = ui.label("").classes(
            "text-sm font-bold mt-2 pm-ortho-preview-label")
        preview_container = ui.column().classes("w-full pm-ortho-preview")

    # --- Multi-file helpers ------------------------------------------------
    def _ortho_snapshot_current():
        """Cache the active photo's edit context so switching back restores it."""
        if not (0 <= state.ortho_image_idx < len(state.ortho_image_list)):
            return
        fname = state.ortho_image_list[state.ortho_image_idx]
        prev = state.ortho_per_image.get(fname) or {}
        state.ortho_per_image[fname] = {
            "corners": [list(c) for c in state.ortho_corners],
            "click_idx": state.ortho_click_idx,
            "seg": [state.ortho_seg_1, state.ortho_seg_2,
                    state.ortho_seg_3, state.ortho_seg_4],
            "override_seg2": _seg_state["override_seg2"],
            "override_seg34": _seg_state["override_seg34"],
            "edge_mode": state.ortho_edge_mode,
            "gsd_m": state.ortho_gsd_m,
            "out_path": state.ortho_out_path,
            # Carried, so switching away and back does not re-arm or disarm
            # the scale warning.
            "seg_confirmed": bool(prev.get("seg_confirmed")),
        }

    def _ortho_default_out_for(image_path: str) -> str:
        """An `orthorectified/` folder beside the raw ones, as a JPEG (its
        EXIF block is read by everything; a PNG's eXIf chunk is not)."""
        src = Path(image_path)
        folder = src.parent.parent / "orthorectified" \
            if src.parent.name.lower() in ("raw", "raws", "original",
                                           "originals", "source", "sources") \
            else src.parent / "orthorectified"
        return str(folder / (src.stem + "_rectified.jpg"))

    def _ortho_activate(idx: int):
        """Switch to image #idx, restoring its cached edit context if any."""
        if not (0 <= idx < len(state.ortho_image_list)):
            return
        fname = state.ortho_image_list[idx]
        full_path = os.path.join(state.ortho_image_dir, fname)
        state.ortho_image_idx = idx
        state.ortho_src_path = full_path

        cached = state.ortho_per_image.get(fname)
        if cached is not None:
            state.ortho_corners = [list(c) for c in cached["corners"]]
            state.ortho_click_idx = cached["click_idx"]
            (state.ortho_seg_1, state.ortho_seg_2,
             state.ortho_seg_3, state.ortho_seg_4) = cached["seg"]
            state.ortho_seg_lengths = list(cached["seg"])
            state.ortho_edge_mode = cached["edge_mode"]
            state.ortho_gsd_m = cached["gsd_m"]
            state.ortho_out_path = cached["out_path"] \
                or _ortho_default_out_for(full_path)
            seg2_checkbox.value = bool(cached.get("override_seg2"))
            seg34_checkbox.value = bool(cached.get("override_seg34"))
        else:
            state.ortho_corners = []
            state.ortho_click_idx = 0
            # Segment lengths carry over: the frame is the same equipment.
            keep = [float(x) for x in (state.ortho_seg_lengths
                                       or [1.0, 1.0, 1.0, 1.0])]
            while len(keep) < 4:
                keep.append(keep[-1] if keep else 1.0)
            (state.ortho_seg_1, state.ortho_seg_2,
             state.ortho_seg_3, state.ortho_seg_4) = keep[:4]
            state.ortho_seg_lengths = keep[:4]
            state.ortho_out_path = _ortho_default_out_for(full_path)
            seg2_checkbox.value = abs(keep[1] - keep[0]) > 1e-9
            seg34_checkbox.value = (abs(keep[2] - keep[0]) > 1e-9
                                    or abs(keep[3] - keep[1]) > 1e-9)
        # Accepting the default follows the file, not the session.
        from functions import orthorectify as _or2
        seg_confirm_checkbox.value = _or2.seg_confirmed_for(
            state.ortho_per_image, fname)
        try:
            log_widget.clear()
        except Exception:
            pass
        try:
            preview_container.clear()
            preview_label.set_text("")
        except Exception:
            pass
        # The dropdown follows Previous/Next. Always rebuilt: the state is
        # shared and another page may have set it already.
        _ortho_select_sync["quiet"] = True
        try:
            state.ortho_selected_image = fname
            _ortho_image_picker.refresh()
        finally:
            _ortho_select_sync["quiet"] = False
        _load_image()
        _ortho_update_nav_summary()

    def _ortho_drop_image(name):
        """Take one photograph out of the working list. Never off the disk."""
        if not name or name not in state.ortho_image_list:
            return
        i = state.ortho_image_idx
        active = (state.ortho_image_list[i]
                  if 0 <= i < len(state.ortho_image_list) else None)
        idx = state.ortho_image_list.index(name)
        if active == name:
            _ortho_snapshot_current()
        state.ortho_image_list = [f for f in state.ortho_image_list
                                  if f != name]
        if not state.ortho_image_list:
            state.ortho_image_idx = -1
            _ortho_select_sync["quiet"] = True
            try:
                state.ortho_selected_image = ""
                _ortho_image_picker.refresh()
            finally:
                _ortho_select_sync["quiet"] = False
            _ortho_update_nav_summary()
        elif active == name:
            state.ortho_image_idx = -1          # so _ortho_activate proceeds
            _ortho_activate(min(idx, len(state.ortho_image_list) - 1))
        else:
            state.ortho_image_idx = state.ortho_image_list.index(active)
            _ortho_update_nav_summary()
        ui.notify(f"{name} removed from the list. The file itself is "
                  f"untouched — press Browse, or retype the folder, to bring "
                  f"it back.", type="info")

    def _switch_to_ortho_image(value):
        if _ortho_select_sync["quiet"]:
            return
        if not value:
            return
        try:
            idx = state.ortho_image_list.index(value)
        except ValueError:
            return
        if idx == state.ortho_image_idx:
            return
        _ortho_snapshot_current()
        _ortho_activate(idx)

    _ortho_picker_handlers["switch"] = _switch_to_ortho_image

    def _step_ortho_image(delta: int):
        if not state.ortho_image_list:
            return
        new_idx = state.ortho_image_idx + delta
        if not (0 <= new_idx < len(state.ortho_image_list)):
            return
        _ortho_snapshot_current()
        _ortho_activate(new_idx)

    def _activate_after_warm(idx):
        """Open photograph #idx once every sidecar in the folder is cached.

        Hundreds of sidecar reads on the event loop hold it past the
        socket's ping and trigger a reconnect storm, so they happen in a
        worker thread first. Without a running loop it all runs inline.
        """
        d = state.ortho_image_dir
        files = list(state.ortho_image_list)
        n = len(files)
        try:
            ortho_nav_summary.set_text(f"Reading {n} photograph(s)…")
        except Exception:
            pass

        def _warm():
            for f in files:
                full = os.path.join(d, f)
                _sidecar_text(_ortho_corners_path(full))
                _sidecar_text(_ortho_segments_path(full))

        def _finish():
            if not _live() or state.ortho_image_dir != d:
                return
            # A slot, so ui.notify / run_javascript inside find the client.
            with ortho_band_b:
                _ortho_activate(idx)

        async def _go():
            try:
                from nicegui import run as _run
                await _run.io_bound(_warm)
            except Exception:
                pass  # the activation below reads them itself, uncached
            _finish()

        try:
            import asyncio
            asyncio.get_running_loop()
            from nicegui import background_tasks as _bg
            _bg.create(_go(), name="ortho-warm-sidecars")
        except RuntimeError:
            _warm()
            _finish()

    def _refresh_ortho_dir():
        """Scan the folder, fill the picker, load its first photograph."""
        if not _live():
            return
        if not state.ortho_image_dir or not os.path.isdir(state.ortho_image_dir):
            state.ortho_image_list = []
            state.ortho_selected_image = ""
            _ortho_image_picker.refresh()
            state.ortho_image_idx = -1
            _ortho_update_nav_summary()
            return
        files = list_images(state.ortho_image_dir,
                            list(_images.PHOTO_EXTENSIONS))
        state.ortho_image_list = files
        # A new folder starts at its first photograph; the same folder on a
        # page rebuild (the state outlives the page) restores the open one.
        changed = (_ortho_dir_seen["dir"] != state.ortho_image_dir)
        _ortho_dir_seen["dir"] = state.ortho_image_dir
        if files and (changed
                      or not (0 <= state.ortho_image_idx < len(files))):
            state.ortho_image_idx = -1
            _activate_after_warm(0)
        elif files:
            _ortho_image_picker.refresh()
            if img_widgets["interactive"] is None:
                _activate_after_warm(state.ortho_image_idx)
            else:
                _ortho_update_nav_summary()
        else:
            _ortho_image_picker.refresh()
            _ortho_update_nav_summary()

    def _seed_ortho_folder(force=False):
        """Fill the folder from the active project's photographs. Never an
        empty folder, and never over a typed one unless ``force``."""
        from functions import project_defaults as _pdf
        proj = state.current_project
        photos = _pdf.raw_photos(proj) if proj else []
        if not photos:
            return
        folder = photos[0].parent
        if state.ortho_image_dir and not force:
            return
        state.ortho_image_idx = -1
        state.ortho_image_list = []
        # State first, then the widget: with the state already equal, the
        # binding propagation stops there and only this input's handler
        # fires, not every element bound to the field on other pages.
        state.ortho_image_dir = str(folder)
        fired = (str(getattr(ortho_dir_inp, "value", "") or "") != str(folder))
        _seed_default(ortho_dir_inp, folder, force=True)
        if not fired:
            _refresh_ortho_dir()

    def _ortho_update_nav_summary():
        if not state.ortho_image_list:
            ortho_nav_summary.set_text(
                Path(state.ortho_src_path).name if state.ortho_src_path
                else "")
            return
        n = len(state.ortho_image_list)
        i = state.ortho_image_idx
        _rows = _ortho_rows()
        n_done = sum(1 for f in _rows if f["done"])
        active = (state.ortho_image_list[i] if 0 <= i < n else "?")
        ortho_nav_summary.set_text(
            f"Image {i + 1} of {n} — {active} · {n_done}/{n} rectified")
        _ortho_refresh_table(_rows)

    def _ortho_rows():
        """One row per photograph: corners recorded, size, rectified or not.
        Corners come from disk as well as from this session."""
        from functions import orthorectify as _or
        rows = []
        # The output folders are listed once for the whole table (per-row
        # listing is quadratic).
        listing = _or.output_listing(_ortho_default_out_for(
            os.path.join(state.ortho_image_dir,
                         state.ortho_image_list[0]))) \
            if state.ortho_image_list else []
        for f in state.ortho_image_list:
            full = os.path.join(state.ortho_image_dir, f)
            cached = state.ortho_per_image.get(f) or {}
            corners = cached.get("corners") or _load_ortho_corners(full) or []
            # This file's own recorded length, never the session's.
            seg = cached.get("seg") or _load_ortho_segments(full)
            rows.append(_or.folder_row(f, corners, seg,
                                        _ortho_default_out_for(full),
                                        cached.get("gsd_m"), listing))
        return rows

    def _ortho_refresh_table(rows=None):
        rows = _ortho_rows() if rows is None else rows
        ortho_file_table.rows = [
            {"photo": r["photo"], "corners": r["corners"],
             "length": r["length"], "output": r["output"], "drop": ""}
            for r in rows
        ]
        ortho_file_table.update()

    def _scroll_to_preview():
        """Scroll to the result only when it would be off-screen."""
        # Deferred and retried: the image is sent in the same flush as this
        # script and must be laid out before it is measured. If the smooth
        # scroll has not landed after 900 ms, an instant one follows.
        _run_js(
            "(function(){var tries=0;"
            "function pos(){"
            "var lab=document.querySelector('.pm-ortho-preview-label');"
            "if(!lab)return null;"
            "var img=document.querySelector('.pm-ortho-preview img');"
            "var bar=document.querySelector('.pm-sticky-toolbar');"
            "var off=(bar?bar.getBoundingClientRect().height:0)+76;"
            "var r=lab.getBoundingClientRect();"
            "var bottom=img?img.getBoundingClientRect().bottom:r.bottom;"
            "return {top:r.top,bottom:bottom,off:off};}"
            "function go(){"
            "var p=pos();"
            "if(!p){if(tries++<10)setTimeout(go,150);return;}"
            "if(p.top>=p.off&&p.bottom<=window.innerHeight)return;"   # on screen
            "window.scrollTo({top:window.scrollY+p.top-p.off,"
            "behavior:document.hidden?'auto':'smooth'});"   # no frames when hidden
            "setTimeout(function(){"
            "var q=pos();if(!q)return;"
            "if(Math.abs(q.top-q.off)>40)"
            "window.scrollTo({top:window.scrollY+q.top-q.off,behavior:'auto'});"
            "},900);}"
            "setTimeout(go,250);})();")

    def _do_rectify():
        if not state.ortho_src_path or not Path(state.ortho_src_path).exists():
            ui.notify("Select a photograph in the toolbar (Work surface), or "
                      "set *Folder of photographs* (Inputs, above).",
                      type="warning")
            return
        if len(state.ortho_corners) != 4:
            ui.notify(f"Place all 4 corners on the image (Work surface, "
                      f"{len(state.ortho_corners)}/4 placed).",
                      type="warning")
            return
        if not state.ortho_out_path:
            ui.notify("Set *Output file* (Inputs, *Output* expander above).",
                      type="warning")
            return
        # Read the segment boxes now: a value typed without the box losing
        # focus has not reached the on_value_change copy.
        state.ortho_seg_lengths = _effective_segments()

        # Ask once, per file, whether the default is really the frame's size.
        from functions import orthorectify as _or
        _seg_key = Path(state.ortho_src_path).name
        if _or.rectifying_at_the_untouched_default(
                state.ortho_seg_lengths,
                _or.seg_confirmed_for(state.ortho_per_image, _seg_key)):
            log_widget.push(
                f"⚠ Rectifying at the default {_or.DEFAULT_SEG_M:.3f} m "
                "segment length. If the quadrat frame is a different size, "
                "stop and set it — every output will be scaled wrong and no "
                "quadrat will place in an ortho.")
            ui.notify(
                f"Using the default {_or.DEFAULT_SEG_M:.3f} m quadrat size. "
                "Is that your frame? If not, set *Quadrat size* (Inputs, "
                "above) and rectify again.", type="warning", timeout=15000)
        _corners_saved = _save_ortho_corners(
            state.ortho_src_path, state.ortho_corners,
            state.ortho_seg_lengths, state.ortho_frame_thickness_m)
        log_widget.clear()
        if _corners_saved:
            log_widget.push(f"Saved corners → {Path(_corners_saved).name}")
        log_widget.push(f"Loading {state.ortho_src_path}…")
        _cam = _dist = None
        if (state.ortho_use_distortion
                and state.ortho_fx > 0 and state.ortho_fy > 0):
            _cam = [[state.ortho_fx, 0.0, state.ortho_cx],
                    [0.0, state.ortho_fy, state.ortho_cy],
                    [0.0, 0.0, 1.0]]
            _dist = [state.ortho_k1, state.ortho_k2, state.ortho_p1,
                     state.ortho_p2, state.ortho_k3]
            log_widget.push("Applying lens distortion correction "
                            f"(fx={state.ortho_fx:.1f}, fy={state.ortho_fy:.1f}).")
        try:
            res = _or.rectify_one(
                state.ortho_src_path, state.ortho_corners,
                state.ortho_seg_lengths, state.ortho_out_path,
                gsd_m=state.ortho_gsd_m,
                camera_matrix=_cam, dist_coeffs=_dist,
                frame_thickness_m=state.ortho_frame_thickness_m,
                seg_confirmed=_or.seg_confirmed_for(
                    state.ortho_per_image, _seg_key),
                overwrite=True, tool_name=_brand.APP_NAME)
            for _n in res.notes:
                log_widget.push(_n)
            for _w in res.warnings:
                log_widget.push("⚠ " + _w)
                ui.notify(f"{_w} Check *Frame geometry* (Inputs, above).",
                          type="warning", timeout=12000)
            if not res.ok:
                log_widget.push(
                    f"❌ Refusing to write: {res.refusal}. The raw "
                    "photographs are field data and cannot be taken again.")
                ui.notify(f"Nothing written: {res.refusal} — see the log (below).",
                          type="negative")
                return
            state.ortho_out_path = res.out_path

            import base64
            import cv2
            import numpy as _np
            _img = cv2.imdecode(_np.fromfile(res.out_path, dtype=_np.uint8),
                                cv2.IMREAD_COLOR)
            if _img is not None:
                ok, buf = cv2.imencode(".png", _img)
                if ok:
                    data_url = ("data:image/png;base64,"
                                + base64.b64encode(buf).decode())
                    preview_label.set_text(
                        f"Rectified preview — {res.shape[1]}×{res.shape[0]} px "
                        f"@ GSD={res.gsd_m * 1000:.3f} mm/px")
                    preview_container.clear()
                    with preview_container:
                        ui.image(data_url).classes("w-full") \
                            .style("max-width: 900px;")
                    _scroll_to_preview()
            # The GSD travels in the sidecar; arm Detection on the output.
            try:
                state.det_mode = QUADRAT
                state.det_saveplot = True
                _outdir = str(Path(res.out_path).parent)
                if state.det_dir != _outdir:
                    state.det_dir = _outdir
            except Exception:
                pass  # arming Detection must never fail a rectification
            ui.notify(
                f"Rectified successfully at GSD={res.gsd_m * 1000:.3f} mm/px. "
                "The Detect tab picks this up automatically — it is set to "
                "Quadrat mode on this output folder, and the image "
                "carries its GSD in a sidecar.",
                type="positive", timeout=10000,
            )
            _ortho_update_nav_summary()
        except Exception as ex:
            import traceback
            log_widget.push(f"❌ {type(ex).__name__}: {ex}")
            log_widget.push(traceback.format_exc())

    # Rectify is the last item of the toolbar attached to the photograph.
    ui.button("Rectify", icon="auto_fix_high", on_click=_do_rectify) \
        .props("color=primary dense no-caps").move(ortho_toolbar)

    # ---------------- The whole folder ---------------------------------- #
    def _ortho_batch_rows():
        """Every photograph with what is recorded for it; no image is opened."""
        rows = []
        for f in state.ortho_image_list:
            full = os.path.join(state.ortho_image_dir, f)
            cached = state.ortho_per_image.get(f) or {}
            corners = cached.get("corners") or _load_ortho_corners(full) or []
            segs = cached.get("seg") or _load_ortho_segments(full)
            confirmed = _or_mod.seg_confirmed_for(state.ortho_per_image, f)
            _, done = _or_mod.rectified_output_for(
                _ortho_default_out_for(full), cached.get("gsd_m"))
            rows.append((f, corners, segs, confirmed,
                         done and not state.ortho_batch_overwrite))
        return rows

    def _ortho_batch_preflight():
        """What the run would do, before it does any of it."""
        if not state.ortho_image_list:
            state.ortho_batch_msg = ""
            return [], []
        rows = _ortho_batch_rows()
        ready, skipped = _or_mod.plan_rectify_folder(rows)
        lengths = _or_mod.lengths_in_play(
            [r for r in rows if r[0] in set(ready)])
        head = ""
        if lengths:
            head = " · ".join(f"{n} at {v:.3f} m" for v, n in lengths)
            if len(lengths) > 1:
                head += "   ← more than one size in this folder"
            head += "\n"
        head += (f"{len(ready)} to rectify, {len(skipped)} skipped."
                 if ready or skipped else "")
        by_reason = {}
        for name, why in skipped:
            by_reason.setdefault(why, []).append(name)
        for why, names in sorted(by_reason.items()):
            shown = ", ".join(names[:4]) + (" …" if len(names) > 4 else "")
            head += f"\n  {len(names)} {why}: {shown}"
        state.ortho_batch_msg = head
        return ready, skipped

    async def _ortho_rectify_folder():
        from nicegui import run as _run
        ready, _skipped = _ortho_batch_preflight()
        if not ready:
            ui.notify(_or_mod.batch_gate_message(_skipped),
                      type="warning", timeout=12000, multi_line=True)
            return
        jobs = []
        for name in ready:
            full = os.path.join(state.ortho_image_dir, name)
            cached = state.ortho_per_image.get(name) or {}
            jobs.append(dict(
                photo=name, src=full,
                corners=cached.get("corners") or _load_ortho_corners(full),
                segs=cached.get("seg") or _load_ortho_segments(full),
                out=_ortho_default_out_for(full),
                gsd=None,
                thickness=_load_ortho_frame_thickness(full),
                confirmed=_or_mod.seg_confirmed_for(
                    state.ortho_per_image, name)))

        state.ortho_batch_stop = False
        state.ortho_batch_done = 0
        state.ortho_batch_total = len(jobs)
        state.ortho_batch_busy = f"Rectifying {len(jobs)} photograph(s)…"
        state.ortho_batch_rows = []

        def _progress(done, total, now):
            state.ortho_batch_done = done
            state.ortho_batch_total = total
            state.ortho_batch_now = now

        def _work():
            # Off the event loop: the JPEG writes would exceed NiceGUI's
            # ping window and drop the client.
            return _or_mod.rectify_folder(
                jobs, overwrite=bool(state.ortho_batch_overwrite),
                progress_fn=_progress,
                should_stop=lambda: bool(state.ortho_batch_stop),
                tool_name=_brand.APP_NAME)

        try:
            items = await _run.io_bound(_work)
        except Exception as ex:                       # noqa: BLE001
            state.ortho_batch_busy = ""
            ui.notify(f"The run stopped: {ex}. See the log (below), then Rectify every ready file again.", type="negative")
            return
        state.ortho_batch_busy = ""
        state.ortho_batch_rows = [
            {"photo": i.photo, "status": i.status,
             "detail": i.reason or "; ".join(i.warnings),
             "output": Path(i.out_path).name if i.out_path else "—"}
            for i in items]

        # A manifest on disk: state does not survive a reconnect.
        try:
            import datetime as _dt
            out_dir = Path(_ortho_default_out_for(
                os.path.join(state.ortho_image_dir,
                             state.ortho_image_list[0]))).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            man = out_dir / "rectification_run.txt"
            man.write_text(_or_mod.batch_manifest(
                items, when=_dt.datetime.now().isoformat(timespec="seconds")),
                encoding="utf-8")
            log_widget.push(f"Wrote {man.name} beside the outputs.")
        except OSError as ex:
            log_widget.push(f"(could not write the run manifest: {ex})")

        # Anything not rectified is named with its reason; nothing is
        # dropped in silence.
        n_ok = sum(1 for i in items if i.status == "rectified")
        others = [i for i in items if i.status != "rectified"]
        summary = f"{n_ok} of {len(items)} rectified."
        for i in others:
            why = i.reason or "; ".join(i.warnings) or i.status
            summary += f"\n  {i.status}: {i.photo} — {why}"
            log_widget.push(f"{i.status}: {i.photo} — {why}")
        # Photographs the plan left out are still part of the accounting.
        for name, why in _skipped:
            summary += f"\n  skipped: {name} — {why}"
            log_widget.push(f"skipped: {name} — {why}")
        for i in items:
            if i.status == "rectified":
                log_widget.push(f"rectified: {i.photo} → "
                                f"{Path(i.out_path).name if i.out_path else '—'}")
        state.ortho_batch_msg = summary
        ui.notify(f"{n_ok} of {len(items)} rectified.", type="positive",
                  timeout=10000)
        _ortho_update_nav_summary()
        try:
            _ortho_refresh_table()
        except Exception:
            pass

    # The folder run consumes the inputs above it and nothing below.
    with ortho_band_a:
        with ui.card().classes("w-full") \
                .style("background-color: #f4f6f9; border: 1px solid #ddd;"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.button("Rectify every ready file", icon="auto_fix_high",
                          on_click=lambda: _ortho_rectify_folder()) \
                    .props("color=primary") \
                    .bind_enabled_from(state, "ortho_batch_busy",
                                       backward=lambda b: not b) \
                    .tooltip("Rectifies every photograph whose corners are "
                             "recorded and whose size somebody chose. A "
                             "photograph still at the default size is skipped "
                             "and named — a folder rectified at a size nobody "
                             "confirmed is the failure that cost a survey.")
                ui.button("Refresh", icon="refresh",
                          on_click=lambda: (_refresh_ortho_dir(),
                                            _ortho_batch_preflight())) \
                    .props("flat dense no-caps")
                ui.button("Stop", icon="stop",
                          on_click=lambda: setattr(state, "ortho_batch_stop",
                                                   True)) \
                    .props("flat dense no-caps color=negative") \
                    .bind_visibility_from(state, "ortho_batch_busy",
                                          backward=lambda b: bool(b))
                ui.checkbox("overwrite existing outputs") \
                    .bind_value(state, "ortho_batch_overwrite") \
                    .on_value_change(lambda _: _ortho_batch_preflight()) \
                    .tooltip("Off by default. An output already on disk is "
                             "work that was done, and replacing it silently "
                             "is how a good rectification becomes a bad one.")
            # Fixed height: a label that grows moves the toolbar below it.
            ui.label("").bind_text_from(state, "ortho_batch_msg") \
                .classes("text-sm w-full") \
                .style("white-space:pre-wrap; height:4.2em; overflow:auto; "
                       "line-height:1.4em;")
            ortho_batch_progress = build_progress()

            def _sync_ortho_batch():
                if state.ortho_batch_busy and state.ortho_batch_total:
                    ortho_batch_progress.update(state.ortho_batch_done,
                                                state.ortho_batch_total,
                                                state.ortho_batch_now)
            ui.timer(0.3, _sync_ortho_batch)

    # Arrive with the folder filled and its first photograph on the
    # surface. Last, because it drives everything above.
    if state.ortho_image_dir and os.path.isdir(state.ortho_image_dir):
        _refresh_ortho_dir()
    else:
        _seed_ortho_folder(force=False)


# ----- Zonal-stats tab: per-polygon and transect summaries ----- #
def build_zonal_tab():
    """Per-polygon zonal statistics and transect profiling.

    Polygon mode: per-polygon mean / std / Dxx (size fields) or Pxx (other
    fields) over the raster pixels inside each polygon. Transect mode:
    equal-distance samples along each LineString, with optional DEM
    co-sampling, as a long-format CSV plus a 2-panel plot per transect.
    Output goes to ``<project>/results/zonal/`` by default.
    """
    from functions import zonal_stats as zs

    # Element refs captured when the analysis inputs are built, so seeded
    # values can carry the "from project" hint.
    _zonal_field_els = {}

    def _seed_zonal(force=False):
        # Newest clast CSV and raster from the active project. The canvas
        # image is left to USE SOURCE / Browse: loading it is heavy.
        from functions import project_defaults as _pdf
        proj = state.current_project
        csv = _pdf.best_clast_csv(proj)
        if csv and (force or not state.zonal_csv):
            state.zonal_csv = str(csv)
            _el = _zonal_field_els.get("csv")
            if _el is not None:
                _seed_default(_el, csv, force=True)
        r = (_pdf.rasters(proj) or [None])[0]
        if r and (force or not state.zonal_raster):
            state.zonal_raster = str(r)
            _el = _zonal_field_els.get("raster")
            if _el is not None:
                _seed_default(_el, r, force=True)

    def _on_proj_change():
        _seed_profile(force=True)
        _zc = state.zonal_zones_canvas
        _zc.vector = ""
        _zc.image = ""
        _zc.out_path = ""
        _zc.pts.clear()
        _zc.features.clear()
        _zc.selected_idx = -1
        state.zonal_dem = ""
        state.zonal_out_csv = ""
        state.zonal_csv = ""
        state.zonal_raster = ""
        _seed_zonal(force=True)

    render_project_strip(on_change=_on_proj_change)

    ui.markdown("### Zonal statistics")
    ui.label("Per-polygon statistics from a clast CSV, raster samples along "
             "transects, or a profile figure from transect CSVs — one "
             "analysis per sub-tab.").classes("text-sm text-grey-7")

    # One chooser for the whole task, at the top.
    with ui.tabs().props("dense no-caps").classes("w-full") as _zonal_subtabs:
        ui.tab("polygons", label="Polygons", icon="category")
        ui.tab("transects", label="Transects", icon="timeline")
        ui.tab("profile", label="Profile figure", icon="show_chart")
    _zonal_subtabs.bind_value(state, "zonal_view")

    # --- Source canvas: the shared build_image_feature_canvas, configured
    # for the named-zone-set save model -------------------------------- #
    _zctx = state.zonal_zones_canvas
    _zonal_canvas_refs: dict = {}
    _zonal_late: dict = {}

    def _zonal_rebuild():
        fn = _zonal_canvas_refs.get("rebuild_canvas")
        if fn is not None:
            fn()

    def _on_zonal_mode_change(e=None):
        # Never write "transect" into _zctx.shape: the polygon toggle bound
        # to it has no such option and Quasar would push null back. The
        # analysis mode is applied by _zonal_effective_shape() instead.
        if _zctx.shape not in ("polygon", "rectangle", "circle"):
            _zctx.shape = "polygon"

    def _zonal_effective_shape():
        # The analysis mode is authoritative: transect mode always draws
        # transects, whatever the hidden polygon toggle holds.
        return zonal_shape_for_mode(state.zonal_mode, _zctx.shape)

    def _on_zonal_view_change(_e=None):
        if state.zonal_view in ("polygons", "transects"):
            if state.zonal_mode != state.zonal_view:
                state.zonal_mode = state.zonal_view
            _on_zonal_mode_change()
    _zonal_subtabs.on_value_change(_on_zonal_view_change)

    _canvas_card = ui.column().classes("w-full")
    _canvas_card.bind_visibility_from(
        state, "zonal_view",
        backward=lambda v: v in ("polygons", "transects"))
    with _canvas_card:
        # ----- Analysis-mode-driven shape toolbar -------------------- #
        # The component's own shape toggle is hidden; these mode-visible
        # toggles are rendered into its sticky toolbar. Two toggles must
        # never bind the same attribute (Quasar pushes an unknown value back
        # as null), so the transect toggle is a binding-less indicator.
        def _zonal_shape_toggles():
            ui.toggle({"polygon": "Polygon", "rectangle": "Rectangle",
                       "circle": "Circle"}) \
                .bind_value(_zctx, "shape") \
                .props("dense no-caps") \
                .bind_visibility_from(state, "zonal_mode", value="polygons") \
                .tooltip("polygon: click each vertex, then Finalise. "
                         "rectangle / circle: 2 clicks, auto-commit.")
            ui.toggle({"transect": "Transect"}, value="transect") \
                .props("no-unset dense no-caps") \
                .bind_visibility_from(state, "zonal_mode", value="transects") \
                .tooltip("transect: click each waypoint, then Finalise.")

        # ----- Source-from-inputs / reload (analysis-aware) ---------- #
        with ui.row().classes("items-center gap-2 mt-2 flex-wrap"):
            def _use_source_from_inputs():
                """Fill the canvas image from the CSV's source ortho or the
                transect raster."""
                target = ""
                if state.zonal_mode == "transects" and state.zonal_raster:
                    target = state.zonal_raster
                elif state.zonal_mode == "polygons" and state.zonal_csv:
                    csv_p = Path(state.zonal_csv)
                    stem = re.sub(r"_individual_clasts$", "", csv_p.stem)
                    stem = naming.image_stem(stem)
                    images_dir = (project_path(state.current_project, "images")
                                  if state.current_project else csv_p.parent)
                    for d in (images_dir, csv_p.parent):
                        for ext in _images.PHOTO_EXTENSIONS:
                            cand = Path(d) / f"{stem}{ext}"
                            if cand.exists():
                                target = str(cand)
                                break
                        if target:
                            break
                if target:
                    # Warm the transcode/probe in a worker subprocess first;
                    # the in-process rebuild then hits a warm cache.
                    outcome = _worker_mod.run_job(
                        "canvas_source",
                        {"src_path": target,
                         "current_project": state.current_project,
                         "max_dim": 4000, "cache_subdir": "canvas_cache"})
                    if not outcome.ok:
                        ui.notify(f"{_job_error_text(outcome)} Pick another source "
                                  "(Inputs, above).",
                                  type="negative", multi_line=True)
                        return
                    _zctx.image = target
                    _zonal_rebuild()
                    ui.notify(f"Using {Path(target).name}", type="positive")
                else:
                    if state.zonal_mode == "polygons" and not state.zonal_csv:
                        msg = ("Set *Detection CSV* (Inputs, above) first; "
                               "this button infers the source ortho from "
                               "that CSV's name.")
                    elif (state.zonal_mode == "transects"
                          and not state.zonal_raster):
                        msg = ("Set *Raster (.tif)* (Inputs, above) first; "
                               "transect mode uses it directly as the "
                               "source.")
                    else:
                        msg = ("No image in the project matches this "
                               + ("CSV's" if state.zonal_mode == "polygons"
                                  else "raster's")
                               + " name. Pick one with *Source image* "
                               "Browse (canvas, below), or check the "
                               "*Detection CSV* / *Raster* (Inputs, above).")
                    ui.notify(msg, type="warning", multi_line=True)

            ui.button("Use source from inputs", icon="link",
                      on_click=_use_source_from_inputs).props("outline") \
                .tooltip("Polygon mode -> infer source ortho from the "
                         "detection-CSV filename. Transect mode -> use the "
                         "picked raster directly.")
            ui.button("Reload image", icon="refresh",
                      on_click=_zonal_rebuild).props("outline") \
                .tooltip("Re-read the current source image and rebuild the "
                         "canvas. Staged features are kept (re-projected).")

        # ----- Zone-set identity ------------------------------------- #
        # A zone set is named, not tied to an image; the save path derives
        # from the name, never from the loaded image.
        with ui.row().classes("w-full items-end gap-2 mt-2"):
            ui.input("Zone set name", placeholder="e.g. upper_beach_zones") \
                .bind_value(state, "zonal_set_name").classes("flex-grow") \
                .tooltip("Save GeoJSON writes input_data/geometries/"
                         "<name>.geojson, reusable on any ortho or date. "
                         "Letters, digits, spaces, dots, hyphens, "
                         "underscores.")

        # ----- Save-path policy injected into the shared component --- #
        def _zonal_resolve_save_path():
            # None when unnamed, so the component prompts instead of guessing.
            path = zonal_set_save_path(
                zonal_target_out_path(_zctx.out_path, state.zonal_set_name),
                state.zonal_set_name, state.current_project, _zctx.image)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def _zonal_on_unnamed_save(manual):
            raw = (state.zonal_set_name or "").strip()
            if raw and not sanitise_zone_set_name(raw):
                ui.notify(
                    f"'{raw}' can't be used as a zone set name. Use letters, "
                    "digits, spaces, dots, hyphens or underscores -- no "
                    "slashes or '..', so the file stays inside the project. Change "
                    "it in *Zone set name* (canvas, above).",
                    type="negative", timeout=10000, multi_line=True,
                    close_button=True)
            elif manual:
                ui.notify(
                    "Type a name in *Zone set name* (canvas, above) first. "
                    "Sets are named, not tied to an image, so the same set "
                    "can be reused on any date.",
                    type="warning", timeout=10000, multi_line=True,
                    close_button=True)
            else:
                ui.notify(
                    "Type a name in *Zone set name* (canvas, above) to save "
                    "it. It is stored in the project's geometries folder and "
                    "can be reused on any ortho or date.",
                    type="info", timeout=10000, multi_line=True,
                    close_button=True)

        def _zonal_on_save(_path):
            fn = _zonal_late.get("refresh_selectors")
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass

        _zonal_canvas_refs.update(build_image_feature_canvas(
            _zctx,
            title="Zones canvas",
            description="Draw or import the zone set on the source image; "
                        "Save GeoJSON writes it under the zone set name and "
                        "the run below consumes it.",
            toolbar_extra=_zonal_shape_toggles,
            allowed_shapes=("polygon", "rectangle", "circle", "transect"),
            current_project_getter=lambda: state.current_project,
            image_picker_default_kind="images",
            vector_picker_default_kind="vectors",
            save_path_resolver=_zonal_resolve_save_path,
            on_unnamed_save=_zonal_on_unnamed_save,
            write_provenance=True,
            disjoint_policy="warn",
            confirm_overwrite=True,
            show_shape_toggle=False,
            overlay_select_recolor=True,
            import_dedup=True,
            effective_shape_getter=_zonal_effective_shape,
            on_save=_zonal_on_save,
            tab_active_check=lambda: state.active_tab == "Zonal"))

        # Sync the draw shape to the persisted mode now that the canvas
        # exists, so an early transect is not committed as a polygon.
        _on_zonal_view_change()


    # --- Analysis section (consumes the canvas's vector) ------------------ #
    _analysis_card = ui.card().classes("w-full")
    _analysis_card.bind_visibility_from(
        state, "zonal_view",
        backward=lambda v: v in ("polygons", "transects"))
    with _analysis_card:
        ui.label("Inputs").classes("text-subtitle2 text-grey-8")
        ui.label("The per-clast CSV (polygons) or raster (transects); the "
                 "zone set comes from the canvas below.") \
            .classes("text-xs text-grey-7")

        # --- Polygon-mode inputs (the clast CSV, not a raster) -------- #
        polygon_box = ui.card().classes("w-full bg-grey-1")
        with polygon_box:
            ui.markdown("**Polygon mode — per-polygon distribution from CSV**")
            with ui.row().classes("w-full items-end gap-2"):
                _zonal_field_els["csv"] = ui.input(
                    "Detection CSV",
                    placeholder="path to *_individual_clasts.csv") \
.bind_value(state, "zonal_csv") \
.classes("flex-grow") \
.tooltip("Per-clast measurements — same file you'd feed "
                             "the Rasterize tab. The output CSV will hold "
                             "one distribution-summary row per polygon.")

                def _pick_zonal_csv():
                    p = native_file_picker(
                        title="Select detection CSV",
                        filetypes=[("CSV", "*.csv"), ("All", "*.*")],
                        initialdir=default_starting_dir("vectors"),
                    )
                    if p:
                        state.zonal_csv = p
                        if not state.zonal_out_csv:
                            from pathlib import Path as _PL
                            _det = _PL(p)
                            state.zonal_out_csv = str(
                                _det.parent / (_det.stem + "_zonal.csv"))

                ui.button("Browse…", icon="folder_open", on_click=_pick_zonal_csv).props("outline")
            ui.input("Percentiles (comma-separated)",
                      placeholder="5, 16, 25, 50, 75, 84, 95") \
.bind_value(state, "zonal_percentiles_text") \
.tooltip("Integer percentiles 0–100. Labels honour the field "
                         "kind: grain-size fields get D5/D16/D50/D84/D95, "
                         "others P5/P16/median/P84/P95. The full output CSV "
                         "always also carries count, density, mean, std, "
                         "median, IQR, CV, range, statistical skew/kurt, and "
                         "Folk-Ward σφ/Skφ/K_G (NaN for non-size fields).")

        # --- Transect-mode inputs (raster + step + interpolation) ----- #
        transect_box = ui.card().classes("w-full bg-grey-1")
        with transect_box:
            ui.markdown("**Transect mode — sample a raster along each LineString**")
            with ui.row().classes("w-full items-end gap-2"):
                _zonal_field_els["raster"] = ui.input(
                    "Raster (.tif)",
                    placeholder="path to GeoTIFF") \
.bind_value(state, "zonal_raster") \
.classes("flex-grow") \
.tooltip("Typically a Rasterize-tab output (D50, "
                             "Clast_length, etc.). Sampled bilinearly (or "
                             "nearest-neighbour) at each step along every "
                             "input LineString.")

                def _pick_raster():
                    p = native_file_picker(
                        title="Select raster (.tif)",
                        filetypes=[("GeoTIFF", "*.tif *.tiff"),
                                   ("All", "*.*")],
                        initialdir=default_starting_dir("rasters"),
                    )
                    if p:
                        state.zonal_raster = p
                        stem = Path(p).stem.lower()
                        for f in ("Clast_length", "Clast_width",
                                   "Equivalent_diameter",
                                   "Ellipse_major_axis",
                                   "Ellipse_minor_axis",
                                   "Surface_area", "Orientation",
                                   "Clast_elongation", "Clast_circularity"):
                            if f.lower() in stem:
                                state.zonal_field = f
                                break

                ui.button("Browse…", icon="folder_open", on_click=_pick_raster).props("outline")
            with ui.row().classes("w-full items-end gap-2"):
                ui.number("Sampling step (m)", min=0.01, step=0.1,
                           format="%.3f") \
.bind_value(state, "zonal_step_m").classes("w-40") \
.tooltip("Distance between samples along each line, "
                             "in raster CRS units.")
                ui.select(["bilinear", "nearest"]) \
.bind_value(state, "zonal_interpolation") \
.classes("w-40") \
.tooltip("Bilinear smooths across cells; nearest returns "
                             "raw cell values (useful for categorical rasters).")
                ui.number("Band", min=1, step=1, format="%d") \
.bind_value(state, "zonal_band") \
.classes("w-24") \
.tooltip("1-based band index (GDAL convention).")

        # --- Common inputs (both modes) -------------------------------- #
        # Field name and ID field list the CSV's columns / the layer's
        # attributes, but accept a typed value too.
        def _zonal_csv_columns() -> list:
            p = state.zonal_csv
            if not p or not Path(p).exists():
                return []
            try:
                import pandas as _pd
                return [str(c) for c in _pd.read_csv(p, nrows=1).columns]
            except Exception:
                return []

        def _zonal_vector_fields() -> list:
            p = state.zonal_zones_canvas.vector
            if not p or not Path(p).exists():
                return []
            try:
                from osgeo import ogr
                vds = ogr.Open(p, 0)
                if vds is None:
                    return []
                defn = vds.GetLayer(0).GetLayerDefn()
                fields = [defn.GetFieldDefn(i).GetName()
                          for i in range(defn.GetFieldCount())]
                vds = None
                return fields
            except Exception:
                return []

        field_select = ui.select(
            options=_zonal_csv_columns(),
            label="Field name",
            with_input=True, new_value_mode="add-unique",
        ).bind_value(state, "zonal_field") \
.tooltip("In polygon mode: column of the detection CSV to "
                     "summarise (the drop-down lists that CSV's columns). "
                     "In transect mode: logical name of what the raster "
                     "represents (drives D vs P labels) — type your own.")

        # Default the ID field to 'name' when the layer carries one.
        _init_id_fields = _zonal_vector_fields()
        if not state.zonal_id_field and "name" in _init_id_fields:
            state.zonal_id_field = "name"
        # The FID fallback is a labelled entry, never a blank row.
        def _id_options(fields):
            return {"": "— feature index (FID) —", **{f: f for f in fields}}

        id_select = ui.select(
            options=_id_options(_init_id_fields),
            label="ID field (label for each zone)",
            with_input=True, new_value_mode="add-unique",
        ).bind_value(state, "zonal_id_field") \
.tooltip("Attribute used to label each polygon / transect in the "
                     "output CSV. Defaults to 'name' (the names you typed in "
                     "the canvas); change it only if you want a different "
                     "attribute. Empty = use the OGR feature index (FID).")

        _zonal_sel_paths = {"csv": None, "vec": None}

        def _refresh_zonal_selectors():
            if state.zonal_csv != _zonal_sel_paths["csv"]:
                _zonal_sel_paths["csv"] = state.zonal_csv
                cols = _zonal_csv_columns()
                if cols:
                    # The select is built before any CSV is known and its
                    # binding writes None over the default: always land on
                    # a column that exists.
                    val = state.zonal_field if state.zonal_field in cols else None
                    if val is None:
                        val = ("Clast_length" if "Clast_length" in cols
                               else next((c for c in cols
                                          if c not in ("x", "y", "clast_ID")),
                                         cols[0]))
                    field_select.set_options(cols, value=val)
                    state.zonal_field = val
            if state.zonal_zones_canvas.vector != _zonal_sel_paths["vec"]:
                _zonal_sel_paths["vec"] = state.zonal_zones_canvas.vector
                _fields = _zonal_vector_fields()
                if not state.zonal_id_field and "name" in _fields:
                    state.zonal_id_field = "name"
                id_select.set_options(
                    _id_options(_fields), value=state.zonal_id_field)

        # The CSV field, the canvas's save and a project switch refresh the
        # lists at once; the timer is the fallback for paths set by code.
        _zonal_field_els["csv"].on_value_change(
            lambda _: _refresh_zonal_selectors())
        ui.timer(1.0, _refresh_zonal_selectors)
        _zonal_late["refresh_selectors"] = _refresh_zonal_selectors

        _seed_zonal()

        # DEM (optional), both modes.
        with ui.row().classes("w-full items-end gap-2"):
            ui.input("DEM (optional, .tif)",
                      placeholder="path to elevation GeoTIFF") \
.bind_value(state, "zonal_dem") \
.classes("flex-grow") \
.tooltip("In polygon mode → emits dem_mean/std/min/max/range "
                         "columns per polygon. In transect mode → adds "
                         "elevation_m to the sample CSV and a topography "
                         "panel to the profile plot.")

            def _pick_dem():
                p = native_file_picker(
                    title="Select DEM",
                    filetypes=[("GeoTIFF", "*.tif *.tiff"),
                               ("All", "*.*")],
                    initialdir=default_starting_dir("dem"),
                )
                if p:
                    state.zonal_dem = p

            ui.button("Browse…", icon="folder_open", on_click=_pick_dem).props("outline")

        polygon_box.bind_visibility_from(
            state, "zonal_mode", value="polygons")
        transect_box.bind_visibility_from(
            state, "zonal_mode", value="transects")

        with ui.row().classes("w-full items-end gap-2"):
            ui.input("Output CSV", placeholder="auto if blank") \
.bind_value(state, "zonal_out_csv") \
.classes("flex-grow") \
.tooltip("If blank, a name is auto-generated under the project's "
                         "results/zonal/ folder.")

            def _pick_out():
                p = native_save_file_picker(
                    title="Save zonal-stats CSV as…",
                    initialdir=default_starting_dir("zonal"),
                    defaultextension=".csv",
                    filetypes=[("CSV", "*.csv"), ("All", "*.*")],
                )
                if p:
                    state.zonal_out_csv = p

            ui.button("Browse…", icon="folder_open", on_click=_pick_out).props("outline")

    # --- Run / Queue ---
    log_widget = live_log("zonal", build_log_console(height="h-64"))

    def _safe_log(msg: str):
        # Ignore stale-client errors if the tab is closed mid-run.
        try:
            log_widget.push(msg)
        except Exception:
            pass

    def _zonal_extent_diagnostic(csv_path: str, vector_path: str) -> str:
        """Explain an empty zonal result by comparing the two extents.
        Returns "" when they do overlap (some other cause)."""
        try:
            import pandas as pd
            from osgeo import ogr
            df = pd.read_csv(csv_path, usecols=["x", "y"])
            cx0, cx1 = float(df["x"].min()), float(df["x"].max())
            cy0, cy1 = float(df["y"].min()), float(df["y"].max())
            vds = ogr.Open(vector_path, 0)
            if vds is None:
                return ""
            vx0, vx1, vy0, vy1 = vds.GetLayer(0).GetExtent()
            vds = None
            if vx0 <= cx1 and vx1 >= cx0 and vy0 <= cy1 and vy1 >= cy0:
                return ""
            return (
                f"The zones and the clasts do not overlap. "
                f"Zones cover X {vx0:,.1f}–{vx1:,.1f}, Y {vy0:,.1f}–{vy1:,.1f}; "
                f"the clasts in this CSV cover X {cx0:,.1f}–{cx1:,.1f}, "
                f"Y {cy0:,.1f}–{cy1:,.1f}. "
                f"These are almost always zones drawn on a different image — "
                f"check that the Vector layer belongs to the same ortho as the "
                f"Detection CSV, or re-draw the zones on this image."
            )
        except Exception:
            return ""

    def _run_zonal_sync(snap: dict):
        result = {
            "mode": snap.get("mode"),
            "field": snap.get("field"),
            "out_csv": "",
            "pngs": [],
            "error": "",
            "warning": "",
        }
        try:
            mode = snap["mode"]
            vector = snap["vector"]
            field_name = snap["field"]
            id_field = snap["id_field"]
            out_csv = snap["out_csv"]
            dem = snap.get("dem") or None

            if not vector or not Path(vector).exists():
                result["error"] = "missing vector layer."
                _safe_log("ERR: missing vector layer.")
                state.zonal_last_result = result
                try: _render_inline_results.refresh()
                except Exception: pass
                return

            if mode == "polygons":
                csv_in = snap.get("csv") or ""
                if not csv_in or not Path(csv_in).exists():
                    result["error"] = "polygon mode requires a detection CSV."
                    _safe_log("ERR: polygon mode requires a detection CSV.")
                    state.zonal_last_result = result
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    return
                base_stem = Path(csv_in).stem
            else:
                raster = snap.get("raster") or ""
                if not raster or not Path(raster).exists():
                    result["error"] = "transect mode requires a raster."
                    _safe_log("ERR: transect mode requires a raster.")
                    state.zonal_last_result = result
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    return
                base_stem = Path(raster).stem
                # The Field selector is filled from the CSV, which transect
                # mode does not read: recover the field from the raster name.
                if not field_name:
                    field_name = field_from_raster(raster)
                    result["field"] = field_name
                    _safe_log(f"  field inferred from raster name: "
                              f"{field_name!r}")

            # A CRS mismatch raises nowhere: the run "succeeds" over an empty
            # result. The per-clast CSV carries no CRS, so polygon mode
            # compares against the displayed ortho.
            crs_ref = (snap.get("raster") if mode == "transects"
                       else state.zonal_zones_canvas.image)
            if crs_ref and Path(str(crs_ref)).exists():
                crs_msg = zs.describe_crs_mismatch(vector, crs_ref)
                if crs_msg:
                    result["warning"] = crs_msg
                    _safe_log(f"  WARNING: {crs_msg}")

            if not out_csv:
                vec_stem = Path(vector).stem
                proj = state.current_project
                if proj:
                    out_dir = project_path(proj, "zonal")
                else:
                    parent = (Path(snap.get("csv") or snap.get("raster") or
                                    vector)).parent
                    out_dir = parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv = str(out_dir / zonal_auto_out_name(
                    base_stem, vec_stem, field_name, mode, id_field))
                _safe_log(f"  out_csv = {out_csv} (auto)")

            if mode == "polygons":
                pct_text = snap.get("percentiles",
                                      "5, 16, 25, 50, 75, 84, 95")
                try:
                    pcts = tuple(int(x.strip()) for x in pct_text.split(",")
                                 if x.strip())
                except ValueError:
                    result["error"] = f"bad percentile list: {pct_text!r}"
                    _safe_log(f"ERR: bad percentile list: {pct_text!r}")
                    state.zonal_last_result = result
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    return
                if not field_name:
                    result["error"] = (
                        "No field selected. Pick the measurement to summarise "
                        "in 'Field name' — it lists the columns of the "
                        "Detection CSV you chose.")
                    _safe_log(f"ERR: {result['error']}")
                    state.zonal_last_result = result
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    return
                _safe_log(f"polygon stats over {Path(snap['csv']).name} "
                          f"by {Path(vector).name} (pct={list(pcts)})"
                          + (f" + DEM {Path(dem).name}" if dem else ""))
                # A worker subprocess: a native fault in GDAL/OGR kills the
                # job, not the app.
                outcome = _worker_mod.run_job(
                    "zonal_polygons",
                    {"detection_csv": snap["csv"], "vector_path": vector,
                     "out_csv": out_csv, "field_name": field_name,
                     "id_field": id_field, "percentiles": list(pcts),
                     "dem_path": dem},
                    log_cb=lambda l: _safe_log(f"  [worker] {l}"))
                if not outcome.ok:
                    result["error"] = _job_error_text(outcome)
                    _safe_log(f"ERR: {result['error']}")
                    state.zonal_last_result = result
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    return
                rows = outcome.summary.get("rows", [])
                # An empty polygon still emits a row (all NaN). A zone inside
                # the surveyed area with no clasts is a measured zero; one
                # outside it is missing data. They must be reported apart.
                empty = [r for r in rows if not r.get("count")]
                outside = [r for r in empty
                           if r.get("coverage", "inside") == "outside"]
                measured_zero = [r for r in empty
                                 if r.get("coverage", "inside") != "outside"]
                if rows and len(empty) == len(rows):
                    detail = _zonal_extent_diagnostic(snap["csv"], vector)
                    result["warning"] = (
                        f"No clasts fell inside any of the {len(rows)} zone(s) — "
                        f"every statistic in the output CSV is NaN. "
                        + (detail or
                           "Check that the Vector layer and the Detection CSV "
                           "describe the same area and use the same CRS.")
                    )
                    _safe_log(f"  WARNING: {result['warning']}")
                elif empty:
                    parts = []
                    if measured_zero:
                        n = len(measured_zero)
                        parts.append(
                            f"{n} {'zone lies' if n == 1 else 'zones lie'} "
                            f"inside the surveyed area and genuinely "
                            f"{'contains' if n == 1 else 'contain'} no clasts "
                            f"(a measured zero)")
                    if outside:
                        n = len(outside)
                        parts.append(
                            f"{n} {'zone falls' if n == 1 else 'zones fall'} "
                            f"outside the surveyed area entirely, so "
                            f"{'its' if n == 1 else 'their'} zero is missing "
                            f"data, not an observation")
                    result["warning"] = (
                        f"{len(empty)} of {len(rows)} zone(s) are empty: "
                        + "; ".join(parts)
                        + ". Every zone still has a row, and the 'coverage' "
                          "column records which case each one is."
                    )
                    _safe_log(f"  WARNING: {result['warning']}")
            else:
                step = float(snap.get("step_m", 0.5))
                interp = snap.get("interpolation", "bilinear")
                band = int(snap.get("band", 1))
                _safe_log(f"transect profile along {Path(vector).name} "
                          f"step={step} m interp={interp}"
                          + (f" + DEM {Path(dem).name}" if dem else ""))
                outcome = _worker_mod.run_job(
                    "zonal_transects",
                    {"raster": snap["raster"], "vector_path": vector,
                     "out_csv": out_csv, "step_m": step, "band": band,
                     "id_field": id_field, "dem_path": dem,
                     "interpolation": interp, "field_name": field_name,
                     "make_plots": True},
                    log_cb=lambda l: _safe_log(f"  [worker] {l}"))
                if not outcome.ok:
                    result["error"] = _job_error_text(outcome)
                    _safe_log(f"ERR: {result['error']}")
                    state.zonal_last_result = result
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    return
                n_samples = int(outcome.summary.get("n_samples", 0))
                n_transects = int(outcome.summary.get("n_transects", 0))
                result["pngs"].extend(outcome.summary.get("pngs", []))
                _safe_log(f"  wrote {n_samples} "
                          f"sample(s) across {n_transects} transect(s) -> {out_csv}")
                # Zero samples writes a header-only CSV and no plots.
                if not n_samples:
                    result["warning"] = (
                        "The transects returned no samples — the output CSV "
                        "contains only a header. This usually means the "
                        "transect lines fall outside the raster, or the raster "
                        "is empty (all no-data) where they cross it. Check that "
                        "the Vector layer belongs to the same ortho as the "
                        "raster."
                    )
                    _safe_log(f"  WARNING: {result['warning']}")
            result["out_csv"] = out_csv
            state.zonal_last_result = result
            try:
                _render_inline_results.refresh()
            except Exception:
                pass
        except Exception as ex:
            result["error"] = f"{type(ex).__name__}: {ex}"
            state.zonal_last_result = result
            try: _render_inline_results.refresh()
            except Exception: pass
            _safe_log(f"ERR: {ex}")

    def _snap_current() -> dict:
        return {
            "csv":            state.zonal_csv,
            "raster":         state.zonal_raster,
            "vector":         state.zonal_zones_canvas.vector,
            "dem":            state.zonal_dem,
            "field":          state.zonal_field,
            "mode":           state.zonal_mode,
            "band":           int(state.zonal_band),
            "id_field":       state.zonal_id_field,
            "percentiles":    state.zonal_percentiles_text,
            "step_m":         float(state.zonal_step_m),
            "interpolation":  state.zonal_interpolation,
            "out_csv":        state.zonal_out_csv,
            "status":         "queued",
        }

    # ---- Run buttons + queue (same pattern as Detection / Rasterize) ----
    def _add_to_queue():
        snap = _snap_current()
        state.zonal_jobs.append(snap)
        _render_queue.refresh()

    def _run_now():
        threading.Thread(
            target=_run_zonal_sync,
            args=(_snap_current(),),
            daemon=True,
        ).start()

    def _run_queue():
        def worker():
            for j in state.zonal_jobs:
                if j.get("status") in (None, "queued", "pending", "error", "interrupted"):
                    j["status"] = "running"
                    _render_queue.refresh()
                    try:
                        _run_zonal_sync(j)
                        j["status"] = "done"
                    except Exception as ex:
                        j["status"] = "error"
                        j["error"] = f"{type(ex).__name__}: {ex}"
                    _render_queue.refresh()
            # Two or more finished transect runs get a combined profile figure.
            done_transect_csvs = [
                j.get("out_csv") for j in state.zonal_jobs
                if j.get("mode") == "transects"
                and j.get("status") == "done"
                and j.get("out_csv")
                and Path(j["out_csv"]).exists()
            ]
            if len(done_transect_csvs) >= 2:
                try:
                    overlay_dir = Path(done_transect_csvs[0]).parent
                    out = zs.build_profile_outputs(
                        done_transect_csvs, overlay_dir,
                        "transects_overlay", bin_width_m=1.0)
                    res = dict(state.zonal_last_result)
                    res["overlay_png"] = str(out["png"])
                    state.zonal_last_result = res
                    try: _render_inline_results.refresh()
                    except Exception: pass
                    _safe_log(f"Profile figure → {out['png'].name}")
                except Exception as ex:
                    _safe_log(f"Overlay plot skipped: {ex}")
        threading.Thread(target=worker, daemon=True).start()

    def _make_zonal_map():
        """Publication map: the product raster with zones and named transects
        overlaid, saved to the project's zonal/ folder."""
        raster = state.zonal_raster
        vector = state.zonal_zones_canvas.vector
        if not raster or not Path(raster).exists():
            ui.notify("Set *Raster (.tif)* (Inputs, above) first — it is the map's "
                      "coloured basemap.", type="warning")
            return
        if not vector or not Path(vector).exists():
            ui.notify("Set or save the *Vector layer* (canvas, above) first.",
                      type="warning")
            return
        field = state.zonal_field or "Clast_length"
        id_field = state.zonal_id_field or ""
        proj = state.current_project
        out_dir = (project_path(proj, "zonal") if proj
                   else Path(raster).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / naming.zonal_map_name(Path(raster).stem,
                                                  Path(vector).stem)

        def _worker():
            try:
                outcome = _worker_mod.run_job(
                    "zonal_map",
                    {"raster": raster, "vector_path": vector,
                     "out_png": str(out_png), "field_name": field,
                     "id_field": id_field,
                     "title": f"{field} — {Path(raster).stem}"},
                    log_cb=_safe_log)
                if not outcome.ok:
                    raise RuntimeError(_job_error_text(outcome))
                res = dict(state.zonal_last_result or {})
                res["zonal_map_png"] = str(out_png)
                state.zonal_last_result = res
                try:
                    _render_inline_results.refresh()
                except Exception:
                    pass
                _safe_log(f"Publication map → {out_png.name}")
            except Exception as ex:
                _safe_log(f"ERR: publication map failed: {ex}")
        threading.Thread(target=_worker, daemon=True).start()
        ui.notify("Building publication map…", type="info")

    # The run/queue controls belong to the two analysis sub-tabs only.
    _run_row = ui.row().classes("items-center gap-2 mt-2")
    _run_row.bind_visibility_from(
        state, "zonal_view",
        backward=lambda v: v in ("polygons", "transects"))
    with _run_row:
        ui.button("Run now", icon="play_arrow",
                   on_click=_run_now).props("color=primary")
        ui.button("Add to queue", icon="add",
                   on_click=_add_to_queue).props("color=primary")
        ui.button("Publication map", icon="map",
                   on_click=_make_zonal_map) \
.props("color=primary outline") \
.tooltip("Render the product raster with the zones and named transects "
                     "overlaid (saved to the project's zonal/ folder).")
        ui.button("Run all queued", icon="playlist_play",
                   on_click=_run_queue) \
.props("color=primary outline") \
.tooltip("Execute every pending job in order. Done jobs "
                     "are skipped; errored jobs are re-tried.")

        def _reset_zonal_queue():
            n = 0
            for j in state.zonal_jobs:
                if j.get("status") in ("done", "error", "queued"):
                    j["status"] = "pending"
                    n += 1
            _render_queue.refresh()
            ui.notify(f"Reset {n} job(s) to pending.",
                       type="info" if n else "warning")
        ui.button("Reset all", icon="restart_alt",
                   on_click=_reset_zonal_queue) \
.props("flat color=primary") \
.tooltip("Mark every done / errored job pending again "
                     "so 'Run all queued' will re-run them.")
        ui.button("Clear queue", icon="delete_sweep",
                   on_click=lambda: (
                       state.zonal_jobs.clear(),
                       _render_queue.refresh())) \
.props("flat color=grey") \
.tooltip("Remove all jobs from the queue.")
        ui.space()
        ui.label().bind_text_from(
            state, "zonal_jobs",
            lambda lst: f"{len(lst)} queued" if lst else "")\
.classes("text-sm text-grey-7")

    queue_container = ui.column().classes("w-full mt-1")

    def _zonal_primary(job, idx):
        src = ((job.get("csv")
                if job.get("mode") == "polygons"
                else job.get("raster")) or "")
        return Path(src).name if src else "(no source)"

    def _zonal_params(job, idx):
        vec_name = (Path(job["vector"]).name
                    if job.get("vector")
                    else "(no vector)")
        return (f"vs {vec_name}   {job.get('mode')} · "
                f"field={job.get('field')}")

    def _zonal_delete(idx):
        state.zonal_jobs.pop(idx)
        _render_queue.refresh()

    @ui.refreshable
    def _render_queue():
        # A missing status defaults to "queued" (an alias for pending).
        render_queue(
            queue_container, state.zonal_jobs,
            title="Queued jobs",
            count_label=None,  # Zonal binds its own count via bind_text_from
            primary_text=_zonal_primary,
            params_text=_zonal_params,
            on_delete=_zonal_delete,
            default_status="queued",
        )

    _render_queue()

    # ---- Inline result display ---------------------------- #
    # Fed by state.zonal_last_result: a CSV table and quick-look composer
    # for polygon runs, the per-transect plots and overlay for transects.
    @ui.refreshable
    def _render_inline_results():
        res = state.zonal_last_result or {}
        if not res:
            return
        with ui.card().classes("w-full mt-2"):
            ui.markdown("#### Last result")
            if res.get("error"):
                ui.label(f"Failed: {res['error']}") \
.classes("text-red-7")
                return
            out_csv = res.get("out_csv", "")
            mode = res.get("mode", "")
            # The warning goes above the result line: an empty-but-successful
            # run must not read as a normal outcome.
            if res.get("warning"):
                with ui.row().classes(
                        "w-full items-start gap-2 q-pa-sm rounded-borders "
                        "bg-orange-1 border border-orange-5"):
                    ui.icon("warning").classes("text-orange-9 text-lg")
                    ui.label(res["warning"]).classes("text-orange-10 text-sm")
            ui.markdown(
                f"**Mode:** `{mode}`  ·  "
                f"**Field:** `{res.get('field', '')}`  ·  "
                f"**Output CSV:** `{out_csv}`"
            ).classes("text-sm text-grey-8")

            df = None
            try:
                import pandas as pd
                df = pd.read_csv(out_csv)
                head = df.head(30)
                columns = [
                    {"name": c, "label": c, "field": c,
                     "align": "left"}
                    for c in head.columns
                ]
                rows = head.to_dict(orient="records")
                ui.label(
                    f"CSV preview — first {len(head)} of "
                    f"{len(df)} rows"
                ).classes("text-sm text-grey-7 mt-2")
                ui.table(columns=columns, rows=rows,
                          row_key=head.columns[0]) \
.classes("w-full") \
.props("dense flat")
            except Exception as ex:
                ui.label(
                    f"Could not load CSV preview: {ex}"
                ).classes("text-red-7")

            # Quick-look (polygon mode): a per-polygon scatter from the
            # summary CSV beside a per-polygon violin clipped from the
            # source detection CSV.
            if (mode == "polygons"
                    and df is not None
                    and len(df.columns) >= 2):
                try:
                    import pandas as _pd
                    numeric_cols = [
                        c for c in df.select_dtypes(include="number").columns
                        if c not in ZONAL_PROVENANCE_COLUMNS]
                except Exception:
                    numeric_cols = []

                def _source_numeric_cols():
                    """Per-clast fields available in the source detection CSV
                    (the violin plots their distribution per polygon)."""
                    try:
                        if state.zonal_csv and Path(state.zonal_csv).exists():
                            sdf = _pd.read_csv(state.zonal_csv, nrows=1)
                            return [c for c in sdf.columns
                                    if c.lower() not in ("x", "y", "clast_id")]
                    except Exception:
                        pass
                    return []

                if len(numeric_cols) >= 2:
                    ui.separator().classes("mt-2")
                    ui.markdown(
                        "**Quick-look — scatter + per-polygon violin**"
                    ).classes("mt-1 text-sm")
                    _x0 = (state.zonal_scatter_x
                            if state.zonal_scatter_x in numeric_cols
                            else numeric_cols[0])
                    _y0 = (state.zonal_scatter_y
                            if state.zonal_scatter_y in numeric_cols
                            else numeric_cols[1])
                    state.zonal_scatter_x = _x0
                    state.zonal_scatter_y = _y0
                    _src_cols = _source_numeric_cols()
                    _vf0 = getattr(state, "zonal_violin_field", "") \
                        or state.zonal_field
                    if _src_cols and _vf0 not in _src_cols:
                        _vf0 = (state.zonal_field
                                if state.zonal_field in _src_cols
                                else _src_cols[0])
                    state.zonal_violin_field = _vf0

                    with ui.row().classes("items-end gap-2 flex-wrap"):
                        x_sel = ui.select(numeric_cols, value=_x0,
                                          label="Scatter X").classes("w-40")
                        y_sel = ui.select(numeric_cols, value=_y0,
                                          label="Scatter Y").classes("w-40")
                        x_sel.on_value_change(lambda e: setattr(
                            state, "zonal_scatter_x", e.value))
                        y_sel.on_value_change(lambda e: setattr(
                            state, "zonal_scatter_y", e.value))
                        if _src_cols:
                            vf_sel = ui.select(
                                _src_cols, value=_vf0,
                                label="Violin field (per-clast)") \
                                .classes("w-52")
                            vf_sel.on_value_change(lambda e: setattr(
                                state, "zonal_violin_field", e.value))
                        else:
                            ui.label("(per-polygon violin needs the source "
                                     "detection CSV set in the canvas)") \
                                .classes("text-xs text-grey-6")

                    def _per_polygon_values(field, _src=state.zonal_csv,
                                            _vec=state.zonal_zones_canvas.vector):
                        """[(label, values-inside-polygon), ...] by clipping the
                        source per-clast CSV to each polygon. Values are in
                        the field's display unit, like the summary CSV."""
                        import numpy as _np
                        from osgeo import ogr as _ogr
                        from functions.units import field_unit_and_factor
                        from functions.zonal_stats import (
                            _ring_to_path_coords as _r2p,
                            _points_in_rings as _pir)
                        _factor = field_unit_and_factor(field)[1]
                        if (not _src or not Path(_src).exists()
                                or not _vec or not Path(_vec).exists()):
                            return []
                        try:
                            sdf = _pd.read_csv(_src)
                        except Exception:
                            return []
                        if (field not in sdf.columns or "x" not in sdf.columns
                                or "y" not in sdf.columns):
                            return []
                        pts = sdf[["x", "y"]].to_numpy(float)
                        vals = _pd.to_numeric(sdf[field],
                                              errors="coerce").to_numpy()
                        vds = _ogr.Open(_vec)
                        if vds is None:
                            return []
                        lyr = vds.GetLayer(0)
                        groups = []
                        for i, feat in enumerate(lyr):
                            geom = feat.GetGeometryRef()
                            if geom is None:
                                continue
                            inside = _pir(pts, _r2p(geom))
                            v = vals[inside] * _factor
                            v = v[_np.isfinite(v)]
                            lbl = None
                            for key in ("name", "id"):
                                try:
                                    fi = feat.GetFieldIndex(key)
                                    if fi >= 0:
                                        fv = feat.GetField(fi)
                                        if fv not in (None, ""):
                                            lbl = str(fv)
                                            break
                                except Exception:
                                    pass
                            groups.append((lbl or f"#{i + 1}", v))
                        vds = None
                        return groups

                    def _gen_plots(_df=df, _num=numeric_cols):
                        import matplotlib.pyplot as plt
                        plt.switch_backend("Agg")
                        import base64, io

                        def _save_quicklook(buffer, kind):
                            """Write the figure beside the zonal CSV it
                            describes, so it outlives the page that shows
                            it."""
                            try:
                                # The CSV this block is previewing, which is
                                # where its figures belong.
                                base = Path(out_csv or state.zonal_out_csv or "")
                                if not base.name:
                                    return None
                                out = base.with_name(f"{base.stem}_{kind}.png")
                                buffer.seek(0)
                                out.write_bytes(buffer.read())
                                return out
                            except Exception:
                                return None
                        # The summary CSV's unit column labels both panels;
                        # absent, the labels stay bare rather than wrong.
                        try:
                            _csv_unit = ""
                            if "unit" in _df.columns and len(_df):
                                _u = _df["unit"].iloc[0]
                                # A dimensionless field is an empty cell (NaN).
                                if _pd.notna(_u):
                                    _csv_unit = str(_u)
                        except Exception:
                            _csv_unit = ""
                        # --- scatter: field-of-interest vs another field ---
                        try:
                            xcol = (state.zonal_scatter_x
                                    if state.zonal_scatter_x in _num
                                    else _num[0])
                            ycol = (state.zonal_scatter_y
                                    if state.zonal_scatter_y in _num
                                    else _num[1])
                            fig, ax = plt.subplots(figsize=(6, 4))
                            tmp = _df[[xcol, ycol]].dropna()
                            ax.scatter(tmp[xcol], tmp[ycol], s=20, alpha=0.6,
                                       color="#225599")
                            ax.set_xlabel(zonal_axis_label(
                                xcol, zonal_column_unit(xcol, _csv_unit)))
                            ax.set_ylabel(zonal_axis_label(
                                ycol, zonal_column_unit(ycol, _csv_unit)))
                            ax.set_title(f"{ycol} vs {xcol} "
                                         "(one point per polygon)", fontsize=10)
                            ax.grid(True, alpha=0.3)
                            fig.tight_layout()
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=140)
                            plt.close(fig); buf.seek(0)
                            state.zonal_scatter_img = (
                                "data:image/png;base64,"
                                + base64.b64encode(buf.read()).decode())
                            _saved = _save_quicklook(buf, "scatter")
                            if _saved is not None:
                                log_widget.push(f"[zonal] wrote {_saved.name}")
                        except Exception as exc:
                            state.zonal_scatter_img = f"__error__:{exc}"
                        # --- companion per-polygon violin ---
                        try:
                            vf = (getattr(state, "zonal_violin_field", "")
                                  or state.zonal_field)
                            groups = [(lbl, v) for lbl, v
                                      in _per_polygon_values(vf)
                                      if len(v) >= 2]
                            fig, ax = plt.subplots(
                                figsize=(max(6.0, 0.5 * len(groups) + 2), 4))
                            if groups:
                                ax.violinplot([v for _, v in groups],
                                              showmedians=True)
                                ax.set_xticks(range(1, len(groups) + 1))
                                ax.set_xticklabels(
                                    [lbl for lbl, _ in groups],
                                    rotation=45, ha="right", fontsize=8)
                                from functions.units import field_unit
                                ax.set_ylabel(
                                    zonal_axis_label(vf, field_unit(vf)))
                                ax.set_title(
                                    f"Per-polygon distribution of {vf} "
                                    f"({len(groups)} polygons)", fontsize=10)
                                ax.grid(True, axis="y", alpha=0.3)
                            else:
                                ax.text(0.5, 0.5,
                                        "No per-polygon clast values.\nSet the "
                                        "source detection CSV + a saved vector "
                                        "in the canvas.",
                                        ha="center", va="center",
                                        transform=ax.transAxes)
                                ax.set_axis_off()
                            fig.tight_layout()
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=140,
                                        bbox_inches="tight")
                            plt.close(fig); buf.seek(0)
                            state.zonal_violin_img = (
                                "data:image/png;base64,"
                                + base64.b64encode(buf.read()).decode())
                            _saved = _save_quicklook(buf, "violin")
                            if _saved is not None:
                                log_widget.push(f"[zonal] wrote {_saved.name}")
                        except Exception as exc:
                            state.zonal_violin_img = f"__error__:{exc}"
                        try:
                            _render_inline_results.refresh()
                        except Exception:
                            pass

                    ui.button("Generate", icon="bar_chart",
                              on_click=_gen_plots) \
.props("color=primary outline").classes("mt-1") \
.tooltip("Render the scatter and its companion per-polygon violin.")

                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        for _im in (state.zonal_scatter_img or "",
                                    getattr(state, "zonal_violin_img", "")
                                    or ""):
                            if _im.startswith("__error__:"):
                                ui.label(_im[len("__error__:"):]) \
.classes("text-red-7 mt-1")
                            elif _im.startswith("data:image"):
                                ui.image(_im).classes("mt-1") \
.style("max-width:520px;")

            overlay_png = res.get("overlay_png") or ""
            if overlay_png and Path(overlay_png).exists():
                ui.separator().classes("mt-2")
                ui.markdown(
                    "**Combined transect overlay**"
                ).classes("mt-1 text-sm")
                try:
                    import base64
                    with open(overlay_png, "rb") as _fh:
                        _b64 = base64.b64encode(
                            _fh.read()).decode()
                    ui.image(
                        f"data:image/png;base64,{_b64}"
                    ).classes("w-full") \
.style("max-width:900px;")
                    ui.label(Path(overlay_png).name) \
.classes("text-xs text-grey-7")
                except Exception as ex:
                    ui.label(
                        f"Could not embed overlay: {ex}"
                    ).classes("text-red-7")

            zonal_map_png = res.get("zonal_map_png") or ""
            if zonal_map_png and Path(zonal_map_png).exists():
                ui.separator().classes("mt-2")
                ui.markdown("**Publication map** — product raster with zones "
                            "and named transects").classes("mt-1 text-sm")
                try:
                    import base64
                    with open(zonal_map_png, "rb") as _fh:
                        _b64 = base64.b64encode(_fh.read()).decode()
                    ui.image(f"data:image/png;base64,{_b64}").classes("w-full") \
.style("max-width:900px;")
                    ui.label(Path(zonal_map_png).name) \
.classes("text-xs text-grey-7")
                except Exception as ex:
                    ui.label(f"Could not embed map: {ex}") \
.classes("text-red-7")

            pngs = res.get("pngs") or []
            if pngs:
                ui.markdown(
                    f"**Per-transect profiles ({len(pngs)})**"
                ).classes("mt-2 text-sm")
                shown = pngs[:6]
                for png in shown:
                    try:
                        import base64
                        with open(png, "rb") as fh:
                            b64 = base64.b64encode(
                                fh.read()).decode()
                        ui.image(f"data:image/png;base64,{b64}") \
.classes("w-full") \
.style("max-width:800px;")
                        ui.label(Path(png).name) \
.classes("text-xs text-grey-7")
                    except Exception as ex:
                        ui.label(
                            f"Could not embed {png}: {ex}"
                        ).classes("text-red-7")
                if len(pngs) > 6:
                    ui.label(
                        f"(+{len(pngs)-6} more PNGs on disk)"
                    ).classes("text-xs text-grey-7")

    # The result readout is bound to the same sub-tabs as the run controls.
    _results_holder = ui.element("div").classes("w-full")
    _results_holder.bind_visibility_from(
        state, "zonal_view",
        backward=lambda v: v in ("polygons", "transects"))
    with _results_holder:
        _render_inline_results()

    # Page order: inputs -> canvas -> run -> queue -> log -> results.
    _kids = _canvas_card.parent_slot.parent.default_slot.children
    _analysis_card.move(target_index=_kids.index(_canvas_card))
    _kids = queue_container.parent_slot.parent.default_slot.children
    log_widget.move(target_index=_kids.index(queue_container) + 1)
    for _el in (queue_container, log_widget):
        _el.bind_visibility_from(
            state, "zonal_view",
            backward=lambda v: v in ("polygons", "transects"))

    # ------ Profile figure ----------------------------------- #
    _profile_card = ui.card().classes("w-full mt-3")
    _profile_card.bind_visibility_from(state, "zonal_view", value="profile")
    with _profile_card:
        ui.markdown("#### Profile figure")
        ui.label('A profile figure (grain size above, elevation below, one line per date) and its three tables, from transect CSVs.').classes("text-sm text-grey-7")

        # ---- Profile inputs -------------------------------------- #
        overlay_card = ui.card().classes("w-full bg-grey-1")
        with overlay_card:
            ui.markdown("**Transect CSVs** — one row per date. Each "
                        "transect_id inside a CSV becomes its own line; "
                        "leave the label blank to use the folder's date.")
            overlay_rows_box = ui.column().classes("w-full")

            def _add_overlay_csv():
                state.zonal_overlay_csvs.append({"csv": "", "label": ""})
                _refresh_overlay_rows()

            def _remove_overlay_csv(idx: int):
                if 0 <= idx < len(state.zonal_overlay_csvs):
                    state.zonal_overlay_csvs.pop(idx)
                    _refresh_overlay_rows()

            def _refresh_overlay_rows():
                overlay_rows_box.clear()
                with overlay_rows_box:
                    if not state.zonal_overlay_csvs:
                        ui.label(
                            "No CSVs added. Click Add CSV.") \
.classes("text-grey-6 italic")
                    for i, row in enumerate(state.zonal_overlay_csvs):
                        with ui.row().classes(
                                "items-end gap-2 w-full"):
                            ui.label(f"{i+1}.").classes("text-grey-7")
                            ui.input("CSV path",
                                      placeholder="transect CSV") \
.bind_value(row, "csv") \
.classes("flex-grow")
                            ui.input("Label (optional)",
                                      placeholder="auto from date folder") \
.bind_value(row, "label") \
.classes("w-52") \
.tooltip("Overrides this line's legend entry. Leave blank "
                                         "to label it with the survey date read "
                                         "from the project's date folder.")

                            # Mutate the dict in place and let the binding
                            # update the input: rebuilding the rows from
                            # inside the callback tears down the live input.
                            def _pick(idx=i):
                                p = native_file_picker(
                                    title="Pick transect CSV",
                                    filetypes=[("CSV", "*.csv"),
                                               ("All", "*.*")],
                                    initialdir=default_starting_dir(
                                        "zonal"),
                                )
                                if p:
                                    state.zonal_overlay_csvs[idx]["csv"] = p

                            ui.button(icon="folder_open", on_click=_pick) \
.props("flat dense") \
.tooltip("Select a transect CSV.")
                            ui.button(
                                icon="delete",
                                on_click=lambda idx=i:
                                    _remove_overlay_csv(idx)
                            ).props("flat dense color=negative")

            _refresh_overlay_rows()
            with ui.row().classes("gap-2 mt-2") as _prof_add_row:
                ui.button("Add CSV", icon="add",
                           on_click=_add_overlay_csv) \
.props("outline color=primary")
                ui.button("Clear", icon="delete_sweep",
                           on_click=lambda: (
                               state.zonal_overlay_csvs.clear(),
                               _refresh_overlay_rows())) \
.props("outline color=negative")

        with ui.row().classes("w-full items-end gap-2 mt-2"):
            ui.number("Table bin width (m)", value=1.0, min=0.01, step=0.5,
                       format="%.2f") \
.bind_value(state, "zonal_profile_bin_m") \
.classes("w-48") \
.tooltip("Distance bin for the binned table — the figure is "
                       "unaffected. Must be greater than zero.")
            ui.input("Caption (what the reader should conclude)",
                      placeholder="e.g. Upper beach coarsens landward; "
                                  "2023 is coarser throughout.") \
.bind_value(state, "zonal_profile_caption") \
.classes("flex-grow") \
.tooltip("Printed under the figure. State the finding, not "
                       "just what is plotted.")

        # ---- Common knobs + render ------------------------------ #
        with ui.row().classes("w-full items-end gap-2 mt-2"):
            ui.input("Field name (y-axis label)",
                      placeholder="Clast_length") \
.bind_value(state, "zonal_plot_field_name") \
.classes("w-56")
            ui.input("Title (optional)") \
.bind_value(state, "zonal_plot_title") \
.classes("flex-grow")

        with ui.row().classes("w-full items-end gap-2"):
            ui.input("Output PNG (auto if blank)") \
.bind_value(state, "zonal_plot_out_png") \
.classes("flex-grow")

            def _pick_plot_out():
                p = native_save_file_picker(
                    title="Save combined plot as…",
                    initialdir=default_starting_dir("zonal"),
                    defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("All", "*.*")],
                )
                if p:
                    state.zonal_plot_out_png = p

            ui.button("Browse…", icon="folder_open", on_click=_pick_plot_out) \
.props("outline")

        plot_log = build_log_console(max_lines=200, height="h-24")
        plot_preview = ui.column().classes("w-full mt-2")

        def _resolve_plot_out() -> str:
            """The output PNG path, defaulted under the project's zonal/."""
            from functions import zonal_stats as _zs_path
            proj = state.current_project
            base = (project_path(proj, "zonal") if proj else Path.cwd())
            return str(_zs_path.profile_out_path(state.zonal_plot_out_png, base))

        def _display_inline_preview(png_path: str):
            """Embed the rendered PNG inline."""
            plot_preview.clear()
            try:
                with open(png_path, "rb") as fh:
                    raw = fh.read()
                import base64
                b64 = base64.b64encode(raw).decode()
                with plot_preview:
                    ui.image(f"data:image/png;base64,{b64}") \
.classes("w-full") \
.style("max-width:900px;")
                    ui.label(f"Saved to {png_path}") \
.classes("text-xs text-grey-7")
            except Exception as ex:
                try:
                    plot_log.push(f"[preview] embed failed: {ex}")
                except Exception:
                    pass

        def _display_profile_tables(res):
            """The three tables, on the page under the figure."""
            import csv as _csv
            with plot_preview:
                for key in ("summary", "change", "binned"):
                    p = res.get(key)
                    if not p:
                        continue
                    try:
                        with open(p, newline="", encoding="utf-8") as fh:
                            rows = list(_csv.reader(fh))
                    except OSError as ex:
                        ui.label(f"{key}: could not read {p} ({ex})") \
                            .classes("text-xs text-negative")
                        continue
                    if not rows:
                        continue
                    head, body = rows[0], rows[1:200]
                    ui.label(f"{key} — {Path(str(p)).name}"
                             + (f" (first 200 of {len(rows) - 1} rows)"
                                if len(rows) - 1 > 200 else "")) \
                        .classes("text-xs font-bold mt-2")
                    ui.table(
                        columns=[{"name": h, "label": h, "field": h,
                                  "align": "left"} for h in head],
                        rows=[dict(zip(head, r)) for r in body],
                        row_key=head[0]).classes("w-full pm-profile-table") \
                        .props("dense flat")

        def _render_profile():
            from functions import zonal_stats as _zs
            rows = [r for r in state.zonal_overlay_csvs if r.get("csv")]
            if not rows:
                ui.notify("Add at least one transect CSV (*Add CSV*, above) first.",
                           type="warning")
                return
            csvs = [r["csv"] for r in rows]
            labels = {r["csv"]: r["label"].strip()
                      for r in rows if (r.get("label") or "").strip()}
            out = Path(_resolve_plot_out())
            stem = out.stem
            try:
                res = _zs.build_profile_outputs(
                    csvs, out.parent, stem,
                    labels=labels or None,
                    # Not `or`: a typed 0 must reach the validator.
                    bin_width_m=(1.0 if state.zonal_profile_bin_m in (None, "")
                                 else state.zonal_profile_bin_m),
                    title=state.zonal_plot_title,
                    caption=state.zonal_profile_caption,
                )
            except ValueError as ex:
                # Mixed fields, a bad bin width, or nothing plottable.
                plot_log.push(f"ERR: {ex}")
                ui.notify(f"{ex} Fix the rows (Profile inputs, above), then Render again.",
                           type="negative", timeout=14000,
                           multi_line=True, close_button=True)
                return
            except Exception as ex:
                plot_log.push(f"ERR: {ex}")
                ui.notify(f"Plot failed: {ex}. See the log (below the button), then Render again.", type="negative")
                return

            state.zonal_plot_last_png = str(res["png"])
            plot_log.push(f"wrote {res['png']}  "
                          f"({len(res['series'])} line(s) from {len(csvs)} CSV)")
            for key in ("summary", "change", "binned"):
                plot_log.push(f"  {key} table -> {res[key]}")
            _display_inline_preview(str(res["png"]))
            _display_profile_tables(res)
            if res["skipped"]:
                msg = ("Left out, no samples (check the rows in Profile inputs, above): "
                       + ", ".join(res["skipped"]))
                plot_log.push(f"  WARNING: {msg}")
                ui.notify(msg, type="warning", timeout=12000,
                           multi_line=True, close_button=True)
            else:
                ui.notify(f"Profile figure and 3 tables saved beside "
                           f"{res['png'].name}", type="positive")

        _profile_render_btn = ui.button(
            "Render profile figure + tables", icon="image",
            on_click=_render_profile).props("color=primary")
        # The log and the figure follow the button that produces them.
        plot_preview.classes("pm-profile-preview")
        _pk = _profile_card.default_slot.children
        plot_log.move(_profile_card, target_index=_pk.index(_profile_render_btn) + 1)
        plot_preview.move(_profile_card, target_index=_pk.index(plot_log) + 1)
        # The Add CSV row precedes the rows it adds to.
        _ok = overlay_card.default_slot.children
        _prof_add_row.move(overlay_card,
                           target_index=_ok.index(overlay_rows_box))

    def _seed_profile(force=False):
        """The project's transect CSVs (newest first, up to six) become the
        profile's rows when none were added by hand."""
        from functions import project_defaults as _pdf
        proj = state.current_project
        if not proj:
            return
        if state.zonal_overlay_csvs and not force:
            return
        csvs = _pdf.transect_csvs(proj)[:6]
        if not csvs:
            return
        state.zonal_overlay_csvs = [{"csv": str(p), "label": ""} for p in csvs]
        try:
            _refresh_overlay_rows()
        except Exception:
            pass
    _seed_profile()


# ----- Georeference tab (place quadrats before spending time on them) ---- #
def build_georeference_tab():
    """Put a quadrat photograph into the ortho's coordinate frame.

    Every long action runs off the event loop through ``run.io_bound``:
    NiceGUI drops the browser connection when the loop is blocked past its
    reconnect timeout, and a spinner cannot paint meanwhile.
    """
    import numpy as np

    render_project_strip(on_change=lambda: _seed_georef(force=True))
    ui.markdown("### Georeference")
    ui.label('Place a quadrat photograph in the ortho before digitizing it; the match uses the photograph alone.').classes("text-sm text-grey-7")
    ui.label("About half of quadrats place when the ortho is near the photograph's resolution; a refusal is normal and writes nothing; an accepted match is good to millimetres.").classes("text-sm text-grey-7")

    # --- 1. Inputs -------------------------------------------------------- #
    ui.markdown("#### Inputs").classes("q-mt-sm")
    _geo_ortho_row = path_input_with_browse(
        "Ortho-image to place quadrats in", "dig_georef_ortho", kind="file",
        filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
        default_kind="images",
        on_change=lambda: _geo_on_ortho_change())
    _geo_dir_row = path_input_with_browse(
        "Folder of quadrat photographs", "dig_photo_dir", kind="dir",
        default_kind="validation",
        on_change=lambda: _geo_reload_photos())

    with ui.row().classes("w-full items-end gap-3"):
        quadrat_select = ui.select(
            [], label="Quadrat to locate", with_input=True,
            on_change=lambda e: _geo_pick_quadrat(e.value)) \
            .classes("w-96").props("dense") \
            .tooltip("Lists the photographs in the folder above. Picking one "
                     "fills the path beside it.")
        ui.button("Load photo list", icon="refresh",
                  on_click=lambda: _geo_reload_photos()) \
            .props("flat dense no-caps") \
            .tooltip("The list also refreshes when the folder changes; this "
                     "is only needed if the folder's contents changed.")
    # The exception: one photograph outside the folder, including the one
    # the Digitize tab hands over through `geo_quadrat_path`.
    with ui.expansion("Work on one photograph from outside this folder",
                      icon="photo").classes("w-full"):
        path_input_with_browse(
            "Quadrat photograph", "geo_quadrat_path", kind="file",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.heic *.heif"),
                       ("All files", "*.*")],
            default_kind="validation",
            on_change=lambda: _geo_on_quadrat_change())
    ui.label("").bind_text_from(state, "geo_quadrat_note") \
        .bind_visibility_from(state, "geo_quadrat_note",
                              backward=lambda v: bool(v)) \
        .classes("text-sm text-grey-8")

    with ui.row().classes("w-full items-end gap-3"):
        ui.number(label="Quadrat resolution (m/pixel)",
                  step=0.0001, min=0.0001, format="%.6f") \
            .bind_value(state, "geo_resolution").classes("w-48") \
            .tooltip("Image scale in metres per pixel. Auto-filled from "
                     "filenames containing _GSD=...m.")
        ui.number(label="Search radius (m)", step=1.0, min=0.5,
                  format="%.1f") \
            .bind_value(state, "dig_georef_radius").classes("w-36") \
            .tooltip("How far from the seed the quadrat may be found. A "
                     "generous radius costs time, not accuracy: measured on "
                     "Bio_Station, seed error up to 5 m placed the same "
                     "quadrats to the same few millimetres.")
        ui.number(label="Frame thickness (m)", step=0.005, min=0.0,
                  format="%.3f") \
            .bind_value(state, "dig_georef_inset").classes("w-40") \
            .tooltip("The quadrat frame is equipment, not ground: it is "
                     "straight and high-contrast, so it attracts features the "
                     "ortho does not contain. This much is cropped from each "
                     "edge before matching. Clast coordinates are unaffected. "
                     "Filled in automatically when the photograph's "
                     "rectification record carries it; type one here and "
                     "yours is kept. Measured on Bio_Station the frame is "
                     "0.055-0.065 m, and setting it there changed neither "
                     "which quadrats were placed nor how accurately, so 0 is "
                     "a fine default.")
    ui.label("").bind_text_from(state, "geo_inset_note") \
        .bind_visibility_from(state, "geo_inset_note",
                              backward=lambda v: bool(v)) \
        .classes("text-sm text-grey-8")

    with ui.row().classes("w-full items-center gap-3"):
        ui.radio({"ortho": "Same as the ortho", "specify": "Specify"},
                 value=state.geo_seed_crs_mode) \
            .bind_value(state, "geo_seed_crs_mode").props("inline dense") \
            .tooltip("Which CRS the coordinates you type — and those in the "
                     "seed list — are in. Outputs are always written in the "
                     "ortho's CRS whichever you choose.")
        ui.input(label="Seed CRS") \
            .bind_value(state, "geo_seed_epsg").classes("w-44") \
            .bind_enabled_from(state, "geo_seed_crs_mode",
                               backward=lambda m: m == "specify") \
            .tooltip("EPSG:4326 is WGS84 latitude/longitude — what a handheld "
                     "GPS gives. Typed as longitude, latitude (x, y).")

    # --- 2. The seed list ------------------------------------------------- #
    ui.markdown("#### Seed list (optional)").classes("q-mt-md")
    ui.label('A CSV of photo, lat, lon (column names read flexibly; an optional crs column); without one, use pins below.').classes("text-sm text-grey-7")
    path_input_with_browse(
        "Seed list (CSV)", "dig_seed_csv", kind="file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        default_kind="validation",
        on_change=lambda: _geo_preview_seeds())
    with ui.row().classes("w-full items-end gap-3"):
        ui.select({"comma": "Comma ,", "semicolon": "Semicolon ;",
                   "tab": "Tab", "pipe": "Pipe |", "auto": "Auto-detect"},
                  label="Column separator") \
            .bind_value(state, "geo_csv_separator").classes("w-40") \
            .props("dense")
        ui.select({"auto": "Auto (either)", ".": "Dot 1.5", ",": "Comma 1,5"},
                  label="Decimal symbol") \
            .bind_value(state, "geo_csv_decimal").classes("w-40") \
            .props("dense") \
            .tooltip("Auto keeps the old tolerant guess. Saying which it is "
                     "matters for a file like 1.234,56, where the guess reads "
                     "1.234.")
        ui.switch("File has a header row") \
            .bind_value(state, "geo_csv_has_header") \
            .tooltip("Without one, the first three columns are read as "
                     "photo, x (longitude/easting), y (latitude/northing).")
        ui.button("Preview", icon="visibility",
                  on_click=lambda: _geo_preview_seeds()) \
            .props("outline dense no-caps") \
            .tooltip("Reads the first rows with the settings above, so a "
                     "wrong separator shows up here rather than as twenty "
                     "failures later.")
    ui.label("").bind_text_from(state, "geo_csv_preview_msg") \
        .classes("text-sm").style("white-space:pre-wrap")
    preview_table = ui.table(
        columns=[
            {"name": "photo", "label": "Photo", "field": "photo",
             "align": "left"},
            # A seed-list row and an EXIF fix are both "a seed"; which one
            # is in play decides whether editing the CSV changes anything.
            {"name": "from", "label": "Seed from", "field": "from",
             "align": "left"},
            {"name": "x", "label": "x (lon/east)", "field": "x",
             "align": "left"},
            {"name": "y", "label": "y (lat/north)", "field": "y",
             "align": "left"},
            {"name": "crs", "label": "CRS", "field": "crs", "align": "left"},
            {"name": "placed", "label": "Already placed", "field": "placed",
             "align": "left"},
        ], rows=[], row_key="photo").classes("w-full")
    preview_table.bind_visibility_from(state, "geo_csv_preview",
                                       backward=lambda r: bool(r))

    # --- 3. The actions --------------------------------------------------- #
    ui.markdown("#### Place them").classes("q-mt-md")
    with ui.row().classes("items-center gap-2") \
            .bind_visibility_from(state, "geo_busy",
                                  backward=lambda b: bool(b)):
        ui.spinner(size="1.4rem")
        ui.label("").bind_text_from(state, "geo_busy") \
            .classes("text-sm text-primary")

    # --- 3a. the whole survey: the default way in ----------------------- #
    with ui.expansion("Check a whole survey against the ortho",
                      icon="checklist", value=True).classes("w-full"):
        ui.markdown(
            "Find out in one pass which of the photographs in the folder "
            "above can be located. Nothing is written — this only tells you "
            "where to spend your digitising time."
        ).classes("text-sm text-grey-8")
        # What this run is about to cost, before it starts.
        ui.label("").bind_text_from(state, "geo_preflight") \
            .bind_visibility_from(state, "geo_preflight",
                                  backward=lambda v: bool(v)) \
            .classes("text-sm text-grey-8 q-mb-xs")
        with ui.row().classes("items-center gap-2"):
            # The button says what it is waiting for.
            _check_btn = ui.button("Check all quadrats", icon="checklist",
                                   on_click=lambda: _geo_check_all()) \
                .props("color=primary")
            _check_btn.bind_enabled_from(
                state, "geo_ready", backward=lambda v: bool(v))
            _check_btn.bind_text_from(
                state, "geo_not_ready",
                backward=lambda why: (f"Check all quadrats — {why}" if why
                                      else "Check all quadrats"))
            ui.button("Stop", icon="stop",
                      on_click=lambda: setattr(state, "geo_batch_stop", True)) \
                .props("flat dense no-caps color=negative") \
                .bind_visibility_from(state, "geo_busy",
                                      backward=lambda b: bool(b))
            # The check writes nothing; this keeps the placements it computed.
            ui.button("Save all located", icon="save",
                      on_click=lambda: _geo_save_all()) \
                .props("outline dense no-caps") \
                .bind_visibility_from(state, "dig_batch_rows",
                                      backward=lambda r: bool(r)) \
                .tooltip("Writes a GeoTIFF, a world-coordinate CSV and an "
                         "audit sidecar for every quadrat the check placed. "
                         "Refusals are never written — an absence of evidence "
                         "must not end up on disk looking like a measurement.")
            ui.checkbox("overwrite existing") \
                .bind_value(state, "geo_save_overwrite") \
                .bind_visibility_from(state, "dig_batch_rows",
                                      backward=lambda r: bool(r)) \
                .tooltip("Off by default: validation/georectified/ holds "
                         "hand-made reference placements, and replacing one "
                         "with an estimate cannot be undone.")
        # Below the run button and priced in visible text: this option
        # turns a four-minute run into an overnight one. A plain column,
        # never a collapsible inside a collapsible.
        with ui.column().classes("gap-0 q-mt-sm"):
            ui.checkbox(
                "Also search the whole ortho for quadrats with no seed") \
                .bind_value(state, "geo_full_grid") \
                .on_value_change(lambda _: _geo_update_readiness())
            ui.label('About an hour per unseeded quadrat (a whole-ortho sweep); seeded ones search ~80 tiles, and quadrats swept together share the reads.').classes("text-sm text-grey-7")
        # The worker thread assigns to `state`; only this event-loop timer
        # touches the widgets.
        batch_progress = build_progress()

        def _sync_batch_progress():
            if state.geo_busy and state.geo_batch_total:
                batch_progress.update(state.geo_batch_done,
                                      state.geo_batch_total,
                                      state.geo_batch_now)
        ui.timer(0.3, _sync_batch_progress)
        ui.label("").bind_text_from(state, "dig_batch_msg") \
            .classes("text-sm").style("white-space:pre-wrap")
        batch_table = ui.table(
            columns=[
                {"name": "photo", "label": "Photo", "field": "photo",
                 "align": "left"},
                {"name": "status", "label": "Result", "field": "status",
                 "align": "left"},
                {"name": "seed", "label": "Seed from", "field": "seed",
                 "align": "left"},
                {"name": "detail", "label": "Detail", "field": "detail",
                 "align": "left"},
            ], rows=[], row_key="photo").classes("w-full")
        batch_table.bind_visibility_from(state, "dig_batch_rows",
                                         backward=lambda r: bool(r))
        # A row opens the placement the batch already computed; no re-match.
        batch_table.on("rowClick", lambda e: _geo_row_clicked(e))
        ui.label('Click a row to open its placement in the editor above (no re-run); a row with candidates lets you choose between them.').classes("text-sm text-grey-7")

        # The placements on the ortho catch what a table cannot (two on the
        # same ground, one in the water, a transposed seed column).
        with ui.row().classes("items-center gap-2") \
                .bind_visibility_from(state, "dig_batch_rows",
                                      backward=lambda r: bool(r)):
            ui.button("Show them on the ortho", icon="map",
                      on_click=lambda: _geo_draw_survey_overlay()) \
                .props("outline dense no-caps")
            ui.label("").bind_text_from(state, "geo_overlay_note") \
                .classes("text-sm text-grey-7")
        survey_overlay_holder = ui.column().classes("w-full")
    # --- 3b. one quadrat: the exception ---------------------------------- #
    # Collapsed: a survey results row opens this panel on its quadrat.
    _locate_panel = ui.expansion("Locate one quadrat", icon="my_location",
                                 value=False).classes("w-full")
    _locate_panel.bind_visibility_from(state, "geo_quadrat_needs_georef")
    with _locate_panel:
        # Name the quadrat here: the chooser in section 1 is off-screen.
        ui.label().bind_text_from(
            state, "geo_quadrat_path",
            backward=lambda p: (f"Placing: {Path(p).name}" if p
                                else "No quadrat chosen — pick one in "
                                     "section 1 above.")) \
            .classes("text-sm text-weight-medium q-mb-xs")
        ui.markdown(
            "Type roughly where it was, or leave the coordinates at zero and "
            "pin it below instead."
        ).classes("text-sm text-grey-8")
        with ui.row().classes("w-full items-end gap-3"):
            ui.number(label="Seed easting / longitude", format="%.6f") \
                .bind_value(state, "dig_georef_seed_x").classes("w-48")
            ui.number(label="Seed northing / latitude", format="%.6f") \
                .bind_value(state, "dig_georef_seed_y").classes("w-48")
            ui.button("Use the seed list / pin", icon="my_location",
                      on_click=lambda: _geo_fill_seed_from_sources()) \
                .props("flat dense no-caps") \
                .tooltip("Fills the two boxes from the pin or the seed-list "
                         "row for the quadrat named above.")
        with ui.row().classes("items-center gap-2"):
            ui.button("Check for a match", icon="search",
                      on_click=lambda: _geo_check()) \
                .props("color=primary") \
                .bind_enabled_from(state, "geo_busy",
                                   backward=lambda b: not b)
            # Enabled only with an accepted match and nothing in flight. The
            # transform reads dig_georef_ok too (NiceGUI re-evaluates it on
            # every refresh step); bound on dig_georef_ok alone the button
            # stayed live during the write and a second click wrote twice.
            ui.button("Save georeferenced outputs", icon="save",
                      on_click=lambda: _geo_write()) \
                .props("color=primary outline") \
                .bind_enabled_from(
                    state, "geo_busy",
                    backward=lambda b: bool(state.dig_georef_ok) and not b) \
                .tooltip("Writes the GeoTIFF, the world-coordinate CSV and an "
                         "audit sidecar. Enabled only once a match has been "
                         "accepted, and not while something is running.")
        # Which source the seed came from: a coordinate read off the
        # photograph must never be mistaken for one the user entered.
        ui.label("").bind_text_from(state, "geo_seed_note") \
            .bind_visibility_from(state, "geo_seed_note",
                                  backward=lambda v: bool(v)) \
            .classes("text-sm text-primary")
        ui.label("").bind_text_from(state, "dig_georef_msg") \
            .classes("text-sm").style("white-space:pre-wrap")

        # ---- the alignment editor -------------------------------------- #
        editor_holder = ui.column().classes("w-full")
        with ui.row().classes("w-full items-center gap-3") \
                .bind_visibility_from(state, "geo_placement",
                                      backward=lambda v: v is not None):
            ui.select({"highpass": "High-pass (default)",
                       "checker": "Checkerboard",
                       "colour": "Plain colour — unreliable for alignment"},
                      label="View") \
                .bind_value(state, "geo_view_mode").classes("w-64") \
                .props("dense") \
                .on_value_change(lambda: _geo_draw_editor()) \
                .tooltip("Plain colour is on record in this project as having "
                         "made a 3 mm placement look wrong: the ortho is "
                         "blurrier and colour-shifted, and the eye judges tone "
                         "rather than position. High-pass removes that.")
            ui.slider(min=0.0, max=1.0, step=0.05) \
                .bind_value(state, "geo_blend_alpha").classes("w-40") \
                .on_value_change(lambda: _geo_draw_editor()) \
                .tooltip("How strongly the quadrat is painted over the ortho.")
            ui.button("Blink", icon="flash_on",
                      on_click=lambda: _geo_blink()) \
                .props("flat dense no-caps") \
                .tooltip("Flick between the two. The eye detects motion far "
                         "better than edge continuity, so a 5 mm shift that a "
                         "blend hides jumps out under a flicker.")
            ui.select({1: "1:1", 2: "2x", 4: "4x", 8: "8x"}, label="Zoom") \
                .bind_value(state, "geo_zoom").classes("w-28").props("dense") \
                .on_value_change(lambda: _geo_draw_editor()) \
                .tooltip("One screen pixel is an ortho pixel at 1:1, which at "
                         "2.4 mm is coarser than the matcher's own 4 mm "
                         "accuracy — so a placement cannot be judged, let "
                         "alone set, that finely without zooming in.")
        with ui.row().classes("w-full items-end gap-2") \
                .bind_visibility_from(state, "geo_placement",
                                      backward=lambda v: v is not None):
            ui.number(label="Easting", format="%.3f") \
                .bind_value(state, "geo_edit_east").classes("w-40") \
                .on_value_change(lambda: _geo_from_fields())
            ui.number(label="Northing", format="%.3f") \
                .bind_value(state, "geo_edit_north").classes("w-40") \
                .on_value_change(lambda: _geo_from_fields())
            ui.number(label="Rotation (deg)", format="%.2f") \
                .bind_value(state, "geo_edit_rot").classes("w-36") \
                .on_value_change(lambda: _geo_from_fields())
            ui.button("+90°", on_click=lambda: _geo_turn(90.0)) \
                .props("flat dense no-caps") \
                .tooltip("A seed-derived placement starts north-up while "
                         "quadrats are laid at arbitrary bearings, so the "
                         "expected starting error is around 45 degrees.")
            ui.button("−90°", on_click=lambda: _geo_turn(-90.0)) \
                .props("flat dense no-caps")
            for _lbl, _dc, _dr in (("←", -1, 0), ("→", 1, 0),
                                   ("↑", 0, -1), ("↓", 0, 1)):
                ui.button(_lbl,
                          on_click=lambda c=_dc, r=_dr: _geo_nudge(c, r)) \
                    .props("flat dense no-caps") \
                    .tooltip("One ortho pixel — finer than a drag can express.")
        ui.label("").bind_text_from(state, "geo_warn_msg") \
            .bind_visibility_from(state, "geo_warn_msg",
                                  backward=lambda v: bool(v)) \
            .classes("text-sm text-orange-9") \
            .style("white-space:pre-wrap")
        ui.label("").bind_text_from(state, "geo_score_msg") \
            .bind_visibility_from(state, "geo_placement",
                                      backward=lambda v: v is not None) \
            .classes("text-sm").style("white-space:pre-wrap")
        ui.label("").bind_text_from(state, "geo_progress") \
            .bind_visibility_from(state, "geo_progress",
                                  backward=lambda v: bool(v)) \
            .classes("text-sm text-grey-7")
        with ui.row().classes("items-center gap-2") \
                .bind_visibility_from(state, "geo_placement",
                                      backward=lambda v: v is not None):
            ui.button("Keep", icon="check",
                      on_click=lambda: _geo_keep()) \
                .props("color=positive") \
                .bind_enabled_from(state, "geo_can_keep")
            ui.button("Discard", icon="close",
                      on_click=lambda: _geo_discard()) \
                .props("color=negative outline")
            ui.button("Reset", icon="undo",
                      on_click=lambda: _geo_reset()) \
                .props("flat dense no-caps") \
                .tooltip("Back to the placement this started from, in one "
                         "step — the fit, or the seed if the matcher refused.")

    # A quadrat handed over by Digitize sets state.geo_quadrat_path from
    # outside this tab; path_input_with_browse fires on_change only from its
    # own picker, so a timer reconciles the arrival and opens the panel.
    _handed_over = {"path": None}

    def _geo_notice_external_quadrat():
        p = (state.geo_quadrat_path or "").strip()
        if p == _handed_over["path"]:
            return
        _handed_over["path"] = p
        if not p:
            return
        _geo_on_quadrat_change()
        try:
            _locate_panel.value = True
        except Exception:
            pass
    ui.timer(0.5, _geo_notice_external_quadrat)

    # --- 3c. pins -------------------------------------------------------- #
    with ui.expansion("Pin quadrats on the ortho", icon="place") \
            .classes("w-full"):
        ui.label('For photographs without a coordinate: pick the photo, click roughly where it was on the ortho below. A pin is a starting point and overrides the seed list.').classes("text-sm text-grey-7")
        with ui.row().classes("w-full items-end gap-3"):
            pin_photo_select = ui.select([], label="Photo this pin is for",
                                         with_input=True) \
                .bind_value(state, "dig_pin_photo").classes("w-96") \
                .props("dense")
            ui.button("Clear pins", icon="delete_sweep",
                      on_click=lambda: _geo_clear_pins()) \
                .props("flat dense no-caps")
        ui.label("").bind_text_from(state, "dig_pin_msg").classes("text-sm")
        pin_canvas_holder = ui.column().classes("w-full")


    # --- The alignment editor --------------------------------------------- #
    # One prepared window per placement session, so a mouse move costs a
    # warp and a dot product rather than a raster read and a filter.
    _ed: dict = {"win": None, "origin": None, "gsd": None, "quad": None,
                 "inset": 0, "png": None, "disp": 1.0, "shape": None,
                 "grab": None}

    def _geo_open_from_result(m, out):
        """Open the editor on a match result, accepted or not.

        A rejected match has no transform (.matrix is None, reasons in
        .quality), so the editor starts from the seed: centred, north-up,
        at the quadrat's own GSD.
        """
        from functions import placement as _pl
        import PIL.Image as _PI
        win, origin, gsd = out.get("win"), out.get("origin"), out.get("gsd")
        if win is None:
            state.geo_editor_on = False
            return
        with _PI.open(str(state.geo_quadrat_path)) as im:
            quad = np.asarray(im.convert("L"), dtype=float)
        inset_px = int(round(float(state.dig_georef_inset or 0.0)
                             / float(state.geo_resolution or 0.001)))
        if m.matrix is not None:
            p = _pl.placement_from_matrix(m.matrix, quad.shape)
        else:
            sx, sy = out.get("seed_world", (None, None))
            if sx is None:
                state.geo_editor_on = False
                return
            p = _pl.placement_from_seed((sx, sy), quad.shape,
                                        float(state.geo_resolution or 0.001))
        _geo_open_editor(p, quad, win, origin, gsd, inset_px)
        # Say what the refusal means before anyone drags: "too few
        # correspondences" and "ambiguous" are different warnings.
        if m.matrix is None:
            why = " ".join(m.quality.reasons).lower()
            notes = []
            if "not in this ortho" in why or "correspondences were found" in why:
                notes.append(
                    "The matcher found no evidence this quadrat is in this "
                    "ortho-image. Placing it by hand is an assertion, not a "
                    "measurement — check it is the right flight and the right "
                    "day before spending time on it.")
            if "almost as good" in why or "separate places" in why:
                notes.append(
                    "The matcher found more than one placement that fits about "
                    "equally well. If the agreement score below is high anyway, "
                    "the quadrat may have been photographed on a different date "
                    "than the ortho — the clasts have moved, and dragging until "
                    "the stones line up fits a scene that is no longer there.")
            if notes:
                state.geo_warn_msg = "⚠ " + "\n⚠ ".join(notes)
            else:
                state.geo_warn_msg = ""
        else:
            state.geo_warn_msg = ""

    def _geo_offer_candidates(photo, cands):
        """Let the user settle an ambiguity the matcher could not: a person
        looking at the ortho usually resolves it at a glance."""
        with ui.dialog() as dlg, ui.card().classes("w-[42rem]"):
            ui.label(f"{photo}: {len(cands)} plausible placements") \
                .classes("text-lg font-bold")
            ui.label('The matcher could not separate these on evidence. Open one to see it on the ortho, move or discard it there; nothing is written until you save.').classes("text-sm text-grey-7")
            for i, c in enumerate(cands):
                d = c.get("from_seed_m")
                with ui.row().classes("items-center gap-3 w-full"):
                    ui.label(f"#{i + 1}").classes("font-bold w-8")
                    ui.label(
                        f"{c.get('n_inliers', 0)} correspondences · "
                        f"{(c.get('residual_m') or 0) * 100:.1f} cm residual · "
                        + (f"{d:.1f} m from the seed"
                           if d == d else "no seed given")
                    ).classes("text-sm flex-1")
                    ui.button(
                        "Open", icon="open_in_new",
                        on_click=lambda _=None, cc=c: (
                            dlg.close(), _geo_open_candidate(photo, cc))) \
                        .props("dense no-caps outline")
            ui.button("Cancel", on_click=dlg.close).props("flat dense no-caps")
        dlg.open()

    def _geo_open_candidate(photo, cand):
        """Open one chosen candidate in the editor, as if it had been found."""
        folder = (state.dig_photo_dir or "").strip()
        path = str(Path(folder) / photo) if folder else ""
        if not path or not Path(path).is_file():
            ui.notify(f"Could not find {photo} in the photo folder (Inputs, above) — pick the folder that holds it.",
                      type="warning")
            return
        state.geo_batch_placements = dict(state.geo_batch_placements or {})
        state.geo_batch_placements[photo] = {
            "matrix": cand.get("matrix"),
            "seed_world": cand.get("centre")}
        # offer_candidates=False, or this re-opens the dialog it was chosen from.
        _geo_open_from_batch(photo, offer_candidates=False)
        state.dig_georef_msg = (
            f"Opened one of {photo}'s candidate placements "
            f"({cand.get('n_inliers', 0)} correspondences). The matcher could "
            "not choose between them — check this is the right one before "
            "saving.")

    def _geo_row_clicked(e):
        """A row of the survey table was clicked."""
        try:
            row = e.args[1] if isinstance(e.args, (list, tuple)) else e.args
            photo = row.get("photo") if isinstance(row, dict) else None
        except Exception:
            photo = None
        _geo_open_from_batch(photo)

    def _geo_open_from_batch(photo, *, offer_candidates=True):
        """Open a placement the batch already found, without matching again."""
        from functions import placement as _pl
        import PIL.Image as _PI
        # An ambiguous refusal carries candidates: offer them.
        if offer_candidates:
            _cands = []
            for _r in (state.geo_batch_results or []):
                if _r.photo == photo:
                    _cands = list(getattr(_r, "candidates", []) or [])
                    break
            if _cands:
                _geo_offer_candidates(photo, _cands)
                return
        rec = (state.geo_batch_placements or {}).get(photo or "")
        if rec is None:
            ui.notify("No stored placement for that row — press *Check all quadrats* (above) first.",
                      type="warning")
            return
        folder = (state.dig_photo_dir or "").strip()
        path = str(Path(folder) / photo) if folder else ""
        if not path or not Path(path).is_file():
            ui.notify(f"Could not find {photo} in the photo folder (Inputs, above) — pick the folder that holds it.",
                      type="warning")
            return
        state.geo_quadrat_path = path
        # The select is the tab's "which photograph am I on"; leaving it on the
        # previous pick while the editor placed another one was a trap.
        try:
            if photo in (quadrat_select.options or []):
                quadrat_select.value = photo
                quadrat_select.update()
        except Exception:
            pass
        _geo_on_quadrat_change()
        with _PI.open(path) as im:
            quad = np.asarray(im.convert("L"), dtype=float)
        gsd = _sd_ortho_gsd()
        inset_px = int(round(float(state.dig_georef_inset or 0.0)
                             / float(state.geo_resolution or 0.001)))
        if rec.get("matrix") is not None:
            p = _pl.placement_from_matrix(np.asarray(rec["matrix"], float),
                                          quad.shape)
        elif rec.get("seed_world"):
            p = _pl.placement_from_seed(rec["seed_world"], quad.shape,
                                        float(state.geo_resolution or 0.001))
        else:
            ui.notify("That quadrat had no placement and no seed — add a seed-list row or a pin (Inputs, above).",
                      type="warning")
            return
        cx, cy = _pl.corners_world(p).mean(axis=0)
        margin = 0.5 + max(quad.shape) * float(state.geo_resolution or 0.001)
        job = dict(_common_job(), radius=margin)
        try:
            win, origin, _g, _crs = _read_ortho_window(job, cx, cy)
        except Exception as ex:
            ui.notify(f"Could not read the ortho there: {ex}. Pick another ortho (Inputs, above).", type="negative")
            return
        _geo_open_editor(p, quad, win, origin, gsd, inset_px)
        state.dig_georef_match = None

        # The editor panel is collapsed by default; the row opens it.
        try:
            _locate_panel.value = True
        except Exception:
            pass

        state.dig_georef_msg = (
            f"Opened {photo} from the survey check — the matcher was not run "
            "again.")

    def _sd_ortho_gsd():
        from functions import seeds as _sd
        return _sd._ortho_gsd(state.dig_georef_ortho)

    def _geo_open_editor(placement, quad, win, origin, gsd, inset_px):
        """Prepare the rasters and build the canvas once; then only move it."""
        _ed.update({"win": win, "origin": origin, "gsd": gsd, "quad": quad,
                    "inset": int(inset_px), "shape": win.shape[:2],
                    "grab": None, "shown": None, "img_el": None})
        state.geo_placement = placement
        state.geo_start_placement = placement
        state.geo_editor_on = True
        state.geo_kept = False
        _geo_sync_fields()
        _geo_prepare_rasters()
        # One persistent element repainted with set_content: rebuilding it
        # per event would be a full teardown on every mouse move.
        editor_holder.clear()
        with editor_holder:
            _ed["img_el"] = ui.interactive_image(
                _ed["base_uri"]["highpass" if state.geo_view_mode == "highpass"
                                else "colour"],
                content="",
                events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                on_mouse=_geo_on_mouse, cross=False,
            ).classes("w-full").style(
                f"max-width:{1000 * int(state.geo_zoom or 1)}px")
        _ed["shown"] = ("highpass" if state.geo_view_mode == "highpass"
                        else "colour")
        _geo_draw_editor()

    def _geo_sync_fields():
        p = state.geo_placement
        if p is None:
            return
        state.geo_edit_east = round(float(p.easting), 3)
        state.geo_edit_north = round(float(p.northing), 3)
        state.geo_edit_rot = round(float(p.rotation_deg), 2)

    def _geo_moved() -> bool:
        """Has the placement left where it started? Geometric, not a latch:
        a latch would survive a Reset and let an untouched seed be Kept."""
        from functions import placement as _pl
        a, b = state.geo_start_placement, state.geo_placement
        if a is None or b is None:
            return False
        d = _pl.placement_delta(a, b)
        return (d["translation_m"] > 0.5 * float(_ed["gsd"] or 0.0024)
                or abs(d["rotation_deg"]) > 0.1)

    def _geo_from_fields():
        from functions import placement as _pl
        p = state.geo_placement
        if p is None:
            return
        try:
            state.geo_placement = _pl.Placement(
                float(state.geo_edit_east), float(state.geo_edit_north),
                float(state.geo_edit_rot), p.scale_m_per_px, p.quad_shape)
        except (TypeError, ValueError):
            return
        _geo_draw_editor()

    def _geo_turn(d):
        from functions import placement as _pl
        if state.geo_placement is None:
            return
        state.geo_placement = _pl.set_rotation(
            state.geo_placement, state.geo_placement.rotation_deg + float(d))
        _geo_sync_fields()
        _geo_draw_editor()

    def _geo_nudge(dcol, drow):
        from functions import placement as _pl
        if state.geo_placement is None:
            return
        state.geo_placement = _pl.nudge(state.geo_placement, dcol, drow,
                                        float(_ed["gsd"] or 0.0024))
        _geo_sync_fields()
        _geo_draw_editor()

    def _geo_reset():
        state.geo_placement = state.geo_start_placement
        _geo_sync_fields()
        _geo_draw_editor()

    def _geo_blink():
        state.geo_blend_alpha = 0.0 if state.geo_blend_alpha > 0.5 else 1.0
        _geo_draw_editor()

    def _geo_decisions_path():
        """Where a season's kept/discarded decisions live, so the refusal
        denominator is recoverable and the season can be resumed."""
        if not state.current_project:
            return None
        return (project_path(state.current_project, "validation")
                / "georeference_decisions.json")

    def _geo_load_decisions():
        import json
        p = _geo_decisions_path()
        if p is None or not p.is_file():
            return
        try:
            state.geo_decisions = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def _geo_record(decision: str):
        import json
        key = Path((state.geo_quadrat_path or "").strip()).name
        if not key:
            return
        d = dict(state.geo_decisions)
        d[key] = {"decision": decision,
                  "when": __import__("datetime").datetime.now().isoformat(
                      timespec="seconds")}
        state.geo_decisions = d
        # Refresh the tally first: persisting needs a project, the count does not.
        _geo_refresh_progress()
        p = _geo_decisions_path()
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _geo_refresh_progress():
        names = _photo_names()
        d = state.geo_decisions or {}
        kept = sum(1 for k in names if (d.get(k) or {}).get("decision") == "kept")
        disc = sum(1 for k in names
                   if (d.get(k) or {}).get("decision") == "discarded")
        left = max(0, len(names) - kept - disc)
        state.geo_progress = (
            f"{len(names)} photograph(s): {kept} kept, {disc} discarded, "
            f"{left} still undecided." if names else "")

    def _geo_discard():
        state.geo_editor_on = False
        state.geo_placement = None
        state.geo_can_keep = False
        state.dig_georef_ok = False
        state.geo_warn_msg = ""
        # The image lives in its own container and must go with the controls.
        editor_holder.clear()
        _ed["img_el"] = None
        _geo_record("discarded")
        state.dig_georef_msg = ("Discarded. Nothing was written.")

    def _geo_keep():
        if not state.geo_can_keep:
            return
        state.geo_kept = True
        state.dig_georef_ok = True
        _geo_record("kept")
        ui.notify("Placement kept — Save is now enabled", type="positive")

    def _geo_prepare_rasters():
        """Rasterise both sides once, in both view modes. The quadrat is then
        moved by an SVG transform, never re-warped or re-encoded per event."""
        from functions import placement as _pl
        import cv2

        def uri(a):
            # A URL, not a data: URI: the overlay carrying it is re-sent on
            # every mouse move.
            ok, buf = cv2.imencode(".png", np.clip(a, 0, 255).astype(np.uint8))
            return _raster_url(buf.tobytes()) if ok else ""

        def hp(a):
            return cv2.normalize(a - cv2.GaussianBlur(a, (0, 0), 3), None,
                                 0, 255, cv2.NORM_MINMAX)

        win = np.asarray(_ed["win"], dtype=np.float32)
        if win.ndim == 3:
            win = win[..., :3].mean(axis=2)
        lo, hi = np.percentile(win, 1), np.percentile(win, 99)
        base = np.clip(255 * (win - lo) / max(1e-9, hi - lo), 0, 255)

        p0 = state.geo_placement
        quad = _pl._prepare_quadrat(_ed["quad"], p0, float(_ed["gsd"]),
                                    int(_ed["inset"]))
        ql, qh_ = np.percentile(quad, 1), np.percentile(quad, 99)
        q = np.clip(255 * (quad - ql) / max(1e-9, qh_ - ql), 0, 255)

        _ed["base_uri"] = {"colour": uri(base), "highpass": uri(hp(base))}
        _ed["quad_uri"] = {"colour": uri(q), "highpass": uri(hp(q))}
        _ed["quad_px"] = (q.shape[1], q.shape[0])
        _ed["win_px"] = (base.shape[1], base.shape[0])

    def _geo_svg():
        """The overlay, as a string: the transform, the outline and the
        handles, about a kilobyte. This is all a mouse move sends."""
        from functions import placement as _pl
        p = state.geo_placement
        gsd, origin = float(_ed["gsd"]), _ed["origin"]
        n = int(_ed["inset"])
        rows, cols = p.quad_shape
        qw, qh = _ed["quad_px"]

        # The quadrat raster covers the inset quadrilateral, so the frame
        # ring (equipment, absent from the ortho) is never drawn.
        M = _pl.matrix_from_placement(p)
        inset = np.array([[n, n], [cols - n, n], [n, rows - n]], dtype=float)
        world = (M[:2, :2] @ inset.T).T + M[:2, 2]
        d = np.column_stack([(world[:, 0] - float(origin[0])) / gsd,
                             (float(origin[1]) - world[:, 1]) / gsd])
        a, b = (d[1] - d[0]) / max(1, qw)
        c, dd = (d[2] - d[0]) / max(1, qh)
        mat = f"matrix({a:.6f} {b:.6f} {c:.6f} {dd:.6f} {d[0][0]:.3f} {d[0][1]:.3f})"

        mode = state.geo_view_mode
        key = "highpass" if mode == "highpass" else "colour"
        alpha = (1.0 if mode == "checker" else float(state.geo_blend_alpha))
        defs, mask = "", ""
        if mode == "checker":
            t = max(8, int(round(0.10 / gsd)))     # 10 cm tiles
            defs = (f'<defs><pattern id="pmck" width="{2*t}" height="{2*t}" '
                    f'patternUnits="userSpaceOnUse">'
                    f'<rect width="{t}" height="{t}" fill="#fff"/>'
                    f'<rect x="{t}" y="{t}" width="{t}" height="{t}" fill="#fff"/>'
                    f'</pattern><mask id="pmckm"><rect width="100%" '
                    f'height="100%" fill="url(#pmck)"/></mask></defs>')
            mask = ' mask="url(#pmckm)"'

        corners = _pl.corners_display(p, origin, gsd, 1.0)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in corners)
        handles = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" '
            f'stroke="#ffd000" stroke-width="3"/>' for x, y in corners)
        return (f'{defs}<image href="{_ed["quad_uri"][key]}" x="0" y="0" '
                f'width="{qw}" height="{qh}" transform="{mat}" '
                f'opacity="{alpha:.2f}"{mask} '
                f'preserveAspectRatio="none"/>'
                f'<polygon points="{pts}" fill="none" stroke="#d62728" '
                f'stroke-width="2"/>{handles}')

    def _geo_draw_editor():
        """Update the overlay: no raster work, no re-encoding, no raster
        sent. The quadrat moves by its own affine as an SVG transform."""
        p = state.geo_placement
        if p is None or _ed["win"] is None or _ed.get("quad_uri") is None:
            return
        el = _ed.get("img_el")
        if el is None:
            return
        z = max(1, int(state.geo_zoom or 1))
        el.style(f"max-width:{1000 * z}px")
        key = "highpass" if state.geo_view_mode == "highpass" else "colour"
        if _ed.get("shown") != key:
            el.set_source(_ed["base_uri"][key])
            _ed["shown"] = key
        el.set_content(_geo_svg())
        state.geo_can_keep = _geo_moved() or bool(state.dig_georef_match)
        _geo_measure()

    def _geo_on_nodata(p):
        """Is this placement on the ortho's blank margin? Returns a reason,
        or "" when the ground under it is real (less than a tenth blank)."""
        from functions import placement as _pl
        if _ed.get("win") is None:
            return ""
        win = np.asarray(_ed["win"], dtype=np.float32)
        if win.ndim == 3:
            blank = (win[..., -1] <= 0) if win.shape[2] == 4 else \
                (win[..., :3].sum(axis=2) <= 0)
        else:
            blank = win <= 0
        _rgb, alpha = _pl.blend_layer(_ed["quad"], p, win.shape[:2],
                                      _ed["origin"], float(_ed["gsd"]),
                                      int(_ed["inset"]))
        under = alpha > 0
        if not under.any():
            return ("This placement falls outside the ortho window entirely.")
        frac = float((blank & under).sum()) / float(under.sum())
        if frac > 0.10:
            return (f"{100 * frac:.0f}% of this placement sits on blank pixels "
                    "outside the ortho's imagery. Inside the raster is not the "
                    "same as inside the picture — move it onto real ground.")
        return ""

    def _geo_measure():
        from functions import placement as _pl
        p = state.geo_placement
        if p is None or _ed["win"] is None:
            return
        s = _pl.agreement_score(p, _ed["quad"], _ed["win"], _ed["origin"],
                                float(_ed["gsd"]), _ed["inset"])
        near = [_pl.agreement_score(_pl.translate(p, dx, dy), _ed["quad"],
                                    _ed["win"], _ed["origin"],
                                    float(_ed["gsd"]), _ed["inset"])
                for dx, dy in ((0.05, 0.0), (-0.05, 0.0))]
        state.geo_edit_score = float(s)
        peak = "on a peak" if s > max(near) else "on a shoulder — not the best fit nearby"
        gsd = float(_ed["gsd"])
        lines = [
            f"Agreement at this placement: {s:.3f}  "
            f"(±5 cm: {near[0]:.3f}, {near[1]:.3f} — {peak})",
            f"For scale: a correct placement reads about 0.40 and falls below "
            f"0.05 within 5 cm or 2°. Below {_pl.SCORE_FLOOR:.2f} a save is "
            f"refused.",
            f"One screen pixel is {gsd * 1000:.1f} mm on the ground; the arrow "
            f"keys move exactly that, which is finer than a drag can express.",
        ]
        if _geo_moved():
            lines.append(
                "⚠ This placement has been moved: the inlier count and "
                "residual above describe the FIT, not this.")
            # A few centimetres pairs each clast with its neighbour rather
            # than itself (pair_csvs' tolerance is half the median NN distance).
            d = _pl.placement_delta(state.geo_start_placement, p)
            if d["translation_m"] > 0.05:
                lines.append(
                    f"⚠ Moved {d['translation_m'] * 100:.1f} cm from the fit. "
                    "Past a few centimetres the validation pairs each clast "
                    "with its neighbour rather than itself — the statistics "
                    "come out mis-paired, not unpaired, so nothing looks wrong.")
        state.geo_score_msg = "\n".join(lines)

    def _geo_on_mouse(e):
        """Drag inside to translate, on a corner to rotate about its opposite."""
        from functions import placement as _pl
        p = state.geo_placement
        if p is None or _ed["win"] is None:
            return
        gsd, origin = float(_ed["gsd"]), _ed["origin"]
        wx, wy = _pl.world_from_display(e.image_x, e.image_y, origin, gsd, 1.0)
        typ = e.type
        if typ == "mousedown":
            cw = _pl.corners_world(p)
            d = np.hypot(cw[:, 0] - wx, cw[:, 1] - wy)
            i = int(np.argmin(d))
            # Hit-test by proximity: the SVG overlay is pointer-events:none.
            if d[i] < 12 * gsd:
                _ed["grab"] = ("corner", i)
            else:
                _ed["grab"] = ("body", (wx - p.easting, wy - p.northing))
            return
        if typ in ("mouseup", "mouseleave"):
            _ed["grab"] = None
            return
        if typ != "mousemove" or _ed["grab"] is None:
            return
        kind, payload = _ed["grab"]
        if kind == "corner":
            state.geo_placement = _pl.drag_corner(p, payload, (wx, wy))
        else:
            ox, oy = payload
            state.geo_placement = _pl.Placement(
                wx - ox, wy - oy, p.rotation_deg, p.scale_m_per_px,
                p.quad_shape)
        _geo_sync_fields()
        _geo_draw_editor()

    def _seed_crs() -> str:
        """The CRS the user's typed coordinates and seed list are in."""
        from functions import georef as _gr
        if state.geo_seed_crs_mode == "ortho":
            return _gr.ortho_crs(state.dig_georef_ortho) or ""
        return (state.geo_seed_epsg or _gr.DEFAULT_SEED_CRS).strip()

    def _load_seed_table():
        """The seed list, read with the settings the user chose."""
        from functions import seeds as _sd
        if not (state.dig_seed_csv or "").strip():
            return None
        dec = state.geo_csv_decimal
        return _sd.load_seed_table(
            state.dig_seed_csv, default_crs=_seed_crs(),
            separator=state.geo_csv_separator,
            has_header=bool(state.geo_csv_has_header),
            decimal=("" if dec in ("", "auto") else dec))

    def _geo_preview_seeds():
        """Every photograph, where its seed comes from, and whether it is
        already placed."""
        from functions import seeds as _sd
        state.geo_csv_preview = []
        preview_table.rows = []
        preview_table.update()
        table = None
        problems = []
        if (state.dig_seed_csv or "").strip():
            try:
                table = _load_seed_table()
                problems = list(table.problems)
            except Exception as ex:
                state.geo_csv_preview_msg = f"Could not read that file: {ex}"
                return

        names = _photo_names()
        if not names:
            state.geo_csv_preview_msg = (
                "Set the folder of quadrat photographs to see which of them "
                "have a seed.")
            return

        folder = Path((state.dig_photo_dir or "").strip())
        # Existing outputs may sit in the current write target or beside the
        # photographs; check both.
        from functions import georef as _gr_placed
        _placed = _gr_placed.georeferenced_listing(
            _geo_results_dir(), folder.parent / "georectified")

        def _already(name):
            hit, ok = _gr_placed.georeferenced_for(name, _placed)
            if not ok:
                return "—"
            # Name the folder when it is an archive, not the write target.
            parent = hit.parent.name
            return "yes" if parent == "georectified" else f"yes ({parent}/)"

        rows, tally = [], {"the seed list": 0, "the photograph": 0,
                           "a pin": 0, "none": 0}
        for n in names:
            r = _sd.resolve_seed(folder / n, table=table,
                                 pins=state.dig_pins or None,
                                 typed=(None, None), typed_crs=_seed_crs(),
                                 search_dirs=[folder])
            if r.ok:
                src = ("a pin" if r.provenance == "a pin"
                       else "the seed list"
                       if r.provenance == "the seed list"
                       else "the photograph")
                rows.append({"photo": n, "from": src,
                             "x": f"{r.x:.6f}", "y": f"{r.y:.6f}",
                             "crs": r.crs or "(the CRS chosen above)",
                             "placed": _already(n)})
            else:
                src = "none"
                rows.append({"photo": n, "from": "— none —", "x": "", "y": "",
                             "crs": "", "placed": _already(n)})
            tally[src] += 1

        state.geo_csv_preview = rows
        preview_table.rows = rows
        preview_table.update()

        head = (f"{len(names)} photograph(s): "
                + ", ".join(f"{v} from {k}" for k, v in tally.items() if v))
        if tally["none"]:
            head += (f" — {tally['none']} cannot be searched at all until they "
                     "get a coordinate.")
        # Seed rows naming files that are not there are invisible unless counted.
        if table is not None:
            keys = {_sd.normalise_photo(n) for n in names}
            orphans = [s.photo for k, s in table.seeds.items() if k not in keys]
            if orphans:
                problems.append(
                    f"{len(orphans)} seed row(s) name files that are not in "
                    f"this folder, so they do nothing: "
                    + ", ".join(orphans[:3])
                    + (" …" if len(orphans) > 3 else ""))
        state.geo_csv_preview_msg = "\n".join(
            [head] + [f"• {m}" for m in problems[:6]])
        _geo_update_readiness(len(names), tally["none"])

    def _geo_update_readiness(n_photos=None, n_unseeded=None):
        """Whether the survey check can run, and what it is about to cost,
        on screen before the button is pressed."""
        missing = []
        if not (state.dig_georef_ortho or "").strip():
            missing.append("an ortho-image")
        if not (state.dig_photo_dir or "").strip():
            missing.append("a folder of photographs")
        if n_photos is None:
            n_photos = len(_photo_names())
        if not missing and not n_photos:
            missing.append("photographs in that folder")
        state.geo_ready = not missing
        state.geo_not_ready = ("need " + " and ".join(missing)) if missing else ""
        if missing:
            state.geo_preflight = ""
            return
        bits = [f"{n_photos} photograph(s) to attempt"]
        if n_unseeded:
            bits.append(
                f"{n_unseeded} with no seed — "
                + ("swept across the whole ortho, roughly an hour each"
                   if state.geo_full_grid
                   else "skipped unless the whole-ortho sweep below is "
                        "turned on"))
        state.geo_preflight = "About to check: " + "; ".join(bits) + "."

    def _photo_names():
        d = (state.dig_photo_dir or "").strip()
        if not d:
            return []
        try:
            return sorted(
                p.name for p in Path(d).glob("*")
                if p.suffix.lower() in _images.PHOTO_EXTENSIONS)
        except OSError:
            return []

    def _geo_reload_photos():
        names = _photo_names()
        quadrat_select.options = names
        quadrat_select.update()
        pin_photo_select.options = names
        pin_photo_select.update()
        if names and not state.dig_pin_photo:
            state.dig_pin_photo = names[0]
        state.dig_pin_msg = (
            f"{len(names)} photo(s) available."
            if names else "Choose the folder of photographs above first.")
        _geo_refresh_progress()
        _geo_update_readiness(len(names))

    def _geo_pick_quadrat(name):
        if not name:
            return
        d = (state.dig_photo_dir or "").strip()
        if d:
            state.geo_quadrat_path = str(Path(d) / name)
            _geo_on_quadrat_change()
            # The editor panel is collapsed by default; choosing opens it.
            try:
                _locate_panel.value = True
            except Exception:
                pass

    def _geo_on_quadrat_change():
        """Auto-fill the resolution, and say whether this one needs placing."""
        from functions import georef as _gr
        # Provenance belongs to one quadrat; never let it stand over the next.
        state.geo_seed_note = ""
        p = (state.geo_quadrat_path or "").strip()
        if not p:
            state.geo_quadrat_note = ""
            return
        try:
            from functions.quadrat_validation import detect_gsd_from_path
            g = detect_gsd_from_path(p).get("gsd")
            if g and abs(float(g) - float(state.geo_resolution or 0)) > 1e-9:
                state.geo_resolution = float(g)
        except Exception:
            pass
        # The frame thickness the photograph was rectified with; a number a
        # person typed outranks the recorded one.
        try:
            from functions import exif_seed as _xs_rec
            thick, why = _gr.frame_thickness_for(
                _xs_rec.read_rectification_record(p) or {},
                state.dig_georef_inset, state.geo_inset_auto)
            if thick is not None:
                state.dig_georef_inset = thick
                state.geo_inset_auto = thick
            state.geo_inset_note = why
        except Exception:
            pass

        info = _gr.describe_georeferencing(p)
        note = info["reason"]
        if info["georeferenced"] and info.get("gsd_m"):
            note += f" Pixel size {info['gsd_m'] * 1000:.2f} mm."
        state.geo_quadrat_note = note
        # An image that already carries a position must not be offered a
        # fitted one: that would replace a survey with an estimate.
        state.geo_quadrat_needs_georef = not info["georeferenced"]
        state.dig_georef_ok = False
        state.dig_georef_match = None
        state.dig_georef_overlay = ""
        state.dig_georef_msg = ""

    def _geo_fill_seed_from_sources():
        """Put the pin, or the seed-list row, into the two coordinate boxes."""
        from functions import seeds as _sd
        name = Path((state.geo_quadrat_path or "").strip()).name
        if not name:
            ui.notify("Choose a quadrat photograph (*Locate one quadrat*, above) first", type="warning")
            return
        try:
            table = _load_seed_table()
        except Exception:
            table = None
        seed = _sd.seed_for_photo(table, name, state.dig_pins or None)
        if seed is None:
            ui.notify("No pin and no seed-list row for that photograph — add one (Inputs, above)",
                      type="warning")
            return
        state.dig_georef_seed_x = float(seed.x)
        state.dig_georef_seed_y = float(seed.y)
        if seed.crs:
            state.geo_seed_crs_mode = "specify"
            state.geo_seed_epsg = seed.crs
        ui.notify(f"Seed taken from {seed.photo}", type="positive")

    def _geo_on_ortho_change():
        _geo_render_pins()
        _geo_update_readiness()

    def _geo_clear_pins():
        state.dig_pins = {}
        state.dig_pin_msg = "All pins cleared."
        _geo_render_pins()

    def _geo_render_pins():
        """Draw the ortho once, with every pin on it."""
        from functions import georef as _gr
        pin_canvas_holder.clear()
        ortho = (state.dig_georef_ortho or "").strip()
        if not ortho:
            with pin_canvas_holder:
                ui.label("Choose the ortho-image above to pin on it.") \
                    .classes("text-sm text-grey-7")
            return
        try:
            png, disp_scale = prepare_browser_image(
                ortho, current_project=state.current_project)
            from osgeo import gdal
            ds = gdal.Open(str(ortho))
            gt = ds.GetGeoTransform()
            ds = None
        except Exception as ex:
            with pin_canvas_holder:
                ui.label(f"Could not render the ortho: {ex}") \
                    .classes("text-sm text-red-7")
            return

        def _on_pin_click(e):
            if not state.dig_pin_photo:
                state.dig_pin_msg = ("Pick which photo this pin is for "
                                     "before clicking.")
                return
            # displayed px -> original raster px -> world
            ox = float(e.image_x) * disp_scale
            oy = float(e.image_y) * disp_scale
            wx = gt[0] + ox * gt[1] + oy * gt[2]
            wy = gt[3] + ox * gt[4] + oy * gt[5]
            from functions import seeds as _sd
            pins = dict(state.dig_pins)
            pins[_sd.normalise_photo(state.dig_pin_photo)] = (
                wx, wy, _gr.ortho_crs(ortho))
            state.dig_pins = pins
            state.dig_pin_msg = (
                f"{state.dig_pin_photo} pinned at {wx:.2f}, {wy:.2f} "
                f"(the ortho's CRS). {len(pins)} pin(s) placed.")
            _geo_render_pins()

        # Markers for every pin, in displayed pixel coordinates.
        marks = []
        for key, (wx, wy, _crs) in (state.dig_pins or {}).items():
            try:
                px = (wx - gt[0]) / gt[1] / disp_scale
                py = (wy - gt[3]) / gt[5] / disp_scale
            except ZeroDivisionError:
                continue
            marks.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" '
                f'fill="none" stroke="#d62728" stroke-width="3"/>'
                f'<text x="{px + 10:.1f}" y="{py - 8:.1f}" fill="#d62728" '
                f'font-size="13">{key}</text>')
        with pin_canvas_holder:
            ui.interactive_image(
                png, content="".join(marks), events=["click"],
                on_mouse=_on_pin_click, cross=True,
            ).classes("w-full").style("max-width:1000px")

    # --- Workers: run on a thread and must not touch the GUI -------------- #
    def _read_ortho_window(job, seed_x, seed_y):
        """Read only the neighbourhood of the seed, plus GSD and CRS."""
        from osgeo import gdal
        ds = gdal.Open(str(job["ortho"]))
        if ds is None:
            raise FileNotFoundError(f"Could not open the ortho: {job['ortho']}")
        gt = ds.GetGeoTransform()
        gsd = abs(float(gt[1]))
        try:
            import PIL.Image as _PI
            with _PI.open(str(job["quadrat"])) as im:
                qw, qh = im.size
        except Exception:
            qw = qh = 0
        quad_m = max(qw, qh) * float(job["resolution"] or 0.001)
        half_px = max(32, int(round((float(job["radius"]) + quad_m) / gsd)))
        cc = int(round((seed_x - gt[0]) / gt[1]))
        rr = int(round((seed_y - gt[3]) / gt[5]))
        c0, r0 = max(0, cc - half_px), max(0, rr - half_px)
        c1 = min(ds.RasterXSize, cc + half_px)
        r1 = min(ds.RasterYSize, rr + half_px)
        if c1 <= c0 or r1 <= r0:
            raise ValueError(
                "The seed position falls outside this ortho-image. Check the "
                "coordinates and their CRS.")
        arr = ds.ReadAsArray(c0, r0, c1 - c0, r1 - r0)
        if arr is not None and arr.ndim == 3:
            arr = np.moveaxis(arr, 0, -1)
        origin = (gt[0] + c0 * gt[1], gt[3] + r0 * gt[5])
        crs = ds.GetProjection() or ""
        ds = None
        return arr, origin, gsd, crs

    def _check_worker(job):
        """Match one quadrat. Returns a plain dict; touches no GUI.

        Searches through ``locate_quadrat``, which tiles outward from the
        seed: one big window swamps the correct correspondences.
        """
        from functions import georef as _gr
        from functions import seeds as _sd
        import PIL.Image as _PI
        with _PI.open(str(job["quadrat"])) as im:
            quad = np.asarray(im.convert("L"), dtype=float)
        # Convert the seed first: the search is centred on it.
        ocrs = _gr.ortho_crs(job["ortho"])
        sx, sy, note = _gr.transform_seed(
            job["seed_x"], job["seed_y"],
            job["seed_crs"] or _gr.DEFAULT_SEED_CRS, ocrs)
        read_window, shape = _sd._window_reader(job["ortho"])
        ogsd = _sd._ortho_gsd(job["ortho"])
        m = _gr.locate_quadrat(
            quad, read_window, shape,
            quadrat_gsd_m=float(job["resolution"] or 0.0),
            ortho_gsd_m=ogsd, seed_xy=(sx, sy),
            ortho_origin_xy=_sd._ortho_origin(job["ortho"]),
            search_radius_m=float(job["radius"]),
            max_radius_m=float(job["radius"]),
            frame_inset_m=float(job["inset"]))
        # The editor needs the window its placement sits in: a tight one
        # around the result, or around the seed when the matcher refused.
        margin = 0.5 + max(quad.shape) * float(job["resolution"] or 0.001)
        if m.matrix is not None:
            cx, cy = _gr._centre_world(m.matrix, quad.shape)
        else:
            cx, cy = sx, sy
        try:
            win, origin, _g, _crs = _read_ortho_window(
                dict(job, radius=margin), cx, cy)
        except Exception:
            win, origin = None, None
        overlay = ""
        return {"match": m, "note": note, "overlay": overlay, "win": win,
                "origin": origin, "gsd": ogsd, "seed_world": (sx, sy)}

    def _write_worker(job):
        """Write the outputs for an accepted match. Touches no GUI."""
        from functions import georef as _gr
        import PIL.Image as _PI
        with _PI.open(str(job["quadrat"])) as im:
            quad_rgb = np.asarray(im.convert("RGB"))
        _, _, _, crs = _read_ortho_window(
            job, float(job["seed_x_world"]), float(job["seed_y_world"]))
        clasts = None
        # Beside the photograph, else where Detect writes it in the
        # project.
        try:
            from functions.layout import get_active_date as _gad
            vectors_dir = (project_path(state.current_project, "vectors")
                           if state.current_project else None)
            csv_found = _gr.find_clasts_csv(
                job["quadrat"], vectors_dir=vectors_dir,
                project=state.current_project, date=_gad())
        except Exception:
            csv_found = None
        if csv_found is not None:
            import pandas as pd
            clasts = pd.read_csv(csv_found)
        return _gr.write_georeferenced(
            job["match"], quad_rgb, job["out_dir"], job["stem"],
            crs_wkt=crs, clasts=clasts,
            hand_placed=job.get("hand_placed"),
            overwrite=bool(job.get("overwrite")),
            source_files={"ortho": job["ortho"], "quadrat": job["quadrat"]})

    def _check_all_worker(job):
        """Check every photograph in the folder. Touches no GUI.

        Tile-major (`check_survey`): the ortho is read once per tile. The
        progress callback only assigns to `state`; the binding loop picks
        it up, which is thread-safe where touching widgets is not.
        """
        from functions import seeds as _sd

        def progress(done, total, msg):
            state.geo_batch_done = int(done)
            state.geo_batch_total = int(total)
            state.geo_batch_now = str(msg)
            state.geo_busy = f"Checking {total} quadrat(s) — {done} of {total}"

        return _sd.check_survey(
            job["photos"], job["ortho"], job["table"],
            # Pins win over the list.
            pins=job["pins"] or None,
            quadrat_gsd_m=float(job["resolution"] or 0.001),
            search_radius_m=float(job["radius"]),
            frame_inset_m=float(job["inset"]),
            progress_fn=progress,
            should_stop=lambda: bool(state.geo_batch_stop),
            full_grid=bool(job.get("full_grid")))

    # --- Handlers: async so the loop keeps breathing while work runs ------ #
    def _common_job():
        return {
            "ortho": state.dig_georef_ortho,
            "quadrat": state.geo_quadrat_path,
            "resolution": float(state.geo_resolution or 0.001),
            "radius": float(state.dig_georef_radius or 10.0),
            "inset": float(state.dig_georef_inset or 0.0),
            "seed_crs": _seed_crs(),
        }

    def _geo_resolve_seed():
        """Where to look, and what said so (see `seeds.resolve_seed`).
        Resolved when the search runs, not when a button was pressed."""
        from functions import seeds as _sd
        folder = (state.dig_photo_dir or "").strip()
        bounds, xform = _geo_ortho_gate()
        r = _sd.resolve_seed(
            state.geo_quadrat_path,
            table=_safe_seed_table(),
            pins=state.dig_pins or None,
            typed=(state.dig_georef_seed_x, state.dig_georef_seed_y),
            typed_crs=_seed_crs(),
            search_dirs=[folder] if folder else [],
            bounds=bounds, crs_transform=xform)
        state.geo_seed_note = (
            f"Seed from {r.provenance}: {r.x:.6f}, {r.y:.6f}" if r.ok else "")
        return r.x, r.y, r.crs, r.provenance

    def _geo_ortho_gate():
        """The ortho's footprint, for refusing a fix that cannot be in it."""
        ortho = (state.dig_georef_ortho or "").strip()
        if not ortho:
            return None, None
        try:
            from functions import seeds as _sd
            from functions import georef as _gr
            bounds = _sd._ortho_bounds(ortho)
            target = _gr.ortho_crs(ortho)

            def xform(lon, lat):
                x, y, _ = _gr.transform_seed(lon, lat, "EPSG:4326", target)
                return x, y
            return bounds, xform
        except Exception:
            return None, None

    def _safe_seed_table():
        try:
            return _load_seed_table()
        except Exception:
            return None

    async def _geo_check():
        from nicegui import run
        state.dig_georef_ok = False
        state.dig_georef_match = None
        if not state.geo_quadrat_path or not state.dig_georef_ortho:
            state.dig_georef_msg = ("Set both the quadrat photograph and the "
                                    "ortho-image first.")
            return
        sx0, sy0, scrs, provenance = _geo_resolve_seed()
        if sx0 is None:
            state.dig_georef_msg = (
                "No seed for this quadrat: " + provenance + " Pin it on the "
                "ortho, add a row to the seed list, or type a coordinate.")
            return
        job = _common_job()
        job["seed_crs"] = scrs
        job["seed_x"] = sx0
        job["seed_y"] = sy0
        # A photograph's fix is looser than a typed coordinate, so give it room.
        if provenance.startswith("the photograph"):
            from functions import exif_seed as _xs
            job["radius"] = max(float(job["radius"]),
                                _xs.EXIF_SEED_RADIUS_M)
        state.geo_busy = f"Locating {Path(job['quadrat']).name}…"
        state.dig_georef_msg = ""
        try:
            out = await run.io_bound(_check_worker, job)
        except Exception as ex:
            state.dig_georef_msg = f"Could not run the match: {ex}"
            return
        finally:
            state.geo_busy = ""
        m, note = out["match"], out["note"]
        q = m.quality
        # Open the editor either way: a refused match can be placed by hand.
        _geo_open_from_result(m, out)
        if m.accepted:
            state.dig_georef_match = m
            # Keep, not Save, is what enables writing.
            state.dig_georef_ok = False
            state.dig_georef_msg = "\n".join([
                "MATCH ACCEPTED — this quadrat is in the ortho, so "
                "digitising it is worthwhile.",
                f"{q.n_inliers} of {q.n_correspondences} correspondences "
                f"agreed ({100 * q.inlier_fraction:.0f}%), average error "
                f"{q.residual_m * 100:.1f} cm, rotation "
                f"{q.rotation_deg:+.1f}°.",
            ])
            ui.notify("Quadrat located in the ortho", type="positive")
        else:
            state.dig_georef_msg = "\n".join(
                ["NO MATCH — do not spend time digitising this quadrat "
                 "until this is resolved."]
                + [f"• {r}" for r in q.reasons]
                # A wrong seed CRS looks exactly like "not in this ortho".
                + ([f"({note})"] if note else []))
            state.dig_georef_overlay = ""
            ui.notify("The quadrat could not be located — read the reasons under "
                      "the editor (above), then Check for a match again.",
                      type="warning")

    async def _geo_write():
        from nicegui import run
        from functions import georef as _gr
        m = state.dig_georef_match
        if m is None or not getattr(m, "accepted", False):
            ui.notify("Press *Check* (this panel) first", type="warning")
            return
        job = _common_job()
        job["match"] = m
        job["stem"] = Path(job["quadrat"]).stem
        job["out_dir"] = (project_path(state.current_project,
                                       "validation_georectified")
                          if state.current_project
                          else Path(job["quadrat"]).parent)
        # The same seed the search used, not the two coordinate boxes (which
        # still hold (0, 0) for a pin or an EXIF seed).
        sx0, sy0, scrs, provenance = _geo_resolve_seed()
        if sx0 is None:
            state.dig_georef_msg = (
                "No seed for this quadrat: " + provenance + " Pin it on the "
                "ortho, add a row to the seed list, or type a coordinate.")
            ui.notify("No seed for this quadrat — add a seed-list row or a pin (Inputs, above)", type="warning")
            return
        sx, sy, _n = _gr.transform_seed(
            sx0, sy0, scrs or _gr.DEFAULT_SEED_CRS,
            _gr.ortho_crs(job["ortho"]))
        job["seed_crs"] = scrs
        job["seed_x_world"], job["seed_y_world"] = sx, sy
        # A moved placement is written through its own argument, and the
        # sidecar says the quality report describes the fit, not it.
        from functions import placement as _pl
        p = state.geo_placement
        # The seed gate the matcher applies; a drag is otherwise bounded
        # only by the raster.
        if p is not None:
            radius = float(state.dig_georef_radius or 10.0)
            if provenance.startswith("the photograph"):
                # The same room the search was given.
                from functions import exif_seed as _xs
                radius = max(radius, _xs.EXIF_SEED_RADIUS_M)
            away = float(np.hypot(p.easting - sx, p.northing - sy))
            if away > radius:
                state.dig_georef_msg = (
                    f"This placement is {away:.1f} m from the seed, beyond the "
                    f"{radius:.0f} m search radius. Move it back, widen the "
                    "radius, or correct the seed.")
                ui.notify("Placement is outside the search radius — drag it back (editor, above) "
                          "or widen *Search radius* (Inputs, above)",
                          type="warning")
                return
            # Inside the raster is not the same as inside the imagery.
            bad = _geo_on_nodata(p)
            if bad:
                state.dig_georef_msg = bad
                ui.notify("Placement is on nodata — drag the quadrat onto imagery (editor, above)", type="warning")
                return
        if p is not None and (_geo_moved() or m is None
                              or getattr(m, "matrix", None) is None):
            start = state.geo_start_placement
            job["hand_placed"] = {
                "matrix": _pl.matrix_from_placement(p),
                "provenance": ("hand-edited" if getattr(m, "matrix", None)
                               is not None else "manual"),
                "agreement_score": float(state.geo_edit_score),
                "delta": (_pl.placement_delta(start, p) if start else None),
                "operator": os.environ.get("USERNAME") or "",
                "timestamp": __import__("datetime").datetime.now().isoformat(
                    timespec="seconds"),
                "view": {"mode": state.geo_view_mode,
                         "opacity": float(state.geo_blend_alpha)},
            }
        state.geo_busy = "Writing georeferenced outputs…"
        try:
            written = await run.io_bound(_write_worker, job)
        except Exception as ex:
            state.dig_georef_msg = f"Could not write outputs: {ex}"
            ui.notify("Writing failed — see the message under the editor, then Save georeferenced outputs again", type="negative")
            return
        finally:
            state.geo_busy = ""
        state.dig_georef_msg = "\n".join(
            ["Written:"] + [f"• {k}: {v}" for k, v in written.items()])
        ui.notify("Georeferenced outputs written", type="positive")

    async def _geo_draw_survey_overlay():
        """Draw every placement the survey found onto the ortho."""
        from nicegui import run
        from functions import georef as _gr
        rows = [r for r in (state.geo_batch_results or [])
                if r.status == "located" and r.matrix]
        if not rows:
            state.geo_overlay_note = ("Nothing was placed, so there is "
                                      "nothing to draw.")
            return
        folder = Path(state.dig_photo_dir or ".")
        shapes = {}
        for r in rows:
            try:
                import PIL.Image as _PI
                with _PI.open(str(folder / r.photo)) as im:
                    shapes[r.photo] = (im.size[1], im.size[0])
            except Exception:
                shapes[r.photo] = (1000, 1000)
        job = {"placements": {r.photo: r.matrix for r in rows},
               "ortho": state.dig_georef_ortho, "shapes": shapes}

        def work(j):
            fig = _gr.survey_overlay_figure(
                j["placements"], j["ortho"], quadrat_shapes=j["shapes"])
            url = _fig_to_image_url(fig, dpi=110)
            import matplotlib.pyplot as _plt
            _plt.close(fig)
            return url

        state.geo_busy = "Drawing the survey…"
        state.geo_overlay_note = ""
        try:
            url = await run.io_bound(work, job)
        except Exception as ex:
            state.geo_overlay_note = f"Could not draw it: {ex}"
            return
        finally:
            state.geo_busy = ""
        survey_overlay_holder.clear()
        with survey_overlay_holder:
            ui.image(url).classes("w-full")
        state.geo_overlay_note = (
            f"{len(rows)} placement(s). Look for two on the same ground, one "
            "off the beach, or a row marching in a line — those are seed "
            "errors a table cannot show.")

    def _geo_results_dir():
        """Where a survey's GeoTIFFs go."""
        return (project_path(state.current_project, "validation_georectified")
                if state.current_project
                else Path(state.dig_photo_dir or ".") / "georectified")

    async def _geo_save_all():
        """Write every placement the check found, in one pass."""
        from nicegui import run
        from functions import seeds as _sd
        rows = state.geo_batch_results or []
        if not rows:
            ui.notify("Press *Check all quadrats* (above) first.", type="warning")
            return
        n_ok = sum(1 for r in rows if r.status == "located")
        if not n_ok:
            ui.notify("The check placed no quadrats, so there is nothing to write. "
                      "Fix the seeds (Inputs, above), then Check all quadrats again.", type="warning")
            return
        out_dir = _geo_results_dir()
        job = dict(rows=rows, photo_dir=state.dig_photo_dir,
                   ortho=state.dig_georef_ortho, out_dir=out_dir,
                   resolution=state.geo_resolution,
                   overwrite=bool(state.geo_save_overwrite))

        def work(j):
            def prog(done, total, msg):
                state.geo_batch_done, state.geo_batch_total = done, total
                state.geo_batch_now = msg
            return _sd.save_located(
                j["rows"], j["photo_dir"], j["ortho"], j["out_dir"],
                quadrat_gsd_m=None, overwrite=j["overwrite"],
                progress_fn=prog)

        state.geo_busy = f"Saving {n_ok} placement(s)…"
        batch_progress.start(n_ok, f"Saving {n_ok} placement(s)…")
        try:
            got = await run.io_bound(work, job)
        except Exception as ex:
            state.dig_batch_msg = f"Saving failed: {ex}"
            batch_progress.hide()
            ui.notify("Saving failed — see the message under the list, then Save all located again", type="negative")
            return
        finally:
            state.geo_busy = ""
        batch_progress.finish(f"{len(got['written'])} written")
        lines = [f"Wrote {len(got['written'])} placement(s) to {out_dir}."]
        for photo, why in got["skipped"]:
            lines.append(f"• {photo}: {why}")
        state.dig_batch_msg = "\n".join(lines)
        ui.notify(f"Wrote {len(got['written'])} of {n_ok}",
                  type="positive" if not got["skipped"] else "warning")

    async def _geo_check_all():
        """Report every quadrat, never stopping at the first failure."""
        from nicegui import run
        state.dig_batch_rows = []
        state.geo_batch_results = []
        batch_table.rows = []
        batch_table.update()
        if not state.dig_photo_dir or not state.dig_georef_ortho:
            state.dig_batch_msg = (
                "Set the folder of photographs and the ortho-image above.")
            return
        photos = sorted(
            p for p in Path(state.dig_photo_dir).glob("*")
            if p.suffix.lower() in _images.PHOTO_EXTENSIONS)
        if not photos:
            state.dig_batch_msg = "No images found in that folder."
            return
        notes = []
        try:
            table = _load_seed_table()
        except Exception as ex:
            state.dig_batch_msg = f"Could not read the seed list: {ex}"
            return
        if table is not None:
            notes.extend(table.problems)
        job = _common_job()
        job.update({"photos": photos, "table": table,
                    "pins": dict(state.dig_pins or {}),
                    "full_grid": bool(state.geo_full_grid)})
        state.geo_busy = f"Checking {len(photos)} quadrat(s)…"
        state.dig_batch_msg = ""
        state.geo_batch_stop = False
        state.geo_batch_done, state.geo_batch_total = 0, len(photos)
        state.geo_batch_now = ""
        batch_progress.start(len(photos),
                             f"Checking {len(photos)} quadrat(s)…")
        try:
            results = await run.io_bound(_check_all_worker, job)
        except Exception as ex:
            state.dig_batch_msg = f"The check failed: {ex}"
            batch_progress.hide()
            return
        finally:
            state.geo_busy = ""
            state.geo_batch_stop = False
        n_stopped = sum(1 for r in results if r.status == "stopped")
        n_checked = len(results) - n_stopped
        batch_progress.finish(
            f"{n_checked} of {len(photos)} checked"
            + (" — stopped early" if n_checked < len(photos) else ""))
        label = {"located": "located",
                 "not_located": "NOT located",
                 "no_seed": "no seed given",
                 "already_placed": "already georeferenced",
                 "unreadable": "could not read",
                 "stopped": "not checked — stopped"}
        rows = [{"photo": r.photo,
                 # Say when a refusal has candidates behind it.
                 "status": label.get(r.status, r.status)
                 + (f" — {len(r.candidates)} candidates to choose from"
                    if getattr(r, "candidates", None) else ""),
                 "seed": r.seed_source or "—",
                 "detail": r.detail} for r in results]
        # Keep the results themselves: Save all and the row editor need the
        # matrices without a second match.
        state.geo_batch_results = list(results)
        state.geo_batch_placements = {
            r.photo: {"matrix": r.matrix, "seed_world": r.seed_world}
            for r in results}
        state.dig_batch_rows = rows
        batch_table.rows = rows
        batch_table.update()
        n_ok = sum(1 for r in results if r.status == "located")
        n_placed = sum(1 for r in results if r.status == "already_placed")
        state.dig_batch_msg = "\n".join(
            [f"{n_ok} of {n_checked} can be located"
             + (f"; {n_placed} already georeferenced." if n_placed else ".")
             + (f" {n_stopped} not checked (stopped)." if n_stopped else "")]
            + ([f"Seed list: {m}" for m in notes] if notes else []))

    # Fill the lists now, and again whenever the folder is edited by hand
    # (the Browse button fires on_change, typing does not).
    _last_dir = {"value": None}

    def _poll_folder():
        cur = (state.dig_photo_dir or "").strip()
        if cur != _last_dir["value"]:
            _last_dir["value"] = cur
            _geo_reload_photos()
    ui.timer(1.0, _poll_folder)
    _geo_reload_photos()
    _geo_load_decisions()
    _geo_refresh_progress()
    _geo_render_pins()

    def _seed_georef(force=False):
        """Seed the project's newest ortho and its quadrat photographs
        (validation/images, else the project's photographs)."""
        from functions import project_defaults as _pdf
        proj = state.current_project
        if not proj:
            return
        ortho = (_pdf.orthos(proj) or [None])[0]
        if ortho is not None and (force or not state.dig_georef_ortho):
            state.dig_georef_ortho = str(ortho)
            _seed_default(_geo_ortho_row._path_input, ortho, force=True)
            try:
                _geo_on_ortho_change()
            except Exception:
                pass
        photos = _pdf.validation_images(proj) or _pdf.photos(proj)
        if photos and (force or not state.dig_photo_dir):
            folder = photos[0].parent
            state.dig_photo_dir = str(folder)
            _seed_default(_geo_dir_row._path_input, folder, force=True)
            try:
                _geo_reload_photos()
            except Exception:
                pass
    _seed_georef()


# ----- Digitize tab (manual clast outlining for ground truth) ------------ #
def build_digitize_tab():
    """Manual clast digitization: outline clasts on a scaled image as
    polygons, circles or ellipses and export a CSV in the detection schema
    (measured by the same _measure_clast, so directly comparable).

    polygon: click each vertex; double-click, Enter, or a click near the
    first vertex closes it. circle: centre, then rim. ellipse: both ends
    of the major axis, then any point for the minor axis. Placed markers
    can be grabbed and moved with two clicks.
    """
    import numpy as np

    def _on_proj_change():
        state.dig_src_path = ""
        _drop_result_card("photo")
        _drop_result_card("sample")
        state.dig_out_path = ""
        # Records and in-progress shapes are kept across a project change;
        # "Clear all records" is the clean slate. The scale segments and
        # the scaling-object library belong to the project.
        state.dig_scale_segments.clear()
        _seg_io["written"].clear()
        _scale["pt"] = None
        _load_scale_library()
        try:
            _library_rows.refresh()
        except Exception:
            pass
        _seed_dig_folder(force=True)
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Digitize")
    ui.label("Outline clasts on a photograph by hand — polygon, circle or "
             "ellipse — and export a CSV in the detection schema for "
             "Validate.").classes("text-sm text-grey-7")

    # --- Is this quadrat placed? ------------------------------------- #
    # Digitising is slow and a quadrat that cannot be placed is not worth
    # outlining, so the answer is stated before the drawing tools.
    def _dig_refresh_georef_state(_e=None):
        """Say whether the loaded image already carries a position — in the
        file itself, or as a placement saved from the Georeference tab
."""
        from functions import georef as _gr
        src = (state.dig_src_path or "").strip()
        saved_dirs = []
        if state.current_project:
            try:
                saved_dirs.append(project_path(state.current_project,
                                               "validation_georectified"))
            except Exception:
                pass
        if src:
            saved_dirs.append(Path(src).parent.parent / "georectified")
            saved_dirs.append(Path(src).parent / "georectified")
        info = _gr.describe_placement(src, *saved_dirs)
        state.dig_src_needs_georef = not info["georeferenced"]
        note = info["reason"]
        if info["georeferenced"] and info.get("gsd_m"):
            note += f" Pixel size {info['gsd_m'] * 1000:.2f} mm."
        state.dig_src_georef_note = note

    def _dig_go_georeference():
        """Hand this image to the Georeference tab and switch to it."""
        if state.dig_src_path and not state.geo_quadrat_path:
            state.geo_quadrat_path = state.dig_src_path
        state.active_tab = "Georeference"

    with ui.row().classes("w-full items-center gap-2 q-mb-sm"):
        ui.label("").bind_text_from(state, "dig_src_georef_note") \
            .bind_visibility_from(state, "dig_src_georef_note",
                                  backward=lambda v: bool(v)) \
            .classes("text-sm text-grey-8")
        ui.button("Place it in the ortho", icon="my_location",
                  on_click=lambda: _dig_go_georeference()) \
            .props("flat dense no-caps color=primary") \
            .bind_visibility_from(state, "dig_src_needs_georef") \
            .tooltip("Opens the Georeference tab with this image, so you "
                     "can find out whether it can be located before "
                     "spending time outlining it.")

    # Settings every photograph shares, in one row under the folder (moved
    # there at the end of the builder).
    with ui.row().classes("w-full items-end gap-3 flex-wrap") \
            as _dig_settings_row:
        _dig_resolution_number = ui.number(
            label="Resolution (m/pixel)",
            step=0.0001, min=0.0001, format="%.5f") \
.bind_value(state, "dig_resolution") \
.classes("w-40") \
.tooltip("Image scale in metres per pixel. Auto-filled from "
                     "the photograph's GSD (sidecar or _GSD=...m in the "
                     "name); emptied for a photograph without one unless "
                     "you typed it.")

    # --- No GSD? Scale from an object ---------------------------------- #
    # A photograph with no GSD (no sidecar, no file-name tag) can still be
    # sized from an object of known length drawn on it: the segments set a
    # pixels-per-unit scale (functions.gauge), Detect writes its CSV and
    # figures in that unit, and the truth CSV follows. The section stays
    # closed, and nothing of it shows elsewhere on the tab, while the
    # photograph carries a GSD.
    from functions import gauge as _gauge
    from functions import naming as _naming
    from functions.gsd import effective_gsd as _effective_gsd
    _scale = {"pt": None, "cursor": None, "scale": None, "problem": "",
              "last_object": "", "quiet": False, "band_at": 0.0}
    # What was last written (or read) for each photograph's segments and for
    # the project library, so a recompute only writes a real change; the CSV
    # paths autosave already warned about.
    _seg_io = {"written": {}, "lib": None, "warned": set()}
    # The selection (Select mode, or rows of the clast table): "many" holds
    # every selected record, "idx" the one picked last, whose handles can be
    # dragged. "press" and "box" are a rubber band being drawn. The
    # deletions Undo can restore: ("delete", [indices], [records], len after).
    _sel = {"idx": None, "many": set(), "press": None, "box": None,
            "box_at": 0.0, "consumed": False}
    # Bumped by every change of a record's geometry, so the outline layer is
    # rebuilt only when an outline changed.
    _geom = {"ver": 0, "sig": None, "base": ""}
    # Every photograph has its truth and, per model, a label set of that
    # model's proposals; each is one entry of the photograph list.
    _TRUTH = "truth"
    _entries = {"map": {}, "order": []}
    _history = []
    _SELECT_COLOUR = "#00e5ff"

    def _notify(msg, **kw):
        """ui.notify from a worker thread has no client; stay quiet then."""
        try:
            ui.notify(msg, **kw)
        except Exception:
            pass

    def _project_root():
        return project_path(state.current_project) if state.current_project else None

    def _image_key():
        return Path(state.dig_src_path).name if state.dig_src_path else ""

    # The result area shows at most two cards: the open photograph's figures
    # (dropped when another photograph is opened) and the latest sample set.
    # A new result replaces its card; there is no history on the page.
    _result_cards = {"photo": None, "photo_key": None, "sample": None}

    def _drop_result_card(slot):
        card = _result_cards.get(slot)
        _result_cards[slot] = None
        if slot == "photo":
            _result_cards["photo_key"] = None
        if card is not None:
            try:
                card.delete()
            except Exception:
                pass

    def _segments_raw():
        """The current photograph's segments, [{"p0", "p1", "object"}]."""
        key = _image_key()
        if not key:
            return []
        return state.dig_scale_segments.setdefault(key, [])

    def _photo_gsd():
        """(metres per pixel, source) the photograph carries about itself,
        from its sidecar or its name; (None, None) for a plain one."""
        if not state.dig_src_path:
            return None, None
        try:
            info = _effective_gsd(state.dig_src_path)
        except Exception:
            return None, None
        return (float(info.gsd), info.source) if info.gsd else (None, None)

    def _library_objs():
        objs = []
        for d in state.dig_scale_library:
            o = _gauge.ScalingObject.from_dict(d)
            if o is not None and all(o.name != x.name for x in objs):
                objs.append(o)
        return objs

    def _load_scale_library():
        import json as _json
        root = _project_root()
        lib = _gauge.load_library(root) if root else _gauge.default_library()
        state.dig_scale_library = [o.to_dict() for o in lib]
        _seg_io["lib"] = _json.dumps(state.dig_scale_library, sort_keys=True,
                                     default=str)

    def _segments_from_disk(fname, csv_path):
        """Read a photograph's saved segments (once per photograph); the
        objects they name join the library when it lacks them."""
        import json as _json
        if not fname or fname in _seg_io["written"]:
            return
        doc = _gauge.read_scale_segments(csv_path)
        if not doc or not doc["segments"]:
            if not state.dig_scale_segments.get(fname):
                _seg_io["written"][fname] = _json.dumps(
                    _gauge.scale_segments_document(fname, [], []), sort_keys=True)
            return
        state.dig_scale_segments[fname] = list(doc["segments"])
        names = {str(d.get("name")) for d in state.dig_scale_library}
        added = False
        for d in doc["objects"]:
            if d.get("name") not in names:
                state.dig_scale_library.append(dict(d))
                names.add(d.get("name"))
                added = True
        if added:
            try:
                _library_rows.refresh()
            except Exception:
                pass
        _seg_io["written"][fname] = _json.dumps(
            _gauge.scale_segments_document(fname, state.dig_scale_segments[fname],
                                           state.dig_scale_library),
            sort_keys=True)

    def _persist_scale():
        """Write what changed: each photograph's segments beside its CSV
        (with the objects they name), and the library to the project."""
        import json as _json
        known = set(state.dig_image_list or [])
        for fname, segs in list(state.dig_scale_segments.items()):
            if fname not in known or not state.dig_image_dir:
                continue
            doc = _gauge.scale_segments_document(fname, segs,
                                                 state.dig_scale_library)
            body = _json.dumps(doc, sort_keys=True)
            if _seg_io["written"].get(fname) == body:
                continue
            # A photograph whose sidecar was never read is left alone
            # unless it now has segments of its own.
            if fname not in _seg_io["written"] and not doc["segments"]:
                continue
            csv = _saved_csv_for(fname)
            try:
                _gauge.write_scale_segments(csv, fname, segs,
                                            state.dig_scale_library)
                _seg_io["written"][fname] = body
                _refresh_detect_files()
            except Exception as ex:
                print(f"[digitize] could not save the scale segments of {fname}: {ex}")
        if _project_root() is not None:
            lib = _json.dumps(state.dig_scale_library, sort_keys=True, default=str)
            if _seg_io["lib"] is not None and lib != _seg_io["lib"]:
                _save_scale_library(quiet=True)
            _seg_io["lib"] = lib

    def _save_scale_library(quiet=False):
        root = _project_root()
        if root is None:
            if not quiet:
                ui.notify("No active project: the library lives in the "
                          "project's scaling_objects.json. Pick a project "
                          "in the drawer to keep it.", type="warning")
            return
        try:
            p = _gauge.save_library(root, state.dig_scale_library)
        except Exception as ex:
            _notify(f"Could not write the library: {ex}", type="negative")
            return
        if not quiet:
            ui.notify(f"Library saved to {p.name}.", type="positive")
        _refresh_detect_files()

    def _default_object_name():
        names = [str(d.get("name")) for d in state.dig_scale_library
                 if d.get("name")]
        if _scale["last_object"] in names:
            return _scale["last_object"]
        return names[0] if names else ""

    def _gauge_segments():
        segs = []
        for s in _segments_raw():
            a, b = s.get("p0"), s.get("p1")
            if not a or not b:
                continue
            segs.append(_gauge.GaugeSegment(
                (float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                str(s.get("object") or "")))
        return segs

    def _assign_missing_objects():
        for s in _segments_raw():
            if not s.get("object"):
                s["object"] = _default_object_name()
                if s["object"]:
                    _scale["last_object"] = s["object"]

    def _resolve_object_scale():
        """The scale the segments give (None without one); the result unit
        is kept within what the segments and the library allow."""
        _assign_missing_objects()
        segs = _gauge_segments()
        lib = _library_objs()
        opts = _gauge.result_unit_options(segs, lib)
        ru = state.dig_scale_result_unit
        if ru not in opts["options"]:
            ru = opts["default"]
            state.dig_scale_result_unit = ru
        scale, problem = None, ""
        if segs:
            try:
                scale = _gauge.resolve_scale(segs, lib, ru or None)
            except ValueError as ex:
                problem = str(ex)
        _scale["scale"], _scale["problem"] = scale, problem
        return scale

    def _effective_scale():
        """Which scale is in force for this photograph: ("object",
        GaugeScale) when it carries at least one assigned segment (drawn on
        purpose, so it wins); ("gsd", metres per pixel) from the
        photograph's own GSD, else from the Resolution field; (None, None)
        with nothing usable."""
        if _scale["scale"] is not None:
            return "object", _scale["scale"]
        gsd, _src = _photo_gsd()
        if gsd:
            return "gsd", gsd
        try:
            r = float(state.dig_resolution or 0)
        except (TypeError, ValueError):
            r = 0.0
        return ("gsd", r) if r > 0 else (None, None)

    def _scale_source_label():
        kind, val = _effective_scale()
        if kind == "object":
            return (f"object scale, {val.px_per_unit:.1f} px per "
                    f"{val.unit_label}")
        if kind == "gsd":
            gsd, src = _photo_gsd()
            where = {"sidecar": "sidecar", "filename": "file name"}.get(
                src, "Resolution field")
            return f"GSD {val * 1000:.3f} mm/px ({where})"
        return "no scale"

    def _truth_scale():
        """(units per pixel for the truth CSV, unit label or None): metres
        from the Resolution field or a metric-known object as usual; a
        custom unit when the object's size is unknown."""
        kind, val = _effective_scale()
        if kind == "object":
            if val.is_metric:
                return float(val.metres_per_px), None
            return 1.0 / float(val.px_per_unit), val.unit_label
        try:
            return float(state.dig_resolution or 0), None
        except (TypeError, ValueError):
            return 0.0, None

    def _scale_recompute():
        _resolve_object_scale()
        try:
            _persist_scale()
        except Exception as ex:
            print(f"[digitize] scale not saved: {ex}")
        for fn in (_refresh_scale_status, _result_unit_select.refresh,
                   _segment_rows.refresh):
            try:
                fn()
            except Exception:
                pass
        _update_overlay()
        _update_info()

    def _refresh_scale_status():
        segs = _gauge_segments()
        scale, problem = _scale["scale"], _scale["problem"]
        n = len(segs)
        if n == 0:
            txt = ("No scale segment yet: turn on Draw scale segment and "
                   "click both ends of the object on the photograph.")
            cls = "text-grey-7"
        else:
            parts = [f"{s.object_name or f'segment {k}'}, {s.px_length:.0f} px"
                     for k, s in enumerate(segs, start=1)]
            txt = f"{n} segment{'s' if n > 1 else ''}: " + "; ".join(parts)
            if scale is None:
                txt += f" — no scale: {problem}"
                cls = "text-negative"
            else:
                txt += f" — scale = {scale.px_per_unit:.1f} px per {scale.unit_label}"
                if n > 1:
                    txt += f"; segments disagree by {scale.spread * 100:.1f} %"
                    if scale.spread > 0.05:
                        txt += " (the camera is probably tilted)"
                if scale.note:
                    txt += f". {scale.note}"
                if scale.metres_per_px is not None:
                    txt += f" · 1 px = {scale.metres_per_px * 1000:.3f} mm"
                cls = "text-warning" if scale.spread > 0.05 else "text-grey-8"
        scale_status.set_text(txt)
        scale_status.classes(remove="text-grey-7 text-grey-8 text-warning "
                                    "text-negative", add=cls)

    def _clear_segments():
        _segments_raw().clear()
        _scale["pt"] = None
        _scale["cursor"] = None
        _scale_recompute()

    def _on_draw_toggle(_e=None):
        # Leaving the toggle returns the canvas to the clast mode it was in.
        _scale["pt"] = None
        _scale["cursor"] = None
        _update_overlay()
        _update_info()

    def _segment_svg(segs, k):
        """The bold segment style of the shared canvas: on-screen sizes
        times ``k`` (image pixels per screen pixel), so a segment reads
        the same at any zoom. Committed: a 4 px line on a 10 px white
        halo, filled endpoints, the object's name in a box laid along the
        segment; in progress: a haloed marker and a dashed band."""
        import html as _html
        vr, lw, halo, er, fs = 7.0 * k, 4.0 * k, 10.0 * k, 5.0 * k, 12.0 * k
        thin = max(1.0, 1.5 * k)
        sc = _scale["scale"]
        ignored = set(sc.ignored) if sc is not None else set()
        out = []

        def _box(x, y, text, colour):
            # The width is estimated from the character count (no text
            # metrics server-side).
            w = fs * (0.62 * len(text) + 1.2)
            h = fs * 1.5
            x0, y0 = x - w / 2.0, y - h
            return (
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" '
                f'height="{h:.1f}" rx="{3.0 * k:.1f}" '
                f'fill="white" fill-opacity="0.92" stroke="{colour}" '
                f'stroke-width="{thin:.1f}"/>'
                f'<text x="{x:.1f}" y="{y - h * 0.36:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="{fs:.1f}" font-weight="600" fill="#222">'
                f'{_html.escape(text)}</text>')

        for i, s in enumerate(segs):
            a, b = s.get("p0"), s.get("p1")
            if not a or not b:
                continue
            ax_, ay_, bx_, by_ = (float(a[0]), float(a[1]),
                                  float(b[0]), float(b[1]))
            name = str(s.get("object") or "")
            colour = "#9a9a9a" if name and name in ignored else "#ff8800"
            pts = f"{ax_:.1f},{ay_:.1f} {bx_:.1f},{by_:.1f}"
            out.append(
                f'<polyline points="{pts}" fill="none" '
                f'stroke="white" stroke-width="{halo:.1f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')
            out.append(
                f'<polyline points="{pts}" fill="none" '
                f'stroke="{colour}" stroke-width="{lw:.1f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')
            for px_, py_ in ((ax_, ay_), (bx_, by_)):
                out.append(
                    f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="{er:.1f}" '
                    f'fill="{colour}" stroke="white" '
                    f'stroke-width="{thin:.1f}"/>')
            # The label lies along the segment: drawn a few screen px
            # above the midpoint, then rotated about it by the segment's
            # on-screen angle (SVG rotate() is clockwise-positive, hence
            # the sign; the angle is flipped to stay readable).
            label = name or f"segment {i + 1}"
            mx, my = (ax_ + bx_) / 2.0, (ay_ + by_) / 2.0
            ang = -_gauge.segment_label_angle((ax_, ay_), (bx_, by_))
            out.append(
                f'<g transform="rotate({ang:.1f} {mx:.1f} {my:.1f})">'
                + _box(mx, my - er - 4.0 * k, label, colour) + '</g>')
        pt = _scale["pt"] if state.dig_scale_draw else None
        if pt:
            cur = _scale["cursor"]
            if cur is not None:
                band = f"{pt[0]:.1f},{pt[1]:.1f} {cur[0]:.1f},{cur[1]:.1f}"
                out.append(
                    f'<polyline points="{band}" fill="none" '
                    f'stroke="white" stroke-width="{halo * 0.8:.1f}" '
                    f'stroke-linecap="round" stroke-opacity="0.8"/>')
                out.append(
                    f'<polyline points="{band}" fill="none" '
                    f'stroke="#22aaff" stroke-width="{lw * 0.75:.1f}" '
                    f'stroke-linecap="round" '
                    f'stroke-dasharray="{8.0 * k:.1f},{5.0 * k:.1f}"/>')
            out.append(
                f'<circle cx="{pt[0]:.1f}" cy="{pt[1]:.1f}" r="{vr:.1f}" '
                f'fill="#ff3030" stroke="white" '
                f'stroke-width="{3.0 * k:.1f}"/>')
        return out

    _load_scale_library()
    _scale_section = ui.expansion("No GSD? Scale from an object",
                                  icon="straighten") \
        .classes("w-full pm-dig-scale") \
        .bind_value(state, "dig_scale_open")
    with _scale_section:
        ui.label("Mark an object of known length on the photograph; sizes "
                 "are then indicative (no perspective or lens correction).") \
            .classes("text-xs text-grey-7")

        with ui.card().classes("w-full") \
                .style("background-color: #f4f6f9; border: 1px solid #ddd;"):
            ui.label("Scaling objects").classes("text-sm font-bold")
            _unit_options = {"": "unknown"}
            _unit_options.update({u: u for u in _gauge.UNITS})

            @ui.refreshable
            def _library_rows():
                for i, d in enumerate(state.dig_scale_library):
                    with ui.row().classes("w-full items-end gap-2 flex-wrap"):
                        name_inp = ui.input(label="Name",
                                            value=str(d.get("name") or ""),
                                            placeholder="e.g. boot width") \
                            .props("dense").classes("w-56")
                        len_inp = ui.number(label="Known length",
                                            value=d.get("length"),
                                            min=0.0, step=0.01, format="%.4g") \
                            .props("dense").classes("w-36") \
                            .tooltip("Leave empty when the size is unknown.")
                        unit_sel = ui.select(_unit_options, label="Unit",
                                             value=(d.get("unit")
                                                    if d.get("unit") in _gauge.UNITS
                                                    else "")) \
                            .props("dense options-dense").classes("w-28")

                        def _set(key, val, d=d):
                            d[key] = val
                            _scale_recompute()

                        def _rename(e, d=d):
                            old, new = str(d.get("name") or ""), (e.value or "").strip()
                            if new and old and new != old:
                                # The segments follow the object they span.
                                for segs in state.dig_scale_segments.values():
                                    for s in segs:
                                        if s.get("object") == old:
                                            s["object"] = new
                                if _scale["last_object"] == old:
                                    _scale["last_object"] = new
                            _set("name", new, d)
                        name_inp.on_value_change(_rename)
                        len_inp.on_value_change(
                            lambda e, d=d: _set("length", e.value, d))
                        unit_sel.on_value_change(
                            lambda e, d=d: _set("unit", e.value or None, d))

                        def _remove(idx=i):
                            if len(state.dig_scale_library) <= 1:
                                ui.notify("Keep at least one object.",
                                          type="warning")
                                return
                            state.dig_scale_library.pop(idx)
                            _library_rows.refresh()
                            _scale_recompute()
                        ui.button(icon="delete", on_click=_remove) \
                            .props("flat dense color=negative") \
                            .tooltip("Remove this object from the library.")
            _library_rows()

            def _add_object():
                base, k = "object", 1
                names = {str(d.get("name")) for d in state.dig_scale_library}
                new = base
                while new in names:
                    k += 1
                    new = f"{base} {k}"
                state.dig_scale_library.append({"name": new, "length": None,
                                                "unit": None})
                _library_rows.refresh()
                _scale_recompute()

            with ui.row().classes("items-center gap-2"):
                ui.button("Add object", icon="add", on_click=_add_object) \
                    .props("outline dense no-caps")
                ui.button("Save to project", icon="save",
                          on_click=lambda: _save_scale_library()) \
                    .props("outline dense no-caps") \
                    .tooltip("Writes scaling_objects.json at the project "
                             "root. Every change is saved there anyway; the "
                             "objects a photograph's segments use are also "
                             "kept beside its CSV.")

        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            ui.switch("Draw scale segment") \
                .bind_value(state, "dig_scale_draw") \
                .on_value_change(_on_draw_toggle) \
                .tooltip("Two clicks on the photograph, one at each end of "
                         "the object, make a scale segment; the clast tools "
                         "wait until this is off.")

            @ui.refreshable
            def _result_unit_select():
                # Rebuilt with options and value together, never bound: a
                # select whose options lack its value nulls the state.
                segs = _gauge_segments()
                opts = _gauge.result_unit_options(segs, _library_objs())
                options = list(opts["options"])
                current = state.dig_scale_result_unit
                if current not in options:
                    current = opts["default"]

                def _on_unit(e):
                    if _scale["quiet"]:
                        return
                    if e.value:
                        state.dig_scale_result_unit = e.value
                        _scale_recompute()
                ui.select({o: o for o in options}, label="Result unit",
                          value=current if current in options else None,
                          on_change=_on_unit) \
                    .props("dense options-dense").classes("w-44") \
                    .tooltip("A physical unit when every object used has a "
                             "known length; otherwise the object that "
                             "defines the unit.")
            _result_unit_select()
            ui.button("Clear segments", icon="delete_sweep",
                      on_click=_clear_segments) \
                .props("outline dense no-caps") \
                .tooltip("Remove every scale segment from this photograph.")
        scale_status = ui.label("").classes("text-sm text-grey-7 "
                                            "pm-dig-scale-status")

        @ui.refreshable
        def _segment_rows():
            names = [str(d.get("name")) for d in state.dig_scale_library
                     if d.get("name")]
            segs = _gauge_segments()
            if not segs:
                return
            ui.label("Which object does each segment span?") \
                .classes("text-xs text-grey-7")
            for k, (raw, s) in enumerate(zip(_segments_raw(), segs), start=1):
                with ui.row().classes("items-center gap-3 flex-wrap"):
                    ui.label(f"Segment {k}").classes("text-sm font-bold w-24")
                    ui.label(f"{s.px_length:.0f} px") \
                        .classes("text-sm text-grey-7 w-20")
                    cur = raw.get("object") if raw.get("object") in names else None

                    def _on_obj(e, raw=raw):
                        if _scale["quiet"] or not e.value:
                            return
                        raw["object"] = e.value
                        _scale["last_object"] = e.value
                        _scale_recompute()
                    ui.select({n: n for n in names}, label="Scaling object",
                              value=cur, on_change=_on_obj) \
                        .props("dense options-dense").classes("w-56")
        _segment_rows()

        with ui.row().classes("items-center gap-3 flex-wrap"):
            ui.select({k: k for k in _gauge.DETECT_SCALE_OPTIONS},
                      label="Detection scale",
                      value=_gauge.DETECT_SCALE_DEFAULT) \
                .bind_value(state, "dig_detect_scale") \
                .props("dense options-dense").classes("w-36") \
                .tooltip(_gauge.DETECT_SCALE_TOOLTIP)
            ui.label(_gauge.DETECT_SCALE_HINT).classes("text-xs text-grey-7")

    # --- The folder. Each image keeps its own records.
    with ui.card().classes("w-full") \
.style("background-color: #f4f6f9; border: 1px solid #ddd;") \
            as _dig_folder_card:
        ui.label("Photographs").classes("text-sm font-bold")
        with ui.row().classes("w-full items-center gap-2"):
            dir_inp = ui.input(label="Image directory").classes("flex-grow") \
.bind_value(state, "dig_image_dir") \
.tooltip("The photographs to digitize. Pick one in the toolbar "
                         "below; each keeps its own records.")
            def _browse_dig_dir():
                current = state.dig_image_dir
                initialdir = (current if current
                              else default_starting_dir("validation"))
                picked = native_dir_picker("Pick image directory", initialdir=initialdir)
                if picked:
                    state.dig_image_dir = picked
                    _refresh_dig_dir()
            ui.button("Browse…", icon="folder_open", on_click=_browse_dig_dir).props("outline")
        dir_inp.on_value_change(lambda _: _refresh_dig_dir())

        nav_row = ui.row().classes("w-full items-center gap-2")
        with nav_row:
            prev_btn = ui.button(icon="chevron_left",
                                 on_click=lambda: _step_image(-1)) \
.props("outline dense").tooltip("Previous photograph")
            next_btn = ui.button(icon="chevron_right",
                                 on_click=lambda: _step_image(+1)) \
.props("outline dense").tooltip("Next photograph")
            image_select = ui.select([], label="Image",
                                     on_change=lambda e: _switch_to_image(e.value)) \
.classes("flex-grow")
            ui.checkbox("Autosave after each commit") \
.bind_value(state, "dig_autosave") \
.tooltip("After every clast you commit, write the CSV "
                         "automatically. Recommended — a crash never loses work.")
        nav_summary = ui.label("").classes("text-sm text-grey-7")

    # Marker / hit-zone size override, applied at _load_image time.
    with ui.row().classes("items-center gap-2 mt-1"):
        ui.label("Marker size:").classes("text-sm text-grey-7")
        size_slider = ui.slider(min=0.1, max=3.0, step=0.05) \
.bind_value(state, "dig_size_scale") \
.classes("w-48") \
.tooltip("Scales marker radius and click-buffer 'near-vertex' "
                     "zone. 1 = auto-default (~0.6 % of image diagonal). "
                     "**Dial down (0.1–0.5) for small clasts on low-res "
                     "images** — when a small image is stretched to fill "
                     "the page, the default markers become large in "
                     "screen pixels and their click buffer prevents "
                     "distinct vertex placement. Dial up for coarse data "
                     "or finger-friendly clicking. Takes effect on the "
                     "next image load (Apply).")
        ui.label().bind_text_from(state, "dig_size_scale",
                                  lambda v: f"×{v:.2f}")
        def _reload_for_scale():
            if state.dig_src_path:
                _load_image()
        ui.button("Apply", icon="refresh", on_click=_reload_for_scale).props("dense outline") \
.tooltip("Reload the current image with the new marker scale.")

    # The field's value is automatic (boot default, a photograph's own GSD)
    # until the user types one; an automatic value is emptied for a
    # photograph that carries no GSD.
    _res_auto = {"last": state.dig_resolution, "typed": False}

    def _res_typed(e):
        v = e.value
        last = _res_auto["last"]
        try:
            _res_auto["typed"] = (v is not None and v != ""
                                  and (last is None
                                       or abs(float(v) - float(last)) > 1e-9))
        except (TypeError, ValueError):
            _res_auto["typed"] = False
    _dig_resolution_number.on_value_change(_res_typed)

    def _autofill_resolution():
        if not state.dig_src_path:
            return
        g, _src = _photo_gsd()
        from functions.gsd import resolution_after_open as _after_open
        value, last, typed = _after_open(g, state.dig_resolution,
                                         _res_auto["last"], _res_auto["typed"])
        _res_auto["last"], _res_auto["typed"] = last, typed
        cur = state.dig_resolution
        if value is None:
            if cur is not None:
                state.dig_resolution = None
        elif cur is None or abs(float(value) - float(cur or 0)) > 1e-9:
            state.dig_resolution = value
    ui.timer(1.0, _autofill_resolution)

    # ---- Multi-file helpers ----
    def _expected_csv_for(image_path: str) -> str:
        """<project>/validation/<stem>_truth.csv, or beside the image when
        there is no active project."""
        stem = Path(image_path).stem
        if state.current_project:
            return str(project_path(state.current_project, "validation") / f"{stem}_truth.csv")
        return str(Path(image_path).with_name(f"{stem}_truth.csv"))

    def _set_csv_for(image_path, set_id):
        """Where a label set of this photograph is saved: the truth CSV, or
        ``<stem>_labels=<model>.csv`` beside it."""
        truth = _expected_csv_for(image_path)
        if not set_id or set_id == _TRUTH:
            return truth
        from functions import naming as _nm
        return str(Path(truth).with_name(
            _nm.label_set_csv_name(Path(image_path).stem, set_id)))

    def _model_display(set_id):
        try:
            from detectors import get_backend as _gb
            be = _gb(set_id)
            if be is not None:
                return be.info.display_name
        except Exception:
            pass
        return str(set_id)

    def _detect_outputs_for(image_path):
        """``{model: csv}`` of the Detect tab's outputs for this photograph,
        the newest per model."""
        from functions import naming as _nm
        import json as _json
        if not state.current_project:
            return {}
        vec = project_path(state.current_project, "vectors")
        stem = Path(image_path).stem
        out = {}
        if not vec.is_dir():
            return out
        found = [p for p in vec.glob(f"*{stem}*.csv")
                 if _nm.image_stem(p.name) == stem]
        for csv in sorted(found, key=lambda p: p.stat().st_mtime):
            try:
                man = _json.loads(Path(str(csv) + ".manifest.json")
                                  .read_text(encoding="utf-8"))
            except Exception:
                continue
            if man.get("model"):
                out[str(man["model"])] = str(csv)
        return out

    def _label_sets_for(fname):
        """``[(set_id, display, source csv)]`` of a photograph: the truth,
        then every model's set, saved, found among the Detect tab's
        outputs, or only in this session's memory."""
        from functions import naming as _nm
        full = os.path.join(state.dig_image_dir, fname)
        stem = Path(full).stem
        truth = Path(_expected_csv_for(full))
        sets = {}
        if truth.parent.is_dir():
            for p in truth.parent.glob(f"{stem}_labels=*.csv"):
                parsed = _nm.parse_label_set_name(p.name)
                if parsed and parsed[0] == stem:
                    sets[parsed[1]] = str(p)
        for model, csv in _detect_outputs_for(full).items():
            sets.setdefault(model, csv)
        prefix = _cache_key(fname) + "::"
        for key in state.dig_per_image:
            if key.startswith(prefix):
                sets.setdefault(key[len(prefix):], "")
        rows = [(_TRUTH, "truth", str(truth))]
        rows += sorted(((m, _model_display(m), src) for m, src in sets.items()),
                       key=lambda r: r[1].lower())
        return rows

    def _entry_key(fname, set_id):
        """The truth keeps the photograph's name; a set adds its model."""
        return fname if not set_id or set_id == _TRUTH else f"{fname}\t{set_id}"

    def _current_entry_key():
        if not (0 <= state.dig_image_idx < len(state.dig_image_list)):
            return None
        return _entry_key(state.dig_image_list[state.dig_image_idx],
                          state.dig_label_set or _TRUTH)

    def _rebuild_entries():
        """The photograph list: every photograph, once per label set."""
        order, emap, opts = [], {}, {}
        for fname in state.dig_image_list:
            rows = _label_sets_for(fname)
            for set_id, display, src in rows:
                key = _entry_key(fname, set_id)
                order.append(key)
                emap[key] = (fname, set_id, display, src)
                opts[key] = (f"{fname} · {display}" if len(rows) > 1 else fname)
        _entries["map"], _entries["order"] = emap, order
        image_select.options = opts
        cur = _current_entry_key()
        if cur in emap:
            image_select.value = cur
        image_select.update()

    def _autoload_records(full_path, set_id):
        """A label set's saved clasts, the first time it is opened in a
        session: its own CSV, else (a model's set) the Detect tab's output.
        Loaded, the canvas is the file, so autosave writes it."""
        csv = _set_csv_for(full_path, set_id)
        src, origin = (csv, None) if os.path.exists(csv) else (None, None)
        if src is None and set_id != _TRUTH:
            src = _detect_outputs_for(full_path).get(set_id)
            origin = _model_display(set_id)
        if not src:
            return []
        recs = _records_from_csv(src, full_path)
        if origin:
            for r in recs:
                r["origin"] = origin
                r["edited"] = False
        if src == csv:
            state.dig_csv_loaded.add(csv)
            state.dig_csv_owned.add(csv)
        return recs

    def _enter_model_set(set_id):
        """A model's proposals fill that model's own label set, replacing
        what it held; the truth and the other sets are left alone."""
        if not state.dig_src_path:
            return
        if (state.dig_label_set or _TRUTH) != set_id:
            _snapshot_current_image()
            state.dig_label_set = set_id
        path = _set_csv_for(state.dig_src_path, set_id)
        state.dig_out_path = path
        state.dig_csv_owned.add(path)
        state.dig_csv_loaded.add(path)
        state.dig_records = []
        _sel.update(idx=None, many=set())
        _history.clear()

    def _records_from_csv(csv_path: str, image_path: str):
        """Rebuild records from a saved truth CSV: each clast from its
        outline (``<stem>.contours.json``, image pixels, so no scale is
        needed) as an editable polygon; a clast without one (an older CSV)
        as its fitted ellipse, which needs the scale the CSV was written
        in."""
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return []
        try:
            from PIL import Image
            with Image.open(image_path) as pim:
                W, H = pim.size
        except Exception:
            return []
        try:
            from functions.digitize import (ellipse_to_cropped_mask,
                                            ellipse_params_from_row,
                                            polygon_to_cropped_mask,
                                            mask_centroid)
            from functions import clast_geometry as _CG
        except Exception:
            return []
        outlines = _CG.read_contours(csv_path) or {}
        if outlines and getattr(outlines, "frame", "pixels") != "pixels":
            outlines = {}
        try:
            side = _CG.contours_path_for(csv_path)
            if outlines and side.stat().st_mtime < Path(csv_path).stat().st_mtime - 2.0:
                outlines = {}      # the CSV was rewritten without them
        except OSError:
            outlines = {}
        resolution, unit = _truth_scale()
        ellipse_ok = True
        if "unit" in df.columns and len(df):
            csv_unit = str(df["unit"].iloc[0] or "")
            if csv_unit and csv_unit != (unit or ""):
                ellipse_ok = False
                if "clast_ID" not in df.columns or not all(
                        int(c) in outlines for c in df["clast_ID"]
                        if c == c):
                    _notify(f"{Path(csv_path).name} is in {csv_unit} units "
                            "and some clasts have no saved outline: draw "
                            "that object's scale segment (No GSD? Scale "
                            "from an object) before those can be shown.",
                            type="warning")
        if not resolution > 0:
            resolution = 0.001
        # Origins survive a reopen through the provenance sidecar; a clast
        # without an entry was drawn by hand.
        from functions import provenance as _prov
        prov = _prov.read_provenance(csv_path) or {}
        origins = prov.get("clasts") or {}
        if prov.get("models") and not state.dig_detect_runs.get(Path(image_path).name):
            state.dig_detect_runs[Path(image_path).name] = list(prov["models"])
        records = []
        for k, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                try:
                    cid = int(row["clast_ID"])
                except (KeyError, TypeError, ValueError):
                    cid = k
                o = origins.get(cid) or {}
                rec = None
                ring = outlines.get(cid)
                if ring is not None and len(ring) >= 3:
                    # Outline x, y are pixels with y up.
                    verts = [[round(float(x), 1), round(float(H) - float(y), 1)]
                             for x, y in ring]
                    mask = polygon_to_cropped_mask(verts, (H, W))
                    centre = mask_centroid(mask)
                    if centre is not None:
                        rec = {"shape": "polygon", "params": {"vertices": verts},
                               "mask": mask, "centroid_x": centre[0],
                               "centroid_y": centre[1]}
                if rec is None:
                    if not ellipse_ok:
                        continue
                    # CSV x, y are pixels with y up; Orientation is a bearing
                    # clockwise from image-up (functions.clast_geometry).
                    (cx_px, cy_px), (a, b), angle_deg = ellipse_params_from_row(
                        row, H, resolution)
                    mask = ellipse_to_cropped_mask((cx_px, cy_px), (a, b), angle_deg, (H, W))
                    rec = {"shape": "ellipse",
                           "params": {"center": [cx_px, cy_px], "axes": [a, b],
                                      "angle_deg": angle_deg},
                           "mask": mask, "centroid_x": cx_px,
                           "centroid_y": cy_px}
                rec.update({
                    "origin": o.get("origin") or _prov.HAND,
                    "edited": bool(o.get("edited")),
                    "label": (str(row["Label"]) if "Label" in df.columns
                              and isinstance(row["Label"], str) else ""),
                })
                if "Score" in df.columns:
                    try:
                        s = float(row["Score"])
                        if s == s:
                            rec["score"] = s
                    except (TypeError, ValueError):
                        pass
                records.append(rec)
            except (KeyError, ValueError, TypeError):
                continue
        return records

    def _cache_key(fname, folder=None, set_id=None):
        """The record cache's key for a photograph of the open folder (or
        of ``folder``): its full path, so a same-named photograph of
        another folder or project does not open with this one's clasts on
        the canvas."""
        base = os.path.normcase(os.path.abspath(
            os.path.join(folder or state.dig_image_dir or "", fname)))
        return base if not set_id or set_id == _TRUTH else f"{base}::{set_id}"

    def _snapshot_current_image(folder=None):
        """Save the active image's records + out path to dig_per_image
        (``folder``: the folder the image list belongs to, when the folder
        field has already moved on)."""
        if state.dig_image_idx < 0 or state.dig_image_idx >= len(state.dig_image_list):
            return
        fname = state.dig_image_list[state.dig_image_idx]
        state.dig_per_image[_cache_key(fname, folder,
                                       state.dig_label_set or _TRUTH)] = {
            "records": list(state.dig_records),
            "out_path": state.dig_out_path,
        }

    def _activate_image(idx: int, set_id=None):
        """Switch the editing context to image #idx in dig_image_list, on
        label set ``set_id`` (the one in force when None)."""
        if not (0 <= idx < len(state.dig_image_list)):
            return
        if set_id is not None:
            state.dig_label_set = set_id
        set_id = state.dig_label_set or _TRUTH
        fname = state.dig_image_list[idx]
        full_path = os.path.join(state.dig_image_dir, fname)
        state.dig_image_idx = idx
        state.dig_src_path = full_path
        if _result_cards["photo_key"] not in (None, fname):
            _drop_result_card("photo")
        # The segments are per photograph, read back from beside its CSV.
        _scale["pt"] = None
        _scale["cursor"] = None
        cached = state.dig_per_image.get(_cache_key(fname, set_id=set_id))
        target_csv = _set_csv_for(full_path, set_id)
        state.dig_out_path = (cached["out_path"]
                              if cached and cached.get("out_path")
                              else target_csv)
        # The segments sit beside the truth, whichever set is open.
        _segments_from_disk(fname, state.dig_out_path if set_id == _TRUTH
                            else _expected_csv_for(full_path))
        _resolve_object_scale()
        _autofill_resolution()
        # What this session holds for the set, else its saved clasts.
        state.dig_records = (list(cached["records"]) if cached
                             else _autoload_records(full_path, set_id))
        _sel.update(idx=None, many=set())
        _history.clear()
        state.dig_active_polygon = []
        state.dig_active_circle_center = []
        state.dig_active_ellipse_clicks = []
        state.dig_drag_target = {}
        state.dig_drag_active = False
        state.dig_drag_consumed = False
        _load_image()
        _update_nav_summary()
        _refresh_records_panel()
        _scale_recompute()
        try:
            _dig_refresh_georef_state()
        except Exception:
            pass

    def _switch_to_image(value):
        """An entry (a photograph, on one of its label sets) was picked."""
        if not value:
            return
        ent = _entries["map"].get(value)
        if ent is None:
            if value not in state.dig_image_list:
                return
            ent = (value, _TRUTH, "", "")
        fname, set_id = ent[0], ent[1]
        idx = state.dig_image_list.index(fname)
        if idx == state.dig_image_idx and set_id == (state.dig_label_set or _TRUTH):
            return
        _snapshot_current_image()
        _activate_image(idx, set_id)
        _sync_select()

    def _sync_select():
        cur = _current_entry_key()
        if cur is not None and cur in _entries["map"] and image_select.value != cur:
            image_select.value = cur

    def _step_image(delta: int):
        """Previous / Next walk every entry: each photograph's truth and
        label sets, then the next photograph."""
        if not state.dig_image_list:
            return
        order = _entries["order"] or list(state.dig_image_list)
        cur = _current_entry_key()
        i = order.index(cur) if cur in order else -1
        j = i + delta
        if not (0 <= j < len(order)):
            return
        _switch_to_image(order[j])

    _dig_dir_seen = {"dir": state.dig_image_dir}

    def _refresh_dig_dir():
        """Scan dig_image_dir for images and populate the picker."""
        if not state.dig_image_dir or not os.path.isdir(state.dig_image_dir):
            state.dig_image_list = []
            image_select.options = []
            image_select.update()
            state.dig_image_idx = -1
            _update_nav_summary()
            return
        files = list_images(state.dig_image_dir,
                            list(_images.PHOTO_EXTENSIONS))
        # The photograph open in the previous folder keeps its records
        # for when that folder is opened again.
        if _dig_dir_seen["dir"] != state.dig_image_dir:
            _snapshot_current_image(folder=_dig_dir_seen["dir"])
        state.dig_image_list = files
        # A new folder starts at its first photograph; the same folder on a
        # page rebuild restores the one that was open.
        changed = (_dig_dir_seen["dir"] != state.dig_image_dir)
        _dig_dir_seen["dir"] = state.dig_image_dir
        if files and (changed
                      or not (0 <= state.dig_image_idx < len(files))):
            state.dig_image_idx = -1
            _activate_image(0, _TRUTH)
        elif files and img_widgets.get("interactive") is None:
            _snapshot_current_image()
            _activate_image(state.dig_image_idx)
        _rebuild_entries()
        _update_nav_summary()

    def _saved_csv_for(fname):
        """The Digitize CSV a photograph of the folder saves to and reloads
        from: the one in use for the open photograph, its cached path, else
        the expected one."""
        full = os.path.join(state.dig_image_dir, fname)
        if (fname == _image_key() and state.dig_out_path
                and (state.dig_label_set or _TRUTH) == _TRUTH):
            return state.dig_out_path
        cached = state.dig_per_image.get(_cache_key(fname)) or {}
        return cached.get("out_path") or _expected_csv_for(full)

    def _update_nav_summary():
        if not state.dig_image_list:
            nav_summary.set_text("")
            _sync_sample_set_button(0)
            return
        n = len(state.dig_image_list)
        i = state.dig_image_idx
        n_done = 0
        for f in state.dig_image_list:
            csv_path = _saved_csv_for(f)
            if os.path.exists(csv_path):
                n_done += 1
        active = (state.dig_image_list[i] if 0 <= i < n else "?")
        if (state.dig_label_set or _TRUTH) != _TRUTH:
            active += f" · {_model_display(state.dig_label_set)}"
        nrec = len(state.dig_records)
        nav_summary.set_text(
            f"Image {i + 1} of {n} — {active} ({nrec} clasts so far). "
            f"{n_done}/{n} files have a saved CSV."
        )
        _sync_sample_set_button(n_done)
        _sync_load_button()
        try:
            _sync_frame_controls()
        except NameError:
            pass

    def _sync_load_button():
        """Load needs saved clasts that are not on the canvas already."""
        try:
            csv = str(state.dig_out_path or "")
            if not state.dig_src_path or not csv or not os.path.exists(csv):
                _disable(load_btn, "This photograph has no saved clasts")
            elif csv in state.dig_csv_loaded:
                _disable(load_btn, "The saved clasts are on the canvas")
            else:
                _enable(load_btn)
        except NameError:
            pass
        try:
            if (state.dig_label_set or _TRUTH) == _TRUTH:
                _disable(copy_btn, "The truth is open: open a model's set "
                                   "in the photograph list")
            elif not state.dig_records:
                _disable(copy_btn, "This set holds no clast")
            else:
                _enable(copy_btn)
        except NameError:
            pass

    def _sync_sample_set_button(n_saved):
        """Sample set needs clasts on at least one photograph of the folder."""
        try:
            if n_saved > 0 or state.dig_records:
                _enable(sample_set_btn)
            else:
                _disable(sample_set_btn, "No photograph of this folder has "
                                         "saved clasts yet")
        except NameError:
            pass

    def _truth_dataframe(H, records=None, with_positions=False):
        """The records as a truth table: metres when a GSD or a metric-known
        object is in force; otherwise the object's unit, named in a
        ``unit`` column. ``with_positions`` also returns which record each
        row is."""
        from functions.digitize import measure_records
        resolution, unit = _truth_scale()
        df, positions = measure_records(
            state.dig_records if records is None else records,
            image_height=H, resolution=resolution)
        if unit:
            df["unit"] = unit
        return (df, positions) if with_positions else df

    def _detect_runs():
        """The model runs whose proposals are on this photograph."""
        return list(state.dig_detect_runs.get(_image_key(), []))

    def _write_truth(path, H):
        """Write the truth CSV, its outlines (``.contours.json``, image
        pixels with y up, what Load rebuilds the clasts from) and its
        provenance sidecar; returns the table. The CSV is then this
        session's to autosave over."""
        from functions import provenance as _prov
        from functions import clast_geometry as _CG
        from functions.digitize import measure_records
        records = list(state.dig_records)
        resolution, unit = _truth_scale()
        df, positions, outlines = measure_records(
            records, image_height=H, resolution=resolution, with_contours=True)
        if unit:
            df["unit"] = unit
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.5f")
        _CG.write_contours(path, outlines, frame="pixels")
        _prov.write_provenance(path, _prov.clast_origins(records, positions),
                               _detect_runs(), image=state.dig_src_path)
        state.dig_csv_owned.add(str(path))
        if records and str(path) == str(state.dig_out_path or ""):
            state.dig_csv_loaded.add(str(path))   # the canvas is the file
        return df

    def _autosave_ok(path=None):
        """Autosave may write ``path``: it is on, and the CSV is absent or
        this session loaded or wrote it."""
        path = str(path or state.dig_out_path or "")
        return bool(state.dig_autosave and path
                    and (path in state.dig_csv_owned or not os.path.exists(path)))

    def _autosave_records():
        if not state.dig_autosave:
            return
        if not state.dig_records or not state.dig_out_path:
            return
        H = img_widgets.get("natural_size", (0, 0))[1] if img_widgets else 0
        if H == 0:
            return
        path = str(state.dig_out_path)
        if not _autosave_ok(path):
            if path not in _seg_io["warned"]:
                _seg_io["warned"].add(path)
                _notify(f"{Path(path).name} already holds saved clasts that "
                        "are not loaded, so autosave is paused for this "
                        "photograph and will not overwrite them. Load (bar "
                        "above the photograph) adds them to the canvas and "
                        "autosave resumes; Export CSV replaces them.",
                        type="warning", timeout=12000)
            return
        try:
            _write_truth(path, H)
        except Exception as ex:
            print(f"[digitize autosave] {ex}")

    img_widgets = {
        "interactive": None,
        "loupe": None,
        "info_label": None,
        "natural_size": (0, 0),
        "src_url": None,
        "_last_move": 0,
        # Handle / hit-zone sizes, in image pixels, scaled to the image
        # diagonal by _load_image.
        "marker_r": 6,
        "close_r": 9,
        "drag_threshold": 20,
    }

    def _shape_outline_d(shape, params):
        """(svg_path_d, svg_transform) for a completed clast; only rotated
        ellipses return a transform."""
        if shape == "polygon":
            verts = params["vertices"]
            if not verts:
                return ("", "")
            d = "M " + " L ".join(f"{v[0]} {v[1]}" for v in verts) + " Z"
            return (d, "")
        if shape == "circle":
            cx, cy = params["center"]
            r = params["radius"]
            d = (f"M {cx - r} {cy} "
                 f"A {r} {r} 0 1 0 {cx + r} {cy} "
                 f"A {r} {r} 0 1 0 {cx - r} {cy} Z")
            return (d, "")
        if shape == "ellipse":
            cx, cy = params["center"]
            a, b = params["axes"]
            angle = params.get("angle_deg", 0.0)
            d = (f"M {cx - a} {cy} "
                 f"A {a} {b} 0 1 0 {cx + a} {cy} "
                 f"A {a} {b} 0 1 0 {cx - a} {cy} Z")
            transform = f"rotate({angle} {cx} {cy})" if angle else ""
            return (d, transform)
        return ("", "")

    _frame_cache = {"path": None, "fi": None}

    def _dig_frame_inset():
        """The quadrat frame of the open photograph (functions.quadrat_frame),
        or None; read once per photograph, the overlay is rebuilt on every
        mouse move."""
        path = state.dig_src_path
        if not path:
            return None
        if _frame_cache["path"] != path:
            from functions import quadrat_frame as _qf
            _frame_cache["path"] = path
            try:
                _frame_cache["fi"] = _qf.frame_inset(path)
            except Exception:
                _frame_cache["fi"] = None
        return _frame_cache["fi"]

    def _frame_guide_svg(fi, k):
        """A dashed rectangle around what is measured, inside the frame."""
        x0, y0, x1, y1 = fi.inner
        return (f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" '
                f'height="{y1 - y0}" fill="none" stroke="#00e5ff" '
                f'stroke-width="{1.5 * k:.2f}" '
                f'stroke-dasharray="{8 * k:.1f},{5 * k:.1f}" '
                f'class="pm-dig-frame"/>')

    def _outline_attrs(transform):
        return (f' transform="{transform}"' if transform else "")

    def _outlines_svg():
        """Every record's outline, in one group: what changes only when a
        record is added, removed or reshaped."""
        parts = []
        for rec in state.dig_records:
            d, transform = _shape_outline_d(rec["shape"], rec["params"])
            if d:
                parts.append(f'<path d="{d}"{_outline_attrs(transform)} '
                             f'vector-effect="non-scaling-stroke"/>')
        if not parts:
            return ""
        return ('<g class="pm-dig-outlines" fill="rgba(46,184,114,0.06)" '
                'stroke="#1a8055" stroke-width="1.5">' + "".join(parts)
                + "</g>")

    def _marks_svg():
        """What follows the pointer: the selection, the handles of the one
        selected clast (Select mode), the shape in progress, the rubber
        band, the scale segments and the frame guide."""
        parts = []
        r = img_widgets.get("marker_r", 6)
        tgt = state.dig_drag_target or {}

        def _marker(x, y, grabbed=False):
            color = "red" if grabbed else "orange"
            return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                    f'stroke="black" stroke-width="{max(1, r / 4)}"/>')

        sel = _selected_indices()
        for i in sel:
            rec = state.dig_records[i]
            d, transform = _shape_outline_d(rec["shape"], rec["params"])
            if d:
                # A screen-pixel stroke: neighbouring selected clasts stay
                # apart at any zoom.
                parts.append(
                    f'<path d="{d}"{_outline_attrs(transform)} '
                    f'fill="rgba(0,229,255,0.28)" stroke="{_SELECT_COLOUR}" '
                    f'stroke-width="2.5" vector-effect="non-scaling-stroke" '
                    f'class="pm-dig-selected"/>')
        idx = _sel["idx"]
        if idx is not None and 0 <= idx < len(state.dig_records):
            rec = state.dig_records[idx]
            cx = rec.get("centroid_x", 0)
            cy = rec.get("centroid_y", 0)
            parts.append(
                f'<text x="{cx}" y="{cy + 5}" font-size="14" fill="white" '
                f'stroke="black" stroke-width="3" paint-order="stroke" '
                f'text-anchor="middle">{idx + 1}</text>'
                f'<text x="{cx}" y="{cy + 5}" font-size="14" fill="black" '
                f'text-anchor="middle">{idx + 1}</text>')
            if state.dig_mode == "select" and len(sel) == 1:
                params = rec.get("params", {})
                rshape = rec.get("shape", "")

                def _grabbed(v_idx):
                    return (tgt.get("source") == "record"
                            and tgt.get("record_idx") == idx
                            and tgt.get("index") == v_idx)
                if rshape == "polygon":
                    verts = params.get("vertices", [])
                    # A traced outline has closely spaced vertices: handles
                    # no wider than the spacing stay apart.
                    if len(verts) >= 3:
                        gaps = sorted(float(np.hypot(verts[k][0] - verts[k - 1][0],
                                                     verts[k][1] - verts[k - 1][1]))
                                      for k in range(len(verts)))
                        r = min(r, max(1.5, 0.4 * gaps[len(gaps) // 2]))
                    for v_idx, (vx, vy) in enumerate(verts):
                        parts.append(_marker(vx, vy, _grabbed(v_idx)))
                elif rshape in ("circle", "ellipse"):
                    vx, vy = params.get("center", [cx, cy])
                    parts.append(_marker(vx, vy, _grabbed(0)))

        def _active_marker(x, y, j, shape):
            return _marker(x, y, tgt.get("source", "active") == "active"
                           and tgt.get("shape") == shape
                           and tgt.get("index") == j)

        if state.dig_mode == "polygon" and state.dig_active_polygon:
            verts = state.dig_active_polygon
            d = "M " + " L ".join(f"{v[0]} {v[1]}" for v in verts)
            parts.append(
                f'<path d="{d}" fill="none" stroke="orange" '
                f'stroke-width="2" stroke-dasharray="4,4"/>'
            )
            # The first vertex is ringed: clicking near it closes the polygon.
            close_r = img_widgets.get("close_r", 9)
            for j, (vx, vy) in enumerate(verts):
                if j == 0 and len(verts) >= 3:
                    parts.append(
                        f'<circle cx="{vx}" cy="{vy}" r="{close_r}" '
                        f'fill="none" stroke="orange" stroke-width="2" '
                        f'stroke-dasharray="2,2"/>'
                    )
                parts.append(_active_marker(vx, vy, j, "polygon"))
        elif state.dig_mode == "circle" and state.dig_active_circle_center:
            cx, cy = state.dig_active_circle_center
            parts.append(_active_marker(cx, cy, 0, "circle"))
        elif state.dig_mode == "ellipse" and state.dig_active_ellipse_clicks:
            clicks = state.dig_active_ellipse_clicks
            if len(clicks) >= 2:
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]
                parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="orange" stroke-width="2" '
                    f'stroke-dasharray="4,4"/>'
                )
            for j, (vx, vy) in enumerate(clicks):
                parts.append(_active_marker(vx, vy, j, "ellipse"))
        box = _sel["box"]
        if box is not None:
            x0, y0, x1, y1 = box
            parts.append(
                f'<rect x="{min(x0, x1)}" y="{min(y0, y1)}" '
                f'width="{abs(x1 - x0)}" height="{abs(y1 - y0)}" '
                f'fill="rgba(0,229,255,0.10)" stroke="{_SELECT_COLOUR}" '
                f'stroke-width="1.5" stroke-dasharray="6,4" '
                f'vector-effect="non-scaling-stroke" class="pm-dig-band"/>')
        # The scale segments (No GSD? Scale from an object), in the bold
        # style: screen-pixel sizes, so they read at any zoom.
        segs = _segments_raw() if state.dig_src_path else []
        if segs or (state.dig_scale_draw and _scale["pt"]):
            k = (float(img_widgets.get("disp_scale", 1.0) or 1.0)
                 / max(0.05, float(state.dig_zoom or 1.0)))
            parts.extend(_segment_svg(segs, k))
        # The quadrat frame: what lies outside the dashed rectangle is not
        # measured.
        _fi = _dig_frame_inset()
        if _fi is not None:
            parts.append(_frame_guide_svg(
                _fi, float(img_widgets.get("disp_scale", 1.0) or 1.0)
                / max(0.05, float(state.dig_zoom or 1.0))))
        return "".join(parts)

    def _update_overlay():
        """Redraw the canvas. The outlines sit in their own layer, rebuilt
        only when a record's geometry changed; everything else sits in a
        light layer redrawn on each interaction, so a click on a photograph
        holding thousands of clasts sends a few kilobytes, not all of them."""
        iimg = img_widgets["interactive"]
        if iimg is None:
            return
        # Stored coords are in original-image pixels; the displayed PNG may
        # be downsampled by disp_scale, so the overlay is scaled to match.
        _ds = float(img_widgets.get("disp_scale", 1.0) or 1.0)

        def _wrap(body):
            if body and _ds != 1.0:
                return f'<g transform="scale({1.0 / _ds:.6f})">{body}</g>'
            return body

        sig = (_geom["ver"], _ds, tuple(id(r) for r in state.dig_records))
        base_changed = sig != _geom["sig"]
        if base_changed:
            _geom["sig"] = sig
            _geom["base"] = _wrap(_outlines_svg())
        marks = _wrap(_marks_svg())
        outline_layer = img_widgets.get("outline_layer")
        mark_layer = img_widgets.get("mark_layer")
        if outline_layer is None or mark_layer is None:
            # NiceGUI before 2.17 has no layers: one content, as before.
            iimg.set_content(_geom["base"] + marks)
            return
        if base_changed:
            outline_layer.set_content(_geom["base"])
        mark_layer.set_content(marks)

    def _apply_zoom():
        """Resize the canvas image via CSS width. Click coords stay in
        source-image pixels: NiceGUI derives image_x/y from the natural size."""
        zoom = max(0.05, float(state.dig_zoom))
        W_disp = img_widgets.get("natural_size", (0, 0))[0]
        iimg = img_widgets.get("interactive")
        if iimg is not None and W_disp > 0:
            try:
                iimg.style(
                    f"width:{int(round(W_disp * zoom))}px; "
                    f"height:auto; display:block;"
                )
            except Exception:
                pass
        _update_overlay()

    # ---- The toolbar: one sticky row attached to the image (navigation,
    # mode, zoom, in-progress actions, Detect). Previous/Next/Image are
    # moved in at the end.
    ui.separator()
    dig_toolbar = ui.row().classes("w-full items-center gap-1 flex-wrap "
                                   "pm-sticky-toolbar")
    with dig_toolbar:
        def _on_mode_change(_e):
            if state.dig_mode != "select" and _sel["many"]:
                _select(None)
            state.dig_active_polygon = []
            state.dig_active_circle_center = []
            state.dig_active_ellipse_clicks = []
            state.dig_drag_target = {}
            state.dig_drag_active = False
            state.dig_drag_consumed = False
            _update_overlay()
            _update_info()
        _mode_btns = {}

        def _set_mode(mode):
            if state.dig_mode != mode:
                state.dig_mode = mode
                _on_mode_change(None)
            _sync_mode_buttons()

        def _sync_mode_buttons():
            for m, b in _mode_btns.items():
                if m == state.dig_mode:
                    b.props(remove="flat").props("unelevated color=primary")
                else:
                    b.props(remove="unelevated color=primary").props("flat")

        _mode_btns["select"] = ui.button(
            "Select", icon="near_me", on_click=lambda: _set_mode("select")) \
            .props("dense no-caps").classes("pm-dig-mode") \
            .tooltip("Click a clast to select it, bare ground to clear. "
                     "Shift or Ctrl + click adds or removes a clast; a drag "
                     "across the photograph selects every clast whose centre "
                     "is in the box. Delete removes the selection; the "
                     "handles of a single selected clast reshape it.")
        ui.separator().props("vertical")
        with ui.button_group().props("flat").classes("pm-dig-draw-tools"):
            for _m, _icon, _tip in (
                    ("polygon", "polyline",
                     "Click each vertex; double-click, press Enter, or click "
                     "near the first vertex to close."),
                    ("circle", "radio_button_unchecked",
                     "Click the centre, then the rim."),
                    ("ellipse", "egg",
                     "Click both ends of the major axis, then any point for "
                     "the minor axis.")):
                _mode_btns[_m] = ui.button(
                    _m.capitalize(), icon=_icon,
                    on_click=lambda _e=None, m=_m: _set_mode(m)) \
                    .props("dense no-caps").classes("pm-dig-mode") \
                    .tooltip(_tip)
        _sync_mode_buttons()

    # ---- Zoom (in the same bar) ----
    with dig_toolbar:
        ui.separator().props("vertical")

        def _dig_zoom_out():
            state.dig_zoom = max(0.05, state.dig_zoom / 1.5)
            _apply_zoom()

        def _dig_zoom_reset():
            state.dig_zoom = 1.0
            _apply_zoom()

        def _dig_zoom_fit():
            # The whole photograph inside the 70 % box.
            W, H = img_widgets.get("natural_size", (1000, 1000))
            vh = float(getattr(state, "ortho_viewport_h", 900) or 900)
            target = min(1.0, (0.70 * vh - 8.0) / max(H, 1),
                         1200.0 / max(W, 1))
            state.dig_zoom = max(0.05, target)
            _apply_zoom()

        def _dig_zoom_in():
            state.dig_zoom = min(8.0, state.dig_zoom * 1.5)
            _apply_zoom()

        ui.button(icon="zoom_out", on_click=_dig_zoom_out).props("dense flat") \
.tooltip("Zoom out")
        ui.button(icon="home", on_click=_dig_zoom_reset).props("dense flat") \
.tooltip("Zoom 1.0×")
        ui.button("Fit", icon="fit_screen", on_click=_dig_zoom_fit) \
.props("dense flat no-caps") \
.tooltip("The whole photograph inside the box.")
        ui.button(icon="zoom_in", on_click=_dig_zoom_in).props("dense flat") \
.tooltip("Zoom in")
        zoom_lbl = ui.label().classes("text-grey-7 text-xs min-w-[3rem]") \
.bind_text_from(state, "dig_zoom", lambda v: f"{v:.2f}×")

    info_label = ui.label("").classes(
        "text-xs font-mono px-2 py-1 w-full pm-dig-status") \
.style(f"background:{PANEL_BG}; color:{PANEL_TEXT}; border-radius:4px;")

    def _update_info():
        n_done = len(state.dig_records)
        if state.dig_scale_draw:
            if _scale["pt"] is None:
                msg = (f"{n_done} clasts digitized | Scale segment: click "
                       f"the first end of the object.")
            else:
                msg = (f"{n_done} clasts digitized | Scale segment: first "
                       f"end set, click the other end.")
        elif state.dig_mode == "polygon":
            n_active = len(state.dig_active_polygon)
            if n_active == 0:
                msg = (f"{n_done} clasts digitized. Click vertices around "
                       f"a clast (polygon mode); ‘Done polygon’ to close.")
            else:
                msg = (f"{n_done} digitized | In progress: polygon with "
                       f"{n_active} vertices. Click more or ‘Done polygon’.")
        elif state.dig_mode == "select":
            n_sel = len(_selected_indices())
            if n_sel == 0:
                msg = (f"{n_done} digitized. Click a clast to select it; "
                       f"Shift + click adds one, a drag selects a box.")
            elif n_sel == 1 and _sel["idx"] is not None:
                msg = (f"{n_done} digitized | Clast #{_sel['idx'] + 1} "
                       f"selected: drag a handle to reshape it, Delete "
                       f"removes it.")
            else:
                msg = (f"{n_done} digitized | {n_sel:,} clasts selected: "
                       f"Delete removes them all.")
        elif state.dig_mode == "circle":
            if not state.dig_active_circle_center:
                msg = (f"{n_done} digitized. Click the clast centre, "
                       f"then click the rim.")
            else:
                msg = (f"{n_done} digitized | Centre set. Click the rim "
                       f"to complete.")
        else:  # ellipse
            n_e = len(state.dig_active_ellipse_clicks)
            if n_e == 0:
                msg = (f"{n_done} digitized. Click the first end of the major axis.")
            elif n_e == 1:
                msg = (f"{n_done} digitized | First end set. Click the "
                       f"other end of the major axis.")
            else:  # n_e == 2
                msg = (f"{n_done} digitized | Major axis set. Click any "
                       f"point to set the minor-axis length (perpendicular "
                       f"distance to the major axis).")
        if state.dig_drag_active and state.dig_drag_target:
            src = state.dig_drag_target.get("source", "active")
            suffix = " (committed)" if src == "record" else ""
            msg += f"  Dragging vertex{suffix} — release to place."
        # Which scale is in force, and whether the truth leaves metres.
        msg += f"  |  Scale: {_scale_source_label()}"
        _res, unit = _truth_scale()
        if unit:
            msg += f" — the truth CSV is in {unit} units, not metres"
        info_label.set_text(msg)

    # ---- Vertex drag helpers ----
    def _find_drag_target(x, y, threshold=None, search_records=True):
        """A drag-target dict if (x, y) is near a placed marker, else None.
        In-progress shape first, then committed records. Threshold in
        image pixels."""
        if threshold is None:
            threshold = img_widgets.get("drag_threshold", 20)
        thr2 = threshold * threshold

        # --- In-progress (active) shape vertices ---
        if state.dig_mode == "polygon":
            for i, (vx, vy) in enumerate(state.dig_active_polygon):
                if (vx - x) ** 2 + (vy - y) ** 2 <= thr2:
                    return {"source": "active",
                            "shape": "polygon", "index": i}
        elif state.dig_mode == "circle":
            if state.dig_active_circle_center:
                vx, vy = state.dig_active_circle_center
                if (vx - x) ** 2 + (vy - y) ** 2 <= thr2:
                    return {"source": "active",
                            "shape": "circle", "index": 0}
        elif state.dig_mode == "ellipse":
            for i, (vx, vy) in enumerate(state.dig_active_ellipse_clicks):
                if (vx - x) ** 2 + (vy - y) ** 2 <= thr2:
                    return {"source": "active",
                            "shape": "ellipse", "index": i}

        if not search_records:
            return None

        # --- The handles of the one selected clast, in Select mode ---
        sel = _selected_indices()
        if state.dig_mode != "select" or len(sel) != 1:
            return None
        for rec_idx in sel:
            rec = state.dig_records[rec_idx]
            rshape = rec.get("shape", "")
            params = rec.get("params", {})
            if rshape == "polygon":
                # The nearest vertex: a traced outline has several in reach.
                best, best_d2 = None, thr2
                for v_idx, (vx, vy) in enumerate(
                        params.get("vertices", [])):
                    d2 = (vx - x) ** 2 + (vy - y) ** 2
                    if d2 <= best_d2:
                        best, best_d2 = v_idx, d2
                if best is not None:
                    return {"source": "record",
                            "shape": "polygon",
                            "record_idx": rec_idx,
                            "index": best}
            elif rshape == "circle":
                vx, vy = params.get("center", [0, 0])
                if (vx - x) ** 2 + (vy - y) ** 2 <= thr2:
                    return {"source": "record",
                            "shape": "circle",
                            "record_idx": rec_idx,
                            "index": 0}   # center only
            elif rshape == "ellipse":
                vx, vy = params.get("center", [0, 0])
                if (vx - x) ** 2 + (vy - y) ** 2 <= thr2:
                    return {"source": "record",
                            "shape": "ellipse",
                            "record_idx": rec_idx,
                            "index": 0}   # center only
        return None

    def _apply_drag_move(x, y):
        """Move the dragged vertex to (x, y); the mask is recomputed by
        _finalize_drag() on mouseup."""
        tgt = state.dig_drag_target
        if not tgt:
            return
        src = tgt.get("source", "active")
        if src == "active":
            shape = tgt["shape"]
            idx = tgt["index"]
            if shape == "polygon":
                if 0 <= idx < len(state.dig_active_polygon):
                    state.dig_active_polygon[idx] = [x, y]
            elif shape == "circle":
                state.dig_active_circle_center = [x, y]
            elif shape == "ellipse":
                if 0 <= idx < len(state.dig_active_ellipse_clicks):
                    state.dig_active_ellipse_clicks[idx] = [x, y]
        else:  # "record"
            rec_idx = tgt.get("record_idx", -1)
            if 0 <= rec_idx < len(state.dig_records):
                rec = state.dig_records[rec_idx]
                rshape = rec.get("shape", "")
                params = rec.get("params", {})
                idx = tgt["index"]
                if rshape == "polygon":
                    verts = params.get("vertices", [])
                    if 0 <= idx < len(verts):
                        verts[idx] = [x, y]
                elif rshape in ("circle", "ellipse"):
                    params["center"] = [x, y]

    def _finalize_drag():
        """Recompute mask and centroid for a committed record after a drag."""
        tgt = state.dig_drag_target
        if not tgt or tgt.get("source") != "record":
            return
        rec_idx = tgt.get("record_idx", -1)
        if not (0 <= rec_idx < len(state.dig_records)):
            return
        rec = state.dig_records[rec_idx]
        rshape = rec.get("shape", "")
        params = rec.get("params", {})
        H = img_widgets["natural_size"][1]
        W = img_widgets["natural_size"][0]
        try:
            if rshape == "polygon":
                from functions.digitize import (polygon_to_cropped_mask,
                                                mask_centroid)
                verts = params.get("vertices", [])
                if len(verts) >= 3:
                    new_mask = polygon_to_cropped_mask(verts, (H, W))
                    rec["mask"] = new_mask
                    c = mask_centroid(new_mask)
                    if c is not None:
                        rec["centroid_x"], rec["centroid_y"] = c
            elif rshape == "circle":
                from functions.digitize import circle_to_cropped_mask
                cx, cy = params["center"]
                r = params.get("radius", 10)
                new_mask = circle_to_cropped_mask((cx, cy), r, (H, W))
                rec["mask"] = new_mask
                rec["centroid_x"] = float(cx)
                rec["centroid_y"] = float(cy)
            elif rshape == "ellipse":
                from functions.digitize import ellipse_to_cropped_mask
                cx, cy = params["center"]
                a, b = params.get("axes", [10, 5])
                angle = params.get("angle_deg", 0)
                new_mask = ellipse_to_cropped_mask((cx, cy), (a, b), angle, (H, W))
                rec["mask"] = new_mask
                rec["centroid_x"] = float(cx)
                rec["centroid_y"] = float(cy)
        except Exception as ex:
            ui.notify(f"Mask recompute failed: {ex}. Undo (bar above) and draw the shape again.", type="warning")
        _geom["ver"] += 1
        _autosave_records()

    def _end_drag(x, y):
        """The mouseup of a handle drag: place the handle, rebuild the mask,
        and swallow the click that follows if the pointer moved."""
        tgt = dict(state.dig_drag_target)
        sx, sy = img_widgets.get("_drag_start", (x, y))
        moved = (sx - x) ** 2 + (sy - y) ** 2 > 9  # > ~3 px
        _apply_drag_move(x, y)
        if moved and tgt.get("source") == "record":
            ri = tgt.get("record_idx", -1)
            if 0 <= ri < len(state.dig_records):
                from functions import provenance as _prov
                if _prov.record_origin(state.dig_records[ri]) != _prov.HAND:
                    # A proposal whose geometry was changed by hand.
                    state.dig_records[ri]["edited"] = True
        _finalize_drag()
        state.dig_drag_active = False
        state.dig_drag_target = {}
        # Suppress the following 'click' only if the cursor moved:
        # a stationary press must still close on vertex 0.
        img_widgets.pop("_drag_start", None)
        state.dig_drag_consumed = moved
        _update_overlay()
        _update_info()
        if tgt.get("source") == "record":
            _refresh_records_panel()

    def _on_select_mouse(e, x, y):
        """Select mode. A click picks the smallest clast under it and bare
        ground clears; Shift or Ctrl + click adds or removes one. A drag
        that starts on a handle of the one selected clast reshapes it; any
        other drag draws a box and selects every clast whose centre is in
        it (Shift or Ctrl keeps the selection and adds to it)."""
        additive = bool(getattr(e, "shift", False) or getattr(e, "ctrl", False)
                        or getattr(e, "meta", False))
        if e.type == "mousedown":
            grab = _find_drag_target(x, y)
            if grab:
                state.dig_drag_target = grab
                state.dig_drag_active = True
                state.dig_drag_consumed = False
                img_widgets["_drag_start"] = (x, y)
                _update_overlay()
                return
            _sel["press"] = (x, y)
            _sel["box"] = None
            return
        if e.type == "mousemove":
            if state.dig_drag_active and state.dig_drag_target:
                _apply_drag_move(x, y)
                _update_overlay()
                return
            press = _sel["press"]
            if press is None:
                return
            if not getattr(e, "buttons", 0):
                # The button came up outside the photograph.
                _sel["press"] = None
                if _sel["box"] is not None:
                    _sel["box"] = None
                    _update_overlay()
                return
            # A box starts once the pointer has moved six screen pixels.
            _ds = float(img_widgets.get("disp_scale", 1.0) or 1.0)
            thr = 6.0 * _ds / max(0.05, float(state.dig_zoom or 1.0))
            if (_sel["box"] is None
                    and (x - press[0]) ** 2 + (y - press[1]) ** 2 < thr * thr):
                return
            _sel["box"] = (press[0], press[1], x, y)
            import time as _time
            now = _time.monotonic()
            if now - _sel["box_at"] >= 0.04:
                _sel["box_at"] = now
                _update_overlay()
            return
        if e.type == "mouseup":
            if state.dig_drag_active and state.dig_drag_target:
                _end_drag(x, y)
                return
            box = _sel["box"]
            _sel["press"] = None
            _sel["box"] = None
            if box is not None:
                x0, x1 = sorted((box[0], x))
                y0, y1 = sorted((box[1], y))
                hits = [i for i, rec in enumerate(state.dig_records)
                        if x0 <= rec.get("centroid_x", -1) <= x1
                        and y0 <= rec.get("centroid_y", -1) <= y1]
                _sel["consumed"] = True
                _select_many(hits, add=additive)
            return
        if e.type == "click":
            if _sel["consumed"] or state.dig_drag_consumed:
                _sel["consumed"] = False
                state.dig_drag_consumed = False
                return
            hit = _hit_record(x, y)
            if additive:
                if hit is not None:
                    _toggle_selected(hit)
            else:
                _select(hit)

    def _on_mouse(e):
        # Event coords are in displayed-PNG pixels; scale by disp_scale
        # back to original-image pixels.
        _ds = float(img_widgets.get("disp_scale", 1.0) or 1.0)

        if e.type == "dblclick":
            if state.dig_scale_draw:
                return
            if state.dig_mode == "polygon" and len(state.dig_active_polygon) >= 3:
                _commit_polygon(state.dig_active_polygon)
                state.dig_active_polygon = []
                _update_overlay()
                _update_info()
            return

        # image_x/y may be None on rapid events (a mouseup before NiceGUI
        # attached coords); a TypeError here would silently cancel the drag.
        x = round(float(getattr(e, "image_x", 0) or 0) * _ds)
        y = round(float(getattr(e, "image_y", 0) or 0) * _ds)

        if state.dig_scale_draw:
            # Two-click scale segment (No GSD? Scale from an object): the
            # first click marks one end, the second commits; the clast
            # tools and their drags wait until the toggle is off.
            if e.type == "click":
                if _scale["pt"] is None:
                    _scale["pt"] = [x, y]
                    _scale["cursor"] = None
                else:
                    x0, y0 = _scale["pt"]
                    if (x0 - x) ** 2 + (y0 - y) ** 2 < 1.0:
                        ui.notify("Both ends are on the same pixel: click "
                                  "the other end of the object (image, "
                                  "Work surface).", type="warning")
                        return
                    _segments_raw().append({"p0": [x0, y0], "p1": [x, y],
                                            "object": _default_object_name()})
                    _scale["pt"] = None
                    _scale["cursor"] = None
                    _scale_recompute()
                    return
                _update_overlay()
                _update_info()
            elif e.type == "mousemove" and _scale["pt"] is not None:
                # The band follows the cursor, at most every 40 ms.
                import time as _time
                now = _time.monotonic()
                if now - _scale["band_at"] >= 0.04:
                    _scale["band_at"] = now
                    _scale["cursor"] = (x, y)
                    _update_overlay()
            return

        if state.dig_mode == "select":
            _on_select_mouse(e, x, y)
            return

        if e.type == "mousedown":
            # Committed records are only grabbable when nothing is in progress.
            _active_drawing = bool(
                state.dig_active_polygon
                or state.dig_active_circle_center
                or state.dig_active_ellipse_clicks
            )
            grab = _find_drag_target(x, y,
                                     search_records=not _active_drawing)
            if grab:
                state.dig_drag_target = grab
                state.dig_drag_active = True
                state.dig_drag_consumed = False
                img_widgets["_drag_start"] = (x, y)
                _update_overlay()
                _update_info()
            return

        if e.type == "mousemove":
            if state.dig_drag_active and state.dig_drag_target:
                _apply_drag_move(x, y)
                _update_overlay()
            return

        if e.type == "mouseup":
            if state.dig_drag_active and state.dig_drag_target:
                _end_drag(x, y)
            return

        if e.type == "click":
            # The click that follows a drag's mouseup.
            if state.dig_drag_consumed:
                state.dig_drag_consumed = False
                return

            # Close-on-first-vertex takes priority; its zone is tighter
            # than the drag zone.
            if (state.dig_mode == "polygon"
                    and len(state.dig_active_polygon) >= 3):
                fx, fy = state.dig_active_polygon[0]
                close_thr = max(8, img_widgets.get("drag_threshold", 20) * 0.6)
                if (fx - x) ** 2 + (fy - y) ** 2 <= close_thr * close_thr:
                    _commit_polygon(state.dig_active_polygon)
                    state.dig_active_polygon = []
                    _update_overlay()
                    _update_info()
                    return

            if state.dig_mode == "polygon":
                state.dig_active_polygon.append([x, y])
            elif state.dig_mode == "circle":
                if not state.dig_active_circle_center:
                    state.dig_active_circle_center = [x, y]
                else:
                    cx, cy = state.dig_active_circle_center
                    radius = float(np.hypot(x - cx, y - cy))
                    if radius < 1:
                        ui.notify("Radius too small — click farther from the centre (image, Work surface).", type="warning")
                        return
                    _commit_circle((cx, cy), radius)
                    state.dig_active_circle_center = []
            elif state.dig_mode == "ellipse":
                state.dig_active_ellipse_clicks.append([x, y])
                if len(state.dig_active_ellipse_clicks) == 3:
                    pts = state.dig_active_ellipse_clicks
                    _commit_ellipse_3click(pts[0], pts[1], pts[2])
                    state.dig_active_ellipse_clicks = []
            _update_overlay()
            _update_info()

    # ---- Selection, deletion, undo ----
    def _hit_record(x, y):
        """The index of the smallest record whose mask holds image pixel
        (x, y), or None."""
        best, best_area = None, None
        for i, rec in enumerate(state.dig_records):
            m = rec.get("mask")
            if m is None or not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]):
                continue
            if not m[int(y), int(x)]:
                continue
            from functions.digitize import mask_area
            area = mask_area(m)
            if best is None or area < best_area:
                best, best_area = i, area
        return best

    def _selected_indices():
        """The selected records, in order."""
        n = len(state.dig_records)
        return sorted(i for i in _sel["many"] if 0 <= i < n)

    def _selection_changed(from_table=False):
        _update_overlay()
        _update_info()
        if not from_table:
            _sync_table_selection()
        _sync_selection_controls()

    def _select(idx, *, from_table=False):
        """Select record ``idx`` alone (None clears): the canvas highlight,
        the table row, the label field and the Delete button follow."""
        if idx is not None and not (0 <= idx < len(state.dig_records)):
            idx = None
        _sel["idx"] = idx
        _sel["many"] = set() if idx is None else {idx}
        _selection_changed(from_table)

    def _select_many(indices, *, add=False, from_table=False, primary=None):
        """Select several records; ``add`` keeps the current selection."""
        n = len(state.dig_records)
        many = set(_sel["many"]) if add else set()
        many.update(i for i in indices if 0 <= i < n)
        _sel["many"] = many
        if primary is not None and primary in many:
            _sel["idx"] = primary
        elif _sel["idx"] not in many:
            _sel["idx"] = min(many) if many else None
        _selection_changed(from_table)

    def _toggle_selected(idx):
        if idx in _sel["many"]:
            many = set(_sel["many"]) - {idx}
            _sel["many"] = many
            if _sel["idx"] == idx:
                _sel["idx"] = min(many) if many else None
            _selection_changed()
        else:
            _select_many([idx], add=True, primary=idx)

    def _delete_indices(indices):
        """Remove these records in one step; Undo restores them all."""
        idxs = sorted({i for i in indices if 0 <= i < len(state.dig_records)})
        if not idxs:
            return []
        recs = [state.dig_records[i] for i in idxs]
        for i in reversed(idxs):
            state.dig_records.pop(i)
        _history.append(("delete", idxs, recs, len(state.dig_records)))
        _sel.update(idx=None, many=set())
        _records_changed()
        return recs

    def _delete_record(idx):
        """Remove record ``idx``; Undo can restore it."""
        recs = _delete_indices([idx])
        return recs[0] if recs else None

    def _delete_selected():
        idxs = _selected_indices()
        if not idxs:
            _notify("Select a clast first: Select mode, then click inside "
                    "it, or pick its row in the clast table.", type="info")
            return
        recs = _delete_indices(idxs)
        if len(recs) == 1:
            _notify(f"Deleted clast #{idxs[0] + 1} "
                    f"({recs[0].get('shape', '')}). Undo restores it.",
                    type="info")
        elif recs:
            _notify(f"Deleted {len(recs):,} clasts. Undo restores them.",
                    type="info")

    def _records_changed(autosave=True):
        """Everything that shows the records follows a change."""
        _geom["ver"] += 1
        _update_overlay()
        _update_info()
        _update_nav_summary()
        _refresh_records_panel()
        if autosave:
            _autosave_records()

    # ---- Commit functions ----
    def _commit_polygon(vertices):
        """Build the mask and store the record."""
        if len(vertices) < 3:
            ui.notify("Polygon needs at least 3 vertices — click more on the photograph (Work surface).", type="warning")
            return
        from functions.digitize import polygon_to_cropped_mask, mask_centroid
        H, W = img_widgets["natural_size"][1], img_widgets["natural_size"][0]
        mask = polygon_to_cropped_mask(vertices, (H, W))
        if mask.sum() < 9:
            ui.notify("Polygon mask is essentially empty — draw a larger outline (image, Work surface).", type="warning")
            return
        cx, cy = mask_centroid(mask)
        state.dig_records.append({
            "shape": "polygon",
            "params": {"vertices": list(vertices)},
            "mask": mask,
            "centroid_x": cx,
            "centroid_y": cy,
            "label": "",   # written to the Label column on export
            "origin": "hand",
            "edited": False,
        })
        ui.notify(f"Clast #{len(state.dig_records)} digitized "
                  f"({mask.sum()} px).", type="positive")
        _records_changed()

    def _commit_circle(center, radius):
        from functions.digitize import circle_to_cropped_mask
        H, W = img_widgets["natural_size"][1], img_widgets["natural_size"][0]
        mask = circle_to_cropped_mask(center, radius, (H, W))
        if mask.sum() < 9:
            ui.notify("Circle mask is essentially empty — click farther from the centre (image, Work surface).", type="warning")
            return
        state.dig_records.append({
            "shape": "circle",
            "params": {"center": list(center), "radius": radius},
            "mask": mask,
            "centroid_x": float(center[0]),
            "centroid_y": float(center[1]),
            "label": "",
            "origin": "hand",
            "edited": False,
        })
        ui.notify(f"Clast #{len(state.dig_records)} digitized "
                  f"(circle r={radius:.0f}px).", type="positive")
        _records_changed()

    def _commit_ellipse_3click(p1, p2, p3):
        """A rotated ellipse from 3 clicks: p1, p2 are the ends of the major
        axis; the minor axis is the perpendicular distance from p3."""
        from functions.digitize import ellipse_to_cropped_mask
        H, W = img_widgets["natural_size"][1], img_widgets["natural_size"][0]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        major_len = float(np.hypot(x2 - x1, y2 - y1))
        if major_len < 2:
            ui.notify("Major axis too short — click the ends farther apart (image, Work surface).", type="warning")
            return
        a = major_len / 2.0
        # Degrees in image coords (y down), which is what cv2.ellipse wants.
        angle_rad = float(np.arctan2(y2 - y1, x2 - x1))
        angle_deg = np.degrees(angle_rad)
        # Perpendicular distance from p3 to the major-axis line.
        b = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / major_len
        if b < 1:
            ui.notify("Minor axis too short — click farther from the axis (image, Work surface).", type="warning")
            return
        mask = ellipse_to_cropped_mask((cx, cy), (a, b), angle_deg, (H, W))
        if mask.sum() < 9:
            ui.notify("Ellipse mask is essentially empty — draw a larger ellipse (image, Work surface).", type="warning")
            return
        state.dig_records.append({
            "shape": "ellipse",
            "params": {
                "center": [cx, cy],
                "axes": [a, b],
                "angle_deg": angle_deg,
            },
            "mask": mask,
            "centroid_x": cx,
            "centroid_y": cy,
            "label": "",
            "origin": "hand",
            "edited": False,
        })
        from functions.digitize import axial_bearing
        ui.notify(f"Clast #{len(state.dig_records)} digitized "
                  f"(ellipse {2*a:.0f}×{2*b:.0f}px @ "
                  f"{axial_bearing(angle_deg):.0f}°).",
                  type="positive")
        _records_changed()

    # ---- Image loader ----
    def _load_image():
        if not state.dig_src_path or not Path(state.dig_src_path).exists():
            ui.notify("Pick a photograph in the toolbar (Work surface), or "
                      "set *Image directory* (Inputs, above).",
                      type="warning")
            return
        try:
            from PIL import Image
            with Image.open(state.dig_src_path) as pim:
                W, H = pim.size
        except Exception as e:
            ui.notify(f"Could not read image: {e}"
                      f"{_images.heif_failure_hint(state.dig_src_path)}. "
                      "Pick another photograph (Image, bar above).",
                      type="negative")
            return
        img_widgets["natural_size"] = (W, H)
        img_widgets["disp_scale"] = 1.0

        # Handle sizes in image pixels: scaled to the diagonal and the
        # user's multiplier, floored at 1-2 px so small clasts stay clickable.
        import math
        diag = math.hypot(W, H) or 1000.0
        scale = max(0.1, float(state.dig_size_scale or 1.0))
        img_widgets["natural_diag"] = diag
        img_widgets["marker_r"] = max(1, round(diag * 0.0025 * scale))
        img_widgets["close_r"] = max(2, round(diag * 0.0040 * scale))
        # The click buffer stays just slightly bigger than the marker.
        img_widgets["drag_threshold"] = max(2, img_widgets["marker_r"] + 2)

        # Huge or non-browser-native TIFFs are served as a downsampled
        # cached PNG; disp_scale > 1 maps click coords back to the source.
        try:
            serve_path_str, disp_scale = prepare_browser_image(
                state.dig_src_path,
                current_project=state.current_project,
                max_dim=4000,
                cache_subdir="digitize_cache",
            )
        except Exception as ex:
            ui.notify(
                f"Could not prepare browser image: {ex}. Pick another image (Image, bar above).",
                type="negative")
            return
        img_widgets["disp_scale"] = float(disp_scale)
        serve_path = Path(serve_path_str)

        from nicegui import app as _ngapp
        import hashlib
        src_dir = str(serve_path.parent)
        mount_name = "dig_" + hashlib.md5(src_dir.encode()).hexdigest()[:8]
        try:
            _ngapp.add_static_files(f"/{mount_name}", src_dir)
        except Exception:
            pass
        url = f"/{mount_name}/{serve_path.name}"
        img_widgets["src_url"] = url

        interactive_container.clear()
        with interactive_container:
            # The photograph in a box capped at 70 % of the viewport.
            _dig_box = ui.element("div").classes("w-full pm-dig-box") \
                .style("overflow:auto; max-height:70vh; "
                       "border:1px solid #dfe3e8;")
            with _dig_box:
                interactive = ui.interactive_image(
                    source=url,
                    events=["mousedown", "mousemove", "mouseup",
                            "click", "dblclick"],
                    cross=True,
                    on_mouse=_on_mouse,
                ).style("display:block;").classes("dig-image")
            img_widgets["interactive"] = interactive
            # Outlines in one layer, what follows the pointer in another
            # (NiceGUI 2.17+); an older NiceGUI draws both in the image.
            if hasattr(interactive, "add_layer"):
                img_widgets["outline_layer"] = interactive.add_layer()
                img_widgets["mark_layer"] = interactive.add_layer()
            else:
                img_widgets["outline_layer"] = None
                img_widgets["mark_layer"] = None
            _geom["sig"] = None
            if disp_scale != 1.0:
                ui.label(
                    f"Source: {W} × {H} px (cached PNG, "
                    f"{disp_scale:.2f}× downsample for display)") \
.classes("text-xs text-grey-7 mt-1")
            else:
                ui.label(f"Source: {W} × {H} px") \
.classes("text-xs text-grey-7 mt-1")
            # Floating magnifier: a position:fixed wrapper and a hidden
            # full-res <img> the canvas samples from; innerHTML strips
            # inline handlers, so the script below wires it up. Its URL
            # carries a query so the browser caches it under its own key:
            # two concurrent requests for the same large cached PNG (a
            # transcoded HEIC is ~10 MB) make Chromium fail one of them
            # with ERR_CACHE_WRITE_FAILURE, and it was the canvas's that
            # lost.
            img_widgets["loupe"] = ui.html(
                f'<div id="digloupe-wrapper" '
                f'style="position:fixed; pointer-events:none; z-index:1000; '
                f'display:none; left:0; top:0;">'
                f'<canvas id="digloupe-canvas" width="200" height="200" '
                f'style="display:block; border:2px solid #5a6878; '
                f'background:#222; image-rendering:pixelated; '
                f'box-shadow:0 2px 8px rgba(0,0,0,0.4);"></canvas>'
                f'<div style="background:rgba(0,0,0,0.7); color:white; '
                f'font-size:10px; text-align:center; padding:2px;">'
                f'Magnifier (4×)</div>'
                f'</div>'
                f'<img id="digloupe-source-img" src="{url}?loupe=1" '
                f'style="display:none;" crossorigin="anonymous" />'
            )

            _run_js(
                f"""
                (function() {{
                    // disp_scale: original-image px per displayed-PNG px.
                    window.__dig_disp_scale = {float(disp_scale)};

                    var WRAP_W = 212, WRAP_H = 226, LOUP_SIZE = 200, MAG = 4;

                    window.__digloupe_draw = function(cx, cy) {{
                        var c = document.getElementById('digloupe-canvas');
                        var img = document.getElementById('digloupe-source-img');
                        if (!c || !img || !img.complete || img.naturalWidth === 0) return;
                        var ctx = c.getContext('2d');
                        ctx.imageSmoothingEnabled = false;
                        ctx.fillStyle = '#222';
                        ctx.fillRect(0, 0, LOUP_SIZE, LOUP_SIZE);
                        var srcW = LOUP_SIZE / MAG, srcH = LOUP_SIZE / MAG;
                        var ox = cx - srcW/2, oy = cy - srcH/2;
                        ctx.drawImage(img, ox, oy, srcW, srcH,
                                      0, 0, LOUP_SIZE, LOUP_SIZE);

                        // Overlay shapes are in original-image coords; apply
                        // the same 1/disp_scale the SVG wrapper does.
                        var ds = window.__dig_disp_scale || 1.0;
                        var svg = document.querySelector('.dig-image svg')
                                  || document.querySelector(
                                       'div[class*="dig-image"] svg');
                        if (svg) {{
                            ctx.save();
                            ctx.scale(MAG / ds, MAG / ds);
                            ctx.translate(-ox * ds, -oy * ds);

                            svg.querySelectorAll('path').forEach(function(p) {{
                                var fill   = p.getAttribute('fill')   || 'none';
                                var stroke = p.getAttribute('stroke') || 'none';
                                var sw     = parseFloat(
                                                p.getAttribute('stroke-width') || '1');
                                var d = p.getAttribute('d');
                                if (!d) return;
                                try {{
                                    var p2d = new Path2D(d);
                                    if (fill !== 'none') {{
                                        ctx.fillStyle = fill;
                                        ctx.fill(p2d);
                                    }}
                                    if (stroke !== 'none') {{
                                        ctx.strokeStyle = stroke;
                                        ctx.lineWidth = sw * ds / MAG;
                                        ctx.stroke(p2d);
                                    }}
                                }} catch(e) {{}}
                            }});

                            svg.querySelectorAll('line').forEach(function(ln) {{
                                var x1 = parseFloat(ln.getAttribute('x1')||'0');
                                var y1 = parseFloat(ln.getAttribute('y1')||'0');
                                var x2 = parseFloat(ln.getAttribute('x2')||'0');
                                var y2 = parseFloat(ln.getAttribute('y2')||'0');
                                var stroke = ln.getAttribute('stroke') || 'orange';
                                var sw = parseFloat(
                                            ln.getAttribute('stroke-width') || '1');
                                ctx.strokeStyle = stroke;
                                ctx.lineWidth = sw * ds / MAG;
                                ctx.beginPath();
                                ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
                                ctx.stroke();
                            }});

                            svg.querySelectorAll('circle').forEach(function(circ) {{
                                var cx2 = parseFloat(circ.getAttribute('cx')||'0');
                                var cy2 = parseFloat(circ.getAttribute('cy')||'0');
                                var r   = parseFloat(circ.getAttribute('r')  ||'4');
                                var fill   = circ.getAttribute('fill')   || 'orange';
                                var stroke = circ.getAttribute('stroke') || 'black';
                                var sw = parseFloat(
                                            circ.getAttribute('stroke-width') || '1');
                                ctx.beginPath();
                                ctx.arc(cx2, cy2, r, 0, 2*Math.PI);
                                ctx.fillStyle = fill;
                                ctx.fill();
                                ctx.strokeStyle = stroke;
                                ctx.lineWidth = sw * ds / MAG;
                                ctx.stroke();
                            }});

                            ctx.restore();
                        }}

                        // crosshair
                        ctx.strokeStyle = 'rgba(255,80,80,0.9)';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        var half = LOUP_SIZE / 2;
                        ctx.moveTo(half, half-10); ctx.lineTo(half, half+10);
                        ctx.moveTo(half-10, half); ctx.lineTo(half+10, half);
                        ctx.stroke();
                    }};

                    window.__digloupe_position = function(clientX, clientY) {{
                        var wrap = document.getElementById('digloupe-wrapper');
                        if (!wrap) return;
                        var OFFSET = 16;
                        var x = clientX + OFFSET, y = clientY + OFFSET;
                        if (x + WRAP_W > window.innerWidth)  x = clientX - WRAP_W - OFFSET;
                        if (y + WRAP_H > window.innerHeight) y = clientY - WRAP_H - OFFSET;
                        if (x < 4) x = 4;
                        if (y < 4) y = 4;
                        wrap.style.left = x + 'px';
                        wrap.style.top  = y + 'px';
                    }};

                    // Move the wrapper to document.body so ancestor
                    // transforms can't re-anchor position:fixed.
                    function ensureWrapper() {{
                        var w = document.getElementById('digloupe-wrapper');
                        if (w && w.parentElement !== document.body) {{
                            document.body.appendChild(w);
                        }}
                    }}
                    ensureWrapper();

                    function hookProbe() {{
                        var probe = document.querySelector('.dig-image img')
                                    || document.querySelector('img[src*="/dig_"]');
                        if (!probe) return false;
                        if (probe.__dig_hooked) return true;
                        probe.__dig_hooked = true;
                        var wrap = document.getElementById('digloupe-wrapper');
                        probe.addEventListener('mouseenter', function() {{
                            ensureWrapper();
                            if (wrap) wrap.style.display = 'block';
                        }});
                        probe.addEventListener('mouseleave', function() {{
                            if (wrap) wrap.style.display = 'none';
                        }});
                        probe.addEventListener('mousemove', function(e) {{
                            var rect = probe.getBoundingClientRect();
                            var img_x = (e.clientX - rect.left)
                                        * probe.naturalWidth  / rect.width;
                            var img_y = (e.clientY - rect.top)
                                        * probe.naturalHeight / rect.height;
                            window.__digloupe_draw(img_x, img_y);
                            window.__digloupe_position(e.clientX, e.clientY);
                        }});
                        return true;
                    }}
                    if (!hookProbe()) {{
                        // The image element can land a frame later.
                        setTimeout(hookProbe, 100);
                        setTimeout(hookProbe, 500);
                    }}

                    var src_img = document.getElementById('digloupe-source-img');
                    var center = function() {{
                        window.__digloupe_draw({W//2}, {H//2});
                    }};
                    if (src_img && src_img.complete && src_img.naturalWidth > 0)
                        center();
                    else if (src_img)
                        src_img.addEventListener('load', center);
                }})();
                """
            )
        _update_overlay()
        _dig_zoom_fit()
        _update_info()

    # ---- Action buttons (in the same bar) ----
    with dig_toolbar:
        ui.separator().props("vertical")

        def _done_polygon():
            if state.dig_mode != "polygon":
                ui.notify("Only relevant in polygon mode.", type="info")
                return
            if not state.dig_active_polygon:
                ui.notify("No vertices yet — click the photograph (Work surface).", type="info")
                return
            if len(state.dig_active_polygon) < 3:
                ui.notify("Need at least 3 vertices — click more on the photograph (Work surface).", type="warning")
                return
            _commit_polygon(state.dig_active_polygon)
            state.dig_active_polygon = []
            _update_overlay()
            _update_info()
        ui.button("Done", icon="check", on_click=_done_polygon) \
.props("dense outline no-caps") \
.tooltip("Close the in-progress polygon and add it to the records. "
                     "Shortcuts: double-click anywhere, press Enter, or click "
                     "near the first vertex.")

        # Enter finishes a polygon, Escape cancels; only on the Digitize tab.
        def _on_key(e):
            try:
                if state.active_tab != "Digitize":
                    return
                if not getattr(e.action, "keydown", False):
                    return
                if e.key.name == "Enter":
                    if (state.dig_mode == "polygon"
                            and len(state.dig_active_polygon) >= 3):
                        _done_polygon()
                elif e.key.name == "Escape":
                    if state.dig_scale_draw and _scale["pt"] is not None:
                        _scale["pt"] = None
                        _scale["cursor"] = None
                        _update_overlay()
                        _update_info()
                        return
                    if (state.dig_active_polygon
                            or state.dig_active_circle_center
                            or state.dig_active_ellipse_clicks
                            or state.dig_drag_target
                            or state.dig_drag_active):
                        state.dig_active_polygon = []
                        state.dig_active_circle_center = []
                        state.dig_active_ellipse_clicks = []
                        state.dig_drag_target = {}
                        state.dig_drag_active = False
                        state.dig_drag_consumed = False
                        _update_overlay()
                        _update_info()
            except Exception:
                pass
        ui.keyboard(on_key=_on_key)

        # Delete / Backspace remove the selected clast, only on this tab and
        # never while a text field has focus (the browser side ignores keys
        # typed into input, select and textarea).
        def _on_delete_key(e):
            try:
                if state.active_tab != "Digitize":
                    return
                if not getattr(e.action, "keydown", False):
                    return
                if e.key.name in ("Delete", "Backspace") and _sel["many"]:
                    _delete_selected()
            except Exception:
                pass
        _dig_delete_keys = ui.keyboard(on_key=_on_delete_key,
                                       ignore=["input", "select", "textarea"])

        def _cancel_active():
            state.dig_active_polygon = []
            state.dig_active_circle_center = []
            state.dig_active_ellipse_clicks = []
            state.dig_drag_target = {}
            state.dig_drag_active = False
            state.dig_drag_consumed = False
            _update_overlay()
            _update_info()
            ui.notify("Cancelled in-progress shape.", type="info")
        ui.button("Cancel", icon="close", on_click=_cancel_active) \
.props("dense outline no-caps") \
.tooltip("Drop the in-progress shape.")

        def _undo_last():
            # A deletion is undone first when nothing was added since.
            while _history:
                kind, idx, rec, n_after = _history[-1]
                if len(state.dig_records) == n_after:
                    _history.pop()
                    idxs = idx if isinstance(idx, list) else [idx]
                    recs = rec if isinstance(rec, list) else [rec]
                    # Ascending, each back at its own index.
                    for i, r in sorted(zip(idxs, recs), key=lambda t: t[0]):
                        state.dig_records.insert(
                            min(max(0, i), len(state.dig_records)), r)
                    _sel.update(idx=None, many=set())
                    _records_changed()
                    if len(recs) == 1:
                        _notify(f"Restored clast #{idxs[0] + 1} "
                                f"({recs[0].get('shape', '')}).", type="info")
                    else:
                        _notify(f"Restored {len(recs):,} clasts.", type="info")
                    return
                if len(state.dig_records) > n_after:
                    break               # a clast added since: undo it first
                _history.pop()          # stale (records cleared or switched)
            if not state.dig_records:
                ui.notify("Nothing to undo.", type="info")
                return
            removed = state.dig_records.pop()
            _sel.update(idx=None, many=set())
            _records_changed()
            ui.notify(f"Removed last clast ({removed['shape']}).", type="info")
        ui.button("Undo", icon="undo", on_click=_undo_last) \
.props("dense outline no-caps") \
.tooltip("Remove the last committed clast, or restore the last deleted one.")

        _dig_delete_btn = ui.button("Delete", icon="delete",
                                    on_click=lambda: _delete_selected()) \
.props("dense outline no-caps color=negative") \
.tooltip("Delete the selected clast (Select mode, or a row of the clast "
                 "table). Delete or Backspace does the same; Undo restores it.")

        def _load_saved():
            """Put the photograph's saved clasts on the canvas (added to
            what is already there); autosave may then write over its CSV."""
            csv = str(state.dig_out_path or "")
            if not state.dig_src_path or not csv or not os.path.exists(csv):
                _notify("This photograph has no saved clasts yet.", type="info")
                return
            if csv in state.dig_csv_loaded:
                _notify(f"The clasts of {Path(csv).name} are already on the "
                        "canvas.", type="info")
                return
            recs = _records_from_csv(csv, state.dig_src_path)
            if not recs:
                _notify(f"No clast of {Path(csv).name} could be rebuilt.",
                        type="warning")
                return
            n_before = len(state.dig_records)
            state.dig_records.extend(recs)
            state.dig_csv_loaded.add(csv)
            state.dig_csv_owned.add(csv)
            _seg_io["warned"].discard(csv)
            _sel.update(idx=None, many=set())
            _records_changed(autosave=bool(n_before))
            _notify(f"Loaded {len(recs):,} clast(s) from {Path(csv).name}"
                    + (f", added to the {n_before:,} already on the canvas."
                       if n_before else "."), type="positive")

        load_btn = ui.button("Load", icon="file_upload", on_click=_load_saved) \
.props("dense outline no-caps") \
.tooltip("Put this label set's saved clasts back on the canvas, after "
                 "Clear all took them off. Opening a set loads them the first "
                 "time.")

        def _truth_now(fname, full):
            """The truth's clasts as they are: this session's, else saved."""
            cached = state.dig_per_image.get(_cache_key(fname, set_id=_TRUTH))
            if cached is not None:
                return list(cached["records"]), cached.get("out_path")
            return _autoload_records(full, _TRUTH), None

        def _copy_to_truth(mode):
            """This set's clasts into the truth: added to its clasts, or in
            their place. The truth then opens, and autosave writes it."""
            copy_dlg.close()
            import copy as _copy
            if (state.dig_label_set or _TRUTH) == _TRUTH or not state.dig_src_path:
                return
            fname, full = _image_key(), state.dig_src_path
            recs = [{**r, "params": _copy.deepcopy(r.get("params", {}))}
                    for r in state.dig_records]
            _snapshot_current_image()
            truth, out_path = _truth_now(fname, full)
            new = (truth if mode == "add" else []) + recs
            state.dig_per_image[_cache_key(fname, set_id=_TRUTH)] = {
                "records": new,
                "out_path": out_path or _set_csv_for(full, _TRUTH)}
            src_label = _model_display(state.dig_label_set)
            _activate_image(state.dig_image_list.index(fname), _TRUTH)
            state.dig_csv_owned.add(str(state.dig_out_path))
            state.dig_csv_loaded.add(str(state.dig_out_path))
            _records_changed()
            _rebuild_entries()
            _notify(f"{len(recs):,} clast(s) of {src_label} "
                    + ("added to the truth" if mode == "add" and truth
                       else "are now the truth")
                    + f": {len(new):,} in all.", type="positive")

        def _ask_copy_to_truth():
            if (state.dig_label_set or _TRUTH) == _TRUTH or not state.dig_records:
                return
            truth, _p = _truth_now(_image_key(), state.dig_src_path)
            if not truth:
                _copy_to_truth("replace")
                return
            copy_msg.set_text(
                f"The truth of this photograph already holds {len(truth):,} "
                f"clast(s). Add the {len(state.dig_records):,} of this set to "
                "them, or replace them?")
            copy_dlg.open()

        with ui.dialog() as copy_dlg, ui.card():
            copy_msg = ui.label("").classes("text-sm")
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("Cancel", on_click=copy_dlg.close).props("flat")
                ui.button("Add to them", icon="add",
                          on_click=lambda: _copy_to_truth("add")) \
                    .props("no-caps color=primary")
                ui.button("Replace them", icon="swap_horiz",
                          on_click=lambda: _copy_to_truth("replace")) \
                    .props("no-caps color=negative")
        copy_btn = ui.button("Copy to truth", icon="content_copy",
                             on_click=_ask_copy_to_truth) \
            .props("dense outline no-caps").classes("pm-dig-copy-truth") \
            .tooltip("Start the truth of this photograph from this model's "
                     "set: its clasts are added to the truth, or replace it "
                     "(asks first when the truth holds clasts).")

        def _clear_all():
            if not state.dig_records:
                return
            n = len(state.dig_records)
            unsaved = not _autosave_ok()
            clear_msg.set_text(
                f"Take the {n:,} clast(s) off the canvas? Nothing on disk is "
                "deleted: Load brings the saved clasts back."
                + (" Autosave is off or paused here, so clasts not written "
                   "with Export CSV are lost." if unsaved else ""))
            clear_dlg.open()

        def _clear_all_confirmed():
            clear_dlg.close()
            # The saved CSV is no longer what the canvas holds: give it
            # back, so autosave pauses (and says so) until Load puts its
            # clasts on the canvas again. Kept owned, one clast drawn after
            # Clear all would overwrite everything saved.
            state.dig_csv_loaded.discard(str(state.dig_out_path or ""))
            state.dig_csv_owned.discard(str(state.dig_out_path or ""))
            _seg_io["warned"].discard(str(state.dig_out_path or ""))
            state.dig_records.clear()
            state.dig_detect_runs.pop(_image_key(), None)
            _sel.update(idx=None, many=set())
            _history.clear()
            state.dig_active_polygon = []
            state.dig_active_circle_center = []
            state.dig_active_ellipse_clicks = []
            state.dig_drag_target = {}
            state.dig_drag_active = False
            state.dig_drag_consumed = False
            _records_changed(autosave=False)
            ui.notify("Cleared the canvas; the saved files are untouched.",
                      type="warning")

        with ui.dialog() as clear_dlg, ui.card():
            clear_msg = ui.label("").classes("text-sm")
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("Cancel", on_click=clear_dlg.close).props("flat")
                ui.button("Clear canvas", icon="delete_sweep",
                          on_click=_clear_all_confirmed) \
                    .props("color=negative no-caps")
        ui.button("Clear all", icon="delete_sweep", on_click=_clear_all) \
.props("dense outline no-caps color=negative") \
.tooltip("Take every clast off the canvas (asks first). Saved CSVs and "
                 "figures are not touched: Load brings the saved clasts back. "
                 "Undo cannot restore a clear.")

    # ---- Mask R-CNN first pass: the surface's action, last in the bar.
    # Its threshold is a setting, moved to Inputs; Reload model is in the
    # left panel (Settings), under the model select.
    with dig_toolbar:
        ui.separator().props("vertical")

        def _maskrcnn_masks(image, min_confidence, devicemode, devicenumber):
            """Mask R-CNN on one image: ``(masks H x W x N, scores)``."""
            from functions.clasts_detection import _build_model
            model = _build_model(devicemode, devicenumber,
                                 min_confidence=min_confidence)
            try:
                r = model.detect([image], verbose=0)[0]
            except Exception as ex:
                # The cached model's TF graph can be gone by the next run
                # (a second Detect in another worker thread, a cache
                # clear): rebuild once and retry, as the ortho tile loop.
                if not ("InvalidArgumentError" in type(ex).__name__
                        or "input_image" in str(ex)
                        or "not found in the Graph" in str(ex)):
                    raise
                model = _build_model(devicemode, devicenumber,
                                     min_confidence=min_confidence,
                                     use_cache=False)
                r = model.detect([image], verbose=0)[0]
            return r.get("masks"), r.get("scores")

        class _FilteredLog:
            """The detector's centimetre summary means nothing for a
            pixel-unit run: every route into the console (logging handler,
            captured stdout, log_fn) goes through this filter; the run
            prints its own summary in the result unit."""

            def __init__(self, inner):
                self._inner = inner

            def push(self, line, **kw):
                if _gauge.is_detector_summary_line(line):
                    return
                self._inner.push(line, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def _serve_output(path: str) -> str:
            """A URL for a PNG under the output folder: a static mount per
            folder, cache-busted by mtime."""
            import hashlib as _hashlib
            p = Path(path)
            mount = "gauge_" + _hashlib.md5(str(p.parent).encode()).hexdigest()[:8]
            try:
                app.add_static_files(f"/{mount}", str(p.parent))
            except Exception:
                pass
            try:
                v = int(p.stat().st_mtime)
            except OSError:
                v = 0
            return f"/{mount}/{p.name}?v={v}"

        def _append_result(res: dict, figs: dict, scale, disclaimer,
                           model_names=(), provenance_line: str = ""):
            """The open photograph's card, replacing the previous one: the
            summary line, the model(s) and the counts per origin, both
            figures, the CSV path and, for an object scale, the disclaimer."""
            ul = scale.unit_label
            st = res["stats"]
            line = (f"n = {st['n']:,}, D50 = {_gauge.format_length(st['D50'], ul)}, "
                    f"D84 = {_gauge.format_length(st['D84'], ul)}")
            if scale.is_metric and st.get("sorting_phi") == st.get("sorting_phi"):
                line += f", sorting σφ = {st['sorting_phi']:.2f}"
            n_excl = int(res.get("excluded_by_segments") or 0)
            if n_excl:
                line += f"; {n_excl} crossing a scale segment removed"
            names = [str(n) for n in (model_names or []) if n]
            _drop_result_card("photo")
            with dig_results_col:
                card = ui.card().classes("w-full pm-dig-result")
                with card:
                    _csv_stem = Path(res['csv']).stem
                    if _csv_stem.endswith("_gauge"):
                        _csv_stem = _csv_stem[:-len("_gauge")]
                    ui.label(f"{_csv_stem} — {line}") \
                        .classes("text-sm font-bold")
                    ui.label("Model: " + (", ".join(names) if names
                                          else "none (every clast drawn by hand)")) \
                        .classes("text-xs text-grey-8 pm-dig-result-model")
                    if provenance_line:
                        ui.label(provenance_line) \
                            .classes("text-xs text-grey-8 pm-dig-result-provenance")
                    ui.image(_serve_output(figs["overlay"])).classes("w-full") \
                        .style("max-width: 1100px;")
                    if figs.get("distribution"):
                        ui.image(_serve_output(figs["distribution"])) \
                            .classes("w-full").style("max-width: 1100px;")
                    ui.label(res["csv"]).classes("text-xs font-mono text-grey-8")
                    if disclaimer:
                        ui.label(disclaimer).classes("text-xs text-grey-6")
            card.move(target_index=0)
            _result_cards["photo"] = card
            _result_cards["photo_key"] = _image_key()

        def _import_proposals(masks, scores, factor, mask_ids=None,
                              removed_ids=()):
            """Every detected mask as an editable polygon record, brought
            back to original-image pixels when detection ran on a
            resampled copy. ``mask_ids[i]`` is the clast_ID mask ``i`` was
            measured under; a mask whose clast run_gauge removed (it crosses
            a scale segment) is skipped. A mask too small for the 1.5 %
            polygon keeps its raw outline instead of being dropped. Returns
            how many were added."""
            import numpy as _np
            import cv2
            from functions.digitize import CroppedMask, mask_centroid
            if masks is None or getattr(masks, "ndim", 0) != 3:
                return 0
            W, H = img_widgets["natural_size"]
            gone = set(int(i) for i in (removed_ids or ()))
            n_added = 0
            for i in range(masks.shape[2]):
                if gone and mask_ids is not None and i < len(mask_ids) \
                        and mask_ids[i] in gone:
                    continue
                mask_i = masks[:, :, i].astype(_np.uint8)
                if mask_i.shape != (H, W):
                    mask_i = cv2.resize(mask_i, (W, H),
                                        interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(
                    mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                eps = 0.015 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx) < 3:
                    approx = cnt
                if len(approx) < 3:
                    continue
                vertices = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                cropped = CroppedMask.from_full(mask_i)
                centre = mask_centroid(cropped)
                if centre is None:
                    continue
                score_i = (float(scores[i]) if scores is not None
                           and i < len(scores) else 1.0)
                state.dig_records.append({
                    "shape": "polygon",
                    "params": {"vertices": vertices},
                    "mask": cropped,
                    "centroid_x": centre[0],
                    "centroid_y": centre[1],
                    "score": score_i,
                    "label": "",
                })
                n_added += 1
            return n_added

        def _proposal_record(vertices_full, score_i, raw_ring=None):
            """One editable polygon record from vertices in original-image
            pixels (col, row), its mask rasterised at the canvas size. When
            the simplified polygon degenerates, ``raw_ring`` (the unsimplified
            outline) is used, so a small clast is not lost."""
            from functions.digitize import polygon_to_cropped_mask, mask_centroid
            W, H = img_widgets["natural_size"]
            mask_i, vertices = None, []
            for ring in (vertices_full, raw_ring):
                if ring is None or len(ring) < 3:
                    continue
                vertices = [[int(round(x)), int(round(y))] for x, y in ring]
                mask_i = polygon_to_cropped_mask(vertices, (H, W))
                if mask_i.any():
                    break
            if mask_i is None or not mask_i.any():
                return None
            cx, cy = mask_centroid(mask_i)
            return {"shape": "polygon", "params": {"vertices": vertices},
                    "mask": mask_i, "centroid_x": cx,
                    "centroid_y": cy, "score": float(score_i),
                    "label": ""}

        def _simplify_ring(pts):
            """The same 1.5 %-of-perimeter polygon the mask proposals get."""
            import numpy as _np
            import cv2
            cnt = _np.asarray(pts, dtype=_np.float32).reshape(-1, 1, 2)
            eps = 0.015 * cv2.arcLength(cnt, True)
            return [(float(p[0][0]), float(p[0][1]))
                    for p in cv2.approxPolyDP(cnt, eps, True)]

        def _import_instance_proposals(instances, factor, segments=(),
                                       frame=None):
            """Proposals from a backend's cropped instance masks (detection
            pixels) brought back to original-image pixels. The instances
            carry no clast_ID, so the segment rule is applied to each
            outline directly: one crossing a scale segment is skipped, as
            run_gauge dropped its clast; so is one whose centroid lies on
            the quadrat frame (``frame``, a FrameInset), as run_gauge
            dropped that one too."""
            import numpy as _np
            import cv2
            n_added = 0
            for inst in instances or []:
                m = _np.asarray(inst.mask, dtype=_np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                if not cs:
                    continue
                cnt = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
                if len(cnt) < 3:
                    continue
                ring = [((c + inst.x0) / factor, (r + inst.y0) / factor)
                        for c, r in cnt]
                if segments and _gauge.ring_crosses_segments(ring, segments):
                    continue
                if frame is not None:
                    rows_, cols_ = _np.nonzero(m)
                    if len(cols_) and not frame.contains(
                            (float(cols_.mean()) + inst.x0) / factor,
                            (float(rows_.mean()) + inst.y0) / factor):
                        continue
                rec = _proposal_record(_simplify_ring(ring), inst.score,
                                       raw_ring=ring)
                if rec is not None:
                    state.dig_records.append(rec)
                    n_added += 1
            return n_added

        def _import_contour_proposals(df, contours):
            """Proposals from outlines already in original pixels, y up (the
            gauge CSV frame), for a backend that left no instance masks."""
            W, H = img_widgets["natural_size"]
            n_added = 0
            if not contours or df is None or "clast_ID" not in df.columns:
                return 0
            scores = dict(zip(df["clast_ID"], df.get("Score", [1.0] * len(df))))
            for cid, arr in contours.items():
                ring = [(float(x), H - float(y)) for x, y in arr]
                sc = scores.get(cid, 1.0)
                rec = _proposal_record(_simplify_ring(ring),
                                       sc if sc == sc else 1.0, raw_ring=ring)
                if rec is not None:
                    state.dig_records.append(rec)
                    n_added += 1
            return n_added

        def _scale_in_force():
            """``(scale, disclaimer, source)`` for this photograph, or None
            (with a warning) when there is no scale."""
            kind, val = _effective_scale()
            if kind == "object":
                return val, _gauge.DISCLAIMER, "object"
            if kind == "gsd":
                _g, _src = _photo_gsd()
                return (_gauge.GaugeScale.from_gsd(val, "mm"), None,
                        f"gsd:{_src or 'resolution field'}")
            _notify("No scale: set *Resolution (m/pixel)* (Inputs, "
                    "above) or draw a scale segment (No GSD? Scale "
                    "from an object, above).", type="warning")
            return None

        def _gauge_target(source_path):
            """``(out_dir, stem)`` of the gauge outputs for a photograph."""
            proj = state.current_project
            out_dir = (project_path(proj, "gauge") if proj
                       else Path(source_path).parent / "gauge")
            stem = _naming.origin_stem(Path(source_path).stem, proj,
                                       state.current_date)
            return out_dir, stem

        def _do_detect_maskrcnn():
            """Run the detection model chosen in the drawer on the current
            photograph at the Detection scale, in a background thread, and
            load every clast as an editable polygon record carrying the
            model's name (from Mask R-CNN's masks, a backend's instance
            masks, or its outlines), leaving out the detections a scale
            segment crosses. Nothing is written: Figures draws the figures
            and writes the CSV from the records as they are then."""
            if not state.dig_src_path or not Path(state.dig_src_path).exists():
                ui.notify("Pick a photograph in the toolbar (Work surface), "
                          "or set *Image directory* (Inputs, above).",
                          type="warning")
                return
            W_nat, H_nat = img_widgets.get("natural_size", (0, 0))
            if H_nat == 0 or W_nat == 0:
                ui.notify("The photograph is not on the canvas yet — pick it "
                          "again in the toolbar (Work surface).",
                          type="warning")
                return
            if state.dig_detect_running:
                ui.notify("A detection is already running; wait for its "
                          "proposals on the photograph.", type="info")
                return
            kind, val = _effective_scale()
            if kind == "object":
                scale, disclaimer, source = val, _gauge.DISCLAIMER, "object"
            elif kind == "gsd":
                scale, disclaimer = _gauge.GaugeScale.from_gsd(val, "mm"), None
                _g, _src = _photo_gsd()
                source = f"gsd:{_src or 'resolution field'}"
            else:
                ui.notify("No scale: set *Resolution (m/pixel)* (Inputs, "
                          "above) or draw a scale segment (No GSD? Scale "
                          "from an object, above).", type="warning")
                return
            source_path = state.dig_src_path
            proj = state.current_project
            out_dir = (project_path(proj, "gauge") if proj
                       else Path(source_path).parent / "gauge")
            stem = _naming.origin_stem(Path(source_path).stem, proj,
                                       state.current_date)
            thr = float(state.dig_detect_threshold or 0.70)
            lib = _library_objs()
            if kind == "object":
                _save_scale_library(quiet=True)
            model_name, model_label = current_detection_model()
            backend = None
            if state.dig_mask_detector is None and model_name != "maskrcnn":
                try:
                    from detectors import get_backend as _get_backend
                    backend = _get_backend(model_name)
                except Exception:
                    backend = None
                if backend is None or not backend.is_available():
                    ui.notify(f"Detection model '{model_label}' is not "
                              "available here: pick another in the left "
                              "panel (Settings).", type="negative")
                    return
            if state.dig_mask_detector is not None:
                model_name = model_label = getattr(
                    state.dig_mask_detector, "model_name", model_name)
            mask_detector = state.dig_mask_detector or _maskrcnn_masks
            if state.dig_detect_scale not in _gauge.DETECT_SCALE_OPTIONS:
                state.dig_detect_scale = _gauge.DETECT_SCALE_DEFAULT
            factor = _gauge.resolve_detect_scale(
                state.dig_detect_scale,
                "maskrcnn" if backend is None else model_name)
            devicemode = str(state.devicemode or "cpu").lower()
            devicenumber = int(state.devicenumber or 0)
            state.dig_detect_running = True
            detect_btn.props("loading")
            _disable(detect_btn, "Running")
            dig_log.clear()
            dig_log.push(f"[detect] {Path(source_path).name} → {out_dir} "
                         f"({model_label}; {_scale_source_label()}; detection "
                         f"scale {state.dig_detect_scale}"
                         + (f" = {factor:g}" if state.dig_detect_scale
                            == _gauge.DETECT_SCALE_AUTO else "")
                         + f"; threshold {thr:.2f})")
            console = _FilteredLog(dig_log)
            captured = {}

            def _detect_fn(**kw):
                """run_gauge's detector: measures every mask in pixel units
                the way Quadrat mode does, and keeps the masks for the
                proposals."""
                import numpy as _np
                import pandas as _pd
                from functions.clasts_detection import (
                    _CLAST_COLUMNS, _measure_clast, _mask_mean_intensity,
                    _normalize_to_uint8)
                path = kw["jobs"][0]["path"]
                resolution = float(kw.get("resolution", 1.0))
                image = _normalize_to_uint8(_images.read_rgb(path))
                masks, scores = mask_detector(
                    image, kw.get("min_confidence"),
                    kw.get("devicemode", devicemode),
                    int(kw.get("devicenumber", devicenumber) or 0))
                if masks is None or getattr(masks, "ndim", 0) != 3:
                    masks = _np.zeros(image.shape[:2] + (0,), dtype=bool)
                n = int(masks.shape[2])
                scores = (list(scores) if scores is not None else [1.0] * n)
                captured["masks"], captured["scores"] = masks, scores
                mask_ids = captured["mask_ids"] = [None] * n
                height = image.shape[0]
                from functions import clast_geometry as _CG
                outlines = {}
                records = []
                for i in range(n):
                    meas = _measure_clast(masks[:, :, i], scores[i], resolution)
                    if meas is None:
                        continue
                    mask_ids[i] = len(records) + 1
                    try:
                        _outline = _CG.contour_from_measurement(
                            meas, to_frame=_CG.quadrat_frame(height))
                        if _outline:
                            outlines[len(records) + 1] = _outline
                    except Exception:
                        pass
                    records.append({
                        "clast_ID": len(records) + 1,
                        "x": meas["center_x"],
                        "y": height - meas["center_y"],
                        "Clast_length": meas["Clast_length"],
                        "Clast_width": meas["Clast_width"],
                        "Ellipse_major_axis": meas["Ellipse_major_axis"],
                        "Ellipse_minor_axis": meas["Ellipse_minor_axis"],
                        "Surface_area": meas["Surface_area"],
                        "Perimeter": meas["Perimeter"],
                        "Equivalent_diameter": meas["Equivalent_diameter"],
                        "Eccentricity": meas["Eccentricity"],
                        "Solidity": meas["Solidity"],
                        "Mean_intensity": _mask_mean_intensity(
                            image, masks[:, :, i]),
                        "Score": meas["Score"],
                        "Orientation": meas["Orientation"],
                    })
                return [_CG.attach_contours(
                    _pd.DataFrame(records, columns=_CLAST_COLUMNS), outlines,
                    frame="pixels")]

            work_dir = Path(out_dir) / "_detect_work" / stem
            # Every segment on this photograph, whatever scale is in force:
            # run_gauge drops the detections they cross.
            try:
                photo_segments = _gauge_segments()
            except Exception:
                photo_segments = []
            image_key = _image_key()

            def _worker():
                from functions._logging import (
                    NiceGUILogHandler, attach_gui_handler, detach_gui_handler)
                handler = NiceGUILogHandler(console)
                attach_gui_handler(handler)
                try:
                    kept = {}
                    # The availability check Detect and Express make: without
                    # it the run reached HDF5 and printed a traceback at the
                    # user.
                    # Only for the model that will run: a mask detector set
                    # in state (dig_mask_detector) replaces Mask R-CNN, whose
                    # weights are then not needed.
                    try:
                        from detectors import get_backend as _gb_check
                        _be = backend if backend is not None else (
                            None if state.dig_mask_detector is not None
                            else _gb_check(model_name))
                        if _be is not None and not _be.is_available():
                            console.push("[detect] " + _model_unavailable_text(_be))
                            return
                    except Exception:
                        pass    # the check must never stop a good run
                    detect_fn = (_gauge.backend_detect_fn(
                        backend, work_dir, log_fn=console.push, keep=kept)
                        if backend is not None else _detect_fn)
                    with capture_stdout_to_log(console):
                        try:
                            res = _gauge.run_gauge(
                                source_path, scale, out_dir,
                                min_confidence=thr,
                                devicemode=devicemode, devicenumber=devicenumber,
                                log_fn=console.push, out_stem=stem,
                                detect_scale=factor, detect_fn=detect_fn,
                                scale_source=source, disclaimer=disclaimer,
                                model=model_name, segments=photo_segments,
                                write=False)
                        finally:
                            if backend is not None:
                                import shutil as _shutil
                                _shutil.rmtree(work_dir, ignore_errors=True)
                                try:
                                    work_dir.parent.rmdir()
                                except OSError:
                                    pass
                    from functions import naming as _nm_sets
                    _enter_model_set(_nm_sets.label_set_id(model_name))
                    n_before = len(state.dig_records)
                    if backend is None:
                        n_added = _import_proposals(
                            captured.get("masks"), captured.get("scores"),
                            factor, mask_ids=captured.get("mask_ids"),
                            removed_ids=list(res.get("removed_ids") or ())
                            + list(res.get("frame_removed_ids") or ()))
                    elif kept.get("instances"):
                        n_added = _import_instance_proposals(
                            kept["instances"], factor,
                            segments=photo_segments, frame=res.get("frame"))
                    else:
                        from functions import clast_geometry as _CG
                        n_added = _import_contour_proposals(
                            res["df"], _CG.contours_of(res["df"]))
                    # Every proposal carries the model that made it.
                    for rec in state.dig_records[n_before:]:
                        rec["origin"] = model_label
                        rec["edited"] = False
                    from functions import provenance as _prov
                    info = None
                    try:
                        from detectors import get_backend as _get_backend
                        info = getattr(backend or _get_backend(model_name),
                                       "info", None)
                    except Exception:
                        info = None
                    runs = [r for r in state.dig_detect_runs.get(image_key, [])
                            if r.get("display_name") != model_label]
                    runs.append(_prov.model_entry(
                        model_name, info, display_name=model_label,
                        detect_scale=factor, min_confidence=thr,
                        n_proposals=int(n_added),
                        excluded_by_segments=int(res.get("excluded_by_segments") or 0)))
                    state.dig_detect_runs[image_key] = runs
                    _sel.update(idx=None, many=set())
                    _records_changed()
                    _rebuild_entries()
                    n_excl = int(res.get("excluded_by_segments") or 0)
                    n_frame = int(res.get("excluded_by_frame") or 0)
                    dig_log.push(
                        f"[detect] {n_added} proposal(s) from {model_label}, "
                        f"in its own label set"
                        + (f"; {n_excl} crossing a scale segment removed"
                           if n_excl else "")
                        + (f"; {n_frame} on the quadrat frame removed"
                           if n_frame else "")
                        + ". Edit them there, or Copy to truth.")
                    _notify(
                        f"{n_added} detection(s) from {model_label}, in its "
                        f"own label set (the photograph list shows it). Edit "
                        f"them, or Copy to truth.", type="positive", timeout=6000)
                except Exception as ex:
                    import traceback
                    _tb = traceback.format_exc()
                    print(_tb)
                    try:
                        dig_log.push(f"[detect] FAILED: {ex}")
                        dig_log.push(_tb)
                    except Exception:
                        pass
                    _notify(f"Detection failed: {ex}. See the log (below), "
                            "then Detect (bar above) again.", type="negative")
                finally:
                    detach_gui_handler(handler)
                    state.dig_detect_running = False
                    try:
                        detect_btn.props(remove="loading")
                        _enable(detect_btn)
                    except Exception:
                        pass

            threading.Thread(target=_worker, daemon=True).start()

        def _do_figures():
            """The overlay and distribution figures, the CSV, its contours
            and sidecars from the records on the canvas as they are now
            (hand-drawn and kept proposals, after deletions and edits), in
            the scale in force; a result card under the clast table."""
            if not state.dig_records:
                _notify("No clasts on the canvas: draw some, or Detect "
                        "(bar above).", type="warning")
                return
            if state.dig_figures_running:
                _notify("The figures are being drawn; wait for the result "
                        "card.", type="info")
                return
            W_nat, H_nat = img_widgets.get("natural_size", (0, 0))
            if not state.dig_src_path or H_nat == 0:
                _notify("Pick a photograph in the toolbar (Work surface) "
                        "first.", type="warning")
                return
            got = _scale_in_force()
            if got is None:
                return
            scale, disclaimer, source = got
            source_path = state.dig_src_path
            out_dir, stem = _gauge_target(source_path)
            lib = _library_objs()
            records = list(state.dig_records)
            runs = _detect_runs()
            segs = [(s.p0, s.p1) for s in _gauge_segments()]
            state.dig_figures_running = True
            figures_btn.props("loading")
            _disable(figures_btn, "Drawing")

            def _worker():
                try:
                    from functions import clast_geometry as _CG
                    from functions import provenance as _prov
                    from functions.digitize import measure_records
                    df_px, positions, contours = measure_records(
                        records, H_nat, 1.0, with_contours=True)
                    df_px = df_px.drop(columns=["Label"], errors="ignore")
                    df = _gauge.convert_pixels_to_units(df_px, scale)
                    _CG.attach_contours(df, contours, frame="pixels")
                    stats = _gauge.gauge_stats(df, scale)
                    clasts = _prov.clast_origins(records, positions)
                    line = _prov.summary_line(clasts, runs)
                    names = _prov.model_names(clasts, runs)
                    excluded = sum(int(r.get("excluded_by_segments") or 0)
                                   for r in runs)
                    sidecar = {
                        "image": os.path.abspath(source_path),
                        "image_name": os.path.basename(source_path),
                        "scale_source": source,
                        "records": len(records),
                        "origins": _prov.origin_counts(clasts),
                        "edited": sum(1 for v in clasts.values() if v["edited"]),
                        "models": runs,
                        "provenance": line,
                        "excluded_by_segments": excluded,
                        "exclusion_rule": (_gauge.SEGMENT_EXCLUSION_RULE
                                           if runs and segs else None),
                        "exclusion_segments": [[list(a), list(b)] for a, b in segs],
                        "disclaimer": disclaimer or None,
                    }
                    written = _gauge.write_gauge_table(
                        df, scale, out_dir, stem, contours=contours,
                        stats=stats, sidecar=sidecar,
                        contours_extra={"models": names})
                    _prov.write_provenance(written["csv"], clasts, runs,
                                           image=source_path)
                    figs = _gauge.write_gauge_figures(
                        source_path, df, scale, out_dir, stem, library=lib,
                        log_fn=dig_log.push, disclaimer=disclaimer,
                        contours=contours, note=line)
                    res = dict(written, stats=stats,
                               excluded_by_segments=excluded)
                    _append_result(res, figs, scale, disclaimer, names, line)
                    dig_log.push(
                        f"[figures] {stats['n']} clast(s) from the canvas "
                        f"({line}); wrote {Path(written['csv']).name} and "
                        f"its figures.")
                    _notify(f"Figures written to {Path(out_dir).name}/ "
                            "(result card below the clast table).",
                            type="positive", timeout=6000)
                except Exception as ex:
                    import traceback
                    _tb = traceback.format_exc()
                    print(_tb)
                    try:
                        dig_log.push(f"[figures] FAILED: {ex}")
                        dig_log.push(_tb)
                    except Exception:
                        pass
                    _notify(f"Figures failed: {ex}. See the log (below the "
                            "photograph).", type="negative")
                finally:
                    state.dig_figures_running = False
                    try:
                        figures_btn.props(remove="loading")
                        _sync_selection_controls()
                    except Exception:
                        pass

            threading.Thread(target=_worker, daemon=True).start()

        figures_btn = ui.button("Figures", icon="insert_chart",
                                on_click=_do_figures) \
.props("dense outline no-caps color=primary") \
.tooltip("The overlay and distribution figures, with the CSV and its "
                 "sidecars, from the clasts on the canvas as they are now "
                 "(after your deletions and edits), in the scale in force.")

        def _sample_set_entries():
            """``(entries, notes, n_empty, object_scaled)`` for
            functions.gauge.write_sample_set: each photograph of the folder
            with its saved Digitize CSV and provenance sidecar; the open
            photograph's records are saved first (autosave), or used
            unsaved and said so."""
            import pandas as pd
            from functions import provenance as _prov
            entries, notes, n_empty, obj = [], [], 0, False
            cur = _image_key()
            H = img_widgets.get("natural_size", (0, 0))[1]
            if cur and state.dig_records and _autosave_ok():
                _autosave_records()
            for f in list(state.dig_image_list):
                df, prov = None, None
                csv = _saved_csv_for(f)
                if (f == cur and state.dig_records and not _autosave_ok()
                        and H):
                    records = list(state.dig_records)
                    df, positions = _truth_dataframe(H, records,
                                                     with_positions=True)
                    prov = {"clasts": _prov.clast_origins(records, positions),
                            "models": _detect_runs()}
                    why = ("autosave is off" if not state.dig_autosave else
                           "autosave is paused: its saved clasts are not loaded")
                    notes.append(f"{f}: its records are not saved ({why}); "
                                 "the ones on the canvas were used.")
                elif csv and os.path.exists(csv):
                    try:
                        df = pd.read_csv(csv)
                    except Exception as ex:
                        notes.append(f"{f}: {Path(csv).name} is unreadable ({ex}).")
                        continue
                    prov = _prov.read_provenance(csv)
                else:
                    n_empty += 1
                    continue
                if df is None or not len(df):
                    n_empty += 1
                    continue
                unit = (str(df["unit"].iloc[0]) if "unit" in df.columns
                        else None)
                gsd = None
                try:
                    gsd = _effective_gsd(os.path.join(state.dig_image_dir, f)).gsd
                except Exception:
                    gsd = None
                if unit or (not gsd and state.dig_scale_segments.get(f)):
                    obj = True
                entries.append({"photo": f, "df": df, "provenance": prov})
            return entries, notes, n_empty, obj

        def _do_sample_set():
            """Every clast of every photograph of this folder, pooled:
            the CSV, the per-photograph summary and the distribution figure,
            in a result card under the clast table."""
            if not state.dig_image_dir or not state.dig_image_list:
                _notify("Set *Image directory* (Inputs, above) first.",
                        type="warning")
                return
            entries, notes, n_empty, obj = _sample_set_entries()
            if not entries:
                _notify("No photograph of this folder has saved clasts yet.",
                        type="warning")
                return
            folder = Path(state.dig_image_dir)
            proj = state.current_project
            out_dir = (project_path(proj, "gauge") if proj
                       else folder / "gauge")
            stem = _naming.origin_stem(folder.name, proj, state.current_date)
            disclaimer = _gauge.DISCLAIMER if obj else None
            for n in notes:
                dig_log.push(f"[sample set] {n}")
            try:
                res = _gauge.write_sample_set(entries, out_dir, stem,
                                              disclaimer=disclaimer,
                                              log_fn=dig_log.push)
            except Exception as ex:
                import traceback
                print(traceback.format_exc())
                dig_log.push(f"[sample set] FAILED: {ex}")
                _notify(f"Sample set failed: {ex}. See the log (below the "
                        "photograph).", type="negative")
                return
            if not res["metric"] and not obj:
                disclaimer = _gauge.DISCLAIMER
            _append_sample_set(res, folder.name, notes, n_empty, disclaimer)
            _notify(f"Sample set written to {Path(out_dir).name}/ (result "
                    "card below the clast table).", type="positive")

        def _append_sample_set(res, folder_name, notes, n_empty, disclaimer):
            summary = res["summary_df"]
            unit = res.get("unit") or ""
            pooled_row = summary[summary["photo"] == "(pooled)"]
            n_photos = int((summary["photo"] != "(pooled)").sum())
            _drop_result_card("sample")
            with dig_results_col:
                card = ui.card().classes("w-full pm-dig-result pm-dig-sampleset")
                with card:
                    if len(pooled_row):
                        pr = pooled_row.iloc[0]
                        head = (f"Sample set — {folder_name}: {n_photos} "
                                f"photograph{'s' if n_photos != 1 else ''}, "
                                f"n = {int(pr['n']):,}, D50 = "
                                f"{_gauge.format_length(pr['D50'], unit)}, "
                                f"D84 = {_gauge.format_length(pr['D84'], unit)}")
                        if res.get("metric") and pr["sorting_phi"] == pr["sorting_phi"]:
                            head += f", sorting σφ = {float(pr['sorting_phi']):.2f}"
                    else:
                        head = f"Sample set — {folder_name}: nothing to pool"
                    ui.label(head).classes("text-sm font-bold")
                    for s in res.get("skipped") or []:
                        ui.label(f"Skipped {s['photo']}: {s['reason']}.") \
                            .classes("text-xs text-warning pm-dig-sampleset-skipped")
                    for n in notes:
                        ui.label(n).classes("text-xs text-grey-8")
                    if n_empty:
                        ui.label(f"{n_empty} photograph{'s' if n_empty != 1 else ''} "
                                 "of the folder without clasts.") \
                            .classes("text-xs text-grey-7")
                    if res.get("distribution"):
                        ui.image(_serve_output(res["distribution"])) \
                            .classes("w-full").style("max-width: 1100px;")

                    def _r(v, d=4):
                        try:
                            f = float(v)
                        except (TypeError, ValueError):
                            return v
                        return None if f != f else float(f"{f:.{d}g}")
                    cols = [{"name": c, "field": c, "sortable": True,
                             "label": (f"{c} ({unit})" if c in ("D16", "D50", "D84", "mean")
                                       else ("Sorting σφ" if c == "sorting_phi" else c)),
                             "align": "left" if c in ("photo", "models", "origins") else "right"}
                            for c in ("photo", "n", "D16", "D50", "D84", "mean",
                                      "sorting_phi", "models", "origins")]
                    rows = [{c: (_r(r[c]) if c in ("D16", "D50", "D84", "mean", "sorting_phi")
                                 else (int(r[c]) if c == "n" else str(r[c] or "")))
                             for c in [x["name"] for x in cols]}
                            for _, r in summary.iterrows()]
                    ui.table(rows=rows, columns=cols, row_key="photo") \
                        .classes("w-full pm-dig-sampleset-summary") \
                        .props("dense flat bordered")
                    ui.label(res["csv"]).classes("text-xs font-mono text-grey-8")
                    ui.label(res["summary"]).classes("text-xs font-mono text-grey-8")
                    if disclaimer:
                        ui.label(disclaimer).classes("text-xs text-grey-6")
            # Under the photograph's card, which stays on top.
            card.move(target_index=1 if _result_cards["photo"] is not None else 0)
            _result_cards["sample"] = card

        sample_set_btn = ui.button("Sample set", icon="stacked_bar_chart",
                                   on_click=_do_sample_set) \
.props("dense outline no-caps color=primary") \
.tooltip("Distribution of every clast from every photograph in this folder")

        # The quadrat frame of this photograph. Its thickness is kept in the
        # photograph's record, where Orthorectify writes it, so Detect and
        # Digitize both leave the band out.
        _frame_ui = {"quiet": False}
        _dig_frame_num = ui.number(
            label="Quadrat frame (cm)", min=0.0, max=50.0, step=0.1,
            format="%.1f") \
            .classes("w-40 pm-dig-frame-field") \
            .tooltip("The width of the frame's bars along the edges of this "
                     "photograph. Detection leaves out every clast whose "
                     "centre falls on that band, and the dashed line shows "
                     "what is measured. Kept with the photograph; empty: no "
                     "frame. It needs the photograph's GSD.")

        def _on_dig_frame(e):
            if _frame_ui["quiet"] or not state.dig_src_path:
                return
            try:
                v = float(e.value) if e.value not in (None, "") else 0.0
            except (TypeError, ValueError):
                return
            from functions import quadrat_frame as _qf
            try:
                _qf.set_frame_thickness(state.dig_src_path,
                                        v / 100.0 if v > 0 else None)
            except Exception as ex:
                _notify(f"Could not record the frame: {ex}", type="warning")
                return
            _frame_cache["path"] = None
            _update_overlay()
            _sync_frame_controls(value=False)
        _dig_frame_num.on_value_change(_on_dig_frame)

        def _select_on_frame():
            idxs = _frame_band_indices()
            if not idxs:
                _notify("No clast on the frame band.", type="info")
                return
            if state.dig_mode != "select":
                _set_mode("select")
            _select_many(idxs)
            _notify(f"{len(idxs):,} clast(s) on the frame selected: Delete "
                    "removes them.", type="info")

        _dig_frame_sel_btn = ui.button(
            "On the frame", icon="border_outer",
            on_click=lambda: _select_on_frame()) \
            .props("dense outline no-caps").classes("pm-dig-frame-select") \
            .tooltip("Select the clasts whose centre is on the frame band, "
                     "for Delete to remove them.")

        detect_btn = ui.button("Detect", icon="biotech",
                               on_click=_do_detect_maskrcnn) \
.props("dense color=secondary no-caps") \
.tooltip(
                "First pass with the detection model chosen in the left "
                "panel (Settings): run it on the current image at the "
                "Detection scale and load each detected clast onto the "
                "canvas as an editable polygon record (detections crossing "
                "a scale segment are left out). Figures then writes the CSV "
                "and the figures. The model uses your current Device "
                "settings (GPU/CPU); the table shows each score.")
        _dig_thr = ui.number(
            label="Detect score threshold",
            min=0.05, max=1.0, step=0.05, format="%.2f",
        ).bind_value(state, "dig_detect_threshold") \
.classes("w-40") \
.tooltip("Minimum Mask R-CNN confidence to keep a detection "
                 "(0.05–1.0). Higher = fewer but surer detections.")

    def _frame_band_indices():
        """The records whose centre lies on the frame band."""
        fi = _dig_frame_inset()
        if fi is None:
            return []
        return [i for i, r in enumerate(state.dig_records)
                if not fi.contains(float(r.get("centroid_x", -1)),
                                   float(r.get("centroid_y", -1)))]

    def _sync_frame_controls(value=True):
        """The frame field shows this photograph's thickness; the button
        says how many clasts sit on the band."""
        try:
            from functions import quadrat_frame as _qf
            from functions.gsd import effective_gsd
            path = state.dig_src_path
            if value:
                t = _qf.frame_thickness_m(path) if path else None
                _frame_ui["quiet"] = True
                try:
                    _dig_frame_num.value = round(t * 100.0, 2) if t else None
                finally:
                    _frame_ui["quiet"] = False
            has_gsd = bool(path and effective_gsd(path).gsd)
            _dig_frame_num.set_enabled(has_gsd)
            n = len(_frame_band_indices())
            if n:
                _dig_frame_sel_btn.set_text(f"On the frame ({n:,})")
                _enable(_dig_frame_sel_btn)
            else:
                _dig_frame_sel_btn.set_text("On the frame")
                _disable(_dig_frame_sel_btn, "No clast on the frame band")
        except NameError:
            pass

    # ---- Detection console, then the clast table and the result cards
    # (newest first), moved into place at the end of the builder ----
    dig_log = build_log_console(max_lines=400, height="h-32")
    dig_results_col = ui.column().classes("w-full gap-2 pm-dig-results")

    # ---- Image + loupe area ----
    interactive_container = ui.column().classes("w-full")

    # ---- The clast table: every record, paginated, sortable ----
    _table_block = ui.column().classes("w-full gap-1 pm-dig-table-block")
    with _table_block:
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Digitized clasts").classes("text-sm font-bold")
            _table_count = ui.label("").classes("text-xs text-grey-7 "
                                                "pm-dig-table-count")
        clast_table = ui.table(
            rows=[], columns=[], row_key="id", selection="multiple",
            pagination={"rowsPerPage": 50, "sortBy": "id", "descending": False},
        ).classes("w-full pm-dig-table").props("dense flat bordered")
        with ui.row().classes("w-full items-end gap-2"):
            _label_inp = ui.input(label="Label of the selected clast",
                                  placeholder="label / note") \
                .classes("flex-grow pm-dig-label").props("dense") \
                .tooltip("Optional label or note for the selected clast, "
                         "written to a 'Label' column in the truth CSV when "
                         "any clast has one.")
    _label_guard = {"quiet": False}

    def _table_scale():
        """``(pixels per unit, unit label)`` for the table: millimetres with
        a GSD, the object's unit with an object scale, pixels otherwise."""
        kind, val = _effective_scale()
        if kind == "object" and val is not None and val.valid:
            return float(val.px_per_unit), val.unit_label
        if kind == "gsd" and val:
            return 0.001 / float(val), "mm"
        return 1.0, "px"

    def _table_columns(unit):
        def num(name, label):
            return {"name": name, "label": label, "field": name,
                    "sortable": True, "align": "right"}
        return [
            num("id", "ID"),
            {"name": "origin", "label": "Origin", "field": "origin",
             "sortable": True, "align": "left"},
            num("length", f"Clast length ({unit})"),
            num("width", f"Clast width ({unit})"),
            num("eqd", f"Equivalent diameter ({unit})"),
            num("area", f"Area ({unit}²)"),
            num("orientation", "Orientation (°)"),
            num("score", "Score"),
        ]

    def _sig(v, digits=4):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return float(f"{v:.{digits}g}")

    def _table_rows():
        from functions import provenance as _prov
        from functions.digitize import measure_record_px
        ppu, unit = _table_scale()
        rows = []
        for i, rec in enumerate(state.dig_records):
            try:
                meas = measure_record_px(rec)
            except Exception:
                meas = None
            origin = _prov.record_origin(rec)
            if origin != _prov.HAND and rec.get("edited"):
                origin += " (edited)"
            row = {"id": i + 1, "origin": origin, "length": None,
                   "width": None, "eqd": None, "area": None,
                   "orientation": None, "score": _sig(rec.get("score"), 3)}
            if meas is not None:
                row.update(
                    length=_sig(meas["Clast_length"] / ppu),
                    width=_sig(meas["Clast_width"] / ppu),
                    eqd=_sig(meas["Equivalent_diameter"] / ppu),
                    area=_sig(meas["Surface_area"] / (ppu * ppu)),
                    orientation=_sig(float(meas["Orientation"]) % 180.0, 4))
            rows.append(row)
        return rows, unit

    def _refresh_records_panel():
        """The clast table follows the records: every one of them, in the
        unit in force."""
        rows, unit = _table_rows()
        clast_table.columns = _table_columns(unit)
        clast_table.rows = rows
        n = len(rows)
        _table_count.set_text(
            f"{n:,} clast{'s' if n != 1 else ''}"
            + (f" — sizes in {unit}" if n else " (none yet)"))
        _sync_table_selection()
        _sync_selection_controls()

    def _sync_table_selection():
        idx = _sel["idx"]
        ids = {i + 1 for i in _selected_indices()}
        clast_table.selected = [r for r in clast_table.rows if r["id"] in ids]
        if idx is not None:
            # Show the row: its page, when the table is in ID order.
            try:
                pag = dict(clast_table.pagination or {})
                rpp = int(pag.get("rowsPerPage") or 0)
                if rpp > 0 and pag.get("sortBy") in (None, "id") \
                        and not pag.get("descending"):
                    pag["page"] = idx // rpp + 1
                    clast_table.pagination = pag
            except Exception:
                pass
        clast_table.update()

    def _sync_selection_controls():
        idx = _sel["idx"]
        has = idx is not None and 0 <= idx < len(state.dig_records)
        try:
            if has:
                _enable(_dig_delete_btn)
            else:
                _disable(_dig_delete_btn, "Select a clast first")
            if state.dig_records and not state.dig_figures_running:
                _enable(figures_btn)
            elif not state.dig_figures_running:
                _disable(figures_btn, "No clasts on the canvas")
        except NameError:
            pass
        _label_guard["quiet"] = True
        try:
            _label_inp.value = (state.dig_records[idx].get("label", "")
                                if has else "")
            _label_inp.set_enabled(has)
        finally:
            _label_guard["quiet"] = False

    def _on_table_select(e):
        sel = list(getattr(e, "selection", None) or [])
        _select_many([int(r["id"]) - 1 for r in sel], from_table=True)

    def _on_row_click(e):
        try:
            row = e.args[1]
            _select(int(row["id"]) - 1)
        except Exception:
            pass

    clast_table.on_select(_on_table_select)
    clast_table.on("rowClick", _on_row_click, [[], ["id"], None])

    def _on_label_change(e):
        """The label applies to every selected clast."""
        if _label_guard["quiet"]:
            return
        idxs = _selected_indices()
        if not idxs:
            return
        for i in idxs:
            state.dig_records[i]["label"] = (e.value or "").strip()
        _autosave_records()
    _label_inp.on_value_change(_on_label_change)

    # ---- Export ----
    ui.separator()
    with ui.row().classes("w-full items-end gap-3 mt-2"):
        ui.input(label="Output CSV path") \
.bind_value(state, "dig_out_path") \
.classes("flex-grow") \
.tooltip("Where to save the digitized clast CSV. Schema "
                     "matches detection output so it drops into Validate.")
        def _browse_save():
            current = state.dig_out_path
            if current:
                initial = Path(current).name
                initial_dir = str(Path(current).parent)
            elif state.dig_src_path:
                initial = Path(state.dig_src_path).stem + "_truth.csv"
                initial_dir = default_starting_dir("validation")
            else:
                initial = "truth.csv"
                initial_dir = default_starting_dir("validation")
            picked = native_save_file_picker(
                title="Save digitized CSV as",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
                initialfile=initial,
                defaultextension=".csv",
                initialdir=initial_dir,
            )
            if picked:
                state.dig_out_path = picked
        ui.button("Browse…", icon="folder_open", on_click=_browse_save).props("outline")

    def _autofill_out_path():
        if (not state.dig_out_path) and state.dig_src_path:
            stem = Path(state.dig_src_path).stem
            target = Path(default_starting_dir("validation")) / f"{stem}_truth.csv"
            state.dig_out_path = str(target)
    ui.timer(1.0, _autofill_out_path)

    def _do_export():
        if not state.dig_records:
            ui.notify("No clasts digitized yet — outline some on the photograph (Work surface, above).", type="warning")
            return
        if not state.dig_out_path:
            ui.notify("Set *Output CSV path* (Results, above) first.", type="warning")
            return
        resolution, unit = _truth_scale()
        if resolution <= 0:
            ui.notify("Set *Resolution (m/pixel)* above 0 (Inputs, above), "
                      "or draw a scale segment (No GSD? Scale from an object).",
                      type="warning")
            return
        try:
            H = img_widgets["natural_size"][1]
            if H == 0:
                ui.notify("Pick a photograph in the toolbar (Work surface) first.", type="warning")
                return
            df = _write_truth(state.dig_out_path, H)
            ui.notify(
                f"✅ Wrote {len(df)} clasts to {Path(state.dig_out_path).name}"
                + (f" in {unit} units (not metres)." if unit else ". ")
                + " Drop into Validate as truth CSV.",
                type="positive", timeout=8000,
            )
        except Exception as ex:
            import traceback
            ui.notify(f"Export failed: {ex}. Check the output path (Export, below), then Export CSV again.", type="negative")
            print(traceback.format_exc())

    ui.button("Export CSV", icon="save", on_click=_do_export) \
.props("color=primary")

    # ---- The three bands, by moving what the code above built ---------- #
    # Inputs: folder card, then the shared settings. Surface: the bar
    # (Previous/Next/Image moved in first), the status line, the
    # photograph. Results: the detect console, the records, the export.
    _root = dig_toolbar.parent_slot.parent

    def _kids():
        return _root.default_slot.children
    _dig_settings_row.move(_root, target_index=_kids().index(_dig_folder_card) + 1)
    _scale_section.move(_root, target_index=_kids().index(_dig_settings_row) + 1)
    _dig_thr.move(_dig_settings_row)
    # The model Detect runs, named where its settings are.
    with _dig_settings_row:
        _dig_model_caption = ui.label("").classes(
            "text-xs text-grey-8 pm-model-caption")

    def _dig_model_text():
        _dig_model_caption.set_text(
            f"Detection model: {current_detection_model()[1]} "
            "(left panel, Settings)")
    _dig_model_text()
    ui.timer(2.0, _dig_model_text)
    image_select.move(dig_toolbar, target_index=0)
    prev_btn.move(dig_toolbar, target_index=1)
    next_btn.move(dig_toolbar, target_index=2)
    image_select.classes("min-w-[12rem] max-w-[16rem]").props("dense")
    with dig_toolbar:
        ui.separator().props("vertical").move(dig_toolbar, target_index=3)
    interactive_container.classes("pm-dig-surface")
    # One tight column for bar + status + photograph + console: the bar's
    # sticky range.
    _dig_work = ui.column().classes("w-full gap-1 pm-dig-work")
    _dig_work.move(_root, target_index=_kids().index(dig_toolbar))
    dig_toolbar.move(_dig_work)
    info_label.move(_dig_work)
    interactive_container.move(_dig_work)
    dig_log.move(_dig_work)
    # The clast table above the figures.
    _table_block.move(_dig_work)
    dig_results_col.move(_dig_work)
    _refresh_records_panel()
    _update_nav_summary()

    def _seed_dig_folder(force=False):
        """Fill the folder from the project: validation/images first, else
        its photographs. Never over a typed folder unless ``force``."""
        proj = state.current_project
        if not proj:
            return
        folder = None
        try:
            for cand in (project_path(proj, "validation") / "images",
                         project_path(proj, "validation")):
                if cand.is_dir() and any(
                        p.suffix.lower() in _images.PHOTO_EXTENSIONS
                        for p in cand.iterdir() if p.is_file()):
                    folder = cand
                    break
            if folder is None:
                # The canonical layout keeps the rectified photographs in
                # validation/orthorectified and the raw ones in
                # validation/raw, so a project switch moves the folder with
                # the project.
                from functions import project_defaults as _pdf
                for photos in (_pdf.rectified_photos(proj),
                               _pdf.raw_photos(proj), _pdf.photos(proj)):
                    if photos:
                        folder = photos[0].parent
                        break
        except Exception:
            folder = None
        if folder is None:
            return
        if state.dig_image_dir and not force:
            return
        state.dig_image_idx = -1
        state.dig_image_list = []
        state.dig_image_dir = str(folder)
        fired = (str(getattr(dir_inp, "value", "") or "") != str(folder))
        _seed_default(dir_inp, folder, force=True)
        if not fired:
            _refresh_dig_dir()

    # Arrive with the folder filled and its photograph on the surface.
    if state.dig_image_dir and os.path.isdir(state.dig_image_dir):
        _refresh_dig_dir()
    else:
        _seed_dig_folder(force=False)


# ----- Validate tab (ground-truth comparison) ---------------------------- #
def build_validate_tab():
    """Compare a detection CSV against a manual ground-truth CSV.

    Three layers: distribution comparison (K-S test, percentiles, overlaid
    histograms / CDFs / Q-Q), detection performance (mutual-nearest-
    neighbour matching within a tolerance: recall, precision, F1), and
    paired per-clast stats (regression, R2, RMSE, MAE, Bland-Altman).
    """
    def _on_proj_change():
        state.val_truth_csv = ""
        state.val_detect_csv = ""
        state.val_src_image = ""
        _seed_validate(force=True)
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Validate")
    ui.label('A detection CSV against a hand-measured truth CSV; both need x, y and the target field, in one CRS / unit.').classes("text-sm text-grey-7")

    # ---- Quadrat navigator: a folder of truth CSVs, one per quadrat ---- #
    with ui.card().classes("w-full bg-blue-1"):
        ui.markdown("**Working a list of quadrats?**")
        ui.label('Truth CSVs in a folder step through with Previous / Next; to run them all against one detection CSV use *Batch truth from folder…* at the foot of this tab.').classes("text-sm text-grey-7")
        _val_dir_row = path_input_with_browse(
            "Truth CSV folder (optional)",
            "val_truth_dir",
            kind="dir",
            default_kind="validation",
            # A lambda: the handler is defined below.
            on_change=lambda: _refresh_quadrats(),
        )

        quadrat_select = ui.select(
            [], label="Quadrat", with_input=True,
        ).classes("w-full").props("dense")

        def _refresh_quadrats(_e=None):
            """List the CSVs in the chosen folder."""
            d = (state.val_truth_dir or "").strip()
            files = []
            if d:
                try:
                    files = sorted(
                        str(p) for p in Path(d).glob("*.csv") if p.is_file())
                except OSError as ex:
                    ui.notify(f"Could not read folder: {ex}. Pick another truth folder (Inputs, above).", type="warning")
            state.val_truth_files = files
            quadrat_select.options = files
            quadrat_select.update()
            if files:
                state.val_truth_index = 0
                quadrat_select.value = files[0]
                state.val_truth_csv = files[0]
                ui.notify(f"{len(files)} quadrat CSV(s) found", type="positive")
            else:
                quadrat_select.value = None
                if d:
                    ui.notify("No .csv files in that *Truth folder* (Inputs, above)", type="warning")
            _sync_nav()

        def _on_pick(_e=None):
            v = quadrat_select.value
            if v:
                state.val_truth_csv = v
                if v in state.val_truth_files:
                    state.val_truth_index = state.val_truth_files.index(v)
            _sync_nav()

        def _step(delta: int):
            files = state.val_truth_files or []
            if not files:
                return
            i = max(0, min(len(files) - 1, state.val_truth_index + delta))
            state.val_truth_index = i
            quadrat_select.value = files[i]
            state.val_truth_csv = files[i]
            _sync_nav()

        with ui.row().classes("items-center gap-2"):
            prev_btn = ui.button("Previous", icon="chevron_left",
                                  on_click=lambda: _step(-1)) \
.props("flat dense no-caps")
            next_btn = ui.button("Next", icon="chevron_right",
                                  on_click=lambda: _step(1)) \
.props("flat dense no-caps")
            ui.button("Rescan folder", icon="refresh",
                       on_click=_refresh_quadrats) \
.props("flat dense no-caps")
            nav_label = ui.label("").classes("text-sm text-grey-7")

        def _sync_nav():
            """Disable rather than hide the navigation."""
            files = state.val_truth_files or []
            n = len(files)
            i = state.val_truth_index
            prev_btn.set_enabled(bool(files) and i > 0)
            next_btn.set_enabled(bool(files) and i < n - 1)
            nav_label.set_text(f"{i + 1} of {n}" if n else "no folder set")

        quadrat_select.on_value_change(_on_pick)
        _sync_nav()

    # ---- Inputs ----
    _val_truth_row = path_input_with_browse(
        "Truth CSV (manual measurements)",
        "val_truth_csv",
        kind="file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        default_kind="validation",
    )
    _val_detect_row = path_input_with_browse(
        "Detection CSV (model output)",
        "val_detect_csv",
        kind="file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        default_kind="vectors",
    )
    _val_src_row = path_input_with_browse(
        "Ground-truth source image (quadrat GeoTIFF — overlay foreground)",
        "val_src_image",
        kind="file",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.heic *.heif"),
                   ("All files", "*.*")],
        default_kind="validation_georectified",
    )
    # Optional UAV ortho as the overlay's background, under the
    # semi-transparent truth image; the truth image still drives the
    # nodata mask and the quadrat footprint.
    _val_uav_row = path_input_with_browse(
        "UAV ortho image (optional, background of overlay)",
        "val_uav_image",
        kind="file",
        filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png"),
                   ("All files", "*.*")],
        default_kind="images",
    )
    with ui.row().classes("items-end gap-2"):
        ui.number("Truth-on-ortho transparency (alpha)",
                   min=0.0, max=1.0, step=0.05, format="%.2f") \
.bind_value(state, "val_truth_alpha") \
.classes("w-64") \
.tooltip("Alpha for the quadrat truth image when "
                     "stacked over the ortho background (0 = invisible, "
                     "1 = fully opaque). 0.55 is the default.")

    # Field selector: the common columns once both CSVs are picked, a
    # default list until then.
    DEFAULT_FIELDS = ["Clast_length", "Clast_width", "Equivalent_diameter",
                      "Surface_area", "Orientation"]
    field_select = ui.select(
        options=DEFAULT_FIELDS,
        value="Clast_length",
        label="Field to compare",
    ).bind_value(state, "val_field").classes("w-64") \
.tooltip("Column name present in both CSVs. The dropdown updates "
                 "to show common columns once both files are loaded.")

    # x / y / clast_ID are pairing inputs; Score has no truth equivalent.
    _VAL_EXCLUDED_FIELDS = {"x", "y", "clast_id", "clast_ID", "score",
                             "Score"}

    def _refresh_field_options():
        """Offer the columns common to both CSVs."""
        try:
            import pandas as pd
            if (state.val_truth_csv and state.val_detect_csv
                and Path(state.val_truth_csv).exists()
                and Path(state.val_detect_csv).exists()):
                t_cols = set(pd.read_csv(state.val_truth_csv, nrows=1).columns)
                d_cols = set(pd.read_csv(state.val_detect_csv, nrows=1).columns)
                excluded_lc = {c.lower()
                               for c in _VAL_EXCLUDED_FIELDS}
                common = sorted(c for c in (t_cols & d_cols)
                                if c.lower() not in excluded_lc)
                if common:
                    field_select.options = common
                    if state.val_field not in common:
                        state.val_field = common[0]
                    field_select.update()
        except Exception:
            pass  # a malformed CSV is reported on Run
    ui.timer(2.0, _refresh_field_options)

    # ---- Coordinate frame: GSD per side ----
    # A georeferenced source GeoTIFF means the CSV coords are already in
    # metres: GSD is forced to 1.0 and the inputs are hidden.
    def _detect_image_georef(path: str):
        """``(is_georef, pixel_size_m)``, or ``(False, None)``."""
        if not path or not Path(path).exists():
            return False, None
        ext = Path(path).suffix.lower()
        if ext not in (".tif", ".tiff"):
            return False, None
        try:
            from osgeo import gdal
            ds = gdal.Open(path)
            if ds is None:
                return False, None
            gt = ds.GetGeoTransform()
            wkt = ds.GetProjection()
            ds = None
            if not wkt:
                return False, None
            default_gt = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            is_default = all(abs(a - b) < 1e-12
                             for a, b in zip(gt, default_gt))
            if is_default:
                return False, None
            return True, abs(float(gt[1]))
        except Exception:
            return False, None

    ui.label("Pixel CSVs at different GSDs: set each side's GSD (m/pixel) and both are paired in a common metric frame; a georeferenced GeoTIFF supplies its own and hides the inputs.").classes("text-sm text-grey-7")
    gsd_status_label = ui.label("").classes("text-sm text-grey-7 italic")
    truth_gsd_row = ui.row().classes("items-end gap-4")
    with truth_gsd_row:
        ui.number(label="Truth GSD (m/pixel, 1.0 = already in metres)",
                  step=0.0001, min=0.0001, format="%.5f") \
.bind_value(state, "val_truth_gsd") \
.classes("w-72") \
.tooltip("Ground sample distance of the image whose pixel coords "
                     "the truth CSV references. 1.0 if already in metres.")
        ui.checkbox("Truth size field also in px (scale by GSD)") \
.bind_value(state, "val_truth_size_in_px") \
.tooltip("Enable if the size column (e.g. Clast_length) values "
                     "are in pixels in the truth CSV. They will be "
                     "multiplied by truth GSD before comparison.")
    detect_gsd_row = ui.row().classes("items-end gap-4")
    with detect_gsd_row:
        ui.number(label="Detection GSD (m/pixel, 1.0 = already in metres)",
                  step=0.0001, min=0.0001, format="%.5f") \
.bind_value(state, "val_detect_gsd") \
.classes("w-72") \
.tooltip("GSD of the detection CSV's coordinate system. "
                     "Detection from the Ortho/Quadrat pipeline outputs "
                     "x,y in metres by default, so leave at 1.0 unless the "
                     "CSV is non-standard.")
        ui.checkbox("Detection size field also in px (scale by GSD)") \
.bind_value(state, "val_detect_size_in_px") \
.tooltip("Enable if the size column values are in pixels in the "
                     "detection CSV.")

    # GSD auto-detection from the filename token or the GeoTransform.
    # Filename-derived values are defaults the user may edit.
    from functions.quadrat_validation import detect_gsd_from_path

    _last_paths = {"truth": "", "detect": "", "src": "", "ortho": ""}

    def _auto_detect_gsd():
        if state.val_truth_csv != _last_paths["truth"]:
            _last_paths["truth"] = state.val_truth_csv
            res = detect_gsd_from_path(state.val_truth_csv)
            if res["gsd"] is not None and res["source"] == "filename":
                state.val_truth_gsd = float(res["gsd"])
                state.val_truth_gsd_source = "filename"
            elif res["gsd"] is None:
                state.val_truth_gsd_source = "manual"
        if state.val_detect_csv != _last_paths["detect"]:
            _last_paths["detect"] = state.val_detect_csv
            res = detect_gsd_from_path(state.val_detect_csv)
            if res["gsd"] is not None and res["source"] == "filename":
                state.val_detect_gsd = float(res["gsd"])
                state.val_detect_gsd_source = "filename"
            elif res["gsd"] is None:
                state.val_detect_gsd_source = "manual"
            # The detection CSV's frame is the UAV ortho: drive the
            # truncation panel's UAV GSD unless the user overrode it.
            if (res["gsd"] is not None
                and res["source"] == "filename"
                and state.val_dmin_auto):
                state.val_dmin_uav_gsd = float(res["gsd"])
        if state.val_src_image != _last_paths["src"]:
            _last_paths["src"] = state.val_src_image
            res = detect_gsd_from_path(state.val_src_image)
            if res["gsd"] is not None and res["source"] == "filename":
                # Only while the truth GSD still sits at its default.
                if abs(state.val_truth_gsd - 0.001) < 1e-9:
                    state.val_truth_gsd = float(res["gsd"])
                    state.val_truth_gsd_source = "filename (src image)"
        # The UAV background image is the authoritative UAV GSD.
        if state.val_uav_image != _last_paths["ortho"]:
            _last_paths["ortho"] = state.val_uav_image
            res = detect_gsd_from_path(state.val_uav_image)
            if (res["gsd"] is not None
                and state.val_dmin_auto):
                state.val_dmin_uav_gsd = float(res["gsd"])

    ui.timer(1.0, _auto_detect_gsd)

    def _refresh_gsd_visibility():
        """Hide the GSD inputs and force 1.0 for a georeferenced source
        GeoTIFF; otherwise show them with the auto-detected source."""
        try:
            is_georef, px_size = _detect_image_georef(state.val_src_image)
        except Exception:
            is_georef, px_size = False, None
        if is_georef:
            if abs(state.val_truth_gsd - 1.0) > 1e-9:
                state.val_truth_gsd = 1.0
                state.val_truth_gsd_source = "geotransform"
            if abs(state.val_detect_gsd - 1.0) > 1e-9:
                state.val_detect_gsd = 1.0
                state.val_detect_gsd_source = "geotransform"
            if state.val_truth_size_in_px:
                state.val_truth_size_in_px = False
            if state.val_detect_size_in_px:
                state.val_detect_size_in_px = False
            truth_gsd_row.visible = False
            detect_gsd_row.visible = False
            gsd_status_label.set_text(
                f"Source image is a georeferenced GeoTIFF "
                f"(pixel size ≈ {px_size:.4f} m). GSD = 1.0 (CSV "
                f"coordinates assumed in metres). Inputs hidden.")
        else:
            truth_gsd_row.visible = True
            detect_gsd_row.visible = True
            badges = []
            if state.val_truth_gsd_source != "manual":
                badges.append(
                    f"truth: {state.val_truth_gsd_source}"
                    f" ({state.val_truth_gsd:.5f} m/px)")
            if state.val_detect_gsd_source != "manual":
                badges.append(
                    f"detect: {state.val_detect_gsd_source}"
                    f" ({state.val_detect_gsd:.5f} m/px)")
            if badges:
                gsd_status_label.set_text(
                    "Auto-detected: " + "; ".join(badges)
                    + " — edit the boxes to override.")
            else:
                gsd_status_label.set_text("")
    ui.timer(1.0, _refresh_gsd_visibility)

    # ----- Quadrat footprint ----------------------------------- #
    # Both CSVs are clipped to the footprint before any comparison runs.
    with ui.expansion("Quadrat footprint (restrict comparison to this area)",
                       icon="crop_free").classes("w-full mt-2"):
        ui.markdown(
            "**Three methods for specifying which part of the UAV "
            "ortho-image the ground-truth quadrat covers.** All comparisons "
            "(pairing, K-S, Folk-Ward Δ) run on clasts whose centroid lies "
            "inside the footprint."
        )
        ui.toggle({
            "geotiff_aligned": "Truth is a georeferenced GeoTIFF",
            "centroid_dims":   "Centroid + dimensions",
            "point_corner":    "One point + its corner + dimensions",
        }).bind_value(state, "val_quad_mode") \
.tooltip("Pick the option that matches what the field team recorded.")

        quad_geotiff_card = ui.card().classes("w-full bg-grey-1")
        with quad_geotiff_card:
            ui.markdown(
                "**GeoTIFF-aligned.** The footprint is the truth image's "
                "GeoTransform bounds. No extra inputs — just make sure the "
                "Source image picker above points at the georeferenced "
                "GeoTIFF version of the quadrat.")

        quad_centroid_card = ui.card().classes("w-full bg-grey-1")
        with quad_centroid_card:
            ui.markdown("**Centroid + dimensions** (CRS units, typically metres)")
            with ui.row().classes("gap-2"):
                ui.number("Centroid x", format="%.6g") \
.bind_value(state, "val_quad_cx").classes("w-32")
                ui.number("Centroid y", format="%.6g") \
.bind_value(state, "val_quad_cy").classes("w-32")
                ui.number("Width",  min=0.0, format="%.6g") \
.bind_value(state, "val_quad_width").classes("w-32")
                ui.number("Height (0 = square)", min=0.0, format="%.6g") \
.bind_value(state, "val_quad_height").classes("w-40")

        quad_point_card = ui.card().classes("w-full bg-grey-1")
        with quad_point_card:
            ui.label('One recorded point, its corner and the dimensions: the validator infers the centroid and pairs within a circle of radius ½·diagonal.').classes("text-sm text-grey-7")
            with ui.row().classes("gap-2"):
                ui.number("Point x", format="%.6g") \
.bind_value(state, "val_quad_px").classes("w-32")
                ui.number("Point y", format="%.6g") \
.bind_value(state, "val_quad_py").classes("w-32")
                ui.select(
                    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]) \
.bind_value(state, "val_quad_corner").classes("w-24") \
.tooltip("Which corner the point represents.")
                ui.number("Width",  min=0.0, format="%.6g") \
.bind_value(state, "val_quad_width").classes("w-32")
                ui.number("Height (0 = square)", min=0.0, format="%.6g") \
.bind_value(state, "val_quad_height").classes("w-40")

        def _refresh_quad_card_visibility():
            quad_geotiff_card.visible = (state.val_quad_mode == "geotiff_aligned")
            quad_centroid_card.visible = (state.val_quad_mode == "centroid_dims")
            quad_point_card.visible = (state.val_quad_mode == "point_corner")

        _refresh_quad_card_visibility()
        ui.timer(0.5, _refresh_quad_card_visibility)

    # ---- Matching options ----
    with ui.row().classes("items-end gap-4 mt-2"):
        ui.number(label="Spatial tolerance (m, 0=auto)",
                  value=0.0, step=0.01, min=0.0, format="%.3f") \
.bind_value(state, "val_tolerance") \
.classes("w-56") \
.tooltip("Maximum distance (in METERS, after GSD conversion) "
                     "between a truth and detection for them to count as "
                     "the same clast. 0 = auto (half the median "
                     "nearest-neighbor distance in the truth set).")
        ui.checkbox("Reject size mismatches > 50%") \
.bind_value(state, "val_size_filter") \
.tooltip("Additional matching criterion: a candidate pair is "
                     "rejected if their size differs by more than 50%. "
                     "Use when overlapping clasts of very different sizes "
                     "could confuse the spatial matcher.")
        ui.checkbox("Statistical-only (skip spatial pairing)") \
.bind_value(state, "val_statistical_only") \
.tooltip("When the truth and detection CSVs come from different "
                     "scenes (e.g. an off-site quadrat vs a UAV ortho), "
                     "per-clast pairing is meaningless. Tick this to run "
                     "only the distribution comparison (K-S, percentiles, "
                     "Folk-Ward σφ/Skφ/K_G) and skip pairing, RMSE, MAE, "
                     "R² and the Bland-Altman plot.")

    # ---- Detection-limit truncation ----------------------------- #
    # The UAV detector cannot segment grains below ~k * GSD (Soloy et al.
    # 2020, k = 8); the distribution comparison runs on P(D | D >= D_min).
    with ui.expansion(
        "Detection-limit truncation (k × GSD threshold)",
        icon="filter_alt",
    ).classes("w-full mt-2"):
        ui.label('Clasts shorter than k · GSD are not segmented reliably (Soloy et al. 2020: 4 cm at 5 mm/px, k = 8 by default); below it the comparison means nothing.').classes("text-sm text-grey-7")
        ui.label().bind_text_from(
            state, "val_dmin_k_pixels",
            lambda v: (
                f"Currently active: k = {v} pixels  ·  "
                f"UAV GSD = {state.val_dmin_uav_gsd*1000:.3f} mm/px  ·  "
                f"D_min = "
                f"{(v * state.val_dmin_uav_gsd if state.val_dmin_auto else state.val_dmin_m)*1000:.2f} mm  ·  "
                f"{'APPLIED' if state.val_dmin_apply else 'NOT applied'}"
            )
        ).classes("text-sm font-mono text-grey-8") \
.style("background:#f0f3f7;padding:4px 8px;border-radius:3px;")
        with ui.row().classes("items-center gap-3"):
            ui.checkbox("Apply detection threshold (recommended)") \
.bind_value(state, "val_dmin_apply") \
.tooltip("When enabled, every grain-size distribution "
                         "plot and statistic — histogram, CDF, Q-Q, "
                         "percentiles, K-S, Folk-Ward σφ/Skφ/K_G — "
                         "operates on the conditional distribution "
                         "P(D | D ≥ D_min) on both the truth and the "
                         "detection sides. The CCDF (log-log) plot "
                         "deliberately keeps both populations end-to-end "
                         "so the divergence below D_min stays visible. "
                         "Pairwise metrics (R², RMSE, Bland-Altman) and "
                         "the spatial map / image overlay are not "
                         "affected.")
            ui.checkbox("Auto: D_min = k · GSD (Soloy 2020)") \
.bind_value(state, "val_dmin_auto") \
.tooltip("When ticked, D_min is recomputed every render "
                         "as k_pixels × UAV GSD. Untick to enter a manual "
                         "value below.")
            ui.checkbox(
                "Use max(k·GSD, min detected size)") \
.bind_value(state, "val_dmin_floor_to_observed") \
.tooltip("When the smallest detected size in this "
                         "comparison is larger than k·GSD, lift D_min "
                         "to the observed minimum. The UAV can "
                         "evidently resolve whatever it actually detected, "
                         "so anything below the smallest observed "
                         "value is genuinely beyond its reach for "
                         "this dataset. Default off; enable for the most "
                         "conservative honest threshold for the pair "
                         "under comparison.")
        with ui.row().classes("items-end gap-3"):
            ui.number("k (pixels)", min=1, max=64, step=1, format="%d") \
.bind_value(state, "val_dmin_k_pixels") \
.classes("w-32") \
.tooltip("Soloy et al. 2020 measured k = 8 pixels on the "
                         "long axis. Other workflows / camera setups may "
                         "shift this; 5 is more aggressive, 12 is more "
                         "conservative. Used only when 'Auto' is on.")
            ui.number("UAV GSD (m/pixel)",
                       min=0.0001, step=0.001, format="%.5f") \
.bind_value(state, "val_dmin_uav_gsd") \
.classes("w-40") \
.tooltip("Ground sample distance of the UAV ortho the "
                         "detections came from. Used only when 'Auto' is on.")
            ui.number("D_min (m, manual)",
                       min=0.0, step=0.005, format="%.4f") \
.bind_value(state, "val_dmin_m") \
.classes("w-40") \
.tooltip("Active value when 'Auto' is OFF. The displayed "
                         "value is also kept in sync with k · GSD when "
                         "Auto is ON so you can see what the engine will "
                         "use.")

        dmin_preview = ui.label("").classes("text-sm text-grey-7 italic")

        def _refresh_dmin():
            if state.val_dmin_auto:
                state.val_dmin_m = round(
                    state.val_dmin_k_pixels * state.val_dmin_uav_gsd, 6)
            dmin_preview.set_text(
                f"D_min = {state.val_dmin_m*1000:.2f} mm "
                f"({state.val_dmin_k_pixels} pixels × "
                f"{state.val_dmin_uav_gsd*1000:.3f} mm/px = "
                f"{state.val_dmin_k_pixels * state.val_dmin_uav_gsd*1000:.2f} mm)"
                + (" — applied" if state.val_dmin_apply
                   else " — not applied (toggle above)"))

        _refresh_dmin()
        ui.timer(0.5, _refresh_dmin)

    # ---- Run + output (queue-only: a single comparison is a queue of 1) ----
    with ui.row().classes("items-center gap-2"):
        save_job_btn = ui.button("Add to queue", icon="add") \
.props("color=primary") \
.tooltip("Snapshot the current form into the queue. Then click "
                     "Run all queued to execute. Useful when validating "
                     "several site-by-site truth/detection pairs in one go.")
        add_all_btn = ui.button("Add all common fields",
                                 icon="dynamic_feed") \
.props("flat color=primary") \
.tooltip("Read both CSVs, intersect their per-clast columns "
                     "(excluding x / y / clast_ID / Score) and queue one "
                     "comparison per shared field. Run all queued then "
                     "executes them in one go.")
        batch_truth_btn = ui.button("Batch truth from folder…",
                                    icon="folder_open") \
.props("flat color=primary") \
.tooltip("Pick a folder of truth CSVs. Every CSV found is paired "
                     "with the current Detection CSV and appended to the "
                     "queue as a separate job (one per truth file). "
                     "Requires a Detection CSV to already be set.")
        run_queue_btn = ui.button("Run all queued", icon="playlist_play") \
.props("color=primary outline") \
.tooltip("Run every queued job in order. Headline metrics "
                     "(matched / recall / precision / R² / RMSE) land "
                     "inline in the queue as each finishes.")
        def _reset_val_queue():
            n = 0
            for j in state.val_jobs:
                if j.get("status") in ("done", "error"):
                    j["status"] = "pending"
                    j["error"] = None
                    n += 1
            _render_queue.refresh()
            ui.notify(f"Reset {n} job(s) to pending.",
                       type="info" if n else "warning")
        ui.button("Reset all", icon="restart_alt",
                  on_click=_reset_val_queue) \
.props("flat color=primary") \
.tooltip("Mark every done / errored comparison pending "
                     "again so 'Run all queued' will re-run them.")
        def _clear_val_queue():
            state.val_jobs.clear()
            _render_queue.refresh()
            ui.notify("Validation queue cleared.", type="info")
        ui.button("Clear queue", icon="delete_sweep",
                  on_click=_clear_val_queue) \
.props("flat color=grey") \
.tooltip("Empty the queue. Already-completed comparisons remain "
                     "on disk in validation/results/.")
        queue_count_label = ui.label("").classes("text-sm text-grey-7 ml-2")

    # Mirror of _VAL_EXCLUDED_FIELDS, lower-cased.
    _EXCLUDE_FROM_BATCH = {"x", "y", "clast_id", "clastid", "id",
                            "score"}
    def _add_all_common_fields():
        import pandas as _pd
        t_path = state.val_truth_csv
        d_path = state.val_detect_csv
        if not (t_path and Path(t_path).exists()):
            ui.notify("Set a valid *Truth CSV* (Inputs, above) first.", type="warning")
            return
        if not (d_path and Path(d_path).exists()):
            ui.notify("Set a valid *Detection CSV* (Inputs, above) first.",
                      type="warning")
            return
        try:
            t_cols = _pd.read_csv(t_path, nrows=1).columns
            d_cols = _pd.read_csv(d_path, nrows=1).columns
        except Exception as ex:
            ui.notify(f"Could not read CSV columns: {ex}. Pick other CSVs (Inputs, above).",
                      type="negative")
            return
        common = []
        for c in t_cols:
            if c not in d_cols:
                continue
            if c.lower() in _EXCLUDE_FROM_BATCH:
                continue
            if c not in common:
                common.append(c)
        if not common:
            ui.notify(
                "No common per-clast columns between the two CSVs "
                "(after excluding x / y / clast_ID / Score). Pick CSVs from the "
                      "same pipeline (Inputs, above).",
                type="warning")
            return
        # One job per common field; everything else from the current form.
        base = _config_from_state()
        n_added = 0
        for field in common:
            cfg = dict(base)
            cfg["field"] = field
            cfg.update({"status": "pending", "metrics": None,
                        "error": None})
            state.val_jobs.append(cfg)
            n_added += 1
        ui.notify(
            f"Queued {n_added} comparison(s) — one per shared field: "
            f"{', '.join(common)}.", type="positive", timeout=6000)
        _render_queue.refresh()

    add_all_btn.on_click(_add_all_common_fields)

    def _batch_truth_from_folder():
        """One queue entry per truth CSV in a picked folder."""
        init = ""
        if state.val_truth_csv:
            init = str(Path(state.val_truth_csv).parent)
        elif state.val_detect_csv:
            init = str(Path(state.val_detect_csv).parent)
        folder = native_dir_picker(
            "Pick folder with truth CSVs", initialdir=init)
        if not folder:
            return
        if not state.val_detect_csv:
            ui.notify("Set *Detection CSV* (Inputs, above) first.", type="warning")
            return
        csv_files = sorted(Path(folder).glob("*.csv"))
        if not csv_files:
            ui.notify(
                f"No CSV files found in '{Path(folder).name}/'. Pick another truth folder (Inputs, above).",
                type="warning")
            return
        base = _config_from_state()
        n_added = 0
        for truth_path in csv_files:
            cfg = dict(base)
            cfg["truth"] = str(truth_path)
            cfg.update({"status": "pending", "metrics": None,
                        "error": None})
            state.val_jobs.append(cfg)
            n_added += 1
        _render_queue.refresh()
        ui.notify(
            f"Added {n_added} truth CSV(s) from "
            f"'{Path(folder).name}/' — "
            f"click 'Run all queued' to execute.",
            type="positive", timeout=5000)

    batch_truth_btn.on_click(_batch_truth_from_folder)

    # ---- Job queue panel ----
    queue_container = ui.column().classes("w-full mt-1")

    log_widget = live_log("validate", build_log_console(height="h-32"))

    summary_container = ui.column().classes("w-full")
    dist_container = ui.column().classes("w-full")
    plots_container = ui.column().classes("w-full")
    aggregate_container = ui.column().classes("w-full")

    # The compute runs through ``nicegui.run.io_bound``; the ``await``
    # continuation lands back on the loop with the slot stack in place,
    # so UI calls happen directly.

    def _config_from_state():
        """Snapshot every val_* form field into a config dict."""
        return {
            "truth": state.val_truth_csv,
            "detect": state.val_detect_csv,
            "src_image": state.val_src_image,
            "uav_image": state.val_uav_image,
            "truth_alpha": state.val_truth_alpha,
            "field": state.val_field,
            "tolerance": state.val_tolerance,
            "size_filter": state.val_size_filter,
            "truth_gsd": state.val_truth_gsd,
            "detect_gsd": state.val_detect_gsd,
            "truth_px": state.val_truth_size_in_px,
            "detect_px": state.val_detect_size_in_px,
            "statistical_only": state.val_statistical_only,
            "quad_mode": state.val_quad_mode,
            "quad_cx": state.val_quad_cx, "quad_cy": state.val_quad_cy,
            "quad_px": state.val_quad_px, "quad_py": state.val_quad_py,
            "quad_width": state.val_quad_width,
            "quad_height": state.val_quad_height,
            "quad_corner": state.val_quad_corner,
            # dmin_m is the threshold the engine uses; auto / k / GSD are
            # carried for the report.
            "dmin_apply": state.val_dmin_apply,
            "dmin_auto": state.val_dmin_auto,
            "dmin_k_pixels": int(state.val_dmin_k_pixels),
            "dmin_uav_gsd": float(state.val_dmin_uav_gsd),
            "dmin_m": float(
                state.val_dmin_k_pixels * state.val_dmin_uav_gsd
                if state.val_dmin_auto else state.val_dmin_m),
            "dmin_floor_to_observed": bool(
                state.val_dmin_floor_to_observed),
        }

    def _compute_validation(cfg, log=None):
        """Run one comparison from a config dict; returns everything the
        report renderer needs. Raises on missing inputs / columns."""
        import pandas as pd
        import numpy as np
        from functions import validation as _v
        if log: log(f"Loading truth CSV: {cfg['truth']}")
        truth_df = pd.read_csv(cfg["truth"])
        if log: log(f"  shape = {truth_df.shape}, "
                    f"columns = {list(truth_df.columns)[:6]}…")
        if log: log(f"Loading detection CSV: {cfg['detect']}")
        detect_df = pd.read_csv(cfg["detect"])
        if log: log(f"  shape = {detect_df.shape}")

        for df, name in ((truth_df, "Truth"), (detect_df, "Detection")):
            for col in ("x", "y", cfg["field"]):
                if col not in df.columns:
                    raise ValueError(
                        f"{name} CSV missing column '{col}'. "
                        f"Available: {list(df.columns)}")

        truth_df = truth_df.copy()
        detect_df = detect_df.copy()

        # geotiff_aligned: a hand-digitised truth CSV is in pixel (col, row)
        # of the source image; forward-transform through the GeoTransform
        # (a scalar GSD ignores the origin and the row flip). The
        # reprojection is a no-op on frames already in world coords.
        _truth_reprojected = False
        _detect_reprojected = False
        if (cfg.get("quad_mode", "geotiff_aligned") == "geotiff_aligned"
                and cfg.get("src_image")):
            try:
                from functions import quadrat_validation as _qv
                truth_df, _truth_reprojected = _qv.reproject_pixels_to_world(
                    cfg["src_image"], truth_df, x_col="x", y_col="y")
                detect_df, _detect_reprojected = _qv.reproject_pixels_to_world(
                    cfg["src_image"], detect_df, x_col="x", y_col="y")
                if log and _truth_reprojected:
                    log("  [coords] truth CSV is in source-image PIXEL "
                        "space; forward-transformed to world coords via "
                        "the GeoTransform.")
                if log and _detect_reprojected:
                    log("  [coords] detection CSV is in source-image PIXEL "
                        "space; forward-transformed to world coords via "
                        "the GeoTransform.")
            except Exception as ex:
                if log:
                    log(f"  [coords] pixel→world reprojection skipped: {ex}")

        # Scalar GSD scaling only when not already reprojected above.
        if cfg["truth_gsd"] != 1.0 and not _truth_reprojected:
            truth_df["x"] = truth_df["x"] * cfg["truth_gsd"]
            truth_df["y"] = truth_df["y"] * cfg["truth_gsd"]
            if log: log(f"Scaled truth coords by GSD={cfg['truth_gsd']:.5f} m/px")
        if cfg["detect_gsd"] != 1.0 and not _detect_reprojected:
            detect_df["x"] = detect_df["x"] * cfg["detect_gsd"]
            detect_df["y"] = detect_df["y"] * cfg["detect_gsd"]
            if log: log(f"Scaled detect coords by GSD={cfg['detect_gsd']:.5f} m/px")
        if cfg["truth_px"] and cfg["truth_gsd"] != 1.0:
            truth_df[cfg["field"]] = truth_df[cfg["field"]] * cfg["truth_gsd"]
            if log: log(f"Scaled truth.{cfg['field']} by GSD")
        if cfg["detect_px"] and cfg["detect_gsd"] != 1.0:
            detect_df[cfg["field"]] = detect_df[cfg["field"]] * cfg["detect_gsd"]
            if log: log(f"Scaled detect.{cfg['field']} by GSD")

        # The outlines beside each CSV go through the same mapping as its
        # centroids, for the image overlay.
        try:
            from functions import clast_geometry as _CG
            _gt = None
            if _truth_reprojected or _detect_reprojected:
                from osgeo import gdal as _gdal
                _ds = _gdal.Open(str(cfg["src_image"]))
                _gt = _ds.GetGeoTransform() if _ds is not None else None
                _ds = None
            for _df, _csv, _rep, _g in (
                    (truth_df, cfg["truth"], _truth_reprojected, cfg["truth_gsd"]),
                    (detect_df, cfg["detect"], _detect_reprojected, cfg["detect_gsd"])):
                _cs = _CG.read_contours(_csv)
                if _cs is None:
                    continue
                if _rep and _gt is not None:
                    _cs = _CG.transform_contours(_cs, _CG.world_frame(_gt))
                elif _g != 1.0 and not _rep:
                    _cs = _CG.transform_contours(
                        _cs, lambda xs, ys, _g=_g: (xs * _g, ys * _g))
                _CG.attach_contours(_df, _cs, frame="world")
        except Exception as _cex:
            if log: log(f"  [outlines] not drawn: {_cex}")

        # The quadrat footprint; both frames are clipped to it.
        footprint = None
        try:
            from functions import quadrat_validation as _qv
            qmode = cfg.get("quad_mode", "geotiff_aligned")
            if qmode == "geotiff_aligned":
                if cfg.get("src_image"):
                    try:
                        footprint = _qv.quadrat_from_geotiff(cfg["src_image"])
                    except (FileNotFoundError, ValueError) as ex:
                        if log: log(f"  [quadrat] geotiff_aligned skipped: {ex}")
                    # The placement provenance travels with the raster.
                    prov = _qv.placement_provenance(cfg["src_image"])
                    if log and prov.get("placement"):
                        score = prov.get("agreement_score")
                        log(f"  [quadrat] placement: {prov['placement']}"
                            + (f", agreement {score:.3f}" if score is not None
                               else "")
                            + ("  <- placed by hand, not matched"
                               if prov["placement"] != "fitted" else ""))
            elif qmode == "centroid_dims":
                footprint = _qv.quadrat_from_centroid(
                    cfg["quad_cx"], cfg["quad_cy"],
                    cfg["quad_width"],
                    cfg["quad_height"] if cfg["quad_height"] > 0 else None,
                )
            elif qmode == "point_corner":
                footprint = _qv.quadrat_from_point_and_corner(
                    cfg["quad_px"], cfg["quad_py"],
                    cfg["quad_corner"],
                    cfg["quad_width"],
                    cfg["quad_height"] if cfg["quad_height"] > 0 else None,
                )
            if footprint is not None:
                n_t0, n_d0 = len(truth_df), len(detect_df)
                truth_df = _qv.clip_to_footprint(truth_df, footprint,
                                                   x_col="x", y_col="y")
                detect_df = _qv.clip_to_footprint(detect_df, footprint,
                                                    x_col="x", y_col="y")
                if log:
                    log(f"  [quadrat] mode={qmode} kind={footprint.kind} "
                        f"bounds=({footprint.bounds[0]:.3f}, "
                        f"{footprint.bounds[1]:.3f})–"
                        f"({footprint.bounds[2]:.3f}, "
                        f"{footprint.bounds[3]:.3f})")
                    log(f"  [quadrat] truth clipped {n_t0} → {len(truth_df)}, "
                        f"detect clipped {n_d0} → {len(detect_df)}")

            # The footprint's bounding box extends into nodata margins;
            # detections there are not false positives, so both frames
            # are clipped to the truth image's valid pixels.
            if cfg.get("src_image"):
                try:
                    mb = _qv.valid_pixel_mask_from_image(cfg["src_image"])
                    if mb is not None:
                        n_t1, n_d1 = len(truth_df), len(detect_df)
                        truth_df = _qv.clip_to_valid_pixels(
                            truth_df, mb, x_col="x", y_col="y")
                        detect_df = _qv.clip_to_valid_pixels(
                            detect_df, mb, x_col="x", y_col="y")
                        if log:
                            log(f"  [nodata-mask] valid-pixel mask "
                                f"applied from {Path(cfg['src_image']).name} "
                                f"(W×H = {mb['W']}×{mb['H']}, "
                                f"{int(mb['mask'].sum())} valid px)")
                            log(f"  [nodata-mask] truth {n_t1} → "
                                f"{len(truth_df)}, detect {n_d1} → "
                                f"{len(detect_df)} (dropped pixels in "
                                f"dark/bright nodata)")
                except Exception as ex:
                    if log:
                        log(f"  [nodata-mask] WARN: skipped: {ex}")
        except Exception as ex:
            if log: log(f"  [quadrat] WARN: footprint construction failed: {ex}")

        # Detection-limit truncation before pairing and stats, so every
        # downstream number reflects P(D | D >= D_min). The full frames
        # are kept so the histogram can grey out the sub-D_min bins.
        # Non-size fields are never truncated.
        from functions.units import is_size_field_for_phi as \
            _is_size_for_dmin
        truth_df_full = truth_df.copy()
        detect_df_full = detect_df.copy()
        _dmin_apply_pre = (
            bool(cfg.get("dmin_apply", False))
            and float(cfg.get("dmin_m", 0.0)) > 0
            and _is_size_for_dmin(cfg["field"])
        )
        _dmin_m_pre = float(cfg.get("dmin_m", 0.0))
        # Optional floor: lift D_min to the smallest observed detection.
        _dmin_source_label = "k·GSD"
        if (_dmin_apply_pre
            and cfg.get("dmin_floor_to_observed", False)
            and cfg["field"] in detect_df.columns
            and len(detect_df) > 0):
            try:
                _det_min = float(
                    detect_df[cfg["field"]].dropna().min())
                if _det_min > _dmin_m_pre:
                    if log:
                        log(
                            f"  [trunc] min(detected) = "
                            f"{_det_min*1000:.2f} mm > "
                            f"k·GSD = {_dmin_m_pre*1000:.2f} mm; "
                            f"floor lifts D_min to observed.")
                    _dmin_m_pre = _det_min
                    _dmin_source_label = "min(detected)"
            except Exception:
                pass
        if _dmin_apply_pre:
            n_t0, n_d0 = len(truth_df), len(detect_df)
            truth_df = truth_df[
                truth_df[cfg["field"]] >= _dmin_m_pre].copy()
            detect_df = detect_df[
                detect_df[cfg["field"]] >= _dmin_m_pre].copy()
            if log:
                log(
                    f"  [trunc] D_min = {_dmin_m_pre*1000:.2f} mm "
                    f"({_dmin_source_label}: "
                    f"{cfg.get('dmin_k_pixels', '?')} px x "
                    f"{cfg.get('dmin_uav_gsd', 0)*1000:.3f} mm/px); "
                    f"truth {n_t0} -> {len(truth_df)}, detect "
                    f"{n_d0} -> {len(detect_df)} (sub-D_min dropped "
                    f"before pairing & stats)")

        # Statistical-only: no pairing (the two CSVs are different scenes).
        if cfg.get("statistical_only"):
            if log:
                log("Statistical-only mode: skipping spatial pairing, "
                    "computing distribution comparison only.")
            pair_res = {
                "matched_pairs": [],
                "unmatched_truth": list(range(len(truth_df))),
                "unmatched_detect": list(range(len(detect_df))),
                "used_tolerance": 0.0,
                "statistical_only": True,
            }
            metrics = _v.detection_metrics(pair_res)
        else:
            tol = cfg["tolerance"] if cfg["tolerance"] > 0 else None
            size_col = cfg["field"] if cfg["size_filter"] else None
            if log: log(f"Pairing with tolerance="
                        f"{'auto' if tol is None else f'{tol:.3f} m'}"
                        f"{' + size filter' if size_col else ''}")
            pair_res = _v.pair_csvs(
                truth_df, detect_df,
                x_col="x", y_col="y",
                tolerance=tol, size_col=size_col)
            if log: log(
                f"  used tolerance: {pair_res['used_tolerance']:.3f} m, "
                f"matched {len(pair_res['matched_pairs'])}, "
                f"unmatched truth {len(pair_res['unmatched_truth'])}, "
                f"unmatched detect {len(pair_res['unmatched_detect'])}"
            )
            metrics = _v.detection_metrics(pair_res)

        # Distribution stats on the (possibly truncated) frames, plus a
        # parallel dist_full on the untouched copies for the report.
        t_vals = truth_df[cfg["field"]].to_numpy()
        d_vals = detect_df[cfg["field"]].to_numpy()
        dist = _v.compute_distribution_stats(
            t_vals, d_vals, field=cfg["field"])

        from functions import truncation as _trunc  # noqa: F401
        dmin_apply = _dmin_apply_pre
        dmin_m = _dmin_m_pre
        if dmin_apply:
            t_full = truth_df_full[cfg["field"]].to_numpy()
            d_full = detect_df_full[cfg["field"]].to_numpy()
            dist_full = _v.compute_distribution_stats(
                t_full, d_full, field=cfg["field"])
            dist_full["truth_n_below_dmin"] = int(
                ((t_full < dmin_m) & np.isfinite(t_full)).sum())
            dist_full["detect_n_below_dmin"] = int(
                ((d_full < dmin_m) & np.isfinite(d_full)).sum())
        elif log and cfg.get("dmin_apply", False) and not dist.get(
                "phi_meaningful", False):
            log(f"  [trunc] skipped: field '{cfg['field']}' is not a "
                "grain-size field (phi-statistics undefined); "
                "truncation would not be physically meaningful here.")
            dist_full = None
        else:
            dist_full = None
        if pair_res["matched_pairs"]:
            t_paired = truth_df[cfg["field"]].iloc[
                [p[0] for p in pair_res["matched_pairs"]]].to_numpy()
            d_paired = detect_df[cfg["field"]].iloc[
                [p[1] for p in pair_res["matched_pairs"]]].to_numpy()
            # Orientation is an axis direction on [0, 180): wrap each
            # detection to the representative nearest its truth value so
            # 179 vs 1 is a 2 degree error, not 178.
            is_angle = "orientation" in cfg["field"].lower()
            if is_angle:
                t_paired = np.mod(t_paired, 180.0)
                d_paired = np.mod(d_paired, 180.0)
                diff = d_paired - t_paired
                d_paired = np.where(diff >  90.0, d_paired - 180.0, d_paired)
                d_paired = np.where(diff < -90.0, d_paired + 180.0, d_paired)
                if log:
                    log("  [angle] wrapped orientation differences into "
                        "(-90°, +90°] before RMSE / bias / regression.")
            paired = _v.compute_paired_stats(t_paired, d_paired)
        else:
            paired = None
        # dist["full"] is the untruncated parallel; dist["truncated"] is a
        # copy of dist itself, which the report rendering reads.
        dist["full"] = dist_full
        dist["truncated"] = (dict(dist) if dmin_apply else None)
        if dmin_apply and dist_full is not None:
            dist["truncated"]["truth_n_below_dmin"] = dist_full.get(
                "truth_n_below_dmin", 0)
            dist["truncated"]["detect_n_below_dmin"] = dist_full.get(
                "detect_n_below_dmin", 0)
        dist["d_min_m"] = dmin_m if dmin_apply else None
        dist["d_min_k_pixels"] = (int(cfg.get("dmin_k_pixels", 0))
                                   if dmin_apply else None)
        dist["d_min_uav_gsd"] = (float(cfg.get("dmin_uav_gsd", 0.0))
                                  if dmin_apply else None)
        # The unfiltered frames ride on dist so the histogram can grey
        # out the sub-D_min bins without changing the return tuple.
        dist["_truth_df_full"] = truth_df_full
        dist["_detect_df_full"] = detect_df_full
        return metrics, dist, paired, pair_res, truth_df, detect_df

    # ---- Job queue management ----
    def _val_primary(job, idx):
        return Path(job["truth"]).name

    def _val_params(job, idx):
        return f"vs {Path(job['detect']).name}   field={job['field']}"

    def _val_extra(job, idx):
        m = job.get("metrics")
        if m:
            # RMSE is in the field's native unit.
            rmse_str = ""
            if m.get("r2") is not None:
                rmse_str = (
                    f", R²={m['r2']:.2f}, "
                    f"RMSE={_format_value_with_unit(m['rmse'], job['field'])}"
                )
            ui.label(
                f"→ matched {m['n_matched']}/{m['n_truth']} "
                f"(R={m['recall']:.2f}, P={m['precision']:.2f}, "
                f"F1={m['f1']:.2f}{rmse_str})"
            ).classes("text-xs text-green-7")

    def _val_delete(idx):
        state.val_jobs.pop(idx)
        _render_queue.refresh()

    @ui.refreshable
    def _render_queue():
        render_queue(
            queue_container, state.val_jobs,
            title="Queued jobs",
            count_label=queue_count_label,
            primary_text=_val_primary,
            params_text=_val_params,
            render_extra=_val_extra,
            on_delete=_val_delete,
        )

    def _save_as_job():
        if not state.val_truth_csv or not Path(state.val_truth_csv).exists():
            ui.notify("Set a valid *Truth CSV* (Inputs, above) first.", type="warning")
            return
        if not state.val_detect_csv or not Path(state.val_detect_csv).exists():
            ui.notify("Set a valid *Detection CSV* (Inputs, above) first.", type="warning")
            return
        cfg = _config_from_state()
        cfg.update({"status": "pending", "metrics": None, "error": None})
        state.val_jobs.append(cfg)
        ui.notify(f"Job #{len(state.val_jobs)} added to queue.", type="positive")
        _render_queue.refresh()

    save_job_btn.on_click(_save_as_job)
    _render_queue()

    def _run_one_validation_job(job):
        """One queued comparison, on the calling thread (wrapped in
        ``run.io_bound`` by the caller). Touches no GUI beyond
        ``log_widget.push``. Returns ``(metrics, dist, paired, pair_res,
        truth_df, detect_df, summary)``."""
        m, dist, paired, pair_res, tdf, ddf = _compute_validation(
            job, log=log_widget.push)
        # Headline metrics, including distribution-level ones: the model
        # is conservative, so recall alone under-reports its usefulness.
        summary = {
            "n_truth": m["n_truth"],
            "n_detect": m["n_detect"],
            "n_matched": m["n_matched"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "r2": (paired.get("r_squared") if paired else None),
            "rmse": (paired.get("rmse") if paired else None),
            "bias": (paired.get("bias") if paired else None),
            "truth_D50": dist.get("truth_d50"),
            "detect_D50": dist.get("detect_d50"),
            "truth_D84": dist.get("truth_d84"),
            "detect_D84": dist.get("detect_d84"),
            "D50_relerr": _safe_relerr(
                dist.get("detect_d50"), dist.get("truth_d50")),
            "D84_relerr": _safe_relerr(
                dist.get("detect_d84"), dist.get("truth_d84")),
            "ks_statistic": dist.get("ks_statistic"),
            "ks_p_value": dist.get("ks_p_value"),
        }
        job["metrics"] = summary
        # Retain per-quadrat size arrays for the batch summary (quadrat =
        # truth, ortho = detect; raw = post-clip pre-D_min, filt = post-D_min).
        try:
            import numpy as _np
            _fld = job["field"]

            def _vals(_df):
                try:
                    v = _np.asarray(_df[_fld], dtype=float)
                    return v[_np.isfinite(v)].tolist()
                except Exception:
                    return []
            job["sizes"] = {
                "truth_filt": _vals(tdf),
                "detect_filt": _vals(ddf),
                "truth_raw": _vals(dist.get("_truth_df_full")),
                "detect_raw": _vals(dist.get("_detect_df_full")),
            }
            job["quad_label"] = Path(job["truth"]).stem[:18]
        except Exception:
            job["sizes"] = {}
        job["figures"] = {}

        # Diagnostic figures, from the same builder the live renderer and
        # the PDF report use; closed after saving.
        try:
            figs = _build_validation_figures(
                m, dist, paired, pair_res, tdf, ddf,
                job["field"],
                src_image=job.get("src_image") or None,
                truth_gsd=job.get("truth_gsd", 1.0),
                uav_image=job.get("uav_image") or None,
                truth_alpha=job.get("truth_alpha", 0.55),
                d_min_m=(job.get("dmin_m")
                          if job.get("dmin_apply") else None))
            from pathlib import Path as _P
            import hashlib as _hl
            hi = (f"{job['truth']}|{job['detect']}|"
                  f"{job.get('field')}|{job.get('truth_gsd')}|"
                  f"{job.get('detect_gsd')}|{job.get('tolerance')}")
            short_hash = _hl.sha1(hi.encode("utf-8")).hexdigest()[:8]
            fig_stem = short_hash
            val_root = None
            for p in _P(job["truth"]).resolve().parents:
                if p.name == "validation":
                    val_root = p
                    break
                if (p / "validation").is_dir():
                    val_root = p / "validation"
                    break
            if val_root is None:
                val_root = _P(job["truth"]).parent / "validation"
            fig_dir = val_root / "results" / "figures" / fig_stem
            saved_figs = _save_validation_figures(
                figs, fig_dir, fig_stem, log=log_widget.push)
            import matplotlib.pyplot as _plt
            for _f in figs.values():
                _plt.close(_f)
            job["figures"] = {
                name: str(p.relative_to(val_root / "results"))
.replace("\\", "/")
                for name, p in saved_figs.items()
            }
        except Exception as fig_ex:
            log_widget.push(
                f"  [warn] could not save figures: {fig_ex}")
            import traceback
            log_widget.push(traceback.format_exc())

        # Persist for the Report tab; best-effort.
        try:
            _persist_validation_result(
                job, summary, dist, paired, pair_res,
                figure_paths=job.get("figures"))
        except Exception as save_ex:
            from functions._logging import get_logger as _gl
            _gl("gui.app").warning(
                "Could not save validation result: %s",
                save_ex, exc_info=True)
            log_widget.push(
                f"  [warn] could not save report: {save_ex} "
                "(full traceback in log file)")
            import traceback
            log_widget.push(traceback.format_exc())

        return m, dist, paired, pair_res, tdf, ddf, summary

    async def _run_all_queued():
        """Run every pending job in the queue, sequentially; each compute
        step is awaited via ``run.io_bound``."""
        pending = [j for j in state.val_jobs
                   if j.get("status") in (None, "pending", "error", "interrupted")]
        if not pending:
            ui.notify("Nothing to run (all queued jobs are done).",
                      type="info")
            return
        run_queue_btn.props("loading")
        _disable(run_queue_btn, "Queue running — the log below shows progress")
        _disable(save_job_btn, "Queue running — add pairs once it has finished")
        log_widget.clear()

        from datetime import datetime as _dt
        from nicegui import run

        try:
            log_widget.push("=" * 60)
            log_widget.push(
                f"Batch validation started at "
                f"{_dt.now().isoformat(timespec='seconds')} "
                f"({len(pending)} job(s))")
            log_widget.push("=" * 60)
            summary_container.clear()
            dist_container.clear()
            plots_container.clear()
            aggregate_container.clear()
            _render_queue.refresh()

            for ji, job in enumerate(pending):
                log_widget.push(
                    f"\n[{ji + 1}/{len(pending)}] "
                    f"{Path(job['truth']).name} vs "
                    f"{Path(job['detect']).name}")
                job["status"] = "running"
                _render_queue.refresh()
                try:
                    result = await run.io_bound(
                        _run_one_validation_job, job)
                    m, dist, paired, pair_res, tdf, ddf, summary = result
                    job["status"] = "done"

                    line = (f"  ✓ matched {summary['n_matched']}/"
                            f"{summary['n_truth']} "
                            f"(R={summary['recall']:.2f}, "
                            f"P={summary['precision']:.2f}, "
                            f"F1={summary['f1']:.2f}")
                    if summary['D50_relerr'] is not None:
                        line += (
                            f", D50 err={summary['D50_relerr']*100:+.1f}%, "
                            f"D84 err={summary['D84_relerr']*100:+.1f}%")
                    line += ")"
                    log_widget.push(line)
                    _render_queue.refresh()

                    # The renderer reads the form state: swap in this
                    # job's values, then restore.
                    saved = (state.val_field, state.val_truth_csv,
                             state.val_detect_csv, state.val_src_image)
                    try:
                        state.val_field = job["field"]
                        state.val_truth_csv = job["truth"]
                        state.val_detect_csv = job["detect"]
                        state.val_src_image = (job.get("src_image", "")
                                                or "")
                        _render_report(m, dist, paired, pair_res,
                                        tdf, ddf, job=job)
                    except Exception as render_ex:
                        import traceback
                        log_widget.push(
                            f"  [render_report] {render_ex}")
                        log_widget.push(traceback.format_exc())
                    finally:
                        (state.val_field, state.val_truth_csv,
                         state.val_detect_csv, state.val_src_image) = saved
                except Exception as ex:
                    import traceback
                    job["status"] = "error"
                    job["error"] = f"{type(ex).__name__}: {ex}"
                    log_widget.push(f"  ✗ {job['error']}")
                    log_widget.push(traceback.format_exc())
                    _render_queue.refresh()

            _render_aggregate()
            log_widget.push("\n✅ Batch finished.")
        except Exception as ex:
            import traceback as _tb
            log_widget.push(
                f"\n❌ Batch crashed: {type(ex).__name__}: {ex}")
            log_widget.push(_tb.format_exc())
        finally:
            run_queue_btn.props(remove="loading")
            _enable(run_queue_btn)
            _enable(save_job_btn)

    run_queue_btn.on_click(_run_all_queued)

    def _render_aggregate():
        """Show a compact summary table of every done job + aggregated means."""
        aggregate_container.clear()
        done = [j for j in state.val_jobs
                if j.get("status") == "done" and j.get("metrics")]
        if not done:
            return
        with aggregate_container:
            ui.separator()
            ui.markdown(f"#### Batch summary — {len(done)} job(s)")
            with ui.grid(columns=8).classes("w-full text-sm"):
                for h in ("Job", "Field", "Truth", "Det", "Matched",
                         "Recall", "Precision", "F1"):
                    ui.label(h).classes("font-bold")
                for ji, job in enumerate(done):
                    m = job["metrics"]
                    ui.label(f"#{ji+1}").classes("font-mono")
                    ui.label(job["field"]).classes("text-xs")
                    ui.label(str(m["n_truth"]))
                    ui.label(str(m["n_detect"]))
                    ui.label(str(m["n_matched"]))
                    ui.label(f"{m['recall']:.3f}")
                    ui.label(f"{m['precision']:.3f}")
                    ui.label(f"{m['f1']:.3f}")
            # recall / precision / F1 / R2 are unit-independent and average
            # across fields; RMSE is grouped by field.
            import numpy as _np
            def _mean(key, jobs=None):
                src = jobs if jobs is not None else done
                vals = [d["metrics"][key] for d in src
                        if d["metrics"].get(key) is not None]
                return _np.mean(vals) if vals else None
            mr = _mean("recall"); mp = _mean("precision"); mf = _mean("f1")
            mr2 = _mean("r2")
            parts = [f"mean recall = {mr:.3f}" if mr is not None else None,
                     f"mean precision = {mp:.3f}" if mp is not None else None,
                     f"mean F1 = {mf:.3f}" if mf is not None else None,
                     f"mean R² = {mr2:.3f}" if mr2 is not None else None]
            jobs_by_field = {}
            for d in done:
                jobs_by_field.setdefault(d["field"], []).append(d)
            for field, fjobs in jobs_by_field.items():
                rmse = _mean("rmse", jobs=fjobs)
                if rmse is None:
                    continue
                parts.append(
                    f"mean RMSE ({field}) = "
                    f"{_format_value_with_unit(rmse, field)}"
                )
            parts = [p for p in parts if p]
            if parts:
                ui.label("Aggregate: " + " · ".join(parts)).classes("text-sm text-grey-8 mt-2")
            _render_batch_summary(done)

    def _render_batch_summary(done):
        """Validation QC across the quadrat batch (one quadrat = one point):
        error vs grain size, ortho vs quadrat mean and D50, per-quadrat
        violins, and a written QC discussion. Assumes grain-size fields."""
        from functions import validation as _v
        import numpy as _np
        import base64 as _b64, io as _io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        try:
            summ = _v.compute_batch_summary(done)
        except Exception as ex:
            ui.label(f"Batch summary failed: {ex}").classes("text-red-7")
            return
        pts = summ["points"]
        if not pts:
            return
        SC = 1000.0  # metres → mm for display

        def _img(fig):
            buf = _io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
            _plt.close(fig); buf.seek(0)
            return "data:image/png;base64," + _b64.b64encode(
                buf.read()).decode()

        imgs = []
        mt = _np.array([p["mean_truth"] for p in pts], float) * SC
        md = _np.array([p["mean_detect"] for p in pts], float) * SC
        rmse = _np.array([(p["rmse"] if p["rmse"] is not None else _np.nan)
                          for p in pts], float) * SC

        # 1) per-clast RMSE vs mean grain size
        ok = _np.isfinite(mt) & _np.isfinite(rmse)
        if ok.sum() >= 1:
            fig, ax = _plt.subplots(figsize=(5.2, 4))
            ax.scatter(mt[ok], rmse[ok], s=30, color="#225599")
            for p, x, y in zip(pts, mt, rmse):
                if _np.isfinite(x) and _np.isfinite(y):
                    ax.annotate(p["label"], (x, y), fontsize=6, alpha=0.6)
            ax.set_xlabel("Mean clast size (mm)")
            ax.set_ylabel("Paired RMSE (mm)")
            ax.set_title("Per-clast error vs grain size\n(1 quadrat = 1 point)",
                         fontsize=10)
            ax.grid(True, alpha=0.3)
            imgs.append(_img(fig))

        # 2) ortho mean vs quadrat mean (+ 1:1)
        ok2 = _np.isfinite(mt) & _np.isfinite(md)
        if ok2.sum() >= 1:
            fig, ax = _plt.subplots(figsize=(5.2, 4))
            ax.scatter(mt[ok2], md[ok2], s=30, color="#aa5522")
            hi = max(_np.nanmax(mt[ok2]), _np.nanmax(md[ok2])) * 1.1
            ax.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
            ax.set_xlim(0, hi); ax.set_ylim(0, hi)
            ax.set_xlabel("Quadrat mean size (mm)")
            ax.set_ylabel("Ortho mean size (mm)")
            ax.set_title("Ortho vs Quadrat mean grain size", fontsize=10)
            ax.legend(); ax.grid(True, alpha=0.3)
            imgs.append(_img(fig))

        # 3) ortho D50 vs quadrat D50 (+ 1:1)
        d50t = _np.array([(p["d50_truth"] or _np.nan) for p in pts], float) * SC
        d50d = _np.array([(p["d50_detect"] or _np.nan) for p in pts], float) * SC
        ok3 = _np.isfinite(d50t) & _np.isfinite(d50d)
        if ok3.sum() >= 1:
            fig, ax = _plt.subplots(figsize=(5.2, 4))
            ax.scatter(d50t[ok3], d50d[ok3], s=30, color="#227755")
            hi = max(_np.nanmax(d50t[ok3]), _np.nanmax(d50d[ok3])) * 1.1
            ax.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
            ax.set_xlim(0, hi); ax.set_ylim(0, hi)
            ax.set_xlabel("Quadrat D50 (mm)")
            ax.set_ylabel("Ortho D50 (mm)")
            ax.set_title("Ortho vs Quadrat D50", fontsize=10)
            ax.legend(); ax.grid(True, alpha=0.3)
            imgs.append(_img(fig))

        # 4 & 5) per-quadrat Quadrat-vs-Ortho size violins (raw + filtered)
        from matplotlib.patches import Patch

        def _violin_pair(violins, title):
            groups = [(lbl, t, d) for lbl, t, d in violins
                      if len(t) >= 2 and len(d) >= 2]
            if not groups:
                return None
            fig, ax = _plt.subplots(
                figsize=(max(6.0, 0.9 * len(groups) + 2), 4))
            ticks, labels, pos = [], [], 1.0
            for lbl, t, d in groups:
                vt = ax.violinplot([t * SC], positions=[pos], widths=0.35,
                                   showmedians=True)
                vd = ax.violinplot([d * SC], positions=[pos + 0.4],
                                   widths=0.35, showmedians=True)
                for b in vt["bodies"]:
                    b.set_facecolor("#2a7fff"); b.set_alpha(0.5)
                for b in vd["bodies"]:
                    b.set_facecolor("#ff7f2a"); b.set_alpha(0.5)
                ticks.append(pos + 0.2); labels.append(lbl); pos += 1.2
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Clast size (mm)")
            ax.set_title(title, fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(handles=[
                Patch(color="#2a7fff", alpha=0.5, label="Quadrat"),
                Patch(color="#ff7f2a", alpha=0.5, label="Ortho")], fontsize=8)
            return _img(fig)

        for v in (_violin_pair(summ["violins_raw"],
                               "Per-quadrat size — Quadrat vs Ortho (raw)"),
                  _violin_pair(summ["violins_filt"],
                               "Per-quadrat size — Quadrat vs Ortho "
                               "(after sub-D_min filter)")):
            if v:
                imgs.append(v)

        ui.separator().classes("mt-2")
        ui.markdown(f"#### Large-scale validation summary "
                    f"({summ['n_quadrats']} quadrat(s))").classes("mt-1")
        with ui.row().classes("w-full gap-2 flex-wrap"):
            for im in imgs:
                ui.image(im).classes("mt-1").style("max-width:460px;")
        ui.markdown(summ["qc_text"]).classes("text-sm text-grey-8 mt-2")

    def _render_report(metrics, dist, paired, pair_res, truth_df, detect_df,
                        *, job=None):
        """Append one job's full diagnostic to the page (the caller clears
        the containers at batch start). ``job`` feeds the heading only.
        Figures come from :func:`_build_validation_figures`, the builder
        the PDF report uses, so units agree everywhere."""
        import matplotlib.pyplot as plt
        field = state.val_field
        unit, factor = _field_display_unit(field)
        if job is not None:
            with plots_container:
                ui.separator()
                truth_name = Path(job["truth"]).name if job.get("truth") else "?"
                detect_name = Path(job["detect"]).name if job.get("detect") else "?"
                ui.markdown(f"### Comparison: `{truth_name}`  ⇄  `{detect_name}`") \
.classes("text-h6")
        # ---- Section 1: detection performance summary ----
        with summary_container:
            ui.separator()
            ui.markdown("#### Detection performance")
            with ui.row().classes("gap-4"):
                _stat_card("Truth clasts", f"{metrics['n_truth']}", "")
                _stat_card("Detected clasts", f"{metrics['n_detect']}", "")
                _stat_card("Matched pairs", f"{metrics['n_matched']}", "")
                _stat_card("Recall", f"{metrics['recall']:.3f}",
                           f"{metrics['n_matched']} / {metrics['n_truth']} truth clasts found")
                _stat_card("Precision", f"{metrics['precision']:.3f}",
                           f"{metrics['n_matched']} / {metrics['n_detect']} detections matched a truth clast")
                _stat_card("F1", f"{metrics['f1']:.3f}",
                           "harmonic mean of recall and precision")
            ui.markdown(
                f"_Tolerance used: {pair_res['used_tolerance']:.3f} m_"
            ).classes("text-sm text-grey-7")
            ui.label('This detector favours precision over recall, so low recall mostly means undetected borderline clasts; for grain-size mapping the distribution metrics below matter more than recall or F1.').classes("text-sm text-grey-7")

        # ---- Section 2: distribution comparison ----
        with dist_container:
            ui.separator()
            ui.markdown(f"#### Distribution comparison — `{field}` "
                        f"({unit or 'unit-less'})")
            phi_meaningful = bool(dist.get("phi_meaningful", True))
            # The RMSE is the precision the percentiles are rounded to.
            _u_disp = None
            try:
                _rmse = metrics.get("rmse")
                if _rmse is not None and np.isfinite(_rmse):
                    _u_disp = float(_rmse) * factor
            except Exception:
                _u_disp = None
            with ui.row().classes("gap-4"):
                for label, key in [("Truth D50",   "truth_d50"),
                                   ("Detect D50",  "detect_d50"),
                                   ("Truth D84",   "truth_d84"),
                                   ("Detect D84",  "detect_d84")]:
                    val = dist.get(key)
                    _stat_card(label,
                               _format_value_with_unit(val, field, _u_disp), "")
                if _u_disp is not None:
                    _stat_card("Shown to",
                               _format_value_with_unit(
                                   metrics.get("rmse"), field, _u_disp),
                               "RMSE of this comparison — the precision the "
                               "percentiles above are rounded to")
                # Folk-Ward sorting only makes sense for linear-size fields.
                if phi_meaningful:
                    ts = dist.get("truth_sorting_phi")
                    ds = dist.get("detect_sorting_phi")
                    if ts is not None and ts == ts:  # not NaN
                        _stat_card("Truth sorting (φ)", f"{ts:.3f}",
                                   "Folk-Ward sorting (size fields only)")
                    if ds is not None and ds == ds:
                        _stat_card("Detect sorting (φ)", f"{ds:.3f}", "")
                ks_summary = (f"D = {dist['ks_statistic']:.3f}, "
                              f"p = {dist['ks_p_value']:.3g}")
                ks_verdict = ("distributions DIFFER (p < 0.05)"
                              if dist['ks_p_value'] < 0.05
                              else "no significant difference")
                _stat_card("K-S test", ks_summary, ks_verdict)

            # Signed percentile errors: the headline for grain-size mapping.
            d50_err = _safe_relerr(dist["detect_d50"], dist["truth_d50"])
            d84_err = _safe_relerr(dist["detect_d84"], dist["truth_d84"])
            with ui.row().classes("gap-4 mt-2"):
                if d50_err is not None:
                    sign = "+" if d50_err >= 0 else ""
                    verdict = ("over-estimated"
                               if d50_err > 0.05 else
                               "under-estimated"
                               if d50_err < -0.05 else
                               "within ±5 %")
                    _stat_card("D50 relative error",
                               f"{sign}{d50_err*100:.1f} %",
                               verdict)
                if d84_err is not None:
                    sign = "+" if d84_err >= 0 else ""
                    verdict = ("over-estimated"
                               if d84_err > 0.05 else
                               "under-estimated"
                               if d84_err < -0.05 else
                               "within ±5 %")
                    _stat_card("D84 relative error",
                               f"{sign}{d84_err*100:.1f} %",
                               verdict)
                # Missed fines make the detection look better sorted than
                # the truth; a flag when |delta| > 0.2 phi.
                if (phi_meaningful
                        and dist.get("detect_sorting_phi") is not None
                        and dist.get("truth_sorting_phi")
                        and dist["truth_sorting_phi"] == dist["truth_sorting_phi"]):
                    dsort = (dist["detect_sorting_phi"]
                              - dist["truth_sorting_phi"])
                    sign = "+" if dsort >= 0 else ""
                    _stat_card("Sorting bias",
                               f"{sign}{dsort:.2f} φ",
                               "detect − truth (φ)")

        # ---- Section 2b: paired clast-level error ----------------- #
        if paired and paired.get("n", 0) >= 2:
            with dist_container:
                ui.separator()
                ui.markdown(
                    f"## Paired clast-level error "
                    f"(n = {paired.get('n', 0)} matched pairs)")
                with ui.row().classes("gap-4"):
                    r2 = paired.get("r_squared")
                    if r2 is not None:
                        _stat_card("R²", f"{r2:.3f}",
                                   "per-clast correlation, 1 = perfect")
                    rmse = paired.get("rmse")
                    if rmse is not None:
                        _stat_card(
                            "RMSE",
                            _format_value_with_unit(rmse, field),
                            "per-clast error magnitude")
                    mae = paired.get("mae")
                    if mae is not None:
                        _stat_card(
                            "MAE",
                            _format_value_with_unit(mae, field),
                            "mean absolute error")
                    bias = paired.get("bias")
                    if bias is not None:
                        sign = "+" if bias >= 0 else ""
                        _stat_card(
                            "Bias (paired)",
                            (f"{sign}"
                             f"{_format_value_with_unit(bias, field)}"),
                            "detect − truth, mean of differences")
                    slope = paired.get("slope")
                    if slope is not None:
                        _stat_card(
                            "Regression slope",
                            f"{slope:.3f}",
                            "y = slope·x + b; 1.0 = unbiased")
                    loa_hi = paired.get("bland_altman_loa_hi")
                    loa_lo = paired.get("bland_altman_loa_lo")
                    if loa_hi is not None and loa_lo is not None:
                        _stat_card(
                            "±1.96σ (LoA)",
                            f"{_format_value_with_unit(loa_lo, field)}"
                            f" … {_format_value_with_unit(loa_hi, field)}",
                            "Bland-Altman limits of agreement")

        # ---- Section 3: plots ----
        with plots_container:
            ui.separator()
            ui.markdown("#### Diagnostic plots")
            try:
                _dmin_active = (
                    state.val_dmin_k_pixels * state.val_dmin_uav_gsd
                    if state.val_dmin_auto else state.val_dmin_m)
                figs = _build_validation_figures(
                    metrics, dist, paired, pair_res,
                    truth_df, detect_df, field,
                    src_image=(state.val_src_image
                                if state.val_src_image else None),
                    truth_gsd=state.val_truth_gsd or 1.0,
                    uav_image=(state.val_uav_image
                                if state.val_uav_image else None),
                    truth_alpha=state.val_truth_alpha,
                    d_min_m=(_dmin_active if state.val_dmin_apply
                              else None),
                )
            except Exception as ex:
                import traceback
                ui.label(
                    f"Plot build failed: {type(ex).__name__}: {ex}"
                ).classes("text-red-700")
                figs = {}
                try:
                    log_widget.push(traceback.format_exc())
                except Exception:
                    pass

            # Each figure in its own try/except: one matplotlib hiccup
            # must not blow up the whole results page.
            def _safe_image(key, *, max_width_pct=None, max_width_px=None,
                            classes="flex-1"):
                f = figs.get(key)
                if f is None:
                    return
                try:
                    img = ui.image(_fig_to_image_url(f, dpi=120)) \
.classes(classes)
                    if max_width_pct is not None:
                        img.style(f"max-width:{max_width_pct}%;")
                    if max_width_px is not None:
                        img.style(f"max-width:{max_width_px}px;")
                    plt.close(f)
                except Exception as ex:
                    try:
                        ui.label(
                            f"[plot '{key}' failed: {type(ex).__name__}: {ex}]"
                        ).classes("text-red-700")
                        log_widget.push(
                            f"[render] plot '{key}' failed: {ex}")
                    except Exception:
                        pass

            with ui.row().classes("w-full gap-2"):
                for key in ("hist", "cdf", "qq"):
                    _safe_image(key, max_width_pct=33)

            if paired and paired.get("n", 0) >= 2:
                ui.markdown(
                    f"### Paired comparison (n = {paired['n']} matched clasts)"
                )
                with ui.row().classes("w-full gap-2"):
                    for key in ("paired_scatter", "bland_altman"):
                        _safe_image(key, max_width_pct=50)
            else:
                ui.markdown(
                    "_Paired comparison skipped: fewer than 2 matched pairs._"
                ).classes("text-grey-7 italic")

            if figs.get("ccdf") is not None or \
                    figs.get("detection_function") is not None:
                ui.markdown("### Detection-limit diagnostics")
                _k_used = state.val_dmin_k_pixels
                _dmin_mm = (_k_used * state.val_dmin_uav_gsd * 1000
                             if state.val_dmin_auto
                             else state.val_dmin_m * 1000)
                ui.markdown(
                    f"Truncation-aware view: **k = {_k_used} pixels** "
                    f"× UAV GSD = "
                    f"{state.val_dmin_uav_gsd*1000:.3f} mm/px "
                    f"→ D_min = {_dmin_mm:.2f} mm. **CCDF** "
                    f"(left) shows both populations end-to-end on "
                    f"log-log axes — the divergence to the LEFT of "
                    f"the vertical D_min line is the small-grain tail "
                    f"the UAV cannot resolve. **Detection function** "
                    f"(right) plots UAV recall against grain-size "
                    f"bins; the fitted logistic's 50% point (d50) is "
                    f"an empirical estimate of where the UAV actually "
                    f"starts losing grains, which you can compare "
                    f"against your k · GSD assumption."
                ).classes("text-sm text-grey-7")
                with ui.row().classes("w-full gap-2"):
                    _safe_image("ccdf", max_width_pct=50)
                    _safe_image("detection_function", max_width_pct=50)

            ui.markdown("### Spatial pattern of matches and misses")
            _safe_image("spatial", max_width_px=1000, classes="w-full")

            # The header renders even when the file is missing, so the
            # miss is visible rather than silent.
            if state.val_src_image:
                ui.markdown("### Image overlay")
                ui.label('Truth, detection and pairing geometry over the source image, scaled by the truth GSD — best when the truth was digitized on this image.').classes("text-sm text-grey-7")
                if not Path(state.val_src_image).exists():
                    ui.label(
                        f"Source image not found at: {state.val_src_image}"
                    ).classes("text-red-700")
                    try:
                        log_widget.push(
                            f"[overlay] source image not found: "
                            f"{state.val_src_image}")
                    except Exception:
                        pass
                _safe_image("overlay", max_width_px=1200, classes="w-full")

    def _seed_validate(force=False):
        """Seed truth CSV, detection CSV, source image and ortho from the
        project (newest first; nothing invented)."""
        from functions import project_defaults as _pdf
        proj = state.current_project
        if not proj:
            return

        def _put(attr, row, value):
            if value is None:
                return
            if getattr(state, attr, "") and not force:
                return
            setattr(state, attr, str(value))
            _seed_default(row._path_input, value, force=True)

        truths = _pdf.validation_csvs(proj)
        truth = truths[0] if truths else None
        _put("val_truth_csv", _val_truth_row, truth)
        if truths:
            _put("val_truth_dir", _val_dir_row, truths[0].parent)
        # The detection CSV and the source image of the truth's own
        # photograph when they exist; the newest ones otherwise.
        _put("val_detect_csv", _val_detect_row,
             _pdf.for_truth(truth, _pdf.detection_csvs(proj)) or _pdf.best_clast_csv(proj))
        geo, imgs = _pdf.georectified_images(proj), _pdf.validation_images(proj)
        _put("val_src_image", _val_src_row,
             _pdf.for_truth(truth, geo) or _pdf.for_truth(truth, imgs)
             or (geo or imgs or [None])[0])
        orthos = _pdf.orthos(proj)
        _put("val_uav_image", _val_uav_row, orthos[0] if orthos else None)
    _seed_validate()

    def _refresh_validate_on_open():
        if drop_foreign_paths("val_truth_csv", "val_detect_csv", "val_src_image",
                              "val_uav_image", "val_truth_dir"):
            _seed_validate(force=True)
        else:
            _seed_validate()
    register_tab_refresh("Validate", _refresh_validate_on_open)


def _stat_card(label, value, sub=""):
    """Small stat card for the validation summary."""
    with ui.card().classes("min-w-[10rem]"):
        ui.label(label).classes("text-xs text-grey-7 uppercase")
        ui.label(value).classes("text-h6")
        if sub:
            ui.label(sub).classes("text-xs text-grey-7")


# ----- Report tab -------------------------------------------------------- #
def build_report_tab():
    """Project-level PDF report: a self-contained deliverable covering
    methodology, detections, maps, validation, figures and appendices."""
    import json as _json

    def _on_proj_change():
        state.report_out_path = ""
        state.report_cover_image = ""
        _autofill_out_path()
        _load_report_metadata()
    render_project_strip(on_change=_on_proj_change)
    ui.markdown("### Report")
    ui.label('One PDF for the active project — summary, methods, detections, maps, validation, figures, appendices; untick a section to omit it.').classes("text-sm text-grey-7")

    # ---- Metadata (cover-page free text) ----
    with ui.row().classes("w-full gap-3"):
        ui.input(label="Author") \
.bind_value(state, "report_author") \
.classes("w-64") \
.tooltip("Optional. Appears on the cover page.")
        ui.input(label="Affiliation") \
.bind_value(state, "report_affiliation") \
.classes("w-64") \
.tooltip("Optional. Appears on the cover page.")
    ui.textarea(label="Project description") \
.bind_value(state, "report_description") \
.classes("w-full") \
.props("rows=6 autogrow") \
.tooltip("Short prose that goes on the cover page, between the "
                 "metadata and the cover image. A couple of sentences "
                 "describing the study site and survey aims is the "
                 "intended use.")

    # ---- Cover image override ----
    def _browse_cover():
        current = state.report_cover_image
        initialdir = (str(Path(current).parent) if current
                      else default_starting_dir("maps"))
        picked = native_file_picker(
            "Cover image (PNG/JPG)",
            filetypes=[("Images", "*.png *.jpg *.jpeg"),
                       ("All files", "*.*")],
            initialdir=initialdir)
        if picked:
            state.report_cover_image = picked
    with ui.row().classes("w-full items-end gap-2"):
        ui.input(label="Cover image (optional — auto if blank)") \
.bind_value(state, "report_cover_image") \
.classes("flex-grow") \
.tooltip("PNG or JPG used as the hero image on the cover. "
                     "Leave empty to auto-pick the first publication map "
                     "in results/maps/, falling back to results/figures/ "
                     "or images/.")
        ui.button("Browse…", icon="folder_open", on_click=_browse_cover).props("outline")

    # ---- Illustration images (custom title + description per image) ----
    with ui.expansion("Illustrations (optional)", icon="image") \
.classes("w-full"):
        ui.markdown(
            "Add your own images (field photos, context figures, instrument "
            "set-ups…) to a dedicated **Illustrations** section of the report. "
            "Each image gets a title and a caption/description.") \
.classes("text-sm text-grey-7")
        illus_box = ui.column().classes("w-full gap-2 mt-1")

        @ui.refreshable
        def _render_illustrations():
            illus_box.clear()
            with illus_box:
                items = state.report_illustrations
                if not items:
                    ui.label("No illustrations added yet.") \
.classes("text-grey-7 italic text-sm")
                for i, item in enumerate(items):
                    with ui.row().classes("w-full items-start gap-2 "
                                          "border rounded p-2"):
                        ui.label(f"{i + 1}.").classes("text-sm mt-3")
                        with ui.column().classes("flex-grow gap-1"):
                            ui.label(Path(item.get("path", "")).name
                                     or "(no file)") \
.classes("text-xs text-grey-7")

                            def _make_handlers(idx):
                                def _set_title(e):
                                    state.report_illustrations[idx]["title"] = \
                                        e.value or ""

                                def _set_desc(e):
                                    state.report_illustrations[idx][
                                        "description"] = e.value or ""
                                return _set_title, _set_desc
                            _set_title, _set_desc = _make_handlers(i)
                            ui.input(label="Title",
                                     value=item.get("title", "")) \
.classes("w-full").on_value_change(_set_title)
                            ui.textarea(label="Description / caption",
                                        value=item.get("description", "")) \
.classes("w-full").props("rows=2 autogrow") \
.on_value_change(_set_desc)

                        def _remove(_=None, idx=i):
                            del state.report_illustrations[idx]
                            _render_illustrations.refresh()
                        ui.button(icon="delete", on_click=_remove) \
.props("flat dense round color=negative") \
.tooltip("Remove this image")

        def _add_illustration():
            picked = native_file_picker(
                "Illustration image (PNG/JPG)",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp"),
                           ("All files", "*.*")],
                initialdir=default_starting_dir("images"))
            if not picked:
                return
            state.report_illustrations.append(
                {"path": picked, "title": Path(picked).stem, "description": ""})
            _render_illustrations.refresh()

        _render_illustrations()
        ui.button("Add image…", icon="add_photo_alternate",
                  on_click=_add_illustration).props("outline").classes("mt-1")

    # ---- Section toggles ----
    with ui.expansion("Sections to include", icon="checklist") \
.classes("w-full"):
        with ui.column().classes("gap-1 ml-2"):
            # Data and processing, detection results, interpretation and
            # the references always render; the rest is optional.
            for key, label in [
                ("cover", "Cover and summary"),
                ("toc", "Contents"),
                ("spatial_maps", "Spatial statistics (per-cell surfaces)"),
                ("zonal_statistics", "Zonal statistics (polygons, transects)"),
                ("validation", "Validation (when ground truth exists)"),
                ("appendix_logs", "Appendix A. Run register"),
                ("appendix_samples", "Appendix B. Data dictionary"),
                ("appendix_field_reference", "Appendix C. Glossary"),
            ]:
                ui.checkbox(label).bind_value(state.report_sections, key)

    # ---- Output path ----
    with ui.row().classes("w-full items-end gap-2"):
        ui.input(label="Output PDF path") \
.bind_value(state, "report_out_path") \
.classes("flex-grow") \
.tooltip("Where the PDF lands. Auto-derived from the active "
                     "project name when empty.")
        def _browse_out():
            current = state.report_out_path
            initialfile = (Path(current).name if current
                           else f"{state.current_project or 'project'}_report.pdf")
            initialdir = (str(Path(current).parent) if current
                          else default_starting_dir(""))
            picked = native_save_file_picker(
                "Save report as",
                filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
                initialfile=initialfile,
                defaultextension=".pdf",
                initialdir=initialdir)
            if picked:
                state.report_out_path = picked
        ui.button("Browse…", icon="folder_open", on_click=_browse_out).props("outline")

    def _autofill_out_path():
        if state.report_out_path:
            return
        proj = state.current_project
        if not proj:
            return
        from datetime import datetime as _dt
        stamp = _dt.now().strftime("%Y%m%d")
        repdir = project_path(proj, "reports")
        try:
            repdir.mkdir(parents=True, exist_ok=True)
        except Exception:
            repdir = project_path(proj)
        target = repdir / f"{proj}_report_{stamp}.pdf"
        state.report_out_path = str(target)

    # ---- Persisted metadata next to the PDF ----
    def _metadata_path():
        if not state.current_project:
            return None
        return project_path(state.current_project) / ".report_metadata.json"

    def _load_report_metadata():
        p = _metadata_path()
        if p is None or not p.exists():
            return
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        # Never over a field the user has typed into.
        for k, attr in [("author", "report_author"),
                        ("affiliation", "report_affiliation"),
                        ("description", "report_description"),
                        ("cover_image", "report_cover_image")]:
            if not getattr(state, attr) and data.get(k):
                setattr(state, attr, data[k])
        if not state.report_illustrations and data.get("illustrations"):
            state.report_illustrations = list(data["illustrations"])

    def _save_report_metadata():
        p = _metadata_path()
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_json.dumps({
                "author": state.report_author,
                "affiliation": state.report_affiliation,
                "description": state.report_description,
                "cover_image": state.report_cover_image,
                "illustrations": list(state.report_illustrations),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    ui.timer(2.0, _autofill_out_path)
    _load_report_metadata()

    # ---- Action buttons ----
    ui.separator()
    with ui.row().classes("items-center gap-2"):
        gen_btn = ui.button("Generate report", icon="picture_as_pdf") \
.props("color=primary") \
.tooltip("Walk the project, compute aggregate stats, build the "
                     "PDF. Takes a few seconds for a small project, longer "
                     "if rasters/figures are large.")

    log_widget = live_log("report", build_log_console(height="h-48"))

    preview_container = ui.column().classes("w-full mt-2")

    def _do_generate():
        if not state.current_project:
            ui.notify("Pick a project (Project, left panel) first.", type="warning")
            return
        _autofill_out_path()
        out_path = Path(state.report_out_path)
        if not out_path.parent.exists():
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                ui.notify(f"Can't write to {out_path.parent}: {e}. Change *Output PDF path* (above), then Generate report again.",
                          type="negative")
                return
        _save_report_metadata()
        gen_btn.props("loading")
        _disable(gen_btn, "Generating — the log below shows progress")
        log_widget.clear()
        preview_container.clear()

        opts = {
            "author": state.report_author,
            "affiliation": state.report_affiliation,
            "description": state.report_description,
            "cover_image": state.report_cover_image or None,
            "illustrations": list(state.report_illustrations),
            "include_cover":               state.report_sections.get("cover", True),
            "include_toc":                 state.report_sections.get("toc", True),
            "include_spatial_maps":        state.report_sections.get("spatial_maps", True),
            "include_zonal_statistics":    state.report_sections.get("zonal_statistics", True),
            "include_validation":          state.report_sections.get("validation", True),
            "include_appendix_logs":       state.report_sections.get("appendix_logs", True),
            "include_appendix_samples":    state.report_sections.get("appendix_samples", True),
            "include_appendix_field_reference":
                state.report_sections.get("appendix_field_reference", True),
        }

        def _worker():
            # A long build can outlast the browser tab; a UI call on the
            # deleted client raises RuntimeError, which must not end the
            # build. The PDF still lands on disk.
            def _safe_log(msg):
                try:
                    log_widget.push(msg)
                except RuntimeError:
                    print(f"[report-log] {msg}")
                except Exception:
                    pass

            def _safe_call(callable_, *args, **kwargs):
                """A UI call that returns None on a stale-client RuntimeError."""
                try:
                    return callable_(*args, **kwargs)
                except RuntimeError as ex:
                    if "client this element belongs to" in str(ex):
                        return None
                    raise

            # Refuse an unrecognised project layout before spending minutes
            # building an empty report from it.
            try:
                from functions.layout import (
                    resolve_project_subfolder_checked as _rsf_checked,
                    describe_unrecognised_layout as _describe_unrecognised,
                )
                _proj_root = project_path(state.current_project)
                if _rsf_checked(_proj_root, "images").unrecognised:
                    _msg = _describe_unrecognised(_proj_root)
                    _safe_log(f"[refused] {_msg}")
                    _safe_call(ui.notify, _msg, type="negative",
                               multi_line=True, close_button="Dismiss")
                    return
            except Exception as _layout_ex:
                # A failure of the check must not block the build.
                _safe_log(f"[warn] layout check skipped: {_layout_ex}")

            try:
                # The reportlab build runs in a worker subprocess; its log
                # lines stream back here.
                with capture_stdout_to_log(log_widget):
                    outcome = _worker_mod.run_job(
                        "report",
                        {"project_root": str(project_path(state.current_project)),
                         "out_path": str(out_path),
                         "options": opts},
                        log_cb=_safe_log)
                if not outcome.ok:
                    raise RuntimeError(_job_error_text(outcome))
                pdf_path = Path(outcome.outputs[0]) if outcome.outputs \
                    else Path(out_path)
                _safe_log(f"[done] Report written to {pdf_path}")
                # Inline preview. A background thread has no slot context,
                # so UI mutations go inside `with preview_container:`.
                try:
                    from nicegui import app as _ngapp
                    import hashlib as _h
                    mount = "report_" + _h.md5(
                        str(pdf_path.parent).encode()).hexdigest()[:8]
                    try:
                        _ngapp.add_static_files(f"/{mount}",
                                                 str(pdf_path.parent))
                    except Exception:
                        pass
                    # The mount URL is constant across rebuilds; the mtime
                    # query parameter busts the browser cache.
                    try:
                        mtime_tag = int(pdf_path.stat().st_mtime)
                        pdf_size = int(pdf_path.stat().st_size)
                    except OSError:
                        import time as _t
                        mtime_tag = int(_t.time())
                        pdf_size = 0
                    url = f"/{mount}/{pdf_path.name}?v={mtime_tag}"
                    # A large PDF in the iframe can crash the NiceGUI
                    # server; over the threshold only the link is shown.
                    INLINE_PREVIEW_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
                    pdf_size_mb = pdf_size / (1024 * 1024)
                    inline_ok = (0 < pdf_size
                                 <= INLINE_PREVIEW_MAX_BYTES)
                    preview_container.clear()
                    with preview_container:
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"Preview: {pdf_path.name}") \
.classes("text-sm text-grey-7")
                            ui.label(
                                f"{pdf_size_mb:.1f} MB · build {mtime_tag}"
                            ).classes("text-xs text-grey-5 font-mono") \
.tooltip(
                                "build tag = file mtime. If the preview "
                                "below does not match this, your browser "
                                "is showing a cached copy — Ctrl+Shift+R "
                                "to force a reload.")
                            ui.link("open in new tab", target=url,
                                     new_tab=True) \
.classes("text-sm text-blue-7")
                        if inline_ok:
                            # 95vw escapes the column's flex sizing; 1100 px
                            # shows one A4 portrait page.
                            ui.html(
                                f'<iframe src="{url}#zoom=page-width" '
                                f'style="width:95vw; max-width:1600px; '
                                f'height:1100px; '
                                f'border:1px solid #ccc; '
                                f'display:block;"></iframe>'
                            )
                        else:
                            ui.label(
                                f"Inline preview disabled — PDF is "
                                f"{pdf_size_mb:.1f} MB which is over "
                                f"the {INLINE_PREVIEW_MAX_BYTES // (1024*1024)} "
                                f"MB threshold for the embedded "
                                f"viewer. Large PDFs in an iframe have "
                                f"crashed the GUI server in earlier "
                                f"builds. Use the 'open in new tab' "
                                f"link above to view the file directly "
                                f"in your browser's native PDF reader."
                            ).classes(
                                "text-sm text-grey-7 italic"
                            ).style(
                                "background:#fff9e6; "
                                "border-left:3px solid #d97706; "
                                "padding:0.5em 0.75em;")
                except RuntimeError as cex:
                    if "client this element belongs to" in str(cex):
                        print(f"[report] preview skipped (client gone): "
                              f"{pdf_path}")
                    else:
                        _safe_log(f"[warn] preview unavailable: {cex}")
                except Exception as pex:
                    _safe_log(f"[warn] preview unavailable: {pex}")
            except Exception as ex:
                import traceback
                _safe_log(f"[error] {type(ex).__name__}: {ex}")
                _safe_log(traceback.format_exc())
            finally:
                try:
                    gen_btn.props(remove="loading")
                    _enable(gen_btn)
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    gen_btn.on_click(_do_generate)


# --- Page assembly ------------------------------------------------------- #
# A deliberately fatal endpoint, registered only under the test env flag,
# to prove a real fault leaves a trace and an announcement.
if os.environ.get("PEBBLEMAPPER_ENABLE_TEST_FAULTS") == "1":
    @ui.page("/debug/fault")
    def _debug_fault_page(kind: str = "segv"):
        ui.label(_crashsafe.induce_test_fault(kind))


@ui.page("/")
def index():
    icon_path = Path(__file__).parent / "icon.svg"
    icon_url = f"/static_icon/icon.svg"  # served via app.add_static_files below

    # ---- Header ----
    with ui.header(elevated=True).style("background-color: #5a6878;"):
        if icon_path.exists():
            ui.image(icon_url).style("width: 36px; height: 36px;")
        ui.label(_brand.APP_NAME).classes("text-h6 text-white")
        ui.space()
        ui.label(_brand.APP_TAGLINE) \
.classes("text-caption text-white")

    # ---- Sidebar (left drawer) ----
    with ui.left_drawer().classes("bg-grey-2"):
        render_project_picker()
        ui.separator()
        ui.label("Settings").classes("text-h6")
        render_detection_settings()
        ui.separator()
        ui.label("This application runs locally. No data leaves your machine.") \
.classes("text-caption text-grey-7")
        ui.label(f"Repo root:\n{REPO_ROOT}").classes("text-caption text-grey-6")
        ui.separator()
        ui.label("Feedback").classes("text-h6")
        ui.label("Something wrong? Build a diagnostic bundle and attach it to "
                 "an issue: versions, logs and file names, no data.") \
            .classes("text-caption text-grey-7")
        _fb_note = ui.textarea(label="What happened (optional)") \
            .classes("w-full").props("dense autogrow")

        def _build_feedback():
            from functions import feedback as _fb
            root = (project_path(state.current_project)
                    if state.current_project else None)
            try:
                out = _fb.build_bundle(root, note=_fb_note.value or "")
            except Exception as ex:
                ui.notify(f"Could not build the bundle: {ex}", type="negative")
                return
            ui.notify(f"Bundle written: {out}. Attach it to an issue at "
                      f"{_fb.ISSUES_URL}.", type="positive", timeout=15000,
                      multi_line=True, close_button=True)
        with ui.row().classes("items-center gap-2"):
            ui.button("Build diagnostic bundle", icon="bug_report",
                      on_click=_build_feedback).props("outline dense")
            from functions.feedback import ISSUES_URL as _ISSUES_URL
            ui.link("Issue tracker", _ISSUES_URL, new_tab=True).classes("text-caption")
        ui.separator()
        render_stop_button()

    # ---- Tabs ----
    # Tabs and panels are both bound to state.active_tab; the value is also
    # set at construction, or the first render starts blank.
    with ui.tabs().classes("w-full").bind_value(state, "active_tab") as tabs:
        ui.tab("Overview", icon="home")
        ui.tab("Express", icon="bolt")
        ui.tab("Orthorectify", icon="crop_rotate")
        ui.tab("Detect", icon="photo_camera")
        ui.tab("Merge", icon="merge")
        ui.tab("Rasterize", icon="map")
        ui.tab("Map", icon="picture_as_pdf")
        ui.tab("Zonal", icon="layers")
        ui.tab("Georeference", icon="my_location")
        ui.tab("Digitize", icon="edit")
        ui.tab("Validate", icon="fact_check")
        ui.tab("Report", icon="description")
    tabs.set_value(state.active_tab)
    # Opening a tab re-seeds it: see register_tab_refresh.
    tabs.on_value_change(lambda e: _fire_tab_hooks(getattr(e, "value", "")))

    # No slide between tabs: the sticky toolbars need the panel clip removed
    # (see _PM_LAYOUT_CSS) and an unclipped slide spills past the edge.
    with ui.tab_panels(tabs, value=state.active_tab, animated=False) \
            .classes("w-full") \
.bind_value(state, "active_tab"):
        with ui.tab_panel("Overview"):
            build_overview_tab()
        with ui.tab_panel("Express"):
            build_express_tab()
        with ui.tab_panel("Orthorectify"):
            build_orthorectify_tab()
        with ui.tab_panel("Detect"):
            build_detection_tab()
        with ui.tab_panel("Merge"):
            build_merge_tab()
        with ui.tab_panel("Rasterize"):
            build_rasterize_tab()
        with ui.tab_panel("Map"):
            build_map_tab()
        with ui.tab_panel("Zonal"):
            build_zonal_tab()
        with ui.tab_panel("Georeference"):
            build_georeference_tab()
        with ui.tab_panel("Digitize"):
            build_digitize_tab()
        with ui.tab_panel("Validate"):
            build_validate_tab()
        with ui.tab_panel("Report"):
            build_report_tab()

    # ---- Announce an unclean previous exit, once per boot ----
    # A closed console window, a killed process or a power cut leave an
    # unclean breadcrumb but no crash dump: nothing to say about those (the
    # console notes it). Only a trace with a dump in it earns the warning.
    _prev_crash = _crashsafe.pending_announcement()
    if _prev_crash:
        _crashsafe.mark_announced()
    if _prev_crash and _prev_crash.get("evidence") == "trace":
        ui.notify(
            f"The previous session (started {_prev_crash['started_at']}) "
            "crashed. The per-thread trace and any worker post-mortem are in: "
            f"{_prev_crash.get('trace_path')}",
            type="warning", multi_line=True, close_button="Dismiss", timeout=0)
    _rec_degraded = _crashsafe.degraded_reason()
    if _rec_degraded and not getattr(state, "_crashsafe_degraded_shown", False):
        state._crashsafe_degraded_shown = True
        ui.notify(_rec_degraded, type="info", multi_line=True,
                  close_button="OK", timeout=0)

    # ---- One restore offer for interrupted queues ----
    if not _qrestore["resolved"] and _qrestore["pending"]:
        _pend = _qrestore["pending"]
        with ui.dialog() as _restore_dlg, ui.card().classes("max-w-xl"):
            ui.label("Interrupted queues from the previous session") \
                .classes("text-h6")
            ui.label('Queued jobs from the last session: Restore brings them back (a running job returns as interrupted, finished rows are re-checked on disk); Discard archives them for good.').classes("text-sm text-grey-7")
            for _tab, _info in sorted(_pend.items()):
                if _info.get("version_error"):
                    ui.label(f"• {_tab}: cannot be restored — "
                             f"{_info['version_error']}") \
                        .classes("text-sm text-negative")
                else:
                    _proj = f" (project {_info['project']})" \
                        if _info.get("project") else ""
                    ui.label(f"• {_tab}: {_info['rows']} unfinished of "
                             f"{_info['total']} row(s), saved "
                             f"{_info.get('saved_at') or '?'}{_proj}") \
                        .classes("text-sm")

            def _restore_accept():
                restored = _queue_store.restore_into(state)
                _qrestore["resolved"] = True
                _restore_dlg.close()
                _ensure_queue_watcher()
                total = sum(restored.values())
                ui.notify(f"Restored {total} queued job(s) across "
                          f"{len(restored)} tab(s); interrupted and "
                          "missing-output rows were demoted as described.",
                          type="positive", multi_line=True)
                ui.navigate.reload()

            def _restore_decline():
                n = _queue_store.archive_declined()
                _qrestore["resolved"] = True
                _restore_dlg.close()
                _ensure_queue_watcher()
                ui.notify(f"Discarded {n} stored queue file(s) — archived "
                          "under queues/declined/, not offered again.",
                          type="info", multi_line=True)

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Discard", on_click=_restore_decline) \
                    .props("flat color=grey")
                ui.button("Restore", icon="restore",
                          on_click=_restore_accept).props("color=primary")
        _restore_dlg.props("persistent")
        _restore_dlg.open()

    # ---- Spacebar pan: while Space is held, the page follows the cursor ----
    ui.run_javascript("""
    (function() {
        if (window.__csm_spacepan_installed) return;
        window.__csm_spacepan_installed = true;

        var panActive = false;
        var lastX = 0, lastY = 0;

        function _isEditTarget(el) {
            if (!el) return false;
            var tag = (el.tagName || '').toLowerCase();
            return tag === 'input' || tag === 'textarea'
                || el.isContentEditable
                || el.getAttribute('contenteditable') === 'true';
        }

        document.addEventListener('keydown', function(e) {
            if (e.code !== 'Space' || e.repeat) return;
            if (_isEditTarget(document.activeElement)) return;
            panActive = true;
            e.preventDefault();
            document.body.style.cursor = 'grab';
        }, {capture: true});

        document.addEventListener('keyup', function(e) {
            if (e.code !== 'Space') return;
            panActive = false;
            document.body.style.cursor = '';
        }, {capture: true});

        document.addEventListener('mousemove', function(e) {
            if (panActive) {
                var dx = e.clientX - lastX;
                var dy = e.clientY - lastY;
                // The deepest scrollable ancestor under the cursor, else window.
                var target = e.target;
                var scrolled = false;
                while (target && target !== document.body) {
                    var cs = window.getComputedStyle(target);
                    var ovX = cs.overflowX, ovY = cs.overflowY;
                    if ((ovX === 'auto' || ovX === 'scroll')
                            && target.scrollWidth > target.clientWidth) {
                        target.scrollLeft -= dx;
                        scrolled = true;
                    }
                    if ((ovY === 'auto' || ovY === 'scroll')
                            && target.scrollHeight > target.clientHeight) {
                        target.scrollTop -= dy;
                        scrolled = true;
                    }
                    if (scrolled) break;
                    target = target.parentElement;
                }
                if (!scrolled) window.scrollBy(-dx, -dy);
            }
            lastX = e.clientX;
            lastY = e.clientY;
        }, {passive: true});
    })();
    """)


# --- Entry point --------------------------------------------------------- #
if __name__ in ("__main__", "__mp_main__"):
    icon_path = Path(__file__).parent / "icon.svg"
    favicon_arg = str(icon_path) if icon_path.exists() else "🪨"

    # ---- Crash observability ----
    # Real main module only: multiprocessing children import this file as
    # "__mp_main__", and a child writing the session breadcrumb would make
    # every next boot announce a phantom crash.
    if __name__ == "__main__":
        # A reentrancy hole in NiceGUI's binding propagation lets a NaN in
        # a two-way-bound value recurse to native stack death. Close it
        # before anything can bind.
        _crashsafe.harden_binding_propagation()
        # NiceGUI warns when one propagation takes more than 10 ms; with the
        # ~2,000 links of this page a step takes 20-70 ms, and the warning
        # flooded the terminal and every log console during a run. During
        # a detection the worker thread holds the interpreter for whole
        # tiles and one propagation is measured at 1-2 s (the loop simply
        # waits for it): the threshold sits above that so the consoles
        # show the run, not the warning.
        try:
            from nicegui import binding as _binding
            _binding.MAX_PROPAGATION_TIME = 3.0
        except Exception:
            pass
        # Windows: no traceback for a peer that reset its connection. A
        # zero-argument callable: NiceGUI hands a one-parameter start-up
        # handler its Client.
        app.on_startup(lambda: _crashsafe.quiet_windows_connection_resets())
        _crashsafe.boot(
            project_path(state.current_project, "logs")
            if state.current_project else None)
        from nicegui import app as _ng_app
        _ng_app.on_shutdown(_crashsafe.mark_clean_exit)
        # Stage any interrupted queues before the persistence watcher starts.
        _qrestore["pending"] = _queue_store.pending_restore()
        _qrestore["resolved"] = not _qrestore["pending"]
        print(f"[queue] staged restore offer: "
              f"{sorted(_qrestore['pending']) or 'none'}", flush=True)
        # The watcher's no-delete guard leaves a still-offered queue file
        # untouched while that tab's in-memory queue is empty.
        _ensure_queue_watcher()

    # Static mount for the header logo (/static_icon/icon.svg).
    if icon_path.exists():
        from nicegui import app
        app.add_static_files("/static_icon", str(icon_path.parent))

    # Say so on the console when a client is dropped. NiceGUI builds its
    # socket.io server with logger=False (ERROR), while engine.io reports
    # the drop at INFO; app.on_disconnect never fires for a drop that
    # reconnects. Lifting the whole logger would print every packet, so a
    # filter keeps only the drop lines.
    import logging as _logging

    class _DropsOnly(_logging.Filter):
        # engine.io's own message strings. check_ping_timeout() runs only
        # from send(), so an idle server never notices a silent client.
        WANTED = ("is gone", "closing socket", "socket is closed",
                  "ping timeout", "disconnect", "Invalid session",
                  "rejected connection", "Cannot send to sid")

        def filter(self, record):
            msg = str(record.getMessage())
            return any(w in msg for w in self.WANTED)

    _eio = _logging.getLogger("engineio.server")
    _eio.setLevel(_logging.INFO)
    _eio.propagate = False              # keep it off the root handler
    # On the logger, not the handler: socket.io attaches its own
    # StreamHandler, which a handler-level filter would not cover.
    _eio.addFilter(_DropsOnly())
    if not _eio.handlers:
        _h = _logging.StreamHandler()
        _h.setFormatter(_logging.Formatter("[socket] %(message)s"))
        _eio.addHandler(_h)

    ui.run(
        title=_brand.APP_NAME,
        favicon=favicon_arg,
        host="127.0.0.1",
        port=_listen_port(),
        # Open the default browser, unless PEBBLEMAPPER_NO_BROWSER is set
        # (a remote or scripted launch; the page is then opened by hand).
        show=os.environ.get("PEBBLEMAPPER_NO_BROWSER", "").strip().lower()
        not in ("1", "true", "yes", "on"),
        reload=False,      # disable hot reload (heavy imports)
        dark=False,
        # NiceGUI derives the socket heartbeat from this:
        #   ping_interval = max(reconnect_timeout * 0.8, 4)
        #   ping_timeout  = max(reconnect_timeout * 0.4, 2)
        # so values <= 5 are a no-op. 20 gives a 24 s stall tolerance (a
        # sleeping laptop, a throttled tab) at the cost of noticing a dead
        # connection in ~24 s instead of 6.
        reconnect_timeout=20.0,
        # binding_refresh_interval stays at its 0.1 s default: a cycle costs
        # a few ms, and a drop needs a contiguous stall of seconds.
    )
