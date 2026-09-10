"""Pluggable detection backends: the shared contract.

The universal contract is the per-clast measurement table
(``CANONICAL_COLUMNS``) written by detection and consumed unchanged by every
downstream stage (merge, rasterize, zonal, validation, map, report). A
backend's job is to produce that table in world coordinates (metres); models
that emit instance geometry are measured through the single shared measurement
step (``detectors.measure``) so their numbers are defined identically.
"""
from __future__ import annotations

import abc
import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# The detector module owns the per-clast schema; import it so the two cannot drift.
from functions.clasts_detection import _CLAST_COLUMNS as CANONICAL_COLUMNS  # noqa: F401

try:
    from functions import __version__ as TOOL_VERSION
except Exception:  # pragma: no cover - version is best-effort metadata
    TOOL_VERSION = "dev"


@dataclasses.dataclass
class BackendInfo:
    """Static description of one detection backend (registry + GUI)."""
    name: str                       # registry key, e.g. "maskrcnn"
    display_name: str               # GUI label
    framework: str                  # "tensorflow" | "pytorch" | "classical" | ...
    license: str                    # SPDX-ish summary
    output_type: str = "instance"   # "instance" (per-clast) | "distribution"
    env: Optional[str] = None       # conda env for subprocess backends; None = core env
    in_process: bool = True         # True = runs in the core env; False = subprocess in `env`
    weights: Optional[str] = None   # weights id / DOI / path note
    install_hint: str = ""          # how a user installs it
    description: str = ""
    # A developer/proof backend (fabricates clasts): registered and available
    # to tests and scripts, never offered as a detection model unless
    # PEBBLEMAPPER_SHOW_DEV_BACKENDS=1 (detectors.registry.selector_options).
    dev_only: bool = False
    # The model's own version (weights release, package version), when the
    # backend knows it; recorded in Digitize's provenance sidecar.
    version: Optional[str] = None


@dataclasses.dataclass
class DetectionManifest:
    """Provenance sidecar written next to each detection CSV (best-effort)."""
    model: str
    model_version: str = TOOL_VERSION
    weights: Optional[str] = None
    params: dict = dataclasses.field(default_factory=dict)
    crs: Optional[str] = None
    gsd_m: Optional[float] = None
    n_tiles: Optional[int] = None
    tool_version: str = TOOL_VERSION
    license: str = ""
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def write(self, csv_path) -> Optional[Path]:
        """Write ``<csv_path>.manifest.json``. Never raises: provenance must not
        be able to fail a detection run."""
        try:
            p = Path(str(csv_path) + ".manifest.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(dataclasses.asdict(self), indent=2),
                         encoding="utf-8")
            return p
        except Exception:
            return None

    def write_for_completed_csv(self, csv_path,
                                run_started: float) -> Optional[Path]:
        """Write only beside a CSV this run actually produced.

        Returns None when the CSV is absent or predates ``run_started``, so a
        failed re-run never refreshes the manifest of an older result. The 2 s
        slack absorbs filesystem timestamp granularity.
        """
        try:
            p = Path(csv_path)
            if not p.exists() or p.stat().st_mtime < float(run_started) - 2.0:
                return None
        except OSError:
            return None
        return self.write(csv_path)

    @staticmethod
    def sidecar_path(csv_path) -> Path:
        """``<csv_path>.manifest.json`` — the one place the sidecar name lives."""
        return Path(str(csv_path) + ".manifest.json")

    @staticmethod
    def discard_stale(csv_path) -> bool:
        """Remove a manifest that predates the CSV beside it.

        A manifest describes the CSV it sits next to. When a backend rewrites
        the CSV without refreshing the sidecar (a different model, a driver
        that forgot), the old manifest would credit the wrong model, which is
        worse than no provenance at all. Returns True when a stale sidecar was
        removed. Never raises.
        """
        try:
            csv = Path(csv_path)
            mf = DetectionManifest.sidecar_path(csv)
            if not csv.exists() or not mf.exists():
                return False
            if mf.stat().st_mtime < csv.stat().st_mtime - 2.0:
                mf.unlink()
                return True
        except OSError:
            pass
        return False


def output_csv_name(mode: str, stem: str, window=None) -> str:
    """The canonical CSV filename for one job, whatever the backend.

    Ortho mode: ``<stem>_ws<window>m.csv`` (``window`` = ``metric_cropsize``).
    Quadrat mode: ``<stem>_individual_clasts.csv`` (the window is ignored).
    Every backend must name its CSVs this way, or the Detect tab, Validate and
    the report will not find them.
    """
    from functions import modes, naming
    mode = modes.normalise_mode(mode)
    if mode == modes.QUADRAT:
        return naming.quadrat_csv_name(stem)
    if window is None:
        raise ValueError("ortho mode needs the window size (metric_cropsize)")
    return naming.detection_csv_name(stem, window)


def output_csv_path(mode: str, job: dict, kwargs: dict) -> Optional[Path]:
    """Where ``detect_jobs`` writes one job's CSV, or None without ``output_dir``."""
    out_dir = kwargs.get("output_dir")
    if not out_dir:
        return None
    stem = job.get("out_stem") or Path(job["path"]).stem
    return Path(out_dir) / output_csv_name(mode, stem,
                                          kwargs.get("metric_cropsize"))


TERMINAL_EVENTS = ("done", "stopped", "error")


def readable_jobs_for(backend: "DetectorBackend", jobs: list, work_dir) -> list:
    """Jobs an out-of-process backend can open.

    A backend in its own environment reads the image with its own libraries,
    which usually cannot decode an iPhone HEIC/HEIF. For such a backend each
    HEIF job is handed a lossless PNG copy written in ``work_dir``, in the
    same pixel frame the app uses (``functions.images.read_rgb``), with
    ``out_stem`` pinned to the original name so the CSV, manifest and
    outlines keep it and ``source_path`` pointing at the original. In-process
    backends read HEIF themselves and get the jobs unchanged.
    """
    info = getattr(backend, "info", None)
    if info is None or getattr(info, "in_process", True):
        return jobs
    from functions import images as _images
    out = []
    for job in jobs:
        path = job.get("path")
        if not path or not _images.is_heif(path):
            out.append(job)
            continue
        src = Path(str(path))
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        png = Path(work_dir) / (src.stem + ".png")
        from PIL import Image
        Image.fromarray(_images.read_rgb(src)).save(png, compress_level=1)
        new = dict(job)
        new["path"] = str(png)
        new.setdefault("out_stem", job.get("out_stem") or src.stem)
        new.setdefault("source_path", str(src))
        out.append(new)
    return out


def frame_rois_for(mode: str, jobs: list, work_dir, exclude: bool = True,
                   log_fn=None, fallback_m=None) -> dict:
    """Give each quadrat job without an ROI the inner rectangle of its
    quadrat frame, when its rectification record says where the frame is.

    Orthorectify keeps the frame bars in the rectified photograph and writes
    their thickness in the sidecar; without this every model measures pieces
    of the frame as clasts. The ROI is written into ``work_dir`` in the frame
    quadrat ROIs already use (image pixels), so it works for every backend
    the same way. A job with its own ``roi_path`` is left alone: the user's
    ROI wins. Returns ``{job index: FrameInset}`` for the jobs it changed.
    """
    from functions import modes as _modes
    from functions import quadrat_frame as _qf
    out = {}
    if not exclude or _modes.normalise_mode(mode) != _modes.QUADRAT:
        return out
    for ji, job in enumerate(jobs):
        if job.get("roi_path"):
            continue
        source = job.get("source_path") or job.get("path")
        fi = _qf.frame_inset(source, fallback_m)
        if fi is None:
            continue
        roi = _qf.write_inner_roi(source, work_dir, fallback_m)
        if roi is None:
            continue
        job["roi_path"] = str(roi)
        out[ji] = fi
        if log_fn:
            log_fn(f"[frame] {Path(str(source)).name}: {_qf.describe(fi)}")
    return out


def _note_frame_in_manifest(csv_path, fi) -> None:
    """Record the excluded frame in the manifest the backend wrote (best
    effort: provenance must never fail a run)."""
    try:
        p = Path(str(csv_path) + ".manifest.json")
        if not p.is_file():
            return
        doc = json.loads(p.read_text(encoding="utf-8"))
        params = doc.setdefault("params", {}) if isinstance(doc, dict) else None
        if params is None:
            return
        w_m, h_m = fi.inner_size_m
        params["frame_excluded"] = {
            "thickness_m": fi.thickness_m, "inset_px": fi.inset_px,
            "measured_size_m": [round(w_m, 4), round(h_m, 4)]}
        p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except Exception:
        pass


def run_detect_jobs(backend: "DetectorBackend", mode: str, jobs: list,
                    **kwargs) -> List:
    """Call ``backend.detect_jobs`` and guarantee one terminal progress event
    per job.

    The app's queue rows are driven by ``progress_callback(job_index, event,
    payload)``; a backend that never calls it would leave its rows "running"
    forever. This wrapper forwards the callback, records which jobs received a
    terminal event ("done", "stopped" or "error"), and after the backend
    returns emits "done" (with the returned row count) for the ones that did
    not; when the backend raises it emits "error" for them and re-raises. A
    stale manifest left beside a CSV this run rewrote is discarded.

    Outlines: when a job's returned frame carries ``attrs["contours"]``
    (``detectors.measure.measure_instances`` attaches them) and the CSV this
    run wrote has no up-to-date ``.contours.json``, the wrapper writes it; a
    sidecar older than the rewritten CSV is otherwise removed, so no figure
    draws another run's outlines. Never fails the run.
    """
    import time as _time
    run_started = _time.time()
    from functions import modes
    mode = modes.normalise_mode(mode)
    callback = kwargs.get("progress_callback")
    settled = set()

    def _emit(ji, event, payload):
        if event in TERMINAL_EVENTS:
            settled.add(ji)
        if callback is None:
            return
        try:
            callback(ji, event, payload)
        except Exception:
            pass

    fwd = dict(kwargs)
    if callback is not None:
        fwd["progress_callback"] = _emit
    else:
        fwd.pop("progress_callback", None)
    exclude_frame = bool(fwd.pop("exclude_frame", True))
    frame_fallback_m = fwd.pop("frame_fallback_m", None)

    import shutil as _shutil
    import tempfile as _tempfile
    work_dir = _tempfile.mkdtemp(prefix="pm_readable_")
    framed = {}
    try:
        try:
            call_jobs = readable_jobs_for(backend, jobs, work_dir)
            # The Detect tab hands one job's ROI as the call-wide roi_path
            # kwarg; that ROI is the user's and wins over the frame's.
            framed = frame_rois_for(
                mode, call_jobs, work_dir,
                exclude_frame and not kwargs.get("roi_path"),
                kwargs.get("log_fn"), frame_fallback_m)
            results = backend.detect_jobs(mode, call_jobs, **fwd)
        except Exception as ex:
            import traceback
            for ji in range(len(jobs)):
                if ji not in settled:
                    _emit(ji, "error", {"message": str(ex),
                                        "traceback": traceback.format_exc()})
            raise
    finally:
        _shutil.rmtree(work_dir, ignore_errors=True)
    if results is None:
        results = []
    results = list(results)
    stop_check = kwargs.get("stop_check")
    stopped = False
    try:
        stopped = bool(stop_check is not None and stop_check())
    except Exception:
        stopped = False
    # Save CSV off: nothing was written this run, so an older CSV's manifest
    # and outlines beside it are left alone.
    wrote = kwargs.get("saveresults", True) is not False
    for ji, job in enumerate(jobs):
        try:
            csv_path = output_csv_path(mode, job, kwargs) if wrote else None
        except Exception:
            csv_path = None
        if csv_path is not None:
            DetectionManifest.discard_stale(csv_path)
            _settle_contours(csv_path, results[ji] if ji < len(results)
                             else None, run_started)
            if ji in framed:
                _note_frame_in_manifest(csv_path, framed[ji])
        if ji in settled:
            continue
        if stopped:
            _emit(ji, "stopped", {})
            continue
        frame = results[ji] if ji < len(results) else None
        try:
            n = int(len(frame)) if frame is not None else 0
        except TypeError:
            n = 0
        _emit(ji, "done", {"n_clasts": n})
    return results


def _settle_contours(csv_path, frame, run_started: float) -> None:
    """Persist the outlines a backend attached to its frame beside the CSV it
    wrote this run, unless a fresh sidecar is already there; drop a stale one.
    Never raises."""
    try:
        from functions import clast_geometry as CG
        csv = Path(csv_path)
        if not csv.exists() or csv.stat().st_mtime < float(run_started) - 2.0:
            return
        side = CG.contours_path_for(csv)
        fresh = (side.exists()
                 and side.stat().st_mtime >= csv.stat().st_mtime - 2.0)
        if not fresh:
            cs = CG.contours_of(frame) if frame is not None else None
            if cs is not None:
                CG.write_contours(csv, cs)
            else:
                CG.discard_stale_contours(csv)
    except Exception:
        pass


class DetectorBackend(abc.ABC):
    """One selectable detection model.

    Implementations either run in-process (in the core ``maskrcnn`` env) or
    shell out to their own conda env as a subprocess. Either way the result is
    the canonical per-clast table.
    """
    info: BackendInfo

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this backend can actually run here (weights present /
        environment installed). Unavailable backends are hidden in the GUI."""

    def clear_cache(self) -> bool:
        """Drop whatever weights this backend keeps loaded between runs, so
        the next run loads them fresh from disk (the drawer's *Reload
        model*). Returns True when a cache exists and was dropped, False
        when there is nothing to reload.

        The default does nothing and returns False: a subprocess backend
        starts its own process, and so loads its weights, at every run.
        An in-process backend that caches a model overrides this."""
        return False

    @abc.abstractmethod
    def detect_jobs(self, mode: str, jobs: list, **kwargs) -> List:
        """Run detection for a list of ``{"path", "kstart"}`` jobs.

        ``mode`` is ``functions.modes.ORTHO`` or ``functions.modes.QUADRAT``;
        implementations normalise it with ``functions.modes.normalise_mode``
        so the older spellings (any case) keep working.

        Writes the canonical per-clast CSV(s) (+ a provenance manifest) into
        ``kwargs["output_dir"]``, named with :func:`output_csv_name`, and
        returns one ``pandas.DataFrame`` per job. ``kwargs`` carries the
        detection parameters (``resolution``, ``metric_cropsize``,
        ``overlap``, ``min_confidence``, ``devicemode``, ``output_dir``,
        ``stop_check``, ``progress_callback``, ...); they are Mask R-CNN's and
        a backend ignores the ones it has no use for. Report each job through
        ``progress_callback(job_index, event, payload)`` ("start", "done",
        "stopped", "error"); the app calls this through :func:`run_detect_jobs`,
        which settles unreported jobs from the return value.
        """
