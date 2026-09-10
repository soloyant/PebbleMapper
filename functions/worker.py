"""Worker: isolated execution of the heavy geoprocessing ops.

- **Worker half**: ``python -m functions.worker <envelope.json>`` runs ONE job
  in a fresh interpreter (same env, same repo, no TensorFlow, no ``gui.app``),
  streams progress to the envelope's ``log_path``, and writes a result JSON to
  ``result_path``. Exceptions become ``ok: false`` results; only a process
  death leaves no result file.
- **Parent half**: :func:`run_job` spawns the worker, tails its log into a
  callback, enforces a per-op timeout ceiling, decodes the exit, and, when the
  worker dies without a result, records a post-mortem through
  :mod:`functions.crashsafe`. Whatever kills the worker is confined to the job.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import subprocess as _sp
import sys as _sys
import tempfile as _tempfile
import time as _time
import traceback as _tb
import uuid as _uuid
from pathlib import Path
from typing import Callable, Optional

PROTOCOL_VERSION = 1

#: Per-op ceilings in seconds; a hung worker must eventually be called dead.
OP_TIMEOUTS = {
    "zonal_polygons": 900,
    "zonal_transects": 900,
    "zonal_map": 900,
    "canvas_source": 900,
    "publication_map": 1800,
    "rasterize": 3600,
    "merge": 1800,
    "report": 5400,
}
DEFAULT_TIMEOUT = 1800
_KEEP_JOB_DIRS = 20


# --- Worker half ---

def _w_zonal_polygons(args: dict, log) -> tuple[dict, list]:
    from functions.zonal_stats import zonal_polygon_stats_from_csv
    rows = zonal_polygon_stats_from_csv(
        args["detection_csv"], args["vector_path"], args["out_csv"],
        field_name=args.get("field_name") or "Clast_length",
        id_field=args.get("id_field") or "",
        percentiles=tuple(args.get("percentiles") or (5, 16, 25, 50, 75, 84, 95)),
        dem_path=args.get("dem_path") or None,
    )
    log(f"wrote {len(rows)} polygon row(s) -> {args['out_csv']}")
    meta = [{"count": int(getattr(r, "count", 0) or 0),
             "coverage": str(getattr(r, "coverage", "inside"))}
            for r in rows]
    return {"n_rows": len(rows), "rows": meta}, [args["out_csv"]]


def _w_zonal_transects(args: dict, log) -> tuple[dict, list]:
    from functions.zonal_stats import zonal_transect_profile, plot_transect_profile
    results = zonal_transect_profile(
        args["raster"], args["vector_path"], args["out_csv"],
        step_m=float(args.get("step_m", 0.5)),
        band=int(args.get("band", 1)),
        id_field=args.get("id_field") or "",
        dem_path=args.get("dem_path") or None,
        interpolation=args.get("interpolation") or "bilinear",
        field_name=args.get("field_name") or "",
    )
    n_samples = int(sum(t.distance_m.size for t in results))
    log(f"wrote {n_samples} sample(s) across {len(results)} transect(s) "
        f"-> {args['out_csv']}")
    pngs, plot_errors = [], []
    if args.get("make_plots", True):
        out_dir = Path(args["out_csv"]).parent
        for t in results:
            png = out_dir / f"{Path(args['out_csv']).stem}__{t.transect_id}.png"
            try:
                plot_transect_profile(t, png,
                                      field_name=args.get("field_name") or "")
                log(f"plot -> {png}")
                pngs.append(str(png))
            except Exception as ex:  # per-plot best effort
                msg = f"plot skipped for transect {t.transect_id}: {ex}"
                log(msg)
                plot_errors.append(msg)
    return ({"n_samples": n_samples, "n_transects": len(results),
             "pngs": pngs, "plot_errors": plot_errors},
            [args["out_csv"], *pngs])


def _w_zonal_map(args: dict, log) -> tuple[dict, list]:
    from functions.zonal_stats import plot_zonal_map
    plot_zonal_map(
        args["raster"], args["vector_path"], args["out_png"],
        field_name=args.get("field_name") or "Clast_length",
        polygon_id_field=args.get("id_field") or "",
        transect_id_field=args.get("id_field") or "",
        title=args.get("title") or None,
        log_fn=log,
    )
    return {}, [args["out_png"]]


def _w_canvas_source(args: dict, log) -> tuple[dict, list]:
    from functions.zonal_canvas import render_canvas_image
    info = render_canvas_image(**args)
    log(f"canvas image -> {info['png']}")
    return info, [info["png"]]


def _w_publication_map(args: dict, log) -> tuple[dict, list]:
    """Render one publication map from the Map tab's job dict and save the
    requested artefacts (the figure itself cannot cross the process
    boundary): ``out_path`` at ``dpi``, ``save_png`` at ``dpi``,
    ``preview_png`` at 120 dpi."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from functions.map_export import make_publication_map, DEFAULT_BASEMAPS
    job = args["job"]
    fig = make_publication_map(
        ortho_tif=(job.get("ortho") or None),
        clast_csv=job.get("csv") if job.get("layer") == "vector" else None,
        raster_tif=(job.get("raster")
                    if job.get("layer") == "raster" else None),
        layer=job.get("layer"),
        field=job.get("field"),
        cmap=job.get("cmap"),
        basemap=DEFAULT_BASEMAPS.get(job.get("basemap")),
        show_ortho=job.get("show_ortho"),
        ortho_alpha=job.get("ortho_alpha"),
        raster_alpha=job.get("raster_alpha"),
        show_grid=job.get("show_grid"),
        show_legend=job.get("show_legend"),
        show_crs_info=job.get("show_crs"),
        show_north_arrow=job.get("show_north"),
        show_scale_bar=job.get("show_scale"),
        show_zebra_border=job.get("show_zebra"),
        show_colorbar_extends=job.get("show_colorbar_extends"),
        title=job.get("title") or None,
        point_size=float(job.get("point_size") or 8),
        color_scale=job.get("color_scale"),
        vmin=None if job.get("vmin_auto") else float(job.get("vmin")),
        vmax=None if job.get("vmax_auto") else float(job.get("vmax")),
        unit_system=job.get("unit_system"),
        size_unit=job.get("size_unit"),
        log_fn=log,
    )
    dpi = int(args.get("dpi") or 300)
    outputs = []
    try:
        if args.get("out_path"):
            Path(args["out_path"]).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args["out_path"], dpi=dpi, bbox_inches="tight")
            log(f"wrote {args['out_path']} at {dpi} dpi")
            outputs.append(args["out_path"])
        if args.get("save_png"):
            fig.savefig(args["save_png"], format="png", dpi=dpi,
                        bbox_inches="tight")
            outputs.append(args["save_png"])
        if args.get("preview_png"):
            fig.savefig(args["preview_png"], format="png", dpi=120,
                        bbox_inches="tight")
            outputs.append(args["preview_png"])
    finally:
        plt.close(fig)
    return {"dpi": dpi}, outputs


def _w_rasterize(args: dict, log) -> tuple[dict, list]:
    from functions import clasts_rasterize as _cr
    thumb = args.pop("thumbnail_png", None)
    p = _cr.clasts_rasterize(**args)
    out = args.get("RasterFileWritingPath")
    log(f"raster -> {out}")
    outputs = [out] if out else []
    summary: dict = {"shape": list(getattr(p, "shape", []) or [])}
    if thumb is not None and p is not None:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots(figsize=(6, 4))
            disp = np.flipud(p) if p.ndim == 2 else np.flipud(p[..., 0])
            im = ax.imshow(disp, interpolation="none", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(thumb, dpi=100, bbox_inches="tight")
            plt.close(fig)
            summary["thumbnail"] = str(thumb)
            outputs.append(str(thumb))
        except Exception as ex:
            log(f"thumbnail skipped: {ex}")
            summary["thumbnail_error"] = str(ex)
    return summary, outputs


def _w_merge(args: dict, log) -> tuple[dict, list]:
    from functions import clasts_merge
    ordered = args["ordered"]           # [{path, window_size}] desc by window
    out_path = args["out_path"]
    log(f"Merging {len(ordered)} CSVs by descending window size:")
    for i, e in enumerate(ordered):
        log(f"  #{i + 1}  window={e['window_size']}m  {Path(e['path']).name}")
    merged = None
    with _tempfile.TemporaryDirectory() as tmpdir:
        accumulator = ordered[0]["path"]
        for i in range(1, len(ordered)):
            is_last = (i == len(ordered) - 1)
            step_out = out_path if is_last else \
                _os.path.join(tmpdir, f"acc_step{i}.csv")
            log(f"--- Step {i}/{len(ordered) - 1}: folding in "
                f"window={ordered[i]['window_size']}m ---")
            merged = clasts_merge.merge_csvs(
                input_filepath_small=ordered[i]["path"],
                input_filepath_large=accumulator,
                output_filepath=step_out,
                method=args.get("method", "iou"),
                overlap=float(args.get("overlap", 0.30)),
                n_points=int(args.get("n_points", 24)),
            )
            accumulator = step_out
    n = int(len(merged)) if merged is not None else 0
    log(f"Wrote {out_path} ({n} rows).")
    return {"n_rows": n}, [out_path]


def _w_report(args: dict, log) -> tuple[dict, list]:
    from functions.report import build_pdf
    pdf = build_pdf(args["project_root"], args["out_path"],
                    options=args.get("options"), log_fn=log)
    out = str(pdf) if pdf else str(args["out_path"])
    size = Path(out).stat().st_size if Path(out).exists() else 0
    return {"pdf": out, "bytes": size}, [out]


_HANDLERS = {
    "zonal_polygons": _w_zonal_polygons,
    "zonal_transects": _w_zonal_transects,
    "zonal_map": _w_zonal_map,
    "canvas_source": _w_canvas_source,
    "publication_map": _w_publication_map,
    "rasterize": _w_rasterize,
    "merge": _w_merge,
    "report": _w_report,
}


def _worker_main(envelope_path: str) -> int:
    """Run one job. Always tries to leave a result file; returns exit code 0
    for both success and loud failure."""
    import faulthandler
    try:
        env = _json.loads(Path(envelope_path).read_text(encoding="utf-8"))
    except Exception as exc:
        _sys.stderr.write(f"worker: unreadable envelope {envelope_path}: {exc}\n")
        return 2

    log_path = env.get("log_path")
    log_f = open(log_path, "a", encoding="utf-8", errors="replace") \
        if log_path else None
    # A native fault in library code should at least say where.
    if log_f is not None:
        faulthandler.enable(file=log_f, all_threads=True)

    def log(msg: str) -> None:
        if log_f is not None:
            log_f.write(str(msg).rstrip("\n") + "\n")
            log_f.flush()

    def write_result(payload: dict) -> None:
        payload["version"] = PROTOCOL_VERSION
        rp = env.get("result_path")
        if not rp:
            return
        tmp = str(rp) + ".tmp"
        Path(tmp).write_text(_json.dumps(payload, indent=1), encoding="utf-8")
        _os.replace(tmp, rp)

    if env.get("version") != PROTOCOL_VERSION:
        write_result({"ok": False, "error_kind": "envelope-version",
                      "error": f"envelope version {env.get('version')!r}; "
                               f"this worker speaks {PROTOCOL_VERSION}"})
        return 0

    # Test hook: lets the protocol tests kill a real worker on demand.
    fault = (env.get("args") or {}).get("_test_fault")
    if fault:
        from functions.crashsafe import induce_test_fault
        log(f"inducing test fault: {fault}")
        log(induce_test_fault(fault))  # refuses (and logs) when not enabled

    handler = _HANDLERS.get(env.get("op"))
    if handler is None:
        write_result({"ok": False, "error_kind": "unknown-op",
                      "error": f"unknown op {env.get('op')!r}"})
        return 0

    started = _dt.datetime.now().isoformat(timespec="seconds")
    log(f"worker start op={env['op']} pid={_os.getpid()} at {started}")
    # Record the import environment: a worker missing the USER site-packages
    # fails only on user-site installs, which the log must be able to show.
    try:
        import site as _site
        _usp = _site.getusersitepackages() if _site.ENABLE_USER_SITE else ""
        if isinstance(_usp, (list, tuple)):
            _usp = _usp[0] if _usp else ""
        _on_path = bool(_usp) and any(
            _os.path.normcase(p) == _os.path.normcase(_usp) for p in _sys.path)
        log(f"  exe={_sys.executable}")
        log(f"  prefix={_sys.prefix}")
        log(f"  user_site={'on' if _site.ENABLE_USER_SITE else 'OFF'} "
            f"path={_usp or '(none)'} "
            f"{'(on sys.path)' if _on_path else '(NOT on sys.path)'}")
    except Exception:
        pass
    try:
        summary, outputs = handler(env.get("args") or {}, log)
    except Exception as exc:
        log("".join(_tb.format_exc()))
        write_result({"ok": False, "error_kind": "computation",
                      "error": f"{type(exc).__name__}: {exc}",
                      "detail_path": log_path})
        return 0
    write_result({"ok": True, "outputs": [str(o) for o in outputs if o],
                  "summary": summary})
    log("worker done")
    return 0


# --- Parent half ---

class JobOutcome:
    """What the GUI gets back from :func:`run_job`. Plain data, no behaviour."""

    def __init__(self, *, ok: bool, died: bool = False,
                 exit_code: Optional[int] = None, error: str = "",
                 error_kind: str = "", summary: Optional[dict] = None,
                 outputs: Optional[list] = None,
                 postmortem_path: Optional[str] = None,
                 log_path: Optional[str] = None):
        self.ok = ok
        self.died = died
        self.exit_code = exit_code
        self.error = error
        self.error_kind = error_kind
        self.summary = summary or {}
        self.outputs = outputs or []
        self.postmortem_path = postmortem_path
        self.log_path = log_path

    def __repr__(self):  # pragma: no cover - debugging nicety
        return (f"JobOutcome(ok={self.ok}, died={self.died}, "
                f"exit_code={self.exit_code}, error={self.error!r})")


def _jobs_root() -> Path:
    from functions.layout import app_home
    return app_home() / "worker"


def _prune_job_dirs(keep: int = _KEEP_JOB_DIRS) -> None:
    try:
        dirs = sorted((d for d in _jobs_root().iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime)
        for d in dirs[:-keep] if keep else dirs:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def run_job(op: str, args: dict, *,
            log_cb: Optional[Callable[[str], None]] = None,
            timeout: Optional[float] = None,
            keep_dir: bool = False) -> JobOutcome:
    """Execute one job in a worker subprocess. Never raises for job failure:
    every outcome, a dead worker included, comes back as a :class:`JobOutcome`."""
    from functions.layout import REPO_ROOT
    from functions import crashsafe

    job_dir = _jobs_root() / f"{_dt.datetime.now():%Y%m%dT%H%M%S}_{_uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    _prune_job_dirs()
    envelope_path = job_dir / "envelope.json"
    result_path = job_dir / "result.json"
    log_path = job_dir / "log.txt"
    log_path.touch()
    envelope = {"version": PROTOCOL_VERSION, "op": op, "args": args,
                "result_path": str(result_path), "log_path": str(log_path)}
    envelope_path.write_text(_json.dumps(envelope, indent=1), encoding="utf-8")

    env = dict(_os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the child's import environment identical to the parent's: packages
    # installed in the USER site-packages (a non-writable conda env) would
    # otherwise be missing from the child whatever launcher started the GUI.
    env.pop("PYTHONNOUSERSITE", None)
    try:
        import site as _site
        _usp = _site.getusersitepackages() if _site.ENABLE_USER_SITE else ""
        if isinstance(_usp, (list, tuple)):
            _usp = _usp[0] if _usp else ""
        if _usp and _os.path.isdir(_usp):
            env["PYTHONPATH"] = (
                _usp + _os.pathsep + env.get("PYTHONPATH", "")
            ).rstrip(_os.pathsep)
    except Exception:
        pass

    started = _dt.datetime.now().isoformat(timespec="seconds")
    proc = _sp.Popen([_sys.executable, "-m", "functions.worker",
                      str(envelope_path)],
                     cwd=str(REPO_ROOT), env=env,
                     stdout=_sp.DEVNULL, stderr=_sp.STDOUT,
                     creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))

    ceiling = timeout if timeout is not None else \
        OP_TIMEOUTS.get(op, DEFAULT_TIMEOUT)
    deadline = _time.monotonic() + ceiling
    timed_out = False
    pos = 0

    def _drain() -> None:
        nonlocal pos
        if log_cb is None:
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                if line:
                    log_cb(line)
        except OSError:
            pass

    while True:
        rc = proc.poll()
        _drain()
        if rc is not None:
            break
        if _time.monotonic() > deadline:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except _sp.TimeoutExpired:
                proc.kill()
                proc.wait()
            rc = proc.poll()
            break
        _time.sleep(0.3)
    _drain()
    ended = _dt.datetime.now().isoformat(timespec="seconds")

    result = None
    if result_path.exists():
        try:
            result = _json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            result = None

    if result is not None and rc == 0 and not timed_out:
        if result.get("ok"):
            if not keep_dir:
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
            return JobOutcome(ok=True, summary=result.get("summary"),
                              outputs=result.get("outputs"),
                              log_path=None if not keep_dir else str(log_path))
        return JobOutcome(ok=False,
                          error=result.get("error") or "job failed",
                          error_kind=result.get("error_kind") or "computation",
                          log_path=str(log_path))

    # Dead worker: any nonzero exit, a timeout, or exit 0 with no result.
    code = rc if rc is not None else -1
    if timed_out:
        error = (f"The job's worker process did not finish within "
                 f"{int(ceiling)} s and was terminated.")
    else:
        error = (f"The job's worker process died "
                 f"(exit {code} / 0x{(code or 0) & 0xFFFFFFFF:08X}) "
                 "without reporting a result. The app itself is unaffected.")
    pm = crashsafe.write_worker_postmortem(code, op, str(envelope_path),
                                           started, ended)
    return JobOutcome(ok=False, died=True, exit_code=code, error=error,
                      error_kind="worker-died",
                      postmortem_path=str(pm) if pm else None,
                      log_path=str(log_path))


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    _sys.exit(_worker_main(_sys.argv[1]))
