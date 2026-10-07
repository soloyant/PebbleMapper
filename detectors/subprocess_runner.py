"""Run an out-of-process detection backend in its own conda environment.

A backend with conflicting dependencies or an incompatible licence lives in a
separate conda environment and talks to the core app only through files in
the canonical schema; the core app never imports its code.

Protocol (instance-level detection)
-----------------------------------
1. The in-core driver (the backend's ``detect_jobs``) builds a *job spec*::

       {
         "mode": "ortho",                     # modes.ORTHO or modes.QUADRAT
         "params": {"resolution": 0.004, "metric_cropsize": 2.5, ...},
         "canonical_columns": [...],          # the exact CSV header
         "jobs": [
           {"path": "<image>", "kstart": 0,
            "out_csv": "<dest .csv>",
            "instances_path": "<dest .csv>.instances.npz"},
           ...
         ]
       }

   The ``out_csv`` for each job is computed by the driver with
   ``detectors.base.output_csv_name`` (the core owns the naming convention),
   so the subprocess never needs the core's naming module.
2. It runs ``conda run -n <env> --no-capture-output python <script> --spec <f>``
   (:func:`run_backend`; ``stream=True`` forwards the child's output line by
   line and honours ``stop_check``).
3. The subprocess reads the spec, segments each image and writes
   ``instances_path`` — the *instances file* (:func:`read_instances_npz`)::

       labels  int32 HxW   0 = background, k = instance k (1..N)
       scores  float32 N   per-instance confidence, any monotone proxy
       shape   int64 (H, W)

   It does not measure: the driver measures every instance with the core's
   shared ``detectors.measure`` step and writes the canonical CSV, so sizes
   are defined identically to Mask R-CNN's.
4. Alternatively a subprocess that re-implements measurement writes
   ``out_csv`` itself (header == ``canonical_columns``, world coordinates in
   metres); the driver then reads it through :func:`read_canonical_csv`,
   which enforces the schema at the boundary. Its numbers are then only as
   comparable to the other backends' as its re-implementation is.

Only the subprocess invocation and the two file readers live here; the
per-backend spec building and CSV writing live in each backend's
``detect_jobs``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


class BackendStopped(Exception):
    """Raised by :func:`run_backend` when ``stop_check()`` turned true and the
    subprocess tree was killed. Not a failure: the driver reports "stopped"."""


class Instance(NamedTuple):
    """One instance from an instances file, cropped to its bounding box.

    ``mask`` is a boolean array covering rows ``y0:y0+h`` and columns
    ``x0:x0+w`` of the image (a ``pad``-pixel margin included, holes filled);
    ``measure_mask``'s ``center_x``/``center_y`` plus ``x0``/``y0`` give the
    centroid in image pixels.
    """
    mask: object      # numpy bool array, h x w
    score: float
    y0: int
    x0: int

# Env-list cache keyed by conda exe, so repeated is_available() calls don't
# each shell out; the TTL lets an env created while the app is open be found.
_ENV_LIST_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_ENV_CACHE_TTL_S = 30.0


def find_conda_exe() -> Optional[str]:
    """Locate a conda/mamba executable, or None.

    Honours ``PEBBLE_CONDA_EXE`` first (explicit override), then the PATH, then
    a handful of conventional install locations on Windows and POSIX.
    """
    env_override = os.environ.get("PEBBLE_CONDA_EXE")
    if env_override and Path(env_override).exists():
        return env_override
    # A .bat/.cmd shim (what conda puts on PATH via condabin) does not launch
    # reliably through subprocess.run's list form on Windows; try real exes first.
    candidates = [
        "C:/ProgramData/miniconda3/Scripts/conda.exe",
        "C:/ProgramData/Anaconda3/Scripts/conda.exe",
        os.path.expanduser("~/miniconda3/Scripts/conda.exe"),
        os.path.expanduser("~/anaconda3/Scripts/conda.exe"),
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        "/opt/conda/bin/conda",
    ]
    exe = next((c for c in candidates if Path(c).exists()), None)
    if exe:
        return exe
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _list_env_paths(conda_exe: str) -> List[str]:
    """Folders of all conda envs (cached per conda exe, with a short TTL)."""
    cached = _ENV_LIST_CACHE.get(conda_exe)
    if cached is not None and (time.monotonic() - cached[0]) < _ENV_CACHE_TTL_S:
        return cached[1]
    paths: List[str] = []
    try:
        out = subprocess.run([conda_exe, "env", "list", "--json"],
                             capture_output=True, text=True, timeout=60)
        paths = list(json.loads(out.stdout or "{}").get("envs", []))
    except Exception:
        paths = []
    _ENV_LIST_CACHE[conda_exe] = (time.monotonic(), paths)
    return paths


def _list_envs(conda_exe: str) -> List[str]:
    """Names of all conda envs."""
    return [Path(p).name for p in _list_env_paths(conda_exe)]


def conda_env_prefix(env: str, conda_exe: Optional[str] = None) -> Optional[Path]:
    """The folder of the conda env named ``env``, or None."""
    if not env:
        return None
    conda_exe = conda_exe or find_conda_exe()
    if not conda_exe:
        return None
    for p in _list_env_paths(conda_exe):
        if Path(p).name == env:
            return Path(p)
    return None


def missing_modules(prefix: Path, modules) -> List[str]:
    """Which of ``modules`` the env at ``prefix`` does not hold.

    Read from its site-packages on disk rather than by importing, so the check
    costs no subprocess: an env whose build stopped half way (conda's solver
    running out of memory on GDAL, say) exists by name but lacks the package.
    """
    sites = [prefix / "Lib" / "site-packages"]
    sites += sorted(prefix.glob("lib/python*/site-packages"))
    sites = [s for s in sites if s.is_dir()]
    missing = []
    for mod in modules:
        top = mod.split(".")[0]
        found = any((s / top).is_dir() or (s / f"{top}.py").is_file()
                    or any(s.glob(f"{top}.*.pyd")) or any(s.glob(f"{top}.*.so"))
                    # an editable install leaves only a pointer behind
                    or any(s.glob(f"__editable__*{top}*"))
                    or any(s.glob(f"{top}*.egg-link"))
                    or any(s.glob(f"{top}-*.dist-info"))
                    for s in sites)
        if not found:
            missing.append(mod)
    return missing


def clear_env_cache() -> None:
    """Forget cached env lists (call after creating/removing an env)."""
    _ENV_LIST_CACHE.clear()


def conda_env_exists(env: str, conda_exe: Optional[str] = None) -> bool:
    """True if a conda env named ``env`` is installed and conda is reachable."""
    if not env:
        return False
    conda_exe = conda_exe or find_conda_exe()
    if not conda_exe:
        return False
    return env in _list_envs(conda_exe)


def build_command(env: str, script, spec_path, conda_exe: str) -> List[str]:
    """The exact ``conda run`` argv used to launch a subprocess backend.

    Factored out so tests can assert on it without spawning a process.
    """
    return [conda_exe, "run", "-n", env, "--no-capture-output",
            "python", str(script), "--spec", str(spec_path)]


def kill_process_tree(proc) -> None:
    """Terminate ``proc`` and its children.

    ``conda run`` launches the real interpreter as a grandchild, so killing
    the launcher alone leaves the model running; on Windows ``taskkill /T``
    takes the whole tree. Never raises.
    """
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.kill()
    except Exception:
        pass


_STOP_POLL_S = 0.5


def _run_streaming(cmd, env, *, log_fn, stop_check, timeout):
    """Popen with stdout+stderr streamed line by line into ``log_fn``.

    Polls ``stop_check`` every 0.5 s; on stop kills the process tree and
    raises :class:`BackendStopped`. Returns a ``CompletedProcess`` whose
    ``stdout`` holds the last 200 lines (``stderr`` is merged into it).
    """
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8", TQDM_DISABLE="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            env=child_env)
    tail: List[str] = []

    def _pump():
        try:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                tail.append(line)
                del tail[:-200]
                if log_fn:
                    try:
                        log_fn(line)
                    except Exception:
                        pass
        except Exception:
            pass

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    t0 = time.monotonic()
    while proc.poll() is None:
        if stop_check is not None:
            try:
                wanted = bool(stop_check())
            except Exception:
                wanted = False
            if wanted:
                if log_fn:
                    log_fn(f"[subprocess] {env}: stop requested; "
                           "terminating the process tree")
                kill_process_tree(proc)
                proc.wait()
                pump.join(5)
                raise BackendStopped(env)
        if timeout and (time.monotonic() - t0) > float(timeout):
            kill_process_tree(proc)
            proc.wait()
            pump.join(5)
            raise RuntimeError(
                f"Subprocess backend '{env}' timed out after {timeout}s.")
        time.sleep(_STOP_POLL_S)
    pump.join(5)
    return subprocess.CompletedProcess(cmd, proc.returncode,
                                       stdout="\n".join(tail), stderr="")


def run_backend(env: str, script, spec: dict, *,
                conda_exe: Optional[str] = None,
                timeout: Optional[float] = None,
                log_fn=None,
                stream: bool = False,
                stop_check=None) -> subprocess.CompletedProcess:
    """Write ``spec`` to a temp file and run ``script`` in conda env ``env``.

    Default (``stream=False``): ``subprocess.run`` with captured output; the
    child's output is only seen on failure. ``stream=True``: the child's
    stdout and stderr are forwarded line by line to ``log_fn`` as they
    arrive, ``stop_check()`` is polled every 0.5 s and, when it turns true,
    the whole process tree is killed and :class:`BackendStopped` is raised.
    A long-running model should always be launched with ``stream=True`` so a
    GUI Stop reaches it and its progress shows in the log.

    Raises ``RuntimeError`` if conda can't be found, the subprocess exits
    non-zero or ``timeout`` (seconds, None = no limit) elapses; returns the
    ``CompletedProcess`` on success. The caller then reads back the per-job
    files named in ``spec``.
    """
    conda_exe = conda_exe or find_conda_exe()
    if not conda_exe:
        raise RuntimeError(
            "Could not locate a conda executable to run subprocess backend "
            f"'{env}'. Set PEBBLE_CONDA_EXE to your conda path.")

    # Not a NamedTemporaryFile: on Windows the child could not open it while held.
    fd, spec_path = tempfile.mkstemp(prefix="pebble_detect_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(spec, fh)
        cmd = build_command(env, script, spec_path, conda_exe)
        if log_fn:
            log_fn(f"[subprocess] {env}: {Path(str(script)).name} "
                   f"({len(spec.get('jobs', []))} job(s))")
        if stream:
            proc = _run_streaming(cmd, env, log_fn=log_fn,
                                  stop_check=stop_check, timeout=timeout)
        else:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                # timeout=None (no limit) by default so a long real detection is never killed.
                raise RuntimeError(
                    f"Subprocess backend '{env}' timed out after {timeout}s.")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"Subprocess backend '{env}' failed (exit {proc.returncode}).\n"
                f"{tail}")
        return proc
    finally:
        try:
            os.remove(spec_path)
        except OSError:
            pass


def instances_path_for(csv_path) -> Path:
    """``<csv>.instances.npz`` — where a subprocess leaves its geometry."""
    return Path(str(csv_path) + ".instances.npz")


def read_instances_npz(path, pad: int = 2) -> List[Instance]:
    """Read an instances file into per-instance bounding-box masks.

    ``path`` holds ``labels`` (int32 HxW, 0 = background, k = instance k)
    and ``scores`` (float32 N, one per label 1..N). Each instance comes back
    as :class:`Instance` — a hole-filled boolean mask cropped to its bounding
    box plus ``pad`` pixels, its score and the crop's top-left corner — ready
    for ``detectors.measure.measure_mask``. A label with no pixels is skipped.
    Raises ``RuntimeError`` on a missing or malformed file.
    """
    import numpy as np
    from scipy import ndimage as ndi

    p = Path(path)
    if not p.exists():
        raise RuntimeError(
            f"Subprocess backend exited successfully but did not write its "
            f"instances file: {p}")
    try:
        with np.load(p) as data:
            labels = np.asarray(data["labels"])
            scores = np.asarray(data["scores"], dtype=float).ravel()
    except Exception as ex:
        raise RuntimeError(f"Unreadable instances file {p.name}: {ex}") from ex
    if labels.ndim != 2:
        raise RuntimeError(
            f"Instances file {p.name}: labels must be HxW, got shape "
            f"{labels.shape}")
    n = int(scores.shape[0])
    if labels.size and int(labels.max()) > n:
        raise RuntimeError(
            f"Instances file {p.name}: labels reach {int(labels.max())} but "
            f"only {n} score(s) were written")
    out: List[Instance] = []
    if n == 0:
        return out
    slices = ndi.find_objects(labels, max_label=n)
    h, w = labels.shape
    for k in range(1, n + 1):
        sl = slices[k - 1]
        if sl is None:
            continue
        y0 = max(0, sl[0].start - pad)
        y1 = min(h, sl[0].stop + pad)
        x0 = max(0, sl[1].start - pad)
        x1 = min(w, sl[1].stop + pad)
        mask = ndi.binary_fill_holes(labels[y0:y1, x0:x1] == k)
        out.append(Instance(mask=mask, score=float(scores[k - 1]),
                            y0=int(y0), x0=int(x0)))
    return out


def read_instances_shape(path) -> Optional[Tuple[int, int]]:
    """``(H, W)`` of the image an instances file describes: its ``shape``
    entry, else the label image's shape; None when unreadable. The image
    height maps a centroid row to the Quadrat ``y``."""
    import numpy as np
    try:
        with np.load(Path(path)) as data:
            if "shape" in data.files:
                sh = [int(v) for v in np.asarray(data["shape"]).ravel()[:2]]
                if len(sh) == 2 and sh[0] > 0 and sh[1] > 0:
                    return sh[0], sh[1]
            lab = np.asarray(data["labels"])
            if lab.ndim == 2:
                return int(lab.shape[0]), int(lab.shape[1])
    except Exception:
        return None
    return None


def instances_as_masks(instances, shape) -> Tuple[object, List[float]]:
    """``(H x W x N bool stack, scores)`` from :func:`read_instances_npz`
    records: what an in-process Mask R-CNN returns, for callers that want
    full-frame masks (one small frame; a large image with many instances is
    better kept as the cropped records)."""
    import numpy as np
    H, W = int(shape[0]), int(shape[1])
    masks = np.zeros((H, W, len(instances)), dtype=bool)
    scores = []
    for k, inst in enumerate(instances):
        h, w = inst.mask.shape
        masks[inst.y0:inst.y0 + h, inst.x0:inst.x0 + w, k] = inst.mask
        scores.append(float(inst.score))
    return masks, scores


def read_canonical_csv(path, columns):
    """Read a subprocess backend's output CSV and enforce the canonical schema.

    This is the single boundary where the contract is enforced before data
    flows into merge/rasterize. Returns a ``pandas.DataFrame`` (0 rows is
    valid). Raises ``RuntimeError`` on a missing file or a header that does not
    match ``columns`` exactly.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise RuntimeError(
            f"Subprocess backend exited successfully but did not write its "
            f"expected output: {p}")
    df = pd.read_csv(p)
    expected = list(columns)
    if list(df.columns) != expected:
        raise RuntimeError(
            f"Subprocess backend wrote a non-canonical schema to {p.name}.\n"
            f"  expected: {expected}\n  got:      {list(df.columns)}")
    return df
