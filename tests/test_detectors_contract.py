from pathlib import Path
# tests/test_detectors_contract.py — what the app relies on every backend for.
#
# Mask R-CNN's provenance manifest, the terminal-event guarantee of
# ``run_detect_jobs``, the one CSV naming helper, and the two declaration
# features of user_detectors.json (``path``, the install-hint line).
import json
import logging
import os
import time

import pandas as pd
import pytest

from detectors import base, registry
from detectors.base import DetectionManifest, output_csv_name, run_detect_jobs
from functions import clasts_detection, modes, naming


# --------------------------------------------------------------------------- #
#  Naming                                                                     #
# --------------------------------------------------------------------------- #
def test_output_csv_name_matches_the_core_convention():
    assert output_csv_name(modes.QUADRAT, "img") == naming.quadrat_csv_name("img")
    assert output_csv_name(modes.QUADRAT, "img") == "img_individual_clasts.csv"
    assert output_csv_name(modes.ORTHO, "o", 2.5) == naming.detection_csv_name("o", 2.5)
    assert output_csv_name(modes.ORTHO, "o", 2.5) == "o_ws2.5m.csv"
    # Legacy spellings are normalised like everywhere else.
    assert output_csv_name("terrestrial", "img", 1.0) == "img_individual_clasts.csv"
    assert output_csv_name("UAV", "o", 1) == "o_ws1m.csv"
    with pytest.raises(ValueError):
        output_csv_name(modes.ORTHO, "o")


# --------------------------------------------------------------------------- #
#  Mask R-CNN writes ITS manifest beside the CSV it wrote                     #
# --------------------------------------------------------------------------- #
def _fake_detect_writing_csv(monkeypatch, tmp_path):
    """Stand-in for clasts_detect_jobs that writes the CSV the real one would."""
    def fake(mode, jobs, **kwargs):
        out = []
        for job in jobs:
            csv_path = base.output_csv_path(mode, job, kwargs)
            df = pd.DataFrame([[1, 0.0, 0.0] + [0.1] * 12],
                              columns=list(base.CANONICAL_COLUMNS))
            df.to_csv(csv_path, index=False)
            out.append(df)
        return out
    monkeypatch.setattr(clasts_detection, "clasts_detect_jobs", fake)


@pytest.mark.parametrize("mode,kw", [
    (modes.QUADRAT, {"resolution": 0.001}),
    (modes.ORTHO, {"metric_cropsize": 2.5}),
])
def test_maskrcnn_manifest_names_maskrcnn_and_replaces_a_stale_one(
        monkeypatch, tmp_path, mode, kw):
    _fake_detect_writing_csv(monkeypatch, tmp_path)
    b = registry.get_backend("maskrcnn")
    job = {"path": str(tmp_path / "img.jpg"), "kstart": 0}
    csv_path = base.output_csv_path(mode, job, dict(kw, output_dir=str(tmp_path)))
    # A manifest from another model already sits where the new one goes.
    stale = DetectionManifest.sidecar_path(csv_path)
    stale.write_text(json.dumps({"model": "OLD"}), encoding="utf-8")
    old = time.time() - 3600
    os.utime(stale, (old, old))

    b.detect_jobs(mode, [job], output_dir=str(tmp_path), **kw)

    assert csv_path.exists()
    d = json.loads(stale.read_text(encoding="utf-8"))
    assert d["model"] == "maskrcnn"
    assert d["license"] == b.info.license
    assert d["params"] == kw


def test_maskrcnn_write_manifests_needs_nothing_imported_inside_detect_jobs(
        tmp_path):
    """Call the writer directly: every name it uses is module-level."""
    from detectors import maskrcnn as mr
    b = registry.get_backend("maskrcnn")
    job = {"path": str(tmp_path / "img.jpg"), "kstart": 0}
    csv_path = tmp_path / naming.quadrat_csv_name("img")
    csv_path.write_text("clast_ID,x,y\n", encoding="utf-8")
    b._write_manifests(modes.QUADRAT, [job],
                       {"output_dir": str(tmp_path), "resolution": 0.001},
                       time.time() - 5)
    mf = DetectionManifest.sidecar_path(csv_path)
    assert mf.exists()
    assert json.loads(mf.read_text(encoding="utf-8"))["model"] == mr.INFO.name


def test_discard_stale_removes_only_manifests_older_than_their_csv(tmp_path):
    csv = tmp_path / "a.csv"
    mf = DetectionManifest.sidecar_path(csv)
    csv.write_text("clast_ID,x,y\n", encoding="utf-8")
    mf.write_text("{}", encoding="utf-8")
    assert DetectionManifest.discard_stale(csv) is False       # fresh: kept
    assert mf.exists()
    old = time.time() - 3600
    os.utime(mf, (old, old))
    assert DetectionManifest.discard_stale(csv) is True        # stale: gone
    assert not mf.exists()
    assert DetectionManifest.discard_stale(tmp_path / "none.csv") is False


# --------------------------------------------------------------------------- #
#  run_detect_jobs: one terminal event per job, whatever the backend does     #
# --------------------------------------------------------------------------- #
class _Silent(base.DetectorBackend):
    """A backend written without progress_callback, as the guide once allowed."""
    info = base.BackendInfo(name="_silent", display_name="s",
                            framework="classical", license="MIT")

    def __init__(self, frames=None, raise_with=None, write=False):
        self._frames = frames
        self._raise = raise_with
        self._write = write

    def is_available(self):
        return True

    def detect_jobs(self, mode, jobs, **kwargs):
        if self._raise is not None:
            raise self._raise
        if self._write:
            for job in jobs:
                p = base.output_csv_path(mode, job, kwargs)
                p.write_text("clast_ID,x,y\n1,0,0\n", encoding="utf-8")
        return self._frames


def _collect():
    events = []
    return events, (lambda ji, ev, payload: events.append((ji, ev, payload)))


def test_backend_that_never_reports_is_settled_from_its_return_value(tmp_path):
    events, cb = _collect()
    frames = [pd.DataFrame({"clast_ID": [1, 2, 3]}), pd.DataFrame()]
    jobs = [{"path": str(tmp_path / "a.jpg")}, {"path": str(tmp_path / "b.jpg")}]
    out = run_detect_jobs(_Silent(frames), modes.QUADRAT, jobs,
                          output_dir=str(tmp_path), progress_callback=cb)
    assert out == frames
    assert events == [(0, "done", {"n_clasts": 3}), (1, "done", {"n_clasts": 0})]


def test_backend_that_raises_is_reported_as_error_and_the_error_propagates(tmp_path):
    events, cb = _collect()
    jobs = [{"path": str(tmp_path / "a.jpg")}]
    with pytest.raises(RuntimeError, match="boom"):
        run_detect_jobs(_Silent(raise_with=RuntimeError("boom")),
                        modes.QUADRAT, jobs, output_dir=str(tmp_path),
                        progress_callback=cb)
    assert [(ji, ev) for ji, ev, _ in events] == [(0, "error")]
    assert events[0][2]["message"] == "boom"
    assert "RuntimeError" in events[0][2]["traceback"]


def test_a_backend_that_reports_itself_is_not_reported_twice(tmp_path):
    class _Talkative(_Silent):
        def detect_jobs(self, mode, jobs, **kwargs):
            cb = kwargs["progress_callback"]
            cb(0, "start", {})
            cb(0, "done", {"n_clasts": 7})
            cb(1, "error", {"message": "bad tile"})
            return [pd.DataFrame({"clast_ID": range(7)}), pd.DataFrame()]

    events, cb = _collect()
    jobs = [{"path": str(tmp_path / "a.jpg")}, {"path": str(tmp_path / "b.jpg")}]
    run_detect_jobs(_Talkative(), modes.QUADRAT, jobs, output_dir=str(tmp_path),
                    progress_callback=cb)
    assert [(ji, ev) for ji, ev, _ in events] == [
        (0, "start"), (0, "done"), (1, "error")]


def test_a_stop_during_a_silent_backend_reports_stopped_not_done(tmp_path):
    events, cb = _collect()
    jobs = [{"path": str(tmp_path / "a.jpg")}]
    run_detect_jobs(_Silent([pd.DataFrame()]), modes.QUADRAT, jobs,
                    output_dir=str(tmp_path), progress_callback=cb,
                    stop_check=lambda: True)
    assert [(ji, ev) for ji, ev, _ in events] == [(0, "stopped")]


def test_run_detect_jobs_works_without_a_callback_and_discards_stale_manifests(
        tmp_path):
    job = {"path": str(tmp_path / "a.jpg"), "out_stem": "site__a"}
    csv_path = tmp_path / naming.quadrat_csv_name("site__a")
    mf = DetectionManifest.sidecar_path(csv_path)
    mf.write_text(json.dumps({"model": "OTHER"}), encoding="utf-8")
    old = time.time() - 3600
    os.utime(mf, (old, old))

    out = run_detect_jobs(_Silent([pd.DataFrame()], write=True),
                          modes.QUADRAT, [job], output_dir=str(tmp_path))
    assert len(out) == 1
    assert csv_path.exists()
    assert not mf.exists(), "the other model's manifest survived a rewrite"


def test_run_detect_jobs_forwards_kwargs_and_normalises_the_mode(tmp_path):
    seen = {}

    class _Spy(_Silent):
        def detect_jobs(self, mode, jobs, **kwargs):
            seen["mode"] = mode
            seen["kwargs"] = kwargs
            return [pd.DataFrame()]

    run_detect_jobs(_Spy(), "UAV", [{"path": "x.tif"}], metric_cropsize=2.0,
                    min_confidence=0.7, log_fn=print)
    assert seen["mode"] == modes.ORTHO
    assert seen["kwargs"]["metric_cropsize"] == 2.0
    assert seen["kwargs"]["min_confidence"] == 0.7
    assert "progress_callback" not in seen["kwargs"]


# --------------------------------------------------------------------------- #
#  user_detectors.json: "path" and the install-hint line                      #
# --------------------------------------------------------------------------- #
def _write_backend(pkg_dir, name, available):
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "backend.py").write_text(
        "from detectors.base import BackendInfo, DetectorBackend\n"
        "class B(DetectorBackend):\n"
        f"    info = BackendInfo(name={name!r}, display_name='Path model',\n"
        "                       framework='pytorch', license='MIT',\n"
        "                       install_hint='create the pm-x env and download w.pt')\n"
        f"    def is_available(self): return {available!r}\n"
        "    def detect_jobs(self, mode, jobs, **kwargs): return []\n"
        "def make_backend(): return B()\n",
        encoding="utf-8")


def test_path_key_puts_the_backend_folder_on_sys_path(tmp_path, monkeypatch):
    """Relative to the JSON file's folder; no PYTHONPATH needed."""
    import sys
    name = "_pathed_backend"
    _write_backend(tmp_path / "plugins" / "pathed_pkg", name, True)
    decl_dir = tmp_path / "conf"
    decl_dir.mkdir()
    decl = decl_dir / "user_detectors.json"
    decl.write_text(json.dumps([{"module": "pathed_pkg.backend",
                                 "factory": "make_backend",
                                 "path": "../plugins"}]), encoding="utf-8")
    monkeypatch.setenv("PEBBLEMAPPER_DETECTORS", str(decl))
    sys.modules.pop("pathed_pkg.backend", None)
    sys.modules.pop("pathed_pkg", None)
    assert str((tmp_path / "plugins").resolve()) not in sys.path

    added = registry.load_user_backends()
    try:
        assert name in added
        assert str((tmp_path / "plugins").resolve()) in sys.path
        assert name in [b.info.name for b in registry.available_backends()]
    finally:
        registry._BACKENDS.pop(name, None)
        sys.path.remove(str((tmp_path / "plugins").resolve()))
        sys.modules.pop("pathed_pkg.backend", None)
        sys.modules.pop("pathed_pkg", None)


def test_an_unavailable_declared_backend_logs_its_install_hint(
        tmp_path, monkeypatch, caplog):
    import sys
    name = "_hinted_backend"
    _write_backend(tmp_path / "hinted_pkg", name, False)
    decl = tmp_path / "user_detectors.json"
    decl.write_text(json.dumps([{"module": "hinted_pkg.backend",
                                 "factory": "make_backend",
                                 "path": str(tmp_path)}]), encoding="utf-8")
    monkeypatch.setenv("PEBBLEMAPPER_DETECTORS", str(decl))
    sys.modules.pop("hinted_pkg.backend", None)
    sys.modules.pop("hinted_pkg", None)

    with caplog.at_level(logging.WARNING, logger="csm.detectors"):
        added = registry.load_user_backends()
    try:
        assert name in added                      # registered ...
        assert name not in [b.info.name for b in registry.available_backends()]
        lines = [r.getMessage() for r in caplog.records
                 if r.name == "csm.detectors" and name in r.getMessage()]
        assert len(lines) == 1                    # ... and said why, once
        assert "create the pm-x env and download w.pt" in lines[0]
        assert "hidden from the model selector" in lines[0]
    finally:
        registry._BACKENDS.pop(name, None)
        if str(tmp_path.resolve()) in sys.path:
            sys.path.remove(str(tmp_path.resolve()))
        sys.modules.pop("hinted_pkg.backend", None)
        sys.modules.pop("hinted_pkg", None)


def _heic(tmp_path, w=60, h=40):
    pillow_heif = pytest.importorskip("pillow_heif")
    import numpy as np
    from PIL import Image
    pillow_heif.register_heif_opener()
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:10, :, 0] = 255          # red band at the TOP
    path = tmp_path / "IMG_0001.heic"
    Image.fromarray(arr).save(path, quality=-1, chroma=444)
    return path


class _Recorder(base.DetectorBackend):
    def __init__(self, in_process):
        self.info = base.BackendInfo(name="rec", display_name="Recorder",
                                     framework="none", license="MIT",
                                     in_process=in_process,
                                     env=None if in_process else "somewhere")
        self.seen = []

    def is_available(self):
        return True

    def detect_jobs(self, mode, jobs, **kwargs):
        import numpy as np
        from PIL import Image
        for job in jobs:
            p = job["path"]
            info = {"path": p, "out_stem": job.get("out_stem"),
                    "source_path": job.get("source_path")}
            if str(p).lower().endswith(".png"):
                with Image.open(p) as im:          # plain Pillow, no HEIF plugin needed
                    a = np.asarray(im.convert("RGB"))
                info["shape"] = a.shape
                info["top_red"] = int(a[2, 30, 0])
            self.seen.append(info)
        return [None for _ in jobs]


def test_a_subprocess_backend_gets_a_png_copy_of_a_heic_in_the_same_frame(tmp_path):
    src = _heic(tmp_path)
    rec = _Recorder(in_process=False)
    base.run_detect_jobs(rec, "quadrat", [{"path": str(src)}],
                         output_dir=str(tmp_path / "out"))
    seen = rec.seen[0]
    assert seen["path"].lower().endswith(".png")
    assert seen["out_stem"] == "IMG_0001"
    assert seen["source_path"] == str(src)
    assert seen["shape"] == (40, 60, 3)
    assert seen["top_red"] > 200, "the copy must keep the photograph upright"
    assert not Path(seen["path"]).exists(), "the temporary copy is removed"


def test_an_in_process_backend_gets_the_heic_itself(tmp_path):
    src = _heic(tmp_path)
    rec = _Recorder(in_process=True)
    base.run_detect_jobs(rec, "quadrat", [{"path": str(src)}])
    assert rec.seen[0]["path"] == str(src)
    assert rec.seen[0]["out_stem"] is None


def test_non_heif_jobs_pass_through_for_a_subprocess_backend(tmp_path):
    jpg = tmp_path / "a.jpg"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(jpg)
    rec = _Recorder(in_process=False)
    base.run_detect_jobs(rec, "quadrat", [{"path": str(jpg)}])
    assert rec.seen[0]["path"] == str(jpg)

