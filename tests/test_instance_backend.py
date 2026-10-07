"""detectors.instance_backend: the ready-made driver a plug-in inherits.

The subprocess itself is stood in for (its only duty is to write the
instances file), so these run anywhere; what is checked is everything the
driver does around it -- the CSV where the Detect tab looks, one row per
instance measured by the shared step, the manifest with the model's own
version and licence, the ROI, the overlay, Stop, and a failing script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from detectors import InstanceSubprocessBackend  # noqa: E402
from detectors.base import BackendInfo, CANONICAL_COLUMNS  # noqa: E402


def _photo(path, w=200, h=150):
    rng = np.random.default_rng(1)
    img = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))
    return path


def _labels(h=150, w=200):
    """Three discs, one of them touching the frame edge."""
    lab = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[:h, :w]
    for k, (cx, cy, r) in enumerate([(50, 60, 18), (130, 70, 22), (195, 140, 15)], 1):
        lab[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = k
    return lab


class _Fake(InstanceSubprocessBackend):
    info = BackendInfo(name="fake", display_name="Fake segmenter",
                       framework="none", license="MIT", output_type="instance",
                       env="nowhere", in_process=False, weights="w.bin")
    env_name = "nowhere"
    script = Path("run.py")
    model_version = "fake 1.0"
    log_prefix = "fake"

    def __init__(self, labels=None, fail=False):
        self.labels = _labels() if labels is None else labels
        self.fail = fail
        self.specs = []

    def spec_params(self, mode, kwargs, resolution):
        return {"weights_dir": "w", "extra": kwargs.get("fake_option", 7)}

    def _run_streaming(self, spec, log_fn, stop_check, timeout):
        self.specs.append(spec)
        if self.fail:
            raise RuntimeError("Subprocess backend 'nowhere' failed (exit 1).")
        for job in spec["jobs"]:
            np.savez_compressed(job["instances_path"], labels=self.labels,
                                scores=np.full(int(self.labels.max()), 0.5,
                                               dtype=np.float32),
                                shape=np.array(self.labels.shape))


def test_the_driver_measures_every_instance_where_the_detect_tab_looks(tmp_path):
    img = _photo(tmp_path / "p_GSD=0.002m.jpg")
    out = tmp_path / "out"
    events = []
    be = _Fake()
    frames = be.detect_jobs("quadrat", [{"path": str(img)}], resolution=0.002,
                            output_dir=str(out), saveresults=True,
                            fake_option=3,
                            progress_callback=lambda ji, ev, p: events.append(ev))
    df = frames[0]
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert len(df) == 3 and list(df["clast_ID"]) == [1, 2, 3]
    # Discs of radius 18 and 22 px at 2 mm per px; the third is cut by the
    # frame edge, so it comes out smaller than its 0.06 m.
    ed = sorted(df["Equivalent_diameter"])
    np.testing.assert_allclose(ed[1:], [0.072, 0.088], rtol=0.08)
    assert ed[0] < 0.06
    assert (df["Score"] == 0.5).all()
    # y is measured upward from the bottom edge, as Quadrat mode writes it.
    row = df.iloc[0]
    assert row["x"] == pytest.approx(50, abs=1) and row["y"] == pytest.approx(150 - 60, abs=1)
    csv = out / "p_GSD=0.002m_individual_clasts.csv"
    assert csv.exists()
    assert pd.read_csv(csv).shape == (3, len(CANONICAL_COLUMNS))
    man = json.loads((str(csv) + ".manifest.json") and Path(str(csv) + ".manifest.json").read_text(encoding="utf-8"))
    assert man["model"] == "fake" and man["model_version"] == "fake 1.0"
    assert man["license"] == "MIT" and man["params"]["extra"] == 3
    assert "weights_dir" not in man["params"]
    # The spec the script received: the common keys plus the backend's own.
    spec = be.specs[0]
    assert spec["mode"] == "quadrat" and spec["params"]["resolution"] == 0.002
    assert spec["params"]["weights_dir"] == "w"
    assert spec["jobs"][0]["instances_path"] == str(csv) + ".instances.npz"
    assert events == ["start", "done"]


def test_a_plug_in_result_carries_its_outlines(tmp_path):
    """The driver attaches every instance's outline to the frame, in the CSV
    frame (y up), and the wrapper writes them beside the CSV: the figures and
    the report draw a plug-in's clasts as outlines, not as axes only. Before,
    the driver built the rows itself and no .contours.json existed."""
    from detectors import run_detect_jobs
    from functions import clast_geometry as CG
    img = _photo(tmp_path / "p_GSD=0.002m.jpg")
    out = tmp_path / "out"
    frames = run_detect_jobs(_Fake(), "quadrat", [{"path": str(img)}],
                             resolution=0.002, output_dir=str(out), saveresults=True)
    df = frames[0]
    contours = df.attrs.get("contours")
    assert contours and sorted(contours) == [1, 2, 3]
    # the first disc: centre (50, 60) in rows, radius 18 -> the outline sits
    # on a circle of that radius around (50, 150 - 60) in the CSV frame
    pts = np.asarray(contours[1], dtype=float)
    r = np.hypot(pts[:, 0] - 50, pts[:, 1] - (150 - 60))
    assert 15 < r.min() and r.max() < 21
    csv = out / "p_GSD=0.002m_individual_clasts.csv"
    written = CG.read_contours(csv)
    assert written and len(written) == 3


def test_an_roi_keeps_only_the_instances_inside_it(tmp_path):
    img = _photo(tmp_path / "p.jpg")
    roi = tmp_path / "roi.geojson"
    roi.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [170, 0], [170, 150], [0, 150], [0, 0]]]}}]}),
        encoding="utf-8")
    df = _Fake().detect_jobs("quadrat", [{"path": str(img), "roi_path": str(roi)}],
                             resolution=0.001, output_dir=str(tmp_path))[0]
    assert len(df) == 2, "the disc at x = 195 lies outside the ROI"
    assert list(df["clast_ID"]) == [1, 2]


def test_the_overlay_is_drawn_when_asked(tmp_path):
    img = _photo(tmp_path / "p.jpg")
    figs = tmp_path / "figs"
    _Fake().detect_jobs("quadrat", [{"path": str(img)}], resolution=0.001,
                        output_dir=str(tmp_path), figures_dir=str(figs),
                        saveplot=True)
    assert (figs / "p_overlay.png").exists()


def test_stop_before_the_run_and_a_failing_script(tmp_path):
    img = _photo(tmp_path / "p.jpg")
    events = []
    frames = _Fake().detect_jobs("quadrat", [{"path": str(img)}],
                                 resolution=0.001, output_dir=str(tmp_path),
                                 stop_check=lambda: True,
                                 progress_callback=lambda ji, ev, p: events.append(ev))
    assert frames[0].empty and events == ["stopped"]
    events.clear()
    with pytest.raises(RuntimeError, match="exit 1"):
        _Fake(fail=True).detect_jobs("quadrat", [{"path": str(img)}],
                                     resolution=0.001, output_dir=str(tmp_path),
                                     progress_callback=lambda ji, ev, p: events.append(ev))
    assert events == ["start", "error"]


def test_is_available_needs_the_script_and_the_env(monkeypatch, tmp_path):
    from detectors import subprocess_runner
    be = _Fake()
    assert be.is_available() is False, "run.py does not exist"
    be.script = _photo(tmp_path / "run.py")
    monkeypatch.setattr(subprocess_runner, "conda_env_exists", lambda name: name == "nowhere")
    assert be.is_available() is True


def test_an_env_that_exists_but_lacks_a_required_module_is_not_available(
        monkeypatch, tmp_path):
    """A conda env whose build stopped half way exists by name; the plug-in
    must be refused up front, not fail with ModuleNotFoundError mid-run."""
    from detectors import subprocess_runner
    prefix = tmp_path / "env"
    site = prefix / "Lib" / "site-packages"
    (site / "cv2").mkdir(parents=True)
    be = _Fake()
    be.script = _photo(tmp_path / "run.py")
    be.required_modules = ("cv2", "osgeo")
    monkeypatch.setattr(subprocess_runner, "conda_env_exists", lambda name: True)
    monkeypatch.setattr(subprocess_runner, "conda_env_prefix", lambda name: prefix)
    assert be.missing_modules() == ["osgeo"]
    assert be.is_available() is False
    (site / "osgeo").mkdir()
    assert be.missing_modules() == []
    assert be.is_available() is True


def test_missing_modules_reads_posix_site_packages_and_single_files(tmp_path):
    from detectors import subprocess_runner
    site = tmp_path / "lib" / "python3.9" / "site-packages"
    site.mkdir(parents=True)
    (site / "six.py").write_text("")
    (site / "_fast.cp39-win_amd64.pyd").write_text("")
    got = subprocess_runner.missing_modules(tmp_path, ("six", "_fast", "osgeo.gdal"))
    assert got == ["osgeo.gdal"]


def test_missing_modules_accepts_an_editable_install(tmp_path):
    from detectors import subprocess_runner
    site = tmp_path / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "__editable__.segmenteverygrain-0.2.pth").write_text("")
    assert subprocess_runner.missing_modules(tmp_path, ("segmenteverygrain",)) == []
