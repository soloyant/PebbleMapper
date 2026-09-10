# tests/test_subprocess_backend.py — cross-environment backend proof.
#
# Most tests here run anywhere (they mock the subprocess). The full round-trip
# test runs only where the throwaway 'pebble-stub' conda env has been created
# (`conda env create -f detectors/stub/environment.yml`); elsewhere it skips.
import json

import numpy as np
import pytest

from detectors import registry, subprocess_runner
from detectors.base import CANONICAL_COLUMNS
from detectors.stub import run as stub_run
from functions import modes


def test_stub_is_registered_but_gated():
    # Registered (template/proof) ...
    assert "stub" in [b.info.name for b in registry.all_backends()]
    stub = registry.get_backend("stub")
    assert stub.info.in_process is False
    assert stub.info.env == "pebble-stub"
    # ... and excluded from the available list unless its env exists.
    if not stub.is_available():
        assert "stub" not in [b.info.name for b in registry.available_backends()]


def test_build_command_shape():
    cmd = subprocess_runner.build_command(
        "pebble-stub", "/path/run.py", "/tmp/spec.json", "conda")
    assert cmd[:5] == ["conda", "run", "-n", "pebble-stub", "--no-capture-output"]
    assert cmd[5] == "python"
    assert cmd[-2:] == ["--spec", "/tmp/spec.json"]


def test_run_backend_without_conda_raises(monkeypatch):
    monkeypatch.setattr(subprocess_runner, "find_conda_exe", lambda: None)
    with pytest.raises(RuntimeError, match="conda"):
        subprocess_runner.run_backend("pebble-stub", "/x/run.py", {"jobs": []})


def test_run_backend_serialises_spec_and_checks_returncode(monkeypatch):
    seen = {}

    class _Proc:
        def __init__(self, rc, err=""):
            self.returncode, self.stderr, self.stdout = rc, err, ""

    def fake_run(cmd, capture_output, text, timeout):
        # The spec path is the last arg; it must be valid JSON on disk.
        spec_path = cmd[-1]
        with open(spec_path, encoding="utf-8") as fh:
            seen["spec"] = json.load(fh)
        seen["cmd"] = cmd
        return _Proc(0)

    monkeypatch.setattr(subprocess_runner, "find_conda_exe", lambda: "conda")
    monkeypatch.setattr(subprocess_runner.subprocess, "run", fake_run)

    spec = {"mode": "ortho", "jobs": [{"path": "a.tif", "out_csv": "a.csv"}],
            "canonical_columns": list(CANONICAL_COLUMNS)}
    proc = subprocess_runner.run_backend("pebble-stub", "/x/run.py", spec)
    assert proc.returncode == 0
    assert seen["spec"]["mode"] == "ortho"
    assert seen["spec"]["canonical_columns"] == list(CANONICAL_COLUMNS)
    assert seen["cmd"][3] == "pebble-stub"

    # Non-zero exit must raise with the captured stderr tail.
    def fail_run(cmd, capture_output, text, timeout):
        return _Proc(1, err="boom")
    monkeypatch.setattr(subprocess_runner.subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="boom"):
        subprocess_runner.run_backend("pebble-stub", "/x/run.py", spec)


# --------------------------------------------------------------------------- #
#  Streaming launch: lines reach log_fn, Stop kills the tree                  #
# --------------------------------------------------------------------------- #
class _FakePopen:
    """Enough of Popen for _run_streaming: an iterable stdout, poll/wait, pid."""
    instances = []

    def __init__(self, cmd, lines, exit_code=0, hang=False):
        self.args, self.pid = cmd, 4242
        self._lines, self._exit, self._hang = lines, exit_code, hang
        self.returncode = None
        self.killed = False
        self._polls = 0
        _FakePopen.instances.append(self)

    @property
    def stdout(self):
        for ln in self._lines:
            yield ln + "\n"

    def poll(self):
        self._polls += 1
        if self.killed:
            self.returncode = -9
        elif not self._hang and self._polls > 1:
            self.returncode = self._exit
        return self.returncode

    def wait(self):
        return self.poll()


def _patch_popen(monkeypatch, **kw):
    _FakePopen.instances.clear()
    monkeypatch.setattr(subprocess_runner, "find_conda_exe", lambda: "conda")
    monkeypatch.setattr(subprocess_runner, "_STOP_POLL_S", 0.005)
    monkeypatch.setattr(subprocess_runner.subprocess, "Popen",
                        lambda cmd, **_: _FakePopen(cmd, **kw))


def test_run_backend_stream_forwards_the_child_output_line_by_line(monkeypatch):
    _patch_popen(monkeypatch, lines=["[model] loading", "[model] tile 1/4", ""])
    log = []
    proc = subprocess_runner.run_backend(
        "pm-x", "/x/run.py", {"jobs": [{"path": "a"}]}, log_fn=log.append,
        stream=True)
    assert proc.returncode == 0
    assert log[0].startswith("[subprocess] pm-x: run.py (1 job(s))")
    assert log[1:] == ["[model] loading", "[model] tile 1/4"]  # blanks dropped


def test_run_backend_stream_raises_with_the_tail_on_failure(monkeypatch):
    _patch_popen(monkeypatch, lines=["Traceback", "ValueError: bad weights"],
                 exit_code=3)
    with pytest.raises(RuntimeError, match=r"(?s)exit 3.*bad weights") as ei:
        subprocess_runner.run_backend("pm-x", "/x/run.py", {"jobs": []},
                                      stream=True)
    assert "Traceback" in str(ei.value)


def test_run_backend_stream_kills_the_tree_and_reports_stopped(monkeypatch):
    _patch_popen(monkeypatch, lines=["[model] working"], hang=True)
    killed = []

    def fake_kill(proc):
        killed.append(proc.pid)
        proc.killed = True
    monkeypatch.setattr(subprocess_runner, "kill_process_tree", fake_kill)
    flag = {"stop": False}
    log = []

    def stop_check():
        flag["stop"] = True                       # stop on the first poll
        return flag["stop"]

    with pytest.raises(subprocess_runner.BackendStopped):
        subprocess_runner.run_backend("pm-x", "/x/run.py", {"jobs": []},
                                      stream=True, stop_check=stop_check,
                                      log_fn=log.append)
    assert killed == [4242]
    assert any("stop requested" in ln for ln in log)


def test_kill_process_tree_uses_taskkill_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess_runner.sys, "platform", "win32")
    monkeypatch.setattr(subprocess_runner.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd))

    class _P:
        pid = 77
    subprocess_runner.kill_process_tree(_P())
    assert calls == [["taskkill", "/F", "/T", "/PID", "77"]]


# --------------------------------------------------------------------------- #
#  The instances file: written by the subprocess, read by the core           #
# --------------------------------------------------------------------------- #
def test_stub_axes_scale_with_cropsize():
    # major axis = metric_cropsize x 0.10 for the first clast (0.25 m at
    # crop=2.5) -> proves the detection parameters reach the subprocess.
    axes = stub_run.synthetic_axes_m({"metric_cropsize": 2.5})
    assert len(axes) == 3
    assert abs(axes[0][0] - 0.25) < 1e-9
    assert abs(axes[0][1] - 0.15) < 1e-9


def test_stub_run_main_writes_an_instances_file_numpy_can_read(tmp_path):
    # Exercise run.py's REAL path end to end (no conda env needed): the
    # standard-library .npz it writes must load with numpy and carry one
    # score per label.
    out_csv = tmp_path / "demo_ws2.5m.csv"
    npz = subprocess_runner.instances_path_for(out_csv)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "mode": "ortho", "params": {"metric_cropsize": 2.5, "resolution": 0.004},
        "canonical_columns": list(CANONICAL_COLUMNS),
        "jobs": [{"path": "x.tif", "kstart": 0, "out_csv": str(out_csv),
                  "instances_path": str(npz)}],
    }), encoding="utf-8")

    assert stub_run.main(["--spec", str(spec)]) == 0
    assert npz.exists()
    with np.load(npz) as z:
        labels, scores, shape = z["labels"], z["scores"], z["shape"]
    assert labels.dtype == np.int32 and labels.ndim == 2
    assert scores.dtype == np.float32 and scores.shape == (3,)
    assert tuple(shape) == labels.shape
    assert int(labels.max()) == 3
    assert not out_csv.exists()          # the subprocess does not measure


def test_read_instances_npz_crops_each_label_to_its_box(tmp_path):
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[2:8, 3:9] = 1                 # a 6x6 square
    labels[10:12, 20:28] = 2             # a 2x8 bar
    labels[5, 5] = 0                     # a hole: filled on read
    p = tmp_path / "x.csv.instances.npz"
    np.savez(p, labels=labels, scores=np.array([0.9, 0.5], dtype=np.float32),
             shape=np.array(labels.shape))
    inst = subprocess_runner.read_instances_npz(p, pad=1)
    assert [i.score for i in inst] == pytest.approx([0.9, 0.5])
    a, b = inst
    assert (a.y0, a.x0) == (1, 2) and a.mask.shape == (8, 8)
    assert a.mask[4, 4]                  # the hole is filled
    assert a.mask.sum() == 36
    assert (b.y0, b.x0) == (9, 19) and b.mask.shape == (4, 10)


def test_read_instances_npz_rejects_missing_and_malformed_files(tmp_path):
    with pytest.raises(RuntimeError, match="did not write"):
        subprocess_runner.read_instances_npz(tmp_path / "nope.npz")
    bad = tmp_path / "bad.npz"
    np.savez(bad, labels=np.array([[1, 2]], dtype=np.int32),
             scores=np.array([0.5], dtype=np.float32))
    with pytest.raises(RuntimeError, match="only 1 score"):
        subprocess_runner.read_instances_npz(bad)
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    with pytest.raises(RuntimeError, match="Unreadable"):
        subprocess_runner.read_instances_npz(tmp_path / "junk.npz")
    empty = tmp_path / "empty.npz"
    np.savez(empty, labels=np.zeros((4, 4), dtype=np.int32),
             scores=np.zeros(0, dtype=np.float32))
    assert subprocess_runner.read_instances_npz(empty) == []


def test_measure_instances_uses_the_shared_measurement(tmp_path):
    from detectors import measure
    labels = np.zeros((60, 80), dtype=np.int32)
    yy, xx = np.mgrid[:60, :80]
    labels[((yy - 30) / 10.0) ** 2 + ((xx - 40) / 20.0) ** 2 <= 1.0] = 1
    p = tmp_path / "e.csv.instances.npz"
    np.savez(p, labels=labels, scores=np.array([0.7], dtype=np.float32))
    inst = subprocess_runner.read_instances_npz(p)
    df = measure.measure_instances(inst, 0.01, height=60)
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["Ellipse_major_axis"] == pytest.approx(0.40, rel=0.05)
    assert r["Ellipse_minor_axis"] == pytest.approx(0.20, rel=0.05)
    assert r["x"] == pytest.approx(40, abs=1) and r["y"] == pytest.approx(30, abs=1)
    assert r["Score"] == pytest.approx(0.7)
    # An ortho geotransform maps the same centroid into the CRS.
    df2 = measure.measure_instances(inst, 0.01, geotransform=(500000, 0.01, 0,
                                                              5400000, 0, -0.01))
    assert df2.iloc[0]["x"] == pytest.approx(500000.40, abs=0.02)
    assert df2.iloc[0]["y"] == pytest.approx(5399999.70, abs=0.02)


def test_read_canonical_csv_enforces_schema(tmp_path):
    # The read-back boundary for a subprocess that writes the CSV itself must
    # REJECT a non-canonical header and a missing file.
    good = tmp_path / "good.csv"
    good.write_text(",".join(CANONICAL_COLUMNS) + "\n", encoding="utf-8")
    df = subprocess_runner.read_canonical_csv(good, CANONICAL_COLUMNS)
    assert list(df.columns) == list(CANONICAL_COLUMNS) and len(df) == 0

    bad = tmp_path / "bad.csv"
    bad.write_text("clast_ID,x,y\n1,0,0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-canonical schema"):
        subprocess_runner.read_canonical_csv(bad, CANONICAL_COLUMNS)

    with pytest.raises(RuntimeError, match="did not write"):
        subprocess_runner.read_canonical_csv(tmp_path / "nope.csv",
                                             CANONICAL_COLUMNS)


# --------------------------------------------------------------------------- #
#  The stub driver, with the subprocess replaced by an in-process call        #
# --------------------------------------------------------------------------- #
def _in_process_run_backend(env, script, spec, **kwargs):
    """Stand-in for run_backend: run the stub's main() here and now."""
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    try:
        assert stub_run.main(["--spec", path]) == 0
    finally:
        os.remove(path)
    if kwargs.get("log_fn"):
        kwargs["log_fn"]("[stub] (in-process stand-in)")


@pytest.mark.parametrize("mode,kw,csv_name", [
    (modes.ORTHO, {"metric_cropsize": 2.5, "resolution": 0.004}, "demo_ws2.5m.csv"),
    (modes.QUADRAT, {"resolution": 0.001}, "demo_individual_clasts.csv"),
])
def test_stub_driver_measures_and_names_the_csv_per_mode(
        monkeypatch, tmp_path, mode, kw, csv_name):
    import pandas as pd
    monkeypatch.setattr(subprocess_runner, "run_backend", _in_process_run_backend)
    events, log = [], []
    stub = registry.get_backend("stub")
    out = stub.detect_jobs(
        mode, [{"path": str(tmp_path / "demo.tif"), "kstart": 0}],
        output_dir=str(tmp_path), log_fn=log.append,
        progress_callback=lambda ji, ev, p: events.append((ji, ev)), **kw)
    assert len(out) == 1 and len(out[0]) == 3
    assert list(out[0].columns) == list(CANONICAL_COLUMNS)
    csv = tmp_path / csv_name
    assert csv.exists()
    assert list(pd.read_csv(csv).columns) == list(CANONICAL_COLUMNS)
    mf = json.loads((tmp_path / (csv_name + ".manifest.json")).read_text())
    assert mf["model"] == "stub"
    assert events == [(0, "start"), (0, "done")]
    # Sizes come from the shared measurement of the emitted geometry.
    major = out[0]["Ellipse_major_axis"].iloc[0]
    assert major == pytest.approx(kw.get("metric_cropsize", 1.0) * 0.10, rel=0.03)


def test_stub_driver_reports_stopped_when_the_launch_is_interrupted(
        monkeypatch, tmp_path):
    def stopped(*a, **k):
        raise subprocess_runner.BackendStopped("pebble-stub")
    monkeypatch.setattr(subprocess_runner, "run_backend", stopped)
    events = []
    stub = registry.get_backend("stub")
    out = stub.detect_jobs(
        modes.ORTHO, [{"path": str(tmp_path / "a.tif")},
                      {"path": str(tmp_path / "b.tif")}],
        output_dir=str(tmp_path), metric_cropsize=1.0,
        progress_callback=lambda ji, ev, p: events.append((ji, ev)))
    assert [len(df) for df in out] == [0, 0]
    assert events == [(0, "start"), (1, "start"), (0, "stopped"), (1, "stopped")]


@pytest.mark.skipif(not registry.get_backend("stub").is_available(),
                    reason="pebble-stub conda env not installed")
def test_stub_backend_cross_env_roundtrip(tmp_path):
    import pandas as pd
    subprocess_runner.clear_env_cache()
    stub = registry.get_backend("stub")
    jobs = [{"path": str(tmp_path / "demo.tif"), "kstart": 0}]
    log = []
    out = stub.detect_jobs(modes.ORTHO, jobs, resolution=0.004,
                           metric_cropsize=2.5, overlap=0.0,
                           min_confidence=0.8, output_dir=str(tmp_path),
                           log_fn=log.append)
    assert len(out) == 1
    df = out[0]
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert len(df) == 3
    csv = tmp_path / "demo_ws2.5m.csv"
    assert csv.exists()
    assert (tmp_path / "demo_ws2.5m.csv.manifest.json").exists()
    assert subprocess_runner.instances_path_for(csv).exists()
    # Round-trips through pandas with the canonical schema intact.
    assert list(pd.read_csv(csv).columns) == list(CANONICAL_COLUMNS)
    assert df["Ellipse_major_axis"].iloc[0] == pytest.approx(0.25, rel=0.03)
    # The child's own output was streamed into the log.
    assert any(ln.startswith("[stub] job 1/1") for ln in log)
