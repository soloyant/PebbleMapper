"""functions/worker.py — the Tier-1 worker protocol.

Every test exercises the REAL ``python -m functions.worker`` entry through
``run_job`` (or a hand-built envelope for the version gate) — the
fixtures-must-match-the-writer rule. Inputs are synthetic; no datasets/
dependency.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from functions import worker

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


def _synthetic_inputs(tmp_path):
    """A tiny detection CSV + one square zone that contains its points."""
    csv = tmp_path / "clasts.csv"
    lines = ["clast_ID,x,y,Clast_length"]
    for i in range(12):
        lines.append(f"{i + 1},{1 + (i % 4)},{1 + (i // 4)},{0.05 + 0.01 * i:.3f}")
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    vec = tmp_path / "zones.geojson"
    vec.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Z1"},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [10, 0], [10, 10],
                                          [0, 10], [0, 0]]]},
        }],
    }), encoding="utf-8")
    return csv, vec


def test_ok_true_runs_real_computation(home, tmp_path):
    csv, vec = _synthetic_inputs(tmp_path)
    out = tmp_path / "out.csv"
    lines = []
    got = worker.run_job("zonal_polygons",
                         {"detection_csv": str(csv), "vector_path": str(vec),
                          "out_csv": str(out), "field_name": "Clast_length",
                          "id_field": "name"},
                         log_cb=lines.append)
    assert got.ok and not got.died
    assert out.exists()
    assert got.summary["n_rows"] == 1
    assert got.summary["rows"][0]["count"] == 12
    assert str(out) in got.outputs
    assert any("polygon row" in l for l in lines)  # log streamed to the parent


def test_ok_false_surfaces_one_line_error(home, tmp_path):
    got = worker.run_job("zonal_polygons",
                         {"detection_csv": str(tmp_path / "missing.csv"),
                          "vector_path": str(tmp_path / "missing.geojson"),
                          "out_csv": str(tmp_path / "o.csv")})
    assert not got.ok and not got.died
    assert got.error_kind == "computation"
    assert got.error and "\n" not in got.error
    assert got.log_path and Path(got.log_path).exists()  # full detail kept


def test_unknown_op_refused_by_name(home):
    got = worker.run_job("definitely_not_an_op", {})
    assert not got.ok and not got.died
    assert got.error_kind == "unknown-op"
    assert "definitely_not_an_op" in got.error


def test_worker_death_reports_exit_and_postmortem(home, tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_ENABLE_TEST_FAULTS", "1")
    csv, vec = _synthetic_inputs(tmp_path)
    got = worker.run_job("zonal_polygons",
                         {"detection_csv": str(csv), "vector_path": str(vec),
                          "out_csv": str(tmp_path / "o.csv"),
                          "_test_fault": "hard-exit"})
    assert got.died and not got.ok
    assert got.exit_code == 0xAD
    assert got.error_kind == "worker-died"
    assert "0x000000AD" in got.error
    assert got.postmortem_path and Path(got.postmortem_path).exists()
    text = Path(got.postmortem_path).read_text(encoding="utf-8")
    assert "worker post-mortem" in text and "zonal_polygons" in text
    # The job dir (envelope + log) survives a death for forensics.
    assert got.log_path and Path(got.log_path).exists()


def test_native_fault_in_worker_confined_and_recorded(home, tmp_path, monkeypatch):
    """A real access violation in the worker: parent unharmed, death decoded."""
    monkeypatch.setenv("PEBBLEMAPPER_ENABLE_TEST_FAULTS", "1")
    csv, vec = _synthetic_inputs(tmp_path)
    got = worker.run_job("zonal_polygons",
                         {"detection_csv": str(csv), "vector_path": str(vec),
                          "out_csv": str(tmp_path / "o.csv"),
                          "_test_fault": "segv"})
    assert got.died and got.exit_code not in (0, None)
    assert got.postmortem_path is not None
    # faulthandler in the worker got a chance to name the dying stack.
    log_text = Path(got.log_path).read_text(encoding="utf-8", errors="replace")
    assert "inducing test fault: segv" in log_text


def test_timeout_terminates_and_says_so(home, tmp_path):
    csv, vec = _synthetic_inputs(tmp_path)
    got = worker.run_job("zonal_polygons",
                         {"detection_csv": str(csv), "vector_path": str(vec),
                          "out_csv": str(tmp_path / "o.csv")},
                         timeout=0.4)
    assert got.died and not got.ok
    assert "did not finish within" in got.error


def test_envelope_version_gate(home, tmp_path):
    """A future-version envelope is refused by name, not guessed at."""
    env = {"version": 99, "op": "merge", "args": {},
           "result_path": str(tmp_path / "r.json"),
           "log_path": str(tmp_path / "l.txt")}
    ep = tmp_path / "e.json"
    ep.write_text(json.dumps(env), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "functions.worker", str(ep)],
                          cwd=str(REPO), capture_output=True, timeout=120)
    assert proc.returncode == 0
    result = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["error_kind"] == "envelope-version"
    assert "99" in result["error"]
