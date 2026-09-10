"""A manifest may only describe the CSV beside it.

Unit half: ``write_for_completed_csv`` refuses absent and stale CSVs and
writes for fresh ones. Sweep half (runs only where ``datasets/`` exists —
the rig): no orphaned manifests, and no manifest older than its CSV.
"""
import os
import time
from pathlib import Path

import pytest

from detectors.base import DetectionManifest

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "datasets"


def _manifest():
    return DetectionManifest(model="test", params={"resolution": 0.003})


def test_refuses_when_csv_absent(tmp_path):
    out = _manifest().write_for_completed_csv(tmp_path / "never.csv",
                                              time.time() - 10)
    assert out is None
    assert not (tmp_path / "never.csv.manifest.json").exists()


def test_refuses_when_csv_predates_the_run(tmp_path):
    csv = tmp_path / "old.csv"
    csv.write_text("clast_ID,x,y\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(csv, (old, old))
    out = _manifest().write_for_completed_csv(csv, time.time() - 10)
    assert out is None
    assert not (tmp_path / "old.csv.manifest.json").exists()


def test_writes_for_a_csv_this_run_produced(tmp_path):
    run_started = time.time() - 5
    csv = tmp_path / "fresh.csv"
    csv.write_text("clast_ID,x,y\n", encoding="utf-8")
    out = _manifest().write_for_completed_csv(csv, run_started)
    assert out is not None and out.exists()
    assert out.name == "fresh.csv.manifest.json"


@pytest.mark.skipif(not DATASETS.exists(),
                    reason="datasets/ not present (rig-only sweep)")
def test_no_orphaned_or_backdated_manifests_in_datasets():
    """The on-disk invariant over whatever projects are present."""
    bad = []
    for mf in DATASETS.rglob("*.manifest.json"):
        csv = Path(str(mf)[: -len(".manifest.json")])
        if not csv.exists():
            bad.append(f"orphan: {mf.relative_to(DATASETS)}")
        elif mf.stat().st_mtime < csv.stat().st_mtime - 2.0:
            bad.append(f"older than its CSV: {mf.relative_to(DATASETS)}")
    assert not bad, ("manifests violating the pairing invariant:\n  "
                     + "\n  ".join(bad))
