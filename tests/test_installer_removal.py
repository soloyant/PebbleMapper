"""030 FR-040, FR-041, SC-007: uninstall removes only what the app made,
never through a link, and reports the space it actually freed."""
import os
import subprocess
import sys

import pytest

from installer import jobs, registration, removal
from installer.errors import InstallError


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))


def _installed(tmp_path, size=10_000):
    models = tmp_path / "models"
    root = models / "m-100-g"
    for sub in ("env/conda-meta", "code/pkg", "weights"):
        (root / sub).mkdir(parents=True)
    (root / "weights" / "w.bin").write_bytes(b"x" * size)
    (root / "code" / "pkg" / "a.py").write_text("x = 1\n")
    registration.register({"id": "m", "release": "1.0.0", "variant": "gpu",
                           "root": str(root), "env": str(root / "env"),
                           "code": str(root / "code"), "weights": str(root / "weights"),
                           "backends": [], "files": []})
    return models, root


def test_uninstall_frees_what_it_measured_and_unregisters(tmp_path):
    models, root = _installed(tmp_path)
    before = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    freed = removal.uninstall("m", "1.0.0", "gpu")
    assert freed == before
    assert not root.exists()
    assert models.exists(), "the models folder itself stays"
    assert registration.read_installed() == []


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_a_junction_inside_is_unlinked_not_followed(tmp_path):
    models, root = _installed(tmp_path)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("field data")
    subprocess.run(["cmd", "/c", "mklink", "/J", str(root / "code" / "link"), str(outside)],
                   check=True, capture_output=True)
    removal.uninstall("m", "1.0.0", "gpu")
    assert (outside / "keep.txt").read_text() == "field data"
    assert not root.exists()


def test_never_outside_the_given_folder(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(InstallError):
        removal.safe_rmtree(other, tmp_path / "models")
    with pytest.raises(InstallError):
        removal.safe_rmtree(tmp_path / "models", tmp_path / "models")
    assert other.exists()


def test_a_model_not_registered_is_not_uninstalled(tmp_path):
    with pytest.raises(InstallError):
        removal.uninstall("nope", "1.0.0", "gpu")


def test_removing_an_interrupted_job_deletes_its_partial_folder_only(tmp_path):
    models = tmp_path / "models"
    job = jobs.new_job("install", {"model": "m", "release": "2.0.0", "variant": "gpu",
                                   "models_dir": str(models)})
    part = models / "m-200-g" / "env"
    part.mkdir(parents=True)
    (models / ".work" / job["id"]).mkdir(parents=True)
    _, kept_root = _installed(tmp_path)                 # another, registered model
    removal.remove_job(job)
    assert not (models / "m-200-g").exists()
    assert not (models / ".work" / job["id"]).exists()
    assert kept_root.exists()
    assert jobs.read_job(job["id"]) is None
