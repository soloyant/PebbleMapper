"""030 FR-043, SC-010: a broken model is shown as Needs repair, with why."""
import hashlib
import shutil

import pytest

from installer import health, registration


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))


def _entry(tmp_path):
    root = tmp_path / "models" / "m-100-g"
    (root / "env" / "conda-meta").mkdir(parents=True)
    (root / "env" / "python.exe").write_bytes(b"")
    (root / "code" / "pkg").mkdir(parents=True)
    (root / "code" / "pkg" / "__init__.py").write_text("")
    (root / "weights").mkdir()
    data = b"w" * 100
    (root / "weights" / "w.bin").write_bytes(data)
    e = {"id": "m", "release": "1.0.0", "variant": "gpu", "root": str(root),
         "env": str(root / "env"), "code": str(root / "code"),
         "weights": str(root / "weights"), "runs_in": "own",
         "backends": [{"name": "m", "package": "pkg", "module": "backend",
                       "factory": "make_backend"}],
         "files": [{"path": "weights/w.bin", "size": 100,
                    "sha256": hashlib.sha256(data).hexdigest()}]}
    registration.register(e)
    return e, root


def test_a_whole_model_passes(tmp_path):
    e, _ = _entry(tmp_path)
    assert health.quick_check(e) is None
    assert health.full_check(e) == []
    assert health.check_all() == []


@pytest.mark.parametrize("breakage,words", [
    (lambda r: shutil.rmtree(r / "env"), "environment"),
    (lambda r: shutil.rmtree(r / "code" / "pkg"), "program files"),
    (lambda r: (r / "weights" / "w.bin").unlink(), "missing"),
    (lambda r: (r / "weights" / "w.bin").write_bytes(b"short"), "changed"),
])
def test_breakage_is_named_and_recorded(tmp_path, breakage, words):
    e, root = _entry(tmp_path)
    breakage(root)
    reason = health.quick_check(e)
    assert reason and words in reason
    [b] = health.check_all()
    assert b["state"] == "needs_repair"
    assert registration.read_installed()[0]["state"] == "needs_repair"


def test_full_check_catches_same_size_damage(tmp_path):
    e, root = _entry(tmp_path)
    (root / "weights" / "w.bin").write_bytes(b"x" * 100)
    assert health.quick_check(e) is None, "same size: the quick check cannot see it"
    assert health.full_check(e) == ["weights/w.bin"]


def test_a_repaired_model_returns_to_ok(tmp_path):
    e, root = _entry(tmp_path)
    shutil.rmtree(root / "env")
    health.check_all()
    (root / "env" / "conda-meta").mkdir(parents=True)
    (root / "env" / "python.exe").write_bytes(b"")
    assert health.check_all() == []
    assert registration.read_installed()[0]["state"] == "ok"
