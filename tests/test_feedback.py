"""The diagnostic bundle: versions, configuration, logs and file names only."""
import json
import zipfile
from pathlib import Path


def test_bundle_holds_versions_logs_and_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "crash_20260910T000000.log").write_text("# trace", encoding="utf-8")
    (home / "config.json").write_text(json.dumps({"datasets_root": str(Path.home() / "data")}),
                                      encoding="utf-8")
    project = tmp_path / "Site"
    (project / "output_results" / "vectors").mkdir(parents=True)
    (project / "output_results" / "vectors" / "img_ws1m.csv").write_text(
        "x,y,Clast_length\n1,2,0.05\n", encoding="utf-8")
    (project / "output_results" / "vectors" / "img_ws1m_detection_log.txt").write_text(
        "[Run 2026]\n  outcome : complete\n", encoding="utf-8")

    from functions import feedback
    out = feedback.build_bundle(project, note="the map is blank")
    assert out.exists() and out.parent == home / "feedback"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {"README.txt", "note.txt", "versions.json", "summary.txt",
                "app/config.json", "app/logs/crash_20260910T000000.log",
                "project/inventory.txt", "project/logs/img_ws1m_detection_log.txt"} <= names
        # Measurements are never included; user paths are shortened.
        assert not any(n.endswith(".csv") for n in names)
        cfg = zf.read("app/config.json").decode("utf-8")
        assert str(Path.home()) not in cfg and "~" in cfg
        assert zf.read("note.txt").decode("utf-8") == "the map is blank"
        versions = json.loads(zf.read("versions.json"))
        assert "python" in versions and "numpy" in versions
        # Whether an iPhone photograph opens on this machine, and why not.
        assert "pillow_heif" in versions
        assert isinstance(versions["heif_available"], bool)
        assert isinstance(versions["heif_error"], str)


def test_bundle_without_a_project(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    from functions import feedback
    out = feedback.build_bundle(None, note="")
    with zipfile.ZipFile(out) as zf:
        assert "project/inventory.txt" not in zf.namelist()
        assert zf.read("note.txt").decode("utf-8") == "(no note)"


def test_two_manifests_of_the_same_name_both_reach_the_bundle(tmp_path, monkeypatch):
    """Manifests were stored under their bare name, so two runs in two folders
    collided and zipfile warned about a duplicate entry."""
    import zipfile
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    from functions import feedback
    project = tmp_path / "Site"
    for sub in ("vectors", "vectors/checkpoints"):
        (project / "output_results" / sub).mkdir(parents=True, exist_ok=True)
        (project / "output_results" / sub / "img.csv.manifest.json").write_text(
            '{"model": "maskrcnn"}', encoding="utf-8")
    out = feedback.build_bundle(project, note="two manifests")
    with zipfile.ZipFile(out) as zf:
        entries = [n for n in zf.namelist() if "manifests/" in n]
    assert len(entries) == 2 and len(set(entries)) == 2
