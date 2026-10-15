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


def test_bundle_holds_the_installed_models_and_the_last_failed_install(tmp_path, monkeypatch):
    import zipfile
    from functions import feedback
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)
    (home / "models" / "installed.json").write_text('{"format": 1, "models": []}',
                                                    encoding="utf-8")
    jd = home / "jobs" / "20261014-000000-abcdef"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({"state": "failed"}), encoding="utf-8")
    (jd / "job.log").write_text(f"error under {Path.home() / 'x'}", encoding="utf-8")
    ok = home / "jobs" / "20261014-000001-123456"
    ok.mkdir()
    (ok / "job.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    out = feedback.build_bundle(None, note="")
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "models/installed.json" in names
        assert "models/last-failed-job/job.log" in names
        assert str(Path.home()) not in zf.read("models/last-failed-job/job.log").decode()


def test_bundle_holds_the_app_update_state(tmp_path, monkeypatch):
    """031 T026: the starter's pointer, the file list, the switch journal and
    the last update check."""
    import zipfile
    from functions import feedback
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "updates.json").write_text('{"last_result": "ok"}', encoding="utf-8")
    app = tmp_path / "app"
    (app / ".pm-update").mkdir(parents=True)
    (app / ".pm-version.json").write_text('{"current": "1.1.0"}', encoding="utf-8")
    (app / ".pm-files.json").write_text('{"files": []}', encoding="utf-8")
    (app / ".pm-update" / "journal.json").write_text('{"journal": []}', encoding="utf-8")
    monkeypatch.setattr(feedback, "APP_DIR", app)
    out = feedback.build_bundle(None, note="")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert {"app/pm-version.json", "app/pm-files.json", "app/pm-update-journal.json",
            "app/updates.json"} <= names
