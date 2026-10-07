# tests/test_express.py
import math
from pathlib import Path
import pytest
from functions import express


def test_resolve_window_sizes_are_fixed_per_preset():
    ext = {"width_m": 90, "height_m": 90, "gsd_m": 0.01}
    assert express.resolve_window_sizes(ext, "fast") == [5.0]
    assert express.resolve_window_sizes(ext, "medium") == [2.5, 5.0]
    assert express.resolve_window_sizes(ext, "slow") == [1.0, 2.5, 5.0]

def test_resolve_window_sizes_independent_of_extent():
    small = {"width_m": 20, "height_m": 20, "gsd_m": 0.01}
    large = {"width_m": 400, "height_m": 400, "gsd_m": 0.002}
    for preset in ("fast", "medium", "slow"):
        assert (express.resolve_window_sizes(small, preset)
                == express.resolve_window_sizes(large, preset))

def test_presets_registry_has_three_named_presets():
    assert set(express.PRESETS) == {"fast", "medium", "slow"}


def _make_tif(tmp_path, gt, nx=40, ny=30, proj_epsg=32611):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import osr
    p = str(tmp_path / "o.tif")
    ds = gdal.GetDriverByName("GTiff").Create(p, nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    if proj_epsg:
        sr = osr.SpatialReference(); sr.ImportFromEPSG(proj_epsg)
        ds.SetProjection(sr.ExportToWkt())
    ds.GetRasterBand(1).Fill(100)
    ds = None
    return p

def test_read_ortho_extent_georeferenced(tmp_path):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01))  # 1cm px
    ext = express.read_ortho_extent(p)
    assert ext["is_georeferenced"] is True
    assert ext["gsd_m"] == pytest.approx(0.01)
    assert ext["width_m"] == pytest.approx(40 * 0.01)
    assert ext["height_m"] == pytest.approx(30 * 0.01)

def test_read_ortho_extent_identity_is_not_georeferenced(tmp_path):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (0.0, 1.0, 0.0, 0.0, 0.0, 1.0), proj_epsg=None)
    ext = express.read_ortho_extent(p)
    assert ext["is_georeferenced"] is False
    assert ext["gsd_m"] is None


def test_resolve_plan_fast(tmp_path):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01), nx=90, ny=90)
    plan = express.resolve_plan(p, "fast")
    assert plan.preset == "fast"
    assert plan.merge is False
    assert len(plan.windows) == 1
    assert plan.overlap == 0.0 and plan.min_confidence == 0.80
    # Fast = exactly one D50 raster, one vector map, no basemap
    assert [(r["field"], r["parameter"], r["percentile"]) for r in plan.rasters] == \
           [("Clast_length", "quantile", 0.5)]
    assert len(plan.maps) == 1 and plan.maps[0]["basemap"] is None
    # zonal/validation always off
    assert plan.report_options["include_zonal_statistics"] is False
    assert plan.report_options["include_validation"] is False
    assert plan.report_options["include_detection"] is True

def test_resolve_plan_slow_is_richer_than_fast(tmp_path):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01), nx=90, ny=90)
    fast = express.resolve_plan(p, "fast")
    slow = express.resolve_plan(p, "slow")
    assert slow.merge is True
    assert len(slow.rasters) > len(fast.rasters)
    assert len(slow.maps) > len(fast.maps)
    assert any(m["basemap"] for m in slow.maps)

def test_resolve_plan_rejects_unreferenced_photo(tmp_path):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (0.0, 1.0, 0.0, 0.0, 0.0, 1.0), proj_epsg=None)
    with pytest.raises(ValueError, match="georeferenced"):
        express.resolve_plan(p, "fast")


def test_run_express_pipeline_orchestration(tmp_path, monkeypatch):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01), nx=90, ny=90)
    calls = {"detect": [], "merge": 0, "raster": 0, "map": 0, "report": 0}

    def fake_detect(project, ortho, window, plan, *, log_fn, progress_cb, stop_check):
        calls["detect"].append(window)
        out = Path(tmp_path) / f"det_w{window}.csv"; out.write_text("clast_ID,x,y\n1,0,0\n")
        return out
    def fake_merge(csvs, out): calls["merge"] += 1; Path(out).write_text("clast_ID\n"); return Path(out)
    def fake_raster(ortho, csv, out, spec): calls["raster"] += 1; Path(out).write_text("tif"); return Path(out)
    def fake_map(ortho, src, out, spec): calls["map"] += 1; Path(out).write_text("png"); return Path(out)
    def fake_report(project, out, opts, log_fn): calls["report"] += 1; Path(out).write_text("pdf"); return Path(out)

    monkeypatch.setattr(express, "_stage_detect", fake_detect)
    monkeypatch.setattr(express, "_stage_merge", fake_merge)
    monkeypatch.setattr(express, "_stage_rasterize", fake_raster)
    monkeypatch.setattr(express, "_stage_map", fake_map)
    monkeypatch.setattr(express, "_stage_report", fake_report)

    res = express.run_express_pipeline(str(tmp_path), p, "medium")
    assert calls["detect"] == [2.5, 5.0]          # one detect per window (medium preset)
    assert calls["merge"] == 1                     # medium merges
    assert calls["raster"] == 3 and calls["map"] == 2
    assert calls["report"] == 1
    assert res.report_pdf is not None
    assert all(s.status == "done" for s in res.stages)

def test_run_express_pipeline_best_effort_continues_on_merge_failure(tmp_path, monkeypatch):
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01), nx=90, ny=90)
    def _fake_detect(project, ortho, window, plan, **k):
        out = Path(tmp_path) / f"d{window}.csv"
        out.write_text("x")
        return out
    monkeypatch.setattr(express, "_stage_detect", _fake_detect)
    def boom(*a, **k): raise RuntimeError("merge exploded")
    monkeypatch.setattr(express, "_stage_merge", boom)
    monkeypatch.setattr(express, "_stage_rasterize", lambda *a, **k: Path(tmp_path, "r.tif"))
    monkeypatch.setattr(express, "_stage_map", lambda *a, **k: Path(tmp_path, "m.png"))
    monkeypatch.setattr(express, "_stage_report", lambda *a, **k: Path(tmp_path, "r.pdf"))
    res = express.run_express_pipeline(str(tmp_path), p, "medium")
    statuses = {s.name: s.status for s in res.stages}
    assert statuses["Merge"] == "error"            # failed stage flagged
    assert statuses["Rasterize"] == "done"         # downstream still ran
    assert statuses["Report"] == "done"


def test_express_tab_registered_and_wired():
    src = (Path(__file__).resolve().parent.parent
           / "gui" / "app.py").read_text(encoding="utf-8")
    assert "def build_express_tab" in src
    assert 'ui.tab("Express"' in src
    assert 'with ui.tab_panel("Express")' in src
    assert "run_express_pipeline" in src   # the tab calls the backend orchestrator

def test_app_imports_with_express():
    import gui.app  # must import without raising
    assert hasattr(gui.app, "build_express_tab")


def test_run_express_pipeline_halts_on_stop(tmp_path, monkeypatch):
    """Stop during detection: the window that was running is partial, so
    nothing is merged, rasterised, mapped or reported from it (found by the
: the report was built from a 27-clast partial CSV)."""
    pytest.importorskip("osgeo.gdal")
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01), nx=90, ny=90)
    calls = {"detect": [], "merge": 0, "raster": 0, "map": 0, "report": 0}
    stopped = {"flag": False}

    def fake_detect(project, ortho, window, plan, *, log_fn, progress_cb, stop_check):
        calls["detect"].append(window)
        out = Path(tmp_path) / f"det_w{window}.csv"
        out.write_text("clast_ID,x,y\n1,0,0\n")
        stopped["flag"] = True                 # the user pressed Stop mid-window
        return out
    monkeypatch.setattr(express, "_stage_detect", fake_detect)
    monkeypatch.setattr(express, "_stage_merge", lambda *a, **k: calls.__setitem__("merge", 1))
    monkeypatch.setattr(express, "_stage_rasterize", lambda *a, **k: calls.__setitem__("raster", 1))
    monkeypatch.setattr(express, "_stage_map", lambda *a, **k: calls.__setitem__("map", 1))
    monkeypatch.setattr(express, "_stage_report", lambda *a, **k: calls.__setitem__("report", 1))
    lines = []
    res = express.run_express_pipeline(str(tmp_path), p, "medium",
                                       stop_check=lambda: stopped["flag"], log_fn=lines.append)
    assert calls["detect"] == [2.5]            # the second window never started
    assert calls["merge"] == calls["raster"] == calls["map"] == calls["report"] == 0
    statuses = {s.name: s.status for s in res.stages}
    assert statuses["Detection"] == "stopped"
    assert all(statuses[n] == "skipped" for n in ("Merge", "Rasterize", "Map", "Report"))
    assert res.report_pdf is None
    assert any("stopped" in l and "partial" in l for l in lines)


def test_a_rerun_overwrites_its_maps_instead_of_numbering_them(tmp_path, monkeypatch):
    """The first re-run wrote '<name>_0_map.png' beside the previous map
    because the name check looked at the disk."""
    pytest.importorskip("osgeo.gdal")
    from functions import layout
    p = _make_tif(tmp_path, (500000.0, 0.01, 0.0, 4500000.0, 0.0, -0.01), nx=90, ny=90)
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "P").mkdir(exist_ok=True)

        def fake_detect(project, ortho, window, plan, *, log_fn, progress_cb, stop_check):
            out = Path(tmp_path) / f"det_w{window}.csv"
            out.write_text("clast_ID,x,y\n1,0,0\n")
            return out

        def touch(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("x")
            return Path(path)
        maps = []
        monkeypatch.setattr(express, "_stage_detect", fake_detect)
        monkeypatch.setattr(express, "_stage_merge", lambda csvs, out: touch(out))
        monkeypatch.setattr(express, "_stage_rasterize", lambda ortho, csv, out, spec: touch(out))
        monkeypatch.setattr(express, "_stage_map", lambda ortho, src, out, spec: maps.append(Path(out).name) or touch(out))
        monkeypatch.setattr(express, "_stage_report", lambda *a, **k: None)
        express.run_express_pipeline("P", p, "medium", stop_check=lambda: False, log_fn=lambda s: None)
        first = list(maps)
        maps.clear()
        express.run_express_pipeline("P", p, "medium", stop_check=lambda: False, log_fn=lambda s: None)
        assert maps == first, "the same names again, none numbered"
        assert not any("_0_map" in m for m in maps)
    finally:
        layout.set_datasets_root(original, persist=False)
