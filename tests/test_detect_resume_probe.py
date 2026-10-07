"""Add to queue finds a project image's checkpoint.

The checkpoint carries the origin stem like every other output; the probe
asked for it under the bare image stem, never found it, and offered the
partial final CSV as "done" instead of "Will resume from tile N".
"""
from __future__ import annotations

import numpy as np
import pytest

_FIELDS = ("current_project", "det_dir", "det_mode", "det_files",
           "det_files_checked", "det_jobs", "det_resolution",
           "det_metric_cropsize", "det_kstart", "det_overlap",
           "det_brightness_filter", "det_nodata_max_frac", "det_dark_threshold")


@pytest.fixture
def clean():
    from gui.app import state
    saved = {k: getattr(state, k) for k in _FIELDS}
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    state.det_jobs = []
    state.det_files = []
    state.det_files_checked = {}
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def _geotiff(path):
    from osgeo import gdal, osr
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), 40, 40, 3, gdal.GDT_Byte)
    ds.SetGeoTransform((497970.0, 0.05, 0, 6960100.0, 0, -0.05))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(1, 4):
        ds.GetRasterBand(b).WriteArray(np.full((40, 40), 120, np.uint8))
    ds.FlushCache()
    ds = None


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def _press(col, label):
    from nicegui.events import ClickEventArguments, handle_event
    btn = next(e for e in _walk(col) if e.__class__.__name__ == "Button"
               and str(getattr(e, "text", "")) == label)
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))


def test_add_to_queue_offers_to_resume_a_project_images_checkpoint(clean, tmp_path):
    import functions.layout as layout
    from functions import naming
    from nicegui import ui
    from gui.app import build_detection_tab
    state, built = clean
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "P").mkdir()
        images = layout.project_path("P", "images")
        images.mkdir(parents=True, exist_ok=True)
        _geotiff(images / "ortho.tif")
        vectors = layout.project_path("P", "vectors")
        vectors.mkdir(parents=True, exist_ok=True)
        stem = naming.origin_stem("ortho", "P", layout.get_active_date())
        run_csv = vectors / naming.detection_csv_name(stem, 2.5, run=True)
        run_csv.write_text("clast_ID,x,y,_tile_idx\n1,0,0,0\n2,1,1,1\n3,2,2,2\n",
                           encoding="utf-8")
        # The partial final CSV a stopped run also writes.
        (vectors / naming.detection_csv_name(stem, 2.5)).write_text(
            "clast_ID,x,y\n1,0,0\n2,1,1\n3,2,2\n", encoding="utf-8")

        state.current_project = "P"
        state.det_mode = "ortho"
        state.det_metric_cropsize = 2.5
        state.det_kstart = 0
        state.det_dir = ""
        with ui.column() as col:
            build_detection_tab()
        built.append(col)
        inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
                   and e._props.get("label") == "Image directory")
        inp.value = str(images)
        assert state.det_files == ["ortho.tif"]
        _press(col, "Add to queue")
        assert len(state.det_jobs) == 1
        job = state.det_jobs[0]
        assert job["resume_state"] == "resume", job
        assert job["kstart"] == 3, "last tile 2 -> resume from 3"
        assert "Will resume from tile 3" in job.get("resume_note", "") or job["status"] == "pending"

        # An emptied "Default kstart for fresh jobs" field (None) must not
        # kill Add to queue: a fresh job starts at 0.
        run_csv.unlink()
        state.det_jobs = []
        state.det_kstart = None
        _press(col, "Add to queue")
        assert len(state.det_jobs) == 1 and state.det_jobs[0]["kstart"] == 0
    finally:
        layout.set_datasets_root(original, persist=False)


def test_add_to_queue_does_not_offer_a_checkpoint_of_another_grid(clean, tmp_path):
    import functions.layout as layout
    from functions import naming
    from functions import clasts_detection as CD
    from nicegui import ui
    from gui.app import build_detection_tab
    state, built = clean
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "P").mkdir()
        images = layout.project_path("P", "images")
        images.mkdir(parents=True, exist_ok=True)
        _geotiff(images / "ortho.tif")
        vectors = layout.project_path("P", "vectors")
        vectors.mkdir(parents=True, exist_ok=True)
        stem = naming.origin_stem("ortho", "P", layout.get_active_date())
        run_csv = vectors / naming.detection_csv_name(stem, 2.5, run=True)
        run_csv.write_text("clast_ID,x,y,_tile_idx\n1,0,0,0\n2,1,1,1\n", encoding="utf-8")
        CD.write_run_grid(run_csv, overlap=0.0, cropsize_px=50, stride_px=50,
                          n_tiles_x=1, n_tiles_y=1)
        state.current_project = "P"
        state.det_mode = "ortho"
        state.det_metric_cropsize = 2.5
        state.det_overlap = 0.2          # not the checkpoint's grid
        state.det_kstart = 0
        state.det_dir = ""
        with ui.column() as col:
            build_detection_tab()
        built.append(col)
        inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
                   and e._props.get("label") == "Image directory")
        inp.value = str(images)
        _press(col, "Add to queue")
        job = state.det_jobs[0]
        assert job["resume_state"] == "fresh" and job["kstart"] == 0
        assert "another tile grid" in job.get("resume_note", "")
    finally:
        layout.set_datasets_root(original, persist=False)


def test_emptied_number_fields_do_not_kill_add_to_queue(clean, tmp_path):
    """Every number field of the form may be emptied (None); the job then
    carries the field's default."""
    import functions.layout as layout
    from nicegui import ui
    from gui.app import build_detection_tab
    state, built = clean
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "P").mkdir()
        images = layout.project_path("P", "images")
        images.mkdir(parents=True, exist_ok=True)
        _geotiff(images / "ortho.tif")
        state.current_project = "P"
        state.det_mode = "ortho"
        state.det_dir = ""
        state.det_metric_cropsize = None      # the number inputs; the sliders
        state.det_overlap = 0.2                # (overlap, dedup) cannot be emptied
        state.det_kstart = None
        state.det_brightness_filter = True
        state.det_nodata_max_frac = None
        state.det_dark_threshold = None
        with ui.column() as col:
            build_detection_tab()
        built.append(col)
        inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
                   and e._props.get("label") == "Image directory")
        inp.value = str(images)
        _press(col, "Add to queue")
        job = state.det_jobs[0]
        assert (job["metric_cropsize"], job["overlap"], job["kstart"]) == (1.0, 0.2, 0)
        assert (job["nodata_max_frac"], job["dark_threshold"]) == (0.95, 15)
    finally:
        layout.set_datasets_root(original, persist=False)
