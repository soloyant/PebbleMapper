"""The zones canvas writes where the zone set name says, never into the
file it happened to write last or into an imported layer."""
from __future__ import annotations

from pathlib import Path


def test_the_last_written_file_counts_only_while_the_name_matches():
    from gui.app import zonal_target_out_path
    last = "C:/p/geometries/zones_test.geojson"
    assert zonal_target_out_path(last, "zones_test") == last
    assert zonal_target_out_path(last, "") == ""
    assert zonal_target_out_path(last, "other") == ""
    assert zonal_target_out_path("", "zones_test") == ""


def test_the_resolved_path_follows_the_name(tmp_path):
    import functions.layout as layout
    from gui.app import zonal_set_save_path, zonal_target_out_path
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "P").mkdir()
        last = str(tmp_path / "P" / "input_data" / "geometries" / "zones_test.geojson")
        # No name: nothing to write to, the component prompts.
        assert zonal_set_save_path(zonal_target_out_path(last, ""), "", "P", "") is None
        # Another name: its own file, not the last one written.
        got = zonal_set_save_path(zonal_target_out_path(last, "fresh"), "fresh", "P", "")
        assert Path(got).name == "fresh.geojson" and str(got) != last
        # The same name: the file already written.
        assert str(zonal_set_save_path(zonal_target_out_path(last, "zones_test"), "zones_test", "P", "")) == last
    finally:
        layout.set_datasets_root(original, persist=False)
