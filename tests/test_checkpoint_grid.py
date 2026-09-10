"""An ortho checkpoint records the tile grid it was laid on, and a job on
another grid refuses it."""
from __future__ import annotations

from functions import clasts_detection as CD


def _checkpoint(tmp_path, overlap):
    run = tmp_path / "site__ortho_ws2.5m.run.csv"
    run.write_text("clast_ID,x,y,_tile_idx\n1,0,0,0\n2,1,1,1\n", encoding="utf-8")
    contours = tmp_path / "site__ortho_ws2.5m.run.contours.jsonl"
    contours.write_text('{"tile": 1, "contours": {}}\n', encoding="utf-8")
    CD.write_run_grid(run, overlap=overlap, cropsize_px=732, stride_px=585,
                      n_tiles_x=11, n_tiles_y=11)
    return run, contours


def test_the_grid_is_recorded_beside_the_checkpoint(tmp_path):
    run, _ = _checkpoint(tmp_path, 0.2)
    assert (tmp_path / "site__ortho_ws2.5m.run.grid.json").exists()
    assert CD.read_run_grid(run) == {"overlap": 0.2, "cropsize_px": 732,
                                     "stride_px": 585, "n_tiles_x": 11,
                                     "n_tiles_y": 11}


def test_the_same_grid_resumes_and_another_overlap_does_not(tmp_path):
    run, _ = _checkpoint(tmp_path, 0.2)
    assert CD.run_grid_mismatch(run, overlap=0.2, cropsize_px=732, stride_px=585) is None
    why = CD.run_grid_mismatch(run, overlap=0.0, cropsize_px=732, stride_px=732)
    assert "overlap 0.20" in why and "0.00" in why


def test_a_checkpoint_without_a_record_is_accepted(tmp_path):
    run = tmp_path / "site__ortho_ws2.5m.run.csv"
    run.write_text("clast_ID,x,y,_tile_idx\n1,0,0,0\n", encoding="utf-8")
    assert CD.run_grid_mismatch(run, overlap=0.0) is None
    info = CD.inspect_run_state(str(tmp_path), str(tmp_path / "ortho.tif"),
                                2.5, out_stem="site__ortho", overlap=0.0)
    assert info["exists"] and info["grid_mismatch"] is None


def test_inspect_run_state_flags_the_mismatch_for_the_queue(tmp_path):
    _checkpoint(tmp_path, 0.0)
    info = CD.inspect_run_state(str(tmp_path), str(tmp_path / "ortho.tif"),
                                2.5, out_stem="site__ortho", overlap=0.2)
    assert info["exists"] and info["resume_k"] == 2
    assert "overlap 0.00" in info["grid_mismatch"]
    same = CD.inspect_run_state(str(tmp_path), str(tmp_path / "ortho.tif"),
                                2.5, out_stem="site__ortho", overlap=0.0)
    assert same["grid_mismatch"] is None


def test_the_engine_discards_a_checkpoint_of_another_grid(tmp_path):
    run, contours = _checkpoint(tmp_path, 0.0)
    kept = CD.discard_mismatched_checkpoint(str(run), str(contours), overlap=0.0,
                                            cropsize_px=732, stride_px=585)
    assert kept is None and run.exists() and contours.exists()
    why = CD.discard_mismatched_checkpoint(str(run), str(contours), overlap=0.2,
                                           cropsize_px=732, stride_px=585)
    assert why and not run.exists() and not contours.exists()
    assert not (tmp_path / "site__ortho_ws2.5m.run.grid.json").exists()
