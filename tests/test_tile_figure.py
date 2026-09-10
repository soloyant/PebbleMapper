"""A tile's figure is drawn on its own axes and saved without printing
."""
from __future__ import annotations

import numpy as np

from functions import clasts_detection as CD


def _tile():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_an_empty_tile_is_saved_quietly(tmp_path, capsys):
    r = {"rois": np.zeros((0, 4)), "masks": np.zeros((64, 64, 0), dtype=bool),
         "class_ids": np.zeros((0,), dtype=int), "scores": np.zeros((0,))}
    png = tmp_path / "t.png"
    CD._tile_figure(_tile(), r, ["BG", "Clast"], "Tile k=0", str(png), False)
    assert png.exists() and png.stat().st_size > 0
    assert capsys.readouterr().out == ""


def test_a_tile_with_a_detection_is_saved_quietly(tmp_path, capsys):
    mask = np.zeros((64, 64, 1), dtype=bool)
    mask[20:40, 20:40, 0] = True
    r = {"rois": np.array([[20, 20, 40, 40]]), "masks": mask,
         "class_ids": np.array([1]), "scores": np.array([0.9])}
    png = tmp_path / "t.png"
    CD._tile_figure(_tile(), r, ["BG", "Clast"], "Tile k=1", str(png), False)
    assert png.exists() and png.stat().st_size > 0
    assert capsys.readouterr().out == ""
