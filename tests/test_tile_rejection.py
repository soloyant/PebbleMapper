"""The tile filter names why a tile is left out."""
from __future__ import annotations

import numpy as np

from functions.clasts_detection import _tile_rejection


def _tile(value, noise=True):
    t = np.full((20, 20, 3), value, dtype=np.uint8)
    if noise:
        t[0, 0] = (value + 40) % 255
        t[1, 1] = (value + 80) % 255
    return t


def test_a_uniform_tile_is_named_uniform():
    assert _tile_rejection(_tile(120, noise=False), 15, 245, 0.95) == "uniform"


def test_a_normal_tile_passes():
    assert _tile_rejection(_tile(120), 15, 245, 0.95) is None
    assert _tile_rejection(_tile(5), None, None, 0.95) is None, "no thresholds: kept"


def test_dark_and_bright_tiles_are_named():
    assert _tile_rejection(_tile(5), 15, 245, 0.95) == "dark"
    assert _tile_rejection(_tile(250), 15, 245, 0.95) == "bright"


def test_a_zero_fraction_rejects_any_bad_pixel():
    t = _tile(120)
    t[2, 2] = 0                       # one dark pixel among 400
    assert _tile_rejection(t, 15, 245, 0.0) == "dark"
    assert _tile_rejection(t, 15, 245, 0.95) is None
