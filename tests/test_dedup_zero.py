"""Dedup IoU 0 disables the tile-overlap dedup, as the manual says."""
from __future__ import annotations

import pandas as pd

from functions import clasts_detection as CD


def _two():
    df = pd.DataFrame({"clast_ID": [1, 2], "x": [0.0, 0.1], "y": [0.0, 0.1]})
    contours = {1: [[0, 0]], 2: [[1, 1]]}
    return df, contours


def test_a_zero_threshold_leaves_the_rows_and_outlines_alone(monkeypatch):
    called = []
    monkeypatch.setattr("functions.clasts_merge.dedup_clasts",
                        lambda *a, **k: called.append(k) or a[0].iloc[:1].copy())
    df, contours = _two()
    out, cs = CD._dedup_on_completion(df, contours, "iou", 0.0, 0.2)
    assert len(out) == 2 and cs == contours and called == []
    out, cs = CD._dedup_on_completion(df, contours, "iou", None, 0.2)
    assert len(out) == 2 and called == []


def test_a_positive_threshold_merges_and_renumbers(monkeypatch):
    called = []
    monkeypatch.setattr("functions.clasts_merge.dedup_clasts",
                        lambda df, **k: called.append(k) or df.iloc[1:].copy())
    df, contours = _two()
    out, cs = CD._dedup_on_completion(df, contours, "iou", 0.3, 0.2)
    assert called and called[0]["overlap"] == 0.3
    assert out["clast_ID"].tolist() == [1], "the kept row is renumbered from 1"
    assert cs == {1: [[1, 1]]}, "its outline follows it"
