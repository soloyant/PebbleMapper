"""functions.modes: the two detection modes and every spelling they accept.

The mode called "UAV" is Ortho and the mode called "terrestrial" is Quadrat.
Old spellings must keep working as INPUT (scripts and persisted queues pass
them); everything the app writes is the canonical value.
"""
import pytest

from functions import modes


def test_canonical_values_and_labels():
    assert modes.ORTHO == "ortho"
    assert modes.QUADRAT == "quadrat"
    assert modes.MODES == (modes.ORTHO, modes.QUADRAT)
    assert modes.MODE_LABELS == {"ortho": "Ortho", "quadrat": "Quadrat"}
    assert modes.mode_label("uav") == "Ortho"
    assert modes.mode_label("terrestrial") == "Quadrat"


@pytest.mark.parametrize("spelling", ["ortho", "Ortho", "ORTHO", "uav", "UAV", "Uav",
                                      " uav ", "orthoimage", "ortho-image"])
def test_every_ortho_spelling_normalises_to_ortho(spelling):
    assert modes.normalise_mode(spelling) == modes.ORTHO
    assert modes.is_ortho(spelling) and not modes.is_quadrat(spelling)


@pytest.mark.parametrize("spelling", ["quadrat", "Quadrat", "QUADRAT", "terrestrial",
                                      "Terrestrial", "TERRESTRIAL", "photo", "photograph"])
def test_every_quadrat_spelling_normalises_to_quadrat(spelling):
    assert modes.normalise_mode(spelling) == modes.QUADRAT
    assert modes.is_quadrat(spelling) and not modes.is_ortho(spelling)


def test_normalise_is_idempotent_on_canonical_values():
    for m in modes.MODES:
        assert modes.normalise_mode(m) == m


def test_unknown_mode_raises_unless_a_default_is_given():
    with pytest.raises(ValueError, match="unknown detection mode"):
        modes.normalise_mode("satellite")
    with pytest.raises(ValueError):
        modes.normalise_mode(None)
    with pytest.raises(ValueError):
        modes.normalise_mode("")
    assert modes.normalise_mode("satellite", default="") == ""
    assert modes.normalise_mode(None, default=modes.ORTHO) == modes.ORTHO
    assert modes.is_ortho(None) is False and modes.is_quadrat("") is False
    assert modes.mode_label("satellite") == "satellite"


# --------------------------------------------------------------------------- #
#  The detector entry point accepts the old spellings and routes on the        #
#  canonical value. The model builder and both per-mode workers are replaced  #
#  so no TensorFlow graph is ever built.                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spelling, expected", [
    ("uav", modes.ORTHO), ("UAV", modes.ORTHO), ("ortho", modes.ORTHO),
    ("terrestrial", modes.QUADRAT), ("Quadrat", modes.QUADRAT),
    ("quadrat", modes.QUADRAT),
])
def test_clasts_detect_jobs_accepts_old_spellings(monkeypatch, tmp_path, spelling, expected):
    pytest.importorskip("tensorflow")
    import pandas as pd
    from functions import clasts_detection as CD

    routed = []
    monkeypatch.setattr(CD, "_build_model", lambda *a, **k: object())
    monkeypatch.setattr(CD, "_detect_quadrat",
                        lambda *a, **k: routed.append(modes.QUADRAT) or pd.DataFrame())
    monkeypatch.setattr(CD, "_detect_ortho",
                        lambda *a, **k: routed.append(modes.ORTHO) or pd.DataFrame())

    out = CD.clasts_detect_jobs(spelling, [{"path": str(tmp_path / "img.tif"), "kstart": 0}],
                                plot=False, saveresults=False, output_dir=str(tmp_path))
    assert len(out) == 1
    assert routed == [expected]


def test_clasts_detect_jobs_rejects_an_unknown_mode_before_building_the_model(monkeypatch):
    pytest.importorskip("tensorflow")
    from functions import clasts_detection as CD

    def _no_model(*a, **k):
        raise AssertionError("the model must not be built for an unknown mode")
    monkeypatch.setattr(CD, "_build_model", _no_model)
    with pytest.raises(ValueError, match="unknown detection mode"):
        CD.clasts_detect_jobs("satellite", [{"path": "x.tif", "kstart": 0}])
