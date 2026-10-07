"""The two detection modes and their spellings.

``ORTHO``: a georeferenced ortho-image, processed in tiles, world coordinates.
``QUADRAT``: a photograph of known scale (usually of a quadrat frame),
processed whole, image pixels scaled by the GSD.

``normalise_mode`` maps every spelling the app has ever written (``"uav"``,
``"UAV"``, ``"terrestrial"``, persisted queues, scripts) to the canonical
value, so callers compare against ``ORTHO`` and ``QUADRAT`` only.
"""
from __future__ import annotations

from typing import Optional

ORTHO = "ortho"
QUADRAT = "quadrat"
MODES = (ORTHO, QUADRAT)

MODE_LABELS = {ORTHO: "Ortho", QUADRAT: "Quadrat"}

_ALIASES = {
    "ortho": ORTHO, "uav": ORTHO, "orthoimage": ORTHO, "ortho-image": ORTHO,
    "quadrat": QUADRAT, "terrestrial": QUADRAT, "photo": QUADRAT,
    "photograph": QUADRAT,
}


def normalise_mode(mode: Optional[str], default: Optional[str] = None) -> str:
    """The canonical mode for any accepted spelling, case-insensitive.

    Raises ``ValueError`` for an unknown value unless ``default`` is given.
    """
    key = str(mode or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if default is not None:
        return default
    raise ValueError(f"unknown detection mode {mode!r}; expected 'ortho' or 'quadrat'")


def is_ortho(mode: Optional[str]) -> bool:
    return normalise_mode(mode, default="") == ORTHO


def is_quadrat(mode: Optional[str]) -> bool:
    return normalise_mode(mode, default="") == QUADRAT


def mode_label(mode: Optional[str]) -> str:
    """``"Ortho"`` or ``"Quadrat"`` for display."""
    return MODE_LABELS.get(normalise_mode(mode, default=""), str(mode or ""))


__all__ = ["ORTHO", "QUADRAT", "MODES", "MODE_LABELS", "normalise_mode",
           "is_ortho", "is_quadrat", "mode_label"]
