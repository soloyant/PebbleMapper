"""Displayed precision derived from measured uncertainty.

The validation step measures each field's uncertainty (one JSON per quadrat);
this module reads it, pools it per mission by sample count, and turns it into
a number of decimal places:

    u  = uncertainty in display units
    u1 = u rounded to one significant figure
    the value is rendered to u1's decimal place

So 74.0348 mm +/- 8.521 mm renders as "74", with the document stating +/- 9 mm.
Rounding is a display concern only; stored values are never changed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

__all__ = [
    "Uncertainty",
    "DEFAULT_DISPLAY_DECIMALS",
    "decimals_for",
    "format_value",
    "load_validation_uncertainties",
    "uncertainty_for",
    "pooled",
]

# Used only when neither a validation result nor a GSD is known: a convention,
# not a measurement, and callers label it as such (see describe()).
DEFAULT_DISPLAY_DECIMALS: Dict[str, int] = {
    "mm": 1,
    "cm": 2,
    "m": 3,
}


def from_gsd(gsd_m: float, field: str = "") -> Optional[Uncertainty]:
    """Uncertainty of one pixel, for a project with no validation result.

    Nothing measured from an image is known to better than its GSD. This is
    not the k*GSD detection limit of :func:`functions.truncation.compute_d_min`,
    which says which clasts can be seen at all.
    """
    if not _finite_positive(gsd_m):
        return None
    return Uncertainty(field=field, value=float(gsd_m), scope="gsd",
                       n=0, sources=0, measured=False, gsd_m=float(gsd_m))


@dataclass(frozen=True)
class Uncertainty:
    """One field's measured uncertainty, in STORED units.

    It is the error measured for one mission (scene, model, GSD), so it may
    only be applied to the data it was measured on; ``gsd_m`` records that
    imagery. ``scope`` is 'quadrat' or 'mission'. ``measured`` is False when
    this stands in for an absent validation result.
    """
    field: str
    value: float                 # stored units (metres for lengths)
    scope: str = "mission"
    n: int = 0
    sources: int = 0             # how many validation files contributed
    measured: bool = True
    gsd_m: Optional[float] = None    # the imagery this was measured on

    def in_display(self, factor: float) -> float:
        """The same uncertainty in display units."""
        return float(self.value) * float(factor)

    def effective(self, gsd_m: Optional[float] = None) -> float:
        """Uncertainty floored at one pixel of the imagery, stored units.

        An RMSE finer than the GSD claims a precision the imagery cannot
        carry. The floor is the GSD passed in, else the one measured on.
        """
        g = gsd_m if _finite_positive(gsd_m) else self.gsd_m
        if _finite_positive(g):
            return max(float(self.value), float(g))
        return float(self.value)


def _finite_positive(x) -> bool:
    """A usable uncertainty. Zero, negative and NaN are all degenerate."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def load_validation_uncertainties(results_dir) -> Dict[str, list]:
    """Read every ``*.validation.json`` in one directory.

    Returns ``{field: [record, ...]}`` where each record carries the RMSE, the
    paired sample count, the truth CSV that identifies the quadrat, and the
    generation timestamp. Unreadable, truncated or older-schema files are
    skipped (a broken validation file must never fail a report build) but
    counted; see :func:`unreadable_validation_files`. An absent measurement
    and an unreadable one are different facts.
    """
    out: Dict[str, list] = {}
    dropped: list = []
    d = Path(results_dir)
    if not d.is_dir():
        _UNREADABLE[str(d)] = dropped
        return out
    for p in sorted(d.glob("*.validation.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            dropped.append((p.name, f"{type(ex).__name__}: {ex}"))
            continue
        if not isinstance(payload, dict):
            dropped.append((p.name, "not a JSON object"))
            continue
        field = str(payload.get("field") or "").strip()
        metrics = payload.get("metrics")
        if not field or not isinstance(metrics, dict):
            dropped.append((p.name, "no field or no metrics -- older schema"))
            continue
        rmse = metrics.get("rmse")
        if not _finite_positive(rmse):
            dropped.append((p.name, "no usable RMSE"))
            continue
        paired = payload.get("paired")
        n = 0
        if isinstance(paired, dict):
            try:
                n = int(paired.get("n") or 0)
            except (TypeError, ValueError):
                n = 0
        out.setdefault(field, []).append({
            "rmse": float(rmse),
            "n": n,
            "quadrat": str(payload.get("truth_csv") or p.stem),
            "generated_at": str(payload.get("generated_at") or ""),
            "path": p,
        })
    _UNREADABLE[str(d)] = dropped
    return out


_UNREADABLE: Dict[str, list] = {}
"""Per directory, the validation files the last read could not use, so a
report can say "pooled over 14 of 16; two were unreadable"."""


def unreadable_validation_files(results_dir) -> list:
    """``[(name, why)]`` skipped by the last read of this directory."""
    return list(_UNREADABLE.get(str(Path(results_dir)), []))


def pooled(records: Iterable[dict], field: str) -> Optional[Uncertainty]:
    """Pool per-quadrat RMSEs into one mission uncertainty.

    ``sqrt(sum(n_i * rmse_i^2) / sum(n_i))``, weighted by paired clasts.
    Records with fewer than two pairs are excluded; re-runs of the same
    quadrat are deduplicated to the most recent.
    """
    usable = [r for r in records if r.get("n", 0) >= 2
              and _finite_positive(r.get("rmse"))]
    if not usable:
        return None
    latest: Dict[str, dict] = {}
    for r in usable:
        key = r.get("quadrat") or str(r.get("path"))
        prev = latest.get(key)
        if prev is None or str(r.get("generated_at", "")) >= str(
                prev.get("generated_at", "")):
            latest[key] = r
    chosen = list(latest.values())
    total_n = sum(int(r["n"]) for r in chosen)
    if total_n <= 0:
        return None
    num = sum(int(r["n"]) * float(r["rmse"]) ** 2 for r in chosen)
    return Uncertainty(
        field=field,
        value=math.sqrt(num / total_n),
        scope="mission" if len(chosen) > 1 else "quadrat",
        n=total_n,
        sources=len(chosen),
        measured=True,
    )


def uncertainty_for(field: str, results_dir) -> Optional[Uncertainty]:
    """The pooled uncertainty for one field in one validation directory."""
    by_field = load_validation_uncertainties(results_dir)
    records = by_field.get(field)
    if not records:
        return None
    return pooled(records, field)


def decimals_for(u_display: Optional[float], unit: str = "") -> int:
    """Decimal places such that the last digit sits at the uncertainty.

    ``u_display`` is the uncertainty ALREADY in display units. Rounding it to
    one significant figure gives the magnitude the value may be stated to.
    ``None`` or a degenerate value falls back to the per-unit default.
    """
    if not _finite_positive(u_display):
        return DEFAULT_DISPLAY_DECIMALS.get(unit, 4)
    u = float(u_display)
    # One significant figure of u, then the decimal place that digit occupies.
    exp = math.floor(math.log10(u))
    u1 = round(u / (10.0 ** exp)) * (10.0 ** exp)
    if u1 <= 0:                      # rounding underflowed
        return DEFAULT_DISPLAY_DECIMALS.get(unit, 4)
    exp1 = math.floor(math.log10(u1))
    return max(0, -exp1)


def format_value(value, u_display: Optional[float] = None,
                 unit: str = "") -> str:
    """Render a measured value at the precision its uncertainty supports.

    A value smaller than its own uncertainty renders as the rounded number
    even when that is ``0``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{decimals_for(u_display, unit)}f}"


def describe(u: Optional[Uncertainty], factor: float, unit: str) -> str:
    """One sentence a document can print beside the numbers it governs."""
    if u is None:
        dp = DEFAULT_DISPLAY_DECIMALS.get(unit, 4)
        return (f"Neither a validation result nor a ground sample distance was "
                f"found for this field, so values are shown to a default "
                f"{dp} decimal place(s) in {unit or 'stored units'}. That is a "
                f"convention, not a measured precision.")
    if not u.measured:
        g = f"{u.in_display(factor):g}"
        return (f"No validation result was found for this field, so values are "
                f"shown to one pixel of the imagery: ± {g} {unit} "
                f"(ground sample distance). This bounds the resolution, not the "
                f"detector's actual error, which validation would measure.")
    ud = u.effective() * factor
    exp = math.floor(math.log10(ud))
    u1 = round(ud / (10.0 ** exp)) * (10.0 ** exp)
    shown = f"{u1:g}"
    scope = ("pooled across the mission's quadrats" if u.sources > 1
             else "from a single validation quadrat")
    return (f"Values are rounded to the measured uncertainty: ± {shown} "
            f"{unit} (RMSE {scope}, n = {u.n}"
            + (f" over {u.sources} quadrats" if u.sources > 1 else "") + ").")
