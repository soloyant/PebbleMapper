"""Shared pytest configuration and fixtures for the PebbleMapper suite.

Two responsibilities:

1. **Import path** — put the repo root on ``sys.path`` so ``import functions``
   and ``import gui.app`` work no matter what cwd pytest is invoked from. 
   Centralising it here means every test file gets it for free.

2. **``synthetic_project`` fixture** — builds a minimal, canonical-layout
   project on disk under ``tmp_path``. It is deterministic (seeded RNG) and
   self-contained: no network, no GDAL, no real datasets.

The fixture is consumed by ``tests/test_report_smoke.py`` and
``tests/test_report.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
#  Import path: repo root on sys.path                                          #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
#  Canonical detection-CSV column schema                                       #
# --------------------------------------------------------------------------- #
#  Mirrors the schema written by functions/clasts_merge.py and digitize.py
#  (``Equivalent_diameter`` included). Kept here as the single source of
#  truth for tests.
CLAST_COLUMNS = [
    "clast_ID", "x", "y",
    "Ellipse_major_axis", "Ellipse_minor_axis",
    "Clast_length", "Clast_width",
    "Surface_area", "Equivalent_diameter",
    "Score", "Orientation",
]


def _synth_clasts(rng, n, length_mean_m, x0, y0, span):
    """Return a DataFrame of ``n`` synthetic clasts with the canonical schema.

    Sizes are drawn from a log-normal distribution (realistic grain-size
    shape) so the report's distribution figures have something sane to fit.
    All geometric columns are internally consistent (area derived from the
    ellipse axes, equivalent diameter derived from the area).
    """
    import numpy as np
    import pandas as pd

    length = rng.lognormal(mean=np.log(length_mean_m), sigma=0.45, size=n)
    width = length * rng.uniform(0.5, 0.9, size=n)
    major = length * rng.uniform(1.0, 1.15, size=n)
    minor = width * rng.uniform(0.9, 1.05, size=n)
    area = np.pi * (major / 2) * (minor / 2)
    eqd = 2 * np.sqrt(area / np.pi)
    df = pd.DataFrame({
        "clast_ID": np.arange(1, n + 1),
        "x": x0 + rng.uniform(0, span, size=n),
        "y": y0 + rng.uniform(0, span, size=n),
        "Ellipse_major_axis": major,
        "Ellipse_minor_axis": minor,
        "Clast_length": length,
        "Clast_width": width,
        "Surface_area": area,
        "Equivalent_diameter": eqd,
        "Score": rng.uniform(0.70, 0.99, size=n),
        "Orientation": rng.uniform(-90, 90, size=n),
    })
    return df[CLAST_COLUMNS]


def _detection_log(ortho_name, n_clasts, cropsize):
    return (
        f"[Run 2026-06-10T12:00:00]\n"
        f"[Inputs]\n"
        f"ortho: {ortho_name}\n"
        f"[Parameters]\n"
        f"metric_cropsize: {cropsize}\n"
        f"overlap: 0.20\n"
        f"min_confidence: 0.70\n"
        f"[Outputs]\n"
        f"clasts detected: {n_clasts}\n"
        f"this run elapsed (s): 42.5\n"
        f"Cumulative elapsed (s): 42.5\n"
    )


def _build_synthetic_project(root: Path) -> dict:
    """Materialise a canonical-layout project under ``root``.

    Returns a small manifest dict describing what was written (used by tests
    to cross-check the report inventory without re-deriving it).

    Requires numpy / pandas / matplotlib — callers gate on those via
    ``pytest.importorskip`` so a minimal env skips cleanly rather than erroring.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)  # deterministic

    img_dir = root / "input_data"
    vec_dir = root / "output_results" / "vectors"
    map_dir = root / "output_results" / "maps"
    zon_dir = root / "output_results" / "zonal"
    rep_dir = root / "output_results" / "reports"
    for d in (img_dir, vec_dir, map_dir, zon_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- input images (ortho + quadrat photograph), as small JPGs ---
    image_names = [
        "DummySite_georef.jpg",
        "Q1_corrected_width=1.0m_height=1.0m_GSD=0.0005m_per_px.jpg",
    ]
    for name in image_names:
        arr = rng.uniform(0.3, 0.8, size=(120, 160, 3))
        plt.imsave(img_dir / name, np.clip(arr, 0, 1))

    # --- detection vectors (ortho merged + quadrat) + logs ---
    n_uav, n_terr = 260, 140
    uav = _synth_clasts(rng, n_uav, 0.06, 500000.0, 4500000.0, 50.0)
    uav.to_csv(
        vec_dir / "DummySite_georef_merged_individual_clast_values.csv",
        index=False,
    )
    (vec_dir / "DummySite_georef_detection_log.txt").write_text(
        _detection_log("DummySite_georef.jpg", n_uav, 2.5), encoding="utf-8")

    terr = _synth_clasts(rng, n_terr, 0.03, 0.0, 0.0, 1000.0)
    terr.to_csv(
        vec_dir / "Q1_corrected_individual_clast_values.csv", index=False)
    (vec_dir / "Q1_corrected_detection_log.txt").write_text(
        _detection_log("Q1_corrected.jpg", n_terr, 0.0), encoding="utf-8")

    # --- publication maps (output_results/maps/) ---
    for field, cmap in (("Clast_length_D50", "viridis"),
                        ("folk_ward_sorting", "magma")):
        fig, ax = plt.subplots(figsize=(5, 4))
        grid = rng.uniform(0.02, 0.1, size=(25, 30))
        im = ax.pcolormesh(grid, cmap=cmap)
        ax.set_title(f"{field} (dummy)")
        ax.set_xlabel("Easting (cells)")
        ax.set_ylabel("Northing (cells)")
        fig.colorbar(im, ax=ax, label=field)
        fig.savefig(map_dir / f"DummySite_{field}_map.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # --- zonal stats (polygons CSV, transects CSV, quick-look PNGs) ---
    import pandas as pd
    poly = pd.DataFrame({
        "polygon_id": [f"poly_{i}" for i in range(1, 6)],
        "count": rng.integers(20, 80, size=5),
        "density": rng.uniform(50, 200, size=5).round(1),
        "mean": rng.uniform(0.04, 0.08, size=5).round(4),
        "std": rng.uniform(0.01, 0.03, size=5).round(4),
        "median": rng.uniform(0.04, 0.07, size=5).round(4),
        "iqr": rng.uniform(0.01, 0.04, size=5).round(4),
        "sigma_phi": rng.uniform(0.4, 1.2, size=5).round(3),
        "sk_phi": rng.uniform(-0.3, 0.3, size=5).round(3),
        "kg_phi": rng.uniform(0.8, 1.4, size=5).round(3),
    })
    poly.to_csv(zon_dir / "DummySite__zones.polygons.csv", index=False)

    rows = []
    for tid in ("T1", "T2"):
        for d in np.linspace(0, 20, 21):
            # The Zonal tab's schema: distance_m / raster_value / elevation_m
            # plus the field and unit provenance columns (stored metres).
            rows.append({
                "transect_id": tid,
                "distance_m": round(float(d), 2),
                "raster_value": round(float(0.05 + 0.01 * np.sin(d / 3)), 4),
                "elevation_m": round(float(2.0 + 0.1 * d), 3),
                "field": "Clast_length",
                "unit": "m",
            })
    pd.DataFrame(rows).to_csv(
        zon_dir / "DummySite__transects.transects.csv", index=False)

    d = np.linspace(0, 20, 21)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(d, 0.05 + 0.01 * np.sin(d / 3), "-o", ms=3)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("D50 (m)")
    ax.set_title("Transect T1")
    fig.savefig(zon_dir / "DummySite__T1.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(d, 0.05 + 0.01 * np.sin(d / 3), "-o", ms=3, label="D50")
    ax2 = ax.twinx()
    ax2.plot(d, 2.0 + 0.1 * d, "r--", label="elevation")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("D50 (m)")
    ax2.set_ylabel("Elev (m)")
    ax.set_title("Combined overview")
    fig.savefig(zon_dir / "combined_DummySite_overview.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "root": root,
        "images": image_names,
        "vectors": [
            "DummySite_georef_merged_individual_clast_values.csv",
            "Q1_corrected_individual_clast_values.csv",
        ],
        "logs": [
            "DummySite_georef_detection_log.txt",
            "Q1_corrected_detection_log.txt",
        ],
        "maps": [
            "DummySite_Clast_length_D50_map.png",
            "DummySite_folk_ward_sorting_map.png",
        ],
        "n_clasts_total": n_uav + n_terr,
        "n_uav": n_uav,
        "n_terr": n_terr,
        "reports_dir": rep_dir,
    }


@pytest.fixture
def synthetic_project(tmp_path):
    """A minimal canonical-layout project on disk (deterministic, offline).

    Yields a manifest dict (see ``_build_synthetic_project``). The heavy
    plotting/array deps are imported lazily and skipped cleanly if missing, so
    a stripped-down env does not error on collection.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    return _build_synthetic_project(tmp_path / "synthetic_proj")


@pytest.fixture(autouse=True)
def _fresh_addon_unit_registry():
    """Add-on units registered by one test must not leak into the next."""
    from functions import units
    units.clear_registered_field_units()
    yield
    units.clear_registered_field_units()
