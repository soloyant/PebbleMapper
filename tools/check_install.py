"""Report whether this environment can run PebbleMapper, and say what is missing.

    python tools/check_install.py

Exits non-zero only when something would actually stop the application from
starting. A missing GPU or missing weights is reported but not an error: the
application runs on the processor, and every tab except Detect works without
the weights.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = [
    ("numpy", "1.23.5"),
    ("tensorflow", "2.10.1"),
    ("nicegui", None),
    ("pandas", None),
    ("PIL", None),
    ("cv2", None),
    ("matplotlib", None),
    ("shapely", None),
    ("rasterio", None),
    ("skimage", None),
]
OPTIONAL = [
    ("pillow_heif", "iPhone HEIC photographs"),
    ("reportlab", "PDF reports"),
]


def version(mod) -> str:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "?"


def main() -> int:
    print(f"PebbleMapper install check")
    print(f"  repository  {REPO}")
    print(f"  python      {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"  platform    {platform.platform()}")
    print()

    problems: list[str] = []
    notes: list[str] = []

    print("Required packages")
    for name, want in REQUIRED:
        try:
            mod = importlib.import_module(name)
        except Exception as ex:
            print(f"  MISSING  {name:<12} {type(ex).__name__}: {ex}")
            problems.append(f"{name} does not import")
            continue
        got = version(mod)
        mark = "ok     "
        if want and got != want:
            mark = "WRONG  "
            notes.append(f"{name} is {got}, the environment pins {want}")
        print(f"  {mark}  {name:<12} {got}")

    try:
        from osgeo import gdal
        print(f"  ok       {'gdal':<12} {gdal.__version__}")
    except Exception as ex:
        print(f"  MISSING  {'gdal':<12} {type(ex).__name__}: {ex}")
        problems.append("osgeo.gdal does not import")

    print("\nOptional packages")
    for name, what in OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"  ok       {name:<12} {version(mod)}  ({what})")
        except Exception:
            print(f"  absent   {name:<12} {what} will not work")
            notes.append(f"{name} is absent, so {what} will not work")

    print("\nGPU")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                print(f"  ok       {g.name}")
        else:
            print("  none     detection will run on the processor, more slowly")
            notes.append("no GPU visible to TensorFlow")
    except Exception as ex:
        print(f"  unknown  {type(ex).__name__}: {ex}")

    print("\nModel weights")
    w = REPO / "model_weights" / "mask_rcnn_clasts.h5"
    if w.is_file():
        print(f"  ok       {w.name}  {w.stat().st_size / 1048576:.0f} MB")
    else:
        print("  absent   model_weights/mask_rcnn_clasts.h5")
        import textwrap
        sys.path.insert(0, str(REPO))
        from detectors.download_weights import weights_missing_message
        for line in textwrap.wrap(weights_missing_message(), 76):
            print("           " + line)
        notes.append("the Mask R-CNN weights are not in place, so Detect cannot run yet")

    print("\nExample data")
    examples = sorted(p for p in (REPO / "datasets").glob("example_*") if p.is_dir())
    if examples:
        for p in examples:
            print(f"  ok       {p.name}")
    else:
        print("  absent   datasets/example_* are not present")

    print()
    if problems:
        print("NOT READY")
        for p in problems:
            print(f"  - {p}")
        print("\nRebuild the environment: conda env create -f environment.yml")
        return 1

    print("READY. Start it with: python -m gui.app")
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
