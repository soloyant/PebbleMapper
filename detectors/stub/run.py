"""Stub detection backend: subprocess entry point (runs in env 'pebble-stub').

Not a real detector. It demonstrates the instances pattern a backend living
in a separate conda environment uses: read ``--spec <json>``, segment, and
write each job's *instances file* (``instances_path``, see
``detectors/subprocess_runner.py``) — an int32 label image plus one score per
instance. The in-core driver (``detectors/stub_backend.py``) measures the
instances with the core's shared measurement step and writes the canonical
CSV; this file never measures and never imports the core app.

It depends only on the standard library, so the tiny ``.npz`` writer below
stands in for ``numpy.savez_compressed``, which a real backend would use. The
synthetic "clasts" are three axis-aligned ellipses whose sizes scale with
``metric_cropsize`` and ``resolution``, so parameter pass-through is
verifiable end to end.
"""
import argparse
import json
import math
import os
import struct
import sys
import zipfile


# --------------------------------------------------------------------------- #
#  Synthetic instances                                                        #
# --------------------------------------------------------------------------- #
def synthetic_axes_m(params, n=3):
    """(major, minor) in metres for each synthetic clast; the major axis is
    ``metric_cropsize`` x (0.10 + 0.02 i), the minor 60 % of it."""
    crop = float(params.get("metric_cropsize", 1.0) or 1.0)
    out = []
    for i in range(n):
        major = crop * (0.10 + 0.02 * i)
        out.append((major, major * 0.6))
    return out


def synthetic_instances(params, n=3):
    """Label image (list of rows of ints), scores, (H, W).

    Ellipse i has semi-axes ``major/2`` and ``minor/2`` in pixels (metres /
    ``resolution``), laid out left to right on one row.
    """
    resolution = float(params.get("resolution", 0.001) or 0.001)
    axes_px = [(max(6.0, a / resolution / 2.0), max(4.0, b / resolution / 2.0))
               for a, b in synthetic_axes_m(params, n)]
    # The canvas is capped so a huge window on a fine GSD does not take
    # minutes to rasterise in pure Python; the stub is a proof, not a model.
    scale = 1.0
    width_needed = sum(2 * a + 8 for a, _ in axes_px) + 8
    if width_needed > 1200:
        scale = 1200.0 / width_needed
        axes_px = [(a * scale, b * scale) for a, b in axes_px]
    margin = 4
    width = int(math.ceil(sum(2 * a + 2 * margin for a, _ in axes_px))) + 2 * margin
    height = int(math.ceil(max(2 * b for _, b in axes_px))) + 4 * margin
    cy = height / 2.0
    rows = [[0] * width for _ in range(height)]
    x_cursor = float(margin)
    for k, (a, b) in enumerate(axes_px, start=1):
        cx = x_cursor + margin + a
        for y in range(height):
            dy = (y + 0.5 - cy) / b
            if abs(dy) >= 1.0:
                continue
            half = a * math.sqrt(1.0 - dy * dy)
            x0 = int(math.ceil(cx - half - 0.5))
            x1 = int(math.floor(cx + half - 0.5)) + 1
            x0, x1 = max(0, x0), min(width, x1)
            if x1 > x0:
                rows[y][x0:x1] = [k] * (x1 - x0)
        x_cursor = cx + a + margin
    scores = [round(0.90 - 0.01 * i, 6) for i in range(n)]
    return rows, scores, (height, width)


# --------------------------------------------------------------------------- #
#  Minimal .npy / .npz writer (standard library only)                          #
# --------------------------------------------------------------------------- #
def _npy_bytes(descr, shape, payload):
    """NumPy .npy v1.0: magic, version, header dict padded to 64 bytes, data."""
    header = ("{'descr': '%s', 'fortran_order': False, 'shape': %s, }"
              % (descr, repr(tuple(shape)) if len(shape) != 1
                 else "(%d,)" % shape[0]))
    prefix_len = 6 + 2 + 2  # magic + version + header length
    pad = 64 - ((prefix_len + len(header) + 1) % 64)
    header = header + " " * pad + "\n"
    return (b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header))
            + header.encode("latin1") + payload)


def write_instances_npz(path, rows, scores, shape):
    """Write ``labels`` (int32 HxW), ``scores`` (float32 N) and ``shape``
    (int64 2) into a ``.npz`` that ``numpy.load`` reads back."""
    h, w = shape
    labels = b"".join(struct.pack("<%di" % w, *row) for row in rows)
    scores_b = struct.pack("<%df" % len(scores), *scores)
    shape_b = struct.pack("<2q", h, w)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labels.npy", _npy_bytes("<i4", (h, w), labels))
        zf.writestr("scores.npy", _npy_bytes("<f4", (len(scores),), scores_b))
        zf.writestr("shape.npy", _npy_bytes("<i8", (2,), shape_b))


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Stub detection backend.")
    ap.add_argument("--spec", required=True, help="path to job-spec JSON")
    args = ap.parse_args(argv)

    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    params = spec.get("params", {})
    jobs = spec.get("jobs", [])

    for ji, job in enumerate(jobs, start=1):
        out = job.get("instances_path") or (job["out_csv"] + ".instances.npz")
        rows, scores, shape = synthetic_instances(params)
        write_instances_npz(out, rows, scores, shape)
        # Plain print: the driver streams these lines into the app's log.
        print(f"[stub] job {ji}/{len(jobs)}: {len(scores)} synthetic instances "
              f"on a {shape[1]}x{shape[0]} canvas -> {os.path.basename(out)}",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
