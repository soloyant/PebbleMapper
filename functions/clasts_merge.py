"""
Clast deduplication and dataset merging.

``dedup_clasts`` removes near-duplicate clasts within one DataFrame (tile
overlap, or cross-window re-detections) by a greedy keep-the-best scan over a
KD-tree neighbour query. ``merge_csvs`` concatenates two detection CSVs from
different window sizes and deduplicates, preferring the large-window rows on
conflicts, and carries the outlines of the kept rows into
``<merged stem>.contours.json`` under the renumbered clast_IDs.
"""
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


# Mirrors functions.clasts_detection._CLAST_COLUMNS (duplicated so this light
# module needn't import the TF-heavy detector). Used only to reorder output
# columns; extras are preserved at the end.
_CLAST_COLUMNS = ['clast_ID', 'x', 'y',
                  'Clast_length', 'Clast_width',
                  'Ellipse_major_axis', 'Ellipse_minor_axis',
                  'Surface_area', 'Perimeter', 'Equivalent_diameter',
                  'Eccentricity', 'Solidity', 'Mean_intensity',
                  'Score', 'Orientation']


def _ellipse_polygon(cx, cy, length, width, orientation_deg, n_points=32):
    """n_points-vertex polygon approximating a clast's ellipse footprint.

    ``orientation_deg`` is the CSV's ``Orientation``: the bearing of the long
    axis, clockwise from +y (grid north, or the top of the photograph), in a
    frame whose y counts up (see :mod:`functions.clast_geometry`). ``length``
    and ``width`` are full diameters (as Clast_length / Clast_width), not radii.
    """
    from functions.clast_geometry import ellipse_outline
    try:
        o = float(orientation_deg)
    except (TypeError, ValueError):
        o = float("nan")
    if not math.isfinite(o):
        o = 0.0          # a missing bearing still gives a footprint of the right size
    pts = ellipse_outline(cx, cy, length, width, o, y_down=False,
                          n_points=n_points)
    return Polygon(pts)


def _quality_score(row, priority_col=None):
    """Quality key for greedy keep-the-best selection: priority column (if
    any), then Score, then size. Higher is better."""
    pri = row[priority_col] if priority_col and priority_col in row.index else 0.0
    score = row['Score'] if 'Score' in row.index and pd.notna(row['Score']) else 0.0
    size = row['Clast_length'] if 'Clast_length' in row.index and pd.notna(row['Clast_length']) else 0.0
    return (pri, score, size)


def dedup_clasts(df, method='iou', overlap=0.30, priority_col=None,
                 n_points=32, search_radius_factor=1.0, return_kept_mask=False):
    """Remove near-duplicate clasts via a greedy keep-the-best scan.

    df needs columns x, y, Clast_length, Clast_width, Orientation (Score used
    as a tiebreaker if present).

    method : 'iou' (conflict when ellipse IoU >= overlap; precise, for
        cross-run merges) or 'centroid' (conflict when centroid distance <
        overlap * mean length; fast, for tile-overlap dedup).
    overlap : minimum IoU, or fraction of the average length, per method.
    priority_col : column whose higher values win conflicts (e.g. a
        window-size rank); None for tile-overlap dedup.
    n_points : vertices per ellipse polygon for method='iou'.
    search_radius_factor : multiplier on the neighbour-query radius; raise
        for elongated clasts where the bounding circle is loose.
    return_kept_mask : also return a boolean array marking kept rows.

    Returns the deduped DataFrame (and the kept mask if requested).
    """
    if method not in ('iou', 'centroid'):
        raise ValueError(f"method must be 'iou' or 'centroid', got {method!r}")
    n = len(df)
    if n == 0:
        kept = np.zeros(0, dtype=bool)
        return (df.copy(), kept) if return_kept_mask else df.copy()

    df = df.reset_index(drop=True)
    xs = df['x'].values
    ys = df['y'].values
    lens = df['Clast_length'].values
    wids = df['Clast_width'].values

    quality = [_quality_score(df.iloc[i], priority_col) for i in range(n)]
    order = sorted(range(n), key=lambda i: quality[i], reverse=True)

    # Two ellipses can only overlap if their centroid distance is <= rA + rB,
    # with r = Clast_length/2 an upper bound on the semi-major axis.
    tree = cKDTree(np.column_stack([xs, ys]))
    radii = lens / 2.0
    max_radius = float(radii.max()) if n else 0.0

    if method == 'iou':
        # Polygons built lazily, only for clasts actually touched.
        polys = {}

        def get_poly(i):
            if i not in polys:
                polys[i] = _ellipse_polygon(xs[i], ys[i], lens[i], wids[i],
                                             df['Orientation'].iloc[i], n_points)
            return polys[i]

    kept = np.ones(n, dtype=bool)

    for i in order:
        if not kept[i]:
            continue
        cand = tree.query_ball_point([xs[i], ys[i]],
                                     r=(radii[i] + max_radius) * search_radius_factor)
        for j in cand:
            if j == i or not kept[j]:
                continue
            # i is the higher-priority of the two (scanning in quality order).
            if method == 'centroid':
                d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                threshold = overlap * 0.5 * (lens[i] + lens[j])
                conflict = d < threshold
            else:  # 'iou'
                d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if d >= radii[i] + radii[j]:
                    continue
                pi, pj = get_poly(i), get_poly(j)
                inter = pi.intersection(pj).area
                if inter == 0:
                    continue
                union = pi.area + pj.area - inter
                iou = inter / union if union > 0 else 0.0
                conflict = iou >= overlap
            if conflict:
                kept[j] = False

    deduped = df.iloc[kept].reset_index(drop=True)
    return (deduped, kept) if return_kept_mask else deduped


def merge_csvs(input_filepath_small, input_filepath_large, output_filepath,
               method='iou', overlap=0.30, n_points=32):
    """Concatenate a small-window and a large-window clast CSV, deduplicate
    (large-window rows win conflicts), renumber clast_IDs sequentially and
    write to output_filepath. ``method``, ``overlap``, ``n_points`` are
    forwarded to dedup_clasts(). Returns the merged DataFrame."""
    df_small = pd.read_csv(input_filepath_small)
    df_large = pd.read_csv(input_filepath_large)

    # Drop degenerate measurements first. Only size columns present in BOTH
    # inputs are checked; shape metrics are excluded because a legitimately
    # round clast has eccentricity 0.
    _candidate_cols = ['Clast_length', 'Clast_width',
                       'Ellipse_major_axis', 'Ellipse_minor_axis']
    measurement_cols = [c for c in _candidate_cols
                        if c in df_small.columns and c in df_large.columns]

    def _filter(df, label):
        if not measurement_cols:
            return df.reset_index(drop=True)
        ok = (df[measurement_cols] > 0).all(axis=1)
        n_drop = (~ok).sum()
        if n_drop:
            print(f"  {label}: dropped {n_drop} rows with non-positive measurements")
        return df[ok].reset_index(drop=True)

    print(f"Loading inputs:")
    print(f"  small-window: {len(df_small)} rows from {input_filepath_small}")
    print(f"  large-window: {len(df_large)} rows from {input_filepath_large}")
    df_small = _filter(df_small, "small-window")
    df_large = _filter(df_large, "large-window")

    # Priority column: higher wins conflicts.
    df_small = df_small.copy(); df_small['_ws'] = 1
    df_large = df_large.copy(); df_large['_ws'] = 2

    merged = pd.concat([df_large, df_small], ignore_index=True)
    print(f"Concatenated: {len(merged)} rows. Running {method} dedup at overlap={overlap}...")

    # Remember where every row came from, so the outlines follow the rows
    # the dedup keeps and the new clast_IDs.
    merged['_src'] = [1 if ws == 2 else 0 for ws in merged['_ws']]
    merged['_src_id'] = (merged['clast_ID'] if 'clast_ID' in merged.columns
                         else pd.Series([None] * len(merged)))

    deduped = dedup_clasts(merged, method=method, overlap=overlap,
                           priority_col='_ws', n_points=n_points)

    src_pairs = list(zip(deduped['_src'], deduped['_src_id']))
    deduped = deduped.drop(columns=['_ws', '_src', '_src_id'])
    deduped['clast_ID'] = range(1, len(deduped) + 1)
    standard = [c for c in _CLAST_COLUMNS if c in deduped.columns]
    extras = [c for c in deduped.columns if c not in _CLAST_COLUMNS]
    deduped = deduped[standard + extras]

    print(f"Result: {len(deduped)} rows after dedup "
          f"({len(merged) - len(deduped)} duplicates removed)")
    deduped.to_csv(output_filepath, index=False, float_format='%.5f')
    print(f"Wrote {output_filepath}")
    _carry_contours(input_filepath_small, input_filepath_large,
                    output_filepath, src_pairs)
    return deduped


def _carry_contours(input_small, input_large, output_filepath, src_pairs):
    """Write ``<merged stem>.contours.json`` holding the outlines of the kept
    rows under their new clast_IDs (1..n in output order). ``src_pairs[i]``
    is ``(0 small | 1 large, original clast_ID)`` of output row i. Rows whose
    input had no sidecar get no outline; when neither input had one, no
    sidecar is written and a stale one is removed. Never raises."""
    try:
        from functions import clast_geometry as CG
        sources = [CG.read_contours(input_small), CG.read_contours(input_large)]
        if sources[0] is None and sources[1] is None:
            stale = CG.contours_path_for(output_filepath)
            if stale.exists():
                stale.unlink()
            return None
        frame = next(getattr(c, "frame", "pixels") for c in sources
                     if c is not None)
        out = {}
        for new_id, (src, old_id) in enumerate(src_pairs, start=1):
            table = sources[int(src)]
            if table is None:
                continue
            try:
                key = int(old_id)
            except (TypeError, ValueError):
                continue
            if key in table:
                out[new_id] = table[key]
        return CG.write_contours(output_filepath, out, frame=frame)
    except Exception:
        return None
