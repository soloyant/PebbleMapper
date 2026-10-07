"""Compare a detection CSV against a ground-truth CSV.

Pure functions on pandas DataFrames; no I/O.

1. pair_csvs: mutual nearest-neighbour matching within a tolerance; returns
   paired and unmatched indices on either side.
2. compute_distribution_stats: K-S test, Q-Q points, percentile summary and
   Folk-Ward moments in phi, on the full populations.
3. compute_paired_stats: linear regression, RMSE, MAE, R^2, Bland-Altman data
   on the matched pairs.
4. detection_metrics: recall / precision / F1 from the pair_csvs output.
"""
import numpy as np
from scipy import stats as _scipy_stats
from scipy.spatial import cKDTree

# Used only to decide whether a field is a linear size for which phi (Krumbein)
# and Folk-Ward statistics make physical sense.
from functions.units import (
    is_size_field_for_phi,
    folk_ward_sorting_phi,
    folk_ward_skewness_phi,
    folk_ward_kurtosis_phi,
)


# --- Spatial pairing ---
def pair_csvs(truth_df, detect_df,
              x_col="x", y_col="y", tolerance=None, size_col=None):
    """Match each truth row to at most one detection row by spatial proximity.

    Mutual nearest-neighbour matching: truth -> detection and detection ->
    truth must agree, so a detection is counted at most once.

    Parameters
    ----------
    truth_df, detect_df : pandas.DataFrame
        Must contain `x_col`, `y_col` in the same CRS / units.
    tolerance : float or None
        Maximum distance (x/y units) for a pair. None means half the median
        nearest-neighbour distance in `truth_df`.
    size_col : str or None
        If given, a candidate pair is rejected when the sizes differ by more
        than 50%.

    Returns
    -------
    dict with keys:
        matched_pairs : list of (truth_idx, detect_idx) tuples
        unmatched_truth : list of truth row indices (false negatives)
        unmatched_detect : list of detection row indices (false positives)
        used_tolerance : float (the value actually used)
    """
    if len(truth_df) == 0 or len(detect_df) == 0:
        return {
            "matched_pairs": [],
            "unmatched_truth": list(range(len(truth_df))),
            "unmatched_detect": list(range(len(detect_df))),
            "used_tolerance": tolerance or 0.0,
        }

    truth_xy = truth_df[[x_col, y_col]].to_numpy(dtype=float)
    detect_xy = detect_df[[x_col, y_col]].to_numpy(dtype=float)

    # Clasts pack at ~1 diameter spacing, so half the typical spacing is a
    # safe matching radius without conflating neighbours.
    if tolerance is None:
        if len(truth_df) >= 2:
            tree_t = cKDTree(truth_xy)
            d_nn, _ = tree_t.query(truth_xy, k=2)
            tolerance = float(np.median(d_nn[:, 1]) * 0.5)
        else:
            tolerance = 0.05  # 5cm fallback for a single-truth-point edge case

    tree_d = cKDTree(detect_xy)
    tree_t = cKDTree(truth_xy)
    d_t2d, idx_t2d = tree_d.query(truth_xy, k=1)   # for each truth, closest detection
    d_d2t, idx_d2t = tree_t.query(detect_xy, k=1)  # for each detection, closest truth

    # Mutual best: idx_t2d[i] == j and idx_d2t[j] == i and d_t2d[i] <= tolerance.
    matched_pairs = []
    truth_matched = np.zeros(len(truth_df), dtype=bool)
    detect_matched = np.zeros(len(detect_df), dtype=bool)
    for i in range(len(truth_df)):
        j = int(idx_t2d[i])
        if d_t2d[i] > tolerance:
            continue
        if int(idx_d2t[j]) != i:
            continue
        if size_col is not None:
            s_t = float(truth_df.iloc[i][size_col])
            s_d = float(detect_df.iloc[j][size_col])
            if s_t > 0 and s_d > 0:
                ratio = max(s_t, s_d) / min(s_t, s_d)
                if ratio > 1.5:    # >50% size mismatch
                    continue
        matched_pairs.append((i, j))
        truth_matched[i] = True
        detect_matched[j] = True

    return {
        "matched_pairs": matched_pairs,
        "unmatched_truth": np.flatnonzero(~truth_matched).tolist(),
        "unmatched_detect": np.flatnonzero(~detect_matched).tolist(),
        "used_tolerance": float(tolerance),
    }


# --- Distribution comparison ---
def _phi(D_meters):
    """Convert grain size in meters to Krumbein phi units (D₀ = 1 mm)."""
    D_mm = np.asarray(D_meters, dtype=float) * 1000.0
    D_mm = D_mm[D_mm > 0]   # log of zero/negative is undefined
    return -np.log2(D_mm)


def _folk_skewness_phi(phi_vals):
    """Folk-Ward graphic skewness Sk_phi, range -1..+1 (negative = coarse tail
    heavy). NaN on samples too small to populate the percentiles."""
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_vals = phi_vals[np.isfinite(phi_vals)]
    if phi_vals.size < 5:
        return float("nan")
    p5, p16, p50, p84, p95 = np.percentile(phi_vals, [5, 16, 50, 84, 95])
    return folk_ward_skewness_phi(p5, p16, p50, p84, p95)


def _folk_kurtosis_phi(phi_vals):
    """Folk-Ward graphic kurtosis K_G (< 0.67 very platykurtic, ~1.0
    mesokurtic, > 1.5 very leptokurtic). NaN on degenerate inputs."""
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_vals = phi_vals[np.isfinite(phi_vals)]
    if phi_vals.size < 5:
        return float("nan")
    p5, p25, p75, p95 = np.percentile(phi_vals, [5, 25, 75, 95])
    return folk_ward_kurtosis_phi(p5, p25, p75, p95)


def _folk_sorting_phi(phi_vals):
    """Folk-Ward (1957) graphic sorting sigma_phi. NaN below 5 values."""
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_vals = phi_vals[np.isfinite(phi_vals)]
    if phi_vals.size < 5:
        return float("nan")
    p5, p16, p84, p95 = np.percentile(phi_vals, [5, 16, 84, 95])
    return folk_ward_sorting_phi(p5, p16, p84, p95)


def compute_distribution_stats(truth_vals, detect_vals, field=None):
    """Summary stats + K-S test for two independent samples.

    Parameters
    ----------
    truth_vals, detect_vals : array-like
        Field values. For a size field these are metres (converted to mm for
        the Krumbein phi axis); other fields may be in any consistent unit.
    field : str, optional
        Name of the field. When supplied, the phi block (mean_phi,
        sorting_phi, Folk-Ward moments, K-S on phi) runs only for linear-size
        fields; other fields get NaN there, since phi of an orientation or an
        area is meaningless. When ``None`` the phi block always runs.

    Returns
    -------
    dict with:
        truth_n, detect_n : sample sizes
        truth_d50, detect_d50, truth_d84, detect_d84, truth_d16, detect_d16
            (input units) and the delta_* differences (detect minus truth)
        truth_mean_phi, detect_mean_phi, truth_sorting_phi, detect_sorting_phi,
            *_skewness_phi, *_kurtosis_phi : NaN for non-size fields
        ks_statistic, ks_p_value : two-sample K-S on phi values for size
            fields, on raw values otherwise
        qq_truth_quantiles, qq_detect_quantiles : 101 matched quantiles
            (percentiles 0..100), input units
        phi_meaningful : bool
    """
    t = np.asarray(truth_vals, dtype=float)
    d = np.asarray(detect_vals, dtype=float)
    # Positive values are required only for size fields: phi is undefined for
    # D <= 0, but an orientation can be 0 and a signed ratio negative.
    phi_safe = (field is None) or is_size_field_for_phi(field)
    if phi_safe:
        t = t[np.isfinite(t) & (t > 0)]
        d = d[np.isfinite(d) & (d > 0)]
    else:
        t = t[np.isfinite(t)]
        d = d[np.isfinite(d)]

    out = {
        "truth_n": int(len(t)),
        "detect_n": int(len(d)),
        "phi_meaningful": bool(phi_safe),
    }
    if len(t) == 0 or len(d) == 0:
        for k in ("truth_d50 detect_d50 truth_d84 detect_d84 truth_d16 detect_d16 "
                  "truth_mean_phi detect_mean_phi truth_sorting_phi "
                  "detect_sorting_phi ks_statistic ks_p_value").split():
            out[k] = float("nan")
        out["qq_truth_quantiles"] = np.array([])
        out["qq_detect_quantiles"] = np.array([])
        return out

    out["truth_d50"] = float(np.percentile(t, 50))
    out["detect_d50"] = float(np.percentile(d, 50))
    out["truth_d84"] = float(np.percentile(t, 84))
    out["detect_d84"] = float(np.percentile(d, 84))
    out["truth_d16"] = float(np.percentile(t, 16))
    out["detect_d16"] = float(np.percentile(d, 16))
    out["delta_d50"] = out["detect_d50"] - out["truth_d50"]
    out["delta_d84"] = out["detect_d84"] - out["truth_d84"]
    out["delta_d16"] = out["detect_d16"] - out["truth_d16"]

    # Keys are populated (NaN) for non-size fields so the schema is identical.
    if phi_safe:
        t_phi = _phi(t)
        d_phi = _phi(d)
        out["truth_mean_phi"] = float(np.mean(t_phi))
        out["detect_mean_phi"] = float(np.mean(d_phi))
        out["truth_sorting_phi"]  = _folk_sorting_phi(t_phi)
        out["detect_sorting_phi"] = _folk_sorting_phi(d_phi)
        # Graphic (percentile-based) Folk-Ward moments are robust to bin-edge
        # artefacts at small N.
        out["truth_skewness_phi"] = _folk_skewness_phi(t_phi)
        out["detect_skewness_phi"] = _folk_skewness_phi(d_phi)
        out["truth_kurtosis_phi"] = _folk_kurtosis_phi(t_phi)
        out["detect_kurtosis_phi"] = _folk_kurtosis_phi(d_phi)
        # K-S is rank-based, so the phi transform does not change the result.
        ks_samples_t, ks_samples_d = t_phi, d_phi
    else:
        out["truth_mean_phi"] = float("nan")
        out["detect_mean_phi"] = float("nan")
        out["truth_sorting_phi"] = float("nan")
        out["detect_sorting_phi"] = float("nan")
        out["truth_skewness_phi"] = float("nan")
        out["detect_skewness_phi"] = float("nan")
        out["truth_kurtosis_phi"] = float("nan")
        out["detect_kurtosis_phi"] = float("nan")
        ks_samples_t, ks_samples_d = t, d

    if len(ks_samples_t) >= 2 and len(ks_samples_d) >= 2:
        ks = _scipy_stats.ks_2samp(ks_samples_t, ks_samples_d)
        out["ks_statistic"] = float(ks.statistic)
        out["ks_p_value"] = float(ks.pvalue)
    else:
        out["ks_statistic"] = float("nan")
        out["ks_p_value"] = float("nan")

    qs = np.linspace(0, 100, 101)
    out["qq_truth_quantiles"] = np.percentile(t, qs)
    out["qq_detect_quantiles"] = np.percentile(d, qs)
    return out


# --- Paired stats ---
def compute_paired_stats(truth_paired, detect_paired):
    """Per-pair stats: linear regression, RMSE, MAE, R², Bland-Altman.

    truth_paired, detect_paired : 1D arrays of equal length, paired by index.
    Both should be in the same units (typically meters).

    Returns dict with:
        n : number of pairs
        slope, intercept : linear regression detect = a*truth + b
        r_squared : Pearson r^2
        rmse, mae : in input units
        bias : mean(detect - truth)
        bland_altman_mean, bland_altman_diff : per-pair, for plotting
        bland_altman_loa_lo, bland_altman_loa_hi : ±1.96σ limits of agreement
    """
    t = np.asarray(truth_paired, dtype=float)
    d = np.asarray(detect_paired, dtype=float)
    keep = np.isfinite(t) & np.isfinite(d) & (t > 0) & (d > 0)
    t = t[keep]
    d = d[keep]

    out = {"n": int(len(t))}
    if len(t) < 2:
        for k in ("slope intercept r_squared rmse mae bias "
                  "bland_altman_loa_lo bland_altman_loa_hi").split():
            out[k] = float("nan")
        out["bland_altman_mean"] = np.array([])
        out["bland_altman_diff"] = np.array([])
        return out

    slope, intercept, r_value, _, _ = _scipy_stats.linregress(t, d)
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)
    out["r_squared"] = float(r_value ** 2)

    err = d - t
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))
    out["bias"] = float(np.mean(err))

    # Bland-Altman: x = mean of pair, y = difference. LoA = bias ± 1.96σ.
    mean_pair = (t + d) / 2.0
    out["bland_altman_mean"] = mean_pair
    out["bland_altman_diff"] = err
    sigma = float(np.std(err, ddof=1)) if len(err) > 1 else 0.0
    out["bland_altman_loa_lo"] = out["bias"] - 1.96 * sigma
    out["bland_altman_loa_hi"] = out["bias"] + 1.96 * sigma
    return out


# --- Detection performance ---
def detection_metrics(pair_result):
    """Recall / precision / F1 from a pair_csvs() output."""
    n_matched = len(pair_result["matched_pairs"])
    n_truth = n_matched + len(pair_result["unmatched_truth"])
    n_detect = n_matched + len(pair_result["unmatched_detect"])

    out = {
        "n_matched": n_matched,
        "n_truth": n_truth,
        "n_detect": n_detect,
        "false_negatives": len(pair_result["unmatched_truth"]),
        "false_positives": len(pair_result["unmatched_detect"]),
    }
    out["recall"] = float(n_matched / n_truth) if n_truth > 0 else float("nan")
    out["precision"] = float(n_matched / n_detect) if n_detect > 0 else float("nan")
    if out["recall"] > 0 and out["precision"] > 0:
        out["f1"] = float(
            2 * out["recall"] * out["precision"]
            / (out["recall"] + out["precision"])
        )
    else:
        out["f1"] = float("nan")
    return out


# --- Batch summary across a quadrat queue ---
# Each validation job compares one quadrat (truth = quadrat photograph reference,
# detect = UAV detection); the batch summary aggregates the completed jobs.


def _batch_qc_text(points) -> str:
    """Data-driven QC discussion from the per-quadrat summary points."""
    n = len(points)
    if n == 0:
        return "No completed quadrats to summarise."

    def _arr(key):
        return np.array([p[key] for p in points
                         if p.get(key) is not None
                         and np.isfinite(p.get(key, np.nan))], dtype=float)

    rec, pre = _arr("recall"), _arr("precision")
    mt = np.array([p["mean_truth"] for p in points], dtype=float)
    md = np.array([p["mean_detect"] for p in points], dtype=float)
    ok = np.isfinite(mt) & np.isfinite(md)
    lines = [f"**Quality control — {n} quadrat(s)** compared "
             "(quadrat truth vs UAV detection)."]
    if rec.size:
        lines.append(
            f"Detection recall averages {np.mean(rec):.2f} "
            f"(range {np.min(rec):.2f}–{np.max(rec):.2f})"
            + (f"; precision averages {np.mean(pre):.2f}." if pre.size else ".")
            + " Recall is the fraction of ground-truth clasts the UAV matched; "
            "the model is deliberately conservative (it keeps only fully-"
            "visible grains), so a modest recall does not by itself imply poor "
            "grain-size mapping.")
    if ok.sum() >= 2:
        r = float(np.corrcoef(mt[ok], md[ok])[0, 1])
        rel = (md[ok] - mt[ok]) / mt[ok]
        bias = float(np.mean(rel) * 100.0)
        lines.append(
            f"Across quadrats, UAV mean grain size tracks the quadrat mean "
            f"with r = {r:.2f}; the UAV mean is on average {bias:+.1f}% "
            "relative to the quadrat truth. A consistent offset indicates a "
            "size-dependent detection bias rather than random scatter.")
    lines.append(
        "Interpretation & limitations: the RMSE-vs-size plot shows whether "
        "per-clast error grows with grain size; the per-quadrat violins (raw "
        "vs filtered) show how excluding sub-detection-limit objects tightens "
        "agreement between the two methods. Residual disagreement is dominated "
        "by (1) the conservative detector under-sampling partially-buried / "
        "overlapping clasts, (2) the finite UAV ground sampling distance "
        "truncating the fine tail, and (3) co-registration error between the "
        "quadrat and UAV frames.")
    return "\n\n".join(lines)


def compute_batch_summary(jobs) -> dict:
    """Aggregate completed validation jobs into batch-level QC data.

    Each ``job`` is the GUI's job dict; it must carry ``status == "done"``,
    a ``metrics`` dict, and a ``sizes`` dict with ``truth_raw`` / ``detect_raw``
    / ``truth_filt`` / ``detect_filt`` value lists (quadrat truth vs UAV, before
    and after the sub-D_min filter). Returns per-quadrat points + violin
    arrays + a QC discussion string.
    """
    from pathlib import Path as _Path
    points, violins_raw, violins_filt = [], [], []
    for j in jobs:
        if j.get("status") != "done":
            continue
        m = j.get("metrics") or {}
        sz = j.get("sizes") or {}
        tr = np.asarray(sz.get("truth_filt") or [], dtype=float)
        de = np.asarray(sz.get("detect_filt") or [], dtype=float)
        tr_raw = np.asarray(sz.get("truth_raw") or [], dtype=float)
        de_raw = np.asarray(sz.get("detect_raw") or [], dtype=float)
        label = (j.get("quad_label")
                 or _Path(str(j.get("truth", ""))).stem[:18] or "quad")
        points.append({
            "label": label,
            "mean_truth": float(np.mean(tr)) if tr.size else float("nan"),
            "mean_detect": float(np.mean(de)) if de.size else float("nan"),
            "rmse": m.get("rmse"),
            "recall": m.get("recall"),
            "precision": m.get("precision"),
            "d50_truth": m.get("truth_D50"),
            "d50_detect": m.get("detect_D50"),
            "n_truth": m.get("n_truth"),
            "n_detect": m.get("n_detect"),
        })
        violins_raw.append((label, tr_raw, de_raw))
        violins_filt.append((label, tr, de))
    return {"points": points, "violins_raw": violins_raw,
            "violins_filt": violins_filt, "qc_text": _batch_qc_text(points),
            "n_quadrats": len(points)}
