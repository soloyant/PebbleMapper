"""Detection-limit truncation analytics.

A UAV detector cannot reliably segment grains below ``k * GSD`` on the long
axis (Soloy et al. 2020: 4 cm at 5 mm/pixel, i.e. k ~ 8 pixels), so a
higher-resolution quadrat truth carries a fine tail the UAV cannot
recover and any whole-distribution comparison is biased by that
left-truncation. Pairwise comparisons stay valid because only
mutually-visible grains pair.

This module computes the explicit threshold ``D_min = k * GSD``, truncates
both sides to ``P(D | D >= D_min)``, and draws two diagnostics: a
complementary CDF on log-log axes and an empirical detection function
(recall vs size bin).

Reference
---------
Soloy, A., Turki, I., Fournier, M., Costa, S., Peuziat, B., & Lecoq, N.
(2020). A Deep Learning-Based Method for Quantifying and Mapping the
Grain Size on Pebble Beaches. *Remote Sensing*, 12(21), 3659.
https://doi.org/10.3390/rs12213659
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from functions.units import field_display


def _humanize_size_label(field_name: str) -> str:
    """Axis label for a size column, e.g. ``Clast_length`` -> ``"Clast length [mm]"``.
    The truncation plots scale metres to mm, so the unit is fixed."""
    disp = field_display(field_name)[0] or \
        str(field_name).replace("_", " ").strip() or str(field_name)
    return f"{disp} [mm]"


# Multiplier on the UAV GSD giving the minimum reliably-detectable long axis
# (Soloy et al. 2020: 4 cm at 5 mm/pixel -> 8 pixels).
DEFAULT_K_PIXELS: int = 8


def compute_d_min(gsd_m: float, k_pixels: int = DEFAULT_K_PIXELS) -> float:
    """Detection threshold ``D_min = k_pixels * gsd_m`` in metres.

    ``gsd_m`` is the UAV ortho ground-sampling distance in metres per pixel.
    """
    return float(k_pixels) * float(gsd_m)


def truncate_above_dmin(values: Sequence[float],
                         d_min: float) -> np.ndarray:
    """Return only values ``≥ d_min`` as a finite-only numpy array."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= d_min)]
    return arr


# --- Detection function (recall vs size bin) ---

@dataclass
class DetectionFunctionFit:
    """Result of fitting a logistic detection function to (size, was_detected).

    ``d50`` is the size at which fitted detection probability reaches 50%
    (an effective detection threshold independent of bin choice); ``slope``
    is the logistic steepness. The ``bin_*`` arrays carry the empirical
    binned recall the fit was made from.
    """
    d50: float
    slope: float
    bin_sizes: np.ndarray   # bin centres (metres)
    bin_recall: np.ndarray  # recall fraction per bin
    bin_counts: np.ndarray  # truth-clast count per bin
    fit_ok: bool = True
    # Which estimator produced d50: the SciPy logistic fit and the closed-form
    # logit regression are different methods.
    method: str = "logistic"


def _logistic(x, mid, k):
    """Standard logistic. Used for the detection-function fit only."""
    return 1.0 / (1.0 + np.exp(-k * (x - mid)))


def detection_function_from_pair(
    truth_sizes: Sequence[float],
    matched_truth_idx: Sequence[int],
    *,
    n_bins: int = 10,
    min_bin_count: int = 3,
) -> DetectionFunctionFit:
    """Empirical recall vs grain-size bin + logistic fit.

    Bins ``truth_sizes`` into ``n_bins`` log-spaced bins (grain-size
    distributions are heavy-tailed) and computes ``recall = matched / total``
    per bin, then fits a logistic via scipy.optimize.curve_fit, falling back
    to a least-squares logit fit when SciPy is unavailable.

    Parameters
    ----------
    truth_sizes
        Grain-size field values for every truth clast (metres).
    matched_truth_idx
        Indices into ``truth_sizes`` of truth clasts paired to a detection.
    n_bins
        Number of log-spaced size bins.
    min_bin_count
        Bins with fewer truth clasts are dropped from the fit to avoid
        spurious 0% / 100% recalls.
    """
    sizes = np.asarray(truth_sizes, dtype=float)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if sizes.size < n_bins * min_bin_count:
        return DetectionFunctionFit(
            d50=float("nan"), slope=float("nan"),
            bin_sizes=np.array([]), bin_recall=np.array([]),
            bin_counts=np.array([]), fit_ok=False, method="none",
        )

    matched_mask = np.zeros(len(truth_sizes), dtype=bool)
    for idx in matched_truth_idx:
        if 0 <= int(idx) < len(matched_mask):
            matched_mask[int(idx)] = True

    # Apply the same finite/positive filter to matched_mask.
    finite_mask = np.isfinite(np.asarray(truth_sizes, dtype=float)) & \
                  (np.asarray(truth_sizes, dtype=float) > 0)
    matched_mask = matched_mask[finite_mask]

    s_min = float(np.percentile(sizes, 1))
    s_max = float(np.percentile(sizes, 99))
    if s_max <= s_min:
        return DetectionFunctionFit(
            d50=float("nan"), slope=float("nan"),
            bin_sizes=np.array([]), bin_recall=np.array([]),
            bin_counts=np.array([]), fit_ok=False, method="none",
        )
    edges = np.geomspace(s_min, s_max, n_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])

    counts = np.zeros(n_bins, dtype=int)
    recalls = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        in_bin = (sizes >= edges[i]) & (sizes < edges[i + 1])
        n = int(in_bin.sum())
        counts[i] = n
        if n == 0:
            recalls[i] = float("nan")
        else:
            recalls[i] = float(matched_mask[in_bin].mean())

    valid = (counts >= min_bin_count) & np.isfinite(recalls)
    if valid.sum() < 3:
        return DetectionFunctionFit(
            d50=float("nan"), slope=float("nan"),
            bin_sizes=centres, bin_recall=recalls,
            bin_counts=counts, fit_ok=False,
        )

    fit_x = centres[valid]
    fit_y = np.clip(recalls[valid], 1e-3, 1 - 1e-3)
    try:
        from scipy.optimize import curve_fit
        p0 = [float(np.median(fit_x)), 50.0 / max(fit_x.max(), 1e-6)]
        popt, _ = curve_fit(_logistic, fit_x, fit_y, p0=p0,
                              maxfev=5000)
        d50, slope = float(popt[0]), float(popt[1])
        return DetectionFunctionFit(
            d50=d50, slope=slope,
            bin_sizes=centres, bin_recall=recalls,
            bin_counts=counts, fit_ok=True, method="logistic",
        )
    except Exception:
        # Closed-form logit regression: a different estimator, recorded in .method.
        logit = np.log(fit_y / (1.0 - fit_y))
        # logit(p) = slope * (x - mid) = slope*x - slope*mid
        A = np.vstack([fit_x, np.ones_like(fit_x)]).T
        try:
            m, b = np.linalg.lstsq(A, logit, rcond=None)[0]
            slope = float(m)
            d50 = float(-b / m) if m != 0 else float("nan")
        except Exception:
            return DetectionFunctionFit(
                d50=float("nan"), slope=float("nan"),
                bin_sizes=centres, bin_recall=recalls,
                bin_counts=counts, fit_ok=False, method="none",
            )
        return DetectionFunctionFit(
            d50=d50, slope=slope,
            bin_sizes=centres, bin_recall=recalls,
            bin_counts=counts, fit_ok=True, method="logit-regression",
        )


# --- Plot helpers ---

def plot_ccdf_log_log(
    truth_vals: Sequence[float],
    detect_vals: Sequence[float],
    *,
    field_name: str,
    d_min: Optional[float],
    truth_label: str = "Truth",
    detect_label: str = "Detection",
    figsize: tuple[float, float] = (6.5, 4.5),
):
    """Complementary-CDF survival plot on log-log axes.

    For heavy-tailed grain-size distributions power-law-like tails plot as
    straight lines, and truncation shows as a sharp divergence between the
    two curves below ``d_min``. Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)

    def _ccdf(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return np.array([]), np.array([])
        a_sorted = np.sort(a)
        # Survival = (n - rank + 1) / n, i.e. values 1/n .. 1.
        n = a_sorted.size
        ccdf = 1.0 - np.arange(n) / float(n)
        return a_sorted, ccdf

    tx, ty = _ccdf(truth_vals)
    dx, dy = _ccdf(detect_vals)

    # Split the detection data at D_min so the truncated portion is drawn distinctly.
    has_dmin = (d_min is not None and np.isfinite(d_min) and d_min > 0)
    if has_dmin and dx.size:
        d_arr = np.asarray(detect_vals, dtype=float)
        d_arr = d_arr[np.isfinite(d_arr) & (d_arr > 0)]
        d_above = d_arr[d_arr >= d_min]
        d_below = d_arr[d_arr < d_min]
        dabx, daby = _ccdf(d_above)
        dblx, dbly = _ccdf(d_below)
    else:
        dabx, daby = dx, dy
        dblx, dbly = np.array([]), np.array([])

    if tx.size:
        ax.loglog(tx * 1000.0, ty, "-", linewidth=1.6,
                   color="#225599", label=f"{truth_label} (n={tx.size})")
    if dx.size:
        ax.loglog(dx * 1000.0, dy, "-", linewidth=1.6,
                   color="#cc3300", label=f"{detect_label} — full (n={dx.size})")
    if has_dmin and dabx.size:
        ax.loglog(dabx * 1000.0, daby, "-", linewidth=2.4,
                   color="#cc3300", alpha=0.9,
                   label=f"  ≥ D_min (n={dabx.size}, used in analysis)")
    if has_dmin and dblx.size:
        ax.loglog(dblx * 1000.0, dbly, "--", linewidth=1.4,
                   color="#e07050", alpha=0.6,
                   label=f"  < D_min (n={dblx.size}, excluded)")

    # Shade the below-threshold region and keep the D_min line for the exact boundary.
    if has_dmin:
        all_x = np.concatenate([tx * 1000.0, dx * 1000.0])
        all_x = all_x[all_x > 0]
        if all_x.size:
            x_left = max(all_x.min() * 0.5, 1e-3)
            ax.axvspan(x_left, d_min * 1000.0,
                        color="#dddddd", alpha=0.35,
                        zorder=0,
                        label="UAV detection limit\n(below threshold)")
        ax.axvline(d_min * 1000.0, linestyle="--", color="#555555",
                    linewidth=1.2,
                    label=f"D_min = {d_min*1000.0:.1f} mm")
    ax.set_xlabel(_humanize_size_label(field_name))
    ax.set_ylabel("1 − F(D)   (survival probability)")
    ax.set_title("Complementary CDF (log–log)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_detection_function(
    fit: DetectionFunctionFit,
    *,
    d_min: Optional[float],
    field_name: str = "Clast_length",
    figsize: tuple[float, float] = (6.5, 4.5),
):
    """Empirical recall vs size bin with the fitted logistic, a 0.9 recall
    reference and the assumed ``d_min`` against the fitted ``d50``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    if fit.bin_sizes.size:
        # Marker size grows with bin count.
        sizes_mm = fit.bin_sizes * 1000.0
        sizes_for_marker = (np.sqrt(fit.bin_counts) * 6.0).clip(8.0, 60.0)
        finite = np.isfinite(fit.bin_recall)
        ax.scatter(sizes_mm[finite], fit.bin_recall[finite],
                    s=sizes_for_marker[finite],
                    facecolor="#5577aa", edgecolor="white",
                    linewidth=1.0,
                    label="Empirical recall (size ∝ √N truth)")
    if fit.fit_ok and np.isfinite(fit.d50) and np.isfinite(fit.slope):
        if fit.bin_sizes.size:
            xx = np.geomspace(
                max(fit.bin_sizes[0] * 0.5, 1e-6),
                fit.bin_sizes[-1] * 2.0, 200)
        else:
            xx = np.geomspace(max(fit.d50 * 0.1, 1e-6), fit.d50 * 5.0, 200)
        yy = _logistic(xx, fit.d50, fit.slope)
        ax.semilogx(xx * 1000.0, yy, "-",
                     color="#cc3300", linewidth=1.6,
                     label=f"Logistic fit  (d50 = {fit.d50*1000:.1f} mm)")
    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1.0,
                label="90% recall reference line")
    if d_min is not None and np.isfinite(d_min) and d_min > 0:
        ax.axvline(d_min * 1000.0, linestyle="--", color="#555555",
                    linewidth=1.2,
                    label=f"Assumed D_min = {d_min*1000.0:.1f} mm")
    # A fitted d50 that disagrees with the assumed D_min is the point of the plot.
    if fit.fit_ok and np.isfinite(fit.d50):
        d50_mm = fit.d50 * 1000.0
        ax.axvline(d50_mm, linestyle=":", color="#cc3300",
                    linewidth=1.2,
                    label=f"Fitted d50 (50% recall) = {d50_mm:.1f} mm")
    ax.set_xscale("log")
    ax.set_xlabel(_humanize_size_label(field_name))
    ax.set_ylabel("Empirical UAV recall   (matched / truth per bin)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Detection function — UAV recall vs grain size",
                  fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_K_PIXELS",
    "DetectionFunctionFit",
    "compute_d_min",
    "truncate_above_dmin",
    "detection_function_from_pair",
    "plot_ccdf_log_log",
    "plot_detection_function",
]
