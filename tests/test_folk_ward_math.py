"""Spot-checks for the Folk-Ward grain-size statistics functions.

Covers _phi, _folk_skewness_phi, _folk_kurtosis_phi, _folk_sorting_phi
against known reference values from Folk & Ward (1957) and the Krumbein
phi scale definition.
"""
import numpy as np
import pytest

from functions.validation import (
    _phi,
    _folk_skewness_phi,
    _folk_kurtosis_phi,
    _folk_sorting_phi,
)


# ---------------------------------------------------------------------------
# _phi: Krumbein scale reference values
# ---------------------------------------------------------------------------

def test_phi_1mm_is_zero():
    assert abs(_phi(np.array([0.001])).item() - 0.0) < 1e-10


def test_phi_2mm_is_minus_one():
    assert abs(_phi(np.array([0.002])).item() - (-1.0)) < 1e-10


def test_phi_half_mm_is_plus_one():
    assert abs(_phi(np.array([0.0005])).item() - 1.0) < 1e-10


def test_phi_filters_nonpositive():
    """Zero and negative diameters must be dropped silently."""
    result = _phi(np.array([0.0, -0.001, 0.001]))
    assert result.size == 1
    assert abs(result[0] - 0.0) < 1e-10


# ---------------------------------------------------------------------------
# _folk_sorting_phi: matches rasterize formula for same data
# ---------------------------------------------------------------------------

def test_folk_sorting_matches_rasterize_formula():
    """validation._folk_sorting_phi must equal the rasterize graphic formula."""
    sizes_m = np.array([0.030, 0.045, 0.060, 0.080, 0.100, 0.150, 0.200])
    phi = _phi(sizes_m)
    val_sort = _folk_sorting_phi(phi)
    p5, p16, p84, p95 = np.percentile(phi, [5, 16, 84, 95])
    rast_sort = (p84 - p16) / 4.0 + (p95 - p5) / 6.6
    assert abs(val_sort - rast_sort) < 1e-12


def test_folk_sorting_nan_for_small_n():
    assert np.isnan(_folk_sorting_phi(np.array([1.0, 2.0, 3.0, 4.0])))


def test_folk_sorting_five_samples_ok():
    phi = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    result = _folk_sorting_phi(phi)
    assert np.isfinite(result)
    assert result > 0


# ---------------------------------------------------------------------------
# _folk_skewness_phi
# ---------------------------------------------------------------------------

def test_folk_skewness_symmetric_distribution():
    """Uniform distribution on phi — skewness near 0."""
    phi = np.linspace(0.0, 4.0, 200)
    sk = _folk_skewness_phi(phi)
    assert abs(sk) < 0.05


def test_folk_skewness_nan_for_small_n():
    assert np.isnan(_folk_skewness_phi(np.array([1.0, 2.0])))


def test_folk_skewness_coarse_tail_negative():
    """Folk-Ward Sk_I < 0 when the distribution has a coarse tail (low-phi tail).

    In the phi scale finer grains have higher phi.  A coarse tail means the
    *bulk* of the distribution is at high phi (fine) and the tail extends to
    low phi (coarse), which pulls the mean below the median and gives Sk_I < 0.
    """
    rng = np.random.default_rng(0)
    # Bulk at high phi (fine, ~silt/sand boundary), with a coarse tail at low phi
    phi = np.concatenate([rng.normal(3.0, 0.3, 80), rng.normal(0.5, 0.3, 20)])
    sk = _folk_skewness_phi(phi)
    assert sk < 0


# ---------------------------------------------------------------------------
# _folk_kurtosis_phi
# ---------------------------------------------------------------------------

def test_folk_kurtosis_mesokurtic():
    """Normal-like phi distribution → K_G ≈ 1.0 (mesokurtic)."""
    rng = np.random.default_rng(42)
    phi = rng.normal(2.0, 0.8, 500)
    kg = _folk_kurtosis_phi(phi)
    assert 0.8 < kg < 1.2


def test_folk_kurtosis_nan_for_small_n():
    assert np.isnan(_folk_kurtosis_phi(np.array([1.0, 2.0, 3.0])))


def test_folk_kurtosis_positive():
    """K_G must always be positive for a valid sample."""
    phi = np.linspace(0.0, 4.0, 50)
    kg = _folk_kurtosis_phi(phi)
    assert kg > 0


# ---------------------------------------------------------------------------
# Cross-check: sorting, skewness, kurtosis all finite on typical gravel sample
# ---------------------------------------------------------------------------

def test_all_moments_finite_gravel_sample():
    """Typical gravel sample: D from 8 mm (phi=-3) to 128 mm (phi=-7)."""
    rng = np.random.default_rng(7)
    D_m = rng.lognormal(mean=np.log(0.032), sigma=0.5, size=200).clip(0.004, 0.256)
    phi = _phi(D_m)
    assert np.isfinite(_folk_sorting_phi(phi))
    assert np.isfinite(_folk_skewness_phi(phi))
    assert np.isfinite(_folk_kurtosis_phi(phi))
