"""Tests for greeklab.surface.

Covers: (a) an IV surface built from a BS-generated price grid recovers
the known input volatilities; (b) SVI slice fitting recovers the
smile it was fit to, and round-trips its own total-variance formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from greeklab.black_scholes import bs_price
from greeklab.surface import fit_svi_slice, iv_surface


def test_iv_surface_recovers_known_volatilities() -> None:
    spot, rate, q = 100.0, 0.05, 0.02
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expiries = np.array([0.25, 0.5, 1.0])
    true_sigmas = np.array(
        [
            [0.25, 0.22, 0.20, 0.21, 0.24],
            [0.24, 0.21, 0.19, 0.20, 0.23],
            [0.23, 0.20, 0.18, 0.19, 0.22],
        ]
    )

    price_grid = np.zeros((3, 5))
    for i, t in enumerate(expiries):
        for j, k in enumerate(strikes):
            price_grid[i, j] = float(bs_price(spot, k, rate, q, true_sigmas[i, j], t, "call"))

    recovered = iv_surface(price_grid, spot, strikes, expiries, rate, q, "call")
    assert recovered.shape == (3, 5)
    assert not np.any(np.isnan(recovered))
    np.testing.assert_allclose(recovered, true_sigmas, atol=1e-6)


def test_iv_surface_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        iv_surface(
            price_grid=np.zeros((2, 3)),
            spot=100.0,
            strikes=np.array([90.0, 100.0, 110.0]),
            expiries=np.array([0.5]),  # length 1, but price_grid has 2 rows
            rate=0.03,
            dividend_yield=0.0,
        )


def test_svi_fit_recovers_a_synthetic_smile() -> None:
    # Generate a smile from a known SVI parameterization directly (not
    # via Black-Scholes), then confirm the fitted parameters
    # reconstruct the same total-variance curve to a tight tolerance.
    true_a, true_b, true_rho, true_m, true_sigma = 0.04, 0.4, -0.3, 0.0, 0.2
    t = 1.0
    k = np.linspace(-0.5, 0.5, 25)
    true_w = true_a + true_b * (true_rho * (k - true_m) + np.sqrt((k - true_m) ** 2 + true_sigma**2))
    true_iv = np.sqrt(true_w / t)

    fit = fit_svi_slice(k, true_iv, t)
    assert fit.rmse < 1e-8

    reconstructed_iv = fit.implied_vol(k)
    np.testing.assert_allclose(reconstructed_iv, true_iv, atol=1e-4)


def test_svi_fit_requires_at_least_five_points() -> None:
    with pytest.raises(ValueError):
        fit_svi_slice(np.array([-0.1, 0.0, 0.1]), np.array([0.2, 0.19, 0.2]), time_to_expiry=1.0)


def test_svi_total_variance_is_nonnegative_near_the_fit_region() -> None:
    true_a, true_b, true_rho, true_m, true_sigma = 0.04, 0.4, -0.3, 0.0, 0.2
    t = 1.0
    k = np.linspace(-0.5, 0.5, 25)
    true_w = true_a + true_b * (true_rho * (k - true_m) + np.sqrt((k - true_m) ** 2 + true_sigma**2))
    true_iv = np.sqrt(true_w / t)
    fit = fit_svi_slice(k, true_iv, t)

    w_fitted = fit.total_variance(k)
    assert np.all(w_fitted >= 0.0)
