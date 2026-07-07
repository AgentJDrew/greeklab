"""Tests for greeklab.monte_carlo.

Covers: (a) MC price converges to closed-form Black-Scholes within a
handful of standard errors, seeded for reproducibility; (b) variance
reduction (antithetic + control variate) genuinely lowers the standard
error versus plain MC; (c) basic input validation.
"""

from __future__ import annotations

import pytest

from greeklab.black_scholes import bs_price
from greeklab.monte_carlo import mc_european_price

# How many standard errors the MC estimate is allowed to be away from
# the true BS price before we call it a failure. With a fixed seed
# this is deterministic per run; 4 SE is an extremely generous bound
# (a Gaussian CLT-based 4-sigma event has probability ~6e-5) chosen to
# make the test robust to seed/library-version changes while still
# being a meaningful correctness check.
_MAX_SE_AWAY = 4.0


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize(("spot", "strike"), [(90.0, 100.0), (100.0, 100.0), (110.0, 100.0)])
def test_mc_converges_to_black_scholes(opt: str, spot: float, strike: float) -> None:
    rate, q, sigma, t = 0.05, 0.02, 0.25, 1.0
    bs = float(bs_price(spot, strike, rate, q, sigma, t, opt))
    result = mc_european_price(spot, strike, rate, q, sigma, t, opt, n_paths=200_000, seed=1234)
    n_se_away = abs(result.price - bs) / result.std_error
    assert n_se_away < _MAX_SE_AWAY


def test_variance_reduction_lowers_standard_error() -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0
    with_vr = mc_european_price(
        spot, strike, rate, q, sigma, t, "call", n_paths=100_000, seed=42, antithetic=True, control_variate=True
    )
    without_vr = mc_european_price(
        spot, strike, rate, q, sigma, t, "call", n_paths=100_000, seed=42, antithetic=False, control_variate=False
    )
    assert with_vr.std_error < without_vr.std_error


def test_antithetic_alone_lowers_standard_error() -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.0, 0.3, 1.0
    with_antithetic = mc_european_price(
        spot, strike, rate, q, sigma, t, "call", n_paths=50_000, seed=7, antithetic=True, control_variate=False
    )
    without = mc_european_price(
        spot, strike, rate, q, sigma, t, "call", n_paths=50_000, seed=7, antithetic=False, control_variate=False
    )
    assert with_antithetic.std_error < without.std_error


def test_seeded_result_is_reproducible() -> None:
    args = (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    r1 = mc_european_price(*args, n_paths=10_000, seed=99)
    r2 = mc_european_price(*args, n_paths=10_000, seed=99)
    assert r1.price == r2.price
    assert r1.std_error == r2.std_error


def test_standard_error_shrinks_as_sqrt_n_paths() -> None:
    args = (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    small = mc_european_price(*args, n_paths=10_000, seed=1, antithetic=False, control_variate=False)
    large = mc_european_price(*args, n_paths=40_000, seed=1, antithetic=False, control_variate=False)
    # Quadrupling n_paths should roughly halve the standard error
    # (SE ~ 1/sqrt(n)); allow generous slack since this is itself a
    # Monte Carlo quantity.
    ratio = small.std_error / large.std_error
    assert 1.5 < ratio < 2.5


def test_invalid_n_paths_raises() -> None:
    with pytest.raises(ValueError):
        mc_european_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", n_paths=0)


def test_nonpositive_spot_raises() -> None:
    with pytest.raises(ValueError):
        mc_european_price(0.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
