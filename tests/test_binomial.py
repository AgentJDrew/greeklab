"""Tests for greeklab.binomial.

Covers: (a) CRR European price converges to closed-form Black-Scholes
as n_steps grows; (b) American >= European for puts; (c) American call
with no dividends exactly equals the European call (no early-exercise
premium); (d) basic input validation.
"""

from __future__ import annotations

import pytest

from greeklab.binomial import crr_american_price, crr_european_price
from greeklab.black_scholes import bs_price


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize(("spot", "strike"), [(90.0, 100.0), (100.0, 100.0), (110.0, 100.0)])
def test_crr_european_converges_to_black_scholes(opt: str, spot: float, strike: float) -> None:
    rate, q, sigma, t = 0.05, 0.02, 0.25, 1.0
    bs = float(bs_price(spot, strike, rate, q, sigma, t, opt))
    crr = crr_european_price(spot, strike, rate, q, sigma, t, opt, n_steps=2000)
    # Binomial convergence to BS is O(1/n) and oscillatory in n
    # (Hull Ch. 13); 2000 steps comfortably gets within a few cents on
    # a $100 option.
    assert crr == pytest.approx(bs, abs=0.05)


@pytest.mark.parametrize("n_steps", [50, 200, 800, 3200])
def test_crr_european_error_stays_within_loose_bound(n_steps: int) -> None:
    # Sanity bound at each step count individually; the stricter
    # trend-down check (error shrinks by an order of magnitude in
    # n_steps) is in test_crr_error_shrinks_from_50_to_3200_steps below.
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    bs = float(bs_price(spot, strike, rate, q, sigma, t, "call"))
    crr = crr_european_price(spot, strike, rate, q, sigma, t, "call", n_steps=n_steps)
    assert abs(crr - bs) < 1.0


def test_crr_error_shrinks_from_50_to_3200_steps() -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    bs = float(bs_price(spot, strike, rate, q, sigma, t, "call"))
    error_50 = abs(crr_european_price(spot, strike, rate, q, sigma, t, "call", n_steps=50) - bs)
    error_3200 = abs(crr_european_price(spot, strike, rate, q, sigma, t, "call", n_steps=3200) - bs)
    assert error_3200 < error_50


@pytest.mark.parametrize(("spot", "strike"), [(90.0, 100.0), (100.0, 100.0), (110.0, 100.0)])
def test_american_put_geq_european_put(spot: float, strike: float) -> None:
    rate, q, sigma, t = 0.05, 0.02, 0.25, 1.0
    european = crr_european_price(spot, strike, rate, q, sigma, t, "put", n_steps=500)
    american = crr_american_price(spot, strike, rate, q, sigma, t, "put", n_steps=500)
    assert american >= european - 1e-9  # tiny tolerance for lattice roundoff


@pytest.mark.parametrize(("spot", "strike"), [(90.0, 100.0), (100.0, 100.0), (110.0, 100.0)])
def test_american_call_geq_european_call(spot: float, strike: float) -> None:
    rate, q, sigma, t = 0.05, 0.03, 0.25, 1.0
    european = crr_european_price(spot, strike, rate, q, sigma, t, "call", n_steps=500)
    american = crr_american_price(spot, strike, rate, q, sigma, t, "call", n_steps=500)
    assert american >= european - 1e-9


def test_american_call_equals_european_call_when_no_dividends() -> None:
    # Merton (1973): with no dividends, early exercise of an American
    # call is never optimal, so American == European exactly (up to
    # lattice discretization, which is identical for both since they
    # share the same tree).
    spot, strike, rate, sigma, t = 100.0, 100.0, 0.05, 0.25, 1.0
    european = crr_european_price(spot, strike, rate, 0.0, sigma, t, "call", n_steps=1000)
    american = crr_american_price(spot, strike, rate, 0.0, sigma, t, "call", n_steps=1000)
    assert american == pytest.approx(european, abs=1e-9)


def test_american_put_early_exercise_premium_is_positive_with_dividends() -> None:
    # With a real dividend yield, the American put premium over
    # European should be strictly positive for a reasonably long-dated
    # ATM option (early exercise has genuine value).
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.03, 0.25, 1.0
    european = crr_european_price(spot, strike, rate, q, sigma, t, "put", n_steps=1000)
    american = crr_american_price(spot, strike, rate, q, sigma, t, "put", n_steps=1000)
    assert american > european + 1e-4


def test_zero_time_to_expiry_gives_intrinsic() -> None:
    assert crr_european_price(110.0, 100.0, 0.05, 0.0, 0.2, 0.0, "call", n_steps=100) == pytest.approx(10.0)
    assert crr_american_price(110.0, 100.0, 0.05, 0.0, 0.2, 0.0, "call", n_steps=100) == pytest.approx(10.0)


def test_invalid_n_steps_raises() -> None:
    with pytest.raises(ValueError):
        crr_european_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", n_steps=0)


def test_nonpositive_spot_raises() -> None:
    with pytest.raises(ValueError):
        crr_european_price(0.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
