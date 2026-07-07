"""Tests for greeklab.implied_vol.

Covers: price -> IV -> price round-trip accuracy across a moneyness/
expiry/vol grid (the primary correctness property for an IV solver),
and correct fallback/convergence-flagging behavior.
"""

from __future__ import annotations

import itertools

import pytest

from greeklab.black_scholes import bs_price
from greeklab.implied_vol import ImpliedVolResult, implied_vol

_SPOTS = [100.0]
_STRIKES = [60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0]
_RATES = [0.0, 0.03, 0.08]
_DIVIDENDS = [0.0, 0.02]
_SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8]
_EXPIRIES = [1 / 365, 0.05, 0.25, 1.0, 2.0, 5.0]

_GRID = list(itertools.product(_SPOTS, _STRIKES, _RATES, _DIVIDENDS, _SIGMAS, _EXPIRIES, ["call", "put"]))


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _GRID)
def test_round_trip_price_to_iv_to_price(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    price = float(bs_price(spot, strike, rate, q, sigma, t, opt))
    result = implied_vol(price, spot, strike, rate, q, t, opt)
    assert isinstance(result, ImpliedVolResult)

    if not result.converged:
        # Only expected for prices indistinguishable from intrinsic
        # value at the tested sigma grid (essentially zero time value
        # left to invert), where any sigma in the bracket reproduces
        # ~the same price to float precision.
        intrinsic = max(spot - strike, 0.0) if opt == "call" else max(strike - spot, 0.0)
        assert price == pytest.approx(intrinsic, abs=1e-6)
        return

    recovered_price = float(bs_price(spot, strike, rate, q, result.sigma, t, opt))
    assert recovered_price == pytest.approx(price, abs=1e-8)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _GRID)
def test_round_trip_recovers_original_sigma_when_converged(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    price = float(bs_price(spot, strike, rate, q, sigma, t, opt))
    result = implied_vol(price, spot, strike, rate, q, t, opt)
    if not result.converged:
        pytest.skip("unconverged case (near-zero time value) covered by the price round-trip test")
    # Away from vega ~ 0, the recovered sigma should match the true
    # sigma tightly; very deep ITM/OTM or very short expiry can have
    # multiple sigmas giving ~identical prices (flat vega), so we only
    # assert tight sigma recovery when vega is not tiny.
    from greeklab.black_scholes import bs_greeks

    vega = float(bs_greeks(spot, strike, rate, q, sigma, t, opt).vega)
    if vega < 1e-3:
        pytest.skip("vega too small for sigma itself to be identifiable; price round-trip is the real test")
    assert result.sigma == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_batch_returns_list_for_array_input() -> None:
    import numpy as np

    prices = np.array(
        [
            float(bs_price(100.0, 90.0, 0.03, 0.0, 0.2, 1.0, "call")),
            float(bs_price(100.0, 100.0, 0.03, 0.0, 0.2, 1.0, "call")),
            float(bs_price(100.0, 110.0, 0.03, 0.0, 0.2, 1.0, "call")),
        ]
    )
    results = implied_vol(prices, 100.0, np.array([90.0, 100.0, 110.0]), 0.03, 0.0, 1.0, "call")
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert r.converged
        assert r.sigma == pytest.approx(0.2, abs=1e-4)


def test_price_below_intrinsic_value_does_not_converge() -> None:
    # A price below intrinsic value is not attainable by any positive
    # volatility -- the solver should report converged=False, not
    # silently return a nonsensical answer or raise.
    result = implied_vol(price=5.0, spot=100.0, strike=100.0, rate=0.10, dividend_yield=0.0, time_to_expiry=1.0, option_type="put")
    # Intrinsic value of this put is max(100-100,0)=0, so 5.0 is a
    # perfectly plausible time-value price and should converge; use an
    # actually-impossible price instead: a call cheaper than its
    # discounted intrinsic forward value.
    del result
    impossible_price = 0.0  # zero price for a deep ITM call is unattainable at any positive sigma
    result2 = implied_vol(
        price=impossible_price, spot=150.0, strike=100.0, rate=0.10, dividend_yield=0.0, time_to_expiry=1.0, option_type="call"
    )
    assert not result2.converged


def test_newton_converges_in_few_iterations_near_atm() -> None:
    # The Corrado-Miller seed should get Newton-Raphson to converge in
    # a handful of iterations for a well-behaved ATM case.
    price = float(bs_price(100.0, 100.0, 0.05, 0.0, 0.25, 1.0, "call"))
    result = implied_vol(price, 100.0, 100.0, 0.05, 0.0, 1.0, "call")
    assert result.converged
    assert result.method == "newton"
    assert result.iterations <= 10
