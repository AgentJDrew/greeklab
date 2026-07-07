"""Tests for greeklab.exotics.

Covers the sanity relations for each exotic payoff family: (a) Asian
call < vanilla European call; (b) knock-out barrier <= vanilla, and
knock-in + knock-out reproduces the vanilla price (in/out parity);
(c) American LSM >= European, and LSM American call with no dividends
approximately equals the European call (no early-exercise premium).
"""

from __future__ import annotations

import pytest

from greeklab.binomial import crr_american_price
from greeklab.black_scholes import bs_price
from greeklab.exotics import american_lsm_mc, asian_arithmetic_mc, barrier_mc

_MAX_SE_AWAY = 4.0


@pytest.mark.parametrize("opt", ["call", "put"])
def test_asian_cheaper_than_vanilla(opt: str) -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
    vanilla = float(bs_price(spot, strike, rate, q, sigma, t, opt))
    asian = asian_arithmetic_mc(spot, strike, rate, q, sigma, t, opt, n_fixings=50, n_paths=100_000, seed=1)
    assert asian.price < vanilla


def test_asian_price_is_seed_reproducible() -> None:
    args = (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    r1 = asian_arithmetic_mc(*args, n_fixings=20, n_paths=20_000, seed=5)
    r2 = asian_arithmetic_mc(*args, n_fixings=20, n_paths=20_000, seed=5)
    assert r1.price == r2.price


@pytest.mark.parametrize(
    "barrier_type",
    ["down-and-out", "up-and-out"],
)
def test_knockout_barrier_cheaper_than_vanilla(barrier_type: str) -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
    barrier_level = 80.0 if "down" in barrier_type else 130.0
    vanilla = float(bs_price(spot, strike, rate, q, sigma, t, "call"))
    result = barrier_mc(
        spot, strike, barrier_level, rate, q, sigma, t, "call", barrier_type, n_steps=100, n_paths=100_000, seed=2
    )
    assert result.price <= vanilla
    assert 0.0 <= result.fraction_knocked <= 1.0


@pytest.mark.parametrize(("base_type", "barrier_level"), [("down", 80.0), ("up", 130.0)])
def test_barrier_in_out_parity(base_type: str, barrier_level: float) -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
    vanilla = float(bs_price(spot, strike, rate, q, sigma, t, "call"))

    knock_out = barrier_mc(
        spot, strike, barrier_level, rate, q, sigma, t, "call", f"{base_type}-and-out", n_steps=200, n_paths=200_000, seed=3
    )
    knock_in = barrier_mc(
        spot, strike, barrier_level, rate, q, sigma, t, "call", f"{base_type}-and-in", n_steps=200, n_paths=200_000, seed=3
    )
    # Same seed => same underlying paths => the knock-out and knock-in
    # payoffs are drawn from a *complementary partition* of the exact
    # same path set, so out+in should equal the vanilla price almost
    # exactly (up to independent MC noise in the discounted-payoff
    # averaging, not in the barrier classification itself).
    combined = knock_out.price + knock_in.price
    combined_se = (knock_out.std_error**2 + knock_in.std_error**2) ** 0.5
    n_se_away = abs(combined - vanilla) / combined_se
    assert n_se_away < _MAX_SE_AWAY


def test_invalid_barrier_type_raises() -> None:
    with pytest.raises(ValueError):
        barrier_mc(100.0, 100.0, 80.0, 0.05, 0.0, 0.2, 1.0, "call", "sideways-and-out")


def test_american_lsm_put_geq_european_put() -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
    european = float(bs_price(spot, strike, rate, q, sigma, t, "put"))
    lsm = american_lsm_mc(spot, strike, rate, q, sigma, t, "put", n_steps=50, n_paths=100_000, seed=3)
    # LSM is a lower-biased estimator of the true American price (any
    # suboptimal exercise policy underestimates value), but should
    # still clear the European price given the exercise premium here.
    assert lsm.price > european


def test_american_lsm_put_agrees_with_crr_lattice() -> None:
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
    crr = crr_american_price(spot, strike, rate, q, sigma, t, "put", n_steps=1000)
    lsm = american_lsm_mc(spot, strike, rate, q, sigma, t, "put", n_steps=50, n_paths=200_000, seed=3)
    n_se_away = abs(lsm.price - crr) / lsm.std_error
    assert n_se_away < _MAX_SE_AWAY


def test_american_lsm_call_no_dividend_matches_european() -> None:
    # Merton (1973): no early-exercise premium for a call with no
    # dividends, so LSM's American call price should land close to the
    # closed-form European price.
    spot, strike, rate, sigma, t = 100.0, 100.0, 0.05, 0.25, 1.0
    european = float(bs_price(spot, strike, rate, 0.0, sigma, t, "call"))
    lsm = american_lsm_mc(spot, strike, rate, 0.0, sigma, t, "call", n_steps=50, n_paths=200_000, seed=4)
    n_se_away = abs(lsm.price - european) / lsm.std_error
    assert n_se_away < _MAX_SE_AWAY


def test_lsm_requires_minimum_paths() -> None:
    with pytest.raises(ValueError):
        american_lsm_mc(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put", n_paths=10)
