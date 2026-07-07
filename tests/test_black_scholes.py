"""Tests for greeklab.black_scholes.

Covers: (a) a hand-computable known reference value; (b) put-call
parity across a moneyness/expiry grid; (c) every analytical Greek
cross-checked against central finite differences; (d) degenerate-limit
edge cases (T=0, sigma=0, deep ITM/OTM) with no NaN/inf and no warnings.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from greeklab.black_scholes import Greeks, bs_greeks, bs_price, put_call_parity_residual

# Hull, J.C. "Options, Futures, and Other Derivatives" (11th ed.), Ch. 15,
# Example 15.6: S0=42, K=40, r=0.10, sigma=0.20, T=0.5 years, no dividend.
# Hull reports call = 4.76, put = 0.81 (rounded to 2dp in the text).
HULL_EXAMPLE_15_6 = {
    "spot": 42.0,
    "strike": 40.0,
    "rate": 0.10,
    "dividend_yield": 0.0,
    "sigma": 0.20,
    "time_to_expiry": 0.5,
}
HULL_CALL_REFERENCE = 4.76
HULL_PUT_REFERENCE = 0.81


def test_hull_example_15_6_call() -> None:
    price = bs_price(**HULL_EXAMPLE_15_6, option_type="call")
    assert price == pytest.approx(HULL_CALL_REFERENCE, abs=5e-3)


def test_hull_example_15_6_put() -> None:
    price = bs_price(**HULL_EXAMPLE_15_6, option_type="put")
    assert price == pytest.approx(HULL_PUT_REFERENCE, abs=5e-3)


# --- Put-call parity across a moneyness/expiry/rate/dividend grid ---

_SPOTS = [80.0, 100.0, 120.0]
_STRIKES = [80.0, 100.0, 120.0]
_RATES = [0.0, 0.03, 0.08]
_DIVIDENDS = [0.0, 0.01, 0.04]
_SIGMAS = [0.05, 0.2, 0.6]
_EXPIRIES = [0.01, 0.5, 1.0, 3.0]

_GRID = list(itertools.product(_SPOTS, _STRIKES, _RATES, _DIVIDENDS, _SIGMAS, _EXPIRIES))


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t"), _GRID)
def test_put_call_parity(spot: float, strike: float, rate: float, q: float, sigma: float, t: float) -> None:
    call = bs_price(spot, strike, rate, q, sigma, t, "call")
    put = bs_price(spot, strike, rate, q, sigma, t, "put")
    residual = put_call_parity_residual(call, put, spot, strike, rate, q, t)
    assert abs(float(residual)) < 1e-8


# --- Finite-difference cross-check of every analytical Greek ---

_FD_GRID = list(
    itertools.product(
        [80.0, 100.0, 120.0],  # spot
        [80.0, 100.0, 120.0],  # strike
        [0.0, 0.03, 0.08],  # rate
        [0.0, 0.02],  # dividend
        [0.1, 0.3, 0.6],  # sigma
        [0.1, 1.0, 2.0],  # time_to_expiry
        ["call", "put"],
    )
)


def _central_fd(f, x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_delta_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)
    h = spot * 1e-4
    fd = _central_fd(lambda s: float(bs_price(s, strike, rate, q, sigma, t, opt)), spot, h)
    assert float(greeks.delta) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_gamma_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)
    h = spot * 1e-3
    fd = (
        float(bs_price(spot + h, strike, rate, q, sigma, t, opt))
        - 2.0 * float(bs_price(spot, strike, rate, q, sigma, t, opt))
        + float(bs_price(spot - h, strike, rate, q, sigma, t, opt))
    ) / (h * h)
    assert float(greeks.gamma) == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_vega_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)
    h = 1e-5
    fd = _central_fd(lambda s: float(bs_price(spot, strike, rate, q, s, t, opt)), sigma, h)
    assert float(greeks.vega) == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_theta_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)
    h = 1e-5
    # theta = dV/dt = -dV/dT
    fd = -_central_fd(lambda tt: float(bs_price(spot, strike, rate, q, sigma, tt, opt)), t, h)
    assert float(greeks.theta) == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_rho_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)
    h = 1e-5
    fd = _central_fd(lambda r: float(bs_price(spot, strike, r, q, sigma, t, opt)), rate, h)
    assert float(greeks.rho) == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_vanna_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    # Vanna = d(delta)/d(sigma).
    h = 1e-5
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)

    def delta_of_sigma(s: float) -> float:
        return float(bs_greeks(spot, strike, rate, q, s, t, opt).delta)

    fd = _central_fd(delta_of_sigma, sigma, h)
    assert float(greeks.vanna) == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_volga_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    # Volga (vomma) = d(vega)/d(sigma).
    h = 1e-5
    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)

    def vega_of_sigma(s: float) -> float:
        return float(bs_greeks(spot, strike, rate, q, s, t, opt).vega)

    fd = _central_fd(vega_of_sigma, sigma, h)
    assert float(greeks.volga) == pytest.approx(fd, abs=5e-3)


@pytest.mark.parametrize(("spot", "strike", "rate", "q", "sigma", "t", "opt"), _FD_GRID)
def test_charm_matches_finite_difference(
    spot: float, strike: float, rate: float, q: float, sigma: float, t: float, opt: str
) -> None:
    # Charm = d(delta)/dt = -d(delta)/dT.
    h = 1e-5

    def delta_of_T(tt: float) -> float:
        return float(bs_greeks(spot, strike, rate, q, sigma, tt, opt).delta)

    greeks = bs_greeks(spot, strike, rate, q, sigma, t, opt)
    fd = -_central_fd(delta_of_T, t, h)
    assert float(greeks.charm) == pytest.approx(fd, abs=1e-3)


# --- Degenerate-limit edge cases: no NaN, no inf, no warnings ---


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("moneyness", ["itm", "atm", "otm"])
def test_zero_time_to_expiry_gives_intrinsic_value(opt: str, moneyness: str) -> None:
    spot = {"itm_call": 110.0, "otm_call": 90.0, "itm_put": 90.0, "otm_put": 110.0, "atm": 100.0}
    s = spot["atm"] if moneyness == "atm" else spot[f"{moneyness}_{opt}"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        price = bs_price(s, 100.0, 0.05, 0.02, 0.2, 0.0, opt)
        greeks = bs_greeks(s, 100.0, 0.05, 0.02, 0.2, 0.0, opt)

    expected_intrinsic = max(s - 100.0, 0.0) if opt == "call" else max(100.0 - s, 0.0)
    assert price == pytest.approx(expected_intrinsic, abs=1e-9)
    assert np.isfinite(price)
    for field in ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"):
        value = getattr(greeks, field)
        assert np.all(np.isfinite(value)), f"{field} is not finite at T=0: {value}"


@pytest.mark.parametrize("opt", ["call", "put"])
def test_zero_volatility_no_nan(opt: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        greeks = bs_greeks(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, opt)
    for field in ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"):
        value = getattr(greeks, field)
        assert np.all(np.isfinite(value)), f"{field} is not finite at sigma=0: {value}"


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize(("spot", "strike"), [(1.0, 1000.0), (1000.0, 1.0)])
def test_deep_itm_otm_small_expiry_no_nan(opt: str, spot: float, strike: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        greeks = bs_greeks(spot, strike, 0.05, 0.0, 0.2, 0.01, opt)
    for field in ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"):
        value = getattr(greeks, field)
        assert np.all(np.isfinite(value)), f"{field} is not finite (deep ITM/OTM): {value}"


def test_zero_rate_no_nan() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        greeks = bs_greeks(100.0, 100.0, 0.0, 0.0, 0.2, 1.0, "call")
    assert np.isfinite(float(greeks.price))


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "straddle")


@pytest.mark.parametrize("bad_spot", [0.0, -1.0])
def test_nonpositive_spot_raises(bad_spot: float) -> None:
    with pytest.raises(ValueError):
        bs_price(bad_spot, 100.0, 0.05, 0.0, 0.2, 1.0, "call")


def test_negative_sigma_raises() -> None:
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.05, 0.0, -0.1, 1.0, "call")


def test_negative_time_raises() -> None:
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.05, 0.0, 0.2, -0.1, "call")


def test_greeks_is_vectorized_over_arrays() -> None:
    spots = np.array([90.0, 100.0, 110.0])
    greeks = bs_greeks(spots, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    assert isinstance(greeks, Greeks)
    assert greeks.price.shape == (3,)
    assert greeks.delta.shape == (3,)
    # Delta should be increasing in spot for a call.
    assert np.all(np.diff(greeks.delta) > 0)
