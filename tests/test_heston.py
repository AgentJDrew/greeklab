"""Tests for greeklab.heston.

This is the headline validation of the library: the Fourier-inversion
price is checked against a **published, independently-verified
reference table** (Alan Lewis's high-precision Mathematica-computed
Heston prices), then cross-validated against an entirely independent
simulation route (full-truncation Euler Monte Carlo), and finally
checked to collapse onto Black-Scholes in the constant-volatility
limit.

Reference source
-----------------
Lewis, A. (2000/posted to the Wilmott quant forum; archived at
https://financepress.com/2019/02/15/heston-model-reference-prices/).
Lewis computed these Heston (1993) call prices in Mathematica with
``WorkingPrecision=50`` ("20 good digits"); Wilmott forum member
"zukimaten" subsequently independently confirmed Panel 1 to ~15
digits and Panel 2 to 13-15 leading digits. Lewis's SDE convention is
``dV_t = (omega - theta*V_t) dt + xi*sqrt(V_t) dW_t`` -- note this
**swaps** the usual roles of the two mean-reversion-related Greek
letters versus this library's ``dv = kappa*(theta - v)dt + xi*sqrt(v)
dW`` convention. The conversion used below is ``kappa = theta_lewis``,
``theta_ours = omega_lewis / theta_lewis``, verified to reproduce
Lewis's own worked numbers before being adopted as a test fixture (see
the module-level constants for the exact converted parameter values).
"""

from __future__ import annotations

import pytest

from greeklab.black_scholes import bs_price
from greeklab.heston import HestonParams, heston_mc_price, heston_price_fourier

# --- Lewis reference: Panel 1 ("standard" parameters) ---
# Lewis notation: omega=1, theta_lewis=4, xi=1, rho=-0.5, v0=0.04,
# r=0.01, q=0.02, S0=100, T=1.
# Converted: kappa = theta_lewis = 4.0, theta = omega/theta_lewis = 0.25.
_LEWIS_KAPPA = 4.0
_LEWIS_THETA = 0.25
_LEWIS_XI = 1.0
_LEWIS_RHO = -0.5
_LEWIS_SPOT = 100.0
_LEWIS_RATE = 0.01
_LEWIS_DIV = 0.02

_PANEL_1_V0 = 0.04
_PANEL_1_T = 1.0
_PANEL_1_REFERENCE = {
    80.0: 26.774758743998854221382195325726949201687074848341,
    90.0: 20.933349000596710388139445766564068085476194042256,
    100.0: 16.070154917028834278213466703938231827658768230714,
    110.0: 12.132211516709844867860534767549426052805766831181,
    120.0: 9.024913483457835636553375454092357136489051667150,
}

# --- Lewis reference: Panel 2 ("extreme" parameters: short T, small v0) ---
_PANEL_2_V0 = 0.01
_PANEL_2_T = 0.01
_PANEL_2_REFERENCE = {
    90.0: 9.989001595065276544935948045293485530832966049263,
    95.0: 4.989963479738160122154264702582719627807098780529,
    100.0: 0.467782671512844263098248405184095087949465507760,
    105.0: 2.527447823194706060519991248106500619490942e-6,
    110.0: 1.29932760052624920704881258510264466e-13,
}


@pytest.mark.parametrize(("strike", "reference"), sorted(_PANEL_1_REFERENCE.items()))
def test_heston_fourier_matches_lewis_reference_panel1(strike: float, reference: float) -> None:
    params = HestonParams(kappa=_LEWIS_KAPPA, theta=_LEWIS_THETA, xi=_LEWIS_XI, rho=_LEWIS_RHO, v0=_PANEL_1_V0)
    price = heston_price_fourier(params, _LEWIS_SPOT, strike, _LEWIS_RATE, _LEWIS_DIV, _PANEL_1_T, "call")
    assert price == pytest.approx(reference, abs=1e-7)


@pytest.mark.parametrize(("strike", "reference"), sorted(_PANEL_2_REFERENCE.items()))
def test_heston_fourier_matches_lewis_reference_panel2_extreme(strike: float, reference: float) -> None:
    # Panel 2 stress-tests short time_to_expiry (T=0.01) and small
    # v0=0.01 -- the regime where a too-small Fourier integration
    # truncation silently gives wrong answers (reproduced during
    # development; see greeklab.heston._p_j docstring). Absolute
    # tolerance is looser here since some reference values are
    # themselves as small as ~1e-13.
    params = HestonParams(kappa=_LEWIS_KAPPA, theta=_LEWIS_THETA, xi=_LEWIS_XI, rho=_LEWIS_RHO, v0=_PANEL_2_V0)
    price = heston_price_fourier(params, _LEWIS_SPOT, strike, _LEWIS_RATE, _LEWIS_DIV, _PANEL_2_T, "call")
    assert price == pytest.approx(reference, abs=5e-8)


def test_heston_fourier_cross_validates_against_monte_carlo() -> None:
    # A widely-cited Heston (1993) parameter set (Albrecher, Mayer,
    # Schoutens & Tistaert 2007, "The Little Heston Trap"): two
    # completely independent numerical methods (quadrature vs.
    # simulation) should agree within a few Monte Carlo standard errors.
    params = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
    fourier = heston_price_fourier(params, 100.0, 100.0, 0.0, 0.0, 1.0, "call")
    mc_price, mc_se = heston_mc_price(params, 100.0, 100.0, 0.0, 0.0, 1.0, "call", n_steps=200, n_paths=100_000, seed=42)
    n_se_away = abs(fourier - mc_price) / mc_se
    assert n_se_away < 4.0


@pytest.mark.parametrize("rho", [-0.7, -0.3, 0.0, 0.3, 0.7])
def test_heston_fourier_cross_validates_against_mc_across_correlations(rho: float) -> None:
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=rho, v0=0.04)
    fourier = heston_price_fourier(params, 100.0, 100.0, 0.03, 0.0, 0.75, "call")
    mc_price, mc_se = heston_mc_price(params, 100.0, 100.0, 0.03, 0.0, 0.75, "call", n_steps=150, n_paths=80_000, seed=7)
    n_se_away = abs(fourier - mc_price) / mc_se
    assert n_se_away < 4.0


def test_heston_reduces_to_black_scholes_as_xi_vanishes() -> None:
    # As xi -> 0 with v0 = theta = sigma**2 (i.e. the variance process
    # has no randomness and starts at its own deterministic long-run
    # level), Heston must collapse to Black-Scholes with volatility
    # = sigma exactly. This is checked against Black-Scholes directly,
    # not against another Heston call -- a genuinely independent
    # formula to cross-check against.
    spot, strike, rate, q, sigma, t = 100.0, 100.0, 0.05, 0.01, 0.2, 1.0
    bs = float(bs_price(spot, strike, rate, q, sigma, t, "call"))
    params = HestonParams(kappa=2.0, theta=sigma**2, xi=1e-4, rho=0.0, v0=sigma**2)
    heston = heston_price_fourier(params, spot, strike, rate, q, t, "call")
    assert heston == pytest.approx(bs, abs=1e-6)


def test_heston_put_via_parity_matches_call() -> None:
    import math

    params = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
    spot, strike, rate, q, t = 100.0, 100.0, 0.03, 0.01, 1.0
    call = heston_price_fourier(params, spot, strike, rate, q, t, "call")
    put = heston_price_fourier(params, spot, strike, rate, q, t, "put")
    residual = (call - put) - (spot * math.exp(-q * t) - strike * math.exp(-rate * t))
    assert abs(residual) < 1e-9


def test_feller_condition_flag() -> None:
    satisfied = HestonParams(kappa=2.0, theta=0.04, xi=0.1, rho=-0.5, v0=0.04)
    assert satisfied.feller_satisfied()

    violated = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
    assert not violated.feller_satisfied()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("kappa", 0.0), ("theta", 0.0), ("xi", 0.0), ("rho", 1.5), ("rho", -1.5), ("v0", 0.0)],
)
def test_invalid_heston_params_raise(field: str, bad_value: float) -> None:
    kwargs = {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.5, "v0": 0.04}
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        HestonParams(**kwargs)


def test_heston_mc_reproducible_with_seed() -> None:
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    p1, se1 = heston_mc_price(params, 100.0, 100.0, 0.03, 0.0, 1.0, "call", n_steps=50, n_paths=5_000, seed=11)
    p2, se2 = heston_mc_price(params, 100.0, 100.0, 0.03, 0.0, 1.0, "call", n_steps=50, n_paths=5_000, seed=11)
    assert p1 == p2
    assert se1 == se2
