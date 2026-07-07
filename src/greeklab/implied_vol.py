"""Implied volatility solver: seeded Newton-Raphson with a Brent fallback.

Given an observed market price, solves for the Black-Scholes-Merton
volatility that reproduces it. Uses a closed-form initial guess
(Corrado & Miller 1996, itself a refinement of Brenner & Subrahmanyam
1988) so Newton-Raphson typically converges in 2-4 iterations, then
falls back to bisection-safeguarded Brent's method (``scipy.optimize.
brentq``) for the rare cases where Newton-Raphson fails to converge
(deep ITM/OTM + vega ~ 0, or a bad/non-arbitrage-free input price).

References
----------
- Brenner, M. and Subrahmanyam, M. G. (1988). "A Simple Formula to
  Compute the Implied Standard Deviation." *Financial Analysts
  Journal*, 44(5), 80-83.
- Corrado, C. J. and Miller, T. W. (1996). "A Note on a Simple, Accurate
  Formula to Compute Implied Standard Deviations." *Journal of Banking
  & Finance*, 20(3), 595-603.
- Hull, J. C. *Options, Futures, and Other Derivatives* (11th ed.),
  Section 15.10 (implied volatility).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ._typing import FloatArray
from ._utils import as_float_array
from .black_scholes import bs_greeks, bs_price

__all__ = ["implied_vol", "ImpliedVolResult"]

#: Newton-Raphson convergence tolerance on absolute price error.
_PRICE_TOL = 1e-10
_MAX_NEWTON_ITER = 50
#: Vega floor below which a Newton step is considered unreliable (near
#: deep ITM/OTM or T~0) and we fall through to the Brent bracket search.
_MIN_VEGA = 1e-8
#: Volatility search bracket for the Brent fallback.
_SIGMA_LO = 1e-6
_SIGMA_HI = 5.0


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Result of a single implied-volatility solve.

    Attributes
    ----------
    sigma : float
        Solved annualized implied volatility.
    iterations : int
        Number of Newton-Raphson iterations used (0 if the Brent
        fallback alone was used).
    method : {"newton", "brent"}
        Which method produced the final answer.
    converged : bool
        Whether the solve met :data:`_PRICE_TOL` (Newton) or Brent's
        own convergence criterion.
    """

    sigma: float
    iterations: int
    method: str
    converged: bool


def _corrado_miller_seed(
    price: float, spot: float, strike: float, rate: float, dividend_yield: float, t: float
) -> float:
    """Closed-form initial IV guess (Corrado & Miller 1996).

    Built for undiscounted/forward-style quotes; here adapted to the
    dividend-adjusted forward :math:`F = S e^{(r-q)T}` so it applies
    under the Merton (1973) continuous-dividend extension too. Corrado-
    Miller improves on Brenner-Subrahmanyam by adding a moneyness
    correction term, giving a materially better seed away from
    at-the-money.
    """
    disc_r = np.exp(-rate * t)
    forward = spot * np.exp((rate - dividend_yield) * t)
    # Work in "undiscounted forward price" units, matching the original
    # Corrado-Miller derivation.
    c = price / disc_r
    x = strike
    f = forward
    inner = (c - (f - x) / 2.0) ** 2 - ((f - x) ** 2) / np.pi
    inner = max(inner, 0.0)
    numerator = (c - (f - x) / 2.0) + np.sqrt(inner)
    seed = np.sqrt(2.0 * np.pi / t) / (f + x) * numerator
    if not np.isfinite(seed) or seed <= 0.0:
        # Fallback to the simpler Brenner-Subrahmanyam ATM approximation.
        seed = np.sqrt(2.0 * np.pi / t) * c / spot
    return float(np.clip(seed, _SIGMA_LO, _SIGMA_HI))


def _solve_scalar(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    t: float,
    option_type: str,
) -> ImpliedVolResult:
    seed = _corrado_miller_seed(price, spot, strike, rate, dividend_yield, t)
    sigma = seed

    for i in range(1, _MAX_NEWTON_ITER + 1):
        greeks = bs_greeks(spot, strike, rate, dividend_yield, sigma, t, option_type)
        model_price = float(greeks.price)
        vega = float(greeks.vega)
        error = model_price - price

        if abs(error) < _PRICE_TOL:
            return ImpliedVolResult(sigma=sigma, iterations=i, method="newton", converged=True)

        if vega < _MIN_VEGA or not np.isfinite(vega):
            break  # Newton is unreliable here; drop to Brent below.

        step = error / vega
        new_sigma = sigma - step
        if not np.isfinite(new_sigma) or new_sigma <= 0.0:
            break
        sigma = float(np.clip(new_sigma, _SIGMA_LO, _SIGMA_HI))
    else:
        i = _MAX_NEWTON_ITER

    # Brent fallback: robust bracketed root-find on the same objective,
    # at the cost of more function evaluations than Newton needs when
    # it converges cleanly.
    def objective(s: float) -> float:
        return float(bs_price(spot, strike, rate, dividend_yield, s, t, option_type)) - price

    lo, hi = _SIGMA_LO, _SIGMA_HI
    f_lo, f_hi = objective(lo), objective(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        # No sign change across the bracket: the price is outside what
        # any volatility in [_SIGMA_LO, _SIGMA_HI] can produce (e.g. a
        # price below intrinsic value, or above the max achievable
        # under an extreme vol). Return the best Newton estimate with
        # converged=False rather than raising, so batch/grid callers
        # can filter on `.converged` instead of catching exceptions.
        return ImpliedVolResult(sigma=sigma, iterations=i, method="newton", converged=False)

    root = brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    return ImpliedVolResult(sigma=float(root), iterations=i, method="brent", converged=True)


def implied_vol(
    price: FloatArray | float,
    spot: FloatArray | float,
    strike: FloatArray | float,
    rate: FloatArray | float,
    dividend_yield: FloatArray | float,
    time_to_expiry: FloatArray | float,
    option_type: str = "call",
) -> ImpliedVolResult | list[ImpliedVolResult]:
    """Solve for Black-Scholes-Merton implied volatility.

    Given an observed option price, finds the ``sigma`` such that
    :func:`greeklab.black_scholes.bs_price` reproduces it, using a
    Corrado-Miller (1996) closed-form seed followed by Newton-Raphson,
    with an automatic fallback to Brent's method (bracketed root-find)
    when Newton-Raphson stalls (typically deep ITM/OTM where vega is
    near zero).

    Parameters
    ----------
    price : float or array
        Observed market price(s). If any input is array-like, all
        inputs are broadcast together and a list of
        :class:`ImpliedVolResult` is returned (one per element);
        otherwise a single :class:`ImpliedVolResult` is returned.
    spot, strike, rate, dividend_yield, time_to_expiry
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
    option_type : {"call", "put"}
        Which payoff ``price`` corresponds to.

    Returns
    -------
    ImpliedVolResult or list[ImpliedVolResult]
        See :class:`ImpliedVolResult`. Check ``.converged`` before
        trusting a result — an unconverged result means the observed
        price was not attainable by any volatility in the search
        bracket ``[1e-6, 5.0]`` (e.g. below intrinsic value or a
        stale/erroneous quote).

    Notes
    -----
    Round-trip accuracy: for prices generated by
    :func:`greeklab.black_scholes.bs_price` itself, ``implied_vol``
    recovers the original ``sigma`` to within ~1e-8 across a wide
    moneyness/expiry grid — see ``tests/test_implied_vol.py``.
    """
    scalar_input = all(
        np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
        for x in (price, spot, strike, rate, dividend_yield, time_to_expiry)
    )

    p_arr, s_arr, k_arr, r_arr, q_arr, t_arr = np.broadcast_arrays(
        as_float_array(price),
        as_float_array(spot),
        as_float_array(strike),
        as_float_array(rate),
        as_float_array(dividend_yield),
        as_float_array(time_to_expiry),
    )

    flat_results = [
        _solve_scalar(
            float(p_arr.flat[i]),
            float(s_arr.flat[i]),
            float(k_arr.flat[i]),
            float(r_arr.flat[i]),
            float(q_arr.flat[i]),
            float(t_arr.flat[i]),
            option_type,
        )
        for i in range(p_arr.size)
    ]

    if scalar_input:
        return flat_results[0]
    return flat_results
