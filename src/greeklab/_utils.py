"""Internal numerical helpers shared across pricing models.

Not part of the public API (leading underscore). Centralizes the
edge-case handling that every model needs: standard normal CDF/PDF,
safe division by (possibly zero) volatility-time products, and the
``d1``/``d2`` terms common to Black-Scholes-family formulas.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._typing import ArrayLike, FloatArray

#: Below this total variance (sigma^2 * T), Black-Scholes formulas are
#: numerically unstable (division by ~0) and are replaced with the
#: analytically correct limiting behavior (intrinsic value, zero vega,
#: etc.) instead of propagating NaN/inf.
_MIN_TOTAL_STDEV = 1e-12


def as_float_array(x: ArrayLike) -> FloatArray:
    """Coerce a scalar or array-like into a ``float64`` NumPy array."""
    return np.asarray(x, dtype=np.float64)


def norm_cdf(x: ArrayLike) -> FloatArray:
    """Standard normal CDF, :math:`\\Phi(x)`."""
    # Force the array-in/array-out overload of scipy.stats.norm.cdf
    # (rather than its float-in/float-out scalar overload) so the
    # return type is unambiguously FloatArray, not float | FloatArray.
    return np.asarray(norm.cdf(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def norm_pdf(x: ArrayLike) -> FloatArray:
    """Standard normal PDF, :math:`\\phi(x)`."""
    return np.asarray(norm.pdf(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def d1_d2(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    sigma: ArrayLike,
    time_to_expiry: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the Black-Scholes-Merton ``d1``, ``d2`` terms safely.

    Uses the cost-of-carry form ``b = rate - dividend_yield`` (Merton
    1973 continuous-dividend extension of Black-Scholes 1973):

    .. math::
        d_1 = \\frac{\\ln(S/K) + (b + \\sigma^2/2) T}{\\sigma \\sqrt{T}}, \\qquad
        d_2 = d_1 - \\sigma \\sqrt{T}

    Returns ``(d1, d2, total_stdev)`` where ``total_stdev = sigma *
    sqrt(T)``. Where ``total_stdev`` is at/below :data:`_MIN_TOTAL_STDEV`
    (``T -> 0`` or ``sigma -> 0``), ``d1``/``d2`` are set to ``+/-inf``
    with the sign of moneyness (``log(S/K)``) so that ``norm_cdf(d1)``
    /``norm_cdf(d2)`` correctly collapse to the 0/1 indicator of the
    option finishing in-the-money — giving exact intrinsic-value pricing
    in the limit without any special-cased branch in the callers.
    """
    s = as_float_array(spot)
    k = as_float_array(strike)
    r = as_float_array(rate)
    q = as_float_array(dividend_yield)
    sig = as_float_array(sigma)
    t = as_float_array(time_to_expiry)

    b = r - q
    total_var = sig * sig * t
    total_stdev = np.sqrt(np.clip(total_var, 0.0, None))

    moneyness = np.log(s / k)
    safe_stdev = np.where(total_stdev > _MIN_TOTAL_STDEV, total_stdev, 1.0)

    d1 = (moneyness + (b + 0.5 * sig * sig) * t) / safe_stdev
    d2 = d1 - safe_stdev

    # Degenerate limit: sign of moneyness decides +/-inf (ties -> +inf,
    # i.e. an at-the-money option at T=0 is treated as just in-the-money,
    # matching the standard intrinsic-value convention max(S-K, 0)).
    degenerate = total_stdev <= _MIN_TOTAL_STDEV
    inf_sign = np.where(moneyness >= 0.0, np.inf, -np.inf)
    d1 = np.where(degenerate, inf_sign, d1)
    d2 = np.where(degenerate, inf_sign, d2)

    return d1, d2, total_stdev


__all__ = ["as_float_array", "norm_cdf", "norm_pdf", "d1_d2"]
