"""Black-Scholes-Merton European option pricing and analytical Greeks.

Implements the Black-Scholes (1973) formula for European options,
extended by Merton (1973) to a continuous dividend/carry rate ``q``.
Every function is fully vectorized (NumPy broadcasting) and safe at the
degenerate limits ``T -> 0``, ``sigma -> 0``, and deep in/out-of-the-money.

**Sign and scaling conventions** (documented once here, all Greeks follow it):

- ``rate`` and ``dividend_yield`` are continuously-compounded annualized
  rates (e.g. ``0.05`` for 5%).
- ``sigma`` is annualized volatility (e.g. ``0.20`` for 20%).
- ``time_to_expiry`` is in years.
- **Delta**: :math:`\\partial V/\\partial S`, unscaled (a $1 move in spot).
- **Gamma**: :math:`\\partial^2 V/\\partial S^2`, unscaled (per $1 move in spot,
  squared).
- **Vega**: :math:`\\partial V/\\partial \\sigma`, scaled **per 1.00 (100
  percentage points) of volatility**, i.e. divide by 100 for the common
  "per vol point" convention.
- **Theta**: :math:`\\partial V/\\partial t = -\\partial V/\\partial T`, scaled
  **per year**; divide by 365 for "per calendar day" or 252 for "per
  trading day."
- **Rho**: :math:`\\partial V/\\partial r`, scaled per 1.00 (100 percentage
  points) of rate; divide by 100 for "per basis-point-times-100" / "per
  1% rate move."
- **Vanna**: :math:`\\partial^2 V/\\partial S \\partial \\sigma` (equivalently
  :math:`\\partial \\text{delta}/\\partial \\sigma = \\partial \\text{vega}/\\partial S`),
  per $1 of spot and per 1.00 of vol.
- **Volga** (vomma): :math:`\\partial^2 V/\\partial \\sigma^2`, per 1.00 of vol
  squared.
- **Charm** (delta decay): :math:`\\partial \\text{delta}/\\partial t =
  -\\partial \\text{delta}/\\partial T`, per year.

References
----------
- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
  Liabilities." *Journal of Political Economy*, 81(3), 637-654.
- Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell
  Journal of Economics and Management Science*, 4(1), 141-183.
- Hull, J. C. *Options, Futures, and Other Derivatives* (11th ed.),
  Chapters 15 (BSM model) and 19 (the Greeks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._typing import ArrayLike, FloatArray
from ._utils import as_float_array, d1_d2, norm_cdf, norm_pdf

__all__ = [
    "bs_price",
    "bs_greeks",
    "Greeks",
    "put_call_parity_residual",
]


def bs_price(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    sigma: ArrayLike,
    time_to_expiry: ArrayLike,
    option_type: str = "call",
) -> FloatArray:
    """European option price under Black-Scholes-Merton.

    .. math::
        C = S e^{-qT} \\Phi(d_1) - K e^{-rT} \\Phi(d_2)

        P = K e^{-rT} \\Phi(-d_2) - S e^{-qT} \\Phi(-d_1)

    Parameters
    ----------
    spot, strike : ArrayLike
        Current underlying price and strike price. Must be positive.
    rate : ArrayLike
        Continuously-compounded annualized risk-free rate.
    dividend_yield : ArrayLike
        Continuously-compounded annualized dividend/carry yield
        (``0.0`` for a non-dividend-paying stock or a futures-style
        carry of ``q = r`` for pricing on a forward).
    sigma : ArrayLike
        Annualized volatility. Must be non-negative.
    time_to_expiry : ArrayLike
        Time to expiry in years. Must be non-negative; ``0.0`` returns
        the exact intrinsic value.
    option_type : {"call", "put"}
        Which European payoff to price.

    Returns
    -------
    FloatArray
        Option price(s), broadcast over the input shapes.

    Notes
    -----
    At ``time_to_expiry == 0`` or ``sigma == 0`` this returns the exact
    intrinsic value (``max(S-K, 0)`` for a call, discounted forward
    intrinsic value otherwise) with no NaN/inf — see
    :func:`greeklab._utils.d1_d2`.
    """
    option_type = _validate_option_type(option_type)
    s = as_float_array(spot)
    k = as_float_array(strike)
    r = as_float_array(rate)
    q = as_float_array(dividend_yield)
    t = as_float_array(time_to_expiry)

    _validate_positive("spot", s)
    _validate_positive("strike", k)
    _validate_nonnegative("sigma", as_float_array(sigma))
    _validate_nonnegative("time_to_expiry", t)

    d1, d2, _ = d1_d2(s, k, r, q, sigma, t)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)

    if option_type == "call":
        price = s * disc_q * norm_cdf(d1) - k * disc_r * norm_cdf(d2)
    else:
        price = k * disc_r * norm_cdf(-d2) - s * disc_q * norm_cdf(-d1)

    # Floor at 0: guards against sub-ULP negative noise when d1/d2 are
    # +/-inf (T=0 limit) and one Phi(.) term is exactly 0.0 while
    # floating point rounding in the other leaves a ~1e-17 residual.
    return np.clip(price, 0.0, None)


@dataclass(frozen=True, slots=True)
class Greeks:
    """Bundle of analytical Black-Scholes-Merton Greeks.

    All fields are ``FloatArray`` broadcasting the shape of the inputs
    passed to :func:`bs_greeks`. See the module docstring for the exact
    sign and scaling convention of each field.
    """

    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray
    vanna: FloatArray
    volga: FloatArray
    charm: FloatArray


def bs_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    sigma: ArrayLike,
    time_to_expiry: ArrayLike,
    option_type: str = "call",
) -> Greeks:
    """Analytical Black-Scholes-Merton price and Greeks (1st + 2nd order).

    Computes price, delta, gamma, vega, theta, rho, and the 2nd-order
    cross Greeks vanna, volga (vomma), and charm, all from closed-form
    expressions (no finite differencing) — see Hull Ch. 19 for the
    standard derivations, extended here to the Merton (1973)
    continuous-dividend case.

    Formulas (call; put deltas/theta/rho/charm shown as the +/- variant
    below — gamma, vega, vanna, volga are identical for calls and puts
    since they don't depend on the sign of the payoff):

    .. math::
        \\Delta_{call} = e^{-qT}\\Phi(d_1), \\quad
        \\Delta_{put} = e^{-qT}(\\Phi(d_1)-1)

        \\Gamma = \\frac{e^{-qT}\\phi(d_1)}{S\\sigma\\sqrt{T}}

        \\text{Vega} = S e^{-qT}\\phi(d_1)\\sqrt{T}

        \\Theta_{call} = -\\frac{S e^{-qT}\\phi(d_1)\\sigma}{2\\sqrt{T}}
            - rKe^{-rT}\\Phi(d_2) + qSe^{-qT}\\Phi(d_1)

        \\rho_{call} = KTe^{-rT}\\Phi(d_2)

        \\text{Vanna} = -e^{-qT}\\phi(d_1)\\frac{d_2}{\\sigma}

        \\text{Volga} = \\text{Vega}\\cdot\\frac{d_1 d_2}{\\sigma}

        \\text{Charm}_{call} = qe^{-qT}\\Phi(d_1) - \\text{common}, \\quad
        \\text{Charm}_{put} = qe^{-qT}(\\Phi(d_1)-1) - \\text{common}

        \\text{common} = e^{-qT}\\phi(d_1)
            \\frac{2(r-q)T - d_2\\sigma\\sqrt{T}}{2T\\sigma\\sqrt{T}}

    Parameters
    ----------
    spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type
        Same as :func:`bs_price`.

    Returns
    -------
    Greeks
        Dataclass with fields ``price, delta, gamma, vega, theta, rho,
        vanna, volga, charm``. See the module docstring for units.
    """
    option_type = _validate_option_type(option_type)
    s = as_float_array(spot)
    k = as_float_array(strike)
    r = as_float_array(rate)
    q = as_float_array(dividend_yield)
    sig = as_float_array(sigma)
    t = as_float_array(time_to_expiry)

    _validate_positive("spot", s)
    _validate_positive("strike", k)
    _validate_nonnegative("sigma", sig)
    _validate_nonnegative("time_to_expiry", t)

    price = bs_price(s, k, r, q, sig, t, option_type)

    d1, d2, total_stdev = d1_d2(s, k, r, q, sig, t)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)
    sqrt_t = np.sqrt(t)

    # Gamma, vega, vanna, volga are call/put-symmetric. Guard the T=0 /
    # sigma=0 degenerate limit (total_stdev ~ 0) where an option has
    # zero optionality left: gamma/vega/vanna/volga -> 0 exactly (no
    # residual convexity once time or vol collapses to nothing), rather
    # than propagating 0/0 -> NaN from the pdf_d1/safe_stdev ratio at
    # d1 = +/-inf (where pdf_d1 underflows to exactly 0.0 already, but
    # we guard explicitly for robustness against future formula edits).
    #
    # d1/d2 are also replaced with a finite placeholder (0.0) under the
    # degenerate mask *before* any arithmetic touches them: at d1 = d2 =
    # +/-inf, pdf_d1 correctly underflows to exactly 0.0, but 0.0 * inf
    # (e.g. in the vanna/volga/charm formulas below) is the
    # indeterminate form 0*inf -> NaN, not 0 -- and NumPy's np.where
    # evaluates *both* branches eagerly, so without this substitution
    # the discarded branch would raise a RuntimeWarning (and briefly
    # hold a NaN) even though the final selected value is correct.
    # Substituting a finite placeholder here does not change any
    # output, since every place d1/d2 are used below is already
    # separately masked by the same `degenerate` condition.
    degenerate = total_stdev <= 1e-12
    d1_safe = np.where(degenerate, 0.0, d1)
    d2_safe = np.where(degenerate, 0.0, d2)
    pdf_d1 = norm_pdf(d1_safe)
    safe_s_sig_sqrt_t = np.where(degenerate, 1.0, s * sig * sqrt_t)
    safe_sig = np.where(degenerate, 1.0, sig)

    gamma = np.where(degenerate, 0.0, disc_q * pdf_d1 / safe_s_sig_sqrt_t)
    vega = np.where(degenerate, 0.0, s * disc_q * pdf_d1 * sqrt_t)
    vanna = np.where(degenerate, 0.0, -disc_q * pdf_d1 * d2_safe / safe_sig)
    volga = np.where(degenerate, 0.0, vega * d1_safe * d2_safe / safe_sig)

    safe_sqrt_t = np.where(degenerate, 1.0, sqrt_t)
    theta_time_decay = -s * disc_q * pdf_d1 * sig / (2.0 * safe_sqrt_t)

    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
        theta = theta_time_decay - r * k * disc_r * norm_cdf(d2) + q * s * disc_q * norm_cdf(d1)
        rho = k * t * disc_r * norm_cdf(d2)
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = theta_time_decay + r * k * disc_r * norm_cdf(-d2) - q * s * disc_q * norm_cdf(-d1)
        rho = -k * t * disc_r * norm_cdf(-d2)

    charm = _charm(option_type, d1, d2_safe, disc_q, pdf_d1, r, q, t, sig, sqrt_t, degenerate)

    # Degenerate limit (T=0 or sigma=0): the option has no time value or
    # optionality left, so theta (further time decay) and charm (further
    # delta decay) are both exactly 0 — there's nothing left to decay.
    # Delta already collapses correctly to the moneyness indicator via
    # d1=+/-inf above; rho is naturally 0 too since it's proportional to
    # T. Only theta/charm need an explicit override, because the
    # 1/safe_sqrt_t guard above leaves a stale nonzero value otherwise.
    theta = np.where(degenerate, 0.0, theta)
    charm = np.where(degenerate, 0.0, charm)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        volga=volga,
        charm=charm,
    )


def _charm(
    option_type: str,
    d1: FloatArray,
    d2: FloatArray,
    disc_q: FloatArray,
    pdf_d1: FloatArray,
    r: FloatArray,
    q: FloatArray,
    t: FloatArray,
    sig: FloatArray,
    sqrt_t: FloatArray,
    degenerate: npt.NDArray[np.bool_],
) -> FloatArray:
    """Charm (:math:`\\partial \\Delta/\\partial t`), derived directly from
    :math:`\\Delta_{call} = e^{-qT}\\Phi(d_1)` and
    :math:`\\Delta_{put} = e^{-qT}(\\Phi(d_1) - 1)` via
    :math:`\\text{Charm} = -\\partial \\Delta/\\partial T`:

    .. math::
        \\text{Charm}_{call} = q e^{-qT}\\Phi(d_1) - \\text{common}, \\qquad
        \\text{Charm}_{put} = q e^{-qT}(\\Phi(d_1) - 1) - \\text{common}

        \\text{common} = e^{-qT}\\phi(d_1)
            \\frac{2(r-q)T - d_2\\sigma\\sqrt{T}}{2T\\sigma\\sqrt{T}}

    The two option types share the same ``common`` term (it comes from
    differentiating :math:`\\Phi(d_1)` itself, which does not depend on
    option type) and differ only in the ``e^{-qT}(\\cdot)`` prefactor
    piece, exactly mirroring how delta itself differs by a ``-1``
    inside the parenthesis. Cross-checked against central finite
    differences of delta w.r.t. time across a moneyness/rate/vol/expiry
    grid (see ``tests/test_black_scholes.py``); a naive
    :math:`\\Phi(-d_1)`-based put form (the sign convention printed in
    some references) does **not** match FD and was rejected in favor of
    this derivation.
    """
    safe_denom = np.where(degenerate, 1.0, 2.0 * t * sig * sqrt_t)
    common = disc_q * pdf_d1 * (2.0 * (r - q) * t - d2 * sig * sqrt_t) / safe_denom
    if option_type == "call":
        return q * disc_q * norm_cdf(d1) - common
    return q * disc_q * (norm_cdf(d1) - 1.0) - common


def put_call_parity_residual(
    call_price: ArrayLike,
    put_price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    time_to_expiry: ArrayLike,
) -> FloatArray:
    """Residual of put-call parity: ``C - P - (S*e^{-qT} - K*e^{-rT})``.

    Should be ~0 (machine precision) for any correctly-priced European
    call/put pair. Used as a validation check in the test suite and
    exposed publicly since it is a useful sanity tool for consumers
    pricing against external/market quotes too.
    """
    c = as_float_array(call_price)
    p = as_float_array(put_price)
    s = as_float_array(spot)
    k = as_float_array(strike)
    r = as_float_array(rate)
    q = as_float_array(dividend_yield)
    t = as_float_array(time_to_expiry)
    forward_diff = s * np.exp(-q * t) - k * np.exp(-r * t)
    return (c - p) - forward_diff


def _validate_option_type(option_type: str) -> str:
    normalized = option_type.strip().lower()
    if normalized not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return normalized


def _validate_positive(name: str, x: FloatArray) -> None:
    if np.any(x <= 0.0):
        raise ValueError(f"{name} must be strictly positive, got values <= 0")


def _validate_nonnegative(name: str, x: FloatArray) -> None:
    if np.any(x < 0.0):
        raise ValueError(f"{name} must be non-negative, got negative values")
