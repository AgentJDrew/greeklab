"""Cox-Ross-Rubinstein (1979) binomial tree pricing.

A discrete-time lattice approximation to Black-Scholes-Merton that
additionally supports American-style early exercise, which has no
closed-form solution under continuous-time BSM. As the number of steps
grows, the CRR European price converges to the closed-form BSM price
(this convergence is validated in the test suite); the American price
is always ``>= `` the corresponding European price, since early
exercise is an option the American holder can decline to use.

The implementation is vectorized "backward induction over columns":
each expiry-to-valuation step updates the entire vector of node values
in one NumPy operation, so pricing with a few thousand steps (needed
for tight BSM convergence) stays fast.

References
----------
- Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). "Option Pricing:
  A Simplified Approach." *Journal of Financial Economics*, 7(3),
  229-263.
- Hull, J. C. *Options, Futures, and Other Derivatives* (11th ed.),
  Chapter 13 (binomial trees).
"""

from __future__ import annotations

import numpy as np

from ._typing import FloatArray

__all__ = ["crr_european_price", "crr_american_price"]


def _crr_lattice_params(
    sigma: float, rate: float, dividend_yield: float, time_to_expiry: float, n_steps: int
) -> tuple[float, float, float, float]:
    """Up/down factors, risk-neutral probability, and per-step discount.

    Standard CRR (1979) parameterization:

    .. math::
        u = e^{\\sigma\\sqrt{\\Delta t}}, \\quad d = 1/u, \\quad
        p = \\frac{e^{(r-q)\\Delta t} - d}{u - d}

    which recombines (an up-then-down move returns to the same price
    as down-then-up) and converges to GBM as ``n_steps -> infinity``.
    """
    dt = time_to_expiry / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-rate * dt)
    growth = np.exp((rate - dividend_yield) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.6f} outside [0, 1]: the chosen "
            f"n_steps={n_steps} gives too coarse a time step (dt={dt:.6f}) for "
            f"this sigma/rate/dividend_yield combination. Increase n_steps."
        )
    return u, d, p, disc


def _terminal_payoffs(
    spot: float, u: float, d: float, n_steps: int, strike: float, option_type: str
) -> FloatArray:
    j = np.arange(n_steps + 1)
    terminal_spots = spot * (u ** (n_steps - j)) * (d**j)
    if option_type == "call":
        return np.clip(terminal_spots - strike, 0.0, None)
    return np.clip(strike - terminal_spots, 0.0, None)


def crr_european_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    option_type: str = "call",
    n_steps: int = 500,
) -> float:
    """Price a European option on a Cox-Ross-Rubinstein binomial lattice.

    Parameters
    ----------
    spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
        Scalars only (a single quote) -- for vectorized pricing across a
        grid, prefer :func:`greeklab.black_scholes.bs_price` (exact
        closed form) and use this function only where the lattice
        machinery itself is of interest, or via a Python-level loop.
    n_steps : int, default 500
        Number of time steps in the lattice. Larger ``n_steps`` gives
        closer convergence to the true Black-Scholes price (at
        ``O(n_steps^2)`` cost) but binomial convergence is famously
        oscillatory in ``n_steps`` (Hull Ch. 13) -- 500+ steps gives
        sub-cent accuracy for typical equity-option parameters.

    Returns
    -------
    float
        The lattice-implied European option price.

    Notes
    -----
    Converges to :func:`greeklab.black_scholes.bs_price` as
    ``n_steps -> infinity``; see ``tests/test_binomial.py`` for the
    convergence check against the closed-form BSM price.
    """
    _validate_inputs(spot, strike, sigma, time_to_expiry, n_steps)
    if time_to_expiry == 0.0:
        return float(np.clip(spot - strike, 0.0, None) if option_type == "call" else np.clip(strike - spot, 0.0, None))

    u, d, p, disc = _crr_lattice_params(sigma, rate, dividend_yield, time_to_expiry, n_steps)
    values = _terminal_payoffs(spot, u, d, n_steps, strike, option_type)

    # Backward induction: at each step, discount the risk-neutral
    # expectation of the two child nodes. No early-exercise check here
    # (that's the only difference vs. crr_american_price).
    for _ in range(n_steps):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])

    return float(values[0])


def crr_american_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    option_type: str = "call",
    n_steps: int = 500,
) -> float:
    """Price an American option (early exercise allowed) on a CRR lattice.

    Identical lattice construction to :func:`crr_european_price`, but
    at every interior node the holder's value is
    ``max(continuation_value, intrinsic_value)`` -- the classic
    dynamic-programming treatment of early exercise (Cox-Ross-Rubinstein
    1979; Hull Ch. 13). There is no general closed-form solution for
    American options under continuous-time BSM, which is precisely why
    the lattice/PDE/simulation methods in this library exist.

    Parameters
    ----------
    Same as :func:`crr_european_price`.

    Returns
    -------
    float
        The lattice-implied American option price. Always
        ``>= `` the European price for the same parameters (early
        exercise is a right, never an obligation) -- validated in
        ``tests/test_binomial.py``. For a non-dividend-paying call
        (``dividend_yield == 0``), American and European calls are
        provably equal (never optimal to exercise early with no
        dividends to capture) -- also validated there.
    """
    _validate_inputs(spot, strike, sigma, time_to_expiry, n_steps)
    if time_to_expiry == 0.0:
        return float(np.clip(spot - strike, 0.0, None) if option_type == "call" else np.clip(strike - spot, 0.0, None))

    u, d, p, disc = _crr_lattice_params(sigma, rate, dividend_yield, time_to_expiry, n_steps)
    values = _terminal_payoffs(spot, u, d, n_steps, strike, option_type)

    for step in range(n_steps - 1, -1, -1):
        continuation = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        j = np.arange(step + 1)
        node_spots = spot * (u ** (step - j)) * (d**j)
        intrinsic = (
            np.clip(node_spots - strike, 0.0, None)
            if option_type == "call"
            else np.clip(strike - node_spots, 0.0, None)
        )
        values = np.maximum(continuation, intrinsic)

    return float(values[0])


def _validate_inputs(
    spot: float, strike: float, sigma: float, time_to_expiry: float, n_steps: int
) -> None:
    if spot <= 0.0:
        raise ValueError(f"spot must be strictly positive, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be strictly positive, got {strike}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if time_to_expiry < 0.0:
        raise ValueError(f"time_to_expiry must be non-negative, got {time_to_expiry}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
