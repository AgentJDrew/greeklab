"""Monte Carlo pricing of European options under geometric Brownian motion.

Simulates terminal spot prices exactly (no discretization bias, since
GBM has a closed-form transition density), discounts the average
payoff, and reports the price alongside its Monte Carlo standard
error. Two variance-reduction techniques are combined:

1. **Antithetic variates**: for every standard normal draw ``Z`` used to
   simulate a path, also simulate the mirrored path using ``-Z``. Since
   the payoff is a monotonic (or piecewise-monotonic) function of the
   terminal price for vanilla options, this induces negative
   correlation between paired paths and shrinks variance versus
   independent draws of the same total path count.
2. **Control variate**: the terminal underlying price itself,
   :math:`S_T`, has a known risk-neutral expectation
   :math:`E[S_T] = S_0 e^{(r-q)T}`. Regressing the payoff against
   :math:`S_T` and subtracting the (scaled) simulation error in
   :math:`S_T` removes another source of noise essentially for free
   (Glasserman 2003, Ch. 4; Hull Ch. 21).

References
----------
- Hull, J. C. *Options, Futures, and Other Derivatives* (11th ed.),
  Chapter 21 (Monte Carlo simulation, variance-reduction techniques).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*,
  Springer. Chapter 4 (variance reduction).
- Boyle, P. P. (1977). "Options: A Monte Carlo Approach." *Journal of
  Financial Economics*, 4(3), 323-338 (the original application of MC
  to option pricing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["mc_european_price", "MonteCarloResult"]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Result of a Monte Carlo option-price estimate.

    Attributes
    ----------
    price : float
        The estimated (discounted, averaged) option price.
    std_error : float
        Standard error of the estimate, i.e. the sample standard
        deviation of the discounted payoffs divided by
        ``sqrt(n_effective_paths)``. A price from an independent
        re-run should fall within a few multiples of ``std_error`` of
        this one with high probability (CLT-based confidence interval).
    n_paths : int
        Number of independent path pairs simulated (with antithetic
        variates, ``2 * n_paths`` total path evaluations are used).
    """

    price: float
    std_error: float
    n_paths: int


def mc_european_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MonteCarloResult:
    """Price a European option by Monte Carlo simulation under GBM.

    Simulates :math:`S_T = S_0 \\exp\\left((r - q -
    \\sigma^2/2)T + \\sigma\\sqrt{T} Z\\right)` for :math:`Z \\sim
    N(0,1)` -- the *exact* one-step GBM transition, so there is no
    time-discretization error, only Monte Carlo sampling error (which
    :attr:`MonteCarloResult.std_error` quantifies and variance
    reduction shrinks).

    Parameters
    ----------
    spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
        Scalars only.
    n_paths : int, default 100_000
        Number of independent path pairs. With ``antithetic=True``
        this simulates ``2 * n_paths`` terminal prices total (a path
        and its antithetic mirror); the *reported* standard error
        already accounts for the induced correlation.
    seed : int, optional
        Seed for the NumPy random generator (``numpy.random.default_
        rng``), for reproducible results.
    antithetic : bool, default True
        Whether to use antithetic variates (see module docstring).
    control_variate : bool, default True
        Whether to use :math:`S_T` as a control variate (see module
        docstring). Uses the standard optimal-coefficient control-
        variate estimator (sample covariance / sample variance).

    Returns
    -------
    MonteCarloResult
        Price estimate with standard error -- see
        :class:`MonteCarloResult`.

    Notes
    -----
    Converges to :func:`greeklab.black_scholes.bs_price` as
    ``n_paths -> infinity``; with variance reduction enabled and a
    fixed seed, the estimate should land within a handful of standard
    errors of the closed-form BSM price at ``n_paths=100_000`` for
    typical equity-option parameters -- see
    ``tests/test_monte_carlo.py``.
    """
    _validate_inputs(spot, strike, sigma, time_to_expiry, n_paths)
    option_type = _validate_option_type(option_type)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    drift = (rate - dividend_yield - 0.5 * sigma * sigma) * time_to_expiry
    diffusion_coef = sigma * np.sqrt(time_to_expiry)

    z_all = np.concatenate([z, -z]) if antithetic else z

    terminal_spot = spot * np.exp(drift + diffusion_coef * z_all)
    payoff = _payoff(terminal_spot, strike, option_type)
    discount = np.exp(-rate * time_to_expiry)
    discounted_payoff = discount * payoff

    if control_variate:
        # Control variate Y = S_T, with known mean E[Y] = S0*e^{(r-q)T}.
        # Optimal coefficient c* = Cov(payoff, Y) / Var(Y); the adjusted
        # estimator payoff - c*(Y - E[Y]) has the same expectation as
        # payoff but (weakly) lower variance whenever payoff and S_T are
        # correlated, which they are for any monotonic-in-S_T payoff
        # (Glasserman 2003, Ch. 4.1).
        control_mean = spot * np.exp((rate - dividend_yield) * time_to_expiry)
        cov_matrix = np.cov(discounted_payoff, terminal_spot, ddof=1)
        var_control = cov_matrix[1, 1]
        if var_control > 0.0:
            c_star = cov_matrix[0, 1] / var_control
            adjusted = discounted_payoff - c_star * (terminal_spot - control_mean)
        else:
            adjusted = discounted_payoff
    else:
        adjusted = discounted_payoff

    price = float(np.mean(adjusted))
    # Effective sample count for the standard-error denominator: with
    # antithetic pairing, the *paths* number 2*n_paths, but the
    # reported std_error must reflect the (lower) variance of the
    # *paired-averaged* estimator, not treat all 2*n_paths draws as
    # independent. We therefore average each antithetic pair first,
    # then take the standard error of those n_paths pair-averages --
    # the textbook-correct treatment (Glasserman 2003, Ch. 4.2) that
    # avoids understating the standard error by a spurious sqrt(2).
    if antithetic:
        pair_avg = 0.5 * (adjusted[:n_paths] + adjusted[n_paths:])
        std_error = float(np.std(pair_avg, ddof=1) / np.sqrt(n_paths))
    else:
        std_error = float(np.std(adjusted, ddof=1) / np.sqrt(n_paths))

    return MonteCarloResult(price=price, std_error=std_error, n_paths=n_paths)


def _payoff(terminal_spot: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.clip(terminal_spot - strike, 0.0, None)
    return np.clip(strike - terminal_spot, 0.0, None)


def _validate_option_type(option_type: str) -> str:
    normalized = option_type.strip().lower()
    if normalized not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return normalized


def _validate_inputs(spot: float, strike: float, sigma: float, time_to_expiry: float, n_paths: int) -> None:
    if spot <= 0.0:
        raise ValueError(f"spot must be strictly positive, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be strictly positive, got {strike}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if time_to_expiry < 0.0:
        raise ValueError(f"time_to_expiry must be non-negative, got {time_to_expiry}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
