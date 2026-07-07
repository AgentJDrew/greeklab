"""Path-dependent exotic option pricing via Monte Carlo.

Three payoff families that have no closed-form Black-Scholes analogue
because their value depends on the *path* the underlying takes, not
just its terminal value:

- **Arithmetic Asian** (average-price): payoff depends on the arithmetic
  average of the underlying sampled at a set of fixing dates. No
  closed form exists for the arithmetic average (only the geometric
  average has one, since a product of lognormals is lognormal but a
  sum is not) -- Monte Carlo is the standard approach (Hull Ch. 26).
- **Barrier** (knock-in / knock-out): payoff is extinguished or activated
  if the underlying crosses a barrier level at any point before
  expiry. Priced here via discrete path monitoring at each simulated
  time step (Hull Ch. 26).
- **American** (Longstaff & Schwartz 2001 Least-Squares Monte Carlo,
  "LSM"): estimates the optimal early-exercise policy by regressing
  continuation value on a polynomial basis of the current spot, at
  each exercise date, using only in-the-money paths.

All three simulate full GBM paths (not just terminal values, unlike
:mod:`greeklab.monte_carlo`) since path-dependence requires the
intermediate values.

References
----------
- Hull, J. C. *Options, Futures, and Other Derivatives* (11th ed.),
  Chapter 26 (exotic options: Asian, barrier options).
- Longstaff, F. A. and Schwartz, E. S. (2001). "Valuing American
  Options by Simulation: A Simple Least-Squares Approach." *Review of
  Financial Studies*, 14(1), 113-147.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "asian_arithmetic_mc",
    "AsianResult",
    "barrier_mc",
    "BarrierResult",
    "american_lsm_mc",
    "AmericanLSMResult",
]


def _simulate_gbm_paths(
    spot: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool,
) -> np.ndarray:
    """Simulate GBM paths on a uniform time grid, shape ``(n_sims, n_steps + 1)``.

    Column 0 is ``spot`` (t=0); each subsequent column is one exact GBM
    step (no discretization bias per step, since each step uses the
    exact lognormal transition density of GBM over ``dt``).
    """
    dt = time_to_expiry / n_steps
    drift = (rate - dividend_yield - 0.5 * sigma * sigma) * dt
    diffusion_coef = sigma * np.sqrt(dt)

    z = rng.standard_normal((n_paths, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)

    log_increments = drift + diffusion_coef * z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.concatenate([np.zeros((log_paths.shape[0], 1)), log_paths], axis=1)
    return spot * np.exp(log_paths)


def _pair_averaged_stats(values: np.ndarray, n_paths: int, antithetic: bool) -> tuple[float, float]:
    """Mean and standard error, correctly accounting for antithetic pairing."""
    if antithetic:
        pair_avg = 0.5 * (values[:n_paths] + values[n_paths:])
        return float(np.mean(pair_avg)), float(np.std(pair_avg, ddof=1) / np.sqrt(n_paths))
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(n_paths))


@dataclass(frozen=True, slots=True)
class AsianResult:
    """Result of an arithmetic Asian option Monte Carlo price."""

    price: float
    std_error: float
    n_paths: int


def asian_arithmetic_mc(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    option_type: str = "call",
    n_fixings: int = 50,
    n_paths: int = 50_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> AsianResult:
    """Price an arithmetic-average-price Asian option via Monte Carlo.

    Payoff at expiry: ``max(A - K, 0)`` for a call (``max(K - A, 0)``
    for a put), where :math:`A = \\frac{1}{n}\\sum_{i=1}^{n} S_{t_i}` is
    the arithmetic average of the underlying at ``n_fixings`` equally-
    spaced observation dates from ``t_1`` through expiry (inclusive).

    Parameters
    ----------
    spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
    n_fixings : int, default 50
        Number of equally-spaced averaging observations.
    n_paths, seed, antithetic
        Same as :func:`greeklab.monte_carlo.mc_european_price`.

    Returns
    -------
    AsianResult

    Notes
    -----
    **Sanity check** (validated in ``tests/test_exotics.py``): since
    averaging reduces the effective volatility the payoff is exposed
    to, an arithmetic Asian call is always cheaper than the vanilla
    European call at the same strike (Hull Ch. 26) -- confirmed against
    :func:`greeklab.black_scholes.bs_price` in the test suite.
    """
    rng = np.random.default_rng(seed)
    paths = _simulate_gbm_paths(
        spot, rate, dividend_yield, sigma, time_to_expiry, n_fixings, n_paths, rng, antithetic
    )
    # Exclude t=0 (column 0); average over the n_fixings observations
    # from t_1 through expiry.
    average = paths[:, 1:].mean(axis=1)
    payoff = np.clip(average - strike, 0.0, None) if option_type == "call" else np.clip(strike - average, 0.0, None)
    discounted = np.exp(-rate * time_to_expiry) * payoff

    price, std_error = _pair_averaged_stats(discounted, n_paths, antithetic)
    return AsianResult(price=price, std_error=std_error, n_paths=n_paths)


@dataclass(frozen=True, slots=True)
class BarrierResult:
    """Result of a barrier option Monte Carlo price."""

    price: float
    std_error: float
    n_paths: int
    fraction_knocked: float


def barrier_mc(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    n_steps: int = 100,
    n_paths: int = 50_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> BarrierResult:
    """Price a single-barrier option via discretely-monitored Monte Carlo.

    Supports the four standard barrier types: ``"down-and-out"``,
    ``"down-and-in"``, ``"up-and-out"``, ``"up-and-in"``. The barrier
    is checked at each of ``n_steps`` discrete monitoring dates (a
    discretely-monitored barrier, the standard practical approximation
    to continuous monitoring -- finer ``n_steps`` converges towards
    the continuous-monitoring price, which is itself slightly more
    conservative for "out" barriers since continuous monitoring can
    only knock out *more* paths than discrete monitoring, per
    Broadie-Glasserman-Kou 1997).

    Parameters
    ----------
    spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
    barrier : float
        The barrier level.
    barrier_type : {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}
        Which barrier condition applies.
    n_steps : int, default 100
        Number of discrete monitoring dates.
    n_paths, seed, antithetic
        Same as :func:`greeklab.monte_carlo.mc_european_price`.

    Returns
    -------
    BarrierResult
        Adds ``fraction_knocked``: the fraction of simulated paths that
        hit the barrier (useful as a sanity check that the barrier
        level is actually being tested by the simulation grid).

    Notes
    -----
    **Sanity checks** (validated in ``tests/test_exotics.py``):
    (1) a knock-out barrier option is always worth less than or equal
    to the corresponding vanilla European option, since a "knock-out"
    payoff is the vanilla payoff restricted to a subset of paths
    (Hull Ch. 26); (2) in/out parity: a knock-in plus the matching
    knock-out (same barrier, strike, type) must reproduce the vanilla
    European price, since every path is either knocked in or knocked
    out, never both or neither.
    """
    valid_types = {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}
    if barrier_type not in valid_types:
        raise ValueError(f"barrier_type must be one of {valid_types}, got {barrier_type!r}")

    rng = np.random.default_rng(seed)
    paths = _simulate_gbm_paths(
        spot, rate, dividend_yield, sigma, time_to_expiry, n_steps, n_paths, rng, antithetic
    )

    is_down = barrier_type.startswith("down")
    is_out = barrier_type.endswith("out")
    hit_barrier = (paths <= barrier).any(axis=1) if is_down else (paths >= barrier).any(axis=1)

    terminal = paths[:, -1]
    vanilla_payoff = (
        np.clip(terminal - strike, 0.0, None) if option_type == "call" else np.clip(strike - terminal, 0.0, None)
    )
    # "out" pays the vanilla payoff only on paths that never hit the
    # barrier; "in" pays only on paths that did hit it.
    active = ~hit_barrier if is_out else hit_barrier
    payoff = np.where(active, vanilla_payoff, 0.0)
    discounted = np.exp(-rate * time_to_expiry) * payoff

    price, std_error = _pair_averaged_stats(discounted, n_paths, antithetic)
    fraction_knocked = float(np.mean(hit_barrier))
    return BarrierResult(price=price, std_error=std_error, n_paths=n_paths, fraction_knocked=fraction_knocked)


@dataclass(frozen=True, slots=True)
class AmericanLSMResult:
    """Result of a Longstaff-Schwartz American option Monte Carlo price."""

    price: float
    std_error: float
    n_paths: int


def american_lsm_mc(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    time_to_expiry: float,
    option_type: str = "put",
    n_steps: int = 50,
    n_paths: int = 50_000,
    seed: int | None = None,
    poly_degree: int = 2,
) -> AmericanLSMResult:
    """Price an American option via Longstaff-Schwartz (2001) Least-Squares MC.

    At each of ``n_steps`` equally-spaced exercise dates (working
    backward from expiry), the holder decides whether to exercise
    immediately or continue. The continuation value is not directly
    observable in a simulation, so LSM estimates it by regressing the
    (discounted) realized future cash flow on a polynomial basis of the
    current spot price, **using only in-the-money paths** (the paths
    where the exercise decision is actually non-trivial) -- the core
    idea of Longstaff & Schwartz (2001).

    Parameters
    ----------
    spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
        Defaults to ``"put"`` since the American *call* on a
        non-dividend-paying stock has zero early-exercise premium (see
        Notes) and is a degenerate/uninteresting case for this method.
    n_steps : int, default 50
        Number of (equally-spaced) exercise opportunities before and
        including expiry.
    n_paths : int, default 50_000
        Number of simulated paths. LSM does not use antithetic
        variates here because the regression step's least-squares fit
        already benefits from having independent paths spanning the
        state space; combining antithetic pairing with the polynomial
        regression is a known source of subtle bias if not done
        carefully, so it is intentionally omitted for a simpler and
        provably-consistent implementation (Longstaff & Schwartz 2001;
        Glasserman 2003 Ch. 8 discusses the general dynamic-programming
        LSM setup this follows).
    seed : int, optional
        RNG seed for reproducibility.
    poly_degree : int, default 2
        Degree of the polynomial basis (1, S, S^2, ...) used in the
        continuation-value regression at each step.

    Returns
    -------
    AmericanLSMResult

    Notes
    -----
    **Sanity check** (validated in ``tests/test_exotics.py``): for an
    American *call* on a non-dividend-paying underlying
    (``dividend_yield == 0``), it is never optimal to exercise early
    (Merton 1973's classical no-early-exercise argument: the intrinsic
    value is always dominated by the time value of waiting when
    there's no dividend to capture), so the LSM American call price
    should closely match the vanilla European call price from
    :func:`greeklab.black_scholes.bs_price`, within Monte Carlo
    tolerance.
    """
    if n_paths < 1000:
        raise ValueError("n_paths should be >= 1000 for a stable LSM regression")

    rng = np.random.default_rng(seed)
    paths = _simulate_gbm_paths(
        spot, rate, dividend_yield, sigma, time_to_expiry, n_steps, n_paths, rng, antithetic=False
    )
    dt = time_to_expiry / n_steps
    disc_step = np.exp(-rate * dt)

    def intrinsic(s: np.ndarray) -> np.ndarray:
        return np.clip(s - strike, 0.0, None) if option_type == "call" else np.clip(strike - s, 0.0, None)

    # cash_flow[i] = the (not-yet-discounted-to-t=0) payoff realized by
    # path i at the *time it is currently assumed to exercise*, tracked
    # via `exercise_time` so we can discount each path back to t=0
    # using its own individually-realized exercise date at the end.
    n_sims = paths.shape[0]
    cash_flow = intrinsic(paths[:, -1])
    exercise_step = np.full(n_sims, n_steps)

    for step in range(n_steps - 1, 0, -1):
        current_spot = paths[:, step]
        itm = intrinsic(current_spot) > 0.0
        if not np.any(itm):
            continue

        # Regress the *discounted-to-this-step* future cash flow (using
        # each path's own currently-recorded exercise step) onto a
        # polynomial basis of the in-the-money current spot.
        steps_ahead = exercise_step[itm] - step
        discounted_future = cash_flow[itm] * (disc_step**steps_ahead)

        basis = np.vander(current_spot[itm], N=poly_degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(basis, discounted_future, rcond=None)
        continuation_value = basis @ coeffs

        immediate = intrinsic(current_spot[itm])
        exercise_now = immediate > continuation_value

        itm_indices = np.flatnonzero(itm)
        exercise_indices = itm_indices[exercise_now]
        cash_flow[exercise_indices] = immediate[exercise_now]
        exercise_step[exercise_indices] = step

    steps_ahead_final = exercise_step
    discounted_payoff = cash_flow * (disc_step**steps_ahead_final)

    price = float(np.mean(discounted_payoff))
    std_error = float(np.std(discounted_payoff, ddof=1) / np.sqrt(n_sims))
    return AmericanLSMResult(price=price, std_error=std_error, n_paths=n_paths)
