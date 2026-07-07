"""Implied-volatility surface construction and SVI slice fitting.

Given a grid of observed option prices across strikes and expiries,
builds the corresponding implied-volatility surface by re-using
:func:`greeklab.implied_vol.implied_vol` at every grid point. Also
provides an optional per-expiry parametric smile fit using Gatheral's
(2004) SVI ("stochastic volatility inspired") parameterization, a
five-parameter curve widely used in practice to interpolate/extrapolate
a smile smoothly between (and beyond) quoted strikes.

References
----------
- Gatheral, J. (2004). "A Parsimonious Arbitrage-Free Implied
  Volatility Parameterization with Application to the Valuation of
  Volatility Derivatives." Presentation, Global Derivatives & Risk
  Management, Madrid (the original SVI parameterization).
- Gatheral, J. and Jacquier, A. (2014). "Arbitrage-Free SVI Volatility
  Surfaces." *Quantitative Finance*, 14(1), 59-71 (conditions for a
  single SVI slice to be free of butterfly arbitrage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ._typing import FloatArray
from .implied_vol import implied_vol

__all__ = ["iv_surface", "fit_svi_slice", "SVISliceFit"]


def iv_surface(
    price_grid: FloatArray,
    spot: float,
    strikes: FloatArray,
    expiries: FloatArray,
    rate: float,
    dividend_yield: float,
    option_type: str = "call",
) -> FloatArray:
    """Build an implied-volatility surface from a grid of option prices.

    Parameters
    ----------
    price_grid : FloatArray, shape (n_expiries, n_strikes)
        Observed/model option prices, one row per expiry, one column
        per strike.
    spot : float
        Current underlying price (shared across the whole grid).
    strikes : FloatArray, shape (n_strikes,)
        Strike prices corresponding to the columns of ``price_grid``.
    expiries : FloatArray, shape (n_expiries,)
        Times to expiry (years) corresponding to the rows of
        ``price_grid``.
    rate, dividend_yield : float
        Shared continuously-compounded rate and dividend yield across
        the whole grid.
    option_type : {"call", "put"}
        Which payoff ``price_grid`` holds.

    Returns
    -------
    FloatArray, shape (n_expiries, n_strikes)
        Implied volatility at each grid point. Entries where the
        solver did not converge (see
        :attr:`greeklab.implied_vol.ImpliedVolResult.converged`) are
        set to ``nan`` rather than a misleading/unconverged value.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    price_grid = np.asarray(price_grid, dtype=np.float64)

    if price_grid.shape != (expiries.size, strikes.size):
        raise ValueError(
            f"price_grid shape {price_grid.shape} must equal "
            f"(len(expiries), len(strikes)) = ({expiries.size}, {strikes.size})"
        )

    iv_grid = np.full(price_grid.shape, np.nan)
    for i, t in enumerate(expiries):
        for j, k in enumerate(strikes):
            result = implied_vol(
                float(price_grid[i, j]), spot, float(k), rate, dividend_yield, float(t), option_type
            )
            # implied_vol always returns a single ImpliedVolResult (never
            # a list) when every argument is a scalar, as they all are
            # here -- narrow the type explicitly rather than with a
            # runtime assert (which is stripped under `python -O`).
            if isinstance(result, list):  # pragma: no cover - unreachable with scalar args
                raise TypeError("implied_vol unexpectedly returned a list for scalar inputs")
            if result.converged:
                iv_grid[i, j] = result.sigma
    return iv_grid


@dataclass(frozen=True, slots=True)
class SVISliceFit:
    """Result of fitting Gatheral's (2004) raw SVI parameterization to one expiry slice.

    Attributes
    ----------
    a, b, rho, m, sigma : float
        The five raw-SVI parameters (Gatheral 2004 notation) satisfying

        .. math::
            w(k) = a + b\\left(\\rho(k - m) +
                \\sqrt{(k-m)^2 + \\sigma^2}\\right)

        where :math:`w(k) = \\sigma_{BS}^2(k) \\cdot T` is the total
        implied variance at log-moneyness :math:`k = \\ln(K/F)`
        (:math:`F` the forward price) and :math:`T` is this slice's
        time to expiry.
    time_to_expiry : float
        The expiry this slice was fit to (needed to convert ``w(k)``
        back to Black-Scholes volatility via ``sqrt(w(k) / T)``).
    rmse : float
        Root-mean-squared error of the fit, in total-variance units.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    time_to_expiry: float
    rmse: float

    def total_variance(self, log_moneyness: FloatArray) -> FloatArray:
        """Evaluate the fitted total variance :math:`w(k)` at given log-moneyness."""
        k = np.asarray(log_moneyness, dtype=np.float64)

        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def implied_vol(self, log_moneyness: FloatArray) -> FloatArray:
        """Evaluate the fitted Black-Scholes implied volatility at given log-moneyness."""
        w = self.total_variance(log_moneyness)
        return np.sqrt(np.clip(w, 0.0, None) / self.time_to_expiry)


def fit_svi_slice(
    log_moneyness: FloatArray,
    implied_vols: FloatArray,
    time_to_expiry: float,
    initial_guess: tuple[float, float, float, float, float] | None = None,
) -> SVISliceFit:
    """Fit Gatheral's (2004) raw SVI parameterization to one expiry's smile.

    Minimizes the sum of squared errors between the model's total
    variance :math:`w(k) = a + b(\\rho(k-m) + \\sqrt{(k-m)^2+\\sigma^2})`
    and the observed total variance :math:`\\sigma_{BS}^2(k) \\cdot T`
    at each quoted log-moneyness ``k = ln(K/F)``, via bounded nonlinear
    least squares (``scipy.optimize.least_squares``, trust-region
    reflective algorithm).

    Parameters
    ----------
    log_moneyness : FloatArray
        Log-moneyness :math:`k = \\ln(K/F)` of each quote (``F`` the
        forward price for this expiry).
    implied_vols : FloatArray
        Observed Black-Scholes implied volatility at each
        ``log_moneyness`` point (same length).
    time_to_expiry : float
        This slice's time to expiry, in years.
    initial_guess : tuple of 5 floats, optional
        Starting point ``(a, b, rho, m, sigma)`` for the optimizer. If
        omitted, a data-driven default is used: ``a`` at the minimum
        observed total variance, ``b`` at a modest positive slope,
        ``rho=0``, ``m`` at the mean log-moneyness, ``sigma`` at the
        log-moneyness spread.

    Returns
    -------
    SVISliceFit

    Raises
    ------
    ValueError
        If fewer than 5 quotes are given (the parameterization has 5
        free parameters, so the fit is underdetermined below that).

    Notes
    -----
    This fit only enforces the parameter bounds ``b >= 0``,
    ``rho in [-1, 1]``, ``sigma > 0`` (required for :math:`w(k)` to be
    a valid, non-negative-sloped total-variance curve at its wings) --
    it does **not** verify the full Gatheral-Jacquier (2014)
    butterfly-arbitrage-free conditions on the fitted parameters,
    which is a stricter and separate check. Treat this as a smoothing/
    interpolation tool, not an arbitrage certifier.
    """
    k = np.asarray(log_moneyness, dtype=np.float64)
    iv = np.asarray(implied_vols, dtype=np.float64)
    if k.shape != iv.shape:
        raise ValueError("log_moneyness and implied_vols must have the same shape")
    if k.size < 5:
        raise ValueError(f"SVI has 5 free parameters; need >= 5 quotes, got {k.size}")

    w_observed = iv**2 * time_to_expiry

    if initial_guess is None:
        a0 = float(np.min(w_observed))
        b0 = 0.1
        rho0 = 0.0
        m0 = float(np.mean(k))
        sigma0 = float(max(np.std(k), 0.01))
        initial_guess = (a0, b0, rho0, m0, sigma0)

    lower_bounds = [-np.inf, 0.0, -1.0, -np.inf, 1e-6]
    upper_bounds = [np.inf, np.inf, 1.0, np.inf, np.inf]

    def residuals(params: FloatArray) -> FloatArray:
        a, b, rho, m, sigma = params
        w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return np.asarray(w_model - w_observed, dtype=np.float64)

    result = least_squares(
        residuals, x0=initial_guess, bounds=(lower_bounds, upper_bounds), method="trf"
    )
    a, b, rho, m, sigma = result.x
    rmse = float(np.sqrt(np.mean(result.fun**2)))

    return SVISliceFit(a=a, b=b, rho=rho, m=m, sigma=sigma, time_to_expiry=time_to_expiry, rmse=rmse)
