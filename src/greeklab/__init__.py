"""greeklab: rigorous options and derivatives pricing in pure NumPy/SciPy.

Public API surface (see each submodule's docstring for full detail and
academic references):

- :mod:`greeklab.black_scholes` -- BSM European pricing + analytical Greeks
  (delta/gamma/vega/theta/rho + vanna/volga/charm).
- :mod:`greeklab.implied_vol` -- Newton-Raphson (seeded) + Brent-fallback
  implied volatility solver.
- :mod:`greeklab.binomial` -- Cox-Ross-Rubinstein binomial trees, European
  and American exercise.
- :mod:`greeklab.monte_carlo` -- GBM Monte Carlo for European options with
  antithetic + control-variate variance reduction.
- :mod:`greeklab.exotics` -- Monte Carlo pricing of arithmetic Asian,
  Barrier (knock-in/knock-out), and American options (Longstaff-Schwartz).
- :mod:`greeklab.heston` -- Heston (1993) stochastic-volatility European
  pricing via Fourier/characteristic-function inversion, plus a full-
  truncation Euler Monte Carlo scheme for cross-validation.
- :mod:`greeklab.surface` -- implied-volatility surface construction from
  a price grid, plus an optional SVI-slice fit.
"""

from __future__ import annotations

from .binomial import crr_american_price, crr_european_price
from .black_scholes import Greeks, bs_greeks, bs_price, put_call_parity_residual
from .exotics import (
    AmericanLSMResult,
    AsianResult,
    BarrierResult,
    american_lsm_mc,
    asian_arithmetic_mc,
    barrier_mc,
)
from .heston import HestonParams, heston_mc_price, heston_price_fourier
from .implied_vol import ImpliedVolResult, implied_vol
from .monte_carlo import MonteCarloResult, mc_european_price
from .surface import SVISliceFit, fit_svi_slice, iv_surface

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # black_scholes
    "bs_price",
    "bs_greeks",
    "Greeks",
    "put_call_parity_residual",
    # implied_vol
    "implied_vol",
    "ImpliedVolResult",
    # binomial
    "crr_european_price",
    "crr_american_price",
    # monte_carlo
    "mc_european_price",
    "MonteCarloResult",
    # exotics
    "asian_arithmetic_mc",
    "AsianResult",
    "barrier_mc",
    "BarrierResult",
    "american_lsm_mc",
    "AmericanLSMResult",
    # heston
    "HestonParams",
    "heston_price_fourier",
    "heston_mc_price",
    # surface
    "iv_surface",
    "fit_svi_slice",
    "SVISliceFit",
]
