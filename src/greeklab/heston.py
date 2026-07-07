"""Heston (1993) stochastic-volatility model: Fourier pricing + Monte Carlo.

The Heston model lets volatility itself follow a mean-reverting square-
root (CIR) diffusion, correlated with the spot's own Brownian motion --
capturing the volatility clustering, mean reversion, and volatility
skew/smile that constant-volatility Black-Scholes cannot:

.. math::
    dS_t = (r - q) S_t\\,dt + \\sqrt{v_t}\\,S_t\\,dW_t^S

    dv_t = \\kappa(\\theta - v_t)\\,dt + \\xi\\sqrt{v_t}\\,dW_t^v

    dW_t^S\\,dW_t^v = \\rho\\,dt

with parameters :math:`\\kappa` (mean-reversion speed), :math:`\\theta`
(long-run variance), :math:`\\xi` (vol-of-vol), :math:`\\rho`
(spot-vol correlation, typically negative -- the equity "leverage
effect"), and :math:`v_0` (initial variance).

Two independent pricing routes are implemented and cross-validated
against each other:

1. **Fourier / characteristic-function inversion** (semi-analytical):
   Heston's (1993) original two-probability formulation,
   :math:`C = S e^{-qT} P_1 - K e^{-rT} P_2`, where each
   :math:`P_j` is recovered from the characteristic function of
   :math:`\\ln S_T` by a single real quadrature (Gil-Pelaez inversion).
   This is essentially exact (limited only by quadrature tolerance),
   and is the reference value the Monte Carlo route is checked
   against. As :math:`\\xi \\to 0` with :math:`v_0 = \\theta =
   \\sigma^2`, it reduces to the Black-Scholes price to ~1e-7 (see
   ``tests/test_heston.py``).
2. **Monte Carlo** (full-truncation Euler scheme, Lord, Koekkoek & Van
   Dijk 2010): discretizes the SDE system directly. The CIR variance
   process can (in an Euler discretization) step negative even though
   the true process cannot; "full truncation" replaces :math:`v_t`
   with :math:`\\max(v_t, 0)` inside the drift/diffusion coefficients
   at each step -- the discretization scheme found most robust in Lord
   et al.'s (2010) comparative study of Heston Euler schemes.

References
----------
- Heston, S. L. (1993). "A Closed-Form Solution for Options with
  Stochastic Volatility with Applications to Bond and Currency
  Options." *Review of Financial Studies*, 6(2), 327-343 (the original
  two-probability :math:`P_1`/:math:`P_2` formulation implemented here).
- Albrecher, H., Mayer, P., Schoutens, W., and Tistaert, J. (2007). "The
  Little Heston Trap." *Wilmott Magazine*, January 2007, 83-92 (the
  numerically-stable characteristic-function branch used here -- the
  textbook sign choice for the auxiliary root ``d`` suffers a
  discontinuous complex branch cut that produces wrong prices, or even
  ``NaN``, at long maturities / high vol-of-vol; this was reproduced
  and confirmed during development, see :func:`_heston_charfunc`).
- Lord, R., Koekkoek, R., and Van Dijk, D. (2010). "A Comparison of
  Biased Simulation Schemes for Stochastic Volatility Models."
  *Quantitative Finance*, 10(2), 177-194 (full-truncation Euler
  scheme for the CIR variance process).
- Gatheral, J. *The Volatility Surface: A Practitioner's Guide*, Wiley
  (2006), Ch. 2 (Heston model background).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

__all__ = ["HestonParams", "heston_price_fourier", "heston_mc_price"]


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Heston (1993) stochastic-volatility model parameters.

    Attributes
    ----------
    kappa : float
        Mean-reversion speed of the variance process. Must be positive.
    theta : float
        Long-run mean variance level (note: *variance*, not
        volatility -- ``sqrt(theta)`` is the long-run annualized vol).
        Must be positive.
    xi : float
        Volatility of variance ("vol-of-vol"). Must be positive.
    rho : float
        Correlation between the spot and variance Brownian motions.
        Must be in ``[-1, 1]``; typically negative for equities (the
        "leverage effect": falling prices coincide with rising vol).
    v0 : float
        Initial (t=0) variance. Must be positive.

    Notes
    -----
    The **Feller condition** :math:`2\\kappa\\theta \\geq \\xi^2`
    guarantees the variance process ``v_t`` never reaches exactly zero
    (a strictly positive CIR process). It is *not* enforced here --
    many empirically-calibrated Heston parameter sets violate it (the
    process still remains non-negative, just no longer strictly
    positive at isolated times) -- but :meth:`feller_satisfied` lets
    callers check it.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.xi <= 0.0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.v0 <= 0.0:
            raise ValueError(f"v0 must be positive, got {self.v0}")

    def feller_satisfied(self) -> bool:
        """Whether ``2 * kappa * theta >= xi**2`` (the Feller condition)."""
        return 2.0 * self.kappa * self.theta >= self.xi**2


def _heston_charfunc(
    u: complex | np.ndarray,
    params: HestonParams,
    spot: float,
    rate: float,
    dividend_yield: float,
    time_to_expiry: float,
    branch: int,
) -> npt.NDArray[np.complex128]:
    """Heston (1993) characteristic function of :math:`\\ln S_T`, evaluated
    at (possibly complex/array) frequency ``u``, for probability branch
    ``branch`` (``1`` or ``2`` -- see :func:`_p_j`).

    Uses the numerically-stable root of the auxiliary quantity
    :math:`d` (Albrecher, Mayer, Schoutens & Tistaert 2007, "The Little
    Heston Trap"): the textbook formula picks the ``+d`` root, whose
    ``g = (b - rho*xi*iu + d) / (b - rho*xi*iu - d)`` drives
    :math:`\\log((1 - g e^{dT}) / (1 - g))` through a discontinuous
    complex branch cut at long maturities / high vol-of-vol, producing
    wrong prices or ``NaN``. Using the ``-d`` root instead gives the
    mathematically equivalent (same characteristic function value) but
    numerically stable branch. This was independently reproduced during
    development: at ``T=5`` the naive branch already diverges from an
    independent Monte Carlo price by ~20 points, and the trap worsens
    to outright ``NaN`` by ``T=10`` for the same parameters, while the
    stable branch used here tracks the Monte Carlo price to within a
    fraction of a standard error at every tested maturity up to
    ``T=20`` -- see ``tests/test_heston.py``.
    """
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0
    x0 = np.log(spot)
    iu = 1j * u

    if branch == 1:
        b = kappa - rho * xi
        alpha_term = 0.5 * iu
    else:
        b = kappa
        alpha_term = -0.5 * iu

    a = kappa * theta
    d = np.sqrt((rho * xi * iu - b) ** 2 - xi**2 * (2.0 * alpha_term - u**2))
    d = -d  # stable branch (Albrecher et al. 2007) -- see docstring above.
    g = (b - rho * xi * iu + d) / (b - rho * xi * iu - d)

    exp_dT = np.exp(d * time_to_expiry)
    C = (rate - dividend_yield) * iu * time_to_expiry + (a / xi**2) * (
        (b - rho * xi * iu + d) * time_to_expiry - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = ((b - rho * xi * iu + d) / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.asarray(np.exp(C + D * v0 + iu * x0), dtype=np.complex128)


def _p_j(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_expiry: float,
    params: HestonParams,
    branch: int,
) -> float:
    """Gil-Pelaez inversion of the Heston (1993) "in-the-money" probability
    :math:`P_j`, ``branch`` in ``{1, 2}``:

    .. math::
        P_j = \\frac{1}{2} + \\frac{1}{\\pi}\\int_0^\\infty
            \\text{Re}\\left[\\frac{e^{-iu\\ln K}\\,\\phi_j(u)}{iu}\\right] du
    """

    def integrand(u: float) -> float:
        phi = _heston_charfunc(u, params, spot, rate, dividend_yield, time_to_expiry, branch)
        value = np.exp(-1j * u * np.log(strike)) * phi / (1j * u)
        return float(np.real(value))

    # Lower limit is a small positive epsilon, not 0: the integrand has
    # a removable 0/0 singularity at u=0 (both numerator and 1/(iu)
    # diverge there in a way that cancels in the limit), which
    # `scipy.integrate.quad` handles more reliably when told to start
    # just past it rather than evaluating exactly at the singular point.
    #
    # Upper limit 2000.0 (not the more commonly-seen 100-200): at very
    # short time_to_expiry the integrand decays much more slowly in u,
    # and truncating at u=200 silently under-integrates -- reproduced
    # during development against Alan Lewis's high-precision reference
    # prices (financepress.com/2019/02/15/heston-model-reference-prices,
    # T=0.01 "extreme parameters" panel): truncating at u=200 gives a
    # call price wrong by 4 orders of magnitude (and even a negative,
    # unphysical P_j) at strike=105, while u=2000 recovers the
    # reference value to ~1e-8 relative error. u=2000 costs no
    # measurable extra runtime (quad's adaptive subdivision spends its
    # budget where the integrand actually has mass) and does not
    # perturb the normal-maturity case -- see
    # ``tests/test_heston.py::test_heston_extreme_short_maturity``.
    integral, _ = quad(integrand, 1e-10, 2000.0, limit=500)
    return 0.5 + integral / np.pi


def heston_price_fourier(
    params: HestonParams,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> float:
    """Price a European option under Heston (1993) via Fourier inversion.

    Uses Heston's (1993) original two-probability formulation:

    .. math::
        C = S e^{-qT} P_1 - K e^{-rT} P_2

    where each :math:`P_j \\in [0, 1]` is a risk-neutral "probability
    of finishing in-the-money" under a different numeraire (:math:`P_1`
    under the share-price measure, :math:`P_2` under the money-market
    measure), recovered from the Heston characteristic function via
    Gil-Pelaez Fourier inversion (see :func:`_p_j`). The put price
    follows from put-call parity applied to this call price.

    Parameters
    ----------
    params : HestonParams
        The model parameters.
    spot, strike, rate, dividend_yield, time_to_expiry
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
    option_type : {"call", "put"}

    Returns
    -------
    float
        The Heston European option price.

    Notes
    -----
    **Validated against a published reference value** in
    ``tests/test_heston.py`` (a standard Heston (1993) parameter set
    widely cited in the numerical-methods literature, e.g. Albrecher
    et al. 2007), and cross-validated against :func:`heston_mc_price`
    (an independent simulation-based route) to within a few Monte
    Carlo standard errors. Also validated to reduce to
    :func:`greeklab.black_scholes.bs_price` in the ``xi -> 0``,
    ``v0 = theta = sigma**2`` limit (constant volatility), matching to
    ~1e-7.
    """
    if time_to_expiry <= 0.0:
        return float(np.clip(spot - strike, 0.0, None) if option_type == "call" else np.clip(strike - spot, 0.0, None))

    p1 = _p_j(spot, strike, rate, dividend_yield, time_to_expiry, params, branch=1)
    p2 = _p_j(spot, strike, rate, dividend_yield, time_to_expiry, params, branch=2)

    call = float(spot * np.exp(-dividend_yield * time_to_expiry) * p1 - strike * np.exp(-rate * time_to_expiry) * p2)
    call = max(call, 0.0)
    if option_type == "call":
        return call
    # Put-call parity: P = C - S*e^{-qT} + K*e^{-rT}.
    put = call - float(spot * np.exp(-dividend_yield * time_to_expiry)) + float(strike * np.exp(-rate * time_to_expiry))
    return max(put, 0.0)


def heston_mc_price(
    params: HestonParams,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_expiry: float,
    option_type: str = "call",
    n_steps: int = 200,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price a European option under Heston via full-truncation Euler Monte Carlo.

    Discretizes the joint ``(S, v)`` SDE system with the "full
    truncation" Euler scheme of Lord, Koekkoek & Van Dijk (2010):
    correlated normals drive both processes, and the (possibly
    Euler-negative) variance is floored at 0 -- ``v+ = max(v, 0)`` --
    *inside* both the drift and diffusion coefficients at every step,
    which their comparative study found to have the best bias/
    efficiency trade-off among the common biased Euler schemes for
    Heston.

    Parameters
    ----------
    params : HestonParams
        The model parameters.
    spot, strike, rate, dividend_yield, time_to_expiry, option_type
        Same conventions as :func:`greeklab.black_scholes.bs_price`.
    n_steps : int, default 200
        Number of Euler time steps (discretization bias shrinks as
        this grows; 200 steps keeps bias well within typical Monte
        Carlo standard error at ``n_paths=50_000``).
    n_paths : int, default 50_000
        Number of simulated paths (antithetic variates on the driving
        normals are used to reduce variance, mirroring
        :func:`greeklab.monte_carlo.mc_european_price`).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        ``(price, std_error)``.

    Notes
    -----
    **Cross-validated** against :func:`heston_price_fourier` (the
    independent semi-analytical route) in ``tests/test_heston.py``:
    the two should agree within a handful of Monte Carlo standard
    errors, since they solve the identical model by unrelated
    numerical methods.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    rng = np.random.default_rng(seed)
    dt = time_to_expiry / n_steps
    sqrt_dt = np.sqrt(dt)
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    z1 = rng.standard_normal((n_paths, n_steps))
    z2_indep = rng.standard_normal((n_paths, n_steps))
    z1 = np.concatenate([z1, -z1], axis=0)
    z2_indep = np.concatenate([z2_indep, -z2_indep], axis=0)
    # Correlated Brownian increments: dW^v = z1, dW^S = rho*z1 + sqrt(1-rho^2)*z2.
    z_v = z1
    z_s = rho * z1 + np.sqrt(1.0 - rho * rho) * z2_indep

    n_sims = z1.shape[0]
    log_s = np.full(n_sims, np.log(spot))
    v = np.full(n_sims, v0)

    for step in range(n_steps):
        v_pos = np.clip(v, 0.0, None)
        sqrt_v_pos = np.sqrt(v_pos)

        # Full-truncation Euler (Lord, Koekkoek & Van Dijk 2010): use
        # v+ = max(v, 0) inside both the variance drift/diffusion and
        # the log-spot diffusion, but the *actual* v is what carries
        # forward (so it can dip negative between steps and be
        # re-truncated next step, exactly as their scheme specifies).
        log_s += (rate - dividend_yield - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z_s[:, step]
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * sqrt_dt * z_v[:, step]

    terminal_spot = np.exp(log_s)
    payoff = (
        np.clip(terminal_spot - strike, 0.0, None)
        if option_type == "call"
        else np.clip(strike - terminal_spot, 0.0, None)
    )
    discounted = np.exp(-rate * time_to_expiry) * payoff

    # Antithetic pairing: average each (z, -z) pair before computing
    # the standard error, matching greeklab.monte_carlo's convention.
    half = n_sims // 2
    pair_avg = 0.5 * (discounted[:half] + discounted[half:])
    price = float(np.mean(pair_avg))
    std_error = float(np.std(pair_avg, ddof=1) / np.sqrt(half))
    return price, std_error
