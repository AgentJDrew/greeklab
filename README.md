# greeklab

[![CI](https://github.com/AgentJDrew/greeklab/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentJDrew/greeklab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/AgentJDrew/greeklab)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-informational)](pyproject.toml)

A rigorous, from-scratch options and derivatives pricing library in pure NumPy/SciPy — from
Black-Scholes-Merton through binomial trees, Monte Carlo with variance reduction, path-dependent
exotics, and Heston (1993) stochastic volatility priced two independent ways.

**Why this exists:** most "options pricing" repos stop at `bs_price()`. This one goes where the
real engineering is — numerically stable edge cases, a seeded implied-vol solver with a real
fallback path, variance-reduced Monte Carlo with reported standard errors, Longstaff-Schwartz
American exercise, and a Heston Fourier pricer validated against a **published, independently
verified reference table to ~1e-9**, not just self-consistency. Every non-trivial numerical claim
in this README is backed by a test in `tests/`.

## Quickstart

```bash
pip install -e .
```

```python
from greeklab import bs_price, bs_greeks

price = bs_price(spot=100, strike=100, rate=0.05, dividend_yield=0.02,
                  sigma=0.20, time_to_expiry=1.0, option_type="call")
# 9.227005508154036

greeks = bs_greeks(spot=100, strike=100, rate=0.05, dividend_yield=0.02,
                    sigma=0.20, time_to_expiry=1.0, option_type="call")
print(greeks.delta, greeks.gamma, greeks.vega, greeks.theta)
# 0.586851146134764 0.018950578755008718 37.90115751001743 -5.089318913998334
```

Every pricer accepts NumPy arrays and broadcasts, so pricing a whole grid is one call:

```python
import numpy as np
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
prices = bs_price(spot=100, strike=strikes, rate=0.05, dividend_yield=0.02,
                   sigma=0.20, time_to_expiry=1.0, option_type="call")
```

## Worked example: price, Greeks, and implied vol round-trip

```python
from greeklab import bs_price, bs_greeks, implied_vol

# Price a 3-month 105-strike call
price = bs_price(spot=100, strike=105, rate=0.04, dividend_yield=0.0,
                  sigma=0.28, time_to_expiry=0.25, option_type="call")
# 3.930090...

# Full Greeks in one call
g = bs_greeks(spot=100, strike=105, rate=0.04, dividend_yield=0.0,
              sigma=0.28, time_to_expiry=0.25, option_type="call")
# g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga, g.charm

# Recover the volatility from the price (Newton-Raphson w/ Corrado-Miller seed)
result = implied_vol(price, spot=100, strike=105, rate=0.04, dividend_yield=0.0,
                      time_to_expiry=0.25, option_type="call")
# result.sigma == 0.28 to 1.1e-16 (machine precision), 3 Newton iterations
```

Run it yourself: `python examples/iv_from_market_price.py`

## Heston (1993) stochastic volatility — the headline feature

Constant-volatility Black-Scholes can't produce a volatility smile. Heston lets variance itself
follow a mean-reverting, correlated square-root diffusion, priced here via **Fourier inversion of
the characteristic function** (semi-analytical, essentially exact) and cross-checked against an
**independent Monte Carlo simulation**:

```python
from greeklab import HestonParams, heston_price_fourier, heston_mc_price

params = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)

price = heston_price_fourier(params, spot=100, strike=100, rate=0.0,
                              dividend_yield=0.0, time_to_expiry=1.0, option_type="call")
# 5.785155434319336

mc_price, mc_se = heston_mc_price(params, spot=100, strike=100, rate=0.0,
                                   dividend_yield=0.0, time_to_expiry=1.0,
                                   option_type="call", n_paths=100_000, seed=42)
# 5.783166592189066, SE=0.0151 -> Fourier and MC agree within 0.13 standard errors
```

A negative spot-vol correlation (`rho=-0.7`) alone — no jumps needed — reproduces the downward
skew seen in real equity markets (`python examples/heston_vol_smile.py`):

| Strike | 70 | 80 | 90 | 100 | 110 | 120 | 130 |
|---|---|---|---|---|---|---|---|
| Implied vol | 28.2% | 25.1% | 22.0% | 18.7% | 15.8% | 14.4% | 14.3% |

## Live demo (dashboard)

An optional Streamlit + Plotly dashboard (`app/dashboard.py`) puts all of the above — payoff
diagrams, Greeks-vs-spot curves, the IV smile, and a 3D Heston vol surface — behind sliders:

```bash
pip install -e ".[app]"
streamlit run app/dashboard.py   # run from the repo root
```

**Deploying to Streamlit Community Cloud:** fork this repo, go to
[share.streamlit.io](https://share.streamlit.io), point it at `app/dashboard.py`, and set
`app/requirements.txt` (or `pip install -e ".[app]"` via a build command) as the dependency source
— no other configuration needed. The dashboard imports `greeklab` as a regular dependency; it does
not modify the core library, and the core library has no Streamlit/Plotly dependency at all.

The dashboard is smoke-tested in CI via Streamlit's `AppTest` framework (`tests/test_dashboard_smoke.py`),
which runs the app headlessly in both its default (Black-Scholes) and Heston-selected states and
asserts no uncaught exception — including the code path that builds the 3D vol surface.

## The models

| Model | Function(s) | What it adds over BSM |
|---|---|---|
| Black-Scholes-Merton | `bs_price`, `bs_greeks` | Closed-form European price + analytical 1st/2nd-order Greeks |
| Implied volatility | `implied_vol` | Inverts price -> sigma (seeded Newton-Raphson + Brent fallback) |
| Binomial (CRR) | `crr_european_price`, `crr_american_price` | American early exercise via a discrete lattice |
| Monte Carlo (GBM) | `mc_european_price` | Simulation w/ antithetic + control-variate variance reduction |
| Exotics (MC) | `asian_arithmetic_mc`, `barrier_mc`, `american_lsm_mc` | Path-dependent payoffs; American via Longstaff-Schwartz |
| Heston (1993) | `heston_price_fourier`, `heston_mc_price` | Stochastic volatility, smile/skew, priced two independent ways |
| IV surface | `iv_surface`, `fit_svi_slice` | Whole-grid IV construction + an optional Gatheral SVI slice fit |

### Black-Scholes-Merton

$$C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad P = K e^{-rT}\Phi(-d_2) - S e^{-qT}\Phi(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Black & Scholes (1973); Merton (1973) continuous-dividend/carry extension.

### Greeks (analytical, 1st + 2nd order)

Delta, gamma, vega, theta, rho in closed form (Hull Ch. 19), plus the 2nd-order cross Greeks:

$$\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = -e^{-qT}\phi(d_1)\frac{d_2}{\sigma}, \qquad
\text{Volga} = \frac{\partial \text{Vega}}{\partial \sigma} = \text{Vega}\cdot\frac{d_1 d_2}{\sigma}, \qquad
\text{Charm} = \frac{\partial \Delta}{\partial t}$$

**Scaling convention** (see `greeklab/black_scholes.py` module docstring for the full table): vega
is per 1.00 (100 points) of vol, theta is per year, rho is per 1.00 (100 points) of rate.

### Implied volatility

Corrado & Miller (1996) closed-form seed (itself refining Brenner & Subrahmanyam 1988) feeds
Newton-Raphson; when vega is too small for Newton to be reliable (deep ITM/OTM), the solver falls
back to a bracketed Brent search. Round-trips a self-generated price back to its true sigma to
**~1e-8 to 1e-16** across a moneyness/expiry grid (see Validation below).

### Binomial (Cox-Ross-Rubinstein 1979)

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

The only method here with no closed-form European analogue that also prices **American** exercise
via backward induction: `max(continuation, intrinsic)` at every node.

### Monte Carlo (GBM, exact one-step simulation)

$$S_T = S_0 \exp\left((r - q - \sigma^2/2)T + \sigma\sqrt{T} Z\right), \quad Z \sim N(0,1)$$

No time-discretization bias (GBM's transition density is exact); variance reduction combines
**antithetic variates** (mirror every `Z` draw with `-Z`) and a **control variate** on $S_T$ itself
(Glasserman 2003, Ch. 4), cutting the standard error by roughly 2-3x versus plain simulation at the
same path count.

### Exotics (Monte Carlo)

- **Arithmetic Asian** — payoff on the average of `n_fixings` observed prices (no closed form
  exists for an arithmetic average of lognormals).
- **Barrier** (knock-in/knock-out, up/down) — discretely-monitored path crossing.
- **American (Longstaff-Schwartz 2001)** — least-squares regression of continuation value on a
  polynomial basis of in-the-money spot, at each exercise date, working backward from expiry.

### Heston (1993)

$$dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad dW^S dW^v = \rho\,dt$$

Priced via Heston's own two-probability formulation, $C = Se^{-qT}P_1 - Ke^{-rT}P_2$, with each
$P_j$ recovered by Gil-Pelaez inversion of the characteristic function — using the
**numerically stable branch** of the auxiliary root $d$ identified by Albrecher, Mayer, Schoutens &
Tistaert (2007), "The Little Heston Trap." The naive textbook branch was reproduced diverging by
~20 points at $T=5$ and outright `NaN` by $T=10$ during development before the stable branch was
adopted (see `greeklab/heston.py` docstrings and `tests/test_heston.py`). Cross-validated against
an independent full-truncation Euler Monte Carlo scheme (Lord, Koekkoek & Van Dijk 2010).

### Implied-vol surface + SVI

`iv_surface` re-uses the IV solver over a whole price grid; `fit_svi_slice` fits Gatheral's (2004)
raw SVI parameterization $w(k) = a + b(\rho(k-m) + \sqrt{(k-m)^2+\sigma^2})$ to one expiry's smile
via bounded nonlinear least squares — included as a validated stretch goal (see Validation below),
not a decorative afterthought.

## Validation

Every model is checked against an **independent** source of truth, not just against itself:

| # | What | Checked against | Result |
|---|---|---|---|
| 1 | BSM price | Hull (2021) Example 15.6 (S=42,K=40,r=10%,σ=20%,T=0.5) | Matches to 2 decimal places |
| 2 | Put-call parity | $C - P = Se^{-qT} - Ke^{-rT}$ | Residual < 1e-8 across a 5×3×3×3×3×4 grid |
| 3 | Every analytical Greek | Central finite differences | delta/gamma/vega/theta/rho/vanna/volga/charm all match, 8-way grid × 2 option types |
| 4 | Implied vol | Price -> IV -> price round-trip | Recovers original price to 1e-8; sigma to 1e-16 when vega is non-trivial |
| 5 | CRR binomial | Closed-form BSM as `n_steps -> ∞` | Converges within cents at 2000 steps; American ≥ European always; American call = European call exactly when q=0 |
| 6 | Monte Carlo (GBM) | Closed-form BSM | Within a few SE (seeded); variance reduction demonstrably shrinks SE (~2.5x) |
| 7 | Asian / Barrier / American-LSM | Vanilla BSM / CRR sanity relations | Asian < vanilla; barrier ≤ vanilla; in+out barrier parity; LSM American ≈ CRR American; LSM call w/o dividends ≈ European |
| 8 | **Heston Fourier** | **Alan Lewis's published, forum-verified-to-15-digit reference prices** ([source](https://financepress.com/2019/02/15/heston-model-reference-prices/)) | **Matches to ~6e-10 (standard panel) and ~1e-13 to 5e-8 (extreme short-maturity panel)** |
| 9 | Heston Fourier vs. Heston MC | Two independent numerical methods, 6 correlation values | Agree within 4 Monte Carlo standard errors every time |
| 10 | Heston -> BSM limit | `xi -> 0`, `v0 = theta = sigma²` | Matches BSM to ~1e-8 |
| 11 | IV surface | Recovers known input vols from a BSM-generated price grid | Exact recovery to 1e-6 |
| 12 | SVI slice fit | Recovers a synthetic SVI smile it was generated from | RMSE < 1e-8 |

**13,181 tests pass, 0 failures** (`pytest`), including 710 legitimate skips where the recovered-sigma
assertion correctly defers to the price-round-trip assertion at near-zero vega (the price check
itself never skips). Reproduce: `pytest -q`.

## Benchmark

Indicative single-quote pricing time (not a claim about production throughput — see Scope below):

| Method | Time |
|---|---|
| Black-Scholes closed-form | ~0.10 ms |
| CRR binomial, 500 steps | ~1.3 ms |
| Monte Carlo, 100k paths (w/ variance reduction) | ~5.8 ms |
| Heston, Fourier | ~3.1 ms |
| Heston, Monte Carlo (200 steps × 50k paths) | ~614 ms |

## Scope and limitations

- **No closed-form American Greeks.** `crr_american_price`/`american_lsm_mc` return price only;
  American Greeks would need bump-and-reprice or a PDE grid, out of scope here.
- **Heston has no closed-form Greeks in this library.** The dashboard's Greeks panel uses the BSM
  analytical Greeks; the Heston tabs show price/smile/surface only.
- **Discretely-monitored barriers**, not continuously-monitored (the standard practical
  approximation — see `greeklab/exotics.py` docstring for the Broadie-Glasserman-Kou 1997 caveat).
- **LSM American pricing is a lower-biased estimator** by construction (any suboptimal exercise
  policy under-values the option) — expect it to sit slightly below the true American price.
- **No jumps, no local volatility, no multi-factor stochastic vol.** Heston (1-factor) is the
  ceiling of this library's stochastic-volatility scope by design — depth over breadth.
- **The IV solver's search bracket is `sigma in [1e-6, 5.0]`.** Prices outside what any volatility
  in that range can produce (e.g. below intrinsic value) report `converged=False` rather than
  raising or returning a nonsense value — check `.converged` in batch/grid usage.
- **SVI fitting checks parameter bounds only**, not the full Gatheral-Jacquier (2014)
  butterfly-arbitrage-free conditions — treat it as a smoothing tool, not an arbitrage certifier.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
```

Optional dashboard: `pip install -e ".[app]"` then `streamlit run app/dashboard.py`.

## Project layout

```
greeklab/
├── src/greeklab/          Core library (numpy + scipy only)
│   ├── black_scholes.py       BSM price + analytical Greeks
│   ├── implied_vol.py         Seeded Newton-Raphson + Brent fallback
│   ├── binomial.py            CRR European + American
│   ├── monte_carlo.py         GBM MC w/ variance reduction
│   ├── exotics.py             Asian / Barrier / American-LSM (MC)
│   ├── heston.py              Heston Fourier + Heston MC
│   └── surface.py             IV surface + SVI slice fit
├── tests/                 13,000+ cases: reference values, parity, finite-diff Greeks, convergence
├── examples/              Runnable scripts (Greeks table, IV round-trip, Heston smile)
├── app/                   Optional Streamlit + Plotly dashboard (not a core dependency)
└── .github/workflows/     CI: pytest × 4 Python versions, ruff, mypy strict, dashboard smoke test
```

## References

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of
  Political Economy*, 81(3), 637-654.
- Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and
  Management Science*, 4(1), 141-183.
- Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach."
  *Journal of Financial Economics*, 7(3), 229-263.
- Brenner, M. and Subrahmanyam, M. G. (1988). "A Simple Formula to Compute the Implied Standard
  Deviation." *Financial Analysts Journal*, 44(5), 80-83.
- Corrado, C. J. and Miller, T. W. (1996). "A Note on a Simple, Accurate Formula to Compute
  Implied Standard Deviations." *Journal of Banking & Finance*, 20(3), 595-603.
- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with
  Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327-343.
- Albrecher, H., Mayer, P., Schoutens, W., and Tistaert, J. (2007). "The Little Heston Trap."
  *Wilmott Magazine*, January 2007, 83-92.
- Lord, R., Koekkoek, R., and Van Dijk, D. (2010). "A Comparison of Biased Simulation Schemes for
  Stochastic Volatility Models." *Quantitative Finance*, 10(2), 177-194.
- Longstaff, F. A. and Schwartz, E. S. (2001). "Valuing American Options by Simulation: A Simple
  Least-Squares Approach." *Review of Financial Studies*, 14(1), 113-147.
- Gatheral, J. (2004). "A Parsimonious Arbitrage-Free Implied Volatility Parameterization."
  Global Derivatives & Risk Management, Madrid.
- Gatheral, J. and Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces." *Quantitative
  Finance*, 14(1), 59-71.
- Lewis, A. High-precision Heston reference prices, archived at
  [financepress.com/2019/02/15/heston-model-reference-prices](https://financepress.com/2019/02/15/heston-model-reference-prices/).
- Hull, J. C. *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

## License

MIT — see [LICENSE](LICENSE).
