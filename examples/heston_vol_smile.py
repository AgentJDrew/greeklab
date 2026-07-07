"""Generate implied-vol smile plot data from the Heston (1993) model.

Prices a strike grid under Heston via Fourier inversion, converts each
price back to a Black-Scholes implied volatility, and prints the
resulting smile -- demonstrating how stochastic volatility alone (no
jumps needed) produces the skew/smile shape observed in real markets.

Usage:
    python examples/heston_vol_smile.py

To actually plot this (matplotlib not a dependency of this library,
install it separately), pipe the printed columns into your own
plotting script, or see app/dashboard.py for an interactive Plotly
version of this exact computation (3D across strike AND maturity).
"""

from __future__ import annotations

import numpy as np

from greeklab import HestonParams, heston_price_fourier, implied_vol


def main() -> None:
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    time_to_expiry = 0.5

    # A parameter set with meaningfully negative spot-vol correlation
    # (the equity "leverage effect"), producing a downward-sloping skew.
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)

    strikes = np.linspace(70.0, 130.0, 13)

    print(f"Heston smile: S={spot}, r={rate}, q={dividend_yield}, T={time_to_expiry}")
    print(f"kappa={params.kappa}, theta={params.theta}, xi={params.xi}, rho={params.rho}, v0={params.v0}")
    print(f"Feller condition satisfied: {params.feller_satisfied()}")
    print()
    print(f"{'Strike':>8} {'Price':>10} {'Implied Vol':>12}")
    print("-" * 34)

    for k in strikes:
        price = heston_price_fourier(params, spot, float(k), rate, dividend_yield, time_to_expiry, "call")
        result = implied_vol(price, spot, float(k), rate, dividend_yield, time_to_expiry, "call")
        iv_str = f"{result.sigma * 100:.3f}%" if result.converged else "n/a"
        print(f"{k:>8.1f} {price:>10.4f} {iv_str:>12}")


if __name__ == "__main__":
    main()
