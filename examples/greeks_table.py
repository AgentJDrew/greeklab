"""Print a formatted table of price + all Greeks across a strike grid.

Usage:
    python examples/greeks_table.py
"""

from __future__ import annotations

import numpy as np

from greeklab import bs_greeks


def main() -> None:
    spot = 100.0
    rate = 0.05
    dividend_yield = 0.02
    sigma = 0.20
    time_to_expiry = 0.5
    strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])

    print(f"Black-Scholes-Merton call, S={spot}, r={rate}, q={dividend_yield}, sigma={sigma}, T={time_to_expiry}")
    print()
    header = f"{'Strike':>8} {'Price':>9} {'Delta':>8} {'Gamma':>8} {'Vega':>8} {'Theta':>9} {'Rho':>8} {'Vanna':>9} {'Volga':>8} {'Charm':>9}"
    print(header)
    print("-" * len(header))

    greeks = bs_greeks(spot, strikes, rate, dividend_yield, sigma, time_to_expiry, "call")
    for i, k in enumerate(strikes):
        print(
            f"{k:>8.1f} {greeks.price[i]:>9.4f} {greeks.delta[i]:>8.4f} {greeks.gamma[i]:>8.5f} "
            f"{greeks.vega[i]:>8.4f} {greeks.theta[i]:>9.4f} {greeks.rho[i]:>8.4f} "
            f"{greeks.vanna[i]:>9.5f} {greeks.volga[i]:>8.5f} {greeks.charm[i]:>9.5f}"
        )

    print()
    print("Units: vega per 1.00 vol (divide by 100 for 'per vol point'); theta per year")
    print("(divide by 365 for 'per calendar day'); rho per 1.00 rate (divide by 100 for 'per 1% move').")


if __name__ == "__main__":
    main()
