"""Back out implied volatility from an observed market option price.

Usage:
    python examples/iv_from_market_price.py
"""

from __future__ import annotations

from greeklab import bs_price, implied_vol


def main() -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.04
    dividend_yield = 0.0
    time_to_expiry = 0.25
    option_type = "call"

    # Simulate an "observed market price" by pricing at a known true vol
    # (in a real workflow this would come from a market data feed).
    true_sigma = 0.28
    observed_price = float(bs_price(spot, strike, rate, dividend_yield, true_sigma, time_to_expiry, option_type))

    print(f"Observed {option_type} price: {observed_price:.4f}  (generated at true sigma={true_sigma:.4f})")

    result = implied_vol(observed_price, spot, strike, rate, dividend_yield, time_to_expiry, option_type)
    print(f"Solved implied vol:          {result.sigma:.6f}")
    print(f"Converged:                   {result.converged}  (method={result.method}, iterations={result.iterations})")
    print(f"Recovery error vs true sigma: {abs(result.sigma - true_sigma):.2e}")

    # Round-trip check: re-price at the solved vol and compare to the
    # original observed price.
    round_trip_price = float(bs_price(spot, strike, rate, dividend_yield, result.sigma, time_to_expiry, option_type))
    print(f"Round-trip price error:      {abs(round_trip_price - observed_price):.2e}")


if __name__ == "__main__":
    main()
