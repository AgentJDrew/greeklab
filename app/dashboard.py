"""Interactive dashboard for greeklab: pricing, Greeks, IV smile, Heston surface.

Run from the repo root (so the theme in .streamlit/config.toml is
picked up):

    streamlit run app/dashboard.py

This module is intentionally the *only* place in the repo that imports
streamlit/plotly -- the core ``greeklab`` package stays numpy+scipy
only. Install with the optional ``app`` extra:

    pip install -e ".[app]"
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from greeklab import HestonParams, bs_greeks, bs_price, heston_price_fourier, implied_vol

# --- Palette (mirrors .streamlit/config.toml) ---------------------------
ACCENT = "#2dd4bf"  # teal
PROFIT = "#3fb950"  # green
LOSS = "#f85149"  # red
GRID = "rgba(230, 237, 243, 0.08)"
AXIS_TEXT = "rgba(230, 237, 243, 0.55)"
PLOT_FONT = dict(family="Inter, -apple-system, sans-serif", color="#e6edf3", size=13)

st.set_page_config(page_title="greeklab", page_icon=":material/monitoring:", layout="wide")


def _base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Apply the shared transparent/dark styling to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e6edf3"), x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=AXIS_TEXT, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=AXIS_TEXT, showline=False)
    return fig


# --- Sidebar inputs -------------------------------------------------------
with st.sidebar:
    st.markdown("### Contract")
    option_type = st.radio("Type", ["call", "put"], horizontal=True)
    model = st.radio("Model", ["Black-Scholes-Merton", "Heston (stochastic vol)"], horizontal=False)

    st.markdown("### Market inputs")
    spot = st.number_input("Spot (S)", min_value=0.01, value=100.0, step=1.0)
    strike = st.number_input("Strike (K)", min_value=0.01, value=100.0, step=1.0)
    rate = st.slider("Risk-free rate (r)", 0.0, 0.15, 0.05, 0.005, format="%.3f")
    dividend_yield = st.slider("Dividend yield (q)", 0.0, 0.10, 0.0, 0.005, format="%.3f")
    time_to_expiry = st.slider("Time to expiry, years (T)", 0.01, 3.0, 1.0, 0.01)

    st.markdown("### Volatility")
    sigma = st.slider("Volatility, BSM (sigma)", 0.01, 1.5, 0.20, 0.01)

    heston_params: HestonParams | None = None
    if model.startswith("Heston"):
        st.caption("Heston (1993) stochastic-volatility parameters")
        v0 = st.slider("Initial variance (v0)", 0.001, 0.5, sigma**2, 0.001, format="%.3f")
        kappa = st.slider("Mean-reversion speed (kappa)", 0.1, 8.0, 2.0, 0.1)
        theta = st.slider("Long-run variance (theta)", 0.001, 0.5, sigma**2, 0.001, format="%.3f")
        xi = st.slider("Vol-of-vol (xi)", 0.01, 2.0, 0.4, 0.01)
        rho = st.slider("Spot-vol correlation (rho)", -0.99, 0.99, -0.5, 0.01)
        heston_params = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

    st.markdown("### Market price (optional)")
    st.caption("Enter an observed price to back out its Black-Scholes implied volatility.")
    market_price = st.number_input("Observed option price", min_value=0.0, value=0.0, step=0.1)


# --- Core pricing ----------------------------------------------------------
greeks = bs_greeks(spot, strike, rate, dividend_yield, sigma, time_to_expiry, option_type)
bsm_price = float(greeks.price)

heston_headline_price: float | None = None
if heston_params is not None:
    heston_headline_price = heston_price_fourier(
        heston_params, spot, strike, rate, dividend_yield, time_to_expiry, option_type
    )

headline_price = heston_headline_price if heston_headline_price is not None else bsm_price

st.title("greeklab")
st.caption("Options pricing, Greeks, implied volatility, and Heston stochastic volatility -- interactive explorer.")

# --- Metrics row -----------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"{headline_price:,.4f}")
m2.metric("Delta", f"{float(greeks.delta):,.4f}")
m3.metric("Gamma", f"{float(greeks.gamma):,.5f}")
m4.metric("Vega (per 1.00 vol)", f"{float(greeks.vega):,.4f}")
m5.metric("Theta (per year)", f"{float(greeks.theta):,.4f}")
m6.metric("Rho (per 1.00 rate)", f"{float(greeks.rho):,.4f}")

m7, m8, m9, _ = st.columns(4)
m7.metric("Vanna", f"{float(greeks.vanna):,.5f}")
m8.metric("Volga", f"{float(greeks.volga):,.5f}")
m9.metric("Charm (per year)", f"{float(greeks.charm):,.5f}")

if heston_headline_price is not None:
    diff = heston_headline_price - bsm_price
    st.caption(
        f"Heston (Fourier) price: **{heston_headline_price:,.4f}** vs. Black-Scholes-Merton: "
        f"**{bsm_price:,.4f}** (difference: {diff:+.4f}). The metrics above are the BSM analytical Greeks; "
        "Heston has no closed-form Greeks in this library, so the smile/surface tabs below carry the stochastic-vol view."
    )

if market_price > 0.0:
    iv_result = implied_vol(market_price, spot, strike, rate, dividend_yield, time_to_expiry, option_type)
    if iv_result.converged:
        st.success(
            f"Implied volatility for observed price {market_price:.4f}: "
            f"**{iv_result.sigma * 100:.2f}%** ({iv_result.iterations} Newton iterations, method={iv_result.method})"
        )
    else:
        st.warning("Could not solve for implied volatility -- the observed price is not attainable by any sigma in [1e-6, 5.0] (e.g. below intrinsic value).")

st.divider()

# --- Payoff diagram + Greeks curves ----------------------------------------
col_left, col_right = st.columns(2)

spot_range = np.linspace(max(spot * 0.4, 0.5), spot * 1.6, 200)

with col_left:
    payoff_at_expiry = np.clip(spot_range - strike, 0.0, None) if option_type == "call" else np.clip(strike - spot_range, 0.0, None)
    premium = headline_price
    pnl_at_expiry = payoff_at_expiry - premium

    fig_payoff = go.Figure()
    fig_payoff.add_trace(
        go.Scatter(
            x=spot_range,
            y=payoff_at_expiry,
            name="Payoff at expiry",
            line=dict(color=AXIS_TEXT, width=1.5, dash="dot"),
        )
    )
    fig_payoff.add_trace(
        go.Scatter(
            x=spot_range,
            y=pnl_at_expiry,
            name="P&L (net of premium)",
            line=dict(color=ACCENT, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.12)",
        )
    )
    fig_payoff.add_hline(y=0, line_color=AXIS_TEXT, line_width=1)
    fig_payoff.add_vline(x=strike, line_color=AXIS_TEXT, line_width=1, line_dash="dash", annotation_text="K")
    fig_payoff.add_vline(x=spot, line_color=ACCENT, line_width=1, line_dash="dash", annotation_text="S")
    _base_layout(fig_payoff, f"Payoff diagram ({option_type})")
    fig_payoff.update_xaxes(title_text="Spot at expiry")
    fig_payoff.update_yaxes(title_text="Value")
    st.plotly_chart(fig_payoff, width='stretch')

with col_right:
    greeks_curve = bs_greeks(spot_range, strike, rate, dividend_yield, sigma, time_to_expiry, option_type)
    fig_greeks = go.Figure()
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=greeks_curve.delta, name="Delta", line=dict(color=ACCENT, width=2)))
    fig_greeks.add_trace(
        go.Scatter(x=spot_range, y=greeks_curve.gamma, name="Gamma", yaxis="y2", line=dict(color=PROFIT, width=2))
    )
    fig_greeks.add_trace(
        go.Scatter(x=spot_range, y=greeks_curve.vega, name="Vega", yaxis="y3", line=dict(color="#e3b341", width=2))
    )
    fig_greeks.add_trace(
        go.Scatter(x=spot_range, y=greeks_curve.theta, name="Theta", yaxis="y4", line=dict(color=LOSS, width=2))
    )
    fig_greeks.add_vline(x=spot, line_color=AXIS_TEXT, line_width=1, line_dash="dash")
    _base_layout(fig_greeks, "Greeks vs. spot")
    fig_greeks.update_layout(
        yaxis=dict(title="Delta", showgrid=True, gridcolor=GRID, color=AXIS_TEXT),
        yaxis2=dict(title="Gamma", overlaying="y", side="right", showgrid=False, color=AXIS_TEXT),
        yaxis3=dict(title="Vega", overlaying="y", side="left", position=0.001, showgrid=False, visible=False),
        yaxis4=dict(title="Theta", overlaying="y", side="right", position=0.999, showgrid=False, visible=False),
        xaxis=dict(title="Spot"),
    )
    st.plotly_chart(fig_greeks, width='stretch')

st.divider()

# --- IV smile + Heston vol surface -----------------------------------------
tab_smile, tab_surface = st.tabs(["Implied-vol smile", "Heston 3D vol surface"])

with tab_smile:
    st.caption(
        "A synthetic smile: BSM prices generated at a range of strikes using a simple skew "
        "(higher vol away from the money), then re-solved for implied vol with greeklab's own solver -- "
        "demonstrating the price -> IV round trip."
    )
    strikes_smile = np.linspace(strike * 0.6, strike * 1.4, 41)
    moneyness = np.log(strikes_smile / spot)
    skewed_vols = sigma + 0.15 * sigma * moneyness**2 - 0.05 * sigma * moneyness  # simple smile+skew shape
    skewed_vols = np.clip(skewed_vols, 0.02, None)
    smile_prices = [
        float(bs_price(spot, k, rate, dividend_yield, v, time_to_expiry, option_type))
        for k, v in zip(strikes_smile, skewed_vols, strict=True)
    ]
    recovered = [
        implied_vol(p, spot, k, rate, dividend_yield, time_to_expiry, option_type)
        for p, k in zip(smile_prices, strikes_smile, strict=True)
    ]
    recovered_vols = [r.sigma if r.converged else np.nan for r in recovered]

    fig_smile = go.Figure()
    fig_smile.add_trace(
        go.Scatter(x=strikes_smile, y=np.asarray(skewed_vols) * 100, name="Input vol", line=dict(color=AXIS_TEXT, width=3))
    )
    fig_smile.add_trace(
        go.Scatter(
            x=strikes_smile,
            y=np.asarray(recovered_vols) * 100,
            name="Recovered (solver)",
            mode="markers",
            marker=dict(color=ACCENT, size=6),
        )
    )
    fig_smile.add_vline(x=spot, line_color=ACCENT, line_width=1, line_dash="dash", annotation_text="ATM")
    _base_layout(fig_smile, "Implied volatility smile", height=420)
    fig_smile.update_xaxes(title_text="Strike")
    fig_smile.update_yaxes(title_text="Implied vol (%)")
    st.plotly_chart(fig_smile, width='stretch')

with tab_surface:
    st.caption("The Heston (1993) Fourier-priced European call, converted to Black-Scholes implied vol, across strike and maturity.")
    if heston_params is None:
        st.info("Select **Heston (stochastic vol)** in the sidebar to enable the 3D surface.")
    else:
        strikes_surface = np.linspace(strike * 0.6, strike * 1.4, 25)
        expiries_surface = np.linspace(0.05, max(time_to_expiry, 0.5) * 1.5, 20)
        iv_grid = np.full((len(expiries_surface), len(strikes_surface)), np.nan)

        with st.spinner("Pricing the Heston surface..."):
            for i, t_i in enumerate(expiries_surface):
                for j, k_j in enumerate(strikes_surface):
                    price_ij = heston_price_fourier(heston_params, spot, float(k_j), rate, dividend_yield, float(t_i), option_type)
                    iv_res = implied_vol(price_ij, spot, float(k_j), rate, dividend_yield, float(t_i), option_type)
                    if iv_res.converged:
                        iv_grid[i, j] = iv_res.sigma * 100

        fig_surface = go.Figure(
            data=[
                go.Surface(
                    x=strikes_surface,
                    y=expiries_surface,
                    z=iv_grid,
                    colorscale=[[0, "#161b22"], [0.5, ACCENT], [1, "#e3b341"]],
                    showscale=True,
                    colorbar=dict(title="IV %", tickfont=dict(color=AXIS_TEXT), title_font=dict(color=AXIS_TEXT)),
                )
            ]
        )
        fig_surface.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=PLOT_FONT,
            height=560,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(text="Heston implied-vol surface", font=dict(size=15, color="#e6edf3"), x=0.02),
            scene=dict(
                xaxis=dict(title="Strike", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=AXIS_TEXT),
                yaxis=dict(title="Time to expiry (yrs)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=AXIS_TEXT),
                zaxis=dict(title="Implied vol (%)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=AXIS_TEXT),
            ),
        )
        st.plotly_chart(fig_surface, width='stretch')

st.divider()
st.caption(
    "greeklab -- Black-Scholes-Merton, binomial trees, Monte Carlo, exotics, and Heston stochastic volatility, "
    "validated against published reference values and finite-difference Greeks. "
    "[GitHub](https://github.com/AgentJDrew/greeklab)"
)
