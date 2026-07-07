"""Smoke test for app/dashboard.py using Streamlit's AppTest framework.

Runs the dashboard script headlessly (no real browser/server needed)
and asserts it produces no uncaught exceptions, in both its default
(Black-Scholes-Merton) mode and with the Heston model selected (which
additionally exercises the 3D vol-surface tab). This test requires the
optional ``app`` extra (``pip install -e ".[app]"``); it is collected
only if streamlit is importable, so a plain ``pip install -e ".[dev]"``
test run is unaffected.
"""

from __future__ import annotations

import pathlib

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

_DASHBOARD_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


def test_dashboard_runs_without_exception_bsm_default() -> None:
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"Dashboard raised: {[str(e) for e in at.exception]}"


def test_dashboard_runs_without_exception_heston_mode() -> None:
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Select the Heston radio option to additionally exercise the 3D
    # vol-surface tab (only built when Heston params exist).
    model_radio = at.sidebar.radio[1]  # index 0 = option_type, 1 = model
    model_radio.set_value("Heston (stochastic vol)")
    at.run()
    assert not at.exception, f"Dashboard raised in Heston mode: {[str(e) for e in at.exception]}"

    # Sanity: the metrics row and the smile/surface tabs were produced.
    # (AppTest has no dedicated accessor for st.plotly_chart elements in
    # this Streamlit version, so absence of exceptions plus presence of
    # the surrounding widgets is the available correctness signal.)
    assert len(at.metric) >= 9
    assert len(at.tabs) >= 1
