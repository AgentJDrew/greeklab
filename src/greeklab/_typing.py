"""Shared type aliases used across the ``greeklab`` public API.

All pricing and Greeks functions accept either Python scalars (``float``)
or NumPy arrays for the state variables (spot, strike, volatility, ...),
and broadcast following standard NumPy rules. ``ArrayLike`` documents that
contract in one place instead of repeating a long union on every signature.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Accepts a Python float/int or a NumPy array of any of those. Every
#: numerical entry point in ``greeklab`` (spot, strike, rate, dividend
#: yield, volatility, time-to-expiry, ...) uses this type and is
#: broadcast-vectorized: pass all scalars for a single quote, or arrays
#: for a whole grid computed in one vectorized call.
ArrayLike = float | npt.NDArray[np.float64]

#: The concrete return type of every pricing/Greeks function: always a
#: NumPy float64 array (0-d for scalar inputs, broadcast shape otherwise).
FloatArray = npt.NDArray[np.float64]

__all__ = ["ArrayLike", "FloatArray"]
