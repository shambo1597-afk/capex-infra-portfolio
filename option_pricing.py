"""
Black-Scholes prices, deltas and implied volatilities for Nifty options (European, no dividend yield).

Used to value the puts held at a future date (the horizon), to give them a delta in the VaR, and to read
the market's implied volatility, which turns option prices into probabilities for the hedging decision.
"""

from math import erf, exp, log, sqrt

import numpy as np


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_price(spot, strike: float, t: float, r: float, vol: float, kind: str = "PE"):
    """European put ('PE') or call ('CE') value; at t <= 0 the intrinsic value. `spot` may be an array."""
    s = np.asarray(spot, dtype=float)
    if t <= 0 or vol <= 0:
        out = np.maximum(strike - s, 0.0) if kind == "PE" else np.maximum(s - strike, 0.0)
        return out if out.ndim else float(out)
    d1 = (np.log(s / strike) + (r + vol * vol / 2) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    n = lambda x: 0.5 * (1 + _erf_vec(x / sqrt(2)))  # noqa: E731
    if kind == "PE":
        out = strike * exp(-r * t) * n(-d2) - s * n(-d1)
    else:
        out = s * n(d1) - strike * exp(-r * t) * n(d2)
    return out if np.ndim(out) else float(out)


def _erf_vec(x):
    x = np.asarray(x, dtype=float)
    return np.vectorize(erf)(x) if x.ndim else erf(float(x))


def bs_delta(spot: float, strike: float, t: float, r: float, vol: float, kind: str = "PE") -> float:
    if t <= 0 or vol <= 0:
        itm = spot < strike if kind == "PE" else spot > strike
        return (-1.0 if kind == "PE" else 1.0) if itm else 0.0
    d1 = (log(spot / strike) + (r + vol * vol / 2) * t) / (vol * sqrt(t))
    return norm_cdf(d1) - 1 if kind == "PE" else norm_cdf(d1)


def implied_vol(price: float, spot: float, strike: float, t: float, r: float, kind: str = "PE") -> float:
    """Volatility that reproduces `price` (bisection); NaN when the price is below intrinsic or not positive."""
    intrinsic = max(strike * exp(-r * t) - spot, 0.0) if kind == "PE" else max(spot - strike * exp(-r * t), 0.0)
    if not price > intrinsic + 1e-6 or t <= 0:
        return float("nan")
    lo, hi = 1e-4, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_price(spot, strike, t, r, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
