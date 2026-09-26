"""
Portfolio weights: equal risk contribution (ERC) within the brief's weight bounds.

Each stock's risk contribution is RC_i = w_i * (C w)_i / (w' C w), its share of portfolio
variance w' C w (C = annualised covariance of daily returns). ERC chooses the weights that make
every RC_i as equal as the bounds allow, so no stock dominates portfolio variance:
  - no return forecast is needed (research/momentum_study.py found stock-level forecasts weak),
  - volatile names get less capital, calm names more,
  - bounds WEIGHT_MIN_PCT..WEIGHT_MAX_PCT keep every stock a real position and cap concentration.
Solved in numpy: fixed-point iteration w <- 1 / (C w) (the ERC condition), each step projected
onto {sum w = 1, lo <= w_i <= hi}.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, WEIGHT_COV_LOOKBACK_DAYS, WEIGHT_MAX_PCT, WEIGHT_MIN_PCT

logger = logging.getLogger("weights")


def project_to_bounded_simplex(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Closest point to v with weights summing to 1 and each in [lo, hi] (bisection on the shift)."""
    n = len(v)
    if not (n * lo <= 1 + 1e-12 and n * hi >= 1 - 1e-12):
        raise ValueError(f"Bounds [{lo}, {hi}] are infeasible for {n} weights summing to 1.")
    a, b = v.min() - hi - 1.0, v.max() - lo + 1.0
    for _ in range(200):
        t = (a + b) / 2
        if np.clip(v - t, lo, hi).sum() > 1:
            a = t
        else:
            b = t
    return np.clip(v - (a + b) / 2, lo, hi)


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each weight's share of portfolio variance (sums to 1)."""
    marginal = cov @ weights
    return weights * marginal / (weights @ marginal)


def equal_risk_contribution(cov: np.ndarray, lo: float, hi: float, iterations: int = 5000) -> np.ndarray:
    """ERC weights under the bounds (see the module docstring)."""
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iterations):
        inv = 1.0 / (cov @ w)
        w_next = project_to_bounded_simplex(0.5 * w + 0.5 * inv / inv.sum(), lo, hi)
        if np.abs(w_next - w).max() < 1e-12:
            break
        w = w_next
    return w


def daily_return_matrix(stock_data: pd.DataFrame, symbols: List[str],
                        lookback: int = WEIGHT_COV_LOOKBACK_DAYS) -> pd.DataFrame:
    """Daily close-to-close returns (sessions x symbols) over the trailing `lookback` sessions."""
    df = stock_data[stock_data["SYMBOL"].isin(symbols)].copy()
    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE"])]
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed")
    closes = df.pivot_table(index="DATE1", columns="SYMBOL", values="CLOSE_PRICE").sort_index()
    return closes.reindex(columns=symbols).pct_change(fill_method=None).iloc[1:].tail(lookback)


def compute_portfolio_weights(stock_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    One row per symbol: weight_pct (ERC within the bounds) and risk_contribution_pct.
    Covariance is pairwise over the trailing window, so a recent listing (QPOWER) uses the sessions it has.
    Falls back to equal weight when any symbol lacks the history for a covariance estimate.
    """
    lo, hi = WEIGHT_MIN_PCT / 100, WEIGHT_MAX_PCT / 100
    n = len(symbols)
    returns = daily_return_matrix(stock_data, symbols) if stock_data is not None and not stock_data.empty else None
    cov: Optional[np.ndarray] = None
    bounds_feasible = n * lo <= 1 <= n * hi
    if bounds_feasible and returns is not None and returns.count().min() >= 60:
        cov = returns.cov(min_periods=60).values * TRADING_DAYS_PER_YEAR
        if np.isnan(cov).any():
            cov = None
    if cov is None:
        logger.warning("No covariance estimate (too little history) or bounds infeasible for %d stocks; "
                       "using equal weights.", n)
        w = np.full(n, 1.0 / n)
        rc = np.full(n, np.nan)
    else:
        w = equal_risk_contribution(cov, lo, hi)
        rc = risk_contributions(w, cov)
    return pd.DataFrame({"symbol": symbols, "weight_pct": np.round(w * 100, 2),
                         "risk_contribution_pct": np.round(rc * 100, 2)})
