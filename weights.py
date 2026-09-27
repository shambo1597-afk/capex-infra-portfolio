"""
Portfolio weights: equal risk contribution (ERC) within the brief's weight bounds.

Each stock's risk contribution is RC_i = w_i * (C w)_i / (w' C w), its share of portfolio
variance w' C w (C = annualised covariance of daily returns). ERC chooses the weights that make
every RC_i as equal as the bounds allow, so no stock dominates portfolio variance:
  - no return forecast is needed (research/momentum_study.py found stock-level forecasts weak),
  - volatile names get less capital, calm names more,
  - bounds WEIGHT_MIN_PCT..WEIGHT_MAX_PCT keep every stock a real position and cap concentration.
Solved in numpy by an active-set method: the free weights are updated multiplicatively,
w_i <- w_i * sqrt(mean RC / RC_i), and rescaled to their budget; a weight that breaches a bound is held
at it and the rest re-solved, so the free stocks end with exactly equal risk contributions.
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


def _erc_free(cov: np.ndarray, w: np.ndarray, free: np.ndarray, budget: float, iterations: int) -> np.ndarray:
    """Equalise w_i * (C w)_i over the free weights (the others held), free weights summing to budget."""
    w = w.copy()
    w[free] = budget * w[free] / w[free].sum()
    for _ in range(iterations):
        rc = w * (cov @ w)
        target = rc[free].mean()
        w_next = w.copy()
        w_next[free] = w[free] * np.sqrt(target / rc[free])
        w_next[free] *= budget / w_next[free].sum()
        if np.abs(w_next - w).max() < 1e-13:
            return w_next
        w = w_next
    return w


def equal_risk_contribution(cov: np.ndarray, lo: float, hi: float, iterations: int = 20000) -> np.ndarray:
    """
    ERC weights under the bounds (see the module docstring). Active set: stocks whose ERC weight would
    breach a bound are held at it, and the rest are solved to exactly equal risk contributions with the
    remaining budget; repeated until no free weight breaches a bound (a stock held at the cap then carries
    less risk than the others, one held at the floor more).
    """
    n = cov.shape[0]
    if not (n * lo <= 1 + 1e-12 and n * hi >= 1 - 1e-12):
        raise ValueError(f"Bounds [{lo}, {hi}] are infeasible for {n} weights summing to 1.")
    w = np.full(n, 1.0 / n)
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        free = ~fixed
        budget = 1.0 - w[fixed].sum()
        w = _erc_free(cov, w, free, budget, iterations)
        over, under = free & (w > hi + 1e-12), free & (w < lo - 1e-12)
        if not over.any() and not under.any():
            break
        # Fix the worst breach first (fixing one changes the others' ERC weights)
        excess = np.where(over, w - hi, 0.0) + np.where(under, lo - w, 0.0)
        i = int(np.argmax(excess))
        w[i] = hi if over[i] else lo
        fixed[i] = True
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
