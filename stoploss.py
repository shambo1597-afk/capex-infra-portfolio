"""
Stop-Loss, Volatility and Historical Return Calculation Engine.

This module implements the portfolio's hybrid stop-loss methodology and the per-stock
risk statistics that feed the portfolio risk summary:
1. Daily returns from official NSE Bhavcopy closes, measured against the exchange's
   PREV_CLOSE so every return is a genuine one-session move.
2. Daily and annualized volatility (sample standard deviation of daily returns).
3. Historical expected return (simple average daily return, annualized).
4. Hybrid stop-loss: the tighter of a support-based level and a volatility-scaled cap.

Pure calculations only (no I/O); analysis.generate_portfolio_risk_summary() assembles
the per-stock table.
"""

import math
from typing import Optional, Tuple

import pandas as pd

from config import (
    RISK_LOOKBACK_TRADING_DAYS,
    STOP_LOSS_HOLDING_PERIOD_DAYS,
    STOP_LOSS_VOL_MULTIPLIER,
    TRADING_DAYS_PER_YEAR,
)

METHOD_SUPPORT = "support"
METHOD_VOLATILITY_CAP = "volatility_cap"
METHOD_UNAVAILABLE = "unavailable"


# -----------------------------------------------------------------------------
# RETURNS & VOLATILITY
# -----------------------------------------------------------------------------

def compute_daily_returns(stock_df: pd.DataFrame, lookback: int = RISK_LOOKBACK_TRADING_DAYS) -> pd.Series:
    """
    Compute simple daily returns over the trailing lookback window.

    Each return is CLOSE_PRICE / PREV_CLOSE - 1, where PREV_CLOSE is the exchange's close
    for the previous trading session. Unlike close-to-close differences between rows, this
    stays a true one-day return when sessions are missing from the local history (a gap
    would otherwise fold a multi-day move into a single "daily" return and inflate
    volatility), and PREV_CLOSE is adjusted by NSE for corporate actions such as splits.
    If PREV_CLOSE is unavailable, falls back to row-to-row close changes.

    Parameters:
        stock_df (pd.DataFrame): One stock's Bhavcopy history (DATE1, CLOSE_PRICE, PREV_CLOSE).
        lookback (int): Number of most recent daily returns to keep (default ~1 year).

    Returns:
        pd.Series: Daily returns as fractions, indexed by date, oldest first.
    """
    if stock_df is None or stock_df.empty:
        return pd.Series(dtype=float)

    df = stock_df.copy()
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed", errors="coerce")
    # Cached Bhavcopy history can repeat a session; keep one row per date
    df = df.dropna(subset=["DATE1"]).drop_duplicates(subset=["DATE1"], keep="last").sort_values("DATE1")

    close = pd.to_numeric(df["CLOSE_PRICE"], errors="coerce")
    if "PREV_CLOSE" in df.columns:
        prev_close = pd.to_numeric(df["PREV_CLOSE"], errors="coerce")
        returns = close / prev_close.where(prev_close > 0) - 1
    else:
        returns = close.pct_change()

    returns.index = df["DATE1"]
    return returns.dropna().tail(lookback)


def compute_daily_volatility(returns: pd.Series) -> Optional[float]:
    """Sample standard deviation (ddof=1) of daily returns, as a fraction; None if < 2 returns."""
    if returns is None or len(returns) < 2:
        return None
    return float(returns.std(ddof=1))


def compute_annualized_volatility(returns: pd.Series) -> Optional[float]:
    """Daily volatility scaled by sqrt(252), as a fraction; None if it cannot be computed."""
    daily_vol = compute_daily_volatility(returns)
    return None if daily_vol is None else daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_historical_expected_return(returns: pd.Series) -> Optional[float]:
    """
    Annualized simple historical average return: mean daily return x 252, as a fraction.

    PLACEHOLDER METHODOLOGY: a trailing average of realized returns is a weak forecast of
    future returns (it mostly reflects the past year's trend). A future iteration may
    replace it with a CAPM-implied expected return, rf + beta x (E[Rm] - rf), once
    portfolio beta against the Nifty 500 is computed.
    """
    if returns is None or len(returns) == 0:
        return None
    return float(returns.mean()) * TRADING_DAYS_PER_YEAR


# -----------------------------------------------------------------------------
# HYBRID STOP-LOSS
# -----------------------------------------------------------------------------

def compute_volatility_cap_stop(
    current_price: float,
    daily_volatility: Optional[float],
    k: float = STOP_LOSS_VOL_MULTIPLIER,
    holding_period_days: int = STOP_LOSS_HOLDING_PERIOD_DAYS,
) -> Optional[float]:
    """
    Volatility-scaled stop: current_price x (1 - k x daily_volatility x sqrt(N)).

    k (default 1.75) and N (default 21 trading days, about one month for a 3-month mandate
    reviewed monthly) are stated, adjustable assumptions; see config.py. The stop sits
    k standard deviations of an N-day move below the current price.

    Returns:
        float, or None when volatility is unknown or the cushion would reach 100%.
    """
    if daily_volatility is None or current_price is None or current_price <= 0:
        return None
    cushion = k * daily_volatility * math.sqrt(holding_period_days)
    if cushion >= 1:
        return None
    return current_price * (1 - cushion)


def _is_valid_stop(candidate: Optional[float], current_price: float) -> bool:
    """A usable stop is a real price strictly between zero and the current price."""
    return candidate is not None and not pd.isna(candidate) and 0 < candidate < current_price


def select_stop_loss(
    current_price: float,
    support_price: Optional[float],
    volatility_cap_price: Optional[float],
) -> Tuple[Optional[float], str]:
    """
    Choose the final stop-loss: whichever candidate is CLOSER to the current price.

    Both candidates sit below the current price, so the closer one is the higher one,
    i.e. the tighter, more conservative stop. A candidate that is missing or not strictly
    below the current price (a stop at or above the price would trigger immediately) is
    ignored. On an exact tie the support level wins, since it is an observed price level.

    Returns:
        (stop_loss_price, method): method is "support" or "volatility_cap", or
        (None, "unavailable") when neither candidate is usable.
    """
    support_ok = _is_valid_stop(support_price, current_price)
    vol_cap_ok = _is_valid_stop(volatility_cap_price, current_price)

    if support_ok and (not vol_cap_ok or support_price >= volatility_cap_price):
        return float(support_price), METHOD_SUPPORT
    if vol_cap_ok:
        return float(volatility_cap_price), METHOD_VOLATILITY_CAP
    return None, METHOD_UNAVAILABLE
