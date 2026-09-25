"""
Stop-Loss, Volatility and Historical Return Calculation Engine.

This module implements the portfolio's ATR stop-loss methodology and the per-stock
risk statistics that feed the portfolio risk summary:
1. Daily returns from official NSE Bhavcopy closes, measured against the exchange's
   PREV_CLOSE so every return is a genuine one-session move.
2. Daily and annualized volatility (sample standard deviation of daily returns).
3. Historical expected return (simple average daily return, annualized).
4. ATR stop-loss for the 3-month mandate: 3 x ATR(14) below price, moved to just below a
   support level that lies slightly beyond it, and trailed up (never down) at each review.

Pure calculations only (no I/O); analysis.generate_portfolio_risk_summary() assembles
the per-stock table.
"""

import math
from typing import Optional, Tuple

import pandas as pd

from config import (
    RISK_LOOKBACK_TRADING_DAYS,
    STOP_LOSS_ATR_MULTIPLE,
    STOP_LOSS_ATR_PERIOD,
    STOP_LOSS_SUPPORT_BAND_ATR,
    STOP_LOSS_SUPPORT_BUFFER_ATR,
    TRADING_DAYS_PER_YEAR,
)
from indicators import wilder_smoothing

METHOD_ATR = "atr"
METHOD_SUPPORT = "support"
METHOD_TRAILED = "trailed"
METHOD_BREACHED = "breached"
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
# ATR STOP-LOSS
# -----------------------------------------------------------------------------

def _one_row_per_session(stock_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.copy()
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed", errors="coerce")
    return df.dropna(subset=["DATE1"]).drop_duplicates(subset=["DATE1"], keep="last").sort_values("DATE1")


def compute_atr(stock_df: pd.DataFrame, period: int = STOP_LOSS_ATR_PERIOD) -> Optional[float]:
    """
    Latest Average True Range: Wilder smoothing of max(High - Low, |High - PrevClose|,
    |Low - PrevClose|), in price units.

    PrevClose is the exchange's PREV_CLOSE (a true one-session reference even when sessions
    are missing locally, and adjusted for corporate actions); falls back to the prior row's
    close when the column is absent.

    Returns:
        float, or None when there are fewer than `period` sessions.
    """
    if stock_df is None or stock_df.empty:
        return None
    df = _one_row_per_session(stock_df)
    high = pd.to_numeric(df["HIGH_PRICE"], errors="coerce")
    low = pd.to_numeric(df["LOW_PRICE"], errors="coerce")
    close = pd.to_numeric(df["CLOSE_PRICE"], errors="coerce")
    prev_close = pd.to_numeric(df["PREV_CLOSE"], errors="coerce") if "PREV_CLOSE" in df.columns else close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    true_range = true_range.dropna().reset_index(drop=True)
    if len(true_range) < period:
        return None
    atr = wilder_smoothing(true_range, period=period).iloc[-1]
    return None if pd.isna(atr) else float(atr)


def compute_atr_stop(
    current_price: float,
    atr: Optional[float],
    support_price: Optional[float] = None,
    atr_multiple: float = STOP_LOSS_ATR_MULTIPLE,
    support_band_atr: float = STOP_LOSS_SUPPORT_BAND_ATR,
    support_buffer_atr: float = STOP_LOSS_SUPPORT_BUFFER_ATR,
) -> Tuple[Optional[float], str]:
    """
    Stop-loss for the 3-month mandate, sized for about one month and trailed at reviews.

    1. Base stop = current_price - atr_multiple x ATR (3 ATR ~ a one-month, one-sigma move).
    2. If a support level lies below the base stop but within support_band_atr ATRs of it,
       the stop moves to support - support_buffer_atr x ATR, just under that level, so an
       ordinary retest of support does not trigger it.
    A support level above the base stop is ignored: it never makes the stop tighter than
    3 ATR (the previous "tighter of support vs cap" rule produced stops 0.1-0.7% below price).

    Returns:
        (stop_price, method): method is "atr" or "support", or (None, "unavailable") when
        the price or ATR is unknown or the stop would not be a positive price.
    """
    if current_price is None or current_price <= 0 or atr is None or atr <= 0:
        return None, METHOD_UNAVAILABLE
    base_stop = current_price - atr_multiple * atr
    stop, method = base_stop, METHOD_ATR
    if (support_price is not None and not pd.isna(support_price)
            and base_stop - support_band_atr * atr <= support_price < base_stop):
        stop, method = support_price - support_buffer_atr * atr, METHOD_SUPPORT
    if stop <= 0:
        return None, METHOD_UNAVAILABLE
    return float(stop), method


def apply_trailing_stop(
    new_stop: Optional[float],
    new_method: str,
    previous_stop: Optional[float],
    current_price: Optional[float],
) -> Tuple[Optional[float], str]:
    """
    Trail the stop: keep the previous review's stop when it is higher than the newly computed
    one (a stop is only ever raised), provided it is still below the current price.

    A previous stop at or above the current price means the stop has been hit: it is returned
    with method "breached" (a negative distance below price) so the exit is not silently lost.
    """
    if previous_stop is None or pd.isna(previous_stop) or current_price is None:
        return new_stop, new_method
    if previous_stop >= current_price:
        return float(previous_stop), METHOD_BREACHED
    # Compare at the 2-decimal precision stops are saved with, so re-reading an unchanged stop
    # from the previous CSV is not mistaken for a trailed one
    if new_stop is None or round(float(previous_stop), 2) > round(new_stop, 2):
        return float(previous_stop), METHOD_TRAILED
    return new_stop, new_method
