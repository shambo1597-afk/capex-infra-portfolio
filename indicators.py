"""
Technical Indicators and Relative Strength Calculation Engine.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module implements transparent, verifiable technical indicators directly in pandas/numpy
without black-box technical analysis libraries:
1. Relative Strength Index (RSI, 14-period) using J. Welles Wilder's exact exponential smoothing.
2. Average Directional Index (ADX, 14-period) with +DI and -DI for trend strength and direction.
3. Relative Strength (RS) spread vs Nifty 500 benchmark over a 63-day trading horizon.
4. Support & Resistance level identification using rolling swing pivot extrema.

Every function is annotated with comprehensive docstrings explaining the underlying
financial intuition, market dynamics, and mathematical mechanics.
"""

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# WILDER'S EXPONENTIAL SMOOTHING ENGINE
# -----------------------------------------------------------------------------

def wilder_smoothing(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute J. Welles Wilder's exponential smoothing method.

    Financial & Mathematical Concept:
    Introduced by J. Welles Wilder Jr. in 'New Concepts in Technical Trading Systems' (1978),
    Wilder's smoothing was designed specifically for price series to retain memory of past
    volatility while decaying historical shocks exponentially.

    Unlike a Simple Moving Average (SMA), which abruptly drops older observations when they
    fall outside the lookback window (causing 'cliff effect' artificial swings), and unlike
    standard Exponential Moving Average (EMA, where alpha = 2/(N+1)), Wilder's smoothing
    uses an effective smoothing factor of alpha = 1 / N:

        Smoothed_t = (Smoothed_{t-1} * (N - 1) + Value_t) / N
                   = Smoothed_{t-1} + (Value_t - Smoothed_{t-1}) / N

    Initialization:
    Wilder specified that the initial smoothed value at period t = N must be the simple
    arithmetic mean of the first N observations:
        Smoothed_{N-1} = (1 / N) * sum_{i=0}^{N-1} Value_i

    Parameters:
        series (pd.Series): The raw numeric series (gains, losses, true range, etc.).
        period (int): Lookback smoothing parameter N (standard is 14 periods).

    Returns:
        pd.Series: Smoothed series with NaN for index < period - 1.
    """
    clean_series = series.astype(float).copy()
    n = len(clean_series)
    smoothed = np.full(n, np.nan, dtype=float)

    if n < period:
        # Not enough data points to establish the initial Wilder's seed
        return pd.Series(smoothed, index=clean_series.index)

    # Step 1: Initialize the seed with the arithmetic mean of the first 'period' values
    values = clean_series.values
    first_window = values[:period]
    smoothed[period - 1] = np.nanmean(first_window)

    # Step 2: Recursively apply Wilder's smoothing formula for all subsequent periods
    # Formula: Smoothed[t] = (Smoothed[t-1] * (period - 1) + values[t]) / period
    for i in range(period, n):
        prev = smoothed[i - 1]
        curr = values[i]
        if np.isnan(curr):
            smoothed[i] = prev
        elif np.isnan(prev):
            smoothed[i] = curr
        else:
            smoothed[i] = (prev * (period - 1) + curr) / period

    return pd.Series(smoothed, index=clean_series.index)


# -----------------------------------------------------------------------------
# 1. RELATIVE STRENGTH INDEX (RSI - 14 PERIOD)
# -----------------------------------------------------------------------------

def compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using Wilder's 14-period smoothing method.

    Financial Concept & Portfolio Application:
    RSI is a bounded momentum oscillator (ranging from 0 to 100) that evaluates the velocity
    and magnitude of directional price movements:
    - Values above 70 indicate that recent upward momentum is abnormally stretched, placing
      the security in 'overbought' territory where mean-reversion risk or consolidation increases.
    - Values below 30 denote 'oversold' conditions, where intense selling pressure may have
      exhausted itself, presenting potential asymmetric entry opportunities.
    - The centerline (50) separates bullish momentum (> 50) from bearish momentum (< 50).
    In active portfolio management, RSI helps determine timing and avoid chasing parabolic moves.

    Formula:
        delta_t = Close_t - Close_{t-1}
        Gain_t = max(delta_t, 0)
        Loss_t = max(-delta_t, 0)
        AvgGain_t = WilderSmooth(Gain, period)
        AvgLoss_t = WilderSmooth(Loss, period)
        RS_t = AvgGain_t / AvgLoss_t
        RSI_t = 100 - (100 / (1 + RS_t))

    Parameters:
        close_series (pd.Series): Time series of daily closing prices.
        period (int): Lookback smoothing period (default = 14 days).

    Returns:
        pd.Series: 14-period RSI time series.
    """
    if len(close_series) < period + 1:
        return pd.Series(np.nan, index=close_series.index)

    # Step 1: Compute price change relative to previous trading day
    delta = close_series.diff()

    # Step 2: Separate price movements into positive gains and positive losses
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Step 3: Apply Wilder's exact smoothing to gains and losses
    avg_gain = wilder_smoothing(gain, period=period)
    avg_loss = wilder_smoothing(loss, period=period)

    # Step 4: Compute Relative Strength (RS) and handle boundary conditions
    # When avg_loss == 0, price only rose; RSI is mathematically 100
    # When avg_gain == 0, price only fell; RSI is mathematically 0
    rs = np.where(avg_loss == 0.0, np.nan, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Clean edge cases
    rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss > 0.0), 0.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), 50.0, rsi)

    return pd.Series(rsi, index=close_series.index)


# -----------------------------------------------------------------------------
# 2. AVERAGE DIRECTIONAL INDEX (ADX, +DI, -DI - 14 PERIOD)
# -----------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX) along with Positive (+DI) and
    Negative (-DI) Directional Indicators using Wilder's 14-period method.

    Financial Concept & Portfolio Application:
    ADX quantifies the STRENGTH of a price trend regardless of direction, while +DI and -DI
    identify the DIRECTION of that trend:
    - Trend Direction:
        * When +DI > -DI, bulls are in control; the prevailing trend is upward.
        * When -DI > +DI, bears dominate; the prevailing trend is downward.
    - Trend Strength:
        * ADX > 25: A strong, actionable trend is underway. Momentum and trend-following strategies
          perform best in this regime.
        * ADX < 20: The market is in a weak trend, consolidation, or non-directional choppy range.
          Breakout signals are prone to whipsaws.
        * 20 <= ADX <= 25: An emerging trend is forming or an existing trend is fading.

    Formula:
        1. True Range (TR): max(High - Low, |High - Close_{t-1}|, |Low - Close_{t-1}|)
        2. Directional Movement:
           UpMove = High_t - High_{t-1}
           DownMove = Low_{t-1} - Low_t
           +DM = UpMove if (UpMove > DownMove and UpMove > 0) else 0
           -DM = DownMove if (DownMove > UpMove and DownMove > 0) else 0
        3. Wilder's Smoothing:
           ATR = WilderSmooth(TR, period)
           Smooth(+DM) = WilderSmooth(+DM, period)
           Smooth(-DM) = WilderSmooth(-DM, period)
        4. Directional Indicators:
           +DI = 100 * (Smooth(+DM) / ATR)
           -DI = 100 * (Smooth(-DM) / ATR)
        5. Directional Index (DX):
           DX = 100 * |+DI - -DI| / (+DI + -DI)
        6. ADX:
           ADX = WilderSmooth(DX, period)

    Parameters:
        df (pd.DataFrame): DataFrame containing 'HIGH_PRICE', 'LOW_PRICE', and 'CLOSE_PRICE'.
        period (int): Lookback smoothing parameter (default = 14 days).

    Returns:
        pd.DataFrame: Contains columns ['TR', 'PLUS_DI', 'MINUS_DI', 'DX', 'ADX', 'TREND_DIR'].
    """
    # Verify required price columns
    required_cols = ["HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' for ADX computation.")

    high = df["HIGH_PRICE"].astype(float)
    low = df["LOW_PRICE"].astype(float)
    close = df["CLOSE_PRICE"].astype(float)
    prev_close = close.shift(1)

    # Step 1: True Range (TR) captures full intraday volatility including overnight gap risk
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Step 2: Directional Movement (+DM and -DM)
    # Measures whether the current bar expanded higher or lower relative to previous bar
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)

    # Step 3: Smooth TR, +DM, and -DM with Wilder's method
    atr = wilder_smoothing(tr, period=period)
    smooth_plus_dm = wilder_smoothing(plus_dm_series, period=period)
    smooth_minus_dm = wilder_smoothing(minus_dm_series, period=period)

    # Step 4: Compute Directional Indicators (+DI and -DI)
    # Scaled by ATR so indicators are normalized against recent market volatility
    plus_di = np.where(atr > 0.0, 100.0 * (smooth_plus_dm / atr), 0.0)
    minus_di = np.where(atr > 0.0, 100.0 * (smooth_minus_dm / atr), 0.0)

    plus_di_series = pd.Series(plus_di, index=df.index)
    minus_di_series = pd.Series(minus_di, index=df.index)

    # Step 5: Compute Directional Movement Index (DX)
    di_diff = (plus_di_series - minus_di_series).abs()
    di_sum = plus_di_series + minus_di_series
    dx = np.where(di_sum > 0.0, 100.0 * (di_diff / di_sum), 0.0)
    dx_series = pd.Series(dx, index=df.index)

    # Step 6: Compute ADX by applying Wilder's smoothing to DX
    adx_series = wilder_smoothing(dx_series, period=period)

    # Step 7: Determine Trend Direction based on +DI vs -DI
    trend_dir = np.where(
        plus_di_series > minus_di_series,
        "Bullish (Uptrend)",
        np.where(minus_di_series > plus_di_series, "Bearish (Downtrend)", "Neutral")
    )
    # Mask initial burn-in periods as NaN/Undetermined
    trend_dir = np.where(adx_series.isna(), "N/A (Insufficient History)", trend_dir)

    return pd.DataFrame({
        "TR": tr,
        "PLUS_DI": plus_di_series,
        "MINUS_DI": minus_di_series,
        "DX": dx_series,
        "ADX": adx_series,
        "TREND_DIR": trend_dir
    }, index=df.index)


# -----------------------------------------------------------------------------
# 3. RELATIVE STRENGTH VS BENCHMARK (63 TRADING DAYS)
# -----------------------------------------------------------------------------

def compute_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_days: int = 63,
    stock_date_col: str = "DATE1",
    stock_price_col: str = "CLOSE_PRICE",
    bench_date_col: str = "Date",
    bench_price_col: str = "Close"
) -> Tuple[float, float, float]:
    """
    Compute Relative Strength (RS) of a stock against the Nifty 500 benchmark over the
    last 63 trading days (~1 calendar quarter / 3 months).

    Financial Concept & Portfolio Rationale:
    In modern portfolio theory and momentum investing (e.g., Jegadeesh & Titman; O'Neil RS),
    Relative Strength isolates pure alpha and market leadership:
    - A stock whose cumulative return exceeds the benchmark index possesses structural buying
      support and institutional accumulation.
    - 63 trading days (approx. 3 months) represents the sweet spot for intermediate-term momentum,
      smoothing out high-frequency 1-week noise while remaining fast enough to reflect quarterly
      earnings catalysts before 252-day annual metrics catch up.
    - Formula:
        Stock_Return (%) = ((P_latest - P_{latest - 63}) / P_{latest - 63}) * 100
        Benchmark_Return (%) = ((B_latest - B_{latest - 63}) / B_{latest - 63}) * 100
        RS_Spread (pp) = Stock_Return - Benchmark_Return

    Parameters:
        stock_df (pd.DataFrame): Time series DataFrame of the individual stock.
        benchmark_df (pd.DataFrame): Time series DataFrame of the benchmark (Nifty 500).
        lookback_days (int): Lookback trading sessions (default = 63).
        stock_date_col (str): Column name for stock date.
        stock_price_col (str): Column name for stock closing price.
        bench_date_col (str): Column name for benchmark date.
        bench_price_col (str): Column name for benchmark closing price.

    Returns:
        Tuple[float, float, float]:
            (rs_spread_pp, stock_cumulative_return_pct, benchmark_cumulative_return_pct)
    """
    if stock_df.empty or benchmark_df.empty:
        return np.nan, np.nan, np.nan

    # Clean and align dates
    s_df = stock_df[[stock_date_col, stock_price_col]].dropna().copy()
    s_df[stock_date_col] = pd.to_datetime(s_df[stock_date_col]).dt.tz_localize(None)
    s_df = s_df.sort_values(stock_date_col).reset_index(drop=True)

    b_df = benchmark_df[[bench_date_col, bench_price_col]].dropna().copy()
    b_df[bench_date_col] = pd.to_datetime(b_df[bench_date_col]).dt.tz_localize(None)
    b_df = b_df.sort_values(bench_date_col).reset_index(drop=True)

    # Merge on exact calendar dates to ensure paired comparisons
    merged = pd.merge(
        s_df,
        b_df,
        left_on=stock_date_col,
        right_on=bench_date_col,
        how="inner"
    ).sort_values(stock_date_col).reset_index(drop=True)

    total_sessions = len(merged)
    if total_sessions < 2:
        return np.nan, np.nan, np.nan

    # If fewer than 63 paired trading sessions exist, use available history
    effective_lookback = min(lookback_days, total_sessions - 1)

    p_latest = float(merged[stock_price_col].iloc[-1])
    p_base = float(merged[stock_price_col].iloc[-1 - effective_lookback])

    b_latest = float(merged[bench_price_col].iloc[-1])
    b_base = float(merged[bench_price_col].iloc[-1 - effective_lookback])

    if p_base <= 0 or b_base <= 0:
        return np.nan, np.nan, np.nan

    # Step 1: Calculate cumulative percentage returns
    stock_return_pct = ((p_latest - p_base) / p_base) * 100.0
    bench_return_pct = ((b_latest - b_base) / b_base) * 100.0

    # Step 2: Compute percentage-point spread (Alpha / Relative Strength)
    rs_spread_pp = stock_return_pct - bench_return_pct

    return round(rs_spread_pp, 2), round(stock_return_pct, 2), round(bench_return_pct, 2)


def compute_sector_relative_strength(
    stock_returns_pct: Mapping[str, float],
) -> Tuple[Dict[str, float], float]:
    """
    Relative strength of each stock against its own sector's equal-weighted average return.

    Same percentage-point spread methodology as compute_relative_strength(), but the benchmark
    is the simple average of the sector constituents' own cumulative returns over the same
    window (e.g. the stock_cumulative_return_pct values from compute_relative_strength):

        sector_avg = mean(R_i)            (equal-weighted, over stocks with a valid return)
        RS_sector_i = R_i - sector_avg    (percentage points)

    By construction the spreads sum to zero across the stocks included in the average. It
    answers "which stock is best positioned among its peers", independent of whether the
    sector as a whole beat the market.

    Parameters:
        stock_returns_pct (Mapping[str, float]): Cumulative return (%) per symbol; NaN/None
            entries are excluded from the average and get a NaN spread.

    Returns:
        Tuple[Dict[str, float], float]: ({symbol: spread in pp}, sector average return %).
    """
    valid = {s: float(r) for s, r in stock_returns_pct.items() if r is not None and not np.isnan(r)}
    if not valid:
        return {s: np.nan for s in stock_returns_pct}, np.nan
    sector_avg = float(np.mean(list(valid.values())))
    spreads = {s: (valid[s] - sector_avg if s in valid else np.nan) for s in stock_returns_pct}
    return spreads, sector_avg


def compute_recent_rs_contribution(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    recent_days: int = 10,
    full_days: int = 63,
) -> Tuple[float, float, float]:
    """
    Share of the full-window RS earned in the most recent sessions: a simple "already run up"
    heuristic (not an established indicator).

        RS_recent = RS vs benchmark over the last `recent_days` sessions        (pp)
        RS_full   = RS vs benchmark over the last `full_days` sessions (63)     (pp)
        recent_contribution_pct = RS_recent / RS_full x 100      (only when RS_full > 0)

    Both RS values come from compute_relative_strength() on the same paired sessions. Spreads
    of compounded returns are not strictly additive, so the percentage is approximate. Reading:
    10 of 63 sessions is ~16% of the window, so steady outperformance earns roughly that share
    recently; well above it (e.g. > 50%) means most of the edge arrived in a recent burst that
    may mean-revert; above 100% means the stock was flat or lagging before the burst; negative
    means it has been giving back ground lately. Undefined (NaN) when RS_full <= 0, since there
    is no outperformance to attribute.

    Returns:
        Tuple[float, float, float]: (RS_recent, RS_full, recent_contribution_pct)
    """
    rs_recent, _, _ = compute_relative_strength(stock_df, benchmark_df, lookback_days=recent_days)
    rs_full, _, _ = compute_relative_strength(stock_df, benchmark_df, lookback_days=full_days)
    if np.isnan(rs_recent) or np.isnan(rs_full) or rs_full <= 0:
        return rs_recent, rs_full, np.nan
    return rs_recent, rs_full, round(rs_recent / rs_full * 100.0, 1)


def compute_rs_at_offsets(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_days: int,
    offsets,
) -> Dict[int, Tuple[float, float]]:
    """
    The lookback-session RS spread and the stock's own return for windows ending `offset` paired
    sessions before the latest one (offset 0 = today), as {offset: (rs_pp, stock_return_pct)}.
    NaN pairs where the history is too short for that window.
    """
    nan = (np.nan, np.nan)
    offsets = list(offsets)
    if stock_df.empty or benchmark_df.empty:
        return {k: nan for k in offsets}
    stock_dates = pd.to_datetime(stock_df["DATE1"]).dt.tz_localize(None)
    bench_dates = pd.to_datetime(benchmark_df["Date"]).dt.tz_localize(None)
    paired = np.sort(np.intersect1d(stock_dates[stock_df["CLOSE_PRICE"].notna()].unique(),
                                    bench_dates[benchmark_df["Close"].notna()].unique()))
    out = {}
    for k in offsets:
        if len(paired) < lookback_days + k + 1:
            out[k] = nan
            continue
        cutoff = pd.Timestamp(paired[-1 - k])
        rs, stock_ret, _ = compute_relative_strength(
            stock_df[stock_dates <= cutoff], benchmark_df[bench_dates <= cutoff], lookback_days=lookback_days)
        out[k] = (rs, stock_ret)
    return out


def compute_rs_momentum(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_days: int = 63,
    momentum_days: int = 10,
    smoothing_days: int = 1,
) -> Tuple[float, float, float, Dict[int, float]]:
    """
    RS-Momentum for a Relative Rotation Graph: the rate of change of the RS spread, smoothed.

        RS(k)       = RS vs benchmark over the `lookback_days` sessions ending k sessions ago (pp)
        RS_avg_now  = mean of RS(0) .. RS(smoothing_days - 1)
        RS_avg_prev = mean of RS(momentum_days) .. RS(momentum_days + smoothing_days - 1)
        RS_momentum = RS_avg_now - RS_avg_prev                                            (pp)

    Positive momentum means relative strength improved over the last `momentum_days` sessions.
    Averaging a few days at each end (smoothing_days, e.g. 5) stops a single session from
    flipping the sign: with smoothing_days=1 this is the plain RS(0) - RS(momentum_days), which
    changed a stock's RRG tier on about 17% of days. Unlike the run-up share
    (compute_recent_rs_contribution), it is defined for negative RS too, which the IMPROVING and
    LAGGING quadrants need.

    Returns:
        (RS_now, RS_avg_prev, RS_momentum, {offset: stock_return_pct}) where RS_now is the
        unsmoothed RS today (the RRG x-axis) and the returns cover every offset used, for the
        same calculation against a sector average. NaNs when there is too little history.
    """
    now_offsets = list(range(smoothing_days))
    prev_offsets = list(range(momentum_days, momentum_days + smoothing_days))
    points = compute_rs_at_offsets(stock_df, benchmark_df, lookback_days, now_offsets + prev_offsets)
    returns = {k: v[1] for k, v in points.items()}
    now_vals = [points[k][0] for k in now_offsets]
    prev_vals = [points[k][0] for k in prev_offsets]
    if any(np.isnan(v) for v in now_vals + prev_vals):
        return np.nan, np.nan, np.nan, returns
    rs_now = points[0][0]
    rs_prev = float(np.mean(prev_vals))
    return rs_now, round(rs_prev, 2), round(float(np.mean(now_vals)) - rs_prev, 2), returns


# -----------------------------------------------------------------------------
# 4. SUPPORT & RESISTANCE IDENTIFICATION (20-DAY ROLLING WINDOW)
# -----------------------------------------------------------------------------

def compute_support_resistance(
    df: pd.DataFrame,
    rolling_window: int = 20,
    current_price: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Identify nearest Support and Resistance price levels using rolling swing pivots.

    Financial Concept & Risk Management Application:
    Support and resistance represent structural supply and demand imbalances in order books:
    - Support: A price floor where aggregate institutional demand has historically overcome
      selling volume, preventing further downward slides.
    - Resistance: A price ceiling where supply overhang from profit-takers and trapped overhead
      buyers halts rallies.
    - Identifying the nearest support level informs stop-loss placement, downside risk evaluation,
      and Value-at-Risk (VaR) parameters.
    - Identifying the nearest resistance level defines target exit prices and upside reward potential,
      enabling strict Risk/Reward Ratio (RRR) filtering.

    Algorithm:
    1. Scan price history over rolling windows (default 20 trading sessions, ~1 calendar month).
    2. Local Swing High: A session high that matches the rolling window peak.
    3. Local Swing Low: A session low that matches the rolling window trough.
    4. Current Reference Price: Latest closing price.
    5. Nearest Support: The highest swing low strictly below current price:
       Support = max({low_t | low_t is local swing low and low_t < Current_Price})
    6. Nearest Resistance: The lowest swing high strictly above current price:
       Resistance = min({high_t | high_t is local swing high and high_t > Current_Price})
    7. Edge Cases: If current price is printing new 52-week highs (blue-sky breakout) or lows,
       nearest boundaries fall back to the window extremes.

    Parameters:
        df (pd.DataFrame): Historical DataFrame containing 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE'.
        rolling_window (int): Size of rolling lookback window for swing detection (default = 20).
        current_price (float, optional): Reference price. If None, uses latest 'CLOSE_PRICE'.

    Returns:
        Tuple[Optional[float], Optional[float]]: (nearest_support, nearest_resistance)
    """
    if len(df) < 5:
        return None, None

    if current_price is None:
        current_price = float(df["CLOSE_PRICE"].iloc[-1])

    highs = df["HIGH_PRICE"].astype(float)
    lows = df["LOW_PRICE"].astype(float)

    # Compute rolling maximums and minimums
    # A local swing peak occurs where high[t] == rolling_max[t]
    # A local swing trough occurs where low[t] == rolling_min[t]
    rolling_highs = highs.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).max()
    rolling_lows = lows.rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).min()

    # Collect distinct swing high and swing low price levels
    swing_highs_set = set(rolling_highs.dropna().round(2).unique())
    swing_lows_set = set(rolling_lows.dropna().round(2).unique())

    # Support candidates must be strictly below current price
    valid_supports = [lvl for lvl in swing_lows_set if lvl < current_price]
    # Resistance candidates must be strictly above current price
    valid_resistances = [lvl for lvl in swing_highs_set if lvl > current_price]

    # Nearest support is the highest level beneath current price (closest floor)
    nearest_support = max(valid_supports) if valid_supports else None

    # Nearest resistance is the lowest level above current price (closest ceiling)
    nearest_resistance = min(valid_resistances) if valid_resistances else None

    # Fallback handlers for extreme boundary conditions:
    # 1. If at 52-week low with no swing low below current price:
    if nearest_support is None:
        min_seen = float(lows.min())
        if min_seen < current_price:
            nearest_support = round(min_seen, 2)
        else:
            # Current price is at absolute historical trough
            nearest_support = round(current_price * 0.95, 2)  # 5% psychological buffer

    # 2. If at 52-week high with no swing high above current price (All-Time High / Blue Sky):
    if nearest_resistance is None:
        max_seen = float(highs.max())
        if max_seen > current_price:
            nearest_resistance = round(max_seen, 2)
        else:
            # Breaking out into uncharted territory
            nearest_resistance = None  # None indicates All-Time High / Blue Sky

    return nearest_support, nearest_resistance
