"""
Portfolio Analysis and Technical Summary Engine.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module ingests processed historical OHLCV data for the target stock universe,
runs the technical indicators (RSI, ADX, Relative Strength vs Nifty 500, Support/Resistance),
assembles the consolidated portfolio summary row per stock, saves the output CSV,
and renders a readable terminal summary table. It also builds the locked portfolio's
risk summary (volatility, historical expected return, weight, ATR stop-loss).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import (
    EQUITY_ALLOCATION_PCT,
    INITIAL_HOLDINGS,
    PRINCIPAL_INR,
    RISK_SUMMARY_OUTPUT_CSV,
    STOP_LOSS_ATR_MULTIPLE,
    STOP_LOSS_ATR_PERIOD,
    STOP_LOSS_SUPPORT_BAND_ATR,
    SUMMARY_OUTPUT_CSV,
    TECHNICAL_RS_LOOKBACK_DAYS,
    sector_of,
)
from indicators import (
    compute_adx,
    compute_relative_strength,
    compute_rsi,
    compute_support_resistance,
)
from weights import compute_portfolio_weights
from stoploss import (
    compute_annualized_volatility,
    compute_daily_returns,
    compute_daily_volatility,
    apply_trailing_stop,
    compute_atr,
    compute_atr_stop,
    compute_historical_expected_return,
)

logger = logging.getLogger("analysis")


def evaluate_stock_technicals(
    symbol: str,
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    rsi_period: int = 14,
    adx_period: int = 14,
    rs_lookback: int = TECHNICAL_RS_LOOKBACK_DAYS,
    sr_window: int = 20
) -> Dict[str, Union[str, float, None]]:
    """
    Compute all required technical metrics for a single stock and construct its summary record.

    Financial Concept & Logic:
    - Current Price: The latest official close price from NSE Bhavcopy.
    - RSI (14-period Wilder): Gauges overbought (>70) or oversold (<30) momentum velocity.
    - ADX & Direction (14-period Wilder): Gauges trend strength (>25 indicates strong trend)
      and direction (+DI vs -DI).
    - Relative Strength (63-day spread): Quantifies 1-quarter alpha vs the Nifty 500 benchmark.
    - Nearest Support: Identifies the nearest price floor beneath current price for risk management.
    - Nearest Resistance: Identifies the overhead ceiling for target price forecasting.

    Parameters:
        symbol (str): Equity ticker.
        stock_df (pd.DataFrame): Historical daily OHLCV series.
        benchmark_df (pd.DataFrame): Benchmark daily index series.
        rsi_period (int): Period for Wilder's RSI (default 14).
        adx_period (int): Period for Wilder's ADX (default 14).
        rs_lookback (int): Trading sessions for RS spread (default config.TECHNICAL_RS_LOOKBACK_DAYS).
        sr_window (int): Rolling window for support/resistance (default 20).

    Returns:
        Dict: Single summary dictionary for the stock.
    """
    if stock_df.empty or len(stock_df) < 5:
        logger.warning("Insufficient historical data for symbol %s to evaluate indicators.", symbol)
        return {
            "symbol": symbol,
            "current_price": np.nan,
            "latest_rsi": np.nan,
            "latest_adx": np.nan,
            "trend_direction": "N/A (Insufficient Data)",
            "plus_di": np.nan,
            "minus_di": np.nan,
            "rs_score_vs_nifty500": np.nan,
            "nearest_support": None,
            "nearest_resistance": None,
            "sector": sector_of(symbol),
        }

    # Sort stock history by date
    clean_stock = stock_df.sort_values("DATE1").reset_index(drop=True)
    latest_close = float(clean_stock["CLOSE_PRICE"].iloc[-1])

    # 1. RSI (14-period Wilder)
    rsi_series = compute_rsi(clean_stock["CLOSE_PRICE"], period=rsi_period)
    latest_rsi = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else np.nan

    # 2. ADX, +DI, -DI (14-period Wilder)
    adx_df = compute_adx(clean_stock, period=adx_period)
    latest_adx = float(adx_df["ADX"].iloc[-1]) if not adx_df["ADX"].isna().all() else np.nan
    latest_plus_di = float(adx_df["PLUS_DI"].iloc[-1]) if not adx_df["PLUS_DI"].isna().all() else np.nan
    latest_minus_di = float(adx_df["MINUS_DI"].iloc[-1]) if not adx_df["MINUS_DI"].isna().all() else np.nan
    latest_trend_dir = str(adx_df["TREND_DIR"].iloc[-1])

    # 3. Relative Strength Spread vs Nifty 500 (63-day lookback)
    rs_spread_pp, stock_ret, bench_ret = compute_relative_strength(
        stock_df=clean_stock,
        benchmark_df=benchmark_df,
        lookback_days=rs_lookback
    )

    # 4. Support and Resistance levels (20-day rolling window)
    support_lvl, resistance_lvl = compute_support_resistance(
        df=clean_stock,
        rolling_window=sr_window,
        current_price=latest_close
    )

    return {
        "symbol": symbol,
        "sector": sector_of(symbol),
        "current_price": round(latest_close, 2),
        "latest_rsi": round(latest_rsi, 2) if not np.isnan(latest_rsi) else np.nan,
        "latest_adx": round(latest_adx, 2) if not np.isnan(latest_adx) else np.nan,
        "plus_di": round(latest_plus_di, 2) if not np.isnan(latest_plus_di) else np.nan,
        "minus_di": round(latest_minus_di, 2) if not np.isnan(latest_minus_di) else np.nan,
        "trend_direction": latest_trend_dir,
        "rs_score_vs_nifty500": round(rs_spread_pp, 2) if not np.isnan(rs_spread_pp) else np.nan,
        "nearest_support": support_lvl,
        "nearest_resistance": resistance_lvl,
    }


def generate_portfolio_summary(
    stock_data: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    target_symbols: Optional[List[str]] = None,
    output_csv_path: Optional[Path] = SUMMARY_OUTPUT_CSV
) -> pd.DataFrame:
    """
    Generate the complete portfolio technical summary across all target stocks,
    save the clean CSV output, and return the summary DataFrame.

    Parameters:
        stock_data (pd.DataFrame): Consolidated historical OHLCV data.
        benchmark_df (pd.DataFrame): Benchmark Nifty 500 DataFrame.
        target_symbols (List[str], optional): List of symbols to evaluate.
        output_csv_path (Path, optional): Destination file path for summary CSV.

    Returns:
        pd.DataFrame: Portfolio summary DataFrame.
    """
    if target_symbols is None:
        if "SYMBOL" in stock_data.columns:
            target_symbols = sorted(stock_data["SYMBOL"].unique().tolist())
        else:
            target_symbols = []

    logger.info("Evaluating portfolio technical indicators for %d symbols...", len(target_symbols))

    summary_records = []
    for symbol in target_symbols:
        sym_df = stock_data[stock_data["SYMBOL"] == symbol] if not stock_data.empty else pd.DataFrame()
        record = evaluate_stock_technicals(
            symbol=symbol,
            stock_df=sym_df,
            benchmark_df=benchmark_df
        )
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    # Order columns cleanly as specified in coursework requirements:
    # symbol, latest RSI, latest ADX (+ trend direction based on +DI vs -DI),
    # RS score vs Nifty 500, current price, nearest support level, nearest resistance level
    display_columns = [
        "symbol",
        "sector",
        "current_price",
        "latest_rsi",
        "latest_adx",
        "trend_direction",
        "rs_score_vs_nifty500",
        "nearest_support",
        "nearest_resistance"
    ]
    # Filter to existing columns
    ordered_cols = [c for c in display_columns if c in summary_df.columns]
    summary_df = summary_df[ordered_cols]

    # Save to clean CSV if path provided
    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        logger.info("Saved portfolio technical summary to: %s", out_path.resolve())

    return summary_df


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """
    Format and print a terminal table of the portfolio technical summary.

    Parameters:
        summary_df (pd.DataFrame): Summary DataFrame to render.
    """
    print("\n" + "=" * 115)
    print(" INDIAN EQUITY PORTFOLIO TECHNICAL ANALYSIS SUMMARY (IIM BODH GAYA - SAPM)")
    print(" Benchmark: Nifty 500 (NSE official closes) | Lookback: 1-Year Bhavcopy History | RS Window: 63 Days")
    print("=" * 115)

    if summary_df.empty:
        print("No summary records to display.")
        print("=" * 115 + "\n")
        return

    # Try using tabulate for beautiful formatting if available
    try:
        from tabulate import tabulate
        # Format None / NaN resistance as 'ATH / Blue Sky'
        formatted_df = summary_df.copy()
        formatted_df["nearest_resistance"] = formatted_df["nearest_resistance"].apply(
            lambda x: "ATH / Blue Sky" if pd.isna(x) or x is None else f"{x:,.2f}"
        )
        formatted_df["nearest_support"] = formatted_df["nearest_support"].apply(
            lambda x: "N/A" if pd.isna(x) or x is None else f"{x:,.2f}"
        )
        formatted_df["current_price"] = formatted_df["current_price"].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
        )
        formatted_df["rs_score_vs_nifty500"] = formatted_df["rs_score_vs_nifty500"].apply(
            lambda x: f"{x:+.2f} pp" if pd.notna(x) else "N/A"
        )
        table_str = tabulate(
            formatted_df,
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False,
            numalign="right",
            stralign="left"
        )
        try:
            print(table_str)
        except UnicodeEncodeError:
            # Windows cp1252 console fallback for box-drawing characters
            print(tabulate(
                formatted_df,
                headers="keys",
                tablefmt="grid",
                showindex=False,
                numalign="right",
                stralign="left"
            ))
    except ImportError:
        # Fallback to pandas string rendering
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(summary_df.to_string(index=False))

    print("=" * 115)
    print(" Key Takeaways:")
    print(" - RSI > 70: Overbought momentum | RSI < 30: Oversold / Mean-reversion candidate")
    print(" - ADX > 25: Strong directional trend | ADX < 20: Consolidating / Range-bound")
    print(f" - RS Score: Percentage-point excess return over Nifty 500 during the last {TECHNICAL_RS_LOOKBACK_DAYS} trading days")
    print("=" * 115 + "\n")


RISK_SUMMARY_COLUMNS = [
    "symbol",
    "sector",
    "current_price",
    "annualized_volatility_pct",
    "historical_expected_return_pct",
    "weight_pct",
    "risk_contribution_pct",
    "shares",
    "invested_inr",
    "atr_14",
    "atr_pct",
    "stop_loss_price",
    "stop_loss_pct_below_current",
    "stop_loss_method",
]


def _pct(fraction: Optional[float]) -> Optional[float]:
    return None if fraction is None else round(fraction * 100, 2)


def generate_portfolio_risk_summary(
    stock_data: pd.DataFrame,
    technical_summary: pd.DataFrame,
    symbols: Optional[List[str]] = None,
    output_csv_path: Optional[Path] = RISK_SUMMARY_OUTPUT_CSV,
    previous_stops: Optional[Dict[str, float]] = None,
    held_shares: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build one risk/sizing row per locked portfolio stock and save it as a CSV.
    Before the snapshot, shares are the whole shares each weight buys at the latest close; once
    invested, `held_shares` (net quantities from the trade ledger) are shown instead, valued at the close.

    Columns: symbol, sector, current_price, annualized_volatility_pct,
    historical_expected_return_pct, weight_pct, stop_loss_price,
    stop_loss_pct_below_current, stop_loss_method. See stoploss.py for the methodology.

    Parameters:
        stock_data (pd.DataFrame): Consolidated historical OHLCV (Bhavcopy) data.
        technical_summary (pd.DataFrame): Output of generate_portfolio_summary(); supplies
            current_price and the support-based stop candidate (nearest_support).
        symbols (List[str], optional): Stocks to include. Defaults to INITIAL_HOLDINGS (the invested stocks).
        output_csv_path (Path, optional): Destination CSV; None skips saving.
        previous_stops (dict, optional): symbol -> stop from the previous review. A stop is
            only ever raised: a higher previous stop is kept ("trailed"), and one at or above
            the current price is reported as "breached".

    Returns:
        pd.DataFrame: The risk summary, one row per symbol.
    """
    if symbols is None:
        symbols = INITIAL_HOLDINGS

    # Weights: equal risk contribution within WEIGHT_MIN_PCT..WEIGHT_MAX_PCT (weights.py). They are
    # weights of the equity sleeve (EQUITY_ALLOCATION_PCT of PRINCIPAL_INR); shares are whole shares of
    # each stock's rupee amount at the latest close.
    weights = compute_portfolio_weights(stock_data, symbols).set_index("symbol")
    equity_inr = PRINCIPAL_INR * EQUITY_ALLOCATION_PCT / 100

    technicals = technical_summary.set_index("symbol") if not technical_summary.empty else pd.DataFrame()
    records = []
    for symbol in symbols:
        tech = technicals.loc[symbol] if symbol in technicals.index else None
        current_price = float(tech["current_price"]) if tech is not None and pd.notna(tech["current_price"]) else None
        support = float(tech["nearest_support"]) if tech is not None and pd.notna(tech["nearest_support"]) else None

        sym_df = stock_data[stock_data["SYMBOL"] == symbol] if not stock_data.empty else pd.DataFrame()
        returns = compute_daily_returns(sym_df)
        daily_vol = compute_daily_volatility(returns)

        atr = compute_atr(sym_df)
        stop_price, method = compute_atr_stop(current_price, atr, support)
        stop_price, method = apply_trailing_stop(
            stop_price, method, (previous_stops or {}).get(symbol), current_price)
        logger.info(
            "%s stop-loss: ATR(%d)=%s, support=%s -> %s at %s",
            symbol,
            STOP_LOSS_ATR_PERIOD,
            "n/a" if atr is None else f"{atr:,.2f}",
            "n/a" if support is None else f"{support:,.2f}",
            method,
            "n/a" if stop_price is None else f"{stop_price:,.2f}",
        )
        if len(returns) < 2:
            logger.warning("%s: insufficient price history for volatility (%d daily returns).", symbol, len(returns))

        weight_pct = float(weights.loc[symbol, "weight_pct"])
        if held_shares is not None:
            shares = int(held_shares.get(symbol, 0))
        else:
            shares = int(equity_inr * weight_pct / 100 // current_price) if current_price else None
        records.append({
            "symbol": symbol,
            "sector": sector_of(symbol),
            "current_price": current_price,
            "annualized_volatility_pct": _pct(compute_annualized_volatility(returns)),
            "historical_expected_return_pct": _pct(compute_historical_expected_return(returns)),
            "weight_pct": weight_pct,
            "risk_contribution_pct": weights.loc[symbol, "risk_contribution_pct"],
            # Whole shares buyable with this stock's share of the equity sleeve at the latest close
            "shares": shares,
            "invested_inr": None if shares is None else round(shares * current_price, 2),
            "atr_14": None if atr is None else round(atr, 2),
            "atr_pct": None if atr is None or not current_price else round(atr / current_price * 100, 2),
            "stop_loss_price": None if stop_price is None else round(stop_price, 2),
            "stop_loss_pct_below_current": (
                None if stop_price is None or not current_price
                else round((current_price - stop_price) / current_price * 100, 2)
            ),
            "stop_loss_method": method,
        })

    risk_df = pd.DataFrame(records, columns=RISK_SUMMARY_COLUMNS)

    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        risk_df.to_csv(out_path, index=False)
        logger.info("Saved portfolio risk summary to: %s", out_path.resolve())

    return risk_df


def print_risk_summary_table(risk_df: pd.DataFrame) -> None:
    """Print the locked portfolio's risk, sizing and stop-loss table to the terminal."""
    print("\n" + "=" * 115)
    print(" LOCKED PORTFOLIO RISK, SIZING & STOP-LOSS SUMMARY")
    print(f" Stop-loss: price - {STOP_LOSS_ATR_MULTIPLE:g} x ATR({STOP_LOSS_ATR_PERIOD}), moved just below a support "
          f"level up to {STOP_LOSS_SUPPORT_BAND_ATR:g} ATR beyond it; trailed up (never down) at monthly reviews")
    print("=" * 115)
    if risk_df.empty:
        print("No risk records to display.")
    else:
        try:
            from tabulate import tabulate
            print(tabulate(risk_df, headers="keys", tablefmt="grid", showindex=False,
                           numalign="right", stralign="left", floatfmt=",.2f"))
        except ImportError:
            print(risk_df.to_string(index=False))
    print("=" * 115)
    if not risk_df.empty and risk_df["invested_inr"].notna().any():
        invested = risk_df["invested_inr"].sum()
        print(f" Allocation of Rs {PRINCIPAL_INR:,.0f}: invested Rs {invested:,.2f} ({invested / PRINCIPAL_INR * 100:.2f}%), "
              f"cash Rs {PRINCIPAL_INR - invested:,.2f} (hedge budget {100 - EQUITY_ALLOCATION_PCT:g}% plus rounding)")
    print(" historical_expected_return_pct: past year's average (not a forecast); the CAPM expected return is in")
    print(" output/capm_expected_returns.csv (risk_model.py)")
    print(" Weights: equal risk contribution within the 5-15% bounds (weights.py).")
    print("=" * 115 + "\n")

