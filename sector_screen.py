"""
Sector Screen: technical screen first, then fundamental safety screen.

Implements the project brief's order ("Technical analysis, then financial analysis") identically
for each sector:

1. Universe: the official Nifty sector index constituents only (config.SECTOR_SCREENS), with no
   manual additions.
2. Technical screen on every constituent, using the existing indicator pipeline
   (analysis.evaluate_stock_technicals) on the complete NSE Bhavcopy history:
   passes when RS vs Nifty 500 (63 sessions) > 0 AND trend direction (+DI vs -DI) is Bullish.
   ADX is reported for tie-breaking but is not a cutoff.
3. Fundamental safety screen, run live (fundamentals.get_fundamentals_summary, no cache) ONLY for
   stocks that passed step 2, against that sector's criteria in config.py.

Writes one CSV per sector (output/cement_full_screen.csv, output/capital_goods_full_screen.csv,
output/power_full_screen.csv) listing every constituent.

Usage:
    python sector_screen.py
"""

import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis import evaluate_stock_technicals
from config import SECTOR_SCREENS, TECHNICAL_RS_LOOKBACK_DAYS
from fetch_data import NSEBhavcopyFetcher, fetch_benchmark_nifty500, get_one_year_date_range
from fundamentals import evaluate_fundamental_screen, get_fundamentals_summary

logger = logging.getLogger("sector_screen")


def load_constituents(csv_path: Path) -> pd.DataFrame:
    """Load an official niftyindices.com constituent file (Company Name, Industry, Symbol, ...)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df


def technical_screen_result(rs_score: float, trend_direction: str) -> Tuple[bool, str]:
    """
    Apply the technical screen: RS vs Nifty 500 > 0 AND trend direction Bullish.

    Returns:
        (passed, reason): reason is "" when passed, else why it failed.
    """
    reasons = []
    if rs_score is None or pd.isna(rs_score):
        reasons.append("RS unavailable")
    elif rs_score <= 0:
        reasons.append("RS <= 0")
    if not str(trend_direction).startswith("Bullish"):
        reasons.append(f"trend {str(trend_direction).split(' (')[0]}")
    return not reasons, "; ".join(reasons)


def screen_sector(
    sector: str,
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    criteria: List[Tuple[str, str, float, str]],
) -> pd.DataFrame:
    """
    Run the technical screen on every constituent, then the live fundamental screen on the
    technical passers only. Returns one row per constituent.
    """
    rows: List[Dict] = []
    for _, member in constituents.iterrows():
        symbol = member["Symbol"]
        sym_prices = prices[prices["SYMBOL"] == symbol] if not prices.empty else pd.DataFrame()
        tech = evaluate_stock_technicals(symbol, sym_prices, benchmark, rs_lookback=TECHNICAL_RS_LOOKBACK_DAYS)
        passed, reason = technical_screen_result(tech["rs_score_vs_nifty500"], tech["trend_direction"])
        rows.append({
            "symbol": symbol,
            "company_name": member.get("Company Name"),
            "index": f"Nifty {sector}",
            "price_sessions": int(sym_prices["DATE1"].nunique()) if not sym_prices.empty else 0,
            "current_price": tech["current_price"],
            "passed_technical_screen": passed,
            "technical_fail_reason": reason,
            "rs_score": tech["rs_score_vs_nifty500"],
            "adx": tech["latest_adx"],
            "plus_di": tech.get("plus_di"),
            "minus_di": tech.get("minus_di"),
            "trend_direction": tech["trend_direction"],
            "rsi": tech["latest_rsi"],
        })
    screen = pd.DataFrame(rows)

    # Fundamental safety screen: live Screener.in data, technical passers only
    passers = screen.loc[screen["passed_technical_screen"], "symbol"].tolist()
    fundamentals = get_fundamentals_summary(passers, use_cache=False) if passers else pd.DataFrame()
    records = {r["symbol"]: r for r in fundamentals.to_dict("records")} if not fundamentals.empty else {}

    metric_cols = [c for field, *_ in criteria for c in (field, f"pass_{field}")]
    for col in ["fundamentals_as_of", "fundamentals_status", *metric_cols, "passed_fundamental_screen", "failed_criteria"]:
        screen[col] = pd.Series([None] * len(screen), dtype="object")
    for i, row in screen.iterrows():
        if not row["passed_technical_screen"]:
            continue
        record = records.get(row["symbol"], {"status": "Data Unavailable"})
        result = evaluate_fundamental_screen(record, criteria)
        failed = result.pop("failed_criteria")
        for col, value in result.items():
            if isinstance(value, float) and math.isinf(value):
                value = "inf"  # e.g. zero interest expense
            screen.at[i, col] = value
        screen.at[i, "fundamentals_as_of"] = date.today().isoformat()
        screen.at[i, "fundamentals_status"] = record.get("status")
        screen.at[i, "passed_fundamental_screen"] = not failed
        screen.at[i, "failed_criteria"] = "; ".join(failed)

    screen["passes_both_screens"] = screen["passed_technical_screen"] & (screen["passed_fundamental_screen"] == True)  # noqa: E712
    return screen


def run_sector_screens(sectors: Optional[List[str]] = None, use_price_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Screen each configured sector and write its CSV. Returns {sector: screen DataFrame}."""
    sectors = sectors or list(SECTOR_SCREENS)
    constituents = {s: load_constituents(SECTOR_SCREENS[s]["constituents_csv"]) for s in sectors}
    all_symbols = sorted({sym for df in constituents.values() for sym in df["Symbol"]})

    start, end = get_one_year_date_range()
    prices = NSEBhavcopyFetcher().fetch_date_range(start, end, all_symbols, use_cache=use_price_cache)
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.isoformat())
    if benchmark.empty:
        raise RuntimeError("Nifty 500 benchmark unavailable; RS cannot be computed.")

    results = {}
    for sector in sectors:
        cfg = SECTOR_SCREENS[sector]
        screen = screen_sector(sector, constituents[sector], prices, benchmark, cfg["criteria"])
        out = Path(cfg["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        screen.to_csv(out, index=False)
        logger.info("Saved %s screen (%d constituents) to %s", sector, len(screen), out.resolve())
        results[sector] = screen
    return results


def print_screen_summary(results: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 115)
    print(" SECTOR SCREEN: technical screen first (RS > 0 AND Bullish), then live fundamental safety screen")
    print("=" * 115)
    for sector, df in results.items():
        tech = df[df["passed_technical_screen"]]
        both = df[df["passes_both_screens"]]
        print(f" {sector:14s} constituents {len(df):3d} | passed technical {len(tech):3d} | passed both {len(both):3d}"
              f" -> {', '.join(both['symbol']) or '-'}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    print_screen_summary(run_sector_screens())
    sys.exit(0)
