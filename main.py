"""
Main Execution Pipeline for Indian Equity Portfolio Toolkit.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

Usage:
    python main.py [OPTIONS]

Examples:
    # Run the default 1-year analysis pipeline:
    python main.py

    # Run analysis using locally cached data (skip network calls):
    python main.py --skip-fetch

    # Point to a custom Nifty 500 TRI CSV:
    python main.py --tri-csv path/to/nifty500_tri.csv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis import generate_portfolio_summary, print_summary_table
from config import (
    BENCHMARK_PRICE_TICKER,
    CEMENT_STOCKS,
    CAPITAL_GOODS_EPC_STOCKS,
    DEFAULT_TRI_CSV_PATH,
    HISTORICAL_OHLCV_CSV,
    LOCKED_PORTFOLIO_SYMBOLS,
    PORTFOLIO_SYMBOLS,
    POWER_SECTOR_STOCKS,
    PROCESSED_DATA_DIR,
    SUMMARY_OUTPUT_CSV,
)
from fetch_data import (
    NSEBhavcopyFetcher,
    fetch_benchmark_nifty500,
    fetch_benchmark_tri_automated,
    get_one_year_date_range,
    load_benchmark_tri,
)

logger = logging.getLogger("main")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the equity portfolio pipeline."""
    parser = argparse.ArgumentParser(
        description="Indian Equity Portfolio Data Pipeline & Technical Toolkit (IIM Bodh Gaya - SAPM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_start, default_end = get_one_year_date_range()

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start.strftime("%Y-%m-%d"),
        help="Pipeline start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end.strftime("%Y-%m-%d"),
        help="Pipeline end date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip NSE Bhavcopy network download and use locally cached data."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of all NSE daily bhavcopies ignoring existing cache."
    )
    parser.add_argument(
        "--tri-source",
        type=str,
        choices=["manual", "automated"],
        default="manual",
        help="Source for Nifty 500 TRI data: 'manual' (default local CSV) or 'automated' (browser automation via Playwright)."
    )
    parser.add_argument(
        "--tri-csv",
        type=str,
        default=str(DEFAULT_TRI_CSV_PATH),
        help="Path to local manual Nifty 500 TRI CSV downloaded from niftyindices.com."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(SUMMARY_OUTPUT_CSV),
        help="File path to write the portfolio technical summary CSV."
    )
    parser.add_argument(
        "--ohlcv-csv",
        type=str,
        default=str(HISTORICAL_OHLCV_CSV),
        help="File path to write consolidated historical OHLCV data."
    )
    return parser.parse_args()


def display_project_header(tri_source: str = "manual") -> None:
    """Print academic project banner and universe overview."""
    print("=" * 115)
    print(" INDIAN EQUITY PORTFOLIO DATA PIPELINE & ANALYSIS TOOLKIT")
    print(" Coursework: Security Analysis & Portfolio Management (SAPM) / Derivatives")
    print(" Institution: Indian Institute of Management (IIM) Bodh Gaya")
    print("=" * 115)
    print(f" [*] Cement Universe (Locked):          {', '.join(CEMENT_STOCKS)}")
    print(f" [*] Capital Goods / EPC Candidates:   {', '.join(CAPITAL_GOODS_EPC_STOCKS)}")
    if POWER_SECTOR_STOCKS:
        print(f" [*] Power Sector Candidates:          {', '.join(POWER_SECTOR_STOCKS)}")
    else:
        print(" [*] Power Sector Candidates:          [Pending / Placeholder: Not yet finalized]")
    print("-" * 115)
    print(f" [*] FINAL LOCKED PORTFOLIO (8 stocks): {', '.join(LOCKED_PORTFOLIO_SYMBOLS)}")
    print("-" * 115)
    print(f" [*] Benchmark 1 (Price Return):       Nifty 500 ({BENCHMARK_PRICE_TICKER} via yfinance)")
    tri_desc = "Local Nifty 500 TRI CSV (manual)" if tri_source == "manual" else "Automated Browser Fetch (niftyindices.com)"
    print(f" [*] Benchmark 2 (Total Return Index): {tri_desc}")
    print("=" * 115 + "\n")


def run_pipeline() -> int:
    """
    Execute the end-to-end data ingestion, benchmark alignment, and technical analysis workflow.
    """
    args = parse_arguments()
    display_project_header(args.tri_source)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if start_date >= end_date:
        logger.error("Start date (%s) must precede end date (%s).", start_date, end_date)
        return 1

    # -------------------------------------------------------------------------
    # STEP 1: Benchmark Data Ingestion
    # -------------------------------------------------------------------------
    print(">>> [Step 1/4] Ingesting Benchmark Series (Nifty 500)...")

    # 1A. Benchmark Price Return Index via yfinance
    benchmark_price_df = fetch_benchmark_nifty500(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker=BENCHMARK_PRICE_TICKER
    )

    if benchmark_price_df.empty:
        logger.warning("Could not download benchmark price series. Proceeding with caution.")

    # 1B. Benchmark Total Return Index (TRI) via Local CSV or Browser Automation
    if args.tri_source == "automated":
        print(f"     [+] Fetching Benchmark TRI via automated browser session ({args.start_date} to {args.end_date})...")
        benchmark_tri_df = fetch_benchmark_tri_automated(
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        benchmark_tri_df = load_benchmark_tri(Path(args.tri_csv), as_of=end_date)

    if not benchmark_tri_df.empty:
        print(f"     [+] Benchmark TRI verified: {len(benchmark_tri_df)} daily sessions loaded.")
    else:
        print("     [!] Notice: Benchmark TRI data not available. "
              "Please supply the official CSV from niftyindices.com via --tri-csv or use --tri-source automated.")

    # -------------------------------------------------------------------------
    # STEP 2: Individual Stock Price History via Direct NSE Bhavcopy
    # -------------------------------------------------------------------------
    print("\n>>> [Step 2/4] Ingesting Individual Stock OHLCV via Direct NSE Bhavcopy...")
    print("     [+] Methodology: Official 'Full Bhavcopy and Security Deliverable data' direct HTTP GET.")
    print("     [+] Anti-distortion guarantee: No 3rd-party wrapper libraries; pure exchange clearing records.")

    consolidated_cache_file = Path(PROCESSED_DATA_DIR) / "portfolio_ohlcv.csv"
    stock_df = pd.DataFrame()

    if args.skip_fetch and consolidated_cache_file.exists():
        print(f"     [+] --skip-fetch active: Loading cached historical data from {consolidated_cache_file}")
        stock_df = pd.read_csv(consolidated_cache_file)
        stock_df["DATE1"] = pd.to_datetime(stock_df["DATE1"], format="mixed", errors="coerce")
    else:
        fetcher = NSEBhavcopyFetcher()
        stock_df = fetcher.fetch_date_range(
            start_date=start_date,
            end_date=end_date,
            target_symbols=PORTFOLIO_SYMBOLS,
            use_cache=not args.force_refresh
        )

        if not stock_df.empty:
            # Save processed data for rapid access and pipeline persistence
            stock_df.to_csv(consolidated_cache_file, index=False)
            stock_df.to_csv(Path(args.ohlcv_csv), index=False)
            print(f"     [+] Consolidated {len(stock_df)} historical records across target universe.")

    if stock_df.empty:
        logger.error("No stock data available. Cannot proceed with technical evaluation.")
        return 1

    # -------------------------------------------------------------------------
    # STEP 3: Technical Indicators & Relative Strength Evaluation
    # -------------------------------------------------------------------------
    print("\n>>> [Step 3/4] Computing Technical Indicators & Relative Strength...")
    print("     [+] RSI: 14-period Wilder's smoothing (exponential, exact)")
    print("     [+] ADX: 14-period Wilder's method with +DI and -DI directional strength")
    print("     [+] Relative Strength: 63-day cumulative return spread vs Nifty 500 (pp)")
    print("     [+] Support / Resistance: 20-day rolling swing highs & lows")

    summary_df = generate_portfolio_summary(
        stock_data=stock_df,
        benchmark_df=benchmark_price_df,
        target_symbols=PORTFOLIO_SYMBOLS,
        output_csv_path=Path(args.output_csv)
    )

    # -------------------------------------------------------------------------
    # STEP 4: Render Output Table & Save Clean CSV
    # -------------------------------------------------------------------------
    print("\n>>> [Step 4/4] Output Generation Complete.")
    print_summary_table(summary_df)

    print(f" [OK] Summary CSV successfully written to: {Path(args.output_csv).resolve()}")
    print(f" [OK] Complete Historical OHLCV saved to:    {Path(args.ohlcv_csv).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
