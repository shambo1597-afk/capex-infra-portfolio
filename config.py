"""
Configuration Module for Indian Equity Portfolio Pipeline & Analysis Toolkit.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module centralizes all portfolio universe configurations, benchmark settings,
NSE endpoint specifications, header requirements, and directory paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_BHAVCOPY_DIR = DATA_DIR / "raw_bhavcopy"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FUNDAMENTALS_CACHE_DIR = DATA_DIR / "fundamentals_cache"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure runtime directories exist
RAW_BHAVCOPY_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# PORTFOLIO STOCK UNIVERSE CONFIGURATION
# -----------------------------------------------------------------------------

# Cement Sector (Locked universe for coursework portfolio allocation)
CEMENT_STOCKS = [
    "JKCEMENT",    # JK Cement Ltd.
    "ULTRACEMCO",  # UltraTech Cement Ltd. (Market Leader)
    "STARCEMENT",  # Star Cement Ltd. (Regional Growth Play)
]

# Capital Goods / EPC Candidates (Evaluating for portfolio selection)
# 10 candidate symbols identified for technical and relative strength screening
CAPITAL_GOODS_EPC_STOCKS = [
    "ABB",         # ABB India Ltd. (Power & Automation)
    "CGPOWER",     # CG Power and Industrial Solutions Ltd.
    "GVT&D",       # GE Vernova T&D India Ltd. (Formerly GET&D)
    "POWERINDIA",  # Hitachi Energy India Ltd.
    "ELECON",      # Elecon Engineering Company Ltd. (Gears & Material Handling)
    "TRITURBINE",  # Triveni Turbine Ltd. (Industrial Steam Turbines)
    "TDPOWERSYS",  # TD Power Systems Ltd. (Generators)
    "SIEMENS",     # Siemens Ltd. (Industrial conglomerate)
    "SCHNEIDER",   # Schneider Electric Infrastructure Ltd.
    "CEMPRO",      # Cemindia Projects Ltd. (Formerly ITD Cementation India Ltd. - ITDCEM)
]

# Power Sector Stocks (Placeholder: Not yet finalized, to be populated later)
# Leave clearly marked as requested in coursework specifications.
POWER_SECTOR_STOCKS = [
    # Placeholder: Add power sector candidate symbols here once finalized
    # e.g., "NTPC", "POWERGRID", "TATAPOWER"
]

# -----------------------------------------------------------------------------
# LOCKED PORTFOLIO (8 STOCKS)
# Coursework Final Portfolio Allocation across 3 key Infra/Capex Sectors
# -----------------------------------------------------------------------------

LOCKED_PORTFOLIO = {
    "JKCEMENT": {"sector": "Cement", "name": "J K Cement"},
    "ULTRACEMCO": {"sector": "Cement", "name": "UltraTech Cement"},
    "STARCEMENT": {"sector": "Cement", "name": "Star Cement"},
    "BHEL": {"sector": "Capital Goods/EPC", "name": "Bharat Heavy Electricals"},
    "VOLTAMP": {"sector": "Capital Goods/EPC", "name": "Voltamp Transformers"},
    "POWERGRID": {"sector": "Power", "name": "Power Grid Corp."},
    "TATAPOWER": {"sector": "Power", "name": "Tata Power"},
    "NTPC": {"sector": "Power", "name": "NTPC"},
}

LOCKED_PORTFOLIO_SYMBOLS = list(LOCKED_PORTFOLIO.keys())

# Documented investment committee caveat for NTPC
NTPC_CAVEAT = (
    "Note: Shows a bearish near-term technical trend but was included "
    "for its strong fundamentals during a sector-wide downturn."
)

# Complete portfolio watchlist to fetch and analyze
PORTFOLIO_SYMBOLS = CEMENT_STOCKS + CAPITAL_GOODS_EPC_STOCKS + POWER_SECTOR_STOCKS
ALL_TARGET_STOCKS = PORTFOLIO_SYMBOLS

# Corporate Action / Symbol Change Alias Mapping
# On the NSE, corporate renamings change the trading symbol. For continuous 1-year historical
# time-series analysis without missing dates, map historical tickers to current tickers:
# - GE T&D India Ltd. (GET&D) -> GE Vernova T&D India Ltd. (GVT&D) [effective late 2024]
# - ITD Cementation India Ltd. (ITDCEM) -> Cemindia Projects Ltd. (CEMPRO) [effective 17-Sep-2025]
SYMBOL_ALIASES = {
    "GET&D": "GVT&D",
    "ITDCEM": "CEMPRO",
}

# -----------------------------------------------------------------------------
# BENCHMARK CONFIGURATION
# -----------------------------------------------------------------------------

# Benchmark 1: Price Return Index via yfinance
# ^CRSLDX is the official Yahoo Finance ticker for the Nifty 500 Index
BENCHMARK_PRICE_TICKER = "^CRSLDX"

# Benchmark 2: Total Return Index (TRI) via local manual CSV
# niftyindices.com automated scraping is blocked/fragile; local CSV ingestion is required
DEFAULT_TRI_CSV_PATH = DATA_DIR / "nifty500_tri.csv"

# -----------------------------------------------------------------------------
# NSE BHAVCOPY ENDPOINTS & NETWORK CONFIGURATION
# -----------------------------------------------------------------------------

NSE_HOME_URL = "https://www.nseindia.com/"
NSE_BHAVCOPY_URL_TEMPLATE = (
    "https://www.nseindia.com/api/reports?archives=%5B%7B%22name%22%3A%22Full%20Bhavcopy"
    "%20and%20Security%20Deliverable%20data%22%2C%22type%22%3A%22daily-reports%22%2C"
    "%22category%22%3A%22capital-market%22%2C%22section%22%3A%22equities%22%7D%5D"
    "&date={date_str}&type=equities&mode=single"
)

# Mandatory real browser headers to bypass NSE Cloudflare / Akamai bot detection
NSE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports",
    "Connection": "keep-alive",
}

# Rate limiting delay between successive daily bhavcopy requests (seconds)
REQUEST_DELAY_SECONDS = 0.5

# Standard expected columns in official NSE Full Bhavcopy CSV
BHAVCOPY_EXPECTED_COLUMNS = [
    "SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE",
    "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
    "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"
]

# Output files
SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "portfolio_technical_summary.csv"
HISTORICAL_OHLCV_CSV = OUTPUT_DIR / "portfolio_historical_ohlcv.csv"
