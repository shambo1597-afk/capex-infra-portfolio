"""
Configuration Module for Indian Equity Portfolio Pipeline & Analysis Toolkit.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module centralizes all portfolio universe configurations, benchmark settings,
NSE endpoint specifications, header requirements, and directory paths.
"""

from datetime import date
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

# Capital Goods / EPC Candidates (Screened candidate universe - 15 stocks)
# The 13 official Nifty Capital Goods index constituents that passed the fundamental screen,
# plus CEMPRO and SCHNEIDER: not index members, but they passed the same screen on their own
# merits (index membership is a methodological preference, not a requirement).
CAPITAL_GOODS_EPC_STOCKS = [
    "ABB", "CGPOWER", "GVT&D", "POWERINDIA", "TRITURBINE", "TDPOWERSYS",
    "SIEMENS", "BHEL", "INOXWIND", "ENRIN", "SUZLON", "THERMAX", "VOLTAMP",
    "CEMPRO",     # Cemindia Projects Ltd. (formerly ITD Cementation; see SYMBOL_ALIASES)
    "SCHNEIDER",  # Schneider Electric Infrastructure Ltd.
]

# Power Sector Stocks (Screened candidate universe - 11 stocks)
# Official Nifty Power index candidates that passed the fundamental screen
POWER_SECTOR_STOCKS = [
    "ADANIENSOL", "ADANIPOWER", "CESC", "KPIGREEN", "NAVA", "NLCINDIA",
    "NTPC", "POWERGRID", "PTC", "TATAPOWER", "TORNTPOWER",
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
# niftyindices.com automated retrieval is unreliable (Akamai bot protection); the local CSV is the default
DEFAULT_TRI_CSV_PATH = DATA_DIR / "nifty500_tri.csv"

# The manual TRI CSV is flagged stale when its last date trails the analysis end date
# by more than this many trading days (weekdays; exchange holidays are not modelled)
TRI_STALE_THRESHOLD_TRADING_DAYS = 3

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

# Official daily closing values of all NSE indices (source for the Nifty 500 benchmark series)
NSE_INDEX_CLOSE_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"
BENCHMARK_INDEX_NAME = "Nifty 500"

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

# NSE's bot protection intermittently answers 403 to valid trading days; such requests are
# retried (with backoff and a fresh cookie warm-up) rather than treated as holidays
BHAVCOPY_FETCH_ATTEMPTS = 4
BHAVCOPY_RETRY_BACKOFF_SECONDS = 3

# NSE normally trades Monday-Friday, so weekends are not requested. Exceptional weekend
# sessions (e.g. Union Budget day) must be listed here to be included in the history.
NSE_SPECIAL_WEEKEND_SESSIONS = frozenset({
    date(2025, 2, 1),  # Saturday: Union Budget special session
    date(2026, 2, 1),  # Sunday: Union Budget special session
})

# Standard expected columns in official NSE Full Bhavcopy CSV
BHAVCOPY_EXPECTED_COLUMNS = [
    "SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE",
    "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
    "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"
]

# Output files
SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "portfolio_technical_summary.csv"
HISTORICAL_OHLCV_CSV = OUTPUT_DIR / "portfolio_historical_ohlcv.csv"
RISK_SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "portfolio_risk_summary.csv"
FUNDAMENTALS_SCREEN_OUTPUT_CSV = OUTPUT_DIR / "fundamentals_screen_check.csv"

# Capital Goods / EPC fundamental screen (Screener.in basis): (field, comparison, threshold, label)
CAPITAL_GOODS_SCREEN_CRITERIA = [
    ("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"),
    ("sales_growth_3yr", ">", 8, "Sales growth 3Years > 8%"),
    ("profit_growth_3yr", ">", 8, "Profit growth 3Years > 8%"),
    ("roce_3yr_avg", ">", 13, "Average ROCE 3Years > 13%"),
    ("opm", ">", 9, "OPM > 9%"),
    ("operating_cash_flow_3yr", ">", 0, "Operating cash flow 3years > 0 (Rs Cr)"),
    ("debt_to_equity", "<", 1.2, "Debt to equity < 1.2"),
]

# Cement fundamental safety screen. No price/DMA condition: that is a technical filter and
# belongs to the technical screen, which runs first.
CEMENT_SCREEN_CRITERIA = [
    ("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"),
    ("roce", ">", 11, "ROCE > 11%"),
    ("roce_3yr_avg", ">", 10, "Average ROCE 3Years > 10%"),
    ("opm", ">", 13, "OPM > 13%"),
    ("debt_to_equity", "<", 1, "Debt to equity < 1"),
    ("operating_cash_flow", ">", 0, "Cash from operations last year > 0 (Rs Cr)"),
    ("sales_growth_3yr", ">", 6, "Sales growth 3Years > 6%"),
    ("profit_growth_3yr", ">", 5, "Profit growth 3Years > 5%"),
]

# Power fundamental safety screen. No sales/profit growth filters: under regulated cost-plus
# tariffs, revenue can fall when pass-through input costs fall while profitability holds, so
# growth is a misleading signal for this sector. ROCE floors are lowered for the regulated,
# debt-heavy (normative 70:30) capital structure. Interest coverage is EBIT / interest for the
# latest full financial year (fundamentals.parse_interest_coverage).
POWER_SCREEN_CRITERIA = [
    ("market_cap", ">", 2000, "Market Cap > 2000 (Rs Cr)"),
    ("roce", ">", 7, "ROCE > 7%"),
    ("roce_3yr_avg", ">", 7, "Average ROCE 3Years > 7%"),
    ("interest_coverage", ">", 2, "Interest coverage > 2"),
    ("operating_cash_flow", ">", 0, "Cash from operations last year > 0 (Rs Cr)"),
]

# -----------------------------------------------------------------------------
# SECTOR SCREEN (technical screen FIRST, then fundamental safety screen; see sector_screen.py)
# -----------------------------------------------------------------------------

# Official Nifty sector index constituent files, downloaded from niftyindices.com
# (https://www.niftyindices.com/IndexConstituent/ind_nifty<Name>_list.csv) on 24-Sep-2026.
INDEX_CONSTITUENTS_DIR = DATA_DIR / "index_constituents"

SECTOR_SCREENS = {
    "Cement": {
        "constituents_csv": INDEX_CONSTITUENTS_DIR / "ind_niftyCement_list.csv",
        "criteria": CEMENT_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "cement_full_screen.csv",
    },
    "Capital Goods": {
        "constituents_csv": INDEX_CONSTITUENTS_DIR / "ind_niftyCapitalGoods_list.csv",
        "criteria": CAPITAL_GOODS_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "capital_goods_full_screen.csv",
    },
    "Power": {
        "constituents_csv": INDEX_CONSTITUENTS_DIR / "ind_niftyPower_list.csv",
        "criteria": POWER_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "power_full_screen.csv",
    },
}

# Technical screen: passes when RS vs Nifty 500 over TECHNICAL_RS_LOOKBACK_DAYS is > 0 AND the
# trend direction (+DI vs -DI) is Bullish. ADX is reported (tiebreaker) but not a cutoff.
TECHNICAL_RS_LOOKBACK_DAYS = 63

# -----------------------------------------------------------------------------
# RISK, STOP-LOSS & SIZING ASSUMPTIONS (see stoploss.py)
# These are stated, adjustable modelling choices, not universally "correct" values.
# -----------------------------------------------------------------------------

# Trading sessions per year, used to annualize daily volatility and mean returns
TRADING_DAYS_PER_YEAR = 252

# Lookback for volatility / expected return: the trailing ~1 year of daily returns
RISK_LOOKBACK_TRADING_DAYS = 252

# k: volatility multiplier for the volatility-capped stop. 1.75 daily-sigma-scaled
# moves leaves a reasonable cushion against ordinary noise without letting a loss run;
# raise it for a looser stop, lower it for a tighter one.
STOP_LOSS_VOL_MULTIPLIER = 1.75

# N: holding-period scaling in trading days (daily sigma is scaled by sqrt(N)).
# 21 ~ one trading month: a middle ground for the 3-month tactical mandate, where stops
# are reviewed and potentially tightened monthly rather than held static for the quarter.
STOP_LOSS_HOLDING_PERIOD_DAYS = 21
