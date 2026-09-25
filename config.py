"""
Configuration Module for Indian Equity Portfolio Pipeline & Analysis Toolkit.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module centralizes all portfolio universe configurations, benchmark settings,
NSE endpoint specifications, header requirements, and directory paths.
"""

import csv
from datetime import date
from pathlib import Path
from typing import Dict, List

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

# Official Nifty sector index constituent files, downloaded from niftyindices.com
# (https://www.niftyindices.com/IndexConstituent/ind_nifty<Name>_list.csv) on 24-Sep-2026.
# These files, plus the named Nifty Infrastructure additions below, are the single source of
# truth for the three sector universes.
INDEX_CONSTITUENTS_DIR = DATA_DIR / "index_constituents"
SECTOR_CONSTITUENT_FILES = {
    "Cement": INDEX_CONSTITUENTS_DIR / "ind_niftyCement_list.csv",
    "Capital Goods": INDEX_CONSTITUENTS_DIR / "ind_niftyCapitalGoods_list.csv",
    "Power": INDEX_CONSTITUENTS_DIR / "ind_niftyPower_list.csv",
}


NIFTY_INFRA_CONSTITUENTS_FILE = INDEX_CONSTITUENTS_DIR / "ind_niftyinfralist.csv"  # downloaded 25-Sep-2026


def _read_constituents(csv_path: Path, index_name: str) -> List[Dict[str, str]]:
    """Rows (symbol, name, index) of an official constituent file; empty if the file is missing."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return [{"symbol": row["Symbol"].strip(), "name": row["Company Name"].strip(), "index": index_name}
                for row in csv.DictReader(f) if row.get("Symbol", "").strip()]


_CONSTITUENTS = {sector: _read_constituents(path, f"Nifty {sector}")
                 for sector, path in SECTOR_CONSTITUENT_FILES.items()}

# Nifty Infrastructure constituents added to a sector universe. The full index is NOT adopted
# (it spans ports, aviation, oil & gas, telecom, healthcare, realty and hotels); of its 30
# constituents, 12 are already in the three sector universes and only these two others have a
# primary business in Cement, Capital Goods/EPC or Power. Each is screened with that sector's
# thresholds and ranked against its universe; see BUSINESS_FOCUS_NOTES before selecting one.
NIFTY_INFRA_SECTOR_ADDITIONS = {
    "Capital Goods": ["LT", "BHARATFORG"],
}

# Business-focus review notes (conglomerate / classification concerns), shown in the review
# tables' business_focus_note column. They are flags for a manual decision, not exclusions.
BUSINESS_FOCUS_NOTES = {
    "LT": (
        "CONGLOMERATE CONCERN - manual review. NSE industry: Construction. Core business is EPC "
        "(infrastructure and energy projects, hi-tech manufacturing), but consolidated revenue also "
        "includes IT & technology services (listed subsidiaries LTIMindtree, L&T Technology Services) "
        "and financial services (L&T Finance). Verify the latest segment split in the annual report."
    ),
    "BHARATFORG": (
        "CLASSIFICATION CONCERN - manual review. NSE industry: Automobile and Auto Components, not "
        "Capital Goods. Forgings serve automotive (commercial and passenger vehicles, including overseas "
        "auto subsidiaries) as well as industrial, defence and aerospace customers. Verify the auto vs "
        "non-auto revenue split before treating it as a capital-goods name."
    ),
    "GRASIM": (
        "CONGLOMERATE CONCERN (same treatment as NAVA). Consolidated results include UltraTech (cement) "
        "but also VSF and chemicals, paints, B2B e-commerce and financial services (Aditya Birla Capital)."
    ),
}


def _add_infra_constituents() -> None:
    infra = {r["symbol"]: r for r in _read_constituents(NIFTY_INFRA_CONSTITUENTS_FILE, "Nifty Infrastructure")}
    for sector, symbols in NIFTY_INFRA_SECTOR_ADDITIONS.items():
        present = {r["symbol"] for rows in _CONSTITUENTS.values() for r in rows}
        for symbol in symbols:
            if symbol in infra and symbol not in present:
                _CONSTITUENTS[sector].append(infra[symbol])


_add_infra_constituents()

CEMENT_STOCKS = [r["symbol"] for r in _CONSTITUENTS["Cement"]]
CAPITAL_GOODS_EPC_STOCKS = [r["symbol"] for r in _CONSTITUENTS["Capital Goods"]]
POWER_SECTOR_STOCKS = [r["symbol"] for r in _CONSTITUENTS["Power"]]


def sector_universe(sector: str) -> List[Dict[str, str]]:
    """Every stock screened for a sector: its official index constituents plus any Nifty
    Infrastructure additions, as rows {symbol, name, index}."""
    return list(_CONSTITUENTS.get(sector, []))


# Symbol -> sector / company name lookups shared by every module
SYMBOL_SECTOR = {r["symbol"]: sector for sector, rows in _CONSTITUENTS.items() for r in rows}
SYMBOL_NAME = {r["symbol"]: r["name"] for rows in _CONSTITUENTS.values() for r in rows}


def sector_of(symbol: str) -> str:
    """Sector of a symbol in the three official universes, or "Other"."""
    return SYMBOL_SECTOR.get(symbol, "Other")


# -----------------------------------------------------------------------------
# LOCKED PORTFOLIO
# The final 8 picks, chosen from the technical and fundamental screens (sector_screen.py,
# output/*_full_review_table.csv) and the RRG analysis (rrg.py). Not every pick passes every
# screen: JKCEMENT and TATAPOWER have RS vs Nifty 500 below +2 pp, and APLAPOLLO fails only
# the OPM criterion (flagged high-turnover, held after manual review). The dashboard lists
# these exceptions from the review tables. Names and sectors come from the constituent files.
# -----------------------------------------------------------------------------

LOCKED_PORTFOLIO_SYMBOLS = [
    "JKCEMENT",                                                    # Cement
    "VOLTAMP", "FINCABLES", "APARINDS", "WELCORP", "APLAPOLLO",    # Capital Goods
    "ACMESOLAR", "TATAPOWER",                                      # Power
]

LOCKED_PORTFOLIO = {
    symbol: {"sector": sector_of(symbol), "name": SYMBOL_NAME.get(symbol, symbol)}
    for symbol in LOCKED_PORTFOLIO_SYMBOLS
}

# Complete watchlist to fetch and analyze: every constituent of the three sector indices
PORTFOLIO_SYMBOLS = CEMENT_STOCKS + CAPITAL_GOODS_EPC_STOCKS + POWER_SECTOR_STOCKS

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
# Promoter pledge disclosures (JSON); the page URL seeds the Akamai cookies the API requires
NSE_PLEDGE_PAGE_URL = "https://www.nseindia.com/companies-listing/corporate-filings-pledged-data"
NSE_PLEDGE_API_URL = "https://www.nseindia.com/api/corporate-pledgedata?index=equities&symbol={symbol}"
# Board-meeting disclosures (upcoming results dates are announced here in advance)
NSE_BOARD_MEETINGS_API_URL = "https://www.nseindia.com/api/corporate-board-meetings?index=equities&symbol={symbol}"
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

# -----------------------------------------------------------------------------
# FUNDAMENTAL SAFETY SCREENS: (field, comparison, threshold, label)
#
# These are deliberately LIGHTER than a long-term investment screen. The mandate is a 3-month
# tactical portfolio with a hard freeze date (and a known mid-October earnings catalyst inside
# the evaluation window), so technicals (RS vs Nifty 500, trend, ADX) drive selection and
# fundamentals only answer "is this company solvent and safe to hold for a quarter?". Criteria
# are therefore current-year only: no 3-year averages of ROCE and no 3-year sales/profit growth,
# which measure multi-year durability and are miscalibrated for a one-quarter holding period.
# Promoter pledge is a hard criterion in every sector: heavily pledged promoter stakes can force
# selling (margin calls) regardless of fundamentals, a real risk over any horizon.
#   roce: latest-year ROCE (Screener headline); opm: latest column (TTM where shown), exact;
#   operating_cash_flow: last financial year; debt_to_equity: latest balance sheet;
#   interest_coverage: EBIT / interest, latest full year; pledged_pct: % of promoter holding
#   pledged, from NSE's pledge disclosures (fundamentals.fetch_pledged_percentage).
# -----------------------------------------------------------------------------

CEMENT_SCREEN_CRITERIA = [
    ("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"),
    ("roce", ">", 8, "ROCE > 8%"),
    ("opm", ">", 10, "OPM > 10%"),
    ("operating_cash_flow", ">", 0, "Cash from operations last year > 0 (Rs Cr)"),
    ("debt_to_equity", "<", 1.5, "Debt to equity < 1.5"),
    ("pledged_pct", "<", 15, "Pledged percentage < 15%"),
]

CAPITAL_GOODS_SCREEN_CRITERIA = [
    ("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"),
    ("roce", ">", 8, "ROCE > 8%"),
    ("opm", ">", 8, "OPM > 8%"),
    ("operating_cash_flow", ">", 0, "Cash from operations last year > 0 (Rs Cr)"),
    ("debt_to_equity", "<", 1.5, "Debt to equity < 1.5"),
    ("pledged_pct", "<", 15, "Pledged percentage < 15%"),
]

# Power: no sales/profit growth filter (under regulated cost-plus tariffs, revenue can fall when
# pass-through input costs fall while profitability holds, so growth misleads for this sector),
# and interest coverage instead of a flat debt/equity cap (the regulated capital structure is
# normatively debt-heavy, ~70:30, so leverage is structural; the ability to service it is what
# matters). The ROCE floor is lower for the same regulated, leveraged capital structure.
POWER_SCREEN_CRITERIA = [
    ("market_cap", ">", 2000, "Market Cap > 2000 (Rs Cr)"),
    ("roce", ">", 6, "ROCE > 6%"),
    ("interest_coverage", ">", 1.5, "Interest coverage > 1.5"),
    ("operating_cash_flow", ">", 0, "Cash from operations last year > 0 (Rs Cr)"),
    ("pledged_pct", "<", 15, "Pledged percentage < 15%"),
]

# -----------------------------------------------------------------------------
# SECTOR SCREEN (technical screen FIRST, then fundamental safety screen; see sector_screen.py)
# -----------------------------------------------------------------------------

SECTOR_SCREENS = {
    "Cement": {
        "constituents_csv": SECTOR_CONSTITUENT_FILES["Cement"],
        "criteria": CEMENT_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "cement_full_screen.csv",
    },
    "Capital Goods": {
        "constituents_csv": SECTOR_CONSTITUENT_FILES["Capital Goods"],
        "criteria": CAPITAL_GOODS_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "capital_goods_full_screen.csv",
    },
    "Power": {
        "constituents_csv": SECTOR_CONSTITUENT_FILES["Power"],
        "criteria": POWER_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "power_full_screen.csv",
    },
}

# Technical screen ("technically attractive"): passes when RS vs Nifty 500 over
# TECHNICAL_RS_LOOKBACK_DAYS exceeds TECHNICAL_RS_MARGIN_PP AND the trend direction (+DI vs -DI)
# is Bullish. ADX is reported (tiebreaker) but not a cutoff.
TECHNICAL_RS_LOOKBACK_DAYS = 63

# Run-up heuristic window: share of the 63-session RS earned in the last this-many sessions
RUNUP_RECENT_DAYS = 10

# Minimum RS vs Nifty 500 (percentage points) to count as outperforming. A bare zero threshold is
# too sensitive to single-day noise: RS is a 63-day cumulative spread, and one new trading day's
# return alone can shift it meaningfully (moving the window end by one session changed RS by up
# to +/-11 pp and flipped several stocks' pass/fail). A +2 pp margin requires the outperformance
# to be more than marginal before a stock is called technically attractive. Applies only to this
# absolute pass/fail test, not to rs_score_vs_sector_avg / sector_rank (a relative ranking).
TECHNICAL_RS_MARGIN_PP = 2.0

# -----------------------------------------------------------------------------
# RELATIVE ROTATION GRAPH & TREND-QUALITY CHECKS (see rrg.py)
# -----------------------------------------------------------------------------

# RS-Momentum = 63-session RS today minus the 63-session RS this many sessions ago (pp)
RRG_MOMENTUM_DAYS = 10

# |+DI - -DI| below this is a thin, borderline trend whichever way it points: the direction
# label can flip on a single bar (e.g. JKCEMENT +DI 19.50 vs -DI 18.95)
DI_GAP_THIN_THRESHOLD = 2.0

# OPM-exception review flag: a stock failing ONLY the OPM criterion with ROCE above this may be
# a high-turnover, low-margin business (e.g. a steel-tube fabricator) for which OPM is the
# wrong yardstick. Flag for a manual business-model check; never an automatic pass.
HIGH_TURNOVER_ROCE_MIN = 20.0

# -----------------------------------------------------------------------------
# RISK, STOP-LOSS & SIZING ASSUMPTIONS (see stoploss.py)
# These are stated, adjustable modelling choices, not universally "correct" values.
# -----------------------------------------------------------------------------

# Trading sessions per year, used to annualize daily volatility and mean returns
TRADING_DAYS_PER_YEAR = 252

# Lookback for volatility / expected return: the trailing ~1 year of daily returns
RISK_LOOKBACK_TRADING_DAYS = 252

# Stop-loss for the 3-month mandate (see stoploss.py). The stop is sized for about one month
# and trailed up at each monthly review, rather than sized for the full quarter: a 63-session
# volatility stop sits ~19-41% below price for these stocks, a larger loss than a 3-month
# tactical trade is expected to earn.
# ATR period (Wilder smoothing of the true range, in sessions)
STOP_LOSS_ATR_PERIOD = 14
# Base stop = price - 3 x ATR(14). 3 ATR roughly equals a one-month, one-standard-deviation
# move (e.g. JKCEMENT: 3 ATR = 8.5% vs sigma_annual x sqrt(21/252) = 9.4%), so the stop sits
# just outside ordinary noise. Raise it for a looser stop, lower it for a tighter one.
STOP_LOSS_ATR_MULTIPLE = 3.0
# Support adjustment: a clear support level lying up to this many ATRs beyond the base stop
# pulls the stop down to just below that support (so the stop is not parked just above it)...
STOP_LOSS_SUPPORT_BAND_ATR = 1.0
# ...placed this many ATRs under the support level
STOP_LOSS_SUPPORT_BUFFER_ATR = 0.25
