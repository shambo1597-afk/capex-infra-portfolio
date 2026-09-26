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

# SINGLE SOURCE OF TRUTH: Screener.in sector exports (data/screener/, downloaded 26-Sep-2026, see
# data/screener/SOURCE.md). They define the universe AND supply the fundamentals (market cap, ROCE,
# OPM, debt/equity, interest cover, pledge %, operating cash flow), so no other list or scrape is used.
# Universe rule (agreed with the group before any results were seen):
#   - NSE-listed (an NSE code) with market cap >= UNIVERSE_MIN_MARKET_CAP_CR
#   - Capital Goods without the Aerospace & Defense group and without the industries in
#     CAPITAL_GOODS_EXCLUDED_INDUSTRIES (not capex/infrastructure products)
#   - no infrastructure investment trusts (NON_EQUITY_INSTRUMENTS: units, not company shares)
#   - no company whose main business is outside the theme (THEME_EXCLUSIONS, one stated rule
#     applied to the whole universe, with the reason for each name)
SCREENER_DIR = DATA_DIR / "screener"
SCREENER_DOWNLOAD_DATE = "2026-09-26"
SCREENER_FILES = {
    "Cement": SCREENER_DIR / "cement.csv",
    "Capital Goods": SCREENER_DIR / "capital_goods.csv",
    "Power": SCREENER_DIR / "power.csv",
}
UNIVERSE_MIN_MARKET_CAP_CR = 5000
CAPITAL_GOODS_EXCLUDED_GROUPS = {"Aerospace & Defense"}
CAPITAL_GOODS_EXCLUDED_INDUSTRIES = {
    "Packaging", "Rubber", "Glass - Industrial", "Aluminium, Copper & Zinc Products",
    "Commercial Vehicles", "Tractors", "Dealers-Commercial Vehicles, Tractors, Construction Vehicles",
}
NON_EQUITY_INSTRUMENTS = {
    "INDIGRID": "Infrastructure investment trust (InvIT units, not company shares)",
    "PGINVIT": "Infrastructure investment trust (InvIT units, not company shares)",
}
# Theme rule: exclude companies whose main business is (1) electronics manufacturing or consumer
# electronics, (2) automotive or consumer components, (3) defence or shipbuilding, (4) primary steel
# making (a commodity metal, not equipment), or (5) packaging and films.
THEME_EXCLUSIONS = {
    "CPPLUS": "(1) CCTV and security electronics",
    "SYRMA": "(1) electronics manufacturing services",
    "KAYNES": "(1) electronics manufacturing services",
    "AVALON": "(1) electronics manufacturing services",
    "INDOMIM": "(2) metal-injection-moulded parts, mainly automotive and consumer",
    "SPECTRUM": "(2) components for appliances and vehicles",
    "KINGFA": "(2) engineering plastics for automotive and appliances",
    "RAYMOND": "(2) engineering business after the demergers is mainly automotive and aerospace components",
    "HAPPYFORGE": "(2) forgings, mainly automotive",
    "MARINE": "(3) electrical systems for ships and naval vessels",
    "MAZDOCK": "(3) defence shipbuilding",
    "COCHINSHIP": "(3) defence and commercial shipbuilding",
    "SWANDEF": "(3) defence shipbuilding",
    "SHYAMMETL": "(4) primary steel and metals",
    "GPIL": "(4) primary steel and iron ore",
    "GALLANTT": "(4) primary steel",
    "JAYNECOIND": "(4) primary steel and castings",
    "SUNFLAG": "(4) alloy steel, mainly for automotive",
    "GRWRHITECH": "(5) plastic films",
    "TIMETECHNO": "(5) packaging and polymer products",
}


def _read_screener(sector: str) -> List[Dict[str, object]]:
    """Universe rows {symbol, name, index, industry, record} of one Screener export after the universe rule."""
    path = SCREENER_FILES[sector]
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            symbol = (row.get("NSE Code") or "").strip()
            try:
                mcap = float(row.get("Market Capitalization") or "nan")
            except ValueError:
                mcap = float("nan")
            if not symbol or not mcap >= UNIVERSE_MIN_MARKET_CAP_CR:
                continue
            if sector == "Capital Goods" and (row["Industry Group"] in CAPITAL_GOODS_EXCLUDED_GROUPS
                                              or row["Industry"] in CAPITAL_GOODS_EXCLUDED_INDUSTRIES):
                continue
            if symbol in NON_EQUITY_INSTRUMENTS or symbol in THEME_EXCLUSIONS:
                continue
            rows.append({"symbol": symbol, "name": row["Name"].strip(), "index": f"Screener: {row['Industry']}",
                         "industry": row["Industry"], "record": row})
    return rows


_CONSTITUENTS = {sector: _read_screener(sector) for sector in SCREENER_FILES}

# Screener row per symbol: the fundamentals source (fundamentals.extract_stock_fundamentals)
SCREENER_RECORDS = {r["symbol"]: r["record"] for rows in _CONSTITUENTS.values() for r in rows}

# Business-focus notes: why a borderline business was kept in the theme (shown in the review tables)
BUSINESS_FOCUS_NOTES = {
    "GREAVESCOT": (
        "KEPT (borderline): engines, gensets and farm equipment are most of revenue; it also owns an "
        "electric two-wheeler business (Greaves Electric Mobility). Verify the segment split."
    ),
    "UTLSOLAR": "KEPT: solar power equipment (panels, inverters, batteries), part of the power build-out.",
    "RPEL": "KEPT: refractory ramming mass used inside steel plants' induction furnaces (an industrial consumable).",
    "GRASIM": (
        "CONGLOMERATE CONCERN. Consolidated results include UltraTech (cement) but also VSF and chemicals, "
        "paints, B2B e-commerce and financial services (Aditya Birla Capital)."
    ),
    "NAVA": "CONGLOMERATE CONCERN: power generation plus ferro alloys and mining (Zambia).",
    "RRKABEL": "Wires and cables are most of revenue, with a smaller consumer-electricals segment.",
}

CEMENT_STOCKS = [r["symbol"] for r in _CONSTITUENTS["Cement"]]
CAPITAL_GOODS_EPC_STOCKS = [r["symbol"] for r in _CONSTITUENTS["Capital Goods"]]
POWER_SECTOR_STOCKS = [r["symbol"] for r in _CONSTITUENTS["Power"]]


def sector_universe(sector: str) -> List[Dict[str, str]]:
    """Every stock screened for a sector (the Screener export after the universe rule), as rows
    {symbol, name, index, industry}."""
    return [{k: r[k] for k in ("symbol", "name", "index", "industry")} for r in _CONSTITUENTS.get(sector, [])]


# Symbol -> sector / company name lookups shared by every module
SYMBOL_SECTOR = {r["symbol"]: sector for sector, rows in _CONSTITUENTS.items() for r in rows}
SYMBOL_NAME = {r["symbol"]: r["name"] for rows in _CONSTITUENTS.values() for r in rows}


def sector_of(symbol: str) -> str:
    """Sector of a symbol in the three official universes, or "Other"."""
    return SYMBOL_SECTOR.get(symbol, "Other")


# -----------------------------------------------------------------------------
# LOCKED PORTFOLIO: 15 tracked stocks, money in the top 8, the other 7 in reserve
# The 15 picks (decided with the group 26-Sep-2026), exactly what the selection rule in
# sector_screen.py gives (select_portfolio; output/selection_ranking.csv, column rule_pick),
# listed in rank order:
#   eligible  = in the Screener universe (theme rule above), HARD fundamental rules pass, bullish
#               trend with a real DI gap (>= 2), at least MIN_HISTORY_SESSIONS of prices;
#   ranked    by 6-month relative strength vs the Nifty 500 excluding the latest month (momentum;
#               research/momentum_study.py found a small positive but statistically inconclusive
#               effect, so this is a stated method, not a proven edge);
#   confirmed by the RRG: conviction High or Moderate (rrg.conviction_tier: not weakening/lagging in
#               both views vs the Nifty 500 and vs the sector), so names whose momentum is visibly
#               fading in both views are skipped (a judgement filter, not a back-tested one);
#   the top PORTFOLIO_SIZE confirmed names.
# The list of 15 is frozen after 5-Oct-2026; all 15 are tracked. The money goes into the top
# INVESTED_COUNT (INITIAL_HOLDINGS); the rest are the reserve, in rank order (RESERVE_SYMBOLS).
# When a holding closes at or below its stop-loss, it is sold and its proceeds buy the first reserve
# stock that still passes the selection rule that day (tracker.plan_replacements); a stock that has
# been sold never comes back. No sector minimums: Power has one stock (ACMESOLAR) and Cement none
# (its best confirmed uptrend, NUVOCO, ranks #32); the group will ask the professor whether a Cement
# leg is required. Soft exceptions (shown on the dashboard): BEML (ROCE/OPM), GREAVESCOT (OPM),
# BANSALWIRE (OPM). UTLSOLAR (211 sessions of prices) fails the one-year history rule and was
# replaced by the rule's pick SBCL (group decision 26-Sep-2026).
# -----------------------------------------------------------------------------

LOCKED_PORTFOLIO_SYMBOLS = [
    "WELCORP", "RPEL", "SBCL", "ACMESOLAR", "FINCABLES", "GREAVESCOT", "ACE", "CARBORUNIV",  # invested
    "GOODLUCK", "BEML", "BANSALWIRE", "TEXRAIL", "SHANTIGEAR", "USHAMART", "AJAXENGG",       # reserve
]

PORTFOLIO_SIZE = 15                          # the brief's maximum; all 15 are tracked
INVESTED_COUNT = 8                           # the brief's minimum holds the money
INITIAL_HOLDINGS = LOCKED_PORTFOLIO_SYMBOLS[:INVESTED_COUNT]
RESERVE_SYMBOLS = LOCKED_PORTFOLIO_SYMBOLS[INVESTED_COUNT:]  # replacement queue, in rank order
SECTOR_MIN_HOLDINGS: Dict[str, int] = {}     # none: picked on merit (the group may add a Cement leg)
SELECTION_CONVICTION_TIERS = ("High", "Moderate")  # RRG confirmation of the momentum ranking
MIN_HISTORY_SESSIONS = 240                   # about a year of NSE sessions, so 6-month RS is measurable

LOCKED_PORTFOLIO = {
    symbol: {"sector": sector_of(symbol), "name": SYMBOL_NAME.get(symbol, symbol)}
    for symbol in LOCKED_PORTFOLIO_SYMBOLS
}

# Complete watchlist to fetch and analyze: the whole universe
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
    ("market_cap", ">=", 5000, "Market Cap >= 5000 (Rs Cr)"),
    ("roce", ">", 8, "ROCE > 8%"),
    ("opm", ">", 10, "OPM > 10%"),
    ("operating_cash_flow", ">", 0, "Cash from operations last year > 0 (Rs Cr)"),
    ("debt_to_equity", "<", 1.5, "Debt to equity < 1.5"),
    ("pledged_pct", "<", 15, "Pledged percentage < 15%"),
]

CAPITAL_GOODS_SCREEN_CRITERIA = [
    ("market_cap", ">=", 5000, "Market Cap >= 5000 (Rs Cr)"),
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
# HARD vs SOFT criteria for a 3-month holding period. HARD failures can turn a bad quarterly result
# into a crash (pledged shares face margin calls and forced selling; heavy debt or thin interest
# cover leaves no cushion; tiny companies are illiquid): never held. SOFT failures (ROCE, OPM,
# one year's operating cash flow) describe business quality over years, are already in the price,
# and matter little over 3 months: acceptable exceptions, shown on the dashboard.
FUNDAMENTAL_HARD_FIELDS = {"pledged_pct", "debt_to_equity", "interest_coverage", "market_cap"}

POWER_SCREEN_CRITERIA = [
    ("market_cap", ">=", 5000, "Market Cap >= 5000 (Rs Cr)"),
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
        "source_csv": SCREENER_FILES["Cement"],
        "criteria": CEMENT_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "cement_full_screen.csv",
    },
    "Capital Goods": {
        "source_csv": SCREENER_FILES["Capital Goods"],
        "criteria": CAPITAL_GOODS_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "capital_goods_full_screen.csv",
    },
    "Power": {
        "source_csv": SCREENER_FILES["Power"],
        "criteria": POWER_SCREEN_CRITERIA,
        "output_csv": OUTPUT_DIR / "power_full_screen.csv",
    },
}

# Technical screen ("technically attractive"): passes when RS vs Nifty 500 over
# TECHNICAL_RS_LOOKBACK_DAYS exceeds TECHNICAL_RS_MARGIN_PP AND the trend direction (+DI vs -DI)
# is Bullish. ADX is reported (tiebreaker) but not a cutoff.
TECHNICAL_RS_LOOKBACK_DAYS = 63

# Selection ranking: RS vs Nifty 500 over SELECTION_RS_DAYS sessions ending SELECTION_SKIP_DAYS
# sessions ago (about 6 months, skipping the latest month), per research/momentum_study.py
SELECTION_RS_DAYS = 105
SELECTION_SKIP_DAYS = 21

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

# RS-Momentum = 63-session RS now minus the 63-session RS this many sessions earlier (pp)...
RRG_MOMENTUM_DAYS = 10
# ...with the RS averaged over this many sessions at each end. Unsmoothed (1), one session's
# move flipped a stock's RRG tier on ~17% of days (WELCORP and FINCABLES fell from High to Low
# on 25-Sep-2026 and back); a 5-session average halves the flip rate while staying responsive.
RRG_MOMENTUM_SMOOTHING_DAYS = 5
# RRG tails (rrg_tails.py): weekly points over the past 8 weeks, and the NSE sector indices shown
# with our sub-themes on the sector-rotation RRG
RRG_TAIL_WEEKS = 8        # weekly points kept (dashboard slider range)
RRG_PLOT_TAIL_WEEKS = 4   # tail length on the static PNGs (longer tails tangle)
RRG_SECTOR_INDICES = ["Nifty Auto", "Nifty Bank", "Nifty IT", "Nifty Metal", "Nifty Pharma", "Nifty FMCG",
                      "Nifty Realty", "Nifty Energy", "Nifty Infrastructure", "Nifty India Defence"]

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

PRINCIPAL_INR = 10_000_000  # Rs 1 crore

# Share of the principal put into the stocks. The remaining 3% is the hedge reserve: the day-0
# Nifty puts (~0.7%, risk_model.py) and one profit-trigger roll-up (~0.7%), plus whole-share rounding;
# it waits in a liquid ETF at the overnight rate, not idle. Market exposure stays well above 90%.
EQUITY_ALLOCATION_PCT = 97.0

# Evaluation window: first snapshot 28-Sep-2026, three months to 28-Dec-2026. tracker.py values the
# Rs 1 crore from the snapshot close using the trade ledger TRADES_CSV (editable: record real fills,
# stop-loss exits and their reserve replacements there).
EVALUATION_START_DATE = "2026-09-28"
EVALUATION_END_DATE = "2026-12-28"
TRADES_CSV = DATA_DIR / "trades.csv"

# -----------------------------------------------------------------------------
# REGRESSION & HEDGING (risk_model.py)
# Factor series are built from NSE's daily index files (plus Brent from yfinance) and committed
# as a small CSV, so the dashboard works without the raw download cache.
# -----------------------------------------------------------------------------
FACTORS_CSV = DATA_DIR / "factors" / "daily_factors.csv"
DERIVATIVES_DIR = DATA_DIR / "derivatives"          # Nifty futures/options rows of the NSE F&O bhavcopy
NSE_FO_BHAVCOPY_URL_TEMPLATE = ("https://nsearchives.nseindia.com/content/fo/"
                                "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip")
HEDGE_INDEX_NAME = "Nifty 50"                        # the hedge instrument's index (Nifty futures/options)
HEDGE_INDEX_SYMBOL = "NIFTY"
GSEC_INDEX_NAME = "Nifty 10 yr Benchmark G-Sec (Clean Price)"  # rate factor: price return ~ -duration x change in yield
RISK_FREE_INDEX_NAME = "Nifty 1D Rate Index"         # overnight money-market rate (risk-free, cash sleeve)
CRUDE_TICKER = "BZ=F"                                # Brent crude front-month futures (USD), via yfinance
TAIL_QUANTILE = 0.10          # tail beta: regression on the worst 10% of Nifty 50 days
TAIL_HEDGE_OTM_PCT = 5.0      # protective puts struck about 5% below spot: insure the tail, not every wobble
HEDGE_PROFIT_TRIGGER_PCT = 10.0  # once the portfolio is up 10%, roll the puts up to lock in part of the gain
NEAR_STOP_PCT = 3.0              # amber warning when a holding closes within this % above its stop-loss
HEDGE_MAX_ROLLS = 1              # roll up once (a team decision; raise to roll again at each further trigger)
# CAPM expected return E[r_i] = r_f + beta_i x MRP (risk_model.py): r_f = the Nifty 1D Rate index
# annualised over the last quarter; MRP = India's total equity risk premium from Damodaran's country
# risk premium table (pages.stern.nyu.edu/~adamodar, "Country Default Spreads and Risk Premiums",
# updated 5-Jan-2026: mature-market ERP plus India's country risk premium 2.85%, rating Baa3).
MARKET_RISK_PREMIUM_PCT = 7.08
MARKET_RISK_PREMIUM_SOURCE = "Damodaran, country risk premiums, India total ERP, updated 5-Jan-2026"
AUTOCORRELATION_LAGS = 5  # Ljung-Box Q over lags 1..5 (5% critical value of chi-square(5) = 11.07)
FUTURES_MARGIN_PCT_ASSUMED = 12.0  # ASSUMPTION: SPAN + exposure margin on a short index future, % of notional

# Stock weights (weights.py): equal risk contribution within the brief's bounds, from the
# covariance of the trailing ~1 year of daily returns.
WEIGHT_MIN_PCT = 5.0
WEIGHT_MAX_PCT = 15.0
WEIGHT_COV_LOOKBACK_DAYS = 252

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
