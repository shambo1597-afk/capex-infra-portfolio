"""
Fundamental Analysis & Screener.in Direct Ingestion Engine.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module provides direct web retrieval and structured parsing of corporate fundamentals
from Screener.in without requiring login credentials or uploaded CSV export files:
1. Direct HTTP fetching using authentic browser User-Agent headers and persistent sessions.
2. Local caching of company HTML pages under 'data/fundamentals_cache/' for fast re-runs.
3. Automatic fallback from consolidated (/consolidated/) to standalone (/) pages when
   consolidated pages 404 or contain empty ratio shells (e.g. single-entity firms like VOLTAMP).
4. Rigorous extraction of core academic quality & valuation criteria:
   - Market Capitalization (₹ Cr)
   - Current Market Price (₹)
   - ROCE (%) and ROE (%)
   - 3-Year Simple Average ROCE trend from the 'Ratios' table
   - Cash from Operating Activity (CFO, ₹ Cr) from the 'Cash Flows' table
   - Debt-to-Equity (Borrowings / [Equity Capital + Reserves]) from the 'Balance Sheet' table
   - Operating Profit Margin (OPM %) from the 'Profit & Loss' table
   - 3-Year Compounded Sales and Profit Growth (%) from the 'Ranges' tables

Every function contains detailed academic docstrings explaining the financial intuition,
accounting mechanics, and investment risk management applications.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    FUNDAMENTALS_CACHE_DIR,
    LOCKED_PORTFOLIO,
    LOCKED_PORTFOLIO_SYMBOLS,
    NSE_REQUEST_HEADERS,
)

logger = logging.getLogger("fundamentals")

# Screener.in URL templates
SCREENER_CONSOLIDATED_URL = "https://www.screener.in/company/{symbol}/consolidated/"
SCREENER_STANDALONE_URL = "https://www.screener.in/company/{symbol}/"


def _clean_numeric(value_str: Any) -> Optional[float]:
    """
    Clean and convert string representations of financial numbers to float.
    Removes currency symbols (₹, $), percentage signs (%), commas, and whitespace.
    """
    if value_str is None or pd.isna(value_str):
        return None
    val_clean = re.sub(r"[^\d.-]", "", str(value_str).strip())
    if not val_clean or val_clean == "-" or val_clean == ".":
        return None
    try:
        return float(val_clean)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# DIRECT HTML INGESTION & CACHING ENGINE
# -----------------------------------------------------------------------------

def fetch_screener_page(
    symbol: str,
    cache_dir: Path = FUNDAMENTALS_CACHE_DIR,
    use_cache: bool = True
) -> Optional[str]:
    """
    Fetch the complete company HTML page directly from Screener.in with local caching.

    Data Acquisition & Fallback Protocol:
    1. First attempts to load cached HTML from 'data/fundamentals_cache/{SYMBOL}.html'.
    2. If not cached, executes an HTTP GET to the consolidated endpoint:
       https://www.screener.in/company/{SYMBOL}/consolidated/
    3. Verifies whether the returned consolidated page contains populated financial ratios.
       Companies with no operating subsidiaries (e.g. Voltamp Transformers) return an empty
       ratio shell at /consolidated/. If the page 404s OR contains blank ratios, the engine
       automatically falls back to the authoritative standalone endpoint:
       https://www.screener.in/company/{SYMBOL}/
    4. Caches the valid HTML locally for instantaneous future executions.

    Parameters:
        symbol (str): Official NSE equity symbol (e.g., 'JKCEMENT', 'VOLTAMP').
        cache_dir (Path): Local directory where HTML pages are stored.
        use_cache (bool): Whether to load from cache if available.

    Returns:
        Optional[str]: Raw HTML text of the company page, or None if network query fails.
    """
    cache_path = Path(cache_dir) / f"{symbol.upper()}.html"

    # Step 1: Check local cache first
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                html = f.read()
            if len(html) > 5000:
                logger.debug("Loaded fundamental HTML for %s from local cache.", symbol)
                return html
        except Exception as exc:
            logger.warning("Error reading cache for %s (%s). Re-fetching from web.", symbol, exc)

    session = requests.Session()
    session.headers.update(NSE_REQUEST_HEADERS)

    # Step 2: Query consolidated endpoint first
    url_cons = SCREENER_CONSOLIDATED_URL.format(symbol=symbol.upper())
    logger.info("Fetching Screener.in page for %s: %s", symbol, url_cons)

    try:
        resp = session.get(url_cons, timeout=12)
    except Exception as exc:
        logger.warning("Network error fetching consolidated page for %s: %s", symbol, exc)
        resp = None

    needs_fallback = False
    if resp is None or resp.status_code != 200:
        needs_fallback = True
    else:
        # Check if consolidated page is an empty shell with no ratio values
        soup = BeautifulSoup(resp.text, "html.parser")
        top_el = soup.find("ul", id="top-ratios")
        if not top_el:
            needs_fallback = True
        else:
            numbers = [li.find("span", class_="number") for li in top_el.find_all("li")]
            has_valid_nums = any(n and n.text.strip() != "" for n in numbers)
            if not has_valid_nums:
                needs_fallback = True

    # Step 3: Fallback to standalone endpoint if consolidated is unavailable or blank
    if needs_fallback:
        url_stand = SCREENER_STANDALONE_URL.format(symbol=symbol.upper())
        logger.info("Falling back to standalone Screener page for %s: %s", symbol, url_stand)
        try:
            resp = session.get(url_stand, timeout=12)
            if resp.status_code != 200:
                logger.error("Standalone page for %s returned HTTP status %d.", symbol, resp.status_code)
                return None
        except Exception as exc:
            logger.error("Network error fetching standalone page for %s: %s", symbol, exc)
            return None

    html_content = resp.text

    # Step 4: Write to local fundamentals cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.debug("Cached fundamental HTML for %s at: %s", symbol, cache_path)
    except Exception as exc:
        logger.warning("Could not cache HTML for %s: %s", symbol, exc)

    return html_content


# -----------------------------------------------------------------------------
# PARSING ENGINES WITH FINANCIAL & ACADEMIC DOCUMENTATION
# -----------------------------------------------------------------------------

def parse_top_ratios(soup: BeautifulSoup) -> Dict[str, Optional[float]]:
    """
    Extract top-level headline ratios from the '#top-ratios' list element.

    Financial Concepts Parsed:
    1. Market Capitalization (₹ Cr):
       The aggregate market value of the company's outstanding equity shares:
       Market Cap = Share Price * Total Outstanding Shares.
       In portfolio construction, market cap determines liquidity thresholds, index weightings,
       and categorizes the firm across Large-cap (UltraTech, NTPC) vs Mid-cap (Voltamp, Star Cement).
    2. Current Price (₹):
       The latest traded price per share on the National Stock Exchange.
    3. ROCE (%) - Return on Capital Employed:
       ROCE = Earnings Before Interest and Tax (EBIT) / Capital Employed, where
       Capital Employed = Total Assets - Current Liabilities = Equity + Long-term Debt.
       It measures how efficiently management generates operational returns from all providers
       of capital (both equity holders and lenders). Values above 15% signify competitive moats.
    4. ROE (%) - Return on Equity:
       ROE = Net Profit After Tax / Shareholders' Net Worth.
       Measures the rate of return generated exclusively on common equity capital, driving
       the long-term intrinsic compounding rate of the equity portfolio.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.

    Returns:
        Dict[str, Optional[float]]: Extracted metrics mapped by standard name.
    """
    ratios_dict = {
        "market_cap": None,
        "current_price": None,
        "roce": None,
        "roe": None,
    }
    top_el = soup.find("ul", id="top-ratios")
    if not top_el:
        return ratios_dict

    for li in top_el.find_all("li"):
        name_el = li.find("span", class_="name")
        val_el = li.find("span", class_="number")
        if not name_el or not val_el:
            continue

        raw_name = name_el.text.strip().lower()
        val = _clean_numeric(val_el.text.strip())

        if "market cap" in raw_name:
            ratios_dict["market_cap"] = val
        elif "current price" in raw_name:
            ratios_dict["current_price"] = val
        elif "roce" in raw_name and "3" not in raw_name:
            ratios_dict["roce"] = val
        elif "roe" in raw_name and "3" not in raw_name:
            ratios_dict["roe"] = val

    return ratios_dict


def parse_roce_3yr_average(soup: BeautifulSoup) -> Optional[float]:
    """
    Extract the multi-year ROCE time series from the 'Ratios' table section and compute
    a simple 3-year historical arithmetic average.

    Financial Rationale:
    Capital expenditure and infrastructure sectors (cement manufacturing, power generation,
    heavy electrical transformers) are subject to multi-year capex and capacity utilization cycles:
    - During initial plant construction or commissioning, large capital is locked in Capital
      Work-In-Progress (CWIP) while earnings have not yet begun, causing single-year ROCE to dip.
    - Conversely, cyclical supply shortages temporarily create peak operating margins.
    - A 3-Year Simple Average ROCE:
          3Yr Avg ROCE = (ROCE_t + ROCE_{t-1} + ROCE_{t-2}) / 3
      smooths out lumpiness, revealing underlying normalized economic profitability.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.

    Returns:
        Optional[float]: 3-year average ROCE rounded to 2 decimal places.
    """
    sec = soup.find("section", id="ratios")
    if not sec:
        return None
    table = sec.find("table")
    if not table:
        return None

    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if tds and "roce" in tds[0].lower():
            # Extract valid yearly numeric values from subsequent columns
            numeric_vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if len(numeric_vals) >= 3:
                # Average of the most recent 3 fiscal years
                recent_3 = numeric_vals[-3:]
                return round(sum(recent_3) / 3.0, 2)
            elif numeric_vals:
                return round(sum(numeric_vals) / len(numeric_vals), 2)

    return None


def parse_operating_cash_flow(soup: BeautifulSoup) -> Optional[float]:
    """
    Extract Cash Flow from Operating Activities (CFO, in ₹ Crores) for the most recent
    financial year from the 'Cash Flows' table.

    Financial Rationale:
    In Security Analysis & Portfolio Management, Operating Cash Flow serves as the primary
    acid test for earnings quality:
    - Accrual accounting allows management discretion over revenue recognition, provisioning,
      and depreciation schedules in net profit figures.
    - Cash from Operating Activity measures actual hard cash collected from core commercial operations
      after adjusting for working capital changes (inventories, trade receivables, trade payables).
    - Robust CFO verifies that reported profits are genuine and available to service debt, pay
      dividends, and fund new capital investments without requiring dilutive equity raises.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.

    Returns:
        Optional[float]: Latest annual Operating Cash Flow in ₹ Crores.
    """
    sec = soup.find("section", id="cash-flow")
    if not sec:
        return None
    table = sec.find("table")
    if not table:
        return None

    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if tds and "operating activity" in tds[0].lower():
            numeric_vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if numeric_vals:
                # Return the most recent annual reporting period
                return numeric_vals[-1]

    return None


def parse_debt_to_equity(soup: BeautifulSoup) -> Optional[float]:
    """
    Extract Total Borrowings and Shareholders' Net Worth from the 'Balance Sheet' table
    and calculate the financial leverage ratio: Debt / Equity.

    Financial Rationale:
    The Debt-to-Equity (D/E) ratio quantifies the degree to which a company finances its assets
    through debt versus wholly owned equity capital:
        Debt / Equity = Total Borrowings / (Equity Share Capital + Reserves & Surplus)
    - In infrastructure and capital goods, heavy interest debt burdens during sector downturns
      amplify insolvency risk and compress equity returns through financial leverage.
    - Conversely, low D/E (e.g. Voltamp Transformers at ~0.0, UltraTech Cement at ~0.3) reflects
      a pristine balance sheet with substantial safety margin during high interest rate cycles.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.

    Returns:
        Optional[float]: Calculated Debt/Equity ratio rounded to 2 decimal places.
    """
    sec = soup.find("section", id="balance-sheet")
    if not sec:
        return None
    table = sec.find("table")
    if not table:
        return None

    borrowings: Optional[float] = None
    equity_capital: Optional[float] = None
    reserves: Optional[float] = None

    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if not tds:
            continue
        row_title = tds[0].lower()

        if "borrowing" in row_title:
            vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if vals:
                borrowings = vals[-1]
        elif "equity capital" in row_title:
            vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if vals:
                equity_capital = vals[-1]
        elif "reserves" in row_title:
            vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if vals:
                reserves = vals[-1]

    if borrowings is not None and equity_capital is not None and reserves is not None:
        total_equity = equity_capital + reserves
        if total_equity > 0:
            return round(borrowings / total_equity, 2)

    return None


def parse_opm(soup: BeautifulSoup) -> Optional[float]:
    """
    Extract Operating Profit Margin (OPM %) for the most recent year from the 'Profit & Loss' table.

    Financial Rationale:
    OPM = Operating Profit (EBITDA) / Net Sales * 100.
    Evaluates pricing power, raw material pass-through capabilities, and operating leverage.
    High OPM buffers companies against input cost inflation (coal, petcoke, power tariffs).

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.

    Returns:
        Optional[float]: Latest OPM percentage.
    """
    sec = soup.find("section", id="profit-loss")
    if not sec:
        return None
    table = sec.find("table")
    if not table:
        return None

    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if tds and "opm" in tds[0].lower():
            vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if vals:
                return vals[-1]

    return None


def parse_sales_and_profit_growth(soup: BeautifulSoup) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract 3-Year Compounded Annual Growth Rate (CAGR) for Sales and Net Profit from
    the 'ranges-table' elements on the company page.

    Financial Rationale:
    3-Year Sales Growth verifies commercial revenue scaling and order backlog conversion,
    while 3-Year Profit Growth demonstrates whether operational scale yields positive
    operating leverage and expanded bottom-line returns for equity shareholders.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.

    Returns:
        Tuple[Optional[float], Optional[float]]: (sales_growth_3yr, profit_growth_3yr)
    """
    sales_growth_3y: Optional[float] = None
    profit_growth_3y: Optional[float] = None

    for tbl in soup.find_all("table", class_="ranges-table"):
        th = tbl.find("th")
        if not th:
            continue
        th_text = th.text.strip().lower()

        if "sales growth" in th_text:
            for tr in tbl.find_all("tr"):
                tds = [td.text.strip() for td in tr.find_all("td")]
                if tds and "3 years" in tds[0].lower():
                    sales_growth_3y = _clean_numeric(tds[1]) if len(tds) > 1 else None

        elif "profit growth" in th_text:
            for tr in tbl.find_all("tr"):
                tds = [td.text.strip() for td in tr.find_all("td")]
                if tds and "3 years" in tds[0].lower():
                    profit_growth_3y = _clean_numeric(tds[1]) if len(tds) > 1 else None

    return sales_growth_3y, profit_growth_3y


# -----------------------------------------------------------------------------
# HIGH-LEVEL AGGREGATOR ENGINE
# -----------------------------------------------------------------------------

def extract_stock_fundamentals(symbol: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Retrieve and parse all required fundamental metrics for a single stock symbol.

    If retrieval or parsing fails for any reason, returns a graceful record with
    'Data Unavailable' values rather than raising unhandled exceptions.

    Parameters:
        symbol (str): Stock symbol.
        use_cache (bool): Whether to use cached HTML.

    Returns:
        Dict[str, Any]: Dictionary of parsed fundamental fields.
    """
    default_record = {
        "symbol": symbol,
        "name": LOCKED_PORTFOLIO.get(symbol, {}).get("name", symbol),
        "sector": LOCKED_PORTFOLIO.get(symbol, {}).get("sector", "Other"),
        "market_cap": None,
        "current_price": None,
        "roce": None,
        "roe": None,
        "roce_3yr_avg": None,
        "opm": None,
        "debt_to_equity": None,
        "operating_cash_flow": None,
        "sales_growth_3yr": None,
        "profit_growth_3yr": None,
        "status": "Data Unavailable",
    }

    html = fetch_screener_page(symbol, use_cache=use_cache)
    if not html:
        return default_record

    try:
        soup = BeautifulSoup(html, "html.parser")
        top_ratios = parse_top_ratios(soup)
        roce_3yr = parse_roce_3yr_average(soup)
        cfo = parse_operating_cash_flow(soup)
        debt_eq = parse_debt_to_equity(soup)
        opm = parse_opm(soup)
        sales_3y, profit_3y = parse_sales_and_profit_growth(soup)

        return {
            "symbol": symbol,
            "name": LOCKED_PORTFOLIO.get(symbol, {}).get("name", symbol),
            "sector": LOCKED_PORTFOLIO.get(symbol, {}).get("sector", "Other"),
            "market_cap": top_ratios.get("market_cap"),
            "current_price": top_ratios.get("current_price"),
            "roce": top_ratios.get("roce"),
            "roe": top_ratios.get("roe"),
            "roce_3yr_avg": roce_3yr,
            "opm": opm,
            "debt_to_equity": debt_eq,
            "operating_cash_flow": cfo,
            "sales_growth_3yr": sales_3y,
            "profit_growth_3yr": profit_3y,
            "status": "OK",
        }
    except Exception as exc:
        logger.error("Failed to parse fundamentals for %s: %s", symbol, exc)
        return default_record


def get_fundamentals_summary(
    symbols: Optional[List[str]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Generate a consolidated fundamental summary DataFrame for all specified portfolio symbols.

    Directly utilized by Tab 2 (Fundamentals) in the Streamlit application.

    Parameters:
        symbols (List[str], optional): List of symbols. Defaults to LOCKED_PORTFOLIO_SYMBOLS.
        use_cache (bool): Whether to load from cached HTML files.

    Returns:
        pd.DataFrame: Clean, structured DataFrame with one row per stock.
    """
    if symbols is None:
        symbols = LOCKED_PORTFOLIO_SYMBOLS

    records = []
    for sym in symbols:
        rec = extract_stock_fundamentals(sym, use_cache=use_cache)
        records.append(rec)

    df = pd.DataFrame(records)
    logger.info("Extracted fundamentals summary for %d symbols.", len(df))
    return df
