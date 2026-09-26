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

import json
import logging
import operator
import re
import time
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    CAPITAL_GOODS_SCREEN_CRITERIA,
    FUNDAMENTALS_CACHE_DIR,
    LOCKED_PORTFOLIO_SYMBOLS,
    NSE_BOARD_MEETINGS_API_URL,
    NSE_PLEDGE_API_URL,
    NSE_PLEDGE_PAGE_URL,
    NSE_REQUEST_HEADERS,
    SCREENER_DOWNLOAD_DATE,
    SCREENER_RECORDS,
    SYMBOL_NAME,
    sector_of,
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

def _latest_statement_year(soup: BeautifulSoup) -> Optional[int]:
    """Latest financial year shown in the Profit & Loss table header (TTM ignored)."""
    sec = soup.find("section", id="profit-loss")
    table = sec.find("table") if sec else None
    if not table:
        return None
    years = [int(y) for th in table.find_all("th") for y in re.findall(r"\b(19\d{2}|20\d{2})\b", th.text)]
    return max(years) if years else None


def _statements_are_stale(soup: BeautifulSoup, max_age_years: int = 2) -> bool:
    """True when a page's financial statements end more than max_age_years ago (or are absent)."""
    latest = _latest_statement_year(soup)
    return latest is None or latest < date.today().year - max_age_years


def _has_3yr_growth(soup: BeautifulSoup) -> bool:
    """True when the page reports a 3-year compounded sales growth figure."""
    sales_3y, _ = parse_sales_and_profit_growth(soup)
    return sales_3y is not None


def _get_with_retry(session: requests.Session, url: str, attempts: int = 4, backoff_seconds: float = 5.0):
    """
    GET a Screener.in page, retrying when rate-limited (HTTP 429) or on transient 5xx errors.

    Honors a numeric Retry-After header; otherwise waits backoff_seconds x attempt. Returns the
    last response (which may still be an error status); raises the last network error if every
    attempt failed to connect.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, timeout=12)
            if resp.status_code != 429 and resp.status_code < 500:
                return resp
            retry_after = resp.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else backoff_seconds * attempt
            logger.info("Screener.in returned HTTP %d for %s; retrying in %.0fs (attempt %d/%d).",
                        resp.status_code, url, wait, attempt, attempts)
        except requests.RequestException as exc:
            last_exc, resp = exc, None
            wait = backoff_seconds * attempt
        if attempt < attempts:
            time.sleep(wait)
    if resp is None and last_exc is not None:
        raise last_exc
    return resp


# Screener.in pages embed pre-signed Amazon S3 links (e.g. concall recordings) whose query strings
# carry a third party's AWS access key ID and request signature. They are irrelevant to the parsed
# fundamentals and must not be stored in the repository.
_PRESIGNED_PARAMS = re.compile(r"(X-Amz-(?:Credential|Signature|Security-Token)=)[^&\"'\s<>]+", re.IGNORECASE)


def sanitize_screener_html(html: str) -> str:
    """Redact pre-signed S3 credentials/signatures from a Screener.in page before caching it."""
    return _PRESIGNED_PARAMS.sub(r"\1REDACTED", html)


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
        resp = _get_with_retry(session, url_cons)
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
            elif _statements_are_stale(soup):
                # Headline ratios are live, but the consolidated statements stopped years ago
                # (e.g. GVT&D: consolidated P&L ends Dec 2010); growth and ROCE history would be
                # missing or wrong, so use the standalone page
                logger.info("Consolidated statements for %s are stale; using standalone page.", symbol)
                needs_fallback = True
            elif not _has_3yr_growth(soup):
                # Consolidated history too short or broken for 3-year growth (e.g. TIMKEN:
                # consolidated reporting starts FY2025; ABB: gap between 2012 and 2024), while
                # the standalone page carries the full history
                logger.info("Consolidated page for %s lacks 3-year history; using standalone page.", symbol)
                needs_fallback = True

    # Step 3: Fallback to standalone endpoint if consolidated is unavailable or blank
    if needs_fallback:
        url_stand = SCREENER_STANDALONE_URL.format(symbol=symbol.upper())
        logger.info("Falling back to standalone Screener page for %s: %s", symbol, url_stand)
        try:
            resp = _get_with_retry(session, url_stand)
            if resp.status_code != 200:
                logger.error("Standalone page for %s returned HTTP status %d.", symbol, resp.status_code)
                return None
        except Exception as exc:
            logger.error("Network error fetching standalone page for %s: %s", symbol, exc)
            return None

    html_content = sanitize_screener_html(resp.text)

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
       and categorizes the firm across large-cap vs mid-cap.
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


def parse_operating_cash_flow_3yr(soup: BeautifulSoup) -> Optional[float]:
    """
    Sum of Cash from Operating Activity over the last 3 financial years (Rs Crores), from the
    same 'Cash Flows' row as parse_operating_cash_flow(). Matches Screener.in's
    "Operating cash flow 3years" screen field; None if fewer than 3 years are reported.
    """
    sec = soup.find("section", id="cash-flow")
    table = sec.find("table") if sec else None
    if not table:
        return None
    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if tds and "operating activity" in tds[0].lower():
            numeric_vals = [_clean_numeric(v) for v in tds[1:] if _clean_numeric(v) is not None]
            if len(numeric_vals) >= 3:
                return round(sum(numeric_vals[-3:]), 2)
    return None


def parse_interest_coverage(soup: BeautifulSoup) -> Optional[float]:
    """
    Interest coverage for the latest full financial year: (Profit before tax + Interest) / Interest,
    i.e. EBIT / Interest, from the 'Profit & Loss' table (a trailing TTM column is skipped).

    Returns:
        Optional[float]: Coverage rounded to 2 decimals; inf when interest is zero; None if the
        rows are not reported.
    """
    sec = soup.find("section", id="profit-loss")
    table = sec.find("table") if sec else None
    if not table:
        return None
    headers = [th.text.strip() for th in table.find_all("th")][1:]
    rows = {}
    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if tds:
            rows[tds[0].rstrip(" +").strip().lower()] = [_clean_numeric(v) for v in tds[1:]]
    interest, pbt = rows.get("interest"), rows.get("profit before tax")
    if not interest or not pbt or not headers:
        return None
    idx = len(headers) - 2 if headers[-1].upper() == "TTM" else len(headers) - 1
    if idx < 0 or idx >= min(len(interest), len(pbt)) or interest[idx] is None or pbt[idx] is None:
        return None
    if interest[idx] == 0:
        return float("inf")
    return round((pbt[idx] + interest[idx]) / interest[idx], 2)


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

    rows = {}
    for tr in table.find_all("tr"):
        tds = [td.text.strip() for td in tr.find_all("td")]
        if tds:
            rows.setdefault(tds[0].rstrip(" +").strip().lower(), tds[1:])

    # Screener displays OPM % rounded to a whole number, which can flip a strict threshold
    # (e.g. "9%" vs OPM > 9); compute it exactly from the same (latest) column when possible
    sales, op_profit = rows.get("sales"), rows.get("operating profit")
    if sales and op_profit:
        s_last, o_last = _clean_numeric(sales[-1]), _clean_numeric(op_profit[-1])
        if s_last and o_last is not None:
            return round(o_last / s_last * 100, 2)

    for title, cells in rows.items():
        if "opm" in title:
            vals = [_clean_numeric(v) for v in cells if _clean_numeric(v) is not None]
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

# -----------------------------------------------------------------------------
# PROMOTER PLEDGE (NSE disclosures)
# -----------------------------------------------------------------------------
# Screener.in's public company page does not report pledge as a ratio; it only mentions it in
# machine-generated "Cons" text when pledge is high (seen only at 40%+), so its absence does not
# mean zero. NSE's corporate pledge disclosures are the authoritative source.

_nse_pledge_session: Optional[requests.Session] = None


def _get_nse_pledge_session(refresh: bool = False) -> requests.Session:
    """Session carrying NSE's Akamai cookies, seeded from the pledge-data page (the homepage 403s)."""
    global _nse_pledge_session
    if _nse_pledge_session is None or refresh:
        session = requests.Session()
        session.headers.update({**NSE_REQUEST_HEADERS, "Accept": "application/json, text/plain, */*",
                                "Referer": NSE_PLEDGE_PAGE_URL})
        try:
            session.get(NSE_PLEDGE_PAGE_URL, timeout=20)
        except requests.RequestException as exc:
            logger.warning("NSE pledge page warm-up failed: %s", exc)
        _nse_pledge_session = session
    return _nse_pledge_session


def parse_pledge_records(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """
    Pledged percentage (% of promoter holding pledged, NSE 'percPromoterShares') and its
    shareholding-pattern date from an NSE corporate-pledgedata response.

    An empty record list means no promoter pledge is disclosed: (0.0, None).
    """
    records = payload.get("data") or []
    if not records:
        return 0.0, None

    def shp_date(rec):
        return pd.to_datetime(rec.get("shp"), format="%d-%b-%Y", errors="coerce")

    latest = max(records, key=lambda rec: (shp_date(rec) if pd.notna(shp_date(rec)) else pd.Timestamp.min))
    value = _clean_numeric(str(latest.get("percPromoterShares", "")).strip())
    return (value if value is not None else None), latest.get("shp")


def _fetch_nse_json(url: str, cache_path: Path, label: str, use_cache: bool = True, attempts: int = 4) -> Optional[Any]:
    """
    GET an NSE JSON API with the cookie-seeded session, retrying Akamai blocks, and cache the
    payload at cache_path. Returns None when NSE could not be reached (never a fabricated value).
    """
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable cache for %s (%s); refetching.", label, exc)
    payload = None
    for attempt in range(1, attempts + 1):
        try:
            resp = _get_nse_pledge_session(refresh=attempt > 1).get(url, timeout=20)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/json"):
                payload = resp.json()
                break
            logger.info("NSE %s: HTTP %d (attempt %d/%d).", label, resp.status_code, attempt, attempts)
        except (requests.RequestException, ValueError) as exc:
            logger.info("NSE %s failed (%s), attempt %d/%d.", label, exc, attempt, attempts)
        if attempt < attempts:
            time.sleep(3 * attempt)
    if payload is None:
        logger.warning("Could not fetch NSE %s.", label)
        return None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache NSE %s: %s", label, exc)
    return payload


def parse_results_meetings(records: List[Dict[str, Any]], as_of: date) -> Dict[str, Any]:
    """
    From NSE board-meeting records, find the next announced results meeting on/after as_of and
    last year's actual meeting for the same (July-September) quarter.

    Only meetings NSE lists are used: if no upcoming results meeting has been announced, the next
    date is None ("not announced"), never an estimate.
    """
    def is_results(rec):
        text = f"{rec.get('bm_purpose', '')} {rec.get('bm_desc', '')}".lower()
        return "result" in text

    meetings = []
    for rec in records or []:
        when = pd.to_datetime(rec.get("bm_date"), format="%d-%b-%Y", errors="coerce")
        if pd.notna(when) and is_results(rec):
            meetings.append((when.date(), rec))
    upcoming = sorted((m for m in meetings if m[0] >= as_of), key=lambda m: m[0])
    past = sorted((m for m in meetings if m[0] < as_of), key=lambda m: m[0])
    # Last year's September-quarter results: a results meeting held Oct-Dec of the previous year
    prior = sorted((m for m in meetings if m[0].year == as_of.year - 1 and m[0].month >= 10), key=lambda m: m[0])
    return {
        "next_results_date": upcoming[0][0].isoformat() if upcoming else None,
        "next_results_desc": (upcoming[0][1].get("bm_desc") or "")[:160] if upcoming else None,
        "prior_year_sep_qtr_results_date": prior[0][0].isoformat() if prior else None,
        "last_results_date": past[-1][0].isoformat() if past else None,
    }


BOARD_MEETINGS_MAX_AGE_HOURS = 20


def fetch_results_calendar(
    symbol: str,
    as_of: Optional[date] = None,
    cache_dir: Path = FUNDAMENTALS_CACHE_DIR,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Upcoming quarterly-results board meeting for one symbol from NSE's board-meeting disclosures
    (see parse_results_meetings). 'results_date_status' is 'announced', 'not announced', or
    'unavailable' (NSE unreachable); dates are never guessed.
    """
    as_of = as_of or date.today()
    url = NSE_BOARD_MEETINGS_API_URL.format(symbol=urllib.parse.quote(symbol.upper(), safe=""))
    cache = Path(cache_dir) / f"{symbol.upper()}.board_meetings.json"
    # Results dates are announced during the window: re-fetch a cache older than a day (stale fallback)
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < BOARD_MEETINGS_MAX_AGE_HOURS * 3600
    payload = _fetch_nse_json(url, cache, f"board meetings for {symbol}", use_cache=use_cache and fresh)
    if payload is None and cache.exists():
        logger.warning("Using stale board-meeting cache for %s (NSE unreachable).", symbol)
        payload = json.loads(cache.read_text(encoding="utf-8"))
    if payload is None:
        return {"next_results_date": None, "next_results_desc": None, "prior_year_sep_qtr_results_date": None,
                "last_results_date": None, "results_date_status": "unavailable"}
    records = payload if isinstance(payload, list) else payload.get("data", [])
    info = parse_results_meetings(records, as_of)
    info["results_date_status"] = "announced" if info["next_results_date"] else "not announced"
    return info


def fetch_pledged_percentage(
    symbol: str,
    cache_dir: Path = FUNDAMENTALS_CACHE_DIR,
    use_cache: bool = True,
    attempts: int = 4,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Promoter pledge for one NSE symbol: (% of promoter holding pledged, shareholding date).

    Cached as data/fundamentals_cache/{SYMBOL}.pledge.json. Returns (None, None) when NSE could
    not be reached, so a screen treats it as missing (a fail), never as zero.
    """
    url = NSE_PLEDGE_API_URL.format(symbol=urllib.parse.quote(symbol.upper(), safe=""))
    cache = Path(cache_dir) / f"{symbol.upper()}.pledge.json"
    previous = None
    if cache.exists():
        try:
            previous = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
    payload = _fetch_nse_json(url, cache, f"pledge data for {symbol}", use_cache=use_cache, attempts=attempts)
    # NSE answers {"data": []} both for "no pledge disclosed" and, at times, for every symbol (an outage).
    # An empty reply never overwrites a cached record that had data: keep the last real disclosure.
    if (isinstance(payload, dict) and not payload.get("data")
            and isinstance(previous, dict) and previous.get("data")):
        logger.warning("NSE returned no pledge records for %s but the cache has a disclosure; keeping the cache.",
                       symbol)
        payload = previous
        cache.write_text(json.dumps(previous), encoding="utf-8")
    if payload is None:
        return None, None
    if not isinstance(payload, dict) or "data" not in payload:
        return None, None
    return parse_pledge_records(payload)


SCREENER_FIELDS = {
    "market_cap": "Market Capitalization",
    "current_price": "Current Price",
    "roce": "Return on capital employed",
    "roe": "Return on equity",
    "roce_3yr_avg": "Average return on capital employed 3Years",
    "opm": "OPM",
    "debt_to_equity": "Debt to equity",
    "operating_cash_flow": "Cash from operations last year",
    "operating_cash_flow_3yr": "Operating cash flow 3years",
    "interest_coverage": "Interest Coverage Ratio",
    "pledged_pct": "Pledged percentage",
    "sales_growth_3yr": "Sales growth 3Years",
    "profit_growth_3yr": "Profit growth 3Years",
}


def screener_record(symbol: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """A fundamentals record from one row of a Screener sector export (config.SCREENER_FILES).
    Blank cells are missing values (None), which fail a screen criterion, never pass it."""
    rec: Dict[str, Any] = {"symbol": symbol, "name": SYMBOL_NAME.get(symbol, symbol), "sector": sector_of(symbol)}
    for field, column in SCREENER_FIELDS.items():
        rec[field] = _clean_numeric(str(row.get(column, "") or "").strip())
    rec["pledged_as_of"] = f"Screener {SCREENER_DOWNLOAD_DATE}"
    rec["status"] = "OK"
    rec["source"] = f"Screener export {SCREENER_DOWNLOAD_DATE}"
    return rec


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
        "name": SYMBOL_NAME.get(symbol, symbol),
        "sector": sector_of(symbol),
        "market_cap": None,
        "current_price": None,
        "roce": None,
        "roe": None,
        "roce_3yr_avg": None,
        "opm": None,
        "debt_to_equity": None,
        "operating_cash_flow": None,
        "operating_cash_flow_3yr": None,
        "interest_coverage": None,
        "pledged_pct": None,
        "pledged_as_of": None,
        "sales_growth_3yr": None,
        "profit_growth_3yr": None,
        "status": "Data Unavailable",
    }

    if symbol in SCREENER_RECORDS:  # the single source of truth: the Screener sector export
        return screener_record(symbol, SCREENER_RECORDS[symbol])

    pledged_pct, pledged_as_of = fetch_pledged_percentage(symbol, use_cache=use_cache)
    default_record.update({"pledged_pct": pledged_pct, "pledged_as_of": pledged_as_of})

    html = fetch_screener_page(symbol, use_cache=use_cache)
    if not html:
        return default_record

    try:
        soup = BeautifulSoup(html, "html.parser")
        top_ratios = parse_top_ratios(soup)
        roce_3yr = parse_roce_3yr_average(soup)
        cfo = parse_operating_cash_flow(soup)
        cfo_3yr = parse_operating_cash_flow_3yr(soup)
        interest_cov = parse_interest_coverage(soup)
        debt_eq = parse_debt_to_equity(soup)
        opm = parse_opm(soup)
        sales_3y, profit_3y = parse_sales_and_profit_growth(soup)

        return {
            "symbol": symbol,
            "name": SYMBOL_NAME.get(symbol, symbol),
            "sector": sector_of(symbol),
            "market_cap": top_ratios.get("market_cap"),
            "current_price": top_ratios.get("current_price"),
            "roce": top_ratios.get("roce"),
            "roe": top_ratios.get("roe"),
            "roce_3yr_avg": roce_3yr,
            "opm": opm,
            "debt_to_equity": debt_eq,
            "operating_cash_flow": cfo,
            "operating_cash_flow_3yr": cfo_3yr,
            "interest_coverage": interest_cov,
            "pledged_pct": pledged_pct,
            "pledged_as_of": pledged_as_of,
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


_SCREEN_COMPARISONS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}


def evaluate_fundamental_screen(record: Dict[str, Any], criteria: List[Tuple[str, str, float, str]]) -> Dict[str, Any]:
    """
    Score one stock's fundamentals record against screen criteria.

    Returns a dict with each criterion's metric value, a pass_<field> flag per criterion, and
    'failed_criteria' (list of labels). A metric that could not be read fails as
    "<label> (missing)"; it never passes by default.
    """
    result: Dict[str, Any] = {}
    failed: List[str] = []
    for field, comparison, threshold, label in criteria:
        value = record.get(field)
        result[field] = value
        present = value is not None and not pd.isna(value)
        passed = present and _SCREEN_COMPARISONS[comparison](value, threshold)
        result[f"pass_{field}"] = bool(passed)
        if not passed:
            failed.append(label if present else f"{label} (missing)")
    result["failed_criteria"] = failed
    return result


def generate_fundamentals_screen_check(
    symbols: List[str],
    criteria: Optional[List[Tuple[str, str, float, str]]] = None,
    use_cache: bool = False,
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Check stocks against a fundamental screen and save one row per symbol.

    Columns: symbol, sector, as_of, status, each screened metric, pass_<metric> for each
    criterion, passes_screen, failed_criteria. A metric that could not be read counts as a
    failure ("<label> (missing)"), never as a pass.

    Parameters:
        symbols (List[str]): Stocks to check.
        criteria (list, optional): (field, comparison, threshold, label) tuples.
            Defaults to CAPITAL_GOODS_SCREEN_CRITERIA.
        use_cache (bool): False (default) fetches live Screener.in pages.
        output_csv_path (Path, optional): Destination CSV; None (default) skips saving.

    Returns:
        pd.DataFrame: The screen check table.
    """
    if criteria is None:
        criteria = CAPITAL_GOODS_SCREEN_CRITERIA

    rows = []
    for symbol in symbols:
        record = extract_stock_fundamentals(symbol, use_cache=use_cache)
        row = {"symbol": symbol, "sector": sector_of(symbol),
               "as_of": date.today().isoformat(), "status": record["status"]}
        result = evaluate_fundamental_screen(record, criteria)
        failed = result.pop("failed_criteria")
        row.update(result)
        row["passes_screen"] = not failed
        row["failed_criteria"] = "; ".join(failed)
        rows.append(row)
        logger.info("%s fundamental screen: %s%s", symbol, "PASS" if not failed else "FAIL",
                    "" if not failed else f" ({row['failed_criteria']})")

    screen_df = pd.DataFrame(rows)
    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        screen_df.to_csv(out_path, index=False)
        logger.info("Saved fundamentals screen check to: %s", out_path.resolve())
    return screen_df

