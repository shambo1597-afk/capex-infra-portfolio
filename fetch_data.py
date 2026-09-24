"""
Data Ingestion and Acquisition Module for Indian Equity Portfolio.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)

This module handles:
1. Direct, authentic daily NSE Bhavcopy retrieval with session warm-up, cookie
   persistence, and custom header masking (strictly bypassing 3rd party wrappers
   to eliminate corporate action adjustment distortions).
2. Benchmark Nifty 500 Price Return retrieval via yfinance (^CRSLDX).
3. Benchmark Nifty 500 Total Return Index (TRI) ingestion from local official CSV.
"""

import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from config import (
    BENCHMARK_PRICE_TICKER,
    BHAVCOPY_EXPECTED_COLUMNS,
    CEMENT_STOCKS,
    CAPITAL_GOODS_EPC_STOCKS,
    DEFAULT_TRI_CSV_PATH,
    NSE_BHAVCOPY_URL_TEMPLATE,
    NSE_HOME_URL,
    NSE_REQUEST_HEADERS,
    PORTFOLIO_SYMBOLS,
    PROCESSED_DATA_DIR,
    RAW_BHAVCOPY_DIR,
    REQUEST_DELAY_SECONDS,
    SYMBOL_ALIASES,
)

# Set up logging for transparent pipeline execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fetch_data")


class NSEBhavcopyFetcher:
    """
    Direct HTTP client for downloading and parsing the National Stock Exchange (NSE)
    official 'Full Bhavcopy and Security Deliverable data' daily report.

    Financial Rationale:
    Third-party APIs and wrapper libraries (like nsepy or unadjusted web scrapers) frequently
    introduce corporate action distortion (unannounced stock splits, bonus adjustments, or
    stale tick caches). By pulling official daily closing bhavcopies directly from the
    clearing house/exchange feed, we ensure 100% auditable, authoritative pricing data for
    academic valuation and portfolio optimization.
    """

    def __init__(self, cache_dir: Path = RAW_BHAVCOPY_DIR, delay_seconds: float = REQUEST_DELAY_SECONDS):
        """
        Initialize the NSE Bhavcopy Fetcher with a persistent HTTP Session.

        Parameters:
            cache_dir (Path): Directory where raw/filtered daily files are preserved.
            delay_seconds (float): Throttle delay between requests to comply with exchange etiquette.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update(NSE_REQUEST_HEADERS)
        self.session_initialized = False

    def warm_up_session(self) -> None:
        """
        Execute a preliminary warm-up request to the NSE homepage.

        Technical Rationale:
        The NSE web infrastructure sits behind anti-bot and security firewalls (Akamai / Cloudflare).
        Calling the reports API directly without first visiting the root domain results in HTTP 401/403
        due to missing security cookies ('nsit', 'nseappid', etc.).
        Visiting the root domain seeds the persistent requests.Session cookie jar.
        Even if the warm-up call returns a 403, cookies are often set, allowing subsequent API queries.
        """
        logger.info("Initializing NSE session cookies via homepage warm-up...")
        try:
            response = self.session.get(NSE_HOME_URL, timeout=12)
            logger.info("Warm-up request finished with status code %d", response.status_code)
        except Exception as exc:
            # As per project requirements: 403 or handshake noise on warm-up is expected; proceed anyway
            logger.warning("Session warm-up encountered notice (%s); proceeding with initialized session.", exc)
        self.session_initialized = True

    def _format_date(self, target_date: date) -> str:
        """
        Format a date object into NSE's required URL format: 'DD-Mon-YYYY' (e.g. '24-Sep-2025').
        """
        return target_date.strftime("%d-%b-%Y")

    def fetch_daily_bhavcopy(
        self,
        target_date: date,
        target_symbols: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and parse the NSE Full Bhavcopy for a single calendar date.

        Parameters:
            target_date (date): The trading date to query.
            target_symbols (List[str], optional): List of equity symbols to extract. If None, uses PORTFOLIO_SYMBOLS.
            use_cache (bool): If True, checks local storage first to prevent repeated queries.

        Returns:
            Optional[pd.DataFrame]: Filtered DataFrame of target stock records, or None if market holiday/unavailable.
        """
        if target_symbols is None:
            target_symbols = PORTFOLIO_SYMBOLS

        # Account for historical ticker aliases (e.g., GET&D -> GVT&D, ITDCEM -> CEMPRO)
        all_query_symbols = set(target_symbols)
        for old_sym, new_sym in SYMBOL_ALIASES.items():
            if new_sym in all_query_symbols or old_sym in all_query_symbols:
                all_query_symbols.add(old_sym)
                all_query_symbols.add(new_sym)

        date_str = self._format_date(target_date)
        cache_file = self.cache_dir / f"bhav_{date_str}.csv"
        holiday_flag_file = self.cache_dir / f"holiday_{date_str}.flag"

        # Check local cache first
        if use_cache:
            if holiday_flag_file.exists():
                logger.debug("Date %s is a known market holiday (cached).", date_str)
                return None
            if cache_file.exists():
                logger.debug("Reading cached bhavcopy for %s", date_str)
                try:
                    df = pd.read_csv(cache_file)
                    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed", errors="coerce")
                    return df
                except Exception as exc:
                    logger.warning("Failed to parse cached file %s (%s). Re-fetching.", cache_file, exc)

        if not self.session_initialized:
            self.warm_up_session()

        url = NSE_BHAVCOPY_URL_TEMPLATE.format(date_str=date_str)
        logger.debug("Requesting Bhavcopy for %s from NSE...", date_str)

        try:
            response = self.session.get(url, timeout=15)
        except Exception as exc:
            logger.error("Network error fetching bhavcopy for %s: %s", date_str, exc)
            return None

        # Handle Market Holidays or Missing Reports
        # NSE returns 404 or an HTML error page on holidays / non-trading days
        if response.status_code == 404 or "Full Bhavcopy" not in response.text and "SYMBOL" not in response.text:
            if response.status_code == 404 or "<html" in response.text.lower():
                logger.debug("No bhavcopy data for %s (Status %d - likely market holiday/weekend).", date_str, response.status_code)
                # Mark as holiday so we don't query again
                try:
                    holiday_flag_file.touch()
                except OSError:
                    pass
                return None

        if response.status_code != 200:
            logger.warning("Unexpected status code %d for %s. Skipping.", response.status_code, date_str)
            return None

        # Parse CSV content from response text
        try:
            raw_df = pd.read_csv(io.StringIO(response.text))
        except Exception as exc:
            logger.error("Failed to parse CSV response for %s: %s", date_str, exc)
            return None

        # Clean column names (strip whitespace)
        raw_df.columns = [c.strip() for c in raw_df.columns]

        # Ensure required columns are present
        if "SYMBOL" not in raw_df.columns:
            logger.warning("Malformed CSV received for %s: 'SYMBOL' column missing.", date_str)
            return None

        # Strip whitespace from string columns
        for col in raw_df.columns:
            if raw_df[col].dtype == object or str(raw_df[col].dtype).startswith("str"):
                raw_df[col] = raw_df[col].astype(str).str.strip()

        # Filter strictly for EQ series (standard equity) or BE (book-entry trade-to-trade)
        # to exclude bonds, sovereign debt (GS), and derivative series
        if "SERIES" in raw_df.columns:
            series_mask = raw_df["SERIES"].isin(["EQ", "BE", "SM"])
            filtered_df = raw_df[series_mask].copy()
        else:
            filtered_df = raw_df.copy()

        # Map historical ticker aliases to current unified ticker
        filtered_df["SYMBOL"] = filtered_df["SYMBOL"].replace(SYMBOL_ALIASES)

        # Filter to only the needed portfolio SYMBOLS
        portfolio_mask = filtered_df["SYMBOL"].isin(target_symbols)
        target_df = filtered_df[portfolio_mask].copy()

        if target_df.empty:
            logger.debug("Bhavcopy for %s valid, but contains no target portfolio symbols.", date_str)
            return None

        # Parse and standardize numeric fields
        numeric_cols = [
            "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE",
            "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS",
            "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"
        ]
        for ncol in numeric_cols:
            if ncol in target_df.columns:
                target_df[ncol] = pd.to_numeric(target_df[ncol].astype(str).str.replace(",", ""), errors="coerce")

        # Standardize date column
        target_df["DATE1"] = pd.to_datetime(target_df["DATE1"], format="mixed", errors="coerce")

        # Cache the filtered daily slice locally for fast re-runs
        try:
            target_df.to_csv(cache_file, index=False)
        except Exception as exc:
            logger.warning("Could not write cache file for %s: %s", date_str, exc)

        # Respectful throttle delay between queries
        time.sleep(self.delay_seconds)

        return target_df

    def fetch_date_range(
        self,
        start_date: date,
        end_date: date,
        target_symbols: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Iterate across each weekday between start_date and end_date, fetch daily bhavcopies,
        and assemble a consolidated historical DataFrame.

        Parameters:
            start_date (date): Start date (inclusive).
            end_date (date): End date (inclusive).
            target_symbols (List[str], optional): Portfolio symbols to retrieve.
            use_cache (bool): Whether to use cached daily files.

        Returns:
            pd.DataFrame: Consolidated historical OHLCV DataFrame sorted by (SYMBOL, DATE1).
        """
        if target_symbols is None:
            target_symbols = PORTFOLIO_SYMBOLS

        logger.info(
            "Commencing NSE Bhavcopy pipeline from %s to %s for %d symbols...",
            start_date.strftime("%d-%b-%Y"),
            end_date.strftime("%d-%b-%Y"),
            len(target_symbols)
        )

        daily_dfs: List[pd.DataFrame] = []
        current_date = start_date
        total_weekdays = 0
        fetched_days = 0

        while current_date <= end_date:
            # Skip Saturday (5) and Sunday (6)
            if current_date.weekday() < 5:
                total_weekdays += 1
                df_day = self.fetch_daily_bhavcopy(
                    target_date=current_date,
                    target_symbols=target_symbols,
                    use_cache=use_cache
                )
                if df_day is not None and not df_day.empty:
                    daily_dfs.append(df_day)
                    fetched_days += 1
            current_date += timedelta(days=1)

        if not daily_dfs:
            logger.warning("No data retrieved for the specified date range.")
            return pd.DataFrame(columns=BHAVCOPY_EXPECTED_COLUMNS)

        consolidated = pd.concat(daily_dfs, ignore_index=True)
        # Drop duplicates if any and sort by Symbol and Date
        consolidated = consolidated.sort_values(by=["SYMBOL", "DATE1"]).reset_index(drop=True)
        logger.info(
            "Bhavcopy ingestion complete: %d trading records retrieved across %d active trading sessions.",
            len(consolidated),
            fetched_days
        )
        return consolidated


# -----------------------------------------------------------------------------
# BENCHMARK DATA FETCHERS
# -----------------------------------------------------------------------------

def fetch_benchmark_nifty500(
    start_date: str,
    end_date: str,
    ticker: str = BENCHMARK_PRICE_TICKER
) -> pd.DataFrame:
    """
    Fetch the Nifty 500 Price Return benchmark index series using yfinance.

    Financial & Academic Rationale:
    Index series do not undergo corporate actions (such as stock splits, bonus shares,
    rights offerings, or spin-offs) that distort individual stock price series. Therefore,
    yfinance is reliable, standardized, and vetted specifically for the Nifty 500 (^CRSLDX)
    price return series.

    Parameters:
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        ticker (str): Index symbol, defaults to '^CRSLDX' (Nifty 500)

    Returns:
        pd.DataFrame: Cleaned DataFrame with Date as DatetimeIndex, and OHLCV columns.
    """
    logger.info("Downloading Nifty 500 benchmark (%s) from yfinance (%s to %s)...", ticker, start_date, end_date)
    raw_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if raw_df.empty:
        logger.error("yfinance returned an empty dataset for benchmark ticker %s.", ticker)
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (common in recent yfinance versions)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    # Reset index to make Date a column, then standardize column names
    df = raw_df.reset_index()
    df.columns = [str(c).capitalize() for c in df.columns]

    # Standardize Date column to tz-naive UTC midnight
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Benchmark downloaded successfully: %d trading sessions.", len(df))
    return df


def load_benchmark_tri(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the benchmark Nifty 500 Total Returns Index (TRI) from a local CSV file.

    Financial & Academic Rationale:
    The Total Return Index (TRI) accounts for dividend reinvestment in addition to price
    appreciation, representing the true opportunity cost and hurdle rate for an active
    equity portfolio manager. Automated scraping of niftyindices.com is intentionally
    avoided because the portal relies on heavy client-side JavaScript rendering and frequently
    changes internal endpoints. Reading the official CSV directly guarantees accuracy.

    Expected CSV columns:
        IndexName, Date, Total Returns Index, Net Total Return Index

    Parameters:
        csv_path (Path, optional): Path to the TRI CSV. Defaults to DEFAULT_TRI_CSV_PATH.

    Returns:
        pd.DataFrame: Cleaned TRI DataFrame with standardized Date and Total Returns Index.
    """
    if csv_path is None:
        csv_path = DEFAULT_TRI_CSV_PATH

    path_obj = Path(csv_path)
    if not path_obj.exists():
        logger.warning(
            "Benchmark TRI CSV file not found at '%s'. If you have downloaded it from "
            "niftyindices.com, please place it at this path or specify --tri-csv.",
            path_obj
        )
        return pd.DataFrame()

    logger.info("Loading benchmark TRI data from local CSV: %s", path_obj)
    df = pd.read_csv(path_obj)
    df.columns = [c.strip() for c in df.columns]

    # Validate required columns
    required_cols = ["IndexName", "Date", "Total Returns Index", "Net Total Return Index"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("TRI CSV missing expected columns: %s. Found: %s", missing, df.columns.tolist())
        return pd.DataFrame()

    # Parse dates and numeric columns
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce").dt.tz_localize(None)
    df["Total Returns Index"] = pd.to_numeric(df["Total Returns Index"].astype(str).str.replace(",", ""), errors="coerce")
    df["Net Total Return Index"] = pd.to_numeric(df["Net Total Return Index"].astype(str).str.replace(",", ""), errors="coerce")

    df = df.dropna(subset=["Date", "Total Returns Index"]).sort_values("Date").reset_index(drop=True)
    logger.info("Loaded %d TRI trading sessions from CSV.", len(df))
    return df


def fetch_benchmark_tri_automated(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Nifty 500 TRI historical data via browser automation, since
    niftyindices.com's TRI data is only reachable through its
    JavaScript-rendered UI, not a stable public API.
    """
    import tempfile
    import urllib.parse

    logger.info("Attempting automated browser retrieval of Nifty 500 TRI from niftyindices.com...")
    try:
        from playwright.sync_api import sync_playwright

        # Standardize dates into DD-Mon-YYYY (e.g. '24-Sep-2024') required by niftyindices portal
        if isinstance(start_date, (datetime, date)):
            start_dt = start_date
        else:
            start_dt = pd.to_datetime(start_date).date()

        if isinstance(end_date, (datetime, date)):
            end_dt = end_date
        else:
            end_dt = pd.to_datetime(end_date).date()

        start_str = start_dt.strftime("%d-%b-%Y")
        end_str = end_dt.strftime("%d-%b-%Y")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ]
            )
            try:
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 800},
                    accept_downloads=True,
                )
                page = context.new_page()
                page.set_default_timeout(30000)

                # Auto-accept dialog alerts (e.g., date range limit notices)
                page.on("dialog", lambda dialog: dialog.accept())

                # 1. Launch headless Chromium, navigate to historical data page
                logger.info("Navigating to https://niftyindices.com/reports/historical-data...")
                try:
                    page.goto("https://niftyindices.com/reports/historical-data", wait_until="domcontentloaded", timeout=30000)
                except Exception:
                    # Retry with commit if domcontentloaded stalls on external trackers
                    page.goto("https://niftyindices.com/reports/historical-data", wait_until="commit", timeout=20000)
                page.wait_for_timeout(1500)

                # 2. Click dropdown currently showing 'Historical Index Data' and select 'Total returns Index Values'
                menu_btn = page.locator("#HistoricalMenu, a.btn.btn-select")
                if menu_btn.count() > 0:
                    menu_btn.first.click()
                    page.wait_for_timeout(500)

                tri_opt = page.locator("#maindd li.form5, #maindd li:has-text('Total returns Index Values')")
                if tri_opt.count() > 0:
                    tri_opt.first.click()
                else:
                    page.evaluate("if (typeof ReturnIndextype === 'function') ReturnIndextype();")
                page.wait_for_timeout(1000)

                # 3. Set 'Select an Index Type' to 'Equity'
                page.wait_for_selector("#ddlHistoricalreturntypee option[value='Equity']", timeout=15000)
                page.select_option("#ddlHistoricalreturntypee", value="Equity")
                page.wait_for_timeout(1000)

                # 4. Set 'Select a Sub-Index' to 'Broad Based Indices' / 'Broad Market Indices'
                page.wait_for_selector("#ddlHistoricalreturntypeeSubindex option:not([value='0'])", timeout=15000)
                subindex_select = page.locator("#ddlHistoricalreturntypeeSubindex")
                options = subindex_select.locator("option").all_inner_texts()
                matched_sub = None
                for opt in options:
                    if "broad" in opt.lower():
                        matched_sub = opt.strip()
                        break
                if matched_sub:
                    subindex_select.select_option(label=matched_sub)
                else:
                    subindex_select.select_option(label="Broad Market Indices")
                page.wait_for_timeout(1000)

                # 5. Set 'Select an Index' to 'NIFTY 500'
                page.wait_for_selector("#ddlHistoricalreturntypeeindex option:not([value='0'])", timeout=15000)
                index_select = page.locator("#ddlHistoricalreturntypeeindex")
                idx_options = index_select.locator("option").all_inner_texts()
                matched_idx = None
                for opt in idx_options:
                    if "500" in opt:
                        matched_idx = opt.strip()
                        break
                if matched_idx:
                    index_select.select_option(label=matched_idx)
                else:
                    index_select.select_option(label="NIFTY 500")
                page.wait_for_timeout(500)

                # 6. Set the two date pickers to start_date and end_date
                page.evaluate(
                    """([startVal, endVal]) => {
                        $('#datepickerFromtotalindex').datepicker('setDate', startVal);
                        $('#datepickerTototalindex').datepicker('setDate', endVal);
                        $('#datepickerFromtotalindex').val(startVal);
                        $('#datepickerTototalindex').val(endVal);
                    }""",
                    [start_str, end_str]
                )
                page.wait_for_timeout(500)

                # 7. Click 'Submit'
                submit_btn = page.locator("#submit_totalindexhistorical, a.submitBtn:has-text('Submit')")
                submit_btn.first.click()

                # 8. Wait for results table to render and export CSV button to appear
                page.wait_for_selector("#exportTotalindex:visible", timeout=20000)
                page.wait_for_timeout(1000)

                # 9. Capture the downloaded CSV (Playwright expect_download pattern)
                df = None
                try:
                    with page.expect_download(timeout=10000) as download_info:
                        page.locator("#exportTotalindex").click()
                    download = download_info.value
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                        tmp_csv_path = Path(tf.name)
                    download.save_as(str(tmp_csv_path))
                    try:
                        df = pd.read_csv(tmp_csv_path)
                    finally:
                        if tmp_csv_path.exists():
                            tmp_csv_path.unlink()
                except Exception as dl_err:
                    logger.debug("Download capture fallback triggered: %s", dl_err)
                    export_elem = page.locator("#exportTotalindex")
                    href = export_elem.get_attribute("href")
                    if href and href.startswith("data:"):
                        csv_payload = urllib.parse.unquote(href.split(",", 1)[1])
                        df = pd.read_csv(io.StringIO(csv_payload))
                    else:
                        table_elem = page.locator("#historytotalindexexport")
                        if table_elem.count() > 0:
                            html_content = table_elem.inner_html()
                            dfs = pd.read_html(io.StringIO(f"<table>{html_content}</table>"))
                            if dfs:
                                df = dfs[0]

                if df is None or df.empty:
                    raise ValueError("No records extracted from niftyindices.com automated TRI response.")

                # Standardize column names to match: IndexName, Date, Total Returns Index, Net Total Return Index
                df.columns = [c.strip() for c in df.columns]
                col_rename = {}
                for c in df.columns:
                    clean_c = c.replace("_", " ").title()
                    if "Total Returns" in clean_c or "Total Return Index" in clean_c:
                        if "Net" in clean_c:
                            col_rename[c] = "Net Total Return Index"
                        else:
                            col_rename[c] = "Total Returns Index"
                    elif "Index Name" in clean_c or "Indexname" in clean_c:
                        col_rename[c] = "IndexName"
                    elif "Date" in clean_c:
                        col_rename[c] = "Date"
                    elif "Ntr" in clean_c:
                        col_rename[c] = "Net Total Return Index"
                df = df.rename(columns=col_rename)

                # Validate expected columns
                required_cols = ["IndexName", "Date", "Total Returns Index"]
                for rc in required_cols:
                    if rc not in df.columns:
                        raise ValueError(f"TRI DataFrame missing column '{rc}'. Available: {df.columns.tolist()}")

                if "Net Total Return Index" not in df.columns:
                    df["Net Total Return Index"] = float("nan")

                # Parse dates and numeric columns
                df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce").dt.tz_localize(None)
                df["Total Returns Index"] = pd.to_numeric(df["Total Returns Index"].astype(str).str.replace(",", ""), errors="coerce")
                df["Net Total Return Index"] = pd.to_numeric(df["Net Total Return Index"].astype(str).str.replace(",", ""), errors="coerce")

                df = df.dropna(subset=["Date", "Total Returns Index"]).sort_values("Date").reset_index(drop=True)
                logger.info("Automated TRI retrieval successful: %d daily sessions acquired.", len(df))
                return df
            finally:
                browser.close()

    except Exception as exc:
        logger.warning(
            "Automated benchmark TRI fetching failed (%s). Falling back to local manual CSV at '%s'.",
            exc,
            DEFAULT_TRI_CSV_PATH,
            exc_info=True
        )
        return load_benchmark_tri(DEFAULT_TRI_CSV_PATH)


def get_one_year_date_range() -> Tuple[date, date]:
    """
    Calculate the standard 1-year back date window from current execution date.
    Returns:
        Tuple[date, date]: (start_date, end_date)
    """
    today = date.today()
    # 365 days back
    one_year_ago = today - timedelta(days=365)
    return one_year_ago, today
