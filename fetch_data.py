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
import logging
import textwrap
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    BENCHMARK_PRICE_TICKER,
    BHAVCOPY_EXPECTED_COLUMNS,
    BHAVCOPY_FETCH_ATTEMPTS,
    BHAVCOPY_RETRY_BACKOFF_SECONDS,
    DEFAULT_TRI_CSV_PATH,
    NSE_BHAVCOPY_URL_TEMPLATE,
    NSE_HOME_URL,
    NSE_REQUEST_HEADERS,
    PORTFOLIO_SYMBOLS,
    RAW_BHAVCOPY_DIR,
    REQUEST_DELAY_SECONDS,
    SYMBOL_ALIASES,
    TRI_STALE_THRESHOLD_TRADING_DAYS,
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
        # Trading dates that could not be downloaded (bot-protection blocks, network errors)
        self.failed_dates: List[date] = []

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

    @staticmethod
    def _is_other_session(df: pd.DataFrame, target_date: date) -> bool:
        """True when a bhavcopy's DATE1 values belong to a session other than target_date."""
        dates = pd.to_datetime(df["DATE1"], format="mixed", errors="coerce").dropna().dt.date
        return not dates.empty and (dates != target_date).all()

    @staticmethod
    def _mark_holiday(holiday_flag_file: Path) -> None:
        try:
            holiday_flag_file.touch()
        except OSError:
            pass

    def _request_bhavcopy(self, target_date: date) -> Optional[requests.Response]:
        """
        Request one day's bhavcopy, retrying when the response is not a usable answer.

        Returns the response when NSE's origin answered (a CSV, or 404 for a holiday), or
        None when every attempt was blocked or failed. NSE's Akamai layer intermittently
        answers 403 'Access Denied' (an HTML page) to trading days; such responses are
        retried with backoff and a fresh cookie warm-up, and are never cached as holidays.
        """
        date_str = self._format_date(target_date)
        url = NSE_BHAVCOPY_URL_TEMPLATE.format(date_str=date_str)
        last_problem = None
        for attempt in range(1, BHAVCOPY_FETCH_ATTEMPTS + 1):
            if not self.session_initialized:
                self.warm_up_session()
            logger.debug("Requesting Bhavcopy for %s from NSE (attempt %d)...", date_str, attempt)
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 404:
                    return response
                if response.status_code == 200 and "SYMBOL" in response.text[:500]:
                    return response
                last_problem = f"HTTP {response.status_code}, non-CSV response"
                # Blocked or unexpected reply: refresh cookies before the next attempt
                self.session_initialized = False
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            if attempt < BHAVCOPY_FETCH_ATTEMPTS:
                time.sleep(BHAVCOPY_RETRY_BACKOFF_SECONDS * attempt)
        logger.warning("Could not fetch bhavcopy for %s after %d attempts (%s).",
                       date_str, BHAVCOPY_FETCH_ATTEMPTS, last_problem)
        return None

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
        # v2: flags written before holiday detection was fixed may mark blocked trading days
        # as holidays; the new name ignores them so those dates are re-checked.
        holiday_flag_file = self.cache_dir / f"holiday_v2_{date_str}.flag"

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
                    if self._is_other_session(df, target_date):
                        # Cached before date validation existed: a holiday served the previous session
                        self._mark_holiday(holiday_flag_file)
                        cache_file.unlink(missing_ok=True)
                        return None
                    return df
                except Exception as exc:
                    logger.warning("Failed to parse cached file %s (%s). Re-fetching.", cache_file, exc)

        response = self._request_bhavcopy(target_date)
        if response is None:
            self.failed_dates.append(target_date)
            return None
        if response.status_code == 404:
            # NSE's origin answers 404 for weekends and exchange holidays
            logger.debug("No bhavcopy for %s (404: market holiday/weekend).", date_str)
            self._mark_holiday(holiday_flag_file)
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

        # On some holidays NSE serves the previous session's file (e.g. 25-Dec returns 24-Dec);
        # accepting it would duplicate that session under a second request date
        if self._is_other_session(target_df, target_date):
            logger.debug("Bhavcopy requested for %s contains another session; treating as holiday.", date_str)
            self._mark_holiday(holiday_flag_file)
            return None

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
        # Drop duplicate sessions if any and sort by Symbol and Date
        consolidated = (
            consolidated.drop_duplicates(subset=["SYMBOL", "DATE1"], keep="last")
            .sort_values(by=["SYMBOL", "DATE1"])
            .reset_index(drop=True)
        )
        logger.info(
            "Bhavcopy ingestion complete: %d trading records retrieved across %d active trading sessions.",
            len(consolidated),
            fetched_days
        )
        if self.failed_dates:
            logger.warning(
                "%d trading day(s) could not be downloaded and are missing from the history "
                "(re-run to retry; they are not cached as holidays): %s",
                len(self.failed_dates),
                ", ".join(self._format_date(d) for d in self.failed_dates),
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


TRI_REDOWNLOAD_INSTRUCTIONS = (
    "Re-download the CSV from niftyindices.com (Historical Index Data -> Total returns Index Values "
    "-> Equity -> Broad Market Indices -> NIFTY 500) and replace data/nifty500_tri.csv before trusting "
    "any benchmark-relative performance figures (Sharpe, Treynor, XIRR vs benchmark)."
)
CONSOLE_BANNER_WIDTH = 115  # Matches main.py's console banners


@dataclass(frozen=True)
class TriStaleness:
    """How far the TRI series trails the date an analysis needs it to cover."""
    last_date: date
    reference_date: date
    trading_days_behind: int
    threshold: int = TRI_STALE_THRESHOLD_TRADING_DAYS

    @property
    def is_stale(self) -> bool:
        return self.trading_days_behind > self.threshold


def assess_tri_staleness(tri_df: pd.DataFrame, reference_date: Optional[date] = None) -> Optional[TriStaleness]:
    """
    Measure how many trading days the TRI data trails the analysis end date.

    The gap counts weekdays after the last available TRI date up to and including the
    reference date (numpy busday_count); exchange holidays are not modelled, so the
    figure can overstate the true gap by the number of holidays in between.

    Parameters:
        tri_df (pd.DataFrame): TRI data with a datetime 'Date' column.
        reference_date (date, optional): The date the analysis needs data through
            (e.g. the pipeline's end date). Defaults to today.

    Returns:
        TriStaleness, or None when there is no TRI data to assess.
    """
    if tri_df is None or tri_df.empty or "Date" not in tri_df.columns:
        return None
    last_date = pd.Timestamp(tri_df["Date"].max()).date()
    reference_date = pd.Timestamp(reference_date).date() if reference_date is not None else date.today()
    gap = 0
    if reference_date > last_date:
        gap = int(np.busday_count(last_date + timedelta(days=1), reference_date + timedelta(days=1)))
    return TriStaleness(last_date=last_date, reference_date=reference_date, trading_days_behind=gap)


def format_tri_staleness_warning(staleness: TriStaleness) -> str:
    """Build the console banner shown when the TRI benchmark data is stale."""
    body = (
        f"Last available date: {staleness.last_date:%d-%b-%Y}. This is {staleness.trading_days_behind} "
        f"trading days behind the requested analysis end date ({staleness.reference_date:%d-%b-%Y}). "
        + TRI_REDOWNLOAD_INSTRUCTIONS
    )
    lines = [" [!] WARNING: TRI benchmark data is stale."]
    lines += textwrap.wrap(body, width=CONSOLE_BANNER_WIDTH - 6,
                           initial_indent=" [!] ", subsequent_indent=" [!] ")
    border = "!" * CONSOLE_BANNER_WIDTH
    return "\n".join([border, *lines, border])


def load_benchmark_tri(
    csv_path: Optional[Path] = None,
    as_of: Optional[date] = None,
    warn_if_stale: bool = True,
) -> pd.DataFrame:
    """
    Load the benchmark Nifty 500 Total Returns Index (TRI) from a local CSV file.

    Financial & Academic Rationale:
    The Total Return Index (TRI) accounts for dividend reinvestment in addition to price
    appreciation, representing the true opportunity cost and hurdle rate for an active
    equity portfolio manager. This local CSV (downloaded from niftyindices.com) is the
    default source because it needs no browser and is immune to the portal's bot protection;
    fetch_benchmark_tri_automated() produces the same schema and parses through this function.

    Expected CSV columns:
        IndexName, Date, Total Returns Index, Net Total Return Index

    The file is maintained by hand, so after loading it is checked for staleness against
    the analysis end date; a stale file triggers a console warning banner but is still
    returned, since older benchmark data remains usable for historical regression work.

    Parameters:
        csv_path (Path, optional): Path to the TRI CSV. Defaults to DEFAULT_TRI_CSV_PATH.
        as_of (date, optional): The date this run needs TRI data through (the pipeline's
            end date). Defaults to today.
        warn_if_stale (bool): Print the staleness banner when the data trails as_of by more
            than TRI_STALE_THRESHOLD_TRADING_DAYS trading days.

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

    if warn_if_stale:
        try:
            staleness = assess_tri_staleness(df, as_of)
        except Exception as exc:  # The check is advisory; never let it fail the load
            logger.warning("Could not assess TRI staleness against %r: %s", as_of, exc)
            staleness = None
        if staleness is not None and staleness.is_stale:
            logger.warning("TRI data is stale: last date %s, %d trading days behind %s.",
                           staleness.last_date, staleness.trading_days_behind, staleness.reference_date)
            print(format_tri_staleness_warning(staleness))
    return df


TRI_HISTORICAL_URL = "https://niftyindices.com/reports/historical-data"
TRI_ENDPOINT_FRAGMENT = "/BackPage/getTotalReturnIndexString"
TRI_MAX_WINDOW_DAYS = 365        # Portal rejects ranges over 365 days (alert since Aug-2025)
TRI_SESSION_ATTEMPTS = 4         # Akamai intermittently rejects a fresh browser session
TRI_SESSION_BACKOFF_SECONDS = 15
TRI_UI_PAUSE_MS = 800            # Unhurried pacing between UI steps; rapid-fire input gets flagged


class _TriSessionRejected(RuntimeError):
    """niftyindices.com's bot protection rejected the current browser session (retryable)."""


def fetch_benchmark_tri_automated(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Nifty 500 TRI historical data via browser automation, since
    niftyindices.com's TRI data is only reachable through its
    JavaScript-rendered UI, not a stable public API.

    Findings that shape this implementation (verified against the live site, Sep-2026):
      - The site sits behind Akamai Bot Manager. Playwright's default headless build
        (chromium_headless_shell) is blocked outright (403 'Access Denied' or a stalled
        connection that surfaces as a navigation timeout). The full Chromium build in
        new-headless mode (channel="chromium") with 'HeadlessChrome' stripped from the
        user agent passes, though Akamai still rejects some sessions (at page load or
        later on the data call), hence whole-session retries with a fresh browser context,
        keeping any windows already captured.
      - The data endpoint (/BackPage/getTotalReturnIndexString) also requires the Akamai
        session cookies, so direct HTTP POSTs are rejected; driving the real page is needed.
      - The portal rejects date ranges longer than 365 days, so requests are chunked.
      - The 'csv format' link builds the file client-side from the rendered table, so
        expect_download() captures exactly the manual-download CSV format.

    Falls back to the local manual CSV (load_benchmark_tri) on any failure.
    """
    import tempfile

    logger.info("Attempting automated browser retrieval of Nifty 500 TRI from niftyindices.com...")
    try:
        from playwright.sync_api import sync_playwright

        start_dt = pd.Timestamp(start_date).date()  # accepts str, date or datetime
        end_dt = pd.Timestamp(end_date).date()
        if start_dt > end_dt:
            raise ValueError(f"start_date {start_dt} is after end_date {end_dt}")

        # Split into windows the portal accepts (<= 365 days each)
        windows = []
        chunk_start = start_dt
        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + timedelta(days=TRI_MAX_WINDOW_DAYS), end_dt)
            windows.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)

        with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp_dir:
            # Full Chromium build (new headless), not the headless shell Akamai blocks
            browser = p.chromium.launch(
                headless=True,
                channel="chromium",
                args=["--disable-blink-features=AutomationControlled"],
            )
            chunk_frames = {}  # (start, end) window -> DataFrame, or None when the window has no data
            try:
                probe = browser.new_page()
                user_agent = str(probe.evaluate("navigator.userAgent")).replace("HeadlessChrome", "Chrome")
                probe.close()

                for attempt in range(1, TRI_SESSION_ATTEMPTS + 1):
                    context = browser.new_context(
                        user_agent=user_agent,
                        viewport={"width": 1366, "height": 900},
                        locale="en-IN",
                        timezone_id="Asia/Kolkata",
                        accept_downloads=True,
                    )
                    blocked_urls = []  # Akamai 403/429s seen in this session, on any request
                    try:
                        page = context.new_page()
                        page.set_default_timeout(30000)
                        page.on("dialog", lambda dialog: (
                            logger.warning("niftyindices.com alert: %s", dialog.message), dialog.accept()))
                        page.on("response", lambda r: blocked_urls.append(r.url)
                                if r.status in (403, 429) and "niftyindices.com" in r.url else None)

                        # 1. Navigate. Akamai rejects some sessions outright (403), stalls others,
                        #    or passes the document but blocks its scripts; all count as rejections.
                        try:
                            response = page.goto(TRI_HISTORICAL_URL, wait_until="domcontentloaded", timeout=45000)
                            status = response.status if response else None
                            if status != 200:
                                raise _TriSessionRejected(f"page returned HTTP {status}")
                            page.wait_for_selector("#HistoricalMenu", state="attached", timeout=15000)
                            page.wait_for_load_state("load", timeout=45000)
                            page.wait_for_function(
                                "() => !!(window.jQuery && window.jQuery.fn && window.jQuery.fn.datepicker)",
                                timeout=20000)
                        except _TriSessionRejected:
                            raise
                        except Exception as nav_err:
                            raise _TriSessionRejected(
                                f"{type(nav_err).__name__}: {str(nav_err).splitlines()[0]}") from nav_err
                        page.wait_for_timeout(TRI_UI_PAUSE_MS)

                        # 2. Open the custom report-type menu and choose 'Total returns Index Values'
                        #    (a jQuery btn-select widget, not a <select>: the click target is the parent anchor)
                        page.locator("a.btn-select:has(#HistoricalMenu)").click()
                        tri_option = page.locator("#maindd li.form5")
                        try:
                            tri_option.wait_for(state="visible", timeout=5000)
                            tri_option.click()
                        except Exception:
                            # Menu animation did not open; fire the site's own <li> click handler instead
                            page.evaluate("$('#maindd li.form5').trigger('click')")
                        page.locator("#TotalReturnindexvalue").wait_for(state="visible", timeout=15000)
                        page.wait_for_timeout(TRI_UI_PAUSE_MS)

                        # 3-5. Native <select> cascade: Equity -> Broad Market Indices -> NIFTY 500
                        page.wait_for_selector("#ddlHistoricalreturntypee option[value='Equity']", state="attached", timeout=15000)
                        page.select_option("#ddlHistoricalreturntypee", value="Equity")
                        page.wait_for_timeout(TRI_UI_PAUSE_MS)

                        subindex_select = page.locator("#ddlHistoricalreturntypeeSubindex")
                        page.wait_for_selector("#ddlHistoricalreturntypeeSubindex option:nth-child(2)", state="attached", timeout=15000)
                        sub_labels = [o.strip() for o in subindex_select.locator("option").all_inner_texts()]
                        broad_label = next((o for o in sub_labels if "broad" in o.lower()), "Broad Market Indices")
                        subindex_select.select_option(label=broad_label)
                        page.wait_for_timeout(TRI_UI_PAUSE_MS)

                        index_select = page.locator("#ddlHistoricalreturntypeeindex")
                        page.wait_for_selector("#ddlHistoricalreturntypeeindex option:nth-child(2)", state="attached", timeout=15000)
                        index_select.select_option(label="NIFTY 500")
                        page.wait_for_timeout(TRI_UI_PAUSE_MS)

                        for win_start, win_end in windows:
                            if (win_start, win_end) in chunk_frames:
                                continue  # captured by an earlier session

                            # 6. Set dates as JS Date objects (the widget's own format is mm/dd/yy;
                            #    passing 'dd-Mon-yyyy' strings is silently misparsed)
                            page.evaluate(
                                """([s, e]) => {
                                    $('#datepickerFromtotalindex').datepicker('setDate', new Date(s[0], s[1] - 1, s[2]));
                                    $('#datepickerTototalindex').datepicker('setDate', new Date(e[0], e[1] - 1, e[2]));
                                }""",
                                [[win_start.year, win_start.month, win_start.day],
                                 [win_end.year, win_end.month, win_end.day]],
                            )
                            page.wait_for_timeout(TRI_UI_PAUSE_MS)

                            # 7-8. Submit and wait for the data call to return
                            with page.expect_response(lambda r: TRI_ENDPOINT_FRAGMENT in r.url, timeout=45000) as resp_info:
                                page.locator("#submit_totalindexhistorical").click()
                            api_response = resp_info.value
                            if api_response.status in (403, 429):
                                # Akamai flagged the session mid-flow
                                raise _TriSessionRejected(f"data endpoint returned HTTP {api_response.status}")
                            if api_response.status != 200:
                                raise RuntimeError(f"TRI data endpoint returned HTTP {api_response.status}")
                            try:
                                records = api_response.json()
                            except Exception:
                                records = None
                            if isinstance(records, list) and not records:
                                logger.info("No TRI records for %s to %s; skipping window.", win_start, win_end)
                                chunk_frames[(win_start, win_end)] = None
                                continue

                            page.locator("#exportTotalindex").wait_for(state="visible", timeout=15000)
                            page.wait_for_function(
                                "document.querySelectorAll('#historytotalindexexport tbody tr').length > 0",
                                timeout=15000)

                            # 9. Capture the client-side generated CSV via expect_download
                            with page.expect_download(timeout=15000) as download_info:
                                page.locator("#exportTotalindex").click()
                            chunk_path = Path(tmp_dir) / f"tri_{win_start:%Y%m%d}_{win_end:%Y%m%d}.csv"
                            download_info.value.save_as(str(chunk_path))
                            chunk_frames[(win_start, win_end)] = pd.read_csv(chunk_path)
                            logger.info("Captured TRI CSV for %s to %s (%d rows).",
                                        win_start, win_end, len(chunk_frames[(win_start, win_end)]))
                        break  # every window captured
                    except Exception as err:
                        # Akamai can block any request mid-flow (e.g. the dropdown-population
                        # calls, surfacing only as a site 'Error Occurred' alert and a timeout).
                        # Failures in a session that saw such blocks are retryable; anything
                        # else is a genuine breakage and goes straight to the fallback.
                        if isinstance(err, _TriSessionRejected):
                            rejection = err
                        elif blocked_urls:
                            rejection = _TriSessionRejected(
                                f"{type(err).__name__} after blocked request(s): {blocked_urls[0]}")
                        else:
                            raise
                        logger.info("niftyindices.com session %d/%d rejected by bot protection (%s).",
                                    attempt, TRI_SESSION_ATTEMPTS, rejection)
                        if attempt == TRI_SESSION_ATTEMPTS:
                            raise RuntimeError(
                                f"niftyindices.com rejected {TRI_SESSION_ATTEMPTS} browser sessions "
                                f"(Akamai bot protection); last: {rejection}") from rejection
                        time.sleep(TRI_SESSION_BACKOFF_SECONDS * attempt)
                    finally:
                        context.close()
            finally:
                browser.close()

            frames = [f for f in chunk_frames.values() if f is not None]
            if not frames:
                raise ValueError("No records extracted from niftyindices.com automated TRI response.")

            # Parse through the same routine as the manual CSV so both paths share one contract
            combined_path = Path(tmp_dir) / "tri_combined.csv"
            pd.concat(frames, ignore_index=True).to_csv(combined_path, index=False)
            df = load_benchmark_tri(combined_path, warn_if_stale=False)

        if df.empty:
            raise ValueError("Automated TRI CSV could not be parsed into the expected schema.")
        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        logger.info("Automated TRI retrieval successful: %d daily sessions acquired.", len(df))
        return df

    except Exception as exc:
        logger.warning(
            "Automated benchmark TRI fetching failed (%s). Falling back to local manual CSV at '%s'.",
            exc,
            DEFAULT_TRI_CSV_PATH,
            exc_info=True
        )
        return load_benchmark_tri(DEFAULT_TRI_CSV_PATH, as_of=end_date)


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
