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
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    BENCHMARK_INDEX_NAME,
    BENCHMARK_PRICE_TICKER,
    BHAVCOPY_EXPECTED_COLUMNS,
    BHAVCOPY_FETCH_ATTEMPTS,
    BHAVCOPY_RETRY_BACKOFF_SECONDS,
    DEFAULT_TRI_CSV_PATH,
    NSE_BHAVCOPY_URL_TEMPLATE,
    NSE_HOME_URL,
    NSE_INDEX_CLOSE_URL_TEMPLATE,
    NSE_REQUEST_HEADERS,
    NSE_SPECIAL_WEEKEND_SESSIONS,
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
    def _select_symbols(day_df: pd.DataFrame, target_symbols: List[str], date_str: str) -> Optional[pd.DataFrame]:
        """
        Map historical ticker aliases to current tickers (e.g. GET&D -> GVT&D, ITDCEM -> CEMPRO)
        and keep only the requested symbols; None when none of them traded that day.
        """
        df = day_df.copy()
        df["SYMBOL"] = df["SYMBOL"].replace(SYMBOL_ALIASES)
        target_df = df[df["SYMBOL"].isin(target_symbols)].copy()
        if target_df.empty:
            logger.debug("Bhavcopy for %s valid, but contains no target portfolio symbols.", date_str)
            return None
        return target_df

    @staticmethod
    def _mark_holiday(holiday_flag_file: Path, session_date: date) -> None:
        """
        Cache session_date as a market holiday, except for today or yesterday: NSE publishes a
        day's files in the evening (IST), so a 404 or previous-session file for a very recent
        date may just mean "not published yet". Caching that would drop a real trading day
        permanently; leaving it unflagged means it is simply re-checked on the next run.
        """
        if session_date >= date.today() - timedelta(days=1):
            logger.info("No NSE file yet for %s; not caching it as a holiday (may be unpublished).",
                        session_date.strftime("%d-%b-%Y"))
            return
        try:
            holiday_flag_file.touch()
        except OSError:
            pass

    def _request_bhavcopy(self, target_date: date) -> Optional[requests.Response]:
        """Request one day's Full Bhavcopy (see _request_archive for retry semantics)."""
        date_str = self._format_date(target_date)
        return self._request_archive(NSE_BHAVCOPY_URL_TEMPLATE.format(date_str=date_str), f"bhavcopy for {date_str}", "SYMBOL")

    def _request_archive(self, url: str, label: str, header_token: str) -> Optional[requests.Response]:
        """
        Request one NSE archive CSV, retrying when the response is not a usable answer.

        Returns the response when NSE's origin answered (a CSV whose first line contains
        header_token, or 404 for a holiday), or None when every attempt was blocked or failed.
        NSE's Akamai layer intermittently answers 403 'Access Denied' (an HTML page) to trading
        days; such responses are retried with backoff and a fresh cookie warm-up, and are never
        cached as holidays.
        """
        last_problem = None
        for attempt in range(1, BHAVCOPY_FETCH_ATTEMPTS + 1):
            if not self.session_initialized:
                self.warm_up_session()
            logger.debug("Requesting %s from NSE (attempt %d)...", label, attempt)
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 404:
                    return response
                if response.status_code == 200 and header_token in response.text[:500]:
                    return response
                last_problem = f"HTTP {response.status_code}, non-CSV response"
                # Blocked or unexpected reply: refresh cookies before the next attempt
                self.session_initialized = False
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            if attempt < BHAVCOPY_FETCH_ATTEMPTS:
                time.sleep(BHAVCOPY_RETRY_BACKOFF_SECONDS * attempt)
        logger.warning("Could not fetch %s after %d attempts (%s).", label, BHAVCOPY_FETCH_ATTEMPTS, last_problem)
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

        date_str = self._format_date(target_date)
        # The cache holds the whole day's equity file (every symbol), so a later change to the
        # symbol universe is still served correctly from cache. ("bhav_eq_" replaces the older
        # "bhav_" files, which held only the universe current at download time and would
        # silently return no rows for symbols added later.)
        cache_file = self.cache_dir / f"bhav_eq_{date_str}.csv"
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
                    day_df = pd.read_csv(cache_file)
                    day_df["DATE1"] = pd.to_datetime(day_df["DATE1"], format="mixed", errors="coerce")
                    return self._select_symbols(day_df, target_symbols, date_str)
                except Exception as exc:
                    logger.warning("Failed to parse cached file %s (%s). Re-fetching.", cache_file, exc)

        response = self._request_bhavcopy(target_date)
        if response is None:
            self.failed_dates.append(target_date)
            return None
        if response.status_code == 404:
            # NSE's origin answers 404 for weekends and exchange holidays
            logger.debug("No bhavcopy for %s (404: market holiday/weekend).", date_str)
            self._mark_holiday(holiday_flag_file, target_date)
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
            day_df = raw_df[series_mask].copy()
        else:
            day_df = raw_df.copy()

        # Parse and standardize numeric fields
        numeric_cols = [
            "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE",
            "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS",
            "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"
        ]
        for ncol in numeric_cols:
            if ncol in day_df.columns:
                day_df[ncol] = pd.to_numeric(day_df[ncol].astype(str).str.replace(",", ""), errors="coerce")

        # Standardize date column
        day_df["DATE1"] = pd.to_datetime(day_df["DATE1"], format="mixed", errors="coerce")

        # On some holidays NSE serves the previous session's file (e.g. 25-Dec returns 24-Dec);
        # accepting it would duplicate that session under a second request date
        if self._is_other_session(day_df, target_date):
            logger.debug("Bhavcopy requested for %s contains another session; treating as holiday.", date_str)
            self._mark_holiday(holiday_flag_file, target_date)
            return None

        # Cache the full day's equity file locally for fast re-runs
        try:
            day_df.to_csv(cache_file, index=False)
        except Exception as exc:
            logger.warning("Could not write cache file for %s: %s", date_str, exc)

        # Respectful throttle delay between queries
        time.sleep(self.delay_seconds)

        return self._select_symbols(day_df, target_symbols, date_str)

    def _trading_calendar(self, start_date: date, end_date: date) -> List[date]:
        """Weekdays plus listed special weekend sessions (the same calendar as fetch_date_range)."""
        days, current = [], start_date
        while current <= end_date:
            if current.weekday() < 5 or current in NSE_SPECIAL_WEEKEND_SESSIONS:
                days.append(current)
            current += timedelta(days=1)
        return days

    def fetch_index_closes(
        self,
        start_date: date,
        end_date: date,
        index_name: str = BENCHMARK_INDEX_NAME,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Daily OHLC of one NSE index from NSE's official 'ind_close_all' archive files.

        Uses the same trading calendar, holiday flags (404 = holiday), block-retry logic and
        wrong-session check as the Bhavcopy download, so benchmark sessions line up exactly
        with the stock price history. Each day's full file (all indices) is cached.

        Returns:
            pd.DataFrame: Date, Open, High, Low, Close for index_name, oldest first
            (empty when nothing could be retrieved).
        """
        rows, failed = [], []
        for day in self._trading_calendar(start_date, end_date):
            date_str = self._format_date(day)
            holiday_flag_file = self.cache_dir / f"holiday_v2_{date_str}.flag"
            cache_file = self.cache_dir / f"indices_{date_str}.csv"
            if use_cache and holiday_flag_file.exists():
                continue
            day_df = None
            if use_cache and cache_file.exists():
                try:
                    day_df = pd.read_csv(cache_file)
                except Exception as exc:
                    logger.warning("Failed to parse cached index file %s (%s). Re-fetching.", cache_file, exc)
            if day_df is None:
                response = self._request_archive(NSE_INDEX_CLOSE_URL_TEMPLATE.format(ddmmyyyy=day.strftime("%d%m%Y")),
                                                 f"index closes for {date_str}", "Index Name")
                if response is None:
                    failed.append(day)
                    continue
                if response.status_code == 404:
                    self._mark_holiday(holiday_flag_file, day)
                    continue
                day_df = pd.read_csv(io.StringIO(response.text))
                day_df.columns = [c.strip() for c in day_df.columns]
                file_dates = pd.to_datetime(day_df["Index Date"], format="%d-%m-%Y", errors="coerce").dropna().dt.date
                if not file_dates.empty and (file_dates != day).all():
                    logger.debug("Index file requested for %s contains another session; treating as holiday.", date_str)
                    self._mark_holiday(holiday_flag_file, day)
                    continue
                try:
                    day_df.to_csv(cache_file, index=False)
                except OSError as exc:
                    logger.warning("Could not write index cache for %s: %s", date_str, exc)
                time.sleep(self.delay_seconds)
            day_df.columns = [c.strip() for c in day_df.columns]
            match = day_df[day_df["Index Name"].astype(str).str.strip().str.lower() == index_name.lower()]
            if match.empty:
                logger.warning("%s not found in NSE index file for %s.", index_name, date_str)
                continue
            r = match.iloc[0]
            rows.append({"Date": pd.Timestamp(day), "Open": r["Open Index Value"], "High": r["High Index Value"],
                         "Low": r["Low Index Value"], "Close": r["Closing Index Value"]})
        if failed:
            logger.warning("%d session(s) of %s closes could not be downloaded: %s", len(failed), index_name,
                           ", ".join(self._format_date(d) for d in failed))
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        return df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

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
            # Skip Saturday (5) and Sunday (6) unless NSE held a special session that day
            if current_date.weekday() < 5 or current_date in NSE_SPECIAL_WEEKEND_SESSIONS:
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
    Fetch the Nifty 500 price-return index series (start_date to end_date, inclusive).

    Source: NSE's official daily index closing files (NSEBhavcopyFetcher.fetch_index_closes),
    on exactly the same session calendar as the Bhavcopy stock prices. yfinance (^CRSLDX) is
    used only as a fallback if the official files cannot be retrieved; it has been observed to
    skip real NSE sessions (e.g. special and recent sessions), which silently shifts the RS window.

    Parameters:
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD' (inclusive)
        ticker (str): yfinance fallback symbol, defaults to '^CRSLDX' (Nifty 500)

    Returns:
        pd.DataFrame: Date (tz-naive) plus Open, High, Low, Close columns, oldest first.
    """
    start_d, end_d = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    # Primary source: NSE's official daily index closes, on the same session calendar as the
    # Bhavcopy stock prices (yfinance's ^CRSLDX has skipped real NSE sessions)
    try:
        official = NSEBhavcopyFetcher().fetch_index_closes(start_d, end_d)
    except Exception as exc:
        logger.warning("Official NSE index closes unavailable (%s); falling back to yfinance.", exc)
        official = pd.DataFrame()
    if not official.empty:
        logger.info("Nifty 500 benchmark from NSE official index closes: %d sessions (%s to %s).",
                    len(official), official["Date"].min().date(), official["Date"].max().date())
        return official

    logger.info("Downloading Nifty 500 benchmark (%s) from yfinance (%s to %s)...", ticker, start_date, end_date)
    # yfinance treats `end` as exclusive; add a day so end_date itself is included
    yf_end = (end_d + timedelta(days=1)).isoformat()
    raw_df = yf.download(ticker, start=start_date, end=yf_end, auto_adjust=True, progress=False)

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
    logger.info("Attempting automated browser retrieval of Nifty 500 TRI from niftyindices.com...")
    try:
        return _fetch_tri_via_browser(start_date, end_date)
    except Exception as exc:
        logger.warning(
            "Automated benchmark TRI fetching failed (%s). Falling back to local manual CSV at '%s'.",
            exc,
            DEFAULT_TRI_CSV_PATH,
            exc_info=True
        )
        return load_benchmark_tri(DEFAULT_TRI_CSV_PATH, as_of=end_date)


def _fetch_tri_via_browser(start_date, end_date) -> pd.DataFrame:
    """The browser retrieval behind fetch_benchmark_tri_automated(); raises on any failure
    instead of falling back, so callers that must know whether fresh data arrived can tell."""
    import tempfile

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



def get_one_year_date_range(end_date: Optional[date] = None) -> Tuple[date, date]:
    """
    Calculate the standard 1-year window ending on end_date (default: today).

    Returns:
        Tuple[date, date]: (start_date, end_date)
    """
    end = pd.Timestamp(end_date).date() if end_date is not None else date.today()
    return end - timedelta(days=365), end


def update_tri_csv(end_date: Optional[date] = None, csv_path: Optional[Path] = None,
                   fetch: Callable[[date, date], pd.DataFrame] = None) -> Tuple[int, Optional[date]]:
    """
    Append the latest Nifty 500 TRI sessions from niftyindices.com to the local TRI CSV (the file
    the pipeline and dashboard read), keeping its official manual-download format.

    Fetches from a week before the CSV's last date to end_date. Overlapping sessions must agree
    with the stored values (within 0.01%); if they do not, nothing is written, since a mismatch
    means the source changed or the parse is wrong. Raises on any failure and leaves the CSV
    untouched; the file is replaced atomically.

    Returns:
        (sessions_added, last_date_in_csv)
    """
    csv_path = Path(csv_path or DEFAULT_TRI_CSV_PATH)
    end = pd.Timestamp(end_date).date() if end_date is not None else date.today()
    fetch = fetch or _fetch_tri_via_browser
    existing = load_benchmark_tri(csv_path, warn_if_stale=False) if csv_path.exists() else pd.DataFrame()
    last = existing["Date"].max().date() if not existing.empty else None
    start = (last - timedelta(days=7)) if last else end - timedelta(days=365)
    if last and last >= end:
        return 0, last

    fetched = fetch(start, end)
    if fetched is None or fetched.empty:
        raise ValueError("niftyindices.com returned no TRI sessions")
    fetched = fetched.copy()
    fetched["Date"] = pd.to_datetime(fetched["Date"]).dt.tz_localize(None)

    if not existing.empty:
        overlap = existing.merge(fetched, on="Date", suffixes=("_old", "_new"))
        for col in ["Total Returns Index", "Net Total Return Index"]:
            rel = ((overlap[f"{col}_new"] - overlap[f"{col}_old"]).abs() / overlap[f"{col}_old"])
            if (rel > 1e-4).any():
                bad = overlap.loc[rel > 1e-4, "Date"].dt.date.tolist()
                raise ValueError(f"Fetched {col} disagrees with the stored CSV on {bad}; CSV left unchanged")

    merged = (pd.concat([existing, fetched], ignore_index=True)
              .drop_duplicates(subset=["Date"], keep="first").sort_values("Date").reset_index(drop=True))
    added = len(merged) - len(existing)
    if added == 0:
        return 0, last  # nothing new (e.g. today's value not published yet): leave the file as is
    out = pd.DataFrame({
        "IndexName": merged["IndexName"].fillna("NIFTY 500") if "IndexName" in merged else "NIFTY 500",
        "Date": merged["Date"].dt.strftime("%d-%b-%Y"),
        "Total Returns Index": merged["Total Returns Index"].round(2),
        "Net Total Return Index": merged["Net Total Return Index"].round(2),
    })
    tmp = csv_path.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False, float_format="%.2f")  # two decimals, as in the official download
    tmp.replace(csv_path)
    new_last = merged["Date"].max().date()
    logger.info("TRI CSV updated: %d session(s) added, now through %s.", added, new_last)
    return added, new_last


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["--update-tri"]:
        # python fetch_data.py --update-tri [YYYY-MM-DD]: append new TRI sessions to the local CSV
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        target = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
        n, through = update_tri_csv(target)
        print(f"Nifty 500 TRI: {n} new session(s); CSV now through {through}")
