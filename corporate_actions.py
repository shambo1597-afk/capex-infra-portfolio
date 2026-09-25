"""
Corporate-action (split / bonus / demerger) adjustment of NSE Bhavcopy prices.

NSE's Bhavcopy is NOT adjusted for corporate actions: on an ex-date, PREV_CLOSE is the previous
session's unadjusted close. A 1-for-10 split therefore shows as a -90% day (CESC, 17-Sep-2021),
a 2:1 bonus as -67% (BEL, 15-Sep-2022), and a 1:3 bonus as -25% (POWERGRID, 29-Jul-2021).
Returns, relative strength, volatility, ATR, ADX and support/resistance computed across such a
date are wrong unless the earlier prices are scaled.

Source: NSE's corporate-actions API (one JSON per symbol, cached under data/corporate_actions/).
Adjustment factor F applied to every price before the ex-date (quantities are divided by F):
  - Face value split / consolidation "From Rs A To Rs B":   F = B / A
  - Bonus "a:b" (a new shares for every b held):              F = b / (a + b)
  - Demerger / scheme of arrangement (the price drop is the value spun off, which the text does
    not state): F = the stock's own ex-date close / previous close, i.e. that one session's
    move is neutralised (its genuine market move is lost; documented, not guessed)
Rights issues and dividends are not adjusted (their price effect is small and needs the issue
price and entitlement, which are not modelled).
"""

import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger("corporate_actions")

CORPORATE_ACTIONS_DIR = DATA_DIR / "corporate_actions"
NSE_CORPORATE_ACTIONS_URL = ("https://www.nseindia.com/api/corporates-corporateActions?index=equities"
                             "&symbol={symbol}&from_date={from_date}&to_date={to_date}")
HISTORY_START = date(2019, 1, 1)
CACHE_MAX_AGE_HOURS = 20  # re-fetched about daily, so newly announced actions are picked up
PRICE_COLUMNS = ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE"]
QUANTITY_COLUMNS = ["TTL_TRD_QNTY", "DELIV_QTY"]

_SPLIT = re.compile(r"(?:split|sub-division|consolidation).*?from\s*r[es]\.?\s*([\d.]+).*?to\s*r[es]\.?\s*([\d.]+)",
                    re.IGNORECASE | re.DOTALL)
_BONUS = re.compile(r"bonus\s*(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_GAP = re.compile(r"demerger|scheme of arrangement|composite scheme", re.IGNORECASE)


def parse_action(subject: str) -> Optional[dict]:
    """
    Classify one corporate-action subject. Returns {"kind": "split"|"bonus", "factor": F} or
    {"kind": "gap"} (demerger/arrangement: factor taken from the ex-date price gap), or None for
    actions that are not adjusted (dividends, rights, AGMs, ...).
    """
    text = " ".join(str(subject).split())
    m = _SPLIT.search(text)
    if m:
        old, new = float(m.group(1)), float(m.group(2))
        if old > 0 and new > 0 and old != new:
            return {"kind": "split", "factor": new / old}
    m = _BONUS.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0 and b > 0:
            return {"kind": "bonus", "factor": b / (a + b)}
    if _GAP.search(text):
        return {"kind": "gap"}
    return None


def fetch_corporate_actions(symbol: str, use_cache: bool = True, to_date: Optional[date] = None) -> Optional[List[dict]]:
    """
    Adjusting actions for one symbol as [{"ex_date": date, "kind", "factor"?, "subject"}], oldest
    first; None when NSE could not be reached and there is no cache (never a guessed empty list).
    """
    from fundamentals import _fetch_nse_json  # NSE session with the Akamai cookies

    CORPORATE_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache = CORPORATE_ACTIONS_DIR / f"{symbol}.json"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600
    url = NSE_CORPORATE_ACTIONS_URL.format(symbol=requests_quote(symbol), from_date=f"{HISTORY_START:%d-%m-%Y}",
                                           to_date=f"{(to_date or date.today()) + timedelta(days=60):%d-%m-%Y}")
    payload = _fetch_nse_json(url, cache, f"corporate actions for {symbol}", use_cache=use_cache and fresh)
    if payload is None and cache.exists():
        logger.warning("Using stale corporate-action cache for %s (NSE unreachable).", symbol)
        payload = json.loads(cache.read_text(encoding="utf-8"))
    if payload is None:
        return None
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    actions = []
    for row in rows:
        parsed = parse_action(row.get("subject", ""))
        ex = pd.to_datetime(row.get("exDate"), format="%d-%b-%Y", errors="coerce")
        if parsed and pd.notna(ex):
            actions.append({"ex_date": ex.date(), "subject": " ".join(str(row.get("subject")).split()), **parsed})
    return sorted(actions, key=lambda a: a["ex_date"])


def requests_quote(symbol: str) -> str:
    from urllib.parse import quote
    return quote(symbol, safe="")


def adjust_for_corporate_actions(prices: pd.DataFrame, actions: Optional[Dict[str, List[dict]]] = None,
                                 use_cache: bool = True) -> pd.DataFrame:
    """
    Return a copy of Bhavcopy rows (SYMBOL, DATE1, price and quantity columns) with every price
    before each split/bonus/demerger ex-date scaled by its factor, and PREV_CLOSE on the ex-date
    scaled too, so CLOSE / PREV_CLOSE is a true one-day return and price history is continuous.
    `actions` maps symbol -> fetch_corporate_actions() output (fetched when not given).
    Symbols whose actions could not be fetched are left unadjusted, with a warning.
    """
    if prices is None or prices.empty:
        return prices
    # Unique row labels: price frames stitched from daily files repeat index labels, and label-based
    # selection would then scale rows of other symbols
    df = prices.reset_index(drop=True)
    df["_date"] = pd.to_datetime(df["DATE1"], format="mixed").dt.date
    for col in PRICE_COLUMNS + QUANTITY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    last_date = max(df["_date"])
    for symbol, idx in df.groupby("SYMBOL").groups.items():
        sym_actions = actions.get(symbol) if actions is not None else fetch_corporate_actions(
            symbol, use_cache=use_cache, to_date=last_date)
        if sym_actions is None:
            logger.warning("%s: corporate actions unavailable; prices left unadjusted.", symbol)
            continue
        rows = df.loc[idx].sort_values("_date")
        for action in sym_actions:
            ex = action["ex_date"]
            on_ex = rows[rows["_date"] == ex]
            before = rows.index[rows["_date"] < ex]
            if on_ex.empty or len(before) == 0:
                continue  # ex-date outside the loaded window: nothing to join up
            if action["kind"] == "gap":
                ex_row = on_ex.iloc[0]
                factor = ex_row["CLOSE_PRICE"] / ex_row["PREV_CLOSE"]
            else:
                factor = action["factor"]
            price_cols = [c for c in PRICE_COLUMNS if c in df.columns]
            df.loc[before, price_cols] *= factor
            if "PREV_CLOSE" in df.columns:
                df.loc[on_ex.index, "PREV_CLOSE"] *= factor
            for q in [c for c in QUANTITY_COLUMNS if c in df.columns]:
                df.loc[before, q] /= factor
            rows = df.loc[idx].sort_values("_date")
    return df.drop(columns="_date")


def unexplained_jumps(prices: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Rows whose one-day return CLOSE / PREV_CLOSE - 1 exceeds `threshold` in size, excluding each
    symbol's first row (listing day, where PREV_CLOSE is the issue price). After adjustment these
    should be real events (results-day moves, circuit limits), so review any that appear.
    """
    df = prices.copy()
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed")
    df = df.sort_values(["SYMBOL", "DATE1"])
    df["ret"] = df["CLOSE_PRICE"] / df["PREV_CLOSE"] - 1
    first = df.groupby("SYMBOL").cumcount() == 0
    return df[~first & (df["ret"].abs() > threshold)][["SYMBOL", "DATE1", "PREV_CLOSE", "CLOSE_PRICE", "ret"]]


if __name__ == "__main__":
    # python corporate_actions.py SYMBOL ...: print the adjusting actions NSE reports
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for sym in sys.argv[1:]:
        for a in fetch_corporate_actions(sym, use_cache=False) or []:
            print(sym, a["ex_date"], a["kind"], round(a.get("factor", float("nan")), 4), "|", a["subject"])
