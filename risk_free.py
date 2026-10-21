"""
The 3-month risk-free rate: the 91-day Treasury Bill cut-off yield at RBI's latest weekly auction.

The professor's rule (14-Oct-2026): the risk-free rate is the annualised rate for the 3-month horizon, so a
91-day T-bill yield, not an overnight rate. RBI publishes each Wednesday auction as a press release
"Treasury Bills: Full Auction Result" with the cut-off price and its yield (YTM, annualised). This module
reads the newest release that has a 91-day yield, caches every value it reads in TBILL_CSV, and falls back
to the cached value (or config.RISK_FREE_3M_FALLBACK_PCT) when RBI cannot be reached.

The overnight rate (Nifty 1D Rate index) stays the return of the cash sleeve and of the liquid-fund
comparator: that is what parked cash actually earns.

Usage:  python risk_free.py
"""

import html
import logging
import re
from datetime import date
from typing import Optional, Tuple

import pandas as pd
import requests

from config import DATA_DIR, RISK_FREE_3M_FALLBACK_DATE, RISK_FREE_3M_FALLBACK_PCT, TRADING_DAYS_PER_YEAR

logger = logging.getLogger("risk_free")

TBILL_CSV = DATA_DIR / "factors" / "tbill_91d.csv"
RBI_LIST_URL = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _text(page: str) -> str:
    page = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))


def parse_auction(page: str) -> Optional[Tuple[date, float]]:
    """(release date, 91-day cut-off yield %) from one 'Full Auction Result' release; None if not filled in."""
    t = _text(page)
    d = re.search(r"Date\s*:\s*([A-Z][a-z]{2} \d{1,2}, \d{4})", t)
    y = re.search(r"Cut-off price \(₹\) / Yield.*?\(YTM:\s*([\d.]+)%\)", t)
    if not d or not y:
        return None
    return pd.Timestamp(d.group(1)).date(), float(y.group(1))


def fetch_latest(max_releases: int = 4, timeout: int = 30) -> Optional[Tuple[date, float]]:
    """The newest 91-day T-bill cut-off yield published by RBI (looks back over a few releases)."""
    listing = requests.get(RBI_LIST_URL, headers=HEADERS, timeout=timeout).text
    ids = re.findall(r"prid=(\d+)>Treasury Bills: Full Auction Result<", html.unescape(listing))
    for prid in ids[:max_releases]:
        page = requests.get(f"{RBI_LIST_URL}?prid={prid}", headers=HEADERS, timeout=timeout).text
        got = parse_auction(page)
        if got:
            return got
    return None


def update_cache() -> Optional[Tuple[date, float]]:
    """Fetch the latest auction and add it to TBILL_CSV (one row per auction date)."""
    try:
        got = fetch_latest()
    except requests.RequestException as exc:
        logger.warning("RBI T-bill auction results unavailable (%s); using the cached rate.", exc)
        return None
    if got is None:
        logger.warning("No 91-day T-bill yield in RBI's latest auction releases; using the cached rate.")
        return None
    cached = pd.read_csv(TBILL_CSV) if TBILL_CSV.exists() else pd.DataFrame(columns=["auction_date", "yield_pct"])
    row = pd.DataFrame([{"auction_date": got[0].isoformat(), "yield_pct": got[1]}])
    out = pd.concat([cached, row]).drop_duplicates("auction_date", keep="last").sort_values("auction_date")
    TBILL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TBILL_CSV, index=False)
    logger.info("91-day T-bill: %.4f%% (auction released %s).", got[1], got[0])
    return got


def risk_free_3m() -> Tuple[float, str]:
    """(annual 3-month risk-free rate as a fraction, its source label) from the cache or the fallback."""
    if TBILL_CSV.exists():
        cached = pd.read_csv(TBILL_CSV).sort_values("auction_date")
        if not cached.empty:
            last = cached.iloc[-1]
            return float(last["yield_pct"]) / 100, f"91-day T-bill cut-off yield, RBI auction of {last['auction_date']}"
    return RISK_FREE_3M_FALLBACK_PCT / 100, f"91-day T-bill cut-off yield, RBI auction of {RISK_FREE_3M_FALLBACK_DATE}"


def daily_rate(annual: float) -> float:
    """The constant daily rate that compounds to `annual` over a trading year."""
    return (1 + annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    update_cache()
    rate, source = risk_free_3m()
    print(f"{rate * 100:.4f}%  ({source})")
