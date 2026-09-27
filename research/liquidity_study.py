"""
Market cap vs trading liquidity in the theme universe (27-Sep-2026 study).

Question: is the Rs 5,000 cr market-cap floor a good proxy for what actually constrains a Rs 1 crore,
3-month book, the ability to enter and exit a position cleanly?

Data: the Screener exports (data/screener/) after every universe rule except the market-cap floor, and
the median daily traded value (NSE full bhavcopy TURNOVER_LACS, EQ/BE series) over the last 30 cached
sessions. Output: output/liquidity_vs_marketcap.csv (one row per stock) and a bucket table on stdout.

Position sizing behind the Rs 5 cr/day threshold (config.MIN_TURNOVER_CR): the largest position is
15% of the Rs 97 lakh equity sleeve, about Rs 14.6 lakh. Trading at most ~5% of a day's value keeps
price impact small (impact grows roughly with the square root of the participation rate); allowing
for turnover halving in a sell-off (when stops are hit), 2 x 14.6 lakh / 5% = Rs 5.8 cr, so Rs 5 cr
median turnover puts a normal-day exit at under 3% of the day's value.
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SESSIONS = 30


def median_turnover_cr(sessions: int = SESSIONS) -> pd.Series:
    files = glob.glob(str(config.DATA_DIR / "raw_bhavcopy" / "bhav_eq_*.csv"))
    files = sorted(files, key=lambda f: pd.to_datetime(re.search(r"bhav_eq_(.+)\.csv", f).group(1),
                                                         format="%d-%b-%Y"))[-sessions:]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = d.columns.str.strip()
        for col in ("SYMBOL", "SERIES"):
            d[col] = d[col].astype(str).str.strip()
        frames.append(d[d["SERIES"].isin(["EQ", "BE"])][["SYMBOL", "TURNOVER_LACS"]])
    return (pd.concat(frames).groupby("SYMBOL")["TURNOVER_LACS"].median() / 100).rename("median_turnover_cr")


def run() -> pd.DataFrame:
    floor = config.UNIVERSE_MIN_MARKET_CAP_CR
    config.UNIVERSE_MIN_MARKET_CAP_CR = 0  # every other universe rule still applies
    rows = [{"symbol": r["symbol"], "sector": s, "market_cap_cr": float(r["record"]["Market Capitalization"])}
            for s in config.SCREENER_FILES for r in config._read_screener(s)]
    config.UNIVERSE_MIN_MARKET_CAP_CR = floor
    df = pd.DataFrame(rows).set_index("symbol").join(median_turnover_cr())
    df = df[df["market_cap_cr"] >= 500].dropna(subset=["median_turnover_cr"])
    df["above_cap_floor"] = df["market_cap_cr"] >= floor
    df["passes_turnover_rule"] = df["median_turnover_cr"] >= config.MIN_TURNOVER_CR
    df.sort_values("market_cap_cr", ascending=False).round(2).to_csv(config.OUTPUT_DIR / "liquidity_vs_marketcap.csv")
    print(f"{len(df)} theme stocks >= Rs 500 cr with turnover data; rank correlation market cap vs turnover "
          f"{df['market_cap_cr'].rank().corr(df['median_turnover_cr'].rank()):.2f}")
    buckets = pd.cut(df["market_cap_cr"], [500, 1000, 2000, 3000, 5000, 10000, 25000, np.inf])
    g = df.groupby(buckets, observed=True)["median_turnover_cr"]
    print(pd.DataFrame({"stocks": g.size(), "median_turnover_cr": g.median().round(1),
                        f"share_>=_{config.MIN_TURNOVER_CR:g}cr": g.apply(lambda s: (s >= config.MIN_TURNOVER_CR).mean()).round(2)}))
    print("Above the floor but thin:", ", ".join(df[df["above_cap_floor"] & ~df["passes_turnover_rule"]].index))
    return df


if __name__ == "__main__":
    run()
