"""
How much does survivorship bias flatter our backtests? (28-Sep-2026)

Every study in research/ runs on TODAY's universe: companies in today's Screener exports with a market cap
of at least Rs 5,000 cr TODAY. A stock that was small in 2021 and is big now (a winner) is in all of 2021;
a stock that was big in 2021 and has since shrunk below the floor (a loser) is missing throughout. That
flatters every absolute return figure. This study rebuilds the universe point in time and compares.

DESIGN (written and run on 28-Sep-2026, after the list was frozen; not pre-registered)
  Candidates  every NSE-listed company in the four Screener exports (data/screener/), with the same
              industry, InvIT and theme exclusions as the live universe but NO market-cap filter today.
  Point-in-time market cap
              approx. market cap on date t = today's market cap x adjusted close(t) / latest adjusted
              close (adjusted for splits and bonuses, so it assumes no new share issuance; QIPs and
              buybacks make it slightly off).
  Point-in-time liquidity
              median daily traded value over the previous 63 sessions >= Rs 5 cr (config.MIN_TURNOVER_CR).
  Universes   S  survivor: today's universe (config.PORTFOLIO_SYMBOLS), as in the other studies
              P  point in time: candidates with approx. market cap >= Rs 5,000 cr AND >= Rs 5 cr/day on t
  Eligibility DI gap >= 2 and 240 sessions of prices on t (as in the other studies); the profit and
              fundamental rules need point-in-time accounts data and are not applied.
  Rule        top 8 by rs_126_skip21 (the live rule), held 63 sessions; outcome = RS vs the Nifty 500.
  Measures    mean forward RS of the 8 (pp), share of dates beating the Nifty 500, rank IC of the signal,
              top-minus-bottom quintile spread, per period (IS 2021-2023 / OOS 2024 onwards).

REMAINING BIAS: companies delisted or merged away since 2020 are not in today's exports and cannot be
classified by sector, so they are still missing; corporate actions of names outside the live universe
are corrected by treating a daily move beyond +/-35% as a split or bonus (NSE price bands cap most
genuine moves at 20%).

Usage:  python research/survivorship_study.py      Output: research/survivorship_results.md
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from config import DI_GAP_THIN_THRESHOLD, MIN_HISTORY_SESSIONS, MIN_TURNOVER_CR, UNIVERSE_MIN_MARKET_CAP_CR  # noqa: E402
from fetch_data import NSEBhavcopyFetcher  # noqa: E402
from indicators import compute_adx  # noqa: E402
from momentum_study import END, HORIZON, IS_END, OUT_DIR, REBALANCE_EVERY, START, WARMUP  # noqa: E402

TOP_N = 8
SPLIT_MOVE = 0.35
TURNOVER_WINDOW = 63


def candidates() -> dict:
    """symbol -> (sector, today's market cap) for every NSE company in the exports, before the market-cap floor."""
    out = {}
    files = {sec: [path, *config.SCREENER_EXTRA_FILES.get(sec, [])] for sec, path in config.SCREENER_FILES.items()}
    for sector, paths in files.items():
        for path in paths:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    sym = (row.get("NSE Code") or "").strip()
                    try:
                        mcap = float(row.get("Market Capitalization") or "nan")
                    except ValueError:
                        mcap = float("nan")
                    if not sym or not mcap > 0:
                        continue
                    if sector == "Capital Goods" and (row["Industry Group"] in config.CAPITAL_GOODS_EXCLUDED_GROUPS
                                                      or row["Industry"] in config.CAPITAL_GOODS_EXCLUDED_INDUSTRIES):
                        continue
                    if sym in config.NON_EQUITY_INSTRUMENTS or sym in config.THEME_EXCLUSIONS:
                        continue
                    out[sym] = (sector, mcap)
    return out


def panels(raw: pd.DataFrame, dates: pd.DatetimeIndex):
    """Adjusted close/high/low and traded value (Rs cr) panels; moves beyond SPLIT_MOVE are treated as corporate actions."""
    raw = raw.copy()
    raw["DATE1"] = pd.to_datetime(raw["DATE1"])
    raw = raw[raw["SERIES"].isin(["EQ", "BE"])] if "SERIES" in raw else raw
    raw = raw.drop_duplicates(["SYMBOL", "DATE1"], keep="last").sort_values(["SYMBOL", "DATE1"])
    close, high, low, value, n_fixed = {}, {}, {}, {}, 0
    for sym, g in raw.groupby("SYMBOL"):
        g = g[g["DATE1"].isin(dates)]
        if len(g) < 2:
            continue
        ret = np.array((g["CLOSE_PRICE"] / g["PREV_CLOSE"] - 1).fillna(0.0), dtype=float)
        ret[0] = 0.0
        jumps = np.abs(ret) > SPLIT_MOVE
        n_fixed += int(jumps.sum())
        ret[jumps] = 0.0
        adj = g["CLOSE_PRICE"].iloc[0] * np.cumprod(1 + ret)
        factor = adj / g["CLOSE_PRICE"].values
        idx = g["DATE1"].values
        close[sym] = pd.Series(adj, index=idx)
        high[sym] = pd.Series(g["HIGH_PRICE"].values * factor, index=idx)
        low[sym] = pd.Series(g["LOW_PRICE"].values * factor, index=idx)
        value[sym] = pd.Series(pd.to_numeric(g["TURNOVER_LACS"], errors="coerce").values / 100, index=idx)
    to = lambda d: pd.DataFrame(d).reindex(dates)  # noqa: E731
    return to(close), to(high), to(low), to(value), n_fixed


def main() -> None:
    cands = candidates()
    fetcher = NSEBhavcopyFetcher()
    b = fetcher.fetch_index_closes(START, END)
    b["Date"] = pd.to_datetime(b["Date"])
    bench = b.drop_duplicates("Date").set_index("Date")["Close"].sort_index()
    raw = fetcher.fetch_date_range(START, END, list(cands), use_cache=True)
    close, high, low, value, n_fixed = panels(raw, bench.index)
    syms = [s for s in close.columns if s in cands]
    close, high, low, value = close[syms], high[syms], low[syms], value[syms]

    last = close.ffill().iloc[-1]
    mcap_now = pd.Series({s: cands[s][1] for s in syms})
    mcap_t = close.div(last, axis=1).mul(mcap_now, axis=1)
    turnover = value.rolling(TURNOVER_WINDOW, min_periods=40).median()
    history = close.notna().cumsum()
    rs_skip = ((close.shift(21) / close.shift(126) - 1).sub(bench.shift(21) / bench.shift(126) - 1, axis=0)) * 100
    fwd = (close.shift(-HORIZON) / close - 1).sub(bench.shift(-HORIZON) / bench - 1, axis=0) * 100
    di = {}
    for sym in syms:
        ok = close[sym].notna() & high[sym].notna() & low[sym].notna()
        if ok.sum() < 40:
            continue
        a = compute_adx(pd.DataFrame({"HIGH_PRICE": high[sym][ok], "LOW_PRICE": low[sym][ok], "CLOSE_PRICE": close[sym][ok]}))
        di[sym] = a["PLUS_DI"] - a["MINUS_DI"]
    di = pd.DataFrame(di).reindex(close.index)

    survivor = set(config.PORTFOLIO_SYMBOLS)
    rows, ics, qs, sizes = [], {"S": [], "P": []}, {"S": [], "P": []}, []
    dates = [d for d in close.index[WARMUP::REBALANCE_EVERY] if close.index.get_loc(d) + HORIZON < len(close.index)]
    for d in dates:
        f = pd.DataFrame({"rs": rs_skip.loc[d], "fwd": fwd.loc[d], "di": di.loc[d], "n": history.loc[d],
                          "mcap": mcap_t.loc[d], "tv": turnover.loc[d]}).dropna(subset=["rs", "fwd", "di"])
        tech = f[(f["di"] >= DI_GAP_THIN_THRESHOLD) & (f["n"] >= MIN_HISTORY_SESSIONS)]
        uni = {"S": tech[tech.index.isin(survivor)],
               "P": tech[(tech["mcap"] >= UNIVERSE_MIN_MARKET_CAP_CR) & (tech["tv"] >= MIN_TURNOVER_CR)]}
        row = {"date": d, "period": "IS 2021-23" if d <= IS_END else "OOS 2024-26"}
        for k, g in uni.items():
            row[k] = g.nlargest(TOP_N, "rs")["fwd"].mean() if len(g) >= TOP_N else np.nan
            if len(g) >= 10:
                ics[k].append((d, g["rs"].rank().corr(g["fwd"].rank())))
                q = pd.qcut(g["rs"].rank(method="first"), 5, labels=False)
                qs[k].append((d, g.loc[q == 4, "fwd"].mean() - g.loc[q == 0, "fwd"].mean()))
        sizes.append({"date": d, "S": len(uni["S"]), "P": len(uni["P"]),
                      "P not in S": len(set(uni["P"].index) - survivor)})
        rows.append(row)
    res = pd.DataFrame(rows)
    ic = {k: pd.Series(dict(v)) for k, v in ics.items()}
    qd = {k: pd.Series(dict(v)) for k, v in qs.items()}
    sz = pd.DataFrame(sizes)
    table = {}
    for label, g in [("All", res)] + list(res.groupby("period")):
        sel = g["date"]
        for k, name in [("S", "S survivor (today's universe)"), ("P", "P point in time")]:
            table[(label, name)] = {
                "mean fwd RS of top 8 (pp)": round(g[k].mean(), 2),
                "beat Nifty 500 %": round((g[k] > 0).mean() * 100, 1),
                "rank IC": round(ic[k].reindex(sel).mean(), 3),
                "Q5-Q1 spread (pp)": round(qd[k].reindex(sel).mean(), 2)}
    dd = (res["P"] - res["S"]).dropna().iloc[::3]
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    lines = ["# Survivorship bias: today's universe vs a point-in-time universe", "",
             "Design and limitations: see the docstring of research/survivorship_study.py.",
             f"{len(res)} rebalance dates; {len(cands)} candidate companies ({len(syms)} with NSE prices); "
             f"{n_fixed} daily moves beyond +/-{SPLIT_MOVE:.0%} treated as splits/bonuses.",
             f"Eligible stocks per date (after the trend and history rules): survivor {sz['S'].mean():.0f}, "
             f"point in time {sz['P'].mean():.0f}, of which {sz['P not in S'].mean():.0f} are not in today's universe "
             "(big then, small now, or illiquid then).", "",
             pd.DataFrame(table).T.to_markdown(), "",
             f"Top 8, point in time minus survivor: {(res['P'] - res['S']).mean():+.2f} pp per quarter "
             f"(t = {t:.2f}, non-overlapping quarters).", ""]
    (OUT_DIR / "survivorship_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
