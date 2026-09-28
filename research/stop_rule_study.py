"""
Do the stop-loss and replacement rules help, and what do trading costs take? (28-Sep-2026)

The selection rule was backtested (research/*); the rules that run the money afterwards were not: a stop
at 3 x ATR(14) below the close that only ever rises, checked on the close; the sale proceeds buy the
first reserve stock (ranks 9-15 at the start) still in an uptrend that day. This study replays them.

DESIGN (written and run on 28-Sep-2026, after the list was frozen; not pre-registered)
  Universe    point in time (research/survivorship_study.py: approx. market cap >= Rs 5,000 cr and
              >= Rs 5 cr/day on the start date, DI gap >= 2, 240 sessions), so survivorship is reduced.
  Portfolios  each month from 2021: the top 8 by rs_126_skip21, equal weight (ERC weights are within a few
              points of equal), held 63 sessions. Reserve = ranks 9-15 on the start date.
    H  hold      no stops
    S  stops     3 x ATR(14) trailing stop on the close; exit at the NEXT session's close (the order goes
                 in after the close); proceeds wait in cash (the stopped stock's weight earns 0)
    R  stops + replacement   as S, the proceeds buy the first reserve stock not yet bought whose
                 DI gap is >= 2 that day, at that close, with its own trailing stop
  Costs       0.25% of each trade's value (STT 0.1% + exchange/brokerage/impact), charged in S and R on
              every exit and replacement purchase; H pays only the entry and exit common to all three,
              which is left out of every rule.
  Outcome     3-month portfolio return minus the Nifty 500's (pp); share of quarters beating it; the worst
              quarter; per period (IS 2021-2023 / OOS 2024 onwards).

LIMITATIONS: equal weights, no hedge; stops on adjusted closes (no intraday lows, no circuits, so real
fills after a gap can be worse); delisted companies still missing; ~22 independent quarters.

Usage:  python research/stop_rule_study.py      Output: research/stop_rule_results.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DI_GAP_THIN_THRESHOLD, MIN_HISTORY_SESSIONS, MIN_TURNOVER_CR,  # noqa: E402
                    STOP_LOSS_ATR_MULTIPLE, UNIVERSE_MIN_MARKET_CAP_CR)
from fetch_data import NSEBhavcopyFetcher  # noqa: E402
from indicators import compute_adx  # noqa: E402
from momentum_study import END, HORIZON, IS_END, OUT_DIR, REBALANCE_EVERY, START, WARMUP  # noqa: E402
from survivorship_study import TURNOVER_WINDOW, candidates, panels  # noqa: E402

TOP_N, RESERVE_N, COST = 8, 7, 0.0025
# Sensitivity of the live rule (R) to the stop's width and to trailing (checked after the main result)
VARIANTS = {"R 4xATR trailing": (4.0, True), "R 5xATR trailing": (5.0, True), "R 6xATR trailing": (6.0, True),
            "R 3xATR fixed": (3.0, False), "R 4xATR fixed": (4.0, False), "R 5xATR fixed": (5.0, False)}


def atr_panel(close, high, low, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    tr = tr.reindex(close.index)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def run_quarter(i0, picks, reserve, close, atr, di, mode, mult=STOP_LOSS_ATR_MULTIPLE, trail=True):
    """Portfolio value path (start 1.0) over HORIZON sessions from row i0 for mode H / S / R."""
    idx = close.index
    slots = [{"sym": s, "w": 1.0 / TOP_N, "px": close.iat[i0, close.columns.get_loc(s)],
              "stop": close.iat[i0, close.columns.get_loc(s)] - mult * atr.iat[i0, atr.columns.get_loc(s)],
              "open": True, "pending": False} for s in picks]
    queue, cash, costs = list(reserve), 0.0, 0.0
    for i in range(i0 + 1, i0 + HORIZON + 1):
        for slot in [s for s in slots if s["open"]]:
            c = close.iat[i, close.columns.get_loc(slot["sym"])]
            if np.isnan(c):
                continue
            if slot["pending"]:  # exit at this close (the session after the stop was hit)
                value = slot["w"] * c / slot["px"]
                costs += value * COST
                slot["open"] = False
                cash += value
                if mode == "R":
                    while queue:
                        cand = queue.pop(0)
                        j = close.columns.get_loc(cand)
                        cp, g = close.iat[i, j], di.iat[i, di.columns.get_loc(cand)] if cand in di else np.nan
                        if not np.isnan(cp) and g >= DI_GAP_THIN_THRESHOLD:
                            costs += value * COST
                            slots.append({"sym": cand, "w": value, "px": cp, "open": True, "pending": False,
                                          "stop": cp - mult * atr.iat[i, j]})
                            cash -= value
                            break
                continue
            if mode == "H":
                continue
            a = atr.iat[i, atr.columns.get_loc(slot["sym"])]
            if trail and not np.isnan(a):
                slot["stop"] = max(slot["stop"], c - mult * a)
            if c <= slot["stop"]:
                slot["pending"] = True
    end = i0 + HORIZON
    value = cash - costs
    for slot in slots:
        if slot["open"]:
            c = close.iloc[:end + 1][slot["sym"]].dropna().iloc[-1]
            value += slot["w"] * c / slot["px"]
    exits = sum(1 for s in slots if not s["open"])
    return value, exits


def main() -> None:
    cands = candidates()
    fetcher = NSEBhavcopyFetcher()
    b = fetcher.fetch_index_closes(START, END)
    b["Date"] = pd.to_datetime(b["Date"])
    bench = b.drop_duplicates("Date").set_index("Date")["Close"].sort_index()
    raw = fetcher.fetch_date_range(START, END, list(cands), use_cache=True)
    close, high, low, value, _ = panels(raw, bench.index)
    syms = [s for s in close.columns if s in cands]
    close, high, low, value = close[syms], high[syms], low[syms], value[syms]
    mcap_t = close.div(close.ffill().iloc[-1], axis=1).mul(pd.Series({s: cands[s][1] for s in syms}), axis=1)
    turnover = value.rolling(TURNOVER_WINDOW, min_periods=40).median()
    history = close.notna().cumsum()
    rs_skip = ((close.shift(21) / close.shift(126) - 1).sub(bench.shift(21) / bench.shift(126) - 1, axis=0)) * 100
    atr = atr_panel(close, high, low)
    di = {}
    for sym in syms:
        ok = close[sym].notna() & high[sym].notna() & low[sym].notna()
        if ok.sum() < 40:
            continue
        a = compute_adx(pd.DataFrame({"HIGH_PRICE": high[sym][ok], "LOW_PRICE": low[sym][ok], "CLOSE_PRICE": close[sym][ok]}))
        di[sym] = a["PLUS_DI"] - a["MINUS_DI"]
    di = pd.DataFrame(di).reindex(close.index)

    rows = []
    for d in [d for d in close.index[WARMUP::REBALANCE_EVERY] if close.index.get_loc(d) + HORIZON < len(close.index)]:
        i0 = close.index.get_loc(d)
        f = pd.DataFrame({"rs": rs_skip.loc[d], "di": di.loc[d], "n": history.loc[d], "mcap": mcap_t.loc[d],
                          "tv": turnover.loc[d], "atr": atr.loc[d], "px": close.loc[d]}).dropna()
        f = f[(f["di"] >= DI_GAP_THIN_THRESHOLD) & (f["n"] >= MIN_HISTORY_SESSIONS)
              & (f["mcap"] >= UNIVERSE_MIN_MARKET_CAP_CR) & (f["tv"] >= MIN_TURNOVER_CR)].sort_values("rs", ascending=False)
        if len(f) < TOP_N:
            continue
        picks, reserve = list(f.index[:TOP_N]), list(f.index[TOP_N:TOP_N + RESERVE_N])
        mkt = bench.iloc[i0 + HORIZON] / bench.iloc[i0] - 1
        row = {"date": d, "period": "IS 2021-23" if d <= IS_END else "OOS 2024-26"}
        for mode in "HSR":
            v, n_exit = run_quarter(i0, picks, reserve, close, atr, di, mode)
            row[mode] = (v - 1 - mkt) * 100
            row[f"{mode} exits"] = n_exit
        for key, (mult, trail) in VARIANTS.items():
            v, n_exit = run_quarter(i0, picks, reserve, close, atr, di, "R", mult, trail)
            row[key] = (v - 1 - mkt) * 100
            row[f"{key} exits"] = n_exit
        rows.append(row)
    res = pd.DataFrame(rows)
    names = {"H": "H hold (no stops)", "S": "S stops, cash", "R": "R stops + replacement (live rule)",
             **{k: k for k in VARIANTS}}
    table = {}
    for label, g in [("All", res)] + list(res.groupby("period")):
        for k, name in names.items():
            table[(label, name)] = {"mean 3m RS vs Nifty 500 (pp)": round(g[k].mean(), 2),
                                    "beat Nifty 500 %": round((g[k] > 0).mean() * 100, 1),
                                    "worst quarter (pp)": round(g[k].min(), 2),
                                    "stock exits per quarter": round(g[f"{k} exits"].mean(), 1)}
    diffs = {}
    for k in ["S", "R", *VARIANTS]:
        dd = res[k] - res["H"]
        s = dd.iloc[::3]
        diffs[f"{names[k]} minus H"] = {
            **{f"mean diff {p} (pp)": round(dd[res["period"] == p].mean(), 2) for p in sorted(res["period"].unique())},
            "t (full, non-overlapping)": round(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))), 2)}
    lines = ["# Stop-loss and replacement rules vs holding", "",
             "Design and limitations: see the docstring of research/stop_rule_study.py.",
             f"{len(res)} rebalance dates, point-in-time universe, 0.25% cost per trade.", "",
             pd.DataFrame(table).T.to_markdown(), "", pd.DataFrame(diffs).T.to_markdown(), ""]
    (OUT_DIR / "stop_rule_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
