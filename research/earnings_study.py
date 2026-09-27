"""
Earnings surprise via the market's reaction to results (27-Sep-2026).

Analyst consensus estimates (and so "profit vs expectations") are not available to us point-in-time.
The documented substitute is the earnings announcement return (EAR): the stock's return minus the
market's over the few sessions around the results. It measures the surprise as the market saw it, and
predicts the post-results drift about as well as estimate-based surprise (Chan, Jegadeesh & Lakonishok
1996; Brandt, Kishore, Santa-Clara & Venkatachalam 2008). Both inputs are known at the time: NSE's
board-meeting disclosures give each results date (data/results_history/, fetched from 2020), and prices
give the reaction.

PRE-REGISTERED DESIGN (fixed before any result was seen; data, dates, outcome and eligibility as in
research/conviction_study.py: monthly rebalances 2021-2026, forward 63-session RS vs the Nifty 500,
DI gap >= 2 and 240 sessions of prices; IS 2021-2023 / OOS 2024 onwards)
  EAR        For the latest results board meeting on date d with d+2 sessions before the rebalance and
             d within the last 100 sessions: stock return minus Nifty 500 return from the close before
             d to the close of the second session from d (results are often released after hours, so
             the reaction can land a session later). No results in that window: EAR missing.
  Rules      B  baseline      top 8 by rs_126_skip21 (the live ranking)
             E1 EAR alone     top 8 by EAR
             E2 confirmed     among the top 16 by rs_126_skip21, the 8 with the highest EAR
             E3 no bad news   the baseline ranking with negative-EAR stocks removed (a stock the market
                              marked down on its latest results, like Tata Power earlier), top 8
  Decision   A rule is adopted only if it beats B in BOTH periods and the paired full-sample t (every
             third date, non-overlapping quarters) is at least 2.

LIMITATIONS: survivorship (today's universe), no fundamentals rules, ~22 independent quarters; the board
meeting date is taken as the results date (meetings are occasionally postponed, and a few firms report
without a separate disclosure).

Usage:  python research/earnings_study.py      Output: research/earnings_study_results.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, DI_GAP_THIN_THRESHOLD, MIN_HISTORY_SESSIONS  # noqa: E402
from indicators import compute_adx  # noqa: E402
from momentum_study import HORIZON, IS_END, OUT_DIR, REBALANCE_EVERY, WARMUP, load_data  # noqa: E402

TOP_N, POOL_N, RECENT = 8, 16, 100
HISTORY_DIR = DATA_DIR / "results_history"


def results_dates(symbol: str) -> list:
    """Board meetings held to approve financial results (NSE disclosures), as dates."""
    path = HISTORY_DIR / f"{symbol}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("data", [])
    dates = set()
    for r in records:
        text = f"{r.get('bm_purpose', '')} {r.get('bm_desc', '')}".lower()
        if "result" in text:
            d = pd.to_datetime(r.get("bm_date"), format="%d-%b-%Y", errors="coerce")
            if pd.notna(d):
                dates.add(d)
    return sorted(dates)


def ear_panel(close: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """Per (session, symbol): EAR of the latest results reaction complete by that session (else NaN)."""
    idx = close.index
    ear = pd.DataFrame(np.nan, index=idx, columns=close.columns)
    for sym in close.columns:
        events = []
        for d in results_dates(sym):
            pos = idx.searchsorted(d)                      # first session on/after the meeting date
            if pos < 1 or pos + 1 >= len(idx):
                continue
            pre, post = idx[pos - 1], idx[pos + 1]
            r = close.at[post, sym] / close.at[pre, sym] - 1 - (bench[post] / bench[pre] - 1)
            if pd.notna(r):
                events.append((pos + 1, r * 100))          # known from the close of the second session
        for known, value in events:
            ear.iloc[known:known + RECENT, ear.columns.get_loc(sym)] = value  # later events overwrite
    return ear


def build(bench, close, high, low) -> pd.DataFrame:
    rs_skip = ((close.shift(21) / close.shift(126) - 1).sub(bench.shift(21) / bench.shift(126) - 1, axis=0)) * 100
    fwd = (close.shift(-HORIZON) / close - 1).sub(bench.shift(-HORIZON) / bench - 1, axis=0) * 100
    history = close.notna().cumsum()
    ear = ear_panel(close, bench)
    di = {}
    for sym in close.columns:
        ok = close[sym].notna() & high[sym].notna() & low[sym].notna()
        if ok.sum() < 40:
            continue
        a = compute_adx(pd.DataFrame({"HIGH_PRICE": high[sym][ok], "LOW_PRICE": low[sym][ok], "CLOSE_PRICE": close[sym][ok]}))
        di[sym] = a["PLUS_DI"] - a["MINUS_DI"]
    di = pd.DataFrame(di).reindex(close.index)
    rows = []
    for d in [d for d in close.index[WARMUP::REBALANCE_EVERY] if close.index.get_loc(d) + HORIZON < len(close.index)]:
        f = pd.DataFrame({"rs_skip": rs_skip.loc[d], "ear": ear.loc[d], "fwd_rs": fwd.loc[d],
                          "di_gap": di.loc[d], "sessions": history.loc[d]}).dropna(subset=["rs_skip", "fwd_rs", "di_gap"])
        f["date"] = d
        rows.append(f)
    p = pd.concat(rows).rename_axis("symbol").reset_index()
    p["period"] = np.where(p["date"] <= IS_END, "IS 2021-23", "OOS 2024-26")
    return p[(p["di_gap"] >= DI_GAP_THIN_THRESHOLD) & (p["sessions"] >= MIN_HISTORY_SESSIONS)]


RULES = {
    "B baseline (rs_126_skip21)": lambda g: g.nlargest(TOP_N, "rs_skip"),
    "E1 EAR alone": lambda g: g.dropna(subset=["ear"]).nlargest(TOP_N, "ear"),
    "E2 top-16 momentum, best 8 EAR": lambda g: g.nlargest(POOL_N, "rs_skip").dropna(subset=["ear"]).nlargest(TOP_N, "ear"),
    "E3 momentum, no negative EAR": lambda g: g[~(g["ear"] < 0)].nlargest(TOP_N, "rs_skip"),
}


def paired_t(diff: pd.Series) -> float:
    d = diff.dropna().iloc[::3]
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def main() -> None:
    bench, close, high, low, _, _ = load_data()
    p = build(bench, close, high, low)
    res = pd.DataFrame([{"date": d, "period": g["period"].iloc[0], **{k: f(g)["fwd_rs"].mean() for k, f in RULES.items()}}
                        for d, g in p.groupby("date")])
    coverage = p["ear"].notna().mean() * 100
    table = {}
    for label, g in [("All", res)] + list(res.groupby("period")):
        for k in RULES:
            table[(label, k)] = {"mean fwd RS of top 8 (pp)": round(g[k].mean(), 2),
                                 "beat Nifty 500 %": round((g[k] > 0).mean() * 100, 1)}
    ic = p.dropna(subset=["ear"]).groupby("date").apply(lambda g: g["ear"].rank().corr(g["fwd_rs"].rank()))
    diffs = {}
    for k in list(RULES)[1:]:
        dd = res[k] - res["B baseline (rs_126_skip21)"]
        diffs[f"{k} minus B"] = {**{f"mean diff {per} (pp)": round(dd[res["period"] == per].mean(), 2)
                                    for per in sorted(res["period"].unique())},
                                 "t (full, non-overlapping)": round(paired_t(dd), 2),
                                 "adopt?": "YES" if all(dd[res["period"] == per].mean() > 0 for per in res["period"].unique())
                                 and paired_t(dd) >= 2 else "no"}
    lines = ["# Earnings surprise (announcement return) vs our ranking", "",
             "Design and limitations: see the docstring of research/earnings_study.py.",
             f"{len(res)} rebalance dates; EAR available for {coverage:.0f}% of eligible stock-dates. "
             f"Mean rank IC of EAR with the forward 3-month RS: {ic.mean():.3f} "
             f"(IS {ic[ic.index <= IS_END].mean():.3f}, OOS {ic[ic.index > IS_END].mean():.3f}).", "",
             pd.DataFrame(table).T.to_markdown(), "", pd.DataFrame(diffs).T.to_markdown(), ""]
    (OUT_DIR / "earnings_study_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
