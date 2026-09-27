"""
Two published refinements of momentum, tested against our ranking (27-Sep-2026).

PRE-REGISTERED DESIGN (fixed before any result was seen; same data, dates, outcome and eligibility as
research/conviction_study.py: monthly rebalances, forward 63-session RS vs the Nifty 500, DI gap >= 2 and
240 sessions of prices, IS 2021-2023 / OOS 2024 onwards)
  B  baseline   top 8 by rs_126_skip21 (the live rule): RS vs the Nifty 500 over sessions t-126..t-21.
  R  residual   top 8 by residual momentum (Blitz, Huij & Martens 2011): alpha and beta vs the Nifty 500
                from daily returns over the 252 sessions ending t-21; residual returns over t-126..t-21
                summed and divided by their standard deviation. Momentum with the market's part removed,
                reported to be steadier and less crash-prone than total-return momentum.
  F  frog       among the top 16 by rs_126_skip21, the 8 with the most continuous path (Da, Gurun &
                Warachka 2014, "frog in the pan"): information discreteness
                ID = sign(return) x (% negative days - % positive days) over t-126..t-21; lower = many
                small moves, which investors under-react to and which keep trending.
  Decision      a rule replaces B only if it beats B in BOTH periods and the paired full-sample t
                (every third date, non-overlapping) is at least 2.

LIMITATIONS: as in conviction_study.py (survivorship, no fundamentals, ~22 independent quarters).

Usage:  python research/signal_extensions_study.py   Output: research/signal_extensions_results.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DI_GAP_THIN_THRESHOLD, MIN_HISTORY_SESSIONS  # noqa: E402
from indicators import compute_adx  # noqa: E402
from momentum_study import HORIZON, IS_END, OUT_DIR, REBALANCE_EVERY, WARMUP, load_data  # noqa: E402

TOP_N, POOL_N = 8, 16
SKIP, WINDOW, BETA_WINDOW = 21, 126, 252


def build(bench, close, high, low) -> pd.DataFrame:
    r = close.pct_change(fill_method=None)
    m = bench.pct_change()
    rs_skip = ((close.shift(SKIP) / close.shift(WINDOW) - 1).sub(bench.shift(SKIP) / bench.shift(WINDOW) - 1, axis=0)) * 100
    fwd = (close.shift(-HORIZON) / close - 1).sub(bench.shift(-HORIZON) / bench - 1, axis=0) * 100
    history = close.notna().cumsum()
    di = {}
    for sym in close.columns:
        ok = close[sym].notna() & high[sym].notna() & low[sym].notna()
        if ok.sum() < 40:
            continue
        a = compute_adx(pd.DataFrame({"HIGH_PRICE": high[sym][ok], "LOW_PRICE": low[sym][ok], "CLOSE_PRICE": close[sym][ok]}))
        di[sym] = a["PLUS_DI"] - a["MINUS_DI"]
    di = pd.DataFrame(di).reindex(close.index)

    rows = []
    rebal = [d for d in close.index[WARMUP::REBALANCE_EVERY] if close.index.get_loc(d) + HORIZON < len(close.index)]
    for d in rebal:
        i = close.index.get_loc(d)
        est = slice(i - SKIP - BETA_WINDOW + 1, i - SKIP + 1)   # 252 sessions ending t-21
        win = slice(i - WINDOW + 1, i - SKIP + 1)               # sessions t-125..t-21 (105 returns)
        R, M = r.iloc[est], m.iloc[est]
        mv = M.var()
        beta = R.apply(lambda s: s.cov(M)) / mv
        alpha = R.mean() - beta * M.mean()
        resid = r.iloc[win] - np.outer(m.iloc[win].values, beta.values) - alpha.values
        resid_mom = resid.sum() / resid.std()
        rw = r.iloc[win]
        n = rw.notna().sum()
        pos, neg = (rw > 0).sum() / n, (rw < 0).sum() / n
        total = (1 + rw).prod() - 1
        idisc = np.sign(total) * (neg - pos)
        f = pd.DataFrame({"rs_skip": rs_skip.loc[d], "resid_mom": resid_mom, "id": idisc, "fwd_rs": fwd.loc[d],
                          "di_gap": di.loc[d], "sessions": history.loc[d]}).dropna()
        f["date"] = d
        rows.append(f)
    p = pd.concat(rows).rename_axis("symbol").reset_index()
    p["period"] = np.where(p["date"] <= IS_END, "IS 2021-23", "OOS 2024-26")
    p = p[(p["di_gap"] >= DI_GAP_THIN_THRESHOLD) & (p["sessions"] >= MIN_HISTORY_SESSIONS)]
    return p


RULES = {
    "B baseline (rs_126_skip21)": lambda g: g.nlargest(TOP_N, "rs_skip"),
    "R residual momentum": lambda g: g.nlargest(TOP_N, "resid_mom"),
    "F frog-in-the-pan": lambda g: g.nlargest(POOL_N, "rs_skip").nsmallest(TOP_N, "id"),
}


def paired_t(diff: pd.Series) -> float:
    d = diff.dropna().iloc[::3]
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def main() -> None:
    bench, close, high, low, _, _ = load_data()
    p = build(bench, close, high, low)
    res = pd.DataFrame([{"date": d, "period": g["period"].iloc[0], **{k: f(g)["fwd_rs"].mean() for k, f in RULES.items()}}
                        for d, g in p.groupby("date")])
    table = {}
    for label, g in [("All", res)] + list(res.groupby("period")):
        for k in RULES:
            table[(label, k)] = {"mean fwd RS of top 8 (pp)": round(g[k].mean(), 2),
                                 "beat Nifty 500 %": round((g[k] > 0).mean() * 100, 1)}
    diffs = {}
    for k in list(RULES)[1:]:
        dd = res[k] - res["B baseline (rs_126_skip21)"]
        diffs[f"{k} minus B"] = {**{f"mean diff {per} (pp)": round(dd[res["period"] == per].mean(), 2)
                                    for per in sorted(res["period"].unique())},
                                 "t (full, non-overlapping)": round(paired_t(dd), 2),
                                 "adopt?": "YES" if all(dd[res["period"] == per].mean() > 0 for per in res["period"].unique())
                                 and paired_t(dd) >= 2 else "no"}
    lines = ["# Momentum refinements: residual momentum and frog-in-the-pan", "",
             "Design and limitations: see the docstring of research/signal_extensions_study.py.",
             f"{len(res)} rebalance dates.", "", pd.DataFrame(table).T.to_markdown(), "",
             pd.DataFrame(diffs).T.to_markdown(), ""]
    (OUT_DIR / "signal_extensions_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
