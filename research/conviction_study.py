"""
Does the RRG conviction filter improve the selection rule? (27-Sep-2026)

The live rule ranks eligible stocks by 6-month RS vs the Nifty 500 excluding the latest month
(rs_126_skip21) and then skips names whose RRG conviction is Low (WEAKENING or LAGGING against both
the Nifty 500 and the equal-weighted sector average). The conviction label comes from 63-session RS
and its 10-session momentum, and 63-session RS was the weakest signal in research/momentum_study.py,
so the filter needs its own test.

PRE-REGISTERED DESIGN (fixed before any result was seen)
  Data       Same as research/momentum_study.py: today's universe (config.PORTFOLIO_SYMBOLS), NSE
             bhavcopy prices adjusted for corporate actions, Nifty 500 closes, monthly rebalances
             (every 21 sessions) once 252 sessions of history exist, forward 63-session RS vs the
             Nifty 500 as the outcome; IS = 2021-2023, OOS = 2024 onwards.
  RRG        Exactly the live definitions: x = 63-session RS (vs Nifty 500; vs the equal-weighted
             63-session return of the stock's sector universe), y = mean of x over the latest 5
             sessions minus its mean 10 sessions earlier (rrg.classify_quadrant /
             rrg.conviction_tier).
  Eligible   Technical part of the live rule only: DI gap (+DI - -DI) >= 2 and 240 sessions of
             prices. The fundamental, liquidity and profit rules need point-in-time accounts data
             that this study does not have.
  Rules      Each date, the top 8 eligible stocks by rs_126_skip21 (equal weight), under:
               A  current   conviction High or Moderate
               B  none      no conviction filter
               C  rs63>0    the stock still beats the Nifty 500 over 63 sessions (drops the
                            short-term momentum term, which is the noisy part)
  Measures   Mean forward RS of the 8 (pp), share of dates the 8 beat the Nifty 500, and the paired
             difference A-B and C-B per date with a t-stat on every third date (non-overlapping).
  Decision   A filter is kept only if it beats B in BOTH periods and the paired full-sample
             |t| >= 2; otherwise the simpler rule (B) is the default.

KNOWN LIMITATIONS: survivorship bias (today's universe), no fundamentals or liquidity rule, and
about 65 monthly dates, i.e. roughly 22 independent 3-month periods: modest differences cannot be
told apart from noise.

Usage:  python research/conviction_study.py      Output: research/conviction_study_results.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DI_GAP_THIN_THRESHOLD, MIN_HISTORY_SESSIONS, RRG_MOMENTUM_DAYS,  # noqa: E402
                    RRG_MOMENTUM_SMOOTHING_DAYS, sector_of)
from indicators import compute_adx  # noqa: E402
from momentum_study import HORIZON, IS_END, OUT_DIR, REBALANCE_EVERY, WARMUP, load_data  # noqa: E402
from rrg import conviction_tier, classify_quadrant  # noqa: E402

TOP_N = 8
LOOKBACK = 63


def rrg_axes(ret63: pd.DataFrame, base63: pd.DataFrame) -> tuple:
    """x = 63-session spread vs a base (pp); y = mean of x over the latest 5 sessions minus 10 earlier."""
    x = (ret63 - base63) * 100
    s = RRG_MOMENTUM_SMOOTHING_DAYS
    y = x.rolling(s).mean() - x.shift(RRG_MOMENTUM_DAYS).rolling(s).mean()
    return x, y


def build(bench, close, high, low) -> pd.DataFrame:
    ret63 = close / close.shift(LOOKBACK) - 1
    mkt63 = bench / bench.shift(LOOKBACK) - 1
    x_n, y_n = rrg_axes(ret63, pd.DataFrame({c: mkt63 for c in close.columns}))
    sectors = pd.Series({c: sector_of(c) for c in close.columns})
    sector_avg = pd.DataFrame({c: ret63[sectors[sectors == sectors[c]].index].mean(axis=1) for c in close.columns})
    x_s, y_s = rrg_axes(ret63, sector_avg)
    rank_sig = ((close.shift(21) / close.shift(126) - 1).sub(bench.shift(21) / bench.shift(126) - 1, axis=0)) * 100
    fwd = (close.shift(-HORIZON) / close - 1).sub(bench.shift(-HORIZON) / bench - 1, axis=0) * 100
    history = close.notna().cumsum()
    di_gap = {}
    for sym in close.columns:
        ok = close[sym].notna() & high[sym].notna() & low[sym].notna()
        if ok.sum() < 40:
            continue
        a = compute_adx(pd.DataFrame({"HIGH_PRICE": high[sym][ok], "LOW_PRICE": low[sym][ok],
                                      "CLOSE_PRICE": close[sym][ok]}))
        di_gap[sym] = a["PLUS_DI"] - a["MINUS_DI"]
    di_gap = pd.DataFrame(di_gap).reindex(close.index)

    rebal = [d for d in close.index[WARMUP::REBALANCE_EVERY] if close.index.get_loc(d) + HORIZON < len(close.index)]
    rows = []
    for d in rebal:
        f = pd.DataFrame({"rank_sig": rank_sig.loc[d], "fwd_rs": fwd.loc[d], "di_gap": di_gap.loc[d],
                          "sessions": history.loc[d], "x_n": x_n.loc[d], "y_n": y_n.loc[d],
                          "x_s": x_s.loc[d], "y_s": y_s.loc[d]}).dropna()
        f["conviction"] = [conviction_tier(classify_quadrant(a, b), classify_quadrant(c, e))
                           for a, b, c, e in zip(f["x_n"], f["y_n"], f["x_s"], f["y_s"])]
        f["date"] = d
        rows.append(f)
    panel = pd.concat(rows).rename_axis("symbol").reset_index()
    panel["period"] = np.where(panel["date"] <= IS_END, "IS 2021-23", "OOS 2024-26")
    panel["eligible"] = (panel["di_gap"] >= DI_GAP_THIN_THRESHOLD) & (panel["sessions"] >= MIN_HISTORY_SESSIONS)
    return panel


RULES = {
    "A current (High/Moderate)": lambda g: g[g["conviction"].isin(["High", "Moderate"])],
    "B no filter": lambda g: g,
    "C rs63 > 0": lambda g: g[g["x_n"] > 0],
}


def per_date(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d, g in panel[panel["eligible"]].groupby("date"):
        g = g.sort_values("rank_sig", ascending=False)
        row = {"date": d, "period": g["period"].iloc[0]}
        for name, rule in RULES.items():
            top = rule(g).head(TOP_N)
            row[name] = top["fwd_rs"].mean() if len(top) else np.nan
        # How often the filter changes the top 8 at all
        row["A changes top 8"] = set(RULES["A current (High/Moderate)"](g).head(TOP_N).index) != set(g.head(TOP_N).index)
        out.append(row)
    return pd.DataFrame(out)


def paired_t(diff: pd.Series) -> float:
    d = diff.dropna().iloc[::3]  # non-overlapping 3-month periods
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def main() -> None:
    bench, close, high, low, _, _ = load_data()
    res = per_date(build(bench, close, high, low))
    lines = ["# Conviction filter study", "", "Design and limitations: see the docstring of research/conviction_study.py.",
             f"{len(res)} rebalance dates; the current filter changed the top 8 on "
             f"{res['A changes top 8'].mean() * 100:.0f}% of them.", ""]
    table = {}
    for label, g in [("All", res)] + list(res.groupby("period")):
        for name in RULES:
            table[(label, name)] = {"mean fwd RS of top 8 (pp)": round(g[name].mean(), 2),
                                    "beat Nifty 500 %": round((g[name] > 0).mean() * 100, 1)}
    t = pd.DataFrame(table).T
    lines += [t.to_markdown(), ""]
    diffs = {}
    for name in ["A current (High/Moderate)", "C rs63 > 0"]:
        dd = res[name] - res["B no filter"]
        diffs[f"{name} minus B"] = {
            **{f"mean diff {p} (pp)": round(dd[res["period"] == p].mean(), 2) for p in sorted(res["period"].unique())},
            "t (full, non-overlapping)": round(paired_t(dd), 2),
            "keep?": "YES" if all(dd[res["period"] == p].mean() > 0 for p in res["period"].unique())
                     and abs(paired_t(dd)) >= 2 else "no"}
    lines += [pd.DataFrame(diffs).T.to_markdown(), ""]
    (OUT_DIR / "conviction_study_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
