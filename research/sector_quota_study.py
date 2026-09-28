"""
Does forcing the theme's pillars (Cement, Capital Goods & EPC, Power) into the top 8 help? (28-Sep-2026)

The live rule has no sector quotas: 7 of the 8 invested stocks are Capital Goods, one is Power and none
is Cement. The critique is that a three-pillar theme should hold all three pillars. This study asks
whether doing so would have earned more, or at least lowered risk, over 3-month holding periods.

DESIGN (NOT pre-registered: written and first run on 28-Sep-2026, after the list was frozen, in answer to
that critique; data, dates, outcome and eligibility as in research/conviction_study.py: monthly rebalances
2021-2026, forward 63-session RS vs the Nifty 500 with no rebalancing inside the 3 months, DI gap >= 2
and 240 sessions of prices; IS 2021-2023 / OOS 2024 onwards)
  B  baseline   top 8 by rs_126_skip21 (the live rule)
  Q  quota      the best-ranked Cement and the best-ranked Power stock, then the top of the ranking to
                fill 8 (a minimal "every pillar present" rule)
  P  pillars    top 3 Capital Goods & EPC, top 3 Power, top 2 Cement (balanced pillars), filled from the
                ranking if a sector has too few eligible stocks
  Measures      mean forward RS of the 8 (pp), share of dates beating the Nifty 500, the spread (standard
                deviation) of the 8's quarterly outcome across dates, and the paired difference to B with
                a t-stat on every third date (non-overlapping quarters).
  Decision      a quota rule would be adopted only if it beat B in BOTH periods with t >= 2, or cut the
                spread of outcomes without costing return.

LIMITATIONS: survivorship (today's universe, including the EPC stocks added on 27-Sep), no fundamentals
or liquidity rule, ~22 independent quarters.

Usage:  python research/sector_quota_study.py      Output: research/sector_quota_results.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import sector_of  # noqa: E402
from conviction_study import build  # noqa: E402
from momentum_study import OUT_DIR, load_data  # noqa: E402

TOP_N = 8
PILLARS = {"Capital Goods": 3, "Power": 3, "Cement": 2}


def baseline(g: pd.DataFrame) -> pd.DataFrame:
    return g.head(TOP_N)


def quota(g: pd.DataFrame) -> pd.DataFrame:
    forced = [g[g["sector"] == sec].index[0] for sec in ("Cement", "Power") if (g["sector"] == sec).any()]
    rest = [i for i in g.index if i not in forced][:TOP_N - len(forced)]
    return g.loc[forced + rest]


def pillars(g: pd.DataFrame) -> pd.DataFrame:
    picked = pd.concat([g[g["sector"] == sec].head(n) for sec, n in PILLARS.items()])
    if len(picked) < TOP_N:
        picked = pd.concat([picked, g.drop(picked.index).head(TOP_N - len(picked))])
    return picked


RULES = {"B baseline (no quotas)": baseline, "Q one Cement + one Power": quota, "P balanced pillars 3/3/2": pillars}


def paired_t(diff: pd.Series) -> float:
    d = diff.dropna().iloc[::3]
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else float("nan")


def main() -> None:
    bench, close, high, low, _, _ = load_data()
    panel = build(bench, close, high, low)
    panel = panel[panel["eligible"]].copy()
    panel["sector"] = panel["symbol"].map(sector_of)
    rows = []
    for d, g in panel.groupby("date"):
        g = g.sort_values("rank_sig", ascending=False).reset_index(drop=True)
        row = {"date": d, "period": g["period"].iloc[0]}
        for name, rule in RULES.items():
            row[name] = rule(g)["fwd_rs"].mean()
        row["B share Capital Goods"] = (baseline(g)["sector"] == "Capital Goods").mean() * 100
        rows.append(row)
    res = pd.DataFrame(rows)
    table = {}
    for label, g in [("All", res)] + list(res.groupby("period")):
        for name in RULES:
            table[(label, name)] = {"mean fwd RS of top 8 (pp)": round(g[name].mean(), 2),
                                    "beat Nifty 500 %": round((g[name] > 0).mean() * 100, 1),
                                    "spread of outcomes (sd, pp)": round(g[name].std(), 2)}
    diffs = {}
    for name in list(RULES)[1:]:
        dd = res[name] - res["B baseline (no quotas)"]
        diffs[f"{name} minus B"] = {
            **{f"mean diff {p} (pp)": round(dd[res["period"] == p].mean(), 2) for p in sorted(res["period"].unique())},
            "t (full, non-overlapping)": round(paired_t(dd), 2),
            "adopt?": "YES" if all(dd[res["period"] == p].mean() > 0 for p in res["period"].unique())
            and paired_t(dd) >= 2 else "no"}
    lines = ["# Sector quotas vs the pure ranking", "",
             "Design and limitations: see the docstring of research/sector_quota_study.py.",
             f"{len(res)} rebalance dates; under the live rule Capital Goods averaged "
             f"{res['B share Capital Goods'].mean():.0f}% of the top 8.", "",
             pd.DataFrame(table).T.to_markdown(), "", pd.DataFrame(diffs).T.to_markdown(), ""]
    (OUT_DIR / "sector_quota_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
