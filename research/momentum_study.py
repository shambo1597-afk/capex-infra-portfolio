"""
Momentum study: which of the brief's technical indicators predict the next 3 months of
relative performance for our Cement / Capital Goods / Power universe?

PRE-REGISTERED DESIGN (fixed before any results were seen, to avoid mining the data for patterns)
  Universe   The current sector universes (config.PORTFOLIO_SYMBOLS), prices from NSE Bhavcopy.
  Prices     Split/bonus-adjusted by chaining the exchange's CLOSE / PREV_CLOSE daily returns.
  Benchmark  Nifty 500 official daily closes (price index).
  Target     Forward relative strength: stock return minus Nifty 500 return over the next
             63 sessions (about 3 months, the mandate), in percentage points.
  Dates      A rebalance every 21 sessions (about monthly) once 252 sessions of history exist.
  Periods    In-sample (IS): rebalances in 2021-2023. Out-of-sample (OOS): 2024 onwards.
             A pattern counts only if it has the same sign and a meaningful size in both.
  Signals    Only the brief's four indicators:
             - RS vs Nifty 500 over 21 / 63 / 126 / 189 / 252 sessions, and 126 sessions
               excluding the latest 21 ("skip the last month", the classic momentum variant)
             - burst: share of the 63-session RS earned in the last 10 sessions (> 50%)
             - ADX / DI: trend strength and direction (ADX > 25 with +DI > -DI)
             - RSI: overbought (> 70)
             - Support / resistance: within 2% of the 20-session high (at resistance / breakout)
  Measures   IC: rank correlation between a signal and forward RS across stocks, per date.
             Top-fifth spread: forward RS of the top fifth of stocks minus the universe average.
             Top-8 hit rate: share of dates on which an equal-weight 8-stock portfolio of the
             top-ranked stocks beat the Nifty 500 over the next 63 sessions.
             Conditional effects are tested inside the top fifth by 126-session RS.
  Significance  Consecutive rebalances overlap (3-month horizon, monthly dates), so the t-stat
             for the mean IC uses every third date only (non-overlapping quarters).

KNOWN LIMITATION: survivorship bias. The universe is today's index constituents plus theme
additions, so stocks that fell out of the indices since 2020 are missing. That inflates absolute
returns; comparisons between signals are less affected but not immune.

Usage:
    python research/momentum_study.py      # uses cached NSE files, downloads any missing day
Outputs: research/momentum_study_results.md and research/momentum_study_panel.csv
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PORTFOLIO_SYMBOLS, sector_of  # noqa: E402
from fetch_data import NSEBhavcopyFetcher  # noqa: E402
from indicators import compute_adx, compute_rsi  # noqa: E402

START, END = date(2020, 1, 1), date(2026, 9, 25)
HORIZON = 63
REBALANCE_EVERY = 21
WARMUP = 252
RS_WINDOWS = [21, 63, 126, 189, 252]
IS_END = pd.Timestamp("2023-12-31")
TOP_N = 8
OUT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

def adjusted_panels(raw: pd.DataFrame, dates: pd.DatetimeIndex):
    """Close / high / low panels (dates x symbols), split-adjusted via CLOSE / PREV_CLOSE returns."""
    raw = raw.copy()
    raw["DATE1"] = pd.to_datetime(raw["DATE1"])
    raw = raw.drop_duplicates(["SYMBOL", "DATE1"], keep="last").sort_values(["SYMBOL", "DATE1"])
    closes, highs, lows, big_moves = {}, {}, {}, 0
    for sym, g in raw.groupby("SYMBOL"):
        g = g[g["DATE1"].isin(dates)]
        if len(g) < 2:
            continue
        ret = (g["CLOSE_PRICE"] / g["PREV_CLOSE"] - 1).fillna(0.0).values
        ret[0] = 0.0
        big_moves += int((np.abs(ret) > 0.35).sum())
        adj = g["CLOSE_PRICE"].iloc[0] * np.cumprod(1 + ret)
        factor = adj / g["CLOSE_PRICE"].values
        idx = g["DATE1"].values
        closes[sym] = pd.Series(adj, index=idx)
        highs[sym] = pd.Series(g["HIGH_PRICE"].values * factor, index=idx)
        lows[sym] = pd.Series(g["LOW_PRICE"].values * factor, index=idx)
    to_panel = lambda d: pd.DataFrame(d).reindex(dates)  # noqa: E731
    return to_panel(closes), to_panel(highs), to_panel(lows), big_moves


def load_data():
    fetcher = NSEBhavcopyFetcher()
    bench = fetcher.fetch_index_closes(START, END)
    bench["Date"] = pd.to_datetime(bench["Date"])
    bench = bench.drop_duplicates("Date").set_index("Date")["Close"].sort_index()
    raw = fetcher.fetch_date_range(START, END, PORTFOLIO_SYMBOLS, use_cache=True)
    close, high, low, big_moves = adjusted_panels(raw, bench.index)
    return bench, close, high, low, big_moves


# -----------------------------------------------------------------------------
# SIGNALS
# -----------------------------------------------------------------------------

def build_panel(bench: pd.Series, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """One row per (rebalance date, stock) with every signal and the forward 63-session RS."""
    def rs(n, lag=0):
        stock = close.shift(lag) / close.shift(lag + n) - 1
        mkt = bench.shift(lag) / bench.shift(lag + n) - 1
        return stock.sub(mkt, axis=0) * 100

    signals = {f"rs_{n}": rs(n) for n in RS_WINDOWS}
    signals["rs_126_skip21"] = rs(105, lag=21)
    rs10 = rs(10)
    signals["burst_share"] = (rs10 / signals["rs_63"] * 100).where(signals["rs_63"] > 0)
    fwd = (close.shift(-HORIZON) / close - 1).sub(bench.shift(-HORIZON) / bench - 1, axis=0) * 100
    signals["dist_20d_high"] = (close / high.rolling(20, min_periods=15).max() - 1) * 100

    adx, di_gap, rsi = {}, {}, {}
    for sym in close.columns:
        ok = close[sym].notna() & high[sym].notna() & low[sym].notna()
        if ok.sum() < 40:
            continue
        df = pd.DataFrame({"HIGH_PRICE": high[sym][ok], "LOW_PRICE": low[sym][ok], "CLOSE_PRICE": close[sym][ok]})
        a = compute_adx(df)
        adx[sym], di_gap[sym] = a["ADX"], a["PLUS_DI"] - a["MINUS_DI"]
        rsi[sym] = compute_rsi(close[sym][ok])
    signals["adx"] = pd.DataFrame(adx).reindex(close.index)
    signals["di_gap"] = pd.DataFrame(di_gap).reindex(close.index)
    signals["rsi"] = pd.DataFrame(rsi).reindex(close.index)

    rebal = close.index[WARMUP::REBALANCE_EVERY]
    rebal = [d for d in rebal if close.index.get_loc(d) + HORIZON < len(close.index)]
    rows = []
    for d in rebal:
        frame = pd.DataFrame({k: v.loc[d] for k, v in signals.items()})
        frame["fwd_rs"] = fwd.loc[d]
        frame["date"] = d
        rows.append(frame.dropna(subset=["fwd_rs", "rs_126"]))
    panel = pd.concat(rows).rename_axis("symbol").reset_index()
    panel["sector"] = panel["symbol"].map(sector_of)
    panel["period"] = np.where(panel["date"] <= IS_END, "IS 2021-23", "OOS 2024-26")
    panel["burst"] = panel["burst_share"] > 50
    panel["strong_trend"] = (panel["adx"] > 25) & (panel["di_gap"] > 0)
    panel["overbought"] = panel["rsi"] > 70
    panel["near_high"] = panel["dist_20d_high"] >= -2
    return panel


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

def rank_signal(panel: pd.DataFrame, signal: str) -> dict:
    ics, spreads, hits, dates = [], [], [], []
    for d, g in panel.groupby("date"):
        g = g.dropna(subset=[signal])
        if len(g) < 25:
            continue
        ics.append(g[signal].rank().corr(g["fwd_rs"].rank()))
        top = g[g[signal] >= g[signal].quantile(0.8)]
        spreads.append(top["fwd_rs"].mean() - g["fwd_rs"].mean())
        hits.append(g.nlargest(TOP_N, signal)["fwd_rs"].mean() > 0)
        dates.append(d)
    ics = np.array(ics)
    q = ics[::3]  # non-overlapping quarters
    t = q.mean() / q.std(ddof=1) * np.sqrt(len(q)) if len(q) > 2 and q.std(ddof=1) > 0 else np.nan
    return {"dates": len(ics), "mean IC": ics.mean(), "IC>0 %": 100 * (ics > 0).mean(), "t (quarterly)": t,
            "top-fifth vs universe pp": np.mean(spreads), "top-8 beat Nifty 500 %": 100 * np.mean(hits)}


def conditional(panel: pd.DataFrame, flag: str) -> dict:
    """Inside the top fifth by 126-session RS on each date: forward RS with vs without the flag."""
    tops = []
    for d, g in panel.groupby("date"):
        tops.append(g[g["rs_126"] >= g["rs_126"].quantile(0.8)])
    top = pd.concat(tops)
    yes, no = top[top[flag] == True]["fwd_rs"], top[top[flag] == False]["fwd_rs"]  # noqa: E712
    return {"with": f"{yes.mean():+.2f} pp (n={len(yes)})", "without": f"{no.mean():+.2f} pp (n={len(no)})",
            "difference pp": yes.mean() - no.mean(), "with: beat market %": 100 * (yes > 0).mean(),
            "without: beat market %": 100 * (no > 0).mean()}


def current_rule(panel: pd.DataFrame) -> dict:
    """Our screen's technical rule: RS63 > +2 pp and +DI > -DI (all passers, equal weight)."""
    out = []
    for d, g in panel.groupby("date"):
        passers = g[(g["rs_63"] > 2) & (g["di_gap"] > 0)]
        if len(passers):
            out.append((passers["fwd_rs"].mean() - g["fwd_rs"].mean(), passers["fwd_rs"].mean() > 0))
    arr = np.array(out, dtype=float)
    return {"vs universe pp": arr[:, 0].mean(), "beat Nifty 500 %": 100 * arr[:, 1].mean(), "dates": len(arr)}


def fmt_table(rows: dict) -> str:
    df = pd.DataFrame(rows).T
    return df.to_markdown(floatfmt=".2f")


def main() -> None:
    bench, close, high, low, big_moves = load_data()
    panel = build_panel(bench, close, high, low)
    panel.to_csv(OUT_DIR / "momentum_study_panel.csv", index=False)

    lines = ["# Momentum study results", "",
             f"Data: {close.index.min():%d-%b-%Y} to {close.index.max():%d-%b-%Y}, {close.shape[1]} stocks, "
             f"{panel['date'].nunique()} rebalance dates ({panel.groupby('period')['date'].nunique().to_dict()}), "
             f"{len(panel)} stock-dates. Daily moves above 35% after adjustment: {big_moves}.",
             "", "Design, measures and limitations: see the docstring of research/momentum_study.py.", ""]

    lines += ["## 1. Which RS window predicts the next 63 sessions?", ""]
    for period, g in [("All", panel)] + list(panel.groupby("period")):
        rows = {s: rank_signal(g, s) for s in [f"rs_{n}" for n in RS_WINDOWS] + ["rs_126_skip21"]}
        lines += [f"### {period}", "", fmt_table(rows), ""]

    lines += ["## 2. Inside the top fifth by 126-session RS: do the other indicators help?", ""]
    for period, g in [("All", panel)] + list(panel.groupby("period")):
        rows = {f: conditional(g, f) for f in ["burst", "strong_trend", "overbought", "near_high"]}
        lines += [f"### {period}", "", fmt_table(rows), ""]

    lines += ["## 3. Our current technical rule (RS63 > +2 pp and bullish DI)", ""]
    rows = {p: current_rule(g) for p, g in [("All", panel)] + list(panel.groupby("period"))}
    lines += [fmt_table(rows), ""]

    (OUT_DIR / "momentum_study_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
