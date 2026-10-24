"""
Three timelines from the 28-Sep-2026 snapshot (added 21-Oct-2026 at the professor's request, 14-Oct):

  1. Ours (technical-led)      what we hold: the frozen 25-Sep ranking, technicals rank the stocks (6-month
                               relative strength, skipping the last month) and fundamentals only screen them.
  2. Reverse (fundamental-led) the same screens and the same eligible stocks, but the order reversed:
                               fundamentals rank the stocks and technicals only screen them. Rank = average
                               percentile of ROCE, OPM, latest-quarter profit and sales growth (YoY) and low
                               debt/equity (a missing figure counts as the middle); top 8, equal-risk weights.
  3. Original strict 4         the first screen of 24-Sep (git 8b86319): technicals first (RS vs Nifty 500
                               > +2 pp and a bullish trend), then the full multi-year fundamental screen.
                               Only 4 stocks passed (WELCORP, FINCABLES, VOLTAMP, ELGIEQUIP); we then widened
                               the criteria to reach 8. The professor: the minimum does not matter. Equal
                               weights (four stocks cannot meet a 15% cap).

All three are valued the same way, so only the stock selection differs: 97% of Rs 1 crore in whole shares
at the 28-Sep close, the rest in cash (no interest, no hedge, no stop-loss replacements: buy-and-hold), held to the latest
close; the same Rs 1 crore in the Nifty 500 TRI alongside. Risk-free: the 91-day T-bill. Beta for alpha:
each portfolio's one-year beta before the snapshot (a regression on a few weeks is noise).

Outputs: output/timelines_growth.csv, output/timelines_summary.csv, output/timelines_holdings.csv.
Usage:   python timelines.py
"""

import glob
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (EQUITY_ALLOCATION_PCT, EVALUATION_START_DATE, HISTORICAL_OHLCV_CSV, INVESTED_COUNT,
                    LOCKED_PORTFOLIO_SYMBOLS, OUTPUT_DIR, PRINCIPAL_INR)

logger = logging.getLogger("timelines")

GROWTH_CSV = OUTPUT_DIR / "timelines_growth.csv"
SUMMARY_CSV = OUTPUT_DIR / "timelines_summary.csv"
HOLDINGS_CSV = OUTPUT_DIR / "timelines_holdings.csv"
FROZEN_RANKING = "data/locked/selection_ranking_2026-09-25.csv"
ORIGINAL_STRICT = ["WELCORP", "FINCABLES", "VOLTAMP", "ELGIEQUIP"]
LABELS = {"ours": "1. Ours: technicals rank, fundamentals screen",
          "reverse": "2. Reverse: fundamentals rank, technicals screen",
          "original": "3. Original strict screen (4 stocks)"}
RULES = {"ours": "Top 8 of the eligible stocks by 6-month RS (skip 1 month); fundamentals are a pass/fail screen.",
         "reverse": "Top 8 of the same eligible stocks by a fundamental quality score (ROCE, OPM, quarterly profit "
                    "and sales growth, low debt/equity); technicals are a pass/fail screen.",
         "original": "Technical screen first (RS vs Nifty 500 > +2 pp and bullish), then the full multi-year "
                     "fundamental screen (24-Sep): only 4 passed."}


def fundamental_score(eligible: pd.DataFrame, review: pd.DataFrame) -> pd.Series:
    """Average percentile (0-1) of ROCE, OPM, quarterly profit and sales growth, and low debt/equity."""
    f = pd.DataFrame(index=eligible["symbol"])
    f["roce"] = review["roce"].reindex(f.index)
    f["opm"] = review["opm"].reindex(f.index)
    f["profit_yoy"] = eligible.set_index("symbol")["qtr_profit_yoy"]
    f["sales_yoy"] = eligible.set_index("symbol")["qtr_sales_yoy"]
    f["low_de"] = -review["debt_to_equity"].reindex(f.index)
    return f.rank(pct=True).fillna(0.5).mean(axis=1).sort_values(ascending=False)


def pick(name: str, closes: pd.DataFrame) -> pd.Series:
    """Weights (fractions of the equity sleeve) of one timeline's stocks."""
    from weights import compute_portfolio_weights

    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    stock_data = stock_data[pd.to_datetime(stock_data["DATE1"], format="mixed") <= pd.Timestamp(EVALUATION_START_DATE)]
    if name == "original":
        return pd.Series(1 / len(ORIGINAL_STRICT), index=ORIGINAL_STRICT)
    if name == "ours":
        symbols = list(LOCKED_PORTFOLIO_SYMBOLS[:INVESTED_COUNT])
    else:
        eligible = pd.read_csv(FROZEN_RANKING)
        review = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUTPUT_DIR / "*_full_review_table.csv"))])
        review = review.drop_duplicates("symbol").set_index("symbol")
        symbols = list(fundamental_score(eligible, review).index[:INVESTED_COUNT])
    w = compute_portfolio_weights(stock_data, symbols).set_index("symbol")["weight_pct"] / 100
    return w / w.sum()


def value_path(weights: pd.Series, closes: pd.DataFrame) -> Dict[str, object]:
    """Rs value each session: whole shares bought at the snapshot close with 97% of the principal, plus cash."""
    start = pd.Timestamp(EVALUATION_START_DATE)
    px = closes.loc[closes.index >= start, weights.index].ffill()
    sleeve = PRINCIPAL_INR * EQUITY_ALLOCATION_PCT / 100
    shares = np.floor(sleeve * weights / px.iloc[0])
    cash = PRINCIPAL_INR - float((shares * px.iloc[0]).sum())
    value = px.mul(shares, axis=1).sum(axis=1) + cash
    holdings = pd.DataFrame({"symbol": weights.index, "weight_pct": (weights * 100).round(2).values,
                             "shares": shares.astype(int).values, "entry_price": px.iloc[0].round(2).values,
                             "latest_price": px.iloc[-1].round(2).values,
                             "return_pct": ((px.iloc[-1] / px.iloc[0] - 1) * 100).round(2).values})
    return {"value": value, "holdings": holdings}


def one_year_beta(weights: pd.Series, mkt_excess: pd.Series, rf: pd.Series) -> float:
    from risk_model import ols, portfolio_returns, stock_return_matrix

    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    r = stock_return_matrix(stock_data, list(weights.index))
    r = r[r.index <= pd.Timestamp(EVALUATION_START_DATE)].iloc[-245:]
    p = portfolio_returns(r, weights)
    j = pd.concat([(p - rf.reindex(p.index)).rename("y"), mkt_excess.reindex(p.index).rename("m")], axis=1).dropna()
    return float(ols(j["y"], j[["m"]])["coef"]["m"])


def run() -> Dict[str, pd.DataFrame]:
    from performance import window_metrics
    from risk_free import daily_rate, risk_free_3m
    from risk_model import excess_returns, load_factor_series
    from tracker import load_closes

    closes = load_closes()
    factors = load_factor_series()
    fx = excess_returns(factors)
    start = pd.Timestamp(EVALUATION_START_DATE)
    tri = factors["nifty500_tri"].reindex(closes.index[closes.index >= start]).ffill()
    growth = pd.DataFrame({"Nifty 500 TRI": PRINCIPAL_INR * tri / tri.iloc[0]})
    rows: List[Dict] = []
    holdings = []
    rf_annual, _ = risk_free_3m()
    for key in ("ours", "reverse", "original"):
        w = pick(key, closes)
        path = value_path(w, closes)
        growth[LABELS[key]] = path["value"]
        port_r = path["value"].pct_change().iloc[1:]
        mkt_r = tri.pct_change().iloc[1:]
        rf = pd.Series(daily_rate(rf_annual), index=port_r.index)
        beta = one_year_beta(w, fx["mkt_excess"], fx["rf"])
        m = window_metrics(port_r, mkt_r, rf, LABELS[key], beta_override=beta)
        m["stocks"] = ", ".join(w.index)
        m["rule"] = RULES[key]
        m["value_inr"] = round(float(path["value"].iloc[-1]))
        m["vs_nifty500_inr"] = round(float(path["value"].iloc[-1] - growth["Nifty 500 TRI"].iloc[-1]))
        rows.append(m)
        holdings.append(path["holdings"].assign(timeline=LABELS[key]))
    summary = pd.DataFrame(rows)
    growth.index.name = "date"
    growth.round(2).reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(GROWTH_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    pd.concat(holdings).to_csv(HOLDINGS_CSV, index=False)
    for r in rows:
        logger.info("%s: %+.2f%% (%s vs Nifty 500).", r["window"], r["portfolio_return_pct"], f"{r['vs_nifty500_inr']:+,}")
    return {"growth": growth, "summary": summary, "holdings": pd.concat(holdings)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out = run()
    pd.set_option("display.width", 250)
    print(out["summary"][["window", "stocks", "portfolio_return_pct", "benchmark_return_pct", "excess_return_pp",
                          "jensen_alpha_period_pp", "beta_vs_nifty500", "sortino_portfolio", "max_drawdown_portfolio_pct",
                          "xirr_portfolio_pct", "xirr_benchmark_pct", "vs_nifty500_inr"]].to_string(index=False))
    print(out["holdings"].to_string(index=False))
