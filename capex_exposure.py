"""
Does the portfolio capture the capex cycle? (added 29-Sep-2026: the professor's test for the theme)

Daily returns of the invested portfolio (current weights, past year) are regressed on candidate benchmarks,
one at a time: our capex universe (all stocks of the Screener universe, equal weight: the capex cycle as
our theme defines it), NSE's sector and style indices (Capital Goods, India Manufacturing, Multicap
Infrastructure, Infrastructure, Energy, Power, Cement, Metal, Smallcap 500, Nifty500 Momentum 50) and the
Nifty 500. The benchmark with the highest R^2 is what the portfolio actually behaves like.

A factor model then separates the capex exposure from the two styles it could be confused with:
    r_p = a + b_m N500 + b_capex (capex universe - N500) + b_size (Smallcap 500 - N500)
            + b_mom (Momentum 50 - N500) + e
A significant b_capex with the styles controlled for means the portfolio carries the capex cycle itself,
not only a small-cap or momentum tilt. The capex and size factors are correlated (mid-caps dominate the
universe), which widens the standard errors; the correlation is reported.

Outputs: output/capex_exposure.csv (one row per benchmark) and output/capex_factor_model.csv.
Usage:   python capex_exposure.py [--as-of YYYY-MM-DD]
"""

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import BENCHMARK_INDEX_NAME, HISTORICAL_OHLCV_CSV, OUTPUT_DIR, PORTFOLIO_SYMBOLS, RISK_SUMMARY_OUTPUT_CSV
from risk_model import ols, portfolio_returns, stock_return_matrix

logger = logging.getLogger("capex_exposure")

EXPOSURE_CSV = OUTPUT_DIR / "capex_exposure.csv"
FACTOR_MODEL_CSV = OUTPUT_DIR / "capex_factor_model.csv"
CAPEX_LABEL = "Our capex universe (equal weight)"
SMALLCAP_INDEX, MOMENTUM_INDEX = "Nifty Smallcap 500", "Nifty500 Momentum 50"
INDICES = ["Nifty Capital Goods", "Nifty India Manufacturing", "Nifty500 Multicap Infrastructure 50:30:20",
           "Nifty Infrastructure", "Nifty Energy", "Nifty Power", "Nifty Cement", "Nifty Metal",
           SMALLCAP_INDEX, MOMENTUM_INDEX, BENCHMARK_INDEX_NAME]
WINDOW = 245           # about a year of sessions
MIN_DAYS = 60          # an index with a shorter history (e.g. a recent launch) is reported with its day count


def single_index_table(port: pd.Series, bench: Dict[str, pd.Series]) -> pd.DataFrame:
    """Beta and R^2 of the portfolio on each benchmark over their common days (up to WINDOW)."""
    rows = []
    for name, x in bench.items():
        j = pd.concat([port, x], axis=1, sort=True).dropna().iloc[-WINDOW:]
        if len(j) < MIN_DAYS:
            continue
        beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
        rows.append({"benchmark": name, "days": len(j), "beta": round(beta, 2),
                     "r_squared": round(float(j.corr().iloc[0, 1] ** 2), 3),
                     "benchmark_return_pct": round(float((1 + j.iloc[:, 1]).prod() - 1) * 100, 1)})
    return pd.DataFrame(rows).sort_values("r_squared", ascending=False).reset_index(drop=True)


def factor_model(port: pd.Series, market: pd.Series, capex: pd.Series, size: Optional[pd.Series],
                 mom: Optional[pd.Series]) -> pd.DataFrame:
    """Market + capex (+ size, momentum where available) regression of the portfolio's daily returns."""
    X = pd.DataFrame({"market": market, "capex": capex - market})
    if size is not None:
        X["size"] = size - market
    if mom is not None:
        X["momentum"] = mom - market
    j = pd.concat([port.rename("p"), X], axis=1, sort=True).dropna().iloc[-WINDOW:]
    fit = ols(j["p"], j.drop(columns="p"))
    rows = [{"term": "alpha (annualised %)", "coef": round(fit["coef"]["const"] * 252 * 100, 2),
             "t": round(fit["t"]["const"], 2)}]
    rows += [{"term": k, "coef": round(fit["coef"][k], 3), "t": round(fit["t"][k], 2)} for k in X.columns]
    rows += [{"term": "R^2", "coef": round(fit["r2"], 3), "t": np.nan},
             {"term": "days", "coef": fit["n"], "t": np.nan}]
    if "size" in X:
        rows.append({"term": "corr(capex, size)", "coef": round(float(j["capex"].corr(j["size"])), 2), "t": np.nan})
    return pd.DataFrame(rows)


def run(as_of: Optional[date] = None) -> Dict[str, pd.DataFrame]:
    from fetch_data import NSEBhavcopyFetcher

    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV)
    universe = stock_return_matrix(stock_data, [s for s in PORTFOLIO_SYMBOLS if s in set(stock_data["SYMBOL"])])
    holdings = stock_return_matrix(stock_data, risk["symbol"].tolist())
    if as_of is not None:
        universe, holdings = universe.loc[:pd.Timestamp(as_of)], holdings.loc[:pd.Timestamp(as_of)]
    port = portfolio_returns(holdings, risk.set_index("symbol")["weight_pct"] / 100)
    capex = universe.mean(axis=1).rename(CAPEX_LABEL)

    end = (as_of or holdings.index.max().date())
    fetcher = NSEBhavcopyFetcher()
    bench = {CAPEX_LABEL: capex}
    for name in INDICES:
        closes = fetcher.fetch_index_closes(end - timedelta(days=400), end, index_name=name)
        if closes is None or closes.empty:
            logger.warning("No closes for %s; left out.", name)
            continue
        closes["Date"] = pd.to_datetime(closes["Date"])
        # Returns only between consecutive sessions: an index missing from a day's file (some NSE indices have
        # gaps) gives NaN, not a multi-day move set against a one-day portfolio return
        level = closes.drop_duplicates("Date").set_index("Date")["Close"].reindex(port.index)
        bench[name] = level.pct_change(fill_method=None)

    table = single_index_table(port, bench)
    model = factor_model(port, bench[BENCHMARK_INDEX_NAME], capex, bench.get(SMALLCAP_INDEX), bench.get(MOMENTUM_INDEX))
    table.to_csv(EXPOSURE_CSV, index=False)
    model.to_csv(FACTOR_MODEL_CSV, index=False)
    logger.info("Capex exposure: best fit %s (R^2 %.2f).", table.iloc[0]["benchmark"], table.iloc[0]["r_squared"])
    return {"table": table, "model": model}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    as_of_arg = date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1]) if "--as-of" in sys.argv else None
    out = run(as_of_arg)
    print(out["table"].to_string(index=False))
    print(out["model"].to_string(index=False))
