"""
Where did the return come from? Live P&L attribution against the Nifty 500 (added 24-Oct-2026).

From the trade ledger, each day's holdings at the previous close earn that day's return. Every stock's
daily P&L is split with its beta vs the Nifty 500 TRI (one year of daily returns up to the snapshot, so the
split uses what was known when we invested):

  market part     value(t-1) x beta x Nifty 500 return(t)   what the market gave the stock
  stock-specific  the rest                                   the stock's own move: our selection, split into
    theme         value(t-1) x the capex universe's own move that day (equal weight of the 157 stocks, minus its
                  beta x the Nifty 500): what any capex stock got for being a capex stock
    picks         the rest: our stocks against the average capex stock

The gap between the portfolio's P&L and the same Rs 1 crore in the Nifty 500 TRI then splits exactly into:

  Market exposure   market part of the stocks - the benchmark's P&L   (beta and the money not in stocks)
  Stock selection   sum of the stock-specific parts = capex theme + our picks within the theme
  Hedge             the puts' P&L (bought, sold, marked at settlement)
  Cash              interest on the cash

Outputs: output/attribution_summary.csv (the four effects), output/attribution_stocks.csv (one row per stock
ever held: P&L, market and specific parts, holding period).
Usage:   python attribution.py
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import EVALUATION_START_DATE, HISTORICAL_OHLCV_CSV, OUTPUT_DIR, PRINCIPAL_INR

logger = logging.getLogger("attribution")

SUMMARY_CSV = OUTPUT_DIR / "attribution_summary.csv"
STOCKS_CSV = OUTPUT_DIR / "attribution_stocks.csv"


def split_pnl(shares: pd.DataFrame, closes: pd.DataFrame, mkt_r: pd.Series, beta: pd.Series,
              theme_r: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per stock: total, market and specific P&L from daily holdings (`shares`: end-of-day shares, sessions x
    symbols) and closes. A day's P&L = shares held at the previous close x the price change. `theme_r`: the
    theme's own daily move beyond its beta (splits the specific part into theme and picks)."""
    held = shares.shift(1).fillna(0)
    pnl = (held * closes.diff()).fillna(0)
    value_prev = held * closes.shift(1)
    market = value_prev.mul(mkt_r, axis=0).mul(beta.reindex(shares.columns), axis=1).fillna(0)
    out = pd.DataFrame({"pnl_inr": pnl.sum(), "market_part_inr": market.sum()})
    out["specific_part_inr"] = out["pnl_inr"] - out["market_part_inr"]
    theme = value_prev.mul(theme_r if theme_r is not None else 0.0, axis=0).fillna(0).sum()
    out["theme_part_inr"] = theme
    out["picks_part_inr"] = out["specific_part_inr"] - theme
    return out


def run() -> Optional[Dict[str, pd.DataFrame]]:
    from greeks import stock_betas
    from risk_model import load_factor_series, stock_return_matrix
    from tracker import TRACKER_DAILY_CSV, TRADES_CSV, load_closes, parse_option

    if not TRADES_CSV.exists() or not TRACKER_DAILY_CSV.exists():
        return None
    ledger = pd.read_csv(TRADES_CSV)
    ledger["date"] = pd.to_datetime(ledger["date"])
    daily = pd.read_csv(TRACKER_DAILY_CSV, parse_dates=["date"]).set_index("date")
    sessions = daily.index
    stock_trades = ledger[~ledger["instrument"].map(lambda i: bool(parse_option(i)))]
    symbols = sorted(stock_trades["instrument"].unique())
    closes = load_closes().reindex(sessions).ffill()[symbols]

    signed = stock_trades.assign(q=np.where(stock_trades["action"].str.upper() == "BUY", 1, -1) * stock_trades["quantity"])
    flows = signed.pivot_table(index="date", columns="instrument", values="q", aggfunc="sum").reindex(sessions).fillna(0)
    shares = flows.cumsum()[symbols]
    # A trade's own day: bought at its price, not the previous close; correct the day's P&L to the fill price
    fill_adj = pd.Series(0.0, index=symbols)
    for _, t in stock_trades.iterrows():
        if t["date"] in closes.index and t["date"] > sessions[0]:
            q = (1 if str(t["action"]).upper() == "BUY" else -1) * float(t["quantity"])
            fill_adj[t["instrument"]] += q * (float(closes.loc[t["date"], t["instrument"]]) - float(t["price"]))

    factors = load_factor_series()
    tri = factors["nifty500_tri"].reindex(sessions).ffill()
    mkt_r = tri.pct_change().fillna(0)
    rets = stock_return_matrix(pd.read_csv(HISTORICAL_OHLCV_CSV), symbols)
    rets = rets[rets.index <= pd.Timestamp(EVALUATION_START_DATE)]
    beta = stock_betas(rets, factors["nifty500_tri"].pct_change(fill_method=None))

    # The capex theme: our 157-stock universe, equal weight, net of its own beta x the Nifty 500
    from config import PORTFOLIO_SYMBOLS
    all_closes = load_closes()
    uni = all_closes[[c for c in PORTFOLIO_SYMBOLS if c in all_closes.columns]].pct_change(fill_method=None)
    uni_r = uni.mean(axis=1)
    beta_theme = float(stock_betas(uni_r[uni_r.index <= pd.Timestamp(EVALUATION_START_DATE)].to_frame("theme"),
                                   factors["nifty500_tri"].pct_change(fill_method=None))["theme"])
    theme_r = (uni_r.reindex(sessions) - beta_theme * mkt_r).fillna(0)
    theme_r.iloc[0] = 0.0

    per = split_pnl(shares, closes, mkt_r, beta, theme_r)
    per["pnl_inr"] += fill_adj
    per["specific_part_inr"] += fill_adj
    per["picks_part_inr"] += fill_adj
    first_buy = stock_trades[stock_trades["action"].str.upper() == "BUY"].groupby("instrument")["date"].min()
    last_sell = stock_trades[stock_trades["action"].str.upper() == "SELL"].groupby("instrument")["date"].max()
    invested = stock_trades[stock_trades["action"].str.upper() == "BUY"].assign(
        v=lambda d: d["quantity"] * d["price"]).groupby("instrument")["v"].sum()
    held_now = shares.iloc[-1] > 0
    stocks = pd.DataFrame({
        "symbol": symbols,
        "status": ["held" if held_now[s] else "sold" for s in symbols],
        "from": [first_buy[s].date().isoformat() for s in symbols],
        "to": [sessions[-1].date().isoformat() if held_now[s] else last_sell[s].date().isoformat() for s in symbols],
        "invested_inr": invested.reindex(symbols).round(0).values,
        "beta_vs_nifty500": beta.reindex(symbols).round(2).values,
        "pnl_inr": per["pnl_inr"].round(0).values,
        "market_part_inr": per["market_part_inr"].round(0).values,
        "specific_part_inr": per["specific_part_inr"].round(0).values,
        "theme_part_inr": per["theme_part_inr"].round(0).values,
        "picks_part_inr": per["picks_part_inr"].round(0).values,
    })
    stocks["return_pct"] = (stocks["pnl_inr"] / stocks["invested_inr"] * 100).round(2)
    stocks["contribution_pct_of_capital"] = (stocks["pnl_inr"] / PRINCIPAL_INR * 100).round(2)
    stocks = stocks.sort_values("pnl_inr").reset_index(drop=True)

    total_pnl = float(daily["total_inr"].iloc[-1] - PRINCIPAL_INR)
    bench_pnl = float(daily["nifty500_inr"].iloc[-1] - PRINCIPAL_INR)
    opt = ledger[ledger["instrument"].map(lambda i: bool(parse_option(i)))]
    opt_cash = float((np.where(opt["action"].str.upper() == "SELL", 1, -1) * opt["quantity"] * opt["price"]).sum())
    hedge = opt_cash + float(daily["options_inr"].iloc[-1])
    stock_pnl = float(stocks["pnl_inr"].sum())
    cash = total_pnl - stock_pnl - hedge
    market = float(stocks["market_part_inr"].sum())
    effects = [
        ("Portfolio P&L", total_pnl, "stocks + puts + cash, from the ledger"),
        ("Same Rs 1 crore in the Nifty 500 TRI", bench_pnl, "the benchmark"),
        ("Gap vs the Nifty 500", total_pnl - bench_pnl, "= the four effects below"),
        ("1. Market exposure", market - bench_pnl,
         "market part of the stocks minus the benchmark: beta above or below 1 and the money held outside stocks"),
        ("2. Stock selection", float(stocks["specific_part_inr"].sum()),
         "the stocks' own moves beyond what their beta explains (the picks and the capex theme)"),
        ("2a. of which the capex theme", float(stocks["theme_part_inr"].sum()),
         f"the capex universe's own move (equal weight, net of its beta {beta_theme:.2f} x the Nifty 500)"),
        ("2b. of which our picks within the theme", float(stocks["picks_part_inr"].sum()),
         "our stocks against the average capex stock"),
        ("3. Hedge (Nifty puts)", hedge, "premium paid and received, puts marked at settlement"),
        ("4. Cash", cash, "interest on the cash (liquid ETF / overnight rate)"),
    ]
    summary = pd.DataFrame(effects, columns=["item", "inr", "note"])
    summary["inr"] = summary["inr"].round(0)
    summary["pct_of_capital"] = (summary["inr"] / PRINCIPAL_INR * 100).round(2)
    summary.insert(0, "as_of", sessions[-1].date().isoformat())
    summary.to_csv(SUMMARY_CSV, index=False)
    stocks.to_csv(STOCKS_CSV, index=False)
    logger.info("Gap vs Nifty 500 Rs %s: market %s, selection %s (theme %s, picks %s), hedge %s, cash %s.",
                *(f"{v:,.0f}" for v in summary["inr"].iloc[2:]))
    return {"summary": summary, "stocks": stocks}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    res = run()
    if res:
        pd.set_option("display.width", 220)
        print(res["summary"].to_string(index=False))
        print(res["stocks"].to_string(index=False))
