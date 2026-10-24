"""
Portfolio delta, the puts' Greeks and a sensitivity table (added 24-Oct-2026).

The portfolio's one market risk factor is the Nifty 50 (the index the puts are written on), so every
position is expressed in Nifty terms:

  Stocks   delta = market value x beta vs the Nifty 50 (one year of daily returns): the rupees the stock
           sleeve gains for a 100% Nifty move, i.e. value x beta / 100 per 1% move. In Nifty units:
           value x beta / spot. Stocks have no gamma, vega or theta.
  Puts     Black-Scholes Greeks at the implied volatility of the day's NSE settlement price, the T-bill
           rate and the time to expiry, times the units held:
             delta  Nifty units (negative: the puts gain when the Nifty falls)
             gamma  change in delta per 1% Nifty move (the puts' cover grows as the Nifty falls)
             vega   Rs per 1 point of implied volatility
             theta  Rs per calendar day (the cost of holding the insurance)
             rho    Rs per 1 percentage point of the interest rate
  Net      stocks + puts; "share of the stocks' delta hedged" = -puts delta / stocks delta.

Sensitivity: the Rs P&L of the stocks (beta x move) and of the puts (full Black-Scholes revaluation, not the
delta approximation) for Nifty moves of -10% to +10%, now and at the professor's exit (29-Oct, with the time
decay to that date), plus the puts at implied volatility +-5 points. The stocks' own (non-market) risk is
left out: the VaR section covers it.

Outputs: output/greeks_summary.csv, output/greeks_stocks.csv, output/greeks_sensitivity.csv.
Usage:   python greeks.py
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import EVALUATION_END_DATE, HISTORICAL_OHLCV_CSV, OUTPUT_DIR

logger = logging.getLogger("greeks")

SUMMARY_CSV = OUTPUT_DIR / "greeks_summary.csv"
STOCKS_CSV = OUTPUT_DIR / "greeks_stocks.csv"
SENSITIVITY_CSV = OUTPUT_DIR / "greeks_sensitivity.csv"
MOVES_PCT = [-10, -7.5, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7.5, 10]
VOL_SHOCK = 0.05
WINDOW = 245


def stock_betas(returns: pd.DataFrame, nifty: pd.Series) -> pd.Series:
    """Beta of each column of `returns` vs `nifty` over the last WINDOW common sessions."""
    out = {}
    for sym in returns.columns:
        j = pd.concat([returns[sym], nifty], axis=1, sort=True).dropna().iloc[-WINDOW:]
        out[sym] = float(j.cov().iloc[0, 1] / j.iloc[:, 1].var()) if len(j) > 20 else np.nan
    return pd.Series(out)


def put_book(units: int, strike: float, t: float, r: float, vol: float, spot: float) -> Dict[str, float]:
    """The puts' Greeks for the whole holding."""
    from option_pricing import bs_greeks
    g = bs_greeks(spot, strike, t, r, vol, "PE")
    return {"delta_units": units * g["delta"], "gamma_units_per_pt": units * g["gamma"],
            "vega_inr": units * g["vega"], "theta_inr": units * g["theta"], "rho_inr": units * g["rho"]}


def sensitivity(stock_delta_inr: float, units: int, strike: float, t_now: float, t_exit: float, r: float,
                vol: float, spot: float) -> pd.DataFrame:
    """Rs P&L by Nifty move: stocks (beta), puts now / at the exit / with IV shocked (full revaluation)."""
    from option_pricing import bs_price
    p0 = bs_price(spot, strike, t_now, r, vol)
    rows = []
    for m in MOVES_PCT:
        s = spot * (1 + m / 100)
        stocks = stock_delta_inr * m / 100
        now = units * (bs_price(s, strike, t_now, r, vol) - p0)
        exit_ = units * (bs_price(s, strike, t_exit, r, vol) - p0)
        rows.append({"nifty_move_pct": m, "nifty_level": round(s), "stocks_pnl_inr": round(stocks),
                     "puts_pnl_now_inr": round(now), "total_now_inr": round(stocks + now),
                     "puts_pnl_at_exit_inr": round(exit_), "total_at_exit_inr": round(stocks + exit_),
                     "puts_pnl_iv_up5_inr": round(units * (bs_price(s, strike, t_now, r, vol + VOL_SHOCK) - p0)),
                     "puts_pnl_iv_down5_inr": round(units * (bs_price(s, strike, t_now, r, max(vol - VOL_SHOCK, 0.01))
                                                             - p0))})
    return pd.DataFrame(rows)


def run() -> Optional[Dict[str, pd.DataFrame]]:
    from option_pricing import implied_vol
    from risk_free import risk_free_3m
    from risk_model import fetch_nifty_derivatives, held_put, load_factor_series, stock_return_matrix
    from tracker import TRACKER_POSITIONS_CSV

    pos = pd.read_csv(TRACKER_POSITIONS_CSV).set_index("symbol")["value_inr"]
    returns = stock_return_matrix(pd.read_csv(HISTORICAL_OHLCV_CSV), list(pos.index))
    as_of = returns.index.max()
    nifty = load_factor_series()["nifty50"].pct_change(fill_method=None)
    beta = stock_betas(returns, nifty)
    stocks = pd.DataFrame({"symbol": pos.index, "value_inr": pos.values.round(0),
                           "beta_vs_nifty50": beta.reindex(pos.index).round(3).values})
    stocks["delta_inr_per_1pct"] = (stocks["value_inr"] * stocks["beta_vs_nifty50"] / 100).round(0)
    stock_delta = float((stocks["value_inr"] * stocks["beta_vs_nifty50"]).sum())  # Rs per 100% Nifty move

    held = held_put()
    fo = fetch_nifty_derivatives(as_of.date(), use_cache=True, save=False) if held else None
    if not held or fo is None or fo.empty:
        logger.warning("No puts held or no F&O file for %s: Greeks skipped.", as_of.date())
        return None
    o = fo[(fo["FinInstrmTp"] == "IDO") & (fo["OptnTp"] == "PE") & (fo["XpryDt"] == held["expiry"])
           & np.isclose(fo["StrkPric"], held["strike"])]
    if o.empty:
        logger.warning("%s not in the F&O file of %s: Greeks skipped.", held["instrument"], as_of.date())
        return None
    spot, price = float(o["UndrlygPric"].iloc[0]), float(o["SttlmPric"].iloc[0])
    expiry = pd.Timestamp(held["expiry"])
    t_now = (expiry - as_of).days / 365
    t_exit = max((expiry - pd.Timestamp(EVALUATION_END_DATE)).days, 0) / 365
    r = risk_free_3m()[0]
    vol = implied_vol(price, spot, held["strike"], t_now, r)
    units = int(held["units"])
    book = put_book(units, held["strike"], t_now, r, vol, spot)
    stock_units = stock_delta / spot
    net_units = stock_units + book["delta_units"]
    gamma_1pct = book["gamma_units_per_pt"] * spot / 100  # delta change (Nifty units) per 1% move

    rows = [
        ("As of", str(as_of.date()), ""),
        ("Nifty 50 spot", round(spot, 2), "NSE F&O file, underlying price"),
        ("Put held", held["instrument"], f"{units} units ({units // 65} lots), settlement Rs {price:.2f}"),
        ("Implied volatility (%)", round(vol * 100, 2), "from the settlement price"),
        ("Days to expiry", int((expiry - as_of).days), ""),
        ("Risk-free (%)", round(r * 100, 4), "91-day T-bill"),
        ("Stocks delta (Nifty units)", round(stock_units), "sum of value x beta / spot"),
        ("Stocks delta (Rs per 1% Nifty move)", round(stock_delta / 100), ""),
        ("Puts delta per unit", round(book["delta_units"] / units, 4), "Black-Scholes"),
        ("Puts delta (Nifty units)", round(book["delta_units"]), "negative: gains when the Nifty falls"),
        ("Puts delta (Rs per 1% Nifty move)", round(book["delta_units"] * spot / 100), ""),
        ("Net delta (Nifty units)", round(net_units), "stocks + puts"),
        ("Net delta (Rs per 1% Nifty move)", round(net_units * spot / 100), ""),
        ("Share of the stocks' delta hedged (%)", round(-book["delta_units"] / stock_units * 100, 1),
         "rises as the Nifty falls (gamma)"),
        ("Puts gamma (delta change per 1% Nifty move, units)", round(gamma_1pct, 1),
         "a 1% fall makes the puts' delta this much more negative"),
        ("Puts vega (Rs per 1 IV point)", round(book["vega_inr"]), "a jump in fear raises the puts' value"),
        ("Puts theta (Rs per calendar day)", round(book["theta_inr"]), "the daily cost of the insurance"),
        ("Puts rho (Rs per 1 pp rate)", round(book["rho_inr"]), "small: rates barely matter"),
        ("Exit date for the sensitivity", EVALUATION_END_DATE, f"{(pd.Timestamp(EVALUATION_END_DATE) - as_of).days} days of time decay"),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value", "note"])
    sens = sensitivity(stock_delta, units, held["strike"], t_now, t_exit, r, vol, spot)
    summary.to_csv(SUMMARY_CSV, index=False)
    stocks.to_csv(STOCKS_CSV, index=False)
    sens.to_csv(SENSITIVITY_CSV, index=False)
    logger.info("Net delta %+.0f Nifty units (stocks %+.0f, puts %+.0f); vega Rs %.0f; theta Rs %.0f/day.",
                net_units, stock_units, book["delta_units"], book["vega_inr"], book["theta_inr"])
    return {"summary": summary, "stocks": stocks, "sensitivity": sens}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    res = run()
    if res:
        pd.set_option("display.width", 220)
        for k in ("summary", "stocks", "sensitivity"):
            print(res[k].to_string(index=False))
