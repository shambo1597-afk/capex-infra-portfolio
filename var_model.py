"""
Portfolio Value-at-Risk, marginal VaR and component VaR (added 21-Oct-2026: the professor's checklist).

Positions are the holdings at the latest close (trade ledger): the stocks at their market value and the
Nifty puts as a delta-equivalent Nifty 50 position (Black-Scholes delta at the implied volatility of the
day's settlement price). Risk comes from one year of daily returns of the stocks and the Nifty 50.

  Parametric (variance-covariance) VaR   z_c x sigma_P x sqrt(h); sigma_P^2 = x' S x (x = rupee positions,
                                         S = daily covariance), mean taken as zero (conservative).
  Historical VaR / CVaR (1 day)          the loss the current positions would have made on each of the past
                                         year's days; VaR = the (1-c) quantile, CVaR = the average beyond it.
  Marginal VaR of stock i                dVaR/dx_i = z_c (S x)_i / sigma_P: the VaR added per rupee more of i.
  Component VaR of stock i               x_i x marginal VaR_i; the components add up to the portfolio VaR,
                                         so each one is the stock's share of the risk.
  Horizon                                h = NSE sessions from the latest close to config.EVALUATION_END_DATE
                                         (29-Oct-2026, the professor's exit), scaled by sqrt(h).

The puts' delta understates their protection in a crash (gamma): the hedged VaR is a conservative figure.

Outputs: output/var_summary.csv, output/var_components.csv.
Usage:   python var_model.py
"""

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import EVALUATION_END_DATE, HISTORICAL_OHLCV_CSV, NSE_TRADING_HOLIDAYS, OUTPUT_DIR, TRADING_DAYS_PER_YEAR

logger = logging.getLogger("var_model")

VAR_SUMMARY_CSV = OUTPUT_DIR / "var_summary.csv"
VAR_COMPONENTS_CSV = OUTPUT_DIR / "var_components.csv"
Z = {0.95: 1.6449, 0.99: 2.3263}
WINDOW = 245
NIFTY = "NIFTY 50 (puts, delta)"


def sessions_to_horizon(as_of: date, horizon: str = EVALUATION_END_DATE) -> int:
    """NSE sessions after `as_of` up to and including the horizon date (at least 1)."""
    end = date.fromisoformat(horizon)
    if end <= as_of:
        return 1
    return max(1, int(np.busday_count(as_of + timedelta(days=1), end + timedelta(days=1),
                                      holidays=NSE_TRADING_HOLIDAYS)))


def var_tables(positions: pd.Series, returns: pd.DataFrame, horizon_sessions: int,
               base_value: Optional[float] = None) -> Dict[str, pd.DataFrame]:
    """VaR summary and per-position components for rupee `positions` (index = columns of `returns`); the
    percentages are of `base_value` (default: the positions' sum), so a hedge does not shrink the base."""
    r = returns[positions.index].dropna().iloc[-WINDOW:]
    cov = r.cov().values
    x = positions.values.astype(float)
    sigma = float(np.sqrt(x @ cov @ x))
    pnl = r.values @ x  # the current positions on each past day
    value = float(base_value if base_value is not None else positions.sum())
    rows = []
    for c in (0.95, 0.99):
        par1 = Z[c] * sigma
        hist1 = -float(np.quantile(pnl, 1 - c))
        cvar1 = -float(pnl[pnl <= -hist1].mean())
        rows += [
            {"measure": f"Parametric VaR {c:.0%}, 1 day", "inr": par1, "pct_of_value": par1 / value * 100},
            {"measure": f"Parametric VaR {c:.0%}, to the horizon ({horizon_sessions} sessions)",
             "inr": par1 * np.sqrt(horizon_sessions), "pct_of_value": par1 * np.sqrt(horizon_sessions) / value * 100},
            {"measure": f"Historical VaR {c:.0%}, 1 day", "inr": hist1, "pct_of_value": hist1 / value * 100},
            {"measure": f"Historical CVaR (expected shortfall) {c:.0%}, 1 day", "inr": cvar1,
             "pct_of_value": cvar1 / value * 100},
        ]
    standalone = Z[0.95] * np.sqrt(np.diag(cov)) * np.abs(x)
    marginal = Z[0.95] * (cov @ x) / sigma  # per rupee
    component = x * marginal
    comp = pd.DataFrame({
        "position": positions.index, "value_inr": x.round(0), "weight_pct": (x / value * 100).round(2),
        "standalone_var95_inr": standalone.round(0),
        "marginal_var95_per_lakh_inr": (marginal * 1e5).round(0),
        "component_var95_inr": component.round(0),
        "share_of_var_pct": (component / component.sum() * 100).round(1),
    })
    summary = pd.DataFrame(rows)
    summary["inr"] = summary["inr"].round(0)
    summary["pct_of_value"] = summary["pct_of_value"].round(2)
    diversification = standalone.sum() - Z[0.95] * sigma
    summary = pd.concat([summary, pd.DataFrame([
        {"measure": "Diversification benefit (sum of standalone 95% VaRs minus portfolio VaR), 1 day",
         "inr": round(diversification), "pct_of_value": round(diversification / value * 100, 2)},
        {"measure": "Stocks valued (Rs)", "inr": round(value), "pct_of_value": 100.0},
    ])], ignore_index=True)
    return {"summary": summary, "components": comp}


def put_delta_position(as_of: date) -> Optional[float]:
    """Rupee Nifty 50 exposure of the puts held: units x delta x spot (negative), from the day's F&O file."""
    from option_pricing import bs_delta, implied_vol
    from risk_free import risk_free_3m
    from risk_model import fetch_nifty_derivatives, held_put

    held = held_put()
    fo = fetch_nifty_derivatives(as_of, use_cache=True, save=False) if held else None
    if not held or fo is None or fo.empty:
        return None
    o = fo[(fo["FinInstrmTp"] == "IDO") & (fo["OptnTp"] == "PE") & (fo["XpryDt"] == held["expiry"])
           & np.isclose(fo["StrkPric"], held["strike"])]
    if o.empty:
        return None
    spot = float(o["UndrlygPric"].iloc[0])
    t = (pd.Timestamp(held["expiry"]) - pd.Timestamp(as_of)).days / 365
    r = risk_free_3m()[0]
    vol = implied_vol(float(o["SttlmPric"].iloc[0]), spot, held["strike"], t, r)
    if not np.isfinite(vol):
        return None
    return held["units"] * bs_delta(spot, held["strike"], t, r, vol) * spot


def run() -> Dict[str, pd.DataFrame]:
    from risk_model import load_factor_series, stock_return_matrix
    from tracker import TRACKER_POSITIONS_CSV

    pos = pd.read_csv(TRACKER_POSITIONS_CSV).set_index("symbol")["value_inr"]
    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    returns = stock_return_matrix(stock_data, list(pos.index))
    as_of = returns.index.max().date()
    nifty = load_factor_series()["nifty50"].pct_change(fill_method=None).rename(NIFTY)
    returns = returns.join(nifty, how="left")
    h = sessions_to_horizon(as_of)

    stocks_only = var_tables(pos, returns, h)
    put_x = put_delta_position(as_of)
    out_summary = stocks_only["summary"].rename(columns={"inr": "stocks_inr", "pct_of_value": "stocks_pct"})
    components = stocks_only["components"]
    if put_x is not None:
        hedged = var_tables(pd.concat([pos, pd.Series({NIFTY: put_x})]), returns, h, base_value=float(pos.sum()))
        hs = hedged["summary"].set_index("measure")
        out_summary["with_puts_inr"] = out_summary["measure"].map(hs["inr"])
        out_summary["with_puts_pct"] = out_summary["measure"].map(hs["pct_of_value"])
        components = hedged["components"]
    meta = pd.DataFrame([{"measure": "As of", "stocks_inr": str(as_of)},
                         {"measure": "Horizon", "stocks_inr": f"{EVALUATION_END_DATE} ({h} sessions)"}])
    out_summary = pd.concat([meta, out_summary], ignore_index=True)
    out_summary.to_csv(VAR_SUMMARY_CSV, index=False)
    components.to_csv(VAR_COMPONENTS_CSV, index=False)
    logger.info("VaR 95%% 1-day: Rs %s (stocks only).", f"{stocks_only['summary'].iloc[0]['inr']:,.0f}")
    return {"summary": out_summary, "components": components}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    res = run()
    print(res["summary"].to_string(index=False))
    print(res["components"].to_string(index=False))
