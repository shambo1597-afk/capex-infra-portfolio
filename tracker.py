"""
Live P&L of the Rs 1 crore from the 28-Sep-2026 snapshot: did we make the money?

The trade ledger (config.TRADES_CSV: date, instrument, action, quantity, price, note) is the record of
what the portfolio holds. It is created once, at the close of the first session on or after
EVALUATION_START_DATE: the stocks at that close with the current weights of the equity sleeve, and
the tail-hedge puts of the hedge plan at that day's NSE settlement price. Everything later (real
purchase prices, stop-loss exits, redeployments, the profit-trigger put roll) is recorded by editing
the ledger; the tracker never trades on its own.

Each session the portfolio is valued as
    stocks (shares x NSE close) + puts (lots x lot size x NSE settlement price) + cash,
cash earning the overnight rate (Nifty 1D Rate index, a liquid ETF's return). The same Rs 1 crore is
also valued in the Nifty 500 TRI and in the liquid fund from the snapshot close, so the answer to
"did we make money, and more than the alternatives?" is one row of output/tracker_summary.csv.
"""

import logging
import re
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    DERIVATIVES_DIR,
    EQUITY_ALLOCATION_PCT,
    EVALUATION_END_DATE,
    EVALUATION_START_DATE,
    HISTORICAL_OHLCV_CSV,
    OUTPUT_DIR,
    PRINCIPAL_INR,
    RISK_SUMMARY_OUTPUT_CSV,
    TRADES_CSV,
)

logger = logging.getLogger("tracker")

TRACKER_DAILY_CSV = OUTPUT_DIR / "tracker_daily.csv"
TRACKER_SUMMARY_CSV = OUTPUT_DIR / "tracker_summary.csv"
TRACKER_POSITIONS_CSV = OUTPUT_DIR / "tracker_positions.csv"
PUT_MARKS_CSV = DERIVATIVES_DIR / "option_marks.csv"
MIN_DAYS_FOR_XIRR = 30  # annualising a few days' return gives meaningless four-digit rates
LEDGER_COLUMNS = ["date", "instrument", "action", "quantity", "price", "note"]
_OPTION = re.compile(r"^NIFTY (\d{4}-\d{2}-\d{2}) (\d+(?:\.\d+)?) (PE|CE)$")


def parse_option(instrument: str) -> Optional[dict]:
    """'NIFTY 2026-12-29 22000 PE' -> {'expiry', 'strike', 'type'}; None for a stock."""
    m = _OPTION.match(str(instrument).strip())
    return None if not m else {"expiry": m.group(1), "strike": float(m.group(2)), "type": m.group(3)}


def load_closes() -> pd.DataFrame:
    """Close prices (sessions x symbols) from the committed price history."""
    df = pd.read_csv(HISTORICAL_OHLCV_CSV)
    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE"])]
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed")
    return df.pivot_table(index="DATE1", columns="SYMBOL", values="CLOSE_PRICE").sort_index()


def option_settlement(instrument: str, session: date, fetch=None) -> Optional[float]:
    """NSE settlement price of one Nifty option on one session (cached in PUT_MARKS_CSV)."""
    key = pd.Timestamp(session).strftime("%Y-%m-%d")
    marks = pd.read_csv(PUT_MARKS_CSV) if PUT_MARKS_CSV.exists() else pd.DataFrame(columns=["date", "instrument", "settle"])
    hit = marks[(marks["date"] == key) & (marks["instrument"] == instrument)]
    if not hit.empty:
        return float(hit["settle"].iloc[0])
    opt = parse_option(instrument)
    if opt is None:
        return None
    if fetch is None:
        from risk_model import fetch_nifty_derivatives
        fetch = lambda d: fetch_nifty_derivatives(d, use_cache=True, save=False)
    fo = fetch(pd.Timestamp(session).date())
    if fo is None or fo.empty:
        return None
    row = fo[(fo["FinInstrmTp"] == "IDO") & (fo["XpryDt"] == opt["expiry"]) & (fo["OptnTp"] == opt["type"])
             & np.isclose(fo["StrkPric"], opt["strike"])]
    if row.empty:
        return None
    settle = float(row["SttlmPric"].iloc[0])
    PUT_MARKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    marks = pd.concat([marks, pd.DataFrame([{"date": key, "instrument": instrument, "settle": settle}])],
                      ignore_index=True).sort_values(["date", "instrument"])
    marks.to_csv(PUT_MARKS_CSV, index=False)
    return settle


def initial_trades(snapshot: pd.Timestamp, closes: pd.Series, weights_pct: pd.Series, put_contract: Optional[str],
                   put_units: int, put_price: Optional[float]) -> pd.DataFrame:
    """The opening trades at the snapshot close: whole shares of each weight of the equity sleeve, then the puts."""
    equity = PRINCIPAL_INR * EQUITY_ALLOCATION_PCT / 100
    day = snapshot.strftime("%Y-%m-%d")
    rows = []
    for symbol, w in weights_pct.items():
        price = float(closes[symbol])
        rows.append({"date": day, "instrument": symbol, "action": "BUY", "quantity": int(equity * w / 100 // price),
                     "price": round(price, 2), "note": f"snapshot: {w:.2f}% of the equity sleeve at the close"})
    if put_contract and put_units and put_price is not None:
        rows.append({"date": day, "instrument": put_contract, "action": "BUY", "quantity": int(put_units),
                     "price": round(put_price, 2), "note": "tail hedge (risk_model.py), NSE settlement price"})
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def ensure_ledger(closes: pd.DataFrame, as_of: pd.Timestamp, fetch=None) -> Optional[pd.DataFrame]:
    """Load the ledger, creating it at the first session on/after EVALUATION_START_DATE if that has closed."""
    if TRADES_CSV.exists():
        return pd.read_csv(TRADES_CSV)
    sessions = closes.index[(closes.index >= pd.Timestamp(EVALUATION_START_DATE)) & (closes.index <= as_of)]
    if len(sessions) == 0:
        logger.info("Snapshot session %s not reached yet; no ledger.", EVALUATION_START_DATE)
        return None
    snapshot = sessions[0]
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV)
    plan_path = OUTPUT_DIR / "hedge_plan.csv"
    plan = pd.read_csv(plan_path).set_index("metric")["value"] if plan_path.exists() else pd.Series(dtype=object)
    contract = plan.get("Puts: contract")
    lot = int(float(plan.get("Nifty lot size", 0) or 0))
    units = int(float(plan.get("Puts: lots", 0) or 0)) * lot
    put_price = option_settlement(contract, snapshot.date(), fetch) if contract else None
    if contract and put_price is None:
        logger.warning("No settlement price for %s on %s yet; ledger not created (retried next refresh).",
                       contract, snapshot.date())
        return None
    ledger = initial_trades(snapshot, closes.loc[snapshot], risk.set_index("symbol")["weight_pct"], contract, units, put_price)
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(TRADES_CSV, index=False)
    logger.info("Created the trade ledger at the %s close: %d trades.", snapshot.date(), len(ledger))
    return ledger


def value_portfolio(ledger: pd.DataFrame, closes: pd.DataFrame, factors: pd.DataFrame, marks: Dict[tuple, float],
                    end: pd.Timestamp) -> pd.DataFrame:
    """
    Daily value from the first ledger date to `end`: positions from the ledger, stocks at the close,
    options at their settlement (the last known mark if a day is missing), cash accruing the 1D rate.
    The comparators put the same principal in the Nifty 500 TRI and the liquid fund at the first close.
    """
    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    start = ledger["date"].min()
    sessions = closes.index[(closes.index >= start) & (closes.index <= end)]
    rate = factors["rate_1d_index"].reindex(sessions).ffill()
    tri = factors["nifty500_tri"].reindex(sessions).ffill()
    closes = closes.reindex(sessions).ffill()
    positions: Dict[str, float] = {}
    last_mark: Dict[str, float] = {}
    cash = float(PRINCIPAL_INR)
    rows = []
    for i, day in enumerate(sessions):
        if i > 0 and pd.notna(rate.iloc[i]) and pd.notna(rate.iloc[i - 1]):
            cash *= rate.iloc[i] / rate.iloc[i - 1]
        for _, t in ledger[ledger["date"] == day].iterrows():
            sign = 1 if str(t["action"]).upper() == "BUY" else -1
            positions[t["instrument"]] = positions.get(t["instrument"], 0) + sign * float(t["quantity"])
            cash -= sign * float(t["quantity"]) * float(t["price"])
            if parse_option(t["instrument"]):
                last_mark.setdefault(t["instrument"], float(t["price"]))
        stocks = options = 0.0
        for inst, qty in positions.items():
            if not qty:
                continue
            if parse_option(inst):
                mark = marks.get((day.strftime("%Y-%m-%d"), inst))
                if mark is not None:
                    last_mark[inst] = mark
                options += qty * last_mark.get(inst, 0.0)
            else:
                stocks += qty * float(closes.loc[day, inst])
        total = stocks + options + cash
        rows.append({
            "date": day.strftime("%Y-%m-%d"), "stocks_inr": round(stocks, 2), "options_inr": round(options, 2),
            "cash_inr": round(cash, 2), "total_inr": round(total, 2), "pnl_inr": round(total - PRINCIPAL_INR, 2),
            "pnl_pct": round((total / PRINCIPAL_INR - 1) * 100, 3),
            "nifty500_inr": round(PRINCIPAL_INR * tri.loc[day] / tri.iloc[0], 2),
            "liquid_fund_inr": round(PRINCIPAL_INR * rate.loc[day] / rate.iloc[0], 2),
        })
    return pd.DataFrame(rows)


def current_positions(ledger: pd.DataFrame, closes: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Open stock positions on `day` with the cost, value, P&L and stop-loss distance in rupees."""
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV).set_index("symbol")
    ledger = ledger[pd.to_datetime(ledger["date"]) <= day]
    rows = []
    for inst, trades in ledger.groupby("instrument"):
        if parse_option(inst):
            continue
        signs = np.where(trades["action"].str.upper() == "BUY", 1, -1)
        qty = float((signs * trades["quantity"]).sum())
        if qty <= 0:
            continue
        buys = trades[trades["action"].str.upper() == "BUY"]
        avg_cost = float((buys["quantity"] * buys["price"]).sum() / buys["quantity"].sum())
        close = float(closes.loc[:day, inst].dropna().iloc[-1])
        stop = float(risk.loc[inst, "stop_loss_price"]) if inst in risk.index else np.nan
        rows.append({
            "symbol": inst, "shares": int(qty), "avg_cost": round(avg_cost, 2), "close": close,
            "value_inr": round(qty * close, 2), "pnl_inr": round(qty * (close - avg_cost), 2),
            "pnl_pct": round((close / avg_cost - 1) * 100, 2), "stop_loss_price": stop,
            "at_risk_to_stop_inr": round(max(close - stop, 0) * qty, 2) if pd.notna(stop) else np.nan,
            "stop_breached": bool(pd.notna(stop) and close <= stop),
        })
    return pd.DataFrame(rows)


def summarise(daily: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """The headline: value, profit, and the same money in the Nifty 500 and a liquid fund."""
    from performance import xirr

    last, first = daily.iloc[-1], daily.iloc[0]
    days = (pd.Timestamp(last["date"]) - pd.Timestamp(first["date"])).days
    port_xirr = (xirr([(pd.Timestamp(first["date"]).date(), -PRINCIPAL_INR),
                       (pd.Timestamp(last["date"]).date(), float(last["total_inr"]))]) * 100
                 if days >= MIN_DAYS_FOR_XIRR else np.nan)
    rows = [
        ("snapshot_date", first["date"]), ("as_of", last["date"]), ("calendar_days", days),
        ("principal_inr", PRINCIPAL_INR), ("value_inr", last["total_inr"]), ("pnl_inr", last["pnl_inr"]),
        ("pnl_pct", last["pnl_pct"]),
        ("nifty500_value_inr", last["nifty500_inr"]), ("liquid_fund_value_inr", last["liquid_fund_inr"]),
        ("vs_nifty500_inr", round(last["total_inr"] - last["nifty500_inr"], 2)),
        ("vs_liquid_fund_inr", round(last["total_inr"] - last["liquid_fund_inr"], 2)),
        ("xirr_pct", round(port_xirr, 2) if pd.notna(port_xirr) else np.nan),
        ("stocks_inr", last["stocks_inr"]), ("options_inr", last["options_inr"]), ("cash_inr", last["cash_inr"]),
        ("at_risk_to_stops_inr", round(positions["at_risk_to_stop_inr"].sum(), 2) if not positions.empty else np.nan),
        ("stops_breached", ", ".join(positions.loc[positions["stop_breached"], "symbol"]) if not positions.empty else ""),
        ("evaluation_end", EVALUATION_END_DATE),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def run(as_of: Optional[date] = None, fetch=None) -> Optional[Dict[str, pd.DataFrame]]:
    from risk_model import load_factor_series

    closes = load_closes()
    end = pd.Timestamp(as_of) if as_of else closes.index.max()
    end = closes.index[closes.index <= end].max()
    ledger = ensure_ledger(closes, end, fetch)
    if ledger is None:
        for path in (TRACKER_DAILY_CSV, TRACKER_SUMMARY_CSV, TRACKER_POSITIONS_CSV):
            path.unlink(missing_ok=True)
        return None
    factors = load_factor_series()
    sessions = closes.index[(closes.index >= pd.to_datetime(ledger["date"]).min()) & (closes.index <= end)]
    marks = {}
    for inst in {i for i in ledger["instrument"] if parse_option(i)}:
        for day in sessions:
            price = option_settlement(inst, day.date(), fetch)
            if price is not None:
                marks[(day.strftime("%Y-%m-%d"), inst)] = price
    daily = value_portfolio(ledger, closes, factors, marks, end)
    positions = current_positions(ledger, closes, end)
    summary = summarise(daily, positions)
    daily.to_csv(TRACKER_DAILY_CSV, index=False)
    positions.to_csv(TRACKER_POSITIONS_CSV, index=False)
    summary.to_csv(TRACKER_SUMMARY_CSV, index=False)
    logger.info("Portfolio value %s: Rs %,.0f (P&L Rs %,.0f).", end.date(), daily.iloc[-1]["total_inr"],
                daily.iloc[-1]["pnl_inr"])
    return {"daily": daily, "positions": positions, "summary": summary}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    as_of_arg = date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1]) if "--as-of" in sys.argv else None
    out = run(as_of_arg)
    if out is None:
        print(f"No snapshot yet: tracking starts at the {EVALUATION_START_DATE} close.")
    else:
        print(out["summary"].to_string(index=False))
        print(out["positions"].to_string(index=False))
