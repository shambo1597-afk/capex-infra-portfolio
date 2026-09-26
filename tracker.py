"""
Live P&L of the Rs 1 crore from the 28-Sep-2026 snapshot: did we make the money?

The trade ledger (config.TRADES_CSV: date, instrument, action, quantity, price, note) is the record of
what the portfolio holds. It is created once, at the close of the first session on or after
EVALUATION_START_DATE: the stocks at that close with the current weights of the equity sleeve, and
the tail-hedge puts of the hedge plan at that day's NSE settlement price. Everything later (real
purchase prices, stop-loss exits, redeployments, the profit-trigger put roll) is recorded by editing
the ledger; the tracker never trades on its own. The stocks held are the ledger's (the top
INVESTED_COUNT of the 15 at the snapshot); when one closes at or below its stop, plan_replacements
names the reserve stock its proceeds buy (output/replacement_plan.csv).

Each session the portfolio is valued as
    stocks (shares x NSE close) + puts (lots x lot size x NSE settlement price) + cash,
cash earning the overnight rate (Nifty 1D Rate index, a liquid ETF's return). The same Rs 1 crore is
also valued in the Nifty 500 TRI and in the liquid fund from the snapshot close, so the answer to
"did we make money, and more than the alternatives?" is one row of output/tracker_summary.csv.
"""

import logging
import re
import subprocess
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
    HEDGE_MAX_ROLLS,
    HEDGE_PROFIT_TRIGGER_PCT,
    INITIAL_HOLDINGS,
    LOCKED_PORTFOLIO_SYMBOLS,
    NEAR_STOP_PCT,
    SELECTION_CONVICTION_TIERS,
    TAIL_HEDGE_OTM_PCT,
    TRADES_CSV,
)

logger = logging.getLogger("tracker")

TRACKER_DAILY_CSV = OUTPUT_DIR / "tracker_daily.csv"
TRACKER_SUMMARY_CSV = OUTPUT_DIR / "tracker_summary.csv"
TRACKER_POSITIONS_CSV = OUTPUT_DIR / "tracker_positions.csv"
PUT_MARKS_CSV = DERIVATIVES_DIR / "option_marks.csv"
HEDGE_ROLL_CSV = OUTPUT_DIR / "hedge_roll.csv"
REPLACEMENT_PLAN_CSV = OUTPUT_DIR / "replacement_plan.csv"
SELECTION_RANKING_CSV = OUTPUT_DIR / "selection_ranking.csv"
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


def ledger_positions(ledger: pd.DataFrame) -> Dict[str, float]:
    """Net quantity held per instrument (buys minus sells) across the whole ledger."""
    signs = np.where(ledger["action"].str.upper() == "BUY", 1, -1)
    net = pd.Series(signs * ledger["quantity"].astype(float)).groupby(ledger["instrument"].values).sum()
    return {k: v for k, v in net.items() if v}


def held_stocks(ledger: Optional[pd.DataFrame]) -> list:
    """Stocks with shares held (net of sales), in LOCKED_PORTFOLIO order; before the snapshot, INITIAL_HOLDINGS."""
    if ledger is None:
        return list(INITIAL_HOLDINGS)
    held = [inst for inst, qty in ledger_positions(ledger).items() if qty > 0 and not parse_option(inst)]
    order = {s: i for i, s in enumerate(LOCKED_PORTFOLIO_SYMBOLS)}
    return sorted(held, key=lambda s: (order.get(s, len(order)), s))


def current_holdings() -> list:
    """The stocks the money is in now: from the trade ledger once it exists, else INITIAL_HOLDINGS."""
    return held_stocks(pd.read_csv(TRADES_CSV) if TRADES_CSV.exists() else None)


def sold_stocks(ledger: Optional[pd.DataFrame]) -> set:
    """Stocks the ledger shows a sale of: a stopped-out stock is never bought back."""
    if ledger is None or ledger.empty:
        return set()
    sells = ledger[ledger["action"].str.upper() == "SELL"]["instrument"]
    return {inst for inst in sells if not parse_option(inst)}


def plan_replacements(positions: pd.DataFrame, ledger: pd.DataFrame, ranking: pd.DataFrame,
                      closes: pd.Series) -> pd.DataFrame:
    """
    Stop-loss replacement rule. Each holding that closed at or below its stop is sold; the sale
    proceeds (shares x today's close, an estimate of tomorrow's fill) buy whole shares of the first
    stock in the reserve queue (LOCKED_PORTFOLIO order after the holdings) that is not held, has
    never been sold, and still passes the selection rule today: listed in `ranking` (the eligible
    stocks of output/selection_ranking.csv) with conviction in SELECTION_CONVICTION_TIERS. With no
    such stock the proceeds wait in the liquid fund. Several stops on one day take the queue in turn.
    """
    columns = ["sell", "sell_shares", "sell_close", "stop_loss_price", "proceeds_inr", "buy", "buy_rank",
               "buy_close", "buy_shares", "buy_inr", "cash_left_inr", "note"]
    sold = sold_stocks(ledger)  # the whole ledger: a sale recorded after the last close is already done
    if positions.empty or not (positions["stop_breached"] & ~positions["symbol"].isin(sold)).any():
        return pd.DataFrame(columns=columns)
    held = set(positions["symbol"]) | set(held_stocks(ledger) if not ledger.empty else [])
    ok = ranking[ranking["conviction"].isin(SELECTION_CONVICTION_TIERS)] if "conviction" in ranking else ranking
    passing = set(ok["symbol"])
    queue = [s for s in LOCKED_PORTFOLIO_SYMBOLS if s not in held and s not in sold]
    skipped = [s for s in queue if s not in passing]
    queue = [s for s in queue if s in passing]
    rows = []
    for _, p in positions[positions["stop_breached"] & ~positions["symbol"].isin(sold)].iterrows():
        proceeds = float(p["shares"]) * float(p["close"])
        row = {"sell": p["symbol"], "sell_shares": int(p["shares"]), "sell_close": float(p["close"]),
               "stop_loss_price": float(p["stop_loss_price"]), "proceeds_inr": round(proceeds, 2)}
        if queue:
            buy = queue.pop(0)
            price = float(closes[buy])
            n = int(proceeds // price)
            row.update({"buy": buy, "buy_rank": LOCKED_PORTFOLIO_SYMBOLS.index(buy) + 1, "buy_close": price,
                        "buy_shares": n, "buy_inr": round(n * price, 2),
                        "cash_left_inr": round(proceeds - n * price, 2),
                        "note": ("skipped (fails the selection rule today): " + ", ".join(skipped)) if skipped else ""})
        else:
            row.update({"buy": None, "buy_rank": None, "buy_close": None, "buy_shares": 0, "buy_inr": 0.0,
                        "cash_left_inr": round(proceeds, 2),
                        "note": "no reserve stock passes the selection rule today: the proceeds wait in the liquid fund"})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


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
            "near_stop": bool(pd.notna(stop) and stop < close <= stop * (1 + NEAR_STOP_PCT / 100)),
            "pct_above_stop": round((close / stop - 1) * 100, 2) if pd.notna(stop) and stop > 0 else np.nan,
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
        ("near_stop", ", ".join(positions.loc[positions["near_stop"], "symbol"])
         if not positions.empty and "near_stop" in positions else ""),
        ("evaluation_end", EVALUATION_END_DATE),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])

def rolls_done(ledger: pd.DataFrame) -> int:
    """Roll-ups recorded: days on which puts were bought at a higher strike than any bought before."""
    puts = ledger[(ledger["action"].str.upper() == "BUY") & ledger["instrument"].map(lambda i: parse_option(i) is not None)]
    best, rolls = None, 0
    for day, grp in puts.sort_values("date").groupby("date", sort=True):
        top = max(parse_option(i)["strike"] for i in grp["instrument"])
        if best is not None and top > best:
            rolls += 1
        best = top if best is None else max(best, top)
    return rolls


def plan_put_roll(summary: Dict[str, object], ledger: pd.DataFrame, as_of: pd.Timestamp, fo: Optional[pd.DataFrame],
                  hedge_ratio: float, trigger_pct: float = HEDGE_PROFIT_TRIGGER_PCT,
                  max_rolls: int = HEDGE_MAX_ROLLS, otm_pct: float = TAIL_HEDGE_OTM_PCT) -> pd.DataFrame:
    """
    The profit-lock rule: once the portfolio is up `trigger_pct`, sell the puts held and buy puts of the same
    expiry about `otm_pct` below the current Nifty, sized to the (larger) stock value x hedge_ratio.
    Returns rows (field, value); status is 'waiting', 'TRIGGERED' (with the trades), 'done' or 'no prices'.
    """
    pnl_pct = float(summary["pnl_pct"])
    done = rolls_done(ledger)
    rows = [("as_of", as_of.strftime("%Y-%m-%d")), ("pnl_pct", round(pnl_pct, 2)), ("trigger_pct", trigger_pct),
            ("rolls_done", done), ("max_rolls", max_rolls)]
    if done >= max_rolls:
        return pd.DataFrame(rows + [("status", "done")], columns=["field", "value"])
    if pnl_pct < trigger_pct:
        return pd.DataFrame(rows + [("status", "waiting"), ("gap_to_trigger_pp", round(trigger_pct - pnl_pct, 2))],
                            columns=["field", "value"])
    if fo is None or fo.empty:
        return pd.DataFrame(rows + [("status", "no prices")], columns=["field", "value"])
    held = {i: q for i, q in ledger_positions(ledger).items() if parse_option(i) and q > 0}
    opts = fo[(fo["FinInstrmTp"] == "IDO") & (fo["OptnTp"] == "PE")]
    sell_rows, expiry = [], None
    for inst, qty in held.items():
        o = parse_option(inst)
        expiry = o["expiry"]
        r = opts[(opts["XpryDt"] == o["expiry"]) & np.isclose(opts["StrkPric"], o["strike"])]
        price = float(r["SttlmPric"].iloc[0]) if not r.empty else 0.0
        sell_rows.append((inst, qty, price))
    fut = fo[fo["FinInstrmTp"] == "IDF"]
    spot = float(fut["UndrlygPric"].iloc[0]) if not fut.empty else float(opts["UndrlygPric"].iloc[0])
    lot = int(fo["NewBrdLotQty"].iloc[0])
    chain = opts[(opts["XpryDt"] == expiry) & (opts["OpnIntrst"] > 0)] if expiry else opts
    if chain.empty:
        return pd.DataFrame(rows + [("status", "no prices")], columns=["field", "value"])
    liquid = chain[chain["OpnIntrst"] >= chain["OpnIntrst"].median()]
    pool = liquid if not liquid.empty else chain
    target = spot * (1 - otm_pct / 100)
    new = pool.iloc[(pool["StrkPric"] - target).abs().argsort().iloc[0]]
    lots = max(1, round(hedge_ratio * float(summary["stocks_inr"]) / (spot * lot)))
    held_strike = max((parse_option(i)["strike"] for i in held), default=0.0)
    if float(new["StrkPric"]) <= held_strike:
        # The Nifty has not risen enough to move the strike up: the gain is stock-specific and the trailing
        # stops protect it. Never sell and rebuy the same contract; only add lots if the larger portfolio needs them.
        inst = max(held, key=lambda i: parse_option(i)["strike"])
        extra = lots * lot - int(held[inst])
        extra = lots * lot - int(held[inst])
        price = float(new["SttlmPric"]) if np.isclose(new["StrkPric"], held_strike) else \
            float(opts[(opts["XpryDt"] == expiry) & np.isclose(opts["StrkPric"], held_strike)]["SttlmPric"].iloc[0])
        rows += [("status", "TRIGGERED" if extra > 0 else "covered"), ("nifty_spot", round(spot, 2)),
                 ("note", "Nifty has not risen enough to raise the strike: the gain is stock-specific and the "
                          "trailing stops protect it; only the lot count is topped up")]
        if extra > 0:
            rows.append(("trade", f"BUY {extra} {inst} @ {price:.2f} ({extra // lot} lots, top-up)"))
        cost = max(extra, 0) * price
        rows += [("sell_value_inr", 0), ("buy_cost_inr", round(cost)), ("net_cost_inr", round(cost)),
                 ("cash_inr", round(float(summary["cash_inr"]))),
                 ("cash_sufficient", bool(float(summary["cash_inr"]) >= cost))]
        return pd.DataFrame(rows, columns=["field", "value"])
    new_inst = f"NIFTY {new['XpryDt']} {new['StrkPric']:.0f} PE"
    buy_cost = lots * lot * float(new["SttlmPric"])
    sell_value = sum(q * p for _, q, p in sell_rows)
    rows += [("status", "TRIGGERED"), ("nifty_spot", round(spot, 2))]
    for inst, qty, price in sell_rows:
        rows.append(("trade", f"SELL {int(qty)} {inst} @ {price:.2f}"))
    rows.append(("trade", f"BUY {lots * lot} {new_inst} @ {float(new['SttlmPric']):.2f} ({lots} lots)"))
    rows += [("new_strike_below_spot_pct", round((1 - new["StrkPric"] / spot) * 100, 2)),
             ("sell_value_inr", round(sell_value)), ("buy_cost_inr", round(buy_cost)),
             ("net_cost_inr", round(buy_cost - sell_value)), ("cash_inr", round(float(summary["cash_inr"]))),
             ("cash_sufficient", bool(float(summary["cash_inr"]) >= buy_cost - sell_value))]
    return pd.DataFrame(rows, columns=["field", "value"])


# ---------------------------------------------------------------------------
# Recording trades from the dashboard (instead of hand-editing data/trades.csv)
# ---------------------------------------------------------------------------

_TRADE_TEXT = re.compile(r"^(BUY|SELL) (\d+) (.+?) @ ([\d.]+)")


def validate_ledger(ledger: pd.DataFrame) -> list:
    """Problems that would make the ledger unusable; empty when it is fine."""
    problems = []
    missing = [c for c in LEDGER_COLUMNS if c not in ledger.columns]
    if missing:
        return [f"missing column(s): {', '.join(missing)}"]
    for i, r in ledger.reset_index(drop=True).iterrows():
        row = f"row {i + 1} ({r['instrument']})"
        if pd.isna(pd.to_datetime(r["date"], errors="coerce")):
            problems.append(f"{row}: date must be YYYY-MM-DD")
        if str(r["action"]).upper() not in ("BUY", "SELL"):
            problems.append(f"{row}: action must be BUY or SELL")
        if not (pd.notna(r["quantity"]) and float(r["quantity"]) > 0 and float(r["quantity"]) == int(float(r["quantity"]))):
            problems.append(f"{row}: quantity must be a whole number above 0")
        if not (pd.notna(r["price"]) and float(r["price"]) > 0):
            problems.append(f"{row}: price must be above 0")
        if not isinstance(r["instrument"], str) or not r["instrument"].strip():
            problems.append(f"row {i + 1}: instrument is empty")
    if not problems:
        held = ledger_positions(ledger.assign(action=ledger["action"].str.upper()))
        short = [k for k, v in held.items() if v < 0]
        if short:
            problems.append(f"sells more than was bought: {', '.join(short)}")
    return problems


def save_ledger(ledger: pd.DataFrame) -> list:
    """Validate, sort by date (keeping the order within a day) and write the ledger; returns the problems."""
    ledger = ledger[LEDGER_COLUMNS].copy() if set(LEDGER_COLUMNS) <= set(ledger.columns) else ledger
    ledger = ledger.dropna(how="all").copy()
    if set(LEDGER_COLUMNS) <= set(ledger.columns):
        ledger["instrument"] = ledger["instrument"].astype("string").str.strip()
        ledger["action"] = ledger["action"].astype("string").str.strip().str.upper()
    problems = validate_ledger(ledger)
    if problems:
        return problems
    ledger["quantity"] = ledger["quantity"].astype(float).astype(int)
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.strftime("%Y-%m-%d")
    ledger["note"] = ledger["note"].fillna("")
    ledger = ledger.sort_values("date", kind="stable")
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(TRADES_CSV, index=False)
    return []


def replacement_trades(plan: pd.DataFrame, day: str, fills: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Ledger rows for a replacement plan: SELL the stopped stock, BUY the reserve stock (fills override closes).
    The buy quantity is recomputed from the actual sale proceeds when fill prices are given."""
    fills = fills or {}
    rows = []
    for _, r in plan.iterrows():
        sell_px = float(fills.get(r["sell"], r["sell_close"]))
        rows.append({"date": day, "instrument": r["sell"], "action": "SELL", "quantity": int(r["sell_shares"]),
                     "price": round(sell_px, 2), "note": f"stop-loss exit (stop {float(r['stop_loss_price']):.2f})"})
        if isinstance(r.get("buy"), str) and r["buy"]:
            buy_px = float(fills.get(r["buy"], r["buy_close"]))
            qty = int(int(r["sell_shares"]) * sell_px // buy_px)
            if qty > 0:
                rows.append({"date": day, "instrument": r["buy"], "action": "BUY", "quantity": qty,
                             "price": round(buy_px, 2), "note": f"replaces {r['sell']} (reserve rank #{int(r['buy_rank'])})"})
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def roll_trades(roll: pd.DataFrame, day: str) -> pd.DataFrame:
    """Ledger rows from the profit-lock plan's trade lines ("BUY 130 NIFTY 2026-12-29 24000 PE @ 45.10 (2 lots)")."""
    rows = []
    for text in roll.loc[roll["field"] == "trade", "value"]:
        m = _TRADE_TEXT.match(str(text))
        if m:
            rows.append({"date": day, "instrument": m.group(3), "action": m.group(1), "quantity": int(m.group(2)),
                         "price": float(m.group(4)), "note": "profit-lock put roll"})
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def publish_ledger(message: str, branch: str = "main") -> str:
    """
    Commit data/trades.csv onto the remote branch without touching the working tree or the local branch
    (a temporary index on top of origin/<branch>), so the evening alert run sees the trades. Returns a
    one-line result; a failure (no git, no network, no permission) is reported, never raised.
    """
    import os
    import tempfile

    repo = TRADES_CSV.resolve().parents[1]
    rel = TRADES_CSV.resolve().relative_to(repo).as_posix()

    def git(*args, env=None):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True,
                              env=env, timeout=60).stdout.strip()

    try:
        git("fetch", "-q", "origin", branch)
        base = git("rev-parse", "FETCH_HEAD")
        blob = git("hash-object", "-w", rel)
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "index")}
            git("read-tree", base, env=env)
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", env=env)
            tree = git("write-tree", env=env)
        if tree == git("rev-parse", f"{base}^{{tree}}"):
            return f"GitHub {branch} already has these trades."
        commit = git("commit-tree", tree, "-p", base, "-m", message)
        git("push", "-q", "origin", f"{commit}:refs/heads/{branch}")
        return f"Saved to GitHub ({branch}): the evening alerts will use these trades."
    except Exception as exc:  # noqa: BLE001 - shown to the user, never fatal
        detail = getattr(exc, "stderr", "") or str(exc)
        return f"Saved on this computer only; pushing to GitHub failed ({str(detail).strip()[:160]})."


def _inr(x: float) -> str:
    neg, x = x < 0, abs(round(x))
    s = str(int(x))
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail = head[-2:] + "," + tail
        head = head[:-2]
    return ("-Rs " if neg else "Rs ") + ((head + "," + tail) if head else tail)


def whatsapp_update(summary: Optional[Dict[str, object]], positions: pd.DataFrame, plan: pd.DataFrame,
                    roll: Optional[Dict[str, object]] = None, risk: Optional[pd.DataFrame] = None) -> str:
    """A short plain-text update for the group chat (WhatsApp *bold* markup)."""
    if not summary:
        lines = ["*Capex portfolio: not invested yet*",
                 f"Snapshot at the {pd.Timestamp(EVALUATION_START_DATE):%d-%b} close; money goes into the top 8 of 15."]
        if risk is not None and not risk.empty:
            near = risk[risk["stop_loss_pct_below_current"] <= NEAR_STOP_PCT]["symbol"].tolist()
            lines.append("Planned: " + ", ".join(f"{r.symbol} {r.weight_pct:.1f}%" for r in risk.itertuples()))
            if near:
                lines.append(f"Stops within {NEAR_STOP_PCT:g}%: " + ", ".join(near))
        return "\n".join(lines)
    pnl = float(summary["pnl_inr"])
    lines = [f"*Capex portfolio, {pd.Timestamp(summary['as_of']):%d-%b-%Y}*",
             f"Rs 1 cr is now *{_inr(float(summary['value_inr']))}* ({'+' if pnl >= 0 else '-'}{_inr(abs(pnl))}, "
             f"{float(summary['pnl_pct']) + 0.0:+.2f}%)".replace("-0.00", "+0.00"),
             f"vs Nifty 500: {'ahead' if float(summary['vs_nifty500_inr']) >= 0 else 'behind'} by "
             f"{_inr(abs(float(summary['vs_nifty500_inr'])))}; vs liquid fund: "
             f"{'ahead' if float(summary['vs_liquid_fund_inr']) >= 0 else 'behind'} by "
             f"{_inr(abs(float(summary['vs_liquid_fund_inr'])))}"]
    if not positions.empty:
        best = positions.loc[positions["pnl_pct"].idxmax()]
        worst = positions.loc[positions["pnl_pct"].idxmin()]
        lines.append(f"Best: {best['symbol']} {best['pnl_pct']:+.1f}% | Worst: {worst['symbol']} {worst['pnl_pct']:+.1f}%")
    alerts = []
    for _, r in plan.iterrows():
        alerts.append(f"STOP HIT {r['sell']}: sell, buy {r['buy']}" if isinstance(r.get("buy"), str) and r["buy"]
                      else f"STOP HIT {r['sell']}: sell, cash waits (no reserve stock qualifies)")
    if not positions.empty and "near_stop" in positions:
        for _, r in positions[positions["near_stop"]].iterrows():
            alerts.append(f"{r['symbol']} is {r['pct_above_stop']:.1f}% above its stop")
    if roll and roll.get("status") == "TRIGGERED":
        alerts.append("Profit lock triggered: roll the Nifty puts up")
    lines.append(("*Action:* " + "; ".join(alerts)) if alerts else "No action needed.")
    return "\n".join(lines)


def run(as_of: Optional[date] = None, fetch=None) -> Optional[Dict[str, pd.DataFrame]]:
    from risk_model import load_factor_series

    closes = load_closes()
    end = pd.Timestamp(as_of) if as_of else closes.index.max()
    end = closes.index[closes.index <= end].max()
    ledger = ensure_ledger(closes, end, fetch)
    if ledger is None:
        for path in (TRACKER_DAILY_CSV, TRACKER_SUMMARY_CSV, TRACKER_POSITIONS_CSV, HEDGE_ROLL_CSV, REPLACEMENT_PLAN_CSV):
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
    logger.info(f"Portfolio value {end.date()}: Rs {daily.iloc[-1]['total_inr']:,.0f} "
                f"(P&L Rs {daily.iloc[-1]['pnl_inr']:,.0f}).")

    # Profit-lock rule: the option chain is only fetched once the trigger is reached
    s = dict(zip(summary["metric"], summary["value"]))
    plan_path = OUTPUT_DIR / "hedge_plan.csv"
    plan = pd.read_csv(plan_path).set_index("metric")["value"] if plan_path.exists() else pd.Series(dtype=object)
    ratio = float(plan.get("Tail hedge ratio", 1.0))
    fo = None
    if float(s["pnl_pct"]) >= HEDGE_PROFIT_TRIGGER_PCT and rolls_done(ledger) < HEDGE_MAX_ROLLS:
        from risk_model import fetch_nifty_derivatives
        fo = (fetch or (lambda d: fetch_nifty_derivatives(d, use_cache=True, save=False)))(end.date())
    roll = plan_put_roll(s, ledger, end, fo, ratio)
    roll.to_csv(HEDGE_ROLL_CSV, index=False)

    # Stop-loss replacement: sell the stopped stock, buy the next reserve stock that still qualifies
    ranking = pd.read_csv(SELECTION_RANKING_CSV) if SELECTION_RANKING_CSV.exists() else pd.DataFrame(columns=["symbol"])
    replacements = plan_replacements(positions, ledger, ranking, closes.loc[end])
    replacements.to_csv(REPLACEMENT_PLAN_CSV, index=False)
    return {"daily": daily, "positions": positions, "summary": summary, "roll": roll, "replacements": replacements}


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
        print(out["roll"].to_string(index=False))
        if not out["replacements"].empty:
            print(out["replacements"].to_string(index=False))
