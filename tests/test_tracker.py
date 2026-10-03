"""Live P&L of the Rs 1 crore (tracker.py): ledger, valuation and comparators."""

import pandas as pd
import pytest

import tracker
from config import PRINCIPAL_INR


def _setup(dates, closes, rates, tri):
    idx = pd.to_datetime(dates)
    closes_df = pd.DataFrame(closes, index=idx)
    factors = pd.DataFrame({"rate_1d_index": rates, "nifty500_tri": tri}, index=idx)
    return closes_df, factors


def test_a_bridged_benchmark_day_is_valued_and_flagged():
    """Snapshot day without the Nifty 500 TRI (niftyindices.com down): the bridged level still gives a number,
    and the day is flagged provisional."""
    dates = ["2026-09-28", "2026-09-29"]
    closes, factors = _setup(dates, {"AAA": [100.0, 110.0]}, [1000.0, 1000.2], [500.0, 505.0])
    factors["nifty500_tri_provisional"] = [True, False]
    ledger = pd.DataFrame([{"date": "2026-09-28", "instrument": "AAA", "action": "BUY", "quantity": 1000,
                            "price": 100.0, "note": ""}])
    daily = tracker.value_portfolio(ledger, closes, factors, {}, pd.Timestamp("2026-09-29")).set_index("date")
    assert daily["benchmark_provisional"].tolist() == [True, False]
    assert daily.loc["2026-09-29", "nifty500_inr"] == pytest.approx(PRINCIPAL_INR * 1.01)


def test_parse_option():
    assert tracker.parse_option("NIFTY 2026-12-29 22000 PE") == {"expiry": "2026-12-29", "strike": 22000.0, "type": "PE"}
    assert tracker.parse_option("WELCORP") is None


def test_value_starts_at_the_principal_and_tracks_trades():
    dates = ["2026-09-28", "2026-09-29", "2026-09-30"]
    closes, factors = _setup(dates, {"AAA": [100.0, 110.0, 120.0]}, [1000.0, 1000.2, 1000.4], [500.0, 505.0, 495.0])
    put = "NIFTY 2026-12-29 22000 PE"
    ledger = pd.DataFrame([
        {"date": "2026-09-28", "instrument": "AAA", "action": "BUY", "quantity": 90000, "price": 100.0, "note": ""},
        {"date": "2026-09-28", "instrument": put, "action": "BUY", "quantity": 455, "price": 150.0, "note": ""},
        {"date": "2026-09-30", "instrument": "AAA", "action": "SELL", "quantity": 10000, "price": 120.0, "note": "stop"},
    ])
    marks = {("2026-09-28", put): 150.0, ("2026-09-29", put): 120.0}  # 30-Sep missing: the last mark is kept
    daily = tracker.value_portfolio(ledger, closes, factors, marks, pd.Timestamp("2026-09-30")).set_index("date")
    assert daily.loc["2026-09-28", "total_inr"] == pytest.approx(PRINCIPAL_INR)
    cash0 = PRINCIPAL_INR - 90000 * 100.0 - 455 * 150.0
    cash1 = cash0 * 1000.2 / 1000.0
    assert daily.loc["2026-09-29", "cash_inr"] == pytest.approx(cash1, abs=0.01)
    assert daily.loc["2026-09-29", "total_inr"] == pytest.approx(90000 * 110 + 455 * 120 + cash1, abs=0.01)
    cash2 = cash1 * 1000.4 / 1000.2 + 10000 * 120.0
    assert daily.loc["2026-09-30", "stocks_inr"] == pytest.approx(80000 * 120.0)
    assert daily.loc["2026-09-30", "options_inr"] == pytest.approx(455 * 120.0)
    assert daily.loc["2026-09-30", "cash_inr"] == pytest.approx(cash2, abs=0.01)
    assert daily.loc["2026-09-30", "nifty500_inr"] == pytest.approx(PRINCIPAL_INR * 495 / 500)
    assert daily.loc["2026-09-30", "liquid_fund_inr"] == pytest.approx(PRINCIPAL_INR * 1000.4 / 1000)


def test_initial_trades_buy_whole_shares_of_the_equity_sleeve():
    from config import EQUITY_ALLOCATION_PCT
    closes = pd.Series({"AAA": 333.0, "BBB": 1000.0})
    t = tracker.initial_trades(pd.Timestamp("2026-09-28"), closes, pd.Series({"AAA": 60.0, "BBB": 40.0}),
                               "NIFTY 2026-12-29 22000 PE", 455, 150.0).set_index("instrument")
    equity = PRINCIPAL_INR * EQUITY_ALLOCATION_PCT / 100
    assert t.loc["AAA", "quantity"] == int(equity * 0.6 // 333.0)
    assert t.loc["BBB", "quantity"] == int(equity * 0.4 // 1000.0)
    assert t.loc["NIFTY 2026-12-29 22000 PE", "quantity"] == 455


def test_no_ledger_before_the_snapshot_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRADES_CSV", tmp_path / "trades.csv")
    monkeypatch.setattr(tracker, "EVALUATION_START_DATE", "2026-09-28")
    closes = pd.DataFrame({"AAA": [100.0]}, index=pd.to_datetime(["2026-09-25"]))
    assert tracker.ensure_ledger(closes, pd.Timestamp("2026-09-25")) is None
    assert not (tmp_path / "trades.csv").exists()


def test_summary_waits_30_days_for_xirr():
    daily = pd.DataFrame({"date": ["2026-09-28", "2026-10-05"], "total_inr": [1e7, 1.01e7], "pnl_inr": [0, 1e5],
                          "pnl_pct": [0, 1.0], "nifty500_inr": [1e7, 1e7], "liquid_fund_inr": [1e7, 1.001e7],
                          "stocks_inr": [9.7e6, 9.8e6], "options_inr": [6e4, 5e4], "cash_inr": [2.4e5, 2.5e5]})
    s = dict(tracker.summarise(daily, pd.DataFrame()).values)
    assert pd.isna(s["xirr_pct"]) and s["vs_liquid_fund_inr"] == pytest.approx(9e4)


def test_ledger_positions_net_buys_and_sells():
    ledger = pd.DataFrame({"instrument": ["AAA", "AAA", "BBB", "BBB"], "action": ["BUY", "SELL", "BUY", "SELL"],
                           "quantity": [100, 40, 10, 10]})
    assert tracker.ledger_positions(ledger) == {"AAA": 60.0}  # BBB fully sold: not held


def _fo(spot=25500.0):
    rows = [{"FinInstrmTp": "IDF", "XpryDt": "2026-11-23", "StrkPric": 0, "OptnTp": None, "SttlmPric": spot + 50,
             "UndrlygPric": spot, "OpnIntrst": 1e6, "NewBrdLotQty": 65}]
    for k, (price, oi) in {22000: (20.0, 3e6), 23000: (60.0, 2e6), 24000: (140.0, 3e6), 24500: (200.0, 1e6)}.items():
        rows.append({"FinInstrmTp": "IDO", "XpryDt": "2026-12-29", "StrkPric": float(k), "OptnTp": "PE", "SttlmPric": price,
                     "UndrlygPric": spot, "OpnIntrst": oi, "NewBrdLotQty": 65})
    return pd.DataFrame(rows)


def _roll_ledger(extra=()):
    rows = [{"date": "2026-09-28", "instrument": "AAA", "action": "BUY", "quantity": 1000, "price": 100.0, "note": ""},
            {"date": "2026-09-28", "instrument": "NIFTY 2026-12-29 22000 PE", "action": "BUY", "quantity": 455,
             "price": 146.5, "note": ""}] + list(extra)
    return pd.DataFrame(rows)


def test_profit_lock_waits_below_the_trigger():
    s = {"pnl_pct": 6.0, "stocks_inr": 1.03e7, "cash_inr": 2.6e5}
    out = dict(tracker.plan_put_roll(s, _roll_ledger(), pd.Timestamp("2026-11-10"), None, 1.12).values)
    assert out["status"] == "waiting" and out["gap_to_trigger_pp"] == pytest.approx(4.0)


def test_profit_lock_rolls_the_puts_up_when_triggered():
    s = {"pnl_pct": 10.5, "stocks_inr": 1.07e7, "cash_inr": 2.6e5}
    roll = tracker.plan_put_roll(s, _roll_ledger(), pd.Timestamp("2026-11-10"), _fo(), 1.12)
    out = dict(roll[roll["field"] != "trade"].values)
    trades = roll.loc[roll["field"] == "trade", "value"].tolist()
    assert out["status"] == "TRIGGERED"
    # 5% below 25,500 = 24,225: the nearest liquid strike (OI >= median) is 24,000
    lots = round(1.12 * 1.07e7 / (25500 * 65))
    assert trades == ["SELL 455 NIFTY 2026-12-29 22000 PE @ 20.00",
                      f"BUY {lots * 65} NIFTY 2026-12-29 24000 PE @ 140.00 ({lots} lots)"]
    assert out["net_cost_inr"] == round(lots * 65 * 140.0 - 455 * 20.0) and out["cash_sufficient"]


def test_profit_lock_is_done_after_the_roll_is_recorded():
    rolled = _roll_ledger([{"date": "2026-11-11", "instrument": "NIFTY 2026-12-29 22000 PE", "action": "SELL",
                            "quantity": 455, "price": 20.0, "note": ""},
                           {"date": "2026-11-11", "instrument": "NIFTY 2026-12-29 24000 PE", "action": "BUY",
                            "quantity": 520, "price": 140.0, "note": ""}])
    s = {"pnl_pct": 12.0, "stocks_inr": 1.1e7, "cash_inr": 2e5}
    assert dict(tracker.plan_put_roll(s, rolled, pd.Timestamp("2026-11-20"), _fo(), 1.12).values)["status"] == "done"


def test_profit_lock_never_churns_the_same_strike():
    s = {"pnl_pct": 11.0, "stocks_inr": 1.2e7, "cash_inr": 2.6e5}
    roll = tracker.plan_put_roll(s, _roll_ledger(), pd.Timestamp("2026-11-10"), _fo(spot=23100.0), 1.12)
    trades = roll.loc[roll["field"] == "trade", "value"].tolist()
    lots = round(1.12 * 1.2e7 / (23100 * 65))
    assert trades == [f"BUY {lots * 65 - 455} NIFTY 2026-12-29 22000 PE @ 20.00 ({(lots * 65 - 455) // 65} lots, top-up)"]
    assert "stock-specific" in dict(roll[roll["field"] != "trade"].values)["note"]


def test_a_same_strike_top_up_is_not_the_roll():
    topped = _roll_ledger([{"date": "2026-11-05", "instrument": "NIFTY 2026-12-29 22000 PE", "action": "BUY",
                            "quantity": 65, "price": 30.0, "note": "top-up"}])
    assert tracker.rolls_done(topped) == 0
    s = {"pnl_pct": 10.5, "stocks_inr": 1.07e7, "cash_inr": 2.6e5}
    out = dict(tracker.plan_put_roll(s, topped, pd.Timestamp("2026-11-10"), _fo(), 1.12)[lambda d: d["field"] != "trade"].values)
    assert out["status"] == "TRIGGERED"  # the Nifty has since risen: the real roll-up is still available


def _drift(stocks, spot=22000.0, roll_status="waiting", price=200.0, ledger=None):
    s = {"stocks_inr": stocks, "cash_inr": 2e5}
    out = tracker.plan_hedge_drift(s, _roll_ledger() if ledger is None else ledger, pd.Timestamp("2026-10-20"), 1.25,
                                   spot, 65, price, roll_status)
    return dict(out.values)


def test_hedge_drift_stays_put_inside_a_full_lot():
    # 455 units = 7 lots held; the stocks need 7.7 lots: inside the band, no trade
    out = _drift(stocks=7.7 * 22000 * 65 / 1.25)
    assert out["status"] == "in band" and out["lots_held"] == 7 and out["lots_needed"] == pytest.approx(7.7)


def test_hedge_drift_buys_back_to_the_rounded_lots():
    out = _drift(stocks=8.2 * 22000 * 65 / 1.25)
    assert out["status"] == "REBALANCE" and out["trade"] == "BUY 65 NIFTY 2026-12-29 22000 PE @ 200.00 (1 lots, hedge ratio)"
    assert out["trade_value_inr"] == 13000 and out["cash_sufficient"]


def test_hedge_drift_sells_when_the_stock_value_falls():
    out = _drift(stocks=5.6 * 22000 * 65 / 1.25)
    assert out["trade"].startswith("SELL 65 NIFTY 2026-12-29 22000 PE")


def test_hedge_drift_leaves_resizing_to_a_triggered_roll_and_needs_puts():
    assert _drift(stocks=9e6, roll_status="TRIGGERED")["status"] == "see profit lock"
    assert _drift(stocks=9e6, ledger=_roll_ledger()[:1])["status"] == "no puts"
    assert _drift(stocks=9.5 * 22000 * 65 / 1.25, price=None)["status"] == "no prices"


def test_hedge_drift_trade_is_recorded_with_its_own_note():
    out = tracker.plan_hedge_drift({"stocks_inr": 8.2 * 22000 * 65 / 1.25, "cash_inr": 2e5}, _roll_ledger(),
                                   pd.Timestamp("2026-10-20"), 1.25, 22000.0, 65, 200.0)
    t = tracker.roll_trades(out, "2026-10-21", note="hedge-ratio rebalance")
    assert t[["instrument", "action", "quantity", "price", "note"]].values.tolist() == [
        ["NIFTY 2026-12-29 22000 PE", "BUY", 65, 200.0, "hedge-ratio rebalance"]]


# ---------------------------------------------------------------------------
# 15 tracked, 8 invested: holdings and the stop-loss replacement rule
# ---------------------------------------------------------------------------

def _positions(rows):
    return pd.DataFrame([{"symbol": s, "shares": n, "close": c, "stop_loss_price": stop, "stop_breached": c <= stop}
                         for s, n, c, stop in rows])


def _ranking(symbols, conviction="High"):
    return pd.DataFrame({"symbol": symbols, "conviction": conviction})


def test_holdings_before_and_after_the_snapshot():
    from config import INITIAL_HOLDINGS, LOCKED_PORTFOLIO_SYMBOLS
    assert tracker.held_stocks(None) == INITIAL_HOLDINGS
    a, b, c = LOCKED_PORTFOLIO_SYMBOLS[0], LOCKED_PORTFOLIO_SYMBOLS[1], LOCKED_PORTFOLIO_SYMBOLS[9]
    ledger = pd.DataFrame([
        {"date": "2026-09-28", "instrument": b, "action": "BUY", "quantity": 10, "price": 1.0, "note": ""},
        {"date": "2026-09-28", "instrument": a, "action": "BUY", "quantity": 10, "price": 1.0, "note": ""},
        {"date": "2026-09-28", "instrument": "NIFTY 2026-12-29 22000 PE", "action": "BUY", "quantity": 65, "price": 1.0,
         "note": ""},
        {"date": "2026-10-10", "instrument": b, "action": "SELL", "quantity": 10, "price": 1.0, "note": "stop"},
        {"date": "2026-10-10", "instrument": c, "action": "BUY", "quantity": 5, "price": 2.0, "note": "replacement"},
    ])
    assert tracker.held_stocks(ledger) == [a, c]  # list order, options and sold stocks left out
    assert tracker.sold_stocks(ledger) == {b}


def test_no_replacement_without_a_stop_hit():
    pos = _positions([("WELCORP", 100, 500.0, 450.0)])
    plan = tracker.plan_replacements(pos, pd.DataFrame(columns=tracker.LEDGER_COLUMNS), _ranking(["GOODLUCK"]),
                                     pd.Series({"GOODLUCK": 100.0}))
    assert plan.empty


def test_stopped_stock_is_replaced_by_the_first_qualifying_reserve_stock():
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    held = INITIAL_HOLDINGS
    pos = _positions([(s, 100, 500.0, 450.0) for s in held[1:]] + [(held[0], 200, 440.0, 450.0)])
    closes = pd.Series({s: (300.0 if s == RESERVE_SYMBOLS[1] else 100.0) for s in RESERVE_SYMBOLS})
    # The first reserve stock no longer passes the selection rule; the second does
    ranking = _ranking(RESERVE_SYMBOLS[1:])  # the ranking lists only stocks passing the selection rule
    plan = tracker.plan_replacements(pos, pd.DataFrame(columns=tracker.LEDGER_COLUMNS), ranking, closes)
    assert len(plan) == 1
    r = plan.iloc[0]
    assert r["sell"] == held[0] and r["proceeds_inr"] == 200 * 440.0
    assert r["buy"] == RESERVE_SYMBOLS[1] and r["buy_rank"] == 10
    assert r["buy_shares"] == 88000 // 300 and r["cash_left_inr"] == pytest.approx(88000 - 293 * 300.0)
    assert RESERVE_SYMBOLS[0] in r["note"]


def test_two_stops_take_the_queue_in_turn_and_sold_stocks_never_return():
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    held = INITIAL_HOLDINGS
    pos = _positions([(held[0], 10, 90.0, 95.0), (held[1], 10, 90.0, 95.0)] + [(s, 10, 100.0, 95.0) for s in held[2:]])
    ledger = pd.DataFrame([{"date": "2026-10-01", "instrument": RESERVE_SYMBOLS[0], "action": "SELL", "quantity": 1,
                            "price": 1.0, "note": "stopped out earlier"}])
    plan = tracker.plan_replacements(pos, ledger, _ranking(RESERVE_SYMBOLS), pd.Series({s: 50.0 for s in RESERVE_SYMBOLS}))
    assert plan["buy"].tolist() == RESERVE_SYMBOLS[1:3]


def test_proceeds_top_up_the_holdings_when_no_reserve_stock_qualifies():
    """Cash cannot sit idle: with no qualifying reserve stock, the proceeds go into the remaining holdings
    in proportion to their weights, and the ledger trades follow the actual fill."""
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    a, b, c = INITIAL_HOLDINGS[:3]
    pos = _positions([(a, 100, 90.0, 95.0), (b, 10, 200.0, 150.0), (c, 10, 100.0, 80.0)])
    closes = pd.Series({b: 200.0, c: 100.0, **{s: 50.0 for s in RESERVE_SYMBOLS}})
    plan = tracker.plan_replacements(pos, pd.DataFrame(columns=tracker.LEDGER_COLUMNS),
                                     _ranking([]), closes, pd.Series({a: 20.0, b: 30.0, c: 10.0}), max_weight_pct=100)
    assert plan["kind"].tolist() == ["top-up", "top-up"] and plan["buy"].tolist() == [b, c]
    assert plan["alloc_pct"].tolist() == [75.0, 25.0]  # 30 : 10 of the holdings that stay
    assert plan["buy_shares"].tolist() == [9000 * 0.75 // 200, 9000 * 0.25 // 100]
    assert plan["cash_left_inr"].iloc[0] == pytest.approx(9000 - 33 * 200 - 22 * 100)
    t = tracker.replacement_trades(plan, "2026-10-05", fills={a: 80.0})
    assert t[["instrument", "action", "quantity"]].values.tolist() == [[a, "SELL", 100], [b, "BUY", 30], [c, "BUY", 20]]
    assert "top-up" in t["note"].iloc[1]


def test_top_ups_skip_holdings_that_are_falling():
    """A top-up only goes to holdings still in an uptrend (DI gap >= 2)."""
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    a, b, c = INITIAL_HOLDINGS[:3]
    pos = _positions([(a, 100, 90.0, 95.0), (b, 10, 200.0, 150.0), (c, 10, 100.0, 80.0)])
    closes = pd.Series({b: 200.0, c: 100.0, **{s: 50.0 for s in RESERVE_SYMBOLS}})
    plan = tracker.plan_replacements(pos, pd.DataFrame(columns=tracker.LEDGER_COLUMNS), _ranking([]), closes,
                                     pd.Series({a: 20.0, b: 30.0, c: 10.0}), trend_ok={c}, max_weight_pct=100)
    assert plan["buy"].tolist() == [c] and plan["alloc_pct"].tolist() == [100.0]
    assert b in plan["note"].iloc[0]


def test_top_ups_respect_the_15_pct_cap():
    """A holding already at the cap gets nothing; one near it gets only its headroom; the surplus goes to the
    others, and what no holding can take stays in cash."""
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    h = INITIAL_HOLDINGS
    # Stock value 100,000: h0 (9,000) is stopped; h1 at 15,000 (cap), h2 at 14,000, the other five at 12,400
    rows = [(h[0], 90, 100.0, 120.0), (h[1], 150, 100.0, 50.0), (h[2], 140, 100.0, 50.0)] + \
           [(s, 124, 100.0, 50.0) for s in h[3:8]]
    closes = pd.Series({**{s: 100.0 for s in h[:8]}, **{s: 50.0 for s in RESERVE_SYMBOLS}})
    plan = tracker.plan_replacements(_positions(rows), pd.DataFrame(columns=tracker.LEDGER_COLUMNS), _ranking([]),
                                     closes)
    alloc = dict(zip(plan["buy"], plan["buy_inr"]))
    assert h[1] not in alloc and alloc[h[2]] == 1000.0  # at the cap / up to the cap
    assert sum(alloc.values()) == 9000.0 and all(v <= 15000 - 12400 for k, v in alloc.items() if k != h[2])
    # Everyone rising is at the cap: the proceeds wait in the liquid ETF
    full = tracker.plan_replacements(_positions(rows), pd.DataFrame(columns=tracker.LEDGER_COLUMNS), _ranking([]),
                                     closes, trend_ok={h[1]})
    assert full["kind"].tolist() == ["parked"]


def test_proceeds_are_parked_then_redeployed_when_a_stock_qualifies():
    """Nothing qualifies: the sale goes ahead and the money waits in the liquid ETF; once a reserve stock
    qualifies, the parked money buys it (no second sale), and after that nothing is left to redeploy."""
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    a, b = INITIAL_HOLDINGS[:2]
    none = pd.DataFrame(columns=tracker.LEDGER_COLUMNS)
    closes = pd.Series({b: 200.0, **{s: 50.0 for s in RESERVE_SYMBOLS}})
    plan = tracker.plan_replacements(_positions([(a, 100, 90.0, 95.0), (b, 10, 200.0, 150.0)]), none, _ranking([]),
                                     closes, trend_ok=set())
    assert plan["kind"].tolist() == ["parked"] and plan["cash_left_inr"].iloc[0] == 9000.0
    t = tracker.replacement_trades(plan, "2026-10-05", fills={a: 88.0})
    assert t[["instrument", "action", "quantity"]].values.tolist() == [[a, "SELL", 100]]
    ledger = pd.concat([pd.DataFrame([{"date": "2026-09-28", "instrument": s, "action": "BUY", "quantity": q,
                                       "price": 100.0, "note": ""} for s, q in [(a, 100), (b, 10)]]), t],
                       ignore_index=True)
    assert tracker.parked_proceeds(ledger) == {a: 8800.0}
    pos_b = _positions([(b, 10, 200.0, 150.0)])
    # Still nothing qualifies: no action
    assert tracker.plan_replacements(pos_b, ledger, _ranking([]), closes, trend_ok=set()).empty
    # A reserve stock qualifies: buy it with the parked money
    plan2 = tracker.plan_replacements(pos_b, ledger, _ranking(RESERVE_SYMBOLS[:1]), closes, trend_ok=set())
    assert plan2["source"].tolist() == ["parked"] and plan2["buy"].tolist() == [RESERVE_SYMBOLS[0]]
    t2 = tracker.replacement_trades(plan2, "2026-10-09")
    assert t2[["instrument", "action", "quantity"]].values.tolist() == [[RESERVE_SYMBOLS[0], "BUY", 176]]
    assert tracker.parked_proceeds(pd.concat([ledger, t2], ignore_index=True)) == {}


def test_trend_ok_symbols_reads_the_review_tables(tmp_path):
    pd.DataFrame({"symbol": ["AAA", "BBB"], "di_gap": [5.0, 1.0]}).to_csv(tmp_path / "x_full_review_table.csv",
                                                                         index=False)
    assert tracker.trend_ok_symbols(tmp_path) == {"AAA"}
    assert tracker.trend_ok_symbols(tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# Dashboard trade recording, near-stop warning, WhatsApp update
# ---------------------------------------------------------------------------

def test_near_stop_flag(tmp_path, monkeypatch):
    risk = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "stop_loss_price": [100.0, 100.0, 100.0]})
    path = tmp_path / "risk.csv"
    risk.to_csv(path, index=False)
    monkeypatch.setattr(tracker, "RISK_SUMMARY_OUTPUT_CSV", path)
    closes = pd.DataFrame({"AAA": [102.0], "BBB": [110.0], "CCC": [99.0]}, index=pd.to_datetime(["2026-10-01"]))
    ledger = pd.DataFrame([{"date": "2026-09-28", "instrument": s, "action": "BUY", "quantity": 10, "price": 105.0,
                            "note": ""} for s in ["AAA", "BBB", "CCC"]])
    pos = tracker.current_positions(ledger, closes, closes.index[-1]).set_index("symbol")
    assert pos["near_stop"].to_dict() == {"AAA": True, "BBB": False, "CCC": False}  # CCC is breached, not near
    assert pos.loc["AAA", "pct_above_stop"] == 2.0


def test_validate_ledger_catches_bad_rows():
    good = pd.DataFrame([{"date": "2026-09-28", "instrument": "AAA", "action": "BUY", "quantity": 10, "price": 5.0,
                          "note": ""}])
    assert tracker.validate_ledger(good) == []
    bad = pd.concat([good, pd.DataFrame([{"date": "28/09", "instrument": "AAA", "action": "SEL", "quantity": 0,
                                          "price": -1, "note": ""}])])
    problems = " ".join(tracker.validate_ledger(bad))
    assert "date" in problems and "BUY or SELL" in problems and "quantity" in problems and "price" in problems
    oversold = pd.concat([good, good.assign(action="SELL", quantity=11)])
    assert "sells more than was bought" in " ".join(tracker.validate_ledger(oversold))


def test_save_ledger_writes_clean_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRADES_CSV", tmp_path / "trades.csv")
    rows = pd.DataFrame([{"date": "2026-10-02", "instrument": " AAA ", "action": "sell", "quantity": 5.0, "price": 6.0,
                          "note": None},
                         {"date": "2026-09-28", "instrument": "AAA", "action": "BUY", "quantity": 10, "price": 5.0,
                          "note": "x"}])
    assert tracker.save_ledger(rows) == []
    out = pd.read_csv(tmp_path / "trades.csv")
    assert out["date"].tolist() == ["2026-09-28", "2026-10-02"] and out["action"].tolist() == ["BUY", "SELL"]
    assert out["instrument"].tolist() == ["AAA", "AAA"] and out["quantity"].tolist() == [10, 5]


def test_replacement_trades_use_actual_fills():
    plan = pd.DataFrame([{"sell": "AAA", "sell_shares": 100, "sell_close": 90.0, "stop_loss_price": 95.0,
                          "proceeds_inr": 9000.0, "buy": "BBB", "buy_rank": 9, "buy_close": 50.0, "buy_shares": 180,
                          "buy_inr": 9000.0, "cash_left_inr": 0.0, "note": ""}])
    t = tracker.replacement_trades(plan, "2026-10-05", fills={"AAA": 88.0, "BBB": 51.0})
    assert t[["instrument", "action", "quantity", "price"]].values.tolist() == [
        ["AAA", "SELL", 100, 88.0], ["BBB", "BUY", 172, 51.0]]  # 8800 // 51


def test_roll_trades_parse_the_plan_lines():
    roll = pd.DataFrame({"field": ["status", "trade", "trade"],
                         "value": ["TRIGGERED", "SELL 65 NIFTY 2026-12-29 22000 PE @ 12.30",
                                   "BUY 130 NIFTY 2026-12-29 24000 PE @ 45.10 (2 lots)"]})
    t = tracker.roll_trades(roll, "2026-11-02")
    assert t[["instrument", "action", "quantity", "price"]].values.tolist() == [
        ["NIFTY 2026-12-29 22000 PE", "SELL", 65, 12.3], ["NIFTY 2026-12-29 24000 PE", "BUY", 130, 45.1]]


def test_whatsapp_update():
    summary = {"as_of": "2026-10-15", "value_inr": 10250000, "pnl_inr": 250000, "pnl_pct": 2.5,
               "vs_nifty500_inr": 100000, "vs_liquid_fund_inr": 180000}
    pos = pd.DataFrame({"symbol": ["AAA", "BBB"], "pnl_pct": [8.0, -3.0], "near_stop": [False, True],
                        "pct_above_stop": [20.0, 1.5]})
    text = tracker.whatsapp_update(summary, pos, pd.DataFrame())
    assert "Rs 1,02,50,000" in text and "+Rs 2,50,000" in text and "ahead by Rs 1,00,000" in text
    assert "Best: AAA +8.0%" in text and "BBB is 1.5% above its stop" in text
    assert "No action needed" in tracker.whatsapp_update(summary, pos.assign(near_stop=False), pd.DataFrame())
    assert "not invested yet" in tracker.whatsapp_update(None, pd.DataFrame(), pd.DataFrame())
    drift = {"status": "REBALANCE", "lots_held": 8.0, "lots_needed": 9.1,
             "trade": "BUY 65 NIFTY 2026-12-29 22000 PE @ 200.00 (1 lots, hedge ratio)"}
    assert "Hedge ratio drifted (8 lots held, 9.10 needed): BUY 65 NIFTY 2026-12-29 22000 PE @ 200.00" in \
        tracker.whatsapp_update(summary, pos.assign(near_stop=False), pd.DataFrame(), drift=drift)


def test_a_recorded_sale_clears_the_replacement_alert():
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    pos = _positions([(INITIAL_HOLDINGS[0], 10, 90.0, 95.0), (INITIAL_HOLDINGS[1], 10, 90.0, 95.0)])
    # The first stop's replacement was recorded after the last close: only the second is still to do,
    # and it takes the next reserve stock, not the one already bought
    ledger = pd.DataFrame([
        {"date": "2026-10-06", "instrument": INITIAL_HOLDINGS[0], "action": "SELL", "quantity": 10, "price": 90.0, "note": ""},
        {"date": "2026-10-06", "instrument": RESERVE_SYMBOLS[0], "action": "BUY", "quantity": 18, "price": 50.0, "note": ""}])
    plan = tracker.plan_replacements(pos, ledger, _ranking(RESERVE_SYMBOLS), pd.Series({s: 50.0 for s in RESERVE_SYMBOLS}))
    assert plan["sell"].tolist() == [INITIAL_HOLDINGS[1]] and plan["buy"].tolist() == [RESERVE_SYMBOLS[1]]


def test_publish_ledger_pushes_only_the_ledger(tmp_path, monkeypatch):
    import subprocess

    def git(*args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()

    remote, work = tmp_path / "remote.git", tmp_path / "work"
    git("init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("clone", "-q", str(remote), str(work), cwd=tmp_path)
    for k, v in [("user.email", "t@example.com"), ("user.name", "t")]:
        git("config", k, v, cwd=work)
    (work / "data").mkdir()
    (work / "data" / "trades.csv").write_text("date,instrument,action,quantity,price,note\n")
    (work / "other.txt").write_text("committed")
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "init", cwd=work)
    git("push", "-q", "origin", "HEAD:main", cwd=work)
    # Local state: the ledger changed and an unrelated file has uncommitted edits
    (work / "data" / "trades.csv").write_text("date,instrument,action,quantity,price,note\n2026-09-28,AAA,BUY,1,2.0,\n")
    (work / "other.txt").write_text("local edit, not for GitHub")
    monkeypatch.setattr(tracker, "TRADES_CSV", work / "data" / "trades.csv")
    assert tracker.publish_ledger("record AAA").startswith("Saved to GitHub")
    assert "AAA" in git("show", "main:data/trades.csv", cwd=remote)
    assert git("show", "main:other.txt", cwd=remote) == "committed"  # only the ledger was pushed
    assert (work / "other.txt").read_text() == "local edit, not for GitHub"  # working tree untouched
    assert tracker.publish_ledger("again").startswith("GitHub main already has")


class _FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload


class _FakeGitHub:
    def __init__(self, existing=None):
        self.existing, self.put_body = existing, None

    def get(self, url, headers, params, timeout):
        if self.existing is None:
            return _FakeResponse(404)
        import base64
        return _FakeResponse(200, {"sha": "abc", "content": base64.b64encode(self.existing).decode()})

    def put(self, url, headers, json, timeout):
        self.put_body = json
        return _FakeResponse(200)


def test_publish_ledger_via_api(tmp_path, monkeypatch):
    import base64
    ledger = tmp_path / "trades.csv"
    ledger.write_bytes(b"date,instrument,action,quantity,price,note\n2026-09-28,AAA,BUY,1,2.0,\n")
    monkeypatch.setattr(tracker, "TRADES_CSV", ledger)
    gh = _FakeGitHub(existing=b"old")
    assert tracker.publish_ledger_via_api("msg", "t", session=gh).startswith("Saved to GitHub")
    assert gh.put_body["sha"] == "abc" and base64.b64decode(gh.put_body["content"]) == ledger.read_bytes()
    new_file = _FakeGitHub(existing=None)
    tracker.publish_ledger_via_api("msg", "t", session=new_file)
    assert "sha" not in new_file.put_body
    same = _FakeGitHub(existing=ledger.read_bytes())
    assert tracker.publish_ledger_via_api("msg", "t", session=same).startswith("GitHub main already has")
    assert same.put_body is None
