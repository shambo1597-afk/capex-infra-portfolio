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
    ranking = pd.concat([_ranking([RESERVE_SYMBOLS[0]], "Low"), _ranking(RESERVE_SYMBOLS[1:])])
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


def test_proceeds_wait_in_cash_when_no_reserve_stock_qualifies():
    from config import INITIAL_HOLDINGS, RESERVE_SYMBOLS
    pos = _positions([(INITIAL_HOLDINGS[0], 10, 90.0, 95.0)])
    plan = tracker.plan_replacements(pos, pd.DataFrame(columns=tracker.LEDGER_COLUMNS),
                                     _ranking(RESERVE_SYMBOLS, "Low"), pd.Series({s: 50.0 for s in RESERVE_SYMBOLS}))
    r = plan.iloc[0]
    assert pd.isna(r["buy"]) and r["buy_shares"] == 0 and r["cash_left_inr"] == 900.0
