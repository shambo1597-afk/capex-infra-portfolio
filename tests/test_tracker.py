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
