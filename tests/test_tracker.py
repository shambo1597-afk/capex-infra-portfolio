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
