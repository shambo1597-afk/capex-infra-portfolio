"""
Tests for the split / bonus / demerger price adjustment (corporate_actions.py).
"""

from datetime import date

import pandas as pd
import pytest

from corporate_actions import adjust_for_corporate_actions, parse_action, unexplained_jumps


class TestParseAction:
    @pytest.mark.parametrize("subject, kind, factor", [
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", "split", 0.1),
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", "split", 0.2),
        ("Face Value Split (Sub-Division) - From Rs10/- Per Share To Rs 5/- Per Share", "split", 0.5),
        ("Consolidation of shares From Rs 1/- Per Share To Rs 10/- Per Share", "split", 10.0),
        ("Bonus 2:1", "bonus", 1 / 3),      # 2 new for every 1 held: price divides by 3
        ("Bonus 1:3", "bonus", 0.75),       # 1 new for every 3 held
        ("Bonus 1:1", "bonus", 0.5),
    ])
    def test_splits_and_bonuses(self, subject, kind, factor):
        parsed = parse_action(subject)
        assert parsed["kind"] == kind and parsed["factor"] == pytest.approx(factor)

    def test_demerger_uses_the_price_gap(self):
        assert parse_action("Demerger") == {"kind": "gap"}
        assert parse_action(" Composite Scheme Of Arrangement") == {"kind": "gap"}

    @pytest.mark.parametrize("subject", ["Interim Dividend - Rs 2 Per Share", "Rights 5:78 @ Premium Rs 110/-",
                                         "Annual General Meeting"])
    def test_other_actions_are_not_adjusted(self, subject):
        assert parse_action(subject) is None


def _rows(symbol, closes, start="2026-01-05"):
    dates = pd.bdate_range(start, periods=len(closes))
    prev = [closes[0]] + closes[:-1]
    return pd.DataFrame({"SYMBOL": symbol, "DATE1": dates, "PREV_CLOSE": prev, "CLOSE_PRICE": closes,
                         "HIGH_PRICE": [c * 1.01 for c in closes], "LOW_PRICE": [c * 0.99 for c in closes],
                         "TTL_TRD_QNTY": [1000.0] * len(closes)})


def test_split_is_joined_up():
    # 1-for-5 split on the 3rd session: 500 -> 102 is really a +2% day
    df = _rows("X", [490.0, 500.0, 102.0, 104.0])
    actions = {"X": [{"ex_date": date(2026, 1, 7), "kind": "split", "factor": 0.2}]}
    adj = adjust_for_corporate_actions(df, actions)
    assert adj["CLOSE_PRICE"].tolist() == pytest.approx([98.0, 100.0, 102.0, 104.0])
    assert (adj["CLOSE_PRICE"] / adj["PREV_CLOSE"] - 1).iloc[2] == pytest.approx(0.02)
    assert adj["TTL_TRD_QNTY"].iloc[0] == pytest.approx(5000.0)  # earlier quantities scale inversely
    assert adj["CLOSE_PRICE"].iloc[3] == 104.0 and len(unexplained_jumps(adj)) == 0
    assert len(unexplained_jumps(df)) == 1  # the raw data shows a fake -80% day


def test_bonus_and_demerger_gap():
    df = pd.concat([_rows("B", [300.0, 303.0, 101.0]), _rows("D", [100.0, 100.0, 60.0, 61.0])])
    actions = {"B": [{"ex_date": date(2026, 1, 7), "kind": "bonus", "factor": 1 / 3}],
               "D": [{"ex_date": date(2026, 1, 7), "kind": "gap"}]}
    adj = adjust_for_corporate_actions(df, actions)
    b = adj[adj["SYMBOL"] == "B"]["CLOSE_PRICE"].tolist()
    d = adj[adj["SYMBOL"] == "D"]
    assert b == pytest.approx([100.0, 101.0, 101.0])
    assert d["CLOSE_PRICE"].tolist() == pytest.approx([60.0, 60.0, 60.0, 61.0])  # ex-date move neutralised
    assert len(unexplained_jumps(adj)) == 0


def test_action_outside_window_or_unavailable_leaves_prices_unchanged():
    df = _rows("X", [100.0, 101.0, 102.0])
    outside = {"X": [{"ex_date": date(2025, 6, 2), "kind": "split", "factor": 0.5}]}
    assert adjust_for_corporate_actions(df, outside)["CLOSE_PRICE"].tolist() == [100.0, 101.0, 102.0]
    assert adjust_for_corporate_actions(df, {"X": None})["CLOSE_PRICE"].tolist() == [100.0, 101.0, 102.0]


def test_listing_day_is_not_an_unexplained_jump():
    df = _rows("NEW", [150.0, 152.0])
    df.loc[0, "PREV_CLOSE"] = 100.0  # issue price on the listing day
    assert unexplained_jumps(df).empty
