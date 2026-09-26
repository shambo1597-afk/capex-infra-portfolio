"""RRG weekly tails (rrg_tails.py)."""

import pandas as pd
import pytest

from config import LOCKED_PORTFOLIO_SYMBOLS, OUTPUT_DIR
from rrg_tails import basket, tail_dates


def test_tail_dates_are_week_ends_then_the_latest_session():
    sessions = pd.bdate_range("2026-08-03", "2026-09-23")  # ends on a Wednesday
    dates = tail_dates(sessions, weeks=3)
    assert dates[-1] == pd.Timestamp("2026-09-23")
    assert dates[:-1] == [pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-11"), pd.Timestamp("2026-09-18")]


def test_basket_is_the_weighted_daily_rebalanced_index():
    closes = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 45.0]},
                          index=pd.bdate_range("2026-01-01", periods=3))
    b = basket(closes, pd.Series({"A": 0.5, "B": 0.5}))
    assert b["CLOSE_PRICE"].tolist() == pytest.approx([100.0, 105.0, 105.0 * (1 + 0.5 * 0.1 + 0.5 * -0.1)])


def test_committed_tails_end_at_the_review_table_values():
    """The latest tail point is the same RS and momentum the review tables and dashboard show."""
    tails = pd.read_csv(OUTPUT_DIR / "rrg_tails_holdings.csv")
    heads = tails.sort_values("date").groupby("name").tail(1).set_index("name")
    review = pd.concat([pd.read_csv(OUTPUT_DIR / f"{s}_full_review_table.csv")
                        for s in ["cement", "capital_goods", "power"]]).set_index("symbol")
    assert set(heads.index) == set(LOCKED_PORTFOLIO_SYMBOLS)
    for sym in LOCKED_PORTFOLIO_SYMBOLS:
        assert heads.loc[sym, "rs"] == pytest.approx(review.loc[sym, "rs_score_vs_nifty500"], abs=0.01)
        assert heads.loc[sym, "momentum"] == pytest.approx(review.loc[sym, "rs_momentum_vs_nifty500"], abs=0.01)
        assert heads.loc[sym, "quadrant"] == review.loc[sym, "rrg_quadrant_vs_nifty500"]
