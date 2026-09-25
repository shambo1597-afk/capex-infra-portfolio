"""
Unit tests for the Relative Rotation Graph (RRG) columns and the full evaluation standard.
"""

import pandas as pd
import pytest

from config import SECTOR_SCREENS
from indicators import compute_relative_strength, compute_rs_momentum
from config import LOCKED_PORTFOLIO_SYMBOLS
from rrg import (
    CONVICTION_HIGH,
    CONVICTION_LOW,
    CONVICTION_MODERATE,
    IMPROVING,
    LAGGING,
    LEADING,
    WEAKENING,
    classify_quadrant,
    conviction_tier,
    plot_rrg,
)
from sector_screen import add_evaluation_columns, review_table_path


def _series(closes, start="2026-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    stock = pd.DataFrame({"DATE1": dates, "CLOSE_PRICE": closes})
    bench = pd.DataFrame({"Date": dates, "Close": [100.0] * len(closes)})  # flat benchmark
    return stock, bench


class TestRsMomentum:
    def test_momentum_is_rs_now_minus_rs_ten_sessions_ago(self):
        # Flat for 64 sessions, then +1% a day for 10: the window ending 10 sessions ago is flat
        closes = [100.0] * 64 + [100.0 * 1.01 ** i for i in range(1, 11)]
        stock, bench = _series(closes)
        rs_now, rs_prev, momentum, ret_prev = compute_rs_momentum(stock, bench, 63, 10)
        assert rs_prev == pytest.approx(0.0) and ret_prev == pytest.approx(0.0)
        assert rs_now == pytest.approx((1.01 ** 10 - 1) * 100, abs=0.01)
        assert momentum == pytest.approx(rs_now - rs_prev, abs=0.01)

    def test_prev_window_matches_truncated_relative_strength(self):
        closes = [100.0 + (i % 7) - 0.2 * i for i in range(90)]
        stock, bench = _series(closes)
        _, rs_prev, _, _ = compute_rs_momentum(stock, bench, 63, 10)
        expected, _, _ = compute_relative_strength(stock.iloc[:-10], bench.iloc[:-10], lookback_days=63)
        assert rs_prev == pytest.approx(expected)

    def test_negative_rs_still_has_momentum(self):
        # Falling for 63 sessions, then recovering: RS < 0 but improving (the IMPROVING quadrant)
        closes = [100.0 - 0.3 * i for i in range(70)] + [79.0 + i for i in range(1, 11)]
        stock, bench = _series(closes)
        rs_now, _, momentum, _ = compute_rs_momentum(stock, bench, 63, 10)
        assert rs_now < 0 and momentum > 0

    def test_insufficient_history_is_nan(self):
        stock, bench = _series([100.0] * 70)
        assert all(pd.isna(v) for v in compute_rs_momentum(stock, bench, 63, 10))


class TestQuadrant:
    @pytest.mark.parametrize("rs, momentum, quadrant", [
        (5.0, 1.0, LEADING), (5.0, -1.0, WEAKENING), (5.0, 0.0, WEAKENING),
        (-5.0, -1.0, LAGGING), (-5.0, 1.0, IMPROVING), (0.0, 1.0, IMPROVING), (0.0, 0.0, LAGGING),
    ])
    def test_classification(self, rs, momentum, quadrant):
        assert classify_quadrant(rs, momentum) == quadrant

    def test_missing_input(self):
        assert classify_quadrant(float("nan"), 1.0) is None and classify_quadrant(1.0, None) is None


CRITERIA = [
    ("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"),
    ("roce", ">", 8, "ROCE > 8%"),
    ("opm", ">", 8, "OPM > 8%"),
]


def _row(symbol, rs=10.0, mom=2.0, rs_sec=10.0, mom_sec=2.0, plus_di=25.0, minus_di=15.0,
         passes=(True, True, True), roce=25.0, recent=20.0):
    return {"symbol": symbol, "rs_score_vs_nifty500": rs, "rs_momentum_vs_nifty500": mom,
            "rs_score_vs_sector_avg": rs_sec, "rs_momentum_vs_sector": mom_sec,
            "plus_di": plus_di, "minus_di": minus_di, "roce": roce, "recent_10day_contribution_pct": recent,
            "pass_market_cap": passes[0], "pass_roce": passes[1], "pass_opm": passes[2],
            "fundamentals_passed_count": sum(passes)}


class TestEvaluationColumns:
    def test_di_gap_and_thin_flag_either_direction(self):
        table = add_evaluation_columns(pd.DataFrame([
            _row("VOLT", plus_di=21.66, minus_di=20.10),   # gap 1.56: bullish but thin
            _row("BEAR", plus_di=18.0, minus_di=19.5),     # gap -1.5: bearish and thin
            _row("REAL", plus_di=25.0, minus_di=15.0),
            _row("EDGE", plus_di=22.0, minus_di=20.0),     # exactly 2.0 is not thin
        ]), CRITERIA).set_index("symbol")
        assert table["di_gap"].to_dict() == pytest.approx({"VOLT": 1.56, "BEAR": -1.5, "REAL": 10.0, "EDGE": 2.0})
        assert table["thin_trend_flag"].to_dict() == {"VOLT": True, "BEAR": True, "REAL": False, "EDGE": False}

    def test_high_turnover_flag_only_opm_failure_with_strong_roce(self):
        table = add_evaluation_columns(pd.DataFrame([
            _row("APL", passes=(True, True, False), roce=31.8),    # only OPM fails, ROCE > 20
            _row("LOWROCE", passes=(True, True, False), roce=15.0),
            _row("TWOFAIL", passes=(False, True, False), roce=40.0),
            _row("CLEAN", passes=(True, True, True), roce=40.0),
        ]), CRITERIA).set_index("symbol")
        assert table["high_turnover_business_flag"].to_dict() == {
            "APL": True, "LOWROCE": False, "TWOFAIL": False, "CLEAN": False}
        # Flagged for review only: it is not a clean fundamental pass or a candidate
        assert not table.loc["APL", "fundamentals_clean"] and not table.loc["APL", "full_standard_candidate"]

    def test_high_turnover_flag_not_applicable_without_opm_criterion(self):
        criteria = [c for c in CRITERIA if c[0] != "opm"]
        rows = [{k: v for k, v in _row("PWR", passes=(True, True, True)).items() if k != "pass_opm"}]
        rows[0]["fundamentals_passed_count"] = 2
        table = add_evaluation_columns(pd.DataFrame(rows), criteria)
        assert pd.isna(table.loc[0, "high_turnover_business_flag"])

    def test_full_standard_candidate(self):
        table = add_evaluation_columns(pd.DataFrame([
            _row("GOOD"),
            _row("WEAKSEC", mom_sec=-1.0),                      # WEAKENING vs sector
            _row("THIN", plus_di=20.0, minus_di=19.0),
            _row("BEARGAP", plus_di=10.0, minus_di=20.0),       # real gap, wrong direction
            _row("FAILFA", passes=(True, False, True)),
        ]), CRITERIA).set_index("symbol")
        assert table["full_standard_candidate"].to_dict() == {
            "GOOD": True, "WEAKSEC": False, "THIN": False, "BEARGAP": False, "FAILFA": False}
        assert table.loc["WEAKSEC", "rrg_quadrant_vs_sector"] == WEAKENING


@pytest.mark.parametrize("sector", list(SECTOR_SCREENS))
def test_committed_review_tables_carry_consistent_evaluation_columns(sector):
    """Every committed row's derived columns agree with the inputs stored beside them."""
    table = pd.read_csv(review_table_path(sector))
    recomputed = add_evaluation_columns(table, SECTOR_SCREENS[sector]["criteria"])
    for col in ["rrg_quadrant_vs_nifty500", "rrg_quadrant_vs_sector", "thin_trend_flag",
                "fundamentals_clean", "full_standard_candidate"]:
        assert table[col].tolist() == recomputed[col].tolist(), col
    assert (table["di_gap"] - (table["plus_di"] - table["minus_di"])).abs().max() <= 0.011
    # Momentum vs sector sums to ~0 (both RS spreads do), up to 2-decimal rounding
    assert abs(table["rs_momentum_vs_sector"].sum()) <= len(table) * 0.01


def test_plot_rrg_writes_png(tmp_path):
    df = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "x": [5, 5, -5, -5], "y": [1, -1, -1, 1]})
    df["q"] = [classify_quadrant(x, y) for x, y in zip(df["x"], df["y"])]
    out = plot_rrg(df, "x", "y", "q", "Test", tmp_path / "rrg.png", highlight_symbols=["A"])
    assert out.exists() and out.stat().st_size > 10_000


class TestConvictionTier:
    @pytest.mark.parametrize("vs_market, vs_sector, tier", [
        (LEADING, LEADING, CONVICTION_HIGH),
        (LEADING, WEAKENING, CONVICTION_MODERATE),     # leading in one view only
        (WEAKENING, LEADING, CONVICTION_MODERATE),
        (IMPROVING, LEADING, CONVICTION_MODERATE),
        (IMPROVING, LAGGING, CONVICTION_MODERATE),     # improving in either view
        (WEAKENING, IMPROVING, CONVICTION_MODERATE),
        (WEAKENING, WEAKENING, CONVICTION_LOW),
        (LAGGING, WEAKENING, CONVICTION_LOW),
        (LAGGING, LAGGING, CONVICTION_LOW),
    ])
    def test_rules(self, vs_market, vs_sector, tier):
        assert conviction_tier(vs_market, vs_sector) == tier

    def test_every_quadrant_pair_gets_exactly_one_tier(self):
        quadrants = [LEADING, WEAKENING, LAGGING, IMPROVING]
        tiers = {conviction_tier(a, b) for a in quadrants for b in quadrants}
        assert tiers == {CONVICTION_HIGH, CONVICTION_MODERATE, CONVICTION_LOW}

    def test_missing_quadrant(self):
        assert conviction_tier(None, LEADING) is None and conviction_tier(LEADING, float("nan")) is None


def test_every_locked_stock_has_rrg_data_in_the_committed_review_tables():
    """The dashboard's conviction tiers need both quadrants for each locked stock (incl. APLAPOLLO)."""
    table = pd.concat([pd.read_csv(review_table_path(s)) for s in SECTOR_SCREENS], ignore_index=True)
    locked = table[table["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)].set_index("symbol")
    assert sorted(locked.index) == sorted(LOCKED_PORTFOLIO_SYMBOLS)
    assert locked[["rrg_quadrant_vs_nifty500", "rrg_quadrant_vs_sector"]].notna().all().all()
    assert (locked["fundamentals_status"] == "OK").all() and (locked["price_sessions"] >= 240).all()
