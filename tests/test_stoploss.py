"""
Unit tests for the ATR stop-loss, volatility and portfolio risk summary calculations.
"""

import math

import pandas as pd
import pytest

from analysis import RISK_SUMMARY_COLUMNS, generate_portfolio_risk_summary
from config import LOCKED_PORTFOLIO_SYMBOLS
from stoploss import (
    apply_trailing_stop,
    compute_annualized_volatility,
    compute_atr,
    compute_atr_stop,
    compute_daily_returns,
    compute_daily_volatility,
    compute_historical_expected_return,
)

# Daily returns of +1%, -2%, +3%, 0%, expressed as Bhavcopy rows (CLOSE vs PREV_CLOSE).
# Hand computation: mean = 0.005; squared deviations sum to 0.0013; sample variance
# (ddof=1) = 0.0013 / 3; annualized variance = 0.0013 / 3 * 252 = 0.1092.
FIXTURE_RETURNS = [0.01, -0.02, 0.03, 0.0]
EXPECTED_ANNUALIZED_VOL = math.sqrt(0.1092)  # 0.330454...


def _bhavcopy_rows(symbol, returns, start_price=100.0, start_date="2026-09-01", day_range=0.0):
    rows, prev_close = [], start_price
    for i, r in enumerate(returns):
        close = prev_close * (1 + r)
        rows.append({
            "SYMBOL": symbol,
            "DATE1": (pd.Timestamp(start_date) + pd.offsets.BDay(i)).strftime("%Y-%m-%d"),
            "PREV_CLOSE": prev_close,
            "CLOSE_PRICE": close,
            "HIGH_PRICE": close + day_range / 2,
            "LOW_PRICE": close - day_range / 2,
        })
        prev_close = close
    return pd.DataFrame(rows)


class TestAtr:
    def test_flat_price_with_constant_range_has_atr_equal_to_range(self):
        # Close never moves, High - Low = 4 every day: every true range is 4, so ATR = 4
        rows = _bhavcopy_rows("TEST", [0.0] * 20, day_range=4.0)
        assert compute_atr(rows) == pytest.approx(4.0)

    def test_gap_uses_previous_close(self):
        # A +10% gap with a 1-point range: TR = |High - PrevClose| = 110.5 - 100 = 10.5 on that day
        rows = _bhavcopy_rows("TEST", [0.0] * 14 + [0.10], day_range=1.0)
        # Wilder: seed = mean of first 14 TRs (1.0), then (1.0 x 13 + 10.5) / 14
        assert compute_atr(rows) == pytest.approx((13 * 1.0 + 10.5) / 14)

    def test_needs_a_full_period(self):
        assert compute_atr(_bhavcopy_rows("TEST", [0.0] * 13, day_range=1.0)) is None
        assert compute_atr(pd.DataFrame()) is None


class TestAtrStop:
    """Base stop = price - 3 ATR; a support up to 1 ATR beyond it pulls the stop to support - 0.25 ATR."""

    def test_base_stop_is_three_atr_below_price(self):
        assert compute_atr_stop(100.0, 2.0) == (pytest.approx(94.0), "atr")

    def test_support_just_beyond_base_stop_moves_stop_below_support(self):
        # Base 94; support 93 is within 1 ATR (92-94) -> stop 93 - 0.5 = 92.5
        assert compute_atr_stop(100.0, 2.0, support_price=93.0) == (pytest.approx(92.5), "support")
        # Band edge (support exactly 1 ATR beyond) still counts
        assert compute_atr_stop(100.0, 2.0, support_price=92.0) == (pytest.approx(91.5), "support")

    def test_support_close_to_price_never_tightens_the_stop(self):
        # The old rule would have put the stop at 99.9 (0.1% below price)
        assert compute_atr_stop(100.0, 2.0, support_price=99.9) == (pytest.approx(94.0), "atr")

    def test_support_far_below_is_ignored(self):
        assert compute_atr_stop(100.0, 2.0, support_price=80.0) == (pytest.approx(94.0), "atr")

    def test_unavailable_without_price_or_atr(self):
        assert compute_atr_stop(100.0, None) == (None, "unavailable")
        assert compute_atr_stop(None, 2.0) == (None, "unavailable")
        assert compute_atr_stop(10.0, 5.0) == (None, "unavailable")  # 3 ATR exceeds the price


class TestTrailingStop:
    def test_higher_previous_stop_is_kept(self):
        assert apply_trailing_stop(90.0, "atr", 93.0, 100.0) == (93.0, "trailed")

    def test_unchanged_stop_read_back_at_csv_precision_is_not_trailed(self):
        assert apply_trailing_stop(472.4077, "atr", 472.41, 509.45) == (472.4077, "atr")

    def test_higher_new_stop_replaces_previous(self):
        assert apply_trailing_stop(95.0, "atr", 93.0, 100.0) == (95.0, "atr")

    def test_previous_stop_at_or_above_price_is_breached(self):
        assert apply_trailing_stop(85.0, "atr", 93.0, 92.0) == (93.0, "breached")

    def test_no_previous_stop(self):
        assert apply_trailing_stop(90.0, "support", None, 100.0) == (90.0, "support")
        assert apply_trailing_stop(90.0, "atr", float("nan"), 100.0) == (90.0, "atr")


class TestVolatilityAndReturns:
    def test_annualized_volatility_matches_hand_computed_value(self):
        returns = compute_daily_returns(_bhavcopy_rows("TEST", FIXTURE_RETURNS))

        assert list(returns.round(10)) == FIXTURE_RETURNS
        assert compute_daily_volatility(returns) == pytest.approx(math.sqrt(0.0013 / 3))
        assert compute_annualized_volatility(returns) == pytest.approx(EXPECTED_ANNUALIZED_VOL)
        assert compute_annualized_volatility(returns) == pytest.approx(0.330454, abs=1e-6)

    def test_historical_expected_return_is_mean_daily_return_x_252(self):
        returns = compute_daily_returns(_bhavcopy_rows("TEST", FIXTURE_RETURNS))
        assert compute_historical_expected_return(returns) == pytest.approx(0.005 * 252)

    def test_missing_sessions_do_not_create_multi_day_returns(self):
        """Returns use PREV_CLOSE, so a session absent from the history is not folded in."""
        rows = _bhavcopy_rows("TEST", FIXTURE_RETURNS).drop(index=1)  # drop the -2% session
        returns = compute_daily_returns(rows)
        assert list(returns.round(10)) == [0.01, 0.03, 0.0]

    def test_duplicate_sessions_are_counted_once(self):
        rows = _bhavcopy_rows("TEST", FIXTURE_RETURNS)
        rows = pd.concat([rows, rows.iloc[[2]]], ignore_index=True)
        assert len(compute_daily_returns(rows)) == 4

    def test_lookback_keeps_most_recent_returns(self):
        returns = compute_daily_returns(_bhavcopy_rows("TEST", FIXTURE_RETURNS), lookback=2)
        assert list(returns.round(10)) == [0.03, 0.0]

    def test_volatility_needs_two_returns(self):
        assert compute_annualized_volatility(pd.Series([0.01])) is None


class TestPortfolioRiskSummary:
    def test_summary_rows_columns_and_csv(self, tmp_path):
        stock_data = pd.concat([
            _bhavcopy_rows("AAA", [0.0] * 20, day_range=2.0),   # ATR 2, close 100
            _bhavcopy_rows("BBB", [0.0] * 20, day_range=2.0),
        ], ignore_index=True)
        technicals = pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "current_price": [100.0, 100.0],
            "nearest_support": [99.0, 93.0],
        })
        out_csv = tmp_path / "portfolio_risk_summary.csv"

        risk = generate_portfolio_risk_summary(stock_data, technicals, symbols=["AAA", "BBB"],
                                               output_csv_path=out_csv, previous_stops={"BBB": 95.0})

        assert list(risk.columns) == RISK_SUMMARY_COLUMNS
        assert out_csv.exists() and len(pd.read_csv(out_csv)) == 2
        aaa, bbb = risk.iloc[0], risk.iloc[1]
        assert aaa["weight_pct"] == bbb["weight_pct"] == 50.0  # equal-weight placeholder
        assert (aaa["atr_14"], aaa["atr_pct"]) == (2.0, 2.0)
        # AAA: support 1% below price is ignored; stop = 100 - 3 x 2
        assert (aaa["stop_loss_method"], aaa["stop_loss_price"], aaa["stop_loss_pct_below_current"]) == ("atr", 94.0, 6.0)
        # BBB: support 93 would give 92.5, but the previous review's 95 is higher and is kept
        assert (bbb["stop_loss_method"], bbb["stop_loss_price"]) == ("trailed", 95.0)

    def test_volatility_and_return_columns(self):
        stock_data = _bhavcopy_rows("AAA", FIXTURE_RETURNS)
        technicals = pd.DataFrame({"symbol": ["AAA"], "current_price": [100.0], "nearest_support": [90.0]})
        aaa = generate_portfolio_risk_summary(stock_data, technicals, symbols=["AAA"], output_csv_path=None).iloc[0]
        assert aaa["annualized_volatility_pct"] == pytest.approx(33.05, abs=0.01)
        assert aaa["historical_expected_return_pct"] == pytest.approx(126.0)
        assert aaa["stop_loss_method"] == "unavailable"  # 4 sessions: too few for ATR(14)

    def test_locked_portfolio_is_equal_weighted(self):
        risk = generate_portfolio_risk_summary(pd.DataFrame(), pd.DataFrame(), output_csv_path=None)
        assert risk["symbol"].tolist() == LOCKED_PORTFOLIO_SYMBOLS
        assert (risk["weight_pct"] == round(100 / len(LOCKED_PORTFOLIO_SYMBOLS), 2)).all()  # 12.5% for 8
        assert (risk["stop_loss_method"] == "unavailable").all()  # no data: no stop, no crash
