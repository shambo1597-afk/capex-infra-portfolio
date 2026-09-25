"""
Unit tests for the hybrid stop-loss, volatility and portfolio risk summary calculations.
"""

import math

import pandas as pd
import pytest

from analysis import RISK_SUMMARY_COLUMNS, generate_portfolio_risk_summary
from config import LOCKED_PORTFOLIO_SYMBOLS
from stoploss import (
    compute_annualized_volatility,
    compute_daily_returns,
    compute_daily_volatility,
    compute_historical_expected_return,
    compute_volatility_cap_stop,
    select_stop_loss,
)

# Daily returns of +1%, -2%, +3%, 0%, expressed as Bhavcopy rows (CLOSE vs PREV_CLOSE).
# Hand computation: mean = 0.005; squared deviations sum to 0.0013; sample variance
# (ddof=1) = 0.0013 / 3; annualized variance = 0.0013 / 3 * 252 = 0.1092.
FIXTURE_RETURNS = [0.01, -0.02, 0.03, 0.0]
EXPECTED_ANNUALIZED_VOL = math.sqrt(0.1092)  # 0.330454...


def _bhavcopy_rows(symbol, returns, start_price=100.0, start_date="2026-09-01"):
    rows, prev_close = [], start_price
    for i, r in enumerate(returns):
        close = prev_close * (1 + r)
        rows.append({
            "SYMBOL": symbol,
            "DATE1": (pd.Timestamp(start_date) + pd.offsets.BDay(i)).strftime("%Y-%m-%d"),
            "PREV_CLOSE": prev_close,
            "CLOSE_PRICE": close,
        })
        prev_close = close
    return pd.DataFrame(rows)


class TestTighterOfTwoCandidates:
    """The final stop is whichever candidate is closer to (i.e. higher below) the price."""

    def test_support_wins_when_tighter(self):
        # Price 100, 2% daily vol: cap = 100 x (1 - 1.75 x 0.02 x sqrt(21)) = 83.96; support 95 is closer
        vol_cap = compute_volatility_cap_stop(100.0, 0.02)
        assert vol_cap == pytest.approx(100 * (1 - 1.75 * 0.02 * math.sqrt(21)))

        stop, method = select_stop_loss(100.0, support_price=95.0, volatility_cap_price=vol_cap)

        assert method == "support"
        assert stop == pytest.approx(95.0)

    def test_volatility_cap_wins_when_tighter(self):
        # Price 100, 0.5% daily vol: cap = 100 x (1 - 1.75 x 0.005 x sqrt(21)) = 95.99; support 80 is far below
        vol_cap = compute_volatility_cap_stop(100.0, 0.005)

        stop, method = select_stop_loss(100.0, support_price=80.0, volatility_cap_price=vol_cap)

        assert method == "volatility_cap"
        assert stop == pytest.approx(100 * (1 - 1.75 * 0.005 * math.sqrt(21)))

    def test_support_at_or_above_price_is_ignored(self):
        stop, method = select_stop_loss(100.0, support_price=100.0, volatility_cap_price=90.0)
        assert (stop, method) == (90.0, "volatility_cap")

    def test_missing_support_falls_back_to_volatility_cap(self):
        assert select_stop_loss(100.0, None, 90.0) == (90.0, "volatility_cap")

    def test_tie_goes_to_support(self):
        assert select_stop_loss(100.0, 90.0, 90.0) == (90.0, "support")

    def test_no_usable_candidate(self):
        assert select_stop_loss(100.0, None, None) == (None, "unavailable")

    def test_volatility_cap_unavailable_when_cushion_reaches_100_pct(self):
        assert compute_volatility_cap_stop(100.0, 0.5) is None
        assert compute_volatility_cap_stop(100.0, None) is None


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
            _bhavcopy_rows("AAA", FIXTURE_RETURNS),
            _bhavcopy_rows("BBB", [0.001, -0.001, 0.002, -0.002]),
        ], ignore_index=True)
        technicals = pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "current_price": [100.0, 100.0],
            "nearest_support": [99.0, 80.0],
        })
        out_csv = tmp_path / "portfolio_risk_summary.csv"

        risk = generate_portfolio_risk_summary(stock_data, technicals, symbols=["AAA", "BBB"],
                                               output_csv_path=out_csv)

        assert list(risk.columns) == RISK_SUMMARY_COLUMNS
        assert out_csv.exists() and len(pd.read_csv(out_csv)) == 2
        aaa, bbb = risk.iloc[0], risk.iloc[1]
        assert aaa["annualized_volatility_pct"] == pytest.approx(33.05, abs=0.01)
        assert aaa["historical_expected_return_pct"] == pytest.approx(126.0)
        assert aaa["weight_pct"] == bbb["weight_pct"] == 50.0  # equal-weight placeholder
        # AAA: support 99 (1% below) beats a ~34% volatility cap
        assert (aaa["stop_loss_method"], aaa["stop_loss_price"], aaa["stop_loss_pct_below_current"]) == ("support", 99.0, 1.0)
        # BBB: low volatility -> cap sits within a few % of price, tighter than support at 80
        assert bbb["stop_loss_method"] == "volatility_cap"
        assert 80.0 < bbb["stop_loss_price"] < 100.0

    def test_locked_portfolio_is_equal_weighted(self):
        risk = generate_portfolio_risk_summary(pd.DataFrame(), pd.DataFrame(), output_csv_path=None)
        assert risk["symbol"].tolist() == LOCKED_PORTFOLIO_SYMBOLS
        assert (risk["weight_pct"] == round(100 / len(LOCKED_PORTFOLIO_SYMBOLS), 2)).all()  # 11.11% for 9
        assert (risk["stop_loss_method"] == "unavailable").all()  # no data: no stop, no crash
