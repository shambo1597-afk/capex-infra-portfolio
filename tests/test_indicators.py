"""
Unit tests for Technical Indicators and Relative Strength Calculations.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)
"""

import numpy as np
import pandas as pd
import pytest

from indicators import (
    compute_adx,
    compute_relative_strength,
    compute_rsi,
    compute_support_resistance,
    wilder_smoothing,
)


class TestWilderSmoothing:
    """Test suite for J. Welles Wilder's exponential smoothing implementation."""

    def test_insufficient_length(self):
        """Verify that series shorter than lookback returns all NaNs."""
        series = pd.Series([10.0, 12.0, 11.0])
        smoothed = wilder_smoothing(series, period=14)
        assert len(smoothed) == 3
        assert smoothed.isna().all()

    def test_seed_value(self):
        """Verify that the initial seed at index period-1 equals the simple arithmetic mean."""
        values = list(range(1, 15))  # 1 to 14
        series = pd.Series(values, dtype=float)
        smoothed = wilder_smoothing(series, period=14)

        expected_seed = np.mean(values)
        assert smoothed.iloc[13] == pytest.approx(expected_seed, rel=1e-5)
        # Prior values must be NaN
        assert smoothed.iloc[:13].isna().all()

    def test_subsequent_smoothing_step(self):
        """Verify the exact recursive step: (prev * (N - 1) + curr) / N."""
        values = [10.0] * 14 + [24.0]
        series = pd.Series(values, dtype=float)
        smoothed = wilder_smoothing(series, period=14)

        # Seed at index 13 is 10.0
        assert smoothed.iloc[13] == pytest.approx(10.0, rel=1e-5)
        # Next value at index 14: (10.0 * 13 + 24.0) / 14 = (130 + 24) / 14 = 154 / 14 = 11.0
        expected_next = (10.0 * 13 + 24.0) / 14.0
        assert smoothed.iloc[14] == pytest.approx(expected_next, rel=1e-5)


class TestRSI:
    """Test suite for Relative Strength Index (RSI)."""

    def test_all_gains_reaches_100(self):
        """A monotonically increasing price series should have RSI = 100."""
        prices = pd.Series([100.0 + i * 2.0 for i in range(30)])
        rsi = compute_rsi(prices, period=14)
        # After burn-in period, RSI should reach 100
        assert rsi.iloc[-1] == pytest.approx(100.0, rel=1e-3)

    def test_all_losses_reaches_0(self):
        """A monotonically decreasing price series should have RSI = 0."""
        prices = pd.Series([200.0 - i * 2.0 for i in range(30)])
        rsi = compute_rsi(prices, period=14)
        assert rsi.iloc[-1] == pytest.approx(0.0, rel=1e-3)

    def test_range_bounds(self):
        """RSI must always stay strictly within [0, 100]."""
        np.random.seed(101)
        random_prices = pd.Series(1000.0 + np.cumsum(np.random.normal(0, 15, 100)))
        rsi = compute_rsi(random_prices, period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0.0).all()
        assert (valid_rsi <= 100.0).all()


class TestADX:
    """Test suite for Average Directional Index (ADX) and Directional Indicators (+DI, -DI)."""

    def test_strong_uptrend(self):
        """In a continuous strong upward move, +DI should significantly exceed -DI."""
        n = 50
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        # Steady strong uptrend: higher highs and higher lows
        highs = [100.0 + i * 5.0 for i in range(n)]
        lows = [95.0 + i * 5.0 for i in range(n)]
        closes = [98.0 + i * 5.0 for i in range(n)]

        df = pd.DataFrame({
            "HIGH_PRICE": highs,
            "LOW_PRICE": lows,
            "CLOSE_PRICE": closes
        }, index=dates)

        adx_res = compute_adx(df, period=14)
        latest_res = adx_res.iloc[-1]

        assert latest_res["PLUS_DI"] > latest_res["MINUS_DI"]
        assert latest_res["TREND_DIR"] == "Bullish (Uptrend)"
        assert latest_res["ADX"] > 25.0  # Strong trend threshold

    def test_strong_downtrend(self):
        """In a continuous downward move, -DI should exceed +DI."""
        n = 50
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        highs = [500.0 - i * 5.0 for i in range(n)]
        lows = [490.0 - i * 5.0 for i in range(n)]
        closes = [492.0 - i * 5.0 for i in range(n)]

        df = pd.DataFrame({
            "HIGH_PRICE": highs,
            "LOW_PRICE": lows,
            "CLOSE_PRICE": closes
        }, index=dates)

        adx_res = compute_adx(df, period=14)
        latest_res = adx_res.iloc[-1]

        assert latest_res["MINUS_DI"] > latest_res["PLUS_DI"]
        assert latest_res["TREND_DIR"] == "Bearish (Downtrend)"


class TestRelativeStrength:
    """Test suite for 63-day Relative Strength spread vs Nifty 500 benchmark."""

    def test_positive_alpha(self):
        """Verify that a stock outperforming benchmark yields positive spread."""
        dates = pd.date_range("2026-01-01", periods=70, freq="B")
        # Stock gained 20%
        stock_df = pd.DataFrame({
            "DATE1": dates,
            "CLOSE_PRICE": [100.0] * 6 + [100.0 + i * (20.0 / 63) for i in range(64)]
        })
        # Benchmark gained only 5%
        bench_df = pd.DataFrame({
            "Date": dates,
            "Close": [20000.0] * 6 + [20000.0 + i * (1000.0 / 63) for i in range(64)]
        })

        spread, stock_ret, bench_ret = compute_relative_strength(stock_df, bench_df, lookback_days=63)
        assert stock_ret == pytest.approx(20.0, rel=1e-2)
        assert bench_ret == pytest.approx(5.0, rel=1e-2)
        assert spread == pytest.approx(15.0, rel=1e-2)  # +15.0 percentage points


class TestSupportResistance:
    """Test suite for Support and Resistance detection."""

    def test_nearest_support_and_resistance(self):
        """Verify that support is the highest level below price, and resistance is lowest above price."""
        # Create a series with known swing lows at 90, 95 and swing highs at 110, 115
        prices = [100.0, 90.0, 100.0, 110.0, 95.0, 105.0, 115.0, 102.0]
        # Expand each to a block of 5 so rolling window picks them up
        expanded = []
        for p in prices:
            expanded.extend([p] * 4)

        df = pd.DataFrame({
            "HIGH_PRICE": [p + 1.0 for p in expanded],
            "LOW_PRICE": [p - 1.0 for p in expanded],
            "CLOSE_PRICE": expanded
        })

        current_price = 102.0
        support, resistance = compute_support_resistance(df, rolling_window=5, current_price=current_price)

        assert support is not None
        assert support < current_price
        assert resistance is not None
        assert resistance > current_price
