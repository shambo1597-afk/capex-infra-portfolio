"""
Pipeline and Integration test suite for Indian Equity Portfolio.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)
"""

from datetime import date
from unittest.mock import patch

import pandas as pd

from config import (
    CEMENT_STOCKS,
    CAPITAL_GOODS_EPC_STOCKS,
    DEFAULT_TRI_CSV_PATH,
    LOCKED_PORTFOLIO,
    LOCKED_PORTFOLIO_SYMBOLS,
    POWER_SECTOR_STOCKS,
    SYMBOL_ALIASES,
)
from fetch_data import (
    NSEBhavcopyFetcher,
    get_one_year_date_range,
    load_benchmark_tri,
)


class TestUniverseConfiguration:
    """Verify stock universe setup and symbol alias mapping."""

    def test_cement_stocks(self):
        assert "JKCEMENT" in CEMENT_STOCKS
        assert "ULTRACEMCO" in CEMENT_STOCKS
        assert "STARCEMENT" in CEMENT_STOCKS
        assert len(CEMENT_STOCKS) == 3

    def test_capital_goods_stocks(self):
        expected_cg = [
            "ABB", "CGPOWER", "GVT&D", "POWERINDIA", "TRITURBINE", "TDPOWERSYS",
            "SIEMENS", "BHEL", "INOXWIND", "ENRIN", "SUZLON", "THERMAX", "VOLTAMP",
        ]
        for sym in expected_cg:
            assert sym in CAPITAL_GOODS_EPC_STOCKS
        assert len(CAPITAL_GOODS_EPC_STOCKS) == 13

    def test_power_sector_stocks(self):
        expected_power = [
            "ADANIENSOL", "ADANIPOWER", "CESC", "KPIGREEN", "NAVA", "NLCINDIA",
            "NTPC", "POWERGRID", "PTC", "TATAPOWER", "TORNTPOWER",
        ]
        for sym in expected_power:
            assert sym in POWER_SECTOR_STOCKS
        assert len(POWER_SECTOR_STOCKS) == 11

    def test_locked_portfolio(self):
        expected_locked = ["JKCEMENT", "ULTRACEMCO", "STARCEMENT", "BHEL", "VOLTAMP", "POWERGRID", "TATAPOWER", "NTPC"]
        assert len(LOCKED_PORTFOLIO) == 8
        assert sorted(LOCKED_PORTFOLIO_SYMBOLS) == sorted(expected_locked)

    def test_symbol_aliases(self):
        # GET&D -> GVT&D (GE Vernova T&D)
        assert SYMBOL_ALIASES.get("GET&D") == "GVT&D"
        # ITDCEM -> CEMPRO (Cemindia Projects)
        assert SYMBOL_ALIASES.get("ITDCEM") == "CEMPRO"


class TestBenchmarkData:
    """Verify benchmark data ingestion functions."""

    def test_tri_loader(self):
        if DEFAULT_TRI_CSV_PATH.exists():
            df = load_benchmark_tri(DEFAULT_TRI_CSV_PATH)
            assert not df.empty
            assert "Date" in df.columns
            assert "Total Returns Index" in df.columns
            assert len(df) > 50

    def test_date_range_helper(self):
        start, end = get_one_year_date_range()
        assert (end - start).days >= 365
        assert start < end


class TestBhavcopyParsing:
    """Verify Bhavcopy CSV parsing and whitespace stripping."""

    def test_bhavcopy_parsing_logic(self, tmp_path):
        sample_csv = """ SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
JKCEMENT, EQ, 23-Sep-2024, 4700.00, 4710.00, 4800.00, 4690.00, 4780.00, 4785.00, 4750.00, 50000, 2375.00, 1500, 25000, 50.00
GET&D, EQ, 23-Sep-2024, 1500.00, 1510.00, 1550.00, 1495.00, 1540.00, 1545.00, 1530.00, 80000, 1224.00, 3000, 40000, 50.00
RANDOMSYM, EQ, 23-Sep-2024, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 100, 0.01, 1, 100, 100.00
"""
        fetcher = NSEBhavcopyFetcher(cache_dir=tmp_path)
        # Mock the session response
        class MockResponse:
            status_code = 200
            text = sample_csv

        fetcher.session.get = lambda url, timeout: MockResponse()
        fetcher.session_initialized = True

        df = fetcher.fetch_daily_bhavcopy(
            target_date=date(2024, 9, 23),
            target_symbols=["JKCEMENT", "GVT&D"],
            use_cache=False
        )

        assert df is not None
        assert len(df) == 2
        # Symbols must be cleaned and GET&D mapped to GVT&D
        symbols = df["SYMBOL"].tolist()
        assert "JKCEMENT" in symbols
        assert "GVT&D" in symbols
        assert "RANDOMSYM" not in symbols
        # Numeric conversions
        assert df.loc[df["SYMBOL"] == "JKCEMENT", "CLOSE_PRICE"].iloc[0] == 4785.00


class _Resp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def _bhav_csv(day_label, close=4785.00):
    return (" SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, "
            "AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
            f"JKCEMENT, EQ, {day_label}, 4700.00, 4710.00, 4800.00, 4690.00, 4780.00, {close}, 4750.00, "
            "50000, 2375.00, 1500, 25000, 50.00\n")


AKAMAI_403 = _Resp(403, "<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY>Access Denied</BODY></HTML>")


class TestBhavcopyHolidayAndBlockHandling:
    """
    NSE answers 404 for holidays, but its Akamai layer also answers 403 'Access Denied' to
    valid trading days. Blocks must be retried and never cached as holidays; files for the
    wrong session (served on some holidays) must be rejected.
    """

    @staticmethod
    def _fetcher(tmp_path, responses):
        fetcher = NSEBhavcopyFetcher(cache_dir=tmp_path, delay_seconds=0)
        queue = list(responses)
        fetcher.session.get = lambda url, timeout: queue.pop(0)
        fetcher.session_initialized = True
        fetcher.warm_up_session = lambda: setattr(fetcher, "session_initialized", True)
        return fetcher

    def test_blocked_request_is_retried_and_not_flagged_as_holiday(self, tmp_path):
        fetcher = self._fetcher(tmp_path, [AKAMAI_403, AKAMAI_403, _Resp(200, _bhav_csv("23-Sep-2024"))])
        with patch("fetch_data.time.sleep"):
            df = fetcher.fetch_daily_bhavcopy(date(2024, 9, 23), ["JKCEMENT"], use_cache=True)

        assert df is not None and len(df) == 1
        assert not list(tmp_path.glob("holiday*"))
        assert fetcher.failed_dates == []

    def test_persistently_blocked_day_is_reported_not_cached_as_holiday(self, tmp_path):
        fetcher = self._fetcher(tmp_path, [AKAMAI_403] * 4)
        with patch("fetch_data.time.sleep"):
            df = fetcher.fetch_daily_bhavcopy(date(2024, 9, 23), ["JKCEMENT"], use_cache=True)

        assert df is None
        assert not list(tmp_path.glob("holiday*"))  # retried on the next run
        assert fetcher.failed_dates == [date(2024, 9, 23)]

    def test_404_is_a_holiday(self, tmp_path):
        fetcher = self._fetcher(tmp_path, [_Resp(404, "<!DOCTYPE html><html>Not found</html>")])
        assert fetcher.fetch_daily_bhavcopy(date(2025, 10, 2), ["JKCEMENT"]) is None
        assert (tmp_path / "holiday_v2_02-Oct-2025.flag").exists()
        assert fetcher.failed_dates == []

    def test_previous_session_served_on_holiday_is_rejected(self, tmp_path):
        """Requesting 25-Dec returns the 24-Dec file; it must not duplicate 24-Dec."""
        fetcher = self._fetcher(tmp_path, [_Resp(200, _bhav_csv("24-Dec-2025"))])
        assert fetcher.fetch_daily_bhavcopy(date(2025, 12, 25), ["JKCEMENT"]) is None
        assert (tmp_path / "holiday_v2_25-Dec-2025.flag").exists()
        assert not (tmp_path / "bhav_25-Dec-2025.csv").exists()

    def test_stale_cached_file_for_other_session_is_discarded(self, tmp_path):
        (tmp_path / "bhav_25-Dec-2025.csv").write_text(
            "SYMBOL,SERIES,DATE1,CLOSE_PRICE\nJKCEMENT,EQ,2025-12-24,4785.0\n", encoding="utf-8")
        fetcher = self._fetcher(tmp_path, [])
        assert fetcher.fetch_daily_bhavcopy(date(2025, 12, 25), ["JKCEMENT"]) is None
        assert not (tmp_path / "bhav_25-Dec-2025.csv").exists()

    def test_legacy_holiday_flags_are_ignored(self, tmp_path):
        """Old-style flags may mark blocked trading days as holidays, so they are re-checked."""
        (tmp_path / "holiday_23-Sep-2024.flag").touch()
        fetcher = self._fetcher(tmp_path, [_Resp(200, _bhav_csv("23-Sep-2024"))])
        df = fetcher.fetch_daily_bhavcopy(date(2024, 9, 23), ["JKCEMENT"])
        assert df is not None and len(df) == 1

    def test_date_range_drops_duplicate_sessions(self, tmp_path):
        fetcher = self._fetcher(tmp_path, [])
        day = pd.DataFrame({"SYMBOL": ["JKCEMENT"], "DATE1": [pd.Timestamp("2024-09-23")], "CLOSE_PRICE": [1.0]})
        fetcher.fetch_daily_bhavcopy = lambda target_date, target_symbols, use_cache: day.copy()

        out = fetcher.fetch_date_range(date(2024, 9, 23), date(2024, 9, 24), ["JKCEMENT"])

        assert len(out) == 1
