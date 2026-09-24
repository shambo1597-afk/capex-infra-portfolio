"""
Pipeline and Integration test suite for Indian Equity Portfolio.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)
"""

import io
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from config import (
    CEMENT_STOCKS,
    CAPITAL_GOODS_EPC_STOCKS,
    DEFAULT_TRI_CSV_PATH,
    LOCKED_PORTFOLIO,
    LOCKED_PORTFOLIO_SYMBOLS,
    PORTFOLIO_SYMBOLS,
    POWER_SECTOR_STOCKS,
    SYMBOL_ALIASES,
)
from fetch_data import (
    NSEBhavcopyFetcher,
    fetch_benchmark_nifty500,
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
