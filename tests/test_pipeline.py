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
    PORTFOLIO_SYMBOLS,
    POWER_SECTOR_STOCKS,
    BUSINESS_FOCUS_NOTES,
    NIFTY_INFRA_CONSTITUENTS_FILE,
    NIFTY_INFRA_SECTOR_ADDITIONS,
    SECTOR_CONSTITUENT_FILES,
    SYMBOL_ALIASES,
    SYMBOL_NAME,
    sector_of,
    sector_universe,
)
from fetch_data import (
    NSEBhavcopyFetcher,
    get_one_year_date_range,
    load_benchmark_tri,
)


class TestUniverseConfiguration:
    """Verify stock universe setup and symbol alias mapping."""

    def test_universes_match_official_constituent_files(self):
        """Each universe is its official niftyindices.com file plus the named Nifty Infrastructure
        additions (appended, never hand-typed), and every stock maps to exactly one sector."""
        for sector, universe in (("Cement", CEMENT_STOCKS), ("Capital Goods", CAPITAL_GOODS_EPC_STOCKS),
                                 ("Power", POWER_SECTOR_STOCKS)):
            official = pd.read_csv(SECTOR_CONSTITUENT_FILES[sector])["Symbol"].str.strip().tolist()
            assert universe == official + NIFTY_INFRA_SECTOR_ADDITIONS.get(sector, [])
            assert all(sector_of(sym) == sector for sym in universe)
        assert (len(CEMENT_STOCKS), len(CAPITAL_GOODS_EPC_STOCKS), len(POWER_SECTOR_STOCKS)) == (16, 52, 21)
        assert len(set(PORTFOLIO_SYMBOLS)) == len(PORTFOLIO_SYMBOLS) == 89
        assert sector_of("NOT_A_SYMBOL") == "Other"

    def test_nifty_infra_additions_are_official_infra_constituents(self):
        infra = pd.read_csv(NIFTY_INFRA_CONSTITUENTS_FILE)
        infra_symbols = set(infra["Symbol"].str.strip())
        for sector, symbols in NIFTY_INFRA_SECTOR_ADDITIONS.items():
            for sym in symbols:
                assert sym in infra_symbols
                assert [r["index"] for r in sector_universe(sector) if r["symbol"] == sym] == ["Nifty Infrastructure"]
                assert sym in BUSINESS_FOCUS_NOTES  # every addition carries a business-focus review note

    def test_locked_portfolio(self):
        expected_locked = ["JKCEMENT", "VOLTAMP", "FINCABLES", "APARINDS", "WELCORP",
                           "APLAPOLLO", "ACMESOLAR", "TATAPOWER"]
        assert LOCKED_PORTFOLIO_SYMBOLS == expected_locked
        assert list(LOCKED_PORTFOLIO) == expected_locked
        # Every pick belongs to one of the three official universes, with name/sector from the files
        for sym, info in LOCKED_PORTFOLIO.items():
            assert sym in PORTFOLIO_SYMBOLS
            assert info["sector"] == sector_of(sym) and info["name"] == SYMBOL_NAME[sym]
        sectors = [info["sector"] for info in LOCKED_PORTFOLIO.values()]
        assert (sectors.count("Cement"), sectors.count("Capital Goods"), sectors.count("Power")) == (1, 5, 2)

    def test_sector_screen_picks_are_the_locked_portfolio(self):
        from sector_screen import CURRENT_PICKS
        assert CURRENT_PICKS == LOCKED_PORTFOLIO_SYMBOLS

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
        assert not list(tmp_path.glob("bhav*25-Dec-2025.csv"))

    def test_legacy_symbol_filtered_cache_is_ignored(self, tmp_path):
        """Old 'bhav_' cache files held only the universe of their day; they must be re-fetched."""
        (tmp_path / "bhav_23-Sep-2024.csv").write_text(
            "SYMBOL,SERIES,DATE1,CLOSE_PRICE\nOTHER,EQ,2024-09-23,1.0\n", encoding="utf-8")
        fetcher = self._fetcher(tmp_path, [_Resp(200, _bhav_csv("23-Sep-2024"))])

        df = fetcher.fetch_daily_bhavcopy(date(2024, 9, 23), ["JKCEMENT"])

        assert df is not None and df["SYMBOL"].tolist() == ["JKCEMENT"]

    def test_symbols_added_later_are_served_from_full_day_cache(self, tmp_path):
        """The cache keeps every symbol of the day, so extending the universe needs no re-download."""
        two_symbols = _bhav_csv("23-Sep-2024") + (
            "ITDCEM, EQ, 23-Sep-2024, 500.00, 505.00, 510.00, 495.00, 507.00, 508.00, 503.00, "
            "10000, 50.00, 400, 5000, 50.00\n")
        fetcher = self._fetcher(tmp_path, [_Resp(200, two_symbols)])
        first = fetcher.fetch_daily_bhavcopy(date(2024, 9, 23), ["JKCEMENT"])
        assert first["SYMBOL"].tolist() == ["JKCEMENT"]

        # No network response queued: this must come from cache, with the ITDCEM -> CEMPRO alias applied
        later = fetcher.fetch_daily_bhavcopy(date(2024, 9, 23), ["JKCEMENT", "CEMPRO"])

        assert sorted(later["SYMBOL"]) == ["CEMPRO", "JKCEMENT"]
        assert later.loc[later["SYMBOL"] == "CEMPRO", "CLOSE_PRICE"].iloc[0] == 508.00

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

    def test_special_weekend_sessions_are_requested(self, tmp_path):
        """Weekends are skipped except listed special sessions (e.g. Budget Sunday 1-Feb-2026)."""
        fetcher = self._fetcher(tmp_path, [])
        requested = []
        fetcher.fetch_daily_bhavcopy = lambda target_date, target_symbols, use_cache: requested.append(target_date)

        fetcher.fetch_date_range(date(2026, 1, 30), date(2026, 2, 2), ["JKCEMENT"])

        assert requested == [date(2026, 1, 30), date(2026, 2, 1), date(2026, 2, 2)]  # Sat 31-Jan skipped



def _index_file(day_label, close):
    return ("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%)\n"
            f"Nifty 50,{day_label},100,110,90,105,1,1\n"
            f"Nifty 500,{day_label},{close - 10},{close + 10},{close - 20},{close},5,0.1\n")


class TestOfficialIndexBenchmark:
    """Nifty 500 benchmark from NSE's official daily index files, on the Bhavcopy session calendar."""

    def test_index_closes_parsed_with_holidays_and_wrong_sessions(self, tmp_path):
        replies = {
            "22092025": _Resp(200, _index_file("22-09-2025", 22794.2)),
            "23092025": _Resp(404, "<html>Not found</html>"),          # holiday
            "24092025": _Resp(200, _index_file("23-09-2025", 22600.0)),  # previous session served
        }
        fetcher = NSEBhavcopyFetcher(cache_dir=tmp_path, delay_seconds=0)
        fetcher.session.get = lambda url, timeout: replies[url.rsplit("_", 1)[1][:8]]
        fetcher.session_initialized = True

        df = fetcher.fetch_index_closes(date(2025, 9, 22), date(2025, 9, 24))

        assert df["Date"].dt.date.tolist() == [date(2025, 9, 22)]
        assert df["Close"].tolist() == [22794.2]
        assert (tmp_path / "holiday_v2_23-Sep-2025.flag").exists()
        assert (tmp_path / "holiday_v2_24-Sep-2025.flag").exists()

    def test_unpublished_recent_day_is_not_cached_as_holiday(self, tmp_path):
        """A 404 for today may only mean NSE hasn't published yet: it must be re-checked later."""
        from datetime import timedelta
        today = date.today()
        fetcher = NSEBhavcopyFetcher(cache_dir=tmp_path, delay_seconds=0)
        fetcher.session.get = lambda url, timeout: _Resp(404, "<html>Not found</html>")
        fetcher.session_initialized = True

        fetcher.fetch_daily_bhavcopy(today, ["JKCEMENT"])
        fetcher.fetch_index_closes(today - timedelta(days=1), today)

        assert not list(tmp_path.glob("holiday*"))

    def test_blocked_index_day_is_retried(self, tmp_path):
        replies = [AKAMAI_403, _Resp(200, _index_file("22-09-2026", 22794.2))]
        fetcher = NSEBhavcopyFetcher(cache_dir=tmp_path, delay_seconds=0)
        fetcher.session.get = lambda url, timeout: replies.pop(0)
        fetcher.session_initialized = True
        fetcher.warm_up_session = lambda: setattr(fetcher, "session_initialized", True)
        with patch("fetch_data.time.sleep"):
            df = fetcher.fetch_index_closes(date(2026, 9, 22), date(2026, 9, 22))
        assert df["Close"].tolist() == [22794.2]

    def test_benchmark_uses_official_series(self):
        from fetch_data import fetch_benchmark_nifty500
        official = pd.DataFrame({"Date": [pd.Timestamp("2026-09-24")], "Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [22552.6]})
        with patch("fetch_data.NSEBhavcopyFetcher.fetch_index_closes", return_value=official), \
                patch("fetch_data.yf.download") as yf_download:
            df = fetch_benchmark_nifty500("2025-09-24", "2026-09-24")
        assert df["Close"].tolist() == [22552.6]
        yf_download.assert_not_called()

    def test_yfinance_fallback_includes_end_date(self):
        from fetch_data import fetch_benchmark_nifty500
        yf_df = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-09-24")], name="Date"))
        with patch("fetch_data.NSEBhavcopyFetcher.fetch_index_closes", return_value=pd.DataFrame()), \
                patch("fetch_data.yf.download", return_value=yf_df) as yf_download:
            fetch_benchmark_nifty500("2025-09-24", "2026-09-24")
        assert yf_download.call_args.kwargs["end"] == "2026-09-25"  # yfinance's end is exclusive
