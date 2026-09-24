"""
Unit tests for the staleness check on the manually maintained Nifty 500 TRI CSV.
"""

from datetime import date

import pandas as pd
import pytest

from config import TRI_STALE_THRESHOLD_TRADING_DAYS
from fetch_data import assess_tri_staleness, format_tri_staleness_warning, load_benchmark_tri

# Last TRI date in the fixtures: Thursday 24-Sep-2026
LAST_TRI_DATE = date(2026, 9, 24)

TRI_CSV_CONTENT = """IndexName,Date,Total Returns Index,Net Total Return Index
NIFTY 500,22-Sep-2026,36666.48,33708.29
NIFTY 500,23-Sep-2026,36893.30,33916.77
NIFTY 500,24-Sep-2026,36950.10,33968.95
"""


@pytest.fixture
def tri_df():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-09-22", "2026-09-23", "2026-09-24"]),
        "Total Returns Index": [36666.48, 36893.30, 36950.10],
    })


@pytest.fixture
def tri_csv(tmp_path):
    path = tmp_path / "nifty500_tri.csv"
    path.write_text(TRI_CSV_CONTENT, encoding="utf-8")
    return path


class TestAssessTriStaleness:
    """Threshold logic: stale only when more than 3 trading days behind the end date."""

    def test_threshold_is_three_trading_days(self):
        assert TRI_STALE_THRESHOLD_TRADING_DAYS == 3

    @pytest.mark.parametrize("reference_date, expected_gap, expected_stale", [
        # Fresh: data covers the end date, or trails it by one session
        (date(2026, 9, 24), 0, False),
        (date(2026, 9, 25), 1, False),
        # Borderline: exactly 3 trading days behind (Fri, Mon, Tue) is still acceptable
        (date(2026, 9, 29), 3, False),
        # Just over: 4 trading days behind
        (date(2026, 9, 30), 4, True),
        # Very stale: a quarter behind
        (date(2026, 12, 31), 70, True),
    ], ids=["same-day", "one-day", "borderline-3", "just-over-4", "very-stale"])
    def test_gap_and_threshold(self, tri_df, reference_date, expected_gap, expected_stale):
        staleness = assess_tri_staleness(tri_df, reference_date)

        assert staleness.last_date == LAST_TRI_DATE
        assert staleness.reference_date == reference_date
        assert staleness.trading_days_behind == expected_gap
        assert staleness.is_stale is expected_stale

    def test_weekends_are_not_counted(self, tri_df):
        """Thu -> following Mon is 4 calendar days but only 2 trading days (Fri, Mon)."""
        staleness = assess_tri_staleness(tri_df, date(2026, 9, 28))
        assert staleness.trading_days_behind == 2
        assert not staleness.is_stale

    def test_weekend_end_date_counts_only_the_friday(self, tri_df):
        staleness = assess_tri_staleness(tri_df, date(2026, 9, 27))  # Sunday
        assert staleness.trading_days_behind == 1

    def test_end_date_before_last_tri_date_is_not_stale(self, tri_df):
        staleness = assess_tri_staleness(tri_df, date(2026, 6, 30))
        assert staleness.trading_days_behind == 0
        assert not staleness.is_stale

    def test_accepts_string_reference_date(self, tri_df):
        assert assess_tri_staleness(tri_df, "2026-09-30").trading_days_behind == 4

    def test_defaults_to_today(self, tri_df):
        assert assess_tri_staleness(tri_df).reference_date == date.today()

    def test_empty_data_returns_none(self):
        assert assess_tri_staleness(pd.DataFrame()) is None


class TestStalenessWarning:
    def test_banner_names_last_date_gap_and_fix(self, tri_df):
        banner = format_tri_staleness_warning(assess_tri_staleness(tri_df, date(2026, 10, 5)))

        assert "WARNING: TRI benchmark data is stale." in banner
        assert "Last available date: 24-Sep-2026" in banner
        assert "7 trading days behind" in banner
        assert "05-Oct-2026" in banner
        assert "niftyindices.com" in banner
        assert all(len(line) <= 115 for line in banner.splitlines())

    def test_stale_csv_prints_banner_but_still_returns_data(self, tri_csv, capsys):
        df = load_benchmark_tri(tri_csv, as_of=date(2026, 10, 5))

        assert "WARNING: TRI benchmark data is stale." in capsys.readouterr().out
        assert len(df) == 3  # Warning only: the stale data is still used

    def test_fresh_csv_prints_nothing(self, tri_csv, capsys):
        load_benchmark_tri(tri_csv, as_of=date(2026, 9, 29))
        assert "stale" not in capsys.readouterr().out

    def test_warning_can_be_disabled(self, tri_csv, capsys):
        load_benchmark_tri(tri_csv, as_of=date(2026, 12, 31), warn_if_stale=False)
        assert "stale" not in capsys.readouterr().out

    def test_unparseable_end_date_never_fails_the_load(self, tri_csv):
        df = load_benchmark_tri(tri_csv, as_of="not-a-date")
        assert len(df) == 3
