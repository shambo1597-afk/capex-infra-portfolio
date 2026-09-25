"""
Unit tests for automated Nifty 500 TRI browser automation and fallback mechanisms.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from datetime import date

import pandas as pd
import pytest

from config import DEFAULT_TRI_CSV_PATH
from fetch_data import fetch_benchmark_tri_automated, load_benchmark_tri


SAMPLE_TRI_CSV_CONTENT = """IndexName,Date,Total Returns Index,Net Total Return Index
NIFTY 500,23-Sep-2024,35000.10,34800.50
NIFTY 500,24-Sep-2024,35250.40,35040.20
NIFTY 500,25-Sep-2024,35400.80,35190.90
"""


def _wire_mock_playwright(mock_sync_playwright, goto_statuses=(200,), api_records=None, api_statuses=None):
    """
    Wire a mocked sync_playwright() so the automated flow runs end to end:
    each navigation attempt returns the next status in goto_statuses, each TRI data
    call returns the next status in api_statuses (all HTTP 200 by default) with
    api_records, and expect_download yields the sample CSV.
    Returns (mock_browser, mock_page) for call assertions.
    """
    mock_download = MagicMock()
    mock_download.save_as = lambda dest: Path(dest).write_text(SAMPLE_TRI_CSV_CONTENT, encoding="utf-8")
    mock_expect_download = MagicMock()
    mock_expect_download.__enter__ = MagicMock(return_value=MagicMock(value=mock_download))
    mock_expect_download.__exit__ = MagicMock(return_value=None)

    def _expect_response_cm(status):
        api_response = MagicMock(status=status)
        api_response.json.return_value = api_records if api_records is not None else [{"Index Name": "Nifty 500"}]
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(value=api_response))
        cm.__exit__ = MagicMock(return_value=None)
        return cm

    mock_page = MagicMock()
    mock_page.goto.side_effect = [MagicMock(status=s) for s in goto_statuses]
    mock_page.expect_download.return_value = mock_expect_download
    if api_statuses is None:
        mock_page.expect_response.side_effect = lambda *a, **k: _expect_response_cm(200)
    else:
        mock_page.expect_response.side_effect = [_expect_response_cm(st) for st in api_statuses]
    mock_page.locator.return_value.count.return_value = 1
    mock_page.locator.return_value.all_inner_texts.return_value = ["--Select--", "Broad Market Indices", "NIFTY 500"]

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page
    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context
    mock_browser.new_page.return_value.evaluate.return_value = "Mozilla/5.0 HeadlessChrome/141.0.0.0"
    mock_p = MagicMock()
    mock_p.chromium.launch.return_value = mock_browser

    mock_sync_playwright.return_value.__enter__.return_value = mock_p
    mock_sync_playwright.return_value.__exit__.return_value = None
    return mock_browser, mock_page


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Navigation retries back off with time.sleep; keep the suite fast."""
    with patch("fetch_data.time.sleep"):
        yield


class TestAutomatedTRI:
    """Test suite verifying automated TRI fetching, mocked Playwright flow, and fallback safety."""

    @patch("playwright.sync_api.sync_playwright")
    def test_fetch_benchmark_tri_automated_success_mocked(self, mock_sync_playwright, tmp_path):
        """
        Verify that when Playwright successfully downloads the TRI CSV, the returned DataFrame
        has expected columns, correct datatypes, and matches the manual CSV contract.
        """
        mock_browser, _ = _wire_mock_playwright(mock_sync_playwright)

        df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 3

        # Column schema verification
        expected_cols = ["IndexName", "Date", "Total Returns Index", "Net Total Return Index"]
        for col in expected_cols:
            assert col in df.columns, f"Expected column '{col}' missing from automated output"

        # Type checks
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert pd.api.types.is_numeric_dtype(df["Total Returns Index"])
        assert pd.api.types.is_numeric_dtype(df["Net Total Return Index"])

        # Values verification
        assert df["IndexName"].iloc[0] == "NIFTY 500"
        assert df["Total Returns Index"].iloc[0] == pytest.approx(35000.10)
        assert df["Net Total Return Index"].iloc[0] == pytest.approx(34800.50)

    @patch("playwright.sync_api.sync_playwright")
    def test_fetch_benchmark_tri_automated_fallback_on_error(self, mock_sync_playwright, caplog):
        """
        Verify that if Playwright raises an error (network drop, element not found, etc.),
        the pipeline logs a warning and falls back to load_benchmark_tri() without crashing.
        """
        # Simulate browser launch failure or unhandled Playwright exception
        mock_sync_playwright.side_effect = RuntimeError("Playwright headless browser failed to launch")

        with caplog.at_level("WARNING"):
            df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        # Verify fallback warning logged
        assert any("Automated benchmark TRI fetching failed" in record.message for record in caplog.records)

        # If DEFAULT_TRI_CSV_PATH exists, verify returned data matches manual CSV
        if DEFAULT_TRI_CSV_PATH.exists():
            manual_df = load_benchmark_tri(DEFAULT_TRI_CSV_PATH)
            assert not df.empty
            assert len(df) == len(manual_df)
            assert list(df.columns) == list(manual_df.columns)
        else:
            assert isinstance(df, pd.DataFrame)

    def test_schema_parity_between_manual_and_automated(self, tmp_path):
        """
        Verify that manual load_benchmark_tri and mock-automated fetch_benchmark_tri_automated
        produce identical schema columns and types so callers can treat them interchangeably.
        """
        dummy_csv = tmp_path / "dummy_tri.csv"
        dummy_csv.write_text(SAMPLE_TRI_CSV_CONTENT, encoding="utf-8")

        manual_df = load_benchmark_tri(dummy_csv)

        with patch("playwright.sync_api.sync_playwright") as mock_pw:
            _wire_mock_playwright(mock_pw)
            automated_df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert list(manual_df.columns) == list(automated_df.columns)
        assert manual_df["Date"].dtype == automated_df["Date"].dtype
        assert manual_df["Total Returns Index"].dtype == automated_df["Total Returns Index"].dtype
        assert manual_df["Net Total Return Index"].dtype == automated_df["Net Total Return Index"].dtype

    @patch("playwright.sync_api.sync_playwright")
    def test_launches_full_chromium_not_headless_shell(self, mock_sync_playwright):
        """Akamai blocks chromium_headless_shell; the full build (channel='chromium') must be used."""
        mock_browser, _ = _wire_mock_playwright(mock_sync_playwright)
        fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        launch_kwargs = mock_sync_playwright.return_value.__enter__.return_value.chromium.launch.call_args.kwargs
        assert launch_kwargs["channel"] == "chromium"
        context_ua = mock_browser.new_context.call_args.kwargs["user_agent"]
        assert "HeadlessChrome" not in context_ua

    @patch("playwright.sync_api.sync_playwright")
    def test_retries_navigation_after_bot_rejection(self, mock_sync_playwright):
        """A 403 from Akamai on the first attempt is retried with a fresh context."""
        mock_browser, mock_page = _wire_mock_playwright(mock_sync_playwright, goto_statuses=(403, 200))

        df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert mock_page.goto.call_count == 2
        assert mock_browser.new_context.call_count == 2
        assert len(df) == 3

    @patch("playwright.sync_api.sync_playwright")
    def test_falls_back_when_all_navigation_attempts_rejected(self, mock_sync_playwright, caplog):
        """Persistent bot rejection ends in the manual CSV fallback, not an exception."""
        _wire_mock_playwright(mock_sync_playwright, goto_statuses=(403, 403, 403, 403))

        with caplog.at_level("WARNING"):
            df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert any("rejected 4 browser sessions" in record.message for record in caplog.records)
        assert isinstance(df, pd.DataFrame)

    @patch("playwright.sync_api.sync_playwright")
    def test_ranges_over_one_year_are_split_into_portal_windows(self, mock_sync_playwright):
        """The portal rejects ranges over 365 days, so a ~26-month request needs 3 submissions."""
        _, mock_page = _wire_mock_playwright(mock_sync_playwright)

        df = fetch_benchmark_tri_automated("2023-01-01", "2025-03-01")

        assert mock_page.expect_response.call_count == 3
        assert mock_page.expect_download.call_count == 3
        # Identical sample rows from each window are de-duplicated by date
        assert len(df) == 3
        assert not df["Date"].duplicated().any()

    @patch("playwright.sync_api.sync_playwright")
    def test_data_call_rejection_retries_session_and_keeps_captured_windows(self, mock_sync_playwright):
        """
        Akamai can pass the page but 403 the data call later. That restarts the session,
        and windows captured before the rejection are not requested again.
        """
        # 2 windows: first captured, second 403'd; new session fetches only the second
        mock_browser, mock_page = _wire_mock_playwright(
            mock_sync_playwright, goto_statuses=(200, 200), api_statuses=(200, 403, 200))

        df = fetch_benchmark_tri_automated("2024-01-01", "2025-03-01")

        assert mock_browser.new_context.call_count == 2
        assert mock_page.expect_response.call_count == 3
        assert mock_page.expect_download.call_count == 2
        assert len(df) == 3


    @patch("playwright.sync_api.sync_playwright")
    def test_failure_after_blocked_request_retries_session(self, mock_sync_playwright):
        """
        Akamai blocking a background call (e.g. dropdown population) surfaces only as a
        later timeout; because the session saw a 403 it is retried, not treated as fatal.
        """
        mock_browser, mock_page = _wire_mock_playwright(mock_sync_playwright, goto_statuses=(200, 200))
        response_handlers = []
        mock_page.on.side_effect = lambda event, handler: response_handlers.append(handler) if event == "response" else None

        def select_option_blocked_once(*args, **kwargs):
            if mock_page.select_option.call_count == 1:
                response_handlers[-1](MagicMock(status=403, url="https://niftyindices.com/BackPage/x"))
                raise TimeoutError("Page.wait_for_selector: Timeout 15000ms exceeded.")
        mock_page.select_option.side_effect = select_option_blocked_once

        df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert mock_browser.new_context.call_count == 2
        assert len(df) == 3

    @patch("playwright.sync_api.sync_playwright")
    def test_genuine_breakage_falls_back_without_retrying(self, mock_sync_playwright, caplog):
        """A failure with no bot-protection signal (e.g. site redesign) is not retried."""
        mock_browser, mock_page = _wire_mock_playwright(mock_sync_playwright)
        mock_page.select_option.side_effect = TimeoutError("element not found")

        with caplog.at_level("WARNING"):
            df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert mock_browser.new_context.call_count == 1
        assert any("Automated benchmark TRI fetching failed" in r.message for r in caplog.records)
        assert isinstance(df, pd.DataFrame)


class TestUpdateTriCsv:
    """update_tri_csv appends new sessions in the manual-download format, or changes nothing."""

    CSV = ("IndexName,Date,Total Returns Index,Net Total Return Index\n"
           "NIFTY 500,21-Sep-2026,36500.00,33500.00\n"
           "NIFTY 500,22-Sep-2026,36666.48,33708.29\n")

    @staticmethod
    def _fetched(rows):
        return pd.DataFrame(rows, columns=["IndexName", "Date", "Total Returns Index", "Net Total Return Index"]) \
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))

    def test_appends_new_sessions_in_manual_format(self, tmp_path):
        from fetch_data import update_tri_csv
        path = tmp_path / "tri.csv"
        path.write_text(self.CSV)
        fetched = self._fetched([("NIFTY 500", "2026-09-22", 36666.48, 33708.29),   # overlap, agrees
                                 ("NIFTY 500", "2026-09-23", 36893.30, 33916.77),
                                 ("NIFTY 500", "2026-09-24", 36278.13, 33351.20)])
        calls = []
        added, last = update_tri_csv(date(2026, 9, 24), csv_path=path,
                                     fetch=lambda s, e: calls.append((s, e)) or fetched)
        assert (added, last) == (2, date(2026, 9, 24))
        assert calls == [(date(2026, 9, 15), date(2026, 9, 24))]  # a week of overlap before the last date
        lines = path.read_text().strip().splitlines()
        assert lines[0] == "IndexName,Date,Total Returns Index,Net Total Return Index"
        assert lines[-1] == "NIFTY 500,24-Sep-2026,36278.13,33351.20"  # trailing zero kept
        assert len(load_benchmark_tri(path, warn_if_stale=False)) == 4

    def test_disagreeing_overlap_leaves_csv_untouched(self, tmp_path):
        from fetch_data import update_tri_csv
        path = tmp_path / "tri.csv"
        path.write_text(self.CSV)
        fetched = self._fetched([("NIFTY 500", "2026-09-22", 37000.00, 33708.29),
                                 ("NIFTY 500", "2026-09-23", 36893.30, 33916.77)])
        with pytest.raises(ValueError, match="disagrees"):
            update_tri_csv(date(2026, 9, 24), csv_path=path, fetch=lambda s, e: fetched)
        assert path.read_text() == self.CSV

    def test_no_new_sessions_leaves_file_byte_identical(self, tmp_path):
        from fetch_data import update_tri_csv
        path = tmp_path / "tri.csv"
        path.write_text(self.CSV)
        fetched = self._fetched([("NIFTY 500", "2026-09-22", 36666.48, 33708.29)])
        assert update_tri_csv(date(2026, 9, 25), csv_path=path, fetch=lambda s, e: fetched) == (0, date(2026, 9, 22))
        assert path.read_text() == self.CSV

    def test_already_current_does_not_fetch(self, tmp_path):
        from fetch_data import update_tri_csv
        path = tmp_path / "tri.csv"
        path.write_text(self.CSV)
        assert update_tri_csv(date(2026, 9, 22), csv_path=path,
                              fetch=lambda s, e: pytest.fail("should not fetch")) == (0, date(2026, 9, 22))

    def test_fetch_failure_propagates_and_leaves_csv_untouched(self, tmp_path):
        from fetch_data import update_tri_csv
        path = tmp_path / "tri.csv"
        path.write_text(self.CSV)

        def blocked(s, e):
            raise RuntimeError("niftyindices.com rejected 4 browser sessions")
        with pytest.raises(RuntimeError):
            update_tri_csv(date(2026, 9, 24), csv_path=path, fetch=blocked)
        assert path.read_text() == self.CSV
