"""
Unit tests for automated Nifty 500 TRI browser automation and fallback mechanisms.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import DEFAULT_TRI_CSV_PATH
from fetch_data import fetch_benchmark_tri_automated, load_benchmark_tri


SAMPLE_TRI_CSV_CONTENT = """IndexName,Date,Total Returns Index,Net Total Return Index
NIFTY 500,23-Sep-2024,35000.10,34800.50
NIFTY 500,24-Sep-2024,35250.40,35040.20
NIFTY 500,25-Sep-2024,35400.80,35190.90
"""


class TestAutomatedTRI:
    """Test suite verifying automated TRI fetching, mocked Playwright flow, and fallback safety."""

    @patch("playwright.sync_api.sync_playwright")
    def test_fetch_benchmark_tri_automated_success_mocked(self, mock_sync_playwright, tmp_path):
        """
        Verify that when Playwright successfully downloads the TRI CSV, the returned DataFrame
        has expected columns, correct datatypes, and matches the manual CSV contract.
        """
        # Create a sample downloaded CSV file
        mock_downloaded_csv = tmp_path / "nifty500_downloaded.csv"
        mock_downloaded_csv.write_text(SAMPLE_TRI_CSV_CONTENT, encoding="utf-8")

        # Mock download object
        mock_download = MagicMock()
        mock_download.save_as = lambda dest: Path(dest).write_text(SAMPLE_TRI_CSV_CONTENT, encoding="utf-8")

        # Mock expect_download context manager
        mock_download_info = MagicMock()
        mock_download_info.value = mock_download

        mock_expect_download = MagicMock()
        mock_expect_download.__enter__ = MagicMock(return_value=mock_download_info)
        mock_expect_download.__exit__ = MagicMock(return_value=None)

        # Mock page locators and actions
        mock_page = MagicMock()
        mock_page.expect_download.return_value = mock_expect_download
        mock_page.locator.return_value.count.return_value = 1
        mock_page.locator.return_value.all_inner_texts.return_value = ["Broad Market Indices", "NIFTY 500"]

        # Mock browser and context
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        # Mock playwright instance
        mock_p = MagicMock()
        mock_p.chromium.launch.return_value = mock_browser

        mock_sync_playwright.return_value.__enter__.return_value = mock_p
        mock_sync_playwright.return_value.__exit__.return_value = None

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
            mock_download = MagicMock()
            mock_download.save_as = lambda dest: Path(dest).write_text(SAMPLE_TRI_CSV_CONTENT, encoding="utf-8")

            mock_expect = MagicMock()
            mock_expect.__enter__ = MagicMock(return_value=MagicMock(value=mock_download))
            mock_expect.__exit__ = MagicMock(return_value=None)

            mock_page = MagicMock()
            mock_page.expect_download.return_value = mock_expect
            mock_page.locator.return_value.count.return_value = 1
            mock_page.locator.return_value.all_inner_texts.return_value = ["Broad Market Indices", "NIFTY 500"]

            mock_context = MagicMock()
            mock_context.new_page.return_value = mock_page
            mock_browser = MagicMock()
            mock_browser.new_context.return_value = mock_context
            mock_p = MagicMock()
            mock_p.chromium.launch.return_value = mock_browser
            mock_pw.return_value.__enter__.return_value = mock_p
            mock_pw.return_value.__exit__.return_value = None

            automated_df = fetch_benchmark_tri_automated("2024-09-23", "2024-09-25")

        assert list(manual_df.columns) == list(automated_df.columns)
        assert manual_df["Date"].dtype == automated_df["Date"].dtype
        assert manual_df["Total Returns Index"].dtype == automated_df["Total Returns Index"].dtype
        assert manual_df["Net Total Return Index"].dtype == automated_df["Net Total Return Index"].dtype
