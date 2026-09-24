"""
Unit tests for the fundamentals extraction and Screener.in parsing module.
Author: Antigravity / SAPM & Derivatives Coursework (IIM Bodh Gaya)
"""

import pytest
from bs4 import BeautifulSoup

from unittest.mock import patch

from fundamentals import (
    _clean_numeric,
    generate_fundamentals_screen_check,
    parse_debt_to_equity,
    parse_operating_cash_flow,
    parse_operating_cash_flow_3yr,
    parse_opm,
    parse_roce_3yr_average,
    parse_sales_and_profit_growth,
    parse_top_ratios,
)

SAMPLE_SCREENER_HTML = """
<!DOCTYPE html>
<html>
<body>
  <ul id="top-ratios">
    <li><span class="name">Market Cap</span> <span class="number">40,184</span> Cr.</li>
    <li><span class="name">Current Price</span> <span class="number">5,201</span></li>
    <li><span class="name">ROCE</span> <span class="number">15.1</span> %</li>
    <li><span class="name">ROE</span> <span class="number">15.6</span> %</li>
  </ul>

  <section id="profit-loss">
    <table>
      <tr>
        <td>Sales</td><td>10,000</td><td>12,000</td><td>14,000</td>
      </tr>
      <tr>
        <td>OPM %</td><td>18%</td><td>17%</td><td>16%</td>
      </tr>
    </table>
    <table class="ranges-table">
      <th>Compounded Sales Growth</th>
      <tr><td>10 Years:</td><td>10%</td></tr>
      <tr><td>3 Years:</td><td>12%</td></tr>
    </table>
    <table class="ranges-table">
      <th>Compounded Profit Growth</th>
      <tr><td>10 Years:</td><td>15%</td></tr>
      <tr><td>3 Years:</td><td>33%</td></tr>
    </table>
  </section>

  <section id="balance-sheet">
    <table>
      <tr>
        <td>Equity Capital</td><td>77</td><td>77</td><td>77</td>
      </tr>
      <tr>
        <td>Reserves</td><td>5,290</td><td>6,012</td><td>6,960</td>
      </tr>
      <tr>
        <td>Borrowings</td><td>5,552</td><td>6,028</td><td>6,183</td>
      </tr>
    </table>
  </section>

  <section id="cash-flow">
    <table>
      <tr>
        <td>Cash from Operating Activity +</td><td>1,377</td><td>1,959</td><td>1,873</td>
      </tr>
    </table>
  </section>

  <section id="ratios">
    <table>
      <tr>
        <td>ROCE %</td><td>10%</td><td>16%</td><td>14%</td><td>15%</td>
      </tr>
    </table>
  </section>
</body>
</html>
"""


class TestNumericCleaning:
    """Test suite for numeric value cleaning."""

    def test_clean_currency_and_percent(self):
        assert _clean_numeric("₹40,184 Cr.") == 40184.0
        assert _clean_numeric("15.1%") == 15.1
        assert _clean_numeric("1,873") == 1873.0
        assert _clean_numeric("-24%") == -24.0
        assert _clean_numeric("-") is None
        assert _clean_numeric(None) is None


class TestScreenerParsing:
    """Test suite for HTML table and ratio parsing."""

    @pytest.fixture
    def soup(self):
        return BeautifulSoup(SAMPLE_SCREENER_HTML, "html.parser")

    def test_top_ratios(self, soup):
        ratios = parse_top_ratios(soup)
        assert ratios["market_cap"] == 40184.0
        assert ratios["current_price"] == 5201.0
        assert ratios["roce"] == 15.1
        assert ratios["roe"] == 15.6

    def test_roce_3yr_average(self, soup):
        # Last 3 years are 16%, 14%, 15% -> Average is (16 + 14 + 15) / 3 = 15.00%
        roce_3y = parse_roce_3yr_average(soup)
        assert roce_3y == pytest.approx(15.0, rel=1e-2)

    def test_operating_cash_flow(self, soup):
        cfo = parse_operating_cash_flow(soup)
        assert cfo == 1873.0

    def test_operating_cash_flow_3yr(self, soup):
        # Screener's "Operating cash flow 3years": 1,377 + 1,959 + 1,873 = 5,209
        assert parse_operating_cash_flow_3yr(soup) == pytest.approx(5209.0)

    def test_debt_to_equity(self, soup):
        # Borrowings = 6183, Net Worth = 77 + 6960 = 7037 -> D/E = 6183 / 7037 = 0.878 -> 0.88
        de = parse_debt_to_equity(soup)
        assert de == pytest.approx(0.88, rel=1e-2)

    def test_opm(self, soup):
        opm = parse_opm(soup)
        assert opm == 16.0

    def test_sales_and_profit_growth(self, soup):
        sales_3y, profit_3y = parse_sales_and_profit_growth(soup)
        assert sales_3y == 12.0
        assert profit_3y == 33.0


class TestGracefulDegradation:
    """Verify that malformed or empty HTML degrades gracefully with 'Data Unavailable'."""

    def test_empty_soup(self):
        empty_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert parse_top_ratios(empty_soup)["market_cap"] is None
        assert parse_roce_3yr_average(empty_soup) is None
        assert parse_operating_cash_flow(empty_soup) is None
        assert parse_debt_to_equity(empty_soup) is None


PASSING_RECORD = {
    "status": "OK", "market_cap": 21808.0, "sales_growth_3yr": 25.0, "profit_growth_3yr": 68.0,
    "roce_3yr_avg": 29.33, "opm": 10.0, "operating_cash_flow_3yr": 1388.0, "debt_to_equity": 0.42,
}


class TestFundamentalsScreenCheck:
    """Capital Goods/EPC screen: every threshold must be strictly met; missing data fails."""

    def _run(self, record):
        with patch("fundamentals.extract_stock_fundamentals", return_value=record):
            return generate_fundamentals_screen_check(["CEMPRO"], output_csv_path=None).iloc[0]

    def test_all_criteria_met_passes(self):
        row = self._run(PASSING_RECORD)
        assert row["passes_screen"] and row["failed_criteria"] == ""
        assert row["sector"] == "Capital Goods/EPC"

    def test_boundary_values_fail_strict_thresholds(self):
        row = self._run({**PASSING_RECORD, "opm": 9.0, "debt_to_equity": 1.2})
        assert not row["passes_screen"]
        assert not row["pass_opm"] and not row["pass_debt_to_equity"]
        assert row["failed_criteria"] == "OPM > 9%; Debt to equity < 1.2"

    def test_missing_metric_fails_and_is_labelled(self):
        row = self._run({**PASSING_RECORD, "operating_cash_flow_3yr": None})
        assert not row["passes_screen"]
        assert row["failed_criteria"] == "Operating cash flow 3years > 0 (Rs Cr) (missing)"

    def test_writes_csv(self, tmp_path):
        out = tmp_path / "fundamentals_screen_check.csv"
        with patch("fundamentals.extract_stock_fundamentals", return_value=PASSING_RECORD):
            generate_fundamentals_screen_check(["CEMPRO", "SCHNEIDER"], output_csv_path=out)
        assert out.exists() and out.read_text().count("\n") == 3

