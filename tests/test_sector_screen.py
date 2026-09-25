"""
Unit tests for the technical-first sector screen and the fundamentals parsers it relies on.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from config import CAPITAL_GOODS_SCREEN_CRITERIA, CEMENT_SCREEN_CRITERIA, POWER_SCREEN_CRITERIA, SECTOR_SCREENS
from fundamentals import _statements_are_stale, parse_interest_coverage, parse_opm
from sector_screen import load_constituents, screen_sector, technical_screen_result

PL_HTML = """
<section id="profit-loss"><table>
  <tr><th></th><th>Mar 2025</th><th>Mar 2026</th><th>TTM</th></tr>
  <tr><td>Sales +</td><td>20,000</td><td>22,902</td><td>24,389</td></tr>
  <tr><td>Operating Profit</td><td>1,700</td><td>1,922</td><td>2,171</td></tr>
  <tr><td>OPM %</td><td>9%</td><td>8%</td><td>9%</td></tr>
  <tr><td>Interest</td><td>400</td><td>500</td><td>520</td></tr>
  <tr><td>Profit before tax</td><td>900</td><td>1,000</td><td>1,100</td></tr>
</table></section>
"""


class TestTechnicalScreenRule:
    """Passes only when RS vs Nifty 500 > +2 pp AND trend is Bullish; ADX is not a cutoff."""

    @pytest.mark.parametrize("rs, trend, passed, reason", [
        (5.0, "Bullish (Uptrend)", True, ""),
        (2.01, "Bullish (Uptrend)", True, ""),
        (2.0, "Bullish (Uptrend)", False, "RS <= +2 pp"),     # margin is strict
        (0.01, "Bullish (Uptrend)", False, "RS <= +2 pp"),    # passed under the old bare-zero rule
        (0.0, "Bullish (Uptrend)", False, "RS <= +2 pp"),
        (-3.0, "Bullish (Uptrend)", False, "RS <= +2 pp"),
        (5.0, "Bearish (Downtrend)", False, "trend Bearish"),
        (-3.0, "Bearish (Downtrend)", False, "RS <= +2 pp; trend Bearish"),
        (float("nan"), "N/A (Insufficient Data)", False, "RS unavailable; trend N/A"),
    ])
    def test_rule(self, rs, trend, passed, reason):
        assert technical_screen_result(rs, trend) == (passed, reason)

    def test_margin_comes_from_config(self):
        from config import TECHNICAL_RS_MARGIN_PP
        assert TECHNICAL_RS_MARGIN_PP == 2.0
        assert technical_screen_result(1.0, "Bullish (Uptrend)", rs_margin_pp=0.5) == (True, "")


class TestScreenSector:
    TECH = {
        "AAA": {"rs_score_vs_nifty500": 10.0, "trend_direction": "Bullish (Uptrend)", "latest_adx": 30.0},
        "BBB": {"rs_score_vs_nifty500": 8.0, "trend_direction": "Bullish (Uptrend)", "latest_adx": 12.0},
        "CCC": {"rs_score_vs_nifty500": -4.0, "trend_direction": "Bullish (Uptrend)", "latest_adx": 25.0},
    }
    CRITERIA = [("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"), ("opm", ">", 9, "OPM > 9%")]

    def _run(self, fundamentals_records):
        def fake_tech(symbol, stock_df, benchmark_df, rs_lookback):
            base = {"current_price": 100.0, "latest_rsi": 55.0, "plus_di": 25.0, "minus_di": 20.0}
            return {**base, **self.TECH[symbol]}

        members = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Company Name": ["A Ltd", "B Ltd", "C Ltd"]})
        with patch("sector_screen.evaluate_stock_technicals", side_effect=fake_tech), \
                patch("sector_screen.get_fundamentals_summary",
                      return_value=pd.DataFrame(fundamentals_records)) as fundamentals:
            screen = screen_sector("Test", members, pd.DataFrame(), pd.DataFrame(), self.CRITERIA)
        return screen.set_index("symbol"), fundamentals

    def test_fundamentals_fetched_live_only_for_technical_passers(self):
        _, fundamentals = self._run([])
        fundamentals.assert_called_once_with(["AAA", "BBB"], use_cache=False)

    def test_both_screens_and_blank_fundamentals_for_technical_failures(self):
        screen, _ = self._run([
            {"symbol": "AAA", "status": "OK", "market_cap": 5000.0, "opm": 12.0},
            {"symbol": "BBB", "status": "OK", "market_cap": 5000.0, "opm": 9.0},
        ])
        assert bool(screen.loc["AAA", "passes_both_screens"]) is True
        assert bool(screen.loc["BBB", "passed_technical_screen"]) is True  # low ADX is not a cutoff
        assert bool(screen.loc["BBB", "passed_fundamental_screen"]) is False
        assert screen.loc["BBB", "failed_criteria"] == "OPM > 9%"
        assert bool(screen.loc["CCC", "passed_technical_screen"]) is False
        assert pd.isna(screen.loc["CCC", "passed_fundamental_screen"])  # never screened
        assert pd.isna(screen.loc["CCC", "opm"])
        assert bool(screen.loc["CCC", "passes_both_screens"]) is False

    def test_missing_fundamentals_record_fails(self):
        screen, _ = self._run([{"symbol": "AAA", "status": "OK", "market_cap": 5000.0, "opm": 12.0}])
        assert bool(screen.loc["BBB", "passed_fundamental_screen"]) is False
        assert "(missing)" in screen.loc["BBB", "failed_criteria"]


class TestSectorScreenConfig:
    def test_criteria_match_the_specified_thresholds(self):
        """Lightened, current-year-only safety screens (no 3-year averages or growth)."""
        as_tuples = lambda crit: [(f, op, t) for f, op, t, _ in crit]  # noqa: E731
        assert as_tuples(CEMENT_SCREEN_CRITERIA) == [
            ("market_cap", ">", 1000), ("roce", ">", 8), ("opm", ">", 10), ("operating_cash_flow", ">", 0),
            ("debt_to_equity", "<", 1.5), ("pledged_pct", "<", 15)]
        assert as_tuples(CAPITAL_GOODS_SCREEN_CRITERIA) == [
            ("market_cap", ">", 1000), ("roce", ">", 8), ("opm", ">", 8), ("operating_cash_flow", ">", 0),
            ("debt_to_equity", "<", 1.5), ("pledged_pct", "<", 15)]
        assert as_tuples(POWER_SCREEN_CRITERIA) == [
            ("market_cap", ">", 2000), ("roce", ">", 6), ("interest_coverage", ">", 1.5),
            ("operating_cash_flow", ">", 0), ("pledged_pct", "<", 15)]
        for crit in (CEMENT_SCREEN_CRITERIA, CAPITAL_GOODS_SCREEN_CRITERIA, POWER_SCREEN_CRITERIA):
            assert not {f for f, *_ in crit} & {"roce_3yr_avg", "sales_growth_3yr", "profit_growth_3yr", "operating_cash_flow_3yr"}

    @pytest.mark.parametrize("sector, count", [("Cement", 16), ("Capital Goods", 50), ("Power", 21)])
    def test_official_constituent_files(self, sector, count):
        members = load_constituents(SECTOR_SCREENS[sector]["constituents_csv"])
        assert len(members) == count
        assert not members["Symbol"].duplicated().any()


class TestFundamentalsParsers:
    @pytest.fixture
    def soup(self):
        return BeautifulSoup(PL_HTML, "html.parser")

    def test_interest_coverage_uses_latest_full_year_not_ttm(self, soup):
        # Mar 2026: (PBT 1,000 + interest 500) / 500 = 3.0
        assert parse_interest_coverage(soup) == pytest.approx(3.0)

    def test_interest_coverage_zero_interest_is_infinite(self):
        html = PL_HTML.replace("<td>500</td>", "<td>0</td>")
        assert parse_interest_coverage(BeautifulSoup(html, "html.parser")) == float("inf")

    def test_opm_is_exact_not_screener_rounded(self, soup):
        # TTM: 2,171 / 24,389 = 8.90%, displayed by Screener as "9%"
        assert parse_opm(soup) == pytest.approx(8.90, abs=0.01)

    def test_stale_consolidated_statements_detected(self):
        stale = '<section id="profit-loss"><table><tr><th></th><th>Dec 2010</th></tr></table></section>'
        assert _statements_are_stale(BeautifulSoup(stale, "html.parser"))
        assert not _statements_are_stale(BeautifulSoup(PL_HTML, "html.parser"))


class TestReviewTable:
    """Unfiltered review: every constituent appears exactly once, however many criteria it fails."""

    def test_no_row_is_excluded_and_sorting(self):
        from sector_screen import build_review_table
        tech = {
            "AAA": (5.0, "Bullish (Uptrend)"), "BBB": (-2.0, "Bearish (Downtrend)"),
            "CCC": (9.0, "Bearish (Downtrend)"), "DDD": (float("nan"), "N/A (Insufficient Data)"),
        }

        def fake_tech(symbol, stock_df, benchmark_df, rs_lookback):
            rs, trend = tech[symbol]
            return {"current_price": 100.0, "latest_rsi": 50.0, "latest_adx": 20.0, "plus_di": 1.0, "minus_di": 1.0,
                    "trend_direction": trend, "rs_score_vs_nifty500": rs,
                    "nearest_support": 95.0, "nearest_resistance": 105.0}

        criteria = [("market_cap", ">", 1000, "Market Cap > 1000 (Rs Cr)"), ("opm", ">", 13, "OPM > 13%")]
        fundamentals = pd.DataFrame([
            {"symbol": "AAA", "status": "OK", "market_cap": 5000.0, "opm": 10.0},
            {"symbol": "BBB", "status": "OK", "market_cap": 5000.0, "opm": 20.0},
            {"symbol": "CCC", "status": "OK", "market_cap": 5000.0, "opm": 20.0},
            # DDD: no fundamentals record at all -> still gets a row, with 0 criteria passed
        ])
        members = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC", "DDD"], "Company Name": list("ABCD")})
        # 63-day stock returns (%) in constituent order: AAA 10, BBB 2, CCC -4, DDD unavailable.
        # Sector average of the valid three = 8/3 = 2.667 -> spreads AAA +7.33, BBB -0.67, CCC -6.67
        returns = [(0.0, 10.0, 0.0), (0.0, 2.0, 0.0), (0.0, -4.0, 0.0), (float("nan"),) * 3]
        with patch("sector_screen.evaluate_stock_technicals", side_effect=fake_tech), \
                patch("sector_screen.compute_relative_strength", side_effect=returns), \
                patch("sector_screen.get_fundamentals_summary", return_value=fundamentals) as live:
            table = build_review_table(members, pd.DataFrame(), pd.DataFrame(), criteria)

        live.assert_called_once_with(["AAA", "BBB", "CCC", "DDD"], use_cache=False)
        # fundamentals_passed_count desc, then sector_rank asc: BBB (rank 2) beats CCC (rank 3)
        # even though CCC's RS vs Nifty 500 is higher
        assert table["symbol"].tolist() == ["BBB", "CCC", "AAA", "DDD"]
        by_sym = table.set_index("symbol")
        assert by_sym.loc["AAA", "rs_score_vs_sector_avg"] == pytest.approx(7.33)
        assert by_sym.loc["BBB", "rs_score_vs_sector_avg"] == pytest.approx(-0.67)
        assert by_sym.loc["CCC", "rs_score_vs_sector_avg"] == pytest.approx(-6.67)
        assert pd.isna(by_sym.loc["DDD", "rs_score_vs_sector_avg"]) and pd.isna(by_sym.loc["DDD", "sector_rank"])
        assert by_sym["sector_rank"].dropna().astype(int).to_dict() == {"AAA": 1, "BBB": 2, "CCC": 3}
        assert table.set_index("symbol")["fundamentals_passed_count"].to_dict() == {"CCC": 2, "BBB": 2, "AAA": 1, "DDD": 0}
        assert table.set_index("symbol")["technically_attractive"].to_dict() == {"CCC": False, "BBB": False, "AAA": True, "DDD": False}


class TestSectorRelativeStrength:
    """RS vs the sector's own equal-weighted average return (indicators.compute_sector_relative_strength)."""

    def test_sector_average_and_spreads(self):
        from indicators import compute_sector_relative_strength
        # Known 63-day returns (%): mean = (10 - 5 + 4 + 7) / 4 = 4.0
        spreads, sector_avg = compute_sector_relative_strength({"A": 10.0, "B": -5.0, "C": 4.0, "D": 7.0})

        assert sector_avg == pytest.approx(4.0)
        assert spreads == pytest.approx({"A": 6.0, "B": -9.0, "C": 0.0, "D": 3.0})
        assert sum(spreads.values()) == pytest.approx(0.0, abs=1e-9)  # spreads from the own average

    def test_spreads_sum_to_zero_for_arbitrary_returns(self):
        from indicators import compute_sector_relative_strength
        rng = np.random.default_rng(7)
        returns = {f"S{i}": float(r) for i, r in enumerate(rng.normal(0, 12, 16))}
        spreads, sector_avg = compute_sector_relative_strength(returns)
        assert sector_avg == pytest.approx(np.mean(list(returns.values())))
        assert sum(spreads.values()) == pytest.approx(0.0, abs=1e-9)

    def test_missing_return_excluded_from_average(self):
        from indicators import compute_sector_relative_strength
        spreads, sector_avg = compute_sector_relative_strength({"A": 6.0, "B": 2.0, "C": float("nan")})
        assert sector_avg == pytest.approx(4.0)
        assert spreads["A"] == pytest.approx(2.0) and spreads["B"] == pytest.approx(-2.0)
        assert np.isnan(spreads["C"])



def _screener_page(sales_3y="16%", years=("Mar 2024", "Mar 2025", "Mar 2026")):
    ths = "".join(f"<th>{y}</th>" for y in years)
    return f"""<html><body>
      <ul id="top-ratios"><li><span class="name">Market Cap</span><span class="number">40,000</span></li></ul>
      <section id="profit-loss"><table><tr><th></th>{ths}</tr></table>
        <table class="ranges-table"><tr><th>Compounded Sales Growth</th></tr>
          <tr><td>3 Years:</td><td>{sales_3y}</td></tr></table>
      </section>{"x" * 5000}</body></html>"""


class _Resp:
    def __init__(self, status_code, text="", headers=None):
        self.status_code, self.text, self.headers = status_code, text, headers or {}


class TestScreenerFetchResilience:
    def test_rate_limited_request_is_retried(self):
        from fundamentals import _get_with_retry
        session = type("S", (), {})()
        replies = [_Resp(429, headers={"Retry-After": "1"}), _Resp(429), _Resp(200, "ok")]
        session.get = lambda url, timeout: replies.pop(0)
        with patch("fundamentals.time.sleep") as sleep:
            resp = _get_with_retry(session, "https://www.screener.in/company/SJVN/")
        assert resp.status_code == 200
        assert [c.args[0] for c in sleep.call_args_list] == [1.0, 10.0]  # Retry-After, then backoff x2

    def test_consolidated_without_3yr_history_falls_back_to_standalone(self, tmp_path):
        from fundamentals import fetch_screener_page, parse_sales_and_profit_growth
        pages = {
            "https://www.screener.in/company/ABB/consolidated/": _Resp(200, _screener_page(sales_3y="%")),
            "https://www.screener.in/company/ABB/": _Resp(200, _screener_page(sales_3y="16%")),
        }
        with patch("fundamentals.requests.Session") as session_cls:
            session_cls.return_value.get.side_effect = lambda url, timeout: pages[url]
            html = fetch_screener_page("ABB", cache_dir=tmp_path, use_cache=False)
        assert parse_sales_and_profit_growth(BeautifulSoup(html, "html.parser"))[0] == 16.0

    def test_consolidated_with_history_is_kept(self, tmp_path):
        from fundamentals import fetch_screener_page
        pages = {"https://www.screener.in/company/ULTRACEMCO/consolidated/": _Resp(200, _screener_page(sales_3y="12%"))}
        with patch("fundamentals.requests.Session") as session_cls:
            session_cls.return_value.get.side_effect = lambda url, timeout: pages[url]
            html = fetch_screener_page("ULTRACEMCO", cache_dir=tmp_path, use_cache=False)
        assert "12%" in html


def test_presigned_s3_credentials_are_redacted_before_caching():
    from fundamentals import sanitize_screener_html
    html = ('<a href="https://x.s3.amazonaws.com/call.mp3?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;'
            'X-Amz-Credential=AKIAABCDEFGHIJKLMNOP%2F20240528%2Fap-south-1%2Fs3%2Faws4_request&amp;'
            'X-Amz-Signature=deadbeef0123">Concall</a><td>OPM %</td>')
    clean = sanitize_screener_html(html)
    assert "AKIA" not in clean and "deadbeef" not in clean
    assert "X-Amz-Credential=REDACTED" in clean and "X-Amz-Signature=REDACTED" in clean
    assert "<td>OPM %</td>" in clean  # page content otherwise untouched


class TestPromoterPledge:
    """Pledged percentage from NSE disclosures (percPromoterShares = % of promoter holding pledged)."""

    def test_latest_shareholding_record_wins(self):
        from fundamentals import parse_pledge_records
        payload = {"data": [
            {"shp": "31-Mar-2026", "percPromoterShares": "    50.00"},
            {"shp": "30-Jun-2026", "percPromoterShares": "    44.74"},
        ]}
        assert parse_pledge_records(payload) == (44.74, "30-Jun-2026")

    def test_no_disclosed_pledge_is_zero(self):
        from fundamentals import parse_pledge_records
        assert parse_pledge_records({"data": []}) == (0.0, None)

    def test_unreachable_nse_is_missing_not_zero(self, tmp_path):
        from fundamentals import fetch_pledged_percentage
        blocked = MagicMock(status_code=403, headers={"content-type": "text/html"})
        with patch("fundamentals._get_nse_pledge_session") as session, patch("fundamentals.time.sleep"):
            session.return_value.get.return_value = blocked
            assert fetch_pledged_percentage("KPIGREEN", cache_dir=tmp_path, use_cache=False) == (None, None)
        assert not list(tmp_path.iterdir())  # nothing cached for a failed fetch

    def test_fetch_encodes_symbol_and_caches(self, tmp_path):
        from fundamentals import fetch_pledged_percentage
        ok = MagicMock(status_code=200, headers={"content-type": "application/json; charset=utf-8"})
        ok.json.return_value = {"data": [{"shp": "30-Jun-2026", "percPromoterShares": "59.96"}]}
        with patch("fundamentals._get_nse_pledge_session") as session:
            session.return_value.get.return_value = ok
            assert fetch_pledged_percentage("GMRP&UI", cache_dir=tmp_path, use_cache=False) == (59.96, "30-Jun-2026")
            assert "symbol=GMRP%26UI" in session.return_value.get.call_args.args[0]
        with patch("fundamentals._get_nse_pledge_session") as session:  # served from cache, no request
            assert fetch_pledged_percentage("GMRP&UI", cache_dir=tmp_path) == (59.96, "30-Jun-2026")
            session.return_value.get.assert_not_called()



@pytest.mark.parametrize("sector", ["Cement", "Capital Goods", "Power"])
def test_committed_review_table_sector_rs_properties(sector):
    """
    The committed review tables satisfy the within-sector RS invariants for every sector: each
    stock is compared only with its own official index's constituents, the spreads from the
    equal-weighted sector average sum to ~0, and (sharing the same stock return) the gap between
    rs_score_vs_sector_avg and rs_score_vs_nifty500 is the same constant for every stock.
    """
    from sector_screen import review_table_path
    table = pd.read_csv(review_table_path(sector))
    members = load_constituents(SECTOR_SCREENS[sector]["constituents_csv"])
    n = len(members)

    assert sorted(table["symbol"]) == sorted(members["Symbol"])  # own universe only, each once
    # Values are stored to 2 decimals, so each carries up to 0.005 rounding error
    assert abs(table["rs_score_vs_sector_avg"].sum()) <= n * 0.005 + 1e-9
    gap = table["rs_score_vs_sector_avg"] - table["rs_score_vs_nifty500"]
    assert gap.max() - gap.min() <= 0.02 + 1e-9
    assert sorted(table["sector_rank"]) == list(range(1, n + 1))
    best_first = table.sort_values("rs_score_vs_sector_avg", ascending=False)["sector_rank"].tolist()
    assert best_first == list(range(1, n + 1))
    expected_order = table.sort_values(["fundamentals_passed_count", "sector_rank"], ascending=[False, True])
    assert table["symbol"].tolist() == expected_order["symbol"].tolist()


class TestTenthCandidateSweepPieces:
    @staticmethod
    def _series(closes, start="2026-06-01"):
        dates = pd.bdate_range(start, periods=len(closes))
        stock = pd.DataFrame({"DATE1": dates, "CLOSE_PRICE": closes})
        bench = pd.DataFrame({"Date": dates, "Close": [100.0] * len(closes)})  # flat benchmark
        return stock, bench

    def test_recent_contribution_for_a_late_burst(self):
        from indicators import compute_recent_rs_contribution
        # Flat at 100 for 54 sessions, then +1% a day for 10 sessions: all outperformance is recent
        closes = [100.0] * 54 + [100.0 * 1.01 ** i for i in range(1, 11)]
        rs10, rs63, pct = compute_recent_rs_contribution(*self._series(closes))
        assert rs10 == pytest.approx(rs63, abs=0.01) and pct == pytest.approx(100.0, abs=0.5)

    def test_recent_contribution_for_steady_trend(self):
        from indicators import compute_recent_rs_contribution
        closes = [100.0 * 1.002 ** i for i in range(64)]  # steady outperformance
        _, _, pct = compute_recent_rs_contribution(*self._series(closes))
        assert 12 < pct < 20  # ~10/63 of the window's edge

    def test_recent_contribution_undefined_without_outperformance(self):
        from indicators import compute_recent_rs_contribution
        closes = [100.0 * 0.999 ** i for i in range(64)]
        assert np.isnan(compute_recent_rs_contribution(*self._series(closes))[2])

    def test_results_meetings_parsed_without_guessing(self):
        from datetime import date
        from fundamentals import parse_results_meetings
        records = [
            {"bm_date": "19-Oct-2026", "bm_purpose": "Board Meeting Intimation", "bm_desc": "approve the Unaudited Financial results"},
            {"bm_date": "30-Sep-2026", "bm_purpose": "Fund Raising", "bm_desc": "raise funds via NCDs"},
            {"bm_date": "18-Oct-2025", "bm_purpose": "Financial Results", "bm_desc": "results for Sep 30, 2025"},
            {"bm_date": "18-Jul-2026", "bm_purpose": "Financial Results", "bm_desc": "results for Jun 30, 2026"},
        ]
        info = parse_results_meetings(records, date(2026, 9, 25))
        assert info["next_results_date"] == "2026-10-19"
        assert info["prior_year_sep_qtr_results_date"] == "2025-10-18"
        assert parse_results_meetings(records[1:], date(2026, 9, 25))["next_results_date"] is None


def test_committed_tenth_candidate_sweep():
    from config import OUTPUT_DIR
    from sector_screen import CURRENT_PICKS, RECENT_SPIKE_THRESHOLD_PCT
    sweep = pd.read_csv(OUTPUT_DIR / "tenth_candidate_sweep.csv")
    universe = set()
    for cfg in SECTOR_SCREENS.values():
        universe |= set(load_constituents(cfg["constituents_csv"])["Symbol"])
    assert set(sweep["symbol"]) == universe - set(CURRENT_PICKS) and not sweep["symbol"].duplicated().any()
    spike = sweep["recent_10day_contribution_pct"].fillna(-1e9) > RECENT_SPIKE_THRESHOLD_PCT
    assert (sweep["recent_spike_flag"] == spike).all()
    expected = sweep["fundamentals_clean"] & sweep["technically_attractive"] & ~sweep["recent_spike_flag"]
    assert (sweep["clean_candidate"] == expected).all()
    assert set(sweep["results_date_status"]) <= {"announced", "not announced", "unavailable"}
