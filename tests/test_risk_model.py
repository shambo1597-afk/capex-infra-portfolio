"""Regression, hedge sizing and performance arithmetic (risk_model.py, performance.py)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from performance import tangency_portfolio, window_metrics, xirr
from risk_model import (choose_expiry, hedge_betas, hedge_scenarios, ols, pick_tail_put, portfolio_returns,
                        single_index_table)


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"m": rng.normal(0, 0.01, 500)})
    y = pd.Series(0.0002 + 1.5 * x["m"] + rng.normal(0, 0.001, 500))
    res = ols(y, x)
    assert res["coef"]["m"] == pytest.approx(1.5, abs=0.02)
    assert res["r2"] > 0.95 and res["adj_r2"] <= res["r2"]


def test_single_index_variance_split_adds_up():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2025-01-01", periods=300)
    m = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    fx = pd.DataFrame({"rf": 0.0002, "mkt_excess": m - 0.0002}, index=idx)
    returns = pd.DataFrame({"S": 1.2 * m + rng.normal(0, 0.015, 300)}, index=idx)
    table = single_index_table(returns, returns["S"].rename("PORTFOLIO"), fx).set_index("symbol")
    row = table.loc["S"]
    assert row["systematic_vol_pct"] ** 2 + row["unsystematic_vol_pct"] ** 2 == pytest.approx(
        row["total_vol_pct"] ** 2, rel=0.01)
    assert row["explained_risk_pct"] + row["unexplained_risk_pct"] == pytest.approx(100, abs=0.2)


def test_portfolio_returns_renormalise_over_trading_stocks():
    r = pd.DataFrame({"A": [0.01, 0.02], "B": [np.nan, 0.04]})
    port = portfolio_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    assert port.tolist() == pytest.approx([0.01, 0.03])  # day 1: B not listed yet, A carries it all


def test_hedge_betas_min_variance_ratio_is_the_regression_beta():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-01", periods=400)
    m = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    port = 0.9 * m + rng.normal(0, 0.005, 400)
    b = hedge_betas(port, pd.DataFrame({"nifty50_ret": m}))
    assert b["min_variance_hedge_ratio"] == pytest.approx(0.9, abs=0.05)
    assert b["tail_days"] == 40


def test_put_choice_and_scenarios():
    cands = pd.DataFrame({"strike": [21000.0, 22000.0, 22500.0], "premium": [74.0, 146.5, 217.7],
                          "open_interest": [2e6, 3e6, 1.7e6], "lots": 7, "strike_below_spot_pct": [9.2, 4.9, 2.8]})
    put = pick_tail_put(cands, otm_pct=5.0)
    assert put["strike"] == 22000.0
    scen = hedge_scenarios(1e7, 1.0, 23140.0, put, 65, moves_pct=(-20, 0, 10), down_beta=1.1).set_index("nifty50_move_pct")
    premium = 146.5 * 7 * 65
    assert scen.loc[0, "hedged_pnl_inr"] == pytest.approx(-premium, abs=1)
    assert scen.loc[10, "unhedged_pnl_inr"] == pytest.approx(1e6, abs=1)
    level = 23140.0 * 0.8
    assert scen.loc[-20, "put_payoff_inr"] == pytest.approx((22000 - level) * 7 * 65, abs=1)
    assert scen.loc[-20, "unhedged_pnl_inr"] == pytest.approx(-1e7 * 1.1 * 0.2, abs=1)


def test_choose_expiry_covers_the_window():
    fo = pd.DataFrame({"FinInstrmTp": ["IDO"] * 3, "XpryDt": ["2026-11-23", "2026-12-29", "2027-03-30"]})
    assert choose_expiry(fo, "IDO", "2026-12-28") == "2026-12-29"


def test_xirr_gain_and_loss():
    assert xirr([(date(2025, 1, 1), -100.0), (date(2026, 1, 1), 110.0)]) == pytest.approx(0.10, abs=1e-6)
    assert xirr([(date(2025, 1, 1), -100.0), (date(2026, 1, 1), 90.0)]) == pytest.approx(-0.10, abs=1e-6)
    # a quarter's +5% annualises by compounding: 1.05^(365/90) - 1
    q = xirr([(date(2026, 1, 1), -100.0), (date(2026, 4, 1), 105.0)])
    assert q == pytest.approx(1.05 ** (365 / 90) - 1, abs=1e-6)


def test_window_metrics_compounding_and_ratios():
    idx = pd.bdate_range("2026-01-01", periods=63)
    rng = np.random.default_rng(4)
    m = pd.Series(rng.normal(0.0005, 0.01, 63), index=idx)
    p = 1.2 * m + 0.001
    rf = pd.Series(0.0002, index=idx)
    out = window_metrics(p, m, rf, "q")
    growth = (1 + p).prod()
    assert out["portfolio_return_pct"] == pytest.approx((growth - 1) * 100, abs=0.01)
    assert out["portfolio_annualised_compound_pct"] == pytest.approx((growth ** (252 / 63) - 1) * 100, abs=0.01)
    assert out["compounding_effect_pp"] > 0
    assert out["beta_vs_nifty500"] == pytest.approx(1.2, abs=0.01)


def test_tangency_prefers_the_better_sharpe():
    mu = np.array([0.30, 0.10])
    cov = np.diag([0.04, 0.04])
    w = tangency_portfolio(mu, cov, rf=0.05)
    assert w[0] > w[1] and w.sum() == pytest.approx(1.0)


def test_capm_expected_return():
    from risk_model import capm_table
    single = pd.DataFrame({"symbol": ["A", "PORTFOLIO"], "beta": [1.5, 1.0]})
    out = capm_table(single, rf_annual=0.05, mrp_pct=7.0).set_index("symbol")
    assert out.loc["A", "capm_expected_return_pct"] == pytest.approx(15.5)
    assert out.loc["A", "capm_3m_return_pct"] == pytest.approx((1.155 ** 0.25 - 1) * 100, abs=0.01)
    assert out.loc["PORTFOLIO", "capm_expected_return_pct"] == pytest.approx(12.0)


def test_autocorrelation_detects_an_ar1_process_and_not_noise():
    from risk_model import autocorrelation_table
    rng = np.random.default_rng(5)
    n = 600
    e = rng.normal(0, 0.01, n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.4 * ar[t - 1] + e[t]
    idx = pd.bdate_range("2024-01-01", periods=n)
    returns = pd.DataFrame({"AR": ar, "NOISE": rng.normal(0, 0.01, n)}, index=idx)
    out = autocorrelation_table(returns, returns["NOISE"]).set_index("symbol")
    assert out.loc["AR", "ar1_phi"] == pytest.approx(0.4, abs=0.08) and out.loc["AR", "predictable_at_5pct"]
    assert not out.loc["NOISE", "predictable_at_5pct"]


def test_gmvp_minimises_variance_within_bounds():
    from performance import gmvp
    cov = np.diag([0.04, 0.09, 0.16])
    w = gmvp(cov)
    inv = 1 / np.diag(cov)
    assert np.allclose(w, inv / inv.sum(), atol=1e-4)  # uncorrelated: weights proportional to 1/variance
    wb = gmvp(cov, 0.2, 0.5)
    assert wb.min() >= 0.2 - 1e-9 and wb.max() <= 0.5 + 1e-9 and wb.sum() == pytest.approx(1.0)
    assert wb @ cov @ wb >= w @ cov @ w - 1e-12  # limits can only add variance


def test_risk_reward_ratio():
    from risk_model import risk_reward_table
    risk = pd.DataFrame({"symbol": ["A"], "current_price": [100.0], "stop_loss_price": [90.0],
                         "stop_loss_pct_below_current": [10.0], "annualized_volatility_pct": [40.0],
                         "atr_pct": [3.0], "invested_inr": [1e6]})
    tech = pd.DataFrame({"symbol": ["A"], "nearest_resistance": [102.0]})
    capm = pd.DataFrame({"symbol": ["A", "PORTFOLIO"], "capm_3m_return_pct": [3.0, 3.0]})
    out = risk_reward_table(risk, tech, capm, portfolio_vol_pct=20.0).set_index("symbol")
    assert out.loc["A", "upside_pct"] == pytest.approx(20.0)  # 40% x sqrt(63/252): resistance is not a cap
    assert out.loc["A", "reward_risk"] == pytest.approx(2.0)
    assert out.loc["A", "first_hurdle_above_pct"] == pytest.approx(2.0)
    assert out.loc["PORTFOLIO (all stocks at once)", "downside_inr"] == pytest.approx(1e5)
    assert out.loc["PORTFOLIO (diversified)", "reward_risk"] == pytest.approx(1.0)


def test_live_metrics_wait_for_enough_sessions_then_use_the_tracker(tmp_path, monkeypatch):
    import tracker
    from performance import live_metrics
    path = tmp_path / "tracker_daily.csv"
    monkeypatch.setattr(tracker, "TRACKER_DAILY_CSV", path)
    idx = pd.bdate_range("2026-09-28", periods=26)
    rng = np.random.default_rng(6)
    m = np.cumprod(np.r_[1.0, 1 + rng.normal(0.0005, 0.01, 25)]) * 1e7
    p = np.cumprod(np.r_[1.0, 1 + 1.1 * (m[1:] / m[:-1] - 1) + 0.0005]) * 1e7
    liquid = 1e7 * (1 + 0.0002) ** np.arange(26)
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "total_inr": p, "nifty500_inr": m,
                  "liquid_fund_inr": liquid}).head(15).to_csv(path, index=False)
    assert live_metrics() is None  # 14 returns: too few
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "total_inr": p, "nifty500_inr": m,
                  "liquid_fund_inr": liquid}).to_csv(path, index=False)
    out = live_metrics()
    assert out["window"].startswith("Live") and out["sessions"] == 25
    assert out["portfolio_return_pct"] == pytest.approx((p[-1] / 1e7 - 1) * 100, abs=0.01)
    assert out["beta_vs_nifty500"] == pytest.approx(1.1, abs=0.02)
