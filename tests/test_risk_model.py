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
