"""Equal-risk-contribution weights within the brief's bounds (weights.py)."""

import numpy as np
import pandas as pd
import pytest

from weights import (compute_portfolio_weights, equal_risk_contribution, project_to_bounded_simplex,
                     risk_contributions)


def test_projection_respects_sum_and_bounds():
    w = project_to_bounded_simplex(np.array([0.9, 0.05, 0.03, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.05, 0.15)
    assert w.sum() == pytest.approx(1.0)
    assert w.min() >= 0.05 - 1e-12 and w.max() <= 0.15 + 1e-12


def test_projection_rejects_infeasible_bounds():
    with pytest.raises(ValueError):
        project_to_bounded_simplex(np.ones(3) / 3, 0.05, 0.15)  # 3 x 15% < 100%


def test_uncorrelated_erc_is_inverse_volatility():
    vols = np.array([0.2, 0.3, 0.4, 0.25, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3])
    cov = np.diag(vols ** 2)
    w = equal_risk_contribution(cov, 0.0, 1.0)
    expected = (1 / vols) / (1 / vols).sum()
    assert np.allclose(w, expected, atol=1e-6)
    assert np.allclose(risk_contributions(w, cov), 0.1, atol=1e-6)


def test_bounds_bind_on_a_very_calm_stock():
    vols = np.array([0.02] + [0.4] * 9)  # the calm stock would take far more than 15% unbounded
    w = equal_risk_contribution(np.diag(vols ** 2), 0.05, 0.15)
    assert w[0] == pytest.approx(0.15)
    assert w.sum() == pytest.approx(1.0) and w.min() >= 0.05 - 1e-12


def _bhav(symbol, closes):
    dates = pd.bdate_range("2025-01-01", periods=len(closes))
    return pd.DataFrame({"SYMBOL": symbol, "SERIES": "EQ", "DATE1": dates.strftime("%Y-%m-%d"), "CLOSE_PRICE": closes})


def test_compute_portfolio_weights_from_prices():
    rng = np.random.default_rng(0)
    frames = []
    for i in range(10):
        vol = 0.01 * (1 + i)  # daily volatility 1% .. 10%
        frames.append(_bhav(f"S{i}", 100 * np.cumprod(1 + rng.normal(0, vol, 200))))
    out = compute_portfolio_weights(pd.concat(frames, ignore_index=True), [f"S{i}" for i in range(10)])
    assert out["weight_pct"].sum() == pytest.approx(100, abs=0.05)
    assert out["weight_pct"].between(5, 15).all()
    assert out["weight_pct"].iloc[0] > out["weight_pct"].iloc[-1]  # calmer stock, bigger weight
