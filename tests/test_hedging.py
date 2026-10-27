"""
Tests for the professor's 14-Oct checklist: option pricing, VaR and marginal VaR, the hedging-strategy
comparison, the 91-day T-bill rate and the three timelines.
"""

from datetime import date
from math import exp

import numpy as np
import pandas as pd
import pytest

from option_pricing import bs_delta, bs_price, implied_vol


def test_put_call_parity_and_implied_vol_round_trip():
    s, k, t, r, v = 22900.0, 22500.0, 15 / 365, 0.0573, 0.137
    put, call = bs_price(s, k, t, r, v, "PE"), bs_price(s, k, t, r, v, "CE")
    assert call - put == pytest.approx(s - k * exp(-r * t), abs=1e-6)
    assert implied_vol(put, s, k, t, r) == pytest.approx(v, abs=1e-6)
    assert -1 < bs_delta(s, k, t, r, v) < 0
    assert np.isnan(implied_vol(0.0, s, k, t, r))  # below intrinsic: no volatility fits


def test_option_values_at_expiry_are_intrinsic():
    assert list(bs_price(np.array([21000.0, 23000.0]), 22000, 0.0, 0.05, 0.2)) == [1000.0, 0.0]


def test_var_components_add_up_and_scale_with_the_horizon():
    from var_model import var_tables
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.02, (300, 3)), columns=["A", "B", "C"])
    pos = pd.Series({"A": 3e6, "B": 2e6, "C": 1e6})
    out = var_tables(pos, returns, horizon_sessions=4)
    s = out["summary"].set_index("measure")["inr"]
    one = s["Parametric VaR 95%, 1 day"]
    assert out["components"]["component_var95_inr"].sum() == pytest.approx(one, rel=1e-3)
    assert s["Parametric VaR 95%, to the horizon (4 sessions)"] == pytest.approx(one * 2, rel=1e-3)
    assert out["components"]["share_of_var_pct"].sum() == pytest.approx(100, abs=0.2)


def test_sessions_to_the_horizon_skip_holidays():
    from var_model import sessions_to_horizon
    # 19-Oct (Mon) to 29-Oct (Thu): 20-Oct is Dussehra, so 21, 22, 23, 26, 27, 28, 29 = 7 sessions
    assert sessions_to_horizon(date(2026, 10, 19), "2026-10-29") == 7
    assert sessions_to_horizon(date(2026, 10, 30), "2026-10-29") == 1


def _chain(spot=22900.0):
    rows = []
    for k, p, oi in [(22900, 232.9, 5e5), (22450, 86.95, 3e5), (22200, 49.85, 2.4e5), (21750, 20.55, 3e5)]:
        rows.append({"FinInstrmTp": "IDO", "XpryDt": "2026-11-03", "StrkPric": float(k), "OptnTp": "PE",
                     "SttlmPric": p, "OpnIntrst": oi, "UndrlygPric": spot})
    rows.append({"FinInstrmTp": "IDO", "XpryDt": "2026-11-03", "StrkPric": 23350.0, "OptnTp": "CE",
                 "SttlmPric": 87.8, "OpnIntrst": 4e5, "UndrlygPric": spot})
    return pd.DataFrame(rows)


def test_bear_put_spread_payoff_is_capped_and_cheaper_than_the_put():
    import hedge_strategies as hs
    t, r = 15 / 365, 0.0573
    c = hs.chain(_chain(), "2026-11-03", 22900.0, t, r)
    strategies = hs.build_strategies(c, 22900.0, None, None)
    names = [s["name"] for s in strategies]
    assert names[0].startswith("No hedge") and any("bear put spread" in n for n in names) and any("collar" in n for n in names)
    curve = hs.payoff_curve(strategies, 585, 22900.0, 1e7, 1.0, 1.3, pd.Timestamp("2026-11-03"), r)
    spread = next(n for n in names if "bear put spread 22900/" in n)
    low = float(spread.rsplit("/", 1)[1])
    prices = {22900: 232.9, 22450: 86.95, 22200: 49.85, 21750: 20.55}
    put = next(n for n in names if n.endswith("protective put 22900"))
    hedge_gain = curve[spread] - curve[names[0]]
    # At expiry the spread pays at most the strike gap less its premium; a protective put keeps paying
    assert hedge_gain.max() == pytest.approx((22900 - low - (232.9 - prices[low])) * 585, abs=1)  # curve is rounded
    assert (curve[put] - curve[names[0]]).iloc[0] > hedge_gain.iloc[0]


def test_recommendation_is_the_cheapest_cover_within_the_budget():
    import hedge_strategies as hs
    t = pd.DataFrame({"strategy": ["No hedge", "A", "B", "C"], "expected_hedge_cost_inr": [0, 2000, 9000, 40000],
                      "cvar95_saved_inr": [0, 110000, 130000, 300000],
                      "cvar_saved_per_rupee_cost": [np.nan, 55.0, 14.4, 7.5]})
    out = hs.recommend(t, value=1e7)
    assert out.loc[out["recommended"], "strategy"].tolist() == ["A"]  # C is over the 0.3% budget
    weak = t.assign(cvar95_saved_inr=[0, 50000, 130000, 300000])
    assert hs.recommend(weak, value=1e7).loc[lambda d: d["recommended"], "strategy"].tolist() == ["B"]  # A cuts < 1%


def test_parse_rbi_auction_reads_the_91_day_yield():
    from risk_free import parse_auction
    page = ("<p>Date : Oct 14, 2026</p><table><tr><td>Cut-off price (₹) / Yield</td><td>98.5920</td>"
            "<td>96.9592</td><td>(YTM: 5.7281%)</td><td>(YTM: 6.2896%)</td></tr></table>")
    assert parse_auction(page) == (date(2026, 10, 14), 5.7281)
    assert parse_auction("<p>Date : Oct 21, 2026</p> Cut-off price (₹) / Yield - - -") is None


def test_fundamental_score_prefers_quality_and_treats_missing_as_middle():
    from timelines import fundamental_score
    eligible = pd.DataFrame({"symbol": ["GOOD", "BAD", "GAP"], "qtr_profit_yoy": [80, 5, 40],
                             "qtr_sales_yoy": [40, 2, 20]})
    review = pd.DataFrame({"roce": [30, 5, np.nan], "opm": [25, 4, np.nan], "debt_to_equity": [0.0, 1.2, np.nan]},
                          index=["GOOD", "BAD", "GAP"])
    assert list(fundamental_score(eligible, review).index) == ["GOOD", "GAP", "BAD"]


def test_calendar_spread_keeps_the_far_put_and_sells_the_near_one():
    import hedge_strategies as hs
    t, r = 15 / 365, 0.0573
    c = hs.chain(_chain(), "2026-11-03", 22900.0, t, r)
    held_leg = {"kind": "PE", "strike": 22200.0, "price": 230.0, "iv": 0.17, "qty": 1, "expiry": "2026-12-29"}
    held = {"instrument": "NIFTY 2026-12-29 22200 PE"}
    cal = [s for s in hs.build_strategies(c, 22900.0, held, held_leg) if s.get("calendar")]
    assert len(cal) == 1
    far, near = cal[0]["legs"]
    assert far is held_leg and near["qty"] == -1 and near["strike"] == 22200 and near["expiry"] == "2026-11-03"


def test_greeks_match_finite_differences():
    from option_pricing import bs_greeks
    s, k, t, r, v = 22600.0, 22000.0, 67 / 365, 0.0573, 0.165
    g = bs_greeks(s, k, t, r, v)
    h = 1.0
    assert g["delta"] == pytest.approx((bs_price(s + h, k, t, r, v) - bs_price(s - h, k, t, r, v)) / (2 * h), abs=1e-4)
    assert g["gamma"] == pytest.approx((bs_delta(s + h, k, t, r, v) - bs_delta(s - h, k, t, r, v)) / (2 * h), rel=1e-3)
    assert g["vega"] == pytest.approx(bs_price(s, k, t, r, v + 0.01) - bs_price(s, k, t, r, v), rel=0.02)
    assert g["theta"] == pytest.approx(bs_price(s, k, t - 1 / 365, r, v) - bs_price(s, k, t, r, v), rel=0.02)
    assert g["rho"] == pytest.approx(bs_price(s, k, t, r + 0.01, v) - bs_price(s, k, t, r, v), rel=0.05)


def test_sensitivity_puts_offset_falls_more_than_their_delta():
    from greeks import sensitivity
    sens = sensitivity(9.1e6, 520, 22000.0, 67 / 365, 61 / 365, 0.0573, 0.165, 22600.0).set_index("nifty_move_pct")
    assert sens.loc[0, "total_now_inr"] == 0 and sens.loc[0, "puts_pnl_at_exit_inr"] < 0  # time decay
    assert sens.loc[-10, "puts_pnl_now_inr"] > 0 > sens.loc[10, "puts_pnl_now_inr"]
    # convexity: the gain in a 10% fall exceeds 10x the gain in a 1% fall
    assert sens.loc[-10, "puts_pnl_now_inr"] > 10 * sens.loc[-1, "puts_pnl_now_inr"]
    assert sens.loc[-5, "puts_pnl_iv_up5_inr"] > sens.loc[-5, "puts_pnl_now_inr"]


def test_attribution_parts_add_up_and_follow_beta():
    from attribution import split_pnl
    idx = pd.date_range("2026-09-28", periods=4, freq="B")
    closes = pd.DataFrame({"A": [100.0, 98.0, 99.0, 95.0], "B": [50.0, 51.0, 50.0, 52.0]}, index=idx)
    shares = pd.DataFrame({"A": [10, 10, 10, 0], "B": [20, 20, 20, 20]}, index=idx, dtype=float)  # A sold on day 4
    mkt = pd.Series([0.0, -0.01, 0.005, -0.02], index=idx)
    theme = pd.Series([0.0, -0.005, 0.0, 0.01], index=idx)
    out = split_pnl(shares, closes, mkt, pd.Series({"A": 2.0, "B": 0.5}), theme)
    assert out.loc["A", "pnl_inr"] == pytest.approx(10 * (95 - 100))
    assert out.loc["A", "market_part_inr"] == pytest.approx(2.0 * (1000 * -0.01 + 980 * 0.005 + 990 * -0.02))
    parts = out["market_part_inr"] + out["theme_part_inr"] + out["picks_part_inr"]
    assert list(parts.round(6)) == list(out["pnl_inr"].round(6))


def test_recommendation_takes_free_cover_first():
    import hedge_strategies as hs
    t = pd.DataFrame({"strategy": ["No hedge", "A", "Free"], "expected_hedge_cost_inr": [0, 2000, -900],
                      "cvar95_saved_inr": [0, 110000, 105000], "cvar_saved_per_rupee_cost": [np.nan, 55.0, np.nan]})
    assert hs.recommend(t, value=1e7).loc[lambda d: d["recommended"], "strategy"].tolist() == ["Free"]
