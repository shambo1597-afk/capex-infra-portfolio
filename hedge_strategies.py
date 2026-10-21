"""
Which hedge, at what cost, for the professor's horizon? (added 21-Oct-2026)

The professor's brief (14-Oct): protect the portfolio's return up to the close of 29-Oct (when he exits),
at minimal cost, managing the cost through the strike; consider spreads; assign probabilities to the
outcomes and then take a call, never apply a strategy blindly.

Method
  Horizon      H = config.EVALUATION_END_DATE; options of the first expiry on or after H (so nothing
               expires before the exit). Prices: NSE settlement of the latest F&O file.
  Outcomes     20,000 simulated paths to H. The Nifty 50 moves lognormally with the at-the-money implied
               volatility of that expiry (the market's own view of the spread of outcomes) and the CAPM
               market return as drift (T-bill + the 7.08% equity risk premium): real-world probabilities, so
               a hedge has a positive expected cost, as insurance should. The stocks move with the
               Nifty 50 through the portfolio's beta (downside beta on falls, the full-sample beta on
               rises) plus their own risk: the residual volatility sigma_P x sqrt(1 - R^2) from the
               one-year regression. That residual is the theme/stock risk no index hedge reaches (15-Oct).
  Valuation    Every option is revalued at H with Black-Scholes at its own implied volatility and the
               time left (sticky strike); the T-bill rate is r. Every strategy holds the same notional as
               the puts held now (units), so the comparison is like for like.
  Strategies   keep the puts held; sell them (no hedge); protective puts; vertical bear put spreads
               (long the higher strike, short the lower: the professor's notes, p. 26); a near-zero-cost
               collar; and the puts held plus a spread.
  Measures     net premium now; cost if the Nifty is unchanged at H (pure time decay); expected cost of the
               hedge to H (premium minus its expected value at H); probability the hedge ends in profit; the portfolio's expected P&L, 5%
               VaR and 5% CVaR at H; the tail loss saved per rupee of expected cost.
  Decision     among strategies whose expected cost stays within HEDGE_COST_BUDGET_PCT of the portfolio and
               that cut the 5% CVaR by at least MIN_CVAR_SAVED_PCT of it, the one that saves the most tail
               loss per rupee of expected cost ("downside coverage at minimal cost"). Every strategy but
               "keep" sells the puts held first, so the net cash today shows what the switch costs.

Outputs: output/hedge_strategies.csv, output/hedge_strategy_buckets.csv, output/hedge_strategy_curve.csv,
         output/hedge_strategy_meta.csv.
Usage:   python hedge_strategies.py
"""

import logging
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import EVALUATION_END_DATE, OUTPUT_DIR, TRADING_DAYS_PER_YEAR

logger = logging.getLogger("hedge_strategies")

STRATEGIES_CSV = OUTPUT_DIR / "hedge_strategies.csv"
BUCKETS_CSV = OUTPUT_DIR / "hedge_strategy_buckets.csv"
CURVE_CSV = OUTPUT_DIR / "hedge_strategy_curve.csv"
META_CSV = OUTPUT_DIR / "hedge_strategy_meta.csv"
HEDGE_COST_BUDGET_PCT = 0.30   # "minimal cost": expected cost to the horizon at most 0.3% of the portfolio
MIN_CVAR_SAVED_PCT = 1.0       # and it must cut the 5% CVaR by at least 1% of the portfolio to count as cover
PATHS = 20000
SEED = 7
BUCKETS = [(-np.inf, -6), (-6, -3), (-3, -1), (-1, 1), (1, 3), (3, np.inf)]


def _leg_value(spot_h: np.ndarray, leg: Dict, t_left: float, r: float) -> np.ndarray:
    from option_pricing import bs_price
    return bs_price(spot_h, leg["strike"], t_left, r, leg["iv"], leg["kind"])


def chain(fo: pd.DataFrame, expiry: str, spot: float, t: float, r: float) -> pd.DataFrame:
    """Strikes of one expiry with settlement price, open interest and implied volatility."""
    from option_pricing import implied_vol
    c = fo[(fo["FinInstrmTp"] == "IDO") & (fo["XpryDt"] == expiry) & (fo["OpnIntrst"] > 0)].copy()
    c["iv"] = [implied_vol(p, spot, k, t, r, kind) for p, k, kind in zip(c["SttlmPric"], c["StrkPric"], c["OptnTp"])]
    return c.dropna(subset=["iv"])


def nearest(c: pd.DataFrame, kind: str, target: float) -> Optional[pd.Series]:
    """The liquid strike (open interest at least the median of its side) closest to `target`."""
    side = c[c["OptnTp"] == kind]
    if side.empty:
        return None
    liquid = side[side["OpnIntrst"] >= side["OpnIntrst"].median()]
    pool = liquid if not liquid.empty else side
    return pool.iloc[(pool["StrkPric"] - target).abs().argsort().iloc[0]]


def build_strategies(c: pd.DataFrame, spot: float, held: Optional[Dict], held_leg: Optional[Dict]) -> List[Dict]:
    """Candidate strategies as lists of legs (qty +1 bought / -1 sold per unit of notional)."""
    def leg(row, qty):
        return {"kind": row["OptnTp"], "strike": float(row["StrkPric"]), "price": float(row["SttlmPric"]),
                "iv": float(row["iv"]), "qty": qty, "expiry": row["XpryDt"]}

    atm, m2, m3, m5 = (nearest(c, "PE", spot * f) for f in (1.0, 0.98, 0.97, 0.95))
    out = []
    if held_leg is not None:
        out.append({"name": f"Keep the puts held ({held['instrument']})", "legs": [held_leg], "keep": True})
    out.append({"name": "No hedge: sell the puts held", "legs": []})
    for row in (atm, m2, m3):
        if row is not None:
            out.append({"name": f"Switch to a protective put {row['StrkPric']:.0f}", "legs": [leg(row, 1)]})
    pairs = [(atm, m3), (atm, m5), (m2, m5)]
    for hi, lo in pairs:
        if hi is not None and lo is not None and hi["StrkPric"] > lo["StrkPric"]:
            out.append({"name": f"Switch to a bear put spread {hi['StrkPric']:.0f}/{lo['StrkPric']:.0f}",
                        "legs": [leg(hi, 1), leg(lo, -1)], "spread": True})
    if m2 is not None:
        calls = c[(c["OptnTp"] == "CE") & (c["StrkPric"] > spot)]
        if not calls.empty:
            call = calls.iloc[(calls["SttlmPric"] - m2["SttlmPric"]).abs().argsort().iloc[0]]
            out.append({"name": f"Switch to a collar: long put {m2['StrkPric']:.0f}, short call {call['StrkPric']:.0f}",
                        "legs": [leg(m2, 1), leg(call, -1)]})
    if held_leg is not None:
        best_spread = next((x for x in out if x.get("spread")), None)
        if best_spread:
            out.append({"name": f"Keep the puts held and add a {best_spread['name'].split('a ', 1)[1]}",
                        "legs": [held_leg] + best_spread["legs"], "keep": True})
    return out


def simulate(spot: float, iv_atm: float, tau: float, sessions: int, beta_up: float, beta_down: float,
             resid_daily: float, drift: float = 0.0, paths: int = PATHS, seed: int = SEED):
    """Nifty level at H and the stock sleeve's return to H, path by path (`drift`: annual expected return)."""
    rng = np.random.default_rng(seed)
    m = iv_atm * np.sqrt(tau) * rng.standard_normal(paths) + (np.log(1 + drift) - iv_atm ** 2 / 2) * tau
    nifty_ret = np.exp(m) - 1
    stock_ret = np.where(nifty_ret < 0, beta_down, beta_up) * nifty_ret \
        + resid_daily * np.sqrt(sessions) * rng.standard_normal(paths)
    return spot * np.exp(m), nifty_ret, stock_ret


def evaluate(strategies: List[Dict], units: int, spot_h: np.ndarray, nifty_ret: np.ndarray, stock_pnl: np.ndarray,
             horizon: pd.Timestamp, r: float, value: float, spot0: float,
             held_value: float = 0.0) -> Dict[str, pd.DataFrame]:
    rows, bucket_rows, results = [], [], {}
    for s in strategies:
        opt = np.zeros_like(spot_h)
        premium = 0.0
        for lg in s["legs"]:
            t_left = max((pd.Timestamp(lg["expiry"]) - horizon).days, 0) / 365
            opt += lg["qty"] * units * (_leg_value(spot_h, lg, t_left, r) - lg["price"])
            premium += lg["qty"] * units * lg["price"]
        total = stock_pnl + opt
        results[s["name"]] = (opt, total)
        flat = sum(lg["qty"] * units * (_leg_value(np.array([spot0]), lg, max((pd.Timestamp(lg["expiry"]) - horizon).days, 0)
                                                     / 365, r)[0] - lg["price"]) for lg in s["legs"])
        q5 = np.quantile(total, 0.05)
        rows.append({
            "strategy": s["name"],
            "legs": "; ".join(f"{'buy' if lg['qty'] > 0 else 'sell'} {units} {lg['expiry']} {lg['strike']:.0f} "
                              f"{lg['kind']} @ {lg['price']:.2f}" for lg in s["legs"]) or "none",
            "net_premium_inr": round(premium),
            "net_cash_today_inr": round((premium - (0 if s.get("keep") else held_value)
                                         - (held_value if s.get("keep") else 0)) * -1),
            "net_premium_pct": round(premium / value * 100, 3),
            "cost_if_nifty_unchanged_inr": round(-flat),
            "expected_hedge_cost_inr": round(-opt.mean()),
            "expected_hedge_cost_pct": round(-opt.mean() / value * 100, 3),
            "prob_hedge_profit_pct": round((opt > 0).mean() * 100, 1),
            "expected_pnl_inr": round(total.mean()),
            "var95_inr": round(-q5),
            "cvar95_inr": round(-total[total <= q5].mean()),
            "worst_1pct_inr": round(-np.quantile(total, 0.01)),
        })
        for lo, hi in BUCKETS:
            mask = (nifty_ret * 100 > lo) & (nifty_ret * 100 <= hi)
            bucket_rows.append({"strategy": s["name"], "nifty_move": _bucket_label(lo, hi),
                                "probability_pct": round(mask.mean() * 100, 1),
                                "mean_portfolio_pnl_inr": round(total[mask].mean()) if mask.any() else np.nan})
    table = pd.DataFrame(rows)
    base = table.loc[table["strategy"].str.startswith("No hedge")].iloc[0]
    table["recommended"] = False
    table["cvar95_saved_inr"] = (base["cvar95_inr"] - table["cvar95_inr"]).round()
    table["cvar_saved_per_rupee_cost"] = np.where(table["expected_hedge_cost_inr"] > 0,
                                                  (table["cvar95_saved_inr"] / table["expected_hedge_cost_inr"]).round(2),
                                                  np.nan)
    return {"table": table, "buckets": pd.DataFrame(bucket_rows), "results": results}


def _bucket_label(lo: float, hi: float) -> str:
    if lo == -np.inf:
        return f"below {hi:+g}%"
    if hi == np.inf:
        return f"above {lo:+g}%"
    return f"{lo:+g}% to {hi:+g}%"


def recommend(table: pd.DataFrame, value: float, budget_pct: float = HEDGE_COST_BUDGET_PCT,
              min_saved_pct: float = MIN_CVAR_SAVED_PCT) -> pd.DataFrame:
    ok = table[(table["expected_hedge_cost_inr"] > 0) & (table["expected_hedge_cost_inr"] <= value * budget_pct / 100)
               & (table["cvar95_saved_inr"] >= value * min_saved_pct / 100)]
    best = ok.sort_values(["cvar_saved_per_rupee_cost", "cvar95_saved_inr"], ascending=False).head(1)
    return table.assign(recommended=table["strategy"].isin(best["strategy"]))


def payoff_curve(strategies: List[Dict], units: int, spot: float, value: float, beta_up: float, beta_down: float,
                 horizon: pd.Timestamp, r: float) -> pd.DataFrame:
    """Portfolio P&L at H for Nifty moves from -10% to +6% (stocks' own risk set to zero)."""
    moves = np.arange(-10, 6.5, 0.5)
    spot_h = spot * (1 + moves / 100)
    stock = value * np.where(moves < 0, beta_down, beta_up) * moves / 100
    out = {"nifty_move_pct": moves, "nifty_level": spot_h.round(0)}
    for s in strategies:
        opt = np.zeros_like(spot_h)
        for lg in s["legs"]:
            t_left = max((pd.Timestamp(lg["expiry"]) - horizon).days, 0) / 365
            opt += lg["qty"] * units * (_leg_value(spot_h, lg, t_left, r) - lg["price"])
        out[s["name"]] = (stock + opt).round(0)
    return pd.DataFrame(out)


def run(as_of: Optional[date] = None) -> Optional[Dict[str, pd.DataFrame]]:
    from option_pricing import implied_vol
    from risk_free import risk_free_3m
    from risk_model import HEDGE_PLAN_CSV, choose_expiry, fetch_nifty_derivatives, held_put, load_factor_series
    from risk_model import portfolio_returns, stock_return_matrix
    from tracker import TRACKER_POSITIONS_CSV
    from config import HISTORICAL_OHLCV_CSV
    from var_model import sessions_to_horizon

    plan = pd.read_csv(HEDGE_PLAN_CSV).set_index("metric")["value"]
    fo_date = pd.Timestamp(plan["Prices as of"]).date() if as_of is None else as_of
    fo = fetch_nifty_derivatives(fo_date, use_cache=True, save=False)
    if fo is None or fo.empty:
        logger.warning("No F&O prices for %s; hedge strategies not rebuilt.", fo_date)
        return None
    r = risk_free_3m()[0]
    spot = float(fo["UndrlygPric"].iloc[0])
    horizon = pd.Timestamp(EVALUATION_END_DATE)
    expiry = choose_expiry(fo, "IDO", EVALUATION_END_DATE)
    t = (pd.Timestamp(expiry) - pd.Timestamp(fo_date)).days / 365
    c = chain(fo, expiry, spot, t, r)
    atm = nearest(c, "PE", spot)
    iv_atm = float(atm["iv"])

    pos = pd.read_csv(TRACKER_POSITIONS_CSV)
    value = float(pos["value_inr"].sum())
    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    port = portfolio_returns(stock_return_matrix(stock_data, pos["symbol"].tolist()),
                             pos.set_index("symbol")["value_inr"] / value)
    sigma_p = float(port.iloc[-245:].std())
    beta_up = float(plan["Minimum-variance hedge ratio h*"])
    beta_down = float(plan["Downside beta (Nifty 50 down days)"])
    r2 = float(plan["Hedge effectiveness (R^2)"])
    resid = sigma_p * np.sqrt(max(1 - r2, 0.0))

    held = held_put()
    held_leg = None
    units = int(held["units"]) if held else int(round(float(plan["Puts: lots"]))) * int(float(plan["Nifty lot size"]))
    if held:
        h = fo[(fo["FinInstrmTp"] == "IDO") & (fo["OptnTp"] == "PE") & (fo["XpryDt"] == held["expiry"])
               & np.isclose(fo["StrkPric"], held["strike"])]
        if not h.empty:
            price = float(h["SttlmPric"].iloc[0])
            th = (pd.Timestamp(held["expiry"]) - pd.Timestamp(fo_date)).days / 365
            held_leg = {"kind": "PE", "strike": held["strike"], "price": price, "expiry": held["expiry"], "qty": 1,
                        "iv": implied_vol(price, spot, held["strike"], th, r)}

    strategies = build_strategies(c, spot, held, held_leg)
    tau = max((horizon - pd.Timestamp(fo_date)).days, 1) / 365
    sessions = sessions_to_horizon(fo_date)
    from config import MARKET_RISK_PREMIUM_PCT
    drift = r + MARKET_RISK_PREMIUM_PCT / 100
    spot_h, nifty_ret, stock_ret = simulate(spot, iv_atm, tau, sessions, beta_up, beta_down, resid, drift)
    held_value = held_leg["price"] * units if held_leg else 0.0
    ev = evaluate(strategies, units, spot_h, nifty_ret, value * stock_ret, horizon, r, value, spot, held_value)
    table = recommend(ev["table"], value)
    curve = payoff_curve(strategies, units, spot, value, beta_up, beta_down, horizon, r)
    from option_pricing import norm_cdf
    probs = {f"P(Nifty below {k:,.0f} on {horizon:%d-%b})":
             round(norm_cdf((np.log(k / spot) + iv_atm ** 2 * tau / 2) / (iv_atm * np.sqrt(tau))) * 100, 1)
             for k in sorted({float(x["strike"]) for s in strategies for x in s["legs"] if x["kind"] == "PE"},
                             reverse=True)}
    meta = pd.DataFrame([
        ("Prices as of", str(fo_date)), ("Horizon (professor exits)", EVALUATION_END_DATE),
        ("Sessions to the horizon", sessions), ("Option expiry used", expiry),
        ("Nifty 50 spot", round(spot, 2)), ("At-the-money implied volatility (%)", round(iv_atm * 100, 2)),
        ("Risk-free (91-day T-bill, %)", round(r * 100, 4)),
        ("Nifty drift in the simulation (CAPM market return, % a year)", round(drift * 100, 2)),
        ("Units per strategy", units),
        ("Stocks valued (Rs)", round(value)), ("Beta on rises (h*)", round(beta_up, 3)),
        ("Beta on falls (downside)", round(beta_down, 3)), ("R^2 vs Nifty 50", round(r2, 3)),
        ("Stocks own daily volatility (%)", round(resid * 100, 3)),
        ("Expected-cost budget (% of portfolio)", HEDGE_COST_BUDGET_PCT),
        ("Minimum CVaR cut to count as cover (% of portfolio)", MIN_CVAR_SAVED_PCT),
        *probs.items(),
    ], columns=["metric", "value"])
    table.to_csv(STRATEGIES_CSV, index=False)
    ev["buckets"].to_csv(BUCKETS_CSV, index=False)
    curve.to_csv(CURVE_CSV, index=False)
    meta.to_csv(META_CSV, index=False)
    rec = table.loc[table["recommended"], "strategy"]
    logger.info("Hedge strategies to %s: recommended %s.", EVALUATION_END_DATE, rec.iloc[0] if not rec.empty else "none")
    return {"table": table, "buckets": ev["buckets"], "curve": curve, "meta": meta}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    res = run()
    if res:
        pd.set_option("display.width", 250)
        print(res["meta"].to_string(index=False))
        print(res["table"].drop(columns=["legs"]).to_string(index=False))
        print(res["table"][["strategy", "legs"]].to_string(index=False))
