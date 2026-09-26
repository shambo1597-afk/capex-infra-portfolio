"""
Regression and hedging: how much of the portfolio's risk is the market, and how to hedge it.

1. Single-index model (vs the Nifty 500 TRI, the benchmark), per stock and for the portfolio:
       r_i - r_f = alpha + beta (r_m - r_f) + e
   Total variance = beta^2 var(r_m)  [explained / systematic]  +  var(e)  [unexplained / idiosyncratic];
   R^2 is the explained share. Only the explained part can be hedged with index derivatives; the
   unexplained part is handled by diversification and the stock-level stop-losses.
2. Multifactor model: adds the change in crude (Brent) and in interest rates (10-year G-sec price
   return, ~ -duration x the change in yield) to the market factor, and compares R^2 and adjusted R^2
   with the single-index model: do crude or rates explain what the market does not?
3. Hedge sizing against the Nifty 50 (the index with liquid futures and options):
   - minimum-variance hedge ratio h* = cov(r_p, r_N50) / var(r_N50) (the regression beta); hedge
     effectiveness = R^2 (the share of variance a futures hedge removes);
   - futures lots N* = h* x portfolio value / (futures price x lot size);
   - tail hedge ratio = beta estimated on the worst TAIL_QUANTILE of Nifty 50 days (correlations rise
     in a sell-off, so the crash beta, not the average beta, sizes the insurance);
   - protective puts: lots = tail hedge ratio x portfolio value / (spot x lot size), expiry covering the
     evaluation window, strike about TAIL_HEDGE_OTM_PCT below spot; cost and a scenario table.

Returns are daily simple returns; annualised with TRADING_DAYS_PER_YEAR. Portfolio returns use the
current weights, held constant (renormalised over the stocks trading on each day).
"""

import io
import logging
import math
import time
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    AUTOCORRELATION_LAGS,
    BENCHMARK_INDEX_NAME,
    CRUDE_TICKER,
    DERIVATIVES_DIR,
    EQUITY_ALLOCATION_PCT,
    EVALUATION_END_DATE,
    FACTORS_CSV,
    FUTURES_MARGIN_PCT_ASSUMED,
    GSEC_INDEX_NAME,
    HEDGE_INDEX_NAME,
    HEDGE_INDEX_SYMBOL,
    HEDGE_PROFIT_TRIGGER_PCT,
    HISTORICAL_OHLCV_CSV,
    MARKET_RISK_PREMIUM_PCT,
    MARKET_RISK_PREMIUM_SOURCE,
    NSE_FO_BHAVCOPY_URL_TEMPLATE,
    OUTPUT_DIR,
    PRINCIPAL_INR,
    RISK_FREE_INDEX_NAME,
    RISK_SUMMARY_OUTPUT_CSV,
    TAIL_HEDGE_OTM_PCT,
    TAIL_QUANTILE,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger("risk_model")

SINGLE_INDEX_CSV = OUTPUT_DIR / "regression_single_index.csv"
MULTIFACTOR_CSV = OUTPUT_DIR / "regression_multifactor.csv"
CAPM_CSV = OUTPUT_DIR / "capm_expected_returns.csv"
AUTOCORR_CSV = OUTPUT_DIR / "autocorrelation.csv"
LJUNG_BOX_CRITICAL_5PCT = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 10: 18.31}  # chi-square 95th percentiles
HEDGE_PLAN_CSV = OUTPUT_DIR / "hedge_plan.csv"
PUT_CANDIDATES_CSV = OUTPUT_DIR / "hedge_put_candidates.csv"
HEDGE_SCENARIOS_CSV = OUTPUT_DIR / "hedge_scenarios.csv"
PORTFOLIO_LABEL = "PORTFOLIO"
FACTOR_COLUMNS = ["nifty500_tri", "nifty50", "gsec10y_clean", "rate_1d_index", "brent_usd"]


# ----------------------------------------------------------------------------- data

def build_factor_series(start: date, end: date, benchmark_tri: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Daily factor levels on NSE sessions: Nifty 500 TRI, Nifty 50, 10-year G-sec (clean price),
    Nifty 1D Rate index, and Brent (last US close strictly before the Indian session, so the day's
    crude move is information the Indian market could trade on). Saved to FACTORS_CSV.
    """
    from fetch_data import NSEBhavcopyFetcher, load_benchmark_tri

    fetcher = NSEBhavcopyFetcher()
    frames = {}
    for column, name in [("nifty50", HEDGE_INDEX_NAME), ("gsec10y_clean", GSEC_INDEX_NAME),
                         ("rate_1d_index", RISK_FREE_INDEX_NAME)]:
        closes = fetcher.fetch_index_closes(start, end, index_name=name)
        frames[column] = closes.set_index("Date")["Close"]
    df = pd.DataFrame(frames)
    df.index.name = "date"

    tri = benchmark_tri if benchmark_tri is not None else load_benchmark_tri(as_of=end, warn_if_stale=False)
    tri = tri.copy()
    tri["Date"] = pd.to_datetime(tri["Date"], format="mixed")
    df["nifty500_tri"] = tri.set_index("Date")["Total Returns Index"].reindex(df.index)
    df["brent_usd"] = _brent_aligned(df.index, start, end)

    FACTORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = df[FACTOR_COLUMNS].reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(FACTORS_CSV, index=False, float_format="%.4f")
    logger.info("Saved %d sessions of factor levels to %s", len(out), FACTORS_CSV)
    return df[FACTOR_COLUMNS]


def _brent_aligned(sessions: pd.DatetimeIndex, start: date, end: date) -> pd.Series:
    """Brent close of the last US session strictly before each Indian session (NaN if unavailable)."""
    try:
        import yfinance as yf
        raw = yf.download(CRUDE_TICKER, start=(pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                          end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=False)
    except Exception as exc:  # network or API failure: the crude factor is then missing, never guessed
        logger.warning("Brent download failed (%s); crude factor unavailable.", exc)
        return pd.Series(np.nan, index=sessions)
    if raw is None or raw.empty:
        logger.warning("No Brent data returned; crude factor unavailable.")
        return pd.Series(np.nan, index=sessions)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.dropna().sort_index()
    positions = close.index.searchsorted(sessions, side="left") - 1  # last date < session
    values = [close.iloc[p] if p >= 0 else np.nan for p in positions]
    return pd.Series(values, index=sessions, dtype=float)


def load_factor_series(path: Path = FACTORS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


def fetch_nifty_derivatives(as_of: date, use_cache: bool = True, save: bool = True) -> Optional[pd.DataFrame]:
    """
    Nifty futures and options rows of NSE's F&O bhavcopy for one session (cached as a small CSV in
    DERIVATIVES_DIR unless save is False). None when NSE could not be reached and nothing is cached.
    """
    cache = DERIVATIVES_DIR / f"nifty_fo_{as_of:%Y-%m-%d}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache)
    from fundamentals import _get_nse_pledge_session  # NSE session with the Akamai cookies

    url = NSE_FO_BHAVCOPY_URL_TEMPLATE.format(yyyymmdd=f"{as_of:%Y%m%d}")
    resp = None
    for attempt in range(4):
        if attempt:
            time.sleep(3 * attempt)  # Akamai blocks are intermittent: back off, refresh the cookies
        try:
            resp = _get_nse_pledge_session(refresh=attempt > 0).get(url, timeout=30)
        except Exception as exc:
            logger.info("F&O bhavcopy attempt %d failed (%s).", attempt + 1, exc)
            continue
        if resp.status_code == 200 and resp.content.startswith(b"PK"):
            break
    if resp is None:
        logger.warning("F&O bhavcopy download failed.")
        return None
    if resp.status_code != 200 or not resp.content.startswith(b"PK"):
        logger.warning("F&O bhavcopy for %s not available (HTTP %d).", as_of, resp.status_code)
        return None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df = df[(df["TckrSymb"] == HEDGE_INDEX_SYMBOL) & df["FinInstrmTp"].isin(["IDF", "IDO"])]
    keep = ["TradDt", "FinInstrmTp", "XpryDt", "StrkPric", "OptnTp", "ClsPric", "SttlmPric", "UndrlygPric",
            "OpnIntrst", "TtlTradgVol", "NewBrdLotQty"]
    df = df[keep].reset_index(drop=True)
    if save:
        DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        logger.info("Saved %d Nifty F&O rows for %s to %s", len(df), as_of, cache)
    return df


def stock_return_matrix(stock_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    df = stock_data[stock_data["SYMBOL"].isin(symbols)].copy()
    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE"])]
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed")
    closes = df.pivot_table(index="DATE1", columns="SYMBOL", values="CLOSE_PRICE").sort_index()
    return closes.reindex(columns=symbols).pct_change(fill_method=None).iloc[1:]


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily return of the weighted portfolio, weights renormalised over the stocks trading that day."""
    w = weights.reindex(returns.columns).fillna(0.0)
    available = returns.notna()
    eff = available.mul(w, axis=1)
    eff = eff.div(eff.sum(axis=1), axis=0)
    return (returns.fillna(0.0) * eff).sum(axis=1).rename(PORTFOLIO_LABEL)


# ----------------------------------------------------------------------------- regression

def ols(y: pd.Series, X: pd.DataFrame) -> Dict[str, object]:
    """OLS with an intercept: coefficients, standard errors, t-stats, R^2, adjusted R^2, residual variance."""
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    n, k = len(data), X.shape[1] + 1
    if n <= k + 2:
        return {"n": n}
    A = np.column_stack([np.ones(n), data[X.columns].values])
    yv = data["y"].values
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ coef
    sse = float(resid @ resid)
    sst = float(((yv - yv.mean()) ** 2).sum())
    sigma2 = sse / (n - k)
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    return {
        "n": n,
        "coef": dict(zip(["const"] + list(X.columns), coef)),
        "se": dict(zip(["const"] + list(X.columns), se)),
        "t": dict(zip(["const"] + list(X.columns), coef / se)),
        "r2": r2,
        "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k),
        "resid_var": sse / (n - 1),
        "y_var": sst / (n - 1),
    }


def excess_returns(factors: pd.DataFrame) -> pd.DataFrame:
    """Daily factor returns: market (Nifty 500 TRI) and Nifty 50 in excess of the 1D rate, crude and G-sec returns."""
    r = factors.pct_change(fill_method=None)
    out = pd.DataFrame(index=factors.index)
    out["rf"] = r["rate_1d_index"]
    out["mkt_excess"] = r["nifty500_tri"] - out["rf"]
    out["nifty50_ret"] = r["nifty50"]
    out["crude_ret"] = r["brent_usd"]
    out["gsec_ret"] = r["gsec10y_clean"]
    return out.iloc[1:]


def single_index_table(returns: pd.DataFrame, port: pd.Series, fx: pd.DataFrame) -> pd.DataFrame:
    """One row per stock and the portfolio: alpha, beta, R^2 and the explained/unexplained variance split."""
    rows = []
    for name, series in list(returns.items()) + [(PORTFOLIO_LABEL, port)]:
        y = (series - fx["rf"]).rename(name)
        res = ols(y, fx[["mkt_excess"]])
        if "coef" not in res:
            continue
        beta = res["coef"]["mkt_excess"]
        mkt_var = fx["mkt_excess"].loc[y.dropna().index].var()
        total = res["y_var"] * TRADING_DAYS_PER_YEAR
        systematic = beta ** 2 * mkt_var * TRADING_DAYS_PER_YEAR
        rows.append({
            "symbol": name,
            "observations": res["n"],
            "alpha_annual_pct": round(res["coef"]["const"] * TRADING_DAYS_PER_YEAR * 100, 2),
            "alpha_t": round(res["t"]["const"], 2),
            "beta": round(beta, 3),
            "beta_se": round(res["se"]["mkt_excess"], 3),
            "r_squared": round(res["r2"], 3),
            "total_vol_pct": round(math.sqrt(total) * 100, 2),
            "systematic_vol_pct": round(math.sqrt(systematic) * 100, 2),
            "unsystematic_vol_pct": round(math.sqrt(max(total - systematic, 0)) * 100, 2),
            "explained_risk_pct": round(res["r2"] * 100, 1),
            "unexplained_risk_pct": round((1 - res["r2"]) * 100, 1),
        })
    return pd.DataFrame(rows)


def multifactor_table(returns: pd.DataFrame, port: pd.Series, fx: pd.DataFrame) -> pd.DataFrame:
    """Market + crude + rates regression per stock and for the portfolio, with R^2 against the single-index model."""
    factors = ["mkt_excess", "crude_ret", "gsec_ret"]
    rows = []
    for name, series in list(returns.items()) + [(PORTFOLIO_LABEL, port)]:
        y = (series - fx["rf"]).rename(name)
        multi = ols(y, fx[factors])
        single = ols(y.loc[fx[factors].dropna().index], fx[["mkt_excess"]].loc[fx[factors].dropna().index])
        if "coef" not in multi or "coef" not in single:
            continue
        rows.append({
            "symbol": name,
            "observations": multi["n"],
            "beta_market": round(multi["coef"]["mkt_excess"], 3),
            "t_market": round(multi["t"]["mkt_excess"], 2),
            "beta_crude": round(multi["coef"]["crude_ret"], 3),
            "t_crude": round(multi["t"]["crude_ret"], 2),
            "beta_gsec": round(multi["coef"]["gsec_ret"], 3),
            "t_gsec": round(multi["t"]["gsec_ret"], 2),
            "r2_single_index": round(single["r2"], 3),
            "r2_multifactor": round(multi["r2"], 3),
            "adj_r2_single_index": round(single["adj_r2"], 3),
            "adj_r2_multifactor": round(multi["adj_r2"], 3),
            "r2_gain_pp": round((multi["r2"] - single["r2"]) * 100, 2),
        })
    return pd.DataFrame(rows)


def annualised_risk_free(fx: pd.DataFrame, sessions: int = 63) -> float:
    """The overnight rate compounded over the last `sessions` sessions, annualised (a fraction)."""
    rf = fx["rf"].dropna().tail(sessions)
    return float((1 + rf).prod() ** (TRADING_DAYS_PER_YEAR / len(rf)) - 1)


def capm_table(single: pd.DataFrame, rf_annual: float, mrp_pct: float = MARKET_RISK_PREMIUM_PCT) -> pd.DataFrame:
    """
    CAPM expected return per stock and for the portfolio: E[r] = r_f + beta x MRP (annual), and the
    same compounded over the 3-month window. Beta is the single-index beta vs the Nifty 500 TRI.
    """
    out = single[["symbol", "beta"]].copy()
    out["risk_free_pct"] = round(rf_annual * 100, 2)
    out["market_risk_premium_pct"] = mrp_pct
    annual = rf_annual + out["beta"] * mrp_pct / 100
    out["capm_expected_return_pct"] = (annual * 100).round(2)
    out["capm_3m_return_pct"] = (((1 + annual) ** 0.25 - 1) * 100).round(2)
    out["source"] = MARKET_RISK_PREMIUM_SOURCE
    return out


def autocorrelation_table(returns: pd.DataFrame, port: pd.Series, lags: int = AUTOCORRELATION_LAGS) -> pd.DataFrame:
    """
    Does a stock's own past return predict its next one? Per stock and for the portfolio:
    AR(1) regression r_t = a + phi r_(t-1) + e (phi and its t-stat), autocorrelations at lags 1..`lags`
    (significant beyond +/-1.96/sqrt(n)), the Ljung-Box Q = n(n+2) sum rho_k^2/(n-k) against its 5%
    critical value, and the lag-1 autocorrelation of non-overlapping weekly (5-session) returns.
    """
    rows = []
    for name, series in list(returns.items()) + [(PORTFOLIO_LABEL, port)]:
        r = series.dropna()
        n = len(r)
        if n < 30:
            continue
        ar = ols(r.iloc[1:].reset_index(drop=True), pd.DataFrame({"lag1": r.iloc[:-1].values}))
        rho = [float(r.autocorr(k)) for k in range(1, lags + 1)]
        q = n * (n + 2) * sum(rk ** 2 / (n - k) for k, rk in enumerate(rho, start=1))
        weekly = (1 + r).groupby(np.arange(n) // 5).prod() - 1
        bound = 1.96 / math.sqrt(n)
        row = {"symbol": name, "observations": n, "ar1_phi": round(ar["coef"]["lag1"], 3),
               "ar1_t": round(ar["t"]["lag1"], 2)}
        row.update({f"rho_{k}": round(v, 3) for k, v in enumerate(rho, start=1)})
        row.update({
            "significance_bound": round(bound, 3),
            "significant_lags": ", ".join(str(k) for k, v in enumerate(rho, start=1) if abs(v) > bound),
            "ljung_box_q": round(q, 2),
            "ljung_box_critical_5pct": LJUNG_BOX_CRITICAL_5PCT[lags],
            "predictable_at_5pct": bool(q > LJUNG_BOX_CRITICAL_5PCT[lags]),
            "weekly_rho_1": round(float(weekly.autocorr(1)), 3),
            "weekly_significance_bound": round(1.96 / math.sqrt(len(weekly)), 3),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- hedging

def hedge_betas(port: pd.Series, fx: pd.DataFrame, tail_quantile: float = TAIL_QUANTILE) -> Dict[str, float]:
    """Minimum-variance hedge ratio vs the Nifty 50, its effectiveness (R^2), downside and tail betas."""
    data = pd.concat([port.rename("p"), fx["nifty50_ret"].rename("m")], axis=1).dropna()
    full = ols(data["p"], data[["m"]])
    down = ols(data.loc[data["m"] < 0, "p"], data.loc[data["m"] < 0, ["m"]])
    cutoff = data["m"].quantile(tail_quantile)
    tail_mask = data["m"] <= cutoff
    tail = ols(data.loc[tail_mask, "p"], data.loc[tail_mask, ["m"]])
    return {
        "observations": full["n"],
        "min_variance_hedge_ratio": full["coef"]["m"],
        "hedge_effectiveness_r2": full["r2"],
        "downside_beta": down["coef"]["m"],
        "tail_beta": tail["coef"]["m"],
        "tail_beta_se": tail["se"]["m"],
        "tail_days": int(tail_mask.sum()),
        "tail_cutoff_pct": cutoff * 100,
    }


def choose_expiry(fo: pd.DataFrame, instrument: str, on_or_after: str) -> Optional[str]:
    """Earliest expiry of the instrument type (IDF/IDO) on or after the given date."""
    expiries = sorted(fo.loc[fo["FinInstrmTp"] == instrument, "XpryDt"].unique())
    later = [x for x in expiries if x >= on_or_after]
    return later[0] if later else None


def put_candidates(fo: pd.DataFrame, expiry: str, spot: float, portfolio_value: float, hedge_ratio: float,
                   lot: int) -> pd.DataFrame:
    """Liquid puts of one expiry from 10% below spot to at the money: lots, cost and protection level."""
    puts = fo[(fo["FinInstrmTp"] == "IDO") & (fo["XpryDt"] == expiry) & (fo["OptnTp"] == "PE")
              & fo["StrkPric"].between(spot * 0.88, spot * 1.001) & (fo["OpnIntrst"] > 0)].copy()
    lots = max(1, round(hedge_ratio * portfolio_value / (spot * lot)))
    puts["lots"] = lots
    puts["premium"] = puts["SttlmPric"]
    puts["cost_inr"] = (puts["premium"] * lots * lot).round(0)
    puts["cost_pct_of_principal"] = (puts["cost_inr"] / PRINCIPAL_INR * 100).round(2)
    puts["strike_below_spot_pct"] = ((1 - puts["StrkPric"] / spot) * 100).round(2)
    # Portfolio loss (beta-scaled) at which the puts start paying off
    puts["portfolio_loss_before_cover_pct"] = (puts["strike_below_spot_pct"] * hedge_ratio).round(2)
    return puts.rename(columns={"StrkPric": "strike", "OpnIntrst": "open_interest"})[
        ["strike", "premium", "open_interest", "lots", "cost_inr", "cost_pct_of_principal",
         "strike_below_spot_pct", "portfolio_loss_before_cover_pct"]].sort_values("strike").reset_index(drop=True)


def pick_tail_put(candidates: pd.DataFrame, otm_pct: float = TAIL_HEDGE_OTM_PCT) -> pd.Series:
    """The listed strike closest to otm_pct below spot, among strikes with at least median open interest."""
    liquid = candidates[candidates["open_interest"] >= candidates["open_interest"].median()]
    pool = liquid if not liquid.empty else candidates
    return pool.iloc[(pool["strike_below_spot_pct"] - otm_pct).abs().argsort().iloc[0]]


def hedge_scenarios(portfolio_value: float, beta: float, spot: float, put: pd.Series, lot: int,
                    moves_pct=(-25, -20, -15, -10, -5, 0, 5, 10, 15),
                    down_beta: Optional[float] = None) -> pd.DataFrame:
    """
    Portfolio P&L at expiry for Nifty 50 moves, unhedged (beta x move; down_beta for falls, since the
    portfolio falls harder than its average beta in a sell-off) and with the puts.
    """
    rows = []
    premium_paid = put["premium"] * put["lots"] * lot
    for move in moves_pct:
        level = spot * (1 + move / 100)
        unhedged = portfolio_value * (down_beta if down_beta is not None and move < 0 else beta) * move / 100
        payoff = max(put["strike"] - level, 0.0) * put["lots"] * lot
        hedged = unhedged + payoff - premium_paid
        rows.append({
            "nifty50_move_pct": move,
            "nifty50_level": round(level, 1),
            "unhedged_pnl_inr": round(unhedged),
            "put_payoff_inr": round(payoff),
            "premium_inr": round(premium_paid),
            "hedged_pnl_inr": round(hedged),
            "unhedged_pnl_pct": round(unhedged / PRINCIPAL_INR * 100, 2),
            "hedged_pnl_pct": round(hedged / PRINCIPAL_INR * 100, 2),
        })
    return pd.DataFrame(rows)


def build_hedge_plan(port: pd.Series, fx: pd.DataFrame, fo: pd.DataFrame, portfolio_value: float,
                     evaluation_end: str = EVALUATION_END_DATE):
    """Hedge plan rows (metric, value, note), put candidates and scenario table."""
    betas = hedge_betas(port, fx)
    fut_expiries = sorted(fo.loc[fo["FinInstrmTp"] == "IDF", "XpryDt"].unique())
    near_fut = fo[(fo["FinInstrmTp"] == "IDF") & (fo["XpryDt"] == fut_expiries[0])].iloc[0]
    far_fut = fo[(fo["FinInstrmTp"] == "IDF") & (fo["XpryDt"] == fut_expiries[-1])].iloc[0]
    spot = float(near_fut["UndrlygPric"])
    lot = int(near_fut["NewBrdLotQty"])
    h = betas["min_variance_hedge_ratio"]
    fut_price = float(far_fut["SttlmPric"])
    fut_lots_exact = h * portfolio_value / (fut_price * lot)
    fut_lots = round(fut_lots_exact)
    fut_notional = fut_lots * fut_price * lot

    # Downside beta (~half the sessions) sizes the insurance; the worst-10% tail beta rests on ~25 days
    # and is too noisy to size a position (reported for information)
    tail_ratio = max(betas["downside_beta"], h)
    put_expiry = choose_expiry(fo, "IDO", evaluation_end)
    cands = put_candidates(fo, put_expiry, spot, portfolio_value, tail_ratio, lot)
    put = pick_tail_put(cands)
    scen = hedge_scenarios(portfolio_value, h, spot, put, lot, down_beta=tail_ratio)
    hedge_budget = PRINCIPAL_INR - portfolio_value
    trade_date = str(fo["TradDt"].iloc[0])

    plan = [
        ("Prices as of", trade_date, "NSE F&O bhavcopy"),
        ("Portfolio value hedged (Rs)", round(portfolio_value), f"stocks at the latest close ({EQUITY_ALLOCATION_PCT:g}% sleeve)"),
        ("Nifty 50 spot", round(spot, 2), ""),
        ("Nifty lot size", lot, "NSE contract specification"),
        ("Regression sessions", betas["observations"], "daily returns, portfolio vs Nifty 50"),
        ("Minimum-variance hedge ratio h*", round(h, 3), "cov(r_p, r_N50) / var(r_N50) = beta vs Nifty 50"),
        ("Hedge effectiveness (R^2)", round(betas["hedge_effectiveness_r2"], 3),
         "share of portfolio variance a full futures hedge removes"),
        ("Downside beta (Nifty 50 down days)", round(betas["downside_beta"], 3), ""),
        (f"Tail beta (worst {TAIL_QUANTILE:.0%} of days)", round(betas["tail_beta"], 3),
         f"{betas['tail_days']} days with Nifty 50 <= {betas['tail_cutoff_pct']:.2f}%; s.e. {betas['tail_beta_se']:.2f}"),
        ("Tail hedge ratio", round(tail_ratio, 3),
         "max(downside beta, h*): puts sized on the sell-off beta; the worst-10% beta is too noisy (few days)"),
        ("Futures: contract", f"NIFTY FUT {far_fut['XpryDt']}",
         f"longest listed expiry; a hedge to {evaluation_end} needs a roll"),
        ("Futures: price", round(fut_price, 2), ""),
        ("Futures: lots for a full hedge (exact)", round(fut_lots_exact, 2), "h* x value / (futures price x lot)"),
        ("Futures: lots for a full hedge (rounded)", fut_lots,
         f"achieved hedge ratio {fut_lots * fut_price * lot / portfolio_value:.2f}"),
        ("Futures: notional (Rs)", round(fut_notional), ""),
        ("Futures: margin needed (Rs, assumed)", round(fut_notional * FUTURES_MARGIN_PCT_ASSUMED / 100),
         f"ASSUMPTION {FUTURES_MARGIN_PCT_ASSUMED:g}% of notional; exceeds the hedge budget"),
        ("Puts: contract", f"NIFTY {put_expiry} {put['strike']:.0f} PE", "expiry covers the evaluation window"),
        ("Puts: premium", round(float(put["premium"]), 2), "settlement price"),
        ("Puts: lots", int(put["lots"]), "tail hedge ratio x value / (spot x lot)"),
        ("Puts: cost (Rs)", int(put["cost_inr"]), f"{put['cost_pct_of_principal']:.2f}% of the principal"),
        ("Puts: strike below spot (%)", float(put["strike_below_spot_pct"]), ""),
        ("Hedge budget (Rs)", round(hedge_budget), "principal minus the stocks"),
        ("Cash after puts (Rs)", round(hedge_budget - float(put["cost_inr"])),
         "parked at the overnight rate (liquid ETF) for redeployment and the profit-trigger roll"),
        ("Profit trigger", f"+{HEDGE_PROFIT_TRIGGER_PCT:g}%",
         f"portfolio up {HEDGE_PROFIT_TRIGGER_PCT:g}%: roll the puts up to about {TAIL_HEDGE_OTM_PCT:g}% below the new "
         "Nifty level, funded from cash, to lock in part of the gain"),
    ]
    return pd.DataFrame(plan, columns=["metric", "value", "note"]), cands, scen


# ----------------------------------------------------------------------------- pipeline

def run(as_of: Optional[date] = None, refresh_factors: bool = True) -> Dict[str, pd.DataFrame]:
    """Build every regression and hedge output from the committed price history and risk summary."""
    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV)
    symbols = risk["symbol"].tolist()
    weights = risk.set_index("symbol")["weight_pct"] / 100
    returns = stock_return_matrix(stock_data, symbols)
    if as_of is not None:
        returns = returns.loc[:pd.Timestamp(as_of)]
    as_of = returns.index.max().date()  # the last price session (a weekend refresh date has no F&O file)
    if refresh_factors or not FACTORS_CSV.exists():
        build_factor_series((returns.index.min() - pd.Timedelta(days=7)).date(), as_of)
    factors = load_factor_series()
    fx = excess_returns(factors).reindex(returns.index)
    port = portfolio_returns(returns, weights)

    single = single_index_table(returns, port, fx)
    multi = multifactor_table(returns, port, fx)
    capm = capm_table(single, annualised_risk_free(fx))
    autocorr = autocorrelation_table(returns, port)
    single.to_csv(SINGLE_INDEX_CSV, index=False)
    multi.to_csv(MULTIFACTOR_CSV, index=False)
    capm.to_csv(CAPM_CSV, index=False)
    autocorr.to_csv(AUTOCORR_CSV, index=False)
    out = {"single": single, "multi": multi, "capm": capm, "autocorr": autocorr}

    fo = fetch_nifty_derivatives(as_of)
    if fo is None or fo.empty:
        cached = sorted(DERIVATIVES_DIR.glob("nifty_fo_*.csv"))
        if not cached:
            logger.warning("No Nifty F&O prices for %s and none cached; hedge plan not rebuilt.", as_of)
            return out
        logger.warning("No Nifty F&O prices for %s; using the latest cached file %s.", as_of, cached[-1].name)
        fo = pd.read_csv(cached[-1])
    plan, cands, scen = build_hedge_plan(port, fx, fo, float(risk["invested_inr"].sum()))
    plan.to_csv(HEDGE_PLAN_CSV, index=False)
    cands.to_csv(PUT_CANDIDATES_CSV, index=False)
    scen.to_csv(HEDGE_SCENARIOS_CSV, index=False)
    out.update({"plan": plan, "puts": cands, "scenarios": scen})
    return out


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    as_of_arg = None
    if "--as-of" in sys.argv:
        as_of_arg = date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1])
    results = run(as_of_arg, refresh_factors="--no-refresh" not in sys.argv)
    pd.set_option("display.width", 200)
    print(results["single"].to_string(index=False))
    print(results["multi"].to_string(index=False))
    print(results["capm"].to_string(index=False))
    print(results["autocorr"].to_string(index=False))
    if "plan" in results:
        print(results["plan"].to_string(index=False))
        print(results["puts"].to_string(index=False))
        print(results["scenarios"].to_string(index=False))
