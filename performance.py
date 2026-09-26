"""
Performance: Sharpe, Treynor, XIRR, the compounding effect, and the Capital Market Line.

Definitions (daily data, TRADING_DAYS_PER_YEAR sessions a year, risk-free = Nifty 1D Rate index):
  r_p            portfolio return: sum_i w_i r_i each day with the current weights (renormalised over
                 the stocks trading that day), compounded over the window: R_p = prod(1 + r_p,t) - 1
  annualised     compound: (1 + R)^(252 / n) - 1; the "simple" figure R x 252 / n is shown next to it,
                 and the gap between them is the compounding effect
  Sharpe         (R_p,ann - R_f,ann) / sigma_p,ann             reward per unit of total risk
  Treynor        (R_p,ann - R_f,ann) / beta_p (vs Nifty 500)     reward per unit of market risk
  Jensen alpha   R_p,ann - [R_f,ann + beta_p (R_m,ann - R_f,ann)]
  XIRR           the annual rate r with sum CF_k / (1 + r)^(days_k / 365) = 0 over dated cash flows
                 (invest at the start, value at the end; a loss gives a negative XIRR, annualised the
                 same way)
The windows before the 28-Sep-2026 snapshot are a BACKTEST of today's portfolio (chosen with
hindsight: its stocks were picked for strong past returns), not realised performance.

CML: the line from the risk-free rate through the market portfolio (Nifty 500 TRI) in (sigma, E[r])
space, E[r] = r_f + (E[r_m] - r_f) / sigma_m x sigma. Plotted with the 11 stocks, the long-only
efficient frontier of the 11, its tangency (maximum-Sharpe) portfolio, and our portfolio.
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import HISTORICAL_OHLCV_CSV, OUTPUT_DIR, RISK_SUMMARY_OUTPUT_CSV, TRADING_DAYS_PER_YEAR
from risk_model import excess_returns, load_factor_series, portfolio_returns, stock_return_matrix
from weights import project_to_bounded_simplex

logger = logging.getLogger("performance")

PERFORMANCE_CSV = OUTPUT_DIR / "performance_summary.csv"
GROWTH_CSV = OUTPUT_DIR / "performance_growth.csv"
CML_CSV = OUTPUT_DIR / "cml_points.csv"
CML_PNG = OUTPUT_DIR / "cml.png"
WINDOWS = {"Last quarter (63 sessions)": 63, "Last year": None}


def xirr(cash_flows: Sequence[Tuple[date, float]]) -> float:
    """Annual internal rate of return of dated cash flows (bisection on the NPV; needs a sign change)."""
    flows = sorted(cash_flows)
    t0 = pd.Timestamp(flows[0][0])
    years = np.array([(pd.Timestamp(d) - t0).days / 365.0 for d, _ in flows])
    amounts = np.array([a for _, a in flows], dtype=float)

    def npv(rate):
        return float((amounts / (1 + rate) ** years).sum())

    lo, hi = -0.9999, 100.0
    if npv(lo) * npv(hi) > 0:
        raise ValueError("XIRR needs cash flows of both signs with a root in (-100%, +10000%).")
    for _ in range(300):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def window_metrics(port: pd.Series, mkt: pd.Series, rf: pd.Series, label: str) -> Dict[str, object]:
    """Return, risk and risk-adjusted metrics of daily portfolio returns against the market over one window."""
    data = pd.concat([port.rename("p"), mkt.rename("m"), rf.rename("rf")], axis=1).dropna()
    n = len(data)
    growth_p = float((1 + data["p"]).prod())
    growth_m = float((1 + data["m"]).prod())
    growth_f = float((1 + data["rf"]).prod())
    ann = lambda g: g ** (TRADING_DAYS_PER_YEAR / n) - 1
    rp, rm, rfa = ann(growth_p), ann(growth_m), ann(growth_f)
    sigma_p = data["p"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sigma_m = data["m"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    beta = np.cov(data["p"] - data["rf"], data["m"] - data["rf"])[0, 1] / (data["m"] - data["rf"]).var()
    start, end = data.index[0], data.index[-1]
    # XIRR: invest Rs 1 the session before the first return, value on the last session
    start_flow_date = (start - pd.tseries.offsets.BDay(1)).date()
    port_xirr = xirr([(start_flow_date, -1.0), (end.date(), growth_p)])
    mkt_xirr = xirr([(start_flow_date, -1.0), (end.date(), growth_m)])
    return {
        "window": label,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "sessions": n,
        "portfolio_return_pct": round((growth_p - 1) * 100, 2),
        "benchmark_return_pct": round((growth_m - 1) * 100, 2),
        "excess_return_pp": round((growth_p - growth_m) * 100, 2),
        "portfolio_annualised_compound_pct": round(rp * 100, 2),
        "portfolio_annualised_simple_pct": round((growth_p - 1) * TRADING_DAYS_PER_YEAR / n * 100, 2),
        "compounding_effect_pp": round((rp - (growth_p - 1) * TRADING_DAYS_PER_YEAR / n) * 100, 2),
        "benchmark_annualised_pct": round(rm * 100, 2),
        "risk_free_annualised_pct": round(rfa * 100, 2),
        "portfolio_vol_pct": round(sigma_p * 100, 2),
        "benchmark_vol_pct": round(sigma_m * 100, 2),
        "beta_vs_nifty500": round(beta, 3),
        "sharpe_portfolio": round((rp - rfa) / sigma_p, 2),
        "sharpe_benchmark": round((rm - rfa) / sigma_m, 2),
        "treynor_portfolio_pct": round((rp - rfa) / beta * 100, 2),
        "treynor_benchmark_pct": round((rm - rfa) * 100, 2),  # benchmark beta = 1
        "jensen_alpha_pct": round((rp - (rfa + beta * (rm - rfa))) * 100, 2),
        "xirr_portfolio_pct": round(port_xirr * 100, 2),
        "xirr_benchmark_pct": round(mkt_xirr * 100, 2),
    }


def tangency_portfolio(mu: np.ndarray, cov: np.ndarray, rf: float, iterations: int = 4000) -> np.ndarray:
    """Long-only maximum-Sharpe weights (projected gradient ascent on the Sharpe ratio)."""
    n = len(mu)
    w = np.full(n, 1.0 / n)
    for _ in range(iterations):
        var = w @ cov @ w
        sd = np.sqrt(var)
        excess = w @ mu - rf
        grad = mu / sd - excess * (cov @ w) / (sd ** 3)
        w = project_to_bounded_simplex(w + 0.01 * grad / (np.abs(grad).max() + 1e-12), 0.0, 1.0)
    return w


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 40, iterations: int = 3000) -> pd.DataFrame:
    """Long-only minimum-variance frontier: minimise w'Cw - lambda w'mu over a grid of lambda."""
    rows = []
    for lam in np.concatenate([[0.0], np.geomspace(0.01, 50, points)]):
        w = np.full(len(mu), 1.0 / len(mu))
        step = 0.5 / (np.linalg.eigvalsh(cov).max() + 1e-12)
        for _ in range(iterations):
            w = project_to_bounded_simplex(w - step * (2 * cov @ w - lam * mu), 0.0, 1.0)
        rows.append({"sigma": float(np.sqrt(w @ cov @ w)), "expected_return": float(w @ mu)})
    return pd.DataFrame(rows).drop_duplicates().sort_values("sigma").reset_index(drop=True)


def cml_points(returns: pd.DataFrame, port: pd.Series, fx: pd.DataFrame) -> Dict[str, object]:
    """Annualised (sigma, mean return) of the stocks, portfolio, market, tangency, frontier and CML (past year)."""
    data = returns.join(fx[["mkt_excess", "rf"]], how="inner")
    mkt = data["mkt_excess"] + data["rf"]
    rf = float(data["rf"].mean() * TRADING_DAYS_PER_YEAR)
    stocks = data[returns.columns]
    mu = stocks.mean().values * TRADING_DAYS_PER_YEAR
    cov = stocks.cov().values * TRADING_DAYS_PER_YEAR
    points = [{"name": s, "kind": "stock", "sigma": float(np.sqrt(cov[i, i])), "expected_return": float(mu[i])}
              for i, s in enumerate(returns.columns)]
    p = port.reindex(stocks.index)
    points.append({"name": "Our portfolio", "kind": "portfolio", "sigma": float(p.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
                   "expected_return": float(p.mean() * TRADING_DAYS_PER_YEAR)})
    m_sigma = float(mkt.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    m_mu = float(mkt.mean() * TRADING_DAYS_PER_YEAR)
    points.append({"name": "Nifty 500 TRI (market)", "kind": "market", "sigma": m_sigma, "expected_return": m_mu})
    w_t = tangency_portfolio(mu, cov, rf)
    points.append({"name": "Tangency (max Sharpe of the 11)", "kind": "tangency",
                   "sigma": float(np.sqrt(w_t @ cov @ w_t)), "expected_return": float(w_t @ mu)})
    points.append({"name": "Risk-free (1D rate)", "kind": "risk_free", "sigma": 0.0, "expected_return": rf})
    return {"points": pd.DataFrame(points), "frontier": efficient_frontier(mu, cov), "rf": rf,
            "market_sharpe": (m_mu - rf) / m_sigma, "tangency_weights": dict(zip(returns.columns, np.round(w_t, 4)))}


def plot_cml(cml: Dict[str, object], path=CML_PNG) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, frontier, rf = cml["points"], cml["frontier"], cml["rf"]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    xmax = max(pts["sigma"].max(), frontier["sigma"].max()) * 1.08
    xs = np.linspace(0, xmax, 50)
    ax.plot(xs * 100, (rf + cml["market_sharpe"] * xs) * 100, color="#1f77b4", lw=2,
            label=f"CML through the market (Sharpe {cml['market_sharpe']:.2f})")
    tan = pts[pts["kind"] == "tangency"].iloc[0]
    tan_sharpe = (tan["expected_return"] - rf) / tan["sigma"]
    ax.plot(xs * 100, (rf + tan_sharpe * xs) * 100, color="#2ca02c", lw=1.2, ls="--",
            label=f"Capital allocation line through the tangency (Sharpe {tan_sharpe:.2f})")
    ax.plot(frontier["sigma"] * 100, frontier["expected_return"] * 100, color="#555", lw=1.5,
            label="Efficient frontier of the 11 stocks (long-only)")
    styles = {"stock": ("o", "#999999", 40), "portfolio": ("*", "#d62728", 260), "market": ("s", "#1f77b4", 90),
              "tangency": ("D", "#2ca02c", 80), "risk_free": ("o", "#000000", 50)}
    for _, r in pts.iterrows():
        marker, color, size = styles[r["kind"]]
        ax.scatter(r["sigma"] * 100, r["expected_return"] * 100, marker=marker, color=color, s=size, zorder=3)
        ax.annotate(r["name"], (r["sigma"] * 100, r["expected_return"] * 100), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised mean return (%)")
    ax.set_title("Capital Market Line: past year of daily returns (ex-post; the stocks were picked on strong past returns)",
                 fontsize=9.5)
    ax.axhline(rf * 100, color="#bbb", lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved CML plot to %s", path)


def run(as_of: Optional[date] = None) -> Dict[str, pd.DataFrame]:
    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV)
    symbols = risk["symbol"].tolist()
    returns = stock_return_matrix(stock_data, symbols)
    if as_of is not None:
        returns = returns.loc[:pd.Timestamp(as_of)]
    fx = excess_returns(load_factor_series()).reindex(returns.index)
    port = portfolio_returns(returns, risk.set_index("symbol")["weight_pct"] / 100)
    mkt = fx["mkt_excess"] + fx["rf"]

    rows = []
    for label, sessions in WINDOWS.items():
        idx = port.index if sessions is None else port.index[-sessions:]
        rows.append(window_metrics(port.loc[idx], mkt.loc[idx], fx["rf"].loc[idx], label))
    summary = pd.DataFrame(rows)
    summary.to_csv(PERFORMANCE_CSV, index=False)

    growth = pd.DataFrame({"portfolio": (1 + port).cumprod(), "nifty500_tri": (1 + mkt.fillna(0)).cumprod()})
    growth.index.name = "date"
    growth.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(
        GROWTH_CSV, index=False, float_format="%.6f")

    cml = cml_points(returns, port, fx)
    cml["points"].to_csv(CML_CSV, index=False, float_format="%.6f")
    plot_cml(cml)
    return {"summary": summary, "cml": cml["points"], "tangency": cml["tangency_weights"]}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    as_of_arg = date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1]) if "--as-of" in sys.argv else None
    out = run(as_of_arg)
    pd.set_option("display.width", 220)
    print(out["summary"].T.to_string())
    print(out["cml"].to_string(index=False))
    print("Tangency weights:", out["tangency"])
