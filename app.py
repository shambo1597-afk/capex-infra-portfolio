"""
Streamlit Portfolio Dashboard for Indian Equity Infrastructure & Capex Portfolio.
Coursework: Security Analysis & Portfolio Management (SAPM) / Derivatives
Institution: Indian Institute of Management (IIM) Bodh Gaya
Author / Pair Programmer: Antigravity

This dashboard serves as the central visual terminal for project defense and presentation.
It imports and reuses the existing pipeline outputs, focusing on the locked portfolio
(config.LOCKED_PORTFOLIO) across the Cement, Capital Goods and Power sectors.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_TRI_CSV_PATH,
    HISTORICAL_OHLCV_CSV,
    LOCKED_PORTFOLIO,
    LOCKED_PORTFOLIO_SYMBOLS,
    SELECTION_CONVICTION_TIERS,
    INVESTED_COUNT,
    NEAR_STOP_PCT,
    TRADES_CSV,
    OUTPUT_DIR,
    PORTFOLIO_SYMBOLS,
    EQUITY_ALLOCATION_PCT,
    EVALUATION_START_DATE,
    MARKET_RISK_PREMIUM_PCT,
    MARKET_RISK_PREMIUM_SOURCE,
    TAIL_HEDGE_OTM_PCT,
    PRINCIPAL_INR,
    WEIGHT_MAX_PCT,
    WEIGHT_MIN_PCT,
    SECTOR_SCREENS,
    RISK_SUMMARY_OUTPUT_CSV,
    STOP_LOSS_ATR_MULTIPLE,
    STOP_LOSS_ATR_PERIOD,
    STOP_LOSS_SUPPORT_BAND_ATR,
    SUMMARY_OUTPUT_CSV,
    TECHNICAL_RS_LOOKBACK_DAYS,
    TECHNICAL_RS_MARGIN_PP,
    DI_GAP_THIN_THRESHOLD,
    HIGH_TURNOVER_ROCE_MIN,
    RRG_MOMENTUM_DAYS,
    RRG_MOMENTUM_SMOOTHING_DAYS,
)
from fetch_data import TRI_REDOWNLOAD_INSTRUCTIONS, TriStaleness, assess_tri_staleness, load_benchmark_tri
from fundamentals import get_fundamentals_summary
from tracker import (LEDGER_COLUMNS, REPLACEMENT_PLAN_CSV, TRACKER_SUMMARY_CSV, held_stocks, ledger_fingerprint,
                     publish_ledger, publish_ledger_via_api, replacement_trades, roll_trades, save_ledger, sold_stocks,
                     whatsapp_update)
from tracker import run as run_tracker
from rrg import CONVICTION_HIGH, CONVICTION_LOW, CONVICTION_MODERATE, QUADRANT_STYLE, conviction_tier

QUADRANT_COLOURS = {q: s["color"] for q, s in QUADRANT_STYLE.items()}
from refresh_data import (
    latest_expected_session,
    load_manifest,
    progress_summary,
    read_log_tail,
    refresh_state,
    start_background_refresh,
)
from performance import MIN_LIVE_SESSIONS
from rrg_tails import smooth_path
from sector_screen import review_table_path

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Indian Equity Portfolio | Capex & Infra (IIMBG)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Institutional CSS for a dense, clean analyst-terminal aesthetic
st.markdown(
    """
    <style>
    /* Global Container Padding & Typography */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }
    
    /* Header & Section Title Styling */
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    .app-header {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        color: #0F172A;
    }
    @media (prefers-color-scheme: dark) {
        .app-title { color: #F8FAFC; }
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    
    /* Sector Badges */
    .badge-cement {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(37, 99, 235, 0.12);
        color: #1D4ED8;
        border: 1px solid rgba(37, 99, 235, 0.25);
    }
    .badge-capital-goods {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(217, 119, 6, 0.12);
        color: #B45309;
        border: 1px solid rgba(217, 119, 6, 0.25);
    }
    .badge-power {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(5, 150, 105, 0.12);
        color: #047857;
        border: 1px solid rgba(5, 150, 105, 0.25);
    }
    
    /* Conviction tier badges (RRG): colour plus a text label, never colour alone */
    .tier {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
        border: 1px solid transparent;
    }
    .tier-high { background: rgba(22, 163, 74, 0.14); color: #166534; border-color: rgba(22, 163, 74, 0.35); }
    .tier-moderate { background: rgba(217, 119, 6, 0.14); color: #92400E; border-color: rgba(217, 119, 6, 0.35); }
    .tier-low { background: rgba(220, 38, 38, 0.12); color: #991B1B; border-color: rgba(220, 38, 38, 0.35); }
    .tier-none { background: rgba(128, 128, 128, 0.12); color: inherit; }
    @media (prefers-color-scheme: dark) {
        .tier-high { color: #86EFAC; }
        .tier-moderate { color: #FCD34D; }
        .tier-low { color: #FCA5A5; }
    }

    /* Per-stock portfolio table (one row per stock, scrolls sideways on narrow screens) */
    .pf-wrap { overflow-x: auto; margin: 0.3rem 0 0.6rem 0; }
    .pf-table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
    .pf-table th {
        text-align: right; font-weight: 600; font-size: 0.7rem; color: #64748B;
        padding: 0.35rem 0.5rem; border-bottom: 1px solid rgba(128, 128, 128, 0.35);
        vertical-align: bottom; line-height: 1.25;
    }
    .pf-table td {
        text-align: right; padding: 0.45rem 0.5rem; white-space: nowrap;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15); font-variant-numeric: tabular-nums;
    }
    .pf-table th:first-child, .pf-table td:first-child,
    .pf-table th:nth-child(2), .pf-table td:nth-child(2) { text-align: left; }
    .pf-table .sub { display: block; font-size: 0.68rem; color: #94A3B8; font-weight: 400; }
    .pf-table .ph { color: #2563EB; font-weight: 600; }

    /* Caveat Callout Box */
    .caveat-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        padding: 0.65rem 0.9rem;
        margin: 0.8rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.82rem;
        color: #92400E;
    }
    @media (prefers-color-scheme: dark) {
        .caveat-box {
            background-color: rgba(245, 158, 11, 0.15);
            color: #FCD34D;
        }
    }
    
    /* Compact Metric Card */
    .metric-card {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        text-align: left;
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.15rem;
    }
    @media (prefers-color-scheme: dark) {
        .metric-value { color: #F8FAFC; }
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.1rem;
    }
    
    /* Placeholder Box for Tabs 4 and 5 */
    .placeholder-container {
        padding: 3rem 2rem;
        text-align: center;
        background-color: rgba(128, 128, 128, 0.03);
        border: 1px dashed rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .placeholder-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.65rem;
        background: rgba(59, 130, 246, 0.1);
        color: #2563EB;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
    .placeholder-text {
        font-size: 1.05rem;
        color: #475569;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.6;
    }
    @media (prefers-color-scheme: dark) {
        .placeholder-text { color: #94A3B8; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA LOADING UTILITIES (CACHED)
# -----------------------------------------------------------------------------

def _file_mtime(path: Path) -> float:
    """Modification time used as a cache key, so rewritten outputs are reloaded without a restart."""
    path = Path(path)
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_summary_data(file_mtime: float) -> pd.DataFrame:
    """
    Load the technical summary table, filter strictly for the locked portfolio
    symbols, and enrich with official company display names and sectors.
    file_mtime is only a cache key (see _file_mtime).
    """
    summary_path = Path(SUMMARY_OUTPUT_CSV)
    if not summary_path.exists():
        st.error(f"Required summary file '{summary_path}' not found. Please run main.py first.")
        return pd.DataFrame()

    df = pd.read_csv(summary_path)
    # Filter strictly for locked portfolio symbols
    df = df[df["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)].copy()

    # Enrich with canonical display names and clean sector groupings
    df["display_name"] = df["symbol"].map(lambda s: LOCKED_PORTFOLIO.get(s, {}).get("name", s))
    df["sector"] = df["symbol"].map(lambda s: LOCKED_PORTFOLIO.get(s, {}).get("sector", "Other"))

    # Preserve consistent order as defined in LOCKED_PORTFOLIO
    df["order"] = df["symbol"].map(lambda s: LOCKED_PORTFOLIO_SYMBOLS.index(s) if s in LOCKED_PORTFOLIO_SYMBOLS else 99)
    df = df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_historical_ohlcv(file_mtime: float) -> pd.DataFrame:
    """
    Load historical daily OHLCV dataset for the locked portfolio stocks.
    file_mtime is only a cache key (see _file_mtime).
    """
    ohlcv_path = Path(HISTORICAL_OHLCV_CSV)
    if not ohlcv_path.exists():
        st.error(f"Required historical file '{ohlcv_path}' not found. Please run main.py first.")
        return pd.DataFrame()

    df = pd.read_csv(ohlcv_path)
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed", errors="coerce")
    df = df[df["SYMBOL"].isin(LOCKED_PORTFOLIO_SYMBOLS)].copy()
    df = df.sort_values(["SYMBOL", "DATE1"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_risk_summary(file_mtime: float) -> pd.DataFrame:
    """
    Load the locked portfolio's risk, sizing and stop-loss table written by main.py.
    file_mtime is only a cache key (see _file_mtime).
    """
    risk_path = Path(RISK_SUMMARY_OUTPUT_CSV)
    if not risk_path.exists():
        return pd.DataFrame()
    return pd.read_csv(risk_path)


RRG_COLUMNS = ["rs_momentum_vs_nifty500", "rrg_quadrant_vs_nifty500", "rs_score_vs_sector_avg",
               "rs_momentum_vs_sector", "rrg_quadrant_vs_sector", "di_gap", "thin_trend_flag",
               "fundamentals_failed", "high_turnover_business_flag", "technically_attractive",
               "hard_fundamentals_pass", "soft_fundamental_fails"]


@st.cache_data(show_spinner=False)
def load_rrg_data(file_mtimes: tuple) -> pd.DataFrame:
    """
    RRG quadrants (vs Nifty 500 and vs sector average), DI gap and fundamental-screen failures
    for the locked stocks, read from the three per-sector review tables, plus the conviction
    tier (rrg.conviction_tier). file_mtimes is only a cache key (see _file_mtime).
    """
    frames = [pd.read_csv(review_table_path(sector)) for sector in SECTOR_SCREENS
              if review_table_path(sector).exists()]
    if not frames:
        return pd.DataFrame(columns=["symbol", *RRG_COLUMNS, "conviction_tier"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)][["symbol", *RRG_COLUMNS]].copy()
    df["conviction_tier"] = [conviction_tier(a, b) for a, b in
                             zip(df["rrg_quadrant_vs_nifty500"], df["rrg_quadrant_vs_sector"])]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_tri_benchmark(file_mtime: float) -> pd.DataFrame:
    """
    Load the manually maintained Nifty 500 TRI CSV. Staleness is shown as a dashboard
    banner (render_tri_staleness_banner), so the loader's console warning is disabled.

    file_mtime is only a cache key: replacing the CSV invalidates the cached copy, so a
    re-downloaded file clears the stale banner without restarting the dashboard.
    """
    return load_benchmark_tri(warn_if_stale=False)


TRI_UI_INSTRUCTIONS = (
    "Click <strong>Refresh all data</strong> at the top: it fetches the new sessions from niftyindices.com. "
    "If that site's bot protection blocks the refresh (the header then shows a warning), try again later or "
    + TRI_REDOWNLOAD_INSTRUCTIONS[0].lower() + TRI_REDOWNLOAD_INSTRUCTIONS[1:]
)


def render_tri_staleness_banner(tri: pd.DataFrame, staleness: Optional[TriStaleness]) -> None:
    """Show a warning banner when the TRI benchmark is missing or trails the analysis end date."""
    if tri.empty:
        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>TRI benchmark data unavailable.</strong> No Nifty 500 TRI CSV could be loaded.
                {TRI_UI_INSTRUCTIONS}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif staleness is not None and staleness.is_stale:
        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>TRI benchmark data is stale.</strong> Last available date:
                {staleness.last_date:%d-%b-%Y}. This is {staleness.trading_days_behind} trading days behind
                the analysis end date ({staleness.reference_date:%d-%b-%Y}). {TRI_UI_INSTRUCTIONS}
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_fundamentals_summary(file_mtimes: tuple, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch and parse fundamental quality, return, leverage, and cash flow metrics
    directly from individual Screener.in company pages with local disk caching
    under data/fundamentals_cache/.
    """
    return get_fundamentals_summary(LOCKED_PORTFOLIO_SYMBOLS, use_cache=not force_refresh)


# -----------------------------------------------------------------------------
# PORTFOLIO TABLE RENDERING
# -----------------------------------------------------------------------------

TIER_CLASSES = {CONVICTION_HIGH: "tier-high", CONVICTION_MODERATE: "tier-moderate", CONVICTION_LOW: "tier-low"}
TIER_LABELS = {CONVICTION_HIGH: "High conviction", CONVICTION_MODERATE: "Moderate conviction",
               CONVICTION_LOW: "Low conviction"}
STOP_METHOD_LABELS = {"atr": f"{STOP_LOSS_ATR_MULTIPLE:g} x ATR", "support": "Below support",
                      "trailed": "Trailed (prev. stop)", "breached": "BREACHED"}


def tier_badge(tier) -> str:
    if tier is None or pd.isna(tier):
        return '<span class="tier tier-none">Unclassified</span>'
    return f'<span class="tier {TIER_CLASSES[tier]}">{TIER_LABELS[tier]}</span>'


def _fmt(value, spec: str, prefix: str = "", suffix: str = "") -> str:
    return "—" if value is None or pd.isna(value) else f"{prefix}{value:{spec}}{suffix}"


def portfolio_table_html(df: pd.DataFrame) -> str:
    """One row per stock: conviction tier plus the eight fields the brief requires."""
    header = (
        "<tr><th>Stock</th><th>Conviction (RRG)</th><th>Price (₹)</th>"
        "<th>Volatility<span class='sub'>annualised</span></th>"
        "<th>Expected return<span class='sub'>CAPM, annual · 3-month</span></th>"
        "<th>Weight<span class='sub'>equal-risk · shares · ₹</span></th>"
        "<th>Stop-loss (₹)<span class='sub'>% below · method</span></th>"
        f"<th>ADX (14)</th><th>RS vs Nifty 500<span class='sub'>{TECHNICAL_RS_LOOKBACK_DAYS} sessions</span></th><th>RSI (14)</th>"
        "<th>Support / Resistance (₹)</th></tr>"
    )
    rows = []
    for _, r in df.iterrows():
        resistance = "ATH / blue sky" if pd.isna(r.get("nearest_resistance")) else _fmt(r["nearest_resistance"], ",.2f")
        method = STOP_METHOD_LABELS.get(r.get("stop_loss_method"), "Unavailable")
        rows.append(
            "<tr>"
            f"<td><strong>{r['symbol']}</strong><span class='sub'>{r['display_name']}</span></td>"
            f"<td>{tier_badge(r.get('conviction_tier'))}</td>"
            f"<td>{_fmt(r['current_price'], ',.2f')}</td>"
            f"<td>{_fmt(r.get('annualized_volatility_pct'), '.2f', suffix='%')}</td>"
            f"<td>{_fmt(r.get('capm_expected_return_pct'), '.2f', suffix='%')}"
            f"<span class='sub'>{_fmt(r.get('capm_3m_return_pct'), '.2f', suffix='%')} over 3 months</span></td>"
            f"<td>{_fmt(r.get('weight_pct'), '.2f', suffix='%')}"
            f"<span class='sub'>{_fmt(r.get('shares'), ',.0f')} sh · {_fmt(r.get('invested_inr'), ',.0f', prefix='₹')}</span></td>"
            f"<td>{_fmt(r.get('stop_loss_price'), ',.2f')}"
            f"<span class='sub'>{_fmt(r.get('stop_loss_pct_below_current'), '.2f', suffix='%')} · {method}</span></td>"
            f"<td>{_fmt(r['latest_adx'], '.2f')}</td>"
            f"<td>{_fmt(r['rs_score_vs_nifty500'], '+.2f', suffix=' pp')}</td>"
            f"<td>{_fmt(r['latest_rsi'], '.2f')}</td>"
            f"<td>{_fmt(r['nearest_support'], ',.2f')} / {resistance}</td>"
            "</tr>"
        )
    return f"<div class='pf-wrap'><table class='pf-table'><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></div>"


# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Indian Equity Portfolio Management Terminal</div>
        <div class="app-subtitle">
            Coursework: Security Analysis & Portfolio Management (SAPM) / Derivatives &bull;
            IIM Bodh Gaya &bull; Infrastructure & Capex Portfolio Universe
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DEPLOYMENT: local (full control) or Streamlit Community Cloud (shared with the group)
# -----------------------------------------------------------------------------

def _secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 - no secrets file when run locally
        return default


# Streamlit Community Cloud runs apps from /mount/src/<repo>; the DEPLOYMENT secret can force either mode
ON_CLOUD = str(_secret("DEPLOYMENT", "cloud" if Path(__file__).resolve().as_posix().startswith("/mount/src")
                       else "local")).lower() == "cloud"
GITHUB_TOKEN = _secret("GITHUB_TOKEN")
EDIT_PASSWORD = _secret("EDIT_PASSWORD")


REFRESH_HINT = ("The web app updates automatically each weekday evening after the NSE close." if ON_CLOUD
                else "Click Refresh all data to update.")


def can_edit() -> bool:
    """Locally anyone at the keyboard can record trades; on the shared web app only after the edit password."""
    return (not ON_CLOUD) or bool(st.session_state.get("edit_unlocked"))


# The P&L outputs follow the trade ledger: if the ledger changed since they were computed (a trade saved
# on the web app, then the app reloaded from GitHub), recompute them once
if TRADES_CSV.exists():
    try:
        _summary = pd.read_csv(TRACKER_SUMMARY_CSV) if TRACKER_SUMMARY_CSV.exists() else pd.DataFrame(columns=["metric"])
        _done = dict(zip(_summary["metric"], _summary.get("value", pd.Series(dtype=object))))
        if _done.get("ledger_sha1") != ledger_fingerprint():
            run_tracker()
    except Exception as _exc:  # noqa: BLE001 - show the stale figures rather than no page
        st.warning(f"The P&L could not be recomputed from the latest trades ({_exc}); showing the last figures.")

# Load data assets
summary_df = load_summary_data(_file_mtime(SUMMARY_OUTPUT_CSV))
ohlcv_df = load_historical_ohlcv(_file_mtime(HISTORICAL_OHLCV_CSV))
tri_df = load_tri_benchmark(_file_mtime(DEFAULT_TRI_CSV_PATH))
risk_df = load_risk_summary(_file_mtime(RISK_SUMMARY_OUTPUT_CSV))
_capm_path = OUTPUT_DIR / "capm_expected_returns.csv"
if not risk_df.empty and _capm_path.exists():
    risk_df = risk_df.merge(pd.read_csv(_capm_path)[["symbol", "capm_expected_return_pct", "capm_3m_return_pct"]],
                            on="symbol", how="left")
rrg_df = load_rrg_data(tuple(_file_mtime(review_table_path(sector)) for sector in SECTOR_SCREENS))

# One row per locked stock: technicals + risk/sizing/stop-loss + RRG, in LOCKED_PORTFOLIO order
portfolio_df = summary_df.copy()
if not risk_df.empty:
    portfolio_df = portfolio_df.merge(
        risk_df.drop(columns=["sector", "current_price"], errors="ignore"), on="symbol", how="left")
portfolio_df = portfolio_df.merge(rrg_df, on="symbol", how="left")
missing_symbols = sorted(set(LOCKED_PORTFOLIO_SYMBOLS) - set(portfolio_df["symbol"]))

# Status of each of the 15: the money is in the holdings (top 8 until a stop-loss replacement), the rest
# wait in the reserve queue in rank order; a stopped-out stock never comes back
_ledger = pd.read_csv(TRADES_CSV) if TRADES_CSV.exists() else None
HELD = held_stocks(_ledger)
SOLD = sold_stocks(_ledger)
RESERVE_QUEUE = [s for s in LOCKED_PORTFOLIO_SYMBOLS if s not in HELD and s not in SOLD]


def _status(symbol: str) -> str:
    if symbol in HELD:
        return "Invested"
    if symbol in SOLD:
        return "Sold (stop-loss)"
    return f"Reserve #{RESERVE_QUEUE.index(symbol) + 1}" if symbol in RESERVE_QUEUE else "—"


portfolio_df["status"] = portfolio_df["symbol"].map(_status)


def _commit_trades(new_ledger: pd.DataFrame, message: str, refresh: bool) -> None:
    """Save the ledger, push it to GitHub for the evening alerts, recompute the P&L, and (for a change of
    holdings) start a full refresh so the weights, stops and hedge follow the new stocks."""
    problems = save_ledger(new_ledger)
    if problems:
        st.error("Not saved:\n\n" + "\n".join(f"- {p}" for p in problems))
        return
    if GITHUB_TOKEN:
        pushed = publish_ledger_via_api(message, GITHUB_TOKEN)
    elif ON_CLOUD:
        pushed = ("NOT saved to GitHub (no GITHUB_TOKEN secret): the web app will forget this trade when it "
                  "restarts. Add the secret, then save again.")
    else:
        pushed = publish_ledger(message)
    try:
        run_tracker()
    except Exception as exc:  # noqa: BLE001 - the ledger is saved; the P&L catches up at the next refresh
        st.warning(f"Trades saved, but recomputing the P&L failed ({exc}); it will update at the next refresh.")
    follow_up = ""
    if refresh and not ON_CLOUD:
        start_background_refresh()
        follow_up = " A full refresh has started so the weights, stops and hedge follow the new holdings."
    elif refresh:
        follow_up = (" The weights, stop-loss and hedge of the new stock appear after this evening's automatic "
                     "refresh (about 7:30 pm IST).")
    st.cache_data.clear()
    st.session_state["trades_saved"] = f"Trades saved. {pushed}{follow_up}"
    st.rerun()


# -----------------------------------------------------------------------------
# DATA FRESHNESS & REFRESH (always visible under the header)
# -----------------------------------------------------------------------------

def _fmt_duration(seconds) -> str:
    seconds = int(round(seconds or 0))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


@st.fragment(run_every=3)
def render_data_freshness() -> None:
    """
    How current the data is, plus the refresh control and live progress. Re-runs every 3 seconds
    (only this section), so a refresh started here, in another tab, or before a reload is tracked
    until it finishes; the whole page then reloads with the new data.
    """
    price_date = pd.Timestamp(ohlcv_df["DATE1"].max()).date() if not ohlcv_df.empty else None
    fund_dates = pd.concat([pd.read_csv(review_table_path(s), usecols=["fundamentals_as_of"])
                            for s in SECTOR_SCREENS if review_table_path(s).exists()], ignore_index=True)
    fund_date = pd.to_datetime(fund_dates["fundamentals_as_of"]).max().date() if not fund_dates.empty else None
    tri_last = pd.Timestamp(tri_df["Date"].max()).date() if not tri_df.empty else None
    manifest = load_manifest()
    state = refresh_state(manifest)

    # A refresh finished since this page loaded its data (here or in another tab): reload everything
    finished = (manifest or {}).get("finished") if state in ("ok", "failed") else None
    if "refresh_seen" not in st.session_state:
        st.session_state["refresh_seen"] = finished
    elif finished and finished != st.session_state["refresh_seen"]:
        st.session_state["refresh_seen"] = finished
        st.session_state["refresh_completed_here"] = finished  # keep the success banner for this refresh
        st.cache_data.clear()
        st.rerun(scope="app")

    expected = latest_expected_session()
    behind = int(np.busday_count(price_date + timedelta(days=1), expected + timedelta(days=1))) \
        if price_date and expected > price_date else 0

    info_col, button_col = st.columns([5, 1.3])
    with info_col:
        st.markdown(
            f"<div style='font-size:0.85rem;padding-top:0.45rem;'><strong>Data as of</strong> &bull; "
            f"Prices through <strong>{price_date:%d-%b-%Y}</strong> &bull; "
            f"Fundamentals fetched <strong>{fund_date:%d-%b-%Y}</strong> &bull; "
            f"Nifty 500 TRI through <strong>{tri_last:%d-%b-%Y}</strong></div>"
            if price_date and fund_date and tri_last else f"<div>Some pipeline outputs are missing. {REFRESH_HINT}</div>",
            unsafe_allow_html=True,
        )
    with button_col:
        if ON_CLOUD:
            st.caption("Refreshed automatically every weekday evening after the NSE close (about 7:30 pm IST).")
        elif st.button("Refresh all data", disabled=state == "running", width="stretch",
                     help="Re-downloads NSE prices, Screener.in fundamentals and new Nifty 500 TRI sessions, "
                          "then rebuilds every table and chart (about 5-10 minutes; needs internet). It runs in "
                          "the background: you can keep using or close this page."):
            start_background_refresh()
            manifest = load_manifest()
            state = refresh_state(manifest)

    if state == "running":
        prog = progress_summary(manifest)
        if prog["remaining_seconds"] is None:
            eta = "time left unknown until one refresh has completed"
        elif prog["overrunning"]:
            eta = (f"this step is taking longer than last time (the website may be slow or retrying); "
                   f"about {_fmt_duration(prog['remaining_seconds'])} for the steps after it")
        else:
            eta = f"about {_fmt_duration(prog['remaining_seconds'])} left (estimate from the last refresh)"
        st.progress(prog["fraction"], text=(
            f"Refreshing data: step {prog['step']} of {prog['total']} ({prog['step_name']}), "
            f"{_fmt_duration(prog['step_elapsed'])} in this step; {eta}. The page updates by itself when it is done."))
        with st.expander("Show live log"):
            st.code("\n".join(read_log_tail(20)) or "Starting...", language=None)
    elif state == "stalled":
        st.error(f"The refresh started at {pd.Timestamp(manifest['started']):%H:%M} IST stopped responding "
                 f"(no progress for over a minute) and will not finish; the app or computer was probably closed "
                 f"or restarted. {REFRESH_HINT}")
    elif state == "failed":
        st.error(f"The last data refresh failed at step: {manifest.get('failed_step')} "
                 f"(finished {pd.Timestamp(manifest['finished']):%d-%b %H:%M} IST). Some tables may be from different "
                 f"runs. {REFRESH_HINT}")
        with st.expander("Show error details"):
            st.code("\n".join(manifest.get("error_tail") or read_log_tail(20)), language=None)
    elif state == "ok":
        took = sum(step["seconds"] for step in manifest.get("steps", []))
        message = (f"Last refresh completed {pd.Timestamp(manifest['finished']):%d-%b-%Y %H:%M} IST: "
                   f"all {len(manifest.get('steps', []))} steps OK in {_fmt_duration(took)}.")
        if st.session_state.get("refresh_completed_here") == manifest.get("finished"):
            st.success("Refresh complete. " + message + " The tables below now show the new data.")
        else:
            st.caption(message)
        for warning in manifest.get("warnings", []):
            st.warning(f"Last refresh: {warning}")
    if state != "running" and behind > 0:
        st.warning(f"Prices are {behind} trading day{'s' if behind > 1 else ''} old (latest expected session: "
                   f"{expected:%d-%b-%Y}; exchange holidays are not modelled). {REFRESH_HINT}")


render_data_freshness()

# TRI must cover the period the last pipeline run analysed (its latest price date)
_last_price_date = ohlcv_df["DATE1"].max() if not ohlcv_df.empty else pd.NaT
analysis_end_date = _last_price_date.date() if pd.notna(_last_price_date) else date.today()
tri_staleness = assess_tri_staleness(tri_df, analysis_end_date)

# -----------------------------------------------------------------------------
# 5 TABS NAVIGATION
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_output_csv(name: str, file_mtime: float) -> pd.DataFrame:
    """One CSV from output/ (empty if missing). file_mtime is only a cache key (see _file_mtime)."""
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _out(name: str) -> pd.DataFrame:
    return load_output_csv(name, _file_mtime(OUTPUT_DIR / name))


def _fit_height(df: pd.DataFrame) -> int:
    """Table height showing every row without an inner scrollbar."""
    return 38 + 35 * len(df)


tab_overview, tab_fundamentals, tab_technicals, tab_risk, tab_performance = st.tabs([
    "Portfolio Overview",
    "Fundamentals",
    "Technicals",
    "Risk & Hedging",
    "Performance",
])

# =============================================================================
# TAB 1: PORTFOLIO OVERVIEW
# =============================================================================
with tab_overview:
    # 0. The bottom line: did the Rs 1 crore make money? (tracker.py, from the snapshot close)
    tracker_summary = _out("tracker_summary.csv")
    def _inr(x: float) -> str:
        """Indian digit grouping: 1,08,74,323."""
        neg, x = x < 0, abs(round(x))
        s = str(int(x))
        head, tail = s[:-3], s[-3:]
        while len(head) > 2:
            tail = head[-2:] + "," + tail if tail else head[-2:]
            head = head[:-2]
        out = (head + "," + tail) if head else tail
        return ("-₹" if neg else "₹") + out

    if tracker_summary.empty:
        perf_q = _out("performance_summary.csv")
        rf_annual = float(perf_q["risk_free_annualised_pct"].iloc[-1]) if not perf_q.empty else float("nan")
        hurdle = PRINCIPAL_INR * ((1 + rf_annual / 100) ** 0.25 - 1)
        at_risk = (float(((risk_df["current_price"] - risk_df["stop_loss_price"]).clip(lower=0) * risk_df["shares"]).sum())
                   if not risk_df.empty else float("nan"))
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #1D4ED8;">
                <div class="metric-title">The ₹1 crore: did we make money?</div>
                <div class="metric-value">Tracking starts at the {pd.Timestamp(EVALUATION_START_DATE):%d-%b-%Y} close</div>
                <div class="metric-sub">The first figures appear after the refresh on that evening. The bar to clear: a liquid fund
                would earn about {_inr(hurdle)} on the ₹1 crore in 3 months. The most the stop-losses let us lose, if every stock
                hit its stop at today's prices: {_inr(at_risk)}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        t = dict(zip(tracker_summary["metric"], tracker_summary["value"]))
        pnl = float(t["pnl_inr"])
        colour = "#15803D" if pnl >= 0 else "#B91C1C"
        verb = "made" if pnl >= 0 else "lost"
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid {colour};">
                <div class="metric-title">The ₹1 crore since the {pd.Timestamp(t['snapshot_date']):%d-%b-%Y} close
                (as of {pd.Timestamp(t['as_of']):%d-%b-%Y})</div>
                <div class="metric-value" style="font-size:2rem;">{_inr(float(t['value_inr']))}
                <span style="color:{colour}; font-size:1.3rem;">&nbsp;We {verb} {_inr(abs(pnl))} ({float(t['pnl_pct']):+.2f}%)</span></div>
                <div class="metric-sub">Stocks {_inr(float(t['stocks_inr']))} &bull; Nifty puts {_inr(float(t['options_inr']))} &bull;
                cash {_inr(float(t['cash_inr']))} (overnight rate)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Same ₹1 crore in the Nifty 500", _inr(float(t["nifty500_value_inr"])),
                  f"we are {_inr(float(t['vs_nifty500_inr']))} ahead" if float(t["vs_nifty500_inr"]) >= 0
                  else f"we are {_inr(-float(t['vs_nifty500_inr']))} behind")
        l2.metric("Same ₹1 crore in a liquid fund", _inr(float(t["liquid_fund_value_inr"])),
                  f"we are {_inr(float(t['vs_liquid_fund_inr']))} ahead" if float(t["vs_liquid_fund_inr"]) >= 0
                  else f"we are {_inr(-float(t['vs_liquid_fund_inr']))} behind")
        l3.metric("Still at risk if every stop is hit", _inr(float(t["at_risk_to_stops_inr"])),
                  help="Sum over the stocks of (close - stop-loss) x shares; a gap below a stop can lose more.")
        xirr_val = t.get("xirr_pct")
        l4.metric("XIRR (annualised)", f"{float(xirr_val):.1f}%" if pd.notna(xirr_val) and str(xirr_val) != "nan"
                  else "after 30 days", help="Annualising only a few days' return is meaningless, so it is shown from day 30.")
        roll_df = _out("hedge_roll.csv")
        if not roll_df.empty:
            rv = dict(zip(roll_df["field"], roll_df["value"]))
            trades = roll_df.loc[roll_df["field"] == "trade", "value"].tolist()
            if rv.get("status") == "TRIGGERED":
                st.warning(
                    f"**Profit lock triggered:** the portfolio is up {float(rv['pnl_pct']):+.2f}% (trigger "
                    f"+{float(rv['trigger_pct']):g}%). Roll the puts up so a crash cannot take the gain back:\n\n"
                    + ("\n".join(f"- {x}" for x in trades) if trades else "- no trade needed")
                    + (f"\n\n_{rv['note']}_" if isinstance(rv.get("note"), str) else "")
                    + f"\n\nNet cost {_inr(float(rv['net_cost_inr']))} from cash ({_inr(float(rv['cash_inr']))} available"
                    + ("" if str(rv.get("cash_sufficient")) == "True" else "; NOT enough: use fewer lots")
                    + "). Prices: NSE settlement today."
                )
                roll_rows = roll_trades(roll_df, pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d"))
                if not roll_rows.empty and not can_edit():
                    st.caption("To record the roll, unlock editing in the Trade ledger section below.")
                if not roll_rows.empty and can_edit():
                    with st.form("record_roll"):
                        st.markdown("**Done the roll? Record it** (edit the prices or quantities to your actual fills):")
                        edited_roll = st.data_editor(roll_rows, hide_index=True, width="stretch",
                                                     disabled=["instrument", "action", "note"])
                        if st.form_submit_button("Record the put roll", type="primary"):
                            _commit_trades(pd.concat([pd.read_csv(TRADES_CSV), edited_roll], ignore_index=True),
                                           "Profit-lock put roll", refresh=False)
            elif rv.get("status") == "waiting":
                st.caption(f"Profit lock: at +{float(rv['trigger_pct']):g}% the puts are rolled up to about "
                           f"{TAIL_HEDGE_OTM_PCT:g}% below the Nifty (now {float(rv['pnl_pct']):+.2f}%, "
                           f"{float(rv['gap_to_trigger_pp']):.2f} pp to go).")
            elif rv.get("status") == "covered":
                st.caption(f"Profit lock: the portfolio is up {float(rv['pnl_pct']):+.2f}% but the Nifty has not risen enough "
                           "to raise the put strike; the gain is stock-specific and the trailing stops protect it. "
                           "The puts already cover the portfolio.")
            elif rv.get("status") == "done":
                st.caption("Profit lock: the puts have been rolled up (recorded in the ledger).")
            elif rv.get("status") == "no prices":
                st.warning("Profit lock triggered, but NSE option prices for today are not available yet; refresh later.")
        plan = pd.read_csv(REPLACEMENT_PLAN_CSV) if REPLACEMENT_PLAN_CSV.exists() else pd.DataFrame()
        if not plan.empty:  # stop hits not yet recorded in the ledger
            lines = []
            for _, r in plan.iterrows():
                sell = (f"SELL {int(r['sell_shares']):,} {r['sell']} (closed ₹{r['sell_close']:,.2f}, stop "
                        f"₹{r['stop_loss_price']:,.2f}; about {_inr(r['proceeds_inr'])})")
                if isinstance(r.get("buy"), str) and r["buy"]:
                    buy = (f"BUY {int(r['buy_shares']):,} {r['buy']} (reserve, rank #{int(r['buy_rank'])} of 15; "
                           f"₹{r['buy_close']:,.2f}, about {_inr(r['buy_inr'])}); {_inr(r['cash_left_inr'])} stays in cash")
                else:
                    buy = "no reserve stock passes the selection rule today: the money waits in the liquid fund"
                note = f" _{r['note']}_" if isinstance(r.get("note"), str) and r["note"] else ""
                lines.append(f"- {sell} → {buy}.{note}")
            st.error(f"**Stop-loss hit: {', '.join(plan['sell'])}.** Replace it from the reserve list:\n\n"
                     + ("\n".join(lines) if lines else "- see output/replacement_plan.csv")
                     + "\n\nPrices are today's closes (the fill will be tomorrow's price).")
            if not plan.empty and not can_edit():
                st.caption("To record these trades, unlock editing in the Trade ledger section below.")
            if not plan.empty and can_edit():
                with st.form("record_replacement"):
                    st.markdown("**Done the trades? Record them here** (enter the actual fill prices):")
                    day = st.date_input("Trade date", value=pd.Timestamp.now(tz="Asia/Kolkata").date())
                    fills = {}
                    cols = st.columns(2)
                    for _, r in plan.iterrows():
                        fills[r["sell"]] = cols[0].number_input(f"Sold {r['sell']} at (₹)", value=float(r["sell_close"]),
                                                                min_value=0.01, format="%.2f")
                        if isinstance(r.get("buy"), str) and r["buy"]:
                            fills[r["buy"]] = cols[1].number_input(f"Bought {r['buy']} at (₹)",
                                                                   value=float(r["buy_close"]), min_value=0.01,
                                                                   format="%.2f")
                    if st.form_submit_button("Record these trades", type="primary"):
                        new = replacement_trades(plan, day.isoformat(), fills)
                        _commit_trades(pd.concat([pd.read_csv(TRADES_CSV), new], ignore_index=True),
                                       f"Stop-loss replacement {day.isoformat()}: " +
                                       ", ".join(f"{a} {i}" for a, i in zip(new["action"], new["instrument"])),
                                       refresh=True)
        track = _out("tracker_daily.csv")
        if len(track) > 1:
            track["date"] = pd.to_datetime(track["date"])
            fig_t = go.Figure()
            for col, name, color in [("total_inr", "Our portfolio", "#B45309"), ("nifty500_inr", "Nifty 500 TRI", "#1D4ED8"),
                                     ("liquid_fund_inr", "Liquid fund", "#64748B")]:
                fig_t.add_trace(go.Scatter(x=track["date"], y=track[col] / 1e5, name=name, line=dict(color=color, width=2),
                                           hovertemplate="%{x|%d-%b}: ₹%{y:,.2f} L<extra>" + name + "</extra>"))
            fig_t.update_layout(height=280, template="plotly_white", hovermode="x unified", yaxis_title="₹ lakh",
                                margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_t, width="stretch")
        pos = _out("tracker_positions.csv")
        if not pos.empty:
            with st.expander("Holdings: cost, value and profit per stock"):
                st.dataframe(pos.rename(columns={
                    "symbol": "Stock", "shares": "Shares", "avg_cost": "Avg cost (₹)", "close": "Close (₹)",
                    "value_inr": "Value (₹)", "pnl_inr": "P&L (₹)", "pnl_pct": "P&L %", "stop_loss_price": "Stop (₹)",
                    "at_risk_to_stop_inr": "At risk to stop (₹)", "stop_breached": "Stop breached",
                    "near_stop": f"Within {NEAR_STOP_PCT:g}% of stop", "pct_above_stop": "% above stop"}),
                    hide_index=True, width="stretch", height=_fit_height(pos))
        if not pos.empty and "near_stop" in pos and pos["near_stop"].any():
            near = pos[pos["near_stop"]]
            st.warning("**Close to the stop-loss:** " + "; ".join(
                f"{r.symbol} closed ₹{r.close:,.2f}, only {r.pct_above_stop:.1f}% above its stop ₹{r.stop_loss_price:,.2f}"
                for r in near.itertuples()) + ". If it closes at or below the stop, the reserve list replaces it.")

    if st.session_state.get("trades_saved"):
        st.success(st.session_state.pop("trades_saved"))
    if tracker_summary.empty and not risk_df.empty:
        near_pre = risk_df[risk_df["stop_loss_pct_below_current"] <= NEAR_STOP_PCT]
        if not near_pre.empty:
            st.warning("**Stop-loss very close:** " + "; ".join(
                f"{r.symbol} (stop {r.stop_loss_pct_below_current:.1f}% below the price)" for r in near_pre.itertuples()))

    # 0a. Trade ledger: record real fills and any trade without editing files
    with st.expander("Trade ledger: record real fill prices and other trades"
                     + ("" if TRADES_CSV.exists() else " (created at the snapshot close)")):
        if not TRADES_CSV.exists():
            st.caption(f"The ledger is created automatically at the {pd.Timestamp(EVALUATION_START_DATE):%d-%b-%Y} close "
                       "with the 8 invested stocks at that day's closing prices and the Nifty puts. After that, correct "
                       "the prices (and share counts) here to what you actually paid.")
        elif not can_edit():
            st.caption("The group can see the trades; recording or correcting them needs the edit password.")
            st.dataframe(pd.read_csv(TRADES_CSV), hide_index=True, width="stretch")
            if EDIT_PASSWORD:
                pw = st.text_input("Edit password", type="password", key="edit_pw")
                if pw:
                    if pw == EDIT_PASSWORD:
                        st.session_state["edit_unlocked"] = True
                        st.rerun()
                    st.error("Wrong password.")
            else:
                st.info("Editing is off on the web app: no EDIT_PASSWORD secret is set.")
        else:
            st.caption("Edit a cell to correct a price or quantity (for example your actual buy prices on the snapshot "
                       "day), or add a row at the bottom for a new trade. Actions are BUY or SELL; stocks by NSE symbol, "
                       "puts as 'NIFTY 2026-12-29 22000 PE'. Saving recomputes the P&L and pushes the ledger to GitHub.")
            ledger_now = pd.read_csv(TRADES_CSV)
            edited = st.data_editor(
                ledger_now, num_rows="dynamic", hide_index=True, width="stretch", key="ledger_editor",
                column_config={
                    "action": st.column_config.SelectboxColumn("action", options=["BUY", "SELL"], required=True),
                    "quantity": st.column_config.NumberColumn("quantity", min_value=1, step=1, required=True),
                    "price": st.column_config.NumberColumn("price (₹)", min_value=0.01, format="%.2f", required=True),
                    "date": st.column_config.TextColumn("date (YYYY-MM-DD)", required=True),
                })
            changed = not edited[LEDGER_COLUMNS].astype(str).reset_index(drop=True).equals(
                ledger_now[LEDGER_COLUMNS].astype(str).reset_index(drop=True))
            if st.button("Save ledger", disabled=not changed, type="primary" if changed else "secondary"):
                try:
                    holdings_changed = set(held_stocks(edited.dropna(how="all"))) != set(held_stocks(ledger_now))
                except Exception:  # noqa: BLE001 - a half-filled row; save_ledger reports it
                    holdings_changed = False
                _commit_trades(edited, "Ledger edited on the dashboard", refresh=holdings_changed)

    # 0c. One-tap update for the group chat
    with st.expander("WhatsApp update for the group"):
        plan_now = pd.read_csv(REPLACEMENT_PLAN_CSV) if REPLACEMENT_PLAN_CSV.exists() else pd.DataFrame()
        roll_now = _out("hedge_roll.csv")
        text = whatsapp_update(
            dict(zip(tracker_summary["metric"], tracker_summary["value"])) if not tracker_summary.empty else None,
            _out("tracker_positions.csv"), plan_now,
            dict(zip(roll_now["field"], roll_now["value"])) if not roll_now.empty else None, risk_df)
        st.code(text, language=None)
        st.caption("Use the copy icon at the top right of the box, then paste into WhatsApp.")

    # 0b. Q2 results calendar: the days a stock can gap through its stop
    lc = _out("locked_portfolio_runup_catalyst_check.csv")
    if not lc.empty and "q2_results_date" in lc.columns:
        cal = lc[["symbol", "q2_results_date", "q2_results_basis"]].dropna(subset=["q2_results_date"]).copy()
        cal["date"] = pd.to_datetime(cal["q2_results_date"])
        today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
        cal["days"] = (cal["date"] - today_ist).dt.days
        if not risk_df.empty:
            cal = cal.merge(risk_df[["symbol", "stop_loss_price", "stop_loss_pct_below_current"]], on="symbol", how="left")
        cal = cal.sort_values("date")
        soon = cal[(cal["q2_results_basis"] != "reported") & cal["days"].between(0, 7)]
        for _, r in soon.iterrows():
            st.warning(f"**{r['symbol']} reports Q2 results {'today' if r['days'] == 0 else f'in {r.days} day(s)'}** "
                       f"({r['date']:%a %d-%b}, {'announced' if r['q2_results_basis'] == 'announced' else 'ESTIMATE from last year'}). "
                       f"A bad result can gap the price straight through its stop at ₹{r['stop_loss_price']:,.2f}.")
        n_est = int((cal["q2_results_basis"] == "estimate").sum())
        with st.expander(f"Q2 results calendar: next {cal.loc[cal['days'] >= 0, 'symbol'].head(1).tolist()[0] if (cal['days'] >= 0).any() else '—'}"
                         f" ({n_est} of {len(cal)} dates are estimates until NSE announces them)", expanded=False):
            basis_label = {"announced": "Announced (NSE)", "estimate": "Estimate: last year's date, same weekday",
                           "reported": "Reported"}
            show_cal = pd.DataFrame({
                "Stock": cal["symbol"],
                "Q2 results": cal["date"].dt.strftime("%a %d-%b-%Y"),
                "Basis": cal["q2_results_basis"].map(lambda b: basis_label.get(b, b)),
                "Days away": cal["days"].map(lambda d: "done" if d < 0 else ("today" if d == 0 else f"{d}")),
                "Stop (₹)": cal.get("stop_loss_price", pd.Series(dtype=float)).map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—"),
                "Stop below price": cal.get("stop_loss_pct_below_current", pd.Series(dtype=float)).map(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "—"),
            })
            st.dataframe(show_cal, hide_index=True, width="stretch", height=_fit_height(show_cal))
            st.caption("From NSE board-meeting announcements (re-checked every refresh). Until a company announces, the date "
                       "is last year's September-quarter results date moved to the same weekday this year, labelled as an "
                       "estimate. Results often come after market hours or on a weekend; the price reacts at the next "
                       "session, and a gap can go through a stop.")

    st.write("")

    # 1. Summary Metric Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Locked list</div>
                <div class="metric-value">{len(LOCKED_PORTFOLIO)} tracked &bull; {len(HELD)} invested</div>
                <div class="metric-sub">{len(RESERVE_QUEUE)} in reserve: a stopped-out stock is replaced by the next one</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        sector_counts = pd.Series([LOCKED_PORTFOLIO[s_]["sector"] for s_ in HELD if s_ in LOCKED_PORTFOLIO]).value_counts()
        sector_breakdown = " &bull; ".join(f"{sec} ({sector_counts.get(sec, 0)})" for sec in SECTOR_SCREENS)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Sectors of the invested stocks</div>
                <div class="metric-value">{len(SECTOR_SCREENS)} Sectors</div>
                <div class="metric-sub">{sector_breakdown}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Benchmark Series</div>
                <div class="metric-value">Nifty 500</div>
                <div class="metric-sub">Price index: NSE official daily closes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        invested_total = float(portfolio_df["invested_inr"].sum()) if "invested_inr" in portfolio_df else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Capital Deployed</div>
                <div class="metric-value">₹{invested_total / 1e5:,.2f} L of ₹{PRINCIPAL_INR / 1e5:,.0f} L</div>
                <div class="metric-sub">{invested_total / PRINCIPAL_INR * 100:.2f}% invested &bull; cash ₹{PRINCIPAL_INR - invested_total:,.0f} ({100 - EQUITY_ALLOCATION_PCT:g}% hedge budget + rounding)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # 2. Conviction tier summary
    tier_counts = portfolio_df["conviction_tier"].value_counts()
    st.markdown(
        " &nbsp; ".join(
            f'{tier_badge(t)} <span style="font-size:0.8rem;">{tier_counts.get(t, 0)} stock(s)</span>'
            for t in (CONVICTION_HIGH, CONVICTION_MODERATE, CONVICTION_LOW)
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Conviction tier from the Relative Rotation Graph: **High** = LEADING vs both the Nifty 500 and the "
        "equal-weighted sector average; **Moderate** = LEADING in one view only, or IMPROVING in either; "
        "**Low conviction** = WEAKENING or LAGGING in both views."
    )
    if missing_symbols:
        st.warning(f"No pipeline data for {', '.join(missing_symbols)}. Run `python main.py` to refresh the outputs.")

    # 3. Sector-grouped, one row per stock with every field the brief requires
    st.markdown("### Portfolio Constituents by Sector")
    st.caption(
        "Prices, technicals and risk figures from daily NSE Bhavcopy files (last pipeline run). "
        f"Stop-loss: price - {STOP_LOSS_ATR_MULTIPLE:g} x ATR({STOP_LOSS_ATR_PERIOD}), moved just below a support level "
        f"up to {STOP_LOSS_SUPPORT_BAND_ATR:g} ATR beyond it; the method column shows which applied. "
        f"Weights: equal risk contribution (each stock carries the same share of portfolio variance), "
        f"bounded {WEIGHT_MIN_PCT:g}-{WEIGHT_MAX_PCT:g}%, applied to the {EQUITY_ALLOCATION_PCT:g}% equity sleeve of the "
        f"₹{PRINCIPAL_INR / 1e7:g} crore; the other {100 - EQUITY_ALLOCATION_PCT:g}% is the hedge budget. Shares are whole "
        "shares at the latest close (the shares actually held once invested). Expected return = CAPM: "
        f"risk-free + beta × {MARKET_RISK_PREMIUM_PCT:g}% India equity risk premium ({MARKET_RISK_PREMIUM_SOURCE})."
    )

    sector_badge_classes = {
        "Cement": "badge-cement",
        "Capital Goods": "badge-capital-goods",
        "Power": "badge-power",
    }
    invested_df = portfolio_df[portfolio_df["status"] == "Invested"]
    for sec in SECTOR_SCREENS:
        sec_df = invested_df[invested_df["sector"] == sec]
        if sec_df.empty:
            continue
        st.markdown(
            f"""
            <div style="margin-top: 1.2rem; margin-bottom: 0.2rem;">
                <span class="{sector_badge_classes.get(sec, 'badge-cement')}">{sec.upper()} ({len(sec_df)} STOCK{'S' if len(sec_df) != 1 else ''})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(portfolio_table_html(sec_df), unsafe_allow_html=True)

    # 3b. The reserve: tracked, no money yet; the first one still passing the selection rule replaces
    # a stopped-out holding
    reserve_df = portfolio_df[portfolio_df["status"] != "Invested"].copy()
    if not reserve_df.empty:
        ranking_path = OUTPUT_DIR / "selection_ranking.csv"
        ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame(columns=["symbol", "conviction"])
        rk = ranking.set_index("symbol")
        passes = reserve_df["symbol"].map(lambda s_: s_ in rk.index and rk.loc[s_, "conviction"] in SELECTION_CONVICTION_TIERS)
        st.markdown(f"### Reserve list ({len(RESERVE_QUEUE)} stocks, tracked, no money yet)")
        st.caption(f"The money is in the top {INVESTED_COUNT} of the {len(LOCKED_PORTFOLIO_SYMBOLS)}. When a holding "
                   "closes at or below its stop-loss, its sale proceeds buy the first reserve stock that still passes "
                   "the selection rule that day (hard fundamentals, bullish trend with DI gap ≥ 2, one year of prices, "
                   "RRG conviction High or Moderate). A stock that has been sold never comes back.")
        show_res = pd.DataFrame({
            "Queue": reserve_df["status"],
            "Stock": reserve_df["symbol"],
            "Name": reserve_df["display_name"],
            "Sector": reserve_df["sector"],
            "Price (₹)": reserve_df["current_price"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—"),
            "6-month RS (skip 1m)": reserve_df["symbol"].map(
                lambda s_: f"{rk.loc[s_, 'rs_6m_skip1m']:+.1f} pp" if s_ in rk.index else "—"),
            "Trend (DI gap)": reserve_df["di_gap"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—"),
            "Conviction": reserve_df["conviction_tier"].fillna("—"),
            "Would be bought today?": passes.map({True: "Yes", False: "No: fails the rule today"}),
        })
        st.dataframe(show_res, hide_index=True, width="stretch", height=_fit_height(show_res))

    # 4. Screen exceptions: locked stocks that do not pass every screen (from the review tables)
    exceptions = []
    for _, r in portfolio_df.iterrows():
        if r.get("technically_attractive") == False:  # noqa: E712
            exceptions.append(
                f"**{r['symbol']}** does not pass the technical screen (RS vs Nifty 500 "
                f"{r['rs_score_vs_nifty500']:+.2f} pp, needs > {TECHNICAL_RS_MARGIN_PP:+g} pp; trend {str(r['trend_direction']).split(' (')[0]}); "
                f"conviction tier: {r.get('conviction_tier') or 'Unclassified'}.")
        failed = r.get("fundamentals_failed")
        if isinstance(failed, str) and failed.strip():
            if r.get("hard_fundamentals_pass") == False:  # noqa: E712
                exceptions.append(f"**{r['symbol']}** fails a HARD fundamental rule ({failed}): not acceptable for a "
                                  "3-month holding (pledge, debt, interest cover or size can turn a bad quarter into a crash).")
            else:
                note = (f" Flagged by the OPM-exception check (fails only OPM, ROCE above {HIGH_TURNOVER_ROCE_MIN:g}%)."
                        if r.get("high_turnover_business_flag") == True else "")  # noqa: E712
                exceptions.append(f"**{r['symbol']}** fails SOFT criteria only ({failed}): acceptable for a 3-month "
                                  f"holding, since long-run quality measures are already in the price.{note}")
    if exceptions:
        st.markdown("#### Screen exceptions")
        st.caption("Locked stocks that do not pass every screen, stated so the selection can be defended. HARD rules "
                   "(pledged shares, debt/equity, interest cover, market cap) are never waived; SOFT criteria (ROCE, OPM, "
                   "one year's operating cash flow) can be.")
        st.markdown("\n".join(f"- {e}" for e in exceptions))


# =============================================================================
# TAB 2: FUNDAMENTALS
# =============================================================================
with tab_fundamentals:
    st.markdown("### Fundamental Analysis & Quality Screening")
    st.caption(
        "Fundamental metrics fetched directly from authentic Screener.in company pages (/consolidated/ with standalone fallback) "
        "and cached locally in data/fundamentals_cache/. Evaluates core financial health, capital productivity, leverage, "
        "and cash flow resilience across the locked portfolio constituents."
    )

    # Locked stocks that do not pass every criterion of their sector's safety screen (review tables)
    screen_exceptions = portfolio_df[portfolio_df["fundamentals_failed"].notna()
                                     & (portfolio_df["fundamentals_failed"].astype(str).str.strip() != "")]
    for _, exc in screen_exceptions.iterrows():
        reason = (" SOFT criteria only: acceptable for a 3-month holding." if exc.get("hard_fundamentals_pass") != False  # noqa: E712
                  else " A HARD rule: not acceptable.")
        if exc.get("high_turnover_business_flag") == True:  # noqa: E712
            reason += (f" The OPM-exception check flagged it (fails only OPM, ROCE above {HIGH_TURNOVER_ROCE_MIN:g}%): "
                       "a high-turnover business for which OPM is the wrong yardstick.")
        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>Screen exception: {exc['symbol']}</strong> fails <em>{exc['fundamentals_failed']}</em>
                on the {exc['sector']} safety screen.{reason}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Keyed on the review tables: a data refresh rewrites them after re-fetching fundamentals
    fund_raw_df = load_fundamentals_summary(tuple(_file_mtime(review_table_path(s)) for s in SECTOR_SCREENS))

    if fund_raw_df is None or fund_raw_df.empty:
        st.warning("Fundamentals data unavailable. Please verify network access to Screener.in.")
    else:
        # Build clean formatted display DataFrame
        formatted_rows = []
        for _, row in fund_raw_df.iterrows():
            sym = row["symbol"]
            status = row.get("status", "OK")

            if status != "OK" and pd.isna(row.get("roce")):
                formatted_rows.append({
                    "Symbol": sym,
                    "Company Name": row.get("name", sym),
                    "Sector": row.get("sector", "Other"),
                    "Market Cap (₹ Cr)": "Data Unavailable",
                    "Price (₹)": "Data Unavailable",
                    "ROCE (%)": "Data Unavailable",
                    "3-Yr Avg ROCE (%)": "Data Unavailable",
                    "ROE (%)": "Data Unavailable",
                    "Debt / Equity": "Data Unavailable",
                    "Operating Cash Flow (₹ Cr)": "Data Unavailable",
                    "OPM (%)": "Data Unavailable",
                    "Interest Coverage (x)": "Data Unavailable",
                    "Pledged (%)": "Data Unavailable",
                    "3-Yr Sales Growth (%)": "Data Unavailable",
                    "3-Yr Profit Growth (%)": "Data Unavailable",
                })
            else:
                formatted_rows.append({
                    "Symbol": sym,
                    "Company Name": row.get("name", sym),
                    "Sector": row.get("sector", "Other"),
                    "Market Cap (₹ Cr)": f"₹{row['market_cap']:,.0f} Cr" if pd.notna(row.get("market_cap")) else "Data Unavailable",
                    "Price (₹)": f"₹{row['current_price']:,.2f}" if pd.notna(row.get("current_price")) else "Data Unavailable",
                    "ROCE (%)": f"{row['roce']:.2f}%" if pd.notna(row.get("roce")) else "Data Unavailable",
                    "3-Yr Avg ROCE (%)": f"{row['roce_3yr_avg']:.2f}%" if pd.notna(row.get("roce_3yr_avg")) else "Data Unavailable",
                    "ROE (%)": f"{row['roe']:.2f}%" if pd.notna(row.get("roe")) else "Data Unavailable",
                    "Debt / Equity": f"{row['debt_to_equity']:.2f}" if pd.notna(row.get("debt_to_equity")) else "Data Unavailable",
                    "Operating Cash Flow (₹ Cr)": f"₹{row['operating_cash_flow']:,.0f} Cr" if pd.notna(row.get("operating_cash_flow")) else "Data Unavailable",
                    "OPM (%)": f"{row['opm']:.2f}%" if pd.notna(row.get("opm")) else "Data Unavailable",
                    "Interest Coverage (x)": f"{row['interest_coverage']:.2f}" if pd.notna(row.get("interest_coverage")) else "Data Unavailable",
                    "Pledged (%)": f"{row['pledged_pct']:.2f}%" if pd.notna(row.get("pledged_pct")) else "Data Unavailable",
                    "3-Yr Sales Growth (%)": f"{row['sales_growth_3yr']:+.1f}%" if pd.notna(row.get("sales_growth_3yr")) else "Data Unavailable",
                    "3-Yr Profit Growth (%)": f"{row['profit_growth_3yr']:+.1f}%" if pd.notna(row.get("profit_growth_3yr")) else "Data Unavailable",
                })

        display_fund_df = pd.DataFrame(formatted_rows)
        st.dataframe(display_fund_df, hide_index=True, width="stretch")

    st.write("")
    st.markdown("#### Fundamental Safety Screen (current-year, per sector)")
    st.caption(
        f"The universe is screened technically (RS vs Nifty 500 > {TECHNICAL_RS_MARGIN_PP:+g} pp and a Bullish trend) and against its "
        "sector's fundamental safety screen below. Every threshold is strict; a metric that cannot be read counts "
        "as a failure. Locked stocks that miss a screen are listed as exceptions on the Portfolio Overview tab. "
        "Full per-sector results: output/*_full_review_table.csv."
    )
    for f_col, (sec, spec) in zip(st.columns(len(SECTOR_SCREENS)), SECTOR_SCREENS.items()):
        with f_col:
            st.markdown(f"**{sec}**\n\n" + "\n".join(f"- {label}" for _, _, _, label in spec["criteria"]))


# =============================================================================
# TAB 3: TECHNICALS
# =============================================================================
with tab_technicals:
    st.markdown("### Quantitative Technical Indicators & Price Structure")
    st.caption(
        "Technical indicators computed directly with NumPy/Pandas from official NSE Bhavcopy data. "
        "RSI & ADX use J. Welles Wilder's exact 14-period exponential smoothing. "
        f"RS score measures {TECHNICAL_RS_LOOKBACK_DAYS}-day cumulative percentage-point alpha over the Nifty 500."
    )

    # 1. Full Technical Summary Table with subtle trend_direction color tinting
    tech_table_df = pd.DataFrame({
        "Symbol": portfolio_df["symbol"],
        "Status": portfolio_df["status"],
        "Name": portfolio_df["display_name"],
        "Sector": portfolio_df["sector"],
        "Current Price (₹)": portfolio_df["current_price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
        "RSI (14)": portfolio_df["latest_rsi"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "ADX (14)": portfolio_df["latest_adx"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "Trend Direction": portfolio_df["trend_direction"],
        "DI Gap (+DI − −DI)": portfolio_df["di_gap"].apply(
            lambda x: "—" if pd.isna(x) else f"{x:+.2f}" + (" (thin)" if abs(x) < DI_GAP_THIN_THRESHOLD else "")),
        "RS Spread vs N500": portfolio_df["rs_score_vs_nifty500"].apply(lambda x: f"{x:+.2f} pp" if pd.notna(x) else "—"),
        "RRG vs Nifty 500": portfolio_df["rrg_quadrant_vs_nifty500"].fillna("—"),
        "RRG vs Sector": portfolio_df["rrg_quadrant_vs_sector"].fillna("—"),
        "Conviction": portfolio_df["conviction_tier"].fillna("Unclassified"),
        "Support (₹)": portfolio_df["nearest_support"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
        "Resistance (₹)": portfolio_df["nearest_resistance"].apply(
            lambda x: "ATH / Blue Sky" if pd.isna(x) or x is None else f"₹{x:,.2f}"
        ),
    })

    def color_trend(val):
        """Apply subtle, analyst-style color tinting to trend direction."""
        if isinstance(val, str):
            if "Bullish" in val:
                return "background-color: rgba(34, 197, 94, 0.15); color: #166534; font-weight: 600;"
            elif "Bearish" in val:
                return "background-color: rgba(239, 68, 68, 0.15); color: #991b1b; font-weight: 600;"
        return ""

    # Use Styler.map (supported in pandas 2.1+) with fallback to applymap
    if hasattr(tech_table_df.style, "map"):
        styled_table = tech_table_df.style.map(color_trend, subset=["Trend Direction"])
    else:
        styled_table = getattr(tech_table_df.style, "applymap")(color_trend, subset=["Trend Direction"])
    st.dataframe(styled_table, hide_index=True, width="stretch")

    # RRG scatter plots (static PNGs written by rrg.py / sector_screen.py --review)
    st.markdown("#### Relative Rotation Graphs")
    st.caption(
        f"x = {TECHNICAL_RS_LOOKBACK_DAYS}-session RS (pp); y = RS-Momentum = RS averaged over the last "
        f"{RRG_MOMENTUM_SMOOTHING_DAYS} sessions minus the same average {RRG_MOMENTUM_DAYS} sessions earlier (pp). "
        "Locked stocks are ringed and bold. Regenerate with `python rrg.py --as-of <date>` after a pipeline run."
    )
    st.markdown("##### Rotation over time (weekly tails)")
    tail_view = st.radio("Show", ["Our holdings", "Sector rotation (NSE sectors + our sub-themes)"], horizontal=True,
                         key="rrg_tail_view", label_visibility="collapsed")
    tails_df = _out("rrg_tails_holdings.csv" if tail_view == "Our holdings" else "rrg_tails_sectors.csv")
    if tails_df.empty:
        st.info("RRG tails not found. Press the refresh button (or run `python rrg_tails.py`).")
    else:
        names = list(dict.fromkeys(tails_df["name"]))
        default = names if tail_view == "Our holdings" else [n for n in names if n.startswith("Our ")]
        tc1, tc2 = st.columns([3, 1])
        picked = tc1.multiselect("Tails for", names, default=default, key=f"tails_pick_{tail_view}")
        n_dates = tails_df["date"].nunique()
        weeks = tc2.slider("Tail length (weeks)", 1, max(n_dates - 1, 1), min(4, max(n_dates - 1, 1)),
                           key="tails_weeks")
        dates = sorted(tails_df["date"].unique())[-(weeks + 1):]
        shown = tails_df[tails_df["date"].isin(dates)].dropna(subset=["rs", "momentum"])
        x_span = max(shown["rs"].max() - shown["rs"].min(), 10.0)
        y_span = max(shown["momentum"].max() - shown["momentum"].min(), 6.0)
        fig_rrg = go.Figure()
        for name in names:
            t = shown[shown["name"] == name].sort_values("date")
            if t.empty:
                continue
            head = t.iloc[-1]
            colour = QUADRANT_COLOURS.get(head["quadrant"], "#6b6a63")
            hover = "<b>" + name + "</b><br>%{customdata}<br>RS %{x:.1f} pp, momentum %{y:+.1f} pp<extra></extra>"
            if name in picked and len(t) > 1:
                cx, cy = smooth_path(t["rs"], t["momentum"], x_span, y_span)
                fig_rrg.add_trace(go.Scatter(x=cx, y=cy, mode="lines", showlegend=False, hoverinfo="skip",
                                             line=dict(color=colour, width=2.2)))
                fig_rrg.add_trace(go.Scatter(
                    x=list(t["rs"].iloc[:-1]) + [cx[-2], cx[-1]], y=list(t["momentum"].iloc[:-1]) + [cy[-2], cy[-1]],
                    mode="markers", name=name, showlegend=False,
                    customdata=list(pd.to_datetime(t["date"]).dt.strftime("%d-%b").iloc[:-1]) + ["", pd.Timestamp(head["date"]).strftime("%d-%b")],
                    marker=dict(size=[7] * (len(t) - 1) + [0, 16], color=colour,
                                symbol=["circle"] * (len(t) - 1) + ["circle", "arrow"], angleref="previous",
                                line=dict(color="#FFFFFF", width=1)),
                    hovertemplate=hover))
            else:
                fig_rrg.add_trace(go.Scatter(
                    x=[head["rs"]], y=[head["momentum"]], mode="markers", name=name, showlegend=False,
                    marker=dict(size=8, color=colour, opacity=0.6), customdata=[pd.Timestamp(head["date"]).strftime("%d-%b")],
                    hovertemplate=hover))
            fig_rrg.add_annotation(x=head["rs"], y=head["momentum"], text=name, showarrow=False, xshift=8, yshift=9,
                                   xanchor="left", font=dict(size=11, color="#1f1f1e" if name in picked else "#6b6a63"))
        for label, x, y, xa, ya in [("LEADING", 1, 1, "right", "top"), ("WEAKENING", 1, 0, "right", "bottom"),
                                    ("LAGGING", 0, 0, "left", "bottom"), ("IMPROVING", 0, 1, "left", "top")]:
            fig_rrg.add_annotation(x=x, y=y, xref="paper", yref="paper", text=f"<b>{label}</b>", showarrow=False,
                                   xanchor=xa, yanchor=ya, font=dict(size=14, color=QUADRANT_COLOURS[label]))
        fig_rrg.add_hline(y=0, line_color="#6b6a63", line_width=1)
        fig_rrg.add_vline(x=0, line_color="#6b6a63", line_width=1)
        fig_rrg.update_layout(height=620, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title=f"RS vs Nifty 500 ({TECHNICAL_RS_LOOKBACK_DAYS}-session return spread, pp)",
                              yaxis_title="RS-Momentum (pp)")
        st.plotly_chart(fig_rrg, width="stretch")
        st.caption(
            f"Each dot is one week's close ({pd.Timestamp(dates[0]):%d-%b} to {pd.Timestamp(dates[-1]):%d-%b-%Y}); the arrow is "
            "the latest week and shows the direction of rotation; colour = the current quadrant. The curve is drawn "
            "through the weekly points (only the line between them is interpolated; every dot is the actual value). Healthy rotation runs "
            "clockwise: IMPROVING → LEADING → WEAKENING → LAGGING. Same RS and momentum as the tables above (percentage "
            "points centred on 0, not StockCharts' proprietary JdK scale centred on 100; the quadrants mean the same). "
            "Our sub-themes are equal-weighted baskets of our universe; 'Our portfolio' uses the current weights."
        )

    rrg_left, rrg_right = st.columns(2)
    with rrg_left:
        combined_png = OUTPUT_DIR / "rrg_all_vs_nifty500.png"
        if combined_png.exists():
            st.image(str(combined_png), caption=f"All {len(PORTFOLIO_SYMBOLS)} universe stocks vs Nifty 500",
                     width="stretch")
        else:
            st.info("output/rrg_all_vs_nifty500.png not found. Run `python rrg.py --as-of <date>`.")
    with rrg_right:
        rrg_sector = st.selectbox("Sector RRG (vs equal-weighted sector average):", list(SECTOR_SCREENS),
                                  index=1, key="rrg_sector")
        sector_png = OUTPUT_DIR / f"rrg_{rrg_sector.lower().replace(' ', '_')}_vs_sector.png"
        if sector_png.exists():
            st.image(str(sector_png), caption=f"Nifty {rrg_sector} vs sector average", width="stretch")
        else:
            st.info(f"{sector_png.name} not found. Run `python rrg.py --as-of <date>`.")

    st.write("")
    st.markdown("---")

    # 2. Risk, Sizing & Stop-Loss (risk/position-sizing data, deliberately separate from momentum)
    with st.container(border=True):
        st.markdown("### Risk, Sizing & Stop-Loss")
        st.caption(
            "Risk and position-sizing figures, not momentum signals. Volatility and historical return use "
            "~1 year of daily returns (close vs. the exchange's previous close). Stop-loss = price - "
            f"{STOP_LOSS_ATR_MULTIPLE:g} x ATR({STOP_LOSS_ATR_PERIOD}), about a one-month, one-standard-deviation "
            f"move; if a support level sits up to {STOP_LOSS_SUPPORT_BAND_ATR:g} ATR below that, the stop moves just under the support. "
            "For the 3-month holding period the stop is sized for one month and, from the 28-Sep snapshot on, trailed "
            "automatically: every refresh keeps the previous stop as a floor, so a stop only ever rises."
        )
        st.markdown(
            f"""
            <span style="font-size: 0.82rem;">
                <strong>Expected return</strong> is CAPM: risk-free (Nifty 1D Rate, last quarter) + beta vs the Nifty 500 ×
                {MARKET_RISK_PREMIUM_PCT:g}% India equity risk premium ({MARKET_RISK_PREMIUM_SOURCE}); the past year's average
                return is shown next to it for comparison only (it is not a forecast). <strong>Weight</strong> is final: equal risk contribution within {WEIGHT_MIN_PCT:g}-{WEIGHT_MAX_PCT:g}%
                (each of the {len(HELD)} invested stocks carries the same share of portfolio variance, from one year
                of daily returns). It needs no return forecast, which no signal provides reliably (research/momentum_study.py).
            </span>
            """,
            unsafe_allow_html=True,
        )

        if risk_df.empty:
            st.info("Risk summary not found. Run `python main.py` to generate output/portfolio_risk_summary.csv.")
        else:
            risk_table_df = pd.DataFrame({
                "Symbol": risk_df["symbol"],
                "Sector": risk_df["sector"],
                "Current Price (₹)": risk_df["current_price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
                "Ann. Volatility (%)": risk_df["annualized_volatility_pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
                "Expected Return: CAPM (annual)": risk_df.get("capm_expected_return_pct", pd.Series(dtype=float)).apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
                "CAPM over 3 months": risk_df.get("capm_3m_return_pct", pd.Series(dtype=float)).apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
                "Past-year average return (not a forecast)": risk_df[
                    "historical_expected_return_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"),
                "Weight (equal risk)": risk_df["weight_pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
                "Risk Contribution (%)": risk_df["risk_contribution_pct"].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "—"),
                "Shares": risk_df["shares"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—"),
                "Invested (₹)": risk_df["invested_inr"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—"),
                f"ATR({STOP_LOSS_ATR_PERIOD}) (%)": risk_df["atr_pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
                "Stop-Loss (₹)": risk_df["stop_loss_price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
                "Stop Below Price (%)": risk_df["stop_loss_pct_below_current"].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
                "Stop Method": risk_df["stop_loss_method"].map(lambda m: STOP_METHOD_LABELS.get(m, "Unavailable")),
            })
            st.dataframe(risk_table_df, hide_index=True, width="stretch")

    st.write("")
    st.markdown("---")

    # 3. Interactive Single-Stock Deep Dive Price Chart with Support / Resistance
    st.markdown("### Stock Technical Deep-Dive: 1-Year Historical Price & Levels")

    # Default to first stock alphabetically
    sorted_symbols = sorted(LOCKED_PORTFOLIO_SYMBOLS)
    selected_symbol = st.selectbox(
        "Select Stock for 1-Year OHLCV & Support/Resistance Analysis:",
        options=sorted_symbols,
        index=0,
        format_func=lambda s: f"{s} — {LOCKED_PORTFOLIO[s]['name']} ({LOCKED_PORTFOLIO[s]['sector']})",
    )

    # Retrieve selected stock's summary and historical series
    stock_info = summary_df[summary_df["symbol"] == selected_symbol].iloc[0] if not summary_df.empty else None
    stock_history = ohlcv_df[ohlcv_df["SYMBOL"] == selected_symbol].copy() if not ohlcv_df.empty else pd.DataFrame()

    if not stock_history.empty and stock_info is not None:
        stock_history = stock_history.sort_values("DATE1").reset_index(drop=True)
        curr_p = float(stock_info["current_price"])
        supp_p = float(stock_info["nearest_support"]) if pd.notna(stock_info["nearest_support"]) else None
        res_p = float(stock_info["nearest_resistance"]) if pd.notna(stock_info["nearest_resistance"]) and stock_info["nearest_resistance"] is not None else None

        # Display key metrics for selected stock in columns
        c_p1, c_p2, c_p3, c_p4, c_p5, c_p6 = st.columns(6)
        with c_p1:
            st.metric("Latest Close", f"₹{curr_p:,.2f}")
        with c_p2:
            rsi_val = stock_info["latest_rsi"]
            st.metric("RSI (14)", f"{rsi_val:.2f}", delta="Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
        with c_p3:
            adx_val = stock_info["latest_adx"]
            st.metric("ADX (14)", f"{adx_val:.2f}", delta="Strong Trend" if adx_val > 25 else "Consolidation")
        with c_p4:
            st.metric("Trend", stock_info["trend_direction"])
        with c_p5:
            dist_supp = f"{((curr_p - supp_p) / curr_p) * 100:.1f}% buffer" if supp_p else "N/A"
            st.metric("Support Floor", f"₹{supp_p:,.2f}" if supp_p else "N/A", delta=dist_supp, delta_color="normal")
        with c_p6:
            dist_res = f"{((res_p - curr_p) / curr_p) * 100:.1f}% to ceiling" if res_p else "Blue Sky"
            st.metric("Resistance Ceiling", f"₹{res_p:,.2f}" if res_p else "ATH / Blue Sky", delta=dist_res, delta_color="normal")

        # Build Interactive Plotly Price Chart with Horizontal Reference Lines
        fig = go.Figure()

        # Closing price line
        fig.add_trace(
            go.Scatter(
                x=stock_history["DATE1"],
                y=stock_history["CLOSE_PRICE"],
                mode="lines",
                name=f"{selected_symbol} Close",
                line=dict(color="#2563EB", width=2),
                hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Close:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        # Nearest Support Reference Line
        if supp_p:
            fig.add_hline(
                y=supp_p,
                line_dash="dash",
                line_color="#10B981",
                line_width=1.8,
                annotation_text=f"Support Floor: ₹{supp_p:,.2f}",
                annotation_position="bottom right",
                annotation_font_color="#10B981",
                annotation_font_size=11,
            )

        # Nearest Resistance Reference Line
        if res_p:
            fig.add_hline(
                y=res_p,
                line_dash="dash",
                line_color="#EF4444",
                line_width=1.8,
                annotation_text=f"Resistance Ceiling: ₹{res_p:,.2f}",
                annotation_position="top right",
                annotation_font_color="#EF4444",
                annotation_font_size=11,
            )

        fig.update_layout(
            title=dict(
                text=f"{LOCKED_PORTFOLIO[selected_symbol]['name']} ({selected_symbol}) — 1-Year Historical Closing Price & Key Levels",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.15)",
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title="Price (INR)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.15)",
                tickprefix="₹",
            ),
            margin=dict(l=40, r=40, t=50, b=30),
            height=460,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.warning(f"No historical price series found for symbol '{selected_symbol}'.")


# =============================================================================
# TAB 4: RISK & HEDGING
# =============================================================================


SECTOR_CHART_COLORS = {"Capital Goods": "#B45309", "Power": "#047857", "Cement": "#1D4ED8"}

with tab_risk:
    single_df = _out("regression_single_index.csv")
    multi_df = _out("regression_multifactor.csv")
    plan_df = _out("hedge_plan.csv")
    puts_df = _out("hedge_put_candidates.csv")
    scen_df = _out("hedge_scenarios.csv")
    if single_df.empty or plan_df.empty:
        st.info("Regression and hedge outputs not found. Press the refresh button (or run `python risk_model.py`).")
    else:
        plan = dict(zip(plan_df["metric"], plan_df["value"]))
        notes = dict(zip(plan_df["metric"], plan_df["note"].fillna("")))

        # 1. Allocation of the Rs 1 crore
        st.markdown("### Allocation of the ₹1 crore")
        put_cost = float(plan.get("Puts: cost (Rs)", 0))
        alloc_rows = []
        if not risk_df.empty:
            for sector, value in risk_df.groupby("sector")["invested_inr"].sum().items():
                alloc_rows.append((f"{sector} stocks", value, SECTOR_CHART_COLORS.get(sector, "#64748B")))
        stocks_total = sum(v for _, v, _ in alloc_rows)
        alloc_rows.append(("Nifty puts (tail hedge)", put_cost, "#475569"))
        alloc_rows.append(("Cash (liquid ETF, overnight rate)", PRINCIPAL_INR - stocks_total - put_cost, "#94A3B8"))
        a_col1, a_col2 = st.columns([1.1, 1])
        with a_col1:
            pie = go.Figure(go.Pie(
                labels=[r[0] for r in alloc_rows], values=[r[1] for r in alloc_rows], hole=0.5, sort=False,
                marker=dict(colors=[r[2] for r in alloc_rows], line=dict(color="#FFFFFF", width=2)),
                texttemplate="%{label}<br>%{percent:.1%}", textposition="outside",
                hovertemplate="%{label}<br>₹%{value:,.0f} (%{percent})<extra></extra>"))
            pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                              template="plotly_white",
                              annotations=[dict(text=f"{stocks_total / PRINCIPAL_INR * 100:.1f}%<br>in stocks",
                                                showarrow=False, font=dict(size=14))])
            st.plotly_chart(pie, width="stretch")
        with a_col2:
            st.dataframe(pd.DataFrame({
                "Bucket": [r[0] for r in alloc_rows],
                "₹": [f"₹{r[1]:,.0f}" for r in alloc_rows],
                "% of principal": [f"{r[1] / PRINCIPAL_INR * 100:.2f}%" for r in alloc_rows],
            }), hide_index=True, width="stretch", height=_fit_height(alloc_rows))
            st.caption(
                f"Market exposure: {stocks_total / PRINCIPAL_INR * 100:.1f}% in stocks (brief: at least 90%). "
                f"The {100 - EQUITY_ALLOCATION_PCT:g}% reserve pays for the day-0 puts and one profit-trigger roll-up; "
                "until then it sits in a liquid ETF earning the overnight rate. A stop-loss exit is replaced from the reserve list "
                "with its own sale proceeds (the rounding stays here). No commodity or other ETF: none has a clear role (copper and aluminium "
                "are input costs for the cable and transformer makers; gold's crash-protection job is done more "
                "directly by the Nifty puts)."
            )

        # 2. Single-index model
        st.markdown("### Beta: explained and unexplained risk (single-index model vs Nifty 500 TRI)")
        st.caption(
            "Daily regression over the past year: r_i - r_f = α + β (r_m - r_f) + ε. Total variance = β²·var(r_m) "
            "(explained, systematic: hedgeable with index futures/options) + var(ε) (unexplained, stock-specific: "
            "handled by diversification and the stop-losses). R² is the explained share. The portfolio row "
            "shows diversification at work: each stock is 60-98% stock-specific risk, the portfolio about half."
        )
        port_row = single_df[single_df["symbol"] == "PORTFOLIO"]
        if not port_row.empty:
            pr = port_row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio beta (Nifty 500)", f"{pr['beta']:.2f}", help=f"standard error {pr['beta_se']:.2f}")
            c2.metric("R² (explained share)", f"{pr['explained_risk_pct']:.0f}%")
            c3.metric("Systematic volatility", f"{pr['systematic_vol_pct']:.1f}%")
            c4.metric("Stock-specific volatility", f"{pr['unsystematic_vol_pct']:.1f}%")
        st.dataframe(single_df.rename(columns={
            "symbol": "Stock", "observations": "Days", "alpha_annual_pct": "Alpha (ann. %)", "alpha_t": "Alpha t",
            "beta": "Beta", "beta_se": "Beta s.e.", "r_squared": "R²", "total_vol_pct": "Total vol %",
            "systematic_vol_pct": "Systematic vol %", "unsystematic_vol_pct": "Unsystematic vol %",
            "explained_risk_pct": "Explained %", "unexplained_risk_pct": "Unexplained %"}),
            hide_index=True, width="stretch", height=_fit_height(single_df))

        # 3. Multifactor model
        st.markdown("### Multifactor model: market + crude + interest rates")
        st.caption(
            "Adds the daily change in Brent crude (last US close before the Indian session) and the 10-year G-sec "
            "price return (≈ -duration × change in yield: positive = yields fell) to the market factor. "
            "Compare R² with the single-index model; |t| > 2 marks a factor that matters."
        )
        if not multi_df.empty:
            mp = multi_df[multi_df["symbol"] == "PORTFOLIO"]
            if not mp.empty:
                m = mp.iloc[0]
                st.markdown(
                    f"**Portfolio:** R² {m['r2_single_index']:.3f} (single index) → {m['r2_multifactor']:.3f} "
                    f"(multifactor), adjusted R² {m['adj_r2_single_index']:.3f} → {m['adj_r2_multifactor']:.3f}. "
                    f"Crude beta {m['beta_crude']:+.3f} (t {m['t_crude']:+.2f}), G-sec beta {m['beta_gsec']:+.3f} "
                    f"(t {m['t_gsec']:+.2f}). Crude and rates add almost nothing beyond the market for this "
                    "portfolio, so the single-index beta is the right basis for the hedge."
                )
            st.dataframe(multi_df.rename(columns={
                "symbol": "Stock", "observations": "Days", "beta_market": "β market", "t_market": "t",
                "beta_crude": "β crude", "t_crude": "t ", "beta_gsec": "β G-sec", "t_gsec": "t  ",
                "r2_single_index": "R² single", "r2_multifactor": "R² multi", "adj_r2_single_index": "Adj R² single",
                "adj_r2_multifactor": "Adj R² multi", "r2_gain_pp": "R² gain (pp)"}),
                hide_index=True, width="stretch", height=_fit_height(multi_df))

        # 3b. CAPM expected return
        capm_df = _out("capm_expected_returns.csv")
        if not capm_df.empty:
            st.markdown("### Expected return (CAPM)")
            cp = capm_df[capm_df["symbol"] == "PORTFOLIO"]
            if not cp.empty:
                c = cp.iloc[0]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Portfolio expected return (annual)", f"{c['capm_expected_return_pct']:.2f}%")
                k2.metric("Over the 3-month window", f"{c['capm_3m_return_pct']:.2f}%",
                          help="(1 + annual)^(1/4) - 1")
                k3.metric("In rupees on ₹1 crore (3 months)", _inr(PRINCIPAL_INR * c["capm_3m_return_pct"] / 100))
                k4.metric("Risk-free / market risk premium", f"{c['risk_free_pct']:.2f}% / {c['market_risk_premium_pct']:.2f}%")
            st.caption(
                "E[r] = r_f + β × MRP. r_f = Nifty 1D Rate index annualised over the last quarter; β = single-index beta vs "
                f"the Nifty 500 TRI (above); MRP = {MARKET_RISK_PREMIUM_SOURCE}. CAPM is what the market pays for the risk "
                "taken, not a forecast of our picks: our thesis is to beat it (alpha)."
            )
            st.dataframe(capm_df.drop(columns=["source"]).rename(columns={
                "symbol": "Stock", "beta": "Beta", "risk_free_pct": "Risk-free %", "market_risk_premium_pct": "MRP %",
                "capm_expected_return_pct": "CAPM annual %", "capm_3m_return_pct": "CAPM 3-month %"}),
                hide_index=True, width="stretch", height=_fit_height(capm_df))

        # 3c. Autocorrelation: stock returns against their own past returns
        ac_df = _out("autocorrelation.csv")
        if not ac_df.empty:
            st.markdown("### Own-return regression (autocorrelation)")
            n_pred = int(ac_df[ac_df["symbol"] != "PORTFOLIO"]["predictable_at_5pct"].sum())
            st.caption(
                "Does yesterday's return predict today's? AR(1): r_t = a + φ·r_(t-1) + e, autocorrelations at lags 1-5 "
                "(significant beyond ±1.96/√n) and the Ljung-Box Q test (Q = n(n+2)·Σρ_k²/(n-k); above the 5% critical value "
                f"= predictable). {n_pred} of {len(ac_df) - 1} stocks show significant daily autocorrelation: daily returns are "
                "close to a random walk, so the edge is not in day-to-day patterns but in the multi-month trend (momentum) the "
                "selection uses. Weekly ρ₁ = lag-1 autocorrelation of non-overlapping 5-session returns (a short-horizon "
                "momentum check; significant beyond its own bound)."
            )
            st.dataframe(ac_df.rename(columns={
                "symbol": "Stock", "observations": "Days", "ar1_phi": "AR(1) φ", "ar1_t": "t", "rho_1": "ρ1", "rho_2": "ρ2",
                "rho_3": "ρ3", "rho_4": "ρ4", "rho_5": "ρ5", "significance_bound": "±bound",
                "significant_lags": "Significant lags", "ljung_box_q": "Ljung-Box Q", "ljung_box_critical_5pct": "Q 5% crit.",
                "predictable_at_5pct": "Predictable?", "weekly_rho_1": "Weekly ρ1", "weekly_significance_bound": "Weekly ±bound"}),
                hide_index=True, width="stretch", height=_fit_height(ac_df))

        # 3d. Risk-reward for the 3-month window
        rr_df = _out("risk_reward.csv")
        if not rr_df.empty:
            st.markdown("### Risk-reward for the next 3 months")
            stocks_rr = rr_df[~rr_df["symbol"].str.startswith("PORTFOLIO")]
            all_once = rr_df[rr_df["symbol"] == "PORTFOLIO (all stocks at once)"]
            diversified = rr_df[rr_df["symbol"] == "PORTFOLIO (diversified)"]
            if not all_once.empty:
                ao = all_once.iloc[0]
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Upside, typical move (all stocks)", _inr(float(ao["upside_inr"])),
                          help="Each stock's one-standard-deviation 3-month move, summed")
                r2.metric("Downside if every stop is hit", _inr(float(ao["downside_inr"])),
                          help="Capped by the stops; a market crash is also covered by the puts")
                r3.metric("Reward : risk (all stocks at once)", f"{float(ao['reward_risk']):.2f} : 1")
                if not diversified.empty:
                    r4.metric("Reward : risk (diversified portfolio)", f"{float(diversified.iloc[0]['reward_risk']):.2f} : 1",
                              help="The portfolio's own 3-month move is smaller because the stocks do not all move together")
            st.caption(
                "Reward : risk = upside ÷ downside for the 3-month window. Downside = distance to the stop-loss (the loss the "
                "stop allows). Upside = a typical 3-month move, one standard deviation = annual volatility × √(63/252); the "
                "nearest resistance (a 20-session swing high) is shown as the first hurdle, not as a cap on a 3-month move. "
                "CAPM reward : risk uses only the market's required return (no stock-picking view), so it is always low. "
                "The case for the design: the downside is cut short by the stops (and a market crash by the puts), while "
                "the upside is left open. A price can gap below a stop on results day."
            )
            show_rr = rr_df.drop(columns=["price", "stop_loss_price", "value_inr"], errors="ignore").rename(columns={
                "symbol": "Stock", "downside_to_stop_pct": "Downside to stop %", "upside_1sd_3m_pct": "Typical 3-month move %",
                "first_hurdle_resistance": "First hurdle (resistance ₹)", "first_hurdle_above_pct": "Hurdle above price %",
                "upside_pct": "Upside %", "reward_risk": "Reward : risk", "capm_3m_pct": "CAPM 3-month %",
                "capm_reward_risk": "CAPM reward : risk", "upside_inr": "Upside ₹", "downside_inr": "Downside ₹"})
            st.dataframe(show_rr, hide_index=True, width="stretch", height=_fit_height(show_rr))

        # 4. Hedge plan
        st.markdown("### Hedge plan (Nifty 50 derivatives)")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Min-variance hedge ratio h*", f"{float(plan['Minimum-variance hedge ratio h*']):.2f}",
                  help=notes.get("Minimum-variance hedge ratio h*"))
        h2.metric("Hedge effectiveness (R²)", f"{float(plan['Hedge effectiveness (R^2)']) * 100:.0f}%",
                  help=notes.get("Hedge effectiveness (R^2)"))
        h3.metric("Tail hedge ratio", f"{float(plan['Tail hedge ratio']):.2f}", help=notes.get("Tail hedge ratio"))
        h4.metric("Puts cost", f"₹{float(plan['Puts: cost (Rs)']):,.0f}",
                  help=notes.get("Puts: cost (Rs)"))
        st.markdown(
            f"""
**Decision: tail-hedge with puts from day 0; no futures hedge.**
- **Day 0:** buy **{int(float(plan['Puts: lots']))} lots of {plan['Puts: contract']}** at ₹{float(plan['Puts: premium']):,.2f}
  (lot {int(float(plan['Nifty lot size']))}), about ₹{float(plan['Puts: cost (Rs)']):,.0f}. Lots = tail hedge ratio × portfolio value /
  (Nifty × lot). The tail hedge ratio is the portfolio's beta on Nifty down days: stocks fall together in a sell-off.
  One expiry covers the whole window, so there is no roll.
- **Why not futures:** a full futures hedge ({int(float(plan['Futures: lots for a full hedge (rounded)']))} lots) would cancel the market
  return we are positioned to earn, remove only {float(plan['Hedge effectiveness (R^2)']) * 100:.0f}% of the variance (the rest is stock-specific),
  needs a roll before the window ends, and ties up about ₹{float(plan['Futures: margin needed (Rs, assumed)']):,.0f} of margin (assumed
  {notes.get('Futures: margin needed (Rs, assumed)', '').split('ASSUMPTION ')[-1].split(' of')[0]} of notional). Forwards on the index are not available to us; exchange futures are the standardised forward.
- **After a profit (not greedy):** {notes.get('Profit trigger', '')}.
- **After a loss (not fearful):** no discretionary hedging; stock-specific falls are cut by each stock's stop-loss, and a market crash
  is covered by the puts.
"""
        )
        if not scen_df.empty:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=scen_df["nifty50_move_pct"], y=scen_df["unhedged_pnl_pct"], mode="lines+markers",
                                       name="Unhedged", line=dict(color="#94A3B8", width=2), marker=dict(size=8),
                                       hovertemplate="Nifty %{x:+}%: %{y:+.2f}% of principal<extra>Unhedged</extra>"))
            fig_s.add_trace(go.Scatter(x=scen_df["nifty50_move_pct"], y=scen_df["hedged_pnl_pct"], mode="lines+markers",
                                       name="With puts", line=dict(color="#1D4ED8", width=2), marker=dict(size=8),
                                       hovertemplate="Nifty %{x:+}%: %{y:+.2f}% of principal<extra>With puts</extra>"))
            fig_s.update_layout(height=340, template="plotly_white", hovermode="x unified",
                                xaxis_title="Nifty 50 move to expiry (%)", yaxis_title="Portfolio P&L (% of ₹1 crore)",
                                margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_s, width="stretch")
            st.caption("Market-driven P&L only (beta × Nifty move; the down-day beta for falls). Stock-specific moves come on top "
                       "and are handled by the stop-losses.")
        with st.expander("Hedge plan details, put strikes compared, scenario table"):
            st.dataframe(plan_df, hide_index=True, width="stretch", height=_fit_height(plan_df))
            st.dataframe(puts_df, hide_index=True, width="stretch", height=_fit_height(puts_df))
            st.dataframe(scen_df, hide_index=True, width="stretch", height=_fit_height(scen_df))


# =============================================================================
# TAB 5: PERFORMANCE
# =============================================================================
with tab_performance:
    render_tri_staleness_banner(tri_df, tri_staleness)
    perf_df = _out("performance_summary.csv")
    growth_df = _out("performance_growth.csv")
    if perf_df.empty:
        st.info("Performance outputs not found. Press the refresh button (or run `python performance.py`).")
    else:
        has_live = perf_df["window"].str.startswith("Live").any()
        if has_live:
            st.markdown(
                "<span style='font-size:0.82rem;'>The <strong>Live</strong> column is the real portfolio since the "
                f"{pd.Timestamp(EVALUATION_START_DATE):%d-%b-%Y} snapshot (stocks + Nifty puts + cash, from the trade ledger) "
                "against the same ₹1 crore in the Nifty 500 TRI; risk-free = the liquid fund. The other columns are a "
                "<strong>backtest</strong> of today's portfolio over past prices (hindsight, not a forecast).</span>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<span class='placeholder-badge' style='margin-bottom:0;'>Backtest until live figures start</span> "
                "<span style='font-size:0.82rem;'>These windows hold <strong>today's</strong> portfolio and weights over past "
                "prices. The stocks were chosen for strong past returns, so the figures are hindsight, not a forecast. "
                f"Live Sharpe, Treynor, alpha and XIRR appear here once the portfolio has {MIN_LIVE_SESSIONS} trading days "
                f"of history from the {pd.Timestamp(EVALUATION_START_DATE):%d-%b-%Y} snapshot.</span>",
                unsafe_allow_html=True,
            )
        show = perf_df.set_index("window").T
        labels = {
            "start": "Start", "end": "End", "sessions": "Sessions",
            "portfolio_return_pct": "Portfolio return (%)", "benchmark_return_pct": "Nifty 500 TRI return (%)",
            "excess_return_pp": "Excess return (pp)",
            "portfolio_annualised_compound_pct": "Portfolio annualised, compounded (%)",
            "portfolio_annualised_simple_pct": "Portfolio annualised, simple (%)",
            "compounding_effect_pp": "Compounding effect (pp)",
            "benchmark_annualised_pct": "Nifty 500 annualised (%)", "risk_free_annualised_pct": "Risk-free, 1D rate (%)",
            "portfolio_vol_pct": "Portfolio volatility (%)", "benchmark_vol_pct": "Nifty 500 volatility (%)",
            "beta_vs_nifty500": "Beta vs Nifty 500", "sharpe_portfolio": "Sharpe: portfolio",
            "sharpe_benchmark": "Sharpe: Nifty 500", "treynor_portfolio_pct": "Treynor: portfolio (%)",
            "treynor_benchmark_pct": "Treynor: Nifty 500 (%)", "jensen_alpha_pct": "Jensen's alpha (%)",
            "xirr_portfolio_pct": "XIRR: portfolio (%)", "xirr_benchmark_pct": "XIRR: Nifty 500 (%)",
        }
        show.index = [labels.get(i, i) for i in show.index]
        st.dataframe(show.astype(str), width="stretch", height=_fit_height(show))
        st.caption(
            "r(p) = Σ wᵢ rᵢ each day (current weights), compounded: R = Π(1 + r_t) - 1. Annualised (compounded) = "
            "(1 + R)^(252/n) - 1; the simple figure R × 252/n understates it, and the gap is the compounding effect. "
            "Sharpe = (R_p - R_f) / σ_p (reward per unit of total risk); Treynor = (R_p - R_f) / β (per unit of market "
            "risk); Jensen's alpha = R_p - [R_f + β (R_m - R_f)]. XIRR solves Σ CF / (1 + r)^(days/365) = 0 for the "
            "dated cash flows (invest at the start, value at the end); a loss gives a negative XIRR, annualised the same way. "
            "Risk-free = Nifty 1D Rate index."
        )
        if not growth_df.empty:
            g = growth_df.copy()
            g["date"] = pd.to_datetime(g["date"])
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=g["date"], y=g["portfolio"] * PRINCIPAL_INR / 1e5, name="Portfolio (today's weights)",
                                       line=dict(color="#B45309", width=2),
                                       hovertemplate="%{x|%d-%b-%Y}: ₹%{y:,.2f} L<extra>Portfolio</extra>"))
            fig_g.add_trace(go.Scatter(x=g["date"], y=g["nifty500_tri"] * PRINCIPAL_INR / 1e5, name="Nifty 500 TRI",
                                       line=dict(color="#1D4ED8", width=2),
                                       hovertemplate="%{x|%d-%b-%Y}: ₹%{y:,.2f} L<extra>Nifty 500 TRI</extra>"))
            fig_g.update_layout(height=360, template="plotly_white", hovermode="x unified",
                                yaxis_title="Value of ₹1 crore (₹ lakh)", margin=dict(l=10, r=10, t=30, b=10),
                                legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_g, width="stretch")
            st.caption("Growth of ₹1 crore over the past year with daily compounding (backtest).")
        st.markdown("### Capital Market Line")
        cml_png = OUTPUT_DIR / "cml.png"
        if cml_png.exists():
            st.image(str(cml_png), width="stretch")
        st.caption(
            "CML: E[r] = r_f + (E[r_m] - r_f)/σ_m × σ, through the Nifty 500 TRI. Over the past year the market returned less "
            "than the risk-free rate, so the CML slopes down (negative market Sharpe). Our portfolio plots far above it; "
            "the dashed line through the tangency portfolio of the invested stocks is the best risk-return trade-off they offered. "
            "Ex-post: a chart of the past, not a forecast."
        )
        cmp_stats = _out("portfolio_weights_compared_stats.csv")
        cmp_w = _out("portfolio_weights_compared.csv")
        if not cmp_stats.empty:
            st.markdown("### Global minimum variance portfolio (GMVP) vs our weights")
            st.caption(
                "GMVP = the combination of the invested stocks with the lowest possible volatility (the leftmost point of the "
                "efficient frontier, purple triangles on the chart), long-only and within the brief's weight limits. "
                "Effective number of stocks = 1 / Σw² (how many equal positions the portfolio behaves like). Our equal-risk "
                "weights give up a little volatility against the GMVP for broader diversification; the tangency portfolio "
                "has the best past Sharpe but bets half the money on one stock chosen with hindsight."
            )
            st.dataframe(cmp_stats.rename(columns={
                "portfolio": "Portfolio", "volatility_pct": "Volatility %", "mean_return_pct": "Past-year mean return %",
                "sharpe": "Sharpe (past year)", "effective_n_stocks": "Effective no. of stocks",
                "largest_weight_pct": "Largest weight %", "stocks_above_1pct": "Stocks above 1%"}),
                hide_index=True, width="stretch", height=_fit_height(cmp_stats))
            if not cmp_w.empty:
                with st.expander("Weights: ours vs GMVP vs tangency"):
                    st.dataframe(cmp_w.rename(columns={
                        "symbol": "Stock", "our_weight_pct": "Ours %", "gmvp_long_only_pct": "GMVP long-only %",
                        "gmvp_bounded_pct": "GMVP 5-15% %", "tangency_pct": "Tangency %"}),
                        hide_index=True, width="stretch", height=_fit_height(cmp_w))
