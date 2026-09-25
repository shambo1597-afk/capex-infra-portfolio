"""
Streamlit Portfolio Dashboard for Indian Equity Infrastructure & Capex Portfolio.
Coursework: Security Analysis & Portfolio Management (SAPM) / Derivatives
Institution: Indian Institute of Management (IIM) Bodh Gaya
Author / Pair Programmer: Antigravity

This dashboard serves as the central visual terminal for project defense and presentation.
It imports and reuses the existing pipeline outputs, focusing on the locked portfolio
(config.LOCKED_PORTFOLIO) across the Cement, Capital Goods and Power sectors.
"""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_TRI_CSV_PATH,
    HISTORICAL_OHLCV_CSV,
    LOCKED_PORTFOLIO,
    LOCKED_PORTFOLIO_SYMBOLS,
    OUTPUT_DIR,
    PORTFOLIO_SYMBOLS,
    SECTOR_SCREENS,
    RISK_SUMMARY_OUTPUT_CSV,
    STOP_LOSS_ATR_MULTIPLE,
    STOP_LOSS_ATR_PERIOD,
    SUMMARY_OUTPUT_CSV,
)
from fetch_data import TRI_REDOWNLOAD_INSTRUCTIONS, TriStaleness, assess_tri_staleness, load_benchmark_tri
from fundamentals import get_fundamentals_summary
from rrg import CONVICTION_HIGH, CONVICTION_LOW, CONVICTION_MODERATE, conviction_tier
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
               "fundamentals_failed", "high_turnover_business_flag", "technically_attractive"]


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


def render_tri_staleness_banner(tri: pd.DataFrame, staleness: Optional[TriStaleness]) -> None:
    """Show a warning banner when the TRI benchmark is missing or trails the analysis end date."""
    if tri.empty:
        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>TRI benchmark data unavailable.</strong> No Nifty 500 TRI CSV could be loaded.
                {TRI_REDOWNLOAD_INSTRUCTIONS}
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
                the analysis end date ({staleness.reference_date:%d-%b-%Y}). {TRI_REDOWNLOAD_INSTRUCTIONS}
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_fundamentals_summary(force_refresh: bool = False) -> pd.DataFrame:
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
               CONVICTION_LOW: "Low: sector-coverage hold"}
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
        "<th>Expected return<span class='sub ph'>Historical average (placeholder pending CAPM)</span></th>"
        "<th>Weight<span class='sub ph'>Placeholder pending final weight assignment</span></th>"
        "<th>Stop-loss (₹)<span class='sub'>% below · method</span></th>"
        "<th>ADX (14)</th><th>RS vs Nifty 500<span class='sub'>63 sessions</span></th><th>RSI (14)</th>"
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
            f"<td class='ph'>{_fmt(r.get('historical_expected_return_pct'), '+.2f', suffix='%')}</td>"
            f"<td class='ph'>{_fmt(r.get('weight_pct'), '.2f', suffix='%')}</td>"
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

# Load data assets
summary_df = load_summary_data(_file_mtime(SUMMARY_OUTPUT_CSV))
ohlcv_df = load_historical_ohlcv(_file_mtime(HISTORICAL_OHLCV_CSV))
tri_df = load_tri_benchmark(_file_mtime(DEFAULT_TRI_CSV_PATH))
risk_df = load_risk_summary(_file_mtime(RISK_SUMMARY_OUTPUT_CSV))
rrg_df = load_rrg_data(tuple(_file_mtime(review_table_path(sector)) for sector in SECTOR_SCREENS))

# One row per locked stock: technicals + risk/sizing/stop-loss + RRG, in LOCKED_PORTFOLIO order
portfolio_df = summary_df.copy()
if not risk_df.empty:
    portfolio_df = portfolio_df.merge(
        risk_df.drop(columns=["sector", "current_price"], errors="ignore"), on="symbol", how="left")
portfolio_df = portfolio_df.merge(rrg_df, on="symbol", how="left")
missing_symbols = sorted(set(LOCKED_PORTFOLIO_SYMBOLS) - set(portfolio_df["symbol"]))

# TRI must cover the period the last pipeline run analysed (its latest price date)
_last_price_date = ohlcv_df["DATE1"].max() if not ohlcv_df.empty else pd.NaT
analysis_end_date = _last_price_date.date() if pd.notna(_last_price_date) else date.today()
tri_staleness = assess_tri_staleness(tri_df, analysis_end_date)

# -----------------------------------------------------------------------------
# 5 TABS NAVIGATION
# -----------------------------------------------------------------------------

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
    # 1. Summary Metric Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Portfolio Universe</div>
                <div class="metric-value">{len(LOCKED_PORTFOLIO)} Stocks</div>
                <div class="metric-sub">Locked for coursework defense</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        sector_counts = pd.Series([v["sector"] for v in LOCKED_PORTFOLIO.values()]).value_counts()
        sector_breakdown = " &bull; ".join(f"{sec} ({sector_counts.get(sec, 0)})" for sec in SECTOR_SCREENS)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Sector Allocation</div>
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
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Price Integrity</div>
                <div class="metric-value">NSE Bhavcopy</div>
                <div class="metric-sub">Direct exchange clearing data</div>
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
        "**Low (sector-coverage hold)** = WEAKENING or LAGGING in both views."
    )
    if missing_symbols:
        st.warning(f"No pipeline data for {', '.join(missing_symbols)}. Run `python main.py` to refresh the outputs.")

    # 3. Sector-grouped, one row per stock with every field the brief requires
    st.markdown("### Portfolio Constituents by Sector")
    st.caption(
        "Prices, technicals and risk figures from daily NSE Bhavcopy files (last pipeline run). "
        f"Stop-loss: price - {STOP_LOSS_ATR_MULTIPLE:g} x ATR({STOP_LOSS_ATR_PERIOD}), moved just below a support level "
        "up to 1 ATR beyond it; the method column shows which applied. Fields marked in blue are placeholders."
    )

    sector_badge_classes = {
        "Cement": "badge-cement",
        "Capital Goods": "badge-capital-goods",
        "Power": "badge-power",
    }
    for sec in SECTOR_SCREENS:
        sec_df = portfolio_df[portfolio_df["sector"] == sec]
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

    # 4. Screen exceptions: locked stocks that do not pass every screen (from the review tables)
    exceptions = []
    for _, r in portfolio_df.iterrows():
        if r.get("technically_attractive") == False:  # noqa: E712
            exceptions.append(
                f"**{r['symbol']}** does not pass the technical screen (RS vs Nifty 500 "
                f"{r['rs_score_vs_nifty500']:+.2f} pp, needs > +2 pp; trend {str(r['trend_direction']).split(' (')[0]}); "
                f"conviction tier: {r.get('conviction_tier') or 'Unclassified'}.")
        failed = r.get("fundamentals_failed")
        if isinstance(failed, str) and failed.strip():
            note = (" Flagged by the OPM-exception check (fails only OPM, ROCE above 20%) and held after a "
                    "manual business-model review." if r.get("high_turnover_business_flag") == True else "")  # noqa: E712
            exceptions.append(f"**{r['symbol']}** fails the {r['sector']} fundamental screen: {failed}.{note}")
    if exceptions:
        st.markdown("#### Screen exceptions")
        st.caption("Locked stocks that do not pass every screen, stated so the selection can be defended.")
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
        reason = (" It is held after a manual business-model review: the OPM-exception check flagged it "
                  "(fails only OPM, ROCE above 20%), a high-turnover business for which OPM is the wrong yardstick."
                  if exc.get("high_turnover_business_flag") == True else "")  # noqa: E712
        st.markdown(
            f"""
            <div class="caveat-box">
                <strong>Screen exception: {exc['symbol']}</strong> fails <em>{exc['fundamentals_failed']}</em>
                on the {exc['sector']} safety screen.{reason}
            </div>
            """,
            unsafe_allow_html=True,
        )

    fund_raw_df = load_fundamentals_summary()

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
        "The universe is screened technically (RS vs Nifty 500 > +2 pp and a Bullish trend) and against its "
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
        "RS score measures 63-day cumulative percentage-point alpha over the Nifty 500."
    )

    # 1. Full Technical Summary Table with subtle trend_direction color tinting
    tech_table_df = pd.DataFrame({
        "Symbol": portfolio_df["symbol"],
        "Name": portfolio_df["display_name"],
        "Sector": portfolio_df["sector"],
        "Current Price (₹)": portfolio_df["current_price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
        "RSI (14)": portfolio_df["latest_rsi"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "ADX (14)": portfolio_df["latest_adx"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "Trend Direction": portfolio_df["trend_direction"],
        "DI Gap (+DI − −DI)": portfolio_df["di_gap"].apply(
            lambda x: "—" if pd.isna(x) else f"{x:+.2f}" + (" (thin)" if abs(x) < 2 else "")),
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
        "x = 63-session RS (pp); y = RS-Momentum = RS today minus the same RS 10 sessions earlier (pp). "
        "Locked stocks are ringed and bold. Regenerate with `python rrg.py --as-of <date>` after a pipeline run."
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
            "move; if a support level sits up to 1 ATR below that, the stop moves just under the support. "
            "For the 3-month holding period the stop is sized for one month and trailed up (never down) at "
            "each monthly review (python main.py --trail-stops <previous risk summary CSV>)."
        )
        st.markdown(
            f"""
            <span class="placeholder-badge" style="margin-bottom: 0;">Placeholder &mdash; pending finalization</span>
            <span style="font-size: 0.82rem;">
                <strong>Expected return</strong> is a historical average (placeholder pending CAPM, once portfolio
                beta is computed). <strong>Weight</strong> is a placeholder pending final weight assignment: equal weighting
                ({100 / len(LOCKED_PORTFOLIO_SYMBOLS):.2f}% each across {len(LOCKED_PORTFOLIO_SYMBOLS)} stocks)
                until formal weight assignment within the capping constraints is completed.
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
                "Expected Return: Historical average (placeholder pending CAPM)": risk_df[
                    "historical_expected_return_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"),
                "Weight: Placeholder pending final weight assignment": risk_df["weight_pct"].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "—"),
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
# TAB 4: RISK & HEDGING (PLACEHOLDER ONLY)
# =============================================================================
with tab_risk:
    st.markdown(
        """
        <div class="placeholder-container">
            <span class="placeholder-badge">Module in Progress</span>
            <div class="placeholder-text">
                Beta regression, explained/unexplained risk decomposition, and hedge
                ratio analysis are in progress and will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 5: PERFORMANCE (PLACEHOLDER ONLY)
# =============================================================================
with tab_performance:
    render_tri_staleness_banner(tri_df, tri_staleness)

    st.markdown(
        """
        <div class="placeholder-container">
            <span class="placeholder-badge">Module in Progress</span>
            <div class="placeholder-text">
                Sharpe ratio, Treynor ratio, XIRR, and the Capital Market Line will
                appear here once the portfolio's first performance snapshot (28th September)
                is available.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
