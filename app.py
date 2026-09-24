"""
Streamlit Portfolio Dashboard for Indian Equity Infrastructure & Capex Portfolio.
Coursework: Security Analysis & Portfolio Management (SAPM) / Derivatives
Institution: Indian Institute of Management (IIM) Bodh Gaya
Author / Pair Programmer: Antigravity

This dashboard serves as the central visual terminal for project defense and presentation.
It imports and reuses the existing pipeline outputs, focusing on the 8 locked portfolio
stocks across Cement, Capital Goods/EPC, and Power sectors.
"""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    BENCHMARK_PRICE_TICKER,
    DEFAULT_TRI_CSV_PATH,
    HISTORICAL_OHLCV_CSV,
    LOCKED_PORTFOLIO,
    LOCKED_PORTFOLIO_SYMBOLS,
    NTPC_CAVEAT,
    SUMMARY_OUTPUT_CSV,
)
from fetch_data import TRI_REDOWNLOAD_INSTRUCTIONS, TriStaleness, assess_tri_staleness, load_benchmark_tri
from fundamentals import get_fundamentals_summary

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
    
    /* NTPC Caveat Callout Box */
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

@st.cache_data(show_spinner=False)
def load_summary_data() -> pd.DataFrame:
    """
    Load the technical summary table, filter strictly for the 8 locked portfolio
    symbols, and enrich with official company display names and sectors.
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
def load_historical_ohlcv() -> pd.DataFrame:
    """
    Load historical daily OHLCV dataset for the locked portfolio stocks.
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
summary_df = load_summary_data()
ohlcv_df = load_historical_ohlcv()
tri_df = load_tri_benchmark(DEFAULT_TRI_CSV_PATH.stat().st_mtime if DEFAULT_TRI_CSV_PATH.exists() else 0.0)

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
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Sector Allocation</div>
                <div class="metric-value">3 Sectors</div>
                <div class="metric-sub">Cement (3) &bull; Cap Goods (2) &bull; Power (3)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Benchmark Series</div>
                <div class="metric-value">Nifty 500</div>
                <div class="metric-sub">Ticker: {BENCHMARK_PRICE_TICKER} (via yfinance)</div>
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

    # Visual NTPC Caveat Notice
    st.markdown(
        f"""
        <div class="caveat-box">
            <strong>Investment Committee Note (NTPC):</strong> {NTPC_CAVEAT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Sector-Grouped Table Display
    st.markdown("### Portfolio Constituents by Sector")
    st.caption(
        "Current market prices and nearest technical support/resistance levels derived from authentic "
        "daily NSE Bhavcopy exchange files. Weights and P&L are deliberately omitted pending capital allocation."
    )

    sectors = ["Cement", "Capital Goods/EPC", "Power"]
    sector_badge_classes = {
        "Cement": "badge-cement",
        "Capital Goods/EPC": "badge-capital-goods",
        "Power": "badge-power",
    }

    # Display clean sector cards / sections
    for sec in sectors:
        sec_df = summary_df[summary_df["sector"] == sec].copy()
        badge_cls = sector_badge_classes.get(sec, "badge-cement")

        st.markdown(
            f"""
            <div style="margin-top: 1.2rem; margin-bottom: 0.4rem;">
                <span class="{badge_cls}">{sec.upper()} SECTOR ({len(sec_df)} STOCKS)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Build clean display table
        display_sec_df = pd.DataFrame({
            "Symbol": sec_df["symbol"],
            "Company Name": sec_df["display_name"],
            "Sector": sec_df["sector"],
            "Current Price (₹)": sec_df["current_price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
            "Nearest Support (₹)": sec_df["nearest_support"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
            "Nearest Resistance (₹)": sec_df["nearest_resistance"].apply(
                lambda x: "ATH / Blue Sky" if pd.isna(x) or x is None else f"₹{x:,.2f}"
            ),
            "Notes": sec_df["symbol"].apply(
                lambda s: "⚠️ Bearish trend overridden on strong fundamentals" if s == "NTPC" else "Locked Constituent"
            ),
        })

        st.dataframe(
            display_sec_df,
            hide_index=True,
            use_container_width=True,
        )


# =============================================================================
# TAB 2: FUNDAMENTALS
# =============================================================================
with tab_fundamentals:
    st.markdown("### Fundamental Analysis & Quality Screening")
    st.caption(
        "Fundamental metrics fetched directly from authentic Screener.in company pages (/consolidated/ with standalone fallback) "
        "and cached locally in data/fundamentals_cache/. Evaluates core financial health, capital productivity, leverage, "
        "and cash flow resilience across the 8 locked portfolio constituents."
    )

    # Visible NTPC Fundamental Justification Banner
    st.markdown(
        """
        <div class="caveat-box">
            <strong>NTPC Fundamental Thesis:</strong> While displaying near-term technical consolidation, NTPC was included
            for its industry-leading operating cash flow generation (<strong>₹50,902 Cr</strong> &mdash; largest in portfolio),
            strong <strong>15.1% ROE</strong>, and regulated tariff cost-plus model providing defensive ballast during sector downturns.
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
                    "OPM (%)": f"{row['opm']:.1f}%" if pd.notna(row.get("opm")) else "Data Unavailable",
                    "3-Yr Sales Growth (%)": f"{row['sales_growth_3yr']:+.1f}%" if pd.notna(row.get("sales_growth_3yr")) else "Data Unavailable",
                    "3-Yr Profit Growth (%)": f"{row['profit_growth_3yr']:+.1f}%" if pd.notna(row.get("profit_growth_3yr")) else "Data Unavailable",
                })

        display_fund_df = pd.DataFrame(formatted_rows)
        st.dataframe(display_fund_df, hide_index=True, use_container_width=True)

    st.write("")
    st.markdown("#### Academic Fundamental Screen Framework & Methodology")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown(
            """
            - **ROCE & 3-Year Avg ROCE (%):** Quantifies core operating profitability per unit of debt + equity capital employed. Multi-year averaging filters out cyclical lumpy capacity additions.
            - **ROE (%):** Evaluates equity compounding power and DuPont efficiency.
            - **OPM (%):** Operating Profit Margin resilience against raw material and power tariff swings.
            """
        )
    with f_col2:
        st.markdown(
            """
            - **Debt to Equity:** Solvency constraint. Capital-goods transformer specialist Voltamp operates debt-free ($0.00$), and UltraTech maintains prudent leverage ($0.31$).
            - **Operating Cash Flow (₹ Cr):** Quality of earnings acid test; confirms cash conversion of accrual profits to service debt and fund capex.
            - **3-Year Compounded Growth:** Demonstrates execution momentum across multi-year infrastructure order books.
            """
        )


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

    # NTPC Caveat Notice in Technicals Tab
    st.markdown(
        f"""
        <div class="caveat-box">
            <strong>NTPC Technical Alert:</strong> {NTPC_CAVEAT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Full Technical Summary Table with subtle trend_direction color tinting
    tech_table_df = pd.DataFrame({
        "Symbol": summary_df["symbol"],
        "Name": summary_df["display_name"],
        "Sector": summary_df["sector"],
        "Current Price (₹)": summary_df["current_price"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
        "RSI (14)": summary_df["latest_rsi"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "ADX (14)": summary_df["latest_adx"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "Trend Direction": summary_df["trend_direction"],
        "RS Spread vs N500": summary_df["rs_score_vs_nifty500"].apply(lambda x: f"{x:+.2f} pp" if pd.notna(x) else "—"),
        "Support (₹)": summary_df["nearest_support"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"),
        "Resistance (₹)": summary_df["nearest_resistance"].apply(
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
    st.dataframe(styled_table, hide_index=True, use_container_width=True)

    st.write("")
    st.markdown("---")

    # 2. Interactive Single-Stock Deep Dive Price Chart with Support / Resistance
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

        # Specific NTPC badge under chart metrics
        if selected_symbol == "NTPC":
            st.markdown(
                f"""
                <div class="caveat-box">
                    <strong>NTPC Caveat:</strong> {NTPC_CAVEAT}
                </div>
                """,
                unsafe_allow_html=True,
            )

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

        st.plotly_chart(fig, use_container_width=True)

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
