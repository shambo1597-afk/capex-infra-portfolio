# Indian Equity Portfolio Data Pipeline & Analysis Toolkit
**Coursework:** Security Analysis & Portfolio Management (SAPM) / Derivatives  
**Institution:** Indian Institute of Management (IIM) Bodh Gaya  
**Author / Pair Programmer:** Antigravity  

A production-grade, mathematically transparent data pipeline and quantitative screening toolkit built for Indian equity portfolio management. The toolkit ingests authentic daily closing Bhavcopy data directly from the National Stock Exchange of India (NSE), downloads benchmark price series via Yahoo Finance, ingests local Total Return Index (TRI) series, and computes custom technical indicators (Wilder's RSI, Wilder's ADX, 63-day Relative Strength spread, Support/Resistance swing pivots) directly in `pandas` and `numpy` without black-box TA libraries.

---

## Table of Contents
1. [Core Features & Architecture](#core-features--architecture)
2. [Data Sources & Anti-Distortion Design](#data-sources--anti-distortion-design)
3. [Portfolio Universe](#portfolio-universe)
4. [Mathematical & Financial Methodology](#mathematical--financial-methodology)
   - [Wilder's Smoothing Method](#1-wilders-exponential-smoothing)
   - [Relative Strength Index (RSI - 14)](#2-relative-strength-index-rsi-14)
   - [Average Directional Index (ADX - 14, +DI, -DI)](#3-average-directional-index-adx-14)
   - [Relative Strength (RS Spread vs Nifty 500 - 63 Days)](#4-relative-strength-spread-63-days)
   - [Support and Resistance Levels](#5-support-and-resistance-swing-pivots)
5. [Project Structure](#project-structure)
6. [Installation & Requirements](#installation--requirements)
7. [How to Run the Pipeline](#how-to-run-the-pipeline)
8. [Output Format & Column Definitions](#output-format--column-definitions)
9. [Running the Test Suite](#running-the-test-suite)
10. [Corporate Action & Symbol Renaming Handling](#corporate-action--symbol-renaming-handling)

---

## Core Features & Architecture

- **Zero Black-Box Libraries:** Every single formula (RSI, ADX, TR, +DM/-DM, DX, Wilder's smoothing, rolling swing pivots) is implemented in clean, vectorized Python/NumPy/Pandas with inline mathematical explanations.
- **Direct NSE Ingestion:** Fetches official daily "Full Bhavcopy and Security Deliverable data" CSVs from NSE's clearing servers, eliminating corporate action distortion from third-party wrappers.
- **Dual Benchmark Support:**
  - **Price Return Index (PRI):** Nifty 500 from NSE's official daily index closes (`yfinance` `^CRSLDX` as fallback).
  - **Total Return Index (TRI):** Ingestion from local official CSV downloaded from `niftyindices.com`.
- **Intelligent Caching:** Daily bhavcopies and consolidated histories are preserved locally under `data/raw_bhavcopy/` and `data/processed/`, enabling instant offline re-analysis.
- **Academic Rigor:** Full docstrings detailing financial theory (momentum, directional volatility, alpha spread, support/resistance barriers) for coursework defense.

---

## Data Sources & Anti-Distortion Design

### 1. Individual Stock Price History — Direct NSE Full Bhavcopy
- **Endpoint:**
  ```
  https://www.nseindia.com/api/reports?archives=[{"name":"Full Bhavcopy and Security Deliverable data","type":"daily-reports","category":"capital-market","section":"equities"}]&date={DD-Mon-YYYY}&type=equities&mode=single
  ```
- **Why Direct Ingestion?**  
  Third-party Python wrappers (`yfinance` for Indian individual equities, `aynse`, `nsepy`, `jugaad-data`) suffer from unannounced corporate action adjustments (splits, bonuses, mergers), stale caches, or broken scrapers. Direct Bhavcopy queries guarantee authentic, exchange-cleared OHLCV and deliverable volumes.
- **Technical Protocol:**
  - Standard Chrome Windows `User-Agent` and header masking.
  - Initial warm-up `GET` request to `https://www.nseindia.com/` to seed session cookies (`nsit`, `nseappid`).
  - Persistent `requests.Session` maintaining cookie state.
  - Weekend filtering (Monday through Friday loop).
  - Polite 0.5s rate-limit sleep between queries.
  - Graceful holiday detection (404 / empty report handling without crashing).

### 2. Benchmark (Price Return) — Nifty 500 from NSE official index closes
- **Source:** NSE's daily `ind_close_all_DDMMYYYY.csv` archive files, fetched and cached on exactly the same session calendar as the Bhavcopy stock prices (same holiday detection, block retries and special weekend sessions).
- **Rationale:** yfinance's `^CRSLDX` matched the official values on every common session but skipped real NSE sessions (1-Jan-2026, the 1-Feb-2026 Budget session, 22-Sep-2026 and the latest day, whose end date yfinance treats as exclusive). That silently shifted the 63-session RS window by a session, which moved RS by up to ±11 pp. yfinance is kept only as a fallback.

### 3. Benchmark (Total Return Index) — Local CSV
- **Rationale:** `niftyindices.com` serves TRI data only to real browser sessions behind Akamai bot protection, so the default path reads the official CSV downloaded from the portal (automated retrieval is available opt-in, see below). Schema:
  `IndexName, Date, Total Returns Index, Net Total Return Index`
- **Staleness check:** the CSV is re-downloaded by hand, so after loading, its last date is compared with the run's `--end-date` (default: today). If it trails by more than 3 trading days (weekdays; holidays not modelled, see `TRI_STALE_THRESHOLD_TRADING_DAYS` in `config.py`), the pipeline prints a warning banner with re-download steps and **continues** with the data it has. The dashboard shows the same warning on the Performance tab, measured against the latest price date from the last pipeline run.

### 4. Fundamental Quality Data — Direct Screener.in Extraction
- **Endpoint Pattern:**
  `https://www.screener.in/company/{SYMBOL}/consolidated/` (with fallback to `https://www.screener.in/company/{SYMBOL}/`)
- **No Uploaded CSVs Required:** Direct HTTP scraping using real browser headers and persistent session cookies.
- **Local Caching:** HTML files cached under `data/fundamentals_cache/{SYMBOL}.html` for instantaneous offline loads.
- **Parsed Metrics:** Market Cap, Current Price, ROCE, ROE, 3-Year Average ROCE trend from Ratios, Cash from Operating Activity (CFO) from Cash Flows, Debt-to-Equity from Balance Sheet, OPM, and 3-Year Compounded Growth.

---

## Portfolio Universe (Locked Allocation — 8 Stocks)

| Sector | Allocation | Symbols | Notes & Investment Committee Thesis |
| :--- | :---: | :--- | :--- |
| **Cement** | 3 Stocks | `JKCEMENT`, `ULTRACEMCO`, `STARCEMENT` | Core capex infrastructure plays with strong capacity expansion and pricing power |
| **Capital Goods / EPC** | 2 Stocks | `BHEL`, `VOLTAMP` | BHEL (power equipment recovery) & Voltamp (debt-free, 28% 3-yr ROCE transformer specialist) |
| **Power** | 3 Stocks | `POWERGRID`, `TATAPOWER`, `NTPC` | Transmission moat & green energy transition. **NTPC Caveat:** Near-term technical consolidation overridden for sector-best cash flows (₹50,902 Cr CFO) |

---

## Mathematical & Financial Methodology

### 1. Wilder's Exponential Smoothing
Introduced by J. Welles Wilder Jr. in *New Concepts in Technical Trading Systems* (1978), this smoothing method eliminates the abrupt 'cliff effect' of Simple Moving Averages (SMA) while avoiding the aggressive weighting of standard EMA ($\alpha = 2 / (N + 1)$). Wilder uses $\alpha = 1 / N$:

$$\text{Smoothed}_t = \frac{\text{Smoothed}_{t-1} \times (N - 1) + \text{Value}_t}{N}$$

**Seed Initialization:** The initial smoothed value at period $t = N - 1$ is the simple arithmetic mean of the first $N$ observations:
$$\text{Smoothed}_{N-1} = \frac{1}{N} \sum_{i=0}^{N-1} \text{Value}_i$$

### 2. Relative Strength Index (RSI - 14)
Measures the velocity and magnitude of price momentum:
$$\Delta_t = \text{Close}_t - \text{Close}_{t-1}$$
$$\text{Gain}_t = \max(\Delta_t, 0), \quad \text{Loss}_t = \max(-\Delta_t, 0)$$
$$\text{AvgGain}_t = \text{WilderSmooth}(\text{Gain}, 14), \quad \text{AvgLoss}_t = \text{WilderSmooth}(\text{Loss}, 14)$$
$$\text{RS}_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}$$
$$\text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$
- **Interpretation:** $\text{RSI} > 70$ signals overbought momentum; $\text{RSI} < 30$ signals oversold conditions.

### 3. Average Directional Index (ADX - 14, +DI, -DI)
Measures trend strength regardless of direction:
1. **True Range (TR):** $\max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$
2. **Directional Movement:**
   - $\text{UpMove} = \text{High}_t - \text{High}_{t-1}$
   - $\text{DownMove} = \text{Low}_{t-1} - \text{Low}_t$
   - $+\text{DM} = \text{UpMove} \text{ if } (\text{UpMove} > \text{DownMove} \text{ and } \text{UpMove} > 0) \text{ else } 0$
   - $-\text{DM} = \text{DownMove} \text{ if } (\text{DownMove} > \text{UpMove} \text{ and } \text{DownMove} > 0) \text{ else } 0$
3. **Smoothed Series:** $\text{ATR} = \text{WilderSmooth}(\text{TR}, 14)$, $\text{Smooth}(+\text{DM})$, $\text{Smooth}(-\text{DM})$
4. **Directional Indicators:**
   $$+\text{DI} = 100 \times \frac{\text{Smooth}(+\text{DM})}{\text{ATR}}, \quad -\text{DI} = 100 \times \frac{\text{Smooth}(-\text{DM})}{\text{ATR}}$$
5. **Directional Index (DX) & ADX:**
   $$\text{DX} = 100 \times \frac{|+\text{DI} - -\text{DI}|}{+\text{DI} + -\text{DI}}, \quad \text{ADX} = \text{WilderSmooth}(\text{DX}, 14)$$
- **Interpretation:** $+\text{DI} > -\text{DI}$ indicates a Bullish Uptrend; $-\text{DI} > +\text{DI}$ indicates a Bearish Downtrend. $\text{ADX} > 25$ denotes a strong trend; $\text{ADX} < 20$ indicates range-bound consolidation.

### 4. Relative Strength Spread (63 Days)
Evaluates medium-term quarterly alpha against the broad market index:
$$\text{Return}_{\text{stock}} = \left( \frac{P_{\text{latest}} - P_{\text{latest}-63}}{P_{\text{latest}-63}} \right) \times 100$$
$$\text{Return}_{\text{Nifty500}} = \left( \frac{B_{\text{latest}} - B_{\text{latest}-63}}{B_{\text{latest}-63}} \right) \times 100$$
$$\text{RS Score (pp)} = \text{Return}_{\text{stock}} - \text{Return}_{\text{Nifty500}}$$
- **Interpretation:** Positive spread indicates institutional accumulation and relative market outperformance over a 1-quarter (~3-month) cycle.

### 5. Support and Resistance (Swing Pivots)
Using a 20-day rolling window:
- **Swing Highs:** Sessions where $\text{High}_t = \max(\text{High}_{t-19 \dots t})$.
- **Swing Lows:** Sessions where $\text{Low}_t = \min(\text{Low}_{t-19 \dots t})$.
- **Nearest Support:** Highest historical swing low strictly below current price:
  $$\text{Support} = \max(\{L \in \text{Swing Lows} \mid L < P_{\text{current}}\})$$
- **Nearest Resistance:** Lowest historical swing high strictly above current price:
  $$\text{Resistance} = \min(\{H \in \text{Swing Highs} \mid H > P_{\text{current}}\})$$

### 6. Volatility, Historical Return & Hybrid Stop-Loss (`stoploss.py`)
- **Daily returns:** $r_t = \text{Close}_t / \text{PrevClose}_t - 1$ over the trailing 252 sessions, using NSE's `PREV_CLOSE` so every return is a true one-session move even when sessions are missing from the local history (and corporate actions are exchange-adjusted).
- **Volatility:** sample standard deviation $\sigma_d$ of daily returns; annualized as $\sigma_d\sqrt{252}$.
- **Historical expected return (placeholder):** mean daily return $\times 252$. May be replaced by a CAPM-implied return once portfolio beta is computed.
- **Weight (placeholder):** equal weighting, $100/8 = 12.5\%$ per stock, until formal weight assignment within the capping constraints is completed.
- **Stop-loss:** the tighter (closer to price) of two candidates, recording which one won:
  1. *Support:* the nearest support level (Section 5).
  2. *Volatility cap:* $P \times (1 - k\,\sigma_d\sqrt{N})$ with $k = 1.75$ and $N = 21$ trading days (about one month, for a 3-month mandate whose stops are reviewed monthly). Both are stated, adjustable assumptions in `config.py`.

---

## Project Structure

```
IAPFDOF/
├── data/
│   ├── raw_bhavcopy/             # Cached daily NSE Bhavcopy slices (bhav_DD-Mon-YYYY.csv)
│   ├── fundamentals_cache/       # Cached company HTML pages from Screener.in
│   ├── processed/                # Unified historical OHLCV dataset
│   └── nifty500_tri.csv          # Official Nifty 500 Total Returns Index CSV
├── output/
│   ├── portfolio_technical_summary.csv # Single-row summary table for portfolio review
│   ├── portfolio_historical_ohlcv.csv  # Clean historical OHLCV data across universe
│   ├── portfolio_risk_summary.csv      # Locked portfolio volatility, placeholder return/weight, stop-loss
│   ├── fundamentals_screen_check.csv   # Candidate fundamentals vs. the Capital Goods/EPC screen thresholds
│   └── *_full_screen.csv               # Technical-first sector screens (cement, capital_goods, power)
├── tests/
│   ├── __init__.py
│   ├── test_automated_tri.py     # Unit tests for automated TRI retrieval, retries, and fallback
│   ├── test_fundamentals.py      # Unit tests for Screener.in extraction and parsing
│   ├── test_indicators.py        # Unit tests for Wilder's smoothing, RSI, ADX, RS, S/R
│   ├── test_pipeline.py          # Integration tests for Bhavcopy parsing, aliases, benchmarks
│   ├── test_stoploss.py          # Unit tests for volatility and the hybrid stop-loss selection
│   └── test_tri_staleness.py     # Unit tests for the TRI CSV staleness check
├── config.py                     # Universe definitions, URLs, headers, symbol alias mapping
├── fetch_data.py                 # NSE Bhavcopy HTTP client, yfinance downloader, TRI loader
├── fundamentals.py               # Direct Screener.in scraper, ratio parser, and local cache
├── indicators.py                 # Pure Pandas/NumPy technical indicator engine
├── analysis.py                   # Portfolio evaluator, table formatter, CSV exporter
├── stoploss.py                   # Volatility, historical return, and hybrid stop-loss calculations
├── sector_screen.py              # Technical-first, then fundamental, screen of official sector indices
├── main.py                       # CLI entry point orchestrating the end-to-end pipeline
├── app.py                        # Streamlit 5-tab institutional portfolio dashboard
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore configurations
└── README.md                     # Comprehensive academic & practical documentation
```

---

## Installation & Requirements

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd IAPFDOF
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   # Run once to install Chromium for automated benchmark retrieval:
   playwright install chromium
   ```

---

## Running the Dashboard

Launch the interactive Streamlit portfolio terminal:
```bash
streamlit run app.py
```

The web dashboard loads instantly from the existing CSV outputs already in the repository (`output/portfolio_technical_summary.csv` and `output/portfolio_historical_ohlcv.csv`) without requiring network re-fetching.

### Dashboard Architecture (5 Tabs)
1. **Portfolio Overview:** Dense, institutional summary metrics and sector-grouped constituent tables for the 8 locked stocks (`Cement`, `Capital Goods/EPC`, `Power`) with visible investment committee notes (including the NTPC fundamental inclusion caveat).
2. **Fundamentals:** Screener criteria view (Market Cap, Price, ROCE, 3-Yr Avg ROCE, ROE, Debt/Equity, Operating Cash Flow, OPM, 3-Yr Sales Growth, 3-Yr Profit Growth) scraped directly from Screener.in company pages by `fundamentals.py` and cached under `data/fundamentals_cache/`.
3. **Technicals:** Full technical summary table with subtle green/red trend direction tinting, a separate **Risk, Sizing & Stop-Loss** section (volatility, placeholder expected return and weight, hybrid stop-loss and the winning method), and an interactive 1-year OHLCV line chart with horizontal Support and Resistance reference levels.
4. **Risk & Hedging:** *(Module in Progress)* Beta regression, explained/unexplained risk decomposition, and hedge ratio analysis.
5. **Performance:** *(Module in Progress)* Sharpe ratio, Treynor ratio, XIRR, and Capital Market Line (scheduled for 28th September snapshot).

---

## How to Run the Pipeline

### 1. Default Run (1-Year Bhavcopy Download & Analysis)
Fetches 1 year of daily bhavcopies directly from NSE, downloads Nifty 500, and outputs the summary:
```bash
python main.py
```

### 2. Fast / Cached Mode (No Network Re-fetching)
Once data is downloaded, re-run indicator analysis instantly without calling NSE:
```bash
python main.py --skip-fetch
```

### 3. Custom Date Range
```bash
python main.py --start-date 2025-09-24 --end-date 2026-09-23
```

### 4. Custom Nifty 500 TRI CSV
```bash
python main.py --tri-csv path/to/nifty500_tri.csv
```

### 5. Automated TRI Retrieval (Opt-In / Experimental)
```bash
python main.py --tri-source automated
```

---

## Automated TRI Fetching (Experimental)

### Why Browser Automation is Needed
The benchmark Nifty 500 Total Returns Index (TRI) series is hosted at [niftyindices.com/reports/historical-data](https://niftyindices.com/reports/historical-data). Unlike standardized price return series available on Yahoo Finance (`^CRSLDX`), TRI historical records are only served to a real browser session:
- The site sits behind **Akamai Bot Manager**. The page's data call (`/BackPage/getTotalReturnIndexString` — the older `Backpage.aspx/...` paths are retired and return the HTML shell) is rejected with `403 Access Denied` unless the request carries the Akamai session cookies that the site's own scripts set in a browser.
- Playwright's default headless build (`chromium_headless_shell`) is blocked outright (403, or a stalled connection that surfaces as a navigation timeout). The routine therefore launches the **full Chromium build in new-headless mode** (`channel="chromium"`) with `HeadlessChrome` stripped from the user agent. Akamai still rejects roughly 1 in 4 fresh sessions, so navigation is retried (up to 4 times, fresh browser context each time, with backoff).
- The portal accepts at most 365 days per query, so longer ranges are requested in ≤365-day windows and stitched together.
- The "csv format" link builds the file client-side from the results table, so the downloaded CSV has exactly the manual schema (`IndexName, Date, Total Returns Index, Net Total Return Index`) and is parsed by the same `load_benchmark_tri()` routine.

### How to Enable
Automated fetching is strictly opt-in and disabled by default:
```bash
# Opt-in to automated browser retrieval:
python main.py --tri-source automated

# Combine with cached stock data:
python main.py --skip-fetch --tri-source automated
```

### Requirements & Setup
Requires Playwright ≥ 1.49 and the full Chromium build (installed alongside the headless shell by):
```bash
playwright install chromium
```
Behind a TLS-intercepting proxy (corporate network, sandboxed CI), Chromium must trust the proxy's CA via its NSS store (`certutil -A -d sql:$HOME/.pki/nssdb -n proxy-ca -t "C,," -i <ca.crt>`); otherwise navigation fails with `net::ERR_CERT_AUTHORITY_INVALID`. Datacenter IPs (e.g. Google Colab) may be scored more harshly by Akamai than residential connections.

### Reliability & Fragility Disclaimer
> [!WARNING]
> **Experimental Feature:** Automated browser retrieval depends directly on `niftyindices.com`'s active DOM layout, element IDs (`#HistoricalMenu`/`#maindd li.form5`, `#ddlHistoricalreturntypee*`, `#submit_totalindexhistorical`, `#exportTotalindex`), client-side scripts, and on Akamai continuing to admit the automated session. If the portal redesigns its layout or tightens bot detection, automated retrieval may fail.

### Fail-Safe Fallback Guarantee
The automated fetch routine in `fetch_benchmark_tri_automated()` is completely wrapped in a defensive `try...except` block. If any step fails (network timeout, element not found, download failure, missing Playwright browser binary), the pipeline logs a detailed warning and **automatically falls back** to `load_benchmark_tri()` using the local manual CSV at `data/nifty500_tri.csv`. Running `python main.py` with no flags remains the default, proven-working, and most reliable production path.

---

## Sector Screen (Technical First, Then Fundamental)

`python sector_screen.py` applies the brief's order ("technical analysis, then financial analysis") identically to each sector:

1. **Universe:** the official Nifty Cement (16), Nifty Capital Goods (50) and Nifty Power (21) constituents, stored as downloaded from niftyindices.com in `data/index_constituents/`. No manual additions.
2. **Technical screen (every constituent):** RS vs Nifty 500 over 63 sessions **> +2 pp** (a margin: RS is a 63-day cumulative spread and one day's return can move it by several points, so a bare `> 0` flips on noise) **and** trend direction (+DI vs −DI) Bullish, computed with the existing indicator pipeline on the complete Bhavcopy history. ADX is reported as a tiebreaker, not a cutoff.
3. **Fundamental safety screen (technical passers only):** fetched live via `get_fundamentals_summary(..., use_cache=False)` (Screener.in for financials, NSE pledge disclosures for promoter pledge) and scored against the sector's criteria in `config.py`. A metric that cannot be read fails. These are deliberately **light, current-year solvency checks** for a 3-month tactical mandate, not a multi-year quality bar (no 3-year averages or growth):

   | Sector | Criteria |
   | :--- | :--- |
   | Cement | Market Cap > 1000 Cr, ROCE > 8%, OPM > 10%, CFO last year > 0, D/E < 1.5, Pledge < 15% |
   | Capital Goods | Market Cap > 1000 Cr, ROCE > 8%, OPM > 8%, CFO last year > 0, D/E < 1.5, Pledge < 15% |
   | Power | Market Cap > 2000 Cr, ROCE > 6%, Interest coverage > 1.5, CFO last year > 0, Pledge < 15% |

   Pledge is "% of promoter holding pledged" from NSE's corporate pledge data: Screener's public page only mentions pledge in its Cons text at high levels (~40%+), so its absence is not evidence of zero. `--as-of YYYY-MM-DD` pins the price window's end date.

Outputs `output/cement_full_screen.csv`, `output/capital_goods_full_screen.csv` and `output/power_full_screen.csv`, one row per constituent: technical metrics and `passed_technical_screen` for all; for technical passers, each fundamental metric with its `pass_<metric>` flag, `passed_fundamental_screen` and `failed_criteria`; and `passes_both_screens`.

## Output Format & Column Definitions

The pipeline prints a formatted table and saves `output/portfolio_technical_summary.csv` containing:

| Column | Data Type | Description & Financial Interpretation |
| :--- | :--- | :--- |
| `symbol` | String | Official NSE equity ticker. |
| `sector` | String | Sector classification: `Cement`, `Capital Goods/EPC`, or `Power`. |
| `current_price` | Float (INR) | Latest official NSE closing price. |
| `latest_rsi` | Float [0-100] | 14-period Wilder's RSI. Above 70 = Overbought; Below 30 = Oversold. |
| `latest_adx` | Float | 14-period Wilder's ADX trend strength. > 25 = Strong Trend; < 20 = Weak / Choppy. |
| `trend_direction` | String | Direction based on $+DI$ vs $-DI$ (`Bullish (Uptrend)` or `Bearish (Downtrend)`). |
| `rs_score_vs_nifty500` | Float (pp) | Relative Strength spread vs Nifty 500 over the past 63 trading days (approx. 1 quarter). |
| `nearest_support` | Float (INR) | Nearest key price floor below current price (stop-loss reference). |
| `nearest_resistance` | Float (INR) | Nearest price ceiling above current price (`ATH / Blue Sky` if breaking new highs). |

It also saves `output/portfolio_risk_summary.csv`, one row per locked portfolio stock:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `symbol`, `sector`, `current_price` | String / Float (INR) | As above. |
| `annualized_volatility_pct` | Float (%) | Sample std. dev. of daily returns x sqrt(252). |
| `historical_expected_return_pct` | Float (%) | **Placeholder:** mean daily return x 252 (may become CAPM-implied). |
| `weight_pct` | Float (%) | **Placeholder:** equal weight (12.5%) pending formal weight assignment. |
| `stop_loss_price` | Float (INR) | Tighter of nearest support and the volatility cap. |
| `stop_loss_pct_below_current` | Float (%) | Distance of the stop below the current price. |
| `stop_loss_method` | String | `support` or `volatility_cap` (whichever was tighter); `unavailable` if neither could be computed. |

`output/fundamentals_screen_check.csv` is an earlier one-off check of CEMPRO and SCHNEIDER against the *previous* long-term Capital Goods screen (3-year ROCE and growth). It is kept for the record; the current screens are the lightened safety checks described under Sector Screen.

---

## Running the Test Suite

Run the full automated unit test suite with `pytest`:
```bash
pytest tests/ -v
```

All 16 tests verify:
- Exact convergence of Wilder's smoothing against recursive mathematical definitions.
- Boundary conditions for RSI ($RSI = 100$ in monotonic gains, $RSI = 0$ in monotonic losses).
- Directional movement calculations and trend indicators for ADX.
- 63-day Relative Strength alpha spread calculations.
- Support and resistance level identification.
- NSE Bhavcopy CSV whitespace stripping and symbol alias replacement.

---

## Corporate Action & Symbol Renaming Handling

To guarantee uninterrupted 1-year historical price series without artificial gaps:
1. **`GET&D` $\rightarrow$ `GVT&D`:** GE T&D India Limited rebranded to **GE Vernova T&D India Limited** in late 2024.
2. **`ITDCEM` $\rightarrow$ `CEMPRO`:** ITD Cementation India Limited was renamed to **Cemindia Projects Limited** on September 17, 2025.

The pipeline automatically maps historical tickers to the unified current tickers during Bhavcopy ingestion via `SYMBOL_ALIASES` in `config.py`.
