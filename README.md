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

## Portfolio Universe & Locked Portfolio

**Universes.** The three sector universes are the official Nifty sector index constituent files in `data/index_constituents/` (niftyindices.com, downloaded 24-Sep-2026): Nifty Cement (16), Nifty Capital Goods (50) and Nifty Power (21), 87 stocks in total. `config.py` reads them at import time, so the universe lists, each symbol's sector and each company's name all come from these files; nothing is hand-typed.

**Locked portfolio (8 stocks, `config.LOCKED_PORTFOLIO`).** Chosen from the technical and fundamental screens (`output/*_full_review_table.csv`) and the Relative Rotation Graph analysis (`rrg.py`).

| Sector | Stocks | Symbols |
| :--- | :---: | :--- |
| **Cement** | 1 | `JKCEMENT` |
| **Capital Goods** | 5 | `VOLTAMP`, `FINCABLES`, `APARINDS`, `WELCORP`, `APLAPOLLO` |
| **Power** | 2 | `ACMESOLAR`, `TATAPOWER` |

Screen exceptions (shown on the dashboard, generated from the review tables): `JKCEMENT` and `TATAPOWER` have RS vs Nifty 500 below the +2 pp technical margin; `APLAPOLLO` fails only the Capital Goods OPM criterion (7.98% vs > 8%) and is held after a manual business-model review, as flagged by `high_turnover_business_flag`.

**Conviction tier** (dashboard badge, `rrg.conviction_tier`): **High** = LEADING vs both the Nifty 500 and the sector average; **Moderate** = LEADING in one view only, or IMPROVING in either; **Low (sector-coverage hold)** = WEAKENING or LAGGING in both views.

`config.LOCKED_PORTFOLIO` is the single definition used by the pipeline's risk summary, the dashboard, and `sector_screen.py` (`--locked-check`, and the exclusion list for `--tenth-sweep`).

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

### 6. Volatility, Historical Return & ATR Stop-Loss (`stoploss.py`)
- **Daily returns:** $r_t = \text{Close}_t / \text{PrevClose}_t - 1$ over the trailing 252 sessions, using NSE's `PREV_CLOSE` so every return is a true one-session move even when sessions are missing from the local history (and corporate actions are exchange-adjusted).
- **Volatility:** sample standard deviation $\sigma_d$ of daily returns; annualized as $\sigma_d\sqrt{252}$.
- **Historical expected return (placeholder):** mean daily return $\times 252$. May be replaced by a CAPM-implied return once portfolio beta is computed.
- **Weight (placeholder):** equal weighting, $100/N$ per stock ($100/8 = 12.5\%$ for the 8 picks), until formal weight assignment within the capping constraints is completed.
- **ATR:** $\text{TR}_t = \max(H_t - L_t,\ |H_t - \text{PrevClose}_t|,\ |L_t - \text{PrevClose}_t|)$, Wilder-smoothed over 14 sessions.
- **Stop-loss (3-month mandate):** sized for about one month and trailed at monthly reviews, rather than sized for the whole quarter. A 63-session volatility stop would sit roughly 19-41% below price for these stocks, a bigger loss than a 3-month tactical trade is expected to earn.
  1. *Base stop:* $P - 3 \times \text{ATR}_{14}$. Three ATRs is close to a one-month, one-standard-deviation move (JKCEMENT: 3 ATR = 8.5% vs $\sigma_{annual}\sqrt{21/252}$ = 9.4%), so the stop sits just outside ordinary noise.
  2. *Support adjustment:* if a support level (Section 5) lies below the base stop but within 1 ATR of it, the stop moves to support $- 0.25 \times$ ATR, just under that level. A support level closer to the price is ignored: it never makes the stop tighter than 3 ATR. (The earlier rule, "the tighter of support and a volatility cap", produced stops 0.1-0.7% below price.)
  3. *Trailing:* at each monthly review, run `python main.py --trail-stops <previous risk summary CSV>`. A stop is only ever raised: a higher previous stop is kept (`trailed`), and a previous stop at or above the current price is reported as `breached`.

  All multiples are stated, adjustable assumptions in `config.py` (`STOP_LOSS_ATR_*`, `STOP_LOSS_SUPPORT_*`).

---

## Project Structure

```
IAPFDOF/
├── data/
│   ├── raw_bhavcopy/             # Cached daily NSE Bhavcopy slices (bhav_DD-Mon-YYYY.csv)
│   ├── fundamentals_cache/       # Cached company HTML pages from Screener.in
│   ├── processed/                # Unified historical OHLCV dataset
│   ├── index_constituents/       # Official Nifty Cement / Capital Goods / Power constituent files
│   └── nifty500_tri.csv          # Official Nifty 500 Total Returns Index CSV
├── output/
│   ├── portfolio_technical_summary.csv # Technical summary, one row per universe stock (87)
│   ├── portfolio_historical_ohlcv.csv  # Clean historical OHLCV data across universe
│   ├── portfolio_risk_summary.csv      # Locked portfolio volatility, placeholder return/weight, stop-loss
│   ├── *_full_screen.csv               # Technical-first sector screens (cement, capital_goods, power)
│   ├── *_full_review_table.csv         # Unfiltered per-sector review tables with RRG / DI-gap columns (+ *_review_notes.md)
│   ├── rrg_*.png                       # Relative Rotation Graphs (combined vs Nifty 500; per sector vs sector average)
│   ├── tenth_candidate_sweep.csv       # Run-up / results-date sweep of non-picked stocks
│   └── locked_portfolio_runup_catalyst_check.csv # Same checks on the locked picks
├── tests/
│   ├── __init__.py
│   ├── test_automated_tri.py     # Unit tests for automated TRI retrieval, retries, and fallback
│   ├── test_fundamentals.py      # Unit tests for Screener.in extraction and parsing
│   ├── test_indicators.py        # Unit tests for Wilder's smoothing, RSI, ADX, RS, S/R
│   ├── test_pipeline.py          # Integration tests for Bhavcopy parsing, aliases, benchmarks
│   ├── test_rrg.py               # RS-Momentum, RRG quadrants, DI-gap and OPM-exception flags
│   ├── test_stoploss.py          # Unit tests for volatility, ATR and the trailing ATR stop-loss
│   └── test_tri_staleness.py     # Unit tests for the TRI CSV staleness check
├── config.py                     # Universe definitions, URLs, headers, symbol alias mapping
├── fetch_data.py                 # NSE Bhavcopy HTTP client, yfinance downloader, TRI loader
├── fundamentals.py               # Direct Screener.in scraper, ratio parser, and local cache
├── indicators.py                 # Pure Pandas/NumPy technical indicator engine
├── analysis.py                   # Portfolio evaluator, table formatter, CSV exporter
├── stoploss.py                   # Volatility, historical return, ATR and stop-loss calculations
├── sector_screen.py              # Technical-first, then fundamental, screen of official sector indices
├── rrg.py                        # Relative Rotation Graph quadrants and plots
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
1. **Portfolio Overview:** the 8 locked stocks grouped by sector, one row each with the RRG conviction badge and every field the brief requires: volatility, expected return (historical average, placeholder pending CAPM), weight (placeholder pending final weight assignment), stop-loss with its method, ADX, RS vs Nifty 500, RSI and support/resistance; followed by the screen exceptions.
2. **Fundamentals:** Fundamentals of the locked picks (Market Cap, Price, ROCE, 3-Yr Avg ROCE, ROE, Debt/Equity, Operating Cash Flow, OPM, Interest Coverage, Pledged %, 3-Yr Sales and Profit Growth) plus each sector's safety-screen thresholds, read from `config.SECTOR_SCREENS`, scraped directly from Screener.in company pages by `fundamentals.py` and cached under `data/fundamentals_cache/`.
3. **Technicals:** technical table for the locked stocks (RSI, ADX, trend, DI gap, RS, both RRG quadrants, conviction tier, support/resistance), the embedded RRG plots (combined vs Nifty 500, and a per-sector selector), a separate **Risk, Sizing & Stop-Loss** section (volatility, placeholder expected return and weight, ATR, stop-loss and its method), and an interactive 1-year OHLCV line chart with horizontal Support and Resistance reference levels.
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

### 6. Monthly Stop-Loss Review (Trailing)
```bash
cp output/portfolio_risk_summary.csv output/risk_summary_previous_review.csv
python main.py --trail-stops output/risk_summary_previous_review.csv
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

### 10th-Candidate Sweep

`python sector_screen.py --tenth-sweep --as-of 2026-09-24` writes `output/tenth_candidate_sweep.csv`: every official-index constituent across the three sectors except the current picks (`sector_screen.CURRENT_PICKS`), reusing the review tables' fundamentals and technicals, plus two new fields:

- **Results catalyst:** the next quarterly-results board meeting announced on NSE (`results_date_status` = announced / not announced / unavailable; never estimated), whether it falls within 30 days, and last year's actual September-quarter results date for reference.
- **Run-up heuristic** (simple, not an established indicator): `recent_10day_contribution_pct = RS_last_10_sessions / RS_63_sessions x 100`, defined only when the 63-session RS is positive. Steady outperformance earns ~16% (10/63) of the edge in the last 10 sessions; `recent_spike_flag` marks > 50% (over 3x that pace) as a recent burst that may mean-revert.

`clean_candidate` = all fundamental criteria pass, technically attractive (RS > +2 pp and Bullish), and no recent spike.

### Relative Rotation Graph & full evaluation standard

`python sector_screen.py --review --as-of 2026-09-24` rebuilds all three review tables with the same standard applied to every constituent, and redraws the RRG plots (`python rrg.py --as-of 2026-09-24` redraws them from the committed tables).

- **RRG axes** (both in percentage points, clearly not the proprietary JdK RS-Ratio index): x = 63-session RS; y = RS-Momentum = RS today minus the same 63-session RS ending 10 sessions earlier (`indicators.compute_rs_momentum`). Unlike the run-up share, it is defined for negative RS, which the IMPROVING and LAGGING quadrants need.
- **Quadrants** at (0, 0): LEADING (RS > 0, momentum > 0), WEAKENING (RS > 0, momentum <= 0), LAGGING (RS <= 0, momentum <= 0), IMPROVING (RS <= 0, momentum > 0), computed against the Nifty 500 and against the equal-weighted sector average (sector rotation first, then stock selection).
- **`di_gap`** = +DI - -DI; **`thin_trend_flag`** when |gap| < 2.0 in either direction.
- **`high_turnover_business_flag`**: fails only the OPM criterion, passes all others, ROCE > 20%: flagged for a manual business-model check, never auto-included.
- **`full_standard_candidate`**: all fundamental criteria pass, LEADING vs both benchmarks, and di_gap >= 2.0. The run-up spike flag is reported beside it.
- **Plots** (`output/`): `rrg_all_vs_nifty500.png` (all 87 stocks) and `rrg_<sector>_vs_sector.png` per sector; locked picks are ringed and bold.

## Output Format & Column Definitions

The pipeline prints a formatted table and saves `output/portfolio_technical_summary.csv` containing:

| Column | Data Type | Description & Financial Interpretation |
| :--- | :--- | :--- |
| `symbol` | String | Official NSE equity ticker. |
| `sector` | String | Sector classification: `Cement`, `Capital Goods`, or `Power` (from the official constituent files). |
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
| `weight_pct` | Float (%) | **Placeholder:** equal weight (100/N; 12.5% for 8 stocks) pending formal weight assignment. |
| `atr_14`, `atr_pct` | Float (INR / %) | 14-session Wilder ATR, in rupees and as % of price. |
| `stop_loss_price` | Float (INR) | Price - 3 ATR, or just below a support level up to 1 ATR beyond that; trailed across reviews. |
| `stop_loss_pct_below_current` | Float (%) | Distance of the stop below the current price. |
| `stop_loss_method` | String | `atr`, `support`, `trailed` (previous review's higher stop kept), `breached` (previous stop at or above price: exit), or `unavailable`. |

---

## Running the Test Suite

Run the full automated unit test suite with `pytest`:
```bash
pytest tests/ -v
```

The 169 tests cover, among other things:
- Exact convergence of Wilder's smoothing against recursive mathematical definitions.
- Boundary conditions for RSI ($RSI = 100$ in monotonic gains, $RSI = 0$ in monotonic losses).
- Directional movement calculations and trend indicators for ADX.
- 63-day Relative Strength alpha spread calculations.
- Support and resistance level identification.
- NSE Bhavcopy CSV whitespace stripping and symbol alias replacement.
- The universes matching the official constituent files, and `sector_screen.CURRENT_PICKS` being `config.LOCKED_PORTFOLIO_SYMBOLS`.

---

## Corporate Action & Symbol Renaming Handling

To guarantee uninterrupted 1-year historical price series without artificial gaps:
1. **`GET&D` $\rightarrow$ `GVT&D`:** GE T&D India Limited rebranded to **GE Vernova T&D India Limited** in late 2024.
2. **`ITDCEM` $\rightarrow$ `CEMPRO`:** ITD Cementation India Limited was renamed to **Cemindia Projects Limited** on September 17, 2025.

The pipeline automatically maps historical tickers to the unified current tickers during Bhavcopy ingestion via `SYMBOL_ALIASES` in `config.py`.
