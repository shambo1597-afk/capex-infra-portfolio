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
  - **Price Return Index (PRI):** Nifty 500 (`^CRSLDX`) via `yfinance`.
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

### 2. Benchmark (Price Return) — Nifty 500 via yfinance
- **Ticker:** `^CRSLDX` (the verified Yahoo Finance ticker for Nifty 500).
- **Rationale:** Index series do not suffer from individual corporate action adjustment errors; yfinance provides an accurate, adjusted historical index series.

### 3. Benchmark (Total Return Index) — Local CSV
- **Rationale:** Automated scraping of `niftyindices.com` is brittle due to heavy client-side JavaScript rendering and frequently updated API payloads. The pipeline reads the official manual CSV with schema:
  `IndexName, Date, Total Returns Index, Net Total Return Index`

---

## Portfolio Universe

| Sector | Status | Symbols | Notes |
| :--- | :--- | :--- | :--- |
| **Cement** | Locked | `JKCEMENT`, `ULTRACEMCO`, `STARCEMENT` | Core portfolio exposure |
| **Capital Goods / EPC** | Candidates | `ABB`, `CGPOWER`, `GVT&D`, `POWERINDIA`, `ELECON`, `TRITURBINE`, `TDPOWERSYS`, `SIEMENS`, `SCHNEIDER`, `CEMPRO` | 10 candidates screened for momentum & breakout |
| **Power** | Placeholder | `POWER_SECTOR_STOCKS = []` | Reserved list variable in `config.py` for future addition |

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

---

## Project Structure

```
IAPFDOF/
├── data/
│   ├── raw_bhavcopy/             # Cached daily NSE Bhavcopy slices (bhav_DD-Mon-YYYY.csv)
│   ├── processed/                # Unified historical OHLCV dataset
│   └── nifty500_tri.csv          # Official Nifty 500 Total Returns Index CSV
├── output/
│   ├── portfolio_technical_summary.csv # Single-row summary table for portfolio review
│   └── portfolio_historical_ohlcv.csv  # Clean historical OHLCV data across universe
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py        # Unit tests for Wilder's smoothing, RSI, ADX, RS, S/R
│   └── test_pipeline.py          # Integration tests for Bhavcopy parsing, aliases, benchmarks
├── config.py                     # Universe definitions, URLs, headers, symbol alias mapping
├── fetch_data.py                 # NSE Bhavcopy HTTP client, yfinance downloader, TRI loader
├── indicators.py                 # Pure Pandas/NumPy technical indicator engine
├── analysis.py                   # Portfolio evaluator, table formatter, CSV exporter
├── main.py                       # CLI entry point orchestrating the end-to-end pipeline
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
2. **Fundamentals:** Screener criteria view (ROCE, 3Yr Avg ROCE, OPM, Debt/Equity, Operating Cash Flow, 3Yr Sales Growth, 3Yr Profit Growth) dynamically loading Screener.in CSV exports from `data/fundamentals/`.
3. **Technicals:** Full technical summary table with subtle green/red trend direction tinting and an interactive 1-year OHLCV line chart with horizontal Support and Resistance reference levels.
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

---

## Output Format & Column Definitions

The pipeline prints a formatted table and saves `output/portfolio_technical_summary.csv` containing:

| Column | Data Type | Description & Financial Interpretation |
| :--- | :--- | :--- |
| `symbol` | String | Official NSE equity ticker. |
| `sector` | String | Sector classification (`Cement (Locked)` vs `Capital Goods / EPC`). |
| `current_price` | Float (INR) | Latest official NSE closing price. |
| `latest_rsi` | Float [0-100] | 14-period Wilder's RSI. Above 70 = Overbought; Below 30 = Oversold. |
| `latest_adx` | Float | 14-period Wilder's ADX trend strength. > 25 = Strong Trend; < 20 = Weak / Choppy. |
| `trend_direction` | String | Direction based on $+DI$ vs $-DI$ (`Bullish (Uptrend)` or `Bearish (Downtrend)`). |
| `rs_score_vs_nifty500` | Float (pp) | Relative Strength spread vs Nifty 500 over the past 63 trading days (approx. 1 quarter). |
| `nearest_support` | Float (INR) | Nearest key price floor below current price (stop-loss reference). |
| `nearest_resistance` | Float (INR) | Nearest price ceiling above current price (`ATH / Blue Sky` if breaking new highs). |

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
