# Screener.in Fundamentals Data Directory (not currently used)

The pipeline and dashboard do **not** read files from this directory. Fundamentals
(ROCE, 3-Yr Avg ROCE, ROE, OPM, Debt/Equity, Operating Cash Flow, 3-Yr Sales and Profit
Growth) are scraped directly from Screener.in company pages by `fundamentals.py` and
cached as HTML under `data/fundamentals_cache/`.

Screener.in CSV exports placed here are kept only for manual reference.
