# Screener.in sector exports (single source of truth for the universe)

Downloaded 26-Sep-2026 from screener.in sector pages, all rows, one file per sector:

- `capital_goods.csv`: Capital Goods (887 companies, all five industry groups)
- `power.csv`: Power (49 companies)
- `cement.csv`: Cement & Cement Products (42 companies)

Universe rule (`config._read_screener`): NSE-listed, market cap >= Rs 5,000 cr; Capital Goods without
Aerospace & Defense, Packaging, Rubber, Glass, Aluminium/Copper/Zinc products, commercial vehicles,
tractors and vehicle dealers; no InvITs; 23 named theme exclusions (`config.THEME_EXCLUSIONS`).

Funnel: 978 rows -> 620 with an NSE code (358 BSE-only, all under Rs 3,300 cr) -> 186 at Rs 5,000 cr
or more -> minus 17 Aerospace & Defense, 8 excluded industries, 2 InvITs and 22 theme exclusions (the 23rd, KSL, is below the floor)
= 137 (Cement 15, Capital Goods 100, Power 22). The one-year-history rule is applied at selection.

Why Rs 5,000 cr (tested 27-Sep-2026, `research/liquidity_study.py`, `output/liquidity_vs_marketcap.csv`):
the floor is kept as a QUALITY floor. It is NOT the liquidity test. Tradability is its own rule at
selection: median daily traded value of at least Rs 5 cr (`config.MIN_TURNOVER_CR`; the largest position,
~Rs 14.6 lakh, is then under 3% of a normal day and under 6% of a day with half the turnover).

- Market cap and 30-day median traded value are strongly related (rank correlation 0.80): 5% of Rs
  500-1,000 cr stocks trade Rs 5 cr a day, 27% at Rs 1,000-2,000 cr, 57-66% at Rs 2,000-5,000 cr, 83% at
  Rs 5,000-10,000 cr and 96-100% above Rs 10,000 cr.
- But not identical, in both directions: 10 stocks above the floor trade under Rs 5 cr a day (BIRLACORPN,
  INDIACEM, PRSMJOHNSN, STARCEMENT, AJAXENGG, BANSALWIRE, ESABINDIA, GRINDWELL, SAATVIKGL, VESUVIUS) and
  are caught by the turnover rule; about 30 below it trade over Rs 10 cr (many recent IPOs).
- Floor Rs 1,000 or 2,000 cr with the turnover rule: identical results (no Rs 1,000-2,000 cr stock passes
  every rule). The universe grows 137 -> 207; the top 15 would gain RATNAVEER, VENUSPIPES, PITTIENG, JASH and
  SAMBHV and put RATNAVEER, VENUSPIPES and PITTIENG in the invested 8 in place of ACE, CARBORUNIV and
  GOODLUCK. Before the profit rule was corrected the two top-ranked newcomers were loss-making
  (KABRAEXTRU, QUADFUTURE); RATNAVEER (stainless sheets and washers) trades 4% of its market value a day.
  The gain is not worth swapping three established, profitable holdings for, so the floor stays.
