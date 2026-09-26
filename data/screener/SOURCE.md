# Screener.in sector exports (single source of truth for the universe)

Downloaded 26-Sep-2026 from screener.in sector pages, all rows, one file per sector:

- `capital_goods.csv`: Capital Goods (887 companies, all five industry groups)
- `power.csv`: Power (49 companies)
- `cement.csv`: Cement & Cement Products (42 companies)

Universe rule (`config._read_screener`): NSE-listed, market cap >= Rs 5,000 cr; Capital Goods without
Aerospace & Defense, Packaging, Rubber, Glass, Aluminium/Copper/Zinc products, commercial vehicles,
tractors and vehicle dealers; no InvITs; 20 named theme exclusions (`config.THEME_EXCLUSIONS`).

Funnel: 978 rows -> 620 with an NSE code (358 BSE-only, all under Rs 3,300 cr) -> 186 at Rs 5,000 cr
or more -> minus 17 Aerospace & Defense, 8 excluded industries, 2 InvITs and 20 theme exclusions
= 139 (Cement 15, Capital Goods 102, Power 22). The one-year-history rule is applied at selection.
