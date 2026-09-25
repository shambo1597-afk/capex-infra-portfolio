# Nifty Power review table: notes

`power_full_review_table.csv` lists every Nifty Power constituent (21 rows,
none excluded), with live fundamentals and technicals as of 24-Sep-2026 (price window
24-Sep-2025 to 24-Sep-2026). It is sorted by `fundamentals_passed_count` (descending),
then `sector_rank` (ascending).

## Two relative strength measures

**`rs_score_vs_nifty500`** measures whether a stock beats the broad market: its 63-session
cumulative return minus the Nifty 500 price index's (NSE official closes), in percentage points. It bears on whether
Power as a theme deserves capital at all, versus simply holding the index.
**`rs_score_vs_sector_avg`** measures which Power stock is best positioned relative to its
Power peers: the same 63-session return minus the equal-weighted average return of all
21 constituents (-9.50% over this window). It is the relevant measure once the
decision to hold Power exposure has been made, per the project's sector-rotation requirement.

`sector_rank` ranks `rs_score_vs_sector_avg` from 1 (best) to 21 (worst); the spreads
sum to approximately zero by construction.
