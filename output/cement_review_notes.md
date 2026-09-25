# Nifty Cement review table: notes

`cement_full_review_table.csv` lists every Nifty Cement constituent (16 rows,
none excluded), with live fundamentals and technicals as of 25-Sep-2026 (price window
25-Sep-2025 to 25-Sep-2026). It is sorted by `fundamentals_passed_count` (descending),
then `sector_rank` (ascending).

## Two relative strength measures

**`rs_score_vs_nifty500`** measures whether a stock beats the broad market: its 63-session
cumulative return minus the Nifty 500 price index's (NSE official closes), in percentage points. It bears on whether
Cement as a theme deserves capital at all, versus simply holding the index.
**`rs_score_vs_sector_avg`** measures which Cement stock is best positioned relative to its
Cement peers: the same 63-session return minus the equal-weighted average return of all
16 constituents (-6.54% over this window). It is the relevant measure once the
decision to hold Cement exposure has been made, per the project's sector-rotation requirement.

`sector_rank` ranks `rs_score_vs_sector_avg` from 1 (best) to 16 (worst); the spreads
sum to approximately zero by construction.

## Full evaluation standard (applied to every row)

- **RRG (Relative Rotation Graph).** x = RS (pp, 63 sessions); y = RS-Momentum = that RS today
  averaged over the last 5 sessions, minus the same average 10 sessions earlier (pp). Both axes are
  plain percentage-point spreads, not the proprietary JdK RS-Ratio index. Quadrants at (0, 0):
  LEADING (RS > 0, momentum > 0), WEAKENING (RS > 0, momentum <= 0), LAGGING (RS <= 0,
  momentum <= 0), IMPROVING (RS <= 0, momentum > 0). Computed against the Nifty 500
  (`rs_momentum_vs_nifty500`, `rrg_quadrant_vs_nifty500`) and against the equal-weighted
  Cement average (`rs_momentum_vs_sector`, `rrg_quadrant_vs_sector`).
- **Trend quality.** `di_gap` = +DI - -DI; `thin_trend_flag` when |di_gap| < 2,
  whichever way it points (the Bullish/Bearish label can flip on one bar).
- **Run-up.** `recent_10day_contribution_pct` = RS over the last 10 sessions / 63-session RS x 100
  (NaN when RS <= 0); `recent_spike_flag` above 50%.
- **`high_turnover_business_flag`** (sectors with an OPM criterion): fails ONLY OPM, passes every
  other criterion, ROCE > 20%. A prompt for a manual business-model check,
  never an automatic pass.
- **`full_standard_candidate`** = every fundamental criterion passed AND LEADING vs both the
  Nifty 500 and the sector AND di_gap >= 2 (a real, bullish trend). The
  spike flag is reported beside it, not folded in.

Plots: `rrg_cement_vs_sector.png` and the combined `rrg_all_vs_nifty500.png`.
