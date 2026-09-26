"""
Sector Screen: technical screen first, then fundamental safety screen.

Implements the project brief's order ("Technical analysis, then financial analysis") identically
for each sector:

1. Universe: the Screener.in sector exports (config.SCREENER_FILES, the single source of truth)
   after the universe rule (NSE-listed, market cap >= Rs 5,000 cr, theme industries and the theme
   exclusions in config); fundamentals come from the same files.
2. Technical screen on every constituent, using the existing indicator pipeline
   (analysis.evaluate_stock_technicals) on the complete NSE Bhavcopy history:
   passes when RS vs Nifty 500 (63 sessions) > +2 pp AND trend direction (+DI vs -DI) is Bullish.
   ADX is reported for tie-breaking but is not a cutoff.
3. Fundamental safety screen (fundamentals.get_fundamentals_summary: the Screener export's values)
   ONLY for stocks that passed step 2, against that sector's criteria in config.py.

Writes one CSV per sector (output/cement_full_screen.csv, output/capital_goods_full_screen.csv,
output/power_full_screen.csv) listing every constituent.

Usage:
    python sector_screen.py                  # technical-first screens for all three sectors
    python sector_screen.py --review-cement  # unfiltered Cement review table (no screening)
    python sector_screen.py --review "Capital Goods" Power   # same review table for other sectors
    python sector_screen.py --review --as-of 2026-09-24      # pin the price window's end date
    python sector_screen.py --tenth-sweep --as-of 2026-09-24 # candidates excluding current picks
    python sector_screen.py --locked-check --as-of 2026-09-24 # run-up / results-date check of the picks
"""

import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis import evaluate_stock_technicals
from config import (
    BUSINESS_FOCUS_NOTES,
    DI_GAP_THIN_THRESHOLD,
    EVALUATION_START_DATE,
    FUNDAMENTAL_HARD_FIELDS,
    HIGH_TURNOVER_ROCE_MIN,
    LOCKED_PORTFOLIO_SYMBOLS,
    MIN_HISTORY_SESSIONS,
    PORTFOLIO_SIZE,
    SCREENER_DOWNLOAD_DATE,
    SELECTION_CONVICTION_TIERS,
    SECTOR_MIN_HOLDINGS,
    OUTPUT_DIR,
    RRG_MOMENTUM_DAYS,
    RRG_MOMENTUM_SMOOTHING_DAYS,
    RUNUP_RECENT_DAYS,
    SELECTION_RS_DAYS,
    SELECTION_SKIP_DAYS,
    SECTOR_SCREENS,
    TECHNICAL_RS_LOOKBACK_DAYS,
    TECHNICAL_RS_MARGIN_PP,
    sector_universe,
)
from corporate_actions import adjust_for_corporate_actions
from fetch_data import NSEBhavcopyFetcher, fetch_benchmark_nifty500, get_one_year_date_range
from fundamentals import evaluate_fundamental_screen, fetch_results_calendar, get_fundamentals_summary
from indicators import (
    compute_recent_rs_contribution,
    compute_relative_strength,
    compute_rs_at_offsets,
    compute_rs_momentum,
    compute_sector_relative_strength,
)
from rrg import LEADING, classify_quadrant, conviction_tier

logger = logging.getLogger("sector_screen")


def load_constituents(csv_path: Path) -> pd.DataFrame:
    """Load an official niftyindices.com constituent file (Company Name, Industry, Symbol, ...)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df


def load_sector_universe(sector: str) -> pd.DataFrame:
    """
    Every stock screened for a sector (config.sector_universe: the Screener export after the universe
    rule), with columns Symbol, Company Name, Index (the Screener industry).
    """
    rows = sector_universe(sector)
    return pd.DataFrame({"Symbol": [r["symbol"] for r in rows], "Company Name": [r["name"] for r in rows],
                         "Index": [r["index"] for r in rows]})


def _fundamentals_date(record: dict) -> str:
    """The date the fundamentals describe: the Screener export's download date, else today (live scrape)."""
    return SCREENER_DOWNLOAD_DATE if record.get("source") else date.today().isoformat()


def technical_screen_result(
    rs_score: float,
    trend_direction: str,
    rs_margin_pp: float = TECHNICAL_RS_MARGIN_PP,
) -> Tuple[bool, str]:
    """
    Apply the technical screen: RS vs Nifty 500 > rs_margin_pp (default +2 pp, see
    config.TECHNICAL_RS_MARGIN_PP for why a bare zero is too noisy) AND trend direction Bullish.

    Returns:
        (passed, reason): reason is "" when passed, else why it failed.
    """
    reasons = []
    if rs_score is None or pd.isna(rs_score):
        reasons.append("RS unavailable")
    elif rs_score <= rs_margin_pp:
        reasons.append(f"RS <= {rs_margin_pp:+g} pp")
    if not str(trend_direction).startswith("Bullish"):
        reasons.append(f"trend {str(trend_direction).split(' (')[0]}")
    return not reasons, "; ".join(reasons)


def screen_sector(
    sector: str,
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    criteria: List[Tuple[str, str, float, str]],
) -> pd.DataFrame:
    """
    Run the technical screen on every constituent, then the live fundamental screen on the
    technical passers only. Returns one row per constituent.
    """
    rows: List[Dict] = []
    for _, member in constituents.iterrows():
        symbol = member["Symbol"]
        sym_prices = prices[prices["SYMBOL"] == symbol] if not prices.empty else pd.DataFrame()
        tech = evaluate_stock_technicals(symbol, sym_prices, benchmark, rs_lookback=TECHNICAL_RS_LOOKBACK_DAYS)
        passed, reason = technical_screen_result(tech["rs_score_vs_nifty500"], tech["trend_direction"])
        rows.append({
            "symbol": symbol,
            "company_name": member.get("Company Name"),
            "index": member.get("Index", f"Nifty {sector}"),
            "price_sessions": int(sym_prices["DATE1"].nunique()) if not sym_prices.empty else 0,
            "current_price": tech["current_price"],
            "passed_technical_screen": passed,
            "technical_fail_reason": reason,
            "rs_score": tech["rs_score_vs_nifty500"],
            "adx": tech["latest_adx"],
            "plus_di": tech.get("plus_di"),
            "minus_di": tech.get("minus_di"),
            "trend_direction": tech["trend_direction"],
            "rsi": tech["latest_rsi"],
        })
    screen = pd.DataFrame(rows)

    # Fundamental safety screen: live Screener.in data, technical passers only
    passers = screen.loc[screen["passed_technical_screen"], "symbol"].tolist()
    fundamentals = get_fundamentals_summary(passers, use_cache=False) if passers else pd.DataFrame()
    records = {r["symbol"]: r for r in fundamentals.to_dict("records")} if not fundamentals.empty else {}

    metric_cols = [c for field, *_ in criteria for c in (field, f"pass_{field}")]
    for col in ["fundamentals_as_of", "fundamentals_status", "pledged_as_of", *metric_cols, "passed_fundamental_screen", "failed_criteria"]:
        screen[col] = pd.Series([None] * len(screen), dtype="object")
    for i, row in screen.iterrows():
        if not row["passed_technical_screen"]:
            continue
        record = records.get(row["symbol"], {"status": "Data Unavailable"})
        result = evaluate_fundamental_screen(record, criteria)
        failed = result.pop("failed_criteria")
        for col, value in result.items():
            if isinstance(value, float) and math.isinf(value):
                value = "inf"  # e.g. zero interest expense
            screen.at[i, col] = value
        screen.at[i, "fundamentals_as_of"] = _fundamentals_date(record)
        screen.at[i, "fundamentals_status"] = record.get("status")
        screen.at[i, "pledged_as_of"] = record.get("pledged_as_of")
        screen.at[i, "passed_fundamental_screen"] = not failed
        screen.at[i, "failed_criteria"] = "; ".join(failed)

    screen["passes_both_screens"] = screen["passed_technical_screen"] & (screen["passed_fundamental_screen"] == True)  # noqa: E712
    return screen


def run_sector_screens(
    sectors: Optional[List[str]] = None,
    use_price_cache: bool = True,
    as_of: Optional[date] = None,
) -> Dict[str, pd.DataFrame]:
    """Screen each configured sector (price window ending as_of, default today) and write its CSV."""
    sectors = sectors or list(SECTOR_SCREENS)
    constituents = {s: load_sector_universe(s) for s in sectors}
    all_symbols = sorted({sym for df in constituents.values() for sym in df["Symbol"]})

    start, end = get_one_year_date_range(as_of)
    prices = adjust_for_corporate_actions(
        NSEBhavcopyFetcher().fetch_date_range(start, end, all_symbols, use_cache=use_price_cache))
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.isoformat())
    if benchmark.empty:
        raise RuntimeError("Nifty 500 benchmark unavailable; RS cannot be computed.")

    results = {}
    for sector in sectors:
        cfg = SECTOR_SCREENS[sector]
        screen = screen_sector(sector, constituents[sector], prices, benchmark, cfg["criteria"])
        out = Path(cfg["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        screen.to_csv(out, index=False)
        logger.info("Saved %s screen (%d constituents) to %s", sector, len(screen), out.resolve())
        results[sector] = screen
    return results


def build_review_table(
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    criteria: List[Tuple[str, str, float, str]],
) -> pd.DataFrame:
    """
    Unfiltered review table: one row per constituent with live fundamentals (each with a
    pass/fail flag against the reference criteria) and technicals side by side.

    Nothing is screened out; the flags, fundamentals_passed_count and technically_attractive
    are informational. Two RS measures over the same 63-session window:
    rs_score_vs_nifty500 (vs the broad market) and rs_score_vs_sector_avg (vs the equal-weighted
    average return of all constituents), with sector_rank 1 = best within the sector.

    Full evaluation standard, applied identically to every constituent (see add_evaluation_columns):
    RRG quadrant and momentum vs both benchmarks, DI-gap trend quality, the 10-day run-up share,
    the OPM-exception review flag and full_standard_candidate.
    Sorted by fundamentals_passed_count descending, then sector_rank ascending.
    """
    symbols = constituents["Symbol"].tolist()
    fundamentals = get_fundamentals_summary(symbols, use_cache=False)
    records = {r["symbol"]: r for r in fundamentals.to_dict("records")} if not fundamentals.empty else {}

    rows = []
    stock_returns: Dict[str, float] = {}
    # {offset: {symbol: 63-session return for the window ending `offset` sessions ago}}
    offset_returns: Dict[int, Dict[str, float]] = {}
    for _, member in constituents.iterrows():
        symbol = member["Symbol"]
        record = records.get(symbol, {"status": "Data Unavailable"})
        result = evaluate_fundamental_screen(record, criteria)
        failed = result.pop("failed_criteria")
        sym_prices = prices[prices["SYMBOL"] == symbol] if not prices.empty else pd.DataFrame()
        tech = evaluate_stock_technicals(symbol, sym_prices, benchmark, rs_lookback=TECHNICAL_RS_LOOKBACK_DAYS)
        attractive, _ = technical_screen_result(tech["rs_score_vs_nifty500"], tech["trend_direction"])
        # The stock's own 63-session return, exactly as used for rs_score_vs_nifty500
        _, stock_return_pct, _ = compute_relative_strength(sym_prices, benchmark, lookback_days=TECHNICAL_RS_LOOKBACK_DAYS)
        stock_returns[symbol] = stock_return_pct
        _, rs_prev, rs_momentum, returns_by_offset = compute_rs_momentum(
            sym_prices, benchmark, TECHNICAL_RS_LOOKBACK_DAYS, RRG_MOMENTUM_DAYS, RRG_MOMENTUM_SMOOTHING_DAYS)
        for offset, ret in returns_by_offset.items():
            offset_returns.setdefault(offset, {})[symbol] = ret
        rs_6m_skip1m = compute_rs_at_offsets(
            sym_prices, benchmark, SELECTION_RS_DAYS, [SELECTION_SKIP_DAYS])[SELECTION_SKIP_DAYS][0]
        rs_10d, _, recent_pct = compute_recent_rs_contribution(
            sym_prices, benchmark, RUNUP_RECENT_DAYS, TECHNICAL_RS_LOOKBACK_DAYS)
        rows.append({
            "symbol": symbol,
            "company_name": member.get("Company Name"),
            "source_index": member.get("Index"),
            "business_focus_note": BUSINESS_FOCUS_NOTES.get(symbol),
            **result,
            "fundamentals_passed_count": len(criteria) - len(failed),
            "fundamentals_failed": "; ".join(failed),
            "fundamentals_status": record.get("status"),
            "fundamentals_as_of": _fundamentals_date(record),
            "pledged_as_of": record.get("pledged_as_of"),
            "current_price": tech["current_price"],
            "latest_rsi": tech["latest_rsi"],
            "latest_adx": tech["latest_adx"],
            "plus_di": tech.get("plus_di"),
            "minus_di": tech.get("minus_di"),
            "trend_direction": tech["trend_direction"],
            "rs_score_vs_nifty500": tech["rs_score_vs_nifty500"],
            "nearest_support": tech["nearest_support"],
            "nearest_resistance": tech["nearest_resistance"],
            "technically_attractive": attractive,
            "price_sessions": int(sym_prices["DATE1"].nunique()) if not sym_prices.empty else 0,
            "rs_score_vs_nifty500_prev_avg": rs_prev,
            "rs_momentum_vs_nifty500": rs_momentum,
            "rs_last_10d": rs_10d,
            "recent_10day_contribution_pct": recent_pct,
            "rs_6m_skip1m": rs_6m_skip1m,
        })
    table = pd.DataFrame(rows)

    # Only stocks with the full RRG history (a 63-session window at every momentum offset) enter the
    # sector average, the same members at every offset: for a recent listing compute_relative_strength
    # falls back to the history it has, a shorter window than everyone else's
    sessions = dict(zip(table["symbol"], table["price_sessions"]))
    min_sessions = TECHNICAL_RS_LOOKBACK_DAYS + RRG_MOMENTUM_DAYS + RRG_MOMENTUM_SMOOTHING_DAYS

    def _full_window(returns: Dict[str, float]) -> Dict[str, float]:
        return {sym: (ret if sessions.get(sym, 0) >= min_sessions else float("nan"))
                for sym, ret in returns.items()}

    # Within-sector relative strength vs the equal-weighted average of all constituents
    sector_rs, sector_avg = compute_sector_relative_strength(_full_window(stock_returns))
    table["rs_score_vs_sector_avg"] = table["symbol"].map(lambda sym: round(sector_rs[sym], 2)
                                                          if not pd.isna(sector_rs[sym]) else float("nan"))
    table["sector_rank"] = table["rs_score_vs_sector_avg"].rank(ascending=False, method="min").astype("Int64")
    table.attrs["sector_avg_return_pct"] = sector_avg

    # Momentum vs sector, smoothed exactly like the vs-Nifty-500 one: the sector spread is recomputed
    # for each window end, then the average of the latest RRG_MOMENTUM_SMOOTHING_DAYS minus the
    # average of the same number of days RRG_MOMENTUM_DAYS sessions earlier
    spreads = {k: compute_sector_relative_strength(_full_window(rets))[0] for k, rets in offset_returns.items()}
    now_k = range(RRG_MOMENTUM_SMOOTHING_DAYS)
    prev_k = range(RRG_MOMENTUM_DAYS, RRG_MOMENTUM_DAYS + RRG_MOMENTUM_SMOOTHING_DAYS)

    def _sector_momentum(symbol: str) -> float:
        vals_now = [spreads.get(k, {}).get(symbol, float("nan")) for k in now_k]
        vals_prev = [spreads.get(k, {}).get(symbol, float("nan")) for k in prev_k]
        if any(pd.isna(v) for v in vals_now + vals_prev):
            return float("nan")
        return round(sum(vals_now) / len(vals_now) - sum(vals_prev) / len(vals_prev), 2)

    table["rs_momentum_vs_sector"] = table["symbol"].map(_sector_momentum)

    table = add_evaluation_columns(table, criteria)
    return table.sort_values(["fundamentals_passed_count", "sector_rank"],
                             ascending=[False, True], na_position="last").reset_index(drop=True)


def add_evaluation_columns(table: pd.DataFrame, criteria: List[Tuple[str, str, float, str]]) -> pd.DataFrame:
    """
    Derived columns of the full evaluation standard (inputs already in the review table):

      rrg_quadrant_vs_nifty500 / rrg_quadrant_vs_sector: RRG quadrant (rrg.classify_quadrant) of
        (RS, RS-Momentum) against the Nifty 500 and against the equal-weighted sector average;
      di_gap: +DI - -DI (signed); thin_trend_flag: |di_gap| < DI_GAP_THIN_THRESHOLD, whichever
        direction it points;
      recent_spike_flag: recent_10day_contribution_pct > RECENT_SPIKE_THRESHOLD_PCT;
      fundamentals_clean: every fundamental criterion passed;
      high_turnover_business_flag: fails ONLY the OPM criterion, passes every other criterion, and
        ROCE > HIGH_TURNOVER_ROCE_MIN -> candidate for a manual business-model check, never an
        automatic pass (NA in sectors without an OPM criterion, i.e. Power);
      full_standard_candidate: fundamentals_clean AND LEADING vs both benchmarks AND
        di_gap >= DI_GAP_THIN_THRESHOLD (a real, bullish trend).
    Informational: nothing is removed from the table.
    """
    table = table.copy()
    table["rrg_quadrant_vs_nifty500"] = [classify_quadrant(r, m) for r, m in
                                         zip(table["rs_score_vs_nifty500"], table["rs_momentum_vs_nifty500"])]
    table["rrg_quadrant_vs_sector"] = [classify_quadrant(r, m) for r, m in
                                       zip(table["rs_score_vs_sector_avg"], table["rs_momentum_vs_sector"])]
    table["di_gap"] = (pd.to_numeric(table["plus_di"]) - pd.to_numeric(table["minus_di"])).round(2)
    table["thin_trend_flag"] = table["di_gap"].abs() < DI_GAP_THIN_THRESHOLD
    table["recent_spike_flag"] = pd.to_numeric(table["recent_10day_contribution_pct"]) > RECENT_SPIKE_THRESHOLD_PCT
    table["fundamentals_clean"] = table["fundamentals_passed_count"] == len(criteria)

    fields = [field for field, *_ in criteria]
    if "opm" in fields:
        others = [f"pass_{f}" for f in fields if f != "opm"]
        only_opm_failed = (table["pass_opm"] == False) & table[others].eq(True).all(axis=1)  # noqa: E712
        roce = pd.to_numeric(table["roce"]) if "roce" in table else pd.Series(float("nan"), index=table.index)
        table["high_turnover_business_flag"] = only_opm_failed & (roce > HIGH_TURNOVER_ROCE_MIN)
    else:
        table["high_turnover_business_flag"] = pd.NA

    # HARD vs SOFT fundamental failures (config.FUNDAMENTAL_HARD_FIELDS) for a 3-month holding
    hard = [f for f in fields if f in FUNDAMENTAL_HARD_FIELDS]
    soft = [f for f in fields if f not in FUNDAMENTAL_HARD_FIELDS]
    labels = {field: label for field, _, _, label in criteria}
    table["hard_fundamentals_pass"] = table[[f"pass_{f}" for f in hard]].eq(True).all(axis=1)
    table["soft_fundamental_fails"] = table.apply(
        lambda r: "; ".join(labels[f] for f in soft if not r.get(f"pass_{f}") == True), axis=1)  # noqa: E712
    # Selection rule: hard rules pass, bullish trend with a real DI gap, a year of prices; ranked by rs_6m_skip1m
    sessions = pd.to_numeric(table.get("price_sessions", pd.Series(MIN_HISTORY_SESSIONS, index=table.index)))
    table["selection_eligible"] = (table["hard_fundamentals_pass"] & (table["di_gap"] >= DI_GAP_THIN_THRESHOLD)
                                   & (sessions >= MIN_HISTORY_SESSIONS)
                                   & (table["rs_6m_skip1m"].notna() if "rs_6m_skip1m" in table else True))

    table["full_standard_candidate"] = (
        table["fundamentals_clean"]
        & (table["rrg_quadrant_vs_nifty500"] == LEADING)
        & (table["rrg_quadrant_vs_sector"] == LEADING)
        & (table["di_gap"] >= DI_GAP_THIN_THRESHOLD)
    )
    return table


def build_selection_ranking(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Universe-wide ranking for portfolio selection: every eligible stock (hard fundamental rules
    pass, bullish trend with di_gap >= DI_GAP_THIN_THRESHOLD) ranked by rs_6m_skip1m (RS vs the
    Nifty 500 over SELECTION_RS_DAYS sessions ending SELECTION_SKIP_DAYS ago), best first, with the
    context needed for the final judgement (soft fundamental fails, 3-month RS, sector, held or not).
    """
    frames = [t.assign(sector=sector) for sector, t in tables.items()]
    table = pd.concat(frames, ignore_index=True)
    table = table[table["selection_eligible"] == True].sort_values("rs_6m_skip1m", ascending=False)  # noqa: E712
    table["selection_rank"] = range(1, len(table) + 1)
    table["conviction"] = [conviction_tier(a, b) for a, b in
                           zip(table["rrg_quadrant_vs_nifty500"], table["rrg_quadrant_vs_sector"])]
    table["locked"] = table["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)  # in the 15 tracked
    table["rule_pick"] = table["symbol"].isin(select_portfolio(table))
    cols = ["selection_rank", "symbol", "company_name", "sector", "locked", "rule_pick", "rs_6m_skip1m", "rs_score_vs_nifty500",
            "di_gap", "latest_adx", "rrg_quadrant_vs_nifty500", "rrg_quadrant_vs_sector", "conviction",
            "soft_fundamental_fails", "business_focus_note"]
    return table[cols].reset_index(drop=True)


def select_portfolio(ranking: pd.DataFrame, size: int = PORTFOLIO_SIZE,
                     sector_minimums: Optional[Dict[str, int]] = None,
                     tiers: Optional[tuple] = SELECTION_CONVICTION_TIERS) -> List[str]:
    """
    The selection rule's picks from a ranking (best first). Only names whose RRG conviction is in
    `tiers` (High / Moderate: momentum still confirmed) are picked; for each sector in
    SECTOR_MIN_HOLDINGS, its best-ranked such names up to the minimum; then the remaining slots by rank.
    """
    minimums = SECTOR_MIN_HOLDINGS if sector_minimums is None else sector_minimums
    if tiers is not None and "conviction" in ranking.columns:
        ranking = ranking[ranking["conviction"].isin(tiers)]
    picks: List[str] = []
    for sector, count in minimums.items():
        picks += ranking[ranking["sector"] == sector]["symbol"].head(count).tolist()
    for symbol in ranking["symbol"]:
        if len(picks) >= size:
            break
        if symbol not in picks:
            picks.append(symbol)
    return picks


def run_review_table(sector: str, output_csv: Path, as_of: Optional[date] = None) -> pd.DataFrame:
    """Build and save the unfiltered review table for one sector (price window ending as_of, default today)."""
    cfg = SECTOR_SCREENS[sector]
    constituents = load_sector_universe(sector)
    start, end = get_one_year_date_range(as_of)
    prices = adjust_for_corporate_actions(
        NSEBhavcopyFetcher().fetch_date_range(start, end, constituents["Symbol"].tolist(), use_cache=True))
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.isoformat())
    if benchmark.empty:
        raise RuntimeError("Nifty 500 benchmark unavailable; RS cannot be computed.")
    table = build_review_table(constituents, prices, benchmark, cfg["criteria"])
    # Label outputs with the last session actually in the data (the requested end date can be a
    # weekend, a holiday, or today before NSE publishes the day's Bhavcopy)
    end = pd.to_datetime(prices["DATE1"]).max().date() if not prices.empty else end
    table.attrs["last_session"] = end
    if len(table) != len(constituents) or set(table["symbol"]) != set(constituents["Symbol"]):
        raise RuntimeError("Review table does not contain exactly one row per constituent.")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    logger.info("Saved %s review table (%d rows) to %s", sector, len(table), output_csv.resolve())
    notes_path = output_csv.with_name(f"{sector.lower().replace(' ', '_')}_review_notes.md")
    notes_path.write_text(_review_notes(sector, table, start, end), encoding="utf-8")
    logger.info("Saved %s review notes to %s", sector, notes_path.resolve())
    return table


def _review_notes(sector: str, table: pd.DataFrame, start: date, end: date) -> str:
    avg = table.attrs.get("sector_avg_return_pct")
    avg_text = f"{avg:+.2f}%" if avg is not None and not pd.isna(avg) else "n/a"
    return f"""# Nifty {sector} review table: notes

`{sector.lower().replace(' ', '_')}_full_review_table.csv` lists every Nifty {sector} constituent ({len(table)} rows,
none excluded), with live fundamentals and technicals as of {end:%d-%b-%Y} (price window
{start:%d-%b-%Y} to {end:%d-%b-%Y}). It is sorted by `fundamentals_passed_count` (descending),
then `sector_rank` (ascending).

## Two relative strength measures

**`rs_score_vs_nifty500`** measures whether a stock beats the broad market: its {TECHNICAL_RS_LOOKBACK_DAYS}-session
cumulative return minus the Nifty 500 price index's (NSE official closes), in percentage points. It bears on whether
{sector} as a theme deserves capital at all, versus simply holding the index.
**`rs_score_vs_sector_avg`** measures which {sector} stock is best positioned relative to its
{sector} peers: the same {TECHNICAL_RS_LOOKBACK_DAYS}-session return minus the equal-weighted average return of all
{len(table)} constituents ({avg_text} over this window). It is the relevant measure once the
decision to hold {sector} exposure has been made, per the project's sector-rotation requirement.

`sector_rank` ranks `rs_score_vs_sector_avg` from 1 (best) to {len(table)} (worst); the spreads
sum to approximately zero by construction.

## Full evaluation standard (applied to every row)

- **RRG (Relative Rotation Graph).** x = RS (pp, {TECHNICAL_RS_LOOKBACK_DAYS} sessions); y = RS-Momentum = that RS today
  averaged over the last {RRG_MOMENTUM_SMOOTHING_DAYS} sessions, minus the same average {RRG_MOMENTUM_DAYS} sessions earlier (pp). Both axes are
  plain percentage-point spreads, not the proprietary JdK RS-Ratio index. Quadrants at (0, 0):
  LEADING (RS > 0, momentum > 0), WEAKENING (RS > 0, momentum <= 0), LAGGING (RS <= 0,
  momentum <= 0), IMPROVING (RS <= 0, momentum > 0). Computed against the Nifty 500
  (`rs_momentum_vs_nifty500`, `rrg_quadrant_vs_nifty500`) and against the equal-weighted
  {sector} average (`rs_momentum_vs_sector`, `rrg_quadrant_vs_sector`).
- **Trend quality.** `di_gap` = +DI - -DI; `thin_trend_flag` when |di_gap| < {DI_GAP_THIN_THRESHOLD:g},
  whichever way it points (the Bullish/Bearish label can flip on one bar).
- **Run-up.** `recent_10day_contribution_pct` = RS over the last {RUNUP_RECENT_DAYS} sessions / {TECHNICAL_RS_LOOKBACK_DAYS}-session RS x 100
  (NaN when RS <= 0); `recent_spike_flag` above {RECENT_SPIKE_THRESHOLD_PCT:g}%.
- **`high_turnover_business_flag`** (sectors with an OPM criterion): fails ONLY OPM, passes every
  other criterion, ROCE > {HIGH_TURNOVER_ROCE_MIN:g}%. A prompt for a manual business-model check,
  never an automatic pass.
- **`full_standard_candidate`** = every fundamental criterion passed AND LEADING vs both the
  Nifty 500 and the sector AND di_gap >= {DI_GAP_THIN_THRESHOLD:g} (a real, bullish trend). The
  spike flag is reported beside it, not folded in.

Plots: `rrg_{sector.lower().replace(' ', '_')}_vs_sector.png` and the combined `rrg_all_vs_nifty500.png`.
"""


# The current picks (config.LOCKED_PORTFOLIO), excluded from the 10th-candidate sweep
CURRENT_PICKS = LOCKED_PORTFOLIO_SYMBOLS

# recent_10day_contribution_pct above this counts as a recent spike rather than sustained
# strength: 10 of 63 sessions is ~16% of the window, so > 50% is over 3x the steady pace
RECENT_SPIKE_THRESHOLD_PCT = 50.0
CATALYST_WINDOW_DAYS = 30


def compute_runup_and_catalyst_info(
    symbols: List[str],
    as_of: date,
    today: Optional[date] = None,
) -> pd.DataFrame:
    """
    Run-up and results-catalyst information for any list of symbols (one row each, same order).
    The single implementation behind the 10th-candidate sweep and the locked-portfolio check.

    Columns:
      rs_score_vs_nifty500 (63 sessions) and rs_last_10d: RS vs the Nifty 500, window ending as_of;
      recent_10day_contribution_pct: rs_last_10d / 63-session RS x 100, a simple run-up heuristic
        (indicators.compute_recent_rs_contribution; NaN when the 63-session RS <= 0);
      recent_spike_flag: contribution above RECENT_SPIKE_THRESHOLD_PCT;
      next_results_date / results_date_status: the next results board meeting announced on NSE
        (announced / not announced / unavailable; never estimated);
      results_within_30_days: announced date within CATALYST_WINDOW_DAYS of `today` (None if none);
      prior_year_sep_qtr_results_date: last year's actual September-quarter results date.

    Informational only: nothing here is a pass/fail gate.
    """
    today = today or date.today()
    start, end = get_one_year_date_range(as_of)
    prices = adjust_for_corporate_actions(
        NSEBhavcopyFetcher().fetch_date_range(start, end, list(symbols), use_cache=True))
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.isoformat())
    rows = []
    for symbol in symbols:
        sym_prices = prices[prices["SYMBOL"] == symbol] if not prices.empty else pd.DataFrame()
        rs10, rs63, pct = compute_recent_rs_contribution(
            sym_prices, benchmark, RUNUP_RECENT_DAYS, TECHNICAL_RS_LOOKBACK_DAYS)
        cal = fetch_results_calendar(symbol, as_of=today)
        next_date = pd.to_datetime(cal["next_results_date"]).date() if cal["next_results_date"] else None
        rows.append({
            "symbol": symbol,
            "rs_score_vs_nifty500": rs63,
            "rs_last_10d": rs10,
            "recent_10day_contribution_pct": pct,
            "recent_spike_flag": bool(pd.notna(pct) and pct > RECENT_SPIKE_THRESHOLD_PCT),
            "next_results_date": cal["next_results_date"],
            "results_date_status": cal["results_date_status"],
            "results_within_30_days": (bool(next_date <= today + timedelta(days=CATALYST_WINDOW_DAYS))
                                       if next_date else None),
            "prior_year_sep_qtr_results_date": cal["prior_year_sep_qtr_results_date"],
            **results_calendar_fields(cal, today),
        })
    return pd.DataFrame(rows)


def results_calendar_fields(cal: dict, today: date) -> dict:
    """
    The Q2 results date for the calendar: 'reported' once a results meeting on/after EVALUATION_START_DATE
    has passed, else the NSE-announced date, else an ESTIMATE = last year's September-quarter date plus
    364 days (the same weekday), labelled as such. days_to_results counts from `today`.
    """
    last = cal.get("last_results_date")
    if last and date.fromisoformat(last) >= date.fromisoformat(EVALUATION_START_DATE):
        when, status = last, "reported"
    elif cal.get("next_results_date"):
        when, status = cal["next_results_date"], "announced"
    elif cal.get("prior_year_sep_qtr_results_date"):
        when = (date.fromisoformat(cal["prior_year_sep_qtr_results_date"]) + timedelta(days=364)).isoformat()
        status = "estimate"
    else:
        when, status = None, "unknown"
    return {
        "q2_results_date": when,
        "q2_results_basis": status,
        "days_to_results": (date.fromisoformat(when) - today).days if when else None,
        "last_results_date": last,
    }


def build_tenth_candidate_sweep(
    exclude: List[str],
    as_of: date,
    today: Optional[date] = None,
) -> pd.DataFrame:
    """
    One row per official-index constituent across all three sectors, excluding `exclude`.

    Reuses each sector's generated review table (lightened fundamentals with per-criterion
    flags, technicals, both RS measures, sector_rank) and adds:
      - the run-up heuristic and NSE results-date fields from compute_runup_and_catalyst_info()
        (same price and benchmark history, window ending as_of), checked against the review
        tables' 63-day RS.
    Sorted by fundamentals_passed_count, then rs_score_vs_nifty500, descending.
    """
    today = today or date.today()
    tables = []
    for sector, cfg in SECTOR_SCREENS.items():
        t = pd.read_csv(review_table_path(sector))
        t.insert(0, "sector", sector)
        t["criteria_total"] = len(cfg["criteria"])
        t["sector_size"] = len(t)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table = table[~table["symbol"].isin(exclude)].reset_index(drop=True)

    info = compute_runup_and_catalyst_info(table["symbol"].tolist(), as_of=as_of, today=today)
    mismatch = (info["rs_score_vs_nifty500"] - table["rs_score_vs_nifty500"]).abs() > 0.011
    if mismatch.any():
        raise RuntimeError(f"Recomputed {TECHNICAL_RS_LOOKBACK_DAYS}-day RS differs from the review tables for {table['symbol'][mismatch].tolist()}")
    info = info.drop(columns=["symbol", "rs_score_vs_nifty500"])
    # The review tables already carry the run-up columns; the freshly computed ones replace them
    table = pd.concat([table.drop(columns=[c for c in info.columns if c in table.columns]), info], axis=1)
    table["fundamentals_clean"] = table["fundamentals_passed_count"] == table["criteria_total"]
    table["clean_candidate"] = (table["fundamentals_clean"] & table["technically_attractive"]
                                & ~table["recent_spike_flag"])
    front = ["sector", "symbol", "company_name", "fundamentals_passed_count", "criteria_total", "fundamentals_clean",
             "fundamentals_failed", "technically_attractive", "rs_score_vs_nifty500", "rs_last_10d",
             "recent_10day_contribution_pct", "recent_spike_flag", "trend_direction", "latest_adx", "latest_rsi",
             "rs_score_vs_sector_avg", "sector_rank", "sector_size", "next_results_date", "results_date_status",
             "results_within_30_days", "prior_year_sep_qtr_results_date", "clean_candidate"]
    table = table[front + [c for c in table.columns if c not in front]]
    return table.sort_values(["fundamentals_passed_count", "rs_score_vs_nifty500"],
                             ascending=[False, False], na_position="last").reset_index(drop=True)


def print_screen_summary(results: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 115)
    print(f" SECTOR SCREEN: technical screen first (RS > {TECHNICAL_RS_MARGIN_PP:+g} pp AND Bullish), then live fundamental safety screen")
    print("=" * 115)
    for sector, df in results.items():
        tech = df[df["passed_technical_screen"]]
        both = df[df["passes_both_screens"]]
        print(f" {sector:14s} constituents {len(df):3d} | passed technical {len(tech):3d} | passed both {len(both):3d}"
              f" -> {', '.join(both['symbol']) or '-'}")
    print("=" * 115 + "\n")


def review_table_path(sector: str) -> Path:
    """output/<sector>_full_review_table.csv, e.g. output/capital_goods_full_review_table.csv."""
    return OUTPUT_DIR / f"{sector.lower().replace(' ', '_')}_full_review_table.csv"


if __name__ == "__main__":
    args = sys.argv[1:]
    as_of = None
    if "--as-of" in args:
        # Pin the price window's end date (YYYY-MM-DD), e.g. to re-run on the same sessions
        i = args.index("--as-of")
        as_of = date.fromisoformat(args[i + 1])
        del args[i:i + 2]
    if args[:1] == ["--locked-check"]:
        # Informational run-up / results-date check of the current picks (no pass/fail, no changes)
        check = compute_runup_and_catalyst_info(CURRENT_PICKS, as_of=as_of or date.today())
        out = OUTPUT_DIR / "locked_portfolio_runup_catalyst_check.csv"
        check.to_csv(out, index=False)
        logger.info("Saved run-up/catalyst check for %d picks to %s", len(check), out.resolve())
        sys.exit(0)
    if args[:1] == ["--tenth-sweep"]:
        # Candidate sweep over all three universes excluding the current picks (no selection)
        sweep = build_tenth_candidate_sweep(CURRENT_PICKS, as_of=as_of or date.today())
        sweep.to_csv(OUTPUT_DIR / "tenth_candidate_sweep.csv", index=False)
        logger.info("Saved 10th-candidate sweep (%d rows) to %s", len(sweep), (OUTPUT_DIR / "tenth_candidate_sweep.csv").resolve())
        sys.exit(0)
    if args[:1] == ["--review-cement"]:
        args = ["--review", "Cement"]
    if args[:1] == ["--review"]:
        # Unfiltered review tables (no screening), one per named sector (default: all three)
        last_sessions = []
        for sector in args[1:] or list(SECTOR_SCREENS):
            if sector not in SECTOR_SCREENS:
                sys.exit(f"Unknown sector {sector!r}; choose from {list(SECTOR_SCREENS)}")
            last_sessions.append(run_review_table(sector, review_table_path(sector), as_of=as_of).attrs["last_session"])
        tables = {sec: pd.read_csv(review_table_path(sec)) for sec in SECTOR_SCREENS}
        build_selection_ranking(tables).to_csv(OUTPUT_DIR / "selection_ranking.csv", index=False)
        # Redraw the RRG plots from all three review tables (whichever were just refreshed)
        from rrg import plot_all_rrgs
        plot_all_rrgs({sec: pd.read_csv(review_table_path(sec)) for sec in SECTOR_SCREENS},
                      as_of_label=f"prices to {max(last_sessions):%d-%b-%Y}")
    else:
        print_screen_summary(run_sector_screens(as_of=as_of))
    sys.exit(0)
