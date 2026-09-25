"""
Sector Screen: technical screen first, then fundamental safety screen.

Implements the project brief's order ("Technical analysis, then financial analysis") identically
for each sector:

1. Universe: the official Nifty sector index constituents only (config.SECTOR_SCREENS), with no
   manual additions.
2. Technical screen on every constituent, using the existing indicator pipeline
   (analysis.evaluate_stock_technicals) on the complete NSE Bhavcopy history:
   passes when RS vs Nifty 500 (63 sessions) > +2 pp AND trend direction (+DI vs -DI) is Bullish.
   ADX is reported for tie-breaking but is not a cutoff.
3. Fundamental safety screen, run live (fundamentals.get_fundamentals_summary, no cache) ONLY for
   stocks that passed step 2, against that sector's criteria in config.py.

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
from config import OUTPUT_DIR, SECTOR_SCREENS, TECHNICAL_RS_LOOKBACK_DAYS, TECHNICAL_RS_MARGIN_PP
from fetch_data import NSEBhavcopyFetcher, fetch_benchmark_nifty500, get_one_year_date_range
from fundamentals import evaluate_fundamental_screen, fetch_results_calendar, get_fundamentals_summary
from indicators import compute_recent_rs_contribution, compute_relative_strength, compute_sector_relative_strength

logger = logging.getLogger("sector_screen")


def load_constituents(csv_path: Path) -> pd.DataFrame:
    """Load an official niftyindices.com constituent file (Company Name, Industry, Symbol, ...)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df


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
            "index": f"Nifty {sector}",
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
        screen.at[i, "fundamentals_as_of"] = date.today().isoformat()
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
    constituents = {s: load_constituents(SECTOR_SCREENS[s]["constituents_csv"]) for s in sectors}
    all_symbols = sorted({sym for df in constituents.values() for sym in df["Symbol"]})

    start, end = get_one_year_date_range(as_of)
    prices = NSEBhavcopyFetcher().fetch_date_range(start, end, all_symbols, use_cache=use_price_cache)
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
    Sorted by fundamentals_passed_count descending, then sector_rank ascending.
    """
    symbols = constituents["Symbol"].tolist()
    fundamentals = get_fundamentals_summary(symbols, use_cache=False)
    records = {r["symbol"]: r for r in fundamentals.to_dict("records")} if not fundamentals.empty else {}

    rows = []
    stock_returns: Dict[str, float] = {}
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
        rows.append({
            "symbol": symbol,
            "company_name": member.get("Company Name"),
            **result,
            "fundamentals_passed_count": len(criteria) - len(failed),
            "fundamentals_failed": "; ".join(failed),
            "fundamentals_status": record.get("status"),
            "fundamentals_as_of": date.today().isoformat(),
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
        })
    table = pd.DataFrame(rows)

    # Within-sector relative strength vs the equal-weighted average of all constituents
    sector_rs, sector_avg = compute_sector_relative_strength(stock_returns)
    table["rs_score_vs_sector_avg"] = table["symbol"].map(lambda sym: round(sector_rs[sym], 2)
                                                          if not pd.isna(sector_rs[sym]) else float("nan"))
    table["sector_rank"] = table["rs_score_vs_sector_avg"].rank(ascending=False, method="min").astype("Int64")
    table.attrs["sector_avg_return_pct"] = sector_avg
    return table.sort_values(["fundamentals_passed_count", "sector_rank"],
                             ascending=[False, True], na_position="last").reset_index(drop=True)


def run_review_table(sector: str, output_csv: Path, as_of: Optional[date] = None) -> pd.DataFrame:
    """Build and save the unfiltered review table for one sector (price window ending as_of, default today)."""
    cfg = SECTOR_SCREENS[sector]
    constituents = load_constituents(cfg["constituents_csv"])
    start, end = get_one_year_date_range(as_of)
    prices = NSEBhavcopyFetcher().fetch_date_range(start, end, constituents["Symbol"].tolist(), use_cache=True)
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.isoformat())
    if benchmark.empty:
        raise RuntimeError("Nifty 500 benchmark unavailable; RS cannot be computed.")
    table = build_review_table(constituents, prices, benchmark, cfg["criteria"])
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

**`rs_score_vs_nifty500`** measures whether a stock beats the broad market: its 63-session
cumulative return minus the Nifty 500 price index's (NSE official closes), in percentage points. It bears on whether
{sector} as a theme deserves capital at all, versus simply holding the index.
**`rs_score_vs_sector_avg`** measures which {sector} stock is best positioned relative to its
{sector} peers: the same 63-session return minus the equal-weighted average return of all
{len(table)} constituents ({avg_text} over this window). It is the relevant measure once the
decision to hold {sector} exposure has been made, per the project's sector-rotation requirement.

`sector_rank` ranks `rs_score_vs_sector_avg` from 1 (best) to {len(table)} (worst); the spreads
sum to approximately zero by construction.
"""


# The 9 current picks (as given on 25-Sep-2026) excluded from the 10th-candidate sweep. Note that
# config.LOCKED_PORTFOLIO has not been updated to this list; it is maintained by the user.
CURRENT_PICKS = ["JKCEMENT", "JKLAKSHMI", "VOLTAMP", "FINCABLES", "ELGIEQUIP", "APARINDS", "WELCORP", "ACMESOLAR", "TATAPOWER"]

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
    prices = NSEBhavcopyFetcher().fetch_date_range(start, end, list(symbols), use_cache=True)
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.isoformat())
    rows = []
    for symbol in symbols:
        sym_prices = prices[prices["SYMBOL"] == symbol] if not prices.empty else pd.DataFrame()
        rs10, rs63, pct = compute_recent_rs_contribution(sym_prices, benchmark, 10, TECHNICAL_RS_LOOKBACK_DAYS)
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
        })
    return pd.DataFrame(rows)


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
        raise RuntimeError(f"Recomputed 63-day RS differs from the review tables for {table['symbol'][mismatch].tolist()}")
    table = pd.concat([table, info.drop(columns=["symbol", "rs_score_vs_nifty500"])], axis=1)
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
        for sector in args[1:] or list(SECTOR_SCREENS):
            if sector not in SECTOR_SCREENS:
                sys.exit(f"Unknown sector {sector!r}; choose from {list(SECTOR_SCREENS)}")
            run_review_table(sector, review_table_path(sector), as_of=as_of)
    else:
        print_screen_summary(run_sector_screens(as_of=as_of))
    sys.exit(0)
