"""
Cross-file integrity of the committed outputs: every file covers the current universe (or the
locked portfolio) and agrees with the others on the shared numbers. These catch an output that
was not regenerated after the universe, the locked portfolio or a methodology parameter changed.
"""

import pandas as pd
import pytest

from config import (
    HISTORICAL_OHLCV_CSV,
    LOCKED_PORTFOLIO_SYMBOLS,
    OUTPUT_DIR,
    PORTFOLIO_SYMBOLS,
    RISK_SUMMARY_OUTPUT_CSV,
    SECTOR_SCREENS,
    SUMMARY_OUTPUT_CSV,
    sector_of,
)
from sector_screen import load_sector_universe, review_table_path


@pytest.fixture(scope="module")
def review():
    return pd.concat([pd.read_csv(review_table_path(s)).assign(_sector=s) for s in SECTOR_SCREENS],
                     ignore_index=True)


@pytest.fixture(scope="module")
def technical_summary():
    return pd.read_csv(SUMMARY_OUTPUT_CSV)


def test_every_output_covers_the_current_universe(review, technical_summary):
    universe = set(PORTFOLIO_SYMBOLS)
    assert set(review["symbol"]) == universe and not review["symbol"].duplicated().any()
    assert set(technical_summary["symbol"]) == universe
    assert set(pd.read_csv(HISTORICAL_OHLCV_CSV, usecols=["SYMBOL"])["SYMBOL"]) == universe
    for sector, cfg in SECTOR_SCREENS.items():
        screen = pd.read_csv(cfg["output_csv"])
        assert sorted(screen["symbol"]) == sorted(load_sector_universe(sector)["Symbol"]), sector


def test_technical_summary_matches_review_tables(review, technical_summary):
    merged = technical_summary.merge(review, on="symbol", suffixes=("_t", "_r"))
    for col in ["current_price", "latest_rsi", "latest_adx", "rs_score_vs_nifty500",
                "nearest_support", "nearest_resistance"]:
        a, b = merged[f"{col}_t"], merged[f"{col}_r"]
        assert (((a - b).abs() < 1e-9) | (a.isna() & b.isna())).all(), col
    assert (merged["trend_direction_t"] == merged["trend_direction_r"]).all()
    assert (merged["sector"] == merged["_sector"]).all()
    assert (merged["sector"] == merged["symbol"].map(sector_of)).all()


def test_sector_screens_match_review_tables(review):
    for sector, cfg in SECTOR_SCREENS.items():
        merged = pd.read_csv(cfg["output_csv"]).merge(review[review["_sector"] == sector], on="symbol")
        assert ((merged["rs_score"] - merged["rs_score_vs_nifty500"]).abs() <= 0.011).all(), sector
        assert (merged["passed_technical_screen"] == merged["technically_attractive"]).all(), sector


def test_locked_portfolio_outputs(review, technical_summary):
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV)
    assert risk["symbol"].tolist() == LOCKED_PORTFOLIO_SYMBOLS
    prices = technical_summary.set_index("symbol")["current_price"]
    assert (risk.set_index("symbol")["current_price"] == prices.loc[LOCKED_PORTFOLIO_SYMBOLS]).all()
    from config import WEIGHT_MAX_PCT, WEIGHT_MIN_PCT
    assert risk["weight_pct"].between(WEIGHT_MIN_PCT, WEIGHT_MAX_PCT).all()
    assert abs(risk["weight_pct"].sum() - 100) < 0.05
    # Equal risk contribution: every stock carries ~1/N of portfolio variance (no bound binds here)
    assert (risk["risk_contribution_pct"] - 100 / len(LOCKED_PORTFOLIO_SYMBOLS)).abs().max() < 0.1

    check = pd.read_csv(OUTPUT_DIR / "locked_portfolio_runup_catalyst_check.csv")
    assert check["symbol"].tolist() == LOCKED_PORTFOLIO_SYMBOLS
    merged = check.merge(review, on="symbol", suffixes=("_c", "_r"))
    assert ((merged["rs_score_vs_nifty500_c"] - merged["rs_score_vs_nifty500_r"]).abs() <= 0.011).all()


def test_candidate_sweep_matches_review_tables(review):
    sweep = pd.read_csv(OUTPUT_DIR / "tenth_candidate_sweep.csv")
    assert set(sweep["symbol"]) == set(PORTFOLIO_SYMBOLS) - set(LOCKED_PORTFOLIO_SYMBOLS)
    merged = sweep.merge(review, on="symbol", suffixes=("_s", "_r"))
    for col in ["rs_score_vs_sector_avg", "sector_rank", "fundamentals_passed_count"]:
        diff = (merged[f"{col}_s"] - merged[f"{col}_r"]).abs()
        both_nan = merged[f"{col}_s"].isna() & merged[f"{col}_r"].isna()
        assert ((diff <= 0.011) | both_nan).all(), col


def test_price_history_is_one_window_without_duplicates():
    ohlcv = pd.read_csv(HISTORICAL_OHLCV_CSV, usecols=["SYMBOL", "DATE1"])
    assert not ohlcv.duplicated().any()
    assert ohlcv.groupby("SYMBOL")["DATE1"].max().nunique() == 1  # every stock ends on the same session


def test_committed_price_history_has_no_unexplained_jumps():
    """Splits / bonuses / demergers are adjusted: no one-day move above 30% remains unexplained."""
    from corporate_actions import unexplained_jumps
    ohlcv = pd.read_csv(HISTORICAL_OHLCV_CSV, usecols=["SYMBOL", "DATE1", "PREV_CLOSE", "CLOSE_PRICE"])
    assert unexplained_jumps(ohlcv).empty


def test_every_holding_passes_the_hard_fundamental_rules(review):
    """HARD rules (pledge, debt, interest cover, size) are never waived for a holding."""
    held = review[review["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)]
    assert held["hard_fundamentals_pass"].all(), held.loc[~held["hard_fundamentals_pass"], "symbol"].tolist()


def test_allocation_of_the_principal():
    from config import EQUITY_ALLOCATION_PCT, PRINCIPAL_INR, TRADES_CSV
    equity_inr = PRINCIPAL_INR * EQUITY_ALLOCATION_PCT / 100
    risk = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV)
    if TRADES_CSV.exists():  # invested: the risk summary shows the shares actually held
        from tracker import ledger_positions
        held = ledger_positions(pd.read_csv(TRADES_CSV))
        assert (risk["shares"] == risk["symbol"].map(lambda s: held.get(s, 0))).all()
        return
    assert (risk["shares"] == (equity_inr * risk["weight_pct"] / 100 // risk["current_price"])).all()
    assert (risk["invested_inr"] - risk["shares"] * risk["current_price"]).abs().max() < 0.01
    invested = risk["invested_inr"].sum()
    assert 0.98 * equity_inr <= invested <= equity_inr  # whole shares leave only a little rounding cash
    assert invested >= 0.90 * PRINCIPAL_INR  # the brief's minimum market exposure


def test_selection_ranking_matches_review_tables(review):
    ranking = pd.read_csv(OUTPUT_DIR / "selection_ranking.csv")
    eligible = review[review["selection_eligible"] == True]  # noqa: E712
    assert sorted(ranking["symbol"]) == sorted(eligible["symbol"])
    assert ranking["rs_6m_skip1m"].is_monotonic_decreasing
    assert (ranking["held"] == ranking["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)).all()


def test_holdings_are_exactly_the_selection_rule_picks():
    """The final portfolio is what the selection rule picks from the ranking: no judgement-call exceptions."""
    from config import PORTFOLIO_SIZE
    ranking = pd.read_csv(OUTPUT_DIR / "selection_ranking.csv")
    assert len(LOCKED_PORTFOLIO_SYMBOLS) == PORTFOLIO_SIZE
    assert set(ranking[ranking["rule_pick"]]["symbol"]) == set(LOCKED_PORTFOLIO_SYMBOLS)
