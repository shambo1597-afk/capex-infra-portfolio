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
    assert (risk["weight_pct"] == round(100 / len(LOCKED_PORTFOLIO_SYMBOLS), 2)).all()

    check = pd.read_csv(OUTPUT_DIR / "locked_portfolio_runup_catalyst_check.csv")
    assert check["symbol"].tolist() == LOCKED_PORTFOLIO_SYMBOLS
    merged = check.merge(review, on="symbol", suffixes=("_c", "_r"))
    assert ((merged["rs_score_vs_nifty500_c"] - merged["rs_score_vs_nifty500_r"]).abs() <= 0.011).all()


def test_candidate_sweep_matches_review_tables(review):
    sweep = pd.read_csv(OUTPUT_DIR / "tenth_candidate_sweep.csv")
    assert set(sweep["symbol"]) == set(PORTFOLIO_SYMBOLS) - set(LOCKED_PORTFOLIO_SYMBOLS)
    merged = sweep.merge(review, on="symbol", suffixes=("_s", "_r"))
    for col in ["rs_score_vs_sector_avg", "sector_rank", "fundamentals_passed_count"]:
        assert ((merged[f"{col}_s"] - merged[f"{col}_r"]).abs() <= 0.011).all(), col


def test_price_history_is_one_window_without_duplicates():
    ohlcv = pd.read_csv(HISTORICAL_OHLCV_CSV, usecols=["SYMBOL", "DATE1"])
    assert not ohlcv.duplicated().any()
    assert ohlcv.groupby("SYMBOL")["DATE1"].max().nunique() == 1  # every stock ends on the same session
