"""
Relative Rotation Graph (RRG) classification: the quadrant columns of the review tables and the conviction
tier that research/conviction_study.py tested (and rejected as a selection filter). Not shown on the dashboard.

Two axes per stock, both in percentage points (not the proprietary JdK RS-Ratio index):
  x  RS:          63-session return minus the benchmark's (rs_score_vs_nifty500, or
                  rs_score_vs_sector_avg against the equal-weighted sector average)
  y  RS-Momentum: that RS averaged over the last RRG_MOMENTUM_SMOOTHING_DAYS (5) sessions minus
                  the same average RRG_MOMENTUM_DAYS (10) sessions earlier
                  (indicators.compute_rs_momentum); positive = relative strength improving

Quadrants (centred at 0, 0):
  LEADING    RS > 0,  momentum > 0   strong and getting stronger
  WEAKENING  RS > 0,  momentum <= 0  strong but losing steam
  LAGGING    RS <= 0, momentum <= 0  weak and getting weaker
  IMPROVING  RS <= 0, momentum > 0   weak but turning up
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("rrg")

LEADING, WEAKENING, LAGGING, IMPROVING = "LEADING", "WEAKENING", "LAGGING", "IMPROVING"



def classify_quadrant(rs: Optional[float], momentum: Optional[float]) -> Optional[str]:
    """RRG quadrant for an RS spread and its momentum (both pp); None if either is missing."""
    if rs is None or momentum is None or pd.isna(rs) or pd.isna(momentum):
        return None
    if rs > 0:
        return LEADING if momentum > 0 else WEAKENING
    return IMPROVING if momentum > 0 else LAGGING


CONVICTION_HIGH = "High"
CONVICTION_MODERATE = "Moderate"
CONVICTION_LOW = "Low"


def conviction_tier(quadrant_vs_nifty500: Optional[str], quadrant_vs_sector: Optional[str]) -> Optional[str]:
    """
    Conviction tier from the two RRG views:
      High      LEADING vs the Nifty 500 AND LEADING vs the sector average
      Moderate  LEADING in exactly one view, or IMPROVING in either view
      Low       WEAKENING or LAGGING in both views
    None when either quadrant is missing. The three rules cover every quadrant pair.
    """
    views = (quadrant_vs_nifty500, quadrant_vs_sector)
    if any(v is None or pd.isna(v) for v in views):
        return None
    if views == (LEADING, LEADING):
        return CONVICTION_HIGH
    if LEADING in views or IMPROVING in views:
        return CONVICTION_MODERATE
    return CONVICTION_LOW
