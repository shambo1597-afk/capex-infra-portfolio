"""
Relative Rotation Graph (RRG) classification and plots.

Two axes per stock, both in percentage points (not the proprietary JdK RS-Ratio index):
  x  RS:          63-session return minus the benchmark's (rs_score_vs_nifty500, or
                  rs_score_vs_sector_avg against the equal-weighted sector average)
  y  RS-Momentum: that RS today minus the same 63-session RS ending RRG_MOMENTUM_DAYS (10)
                  sessions earlier (indicators.compute_rs_momentum); positive = the spread widened

Quadrants (centred at 0, 0):
  LEADING    RS > 0,  momentum > 0   strong and getting stronger
  WEAKENING  RS > 0,  momentum <= 0  strong but losing steam
  LAGGING    RS <= 0, momentum <= 0  weak and getting weaker
  IMPROVING  RS <= 0, momentum > 0   weak but turning up

Run `python rrg.py --as-of 2026-09-24` to redraw the PNGs from the committed review tables
(`python sector_screen.py --review --as-of ...` refreshes the tables and redraws them).
"""

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    LOCKED_PORTFOLIO_SYMBOLS,
    OUTPUT_DIR,
    RRG_MOMENTUM_DAYS,
    SECTOR_SCREENS,
    TECHNICAL_RS_LOOKBACK_DAYS,
)

logger = logging.getLogger("rrg")

LEADING, WEAKENING, LAGGING, IMPROVING = "LEADING", "WEAKENING", "LAGGING", "IMPROVING"

# Validated categorical steps (all-pairs, light surface) in RRG's conventional colours; each
# quadrant also has its own marker shape and a corner label, so identity is never colour alone.
QUADRANT_STYLE = {
    LEADING: {"color": "#008300", "marker": "o"},
    WEAKENING: {"color": "#eda100", "marker": "s"},
    LAGGING: {"color": "#e34948", "marker": "v"},
    IMPROVING: {"color": "#2a78d6", "marker": "^"},
}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#1f1f1e"
TEXT_MUTED = "#6b6a63"
GRID = "#e4e3dc"


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
      Low       WEAKENING or LAGGING in both views (a sector-coverage hold)
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


def plot_rrg(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    quadrant_col: str,
    title: str,
    out_path: Path,
    label_symbols: Optional[Iterable[str]] = None,
    highlight_symbols: Iterable[str] = (),
    x_label: str = "RS (pp)",
    y_label: str = f"RS-Momentum (pp change over {RRG_MOMENTUM_DAYS} sessions)",
) -> Path:
    """
    Scatter one point per stock, coloured and shaped by quadrant. `label_symbols` (default: all)
    get a text label; `highlight_symbols` (e.g. the locked picks) get a dark ring and bold label.
    """
    data = df.dropna(subset=[x_col, y_col]).copy()
    labels = set(data["symbol"]) if label_symbols is None else set(label_symbols)
    highlight = set(highlight_symbols)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Fit the data (always keeping the 0/0 cross in view) with room for labels and corner titles
    x_lo, x_hi = min(data[x_col].min(), 0), max(data[x_col].max(), 0)
    y_lo, y_hi = min(data[y_col].min(), 0), max(data[y_col].max(), 0)
    x_span, y_span = max(x_hi - x_lo, 10), max(y_hi - y_lo, 6)
    ax.set_xlim(x_lo - 0.08 * x_span, x_hi + 0.12 * x_span)
    ax.set_ylim(y_lo - 0.12 * y_span, y_hi + 0.12 * y_span)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1)
    ax.axvline(0, color=TEXT_MUTED, linewidth=1)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    corner = {LEADING: (0.98, 0.97, "right", "top"), WEAKENING: (0.98, 0.03, "right", "bottom"),
              LAGGING: (0.02, 0.03, "left", "bottom"), IMPROVING: (0.02, 0.97, "left", "top")}
    for quadrant, (cx, cy, ha, va) in corner.items():
        ax.text(cx, cy, quadrant, transform=ax.transAxes, ha=ha, va=va, fontsize=13, fontweight="bold",
                color=QUADRANT_STYLE[quadrant]["color"], alpha=0.9)

    for quadrant, style in QUADRANT_STYLE.items():
        sub = data[data[quadrant_col] == quadrant]
        if sub.empty:
            continue
        ringed = sub["symbol"].isin(highlight)
        ax.scatter(sub.loc[~ringed, x_col], sub.loc[~ringed, y_col], s=60, marker=style["marker"],
                   color=style["color"], edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=f"{quadrant.title()} ({len(sub)})")
        ax.scatter(sub.loc[ringed, x_col], sub.loc[ringed, y_col], s=110, marker=style["marker"],
                   color=style["color"], edgecolors=TEXT_PRIMARY, linewidths=1.8, zorder=4)

    _place_labels(fig, ax, data, x_col, y_col, labels, highlight)

    ax.set_xlabel(x_label, color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel(y_label, color=TEXT_PRIMARY, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, fontsize=9)
    if highlight:
        legend.set_title("Ringed + bold: locked portfolio", prop={"size": 8})
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved RRG plot to %s", out_path.resolve())
    return out_path


# Candidate label offsets (points) around a marker, tried in order until one does not collide
_LABEL_OFFSETS = [(6, 4, "left"), (6, -11, "left"), (-6, 4, "right"), (-6, -11, "right"),
                  (6, 13, "left"), (-6, 13, "right"), (6, -20, "left"), (-6, -20, "right")]


def _place_labels(fig, ax, data, x_col, y_col, labels, highlight) -> None:
    """Greedy label placement: highlighted labels first, each at the first offset that overlaps no
    earlier label or marker; falls back to the first offset when every candidate collides."""
    renderer = fig.canvas.get_renderer()
    marker_boxes = []
    for x, y in zip(data[x_col], data[y_col]):
        px, py = ax.transData.transform((x, y))
        marker_boxes.append((px - 5, py - 5, px + 5, py + 5))
    placed = []
    rows = data[data["symbol"].isin(labels | highlight)].copy()
    rows["_first"] = ~rows["symbol"].isin(highlight)
    for _, row in rows.sort_values("_first").iterrows():
        bold = row["symbol"] in highlight
        style = dict(fontsize=8.5 if bold else 7.5, fontweight="bold" if bold else "normal",
                     color=TEXT_PRIMARY if bold else TEXT_MUTED, zorder=5)
        chosen = None
        for dx, dy, ha in _LABEL_OFFSETS:
            text = ax.annotate(row["symbol"], (row[x_col], row[y_col]), xytext=(dx, dy),
                               textcoords="offset points", ha=ha, **style)
            bb = text.get_window_extent(renderer).padded(1)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            others = placed + [m for m in marker_boxes if not _contains_point(m, ax, row, x_col, y_col)]
            if not any(_overlaps(box, o) for o in others):
                chosen = box
                break
            text.remove()
        if chosen is None:
            dx, dy, ha = _LABEL_OFFSETS[0]
            text = ax.annotate(row["symbol"], (row[x_col], row[y_col]), xytext=(dx, dy),
                               textcoords="offset points", ha=ha, **style)
            bb = text.get_window_extent(renderer)
            chosen = (bb.x0, bb.y0, bb.x1, bb.y1)
        placed.append(chosen)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _contains_point(box, ax, row, x_col, y_col) -> bool:
    """True for the marker box of the point being labelled (its own marker is not an obstacle)."""
    px, py = ax.transData.transform((row[x_col], row[y_col]))
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def _slug(sector: str) -> str:
    return sector.lower().replace(" ", "_")


def plot_all_rrgs(tables: dict, as_of_label: str = "") -> list:
    """
    One vs-sector-average RRG per sector (output/rrg_<sector>_vs_sector.png) and one combined
    vs-Nifty-500 RRG across all sectors (output/rrg_all_vs_nifty500.png).
    """
    suffix = f" ({as_of_label})" if as_of_label else ""
    paths = []
    for sector, table in tables.items():
        # Label every stock except in large universes, where only notable ones are labelled
        label = None
        if len(table) > 25:
            notable = (table["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)
                       | (table["high_turnover_business_flag"] == True)  # noqa: E712
                       | (table["rrg_quadrant_vs_sector"] == LEADING)
                       | (table["rs_score_vs_sector_avg"].abs() > 20)
                       | (table["rs_momentum_vs_sector"].abs() > 8))
            label = table.loc[notable, "symbol"]
        paths.append(plot_rrg(
            table, "rs_score_vs_sector_avg", "rs_momentum_vs_sector", "rrg_quadrant_vs_sector",
            f"Nifty {sector}: RRG vs equal-weighted sector average{suffix}",
            OUTPUT_DIR / f"rrg_{_slug(sector)}_vs_sector.png",
            label_symbols=label, highlight_symbols=LOCKED_PORTFOLIO_SYMBOLS,
            x_label=f"RS vs sector average (pp, {TECHNICAL_RS_LOOKBACK_DAYS} sessions)",
        ))

    combined = pd.concat([t.assign(sector=s) for s, t in tables.items()], ignore_index=True)
    notable = (combined["symbol"].isin(LOCKED_PORTFOLIO_SYMBOLS)
               | combined["full_standard_candidate"].fillna(False).astype(bool)
               | (combined["high_turnover_business_flag"] == True)  # noqa: E712
               | (combined["rs_score_vs_nifty500"].abs() > 30)
               | (combined["rs_momentum_vs_nifty500"].abs() > 10))
    paths.append(plot_rrg(
        combined, "rs_score_vs_nifty500", "rs_momentum_vs_nifty500", "rrg_quadrant_vs_nifty500",
        f"Cement, Capital Goods & Power ({len(combined)} stocks): RRG vs Nifty 500{suffix}",
        OUTPUT_DIR / "rrg_all_vs_nifty500.png",
        label_symbols=combined.loc[notable, "symbol"], highlight_symbols=LOCKED_PORTFOLIO_SYMBOLS,
        x_label=f"RS vs Nifty 500 (pp, {TECHNICAL_RS_LOOKBACK_DAYS} sessions)",
    ))
    return paths


if __name__ == "__main__":
    from sector_screen import review_table_path

    import sys
    from datetime import date

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Optional --as-of YYYY-MM-DD: the review tables' price-window end date, shown in the titles
    label = ""
    if "--as-of" in sys.argv:
        label = f"as of {date.fromisoformat(sys.argv[sys.argv.index('--as-of') + 1]):%d-%b-%Y}"
    plot_all_rrgs({sector: pd.read_csv(review_table_path(sector)) for sector in SECTOR_SCREENS}, as_of_label=label)
