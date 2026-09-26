"""
Relative Rotation Graphs with tails: where each stock or sector has been over the last weeks.

rrg.py plots today's point only. Here the same point (RS vs the Nifty 500 and its smoothed
RS-Momentum, from indicators.compute_rs_momentum, so the last point equals the review tables and the
dashboard) is computed at the last session of each of the past RRG_TAIL_WEEKS weeks and joined
into a tail, as on StockCharts-style RRGs. Direction matters: a tail moving clockwise from
IMPROVING towards LEADING is a strengthening trend; one heading from LEADING into WEAKENING is fading.
(Axes are percentage points centred on 0, not the proprietary JdK RS-Ratio centred on 100; the
quadrants mean the same.)

Two charts:
  holdings  the portfolio's stocks
  sectors   NSE sector indices plus our three sub-themes (equal-weighted baskets of our universe)
            and the portfolio itself (current weights): sector rotation across the market
Outputs: output/rrg_tails_{holdings,sectors}.csv and .png.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    HISTORICAL_OHLCV_CSV,
    LOCKED_PORTFOLIO_SYMBOLS,
    OUTPUT_DIR,
    RISK_SUMMARY_OUTPUT_CSV,
    RRG_MOMENTUM_DAYS,
    RRG_MOMENTUM_SMOOTHING_DAYS,
    RRG_PLOT_TAIL_WEEKS,
    RRG_TAIL_WEEKS,
    RRG_SECTOR_INDICES,
    SYMBOL_SECTOR,
    TECHNICAL_RS_LOOKBACK_DAYS,
)
from indicators import compute_rs_momentum  # noqa: E402
from rrg import GRID, QUADRANT_STYLE, SURFACE, TEXT_MUTED, TEXT_PRIMARY, classify_quadrant  # noqa: E402

logger = logging.getLogger("rrg_tails")


def tail_dates(sessions: pd.DatetimeIndex, weeks: int = RRG_TAIL_WEEKS) -> List[pd.Timestamp]:
    """The last session of each of the past `weeks` weeks, then the latest session (oldest first)."""
    s = pd.Series(sessions.sort_values(), index=sessions.sort_values())
    weekly = s.groupby(s.index.to_period("W")).max()
    points = list(weekly.iloc[-(weeks + 1):])
    return [pd.Timestamp(p) for p in points]


def rrg_tail(series_df: pd.DataFrame, benchmark: pd.DataFrame, dates: List[pd.Timestamp]) -> pd.DataFrame:
    """RS and RS-Momentum of one price series (DATE1, CLOSE_PRICE) at each date, with its quadrant."""
    s_dates = pd.to_datetime(series_df["DATE1"])
    b_dates = pd.to_datetime(benchmark["Date"])
    rows = []
    for d in dates:
        rs, _, momentum, _ = compute_rs_momentum(
            series_df[s_dates <= d], benchmark[b_dates <= d], lookback_days=TECHNICAL_RS_LOOKBACK_DAYS,
            momentum_days=RRG_MOMENTUM_DAYS, smoothing_days=RRG_MOMENTUM_SMOOTHING_DAYS)
        rows.append({"date": d.strftime("%Y-%m-%d"), "rs": rs, "momentum": momentum,
                     "quadrant": classify_quadrant(rs, momentum)})
    return pd.DataFrame(rows)


def basket(closes: pd.DataFrame, weights: Optional[pd.Series] = None) -> pd.DataFrame:
    """A price index (DATE1, CLOSE_PRICE) from daily-rebalanced weighted returns of the columns."""
    r = closes.pct_change(fill_method=None).iloc[1:]
    w = (weights.reindex(closes.columns).fillna(0.0) if weights is not None
         else pd.Series(1.0, index=closes.columns))
    eff = r.notna().mul(w, axis=1)
    eff = eff.div(eff.sum(axis=1), axis=0)
    level = (1 + (r.fillna(0.0) * eff).sum(axis=1)).cumprod() * 100
    level = pd.concat([pd.Series([100.0], index=[closes.index[0]]), level])
    return pd.DataFrame({"DATE1": level.index, "CLOSE_PRICE": level.values})


def build_tails(stock_data: pd.DataFrame, benchmark: pd.DataFrame, sector_indices: Dict[str, pd.DataFrame],
                holdings: List[str], weights: Optional[pd.Series] = None,
                weeks: int = RRG_TAIL_WEEKS) -> Dict[str, pd.DataFrame]:
    """Tails for the holdings and for the sector view (NSE indices, our sub-themes, the portfolio)."""
    df = stock_data.copy()
    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE"])]
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="mixed")
    benchmark = benchmark.copy()
    benchmark["Date"] = pd.to_datetime(benchmark["Date"])
    dates = tail_dates(pd.DatetimeIndex(sorted(set(df["DATE1"]) & set(benchmark["Date"]))), weeks)

    holding_rows = []
    for sym in holdings:
        t = rrg_tail(df[df["SYMBOL"] == sym][["DATE1", "CLOSE_PRICE"]], benchmark, dates)
        holding_rows.append(t.assign(name=sym, group="Holding"))

    closes = df.pivot_table(index="DATE1", columns="SYMBOL", values="CLOSE_PRICE").sort_index()
    sector_rows = []
    for name, idx in sector_indices.items():
        series = pd.DataFrame({"DATE1": pd.to_datetime(idx["Date"]), "CLOSE_PRICE": idx["Close"]})
        sector_rows.append(rrg_tail(series, benchmark, dates).assign(name=name, group="NSE sector index"))
    for sector in ["Cement", "Capital Goods", "Power"]:
        members = [s for s in closes.columns if SYMBOL_SECTOR.get(s) == sector]
        if members:
            sector_rows.append(rrg_tail(basket(closes[members]), benchmark, dates)
                               .assign(name=f"Our {sector} universe", group="Our sub-theme"))
    held = [s for s in holdings if s in closes.columns]
    sector_rows.append(rrg_tail(basket(closes[held], weights), benchmark, dates)
                       .assign(name="Our portfolio", group="Our portfolio"))
    cols = ["name", "group", "date", "rs", "momentum", "quadrant"]
    return {"holdings": pd.concat(holding_rows, ignore_index=True)[cols],
            "sectors": pd.concat(sector_rows, ignore_index=True)[cols]}


def plot_tails(tails: pd.DataFrame, title: str, out_path: Path, tail_names: Optional[List[str]] = None,
               emphasis: Optional[List[str]] = None, weeks: int = RRG_PLOT_TAIL_WEEKS) -> Path:
    """
    Tails of the last `weeks` weekly steps with an arrowhead at the latest point, coloured by the
    current quadrant; `tail_names` (default: all) get a tail, the rest only their latest dot.
    Labels use rrg.py's collision-avoiding placement.
    """
    from rrg import _place_labels

    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    data = tails.dropna(subset=["rs", "momentum"])
    shown_dates = sorted(data["date"].unique())[-(weeks + 1):]
    data = data[data["date"].isin(shown_dates)]
    with_tail = set(data["name"]) if tail_names is None else set(tail_names)
    visible = pd.concat([data[data["name"].isin(with_tail)], data[data["date"] == shown_dates[-1]]])
    x_lo, x_hi = min(visible["rs"].min(), 0), max(visible["rs"].max(), 0)
    y_lo, y_hi = min(visible["momentum"].min(), 0), max(visible["momentum"].max(), 0)
    x_span, y_span = max(x_hi - x_lo, 10), max(y_hi - y_lo, 6)
    ax.set_xlim(x_lo - 0.08 * x_span, x_hi + 0.16 * x_span)
    ax.set_ylim(y_lo - 0.1 * y_span, y_hi + 0.1 * y_span)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1)
    ax.axvline(0, color=TEXT_MUTED, linewidth=1)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    corner = {"LEADING": (0.98, 0.97, "right", "top"), "WEAKENING": (0.98, 0.03, "right", "bottom"),
              "LAGGING": (0.02, 0.03, "left", "bottom"), "IMPROVING": (0.02, 0.97, "left", "top")}
    for quadrant, (cx, cy, ha, va) in corner.items():
        ax.text(cx, cy, quadrant, transform=ax.transAxes, ha=ha, va=va, fontsize=13, fontweight="bold",
                color=QUADRANT_STYLE[quadrant]["color"], alpha=0.9)
    emphasis = set(emphasis or [])
    heads = []
    for name, t in data.groupby("name", sort=False):
        t = t.sort_values("date")
        head = t.iloc[-1]
        color = QUADRANT_STYLE.get(head["quadrant"], {"color": TEXT_MUTED})["color"]
        if name in with_tail and len(t) > 1:
            n = len(t)
            for i in range(1, n - 1):  # older segments fade
                ax.plot(t["rs"].iloc[i - 1:i + 1], t["momentum"].iloc[i - 1:i + 1], color=color,
                        alpha=0.3 + 0.6 * i / (n - 1), linewidth=1.6, zorder=2)
            ax.annotate("", xy=(t["rs"].iloc[-1], t["momentum"].iloc[-1]),
                        xytext=(t["rs"].iloc[-2], t["momentum"].iloc[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=14), zorder=3)
            ax.scatter(t["rs"].iloc[:-1], t["momentum"].iloc[:-1], s=12, color=color, alpha=0.7, zorder=3)
        big = name in emphasis
        ax.scatter([head["rs"]], [head["momentum"]], s=85 if big else 45, color=color,
                   edgecolors=TEXT_PRIMARY if big else SURFACE, linewidths=1.6, zorder=4)
        heads.append({"symbol": name, "rs": head["rs"], "momentum": head["momentum"]})
    heads = pd.DataFrame(heads)
    _place_labels(fig, ax, heads, "rs", "momentum", set(heads["symbol"]), emphasis or set(heads["symbol"]) & with_tail)
    ax.set_xlabel(f"RS vs Nifty 500 ({TECHNICAL_RS_LOOKBACK_DAYS}-session return spread, pp)", color=TEXT_PRIMARY)
    ax.set_ylabel(f"RS-Momentum (pp change over {RRG_MOMENTUM_DAYS} sessions, {RRG_MOMENTUM_SMOOTHING_DAYS}-session average)",
                  color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.set_title(f"{title}: {weeks}-week tails to {pd.Timestamp(shown_dates[-1]):%d-%b-%Y} "
                 "(arrow = latest week, colour = current quadrant)", color=TEXT_PRIMARY, fontsize=11.5,
                 fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved RRG tails plot to %s", out_path)
    return out_path


def run(as_of: Optional[date] = None) -> Dict[str, pd.DataFrame]:
    from fetch_data import NSEBhavcopyFetcher, fetch_benchmark_nifty500

    stock_data = pd.read_csv(HISTORICAL_OHLCV_CSV)
    dates = pd.to_datetime(stock_data["DATE1"], format="mixed")
    end = min(pd.Timestamp(as_of), dates.max()) if as_of else dates.max()
    stock_data = stock_data[dates <= end]
    start = pd.to_datetime(stock_data["DATE1"], format="mixed").min().date()
    benchmark = fetch_benchmark_nifty500(start_date=start.isoformat(), end_date=end.date().isoformat())
    fetcher = NSEBhavcopyFetcher()
    sector_indices = {}
    for name in RRG_SECTOR_INDICES:
        idx = fetcher.fetch_index_closes(start, end.date(), index_name=name)
        if not idx.empty:
            sector_indices[name] = idx
    weights = None
    if Path(RISK_SUMMARY_OUTPUT_CSV).exists():
        weights = pd.read_csv(RISK_SUMMARY_OUTPUT_CSV).set_index("symbol")["weight_pct"] / 100
    tails = build_tails(stock_data, benchmark, sector_indices, LOCKED_PORTFOLIO_SYMBOLS, weights)
    ours = ["Our portfolio", "Our Cement universe", "Our Capital Goods universe", "Our Power universe"]
    for key, title, tail_names in [("holdings", "Our holdings vs Nifty 500", None),
                                   ("sectors", "Sector rotation vs Nifty 500", ours)]:
        tails[key].to_csv(OUTPUT_DIR / f"rrg_tails_{key}.csv", index=False, float_format="%.3f")
        plot_tails(tails[key], title, OUTPUT_DIR / f"rrg_tails_{key}.png", tail_names=tail_names,
                   emphasis=ours if key == "sectors" else None)
    return tails


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    as_of_arg = date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1]) if "--as-of" in sys.argv else None
    out = run(as_of_arg)
    for key, t in out.items():
        print(t.groupby("name").tail(1).to_string(index=False))
