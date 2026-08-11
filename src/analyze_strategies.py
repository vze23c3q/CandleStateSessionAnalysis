"""
analyze_strategies.py
Version: 1.2

Purpose:
    Cross-strategy day-of-week comparison. Scans the per-strategy `output`
    folders (MACDTarget, MACDTrail, MACDFlip), pulls daily DayGain from
    sessions.csv in each, aggregates net gain by weekday per strategy, prints
    a summary table, and renders a grouped bar chart (one bar group per
    weekday, one color per strategy) with per-strategy totals in the legend.

Usage (Windows):
    python analyze_strategies.py

    Adjust the CONFIG block below to point at your paths. No CLI args
    required; everything is driven by the constants so this matches the rest
    of your CandleStateSessionAnalysis conventions.

Expected inputs:
    - BASE_DIR / <Strategy> / output / sessions.csv  (or any *sessions*.csv
      under output/). Must contain a date column (SessionStart / SessionDate /
      date) and a gain column (DayGain / DayGainLoss / RealizedPL). First
      match wins per file - if your real headers differ, edit
      DATE_COL_CANDIDATES / GAIN_COL_CANDIDATES below.

Output:
    - Prints a per-strategy day-of-week table (days, wins, losses, win pct,
      net) to console. Win is defined as DayGain >= 0.
    - Prints two data-derived DOW switching results: per weekday, run the
      strategy with (a) the best WinPct and (b) the best Net, each compared
      against the pure strategies, plus the daily oracle ceiling
      (perfect-hindsight best strategy each day) for context.
    - Saves dow_net_by_strategy.png to OUTPUT_DIR and shows the chart.

Design notes:
    - The three strategies share the same buy logic and largely the same
      entry days, so per-weekday agreement across strategies is NOT
      independent confirmation - it's the same days measured three ways.
      Where strategies DIVERGE on a weekday (e.g. Trail positive / Flip
      negative), that's the exit logic talking: a read on whether those days
      trended (Trail's runners pay) or chopped (Flip's whipsaw cap pays).
    - Weekday nets on ~18-20 observations per weekday are concentration-prone.
      A single strong weekday can be a few big shared days, not calendar
      structure. Check uplift concentration before acting on a weekday.
    - Colors are assigned by STRATEGIES list order from a fixed palette, so
      a strategy keeps its color if the list is reordered or extended.
    - The DOW switching results are FOR FUN / in-sample by construction:
      both maps are per-weekday argmax picks from this same sample, i.e.
      maximally greedy fits (the best-Net map is, by definition, the best
      any fixed weekday map can score on Net in this sample - a weekday
      oracle, not a rule). ~18-20 observations per weekday and no mechanism
      for why a weekday should matter. Treat the printed uplifts as
      upper-bound curiosities. The daily oracle line is the hindsight
      ceiling no fixed rule can beat.

Changelog:
    1.0 - Initial version. Loader (date/gain column candidates, *sessions*.csv
          glob) mirrors analyze_strategy_correlations.py so both scripts read
          the same files the same way. Grouped weekday bar chart with
          per-strategy totals in the legend, currency-formatted axis, zero
          baseline, PNG export.
    1.1 - Added DOW_RULE weekday->strategy switching result (default: Trail
          Mon/Tue, Flip Wed/Fri, Target Thu) with uplift vs each pure
          strategy and the daily perfect-hindsight oracle as a ceiling
          reference. Flagged as in-sample/for-fun in the printout.
    1.2 - Replaced the hardcoded DOW_RULE map with two data-derived maps:
          per-weekday best WinPct (win = DayGain >= 0, ties broken by
          STRATEGIES order) and per-weekday best Net. Shared evaluation
          helper prints each map, its total, uplift vs pures, and oracle
          capture. Maps are computed on complete days only (all strategies
          present) so comparisons stay apples-to-apples.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

VERSION = "1.2"

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

BASE_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data")
STRATEGIES = ["MACDTarget", "MACDTrail", "MACDFlip"]

OUTPUT_DIR = BASE_DIR / "strategy_analysis"

DATE_COL_CANDIDATES = ["SessionStart", "SessionDate", "date", "session_date"]
GAIN_COL_CANDIDATES = ["DayGain", "DayGainLoss", "RealizedPL", "daygain"]

# Fixed palette, assigned by STRATEGIES list order. Ordered so the default
# list gets MACDTarget=orange, MACDTrail=blue, MACDFlip=aqua; yellow and
# magenta are spares if you ever run more than five strategies.
STRATEGY_COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4"]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

CHART_PNG = "dow_net_by_strategy.png"

# ------------------------------------------------------------------------- #
# HELPERS
# ------------------------------------------------------------------------- #


def find_first_col(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find a {label} column. Looked for {candidates}, "
        f"found {list(columns)}."
    )


def load_strategy_daygain(strategy: str) -> pd.DataFrame:
    """Load daily DayGain for one strategy from its output folder.

    Returns a DataFrame with Date (normalized) and <strategy> columns, or an
    empty frame if no sessions CSV is found.
    """
    output_dir = BASE_DIR / strategy / "output"
    if not output_dir.exists():
        print(f"  [WARN] {output_dir} not found - skipping {strategy}.")
        return pd.DataFrame(columns=["Date", strategy])

    candidates = sorted(output_dir.glob("*sessions*.csv"))
    if not candidates:
        print(f"  [WARN] No *sessions*.csv under {output_dir} - skipping {strategy}.")
        return pd.DataFrame(columns=["Date", strategy])

    csv_path = candidates[0]
    df = pd.read_csv(csv_path)

    date_col = find_first_col(df.columns, DATE_COL_CANDIDATES, "date")
    gain_col = find_first_col(df.columns, GAIN_COL_CANDIDATES, "gain")

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        strategy: pd.to_numeric(df[gain_col], errors="coerce"),
    }).dropna(subset=["Date"])

    # One row per day: sum in case the file is per-position rather than per-day.
    out = out.groupby("Date", as_index=False)[strategy].sum()
    print(f"  Loaded {strategy:<12} {len(out):>4} days from {csv_path}")
    return out


def build_combined_daygain() -> pd.DataFrame:
    """Outer-merge all strategies into one date-indexed frame."""
    merged = None
    for strategy in STRATEGIES:
        df = load_strategy_daygain(strategy)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="Date", how="outer")

    if merged is None:
        raise SystemExit("No strategy data found - check BASE_DIR and STRATEGIES.")

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["Weekday"] = merged["Date"].dt.day_name()
    return merged


def currency(v, _pos=None) -> str:
    """$1,234 / -$1,234 (sign before the symbol)."""
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


# ------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------- #


def main():
    print(f"analyze_strategies.py v{VERSION}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full = build_combined_daygain()
    strategy_cols = [s for s in STRATEGIES if s in full.columns]

    date_min = full["Date"].min()
    date_max = full["Date"].max()
    print(f"\nPeriod: {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d} ({len(full)} days)")

    # --- Per-strategy day-of-week summary table (win = DayGain >= 0) ---
    for strategy in strategy_cols:
        rows = []
        for weekday in WEEKDAY_ORDER:
            day_gains = full.loc[full["Weekday"] == weekday, strategy].dropna()
            if day_gains.empty:
                continue
            wins = int((day_gains >= 0).sum())
            losses = int((day_gains < 0).sum())
            rows.append({
                "Weekday": weekday,
                "Days": len(day_gains),
                "Wins": wins,
                "Losses": losses,
                "WinPct": f"{wins / len(day_gains) * 100:.0f}%",
                "Net": currency(day_gains.sum()),
            })
        summary = pd.DataFrame(rows)
        total = full[strategy].sum()
        print(f"\n=== {strategy} by day of week (total: {currency(total)}) ===")
        print(summary.to_string(index=False))

    # --- For-fun DOW switching rules: per-weekday argmax picks ---
    # In-sample by construction: both maps are chosen greedily from this same
    # sample (5 slots x per-slot argmax). The best-Net map is by definition
    # the highest Net any fixed weekday map can score here - a weekday
    # oracle, not a candidate rule. Curiosities only.
    rule_df = full.dropna(subset=strategy_cols).copy()
    if not rule_df.empty:
        winpct_map = {}
        net_map = {}
        for weekday, grp in rule_df.groupby("Weekday"):
            win_rates = {s: (grp[s] >= 0).mean() for s in strategy_cols}
            nets = {s: grp[s].sum() for s in strategy_cols}
            # iterating strategy_cols -> ties break by STRATEGIES list order
            winpct_map[weekday] = max(strategy_cols, key=lambda s: win_rates[s])
            net_map[weekday] = max(strategy_cols, key=lambda s: nets[s])

        oracle_total = rule_df[strategy_cols].max(axis=1).sum()

        def evaluate_dow_rule(rule_map: dict, label: str):
            chosen = rule_df["Weekday"].map(rule_map)
            rule_total = rule_df[strategy_cols].to_numpy()[
                np.arange(len(rule_df)),
                [strategy_cols.index(c) for c in chosen],
            ].sum()
            rule_desc = ", ".join(f"{d[:3]}={rule_map[d]}" for d in WEEKDAY_ORDER if d in rule_map)
            print(f"\n=== For-fun DOW switching rule ({label}) ===")
            print(f"Map: {rule_desc}")
            print("[WARNING] In-sample fit (per-weekday argmax from this same sample). "
                  "Curiosity only, not a candidate rule.")
            print(f"Rule total: {currency(rule_total)}  (n={len(rule_df)})")
            for strategy in strategy_cols:
                actual_total = rule_df[strategy].sum()
                diff = rule_total - actual_total
                pct = (diff / abs(actual_total) * 100) if actual_total != 0 else float("nan")
                print(f"  vs pure {strategy:<12} actual: {currency(actual_total):>10}   "
                      f"rule uplift: {currency(diff):>10}  ({pct:+.1f}%)")
            capture = (rule_total / oracle_total * 100) if oracle_total != 0 else float("nan")
            print(f"Daily oracle ceiling (hindsight best each day): {currency(oracle_total)}  "
                  f"- rule captures {capture:.0f}% of it")

        evaluate_dow_rule(winpct_map, "best WinPct per weekday")
        evaluate_dow_rule(net_map, "best Net per weekday")
    else:
        print("\n[WARN] No days with all strategies present - skipping DOW rule results.")

    # --- Grouped bar chart: net by weekday, one color per strategy ---
    dow_net = (
        full.groupby("Weekday")[strategy_cols].sum()
        .reindex(WEEKDAY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(strategy_cols)
    group_width = 0.78
    bar_width = group_width / n
    x = np.arange(len(WEEKDAY_ORDER))

    for i, strategy in enumerate(strategy_cols):
        offset = (i - (n - 1) / 2) * bar_width
        total = full[strategy].sum()
        ax.bar(
            x + offset,
            dow_net[strategy].values,
            width=bar_width * 0.92,
            color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
            label=f"{strategy}  {currency(total)}",
        )

    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:3] for d in WEEKDAY_ORDER])
    ax.yaxis.set_major_formatter(FuncFormatter(currency))
    ax.set_ylabel("Net DayGain")
    ax.set_title(
        f"Net DayGain by weekday  |  {date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}  ({len(full)} days)"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = OUTPUT_DIR / CHART_PNG
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved chart -> {png_path}")

    plt.show()


if __name__ == "__main__":
    main()
