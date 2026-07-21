"""
analyze_strategy_correlations.py
Version: 1.1

Purpose:
    Scan the per-strategy `output` folders for MACDTarget, MACDTrail, and MACDFlip,
    pull daily DayGain from sessions.csv (or *sessions*.csv files) in each, combine
    into a single date-indexed DataFrame (one column per strategy), then correlate
    each strategy's daily performance against SPY trend-regime data (SwingState
    analytics export) and overnight gap.

Usage (Windows):
    python analyze_strategy_correlations.py

    Adjust the CONFIG block below to point at your paths. No CLI args required;
    everything is driven by the constants so this matches the rest of your
    CandleStateSessionAnalysis conventions.

Expected inputs:
    - BASE_DIR / <Strategy> / output / sessions.csv  (or any *sessions*.csv under output/)
      Must contain a date column (SessionStart / SessionDate / date) and a gain
      column (DayGain / DayGainLoss / RealizedPL). First match wins per file - if
      your real headers differ, edit DATE_COL_CANDIDATES / GAIN_COL_CANDIDATES below.
    - SPY_ANALYTICS_JSON: SwingState per-day export. Needs Timestamp, Close, Open
      (for GapBps), RSI, FastSlopeDeg/FastTrend, SlowSlopeDeg/SlowTrend.

Output:
    - Prints a correlation summary to console
    - Writes combined_daily_daygain.csv to OUTPUT_DIR
    - correlation_matrix.csv save + heatmap plot are currently disabled (see main())

Analysis included:
    - Correlation matrix: strategies vs SPY trend fields (same-day and lagged
      _Prior versions) and GapBps.
    - Best-performer summary: totals, means, day counts by strategy.
    - Oracle calc: total DayGain if the best strategy were picked every day,
      plus an uplift-concentration breakdown (top N days' share of the gap
      over the best single-strategy baseline).
    - Trend-regime bucket comparison: mean DayGain by |FastSlopeDeg_Prior|
      tertile (Chop/Moderate/Strong) and direction, with row counts and the
      actual qcut bin edges printed.
    - Gap bucket comparison: same structure, using |GapBps| tertiles
      (Small/Moderate/Large) and gap direction.
    - Candidate switching rule test: 3-way gap-size split (Small -> MACDTarget,
      Moderate -> MACDTarget, Large -> MACDTrail), based on which strategy has
      the higher mean in each tertile - flagged as an in-sample fit, not a
      validated edge.

Design notes:
    - FastSlopeDeg/FastTrend/SlowSlopeDeg/SlowTrend/RSI are lagged by one
      trading day (_Prior columns) because same-day values are computed from
      that day's own closing candle, so they partly reflect that day's own
      price action - same-day correlation against that day's DayGain is
      cause/effect conflation.
    - GapBps is NOT lagged: it's fully known at market open, before any
      intraday trading, so same-day is already the causally correct alignment.
    - Bucket splits use pd.qcut (data-relative tertiles, roughly equal cell
      counts) rather than fixed thresholds, and print their own bin edges so
      the cutoffs are never hidden behind a label.
    - Every bucket/cross-tab includes row counts - thin cells (especially in
      the strength x direction and gap-size x direction cross-tabs) should be
      read as low-confidence regardless of how large the mean looks.
    - The switching rule is chosen by inspecting this exact dataset (picking
      the higher mean per tertile), so its "uplift" is optimistic by
      construction. A 3-way split has more free choices than a 2-way split,
      so it fits this sample more closely and is a WEAKER bet on new data,
      not a stronger one - more free parameters = more overfitting risk, even
      though the in-sample number looks better.

Changelog:
    1.0 - Baseline consolidated version. F&G removed (ruled out - no linear
          relationship, same-day or lagged). Kept: strategy scanning/merge,
          SPY trend fields with proper lag, GapBps (correctly unlagged) with
          validation printout, correlation matrix, best-performer summary,
          oracle + uplift concentration, trend-regime and gap bucket
          comparisons with bin edges and counts, in-sample switching-rule test.
    1.1 - Fixed the switching rule from a 2-way split (Small -> MACDTarget,
          Moderate+Large lumped together -> MACDTrail) to the correct 3-way
          split matching the actual bucket means: MACDTarget wins both Small
          AND Moderate, MACDTrail only wins Large. Merged in matplotlib/
          seaborn import and plot_correlation_heatmap() (currently unused -
          call is commented out in main()). Correlation matrix CSV save is
          currently disabled in main() as well.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

BASE_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data")
STRATEGIES = ["MACDTarget", "MACDTrail", "MACDFlip"]

# SwingState daily analytics export (SPY) - JSON array of per-day records with
# Timestamp, Open, Close, FastMA/FastSlopeDeg/FastTrend, SlowMA/SlowSlopeDeg/
# SlowTrend, RSI, VolumeMA, etc.
ANALYTICS_JSON = Path(r"C:\Git\CandleStateSessionAnalysis\data\analytics_QQQ.json")

OUTPUT_DIR = BASE_DIR / "correlation_analysis"

DATE_COL_CANDIDATES = ["SessionStart", "SessionDate", "date", "session_date"]
GAIN_COL_CANDIDATES = ["DayGain", "DayGainLoss", "RealizedPL", "daygain"]

# ------------------------------------------------------------------------- #
# HELPERS
# ------------------------------------------------------------------------- #


def find_first_col(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find a {label} column. Looked for {candidates}, "
        f"found columns: {list(columns)}"
    )


def find_session_csvs(strategy: str) -> list[Path]:
    """Scan <BASE_DIR>/<strategy>/output for sessions.csv (recursive, any *sessions*.csv)."""
    output_folder = BASE_DIR / strategy / "output"
    if not output_folder.exists():
        print(f"  [WARN] Output folder not found: {output_folder}")
        return []

    matches = sorted(output_folder.rglob("*sessions*.csv"))
    if not matches:
        print(f"  [WARN] No *sessions*.csv files found under {output_folder}")
    return matches


def load_strategy_daygain(strategy: str) -> pd.DataFrame:
    """Return a Date/DayGain df (one row per calendar day, summed) for one strategy."""
    print(f"Scanning {strategy}...")
    csv_paths = find_session_csvs(strategy)
    if not csv_paths:
        return pd.DataFrame(columns=["Date", strategy])

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        date_col = find_first_col(df.columns, DATE_COL_CANDIDATES, "date")
        gain_col = find_first_col(df.columns, GAIN_COL_CANDIDATES, "gain")

        df = df[[date_col, gain_col]].copy()
        df.columns = ["Date", strategy]
        # UTC-normalize then drop to calendar date (matches existing convention)
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["Date"])
        frames.append(df)
        print(f"  loaded {len(df):>5} rows from {p.name}")

    combined = pd.concat(frames, ignore_index=True)
    # Sum in case multiple sessions land on the same calendar day (e.g. paired accounts)
    daily = combined.groupby("Date", as_index=False)[strategy].sum()
    print(f"  -> {len(daily)} unique trading days for {strategy}")
    return daily


def build_combined_daygain() -> pd.DataFrame:
    """Outer-join all three strategies on Date -> wide df with one column per strategy."""
    merged = None
    for strategy in STRATEGIES:
        daily = load_strategy_daygain(strategy)
        if merged is None:
            merged = daily
        else:
            merged = merged.merge(daily, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)

    # Best-performing strategy per day (ignores NaNs)
    strategy_cols = [s for s in STRATEGIES if s in merged.columns]
    merged["BestStrategy"] = merged[strategy_cols].idxmax(axis=1, skipna=True)

    return merged


def load_spy_analytics(path: Path) -> pd.DataFrame:
    """Load SwingState's per-day SPY analytics JSON export.

    Pulls Open/Close plus the trend-regime fields: FastSlopeDeg/FastTrend
    (short MA slope + consecutive-same-direction day count), SlowSlopeDeg/
    SlowTrend (same, longer MA), and RSI. Early rows in the export won't have
    MA-derived fields yet (rolling window warmup) - those come back as NaN,
    which is fine, they just won't fire in the correlation/bucket breakdown.

    Computes:
      - GapBps: overnight gap in basis points, (Open - PriorClose) /
        PriorClose * 10000. NOT lagged - fully known at market open, before
        any intraday trading, so same-day is already causally correct.
      - *_Prior: FastSlopeDeg/FastTrend/SlowSlopeDeg/SlowTrend/RSI lagged one
        trading day, since same-day values are computed from that day's own
        closing candle (cause/effect conflation if used same-day).
    """
    if not path.exists():
        print(f"  [WARN] SPY analytics JSON not found at {path} - skipping.")
        return pd.DataFrame(columns=["Date"])

    with open(path, "r") as f:
        records = json.load(f)

    rows = []
    for r in records:
        rows.append({
            "Date": r.get("Timestamp"),
            "SPY_Open": r.get("Open"),
            "SPY_Close": r.get("Close"),
            "RSI": r.get("RSI"),
            "FastSlopeDeg": r.get("FastSlopeDeg"),
            "FastTrend": r.get("FastTrend"),
            "SlowSlopeDeg": r.get("SlowSlopeDeg"),
            "SlowTrend": r.get("SlowTrend"),
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if df["SPY_Open"].notna().any():
        df["PriorClose"] = df["SPY_Close"].shift(1)
        df["Gap"] = df["SPY_Open"] - df["PriorClose"]
        df["GapBps"] = df["Gap"] / df["PriorClose"] * 10000
    else:
        print("  [WARN] No 'Open' field found in SPY analytics JSON - GapBps will be unavailable. "
              "Add Open to the SwingState export to enable gap analysis.")
        df["Gap"] = np.nan
        df["GapBps"] = np.nan

    for col in ["FastSlopeDeg", "FastTrend", "SlowSlopeDeg", "SlowTrend", "RSI"]:
        df[f"{col}_Prior"] = df[col].shift(1)

    return df


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6})
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------- #


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daygain_df = build_combined_daygain()
    strategy_cols = [s for s in STRATEGIES if s in daygain_df.columns]

    spy_df = load_spy_analytics(ANALYTICS_JSON)
    full = daygain_df.merge(spy_df, on="Date", how="left")
    full = full.sort_values("Date").reset_index(drop=True)

    # --- Gap validation output: eyeball the raw numbers behind GapBps ---
    if "Gap" in full.columns and full["Gap"].notna().any():
        gap_check = full[["Date", "PriorClose", "SPY_Open", "Gap", "GapBps"]].rename(
            columns={"SPY_Open": "Open"}
        )
        print("\n=== Gap validation: Date, PriorClose, Open, Gap, GapBps ===")
        print(gap_check.to_string(index=False))

    combined_path = OUTPUT_DIR / "combined_daily_daygain.csv"
    full.to_csv(combined_path, index=False)
    print(f"\nSaved combined daily df -> {combined_path}")

    # --- Correlations ---
    corr_cols = strategy_cols + [
        "FastSlopeDeg", "FastSlopeDeg_Prior", "FastTrend", "FastTrend_Prior",
        "SlowSlopeDeg", "SlowSlopeDeg_Prior", "SlowTrend", "SlowTrend_Prior",
        "RSI", "RSI_Prior", "GapBps",
    ]
    corr_cols = [c for c in corr_cols if c in full.columns]

    corr_matrix = full[corr_cols].corr(method="pearson")
    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    # corr_matrix.to_csv(corr_path)  # disabled

    print("\n=== Correlation matrix (Pearson) ===")
    print(corr_matrix.round(3).to_string())
    #plot_correlation_heatmap(corr_matrix)  

    # --- Best performer summary ---
    print("\n=== Best strategy by day count ===")
    print(full["BestStrategy"].value_counts(dropna=True).to_string())

    print("\n=== Total DayGain by strategy ===")
    print(full[strategy_cols].sum().sort_values(ascending=False).to_string())

    print("\n=== Mean DayGain by strategy ===")
    print(full[strategy_cols].mean().sort_values(ascending=False).to_string())

    # --- Oracle: total gain if we picked the best strategy every single day ---
    oracle_daily_gain = full[strategy_cols].max(axis=1, skipna=True)
    oracle_total = oracle_daily_gain.sum()

    print("\n=== Oracle: picking the best strategy every day ===")
    print(f"Oracle total DayGain: {oracle_total:,.2f}  (across {len(full)} days)")
    for strategy in strategy_cols:
        actual_total = full[strategy].sum()
        uplift = oracle_total - actual_total
        pct = (uplift / abs(actual_total) * 100) if actual_total != 0 else float("nan")
        print(f"  vs {strategy:<12} actual: {actual_total:>12,.2f}   oracle uplift: {uplift:>12,.2f}  ({pct:+.1f}%)")

    # --- Uplift concentration: is the oracle's edge broad-based or a few outlier days? ---
    baseline_strategy = full[strategy_cols].sum().idxmax()
    full["DailyUplift"] = oracle_daily_gain - full[baseline_strategy]

    top_n = 10
    top_days = full[["Date", baseline_strategy, "BestStrategy", "DailyUplift"]].sort_values(
        "DailyUplift", ascending=False
    ).head(top_n)
    top_uplift_sum = top_days["DailyUplift"].sum()
    total_uplift_sum = full.loc[full["DailyUplift"] > 0, "DailyUplift"].sum()
    pct_from_top = (top_uplift_sum / total_uplift_sum * 100) if total_uplift_sum else float("nan")

    print(f"\n=== Uplift concentration (baseline: {baseline_strategy}, highest actual total) ===")
    print(f"Top {top_n} days account for {pct_from_top:.1f}% of total positive uplift "
          f"({top_uplift_sum:,.2f} of {total_uplift_sum:,.2f})")
    print(top_days.to_string(index=False))

    # --- Trend-regime bucket comparison ---
    if "FastSlopeDeg_Prior" in full.columns and full["FastSlopeDeg_Prior"].notna().sum() > 0:
        trend_df = full.dropna(subset=["FastSlopeDeg_Prior"]).copy()
        trend_df["TrendStrength"], trend_bins = pd.qcut(
            trend_df["FastSlopeDeg_Prior"].abs(), q=3, labels=["Chop", "Moderate", "Strong"], retbins=True
        )
        trend_df["TrendDirection"] = np.where(trend_df["FastSlopeDeg_Prior"] >= 0, "Up", "Down")

        print("\n=== |FastSlopeDeg_Prior| tertile edges (data-relative, from qcut) ===")
        print(f"  Chop:     0.00 to {trend_bins[1]:.3f} deg")
        print(f"  Moderate: {trend_bins[1]:.3f} to {trend_bins[2]:.3f} deg")
        print(f"  Strong:   {trend_bins[2]:.3f} to {trend_bins[3]:.3f} deg")

        print(f"\n=== Mean DayGain by trend strength (|FastSlopeDeg_Prior| tertile, n={len(trend_df)}) ===")
        print(trend_df.groupby("TrendStrength", observed=True)[strategy_cols].agg(["mean", "count"]).round(2).to_string())

        print("\n=== Mean DayGain by trend direction ===")
        print(trend_df.groupby("TrendDirection", observed=True)[strategy_cols].agg(["mean", "count"]).round(2).to_string())

        print("\n=== Mean DayGain by strength x direction (n = cell count, watch for thin cells) ===")
        print(trend_df.groupby(["TrendStrength", "TrendDirection"], observed=True)[strategy_cols]
              .agg(["mean", "count"]).round(2).to_string())
    else:
        print("\n[WARN] No FastSlopeDeg_Prior data available - skipping trend-regime bucket comparison. "
              f"Check ANALYTICS_JSON path: {ANALYTICS_JSON}")

    # --- Overnight gap bucket comparison ---
    if "GapBps" in full.columns and full["GapBps"].notna().sum() > 0:
        gap_df = full.dropna(subset=["GapBps"]).copy()
        gap_df["GapDirection"] = np.where(gap_df["GapBps"] >= 0, "GapUp", "GapDown")
        gap_df["GapSize"], gap_size_bins = pd.qcut(
            gap_df["GapBps"].abs(), q=3, labels=["Small", "Moderate", "Large"], retbins=True
        )

        print("\n=== |GapBps| tertile edges (data-relative, from qcut) ===")
        print(f"  Small:    0.00 to {gap_size_bins[1]:.2f} bps")
        print(f"  Moderate: {gap_size_bins[1]:.2f} to {gap_size_bins[2]:.2f} bps")
        print(f"  Large:    {gap_size_bins[2]:.2f} to {gap_size_bins[3]:.2f} bps")

        print(f"\n=== Mean DayGain by gap direction (n={len(gap_df)}) ===")
        print(gap_df.groupby("GapDirection", observed=True)[strategy_cols].agg(["mean", "count"]).round(2).to_string())

        print("\n=== Mean DayGain by gap size (|GapBps| tertile) ===")
        print(gap_df.groupby("GapSize", observed=True)[strategy_cols].agg(["mean", "count"]).round(2).to_string())

        print("\n=== Mean DayGain by gap size x direction (n = cell count, watch for thin cells) ===")
        print(gap_df.groupby(["GapSize", "GapDirection"], observed=True)[strategy_cols]
              .agg(["mean", "count"]).round(2).to_string())
    else:
        print("\n[WARN] No GapBps data available - skipping gap bucket comparison. "
              "Add 'Open' to the SwingState SPY analytics export to enable this.")

    # --- Candidate switching rule: 3-way gap-size split ---
    # Rule: Small -> MACDTarget, Moderate -> MACDTarget, Large -> MACDTrail.
    # Picked as the higher-mean strategy in EACH tertile (previous version
    # incorrectly lumped Moderate in with Large, handing MACDTrail a bucket
    # where MACDTarget actually had the higher mean).
    #
    # IMPORTANT: this rule was picked by inspecting this SAME sample, so the
    # comparison below is in-sample and optimistic BY CONSTRUCTION. A 3-way
    # split has MORE free choices than the old 2-way split, so it fits this
    # 88-day sample more closely - that makes the printed uplift look better,
    # but makes the rule a WEAKER bet on new data, not a stronger one. Treat
    # as a hypothesis to monitor out-of-sample, not a proven edge.
    if "GapBps" in full.columns and full["GapBps"].notna().sum() > 0:
        switch_df = full.dropna(subset=["GapBps"]).copy()
        switch_df["GapSize"] = pd.qcut(switch_df["GapBps"].abs(), q=3, labels=["Small", "Moderate", "Large"])
        switch_df["ChosenStrategy"] = np.where(switch_df["GapSize"] == "Large", "MACDTrail", "MACDTarget")
        switch_df["SwitchedDayGain"] = switch_df.apply(lambda r: r[r["ChosenStrategy"]], axis=1)

        switched_total = switch_df["SwitchedDayGain"].sum()
        print("\n=== Candidate switching rule: Small/Moderate gap -> MACDTarget, Large -> MACDTrail ===")
        print("[WARNING] In-sample fit (3 free choices) - rule was chosen from this same data. "
              "Not proven out-of-sample.")
        print(f"Switched total: {switched_total:,.2f}  (n={len(switch_df)})")
        for strategy in strategy_cols:
            actual_total = switch_df[strategy].sum()
            diff = switched_total - actual_total
            pct = (diff / abs(actual_total) * 100) if actual_total != 0 else float("nan")
            print(f"  vs pure {strategy:<12} actual: {actual_total:>12,.2f}   switch uplift: {diff:>12,.2f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
