"""
analyze_regime_match.py
Version: 1.0

Purpose:
    Second crack at regime -> strategy matching, built on intraday-character
    persistence rather than single-day snapshots. Computes open-time-knowable
    regime features from the SwingState SPY analytics export (daily candles)
    plus Fear & Greed history, joins them to per-strategy daily DayGain
    (Target / Trail / LVL), and tests whether regime buckets
    separate strategy performance - including against the Trail-minus-Flip
    daily spread, a direct measurement of trend-vs-chop that the shared buy
    logic makes possible.

Usage (Windows):
    python analyze_regime_match.py

    Adjust the CONFIG block below to point at your paths. No CLI args
    required; everything is driven by the constants so this matches the rest
    of your CandleStateSessionAnalysis conventions.

Expected inputs:
    - BASE_DIR / <Strategy> / output / sessions.csv  (or any *sessions*.csv
      under output/). Date column (SessionStart / SessionDate / date) and
      gain column (DayGain / DayGainLoss / RealizedPL); first match wins.
    - ANALYTICS_JSON: SwingState per-day SPY export with Timestamp, Open,
      Close, CandleHeight, CandleBodyPct, RSI, FastTrend, FastSlopeDeg.
    - FEAR_GREED_CSV: Date,Value rows (extra empty columns tolerated).

Output:
    - Prints per-regime-bucket mean DayGain per strategy with cell counts.
    - Prints correlations of each open-time feature vs the Trail-Flip spread
      (the trendiness outcome) and vs each strategy's DayGain.
    - Prints a best-per-bucket switching result with in-sample warning.
    - Writes regime_days.csv (per-day features + labels) to OUTPUT_DIR.

Regime features (ALL knowable at that day's open - prior days only, except
the gap which is known at 9:30):
    - Trail5_BodyPct:  5-day trailing mean of CandleBodyPct. Intraday
      directionality persistence: big bodies = directional days, small
      bodies = wicky indecision. THE candidate axis - July (worst month for
      all three strategies) has the lowest trailing body pct in the sample.
    - Trail5_HeightBps: 5-day trailing mean of candle height in bps of
      price. Intraday range persistence. June pairs low multi-day ER with
      big bodies+height and was the best month - so intraday size matters
      even when the multi-day path nets nowhere.
    - ER10: 10-day Kaufman efficiency ratio through yesterday's close
      (|net move| / sum of |daily moves|). Multi-day trendiness.
    - GapBps: overnight gap, known at open (not lagged).
    - FG / FG_Chg5: Fear & Greed level and its 5-day change.
    - RSI_Prior, FastTrend_Prior, FastSlopeDeg_Prior: lagged one day (same
      convention as analyze_strategy_correlations.py - same-day values are
      computed from that day's own close).

Design notes:
    - Prior finding: same-day/lagged regime features show no LINEAR
      correlation with strategy DayGain. This script tests bucketed
      (nonlinear) effects and uses the Trail-Flip spread as the outcome,
      which isolates exit-logic payoff (trend vs chop) from shared entries.
    - Daily SPY candles are a coarse shadow of what intraday strategies
      harvest: weekday-level checks show SPY daily character does NOT line
      up with strategy weekday P/L (e.g. Tuesdays are among the wickiest
      SPY days yet the most profitable strategy days). If nothing here
      separates the buckets, the honest conclusion is that daily-granularity
      features can't see the intraday texture - and the strategies' own
      session stats (transactions/day, flip counts) are the better regime
      detector.
    - Bucket splits use pd.qcut (data-relative tertiles) and print their own
      bin edges. Every cell prints counts - thin cells are low-confidence.
    - Best-per-bucket switching is in-sample BY CONSTRUCTION (picked from
      this same data). Printed for scale, not as a candidate rule.

Changelog:
    1.0 - Initial version. Open-time feature builder (trailing candle
          character, ER10, gap, F&G, lagged trend fields), strategy loader
          mirroring analyze_strategy_correlations.py, Trail-Flip spread
          outcome, tertile bucket tables, best-per-bucket switch test,
          regime_days.csv export.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

VERSION = "1.0"

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

BASE_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data")
STRATEGIES = ["Target", "Trail", "LVL"]

ANALYTICS_JSON = Path(r"C:\Git\CandleStateSessionAnalysis\data\analytics_SPY.json")
FEAR_GREED_CSV = Path(r"C:\Git\CandleStateSessionAnalysis\data\fear_greed.csv")

OUTPUT_DIR = BASE_DIR / "regime_analysis"

DATE_COL_CANDIDATES = ["SessionStart", "SessionDate", "date", "session_date"]
GAIN_COL_CANDIDATES = ["DayGain", "DayGainLoss", "RealizedPL", "daygain"]

# Spread pair: SpreadTrendChop = TREND_STRATEGY - CHOP_STRATEGY per day.
# Positive = the day paid the runner exit; negative = the whipsaw cap paid.
TREND_STRATEGY = "Trail"
CHOP_STRATEGY = "MACDFlip"

TRAIL_WINDOW = 5    # days for trailing candle-character means
ER_WINDOW = 10      # days for the efficiency ratio

FEATURES = [
    "Trail5_BodyPct", "Trail5_HeightBps", "ER10", "GapBps",
    "FG", "FG_Chg5", "RSI_Prior", "FastTrend_Prior", "FastSlopeDeg_Prior",
]

REGIME_CSV = "regime_days.csv"

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
    output_dir = BASE_DIR / strategy / "output"
    if not output_dir.exists():
        print(f"  [WARN] {output_dir} not found - skipping {strategy}.")
        return pd.DataFrame(columns=["Date", strategy])

    candidates = sorted(output_dir.glob("*sessions*.csv"))
    if not candidates:
        print(f"  [WARN] No *sessions*.csv under {output_dir} - skipping {strategy}.")
        return pd.DataFrame(columns=["Date", strategy])

    df = pd.read_csv(candidates[0])
    date_col = find_first_col(df.columns, DATE_COL_CANDIDATES, "date")
    gain_col = find_first_col(df.columns, GAIN_COL_CANDIDATES, "gain")

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        strategy: pd.to_numeric(df[gain_col], errors="coerce"),
    }).dropna(subset=["Date"])
    out = out.groupby("Date", as_index=False)[strategy].sum()
    print(f"  Loaded {strategy:<12} {len(out):>4} days from {candidates[0]}")
    return out


def build_combined_daygain() -> pd.DataFrame:
    merged = None
    for strategy in STRATEGIES:
        df = load_strategy_daygain(strategy)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="Date", how="outer")
    if merged is None:
        raise SystemExit("No strategy data found - check BASE_DIR and STRATEGIES.")
    return merged.sort_values("Date").reset_index(drop=True)


def build_regime_features() -> pd.DataFrame:
    """Open-time-knowable regime features from SPY analytics + F&G.

    Trailing features use .shift(1) BEFORE rolling so only prior days feed
    them. GapBps uses today's open (known at 9:30) - deliberately unlagged.
    """
    if not ANALYTICS_JSON.exists():
        raise SystemExit(f"SPY analytics JSON not found at {ANALYTICS_JSON}")

    with open(ANALYTICS_JSON, "r") as f:
        records = json.load(f)
    spy = pd.DataFrame(records)
    spy["Date"] = (pd.to_datetime(spy["Timestamp"], utc=True)
                   .dt.tz_localize(None).dt.normalize())
    spy = spy.sort_values("Date").reset_index(drop=True)

    spy["PriorClose"] = spy["Close"].shift(1)
    spy["GapBps"] = (spy["Open"] - spy["PriorClose"]) / spy["PriorClose"] * 10000
    spy["HeightBps"] = spy["CandleHeight"] / spy["Close"] * 10000

    spy["Trail5_BodyPct"] = (spy["CandleBodyPct"].shift(1)
                             .rolling(TRAIL_WINDOW).mean())
    spy["Trail5_HeightBps"] = (spy["HeightBps"].shift(1)
                               .rolling(TRAIL_WINDOW).mean())

    price_path = spy["Close"].diff().abs().shift(1).rolling(ER_WINDOW).sum()
    net_move = (spy["PriorClose"] - spy["Close"].shift(ER_WINDOW + 1)).abs()
    spy["ER10"] = net_move / price_path

    for col in ["RSI", "FastTrend", "FastSlopeDeg"]:
        spy[f"{col}_Prior"] = spy[col].shift(1)

    if FEAR_GREED_CSV.exists():
        fg = pd.read_csv(FEAR_GREED_CSV, usecols=[0, 1], encoding="utf-8-sig")
        fg.columns = ["Date", "FG"]
        fg["Date"] = pd.to_datetime(fg["Date"], errors="coerce")
        fg = fg.dropna().sort_values("Date").reset_index(drop=True)
        fg["FG_Chg5"] = fg["FG"] - fg["FG"].shift(5)
        spy = spy.merge(fg, on="Date", how="left")
    else:
        print(f"  [WARN] Fear & Greed CSV not found at {FEAR_GREED_CSV} - "
              "FG features will be NaN.")
        spy["FG"] = np.nan
        spy["FG_Chg5"] = np.nan

    keep = ["Date"] + [f for f in FEATURES if f in spy.columns]
    return spy[keep]


def tertile_table(df: pd.DataFrame, feature: str, outcome_cols: list) -> None:
    """Print bin edges and per-tertile means (+counts) for one feature."""
    data = df.dropna(subset=[feature]).copy()
    if len(data) < 15:
        print(f"\n[WARN] Too few rows with {feature} ({len(data)}) - skipping.")
        return
    try:
        data["Bucket"], bins = pd.qcut(
            data[feature], q=3, labels=["Low", "Mid", "High"],
            retbins=True, duplicates="drop"
        )
    except ValueError:
        print(f"\n[WARN] Could not form tertiles for {feature} - skipping.")
        return

    edges = "  ".join(f"{b:.2f}" for b in bins)
    print(f"\n=== Mean by {feature} tertile (edges: {edges}) ===")
    print(data.groupby("Bucket", observed=True)[outcome_cols]
          .agg(["mean", "count"]).round(2).to_string())


# ------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------- #


def main():
    print(f"analyze_regime_match.py v{VERSION}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daygain = build_combined_daygain()
    strategy_cols = [s for s in STRATEGIES if s in daygain.columns]

    regime = build_regime_features()
    full = daygain.merge(regime, on="Date", how="left")
    full = full.sort_values("Date").reset_index(drop=True)

    # Trendiness outcome: exit-logic payoff isolated from shared entries.
    spread_col = None
    if TREND_STRATEGY in full.columns and CHOP_STRATEGY in full.columns:
        spread_col = "SpreadTrendChop"
        full[spread_col] = full[TREND_STRATEGY] - full[CHOP_STRATEGY]

    csv_path = OUTPUT_DIR / REGIME_CSV
    full.to_csv(csv_path, index=False)
    print(f"\nSaved per-day regime features -> {csv_path}")

    features = [f for f in FEATURES if f in full.columns]
    outcome_cols = strategy_cols + ([spread_col] if spread_col else [])

    # --- Linear check: features vs strategies and vs the spread ---
    print("\n=== Correlation: open-time features vs DayGain and Trail-Flip spread ===")
    print("(Prior finding was ~zero vs raw DayGain; the spread column is the new test.)")
    corr = full[features + outcome_cols].corr(method="pearson")
    print(corr.loc[features, outcome_cols].round(3).to_string())

    # --- Bucketed (nonlinear) check per feature ---
    for feature in features:
        tertile_table(full, feature, outcome_cols)

    # --- 2-axis regime: directionality x range, best-per-cell switch ---
    axis_a, axis_b = "Trail5_BodyPct", "Trail5_HeightBps"
    if axis_a in full.columns and axis_b in full.columns:
        cell_df = full.dropna(subset=[axis_a, axis_b] + strategy_cols).copy()
        if len(cell_df) >= 30:
            cell_df["Directionality"] = pd.qcut(
                cell_df[axis_a], q=2, labels=["Wicky", "Directional"])
            cell_df["Range"] = pd.qcut(
                cell_df[axis_b], q=2, labels=["Quiet", "Wide"])
            print("\n=== Mean DayGain by directionality x range "
                  "(2x2, n = cell count) ===")
            grouped = (cell_df
                       .groupby(["Directionality", "Range"], observed=True)[strategy_cols]
                       .agg(["mean", "count"]).round(2))
            print(grouped.to_string())

            means = (cell_df
                     .groupby(["Directionality", "Range"], observed=True)[strategy_cols]
                     .mean())
            best_per_cell = means.idxmax(axis=1)
            print("\n=== Best strategy per cell (in-sample - "
                  "4 free choices, scale only, not a rule) ===")
            print(best_per_cell.to_string())

            cell_df["Cell"] = list(zip(cell_df["Directionality"], cell_df["Range"]))
            chosen = {cell: best_per_cell.loc[cell] for cell in best_per_cell.index}
            cell_df["SwitchGain"] = cell_df.apply(
                lambda r: r[chosen[r["Cell"]]], axis=1)
            switch_total = cell_df["SwitchGain"].sum()
            oracle_total = cell_df[strategy_cols].max(axis=1).sum()
            print(f"\nBest-per-cell switch total: {switch_total:,.2f}  "
                  f"(n={len(cell_df)})")
            for strategy in strategy_cols:
                actual = cell_df[strategy].sum()
                diff = switch_total - actual
                print(f"  vs pure {strategy:<12} actual: {actual:>12,.2f}   "
                      f"uplift: {diff:>12,.2f}")
            print(f"Daily oracle ceiling: {oracle_total:,.2f}  - switch captures "
                  f"{switch_total / oracle_total * 100:.0f}% of it"
                  if oracle_total else "")
        else:
            print(f"\n[WARN] Only {len(cell_df)} complete rows for the 2x2 "
                  "regime grid - skipping (need 30+).")


if __name__ == "__main__":
    main()
