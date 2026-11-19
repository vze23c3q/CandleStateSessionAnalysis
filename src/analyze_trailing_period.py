#!/usr/bin/env python3
"""
analyze_trailing_period.py  v1.1

Reactive regime-deterioration monitor for CandleState.

Purpose
-------
CandleState's MACD strategies (Target/Trail/Flip) have no leading indicator
that reliably predicts whipsaw regimes (confirmed: VolatilityBps mean, ADX,
HOD-proximity, and time-of-day were all checked and ruled out). This script
does not try to predict a regime shift. It answers a narrower, reactive
question instead: "is the strategy's most recent trailing window unusual
relative to its own full history?" -- the same logic already used for the
MA-stack-driven Trail<->Target switch, applied to realized results instead
of price structure.

It computes, as of the most recent trading day in the data:
  1. Trailing N-day Target-Hit rate (default N=20, ~1 trading month)
  2. Trailing N-day realized net P/L (sum of DayGain)
  3. Where each trailing metric falls in the full historical distribution
     of all rolling N-day windows for this strategy (percentile rank)
  4. Two independent alert triggers (either one can fire):
       a. PERCENTILE alert: current trailing window is at/below the
          --alert-percentile of all windows in history. Good at catching
          "this is worse than almost anything I've seen," but on a
          mostly-decent dataset it can be too lenient early in a real
          decline (a slow grind doesn't look unusual until it's deep).
       b. DURATION alert: trailing hit rate has stayed below
          --duration-threshold for at least --duration-days consecutive
          trading days. This is closer to how a human eye reads "this
          doesn't look good" -- a plain level, not a percentile -- but
          filtered by sustained duration so it doesn't fire on every
          few-day dip. Tested against 2025-06 to 2026-08 data: a bare
          "below 50%" level alone fires ~9 separate times (many just
          1-6 days, pure noise); requiring 10+ consecutive days below
          50% collapses that down to the two episodes that actually
          corresponded to the real 2025 regime failure and the Feb-Mar
          2026 dip, with no short-lived false positives.

It does NOT recommend a specific position size or trade. It surfaces the
numbers so you can apply your own capital-preservation rule (e.g. revert to
Target, or trigger the half-capital rotation fallback) with an objective
trigger instead of eyeballing monthly tables.

Input
-----
--src points at a DIRECTORY containing a sessions.csv export, one row per
trading day, with at least:
    SessionStart, DayGain, TargetHit
(TradingCapital, DayTarget, MaxDayLoss, SourceFile are used if present but
not required.)

This script assumes ONE sessions.csv = ONE STRATEGY (e.g. a Trail export
and a Target export live in separate folders) -- CandleState's session
exports don't appear to carry a strategy field at the day-summary level.
If one CSV contains multiple strategies, pass --strategy-col / --strategy
to filter.

Usage
-----
    python analyze_trailing_period.py --src "C:\\Git\\CandleStateSessionAnalysis\\data\\Target\\output"

    python analyze_trailing_period.py --src ./data --window 10 --alert-percentile 15

    python analyze_trailing_period.py --src ./data --duration-threshold 50 --duration-days 10

    python analyze_trailing_period.py --src ./data --plot trailing.png --csv-out full_table.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_csv(src_dir: Path, strategy_col: str | None, strategy_value: str | None):
    file = src_dir / "sessions.csv"
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Could not read {file}: {e}", file=sys.stderr)
        sys.exit(1)

    required = ["SessionStart", "DayGain", "TargetHit"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            f"CSV is missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if strategy_col:
        if strategy_col not in df.columns:
            print(f"--strategy-col '{strategy_col}' not found in CSV columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        if strategy_value:
            before = len(df)
            df = df[df[strategy_col].astype(str).str.contains(strategy_value, case=False, na=False)]
            print(f"Filtered {strategy_col} contains '{strategy_value}': {before} -> {len(df)} rows")

    df["date"] = pd.to_datetime(df["SessionStart"]).dt.normalize()
    df["day_gain"] = pd.to_numeric(df["DayGain"], errors="coerce")
    df["target_hit"] = df["TargetHit"].astype(bool)

    bad = df["day_gain"].isna()
    if bad.any():
        print(f"Warning: {bad.sum()} row(s) had unparseable DayGain and will be dropped.", file=sys.stderr)
        df = df[~bad]

    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)

    if df.empty:
        print("No usable rows after loading/cleaning the CSV.", file=sys.stderr)
        sys.exit(1)

    return df


def current_streak_below(series: pd.Series, threshold: float) -> int:
    """Length of the most recent consecutive run where series < threshold,
    counting backward from the last row. 0 if the last row is not below."""
    streak = 0
    for v in series.iloc[::-1]:
        if pd.isna(v):
            break
        if v < threshold:
            streak += 1
        else:
            break
    return streak


def find_episodes(mask: pd.Series, dates: pd.Series):
    """Return list of (start_date, end_date, length) for consecutive True runs."""
    episodes = []
    run_start = None
    mask_list = mask.tolist()
    for i, v in enumerate(mask_list):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            episodes.append((dates.iloc[run_start].date(), dates.iloc[i - 1].date(), i - run_start))
            run_start = None
    if run_start is not None:
        episodes.append((dates.iloc[run_start].date(), dates.iloc[len(mask_list) - 1].date(), len(mask_list) - run_start))
    return episodes


def analyze(df: pd.DataFrame, window: int, alert_percentile: float,
            duration_threshold: float, duration_days: int):
    df = df.copy()
    df["target_hit_int"] = df["target_hit"].astype(int)

    df["trailing_hit_rate"] = (
        df["target_hit_int"].rolling(window, min_periods=window).mean() * 100.0
    )
    df["trailing_net"] = df["day_gain"].rolling(window, min_periods=window).sum()

    hist_hit = df["trailing_hit_rate"].dropna()
    hist_net = df["trailing_net"].dropna()

    if len(hist_hit) < 2:
        print(
            f"\nNot enough trading days ({len(df)}) for a {window}-day trailing "
            f"window comparison. Need at least {window + 1} days of data.",
            file=sys.stderr,
        )
        sys.exit(1)

    current_hit = hist_hit.iloc[-1]
    current_net = hist_net.iloc[-1]

    # compare current window against all PRIOR windows, so it's not
    # compared against itself
    prior_hit_dist = hist_hit.iloc[:-1]
    prior_net_dist = hist_net.iloc[:-1]

    hit_percentile = (prior_hit_dist < current_hit).mean() * 100.0
    net_percentile = (prior_net_dist < current_net).mean() * 100.0

    percentile_alert = (hit_percentile <= alert_percentile) or (net_percentile <= alert_percentile)

    # duration-based alert: how many consecutive trading days (ending today)
    # has the trailing hit rate stayed below duration_threshold
    streak = current_streak_below(df["trailing_hit_rate"], duration_threshold)
    duration_alert = streak >= duration_days

    # historical episodes at this duration threshold, for context
    below_mask = df["trailing_hit_rate"] < duration_threshold
    episodes = [e for e in find_episodes(below_mask, df["date"]) if e[2] >= duration_days]

    return df, {
        "current_hit_rate": current_hit,
        "current_net": current_net,
        "hit_percentile": hit_percentile,
        "net_percentile": net_percentile,
        "n_windows_in_history": len(prior_hit_dist),
        "as_of_date": df["date"].iloc[-1],
        "percentile_alert": percentile_alert,
        "current_streak": streak,
        "duration_alert": duration_alert,
        "duration_episodes": episodes,
        "alert": percentile_alert or duration_alert,
    }


def print_report(df, summary, window, alert_percentile, duration_threshold, duration_days):
    as_of = summary["as_of_date"].strftime("%Y-%m-%d")
    print("=" * 66)
    print(f"Trailing {window}-day regime monitor -- as of {as_of}")
    print("=" * 66)
    print(f"Total trading days loaded: {len(df)}")
    print(f"Rolling windows available for comparison: {summary['n_windows_in_history']}")
    print()
    print(f"Trailing {window}-day Target-Hit rate: {summary['current_hit_rate']:.1f}%"
          f"  (percentile vs. full history: {summary['hit_percentile']:.0f}th)")
    print(f"Trailing {window}-day realized net P/L: ${summary['current_net']:,.0f}"
          f"  (percentile vs. full history: {summary['net_percentile']:.0f}th)")
    print()

    # --- Percentile alert ---
    if summary["percentile_alert"]:
        print(f"** PERCENTILE ALERT: current window is at/below the {alert_percentile:.0f}th "
              f"percentile of all {window}-day windows in this dataset. **")
    else:
        print(f"Percentile check: above the {alert_percentile:.0f}th percentile threshold -- "
              f"not unusual relative to this dataset's own history.")

    # --- Duration alert ---
    streak = summary["current_streak"]
    if summary["duration_alert"]:
        print(f"** DURATION ALERT: trailing hit rate has been below {duration_threshold:.0f}% "
              f"for {streak} consecutive trading days (threshold: {duration_days}). **")
    elif streak > 0:
        print(f"Duration check: currently {streak} consecutive day(s) below {duration_threshold:.0f}% "
              f"(threshold to alert: {duration_days}) -- not yet sustained.")
    else:
        print(f"Duration check: not currently below {duration_threshold:.0f}%.")

    print()
    if summary["alert"]:
        print("This is the kind of reading your capital-preservation default "
              "(Target over Trail, or the half-capital fallback) is meant to react to.")
    print()

    if summary["duration_episodes"]:
        print(f"Historical episodes of {duration_days}+ consecutive days below {duration_threshold:.0f}%:")
        for start, end, length in summary["duration_episodes"]:
            print(f"  {start} to {end}  ({length} days)")
        print()

    tail = df.dropna(subset=["trailing_hit_rate"]).tail(10)
    print(f"Last {len(tail)} trailing-{window}-day readings:")
    print(f"{'Date':<12}{'TrailHit%':>12}{'TrailNet$':>14}")
    for _, r in tail.iterrows():
        print(f"{r['date'].strftime('%Y-%m-%d'):<12}{r['trailing_hit_rate']:>11.1f}%"
              f"{r['trailing_net']:>14,.0f}")
    print("=" * 66)


def make_plot(df, window, out_path, duration_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.dropna(subset=["trailing_hit_rate"])
    if plot_df.empty:
        print("Nothing to plot (not enough data for a full window).", file=sys.stderr)
        return

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(plot_df["date"], plot_df["trailing_hit_rate"], color="tab:blue", label="Trailing Target-Hit %")
    ax1.set_ylabel("Trailing Target-Hit %", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50% (reference)")
    ax1.axhline(duration_threshold, color="red", linewidth=0.8, linestyle=":",
                label=f"{duration_threshold:.0f}% (duration alert level)")

    ax2 = ax1.twinx()
    ax2.plot(plot_df["date"], plot_df["trailing_net"], color="tab:orange", label="Trailing Net $")
    ax2.set_ylabel("Trailing Net P/L ($)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax1.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Trailing {window}-day Target-Hit Rate & Net P/L")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--src", required=True, type=Path, help="Directory containing sessions.csv")
    ap.add_argument("--window", type=int, default=20, help="Trailing window length in trading days (default 20)")
    ap.add_argument("--alert-percentile", type=float, default=10.0,
                     help="Flag current window if at/below this percentile of history (default 10)")
    ap.add_argument("--duration-threshold", type=float, default=50.0,
                     help="Trailing hit-rate level for the duration alert (default 50)")
    ap.add_argument("--duration-days", type=int, default=10,
                     help="Consecutive trading days below --duration-threshold required to alert (default 10)")
    ap.add_argument("--strategy-col", default=None,
                     help="Column to filter on if one CSV contains multiple strategies (e.g. SourceFile)")
    ap.add_argument("--strategy", default=None,
                     help="Substring to match within --strategy-col (e.g. 'Trail')")
    ap.add_argument("--plot", type=Path, default=None, help="Optional filename to save a PNG chart (written under --src)")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional filename to dump the full daily/trailing table as CSV (written under --src)")
    args = ap.parse_args()

    df = load_csv(args.src, args.strategy_col, args.strategy)
    df, summary = analyze(df, args.window, args.alert_percentile, args.duration_threshold, args.duration_days)
    print_report(df, summary, args.window, args.alert_percentile, args.duration_threshold, args.duration_days)

    if args.plot:
        make_plot(df, args.window, args.src / args.plot, args.duration_threshold)

    if args.csv_out:
        out_path = args.src / args.csv_out
        df.to_csv(out_path, index=False)
        print(f"Saved full table to {out_path}")


if __name__ == "__main__":
    main()
