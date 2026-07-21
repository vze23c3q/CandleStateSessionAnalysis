r"""
analyze_regime.py - Version 1.0.1

Regime-conditioned session performance analysis for CandleState.

Answers the question: "Is the eval period (e.g. July) underperforming because
it was dealt adverse regime cards, or underperforming GIVEN its cards?"

Method:
  1. Split session days into BASELINE (before --eval-start) and EVAL (on/after).
  2. Compute per-day regime features from market daily analytics + Fear & Greed:
       - SlopeDeg : SlowSlopeDeg (or FastSlopeDeg with --fast-slope)
       - GapBps   : overnight gap, (Open / prior Close - 1) * 10000
       - FG       : CNN Fear & Greed value
  3. Freeze tertile boundaries on BASELINE days only (eval days cannot shift
     the buckets - honest out-of-sample break detection).
  4. Report:
       a. Regime composition: baseline vs eval bucket mix per factor
          ("what cards was each period dealt?")
       b. Within-bucket performance: baseline vs eval DayGain / hit rate
          per bucket ("how did each period play its cards?")
       c. Per-day eval table with same-bucket baseline z-scores
       d. Monte Carlo: simulate the eval period by drawing, for each eval
          day, a random baseline day from the SAME bucket. Percentile of
          actual eval Net within the simulated distribution is the headline:
          low percentile = underperforming its cards (possible break),
          mid percentile = normal result given the regime mix.
       e. Secondary: Slope x Gap grid (thin cells flagged, interpret with care)

Conventions:
  - Session P&L field is DayGain (session-level realized gain).
  - Win / hit definitions: TargetHit column; win = DayGain >= 0.
  - Tertile labels are relative to baseline: T1 (low) / T2 (mid) / T3 (high),
    with the frozen boundary values printed for interpretation.

USAGE (PowerShell - note backtick line continuation, no trailing spaces after it):
  python analyze_regime.py `
      --sessions   C:\Git\CandleStateSessionAnalysis\data\MACDTrail\output\sessions.csv `
      --analytics  C:\Git\CandleStateSessionAnalysis\data\analytics_SPY.json `
      --fear-greed C:\Git\CandleStateSessionAnalysis\data\fear_greed.csv `
      --eval-start 2026-07-01

USAGE (cmd.exe - caret line continuation):
  python C:\Git\CandleStateSessionAnalysis\src\analyze_regime.py ^
      --sessions   C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output\sessions.csv ^
      --analytics  C:\Git\CandleStateSessionAnalysis\data\analytics_SPY.json ^
      --fear-greed C:\Git\CandleStateSessionAnalysis\data\fear_greed.csv ^
      --eval-start 2026-07-01

  Optional:
      --fast-slope            use FastSlopeDeg instead of SlowSlopeDeg
      --sim-bucket slope      bucket factor for Monte Carlo: slope|gap|fg (default slope)
      --sims 10000            Monte Carlo iterations
      --min-bucket 5          min baseline days in a bucket before falling back to all-baseline draws
      --export PATH           write per-day merged table to CSV
      --seed 42               RNG seed for reproducible simulations
"""

VERSION = "1.0.1"

import argparse
import json
import sys

import numpy as np
import pandas as pd

FG_BANDS = [(-np.inf, 25, "ExtremeFear"), (25, 45, "Fear"),
            (45, 55, "Neutral"), (55, 75, "Greed"), (75, np.inf, "ExtremeGreed")]


# ---------------------------------------------------------------- loading

def load_sessions(path):
    df = pd.read_csv(path, parse_dates=["SessionStart"])
    df["Date"] = df["SessionStart"].dt.normalize()
    df["Win"] = df["DayGain"] >= 0
    df["TargetHit"] = df["TargetHit"].astype(bool)
    return df[["Date", "DayGain", "TargetHit", "Win"]].sort_values("Date").reset_index(drop=True)


def load_analytics(path, use_fast_slope):
    raw = json.load(open(path))
    df = pd.DataFrame(raw)
    df["Date"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df["GapBps"] = (df["Open"] / df["Close"].shift(1) - 1.0) * 10000.0
    slope_col = "FastSlopeDeg" if use_fast_slope else "SlowSlopeDeg"
    df["SlopeDeg"] = df[slope_col]
    return df[["Date", "SlopeDeg", "GapBps", "RSI", "Close"]]


def load_fear_greed(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns={"Value": "FG"})
    return df[["Date", "FG"]]


# ---------------------------------------------------------------- bucketing

def tertile_bounds(series):
    """Frozen tertile boundaries from baseline values (33.3 / 66.7 percentiles)."""
    s = series.dropna()
    return s.quantile(1 / 3), s.quantile(2 / 3)


def tertile_label(value, lo, hi):
    if pd.isna(value):
        return "NA"
    if value <= lo:
        return "T1"
    if value <= hi:
        return "T2"
    return "T3"


def fg_band(value):
    if pd.isna(value):
        return "NA"
    for lo, hi, name in FG_BANDS:
        if lo < value <= hi if lo != -np.inf else value <= hi:
            return name
    return "NA"


# ---------------------------------------------------------------- reporting

def composition_table(df, bucket_col, order):
    base = df[~df["IsEval"]][bucket_col].value_counts()
    ev = df[df["IsEval"]][bucket_col].value_counts()
    rows = []
    for b in order:
        nb, ne = int(base.get(b, 0)), int(ev.get(b, 0))
        rows.append({
            "Bucket": b,
            "BaseDays": nb,
            "BasePct": 100.0 * nb / max(base.sum(), 1),
            "EvalDays": ne,
            "EvalPct": 100.0 * ne / max(ev.sum(), 1),
        })
    return pd.DataFrame(rows)


def within_bucket_table(df, bucket_col, order, min_bucket):
    rows = []
    for b in order:
        base = df[(~df["IsEval"]) & (df[bucket_col] == b)]
        ev = df[(df["IsEval"]) & (df[bucket_col] == b)]
        if len(base) == 0 and len(ev) == 0:
            continue
        rows.append({
            "Bucket": b,
            "BaseN": len(base),
            "BaseMean$": base["DayGain"].mean(),
            "BaseMed$": base["DayGain"].median(),
            "BaseHit%": 100.0 * base["TargetHit"].mean() if len(base) else np.nan,
            "EvalN": len(ev),
            "EvalMean$": ev["DayGain"].mean() if len(ev) else np.nan,
            "EvalHit%": 100.0 * ev["TargetHit"].mean() if len(ev) else np.nan,
            "Thin": "*" if 0 < len(base) < min_bucket else "",
        })
    return pd.DataFrame(rows)


def per_day_eval_table(df, bucket_col):
    """Eval days with z-score vs same-bucket baseline DayGain."""
    base = df[~df["IsEval"]]
    stats = base.groupby(bucket_col)["DayGain"].agg(["mean", "std", "count"])
    rows = []
    for _, r in df[df["IsEval"]].iterrows():
        b = r[bucket_col]
        if b in stats.index and stats.loc[b, "count"] >= 2 and stats.loc[b, "std"] > 0:
            z = (r["DayGain"] - stats.loc[b, "mean"]) / stats.loc[b, "std"]
            exp = stats.loc[b, "mean"]
        else:
            z, exp = np.nan, np.nan
        rows.append({
            "Date": r["Date"].strftime("%Y-%m-%d"),
            "Bucket": b,
            "SlopeDeg": r["SlopeDeg"],
            "GapBps": r["GapBps"],
            "FG": r["FG"],
            "DayGain$": r["DayGain"],
            "Hit": "Y" if r["TargetHit"] else "-",
            "BucketExp$": exp,
            "Z": z,
        })
    return pd.DataFrame(rows)


def monte_carlo(df, bucket_col, sims, min_bucket, rng):
    """Simulate the eval period: for each eval day draw a baseline day from the
    same bucket (fallback: all baseline days if the bucket is thin/absent).
    Returns simulated Net and TargetHit-count distributions plus fallback count."""
    base = df[~df["IsEval"]]
    ev = df[df["IsEval"]]
    pools_gain, pools_hit, fallbacks = [], [], 0
    all_gain = base["DayGain"].to_numpy()
    all_hit = base["TargetHit"].to_numpy()
    for _, r in ev.iterrows():
        b = r[bucket_col]
        pool = base[base[bucket_col] == b]
        if len(pool) >= min_bucket:
            pools_gain.append(pool["DayGain"].to_numpy())
            pools_hit.append(pool["TargetHit"].to_numpy())
        else:
            pools_gain.append(all_gain)
            pools_hit.append(all_hit)
            fallbacks += 1
    n = len(pools_gain)
    sim_net = np.empty(sims)
    sim_hits = np.empty(sims)
    for i in range(sims):
        net = 0.0
        hits = 0
        for d in range(n):
            j = rng.integers(0, len(pools_gain[d]))
            net += pools_gain[d][j]
            hits += pools_hit[d][j]
        sim_net[i] = net
        sim_hits[i] = hits
    return sim_net, sim_hits, fallbacks


def pctile_of(value, dist):
    return 100.0 * (dist < value).mean()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=f"analyze_regime.py v{VERSION}")
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--analytics", required=True, help="daily analytics JSON (e.g. analytics_SPY.json)")
    ap.add_argument("--fear-greed", required=True)
    ap.add_argument("--eval-start", required=True, help="YYYY-MM-DD; days on/after are the eval period")
    ap.add_argument("--fast-slope", action="store_true")
    ap.add_argument("--sim-bucket", choices=["slope", "gap", "fg"], default="slope")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--min-bucket", type=int, default=5)
    ap.add_argument("--export", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.1f}")

    sessions = load_sessions(args.sessions)
    market = load_analytics(args.analytics, args.fast_slope)
    fg = load_fear_greed(args.fear_greed)

    df = sessions.merge(market, on="Date", how="left").merge(fg, on="Date", how="left")
    missing = df[df["SlopeDeg"].isna() | df["GapBps"].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} session day(s) missing market analytics: "
              f"{', '.join(missing['Date'].dt.strftime('%Y-%m-%d'))}", file=sys.stderr)

    eval_start = pd.Timestamp(args.eval_start)
    df["IsEval"] = df["Date"] >= eval_start
    base_mask = ~df["IsEval"]
    n_base, n_eval = int(base_mask.sum()), int(df["IsEval"].sum())
    if n_base < 30:
        print(f"WARNING: baseline has only {n_base} days; tertile boundaries will be noisy.", file=sys.stderr)
    if n_eval == 0:
        sys.exit("ERROR: no eval days on/after --eval-start.")

    # frozen boundaries from baseline only
    slope_lo, slope_hi = tertile_bounds(df.loc[base_mask, "SlopeDeg"])
    gap_lo, gap_hi = tertile_bounds(df.loc[base_mask, "GapBps"])
    df["SlopeBkt"] = df["SlopeDeg"].apply(lambda v: tertile_label(v, slope_lo, slope_hi))
    df["GapBkt"] = df["GapBps"].apply(lambda v: tertile_label(v, gap_lo, gap_hi))
    df["FGBkt"] = df["FG"].apply(fg_band)
    df["GridBkt"] = df["SlopeBkt"] + "/" + df["GapBkt"]

    slope_order = ["T1", "T2", "T3", "NA"]
    gap_order = ["T1", "T2", "T3", "NA"]
    fg_order = [b[2] for b in FG_BANDS] + ["NA"]
    grid_order = [f"{s}/{g}" for s in ["T1", "T2", "T3"] for g in ["T1", "T2", "T3"]]

    sim_bucket_col = {"slope": "SlopeBkt", "gap": "GapBkt", "fg": "FGBkt"}[args.sim_bucket]

    hdr = "=" * 78
    print(hdr)
    print(f"analyze_regime.py v{VERSION}")
    print(f"Baseline: {df.loc[base_mask,'Date'].min():%Y-%m-%d} .. {df.loc[base_mask,'Date'].max():%Y-%m-%d}  ({n_base} days)")
    print(f"Eval:     {df.loc[df['IsEval'],'Date'].min():%Y-%m-%d} .. {df.loc[df['IsEval'],'Date'].max():%Y-%m-%d}  ({n_eval} days)")
    print(f"Slope source: {'FastSlopeDeg' if args.fast_slope else 'SlowSlopeDeg'}")
    print(f"Frozen tertile bounds (baseline only):")
    print(f"  SlopeDeg: T1 <= {slope_lo:.3f} < T2 <= {slope_hi:.3f} < T3")
    print(f"  GapBps:   T1 <= {gap_lo:.1f} < T2 <= {gap_hi:.1f} < T3")
    print(hdr)

    # -- a. regime composition ------------------------------------------------
    print("\n[1] REGIME COMPOSITION - what cards was each period dealt?\n")
    for name, col, order in [("Trend slope tertile", "SlopeBkt", slope_order),
                             ("Overnight gap tertile", "GapBkt", gap_order),
                             ("Fear & Greed band", "FGBkt", fg_order)]:
        t = composition_table(df, col, order)
        t = t[(t["BaseDays"] > 0) | (t["EvalDays"] > 0)]
        print(f"  {name}:")
        print(t.to_string(index=False, formatters={
            "BasePct": "{:5.1f}%".format, "EvalPct": "{:5.1f}%".format}))
        print()

    # -- b. within-bucket performance ----------------------------------------
    print(f"[2] WITHIN-BUCKET PERFORMANCE - how did each period play its cards?")
    print(f"    ('*' = baseline bucket thinner than {args.min_bucket} days)\n")
    for name, col, order in [("Trend slope tertile", "SlopeBkt", slope_order),
                             ("Overnight gap tertile", "GapBkt", gap_order),
                             ("Fear & Greed band", "FGBkt", fg_order)]:
        t = within_bucket_table(df, col, order, args.min_bucket)
        print(f"  {name}:")
        print(t.to_string(index=False))
        print()

    # -- c. per-day eval table ------------------------------------------------
    print(f"[3] EVAL DAYS vs SAME-BUCKET BASELINE  (bucket factor: {args.sim_bucket})\n")
    t = per_day_eval_table(df, sim_bucket_col)
    print(t.to_string(index=False, formatters={"Z": "{:+.2f}".format}))
    zvals = t["Z"].dropna()
    if len(zvals):
        print(f"\n  Mean per-day z-score: {zvals.mean():+.2f}   "
              f"(negative = eval days running below their bucket's baseline)")

    # -- d. Monte Carlo -------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    sim_net, sim_hits, fallbacks = monte_carlo(df, sim_bucket_col, args.sims, args.min_bucket, rng)
    actual_net = df.loc[df["IsEval"], "DayGain"].sum()
    actual_hits = int(df.loc[df["IsEval"], "TargetHit"].sum())
    p_net = pctile_of(actual_net, sim_net)
    p_hit = pctile_of(actual_hits, sim_hits)
    lo5, med, hi95 = np.percentile(sim_net, [5, 50, 95])

    print(f"\n[4] MONTE CARLO - {args.sims:,} simulated eval periods, same-bucket draws "
          f"(factor: {args.sim_bucket}, {fallbacks} day(s) used all-baseline fallback)\n")
    print(f"  Simulated eval Net:  5th pct {lo5:>10,.0f}   median {med:>10,.0f}   95th pct {hi95:>10,.0f}")
    print(f"  ACTUAL eval Net:     {actual_net:>10,.0f}   -> percentile {p_net:.1f}")
    print(f"  ACTUAL TargetHits:   {actual_hits} of {n_eval}          -> percentile {p_hit:.1f}")
    print()
    if p_net < 5:
        verdict = ("BELOW the 5th percentile of regime-matched history. The eval period is\n"
                   "  underperforming even after accounting for its regime mix - treat this as\n"
                   "  evidence consistent with strategy decay. Cross-check [2] for which buckets.")
    elif p_net < 20:
        verdict = ("In the lower tail (5th-20th pct) of regime-matched history. Weak given its\n"
                   "  cards, but within the range a healthy strategy produces. Watch, don't act.")
    else:
        verdict = ("Within the normal range of regime-matched history. The result is explained\n"
                   "  by the regime cards dealt, not by decay in how the strategy played them.")
    print(f"  VERDICT: {verdict}")

    # -- e. thin grid ---------------------------------------------------------
    print(f"\n[5] SLOPE x GAP GRID (secondary; cells are thin - interpret with care)\n")
    t = within_bucket_table(df, "GridBkt", grid_order, args.min_bucket)
    print(t.to_string(index=False))

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nExported merged per-day table -> {args.export}")


if __name__ == "__main__":
    main()
