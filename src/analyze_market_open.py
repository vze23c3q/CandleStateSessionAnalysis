"""
analyze_market_open.py - v1.1.0

Focuses on the FIRST qualifying window of each session: when it arrives,
which side it favors (LONG basket = UPRO+TQQQ vs SHORT basket = SDS+QID),
and how both relate to the overnight gap. Also evaluates the "shut it
down at 11:00" rule: how many days present nothing by then, and what
supply (and optionally P/L) a shutdown forgoes on those days.

v1.1.0 - overnight gaps now come from index analytics files (official
         prior closes): SPY required, QQQ optional. Adds cross-index gap
         agreement (SPY vs QQQ gap sign) as an overnight divergence
         signal. SPY gap in index bps aligns with analyze_regime.py's
         GapBps tertiles.
v1.0.0 - initial version; gap derived from basket legs' 15:52 close.

A window qualifies when a basket rises >= --target-bps within --window
candles without adverse excursion worse than --mae-cap-bps before target
is first reached (same definition as analyze_trend_supply.py v1.1.x).

USAGE:
  python analyze_market_open.py `
      --signals   C:\\Git\\CandleStateSessionAnalysis\\data\\MACDFlip\\output\\trade_signals.csv `
      --spy       C:\\Git\\CandleStateSessionAnalysis\\data\\analytics_SPY.json `
      --qqq       C:\\Git\\CandleStateSessionAnalysis\\data\\analytics_QQQ.json `
      --out       C:\\Git\\CandleStateSessionAnalysis\\data\\MACDFlip\\output\\market_open.csv `
      --sessions  C:\\Git\\CandleStateSessionAnalysis\\data\\MACDFlip\\output\\sessions.csv `
      --window 15 --target-bps 51 --mae-cap-bps 30 --cutoff 11:00

NOTES:
  - CandleOpen is naive US/Eastern wall-clock (normalized upstream by
    parse_sessions.py). Loader hard-fails on out-of-session timestamps.
  - The analytics loader auto-detects field names (Date/Open/Close or a
    precomputed GapBps). If detection fails it exits listing the fields
    it found - send that list back and the loader gets a one-line fix.
"""

import argparse
import json
import os
from datetime import time

import numpy as np
import pandas as pd

VERSION = "1.1.0"

LONG_LEGS = ("UPRO", "TQQQ")
SHORT_LEGS = ("SDS", "QID")

SESSION_OPEN = time(9, 30)
SESSION_LAST = time(15, 54)

DATE_KEYS = ["Date", "SessionDate", "Timestamp", "Day", "TradeDate", "date", "day"]
OPEN_KEYS = ["Open", "DayOpen", "open"]
CLOSE_KEYS = ["Close", "DayClose", "close"]
GAP_KEYS = ["GapBps", "OvernightGapBps", "Gap"]
FLAT_BPS = 3.0  # |gap| below this counts as flat for agreement logic


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["CandleOpen", "Symbol", "Close"])
    df["CandleOpen"] = pd.to_datetime(df["CandleOpen"])
    df = df.drop_duplicates(subset=["Symbol", "CandleOpen"])
    df = df.sort_values("CandleOpen").reset_index(drop=True)
    df["Date"] = df["CandleOpen"].dt.date

    tt = df["CandleOpen"].dt.time
    bad = ((tt < SESSION_OPEN) | (tt > SESSION_LAST)).sum()
    if bad:
        raise SystemExit(f"TIMESTAMP ERROR: {bad} rows outside "
                         f"09:30-15:54 ET. Re-run parse_sessions.py or "
                         f"check for a new CandleOpen format.")
    return df


def _find_key(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_index_gaps(path: str, label: str) -> pd.Series:
    """Daily overnight gap in index bps from an analytics JSON file.
    Uses a precomputed gap field if present, else today's Open vs prior
    day's Close."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):  # records may sit under a top-level key
        for v in raw.values():
            if isinstance(v, list):
                raw = v
                break
    df = pd.json_normalize(raw)

    dk = _find_key(df.columns, DATE_KEYS)
    if dk is None:
        raise SystemExit(f"{label}: no date field found. "
                         f"Fields present: {sorted(df.columns)}")
    df["Date"] = pd.to_datetime(
        df[dk].astype("string").str.slice(0, 10)).dt.date
    df = df.sort_values("Date")

    gk = _find_key(df.columns, GAP_KEYS)
    if gk is not None:
        gap = pd.Series(df[gk].to_numpy(), index=df.Date, name=label)
        return gap

    ok, ck = (_find_key(df.columns, OPEN_KEYS),
              _find_key(df.columns, CLOSE_KEYS))
    if ok is None or ck is None:
        raise SystemExit(f"{label}: no gap field and no Open/Close pair. "
                         f"Fields present: {sorted(df.columns)}")
    gap = (df[ok] / df[ck].shift(1) - 1.0) * 10_000
    return pd.Series(gap.to_numpy(), index=df.Date, name=label)


def build_basket(sig: pd.DataFrame, legs: tuple) -> pd.DataFrame:
    a = sig[sig.Symbol == legs[0]][["CandleOpen", "Date", "Close"]] \
        .rename(columns={"Close": "C0"})
    b = sig[sig.Symbol == legs[1]][["CandleOpen", "Close"]] \
        .rename(columns={"Close": "C1"})
    m = pd.merge(a, b, on="CandleOpen", how="inner").sort_values("CandleOpen")
    parts = []
    for _, day in m.groupby("Date", sort=True):
        day = day.copy()
        day["Basket"] = (day.C0 / day.C0.iloc[0] + day.C1 / day.C1.iloc[0]) / 2
        parts.append(day)
    return pd.concat(parts, ignore_index=True)


def first_window(day, window, target_bps, mae_cap_bps):
    """Earliest qualifying window in one session's basket, else None."""
    bk = day.Basket.to_numpy()
    ts = day.CandleOpen.to_numpy()
    for i in range(len(bk) - window + 1):
        path = (bk[i:i + window] / bk[i] - 1.0) * 10_000
        if path.max() < target_bps:
            continue
        hit = int(np.argmax(path >= target_bps))
        mae = float(path[:hit + 1].min()) if hit > 0 else 0.0
        if mae_cap_bps is not None and mae <= -mae_cap_bps:
            continue
        return {"start": pd.Timestamp(ts[i]), "peak_bps": float(path.max()),
                "mae_bps": mae, "candles_to_hit": hit}
    return None


def count_windows_after(day, after, window, target_bps, mae_cap_bps):
    day = day[day.CandleOpen >= after]
    bk = day.Basket.to_numpy()
    count, i = 0, 0
    while i <= len(bk) - window:
        path = (bk[i:i + window] / bk[i] - 1.0) * 10_000
        if path.max() >= target_bps:
            hit = int(np.argmax(path >= target_bps))
            mae = float(path[:hit + 1].min()) if hit > 0 else 0.0
            if mae_cap_bps is None or mae > -mae_cap_bps:
                count += 1
                i += window
                continue
        i += 1
    return count


def gap_agreement(spy, qqq):
    if pd.isna(spy) or pd.isna(qqq):
        return None
    if abs(spy) < FLAT_BPS and abs(qqq) < FLAT_BPS:
        return "Flat"
    if abs(spy) < FLAT_BPS or abs(qqq) < FLAT_BPS:
        return "Mixed"
    return "Agree" if np.sign(spy) == np.sign(qqq) else "Disagree"


def minutes_after_open(t: pd.Timestamp) -> float:
    return (t.hour - 9) * 60 + t.minute - 30


def main():
    p = argparse.ArgumentParser(description="First-window / open analysis")
    p.add_argument("--signals", required=True)
    p.add_argument("--spy", required=True, help="analytics_SPY.json")
    p.add_argument("--qqq", default=None, help="analytics_QQQ.json")
    p.add_argument("--out", required=True)
    p.add_argument("--sessions", default=None,
                   help="Optional sessions.csv for DayGain join")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--target-bps", type=float, default=51.0)
    p.add_argument("--mae-cap-bps", type=float, default=None)
    p.add_argument("--cutoff", default="11:00")
    args = p.parse_args()

    ch, cm = (int(x) for x in args.cutoff.split(":"))
    cutoff_t = time(ch, cm)

    print("=" * 78)
    print(f"analyze_market_open.py v{VERSION}")
    print(f"Signals: {os.path.abspath(args.signals)}")
    print(f"SPY analytics: {os.path.abspath(args.spy)}")
    if args.qqq:
        print(f"QQQ analytics: {os.path.abspath(args.qqq)}")
    print(f"Window: {args.window}   Target: {args.target_bps} bps   "
          f"MAE cap: {args.mae_cap_bps}   Cutoff: {args.cutoff}")
    print("=" * 78)

    sig = load_signals(args.signals)
    baskets = {"LONG": build_basket(sig, LONG_LEGS),
               "SHORT": build_basket(sig, SHORT_LEGS)}
    spy_gap = load_index_gaps(args.spy, "SpyGapBps")
    qqq_gap = (load_index_gaps(args.qqq, "QqqGapBps")
               if args.qqq else pd.Series(dtype=float, name="QqqGapBps"))

    rows = []
    for date in sorted(sig.Date.unique()):
        firsts = {}
        for side, bsk in baskets.items():
            day = bsk[bsk.Date == date]
            if len(day) >= args.window:
                w = first_window(day, args.window, args.target_bps,
                                 args.mae_cap_bps)
                if w:
                    firsts[side] = w
        sg = spy_gap.get(date, np.nan)
        qg = qqq_gap.get(date, np.nan)
        row = {"Date": date, "SpyGapBps": sg, "QqqGapBps": qg,
               "GapAgreement": gap_agreement(sg, qg),
               "FirstSide": None, "FirstTime": None, "FirstMin": np.nan,
               "PeakBps": np.nan, "CandlesToHit": np.nan, "MAEBps": np.nan,
               "ByCutoff": False, "WindowsAfterCutoff": 0}
        if firsts:
            side = min(firsts, key=lambda s: firsts[s]["start"])
            w = firsts[side]
            if len(firsts) == 2 and \
                    firsts["LONG"]["start"] == firsts["SHORT"]["start"]:
                side = "TIE"
            row.update({"FirstSide": side, "FirstTime": w["start"].time(),
                        "FirstMin": minutes_after_open(w["start"]),
                        "PeakBps": w["peak_bps"],
                        "CandlesToHit": w["candles_to_hit"],
                        "MAEBps": w["mae_bps"],
                        "ByCutoff": w["start"].time() < cutoff_t})
        if not row["ByCutoff"]:
            cut_ts = pd.Timestamp(f"{date} {args.cutoff}")
            row["WindowsAfterCutoff"] = sum(
                count_windows_after(b[b.Date == date], cut_ts, args.window,
                                    args.target_bps, args.mae_cap_bps)
                for b in baskets.values())
        rows.append(row)

    df = pd.DataFrame(rows)
    nd = len(df)

    print(f"\n[1] FIRST-WINDOW TIMING  ({nd} sessions)")
    has = df.FirstSide.notna()
    print(f"  Days with any qualifying window: {has.sum()} ({has.mean():.0%})")
    for label, mins in [("by 10:00", 30), ("by 10:30", 60),
                        ("by 11:00", 90), ("by 12:00", 150)]:
        n = (df.FirstMin <= mins).sum()
        print(f"  First window {label}: {n:4d}  ({n / nd:.0%} of days)")
    fm = df.FirstMin.dropna()
    print(f"  First-window minute after open: median {fm.median():.0f}, "
          f"p75 {fm.quantile(.75):.0f}, p90 {fm.quantile(.90):.0f}")

    print("\n[2] FIRST-WINDOW DIRECTION")
    print(df.FirstSide.value_counts(dropna=False).to_string())

    print("\n[3] GAP vs FIRST WINDOW  (index bps, official closes)")
    g = df.dropna(subset=["SpyGapBps"]).copy()
    g["GapBucket"] = pd.qcut(g.SpyGapBps, 3,
                             labels=["T1_dn", "T2_flat", "T3_up"])
    print("  SPY gap tertile bounds:",
          [round(x, 1) for x in g.SpyGapBps.quantile([1/3, 2/3])])
    print("\n  P(first side | SPY gap tertile):")
    print(pd.crosstab(g.GapBucket, g.FirstSide.fillna("NONE"),
                      normalize="index").round(2).to_string())
    print(f"\n  Mean SPY gap, LONG-first days:  "
          f"{g[g.FirstSide == 'LONG'].SpyGapBps.mean():+.1f} bps")
    print(f"  Mean SPY gap, SHORT-first days: "
          f"{g[g.FirstSide == 'SHORT'].SpyGapBps.mean():+.1f} bps")
    both = g.dropna(subset=["FirstMin"])
    print(f"  Spearman |SPY gap| vs first-window minute: "
          f"{both.SpyGapBps.abs().rank().corr(both.FirstMin.rank()):+.2f}")

    
    if args.qqq:
        ga = df.dropna(subset=["GapAgreement"])
        print("\n  P(first side | SPY/QQQ gap agreement):")
        print(pd.crosstab(ga.GapAgreement, ga.FirstSide.fillna("NONE"),
                          normalize="index").round(2).to_string())
        print("  Days per agreement class:")
        print(ga.GapAgreement.value_counts().to_string())

    print(f"\n[4] SHUTDOWN-AT-{args.cutoff} EVALUATION")
    no_by = df[~df.ByCutoff]
    print(f"  Days with no window by {args.cutoff}: {len(no_by)} "
          f"({len(no_by) / nd:.0%})")
    late = no_by[no_by.WindowsAfterCutoff > 0]
    print(f"  ...of which later supply appeared: {len(late)} "
          f"({len(late) / max(len(no_by), 1):.0%} of shutdown days, "
          f"{late.WindowsAfterCutoff.sum()} windows forgone)")
    if args.sessions:
        ses = pd.read_csv(args.sessions, usecols=["SessionStart", "DayGain"])
        ses["Date"] = pd.to_datetime(
            ses.SessionStart.astype("string").str.slice(0, 19)).dt.date
        m = df.merge(ses[["Date", "DayGain"]], on="Date", how="left")
        for flag, name in [(True, f"window by {args.cutoff}"),
                           (False, "no window (shutdown days)")]:
            sub = m[m.ByCutoff == flag]
            print(f"  Actual DayGain, {name}: n={len(sub)}, "
                  f"mean {sub.DayGain.mean():+.1f}, "
                  f"median {sub.DayGain.median():+.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nPer-day first-window data written: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
