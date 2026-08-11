"""
analyze_opening_drive.py - v1.0.1

v1.0.1 - baskets normalize to first candle OPEN (v1.0.0 anchored to
         first close, zeroing the N=1 signal and lagging N=2/3 by a
         candle); tie-proof magnitude tertiles.
v1.0.0 - initial version.

Tests the opening-drive playbook: read the direction of the first N
candles after 09:30, take one shot in that direction at full deployment,
ride to the day target with a tight trailing stop if wrong.

Three questions, three sections:
  [1] READABILITY - how often does the first-N-candle direction match the
      day's first qualifying window side (per analyze_market_open.py
      definitions)? Reported for N=1..3, and by signal-magnitude tertile
      (strong opens may be more readable -> graded size input).
  [2] SIMULATION GRID - one trade per day: signal read at close of candle
      N, enter the matching basket (up -> UPRO+TQQQ, down -> SDS+QID),
      exit on target (+--target-bps), trailing stop (peak - trail bps,
      which doubles as the initial stop), or time stop (--max-hold
      candles). Grid over N x trail, one-shot and flip-once variants
      (flip: after a stop-out, take the opposite side once, same rules).
  [3] DETAIL - for --detail-n/--detail-trail: first-half vs second-half
      stability split, monthly net, min-signal dead-zone sensitivity.

Exits are evaluated on candle CLOSES (basket H/L is not well-defined
from leg closes) - conservative: real fills on target/stop would be at
least as good intra-candle.

$ conversion: --capital * 0.98 deployed, so $/bps = capital*0.98/10000
(default $4.90/bps; 51 bps = ~$250, trail 15 = ~-$74).

USAGE:
  python C:\\Git\\CandleStateSessionAnalysis\\src\\analyze_opening_drive.py `
      --signals C:\\Git\\CandleStateSessionAnalysis\\data\\MACDFlip\\output\\trade_signals.csv `
      --out     C:\\Git\\CandleStateSessionAnalysis\\data\\MACDFlip\\output\\opening_drive.csv `
      --target-bps 51 --max-hold 15 --detail-n 2 --detail-trail 15

NOTES:
  - CandleOpen is naive US/Eastern wall-clock (normalized upstream by
    parse_sessions.py). Loader hard-fails on out-of-session timestamps.
  - OVERFITTING GUARD: the grid exists to find a PLATEAU, not a peak.
    Pick parameters where neighboring cells agree; distrust any cell
    that beats its neighbors by a lot. Confirm on the [3] halves split.
"""

import argparse
import os
from datetime import time

import numpy as np
import pandas as pd

VERSION = "1.0.1"

LONG_LEGS = ("UPRO", "TQQQ")
SHORT_LEGS = ("SDS", "QID")

SESSION_OPEN = time(9, 30)
SESSION_LAST = time(15, 54)

N_CHOICES = (1, 2, 3)
TRAIL_CHOICES = (10.0, 15.0, 20.0, 25.0, 30.0)


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path,
                     usecols=["CandleOpen", "Symbol", "Open", "Close"])
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


def build_basket(sig: pd.DataFrame, legs: tuple) -> dict:
    """{date: np.array of basket levels, normalized to first candle OPEN
    so candle 1's own move carries into the N=1 signal}."""
    a = sig[sig.Symbol == legs[0]][["CandleOpen", "Date", "Open", "Close"]] \
        .rename(columns={"Open": "O0", "Close": "C0"})
    b = sig[sig.Symbol == legs[1]][["CandleOpen", "Open", "Close"]] \
        .rename(columns={"Open": "O1", "Close": "C1"})
    m = pd.merge(a, b, on="CandleOpen", how="inner").sort_values("CandleOpen")
    out = {}
    for date, day in m.groupby("Date", sort=True):
        bk = (day.C0 / day.O0.iloc[0] + day.C1 / day.O1.iloc[0]).to_numpy() / 2
        out[date] = bk
    return out


def first_window_side(long_bk, short_bk, window, target_bps, mae_cap_bps):
    """Side (LONG/SHORT/None) of the day's earliest qualifying window."""
    best_i, best_side = None, None
    for side, bk in (("LONG", long_bk), ("SHORT", short_bk)):
        n = len(bk)
        for i in range(n - window + 1):
            path = (bk[i:i + window] / bk[i] - 1.0) * 10_000
            if path.max() < target_bps:
                continue
            hit = int(np.argmax(path >= target_bps))
            mae = float(path[:hit + 1].min()) if hit > 0 else 0.0
            if mae_cap_bps is not None and mae <= -mae_cap_bps:
                continue
            if best_i is None or i < best_i:
                best_i, best_side = i, side
            break
    return best_side


def run_trade(bk, entry_idx, target_bps, trail_bps, max_hold):
    """One trade on basket `bk` entered at close of candle entry_idx.
    Returns (result_bps, exit_reason, bars_held)."""
    entry = bk[entry_idx]
    peak = 0.0
    last = min(entry_idx + max_hold, len(bk) - 1)
    for j in range(entry_idx + 1, last + 1):
        move = (bk[j] / entry - 1.0) * 10_000
        if move >= target_bps:
            return target_bps, "target", j - entry_idx
        peak = max(peak, move)
        if move <= peak - trail_bps:
            return move, "stop", j - entry_idx
    return (bk[last] / entry - 1.0) * 10_000, "time", last - entry_idx


def simulate_day(long_bk, short_bk, n, trail_bps, target_bps, max_hold,
                 min_signal_bps, flip):
    """Returns dict for one day's play, or None if untradeable."""
    need = n + 2  # signal candles + at least one bar to trade
    if len(long_bk) < need or len(short_bk) < need:
        return None
    # basket is normalized to the 09:30 open (=1.0), so the signal is
    # candle N's close vs the session open - candle 1's move included
    signal_bps = (long_bk[n - 1] - 1.0) * 10_000
    # direction from first N candles of the LONG (market-direction) basket
    if abs(signal_bps) < min_signal_bps:
        return {"traded": False, "signal_bps": signal_bps}
    direction = "LONG" if signal_bps >= 0 else "SHORT"
    bk = long_bk if direction == "LONG" else short_bk
    res, why, held = run_trade(bk, n - 1, target_bps, trail_bps, max_hold)
    day = {"traded": True, "signal_bps": signal_bps, "direction": direction,
           "trade1_bps": res, "exit1": why, "held1": held,
           "flip_bps": 0.0, "exit2": None}
    if flip and why == "stop":
        flip_dir = "SHORT" if direction == "LONG" else "LONG"
        fbk = long_bk if flip_dir == "LONG" else short_bk
        idx = n - 1 + held
        if idx + 1 < len(fbk):
            res2, why2, _ = run_trade(fbk, idx, target_bps, trail_bps,
                                      max_hold)
            day["flip_bps"] = res2
            day["exit2"] = why2
    day["total_bps"] = day["trade1_bps"] + day["flip_bps"]
    return day


def main():
    p = argparse.ArgumentParser(description="Opening drive analysis")
    p.add_argument("--signals", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-bps", type=float, default=51.0)
    p.add_argument("--mae-cap-bps", type=float, default=30.0,
                   help="Only used for FirstSide labeling in [1]")
    p.add_argument("--window", type=int, default=15,
                   help="Only used for FirstSide labeling in [1]")
    p.add_argument("--max-hold", type=int, default=15)
    p.add_argument("--capital", type=float, default=50_000.0)
    p.add_argument("--min-signal-bps", type=float, default=0.0)
    p.add_argument("--detail-n", type=int, default=2)
    p.add_argument("--detail-trail", type=float, default=15.0)
    args = p.parse_args()

    dollars_per_bps = args.capital * 0.98 / 10_000

    print("=" * 78)
    print(f"analyze_opening_drive.py v{VERSION}")
    print(f"Signals: {os.path.abspath(args.signals)}")
    print(f"Target: {args.target_bps} bps   Max hold: {args.max_hold}"
          f"   $/bps: {dollars_per_bps:.2f}"
          f"   Min signal: {args.min_signal_bps} bps")
    print("=" * 78)

    sig = load_signals(args.signals)
    long_b = build_basket(sig, LONG_LEGS)
    short_b = build_basket(sig, SHORT_LEGS)
    dates = sorted(set(long_b) & set(short_b))

    firstside = {d: first_window_side(long_b[d], short_b[d], args.window,
                                      args.target_bps, args.mae_cap_bps)
                 for d in dates}

    print(f"\n[1] OPEN READABILITY  ({len(dates)} sessions)")
    for n in N_CHOICES:
        recs = []
        for d in dates:
            if len(long_b[d]) < n or firstside[d] is None:
                continue
            s = (long_b[d][n - 1] - 1.0) * 10_000
            pred = "LONG" if s >= 0 else "SHORT"
            recs.append((abs(s), pred == firstside[d]))
        df1 = pd.DataFrame(recs, columns=["mag", "correct"])
        acc = df1.correct.mean()
        print(f"  N={n} (decide {9}:{30 + 2 * n:02d}): "
              f"accuracy {acc:.0%}  (n={len(df1)})")
        df1["magT"] = pd.qcut(df1.mag.rank(method="first"), 3,
                              labels=["weak", "mid", "strong"])
        by = df1.groupby("magT", observed=True).correct.mean()
        print("        by |signal| tertile: "
              + "  ".join(f"{k}={v:.0%}" for k, v in by.items()))

    print("\n[2] SIMULATION GRID  (mean $/day | target-hit % | stop %)")
    for flip in (False, True):
        print(f"\n  {'FLIP-ONCE after stop' if flip else 'ONE-SHOT'}:")
        header = "  N\\trail " + "".join(f"{t:>18.0f}" for t in TRAIL_CHOICES)
        print(header)
        for n in N_CHOICES:
            cells = []
            for t in TRAIL_CHOICES:
                days = [simulate_day(long_b[d], short_b[d], n, t,
                                     args.target_bps, args.max_hold,
                                     args.min_signal_bps, flip)
                        for d in dates]
                days = [x for x in days if x and x["traded"]]
                tot = np.array([x["total_bps"] for x in days])
                hit = np.mean([x["exit1"] == "target" or
                               x["exit2"] == "target" for x in days])
                stp = np.mean([x["exit1"] == "stop" for x in days])
                cells.append(f"{tot.mean() * dollars_per_bps:+7.1f}"
                             f" {hit:>4.0%} {stp:>4.0%}")
            print(f"  {n:>7} " + " ".join(f"{c:>18}" for c in cells))

    n, t = args.detail_n, args.detail_trail
    print(f"\n[3] DETAIL  N={n}, trail={t:.0f} bps, one-shot")
    rows = []
    for d in dates:
        r = simulate_day(long_b[d], short_b[d], n, t, args.target_bps,
                         args.max_hold, args.min_signal_bps, False)
        if r and r["traded"]:
            r["Date"] = d
            r["FirstSide"] = firstside[d]
            r["Correct"] = (firstside[d] == r["direction"]
                            if firstside[d] else None)
            rows.append(r)
    det = pd.DataFrame(rows)
    det["Dollars"] = det.total_bps * dollars_per_bps
    half = len(det) // 2
    for name, sub in [("first half", det.iloc[:half]),
                      ("second half", det.iloc[half:])]:
        print(f"  {name}: mean {sub.Dollars.mean():+.1f}/day, "
              f"median {sub.Dollars.median():+.1f}, "
              f"target {np.mean(sub.exit1 == 'target'):.0%}, "
              f"total {sub.Dollars.sum():+,.0f}")
    det["Month"] = pd.to_datetime(det.Date.astype(str)).dt.to_period("M")
    print("\n  Monthly net $:")
    print(det.groupby("Month").Dollars.sum().round(0).to_string())
    print("\n  Min-signal sensitivity (one-shot, same N/trail):")
    for ms in (0.0, 5.0, 10.0):
        days = [simulate_day(long_b[d], short_b[d], n, t, args.target_bps,
                             args.max_hold, ms, False) for d in dates]
        traded = [x for x in days if x and x["traded"]]
        tot = np.array([x["total_bps"] for x in traded])
        print(f"    min-signal {ms:>4.0f}: traded {len(traded)}/{len(dates)}"
              f" days, mean {tot.mean() * dollars_per_bps:+.1f}/day,"
              f" total {tot.sum() * dollars_per_bps:+,.0f}")

    out_cols = ["Date", "signal_bps", "direction", "FirstSide", "Correct",
                "trade1_bps", "exit1", "held1", "flip_bps", "exit2",
                "total_bps", "Dollars"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    det[out_cols].to_csv(args.out, index=False)
    print(f"\nPer-day results written: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
