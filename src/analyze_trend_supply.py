"""
analyze_trend_supply.py - v1.1.0

Scans trade_signals price history for "qualifying windows": 15-candle
(30 min) stretches where a two-leg basket moved enough to hit the day
target at full deployment. Separates the SUPPLY question (was the move
there?) from the DETECTION question (did MACD catch it?).

v1.1.0 - load from trade_signals CSV (engine's own per-candle output)
         instead of raw candle files; dedupe on (Symbol, CandleOpen) so
         multi-strategy exports are safe; added per-day TwoSided flag
         (qualifying supply in BOTH directions = 30-minute-scale whipsaw).
v1.0.0 - initial version against raw candle CSVs.

Baskets (equal-weight cumulative return, normalized to session start):
  LONG  basket = UPRO + TQQQ     SHORT basket = SDS + QID

A window qualifies when the basket rises >= --target-bps within --window
candles, without adverse excursion worse than --mae-cap-bps before target
is first reached. Windows are non-overlapping. SHORT-side near-miss band
[--nearmiss-lo-bps, target) counts supply a -3x inverse pairing would
capture that the current -2x pairing forgoes.

USAGE:
  python analyze_trend_supply.py `
      --signals C:\\Git\\CandleStateSessionAnalysis\\data\\Target\\output\\trade_signals.csv `
      --out     C:\\Git\\CandleStateSessionAnalysis\\data\\Target\\output\\trend_supply.csv `
      --window 15 --target-bps 51 --mae-cap-bps 30

NOTES:
  - trade_signals timestamps carry offsets; dates/times are used as-is
    (already US/Eastern per engine output).
  - Only CandleOpen, Symbol, Close are read; loads fast even on the full
    multi-strategy export. Convert to Parquet locally for repeat runs.
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

VERSION = "1.2.0"

LONG_LEGS = ("UPRO", "TQQQ")
SHORT_LEGS = ("SDS", "QID")
BOTH_LEGS_FRACTION = 0.40


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["CandleOpen", "Symbol", "Close"])
    df["CandleOpen"] = pd.to_datetime(df["CandleOpen"])
    df = df.drop_duplicates(subset=["Symbol", "CandleOpen"])
    df = df.sort_values("CandleOpen").reset_index(drop=True)
    df["Date"] = df["CandleOpen"].dt.date
    return df

def build_basket(sig: pd.DataFrame, legs: tuple) -> pd.DataFrame:
    a = sig[sig.Symbol == legs[0]][["CandleOpen", "Date", "Close"]] \
        .rename(columns={"Close": "C0"})
    b = sig[sig.Symbol == legs[1]][["CandleOpen", "Close"]] \
        .rename(columns={"Close": "C1"})
    m = pd.merge(a, b, on="CandleOpen", how="inner").sort_values("CandleOpen")
    parts = []
    for _, day in m.groupby("Date", sort=True):
        day = day.copy()
        r0 = day.C0 / day.C0.iloc[0]
        r1 = day.C1 / day.C1.iloc[0]
        day["Basket"] = (r0 + r1) / 2.0
        day["L0"] = (r0 - 1.0) * 10_000
        day["L1"] = (r1 - 1.0) * 10_000
        parts.append(day[["CandleOpen", "Date", "Basket", "L0", "L1"]])
    return pd.concat(parts, ignore_index=True)


def scan_session(day, window, target_bps, mae_cap_bps, nearmiss_lo_bps):
    bk = day.Basket.to_numpy()
    l0 = day.L0.to_numpy()
    l1 = day.L1.to_numpy()
    ts = day.CandleOpen.to_numpy()
    n = len(bk)
    wins, nms = [], []
    i = 0
    while i <= n - window:
        path = (bk[i:i + window] / bk[i] - 1.0) * 10_000
        peak = path.max()
        if peak >= target_bps:
            hit = int(np.argmax(path >= target_bps))
            mae = float(path[:hit + 1].min()) if hit > 0 else 0.0
            if mae_cap_bps is None or mae > -mae_cap_bps:
                a = l0[i + hit] - l0[i]
                b = l1[i + hit] - l1[i]
                both = (a >= BOTH_LEGS_FRACTION * target_bps and
                        b >= BOTH_LEGS_FRACTION * target_bps)
                wins.append({"start": pd.Timestamp(ts[i]),
                             "peak_bps": float(peak), "mae_bps": mae,
                             "candles_to_hit": hit,
                             "tag": "both" if both else "single"})
                i += window
                continue
        elif nearmiss_lo_bps is not None and peak >= nearmiss_lo_bps:
            nms.append(pd.Timestamp(ts[i]))
            i += window
            continue
        i += 1
    return wins, nms


def halfhour_bin(t: pd.Timestamp) -> str:
    return f"{t.hour:02d}:{0 if t.minute < 30 else 30:02d}"


def main():
    p = argparse.ArgumentParser(description="Trend supply scanner")
    p.add_argument("--signals", required=True, help="trade_signals CSV")
    p.add_argument("--out", required=True, help="Per-day CSV output path")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--target-bps", type=float, default=51.0)
    p.add_argument("--mae-cap-bps", type=float, default=None)
    p.add_argument("--nearmiss-lo-bps", type=float, default=34.0)
    args = p.parse_args()

    print("=" * 78)
    print(f"analyze_trend_supply.py v{VERSION}")
    print(f"Signals: {os.path.abspath(args.signals)}")
    print(f"Window: {args.window} candles   Target: {args.target_bps} bps"
          f"   MAE cap: {args.mae_cap_bps}"
          f"   Near-miss lo: {args.nearmiss_lo_bps}")
    print(f"LONG basket: {'+'.join(LONG_LEGS)}   "
          f"SHORT basket: {'+'.join(SHORT_LEGS)}")
    print("=" * 78)

    sig = load_signals(args.signals)
    per_day = defaultdict(lambda: {"L": 0, "S": 0, "first": None,
                                   "both": 0, "single": 0, "nm": 0})
    tod = {"LONG": Counter(), "SHORT": Counter()}
    mae_pool = {"LONG": [], "SHORT": []}
    hit_pool = {"LONG": [], "SHORT": []}

    for side, legs, nm_lo in [("LONG", LONG_LEGS, None),
                              ("SHORT", SHORT_LEGS, args.nearmiss_lo_bps)]:
        basket = build_basket(sig, legs)
        for date, day in basket.groupby("Date", sort=True):
            wins, nms = scan_session(day, args.window, args.target_bps,
                                     args.mae_cap_bps, nm_lo)
            r = per_day[date]
            r["L" if side == "LONG" else "S"] += len(wins)
            r["nm"] += len(nms)
            for w in wins:
                t = w["start"]
                tod[side][halfhour_bin(t)] += 1
                mae_pool[side].append(w["mae_bps"])
                hit_pool[side].append(w["candles_to_hit"])
                r["both" if w["tag"] == "both" else "single"] += 1
                if r["first"] is None or t.time() < r["first"]:
                    r["first"] = t.time()

    dates = sorted(per_day)
    nd = len(dates)
    tl = sum(per_day[d]["L"] for d in dates)
    tsh = sum(per_day[d]["S"] for d in dates)
    zero = sum(1 for d in dates if per_day[d]["L"] + per_day[d]["S"] == 0)
    two = sum(1 for d in dates if per_day[d]["L"] > 0 and per_day[d]["S"] > 0)
    tb = sum(per_day[d]["both"] for d in dates)
    tsg = sum(per_day[d]["single"] for d in dates)
    tnm = sum(per_day[d]["nm"] for d in dates)

    print(f"\n[1] SUPPLY SUMMARY  ({nd} sessions)")
    print(f"  LONG windows:  {tl:5d}  ({tl / nd:.2f}/day)")
    print(f"  SHORT windows: {tsh:5d}  ({tsh / nd:.2f}/day)")
    print(f"  Zero-supply days: {zero} ({zero / nd:.0%})"
          f"   Two-sided days: {two} ({two / nd:.0%})  <- whipsaw signature")
    print(f"  Window tags: both-legs {tb}, single-leg {tsg}")
    print(f"  SHORT near-miss [{args.nearmiss_lo_bps:.0f}"
          f"..{args.target_bps:.0f}): {tnm}"
          f"  <- forgone by -2x vs -3x inverse pairing")

    print("\n[2] TIME-OF-DAY  (window starts, half-hour bins)")
    print(f"  {'Bin':>6} {'LONG':>6} {'SHORT':>6}")
    for b in sorted(set(tod["LONG"]) | set(tod["SHORT"])):
        print(f"  {b:>6} {tod['LONG'][b]:>6} {tod['SHORT'][b]:>6}")

    print("\n[3] WINDOW CHARACTER")
    for side in ("LONG", "SHORT"):
        if not mae_pool[side]:
            print(f"  {side}: no qualifying windows")
            continue
        m = np.array(mae_pool[side])
        h = np.array(hit_pool[side])
        print(f"  {side}: MAE median {np.median(m):+.1f} bps,"
              f" p10 {np.percentile(m, 10):+.1f} bps |"
              f" candles-to-hit median {np.median(h):.0f}")

    rows = [{"Date": d,
             "LongWindows": per_day[d]["L"],
             "ShortWindows": per_day[d]["S"],
             "TwoSided": per_day[d]["L"] > 0 and per_day[d]["S"] > 0,
             "FirstWindowTime": per_day[d]["first"],
             "BothLegTag": per_day[d]["both"],
             "SingleLegTag": per_day[d]["single"],
             "ShortNearMiss": per_day[d]["nm"]} for d in dates]
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nPer-day supply written: {os.path.abspath(args.out)}")
    print("Join on Date against analyze_regime.py bucket labels for the"
          " supply-by-regime table.")


    sup = pd.read_csv(r"C:\Git\CandleStateSessionAnalysis\data\Target\output\trend_supply.csv",
                  parse_dates=["Date"])

    # 1. The original mystery: was Jun-Aug 2025 a supply famine or a detection failure?
    print(sup.groupby(pd.Grouper(key="Date", freq="ME"))
            [["LongWindows", "ShortWindows", "TwoSided"]].mean().round(2))

    # 2. The mechanism: does two-sidedness explain the P/L?
    ses = pd.read_csv(r"C:\Git\CandleStateSessionAnalysis\data\Target\output\sessions.csv",
                    parse_dates=["SessionStart"])
    ses["Date"] = ses["SessionStart"].dt.normalize()
    j = sup.merge(ses[["Date", "DayGain", "TargetHit"]], on="Date")
    print(j.groupby("TwoSided")["DayGain"].agg(["count", "mean", "median"]).round(1))


if __name__ == "__main__":
    main()
