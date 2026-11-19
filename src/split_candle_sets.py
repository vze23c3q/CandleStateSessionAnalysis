"""
split_candle_sets.py
Version: 1.1

Purpose:
    Step 3 of 3
    Split the aggregated 2-minute candle CSVs (from aggregate_candles.py)
    into CandleState tape files: one candle set per RUN DAY, where each set
    contains 3 trading days - the run day plus the PRIOR 2 days to prime the
    analytics. Sets therefore overlap: folder 20250605 holds 6/3-6/5, folder
    20250606 holds 6/4-6/6, and so on. The first two days of the pull only
    prime later sets and get no folder of their own, so the first output is
    20250604 (6/2-6/4) and the last is the final trading day of the data.
    Each folder contains one candleSet_<Symbol>_yyyymmdd.json per symbol.

Usage (Windows):
    python split_candle_sets.py

    Adjust the CONFIG block below. First chunk for the 2025-06-01 pull is
    2025-06-02..04 -> folder 20250604.

Input:
    INPUT_DIR / candles_<SYMBOL>_2min.csv with columns:
        Timestamp, Open, High, Low, Close, Volume (extra columns ignored)

Output:
    OUTPUT_ROOT / <yyyymmdd> / candleSet_<Symbol>_<yyyymmdd>.json
    Envelope: Symbol populated; StartTime/EndTime = first/last candle
    DateTime in the chunk; LoadTime = script run time; Description and
    HighProfitLoss null; Decimals 0; PremarketVolume 0 (not available from
    RTH-only tape). Candles carry DateTime, Open, High, Low, Close, Volume,
    IsCloseCandle=true only - aggregated-candle bookkeeping fields
    (IsCumulative, Created, Completed, ...) are deliberately omitted since
    these are tape candles.

Design notes:
    - The trading-day calendar is the UNION of dates across all symbols, so
      set boundaries and folder names are identical for every symbol.
    - Rolling windows: run day i produces days[i-PRIME_DAYS .. i]. Every set
      is exactly SET_DAYS long by construction - days with fewer than
      PRIME_DAYS predecessors (the start of the pull) produce no set.
    - DateTime strings carry the correct US/Eastern UTC offset per candle
      (-04:00 EDT / -05:00 EST via zoneinfo), so the June-2025 start and the
      Feb-2026 end of the pull serialize correctly across the November DST
      transition.
    - JSON is written pretty-printed (tab indent). CandleState's
      deserializer won't care, and it stays human-diffable.

Changelog:
    1.0 - Initial version. Disjoint 3-day chunking (wrong model), Eastern-
          offset DateTime serialization, tape-only candle fields.
    1.1 - Corrected to ROLLING per-run-day sets: each folder is one run day
          plus PRIME_DAYS prior days (overlapping windows), so every set has
          exactly SET_DAYS days and the last folder is the final trading
          day. First PRIME_DAYS days prime only.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

VERSION = "1.1"

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

INPUT_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data\polygon")
OUTPUT_ROOT = Path(r"C:\Git\CandleStateSessionAnalysis\data\polygon\candles")

# SYMBOLS = ["UPRO", "SDS", "TQQQ", "QID"]
SYMBOLS = ["SPY", "QQQ"]

INPUT_PATTERN = "candles_{symbol}_2min.csv"
FILE_PATTERN = "candleSet_{symbol}_{chunk}.json"

SET_DAYS = 3                 # run day + PRIME_DAYS prior days
PRIME_DAYS = SET_DAYS - 1

EASTERN = ZoneInfo("America/New_York")

# ------------------------------------------------------------------------- #


def eastern_iso(ts: pd.Timestamp) -> str:
    """Naive ET timestamp -> ISO string with correct -04:00/-05:00 offset."""
    return ts.tz_localize(EASTERN).isoformat()


def load_symbol(symbol: str) -> pd.DataFrame | None:
    path = INPUT_DIR / INPUT_PATTERN.format(symbol=symbol)
    if not path.exists():
        print(f"[WARN] {path} not found - {symbol} will be skipped.")
        return None
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Day"] = df["Timestamp"].dt.normalize()
    return df


def build_candle_set(symbol: str, chunk_df: pd.DataFrame) -> dict:
    candles = [
        {
            "DateTime": eastern_iso(row.Timestamp),
            "Open": row.Open,
            "High": row.High,
            "Low": row.Low,
            "Close": row.Close,
            "Volume": int(row.Volume),
            "IsCloseCandle": True,
        }
        for row in chunk_df.itertuples()
    ]
    return {
        "Symbol": symbol,
        "Description": None,
        "Decimals": 0,
        "StartTime": candles[0]["DateTime"],
        "EndTime": candles[-1]["DateTime"],
        "LoadTime": datetime.now(EASTERN).isoformat(),
        "PremarketVolume": 0,
        "HighProfitLoss": None,
        "Candles": candles,
    }


def main():
    print(f"split_candle_sets.py v{VERSION}  |  rolling {SET_DAYS}-day sets "
          f"({PRIME_DAYS} prime + 1 run)\n")

    frames = {s: df for s in SYMBOLS if (df := load_symbol(s)) is not None}
    if not frames:
        raise SystemExit("No input CSVs found - check INPUT_DIR.")

    # Union of trading days across symbols -> identical set boundaries.
    all_days = sorted(set().union(*[set(df["Day"]) for df in frames.values()]))
    if len(all_days) < SET_DAYS:
        raise SystemExit(f"Only {len(all_days)} trading days - need at "
                         f"least {SET_DAYS} for one set.")
    run_days = all_days[PRIME_DAYS:]
    print(f"{len(all_days)} trading days -> {len(run_days)} run-day sets "
          f"(first run {run_days[0]:%Y-%m-%d}, last {run_days[-1]:%Y-%m-%d}; "
          f"{all_days[0]:%Y-%m-%d}..{all_days[PRIME_DAYS-1]:%Y-%m-%d} "
          "prime only)\n")

    for i, run_day in enumerate(run_days, start=PRIME_DAYS):
        chunk_days = all_days[i - PRIME_DAYS:i + 1]
        chunk_name = f"{run_day:%Y%m%d}"
        chunk_dir = OUTPUT_ROOT / chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for symbol, df in frames.items():
            chunk_df = df[df["Day"].isin(chunk_days)]
            if chunk_df.empty:
                print(f"{chunk_name}: [WARN] no {symbol} data - skipped.")
                continue
            payload = build_candle_set(symbol, chunk_df)
            out_path = chunk_dir / FILE_PATTERN.format(symbol=symbol,
                                                       chunk=chunk_name)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent="\t")

        print(f"{chunk_name}: {', '.join(f'{d:%m/%d}' for d in chunk_days)}"
              f"  ->  {len(frames)} file(s)")

    print(f"\nDone. Output under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
