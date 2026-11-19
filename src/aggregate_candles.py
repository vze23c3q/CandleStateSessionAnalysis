"""
aggregate_candles.py
Version: 1.0

Purpose:
    Step 2 of 3
    Aggregate the Polygon 1-minute candle CSVs (from fetch_polygon_candles.py)
    into N-minute candles anchored to the 09:30 ET session open, so bar
    boundaries match the CandleState session clock. Writes one output CSV per
    symbol.  Step 3, split_candle_sets.py, will then split the aggregated CSVs into overlapping 3-day sets for each run day.

Usage (Windows):
    python aggregate_candles.py

    Adjust the CONFIG block below. AGG_MINUTES=2 by default; set 5, 10, etc.
    for other granularities - the same anchoring logic applies.

Input:
    INPUT_DIR / candles_<SYMBOL>_1min.csv with columns:
        Timestamp, Open, High, Low, Close, Volume, VWAP, Trades
    (VWAP and Trades optional - handled if absent.)

Output:
    OUTPUT_DIR / candles_<SYMBOL>_<AGG_MINUTES>min.csv, same columns.
    Aggregation per bin: Open=first, High=max, Low=min, Close=last,
    Volume=sum, Trades=sum, VWAP=volume-weighted mean of the 1-min VWAPs.

Design notes:
    - Anchoring: each bar is assigned to a bin by flooring its minutes-since-
      09:30 to a multiple of AGG_MINUTES, per day. This guarantees bins start
      at 9:30, 9:32, ... regardless of missing minutes, and can never span
      the overnight gap (grouping is per-date). A top-of-hour/epoch anchor
      would phase-shift every MACD crossover downstream - avoided by design.
    - Bars before 09:30 or at/after 16:00 (if present in the input) are
      dropped with a warning rather than binned, so an extended-hours input
      file can't silently contaminate RTH bars.
    - Partial bins are kept: if only one of a bin's minutes exists (e.g. a
      data gap, or an odd trailing minute on some feed), the bin is emitted
      from what's there. The per-day bar count report surfaces these - a full
      RTH day at 2-min is 195 bars, a 13:00 half day is 105.
    - VWAP recombination: sum(VWAP_i * Volume_i) / sum(Volume_i). If a bin's
      total volume is 0, VWAP falls back to the bin Close.

Changelog:
    1.0 - Initial version. Session-anchored flooring aggregation, RTH guard,
          VWAP/Trades recombination, per-day bar-count anomaly report.
"""

from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "1.0"

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

INPUT_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data\polygon")
OUTPUT_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data\polygon")

# SYMBOLS = ["UPRO", "SDS", "TQQQ", "QID"]
SYMBOLS = ["SPY", "QQQ"]

INPUT_PATTERN = "candles_{symbol}_1min.csv"
OUTPUT_PATTERN = "candles_{symbol}_{agg}min.csv"

AGG_MINUTES = 2

SESSION_OPEN_MIN = 9 * 60 + 30    # 09:30
SESSION_CLOSE_MIN = 16 * 60       # 16:00 (exclusive)

# Expected 2-min bars: full day 195, half day (13:00 close) 105. Used only
# for the anomaly report - update if AGG_MINUTES changes the math for you.
EXPECTED_COUNTS = {390 // AGG_MINUTES, 210 // AGG_MINUTES}

# ------------------------------------------------------------------------- #


def aggregate_symbol(symbol: str) -> None:
    in_path = INPUT_DIR / INPUT_PATTERN.format(symbol=symbol)
    if not in_path.exists():
        print(f"{symbol}: [WARN] {in_path} not found - skipping.")
        return

    df = pd.read_csv(in_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    has_vwap = "VWAP" in df.columns and df["VWAP"].notna().any()
    has_trades = "Trades" in df.columns and df["Trades"].notna().any()

    minutes = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    rth = (minutes >= SESSION_OPEN_MIN) & (minutes < SESSION_CLOSE_MIN)
    dropped = int((~rth).sum())
    if dropped:
        print(f"{symbol}: [WARN] dropped {dropped:,} non-RTH bars "
              "found in input.")
    df, minutes = df[rth].copy(), minutes[rth]

    if df.empty:
        print(f"{symbol}: [WARN] no RTH bars - skipping.")
        return

    # Session-anchored bin: floor minutes-since-open to AGG_MINUTES.
    offset = minutes - SESSION_OPEN_MIN
    bin_start_min = SESSION_OPEN_MIN + (offset // AGG_MINUTES) * AGG_MINUTES
    df["BinTime"] = (df["Timestamp"].dt.normalize()
                     + pd.to_timedelta(bin_start_min, unit="m"))

    if has_vwap:
        df["_vwap_x_vol"] = df["VWAP"] * df["Volume"]

    agg_spec = {
        "Open": ("Open", "first"),
        "High": ("High", "max"),
        "Low": ("Low", "min"),
        "Close": ("Close", "last"),
        "Volume": ("Volume", "sum"),
    }
    if has_trades:
        agg_spec["Trades"] = ("Trades", "sum")
    if has_vwap:
        agg_spec["_vv"] = ("_vwap_x_vol", "sum")

    out = (df.groupby("BinTime", as_index=False)
           .agg(**agg_spec)
           .rename(columns={"BinTime": "Timestamp"})
           .sort_values("Timestamp").reset_index(drop=True))

    if has_vwap:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["VWAP"] = np.where(out["Volume"] > 0,
                                   out["_vv"] / out["Volume"], out["Close"])
        out = out.drop(columns=["_vv"])
        out["VWAP"] = out["VWAP"].round(4)

    cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    cols += ["VWAP"] if has_vwap else []
    cols += ["Trades"] if has_trades else []
    out = out[cols]

    out_path = OUTPUT_DIR / OUTPUT_PATTERN.format(symbol=symbol,
                                                  agg=AGG_MINUTES)
    out.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S",
               float_format="%.4f")

    # Per-day bar count anomaly report
    per_day = out.groupby(out["Timestamp"].dt.normalize()).size()
    odd_days = per_day[~per_day.isin(EXPECTED_COUNTS)]
    print(f"{symbol}: {len(df):,} 1-min bars -> {len(out):,} "
          f"{AGG_MINUTES}-min bars across {per_day.size} days "
          f"-> {out_path.name}")
    if odd_days.empty:
        print(f"    All days have expected bar counts "
              f"({sorted(EXPECTED_COUNTS)}).")
    else:
        print(f"    [CHECK] {len(odd_days)} day(s) with unexpected bar "
              "counts (gaps or partial data):")
        for day, n in odd_days.items():
            print(f"      {day:%Y-%m-%d}: {n} bars")


def main():
    print(f"aggregate_candles.py v{VERSION}  |  {AGG_MINUTES}-min bars, "
          "anchored 09:30\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        aggregate_symbol(symbol)
    print("\nDone.")


if __name__ == "__main__":
    main()
