#!/usr/bin/env python3
"""
extract_never_worked.py

Splits closed CandleState positions into labeled subsets, each with a
matching signal-window export for the buy candle and the LOOKBACK candles
before it:

  never_worked  - lost money without ever reaching a meaningful favorable
                   peak (PeakUnrealizedGain <= PUSH_THRESHOLD), as distinct
                   from whipsaw trades (real push that reversed).
                   -> nw_positions.csv / nw_trade_signals.csv

  subset        - big moves either direction (|RealizedGain| > BIG_MOVE_THRESHOLD),
                   tagged with Outcome = BigWin / BigLoss, so lead-in signal
                   shape can be compared between the two.
                   -> subset_positions.csv / subset_trade_signals.csv

Signal-window files are long format: one row per (PositionID, Offset),
Offset 0 = buy candle, Offset -LOOKBACK..-1 = prior candles (if LOOKBACK > 0),
Offset 1..LOOKAHEAD = candles after the buy. Missing candles (e.g. near
session open/close) are left as NaN rows rather than dropped, so gaps stay
visible.

VERSION = "1.8"

USAGE:
  python extract_never_worked.py "C:\\Git\\CandleStateSessionAnalysis\\data\\MACDTrail\\output"

Expects POSITIONS_FILENAME and SIGNALS_FILENAME (below) to exist directly in
the given folder. Writes output files into that same folder. To change any
threshold, the lookback/lookahead window, or the index column name, edit the
constants below rather than passing flags.
"""

VERSION = "1.8"

# --- Hardcoded config: edit here rather than via CLI flags ---
POSITIONS_FILENAME = "positions.csv"
SIGNALS_FILENAME = "trade_signals.csv"
PUSH_THRESHOLD = 25.0        # PeakUnrealizedGain <= this AND RealizedGain < 0 => never_worked
BIG_MOVE_THRESHOLD = 100.0   # |RealizedGain| > this => subset (BigWin / BigLoss)
LOOKBACK = 0                 # candles prior to buy to include (0 = buy candle only, no lead-in)
LOOKAHEAD = 10                # candles after the buy to include, to see how fast things reverse
INDEX_COL = "Index"          # candle index column name in trade_signals.csv

import sys
from pathlib import Path

import pandas as pd


def load_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Duration"] = df["IndexClose"] - df["IndexOpen"]
    return df

def flag_never_worked(df: pd.DataFrame, push_threshold: float) -> pd.DataFrame:
    had_push = df["PeakUnrealizedGain"] > push_threshold
    never_worked = (~had_push) & (df["RealizedGain"] < 0)
    out = df.loc[never_worked].copy()
    out = out.rename(columns={"ID": "PositionID"})
    return out


def flag_subset(df: pd.DataFrame, big_move_threshold: float) -> pd.DataFrame:
    """Big moves either direction, tagged BigWin/BigLoss for comparison."""
    big_move = df["RealizedGain"].abs() > big_move_threshold
    out = df.loc[big_move].copy()
    out = out.rename(columns={"ID": "PositionID"})
    out["Outcome"] = out["RealizedGain"].apply(lambda x: "BigWin" if x > 0 else "BigLoss")
    return out


def build_signal_window(
    positions_subset: pd.DataFrame,
    signals: pd.DataFrame,
    lookback: int,
    lookahead: int,
    index_col: str,
    extra_cols: list = None,
) -> pd.DataFrame:
    """
    For each position in the subset, pull the buy-candle signal row (Offset 0)
    plus up to LOOKBACK candles before it and LOOKAHEAD candles after it,
    joined on SourceFile + Symbol + candle index. extra_cols (e.g. ["Outcome"])
    are carried through from positions_subset onto every offset row.
    """
    extra_cols = extra_cols or []
    join_cols = ["SourceFile", "Symbol"]
    for col in join_cols:
        if col not in signals.columns:
            raise ValueError(
                f"trade_signals.csv is missing expected join column '{col}'. "
                f"Columns found: {list(signals.columns)}"
            )
    if index_col not in signals.columns:
        raise ValueError(
            f"trade_signals.csv is missing index column '{index_col}' "
            f"(edit INDEX_COL to override). Columns found: {list(signals.columns)}"
        )

    offsets = list(range(-lookback, lookahead + 1))  # -lookback, ... -1, 0, 1, ... lookahead

    # Expand each position into one row per offset, with the target candle index.
    base_cols = ["PositionID", "SourceFile", "Symbol", "IndexOpen"] + extra_cols
    expanded = positions_subset[base_cols].copy()
    expanded = expanded.merge(pd.DataFrame({"Offset": offsets}), how="cross")
    expanded["TargetIndex"] = expanded["IndexOpen"] + expanded["Offset"]

    # trade_signals.csv carries its own PositionID (populated only on the
    # candle that actually became a trade). Rename to avoid colliding with
    # the PositionID we're attaching from nw_positions, and keep it as a
    # cross-check: on Offset 0 it should equal our PositionID.
    signals_renamed = signals.rename(columns={index_col: "TargetIndex", "PositionID": "SignalPositionID"})

    merged = expanded.merge(
        signals_renamed,
        on=["SourceFile", "Symbol", "TargetIndex"],
        how="left",  # keep gaps visible (e.g. near session open/close) rather than dropping
    )

    merged = merged.sort_values(["PositionID", "Offset"], ascending=[True, True])

    # Sanity check: at Offset 0, SignalPositionID should match PositionID.
    buy_rows = merged[merged["Offset"] == 0]
    mismatched = buy_rows[
        buy_rows["SignalPositionID"].notna() & (buy_rows["SignalPositionID"] != buy_rows["PositionID"])
    ]
    if len(mismatched):
        print(f"WARNING: {len(mismatched)} buy-candle rows have a SignalPositionID that "
              f"doesn't match the position's own ID - join may be misaligned.")

    return merged


def run_extraction(label, positions_subset, signals, extra_cols, folder):
    positions_out = folder / f"{label}_positions.csv"
    signals_out = folder / f"{label}_trade_signals.csv"

    positions_subset.to_csv(positions_out, index=False)

    signal_window = build_signal_window(positions_subset, signals, LOOKBACK, LOOKAHEAD, INDEX_COL, extra_cols)
    signal_window.to_csv(signals_out, index=False)

    carry_cols = ["PositionID", "SourceFile", "Symbol", "IndexOpen", "Offset", "TargetIndex"] + extra_cols
    missing = signal_window.loc[signal_window["Offset"] == 0].drop(
        columns=[c for c in carry_cols if c in signal_window.columns]
    ).isna().all(axis=1).sum()

    offsets_per_position = LOOKBACK + LOOKAHEAD + 1
    print(f"  {label}: {len(positions_subset)} positions -> {positions_out.name}")
    print(f"           {len(signal_window)} signal rows ({offsets_per_position} offsets each: "
          f"-{LOOKBACK}..+{LOOKAHEAD}) -> {signals_out.name}")
    if missing:
        print(f"           WARNING: {missing} buy-candle rows had no signal match - check join key / SourceFile coverage.")


def main():
    if len(sys.argv) != 2:
        print('Usage: python extract_never_worked.py "C:\\path\\to\\folder"')
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"ERROR: not a folder: {folder}")
        sys.exit(1)

    positions_path = folder / POSITIONS_FILENAME
    signals_path = folder / SIGNALS_FILENAME
    if not positions_path.exists():
        print(f"ERROR: {positions_path} not found")
        sys.exit(1)
    if not signals_path.exists():
        print(f"ERROR: {signals_path} not found")
        sys.exit(1)

    positions = load_positions(positions_path)
    signals = pd.read_csv(signals_path)

    print(f"[extract_never_worked v{VERSION}]")
    print(f"Folder                  : {folder}")
    print(f"Total closed positions  : {len(positions)}")
    print()

    nw = flag_never_worked(positions, PUSH_THRESHOLD)
    run_extraction("nw", nw, signals, [], folder)

    print()

    subset = flag_subset(positions, BIG_MOVE_THRESHOLD)
    win_ct = (subset["Outcome"] == "BigWin").sum()
    loss_ct = (subset["Outcome"] == "BigLoss").sum()
    print(f"  subset filter: |RealizedGain| > {BIG_MOVE_THRESHOLD} -> {win_ct} BigWin, {loss_ct} BigLoss")
    run_extraction("subset", subset, signals, ["Outcome"], folder)


if __name__ == "__main__":
    main()
