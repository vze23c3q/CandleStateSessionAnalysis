"""
analyze_sessions.py

Reads the CSVs produced by parse_sessions.py (transactions.csv and
trade_signals.csv) and checks, for every BUY fill, whether you entered
above or below VWAP at that bar.

WHY THIS JOIN WORKS
--------------------
positions.csv has one row per position, with IndexOpen marking the bar the
position was entered on.
trade_signals.csv has one row per candle/bar, with the VWAP for that bar.

Joining Positions.IndexOpen -> TradeSignals.Index gives one VWAP-at-entry
row per position (as opposed to joining off transactions.csv, which would
give one row per partial-fill buy if a position was built in stages).

Positions.AvgPrice isn't usable for the actual entry price, since it resets
to 0 once a position closes. So the exact entry fill price is pulled
separately from transactions.csv (the first BUY transaction for that
Position's ID) and attached alongside the VWAP.

USAGE
-----
    python analyze_sessions.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output"

(pass the "output" folder that parse_sessions.py created)
"""

import sys
from pathlib import Path

import pandas as pd


def load_tables(output_dir: Path):
    positions = pd.read_csv(output_dir / "positions.csv")
    trade_signals = pd.read_csv(output_dir / "trade_signals.csv")
    transactions = pd.read_csv(output_dir / "transactions.csv")
    return positions, trade_signals, transactions


def buys_vs_vwap(positions: pd.DataFrame, trade_signals: pd.DataFrame,
                  transactions: pd.DataFrame) -> pd.DataFrame:
    # 1. Join each position to the trade_signals row at its entry bar.
    #    Join on SourceFile + Symbol + IndexOpen/Index because Index numbers
    #    reset each session, so Symbol/Index alone could collide across files.
    entry_bars = positions.merge(
        trade_signals[["SourceFile", "Symbol", "Index", "VWAP", "Close"]],
        left_on=["SourceFile", "Symbol", "IndexOpen"],
        right_on=["SourceFile", "Symbol", "Index"],
        how="left",
    )

    # 2. Get the actual entry fill price: the first BUY transaction that
    #    belongs to this position (Positions.AvgPrice resets to 0 once a
    #    position is closed, so it can't be used here).
    buys = transactions[transactions["Quantity"] > 0].sort_values("Created")
    first_buy = (
        buys.groupby(["SourceFile", "PositionID"], as_index=False)
        .first()[["SourceFile", "PositionID", "Price"]]
        .rename(columns={"Price": "EntryPrice"})
    )
    entry_bars = entry_bars.merge(
        first_buy,
        left_on=["SourceFile", "ID"],
        right_on=["SourceFile", "PositionID"],
        how="left",
    )

    entry_bars["EntryVsVWAP"] = entry_bars["EntryPrice"] - entry_bars["VWAP"]
    entry_bars["AboveVWAP"] = entry_bars["EntryVsVWAP"] > 0

    cols = ["SourceFile", "Symbol", "ID", "IndexOpen", "Opened",
            "EntryPrice", "VWAP", "EntryVsVWAP", "AboveVWAP",
            "RealizedGain"]
    return entry_bars[cols].sort_values(["SourceFile", "Opened"])


def win_rate_by_vwap_side(buys: pd.DataFrame) -> pd.DataFrame:
    """
    Groups positions by whether they entered above or below VWAP, and
    reports win rate (share of positions with RealizedGain > 0) for each
    side, along with count, average win size, average loss size, and
    expectancy (winrate * avg_win - (1 - winrate) * avg_loss).

    Win rate alone can mislead: a bucket can win more often but still have
    worse expectancy if its average loss is bigger than its average win.
    """
    grouped = buys.groupby("AboveVWAP").agg(
        Count=("RealizedGain", "size"),
        Wins=("RealizedGain", lambda s: (s >= 0).sum()),
        AvgRealizedGain=("RealizedGain", "mean"),
        TotalRealizedGain=("RealizedGain", "sum"),
    )
    grouped["WinRate"] = grouped["Wins"] / grouped["Count"]

    # Average win size = mean RealizedGain among winning trades only.
    # Average loss size = mean absolute RealizedGain among losing trades only
    # (kept positive so it reads naturally next to AvgWin, and so the
    # expectancy formula below matches the standard textbook form).
    avg_win = buys[buys["RealizedGain"] >= 0].groupby("AboveVWAP")["RealizedGain"].mean()
    avg_loss = buys[buys["RealizedGain"] < 0].groupby("AboveVWAP")["RealizedGain"].mean().abs()
    grouped["AvgWin"] = avg_win
    grouped["AvgLoss"] = avg_loss
    grouped[["AvgWin", "AvgLoss"]] = grouped[["AvgWin", "AvgLoss"]].fillna(0)

    grouped["Expectancy"] = (
        grouped["WinRate"] * grouped["AvgWin"]
        - (1 - grouped["WinRate"]) * grouped["AvgLoss"]
    )
    return grouped


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions, trade_signals, transactions = load_tables(output_dir)
    result = buys_vs_vwap(positions, trade_signals, transactions)

    print(result.to_string(index=False))
    print()
    print(f"Total buys: {len(result)}")
    print(f"Above VWAP: {result['AboveVWAP'].sum()}")
    print(f"Below VWAP: {(~result['AboveVWAP']).sum()}")
    unmatched = result["VWAP"].isna().sum()
    if unmatched:
        print(f"WARNING: {unmatched} buys had no matching VWAP row (check the join).")

    out_path = output_dir / "buys_vs_vwap.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n--- Win rate: entered above VWAP vs below VWAP ---")
    win_rates = win_rate_by_vwap_side(result)
    print(win_rates.to_string())

    win_rate_path = output_dir / "win_rate_by_vwap_side.csv"
    win_rates.to_csv(win_rate_path)
    print(f"\nWrote {win_rate_path}")


if __name__ == "__main__":
    main()
