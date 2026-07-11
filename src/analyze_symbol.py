r"""
analyze_symbol.py

Breaks down performance by Symbol: Count of trades, Net (sum of
RealizedGain), Win/Loss counts, and win rate.

Unlike analyze_DOW.py, this is a per-TRADE summary, not a per-day summary
-- each position already belongs to exactly one symbol, so there's no
day-level collapsing needed here.

Reads positions.csv directly (produced by parse_sessions.py).

USAGE
-----
    python analyze_symbol.py "C:\\path\\to\\output"

(pass the "output" folder that parse_sessions.py created)
"""

import sys
from pathlib import Path

import pandas as pd


def load_positions(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "positions.csv")


def symbol_summary(positions: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol: Count of trades, Net (sum of RealizedGain), Win/Loss
    counts, and win rate. Sorted by Net, best to worst.
    """
    grouped = positions.groupby("Symbol").agg(
        Count=("RealizedGain", "size"),
        Net=("RealizedGain", "sum"),
        Win=("RealizedGain", lambda s: (s >= 0).sum()),
        Loss=("RealizedGain", lambda s: (s < 0).sum()),
    )
    grouped["PctWin"] = grouped["Win"] / grouped["Count"]
    grouped = grouped.sort_values("Net", ascending=False)
    return grouped


def format_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Net formatted like C#'s "C0" (currency, no decimals), PctWin like "P0"
    (percentage, no decimals). Display only -- the CSV keeps raw numbers.
    """
    display = summary.copy()
    display["Net"] = display["Net"].map(
        lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    )
    display["PctWin"] = display["PctWin"].map(lambda v: f"{v:.0%}")
    return display


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions = load_positions(output_dir)
    summary = symbol_summary(positions)

    print("--- Performance by symbol ---")
    print(format_for_display(summary).to_string())

    out_path = output_dir / "symbol_summary.csv"
    summary.to_csv(out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
