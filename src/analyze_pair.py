r"""
analyze_pair.py

Candle-by-candle trace of DayGainLoss across every symbol in one or more
session files, in TRUE chronological order -- not grouped by symbol.

DayGainLoss is a shared/portfolio-level field (see OrderManager.
UpsertPositionList: DayGainLoss = Positions.Sum(NetGain)), so its real
behavior only makes sense viewed as one interleaved timeline across every
symbol, not as separate per-symbol trajectories. This script builds that
timeline so you can see, tick by tick:
    - which symbol's candle triggered each DayGainLoss value
    - whether the value actually changed from the row before it
    - whether that symbol had shares open at that moment (Shares column)

This is meant to be iterated on -- first pass just gets the raw sequence
in front of you. Tell me what's confusing or missing and we'll adjust.

UNLIKE the other analyze_*.py scripts, this one does NOT use
parse_sessions.py's output. It parses the raw session JSON files itself,
directly in memory, and only pulls the 3 sections needed for this check
(TradeSignals, Transactions, Positions) -- skipping Logs, PriceLevels,
Orders, and Sessions entirely, since Logs especially is slow to parse and
isn't needed here. Nothing gets written to disk; this is meant to be
cheap enough to just rerun each time.

USAGE
-----
    python analyze_pair.py "C:\\path\\to\\data\\recon"

(pass the folder containing the raw session .json files directly -- NOT
a parse_sessions.py output folder)
"""

import sys
from pathlib import Path

import pandas as pd

# Reuse the enum-conversion table and JSON loader from parse_sessions.py
# rather than duplicating them. Must be run from the same folder (src/).
import parse_sessions as ps

# Only the 3 sections needed for this check -- Logs is deliberately
# skipped (it's the slowest table by far, thousands of rows per file, and
# not needed for a DayGainLoss trace).
SECTIONS_NEEDED = {
    "TradeSignals": "trade_signals",
    "Transactions": "transactions",
    "Positions": "positions",
}


def parse_light(folder: Path) -> dict:
    """
    Same row-building logic as parse_sessions.py's parse_sessions(), but
    limited to SECTIONS_NEEDED and with no Sessions/Logs tables built.
    Returns {"trade_signals": df, "transactions": df, "positions": df}.
    """
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {folder}")

    section_rows = {name: [] for name in SECTIONS_NEEDED.values()}

    for filepath in json_files:
        data = ps.load_session_file(filepath)

        file_counts = {}
        for json_key, table_name in SECTIONS_NEEDED.items():
            records = data.get(json_key, [])
            file_counts[table_name] = len(records)
            for record in records:
                row = dict(record)
                row["SourceFile"] = filepath.name
                row["SessionStart"] = data.get("SessionStart")
                row["AccountName"] = data.get("AccountName")

                for column, names in ps.ENUM_COLUMNS.items():
                    if column[0] != table_name or column[1] not in row:
                        continue
                    ordinal = row[column[1]]
                    if isinstance(ordinal, int) and 0 <= ordinal < len(names):
                        row[column[1]] = names[ordinal]

                # Position.AvgPrice gets nulled out once Quantity hits 0
                # (position closed), so recompute it from BuyTransactions
                # instead -- a quantity-weighted average across every buy
                # (most positions have just one, but adds create more).
                #
                # Also compute ClosePriceSellActivity here, from the nested
                # SellActivity within each buy lot (FIFO fills against that
                # lot) -- this is the RAW fill price, matching the
                # Description string, as opposed to ClosePriceTrx (added
                # below from Transactions.csv-equivalent data), which is
                # back-derived from NetAmount/CommissionFee and can differ
                # by a fraction of a cent. Both are quantity-weighted
                # averages in case of multiple partial sells.
                if table_name == "positions" and "BuyTransactions" in row:
                    buys = row["BuyTransactions"] or []
                    total_qty = sum(b.get("BuyQuantity", 0) for b in buys)
                    total_cost = sum(
                        b.get("BuyQuantity", 0) * b.get("Price", 0) for b in buys
                    )
                    row["AvgPrice"] = total_cost / total_qty if total_qty else None

                    sell_qty_total = 0
                    sell_cost_total = 0
                    for b in buys:
                        for s in (b.get("SellActivity") or []):
                            qty = s.get("SellQty", 0)
                            sell_qty_total += qty
                            sell_cost_total += qty * s.get("Price", 0)
                    row["ClosePriceSellActivity"] = (
                        sell_cost_total / sell_qty_total if sell_qty_total else None
                    )

                    del row["BuyTransactions"]  # nested list, not needed beyond this

                section_rows[table_name].append(row)

        print(f"Parsed {filepath.name}: "
              + ", ".join(f"{table}={count}" for table, count in file_counts.items()))

    tables = {name: pd.DataFrame(rows) for name, rows in section_rows.items()}

    # ClosePrice per position, pulled from the matching SELL transaction(s)
    # (Quantity < 0) rather than from BuyTransactions' SellActivity -- a
    # position can be trimmed in more than one sell, so this is a
    # quantity-weighted average across all of them, same approach as
    # AvgPrice above.
    positions_df = tables["positions"]
    transactions_df = tables["transactions"]
    if not positions_df.empty and not transactions_df.empty:
        sells = transactions_df[transactions_df["Quantity"] < 0].copy()
        sells["AbsQty"] = sells["Quantity"].abs()
        sells["Cost"] = sells["AbsQty"] * sells["Price"]
        close_totals = sells.groupby(["SourceFile", "PositionID"], as_index=False).agg(
            TotalQty=("AbsQty", "sum"), TotalCost=("Cost", "sum")
        )
        close_totals["ClosePriceTrx"] = close_totals["TotalCost"] / close_totals["TotalQty"]

        positions_df = positions_df.merge(
            close_totals[["SourceFile", "PositionID", "ClosePriceTrx"]],
            left_on=["SourceFile", "ID"], right_on=["SourceFile", "PositionID"],
            how="left", suffixes=("", "_fromTrx"),
        )
        tables["positions"] = positions_df

    return tables


def build_position_timelines(trade_signals: pd.DataFrame, positions: pd.DataFrame,
                              source_file: str) -> list:
    """
    One timeline per position (not one big whole-day timeline). Each
    position row defines the range to show: its own Symbol, from
    IndexOpen to IndexClose. Loops because IndexOpen/IndexClose are
    per-position -- a session can hold multiple positions (even multiple
    for the same symbol, if it's entered more than once in a day), each
    with its own open/close range.

    Returns a list of (position_row, timeline_df) tuples, one per
    position, in the order positions were opened.
    """
    pos_for_file = positions[positions["SourceFile"] == source_file].copy()
    pos_for_file = pos_for_file.sort_values("IndexOpen")

    results = []
    for _, pos in pos_for_file.iterrows():
        symbol = pos["Symbol"]
        index_open = pos["IndexOpen"]
        index_close = pos["IndexClose"]

        df = trade_signals[
            (trade_signals["SourceFile"] == source_file)
            & (trade_signals["Symbol"] == symbol)
            & (trade_signals["Index"] >= index_open)
            & (trade_signals["Index"] <= index_close)
        ].copy()
        df = df.sort_values("CandleOpen").reset_index(drop=True)

        df["DayGainLossChanged"] = df["DayGainLoss"].diff().fillna(1) != 0

        cols = ["CandleOpen", "Index", "Symbol", "Close", "Shares",
                "DayGainLoss", "DayGainLossChanged", "PositionID", "AccountName"]
        cols = [c for c in cols if c in df.columns]
        results.append((pos, df[cols]))

    return results


def print_with_repeated_headers(df: pd.DataFrame, chunk_size: int = 40):
    """
    Prints a DataFrame in chunks, reprinting the column header before each
    chunk -- otherwise the header scrolls off screen almost immediately on
    a long table and every row past the first screenful is unlabeled.
    """
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        print(chunk.to_string(index=False, header=True))
        print()


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    folder = folder.resolve()

    tables = parse_light(folder)
    trade_signals = tables["trade_signals"]
    positions = tables["positions"]
    source_files = trade_signals["SourceFile"].unique()

    print(f"\nFound {len(source_files)} session file(s) in {folder}\n")

    for source_file in source_files:
        print(f"=== {source_file} ===\n")

        position_timelines = build_position_timelines(trade_signals, positions, source_file)
        if not position_timelines:
            print("No positions found in this session.\n")
            continue

        for pos, timeline in position_timelines:
            total = len(timeline)
            changed = timeline["DayGainLossChanged"].sum() if total else 0
            print(f"--- Position {pos.get('ID', '?')}: {pos['Symbol']}  "
                  f"Index {pos['IndexOpen']}-{pos['IndexClose']}  "
                  f"AvgPrice={pos.get('AvgPrice', 0):.4f}  "
                  f"ClosePriceTrx={pos.get('ClosePriceTrx', 0):.4f}  "
                  f"ClosePriceSellActivity={pos.get('ClosePriceSellActivity', 0):.4f}  "
                  f"RealizedGain={pos.get('RealizedGain', 0):.2f} ---")
            print(f"{total} candles, DayGainLoss changed on {changed} of them "
                  f"({changed/total:.0%})\n" if total else "No matching candles found.\n")
            print_with_repeated_headers(timeline)


if __name__ == "__main__":
    main()
