r"""
parse_sessions.py

Loads all CandleState session JSON files from a folder and turns each
section of every file into a pandas DataFrame (a table). The tables from
all files are stacked together and saved as CSVs so you can open them in
Excel, load them into another script, or query them with pandas/SQL.

USAGE
-----
    python parse_sessions.py "C:\Git\CandleStateSessionAnalysis\data\MACD Target"
    python parse_sessions.py "C:\Git\CandleStateSessionAnalysis\data\MACD Trail"

If you don't pass a path, it defaults to the current folder.

WHAT IT PRODUCES
----------------
An "output" folder (created next to your session files) containing:
    sessions.csv        -> one row per session file (metadata: start/end,
                            account, capital, day gain, etc.)
    trade_signals.csv    -> every bar/candle signal row, from every file
    price_levels.csv     -> support/resistance level rows
    orders.csv            -> order rows
    transactions.csv      -> fill/transaction rows
    positions.csv          -> position rows
    logs.csv                -> log message rows

Every row in every table gets two extra columns so you can trace it back
to its source and join it against the `sessions` table:
    SourceFile    -> the JSON filename it came from
    SessionStart  -> the SessionStart timestamp of that session
"""

import json
import sys
from pathlib import Path

import pandas as pd

# The six list-shaped sections found inside every session file.
# (dict key in the JSON) -> (name we'll use for the output table/CSV)
SECTIONS = {
    "TradeSignals": "trade_signals",
    "PriceLevels": "price_levels",
    "Orders": "orders",
    "Transactions": "transactions",
    "Positions": "positions",
    "Logs": "logs",
}

# Enum converters, mirroring C# enums in CandleState. The JSON stores these
# fields as integer ordinals; this maps them back to readable names.
#
# Key = (table_name, column_name) exactly as they appear after SECTIONS
# renames the table. Value = list of names in ordinal order (index 0 = the
# enum's first member, etc).
#
# To add another one, just add another entry here — nothing else in the
# script needs to change.
ENUM_COLUMNS = {
    ("logs", "Category"): [
        "Info", "Warning", "Error", "Trade", "Signal", "Debug",
        "Critical", "Monitor", "Rule", "Json", "Alert",
    ],
    ("trade_signals", "CandleName"): [
        "Undefined", "Doji", "BullElephant", "BearElephant", "ToppingTail",
        "BottomingTail", "Bull180", "Bear180", "BullHammer", "BearHammer",
        "BullColorChange", "BearColorChange", 
    ],
    ("trade_signals", "CandleColor"): ["White", "Green", "Red"
    ],
    ("trade_signals", "Signal"): ["Idle", "MarketOpen", "Buy", "Add", "Remove", 
        "Sell", "Continue", "Gated", "Stopped", "SessionComplete"
    ],
    ("trade_signals", "MACDPhase"): ["RisingGreen", "FallingGreen", "FallingRed", "RisingRed"
    ],
    ("transactions", "Type"): ["BuyToOpen", "BuyToClose", "SellToOpen", "SellToClose"   
    ],
    ("price_levels", "Type"): ["PriorHighOfDay", "PriorLowOfDay", "PriorDayClose", "Intraday",   
    ],
}

# The top-level fields that describe the session itself (not a list).
SESSION_FIELDS = [
    "SessionStart", "SessionEnd", "TradingMode", "AccountName",
    "TradingCapital", "MaxDayLoss", "DayTarget", "CashBalance", "DayGain", "TargetHit",
]


def load_session_file(filepath: Path) -> dict:
    """Read one session JSON file and return the parsed dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_sessions(folder: Path):
    """
    Walk every *.json file in `folder`, and build one combined DataFrame
    per section (plus a `sessions` table of the per-file metadata).

    Returns a dict: {"sessions": df, "trade_signals": df, ...}
    """
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {folder}")

    # Collect rows for each table as we go, then build DataFrames at the end.
    session_rows = []
    section_rows = {name: [] for name in SECTIONS.values()}

    for filepath in json_files:
        data = load_session_file(filepath)

        # --- 1. Session-level metadata becomes one row in `sessions` ---
        session_row = {field: data.get(field) for field in SESSION_FIELDS}
        session_row["SourceFile"] = filepath.name
        session_rows.append(session_row)

        # --- 2. Each list section becomes rows in its own table ---
        file_counts = {}
        for json_key, table_name in SECTIONS.items():
            records = data.get(json_key, [])
            file_counts[table_name] = len(records)
            for record in records:
                # Tag every row so it can be traced back to its session.
                row = dict(record)
                row["SourceFile"] = filepath.name
                row["SessionStart"] = data.get("SessionStart")

                # Translate any enum-backed int columns to readable names.
                # Falls back to the raw number if it's ever out of range,
                # so a mismatched/stale mapping never breaks parsing.
                for column, names in ENUM_COLUMNS.items():
                    if column[0] != table_name or column[1] not in row:
                        continue
                    ordinal = row[column[1]]
                    if isinstance(ordinal, int) and 0 <= ordinal < len(names):
                        row[column[1]] = names[ordinal]

                section_rows[table_name].append(row)

        print(f"Parsed {filepath.name}: "
              + ", ".join(f"{table}={count}" for table, count in file_counts.items()))

    tables = {"sessions": pd.DataFrame(session_rows)}
    for table_name, rows in section_rows.items():
        tables[table_name] = pd.DataFrame(rows)

    return tables


def save_tables(tables: dict, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    for name, df in tables.items():
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(df)} rows, {len(df.columns)} columns)")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    folder = folder.resolve()

    tables = parse_sessions(folder)
    save_tables(tables, folder / "output")


if __name__ == "__main__":
    main()
