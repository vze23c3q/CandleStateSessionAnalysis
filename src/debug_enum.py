"""
debug_enum.py - temporary diagnostic, not part of the real pipeline.

Run: python debug_enum.py "C:\\Git\\CandleStateSessionAnalysis\\data\\Session_File__2026-03-02__2026-07-08_0924.json"
"""
import json
import sys

ENUM_COLUMNS = {
    ("trade_signals", "Signal"): ["Idle", "MarketOpen", "Buy", "Add", "Remove",
        "Sell", "Continue", "Gated", "Stopped", "SessionComplete"
    ],
}

filepath = sys.argv[1]
with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

table_name = "trade_signals"  # this is what SECTIONS maps "TradeSignals" to
records = data.get("TradeSignals", [])
print(f"Found {len(records)} TradeSignals records")

record = records[0]
row = dict(record)
print(f"Raw Signal value: {row.get('Signal')!r}  (type: {type(row.get('Signal')).__name__})")

converted_any = False
for column, names in ENUM_COLUMNS.items():
    print(f"Checking ENUM_COLUMNS entry: {column}")
    print(f"  column[0] == table_name?  {column[0]!r} == {table_name!r} -> {column[0] == table_name}")
    print(f"  column[1] in row?         {column[1]!r} in row -> {column[1] in row}")
    if column[0] != table_name or column[1] not in row:
        print("  -> SKIPPED (continue triggered)")
        continue
    ordinal = row[column[1]]
    print(f"  ordinal = {ordinal!r}, isinstance int = {isinstance(ordinal, int)}, in range = {0 <= ordinal < len(names) if isinstance(ordinal, int) else 'n/a'}")
    if isinstance(ordinal, int) and 0 <= ordinal < len(names):
        row[column[1]] = names[ordinal]
        converted_any = True
        print(f"  -> CONVERTED to {row[column[1]]!r}")

print(f"\nFinal row Signal value: {row.get('Signal')!r}")
print(f"Any conversion happened: {converted_any}")
