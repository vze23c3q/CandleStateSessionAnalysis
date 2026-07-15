r"""
analyze_target_hit.py  v1.7

Looks specifically at sessions where TargetHit == True, and profiles the
position activity that got them there -- how many positions were opened,
and whether the day traded in ONE direction (all long-side symbols, or
all inverse-side symbols, i.e. "Short") or FLIPPED between them.

The Directions column is collapsed -- consecutive positions in the same
direction only show once, so "SDS, UPRO, TQQQ" (Short, Long, Long) prints
as "Short, Long", not "Short, Long, Long". Only actual direction changes
show up.

This reads parse_sessions.py's CSV OUTPUT (sessions.csv, positions.csv) 
rather than the raw session JSON -- this is a downstream analysis over data 
that's already been parsed once, not a from-scratch parse.

USAGE
-----
    python analyze_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output"
    python analyze_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output" S
    python analyze_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output" N
    python analyze_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output" S --miss

INDEX is OPTIONAL: "S" for S&P (UPRO/SDS), "N" for Nasdaq (TQQQ/QID), or
omit it entirely to look at both indexes together (no symbol filtering --
a day that traded both shows all of them, same as pre-index behavior).
When an index IS given, only positions in that index's two symbols are
considered -- a session that also traded the other index that day has
those positions ignored for this profile, and a session with NO
positions in the requested index is dropped entirely (nothing to
profile).

Default is TargetHit == True. Add --miss to flip to TargetHit == False,
for the same stats on days the target was NOT hit.

"""

import sys
from pathlib import Path

import pandas as pd

VERSION = "1.7"

# The only thing that should need editing if the symbol lineup changes.
SYMBOL_DIRECTION = {
    "UPRO": "Long",
    "TQQQ": "Long",
    "SDS": "Short",
    "QID": "Short",
}

INDEX_SYMBOLS = {
    "S": ["UPRO", "SDS"],   # S&P
    "N": ["TQQQ", "QID"],   # Nasdaq
}


def load_tables(output_folder: Path) -> tuple:
    """Reads sessions.csv and positions.csv from a parse_sessions.py output folder."""
    sessions_path = output_folder / "sessions.csv"
    positions_path = output_folder / "positions.csv"

    if not sessions_path.exists():
        raise FileNotFoundError(f"sessions.csv not found in {output_folder}")
    if not positions_path.exists():
        raise FileNotFoundError(f"positions.csv not found in {output_folder}")

    sessions = pd.read_csv(sessions_path)
    positions = pd.read_csv(positions_path)
    return sessions, positions


def profile_target_hit_days(sessions: pd.DataFrame, positions: pd.DataFrame,
                             index_symbols: list = None, target_hit: bool = True) -> pd.DataFrame:
    """
    One row per session matching TargetHit == target_hit. If index_symbols
    is given (e.g. ["UPRO", "SDS"] for S&P), only positions in those two
    symbols are considered -- a day that also traded the other index has
    those positions ignored for this profile, and a session with NO
    positions at all in index_symbols is dropped. If index_symbols is
    None, every position counts regardless of index -- a day that traded
    both shows all of them together.

        PositionCount    -- how many positions (in scope) were opened
                             that day
        Symbols          -- symbols traded, in the order they were opened
        Directions       -- Long/Short, COLLAPSED so consecutive positions
                             in the same direction only show once (e.g.
                             "Short, Long, Long" prints as "Short, Long")
        Flipped          -- True if the day held both a Long and a
                             Short symbol at some point
        DirectionChanges -- how many times direction switched (0 = never
                             flipped) -- same as len(collapsed) - 1
        RealizedPL       -- sum across matching positions that day, pulled
                             from the RealizedGain field (never NetGain --
                             matches every other script in this project).
                             Labeled "PL" here rather than "Gain" since
                             it's frequently negative.
    """
    matching = sessions[sessions["TargetHit"] == target_hit].copy()

    rows = []
    for _, session in matching.iterrows():
        source_file = session["SourceFile"]
        pos_for_day = positions[positions["SourceFile"] == source_file].copy()
        if index_symbols is not None:
            pos_for_day = pos_for_day[pos_for_day["Symbol"].isin(index_symbols)]
        pos_for_day = pos_for_day.sort_values("IndexOpen")

        if pos_for_day.empty:
            continue  # nothing in scope for this day

        symbols = pos_for_day["Symbol"].tolist()
        directions = [SYMBOL_DIRECTION.get(sym, "Unknown") for sym in symbols]

        # Collapse consecutive duplicates -- only an actual change in
        # direction is worth showing (e.g. Short, Long, Long -> Short, Long).
        collapsed_directions = [
            d for i, d in enumerate(directions) if i == 0 or d != directions[i - 1]
        ]
        direction_changes = len(collapsed_directions) - 1

        rows.append({
            "SourceFile": source_file,
            "AccountName": session.get("AccountName"),
            "SessionStart": session.get("SessionStart"),
            "DayTarget": session.get("DayTarget"),
            "DayGain": session.get("DayGain"),
            "PositionCount": len(pos_for_day),
            "Symbols": ", ".join(symbols),
            "Directions": ", ".join(collapsed_directions),
            "Flipped": direction_changes > 0,
            "DirectionChanges": direction_changes,
            "RealizedPL": pos_for_day["RealizedGain"].sum(),
        })

    return pd.DataFrame(rows)


def print_summary(profile: pd.DataFrame, target_hit: bool = True):
    label = "target-hit" if target_hit else "target-missed"

    if profile.empty:
        print(f"No TargetHit=={target_hit} sessions found.")
        return

    print(f"{len(profile)} {label} session(s)\n")

    flipped = profile["Flipped"].sum()
    one_direction = len(profile) - flipped
    print(f"One direction only: {one_direction} ({one_direction/len(profile):.0%})")
    print(f"Flipped direction:  {flipped} ({flipped/len(profile):.0%})\n")

    print("Position count distribution:")
    print(profile["PositionCount"].value_counts().sort_index().to_string())
    print()

    print("Direction combination counts:")
    direction_pl = profile.groupby("Directions")["RealizedPL"].agg(
        Count="count", TotalPL="sum", AvgPL="mean"
    ).sort_values("Count", ascending=False)
    print(direction_pl.to_string())
    print()

    cols = ["SourceFile", "AccountName", "PositionCount", "Symbols",
            "Directions", "Flipped", "RealizedPL"]
    cols = [c for c in cols if c in profile.columns]
    print(profile[cols].to_string(index=False))


def main():
    print(f"analyze_target_hit.py  v{VERSION}")

    args = sys.argv[1:]
    target_hit = "--miss" not in args
    args = [a for a in args if a != "--miss"]

    if len(args) < 1:
        print('Usage: python analyze_target_hit.py "<output_folder>" [S|N] [--miss]')
        print('  S = S&P (UPRO/SDS)   N = Nasdaq (TQQQ/QID)   omit = both together')
        sys.exit(1)

    folder = Path(args[0]).resolve()

    index = None
    index_symbols = None
    if len(args) >= 2:
        index = args[1].upper()
        if index not in INDEX_SYMBOLS:
            print(f'Index must be "S" or "N", got "{args[1]}"')
            sys.exit(1)
        index_symbols = INDEX_SYMBOLS[index]

    sessions, positions = load_tables(folder)
    profile = profile_target_hit_days(sessions, positions, index_symbols, target_hit=target_hit)
    if index:
        print(f"Index: {index} ({'/'.join(index_symbols)})\n")
    else:
        print("Index: All (S&P + Nasdaq)\n")
    print_summary(profile, target_hit=target_hit)


if __name__ == "__main__":
    main()
