r"""
sync_sessions.py - Version 1.2.0

Moves new CandleState session files from the CandleState Sessions folder into
the per-strategy data folders of this repo.

Source: C:\ProgramData\CandleState\Sessions
Dest:   <repo>\data\{MACDTarget,MACDTrail,MACDFlip,SoftMarket}

Only these names are considered, where runStamp is yyyy-MM-dd_HHmm or
yyyy-MM-dd_HHmmss (Session_Live__*, *.zip and other names are ignored):
    Session_File_{Token}_{tradeDate}__{runStamp}.json   (Token: Target, Trail, Flip, Soft)
    Session_File__{tradeDate}__{runStamp}[__{Strategy}].json
If the name carries a Token or Strategy, it must match the strategy in the JSON
or the file is skipped. The strategy is read
from the JSON: every TradeSignals item has "Strategy": "<Strategy>/<Symbol>",
and the prefix before "/" must be one of the strategy folder names. Files
with no signals, more than one prefix, or an unknown prefix are skipped and
reported.

For each strategy:
  1. Find the latest trade date already in the strategy folder.
  2. Take source files for that strategy with a newer trade date.
  3. If several share a trade date, keep the one with the newest runStamp.
  4. Move them into the strategy folder, never overwriting an existing file.
  5. Print a summary of moved and skipped files with the reason.

USAGE:
    python sync_sessions.py              # move files
    python sync_sessions.py --dry-run    # show what would move
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

SOURCE_DIR = Path(r"C:\ProgramData\CandleState\Sessions")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRATEGIES = ["MACDTarget", "MACDTrail", "MACDFlip", "SoftMarket"]

# Short strategy tokens used in file names (Session_File_<Token>_...) -> folder name.
NAME_TOKENS = {"Target": "MACDTarget", "Trail": "MACDTrail", "Flip": "MACDFlip", "Soft": "SoftMarket"}

FILE_GLOB = "Session_File_*.json"
FILE_RE = re.compile(
    r"^Session_File_(?:_|(?P<token>[A-Za-z]+)_)"
    r"(?P<date>\d{4}-\d{2}-\d{2})__(?P<stamp>\d{4}-\d{2}-\d{2}_\d{4}(?:\d{2})?)"
    r"(?:__(?P<suffix>\w+))?\.json$"
)


def parse_name(path):
    """Return (tradeDate, runStamp) for a Session_File name, else None."""
    m = FILE_RE.match(path.name)
    return (m.group("date"), m.group("stamp")) if m else None


def name_strategy(path):
    """Strategy named in the file name (prefix token or __<Strategy> suffix), or None."""
    m = FILE_RE.match(path.name)
    if not m:
        return None
    named = m.group("token") or m.group("suffix")
    return NAME_TOKENS.get(named, named)


def read_strategy(path):
    """Return (strategy, None) or (None, reason) from the file's TradeSignals."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, f"unreadable JSON ({e})"
    signals = data.get("TradeSignals") or []
    prefixes = {str(s.get("Strategy", "")).split("/")[0] for s in signals}
    prefixes.discard("")
    if not prefixes:
        return None, "no TradeSignals with a Strategy"
    if len(prefixes) > 1:
        return None, f"multiple strategies: {', '.join(sorted(prefixes))}"
    return prefixes.pop(), None


def latest_trade_date(folder):
    """Latest tradeDate among Session_File names in folder, or None."""
    dates = [p[0] for p in map(parse_name, folder.glob(FILE_GLOB)) if p]
    return max(dates) if dates else None


def main():
    parser = argparse.ArgumentParser(description="Move new CandleState session files into strategy folders.")
    parser.add_argument("--dry-run", action="store_true", help="show what would move without moving anything")
    args = parser.parse_args()

    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source folder not found: {SOURCE_DIR}")

    latest_by_strategy = {s: latest_trade_date(DATA_DIR / s) for s in STRATEGIES if (DATA_DIR / s).is_dir()}
    # Files at or before every folder's latest date can't be moved anywhere, so don't open them.
    latests = list(latest_by_strategy.values())
    cutoff = min(latests) if latests and None not in latests else None

    # Classify the remaining source Session_Files by strategy.
    by_strategy = defaultdict(list)   # strategy -> [(tradeDate, runStamp, path)]
    unassigned = []                   # (path, reason)
    for path in sorted(SOURCE_DIR.glob(FILE_GLOB)):
        parsed = parse_name(path)
        if not parsed or (cutoff and parsed[0] <= cutoff):
            continue
        strategy, reason = read_strategy(path)
        named = name_strategy(path)
        if strategy is None:
            unassigned.append((path, reason))
        elif named and named != strategy:
            unassigned.append((path, f"file name says {named} but signals are {strategy}"))
        elif strategy not in STRATEGIES:
            unassigned.append((path, f"strategy '{strategy}' has no data folder"))
        else:
            by_strategy[strategy].append((*parsed, path))

    tag = "[DRY RUN] " if args.dry_run else ""
    total_moved = 0

    for strategy in STRATEGIES:
        dest_dir = DATA_DIR / strategy
        print(f"\n=== {strategy} ===")
        if not dest_dir.is_dir():
            print(f"  Destination folder not found: {dest_dir} - skipped")
            continue

        latest = latest_by_strategy[strategy]
        print(f"  Latest trade date in folder: {latest or '(none)'}")

        moved, skipped = [], []
        newest_per_date = {}
        for trade_date, run_stamp, path in by_strategy.get(strategy, []):
            if latest and trade_date <= latest:
                skipped.append((path, f"trade date {trade_date} not newer than {latest}"))
                continue
            prev = newest_per_date.get(trade_date)
            if prev is None or run_stamp > prev[0]:
                if prev:
                    skipped.append((prev[1], f"superseded by newer run {run_stamp}"))
                newest_per_date[trade_date] = (run_stamp, path)
            else:
                skipped.append((path, f"superseded by newer run {prev[0]}"))

        for trade_date in sorted(newest_per_date):
            path = newest_per_date[trade_date][1]
            target = dest_dir / path.name
            if target.exists():
                skipped.append((path, "file already exists in destination"))
                continue
            if not args.dry_run:
                try:
                    shutil.move(str(path), str(target))
                except OSError as e:
                    skipped.append((path, f"move failed ({e})"))
                    continue
            moved.append(path)

        print(f"  {tag}Moved: {len(moved)}")
        for path in moved:
            print(f"    {path.name}")
        print(f"  Skipped: {len(skipped)}")
        for path, reason in sorted(skipped):
            print(f"    {path.name}  - {reason}")
        total_moved += len(moved)

    if unassigned:
        print(f"\n=== Not assigned to a strategy: {len(unassigned)} ===")
        for path, reason in unassigned:
            print(f"    {path.name}  - {reason}")

    verb = "would be moved" if args.dry_run else "moved"
    print(f"\nTotal {verb}: {total_moved}")


if __name__ == "__main__":
    main()
