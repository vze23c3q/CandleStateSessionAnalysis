r"""
finish_day_candles.py - Version 3.0.0

Finishes a partial day's candle sets by appending tape candles to the live
(streamer) candle files. Used when the streamer dies intraday: the live files
hold aggregate candles up to the failure point, and after the close the tape
files (full-day chart history) fill in the remainder of the session.

Given a date parameter (e.g. 20260721), the script locates:
    <root>\<date> Live    - partial-day candle sets (updated in place)
    <root>\<date>         - full-day tape candle sets (read only)
and processes every .json file in the Live folder, pairing each with the
same-named file in the tape folder (falling back to a Symbol-field match if
the filename is not found).

After a fully successful run (and not --dry-run), the folders are renamed so
the finished day is ready to backtest:
    <root>\<date>       -> <root>\<date> Tape    (raw tape archived)
    <root>\<date> Live  -> <root>\<date>         (finished candle sets)
If any file fails, the renames are skipped so the run can be repeated after
fixing the problem.

Merge behavior per file:
  - Appends all tape candles for the target date after the last complete
    live candle.
  - A trailing incomplete live candle (IsCloseCandle false) is REPLACED by
    the tape candle for that slot, since the streamer died mid-candle and
    its OHLC/volume are only partial. Pass --keep-incomplete to keep the
    partial live candle instead. A finished file therefore has every candle
    IsCloseCandle true.
  - Tape candles are appended verbatim (their minimal field set is
    preserved; no streamer-only fields are fabricated).
  - Prior days in the live file are untouched.
  - Each live file is backed up alongside itself with a timestamped .bak
    before being overwritten (skip with --no-backup).
  - Validates symbol match, chronological order, and duplicate timestamps;
    warns about missing candle slots (interval inferred from the data) and
    warns if the tape set itself is incomplete (last candle before 15:52).

USAGE:
  python finish_day_candles.py 20260721

  Optional:
      --root C:\ProgramData\CandleState\Candles   (default shown)
      --keep-incomplete    (keep a trailing IsCloseCandle=false candle
                            instead of replacing it with the tape candle)
      --no-backup          (skip the .bak backup)
      --dry-run            (report only; write nothing)
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

VERSION = "3.0.0"
DEFAULT_ROOT = r"C:\ProgramData\CandleState\Candles"
TAPE_COMPLETE_TIME = "15:52"  # warn if the tape's last candle is before this


def load_candle_set(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def candle_dt(candle: dict) -> datetime:
    return datetime.fromisoformat(candle["DateTime"])


def candle_date(candle: dict) -> str:
    return candle["DateTime"][:10]


def infer_interval_minutes(candles: list) -> int:
    """Infer the candle interval as the most common gap between consecutive candles."""
    if len(candles) < 2:
        return 2
    gaps = []
    for a, b in zip(candles, candles[1:]):
        delta = (candle_dt(b) - candle_dt(a)).total_seconds() / 60
        if delta > 0:
            gaps.append(int(delta))
    if not gaps:
        return 2
    return Counter(gaps).most_common(1)[0][0]


def report_gaps(candles: list, interval_min: int) -> None:
    for a, b in zip(candles, candles[1:]):
        delta = (candle_dt(b) - candle_dt(a)).total_seconds() / 60
        if delta > interval_min:
            missing = int(delta // interval_min) - 1
            print(f"    WARNING: gap of {int(delta)} min "
                  f"({missing} missing candle{'s' if missing != 1 else ''}) "
                  f"between {a['DateTime']} and {b['DateTime']}")


def find_tape_file(live_path: Path, tape_dir: Path, live_symbol: str) -> Path | None:
    """Pair a live file with its tape file: same filename first, Symbol match second."""
    candidate = tape_dir / live_path.name
    if candidate.is_file():
        return candidate
    for path in sorted(tape_dir.glob("*.json")):
        try:
            if load_candle_set(path).get("Symbol") == live_symbol:
                return path
        except (json.JSONDecodeError, OSError):
            continue
    return None


def finish_file(live_path: Path, tape_dir: Path, target_date: str,
                keep_incomplete: bool, no_backup: bool, dry_run: bool) -> bool:
    """Merge one live file with its tape counterpart. Returns True on success."""
    print(f"\n{live_path.name}")

    live_set = load_candle_set(live_path)
    live_symbol = live_set.get("Symbol") or "(unknown)"

    tape_path = find_tape_file(live_path, tape_dir, live_symbol)
    if tape_path is None:
        print(f"    ERROR: No tape file found for {live_symbol} in {tape_dir}")
        return False

    tape_set = load_candle_set(tape_path)

    live_candles = live_set.get("Candles") or []
    tape_candles = tape_set.get("Candles") or []
    if not live_candles:
        print("    ERROR: Live file contains no candles.")
        return False
    if not tape_candles:
        print(f"    ERROR: Tape file contains no candles: {tape_path.name}")
        return False

    t_sym = tape_set.get("Symbol")
    if live_set.get("Symbol") and t_sym and live_set["Symbol"] != t_sym:
        print(f"    ERROR: Symbol mismatch: live={live_set['Symbol']}, tape={t_sym}")
        return False

    day_live = [c for c in live_candles if candle_date(c) == target_date]
    day_tape = [c for c in tape_candles if candle_date(c) == target_date]
    if not day_live:
        print(f"    ERROR: Live file has no candles on {target_date}.")
        return False
    if not day_tape:
        print(f"    ERROR: Tape file has no candles on {target_date}.")
        return False

    last_live = day_live[-1]
    last_is_incomplete = last_live.get("IsCloseCandle") is False

    print(f"    Live: {len(day_live)} candles, "
          f"{day_live[0]['DateTime'][11:16]} -> {last_live['DateTime'][11:16]}"
          f"{' (last candle INCOMPLETE)' if last_is_incomplete else ''}")
    tape_last_time = day_tape[-1]["DateTime"][11:16]
    print(f"    Tape: {len(day_tape)} candles, "
          f"{day_tape[0]['DateTime'][11:16]} -> {tape_last_time}")
    if tape_last_time < TAPE_COMPLETE_TIME:
        print(f"    WARNING: Tape candle set incomplete "
              f"(ends {tape_last_time}, before {TAPE_COMPLETE_TIME}). "
              f"Merged day will not reach the close.")

    # Determine splice point
    drop_last_live = False
    if last_is_incomplete and not keep_incomplete:
        drop_last_live = True
        if len(day_live) >= 2:
            splice_after = candle_dt(day_live[-2])
            to_append = [c for c in day_tape if candle_dt(c) > splice_after]
        else:
            to_append = [c for c in day_tape if candle_dt(c) >= candle_dt(last_live)]
        print(f"    Replacing incomplete live candle at "
              f"{last_live['DateTime'][11:16]} with tape data.")
    else:
        splice_after = candle_dt(last_live)
        to_append = [c for c in day_tape if candle_dt(c) > splice_after]
        if last_is_incomplete:
            print("    NOTE: Keeping incomplete final live candle "
                  "(--keep-incomplete specified).")

    if not to_append:
        print("    Nothing to append: tape has no candles after the live file's "
              "last candle. File left unchanged.")
        return True

    print(f"    Appending {len(to_append)} tape candles: "
          f"{to_append[0]['DateTime'][11:16]} -> {to_append[-1]['DateTime'][11:16]}")

    merged = [c for c in live_candles if not (drop_last_live and c is last_live)]
    merged.extend(to_append)

    # Validate: chronological order and no duplicate timestamps on target date
    day_merged = [c for c in merged if candle_date(c) == target_date]
    times = [c["DateTime"] for c in day_merged]
    dupes = [t for t, n in Counter(times).items() if n > 1]
    if dupes:
        print(f"    ERROR: Duplicate timestamps after merge: {sorted(dupes)[:5]}")
        return False
    if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
        print("    ERROR: Merged candles are not in chronological order.")
        return False

    interval_min = infer_interval_minutes(day_merged)
    print(f"    Merged: {len(day_merged)} candles, "
          f"{day_merged[0]['DateTime'][11:16]} -> {day_merged[-1]['DateTime'][11:16]} "
          f"({interval_min}-min interval)")
    report_gaps(day_merged, interval_min)

    if dry_run:
        print("    Dry run: not written.")
        return True

    live_set["Candles"] = merged

    if not no_backup:
        bak_path = live_path.with_suffix(
            live_path.suffix + f".{datetime.now():%Y%m%d_%H%M%S}.bak")
        shutil.copy2(live_path, bak_path)
        print(f"    Backup: {bak_path.name}")

    with open(live_path, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(live_set, f, indent="\t")
    print(f"    Written: {live_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append tape candles to partial live candle files for a date.")
    parser.add_argument("date",
                        help="Trading date as YYYYMMDD, e.g. 20260721.")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"Candles root folder (default: {DEFAULT_ROOT}).")
    parser.add_argument("--keep-incomplete", action="store_true",
                        help="Keep a trailing IsCloseCandle=false live candle "
                             "instead of replacing it with the tape candle.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the .bak backup before overwriting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen without writing.")
    args = parser.parse_args()

    print(f"finish_day_candles.py v{VERSION}")

    if len(args.date) != 8 or not args.date.isdigit():
        print(f"ERROR: Date must be YYYYMMDD, got '{args.date}'.")
        return 1
    target_date = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"

    root = Path(args.root)
    live_dir = root / f"{args.date} Live"
    tape_dir = root / args.date

    if not live_dir.is_dir():
        print(f"ERROR: Live folder not found: {live_dir}")
        return 1
    if not tape_dir.is_dir():
        print(f"ERROR: Tape folder not found: {tape_dir}")
        return 1

    live_files = sorted(live_dir.glob("*.json"))
    if not live_files:
        print(f"ERROR: No .json files in {live_dir}")
        return 1

    print(f"Target date: {target_date}")
    print(f"Live folder: {live_dir} ({len(live_files)} files)")
    print(f"Tape folder: {tape_dir}")

    results = {}
    for live_path in live_files:
        try:
            results[live_path.name] = finish_file(
                live_path, tape_dir, target_date,
                args.keep_incomplete, args.no_backup, args.dry_run)
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as ex:
            print(f"    ERROR: {type(ex).__name__}: {ex}")
            results[live_path.name] = False

    ok = sum(1 for v in results.values() if v)
    failed = [name for name, v in results.items() if not v]
    print(f"\nDone: {ok}/{len(results)} files succeeded.")
    if failed:
        print("Failed: " + ", ".join(failed))
        print("Folder renames skipped due to failures; rerun after fixing.")
        return 1

    if args.dry_run:
        print("Dry run: folder renames skipped.")
        return 0

    tape_target = root / f"{args.date} Tape"
    if tape_target.exists():
        print(f"ERROR: Cannot rename tape folder; target already exists: {tape_target}")
        return 1
    live_target = root / args.date  # becomes free once tape_dir is renamed
    tape_dir.rename(tape_target)
    print(f"Renamed: {tape_dir.name} -> {tape_target.name}")
    live_dir.rename(live_target)
    print(f"Renamed: {live_dir.name} -> {live_target.name}")
    print("Ready to backtest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
