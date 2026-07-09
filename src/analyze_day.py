r"""
analyze_day.py

Digs into a single weekday: is its average from analyze_DOW.py a handful
of outlier days dragging (or lifting) an otherwise ordinary day, or is
that weekday consistently strong/weak across the board?

Prints:
    1. Every occurrence of that weekday's daily P&L, sorted worst to best,
       with a flag on any day that's an outlier (more than 1 standard
       deviation below that weekday's average).
    2. Every individual position from the worst 3 occurrences, so you can
       see what actually happened on those days (symbol, entry/exit, size).

Reads positions.csv and sessions.csv (both produced by parse_sessions.py).
This script is standalone (doesn't import from analyze_DOW.py) so it can
be run on its own, same as the other analysis scripts.

USAGE
-----
    python analyze_day.py <Weekday> "C:\\path\\to\\output"

    python analyze_day.py Thursday "C:\Git\CandleStateSessionAnalysis\data\output"

<Weekday> is any of Monday/Tuesday/Wednesday/Thursday/Friday (case
insensitive).
"""

import sys
from pathlib import Path

import pandas as pd

VALID_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _parse_local_datetime(series: pd.Series) -> pd.Series:
    """
    Timestamps carry a UTC offset (e.g. -04:00 or -05:00) that changes
    across the Daylight Saving Time boundary. pandas can't parse a column
    with mixed offsets directly, so this normalizes everything to UTC
    first, then converts to US/Eastern so the weekday still reflects the
    actual local trading day (not the UTC day, which can differ near
    midnight).
    """
    return pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York")


def normalize_weekday(raw: str) -> str:
    """
    Matches the given string against the 5 trading weekdays, case
    insensitive. Raises a clear error (rather than silently returning zero
    rows) if it doesn't match, since a typo here would otherwise look
    identical to "no sessions found."
    """
    lookup = {day.lower(): day for day in VALID_WEEKDAYS}
    match = lookup.get(raw.strip().lower())
    if match is None:
        valid = ", ".join(VALID_WEEKDAYS)
        raise ValueError(f"'{raw}' isn't a recognized weekday. Expected one of: {valid}")
    return match


def load_positions_and_sessions(output_dir: Path):
    positions = pd.read_csv(output_dir / "positions.csv")
    sessions = pd.read_csv(output_dir / "sessions.csv")
    return positions, sessions


def day_results(positions: pd.DataFrame, sessions: pd.DataFrame, weekday: str) -> pd.DataFrame:
    """
    One row per occurrence of `weekday`: date, DailyNet (sum of that day's
    RealizedGain), and whether that day is a statistical outlier relative
    to other occurrences of the same weekday (more than 1 standard
    deviation below that weekday's average).
    """
    daily = positions.groupby("SourceFile", as_index=False)["RealizedGain"].sum()
    daily = daily.rename(columns={"RealizedGain": "DailyNet"})

    daily = sessions[["SourceFile", "SessionStart"]].merge(daily, on="SourceFile", how="left")
    daily["DailyNet"] = daily["DailyNet"].fillna(0)

    daily["SessionStart_dt"] = _parse_local_datetime(daily["SessionStart"])
    daily["Weekday"] = daily["SessionStart_dt"].dt.day_name()

    matching_days = daily[daily["Weekday"] == weekday].copy()
    matching_days["Date"] = matching_days["SessionStart_dt"].dt.date

    if len(matching_days) > 1:
        mean = matching_days["DailyNet"].mean()
        std = matching_days["DailyNet"].std()
        matching_days["IsOutlier"] = matching_days["DailyNet"] < (mean - std)
    else:
        matching_days["IsOutlier"] = False

    matching_days = matching_days.sort_values("DailyNet")
    matching_days["DailyNet"] = matching_days["DailyNet"].round(2)
    return matching_days[["Date", "SourceFile", "DailyNet", "IsOutlier"]]


def worst_day_positions(positions: pd.DataFrame, day_daily: pd.DataFrame,
                         n_worst: int = 3) -> pd.DataFrame:
    """
    Every individual position from the n worst occurrences (by DailyNet),
    so you can see the actual symbols/sizes/outcomes behind those days.

    NOTE ON TIMES: some early session files (e.g. 2026-03-06) serialize
    Opened/Closed with a "Z" (UTC) suffix instead of a proper offset like
    the rest -- but the clock values themselves (e.g. 15:54 at close,
    matching every other file's market-close time) show this is actually
    local Eastern time mislabeled as UTC, not real UTC. Converting it as
    true UTC would show entries at ~5am, which can't be right for
    market-hours trades. So this reads the wall-clock digits directly out
    of the timestamp string rather than doing timezone-aware conversion,
    to avoid displaying a wrong shifted time for those files. Worth
    checking on the CandleState side when/why that serialization changed.
    """
    worst = day_daily.head(n_worst)[["SourceFile", "DailyNet"]]
    worst = worst.rename(columns={"DailyNet": "DayTotal"})

    detail = positions.merge(worst, on="SourceFile", how="inner").copy()

    # Pull HH:MM directly out of the ISO string (chars 11:16), ignoring
    # any timezone suffix -- see note above on why.
    detail["EntryTime"] = detail["Opened"].str[11:16]
    detail["ExitTime"] = detail["Closed"].str[11:16]

    opened_naive = pd.to_datetime(detail["Opened"].str[:19])
    closed_naive = pd.to_datetime(detail["Closed"].str[:19])
    detail["HeldMinutes"] = ((closed_naive - opened_naive).dt.total_seconds() / 60).round(0)

    detail["RealizedGain"] = detail["RealizedGain"].round(2)

    cols = ["SourceFile", "DayTotal", "Symbol", "EntryTime", "ExitTime",
            "HeldMinutes", "RealizedGain"]
    detail = detail.sort_values(["DayTotal", "SourceFile", "EntryTime"])
    return detail[cols]


def main():
    if len(sys.argv) < 3:
        valid = ", ".join(VALID_WEEKDAYS)
        print(f"Usage: python analyze_day.py <Weekday> <output_dir>")
        print(f"  <Weekday> must be one of: {valid}")
        sys.exit(1)

    try:
        weekday = normalize_weekday(sys.argv[1])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_dir = Path(sys.argv[2]).resolve()

    positions, sessions = load_positions_and_sessions(output_dir)
    days = day_results(positions, sessions, weekday)

    if days.empty:
        print(f"No {weekday} sessions found in this data.")
        return

    print(f"--- {len(days)} {weekday}s, sorted worst to best ---")
    print(days.to_string(index=False))

    outliers = days[days["IsOutlier"]]
    if not outliers.empty:
        print(f"\n{len(outliers)} outlier day(s) (>1 std below the {weekday} average):")
        print(outliers[["Date", "DailyNet"]].to_string(index=False))
    else:
        print(f"\nNo single-day outliers -- {weekday}'s result looks spread across many days, "
              "not driven by one or two disasters.")

    days_path = output_dir / f"{weekday.lower()}_daily_detail.csv"
    days.to_csv(days_path, index=False)
    print(f"\nWrote {days_path}")

    print(f"\n--- Positions from the 3 worst {weekday}s ---")
    worst = worst_day_positions(positions, days)
    print(worst.to_string(index=False))

    worst_path = output_dir / f"{weekday.lower()}_worst_positions.csv"
    worst.to_csv(worst_path, index=False)
    print(f"\nWrote {worst_path}")


if __name__ == "__main__":
    main()
