r"""
analyze_Friday.py

Digs into Friday specifically: is the -$266 average from analyze_DOW.py a
handful of outlier losing days dragging down an otherwise fine day, or is
Friday consistently mediocre across the board?

Prints:
    1. Every Friday's daily P&L, sorted worst to best, with a flag on any
       day that's an outlier (more than 1 standard deviation below the
       average Friday).
    2. Every individual position from the worst 3 Fridays, so you can see
       what actually happened on those days (symbol, entry/exit, size).

Reads positions.csv and sessions.csv (both produced by parse_sessions.py).
This script is standalone (doesn't import from analyze_DOW.py) so it can
be run on its own, same as the other analysis scripts.

USAGE
-----
    python analyze_Friday.py "C:\\path\\to\\output"

(pass the "output" folder that parse_sessions.py created)
"""

import sys
from pathlib import Path

import pandas as pd


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


def load_positions_and_sessions(output_dir: Path):
    positions = pd.read_csv(output_dir / "positions.csv")
    sessions = pd.read_csv(output_dir / "sessions.csv")
    return positions, sessions


def friday_daily_results(positions: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per Friday: date, DailyNet (sum of that day's RealizedGain),
    and whether that day is a statistical outlier relative to other
    Fridays (more than 1 standard deviation below the Friday average).
    """
    daily = positions.groupby("SourceFile", as_index=False)["RealizedGain"].sum()
    daily = daily.rename(columns={"RealizedGain": "DailyNet"})

    daily = sessions[["SourceFile", "SessionStart"]].merge(daily, on="SourceFile", how="left")
    daily["DailyNet"] = daily["DailyNet"].fillna(0)

    daily["SessionStart_dt"] = _parse_local_datetime(daily["SessionStart"])
    daily["Weekday"] = daily["SessionStart_dt"].dt.day_name()

    fridays = daily[daily["Weekday"] == "Friday"].copy()
    fridays["Date"] = fridays["SessionStart_dt"].dt.date

    if len(fridays) > 1:
        mean = fridays["DailyNet"].mean()
        std = fridays["DailyNet"].std()
        fridays["IsOutlier"] = fridays["DailyNet"] < (mean - std)
    else:
        fridays["IsOutlier"] = False

    fridays = fridays.sort_values("DailyNet")
    fridays["DailyNet"] = fridays["DailyNet"].round(2)
    return fridays[["Date", "SourceFile", "DailyNet", "IsOutlier"]]


def worst_friday_positions(positions: pd.DataFrame, friday_daily: pd.DataFrame,
                            n_worst: int = 3) -> pd.DataFrame:
    """
    Every individual position from the n worst Fridays (by DailyNet), so
    you can see the actual symbols/sizes/outcomes behind those days.

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
    worst = friday_daily.head(n_worst)[["SourceFile", "DailyNet"]]
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
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions, sessions = load_positions_and_sessions(output_dir)
    fridays = friday_daily_results(positions, sessions)

    if fridays.empty:
        print("No Friday sessions found in this data.")
        return

    print(f"--- {len(fridays)} Fridays, sorted worst to best ---")
    print(fridays.to_string(index=False))

    outliers = fridays[fridays["IsOutlier"]]
    if not outliers.empty:
        print(f"\n{len(outliers)} outlier day(s) (>1 std below the Friday average):")
        print(outliers[["Date", "DailyNet"]].to_string(index=False))
    else:
        print("\nNo single-day outliers -- Friday's weakness looks spread across many days, "
              "not driven by one or two disasters.")

    fridays_path = output_dir / "friday_daily_detail.csv"
    fridays.to_csv(fridays_path, index=False)
    print(f"\nWrote {fridays_path}")

    print("\n--- Positions from the 3 worst Fridays ---")
    worst = worst_friday_positions(positions, fridays)
    print(worst.to_string(index=False))

    worst_path = output_dir / "friday_worst_positions.csv"
    worst.to_csv(worst_path, index=False)
    print(f"\nWrote {worst_path}")


if __name__ == "__main__":
    main()
