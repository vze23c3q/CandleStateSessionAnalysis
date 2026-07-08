r"""
analyze_DOW.py

Reproduces the weekly performance table:

        Count   Net   Target Hit   Win   Loss   Pct Win
    Mon   ...
    Tue   ...
    ...

Longest Win Streak / Longest Loss Streak

IMPORTANT: this is a per-DAY summary, not a per-trade summary. Count is
the number of trading days that fell on that weekday; Win/Loss classify
each day as a whole (net-positive or net-negative), not each individual
position. A day with 5 winning trades and 1 big loser can still be a
"Loss" day overall if the loser outweighs the winners.

Reads positions.csv and sessions.csv (both produced by parse_sessions.py):
positions.csv supplies each day's trades (summed into a daily P&L),
sessions.csv supplies each day's date/weekday via SessionStart.

USAGE
-----
    python analyze_DOW.py "C:\\path\\to\\output"

(pass the "output" folder that parse_sessions.py created)
"""

import sys
from pathlib import Path

import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


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


def daily_results(positions: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses positions down to one row per trading day (SourceFile):
    DailyNet = sum of that day's RealizedGain across all its positions,
    DayWin = whether that day finished net-positive.

    Weekday and calendar date come from sessions.csv's SessionStart, which
    is the session's own timestamp -- more reliable than inferring the
    date from a position's Opened time, since a day could theoretically
    have zero closed positions.
    """
    daily = positions.groupby("SourceFile", as_index=False)["RealizedGain"].sum()
    daily = daily.rename(columns={"RealizedGain": "DailyNet"})

    # Preserve every session day even if it had zero positions (DailyNet
    # would just be missing -> filled to 0 below).
    daily = sessions[["SourceFile", "SessionStart"]].merge(daily, on="SourceFile", how="left")
    daily["DailyNet"] = daily["DailyNet"].fillna(0)

    daily["Weekday"] = _parse_local_datetime(daily["SessionStart"]).dt.day_name()
    daily["DayWin"] = daily["DailyNet"] > 0
    return daily


def day_of_week_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each weekday: Count of trading days, Net (sum of that weekday's
    daily P&L), Win/Loss day counts, and win rate (share of days that
    finished net-positive).

    "Target Hit" isn't included yet -- that data doesn't exist in the
    session files until it's added to the session header. Once it is, add
    a TargetHit column here the same way Win/Loss are computed below.
    """
    grouped = daily.groupby("Weekday").agg(
        Count=("DailyNet", "size"),
        Net=("DailyNet", "sum"),
        Win=("DayWin", "sum"),
        Loss=("DayWin", lambda s: (~s).sum()),
    )
    grouped["PctWin"] = grouped["Win"] / grouped["Count"]

    # Reorder Mon-Fri regardless of which weekdays have data.
    grouped = grouped.reindex([d for d in WEEKDAY_ORDER if d in grouped.index])
    return grouped


def longest_streaks(daily: pd.DataFrame) -> dict:
    """
    Sorts trading days chronologically and finds the longest run of
    consecutive net-positive days and the longest run of consecutive
    net-negative (or flat) days.
    """
    df = daily.copy()
    df["SessionStart_dt"] = _parse_local_datetime(df["SessionStart"])
    df = df.sort_values("SessionStart_dt")

    longest_win = current_win = 0
    longest_loss = current_loss = 0
    for day_win in df["DayWin"]:
        if day_win:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)

    return {"LongestWinStreak": longest_win, "LongestLossStreak": longest_loss}


def format_for_display(dow: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the day-of-week table with Net formatted like C#'s
    "C0" (currency, no decimals, e.g. $1,241 / -$266) and PctWin formatted
    like "P0" (percentage, no decimals, e.g. 60%). Purely for display --
    the CSV keeps the raw numeric values so it stays usable for further
    analysis (Excel formulas, re-loading into pandas, etc).
    """
    display = dow.copy()
    display["Net"] = display["Net"].map(
        lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    )
    display["PctWin"] = display["PctWin"].map(lambda v: f"{v:.0%}")
    return display


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions, sessions = load_positions_and_sessions(output_dir)
    daily = daily_results(positions, sessions)

    print("--- Performance by day of week ---")
    dow = day_of_week_summary(daily)
    print(format_for_display(dow).to_string())

    dow_path = output_dir / "day_of_week_summary.csv"
    dow.to_csv(dow_path)
    print(f"\nWrote {dow_path}")

    streaks = longest_streaks(daily)
    print(f"\nLongest Win Streak:  {streaks['LongestWinStreak']}")
    print(f"Longest Loss Streak: {streaks['LongestLossStreak']}")


if __name__ == "__main__":
    main()
