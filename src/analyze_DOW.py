r"""
analyze_DOW.py

Reproduces the weekly performance table:

        Count   Net   Target Hit   Win   Loss   Pct Win
    Mon   ...
    Tue   ...
    ...

Longest Win Streak / Longest Loss Streak
Current Streak
Streak continuation table (historical odds the next day extends a streak
of a given type/length, with the current streak's odds called out)

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
    python analyze_DOW.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output"
    python analyze_DOW.py "C:\Git\CandleStateSessionAnalysis\data\MACDTrail\output"

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
    daily = sessions[["SourceFile", "SessionStart", "TargetHit"]].merge(daily, on="SourceFile", how="left")
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
        Positions=("DailyNet", "size"),
        Net=("DailyNet", "sum"),
        Wins=("DayWin", "sum"),
        Losses=("DayWin", lambda s: (~s).sum()),
        TargetHit=("TargetHit", "sum"),
    )
    grouped["WinPct"] = grouped["Wins"] / grouped["Positions"]
    grouped["TargetHitPct"] = grouped["TargetHit"] / grouped["Positions"]

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


def current_streak(daily: pd.DataFrame) -> dict:
    """
    Sorts trading days chronologically and reports the streak currently
    in progress as of the most recent day: how many consecutive days,
    counting backward from the latest one, share the same Win/Loss
    direction. E.g. if the last 3 days were all net-positive, this
    returns {"Type": "Win", "Length": 3} -- different from
    LongestWinStreak, which reports the best historical run rather than
    what's happening right now.
    """
    df = daily.copy()
    df["SessionStart_dt"] = _parse_local_datetime(df["SessionStart"])
    df = df.sort_values("SessionStart_dt")

    if df.empty:
        return {"Type": None, "Length": 0}

    day_wins = df["DayWin"].tolist()
    current_type_is_win = day_wins[-1]

    length = 0
    for day_win in reversed(day_wins):
        if day_win == current_type_is_win:
            length += 1
        else:
            break

    return {"Type": "Win" if current_type_is_win else "Loss", "Length": length}


def streak_continuation(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Answers: "we're currently sitting at N consecutive Wins (or Losses) --
    historically, how often did the NEXT day extend that streak?"

    Walks the trading days chronologically. After each day closes, the
    account is sitting at some streak state (e.g. "3 Wins"). The following
    day either continues the streak (same direction) or breaks it. Every
    historical day except the most recent one contributes exactly one
    observation to its streak state's bucket.

    Note that continuation is direction-matched: at "2 Losses", a
    continuation means the next day was also a Loss. So for Loss streaks,
    a LOW ContinuePct is good news.

    Sample sizes shrink fast at longer streak lengths, so treat the deep
    rows with suspicion. If day-to-day results were truly independent,
    every Win row would hover near the overall daily win rate no matter
    the length -- deviations in thin buckets are more likely noise than
    momentum/mean-reversion signal.
    """
    df = daily.copy()
    df["SessionStart_dt"] = _parse_local_datetime(df["SessionStart"])
    df = df.sort_values("SessionStart_dt")
    wins = df["DayWin"].tolist()

    records = []
    streak_len = 0
    streak_is_win = None
    for i, day_win in enumerate(wins):
        if day_win == streak_is_win:
            streak_len += 1
        else:
            streak_is_win = day_win
            streak_len = 1
        # The last day has no "next day" yet -- it's the state we're
        # trying to predict, not evidence.
        if i < len(wins) - 1:
            records.append(
                {
                    "Type": "Win" if streak_is_win else "Loss",
                    "Length": streak_len,
                    "Continued": wins[i + 1] == streak_is_win,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["Type", "Length", "Count", "Continued", "ContinuePct"]
        )

    table = (
        pd.DataFrame(records)
        .groupby(["Type", "Length"])
        .agg(Count=("Continued", "size"), Continued=("Continued", "sum"))
        .reset_index()
    )
    table["ContinuePct"] = table["Continued"] / table["Count"]

    # Win streaks first (ascending length), then Loss streaks.
    table = table.sort_values(
        ["Type", "Length"], ascending=[False, True]
    ).reset_index(drop=True)
    return table


def current_streak_summary(daily: pd.DataFrame) -> str:
    """
    One-line sentence combining current_streak() with its row in
    streak_continuation(), e.g.:

        Current streak is 3 Wins. Historically at this state: next day
        continued 4 of 8 times (50%).

    Intended for notebook/report use where the full continuation table
    would be too much -- just the row that matters right now. Falls back
    to a "no precedent" message when the current streak is the longest of
    its kind on record (its state has never had a next-day observation).
    """
    current = current_streak(daily)
    if current["Type"] is None:
        return "No trading days yet -- no current streak."

    plural = {"Win": "Wins", "Loss": "Losses"}
    label = current["Type"] if current["Length"] == 1 else plural[current["Type"]]
    state = f"{current['Length']} {label}"

    cont = streak_continuation(daily)
    match = cont[
        (cont["Type"] == current["Type"]) & (cont["Length"] == current["Length"])
    ]
    if match.empty:
        return (
            f"**Current Streak** is {state}. No historical precedent -- "
            "this is the longest such streak on record."
        )

    row = match.iloc[0]
    return (
        f"**Current Streak** is {state}. Historically at this state: "
        f"next day continued {int(row['Continued'])} of {int(row['Count'])} "
        f"times ({row['ContinuePct']:.0%})."
    )


def format_continuation_for_display(cont: pd.DataFrame) -> pd.DataFrame:
    """
    Display copy of the continuation table: ContinuePct formatted like
    "P0". The CSV keeps raw numerics, same convention as the DOW table.
    """
    display = cont.copy()
    display["ContinuePct"] = display["ContinuePct"].map(lambda v: f"{v:.0%}")
    return display


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
    display["WinPct"] = display["WinPct"].map(lambda v: f"{v:.0%}")
    display["TargetHitPct"] = display["TargetHitPct"].map(lambda v: f"{v:.0%}")
    
    column_order = ["Positions", "Wins", "Losses", "WinPct", "TargetHit", "TargetHitPct", "Net"]
    return display[column_order]
    


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

    current = current_streak(daily)
    if current["Type"] is not None:
        plural = {"Win": "Wins", "Loss": "Losses"}
        label = current["Type"] if current["Length"] == 1 else plural[current["Type"]]
        print(f"Current Streak:      {current['Length']} {label}")

    print("\n--- Streak continuation (chance the NEXT day extends the streak) ---")
    cont = streak_continuation(daily)
    print(format_continuation_for_display(cont).to_string(index=False))

    cont_path = output_dir / "streak_continuation.csv"
    cont.to_csv(cont_path, index=False)
    print(f"\nWrote {cont_path}")

    print(f"\n{current_streak_summary(daily)}")


if __name__ == "__main__":
    main()
