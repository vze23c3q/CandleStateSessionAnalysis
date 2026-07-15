r"""
analyze_stats.py

Summary statistics on daily P&L: StdDev, Average, Annualized (return),
Max Loss, Total, and Target Hit % (share of days that hit DayTarget).

Like analyze_DOW.py, this works on a per-DAY basis (one number per
trading day, summed across that day's positions), not per-trade.

Reads positions.csv and sessions.csv (both produced by parse_sessions.py):
positions.csv supplies each day's trades (summed into a daily P&L),
sessions.csv supplies TradingCapital (for the Annualized figure).

ANNUALIZED
----------
Simple (non-compounding) annualization:

    Annualized = (Total P&L / TradingCapital) * (252 / trading days)

252 is the standard approximate number of US trading days in a year.
TradingCapital is read from sessions.csv -- if it varies across session
files (rare, but possible if capital was adjusted mid-dataset), this uses
the first session's value and prints a warning so it doesn't silently
give a misleading number.

USAGE
-----
    python analyze_stats.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output"
    python analyze_stats.py "C:\Git\CandleStateSessionAnalysis\data\MACDTrail\output"

(pass the "output" folder that parse_sessions.py created)
"""

import sys
from pathlib import Path

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def load_positions_and_sessions(output_dir: Path):
    positions = pd.read_csv(output_dir / "positions.csv")
    sessions = pd.read_csv(output_dir / "sessions.csv")
    return positions, sessions


def daily_results(positions: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses positions down to one row per trading day (SourceFile):
    DailyNet = sum of that day's RealizedGain across all its positions.
    TargetHit comes from sessions.csv (one value per day, not summed from
    positions), same pattern as analyze_DOW.daily_results.
    """
    daily = positions.groupby("SourceFile", as_index=False)["RealizedGain"].sum()
    daily = daily.rename(columns={"RealizedGain": "DailyNet"})

    daily = sessions[["SourceFile", "SessionStart", "TargetHit"]].merge(daily, on="SourceFile", how="left")
    daily["DailyNet"] = daily["DailyNet"].fillna(0)

    daily["SessionStart"] = pd.to_datetime(daily["SessionStart"], utc=True)
    daily = daily.sort_values("SessionStart").reset_index(drop=True)
    
    daily["CumulativeNet"] = daily["DailyNet"].cumsum()
    daily["RunningPeak"] = daily["CumulativeNet"].cummax()
    daily["Drawdown"] = daily["CumulativeNet"] - daily["RunningPeak"]

    return daily


def get_trading_capital(sessions: pd.DataFrame) -> float:
    """
    Reads TradingCapital from sessions.csv. Uses the first session's value;
    warns (rather than silently guessing) if it isn't constant across the
    dataset, since that would make a single Annualized figure misleading.
    """
    capitals = sessions["TradingCapital"].dropna().unique()
    if len(capitals) > 1:
        print(f"WARNING: TradingCapital varies across sessions {sorted(capitals)} "
              f"-- using the first session's value ({capitals[0]:,.0f}) for Annualized.")
    return float(capitals[0])


def compute_stats(daily: pd.DataFrame, trading_capital: float) -> dict:
    """
    StdDev / Average / Max Loss / Total are all computed on DailyNet
    (one data point per trading day). Annualized scales the dataset's
    total return up to a full trading year using a simple (non-
    compounding) annualization -- see module docstring.
    """
    total = daily["DailyNet"].sum()
    average = daily["DailyNet"].mean()
    std_dev = daily["DailyNet"].std()
    max_loss = daily["DailyNet"].min()

    trading_days = len(daily)
    if trading_days == 0 or trading_capital == 0:
        annualized = float("nan")
    else:
        annualized = (total / trading_capital) * (TRADING_DAYS_PER_YEAR / trading_days)

    target_hit_pct = daily["TargetHit"].sum() / trading_days if trading_days else float("nan")
    win_pct = (daily["DailyNet"] >= 0).sum() / trading_days if trading_days else float("nan")
    
    winning_days = daily.loc[daily["DailyNet"] >= 0, "DailyNet"]
    losing_days = daily.loc[daily["DailyNet"] < 0, "DailyNet"]
    average_win = winning_days.mean() if len(winning_days) else float("nan")
    average_loss = losing_days.mean() if len(losing_days) else float("nan")
    
    minCumulativeNet = daily["CumulativeNet"].min()  
    maxDrawDown = daily["Drawdown"].min()  # most negative drawdown 

    return {
        "StdDev": std_dev,
        "Average": average,
        "AverageWin": average_win,
        "AverageLoss": average_loss,    
        "Annualized": annualized,
        "MaxLoss": max_loss,
        "MinCumulativeNet": minCumulativeNet,
        "MaxDrawDown": maxDrawDown,
        "Total": total,
        "TargetHitPct": target_hit_pct,
        "WinPct": win_pct,
    }


def monthly_results(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses `daily` (one row per trading day) down to one row per
    calendar month: trading day count, total P&L, days that hit target,
    and that month's worst single day.
    """
    monthly = daily.copy()
    monthly["Month"] = monthly["SessionStart"].dt.to_period("M")

    summary = monthly.groupby("Month").agg(
        Days=("DailyNet", "count"),
        MaxLoss=("DailyNet", "min"),
        TargetHit=("TargetHit", "sum"),
        Net=("DailyNet", "sum"),
    ).reset_index()
    summary["TargetHitPct"] = summary["TargetHit"] / summary["Days"]
    summary["Month"] = summary["Month"].dt.strftime("%b")

    return summary


def format_for_display(stats: dict) -> pd.Series:
    """
    Formats the stats dict for display: dollar figures as currency
    (no decimals), Annualized as a percentage (no decimals). Mirrors
    analyze_DOW.format_for_display -- the raw dict keeps full precision
    for anything downstream.
    """
    def fmt_dollars(v):
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

    return pd.Series({
        "Total P/L": fmt_dollars(stats["Total"]),
        "Annualized Yield": f"{stats['Annualized']:.0%}",
        "Breakeven-or-Better Days": f"{stats['WinPct']:.0%}",
        "Target Hit Days": f"{stats['TargetHitPct']:.0%}",
        "Daily Average": fmt_dollars(stats["Average"]),
        "Average Win": fmt_dollars(stats["AverageWin"]),
        "Average Loss": fmt_dollars(stats["AverageLoss"]),
        "Max Day Loss": fmt_dollars(stats["MaxLoss"]),
        "Min Running P/L": fmt_dollars(stats["MinCumulativeNet"]),
        "Max Drawdown (peak to trough)": fmt_dollars(stats["MaxDrawDown"]),
        "Volatility": f"{stats['StdDev']:,.0f}",
    })


def format_for_monthly_display(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the monthly summary for display: Total/MaxLoss as currency
    (no decimals). Mirrors format_for_display -- the raw DataFrame keeps
    full precision for anything downstream.
    """
    def fmt_dollars(v):
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

    display = monthly.copy()
    display["Net"] = display["Net"].apply(fmt_dollars)
    display["MaxLoss"] = display["MaxLoss"].apply(fmt_dollars)
    display["TargetHitPct"] = display["TargetHitPct"].map(lambda v: f"{v:.0%}")
    
    column_order = ["Month", "Days", "TargetHit", "TargetHitPct", "MaxLoss", "Net"]
    return display[column_order ]


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions, sessions = load_positions_and_sessions(output_dir)
    daily = daily_results(positions, sessions)
    
    #out_path = output_dir / "daily_summary.csv"
    #daily.to_csv(out_path)
    
    trading_capital = get_trading_capital(sessions)

    stats = compute_stats(daily, trading_capital)
    monthly = monthly_results(daily)

    print("--- Summary statistics ---")
    print(format_for_display(stats).to_string())

    print("--- Monthly results ---")
    print(format_for_monthly_display(monthly).to_string(index=False))


if __name__ == "__main__":
    main()
