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
    python analyze_stats.py "C:\Git\CandleStateSessionAnalysis\data\MACD Target\output"
    python analyze_stats.py "C:\Git\CandleStateSessionAnalysis\data\MACD Trail\output"

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

    daily = sessions[["SourceFile", "TargetHit"]].merge(daily, on="SourceFile", how="left")
    daily["DailyNet"] = daily["DailyNet"].fillna(0)
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

    return {
        "StdDev": std_dev,
        "Average": average,
        "Annualized": annualized,
        "MaxLoss": max_loss,
        "Total": total,
        "TargetHitPct": target_hit_pct,
    }


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
        "StdDev": f"{stats['StdDev']:,.0f}",
        "Average": fmt_dollars(stats["Average"]),
        "Annualized": f"{stats['Annualized']:.0%}",
        "Max Loss": fmt_dollars(stats["MaxLoss"]),
        "Total": fmt_dollars(stats["Total"]),
        "Target Hit %": f"{stats['TargetHitPct']:.0%}",
    })


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions, sessions = load_positions_and_sessions(output_dir)
    daily = daily_results(positions, sessions)
    trading_capital = get_trading_capital(sessions)

    stats = compute_stats(daily, trading_capital)

    print("--- Summary statistics ---")
    print(format_for_display(stats).to_string())


if __name__ == "__main__":
    main()
