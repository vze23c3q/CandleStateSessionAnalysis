r"""
analyze_ADX.py

Checks whether ADX (trend strength) at entry predicts outcome -- the
question behind the never-finished SignalScore idea: can entry quality be
graded well enough to set risk (position size) accordingly?

Buckets each position's entry by ADX at that bar, coarsely rather than by
precise threshold (per the same "coarse bucketing over precise thresholds"
approach used elsewhere in this project -- fine thresholds overfit a
sample this size):

    Weak      ADX < 20   (range-bound / no clear trend)
    Moderate  20-40      (developing trend)
    Strong    ADX >= 40  (strong trend)

For each bucket: Count, win rate, average win/loss, and expectancy --
same shape as the VWAP-side analysis in analyze_sessions.py, so the two
are easy to compare side by side.

WHY THIS JOIN
-------------
Same pattern as analyze_sessions.py: positions.csv's IndexOpen marks the
entry bar; trade_signals.csv has ADX for that bar. Joining
Positions.IndexOpen -> TradeSignals.Index gives one ADX-at-entry value per
position (not per partial-fill transaction).

USAGE
-----
    python analyze_ADX.py "C:\Git\CandleStateSessionAnalysis\data\MACD Target\output"
    python analyze_ADX.py "C:\Git\CandleStateSessionAnalysis\data\MACD Trail\output"

(pass the "output" folder that parse_sessions.py created)
"""

import sys
from pathlib import Path

import pandas as pd


def _parse_local_datetime(series: pd.Series) -> pd.Series:
    """
    Opened timestamps carry a UTC offset (e.g. -04:00 or -05:00) that
    changes across the Daylight Saving Time boundary. pandas can't parse a
    column with mixed offsets directly, so this normalizes everything to
    UTC first, then converts to US/Eastern so the date still reflects the
    actual local trading day.
    """
    return pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York")


def load_tables(output_dir: Path):
    positions = pd.read_csv(output_dir / "positions.csv")
    trade_signals = pd.read_csv(output_dir / "trade_signals.csv")
    return positions, trade_signals


def bucket_adx(adx: float) -> str:
    if adx < 20:
        return "Weak"
    elif adx < 40:
        return "Moderate"
    else:
        return "Strong"


def entries_with_adx(positions: pd.DataFrame, trade_signals: pd.DataFrame) -> pd.DataFrame:
    entry_bars = positions.merge(
        trade_signals[["SourceFile", "Symbol", "Index", "ADX"]],
        left_on=["SourceFile", "Symbol", "IndexOpen"],
        right_on=["SourceFile", "Symbol", "Index"],
        how="left",
    )
    entry_bars["ADXBucket"] = entry_bars["ADX"].apply(bucket_adx)
    entry_bars["Month"] = _parse_local_datetime(entry_bars["Opened"]).dt.strftime("%Y-%m")

    cols = ["SourceFile", "Symbol", "ID", "IndexOpen", "Opened", "Month",
            "ADX", "ADXBucket", "RealizedGain"]
    return entry_bars[cols].sort_values(["SourceFile", "Opened"])


def bucket_summary(entries: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Generic version of the win-rate/expectancy aggregation, grouped by
    whatever columns are passed in. Used for the overall ADX-bucket view,
    the by-symbol view, and the by-month view, so the win/loss/expectancy
    math only lives in one place.
    """
    grouped = entries.groupby(group_cols).agg(
        Count=("RealizedGain", "size"),
        Wins=("RealizedGain", lambda s: (s > 0).sum()),
        Net=("RealizedGain", "sum"),
    )
    grouped["WinRate"] = grouped["Wins"] / grouped["Count"]

    avg_win = entries[entries["RealizedGain"] > 0].groupby(group_cols)["RealizedGain"].mean()
    avg_loss = entries[entries["RealizedGain"] <= 0].groupby(group_cols)["RealizedGain"].mean().abs()
    grouped["AvgWin"] = avg_win
    grouped["AvgLoss"] = avg_loss
    grouped[["AvgWin", "AvgLoss"]] = grouped[["AvgWin", "AvgLoss"]].fillna(0)

    grouped["Expectancy"] = (
        grouped["WinRate"] * grouped["AvgWin"]
        - (1 - grouped["WinRate"]) * grouped["AvgLoss"]
    )
    return grouped


def adx_bucket_summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Same shape as win_rate_by_vwap_side in analyze_sessions.py."""
    grouped = bucket_summary(entries, ["ADXBucket"])

    # Order Weak -> Moderate -> Strong regardless of which buckets have data.
    order = ["Weak", "Moderate", "Strong"]
    grouped = grouped.reindex([b for b in order if b in grouped.index])
    return grouped


def adx_bucket_by_symbol(entries: pd.DataFrame) -> pd.DataFrame:
    """Checks whether the ADX-bucket effect holds up within each symbol,
    or is concentrated in just one or two of them."""
    return bucket_summary(entries, ["Symbol", "ADXBucket"])


def adx_bucket_by_month(entries: pd.DataFrame) -> pd.DataFrame:
    """Checks whether the ADX-bucket effect is stable over time, or
    concentrated in a particular stretch of the dataset."""
    return bucket_summary(entries, ["Month", "ADXBucket"])


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    output_dir = output_dir.resolve()

    positions, trade_signals = load_tables(output_dir)
    entries = entries_with_adx(positions, trade_signals)

    unmatched = entries["ADX"].isna().sum()
    if unmatched:
        print(f"WARNING: {unmatched} positions had no matching ADX row (check the join).")

    print("--- Performance by ADX bucket at entry ---")
    summary = adx_bucket_summary(entries)
    print(summary.to_string())

    out_path = output_dir / "adx_bucket_summary.csv"
    summary.to_csv(out_path)
    print(f"\nWrote {out_path}")

    detail_path = output_dir / "entries_with_adx.csv"
    entries.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path}")

    print("\n--- Performance by ADX bucket, split by symbol ---")
    by_symbol = adx_bucket_by_symbol(entries)
    print(by_symbol.to_string())
    by_symbol_path = output_dir / "adx_bucket_by_symbol.csv"
    by_symbol.to_csv(by_symbol_path)
    print(f"\nWrote {by_symbol_path}")

    print("\n--- Performance by ADX bucket, split by month ---")
    by_month = adx_bucket_by_month(entries)
    print(by_month.to_string())
    by_month_path = output_dir / "adx_bucket_by_month.csv"
    by_month.to_csv(by_month_path)
    print(f"\nWrote {by_month_path}")


if __name__ == "__main__":
    main()
