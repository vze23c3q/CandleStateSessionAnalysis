r"""
analyze_pair.py  v1.4

Candle-by-candle trace of DayGainLoss across every symbol in one or more
session files, in TRUE chronological order -- not grouped by symbol.

DayGainLoss is a shared/portfolio-level field (see OrderManager.
UpsertPositionList: DayGainLoss = Positions.Sum(NetGain)), so its real
behavior only makes sense viewed as one interleaved timeline across every
symbol, not as separate per-symbol trajectories. This script builds that
timeline so you can see, tick by tick:
    - which symbol's candle triggered each DayGainLoss value
    - whether the value actually changed from the row before it
    - whether that symbol had shares open at that moment (Shares column)

This is meant to be iterated on -- first pass just gets the raw sequence
in front of you. Tell me what's confusing or missing and we'll adjust.

UNLIKE the other analyze_*.py scripts, this one does NOT use
parse_sessions.py's output. It parses the raw session JSON files itself,
directly in memory, and only pulls the 3 sections needed for this check
(TradeSignals, Transactions, Positions) -- skipping Logs, PriceLevels,
Orders, and Sessions entirely, since Logs especially is slow to parse and
isn't needed here. Nothing gets written to disk; this is meant to be
cheap enough to just rerun each time.

USAGE
-----
    python analyze_pair.py "C:\Git\CandleStateSessionAnalysis\data\recon"

(pass the folder containing the raw session .json files directly -- NOT
a parse_sessions.py output folder)
"""

import sys
from pathlib import Path

import pandas as pd

# Bump this on every change. Printed at the top of every run's output so
# it's obvious in the console/output window whether you're looking at
# results from the current version of the script or a stale one.
VERSION = "1.4"

# Reuse the enum-conversion table and JSON loader from parse_sessions.py
# rather than duplicating them. Must be run from the same folder (src/).
import parse_sessions as ps

# Only the 3 sections needed for this check -- Logs is deliberately
# skipped (it's the slowest table by far, thousands of rows per file, and
# not needed for a DayGainLoss trace).
SECTIONS_NEEDED = {
    "TradeSignals": "trade_signals",
    "Transactions": "transactions",
    "Positions": "positions",
}


def parse_light(folder: Path) -> dict:
    """
    Same row-building logic as parse_sessions.py's parse_sessions(), but
    limited to SECTIONS_NEEDED and with no Sessions/Logs tables built.
    Returns {"trade_signals": df, "transactions": df, "positions": df,
    "session_meta": df}. session_meta is one row per file with the
    account-level fields needed to compare accounts on equal footing --
    TradingCapital and DayTarget differ per account, but DayTarget is
    set as a fixed % of TradingCapital, so TargetRatio (DayTarget /
    TradingCapital) is the number that's actually comparable across
    accounts of different sizes.
    """
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {folder}")

    section_rows = {name: [] for name in SECTIONS_NEEDED.values()}
    meta_rows = []

    for filepath in json_files:
        data = ps.load_session_file(filepath)

        trading_capital = data.get("TradingCapital")
        day_target = data.get("DayTarget")
        meta_rows.append({
            "SourceFile": filepath.name,
            "AccountName": data.get("AccountName"),
            "TradingCapital": trading_capital,
            "DayTarget": day_target,
            "TargetRatio": (day_target / trading_capital)
                           if trading_capital else None,
        })

        file_counts = {}
        for json_key, table_name in SECTIONS_NEEDED.items():
            records = data.get(json_key, [])
            file_counts[table_name] = len(records)
            for record in records:
                row = dict(record)
                row["SourceFile"] = filepath.name
                row["SessionStart"] = data.get("SessionStart")
                row["AccountName"] = data.get("AccountName")

                for column, names in ps.ENUM_COLUMNS.items():
                    if column[0] != table_name or column[1] not in row:
                        continue
                    ordinal = row[column[1]]
                    if isinstance(ordinal, int) and 0 <= ordinal < len(names):
                        row[column[1]] = names[ordinal]

                # Position.AvgPrice gets nulled out once Quantity hits 0
                # (position closed), so recompute it from BuyTransactions
                # instead -- a quantity-weighted average across every buy
                # (most positions have just one, but adds create more).
                #
                # Also compute ClosePrice here, from the nested
                # SellActivity within each buy lot (FIFO fills against that
                # lot) -- this is the RAW fill price, matching the
                # Description string, as opposed to ClosePriceTrx (added
                # below from Transactions.csv-equivalent data), which is
                # back-derived from NetAmount/CommissionFee and can differ
                # by a fraction of a cent. Both are quantity-weighted
                # averages in case of multiple partial sells.
                if table_name == "positions" and "BuyTransactions" in row:
                    buys = row["BuyTransactions"] or []
                    total_qty = sum(b.get("BuyQuantity", 0) for b in buys)
                    total_cost = sum(
                        b.get("BuyQuantity", 0) * b.get("Price", 0) for b in buys
                    )
                    row["AvgPrice"] = total_cost / total_qty if total_qty else None

                    sell_qty_total = 0
                    sell_cost_total = 0
                    for b in buys:
                        for s in (b.get("SellActivity") or []):
                            qty = s.get("SellQty", 0)
                            sell_qty_total += qty
                            sell_cost_total += qty * s.get("Price", 0)
                    row["ClosePrice"] = (
                        sell_cost_total / sell_qty_total if sell_qty_total else None
                    )

                    del row["BuyTransactions"]  # nested list, not needed beyond this

                section_rows[table_name].append(row)

        print(f"Parsed {filepath.name}: "
              + ", ".join(f"{table}={count}" for table, count in file_counts.items()))

    tables = {name: pd.DataFrame(rows) for name, rows in section_rows.items()}
    tables["session_meta"] = pd.DataFrame(meta_rows)

    # ClosePrice per position, pulled from the matching SELL transaction(s)
    # (Quantity < 0) rather than from BuyTransactions' SellActivity -- a
    # position can be trimmed in more than one sell, so this is a
    # quantity-weighted average across all of them, same approach as
    # AvgPrice above.
    positions_df = tables["positions"]
    transactions_df = tables["transactions"]
    if not positions_df.empty and not transactions_df.empty:
        sells = transactions_df[transactions_df["Quantity"] < 0].copy()
        sells["AbsQty"] = sells["Quantity"].abs()
        sells["Cost"] = sells["AbsQty"] * sells["Price"]
        close_totals = sells.groupby(["SourceFile", "PositionID"], as_index=False).agg(
            TotalQty=("AbsQty", "sum"), TotalCost=("Cost", "sum")
        )
        close_totals["ClosePriceTrx"] = close_totals["TotalCost"] / close_totals["TotalQty"]

        positions_df = positions_df.merge(
            close_totals[["SourceFile", "PositionID", "ClosePriceTrx"]],
            left_on=["SourceFile", "ID"], right_on=["SourceFile", "PositionID"],
            how="left", suffixes=("", "_fromTrx"),
        )
        tables["positions"] = positions_df

    return tables


def build_single_position_timeline(trade_signals: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """
    Candle-by-candle timeline for exactly one position (one row from the
    positions table). Pulled out of build_position_timelines() so the
    same logic can be reused for cross-account comparisons, where we're
    no longer looping positions within a single file.
    """
    df = trade_signals[
        (trade_signals["SourceFile"] == pos["SourceFile"])
        & (trade_signals["Symbol"] == pos["Symbol"])
        & (trade_signals["Index"] >= pos["IndexOpen"])
        & (trade_signals["Index"] <= pos["IndexClose"])
    ].copy()
    df = df.sort_values("CandleOpen").reset_index(drop=True)

    # The closing candle's row comes back with PositionID/Shares as
    # NaN -- the position record zeroes Quantity out once IsOpen goes
    # False, and trade_signals mirrors that. But the close row is
    # exactly the one we care about most (it's the candle that
    # tripped the exit), so backfill both:
    #   - PositionID is constant for every row in this loop anyway
    #   - Shares doesn't change while a position is open, so ffill
    #     just carries the last real value onto the close row
    df["PositionID"] = pos["ID"]
    df["Shares"] = df["Shares"].ffill()

    # AvgPrice is constant for the life of the position (it's a
    # position-level field, not a per-candle one), so this broadcasts
    # the same value down every row of the timeline.
    df["Unrealized"] = (df["Close"] - pos["AvgPrice"]) * df["Shares"]

    return df


def build_position_timelines(trade_signals: pd.DataFrame, positions: pd.DataFrame,
                              source_file: str) -> list:
    """
    One timeline per position (not one big whole-day timeline). Each
    position row defines the range to show: its own Symbol, from
    IndexOpen to IndexClose. Loops because IndexOpen/IndexClose are
    per-position -- a session can hold multiple positions (even multiple
    for the same symbol, if it's entered more than once in a day), each
    with its own open/close range.

    Returns a list of (position_row, timeline_df) tuples, one per
    position, in the order positions were opened.
    """
    pos_for_file = positions[positions["SourceFile"] == source_file].copy()
    pos_for_file = pos_for_file.sort_values("IndexOpen")

    results = []
    for _, pos in pos_for_file.iterrows():
        df = build_single_position_timeline(trade_signals, pos)

        cols = ["AccountName", "Symbol", "PositionID", "Shares", "Unrealized",
                "CandleOpen", "Index", "Close", "DayGainLoss"]
        cols = [c for c in cols if c in df.columns]
        results.append((pos, df[cols]))

    return results


def build_cross_account_comparisons(trade_signals: pd.DataFrame, positions: pd.DataFrame) -> list:
    """
    Finds positions that opened on the SAME candle Index but in
    different accounts (i.e. the same signal fired two live sessions at
    once), and merges each such group's per-position timelines into one
    interleaved table sorted by Index. This is what surfaces cases like
    RM2DR/4CQ1S -- opened together, but one account's exit fired well
    before the other's.

    Returns a list of (symbol, index_open, early_close_index, group_df)
    tuples, one per matched group, in Index order. group_df is trimmed
    down to just the early_close_index candle -- the moment the first
    account's exit fired -- not the full open-to-close span.
    """
    groups = positions.groupby(["Symbol", "IndexOpen"])

    results = []
    for (symbol, index_open), group in groups:
        account_count = group["SourceFile"].nunique()
        if account_count < 2:
            continue  # only one account opened this symbol at this Index -- nothing to compare

        frames = [build_single_position_timeline(trade_signals, pos)
                  for _, pos in group.iterrows()]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["Index", "AccountName"]).reset_index(drop=True)

        # The moment we actually care about is the FIRST account's exit --
        # whatever candle tripped the earliest close in the group. Trim
        # every account's timeline down to just that one Index so the
        # comparison is "what did everyone's numbers look like at the
        # instant the early exit fired," not the full open-to-close span.
        early_close_index = group["IndexClose"].min()
        combined = combined[combined["Index"] == early_close_index]

        cols = ["Index", "CandleOpen", "AccountName", "PositionID", "Shares",
                "Unrealized", "Close", "DayGainLoss"]
        cols = [c for c in cols if c in combined.columns]
        results.append((symbol, index_open, early_close_index, combined[cols]))

    return results


def find_missed_targets(trade_signals: pd.DataFrame, positions: pd.DataFrame,
                         session_meta: pd.DataFrame) -> list:
    """
    Per account, walks every candle where a position was actually OPEN
    (built the same way as everywhere else in this script -- each
    position's own Symbol/IndexOpen/IndexClose slice, via
    build_single_position_timeline -- then combined and put in TRUE
    chronological order by CandleOpen), and flags the moments
    DayGainLoss / TradingCapital crossed UP over that account's
    TargetRatio (DayTarget / TradingCapital -- the number that's
    actually comparable across accounts of different sizes) without a
    position closing on that same candle. That's a "missed" target: the
    aggregate genuinely crossed the line, but the rule engine's
    completed-candle evaluation didn't act on it -- usually because the
    crossing was a momentary mid-candle spike that had already receded
    by the next candle-roll.

    Restricting to open-position candles (rather than every trade_signal
    row for the account) matters here: DayGainLoss is written onto every
    symbol's tick regardless of whether that symbol has a position open,
    so a flat/idle symbol sitting on the sidelines would otherwise show
    up as a duplicate "miss" just for riding along on the same shared
    portfolio value as whatever symbol actually triggered it.

    Only the rising edge of each crossing is flagged (the first tick
    that goes over, not every tick that stays over), since a target
    that's been legitimately sitting above threshold for many candles
    isn't a new miss each tick.

    "Did a position close on this candle" is checked per-symbol (the
    row's own Symbol + Index against that symbol's own IndexClose), so a
    different symbol closing at the same Index number can't wrongly
    explain away this row.

    Returns a list of (source_file, account_name, target_ratio, missed_df)
    tuples, one per file that has at least one miss.
    """
    results = []
    for _, meta in session_meta.iterrows():
        source_file = meta["SourceFile"]
        target_ratio = meta["TargetRatio"]
        if target_ratio is None:
            continue

        pos_for_file = positions[positions["SourceFile"] == source_file]
        if pos_for_file.empty:
            continue

        frames = [build_single_position_timeline(trade_signals, pos)
                  for _, pos in pos_for_file.iterrows()]
        df = pd.concat(frames, ignore_index=True)
        df["CandleOpen"] = pd.to_datetime(df["CandleOpen"])
        df = df.sort_values("CandleOpen").reset_index(drop=True)
        df["Ratio"] = df["DayGainLoss"] / meta["TradingCapital"]

        closed_pairs = set(zip(pos_for_file["Symbol"], pos_for_file["IndexClose"].dropna()))
        closed_this_candle = df.apply(
            lambda row: (row["Symbol"], row["Index"]) in closed_pairs, axis=1
        )

        over_target = df["Ratio"] > target_ratio
        rising_edge = over_target & ~over_target.shift(1, fill_value=False)
        missed = df[rising_edge & ~closed_this_candle].copy()

        if missed.empty:
            continue

        cols = ["Index", "CandleOpen", "Symbol", "PositionID", "DayGainLoss",
                "Ratio", "AccountName"]
        cols = [c for c in cols if c in missed.columns]
        results.append((source_file, meta["AccountName"], target_ratio, missed[cols]))

    return results




def print_with_repeated_headers(df: pd.DataFrame, chunk_size: int = 40):
    """
    Prints a DataFrame in chunks, reprinting the column header before each
    chunk -- otherwise the header scrolls off screen almost immediately on
    a long table and every row past the first screenful is unlabeled.
    """
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        print(chunk.to_string(index=False, header=True))
        print()


def main():
    print(f"analyze_pair.py  v{VERSION}")

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    folder = folder.resolve()

    tables = parse_light(folder)
    trade_signals = tables["trade_signals"]
    positions = tables["positions"]
    session_meta = tables["session_meta"]
    source_files = trade_signals["SourceFile"].unique()

    print(f"\nFound {len(source_files)} session file(s) in {folder}\n")

    for source_file in source_files:
        print(f"=== {source_file} ===\n")

        position_timelines = build_position_timelines(trade_signals, positions, source_file)
        if not position_timelines:
            print("No positions found in this session.\n")
            continue

        for pos, timeline in position_timelines:
            total = len(timeline)
            print(f"--- Position {pos.get('ID', '?')}: {pos['Symbol']}  "
                  f"Index {pos['IndexOpen']}-{pos['IndexClose']}  "
                  f"AvgPrice={pos.get('AvgPrice', 0):.4f}  "
                  f"ClosePrice={pos.get('ClosePrice', 0):.4f}  "
                  f"RealizedGain={pos.get('RealizedGain', 0):.2f} ---")
            print(f"{total} candles\n" if total else "No matching candles found.\n")
            print_with_repeated_headers(timeline)

    cross_account = build_cross_account_comparisons(trade_signals, positions)
    if cross_account:
        print("\n=== Cross-account comparisons (same symbol, same entry Index) ===\n")
        for symbol, index_open, early_close_index, comparison in cross_account:
            accounts = comparison["AccountName"].unique()
            print(f"--- {symbol}  opened at Index {index_open}  "
                  f"early exit at Index {early_close_index}  "
                  f"({len(accounts)} accounts: {', '.join(accounts)}) ---")
            print_with_repeated_headers(comparison)

    missed_targets = find_missed_targets(trade_signals, positions, session_meta)
    if missed_targets:
        print("\n=== Missed targets (DayGainLoss/TradingCapital exceeded "
              "TargetRatio but nothing closed that candle) ===\n")
        for source_file, account_name, target_ratio, missed in missed_targets:
            print(f"--- {source_file}  {account_name}  TargetRatio={target_ratio:.4f} ---")
            print_with_repeated_headers(missed)


if __name__ == "__main__":
    main()
