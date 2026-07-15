r"""
visualize_target_hit.py  v1.0

Combines everything analyze_target_hit.py looks at into one picture:
direction pattern, hit rate, RealizedPL, and flip count, across ALL
sessions (both TargetHit==True and False), plotted together instead of
read off separate console tables.

Three panels:
    1. Hit rate by direction pattern -- stacked bar (hit vs miss count)
       for every pattern seen, sorted by frequency (most common first).
    2. Average RealizedPL by direction pattern -- split hit vs miss,
       colored red/green by sign, so you can see which patterns are
       reliable/profitable vs which are common-but-costly.
    3. Flip count vs RealizedPL -- scatter, colored by TargetHit, one
       point per session. Shows whether more flips trends toward worse
       outcomes, and where any "stop after N flips" idea would actually
       cut the sample.

Reuses analyze_target_hit.py's load_tables/SYMBOL_DIRECTION/INDEX_SYMBOLS
rather than duplicating them -- must be run from the same folder (src/).

USAGE
-----
    python visualize_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output"
    python visualize_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output" S
    python visualize_target_hit.py "C:\Git\CandleStateSessionAnalysis\data\MACDTarget\output" N

INDEX is optional, same as analyze_target_hit.py: "S" for S&P (UPRO/SDS),
"N" for Nasdaq (TQQQ/QID), or omit for both combined (how you actually
trade).

Saves a PNG next to the output folder and opens it in a window.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import analyze_target_hit as ath

VERSION = "1.0"


def build_full_profile(sessions: pd.DataFrame, positions: pd.DataFrame,
                        index_symbols: list = None) -> pd.DataFrame:
    """
    Same per-session profiling as analyze_target_hit.profile_target_hit_days,
    but across EVERY session regardless of TargetHit -- this needs both
    hit and miss rows in one table to plot them together.
    """
    rows = []
    for _, session in sessions.iterrows():
        source_file = session["SourceFile"]
        pos_for_day = positions[positions["SourceFile"] == source_file].copy()
        if index_symbols is not None:
            pos_for_day = pos_for_day[pos_for_day["Symbol"].isin(index_symbols)]
        pos_for_day = pos_for_day.sort_values("IndexOpen")

        if pos_for_day.empty:
            continue  # nothing in scope for this day

        symbols = pos_for_day["Symbol"].tolist()
        directions = [ath.SYMBOL_DIRECTION.get(sym, "Unknown") for sym in symbols]
        collapsed_directions = [
            d for i, d in enumerate(directions) if i == 0 or d != directions[i - 1]
        ]
        num_flips = len(collapsed_directions) - 1

        rows.append({
            "SourceFile": source_file,
            "TargetHit": bool(session["TargetHit"]),
            "PositionCount": len(pos_for_day),
            "Directions": ", ".join(collapsed_directions),
            "NumFlips": num_flips,
            "RealizedPL": pos_for_day["RealizedGain"].sum(),
        })

    return pd.DataFrame(rows)


def plot_profile(profile: pd.DataFrame, index_label: str, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    # --- Panel 1: hit rate by direction pattern (stacked bar) ---
    pattern_order = profile["Directions"].value_counts().index.tolist()
    hit_counts = []
    miss_counts = []
    for pattern in pattern_order:
        subset = profile[profile["Directions"] == pattern]
        hit_counts.append((subset["TargetHit"]).sum())
        miss_counts.append((~subset["TargetHit"]).sum())

    ax = axes[0]
    ax.bar(pattern_order, hit_counts, label="Hit", color="#2e7d32")
    ax.bar(pattern_order, miss_counts, bottom=hit_counts, label="Miss", color="#c62828")
    ax.set_title(f"Hit rate by direction pattern -- {index_label}")
    ax.set_ylabel("Session count")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    for i, pattern in enumerate(pattern_order):
        total = hit_counts[i] + miss_counts[i]
        rate = hit_counts[i] / total if total else 0
        ax.annotate(f"{rate:.0%}", (i, total), ha="center", va="bottom", fontsize=8)

    # --- Panel 2: average RealizedPL by direction pattern, hit vs miss ---
    ax = axes[1]
    width = 0.35
    x = range(len(pattern_order))
    hit_avg = []
    miss_avg = []
    for pattern in pattern_order:
        subset = profile[profile["Directions"] == pattern]
        hit_avg.append(subset[subset["TargetHit"]]["RealizedPL"].mean())
        miss_avg.append(subset[~subset["TargetHit"]]["RealizedPL"].mean())
    hit_avg = [v if pd.notna(v) else 0 for v in hit_avg]
    miss_avg = [v if pd.notna(v) else 0 for v in miss_avg]

    ax.bar([i - width / 2 for i in x], hit_avg, width, label="Hit avg PL", color="#2e7d32")
    ax.bar([i + width / 2 for i in x], miss_avg, width, label="Miss avg PL", color="#c62828")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pattern_order, rotation=30, ha="right")
    ax.set_title(f"Average RealizedPL by direction pattern -- {index_label}")
    ax.set_ylabel("Avg RealizedPL ($)")
    ax.legend()

    # --- Panel 3: flip count vs RealizedPL scatter, colored by TargetHit ---
    ax = axes[2]
    hit_rows = profile[profile["TargetHit"]]
    miss_rows = profile[~profile["TargetHit"]]
    ax.scatter(hit_rows["NumFlips"], hit_rows["RealizedPL"],
               color="#2e7d32", label="Hit", alpha=0.7)
    ax.scatter(miss_rows["NumFlips"], miss_rows["RealizedPL"],
               color="#c62828", label="Miss", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Number of direction flips that day")
    ax.set_ylabel("RealizedPL ($)")
    ax.set_title(f"Flip count vs RealizedPL -- {index_label}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.show()


def main():
    print(f"visualize_target_hit.py  v{VERSION}")

    args = sys.argv[1:]
    if len(args) < 1:
        print('Usage: python visualize_target_hit.py "<output_folder>" [S|N]')
        sys.exit(1)

    folder = Path(args[0]).resolve()

    index_symbols = None
    index_label = "All (S&P + Nasdaq)"
    if len(args) >= 2:
        index = args[1].upper()
        if index not in ath.INDEX_SYMBOLS:
            print(f'Index must be "S" or "N", got "{args[1]}"')
            sys.exit(1)
        index_symbols = ath.INDEX_SYMBOLS[index]
        index_label = f"{index} ({'/'.join(index_symbols)})"

    sessions, positions = ath.load_tables(folder)
    profile = build_full_profile(sessions, positions, index_symbols)

    output_path = folder / "target_hit_visualization.png"
    plot_profile(profile, index_label, output_path)


if __name__ == "__main__":
    main()
