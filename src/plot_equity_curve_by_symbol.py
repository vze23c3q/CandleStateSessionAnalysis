"""
plot_equity_curve_by_symbol.py

Builds and plots a per-symbol equity curve from a CandleState positions CSV
(as produced by parse_sessions.py). Each symbol's cumulative curve is built
from RealizedGain, ordered by Closed timestamp, plotted on shared axes so
symbol contributions can be compared directly.

VERSION: 1.0.0

USAGE:python plot_equity_curve_by_symbol.py "C:\\Git\\CandleStateSessionAnalysis\\data\\Target\\output"
    
"""

from pathlib import Path
import sys
import csv
from datetime import datetime

VERSION = "1.0.0"


def load_positions(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: datetime.fromisoformat(r["Closed"]))
    return rows


def build_curves_by_symbol(rows):
    if not rows:
        raise ValueError("No position rows found in CSV.")

    symbols = sorted(set(r["Symbol"] for r in rows))

    curves = {}
    for symbol in symbols:
        sym_rows = [r for r in rows if r["Symbol"] == symbol]
        dates = []
        cumulative = []
        running_total = 0.0
        for row in sym_rows:
            running_total += float(row["RealizedGain"])
            dates.append(datetime.fromisoformat(row["Closed"]))
            cumulative.append(running_total)
        curves[symbol] = {"dates": dates, "cumulative": cumulative}

    # Combined (all symbols), ordered by Closed across all rows
    dates = []
    cumulative = []
    running_total = 0.0
    for row in rows:
        running_total += float(row["RealizedGain"])
        dates.append(datetime.fromisoformat(row["Closed"]))
        cumulative.append(running_total)
    curves["ALL"] = {"dates": dates, "cumulative": cumulative}

    return curves


SYMBOL_COLORS = {
    "UPRO": "#2166ac",
    "TQQQ": "#1a9850",
    "SDS": "#d62728",
    "QID": "#ff7f0e",
    "ALL": "#555555",
}


def plot_curves_by_symbol(curves, output_path):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for symbol, curve in curves.items():
        if symbol == "ALL":
            continue
        color = SYMBOL_COLORS.get(symbol, None)
        final = curve["cumulative"][-1]
        ax.plot(curve["dates"], curve["cumulative"], color=color, linewidth=1.3,
                label=f"{symbol} (${final:,.0f})")

    # Combined line, dashed, on top for reference
    all_curve = curves["ALL"]
    ax.plot(all_curve["dates"], all_curve["cumulative"], color=SYMBOL_COLORS["ALL"],
            linewidth=1.5, linestyle="--", label=f"ALL (${all_curve['cumulative'][-1]:,.0f})")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Cumulative Realized P/L by Symbol")
    ax.set_xlabel("Close Date")
    ax.set_ylabel("Cumulative Realized P/L ($)")
    ax.legend(title="Symbol", loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_equity_curve_by_symbol.py <output_directory>")
        sys.exit(1)

    csv_path = Path(sys.argv[1], "positions.csv").resolve()
    output_path = Path(sys.argv[1], "equity_curve_by_symbol.png").resolve()

    rows = load_positions(csv_path)
    curves = build_curves_by_symbol(rows)

    for symbol, curve in curves.items():
        final = curve["cumulative"][-1]
        print(f"{symbol}: {len(curve['dates'])} closes, final cumulative RealizedGain: ${final:,.2f}")

    plot_curves_by_symbol(curves, output_path)


if __name__ == "__main__":
    main()
