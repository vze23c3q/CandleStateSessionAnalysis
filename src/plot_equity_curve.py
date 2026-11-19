"""
plot_equity_curve.py

Builds and plots an equity curve from a CandleState sessions CSV
(as produced by parse_sessions.py). Cumulative equity is computed as
starting TradingCapital + running sum of DayGain across sessions,
ordered by SessionStart.

Outputs:
  - A PNG chart of the equity curve
  - A CSV with SessionStart, DayGain, CumulativeGain, Equity, TargetHit

VERSION: 1.0.0

USAGE:
    python plot_equity_curve.py "C:\\Git\\CandleStateSessionAnalysis\\data\\Target\\output"
"""

from pathlib import Path
import sys
import csv
from datetime import datetime

VERSION = "1.0.0"


def load_sessions(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: datetime.fromisoformat(r["SessionStart"]))
    return rows


def build_equity_curve(rows):
    if not rows:
        raise ValueError("No session rows found in CSV.")

    starting_capital = float(rows[0]["TradingCapital"])

    dates = []
    day_gains = []
    cumulative_gains = []
    equity = []
    target_hits = []

    running_total = 0.0
    for row in rows:
        gain = float(row["DayGain"])
        running_total += gain

        dates.append(datetime.fromisoformat(row["SessionStart"]))
        day_gains.append(gain)
        cumulative_gains.append(running_total)
        equity.append(starting_capital + running_total)
        target_hits.append(row["TargetHit"])

    return {
        "starting_capital": starting_capital,
        "dates": dates,
        "day_gains": day_gains,
        "cumulative_gains": cumulative_gains,
        "equity": equity,
        "target_hits": target_hits,
    }


def write_output_csv(curve, out_csv_path):
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SessionStart", "DayGain", "CumulativeGain", "Equity", "TargetHit"])
        for i in range(len(curve["dates"])):
            writer.writerow([
                curve["dates"][i].isoformat(),
                f"{curve['day_gains'][i]:.2f}",
                f"{curve['cumulative_gains'][i]:.2f}",
                f"{curve['equity'][i]:.2f}",
                curve["target_hits"][i],
            ])


def plot_equity_curve(curve):
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = curve["dates"]
    equity = curve["equity"]
    starting_capital = curve["starting_capital"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, equity, color="#2166ac", linewidth=1.5, label="Equity")
    ax.axhline(starting_capital, color="gray", linestyle="--", linewidth=1, label="Starting Capital")

    # Shade drawdowns below the running peak
    running_peak = []
    peak = float("-inf")
    for e in equity:
        peak = max(peak, e)
        running_peak.append(peak)
    ax.fill_between(dates, equity, running_peak, where=[e < p for e, p in zip(equity, running_peak)],
                     color="#b2182b", alpha=0.15, interpolate=True, label="Drawdown")

    final_equity = equity[-1]
    total_gain = final_equity - starting_capital
    total_return_pct = (total_gain / starting_capital) * 100.0
    max_dd = min(e - p for e, p in zip(equity, running_peak))

    ax.set_title(
        f"Equity Curve  |  Start: \\${starting_capital:,.0f}  "
        f"Final: \\${final_equity:,.0f}  "
        f"Total: \\${total_gain:,.0f} ({total_return_pct:+.1f}%)  "
        f"Max DD: \\${max_dd:,.0f}",
        fontsize=11,
    )
    ax.set_xlabel("Session Date")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    #fig.savefig(out_png_path, dpi=150)
    plt.show()


def main():
    if len(sys.argv) < 1:
        print("Usage: python plot_equity_curve.py <output_directory> ")
        sys.exit(1)

    csv_path = Path(sys.argv[1], "sessions.csv") 
    csv_path = csv_path.resolve()


    #out_png_path = output_dir / "equity_curve.png"
    #out_csv_path = output_dir /     "equity_curve.csv"

    rows = load_sessions(csv_path)
    curve = build_equity_curve(rows)

    plot_equity_curve(curve)
    #write_output_csv(curve, out_csv_path)

    starting = curve["starting_capital"]
    final = curve["equity"][-1]
    print(f"Sessions: {len(rows)}")
    print(f"Starting capital: ${starting:,.2f}")
    print(f"Final equity: ${final:,.2f}")
    print(f"Total gain: ${final - starting:,.2f} ({(final - starting) / starting * 100:.2f}%)")
    #print(f"Chart written to: {out_png_path}")
    #print(f"CSV written to: {out_csv_path}")


if __name__ == "__main__":
    main()
