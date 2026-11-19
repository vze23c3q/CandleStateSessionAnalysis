r"""
analyze_pop.py

Probability-of-profit (POP) for the NEXT session, from one pre-open input:
the PRIOR day's LineChart leg count (CompletedLegCount, 15-bps zigzag),
averaged over UPRO and TQQQ.

WHY THIS FEATURE
----------------
Leg count is effectively an intraday swing/volatility measure: with ~191
two-minute bars per day, more completed legs = more 15-bps swings
(r = 0.85 vs. summed |2-min returns|). Volatility clusters day to day
(autocorrelation ~0.5), and Target earns its money on days that move.
So yesterday's leg count is a cheap, reactive read on whether today is
likely to have enough movement to reach target.

EVIDENCE (Oct 2026 study, see claude/PriorDayLegs_POP_Finding.md)
    Spearman(prior-day legs, next-day Net):  2026 +0.29 (p<0.001)
                                             2025 +0.21 (p<0.001)  <- out of sample
    Day-of-week, streak, trailing win rate: no out-of-sample value
    (the Tue/Thu/Fri edge in 2026 reversed in 2025).

Calibration (expanding walk-forward over 2025-2026):
    predicted 45% -> actual 49%   predicted 64% -> actual 66%
    predicted 55% -> actual 58%   predicted 73% -> actual 79%

USAGE
-----
    python analyze_pop.py "C:\Git\CandleStateSessionAnalysis\data\Target\output"

Uses line_chart_summary.csv if it exists; otherwise recomputes leg counts
from trade_signals.csv with a port of LineChart.AddCandle (verified to
match the engine's CompletedLegCount exactly on all 2026 symbol-days).
Pass several output folders to fit on all of them (e.g. 2025 + 2026).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEG_SYMBOLS = ["UPRO", "TQQQ"]
THRESHOLD_BIPS = 15
BUCKETS = [0, 36, 40, 45, 50, 1000]


# ---------------------------------------------------------------- legs
def _count_legs(closes, th=THRESHOLD_BIPS) -> int:
    """Port of LineChart.AddCandle reversal logic (structural 15 bps from the
    leg extreme, or behavioral 2 consecutive opposite closes)."""
    active, direction, completed = [], 0, 0

    def infer():
        if len(active) < 2:
            return 0
        f, l = active[0], active[-1]
        return 1 if l > f else (-1 if l < f else 0)

    for c in closes:
        if not active:
            active.append(c)
            continue
        active.append(c)
        if direction == 0:
            direction = infer()
            continue
        rev = (direction == 1 and c < max(active) * (1 - th / 1e4)) or \
              (direction == -1 and c > min(active) * (1 + th / 1e4))
        if not rev:
            opp, p = 0, c
            for prior in reversed(active[:-1]):
                if (direction == 1 and p < prior) or (direction == -1 and p > prior):
                    opp += 1
                else:
                    break
                if opp >= 2:
                    rev = True
                    break
                p = prior
        if rev:
            ext = max(active) if direction == 1 else min(active)
            ei = len(active) - 1 - active[::-1].index(ext)
            active = active[ei:]          # old and new leg share the extreme
            completed += 1
            direction = infer()
    return completed


def daily_legs(output_dir: Path) -> pd.DataFrame:
    """One row per session date: Legs = mean CompletedLegCount over LEG_SYMBOLS."""
    lcs = output_dir / "line_chart_summary.csv"
    if lcs.exists():
        df = pd.read_csv(lcs)
        df["Date"] = pd.to_datetime(df["SessionStart"].str[:10])
        df = df.rename(columns={"CompletedLegCount": "Legs"})
    else:
        ts = pd.read_csv(output_dir / "trade_signals.csv",
                         usecols=["CandleOpen", "Symbol", "Index", "Close", "IsBootstrap", "SourceFile"])
        ts = ts[~ts["IsBootstrap"].astype(bool) & ts["Symbol"].isin(LEG_SYMBOLS)]
        ts = ts.sort_values(["SourceFile", "Symbol", "Index"]).drop_duplicates(
            ["SourceFile", "Symbol", "Index"], keep="last")
        ts["Date"] = pd.to_datetime(ts["CandleOpen"].str[:10])
        df = (ts.groupby(["Date", "Symbol"])["Close"]
                .apply(lambda s: _count_legs(s.to_list())).rename("Legs").reset_index())
    df = df[df["Symbol"].isin(LEG_SYMBOLS)]
    return df.groupby("Date", as_index=False)["Legs"].mean()


def daily_results(output_dir: Path) -> pd.DataFrame:
    """Daily Net / Win / TargetHit joined to the PRIOR day's legs."""
    sessions = pd.read_csv(output_dir / "sessions.csv")
    positions = pd.read_csv(output_dir / "positions.csv")
    net = positions.groupby("SourceFile", as_index=False)["RealizedGain"].sum()
    d = sessions[["SourceFile", "SessionStart", "TargetHit"]].merge(net, on="SourceFile", how="left")
    d["Net"] = d["RealizedGain"].fillna(0)
    d["Date"] = pd.to_datetime(d["SessionStart"].str[:10])
    d["Win"] = d["Net"] >= 0            # same definition as Breakeven-or-Better
    d["TargetHit"] = d["TargetHit"].astype(bool)
    d = d.merge(daily_legs(output_dir), on="Date", how="left").sort_values("Date")
    d["PriorLegs"] = d["Legs"].shift(1)  # known before today's open
    return d[["Date", "Net", "Win", "TargetHit", "Legs", "PriorLegs"]].reset_index(drop=True)


# ---------------------------------------------------------------- model
def _fit_logistic(x, y, iters=50):
    """One-feature logistic regression via Newton's method (no sklearn needed)."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        b += np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
    return b


def fit_pop(history: pd.DataFrame) -> dict:
    h = history.dropna(subset=["PriorLegs"])
    x = h["PriorLegs"].to_numpy(float)
    return {
        "Win": _fit_logistic(x, h["Win"].to_numpy(float)),
        "TargetHit": _fit_logistic(x, h["TargetHit"].to_numpy(float)),
        "BaseWin": h["Win"].mean(),
        "BaseTargetHit": h["TargetHit"].mean(),
        "N": len(h),
    }


def predict(model: dict, legs: float) -> dict:
    sig = lambda b: 1 / (1 + np.exp(-(b[0] + b[1] * legs)))
    return {"Win": sig(model["Win"]), "TargetHit": sig(model["TargetHit"])}


def bucket_table(history: pd.DataFrame) -> pd.DataFrame:
    h = history.dropna(subset=["PriorLegs"]).copy()
    h["PriorLegs"] = pd.cut(h["PriorLegs"], BUCKETS, labels=["<36", "36-40", "40-45", "45-50", "50+"])
    return h.groupby("PriorLegs", observed=True).agg(
        Days=("Net", "size"), WinPct=("Win", "mean"),
        TargetHitPct=("TargetHit", "mean"), AvgNet=("Net", "mean"))


def current_pop_summary(history: pd.DataFrame, model: dict) -> str:
    """Report line for the NEXT session, using the most recent day's legs."""
    legs = history["Legs"].dropna().iloc[-1]
    p = predict(model, legs)
    flag = "  Below-average setup (low prior-day swing count)." if legs < 36 else ""
    return (f"**Next-session POP:** {p['Win']:.0%} breakeven-or-better, "
            f"{p['TargetHit']:.0%} target hit "
            f"(vs {model['BaseWin']:.0%} / {model['BaseTargetHit']:.0%} unconditional; "
            f"prior-day legs {legs:.1f}).{flag}")


def main():
    dirs = [Path(a).resolve() for a in sys.argv[1:]] or [Path("output").resolve()]
    history = pd.concat([daily_results(d) for d in dirs]).sort_values("Date").reset_index(drop=True)
    model = fit_pop(history)

    print(f"--- POP by prior-day legs ({model['N']} days) ---")
    t = bucket_table(history)
    print(t.to_string(formatters={"WinPct": "{:.0%}".format, "TargetHitPct": "{:.0%}".format,
                                  "AvgNet": lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"}))
    print()
    print(current_pop_summary(history, model).replace("**", ""))


if __name__ == "__main__":
    main()
