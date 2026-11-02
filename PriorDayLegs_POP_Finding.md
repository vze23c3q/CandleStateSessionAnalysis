# Prior-Day Legs as a Next-Session POP Signal (MACDTarget) — 2026-10-28

## Question
Can a pre-open probability-of-profit be derived (originally: from current streak × day-of-week)?

## Data
MACDTarget backtests: 2025 (245 days, Jan 6–Dec 31) and 2026 (198 days, Jan 2–Oct 27). Win = Net >= 0 (Breakeven-or-Better).

## Findings
1. **Day-of-week does not hold out of sample.** 2026: Tue/Thu/Fri +$13,192, Mon/Wed −$742. 2025: Tue/Thu/Fri **−$287**, Mon/Wed **+$1,297** (Tue and Thu were the two worst days). Adding a TTF flag to the legs model worsened 2026→2025 Brier (0.2391 → 0.2422).
2. **Streak, prior-day result, prior-day position count, trailing 5/10/15/20-day win/hit/net rates:** |ρ| ≤ 0.13 with next-day Net in both years and inconsistent. No usable signal.
3. **Prior-day CompletedLegCount (LineChart, 15 bps, avg of UPRO+TQQQ) predicts next-day results in both years.**
   - Spearman vs next-day Net: 2026 +0.29 (p<0.001), 2025 +0.21 (p=0.0008, out of sample). Holds in both halves of each year and within months (not a calendar confound).
   - Legs = price swings (reversals of 15 bps or 2 opposite closes), not positions.
   - Python port of LineChart.AddCandle reproduces engine CompletedLegCount exactly (792/792 symbol-days), so 2025 legs were computed from trade_signals.

| Prior-day legs | Days | Win% | TargetHit% | Avg Net |
|---|---|---|---|---|
| <36 | 90 | 46% | 23% | −$45 |
| 36–40 | 85 | 60% | 45% | $40 |
| 40–45 | 83 | 54% | 43% | $5 |
| 45–50 | 65 | 65% | 60% | $59 |
| 50+ | 118 | 71% | 66% | $79 |

## Volatility, not chop
Legs track *amount* of movement (r = 0.86–0.91 vs path length Σ|2-min returns|) and are nearly uncorrelated with *efficiency* (range ÷ path; low = whipsaw, high = trend; r ≈ −0.1).
- Same-day whipsaw does hurt: at every activity level, trend days beat whipsaw days (2025 active: −$81 whipsaw vs +$149 trend).
- But yesterday's efficiency does not predict today's (ρ ≈ −0.04 both years); yesterday's movement does (ρ +0.22 to +0.29).
- Conclusion: pre-open you can know quiet vs active, not trend vs whipsaw. Direction must come from intraday reaction. Consistent with the August MACD phase-flip finding (intraday whipsaw, not forecastable).

## POP model
One-feature logistic on prior-day legs (2025+2026 pooled): Win b0=−1.439, b1=0.0413; TargetHit b0=−3.047, b1=0.0665.
Cross-year Brier improvement vs constant: Win +3–4%, TargetHit +8–9%. Walk-forward calibration good (pred 45%→49% actual, 73%→79%).

## Capital sizing (the one daily decision; single strategy)
Assumes P&L scales with capital.

| Schedule | 2025 net | 2025 max DD | 2026 net | 2026 max DD |
|---|---|---|---|---|
| Full every day | $729 | −$9,512 | $12,450 | −$4,597 |
| **Skip if legs < 36** | **$2,494** | **−$7,864** | **$14,961** | **−$3,564** |
| Half size if < 36 | $1,612 | −$8,533 | $13,705 | −$4,079 |
| Half < 36, 1.5× if 50+ | $2,333 | −$10,321 | $17,982 | −$4,170 |
| Linear 0.5×–1.5× | $1,992 | −$10,402 | $18,072 | −$3,908 |

- **Skip < 36** is the only schedule that improves both net and drawdown in both years; trades ~80–85% of days.
- Largest skipped winning day only $305–$554 — rarely idles through a big runner.
- Sizing up on 50+ adds dollars in 2026 but worsens 2025 drawdown (50+ bucket only $22/day in 2025) — not recommended.
- Threshold is a zone (36–38), not a tuned value; dollar sweep is noisy outside it.

## Daily routine
Before the open, take yesterday's UPRO/TQQQ average CompletedLegCount. Below 36 → sit out; otherwise run full capital.

## Status / caveats
- 2025 sessions were generated with the Aug-2026 build; strategy version differences between years are possible.
- Half-size positions may hit stops/targets differently than full size (not modeled).
- Not yet tested: entry location within the day's running range (buying near HOD), split by legs bucket; opening-range bips (2026-only, ρ=0.15).
- Script: `src/analyze_pop.py` (prints bucket table + next-session POP line; accepts multiple output folders).
