r"""
run_endofday.py - Version 1.5.0

The end-of-day steps, in order; if a step fails, the later steps don't run:
  1. src\finish_day_candles.py for the current date (YYYYMMDD), to finish the
     day's live candle sets. Skipped when there is no "<date> Live" folder
     (the streamer didn't die), or with --skip-finish. Pass --date to finish
     a different day. If it does run, the job stops after it so the four
     strategies can be re-run in CandleState on the merged candles; then run
     it again (the Live folder was renamed, so this step is skipped).
  2. src\sync_sessions.py, to move new session files into data\<Strategy>.
  3. The notebooks, below.
  4. src\analyze_pop.py on data\MACDTarget\output, for the next session's
     POP. Skipped if the MACDTarget notebook failed in this run.

Step 3 runs notebooks\analyze_backtest.ipynb once per strategy, replacing the daily
routine of editing STRATEGY = strategies[N] and re-running the notebook by hand.

For each strategy the notebook is loaded in memory, the STRATEGY line in its
first code cell is replaced with that strategy, and every cell is executed
(working directory notebooks\, as in Jupyter). The notebook file on disk is
never modified. Each run writes its usual HTML report to data\Reports, and the
executed notebook (with all printed tables) is saved to
data\<Strategy>\output\analyze_backtest_executed.ipynb.

A failing strategy is reported and the remaining strategies still run.

Each run echoes the notebook's data quality checks (one session per day, all
signals tagged with one strategy name) and also checks that the name is the
strategy being run. Any warning is shown in the summary and makes the exit
code 1, but POP is skipped only if the MACDTarget notebook outright failed.

Requires: pip install nbclient nbformat

USAGE:
    python run_endofday.py                         # all four strategies
    python run_endofday.py SoftMarket MACDFlip     # just these
    python run_endofday.py --date 20261028         # finish a different day
    python run_endofday.py --skip-finish           # notebooks only
"""

import argparse
import re
import subprocess
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_DIR / "notebooks" / "analyze_backtest.ipynb"
DATA_DIR = REPO_DIR / "data"
FINISH_DAY_SCRIPT = REPO_DIR / "src" / "finish_day_candles.py"
SYNC_SESSIONS_SCRIPT = REPO_DIR / "src" / "sync_sessions.py"
ANALYZE_POP_SCRIPT = REPO_DIR / "src" / "analyze_pop.py"
POP_STRATEGY = "MACDTarget"  # analyze_pop.py is fit on this strategy's output
CANDLES_ROOT = Path(r"C:\ProgramData\CandleState\Candles")  # finish_day_candles default --root
STRATEGIES = ["MACDTarget", "MACDTrail", "MACDFlip", "SoftMarket"]

STRATEGY_LINE_RE = re.compile(r"^STRATEGY\s*=.*$", re.MULTILINE)

# Harmless on Windows: zmq falls back to a selector thread under the Proactor loop.
warnings.filterwarnings("ignore", message="Proactor event loop", category=RuntimeWarning)


def set_strategy(nb, strategy):
    """Replace the STRATEGY assignment in the first code cell that has one."""
    for cell in nb.cells:
        if cell.cell_type == "code" and STRATEGY_LINE_RE.search(cell.source):
            cell.source = STRATEGY_LINE_RE.sub(f'STRATEGY = "{strategy}"', cell.source, count=1)
            return
    raise ValueError(f"No 'STRATEGY = ...' line found in {NOTEBOOK.name}")


def run_script(title, script, *args):
    """Run a src script with this Python; return True if it exits 0."""
    print(f"=== {title} ===", flush=True)
    return subprocess.run([sys.executable, str(script), *args]).returncode == 0


def run_strategy(strategy):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    set_strategy(nb, strategy)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    try:
        client.execute()
    finally:
        out_dir = DATA_DIR / strategy / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_dir / "analyze_backtest_executed.ipynb")

    return check_output(nb, strategy)


def check_output(nb, strategy):
    """Echo the data quality checks and the "Wrote <report>.html" line from
    the executed notebook; return the number of warnings.

    parse_sessions prints "Data quality check passed: ..." or a "WARNING: ..."
    line followed by indented detail lines. The strategy check passes whenever
    all signals share one name, so also verify it is this strategy's name.
    """
    lines = [line for cell in nb.cells for output in cell.get("outputs", [])
             for line in output.get("text", "").splitlines()]
    warnings_found = 0
    in_warning = False
    for line in lines:
        if "WARNING" in line:
            print(f"  {line}")
            warnings_found += 1
            in_warning = True
        elif in_warning and line.startswith("  "):
            print(f"  {line}")
        else:
            in_warning = False
            if line.startswith("Data quality check") or (line.startswith("Wrote ") and line.endswith(".html")):
                print(f"  {line}")

    expected = f"Data quality check passed: all trade signals are {strategy}."
    if not any(line.startswith("Data quality check passed: all trade signals are") for line in lines):
        print("  WARNING: no strategy-name check in the notebook output.")
        warnings_found += 1
    elif expected not in lines:
        print(f"  WARNING: trade signals are not tagged {strategy} (see above).")
        warnings_found += 1
    return warnings_found


def main():
    parser = argparse.ArgumentParser(description="Run analyze_backtest.ipynb for each strategy.")
    parser.add_argument("strategies", nargs="*", choices=STRATEGIES, metavar="STRATEGY",
                        help=f"strategies to run (default: all of {', '.join(STRATEGIES)})")
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"),
                        help="day to finish as YYYYMMDD (default: today)")
    parser.add_argument("--skip-finish", action="store_true",
                        help="don't run finish_day_candles.py first")
    args = parser.parse_args()

    if args.skip_finish:
        pass
    elif not (CANDLES_ROOT / f"{args.date} Live").is_dir():
        print(f"=== finish_day_candles {args.date} ===")
        print(f"  No '{args.date} Live' folder; skipped.", flush=True)
    elif not run_script(f"finish_day_candles {args.date}", FINISH_DAY_SCRIPT, args.date):
        print("finish_day_candles.py failed; later steps not run.")
        sys.exit(1)
    else:
        # The strategies must be re-run in CandleState on the merged candles
        # before their session files are worth syncing.
        print("\nLive candles finished. Run the four strategies in CandleState, then run")
        print("run_endofday.py again (the Live folder is gone, so this step is skipped).")
        sys.exit(0)

    if not run_script("sync_sessions", SYNC_SESSIONS_SCRIPT):
        print("sync_sessions.py failed; notebooks not run.")
        sys.exit(1)
    print()

    results = []
    for strategy in args.strategies or STRATEGIES:
        print(f"=== {strategy} ===", flush=True)
        start = time.time()
        try:
            warning_count = run_strategy(strategy)
            status = f"WARNING: {warning_count} data quality issue(s)" if warning_count else "OK"
        except CellExecutionError as e:
            # First line of the traceback summary is enough here; the full
            # error is in the saved executed notebook.
            status = "FAILED: " + str(e).strip().splitlines()[-1]
        except Exception as e:
            status = f"FAILED: {e}"
        elapsed = time.time() - start
        print(f"  {status} ({elapsed:.0f}s)", flush=True)
        results.append((strategy, status))

    # POP reads MACDTarget's notebook output, so skip it if that notebook just failed.
    if dict(results).get(POP_STRATEGY, "OK").startswith("FAILED"):
        results.append(("POP", f"SKIPPED: {POP_STRATEGY} failed"))
    else:
        print()
        ok = run_script("analyze_pop", ANALYZE_POP_SCRIPT, str(DATA_DIR / POP_STRATEGY / "output"))
        results.append(("POP", "OK" if ok else "FAILED"))

    print("\n=== Summary ===")
    for strategy, status in results:
        print(f"  {strategy:<12} {status}")
    sys.exit(0 if all(s == "OK" for _, s in results) else 1)


if __name__ == "__main__":
    main()
