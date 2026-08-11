"""
fetch_polygon_candles.py
Version: 1.0

Purpose:
    One-time backfill of historical minute candles (OHLC + Volume) from the
    Polygon.io aggregates API for the CandleState ETF set. Writes one CSV per
    symbol. Handles pagination (Polygon caps each response at 50,000 bars, and
    a year of 1-min RTH bars is ~73,000), free-tier rate limiting (5 requests
    per minute), and 429/5xx retries, so it can run unattended on the free
    plan.

Usage (Windows):
    set POLYGON_API_KEY=YOURKEY
    python fetch_polygon_candles.py

    The API key is read from the POLYGON_API_KEY environment variable, or can
    be pasted into API_KEY below (env var wins if both are set). Adjust the
    CONFIG block for symbols, dates, and granularity.

Output:
    OUTPUT_DIR / candles_<SYMBOL>_<MULTIPLIER>min.csv, columns:
        Timestamp, Open, High, Low, Close, Volume, VWAP, Trades
    Timestamp is US/Eastern (DST-aware), format YYYY-MM-DD HH:MM:SS. VWAP and
    Trades come free from Polygon (vw / n fields) - drop the columns later if
    you don't want them.

Design notes:
    - MULTIPLIER=1 by default: pull the finest granularity and aggregate to
      2-min yourself. When resampling, anchor to 09:30 ET (e.g. pandas
      .resample('2min', origin='start_day', offset='9h30min')) so bar
      boundaries match the CandleState session clock - a top-of-hour anchor
      phase-shifts every MACD crossover.
    - adjusted=true: split-adjusted prices. Leveraged ETFs have split
      histories, and an unadjusted seam mid-backtest silently poisons MACD
      values around the event.
    - INCLUDE_EXTENDED_HOURS=False filters to 09:30:00-15:59:59 ET (bars are
      stamped by open time, so the last RTH minute bar is 15:59). Polygon
      returns the full session otherwise.
    - Resumable: a symbol whose output CSV already exists is skipped, so a
      crashed or rate-limited run can just be restarted.
    - Free-tier pacing: REQUEST_SPACING_SEC=13 keeps under 5 requests/min
      including pagination requests. On a paid plan set it to 0.
    - Timestamps: Polygon returns epoch ms UTC; converted to US/Eastern via
      zoneinfo, so DST transitions inside the range are handled correctly.

Changelog:
    1.0 - Initial version. Paginated aggregates fetch with retry/backoff,
          RTH filter, Eastern timestamps, per-symbol CSV, resume-by-skip.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

VERSION = "1.0"

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

API_KEY = os.environ.get("POLYGON_API_KEY", "x4flepjiLfR5HOkEXUm8cjXKinw7lgMJ")

SYMBOLS = ["UPRO", "SDS", "TQQQ", "QID"]

START_DATE = "2025-01-02"
END_DATE = "2025-06-06"

MULTIPLIER = 1          # bar size in TIMESPAN units (1 = 1-minute bars)
TIMESPAN = "minute"
ADJUSTED = True         # split-adjusted - keep True for leveraged ETFs

INCLUDE_EXTENDED_HOURS = False   # False = regular trading hours only

OUTPUT_DIR = Path(r"C:\Git\CandleStateSessionAnalysis\data\polygon")

REQUEST_SPACING_SEC = 13   # free tier: 5 req/min incl. pagination. Paid: 0.
MAX_RETRIES = 5
RETRY_BASE_SLEEP = 30      # seconds; doubles per retry on 429/5xx

EASTERN = ZoneInfo("America/New_York")
RTH_START = (9, 30)
RTH_END = (16, 0)          # exclusive - last included 1-min bar opens 15:59

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start}/{end}"

# ------------------------------------------------------------------------- #
# FETCH
# ------------------------------------------------------------------------- #


def polygon_get(url: str, params: dict | None) -> dict:
    """GET with retry/backoff for 429 and transient 5xx."""
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            sleep_s = RETRY_BASE_SLEEP * (2 ** attempt)
            print(f"    HTTP {resp.status_code} - retry {attempt + 1}/"
                  f"{MAX_RETRIES} in {sleep_s}s")
            time.sleep(sleep_s)
            continue
        raise RuntimeError(
            f"Polygon returned HTTP {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(f"Gave up after {MAX_RETRIES} retries: {url}")


def fetch_symbol(symbol: str) -> pd.DataFrame:
    """Fetch all bars for one symbol across the full date range, paginated."""
    url = BASE_URL.format(symbol=symbol, mult=MULTIPLIER, span=TIMESPAN,
                          start=START_DATE, end=END_DATE)
    params = {
        "adjusted": str(ADJUSTED).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    rows = []
    page = 1
    while True:
        data = polygon_get(url, params)
        results = data.get("results", [])
        rows.extend(results)
        print(f"    page {page}: {len(results):,} bars "
              f"(total {len(rows):,})")

        next_url = data.get("next_url")
        if not next_url:
            break
        # next_url does not carry the API key; everything else is baked in.
        url, params = next_url, {"apiKey": API_KEY}
        page += 1
        time.sleep(REQUEST_SPACING_SEC)

    if not rows:
        print(f"    [WARN] No data returned for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Polygon fields: t=epoch ms (UTC, bar open), o/h/l/c, v=volume,
    # vw=vwap, n=trade count
    ts_utc = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["Timestamp"] = ts_utc.dt.tz_convert(EASTERN).dt.tz_localize(None)

    out = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Open": df["o"],
        "High": df["h"],
        "Low": df["l"],
        "Close": df["c"],
        "Volume": df["v"].astype("int64"),
        "VWAP": df.get("vw"),
        "Trades": df.get("n"),
    }).sort_values("Timestamp").reset_index(drop=True)

    if not INCLUDE_EXTENDED_HOURS:
        minutes = out["Timestamp"].dt.hour * 60 + out["Timestamp"].dt.minute
        lo = RTH_START[0] * 60 + RTH_START[1]
        hi = RTH_END[0] * 60 + RTH_END[1]
        before = len(out)
        out = out[(minutes >= lo) & (minutes < hi)].reset_index(drop=True)
        print(f"    RTH filter: {before:,} -> {len(out):,} bars")

    return out


# ------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------- #


def main():
    print(f"fetch_polygon_candles.py v{VERSION}")
    print(f"Range: {START_DATE} to {END_DATE}  |  {MULTIPLIER}-{TIMESPAN} "
          f"bars  |  adjusted={ADJUSTED}  |  RTH only="
          f"{not INCLUDE_EXTENDED_HOURS}\n")

    if API_KEY == "PASTE_KEY_HERE":
        sys.exit("No API key: set POLYGON_API_KEY or edit API_KEY in CONFIG.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(SYMBOLS):
        csv_path = OUTPUT_DIR / f"candles_{symbol}_{MULTIPLIER}min.csv"
        if csv_path.exists():
            print(f"{symbol}: {csv_path.name} already exists - skipping "
                  "(delete the file to refetch).")
            continue

        print(f"{symbol}: fetching...")
        df = fetch_symbol(symbol)
        if df.empty:
            continue

        df.to_csv(csv_path, index=False,
                  date_format="%Y-%m-%d %H:%M:%S", float_format="%.4f")
        days = df["Timestamp"].dt.normalize().nunique()
        print(f"    Saved {len(df):,} bars across {days} trading days "
              f"-> {csv_path}")

        if i < len(SYMBOLS) - 1:
            time.sleep(REQUEST_SPACING_SEC)

    print("\nDone.")


if __name__ == "__main__":
    main()
