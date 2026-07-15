"""
fetch_datasets_bulk.py — Download multiple OHLCV datasets in one run.

Windows — run from the repository root:
    .venv\\Scripts\\python.exe ModelWorkbench\\fetch_datasets_bulk.py

Linux — run from the repository root (from the ModelWorkbench directory):
    ../.venv/bin/python fetch_datasets_bulk.py

Because this script lives inside ``ModelWorkbench``, it can import
``Learn.data.fetch_ohlcv_bulk`` directly.
"""

from __future__ import annotations

import pandas as pd

from Learn.data import fetch_ohlcv_bulk


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL_NAMES = [
    'BTCUSD',
    'EURUSD',
    'XAUUSD',
    'XAGUSD',
    'GBPUSD',
    'USDJPY',
    # 'SpotCrude',
    # 'US500',

]

NUM_CHUNKS = 52 * 3 # 3 years of weekly data, split into 1-week chunks
WEEKS_PER_CHUNK = 1
PERIOD_STR = "M1"

# When True, write one CSV per symbol to data/ using the same naming convention
# as fetch_ohlcv(). When False, return DataFrames and print a short preview.
SAVE_CSV = True


def main() -> None:
    result = fetch_ohlcv_bulk(
        symbol_names=SYMBOL_NAMES,
        num_chunks=NUM_CHUNKS,
        weeks_per_chunk=WEEKS_PER_CHUNK,
        period_str=PERIOD_STR,
        save_csv=SAVE_CSV,
    )

    if SAVE_CSV:
        print(f"Saved datasets for {len(SYMBOL_NAMES)} symbols.")
        return

    assert result is not None
    for symbol_name, df in result.items():
        time_min = df["Time"].min() if not df.empty else pd.NaT
        time_max = df["Time"].max() if not df.empty else pd.NaT
        print(f"{symbol_name}: shape={df.shape} range={time_min} -> {time_max}")
        print(df.head(3))
        print("-" * 80)


if __name__ == "__main__":
    main()
