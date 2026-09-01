"""
backfill_underlyings.py — one-time (idempotent) 2-year OHLC/RV backfill for
the whole UNIVERSE, feeding signals.rv_percentile()'s MIN_RV_OBS gate.

underlying_history.csv is explicitly backfillable (AGENT_CONTEXT §5) —
unlike chain_history.csv, which is same-session only and can never be
backfilled. Safe to re-run: per-ticker, only pulls history when the ticker
doesn't already have enough of it; _append_dedupe (snapshot.py) skips rows
already on disk by (date, ticker), so a second run adds zero rows.
"""

from __future__ import annotations

import os

import pandas as pd

from screener import UNIVERSE
from snapshot import snapshot_underlyings, UNDERLYING_SCHEMA
from signals import MIN_RV_OBS

PATH = "underlying_history.csv"
TARGET_YEARS = 2


def _tickers_needing_backfill(tickers, path=PATH, min_obs=MIN_RV_OBS) -> list[str]:
    if not os.path.exists(path):
        return list(tickers)
    df = pd.read_csv(path, usecols=["ticker"])
    counts = df["ticker"].astype(str).str.upper().value_counts()
    return [t for t in tickers if counts.get(t.upper(), 0) < min_obs]


def backfill(tickers=UNIVERSE, path=PATH, years=TARGET_YEARS) -> int:
    need = _tickers_needing_backfill(tickers, path)
    if not need:
        print(f"backfill_underlyings: all {len(tickers)} tickers already have "
              f">= {MIN_RV_OBS} obs — nothing to do")
        return 0
    print(f"backfill_underlyings: pulling {years}y for {len(need)} tickers: {need}")
    n = snapshot_underlyings(need, path=path, backfill_years=years, force_period=f"{years}y")
    print(f"backfill_underlyings: wrote {n} rows")
    return n


if __name__ == "__main__":
    backfill()
