"""
backfill_underlyings.py — one-time (idempotent) 2-year OHLC/RV backfill for
the whole UNIVERSE, feeding signals.rv_percentile()'s MIN_RV_OBS gate.

underlying_history.csv is explicitly backfillable (AGENT_CONTEXT §5) —
unlike chain_history, which is same-session only and can never be
backfilled. Safe to re-run: per-ticker, only pulls history when the ticker
doesn't already have enough of it; _append_dedupe (snapshot.py) skips rows
already on disk by (date, ticker), so a second run adds zero rows.

Also repairs rv20_pctile cells left nan by the old 3mo incremental pull
(every daily row from 2026-08-24 on) — see repair_rv_pctile().
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from screener import UNIVERSE, vol_percentile
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


def repair_rv_pctile(path=PATH) -> int:
    """Fill nan rv20_pctile cells from the stored rv20_yz series with the same
    vol_percentile() snapshot.py uses (trailing 252 obs, as of that row).
    Only nan cells with a computable percentile are touched; idempotent."""
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    filled = 0
    for _, g in df.sort_values("date").groupby("ticker"):
        rv = g["rv20_yz"].astype(float)
        for i, idx in enumerate(g.index):
            if pd.notna(df.at[idx, "rv20_pctile"]):
                continue
            pct = vol_percentile(rv.iloc[: i + 1])
            if np.isfinite(pct):
                df.at[idx, "rv20_pctile"] = round(pct, 2)
                filled += 1
    if filled:
        df[UNDERLYING_SCHEMA].to_csv(path, index=False)
    print(f"repair_rv_pctile: filled {filled} rv20_pctile cells")
    return filled


if __name__ == "__main__":
    backfill()
    repair_rv_pctile()
