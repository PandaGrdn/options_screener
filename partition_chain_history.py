"""
partition_chain_history.py — one-time (idempotent) split of the single
chain_history.csv into chain_history/YYYY-MM.csv.

Why: the single file grew ~0.8MB per session (38MB by 2026-10) and GitHub
rejects pushes of files >100MB, which would have broken the daily cron around
Feb 2027. Monthly partitions stay ~20MB each.

Rows dated on a non-NYSE session are dropped (only 2026-09-07, Labor Day: a
byte-identical copy of 2026-09-04's quotes stored under the holiday date).
Every other row is moved unchanged; row counts are verified before the
legacy file is removed. Re-running with no legacy file is a no-op.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from snapshot import CHAIN_SCHEMA, chain_month_path, is_trading_day

LEGACY = Path("chain_history.csv")
ROOT = Path("chain_history")


def partition(legacy: Path = LEGACY, root: Path = ROOT) -> int:
    if not legacy.exists():
        print(f"partition_chain_history: no {legacy} — nothing to do")
        return 0
    df = pd.read_csv(legacy)
    dates = df["date"].astype(str)
    off = sorted(d for d in dates.unique() if not is_trading_day(dt.date.fromisoformat(d)))
    if off:
        print(f"dropping {int(dates.isin(off).sum())} rows dated on non-sessions: {off}")
        df = df[~dates.isin(off)]
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    for month, part in df.groupby(df["date"].astype(str).str[:7]):
        out = chain_month_path(root, dt.date.fromisoformat(f"{month}-01"))
        if out.exists():
            raise RuntimeError(f"{out} already exists — refusing to merge into it")
        part[CHAIN_SCHEMA].to_csv(out, index=False)
        written += len(part)
        print(f"  {out}: {len(part)} rows")
    check = sum(len(pd.read_csv(f)) for f in sorted(root.glob("*.csv")))
    if check != len(df):
        raise RuntimeError(f"row count mismatch: wrote {check}, expected {len(df)}")
    legacy.unlink()
    print(f"partition_chain_history: {written} rows -> {root}/, removed {legacy}")
    return written


if __name__ == "__main__":
    partition()
