"""
migrate_model_version.py — ONE-TIME backfill of the model_version column
added by Patch 1 (spread_eval.py path-dependent Monte Carlo rewrite).

forecasts.csv is append-only by hard constraint (AGENT_CONTEXT §2) — no
edit/update API exists on purpose, and this script does not add one. The
Patch 1 migration note is an explicit, narrow exception: backfill is
permitted for this ONE new column only, on rows a model actually scored
(regime=kelly, pred_prob_profit set) with model_version still blank. Every
other cell in every row is left byte-for-byte identical. trades.csv already
allows rewriting non-forecast fields (rewrite_trades) — same rule applies:
model_version only.

Idempotent: rows that already have a model_version are skipped, so re-running
after Patch 1 code is live (which stamps model_version on new rows itself)
is a no-op.
"""

from __future__ import annotations

import pandas as pd

from paper import FORECASTS, TRADES
from paper.models import (
    is_kelly_regime, read_forecasts, read_trades, TRADE_FIELDS, FORECAST_FIELDS,
)
from spread_eval import MODEL_VERSION_LEGACY


def migrate_forecasts() -> int:
    rows = read_forecasts()
    if not rows:
        return 0
    n = 0
    for r in rows:
        if str(r.get("model_version") or "").strip():
            continue  # already tagged (by this script or by new Patch-1 code)
        if not is_kelly_regime(r):
            continue  # v1/legacy regime — never Kelly-scored, leave untouched
        if not str(r.get("pred_prob_profit") or "").strip():
            continue  # no model probability was ever recorded for this row
        r["model_version"] = MODEL_VERSION_LEGACY
        n += 1
    if n:
        df = pd.DataFrame(rows)
        for col in FORECAST_FIELDS:
            if col not in df.columns:
                df[col] = ""
        df[FORECAST_FIELDS].to_csv(FORECASTS, index=False)
    return n


def migrate_trades() -> int:
    rows = read_trades()
    if not rows:
        return 0
    n = 0
    for r in rows:
        if str(r.get("model_version") or "").strip():
            continue
        r["model_version"] = MODEL_VERSION_LEGACY
        n += 1
    if n:
        df = pd.DataFrame(rows)
        for col in TRADE_FIELDS:
            if col not in df.columns:
                df[col] = ""
        df[TRADE_FIELDS].to_csv(TRADES, index=False)
    return n


if __name__ == "__main__":
    nf = migrate_forecasts()
    nt = migrate_trades()
    print(f"migrate_model_version: forecasts backfilled={nf} trades backfilled={nt}")
