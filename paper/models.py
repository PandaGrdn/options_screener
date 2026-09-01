"""Schemas, fill helpers, CSV IO. forecasts.csv is APPEND-ONLY — no edit API."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

from paper import FORECASTS, TRADES, MARKS, ensure_data_dir

FORECAST_FIELDS = [
    "forecast_id", "ts_utc", "ticker", "horizon_days", "direction",
    "pred_move_pct", "pred_vol_annual", "pred_prob_profit",
    "iv_at_forecast", "iv_rank", "rationale", "decision", "skip_reason",
    "earnings_trade", "source",  # source=model|human
    "regime",  # kelly = current scorecard; blank = v1 RV+p/EV (kept, not scored)
    "gate_reason",  # cheapness_gate() human-readable reason (signals.py, Patch 2)
    "hypothesis",  # trend|mean_reversion|vol_expansion|post_earnings|catalyst|other (Patch 3)
    "model_version",  # mc_terminal_v1 (pre-patch, biased) | mc_path_v2 (Patch 1)
]

REGIME_KELLY = "kelly"
REGIME_V1 = "v1_rv_pev"  # inferred when regime column is blank

HYPOTHESES = ("trend", "mean_reversion", "vol_expansion", "post_earnings", "catalyst", "other")


def forecast_regime(row: dict) -> str:
    r = str(row.get("regime") or "").strip()
    return r if r else REGIME_V1


def is_kelly_regime(row: dict) -> bool:
    return forecast_regime(row) == REGIME_KELLY

TRADE_FIELDS = [
    "trade_id", "forecast_id", "opened_utc", "ticker", "structure",
    "expiry", "dte_at_entry", "long_strike", "short_strike",
    "entry_debit", "entry_mid", "contracts", "capital_at_risk",
    "model_prob_profit", "model_ev", "model_log_growth",
    "tp_level", "sl_level", "time_stop_date",
    "status", "closed_utc", "exit_credit", "exit_reason", "pnl", "return_pct",
    "override", "override_reason", "earnings_trade",
    "model_version",  # mc_terminal_v1 (pre-patch, biased) | mc_path_v2 (Patch 1)
]

MARK_FIELDS = [
    "mark_date", "trade_id", "spot", "spread_mid", "spread_conservative",
    "unrealized_pnl", "dte_left", "carried_forward", "flag",
]


@dataclass
class Forecast:
    forecast_id: str
    ts_utc: str
    ticker: str
    horizon_days: int
    direction: str
    pred_move_pct: float
    pred_vol_annual: float
    pred_prob_profit: float
    iv_at_forecast: float
    iv_rank: float
    rationale: str
    decision: str
    skip_reason: str = ""
    earnings_trade: bool = False


def _ensure_header(path: Path, fieldnames: list[str]) -> None:
    ensure_data_dir()
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    # Widen schema if new columns were added (does not alter existing cell values).
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        rows = list(reader)
    if list(old_fields) == list(fieldnames):
        return
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def append_forecast(row: dict) -> None:
    """ONLY write path for forecasts. No update/delete helpers exist on purpose."""
    _ensure_header(FORECASTS, FORECAST_FIELDS)
    with FORECASTS.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FORECAST_FIELDS).writerow(
            {k: row.get(k, "") for k in FORECAST_FIELDS}
        )


def read_forecasts() -> list[dict]:
    if not FORECASTS.exists():
        return []
    with FORECASTS.open(newline="") as f:
        return list(csv.DictReader(f))


def get_forecast(forecast_id: str) -> Optional[dict]:
    for r in read_forecasts():
        if r["forecast_id"] == forecast_id:
            return r
    return None


def append_trade(row: dict) -> None:
    _ensure_header(TRADES, TRADE_FIELDS)
    with TRADES.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_FIELDS).writerow(
            {k: row.get(k, "") for k in TRADE_FIELDS}
        )


def read_trades() -> list[dict]:
    if not TRADES.exists():
        return []
    with TRADES.open(newline="") as f:
        return list(csv.DictReader(f))


def rewrite_trades(rows: Iterable[dict]) -> None:
    """Allowed: update trade status/exit fields only (not forecasts)."""
    ensure_data_dir()
    tmp = TRADES.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRADE_FIELDS})
    os.replace(tmp, TRADES)


def append_mark(row: dict) -> None:
    _ensure_header(MARKS, MARK_FIELDS)
    with MARKS.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=MARK_FIELDS).writerow(
            {k: row.get(k, "") for k in MARK_FIELDS}
        )


def open_capital_at_risk(trades: Optional[list[dict]] = None) -> float:
    trades = trades if trades is not None else read_trades()
    total = 0.0
    for t in trades:
        if t.get("status") == "open":
            try:
                total += float(t["capital_at_risk"])
            except (TypeError, ValueError):
                pass
    return total


def _f(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default
