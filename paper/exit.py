"""Exit rule evaluation. TP/SL/time_stop are immutable after open."""

from __future__ import annotations

import datetime as dt
from typing import Optional


def check_exit(trade: dict, spread_conservative: float, mark_date: str) -> Optional[str]:
    """
    Return exit_reason or None.
    TP/SL compare mark value (conservative) to levels set at entry.
    Levels themselves are never modified here.
    """
    tp = float(trade["tp_level"])
    sl = float(trade["sl_level"])
    if spread_conservative >= tp:
        return "tp"
    if spread_conservative <= sl:
        return "sl"
    if mark_date >= str(trade["time_stop_date"]):
        return "time_stop"
    # expiry day
    if mark_date >= str(trade["expiry"]):
        return "expiry"
    return None


def apply_close(trade: dict, exit_credit: float, reason: str,
                closed_utc: Optional[str] = None) -> dict:
    """Write exit fields only; do not touch tp/sl/time_stop."""
    out = dict(trade)
    legs = 1 if not trade.get("short_strike") else 2
    from spread_eval import FEE_PER_CONTRACT
    contracts = int(trade["contracts"])
    fees = legs * 2 * FEE_PER_CONTRACT * contracts  # open+close already partly in sim; charge close here
    # entry already paid; pnl on premium difference * 100 * contracts - close fees
    entry = float(trade["entry_debit"])
    pnl = (exit_credit - entry) * 100 * contracts - legs * FEE_PER_CONTRACT * contracts
    out["status"] = "closed"
    out["closed_utc"] = closed_utc or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out["exit_credit"] = exit_credit
    out["exit_reason"] = reason
    out["pnl"] = round(pnl, 2)
    risk = float(trade["capital_at_risk"]) or 1.0
    out["return_pct"] = round(pnl / risk, 6)
    return out
