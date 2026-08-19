"""Daily mark-to-market from chain_history; auto-close on TP/SL/time stop."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from paper import chain_history_path
from paper.models import read_trades, rewrite_trades, append_mark
from paper.exit import check_exit, apply_close
from spread_eval import credit_exit_fill


def _leg_quote(day: pd.DataFrame, expiry: str, kind: str, strike: float):
    sub = day[
        (day["expiry"].astype(str) == str(expiry))
        & (day["type"] == kind)
        & (np.isclose(day["strike"].astype(float), float(strike), atol=1e-6))
    ]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return float(r["bid"]), float(r["ask"]), float(r["mid"])


def price_spread(day: pd.DataFrame, trade: dict, stop: bool = False) -> tuple[Optional[float], Optional[float], bool]:
    """
    Returns (conservative_credit, mid_credit, missing_strike).
    Missing strike → (None, None, True) — caller carries forward.
    """
    long_q = _leg_quote(day, trade["expiry"], "C", trade["long_strike"])
    if long_q is None:
        return None, None, True
    short_strike = trade.get("short_strike")
    if short_strike in ("", None):
        cons, mid = credit_exit_fill(long_q[0], long_q[1], stop=stop)
        return cons, mid, False
    short_q = _leg_quote(day, trade["expiry"], "C", short_strike)
    if short_q is None:
        return None, None, True
    cons, mid = credit_exit_fill(
        long_q[0], long_q[1], short_q[1], short_q[0], stop=stop,
    )
    return cons, mid, False


def last_mark(trade_id: str) -> Optional[dict]:
    from paper import MARKS
    import csv
    if not MARKS.exists():
        return None
    last = None
    with MARKS.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["trade_id"] == trade_id:
                last = row
    return last


def run_mark(asof: Optional[str] = None) -> list[dict]:
    path = chain_history_path()
    if not path.exists():
        raise FileNotFoundError(path)
    chain = pd.read_csv(path)
    asof = asof or chain["date"].astype(str).max()
    day_all = chain[chain["date"].astype(str) == asof]
    trades = read_trades()
    updated = []
    marks = []

    for t in trades:
        if t.get("status") != "open":
            updated.append(t)
            continue
        ticker = t["ticker"]
        day = day_all[day_all["ticker"].str.upper() == ticker.upper()]
        carried = False
        flag = ""
        spot = float(day["spot"].iloc[0]) if not day.empty else float("nan")

        cons, mid, missing = (None, None, True)
        if not day.empty:
            cons, mid, missing = price_spread(day, t)

        if missing or cons is None:
            prev = last_mark(t["trade_id"])
            if prev is None:
                flag = "missing_strike_no_prior_mark"
                # cannot mark — skip close checks
                marks.append({
                    "mark_date": asof, "trade_id": t["trade_id"], "spot": spot,
                    "spread_mid": "", "spread_conservative": "",
                    "unrealized_pnl": "", "dte_left": "",
                    "carried_forward": "true", "flag": flag,
                })
                updated.append(t)
                continue
            cons = float(prev["spread_conservative"])
            mid = float(prev["spread_mid"]) if prev.get("spread_mid") not in ("", None) else cons
            carried = True
            flag = "missing_strike_carried_forward"
            if not np.isfinite(spot) and prev.get("spot") not in ("", None):
                spot = float(prev["spot"])

        entry = float(t["entry_debit"])
        contracts = int(t["contracts"])
        unreal = (cons - entry) * 100 * contracts
        try:
            dte_left = (dt.date.fromisoformat(str(t["expiry"])) - dt.date.fromisoformat(asof)).days
        except ValueError:
            dte_left = ""

        marks.append({
            "mark_date": asof,
            "trade_id": t["trade_id"],
            "spot": spot,
            "spread_mid": round(mid, 4),
            "spread_conservative": round(cons, 4),
            "unrealized_pnl": round(unreal, 2),
            "dte_left": dte_left,
            "carried_forward": str(carried).lower(),
            "flag": flag,
        })

        reason = check_exit(t, cons, asof)
        if reason:
            # stops get worse fill
            exit_cons = cons
            if reason == "sl":
                exit_cons, _ = price_spread(day, t, stop=True)[:2] if not day.empty else (cons, mid)
                if exit_cons is None:
                    exit_cons = cons
            closed = apply_close(t, float(exit_cons), reason)
            updated.append(closed)
            print(f"AUTO-CLOSE {t['trade_id'][:8]}… {ticker} reason={reason} "
                  f"exit={exit_cons:.4f} pnl={closed['pnl']}")
        else:
            updated.append(t)

    for m in marks:
        append_mark(m)
    rewrite_trades(updated)
    print(f"marked {len(marks)} open trades for {asof}")
    return marks
