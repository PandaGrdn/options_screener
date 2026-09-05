"""
Daily cron: snapshot already ran. Score the universe at market (IV + 0 drift),
open a *shadow* (mark-only) book on cheapness-pass names so Brier has outcomes,
mark everything, write the report.

Real paper opens still require a human forecast that beats Kelly. Shadows
do not use capital.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from paper import DATA, REFERENCE_ONLY, ensure_data_dir
from paper.entry import context_for_forecast, open_shadow
from paper.models import (
    append_forecast, read_forecasts, read_trades, open_capital_at_risk,
    REGIME_KELLY,
)
from paper.mark import run_mark
from paper.score import report as print_report
from spread_eval import evaluate, MODEL_VERSION
from screener import UNIVERSE

MAX_NEW_TRADES_PER_DAY = 1  # ignored: auto never opens (no forecast)
DEFAULT_HORIZON = 21


def _today() -> str:
    return dt.date.today().isoformat()


def _already_decided_today(ticker: str, today: str) -> bool:
    for f in read_forecasts():
        if f.get("ticker", "").upper() != ticker.upper():
            continue
        ts = str(f.get("ts_utc", ""))
        if ts.startswith(today):
            return True
        # also tolerate date-only fields if present
        if str(f.get("date", "")) == today:
            return True
    return False


def _open_tickers() -> set[str]:
    return {t["ticker"].upper() for t in read_trades() if t.get("status") == "open"}


def auto_decide_universe(horizon_days: int = DEFAULT_HORIZON,
                         max_new: int = MAX_NEW_TRADES_PER_DAY) -> dict:
    """Model-first, noninteractive. Returns summary counts."""
    today = _today()
    opened = []
    shadowed = []
    skipped = []
    errors = []
    open_names = _open_tickers()
    new_opens = 0

    tradeable = [t for t in UNIVERSE if t.upper() not in REFERENCE_ONLY]

    for ticker in tradeable:
        try:
            if _already_decided_today(ticker, today):
                continue
            if ticker.upper() in open_names:
                # still log? skip quietly — already in a position
                continue

            ctx = context_for_forecast(ticker)
            iv = ctx["iv"]
            ivr = ctx["iv_rank"]
            ed = ctx["earn_days"]
            gate_passed, gate_reason = ctx["gate_passed"], ctx["gate_reason"]
            earnings_block = ed is not None and 0 <= ed <= horizon_days

            # Market default: ATM IV, 0 drift. No RV-as-forecast.
            result = evaluate(
                ticker=ticker,
                spot=ctx["spot"],
                chain_rows=ctx["chain_day"],
                pred_vol_annual=None,
                pred_move_pct=0.0,
                horizon_days=horizon_days,
                already_deployed=open_capital_at_risk(),
            )
            model_p = float(result.get("prob_profit", float("nan")))
            vol = result.get("forecast_vol")
            vol_s = float(vol) if vol is not None and np.isfinite(vol) else float("nan")
            verdict = result.get("verdict")

            decision, skip_reason = "skip", result.get("skip_reason") or "model SKIP"
            if earnings_block:
                skip_reason = f"earnings in {ed}d (auto)"
            elif not gate_passed:
                skip_reason = gate_reason
            elif verdict == "TRADE":
                # Cron has no forecast. Even a +Kelly market misprice is not an auto-open.
                skip_reason = "market-default: no forecast, auto will not open"

            row = {
                "forecast_id": str(uuid.uuid4()),
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "ticker": ticker,
                "horizon_days": horizon_days,
                "direction": "up",
                "pred_move_pct": 0.0,
                "pred_vol_annual": vol_s if np.isfinite(vol_s) else "",
                "pred_prob_profit": model_p if np.isfinite(model_p) else "",
                "iv_at_forecast": iv if np.isfinite(iv) else "",
                "iv_rank": ivr if np.isfinite(ivr) else "",
                "rationale": (
                    f"auto cron market-default verdict={verdict} p={model_p:.3f} "
                    f"vol={vol_s:.2f}" if np.isfinite(vol_s) else
                    f"auto cron market-default verdict={verdict} p={model_p:.3f}"
                ),
                "decision": decision,
                "skip_reason": skip_reason,
                "earnings_trade": "false",
                "source": "model",
                "regime": REGIME_KELLY,
                "gate_reason": gate_reason,
                "hypothesis": "",  # cron has no trade idea to tag
                "model_version": MODEL_VERSION,
            }
            append_forecast(row)
            skipped.append({"ticker": ticker, "reason": skip_reason, "verdict": verdict})
            print(f"SKIP  {ticker} [{verdict}] {skip_reason}")
            # Ghosts only on names that pass cheapness — same screen a real
            # TRADE would need. Failures still get a forecast row, no shadow.
            if not gate_passed:
                print(f"NO SHADOW {ticker} cheapness fail — {gate_reason}")
            else:
                shadow = open_shadow(row["forecast_id"], result)
                if shadow:
                    shadowed.append(shadow)
                    open_names.add(ticker.upper())

        except Exception as ex:
            errors.append({"ticker": ticker, "error": str(ex)[:120]})
            print(f"ERR   {ticker}: {ex}")

    return {
        "opened": opened,
        "shadowed": shadowed,
        "skipped": skipped,
        "errors": errors,
        "new_opens": new_opens,
    }


def write_report(path: Path | None = None) -> str:
    ensure_data_dir()
    path = path or (DATA / "latest_report.txt")
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        print(f"generated_utc {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
        print_report()
    finally:
        sys.stdout = old
    text = buf.getvalue()
    path.write_text(text)
    from paper.dashboard import write_dashboard
    dash = write_dashboard()
    print(f"wrote {dash}")
    # append a dated copy for history
    hist = DATA / "report_history.txt"
    with hist.open("a") as f:
        f.write("\n" + "=" * 72 + "\n")
        f.write(text)
        f.write("\n")
    return text


def run_daily(horizon_days: int = DEFAULT_HORIZON,
              max_new: int = MAX_NEW_TRADES_PER_DAY) -> None:
    print(f"=== paper run-daily {dt.datetime.now(dt.timezone.utc).isoformat()} ===")
    summary = auto_decide_universe(horizon_days=horizon_days, max_new=max_new)
    print(f"\nnew opens: {summary['new_opens']}  "
          f"shadows: {len(summary.get('shadowed', []))}  "
          f"skips: {len(summary['skipped'])}  errors: {len(summary['errors'])}")
    print("\n--- mark ---")
    run_mark()
    print("\n--- report ---")
    text = write_report()
    print(text)
    # GitHub Actions job summary if present
    ghs = os.environ.get("GITHUB_STEP_SUMMARY")
    if ghs:
        Path(ghs).write_text("```\n" + text + "\n```\n")
