"""
Fully automated daily paper loop for GitHub Actions.

1. For each tradeable name: run model, log trade/skip, open if TRADE under caps
2. Mark open positions (TP/SL/time stop)
3. Write metrics report to data/latest_report.txt

No human prompts. You read the report.
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
from paper.entry import context_for_forecast, open_trade
from paper.models import (
    append_forecast, read_forecasts, read_trades, open_capital_at_risk,
)
from paper.mark import run_mark
from paper.score import report as print_report
from spread_eval import evaluate
from screener import UNIVERSE

MAX_NEW_TRADES_PER_DAY = 1  # ~2–4/week pacing
DEFAULT_HORIZON = 30


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
            rv, iv = ctx["rv20"], ctx["iv"]
            vol = float(rv) if np.isfinite(rv) else (float(iv) if np.isfinite(iv) else 0.4)
            ed = ctx["earn_days"]

            # Auto policy: never enter into earnings window
            earnings_block = ed is not None and 0 <= ed <= horizon_days

            result = evaluate(
                ticker=ticker,
                spot=ctx["spot"],
                chain_rows=ctx["chain_day"],
                pred_vol_annual=vol,
                pred_move_pct=0.0,
                horizon_days=horizon_days,
                already_deployed=open_capital_at_risk(),
            )
            model_p = float(result.get("prob_profit", float("nan")))
            verdict = result.get("verdict")

            decision = "skip"
            skip_reason = result.get("skip_reason") or ""
            if earnings_block:
                decision, skip_reason = "skip", f"earnings in {ed}d (auto)"
            elif verdict == "TRADE" and new_opens < max_new:
                decision = "trade"
                skip_reason = ""
            elif verdict == "TRADE" and new_opens >= max_new:
                decision, skip_reason = "skip", f"daily new-trade cap ({max_new})"
            else:
                decision, skip_reason = "skip", skip_reason or "model SKIP"

            row = {
                "forecast_id": str(uuid.uuid4()),
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "ticker": ticker,
                "horizon_days": horizon_days,
                "direction": "up",
                "pred_move_pct": 0.0,
                "pred_vol_annual": vol,
                "pred_prob_profit": model_p if np.isfinite(model_p) else "",
                "iv_at_forecast": iv if np.isfinite(iv) else "",
                "iv_rank": ctx["iv_rank"] if np.isfinite(ctx["iv_rank"]) else "",
                "rationale": f"auto cron model={verdict} p={model_p:.3f} vol={vol:.2f}",
                "decision": decision,
                "skip_reason": skip_reason,
                "earnings_trade": "false",
                "source": "model",
            }
            append_forecast(row)

            if decision == "trade":
                trade = open_trade(row["forecast_id"], eval_result=result)
                opened.append(trade)
                open_names.add(ticker.upper())
                new_opens += 1
                print(f"OPEN  {ticker} {result.get('structure')} "
                      f"p={model_p:.3f} debit={result.get('entry_debit')}")
            else:
                skipped.append({"ticker": ticker, "reason": skip_reason, "verdict": verdict})
                print(f"SKIP  {ticker} [{verdict}] {skip_reason}")

        except Exception as ex:
            errors.append({"ticker": ticker, "error": str(ex)[:120]})
            print(f"ERR   {ticker}: {ex}")

    return {
        "opened": opened,
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
