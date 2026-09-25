"""
Daily cron: snapshot already ran. Score the universe at market (IV + 0 drift).
Among cheapness-pass names, open the top max_new as real 1-lot paper trades
(ranked by Kelly log_growth). Shadow the rest. Mark and write the report.
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
from paper.entry import context_for_forecast, open_shadow, open_auto_trade
from paper.models import (
    append_forecast, read_forecasts, read_trades, open_capital_at_risk,
    REGIME_KELLY,
)
from paper.mark import run_mark
from paper.score import report as print_report
from spread_eval import evaluate, MODEL_VERSION
from screener import UNIVERSE

MAX_NEW_TRADES_PER_DAY = 1
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


def _finite(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _viable_spread(result: dict) -> bool:
    if result.get("structure") != "call_debit_spread":
        return False
    debit = _finite(result.get("entry_debit"))
    return debit is not None and debit > 0


def _auto_rank_key(result: dict) -> tuple:
    return (
        _finite(result.get("log_growth"), -1e9),
        _finite(result.get("prob_profit"), -1.0),
    )


def auto_decide_universe(horizon_days: int = DEFAULT_HORIZON,
                         max_new: int = MAX_NEW_TRADES_PER_DAY) -> dict:
    """Score the universe, auto-open the best cheapness-pass names, shadow the rest."""
    today = _today()
    opened = []
    shadowed = []
    skipped = []
    errors = []
    open_names = _open_tickers()
    new_opens = 0

    tradeable = [t for t in UNIVERSE if t.upper() not in REFERENCE_ONLY]
    pending = []

    for ticker in tradeable:
        try:
            if _already_decided_today(ticker, today):
                continue
            if ticker.upper() in open_names:
                continue

            ctx = context_for_forecast(ticker)
            ed = ctx["earn_days"]
            earnings_block = ed is not None and 0 <= ed <= horizon_days
            result = evaluate(
                ticker=ticker,
                spot=ctx["spot"],
                chain_rows=ctx["chain_day"],
                pred_vol_annual=None,
                pred_move_pct=0.0,
                horizon_days=horizon_days,
                already_deployed=open_capital_at_risk(),
            )
            pending.append({
                "ticker": ticker,
                "ctx": ctx,
                "ed": ed,
                "earnings_block": earnings_block,
                "result": result,
            })
        except Exception as ex:
            errors.append({"ticker": ticker, "error": str(ex)[:120]})
            print(f"ERR   {ticker}: {ex}")

    ranked = [
        p for p in pending
        if p["ctx"]["gate_passed"] and not p["earnings_block"] and _viable_spread(p["result"])
    ]
    ranked.sort(key=lambda p: _auto_rank_key(p["result"]), reverse=True)
    auto_tickers = {p["ticker"].upper() for p in ranked[: max(0, int(max_new))]}

    for item in pending:
        ticker = item["ticker"]
        ctx = item["ctx"]
        result = item["result"]
        iv = ctx["iv"]
        ivr = ctx["iv_rank"]
        ed = item["ed"]
        gate_passed, gate_reason = ctx["gate_passed"], ctx["gate_reason"]
        earnings_block = item["earnings_block"]
        model_p = float(result.get("prob_profit", float("nan")))
        vol = result.get("forecast_vol")
        vol_s = float(vol) if vol is not None and np.isfinite(vol) else float("nan")
        verdict = result.get("verdict")
        log_g = _finite(result.get("log_growth"))

        take = (
            ticker.upper() in auto_tickers
            and new_opens < max_new
            and ticker.upper() not in open_names
        )
        decision = "trade" if take else "skip"
        skip_reason = ""
        if not take:
            skip_reason = result.get("skip_reason") or "model SKIP"
            if earnings_block:
                skip_reason = f"earnings in {ed}d (auto)"
            elif not gate_passed:
                skip_reason = gate_reason

        rationale = (
            f"auto cron market-default verdict={verdict} p={model_p:.3f} "
            f"vol={vol_s:.2f}" if np.isfinite(vol_s) else
            f"auto cron market-default verdict={verdict} p={model_p:.3f}"
        )
        if take and log_g is not None:
            rationale = f"auto cron OPEN log_growth={log_g:.4f} p={model_p:.3f} vol={vol_s:.2f}"

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
            "rationale": rationale,
            "decision": decision,
            "skip_reason": skip_reason,
            "earnings_trade": "false",
            "source": "model",
            "regime": REGIME_KELLY,
            "gate_reason": gate_reason,
            "hypothesis": "",
            "model_version": MODEL_VERSION,
        }
        append_forecast(row)

        if take:
            trade = open_auto_trade(row["forecast_id"], result)
            if trade:
                opened.append(trade)
                open_names.add(ticker.upper())
                new_opens += 1
                print(f"OPEN  {ticker} [{verdict}] auto rank log_growth={log_g}")
                continue
            print(f"AUTO-OPEN FAIL {ticker} — shadowing instead")
            take = False
            # forecast already says trade; still shadow so we get an outcome

        skipped.append({"ticker": ticker, "reason": skip_reason or "auto-open failed", "verdict": verdict})
        print(f"SKIP  {ticker} [{verdict}] {skip_reason or 'auto-open failed'}")
        if not gate_passed:
            print(f"NO SHADOW {ticker} cheapness fail — {gate_reason}")
        else:
            shadow = open_shadow(row["forecast_id"], result)
            if shadow:
                shadowed.append(shadow)
                open_names.add(ticker.upper())

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
