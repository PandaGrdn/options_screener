"""Forecast capture (model-blind) + trade open."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Optional

import numpy as np
import pandas as pd

from paper import (
    REFERENCE_ONLY, PORTFOLIO, MAX_DEPLOYED_PCT,
    chain_history_path, underlying_history_path, earnings_calendar_path,
)
from paper.models import (
    append_forecast, append_trade, get_forecast, read_trades,
    open_capital_at_risk, FORECAST_FIELDS,
)
from spread_eval import implied_vol, evaluate, FEE_PER_CONTRACT
from screener import yang_zhang_vol, TRADING_DAYS


def load_chain(ticker: Optional[str] = None) -> pd.DataFrame:
    path = chain_history_path()
    if not path.exists():
        raise FileNotFoundError(f"chain history not found at {path}")
    df = pd.read_csv(path)
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]
    return df


def latest_spot_and_chain(ticker: str) -> tuple[float, pd.DataFrame, str]:
    df = load_chain(ticker)
    if df.empty:
        raise ValueError(f"no chain rows for {ticker}")
    latest_date = df["date"].astype(str).max()
    day = df[df["date"].astype(str) == latest_date]
    spot = float(day["spot"].iloc[0])
    return spot, day, latest_date


def atm_iv_from_chain(day: pd.DataFrame, spot: float) -> float:
    """Recompute ATM call IV from mid — not yfinance's field."""
    calls = day[day["type"] == "C"].copy()
    if calls.empty:
        return float("nan")
    # nearest expiry ~30d, ATM strike
    calls["dte_gap"] = (calls["dte"] - 30).abs()
    exp = calls.sort_values("dte_gap").iloc[0]["expiry"]
    book = calls[calls["expiry"] == exp]
    row = book.iloc[(book["strike"] - spot).abs().argmin()]
    T = max(int(row["dte"]), 1) / TRADING_DAYS
    mid = float(row["mid"]) if np.isfinite(row["mid"]) else (float(row["bid"]) + float(row["ask"])) / 2
    return implied_vol(mid, spot, float(row["strike"]), T, kind="C")


def iv_series_for_ticker(ticker: str) -> pd.Series:
    """Daily ATM IV from chain_history (recomputed)."""
    df = load_chain(ticker)
    if df.empty:
        return pd.Series(dtype=float)
    rows = []
    for date, day in df.groupby(df["date"].astype(str)):
        spot = float(day["spot"].iloc[0])
        iv = atm_iv_from_chain(day, spot)
        rows.append((date, iv))
    s = pd.Series({d: v for d, v in rows}).sort_index()
    return s


def iv_rank(ticker: str, lookback: int = 252) -> float:
    s = iv_series_for_ticker(ticker).dropna()
    if len(s) < 5:
        return float("nan")
    s = s.tail(lookback)
    return float((s < s.iloc[-1]).mean() * 100)


def realized_vol(ticker: str, window: int = 20) -> float:
    """Prefer stored underlying_history; fall back to live yfinance."""
    path = underlying_history_path()
    col = "rv20_yz" if window <= 20 else "rv60_yz"
    if path.exists():
        try:
            df = pd.read_csv(path)
            sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
            if not sub.empty and col in sub.columns:
                v = float(sub.sort_values("date").iloc[-1][col])
                if np.isfinite(v):
                    return v
        except Exception:
            pass
    import yfinance as yf
    px = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if px is None or len(px) < window + 5:
        return float("nan")
    return float(yang_zhang_vol(px, window).iloc[-1])


def earnings_days(ticker: str) -> Optional[int]:
    """Prefer stored earnings_calendar; fall back to live yfinance."""
    path = earnings_calendar_path()
    if path.exists():
        try:
            df = pd.read_csv(path)
            sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
            if not sub.empty:
                row = sub.sort_values("asof").iloc[-1]
                ed = dt.date.fromisoformat(str(row["earnings_date"])[:10])
                return (ed - dt.date.today()).days
        except Exception:
            pass
    import yfinance as yf
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            ed = cal["Earnings Date"]
            ed = ed[0] if isinstance(ed, list) else ed
            if hasattr(ed, "date"):
                ed = ed.date()
            return (ed - dt.date.today()).days
        if hasattr(cal, "empty") and not cal.empty:
            return (cal.iloc[0, 0].date() - dt.date.today()).days
    except Exception:
        return None
    return None


def context_for_forecast(ticker: str) -> dict:
    """Market context ONLY — no model output."""
    ticker = ticker.upper()
    spot, day, asof = latest_spot_and_chain(ticker)
    iv = atm_iv_from_chain(day, spot)
    return {
        "ticker": ticker,
        "asof": asof,
        "spot": spot,
        "iv": iv,
        "iv_rank": iv_rank(ticker),
        "rv20": realized_vol(ticker),
        "earn_days": earnings_days(ticker),
        "chain_day": day,
    }


def prompt_forecast(ticker: str, noninteractive: Optional[dict] = None) -> dict:
    """
    Show context, collect forecast, APPEND to forecasts.csv.
    noninteractive: dict of fields for tests / scripting (still no model).
    """
    ticker = ticker.upper()
    ctx = context_for_forecast(ticker)

    print(f"\n=== PAPER FORECAST (no model) — {ticker} ===")
    print(f"asof          {ctx['asof']}")
    print(f"spot          {ctx['spot']:.2f}")
    print(f"IV (recomputed) {ctx['iv']:.1%}" if np.isfinite(ctx["iv"]) else "IV (recomputed) n/a")
    ivr = ctx["iv_rank"]
    print(f"IV rank       {ivr:.0f}" if np.isfinite(ivr) else "IV rank       n/a (need more IV history)")
    print(f"RV20 (YZ)     {ctx['rv20']:.1%}" if np.isfinite(ctx["rv20"]) else "RV20          n/a")
    ed = ctx["earn_days"]
    print(f"earnings      in {ed}d" if ed is not None else "earnings      n/a")
    print("(No model probability, EV, or verdict is shown here on purpose.)\n")

    def ask(key, cast, prompt, default=None):
        if noninteractive is not None and key in noninteractive:
            return cast(noninteractive[key])
        while True:
            raw = input(prompt if default is None else f"{prompt} [{default}]: ").strip()
            if raw == "" and default is not None:
                raw = str(default)
            try:
                return cast(raw)
            except Exception:
                print("  invalid, try again")

    horizon = ask("horizon_days", int, "horizon_days")
    direction = ask("direction", str, "direction (up/down/flat)").lower()
    if direction not in ("up", "down", "flat"):
        raise ValueError("direction must be up/down/flat")
    pred_move = ask("pred_move_pct", float, "pred_move_pct (e.g. 0.05 for +5%)")
    pred_vol = ask("pred_vol_annual", float, "pred_vol_annual (e.g. 0.40)")
    pred_prob = ask("pred_prob_profit", float, "pred_prob_profit (0-1)")
    if not 0 <= pred_prob <= 1:
        raise ValueError("pred_prob_profit must be in [0,1]")
    rationale = ask("rationale", str, "rationale (>=20 chars)")
    if len(rationale) < 20:
        raise ValueError("rationale must be at least 20 characters")
    decision = ask("decision", str, "decision (trade/skip)").lower()
    if decision not in ("trade", "skip"):
        raise ValueError("decision must be trade or skip")
    skip_reason = ""
    if decision == "skip":
        skip_reason = ask("skip_reason", str, "skip_reason")
        if not skip_reason:
            raise ValueError("skip_reason required when decision=skip")
    earnings_trade = False
    if ed is not None and 0 <= ed <= horizon:
        flag = ask("earnings_trade", str, "earnings inside hold — flag as earnings_trade? (y/n)", "n")
        earnings_trade = flag.lower().startswith("y")
        if not earnings_trade and decision == "trade":
            print("Refusing trade decision with earnings in window unless earnings_trade=y")
            decision = "skip"
            skip_reason = skip_reason or "earnings inside hold window"

    row = {
        "forecast_id": str(uuid.uuid4()),
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "horizon_days": horizon,
        "direction": direction,
        "pred_move_pct": pred_move,
        "pred_vol_annual": pred_vol,
        "pred_prob_profit": pred_prob,
        "iv_at_forecast": ctx["iv"] if np.isfinite(ctx["iv"]) else "",
        "iv_rank": ivr if np.isfinite(ivr) else "",
        "rationale": rationale,
        "decision": decision,
        "skip_reason": skip_reason,
        "earnings_trade": str(bool(earnings_trade)).lower(),
        "source": "human",
    }
    append_forecast(row)
    print(f"\nappended forecast_id={row['forecast_id']} decision={decision}")
    print("Next: paper evaluate --forecast-id", row["forecast_id"])
    return row


def _ask(noninteractive, key, cast, prompt, default=None):
    if noninteractive is not None and key in noninteractive:
        return cast(noninteractive[key])
    while True:
        raw = input(prompt if default is None else f"{prompt} [{default}]: ").strip()
        if raw == "" and default is not None:
            raw = str(default)
        try:
            return cast(raw)
        except Exception:
            print("  invalid, try again")


def decide(ticker: str, horizon_days: int = 30, auto_open: bool = False,
           noninteractive: Optional[dict] = None) -> dict:
    """
    Model-first workflow: show model verdict, then you accept or skip.
    Inputs to the model are market-derived (RV20 vol, flat drift) — not your gut.
    """
    ticker = ticker.upper()
    if ticker in REFERENCE_ONLY:
        raise ValueError(f"{ticker} is reference-only — refuse")

    ctx = context_for_forecast(ticker)
    rv = ctx["rv20"]
    iv = ctx["iv"]
    # Model assumptions when you're not forecasting: use RV as vol; flat underlying drift.
    # If RV missing, fall back to IV.
    vol = float(rv) if np.isfinite(rv) else (float(iv) if np.isfinite(iv) else 0.4)
    move = 0.0
    direction = "up"
    ed = ctx["earn_days"]

    print(f"\n=== PAPER DECIDE (model-first) — {ticker} ===")
    print(f"asof          {ctx['asof']}")
    print(f"spot          {ctx['spot']:.2f}")
    print(f"IV            {iv:.1%}" if np.isfinite(iv) else "IV            n/a")
    ivr = ctx["iv_rank"]
    print(f"IV rank       {ivr:.0f}" if np.isfinite(ivr) else "IV rank       n/a")
    print(f"RV20 → model vol  {vol:.1%}")
    print(f"model drift   {move:.1%} over {horizon_days}d (flat default)")
    print(f"earnings      in {ed}d" if ed is not None else "earnings      n/a")

    deployed = open_capital_at_risk()
    result = evaluate(
        ticker=ticker,
        spot=ctx["spot"],
        chain_rows=ctx["chain_day"],
        pred_vol_annual=vol,
        pred_move_pct=move,
        horizon_days=horizon_days,
        already_deployed=deployed,
    )
    model_p = float(result.get("prob_profit", float("nan")))

    print(f"\n--- MODEL ---")
    print(f"verdict              {result.get('verdict')}")
    if result.get("skip_reason"):
        print(f"skip_reason          {result.get('skip_reason')}")
    print(f"model_prob_profit    {model_p:.3f}" if np.isfinite(model_p) else "model_prob_profit    n/a")
    print(f"structure            {result.get('structure')}")
    print(f"expiry / dte         {result.get('expiry')} / {result.get('dte')}")
    print(f"strikes              long={result.get('long_strike')} short={result.get('short_strike')}")
    print(f"entry_debit (consol) {result.get('entry_debit')}")
    print(f"contracts / capital  {result.get('contracts')} / ${float(result.get('capital_at_risk') or 0):.0f}")
    print(f"EV / log_growth      {float(result.get('ev') or 0):.4f} / {float(result.get('log_growth') or 0):.4f}")
    print(f"TP / SL              {result.get('tp_level'):.4f} / {result.get('sl_level'):.4f}")

    # Default decision follows the model
    default_decision = "trade" if result.get("verdict") == "TRADE" else "skip"
    decision = _ask(noninteractive, "decision", str,
                    f"decision (trade/skip) — model says {result.get('verdict')}",
                    default_decision).lower()
    if decision not in ("trade", "skip"):
        raise ValueError("decision must be trade or skip")

    skip_reason = ""
    if decision == "skip":
        skip_reason = _ask(noninteractive, "skip_reason", str, "skip_reason",
                           result.get("skip_reason") or "model skip / user declined")
    rationale = _ask(noninteractive, "rationale", str, "note (>=20 chars)",
                     f"model-first {result.get('verdict')} {result.get('structure')} p={model_p:.2f}")
    if len(rationale) < 20:
        rationale = (rationale + " " + "model-driven decision").strip()
        if len(rationale) < 20:
            rationale = rationale.ljust(20, ".")

    earnings_trade = False
    if ed is not None and 0 <= ed <= horizon_days:
        flag = _ask(noninteractive, "earnings_trade", str,
                    "earnings inside hold — flag earnings_trade? (y/n)", "n")
        earnings_trade = flag.lower().startswith("y")
        if decision == "trade" and not earnings_trade:
            print("Earnings in window → forcing skip (pass earnings_trade=y to allow)")
            decision = "skip"
            skip_reason = skip_reason or "earnings inside hold window"

    row = {
        "forecast_id": str(uuid.uuid4()),
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "horizon_days": horizon_days,
        "direction": direction,
        "pred_move_pct": move,
        "pred_vol_annual": vol,
        "pred_prob_profit": model_p if np.isfinite(model_p) else "",
        "iv_at_forecast": iv if np.isfinite(iv) else "",
        "iv_rank": ivr if np.isfinite(ivr) else "",
        "rationale": rationale,
        "decision": decision,
        "skip_reason": skip_reason,
        "earnings_trade": str(bool(earnings_trade)).lower(),
        "source": "model",
    }
    append_forecast(row)
    print(f"\nlogged forecast_id={row['forecast_id']} decision={decision} source=model")

    out = {"forecast": row, "eval": result, "trade": None}
    if decision == "trade":
        if result.get("verdict") != "TRADE":
            print("Model verdict is SKIP — not auto-opening. Use open --override if you insist.")
            return out
        do_open = auto_open
        if not auto_open:
            ans = _ask(noninteractive, "open_now", str, "open this trade now? (y/n)", "y")
            do_open = ans.lower().startswith("y")
        if do_open:
            out["trade"] = open_trade(row["forecast_id"], eval_result=result)
    return out


def run_evaluate(forecast_id: str) -> dict:
    fc = get_forecast(forecast_id)
    if fc is None:
        raise ValueError(f"unknown forecast_id {forecast_id}")
    ticker = fc["ticker"].upper()
    spot, day, _ = latest_spot_and_chain(ticker)
    deployed = open_capital_at_risk()
    result = evaluate(
        ticker=ticker,
        spot=spot,
        chain_rows=day,
        pred_vol_annual=float(fc["pred_vol_annual"]),
        pred_move_pct=float(fc["pred_move_pct"]),
        horizon_days=int(fc["horizon_days"]),
        already_deployed=deployed,
    )
    pred = float(fc["pred_prob_profit"])
    model_p = float(result.get("prob_profit", float("nan")))
    print(f"\n=== MODEL EVALUATE — {ticker} / {forecast_id[:8]}… ===")
    print(f"your pred_prob_profit   {pred:.3f}")
    print(f"model_prob_profit       {model_p:.3f}" if np.isfinite(model_p) else "model_prob_profit       n/a")
    if np.isfinite(model_p):
        print(f"delta (you - model)     {pred - model_p:+.3f}")
    print(f"verdict                 {result.get('verdict')}")
    if result.get("skip_reason"):
        print(f"skip_reason             {result['skip_reason']}")
    print(f"structure               {result.get('structure')}")
    print(f"expiry / dte            {result.get('expiry')} / {result.get('dte')}")
    print(f"strikes                 long={result.get('long_strike')} short={result.get('short_strike')}")
    print(f"entry_debit (consol)    {result.get('entry_debit')}")
    print(f"contracts / capital     {result.get('contracts')} / ${result.get('capital_at_risk'):.0f}")
    print(f"EV / log_growth         {result.get('ev'):.4f} / {result.get('log_growth'):.4f}")
    print(f"TP / SL levels          {result.get('tp_level'):.4f} / {result.get('sl_level'):.4f}")
    return result


def open_trade(forecast_id: str, override: bool = False, override_reason: str = "",
               eval_result: Optional[dict] = None) -> dict:
    fc = get_forecast(forecast_id)
    if fc is None:
        raise ValueError(f"unknown forecast_id {forecast_id}")
    if fc["decision"] != "trade":
        raise ValueError("forecast decision is skip — not opening a trade from a skip")

    ticker = fc["ticker"].upper()
    if ticker in REFERENCE_ONLY:
        raise ValueError(f"{ticker} is reference-only (SPY/QQQ/IWM) — refuse entry")

    result = eval_result or run_evaluate(forecast_id)
    if result.get("verdict") != "TRADE":
        if not override:
            raise ValueError(
                f"model says SKIP ({result.get('skip_reason')}). "
                "Pass --override and --override-reason to force."
            )
        if not override_reason or len(override_reason) < 10:
            raise ValueError("override requires --override-reason (>=10 chars)")

    contracts = int(result.get("contracts") or 0)
    if contracts == 0:
        raise ValueError("contracts == 0 — refuse entry")

    capital = float(result["capital_at_risk"])
    deployed = open_capital_at_risk()
    if deployed + capital > PORTFOLIO * MAX_DEPLOYED_PCT + 1e-6:
        raise ValueError(
            f"would exceed 20% deployed ({deployed:.0f}+{capital:.0f} > {PORTFOLIO*MAX_DEPLOYED_PCT:.0f})"
        )

    earn = fc.get("earnings_trade", "").lower() in ("1", "true", "yes")
    ed = earnings_days(ticker)
    horizon = int(fc["horizon_days"])
    if ed is not None and 0 <= ed <= horizon and not earn:
        raise ValueError("earnings inside hold window — flag forecast as earnings_trade or skip")

    opened = dt.datetime.now(dt.timezone.utc)
    time_stop = (opened.date() + dt.timedelta(days=horizon)).isoformat()
    short = result.get("short_strike")
    row = {
        "trade_id": str(uuid.uuid4()),
        "forecast_id": forecast_id,
        "opened_utc": opened.isoformat(timespec="seconds"),
        "ticker": ticker,
        "structure": result["structure"],
        "expiry": result["expiry"],
        "dte_at_entry": result["dte"],
        "long_strike": result["long_strike"],
        "short_strike": "" if short is None else short,
        "entry_debit": result["entry_debit"],
        "entry_mid": result.get("entry_mid", result["entry_debit"]),
        "contracts": contracts,
        "capital_at_risk": capital,
        "model_prob_profit": result["prob_profit"],
        "model_ev": result["ev"],
        "model_log_growth": result["log_growth"],
        "tp_level": result["tp_level"],
        "sl_level": result["sl_level"],
        "time_stop_date": time_stop,
        "status": "open",
        "closed_utc": "",
        "exit_credit": "",
        "exit_reason": "",
        "pnl": "",
        "return_pct": "",
        "override": str(bool(override)).lower(),
        "override_reason": override_reason if override else "",
        "earnings_trade": str(earn).lower(),
    }
    append_trade(row)
    print(f"\nopened trade_id={row['trade_id']} {ticker} {row['structure']} "
          f"x{contracts} debit={row['entry_debit']:.4f}")
    return row
