"""
snapshot.py — daily market data capture (not model outputs).

Stores:
  chain_history/YYYY-MM.csv  raw quotes + own IV/greeks across a wider surface,
                             one file per month (GitHub rejects files >100MB)
                             (same-session; cannot be backfilled)
  underlying_history.csv     OHLC + realized vol for the *previous* completed
                             session (Yahoo's daily bar is reliable on T+1)
  earnings_calendar.csv      next earnings date as known each asof day

yfinance's impliedVolatility is frequently stale; we still keep yf_iv but also
solve our own IV from mid and store BS greeks. You cannot backfill option
chains, so capture them from day one. Underlying OHLC *can* be backfilled.

Chains are captured only after the close of a real NYSE session (holidays and
early closes from exchange_calendars). Off-session runs return the previous
session's quotes under today's date — that is how 2026-09-07 (Labor Day)
ended up as a byte-identical copy of 09-04.
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from screener import close_to_close_vol, yang_zhang_vol, vol_percentile
from spread_eval import implied_vol, bs_greeks

NY = ZoneInfo("America/New_York")

# ETF / index options keep quoting until 4:15pm ET; wait for those to settle.
CLOSE_BUFFER = dt.timedelta(minutes=15)
# Fraction of overlapping quotes identical to the previous stored session above
# which we refuse to write (market closed, or Yahoo serving a frozen chain).
# Normal days run ~0.1–0.3% identical.
STALE_IDENTICAL_FRAC = 0.95
STALE_MIN_OVERLAP = 200  # below this the comparison is too thin to judge

_NYSE = None


def nyse():
    global _NYSE
    if _NYSE is None:
        import exchange_calendars as xcals
        _NYSE = xcals.get_calendar("XNYS")
    return _NYSE


def is_trading_day(d: dt.date) -> bool:
    return bool(nyse().is_session(pd.Timestamp(d)))


def session_date(now: dt.datetime | None = None) -> dt.date:
    """US equity session date (America/New_York), not the runner's UTC date."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(NY).date()


def chain_capture_status(now: dt.datetime | None = None) -> tuple[bool, str]:
    """(ok, reason). Chains are captured only on an NYSE session, after its close
    (+ CLOSE_BUFFER), on the same NY calendar day — anything else would store
    stale or intraday quotes under the wrong session date."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    d = session_date(now)
    if not is_trading_day(d):
        return False, f"{d} is not an NYSE session (weekend/holiday)"
    close = nyse().session_close(pd.Timestamp(d)).to_pydatetime()
    ready = close + CLOSE_BUFFER
    if now < ready:
        return False, (f"{d} session not settled yet (close {close.astimezone(NY):%H:%M} ET, "
                       f"capture after {ready.astimezone(NY):%H:%M} ET)")
    return True, f"{d} session closed {close.astimezone(NY):%H:%M} ET"


def index_session_date(idx) -> dt.date:
    """Map a yfinance DatetimeIndex label to the NY session date."""
    ts = pd.Timestamp(idx)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY)
    return ts.date()


def previous_session_date(asof: dt.date | None = None) -> dt.date:
    """Last NYSE session strictly before asof (skips weekends and holidays)."""
    d = (asof or session_date()) - dt.timedelta(days=1)
    return nyse().date_to_session(pd.Timestamp(d), direction="previous").date()


RISK_FREE = 0.05

CHAIN_SCHEMA = [
    "date", "ts_utc", "ticker", "spot", "expiry", "dte", "type",
    "strike", "moneyness", "bid", "ask", "mid", "spread_pct",
    "volume", "open_interest", "yf_iv",
    "iv", "delta", "gamma", "vega", "theta",
]

UNDERLYING_SCHEMA = [
    "date", "ticker", "open", "high", "low", "close", "volume",
    "rv20_yz", "rv60_yz", "rv20_c2c", "rv20_pctile",
]

EARNINGS_SCHEMA = ["asof", "ticker", "earnings_date", "days_to_earnings"]


def _safe_int(x) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        return int(x)
    except (TypeError, ValueError):
        return 0


def _ensure_csv_schema(path: str, schema: list[str]) -> None:
    """Add any missing columns (NaN) so appends stay consistent after schema upgrades."""
    if not os.path.exists(path):
        return
    prev = pd.read_csv(path)
    missing = [c for c in schema if c not in prev.columns]
    if not missing and list(prev.columns) == schema:
        return
    for c in missing:
        prev[c] = np.nan
    prev[schema].to_csv(path, index=False)


def _append_dedupe(path: str, df: pd.DataFrame, schema: list[str], key_cols: list[str]) -> int:
    if df.empty:
        return 0
    df = df[schema].copy()
    _ensure_csv_schema(path, schema)
    if os.path.exists(path):
        try:
            prev = pd.read_csv(path, usecols=key_cols)
            have = set(zip(*(prev[c].astype(str) for c in key_cols)))
            mask = ~df.apply(lambda r: tuple(str(r[c]) for c in key_cols) in have, axis=1)
            df = df[mask]
        except Exception:
            pass
    if df.empty:
        return 0
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(df)


def chain_month_path(root, d: dt.date) -> Path:
    """chain_history/<YYYY-MM>.csv for session date d."""
    return Path(root) / f"{d:%Y-%m}.csv"


def open_trade_legs(trades: list[dict] | None = None) -> dict[str, dict[str, set[float]]]:
    """{ticker: {expiry: {strikes}}} for every open trade, shadows included.

    The nearest-target expiries and the ±band strike window drift away from a
    held spread (a 21-DTE expiry stops being "nearest 14" a week later; a strike
    10% OTM leaves the band on a down move), so without this mark.py carries
    forward stale marks and TP/SL/expiry outcomes are wrong.
    """
    if trades is None:
        from paper.models import read_trades
        trades = read_trades()
    legs: dict[str, dict[str, set[float]]] = {}
    for t in trades:
        if t.get("status") != "open" or not t.get("expiry"):
            continue
        strikes = legs.setdefault(str(t["ticker"]).upper(), {}).setdefault(str(t["expiry"]), set())
        for k in (t.get("long_strike"), t.get("short_strike")):
            try:
                strikes.add(round(float(k), 4))
            except (TypeError, ValueError):
                pass
    return legs


def _one_expiry(tk, ticker, expiry, spot, today, ts, band, keep_strikes=frozenset()):
    try:
        ch = tk.option_chain(expiry)
    except Exception:
        return []
    dte = (dt.date.fromisoformat(expiry) - today).days
    T = max(dte, 1) / 365.0
    rows = []
    for kind, frame in (("C", ch.calls), ("P", ch.puts)):
        if frame is None or frame.empty:
            continue
        df = frame.copy()
        df["moneyness"] = df["strike"] / spot - 1.0
        in_band = df["moneyness"].abs() <= band
        held = df["strike"].astype(float).round(4).isin(keep_strikes)
        df = df[in_band | held]
        for _, r in df.iterrows():
            bid, ask = float(r.get("bid") or 0), float(r.get("ask") or 0)
            if ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            strike = float(r["strike"])
            iv = implied_vol(mid, spot, strike, T, r=RISK_FREE, kind=kind)
            g = bs_greeks(spot, strike, T, RISK_FREE, iv, kind=kind)
            rows.append({
                "date": today.isoformat(), "ts_utc": ts, "ticker": ticker,
                "spot": round(spot, 4), "expiry": expiry, "dte": dte,
                "type": kind, "strike": strike,
                "moneyness": round(float(r["moneyness"]), 5),
                "bid": bid, "ask": ask, "mid": round(mid, 4),
                "spread_pct": round((ask - bid) / mid, 4) if mid > 0 else None,
                "volume": _safe_int(r.get("volume")),
                "open_interest": _safe_int(r.get("openInterest")),
                "yf_iv": float(r.get("impliedVolatility") or np.nan),
                "iv": None if not np.isfinite(iv) else round(iv, 6),
                "delta": None if not np.isfinite(g["delta"]) else round(g["delta"], 6),
                "gamma": None if not np.isfinite(g["gamma"]) else round(g["gamma"], 8),
                "vega": None if not np.isfinite(g["vega"]) else round(g["vega"], 6),
                "theta": None if not np.isfinite(g["theta"]) else round(g["theta"], 6),
            })
    return rows


def _assert_not_stale(df: pd.DataFrame, root, today: dt.date) -> None:
    """Refuse to write a chain that is a copy of the last stored session — the
    calendar can miss an unscheduled closure, and Yahoo can serve a frozen chain."""
    files = sorted(Path(root).glob("*.csv"))[-2:]
    if not files:
        return
    prev = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    prev = prev[prev["date"].astype(str) < today.isoformat()]
    if prev.empty:
        return
    last = prev["date"].astype(str).max()
    key = ["ticker", "expiry", "type", "strike"]
    j = df[key + ["bid", "ask"]].merge(
        prev.loc[prev["date"].astype(str) == last, key + ["bid", "ask"]],
        on=key, suffixes=("", "_prev"))
    if len(j) < STALE_MIN_OVERLAP:
        return
    same = ((j["bid"] == j["bid_prev"]) & (j["ask"] == j["ask_prev"])).mean()
    if same >= STALE_IDENTICAL_FRAC:
        raise RuntimeError(
            f"{today}: {same:.1%} of {len(j)} quotes identical to {last} — "
            "market closed or stale feed; refusing to write")


def snapshot_chains(tickers, path="chain_history",
                    target_dtes=(14, 21, 30, 60), band=0.10,
                    dte_window=(7, 120), trade_legs=None, now=None):
    """
    Append today's option surface for each ticker to path/<YYYY-MM>.csv.
    band=0.10 -> strikes within +/-10% of spot.
    target_dtes -> expiry nearest each target (2–4 week tenors + 60d).
    trade_legs -> {ticker: {expiry: {strikes}}} always captured regardless of
                  band/target (default: open + shadow trades in data/trades.csv).
    """
    import yfinance as yf
    ok, why = chain_capture_status(now)
    if not ok:
        print(f"chains: skipped — {why}")
        return 0
    today = session_date(now)
    ts = (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")
    if trade_legs is None:
        trade_legs = open_trade_legs()
    all_rows, failed = [], []

    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="1d")
            if hist.empty:
                failed.append((t, "no spot")); continue
            spot = float(hist["Close"].iloc[-1])

            exps = list(tk.options or [])
            if not exps:
                failed.append((t, "no expiries")); continue

            chosen = set()
            for target in target_dtes:
                best, gap = None, 1e9
                for e in exps:
                    try:
                        d = (dt.date.fromisoformat(e) - today).days
                    except ValueError:
                        continue
                    if dte_window[0] <= d <= dte_window[1] and abs(d - target) < gap:
                        best, gap = e, abs(d - target)
                if best:
                    chosen.add(best)

            held = trade_legs.get(t.upper(), {})
            for e in held:
                if e in exps:
                    chosen.add(e)
                elif e >= today.isoformat():
                    failed.append((t, f"held expiry {e} not listed"))

            if not chosen:
                failed.append((t, "no expiry in window")); continue

            for e in sorted(chosen):
                all_rows += _one_expiry(tk, t, e, spot, today, ts, band,
                                        keep_strikes=frozenset(held.get(e, ())))

        except Exception as ex:
            failed.append((t, str(ex)[:60]))

    if not all_rows:
        raise RuntimeError(f"snapshot captured 0 rows. failures: {failed}")

    df = pd.DataFrame(all_rows)[CHAIN_SCHEMA]
    _assert_not_stale(df, path, today)
    out = chain_month_path(path, today)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = _append_dedupe(str(out), df, CHAIN_SCHEMA, ["date", "ticker"])
    if n == 0:
        print("already snapshotted today; nothing written")
    else:
        print(f"wrote {n} chain rows / {df['ticker'].nunique()} tickers -> {out}")
    missing = [
        (t, e, k) for t, by_exp in trade_legs.items() for e, ks in by_exp.items()
        if e >= today.isoformat() for k in ks
        if not ((df["ticker"] == t) & (df["expiry"] == e) & (df["type"] == "C")
                & np.isclose(df["strike"].astype(float), k)).any()
    ]
    if missing:
        print(f"held legs not captured: {missing}")
    if failed:
        print(f"failed: {failed}")
    return n


def snapshot_underlyings(tickers, path="underlying_history.csv", backfill_years: int = 2,
                         force_period: str | None = None):
    """
    Append completed daily OHLC + realized vol. Incremental runs persist the
    previous session (T-1), not today — Yahoo often omits the current daily
    bar until late evening, and a mid-day 1d quote is not a real OHLC.

    force_period overrides the first-write-vs-incremental period choice —
    used by backfill_underlyings.py to pull a full history window for
    specific tickers even when the file already has other tickers in it
    (the normal "first_write" flag is file-global, not per-ticker).
    """
    import yfinance as yf
    asof = session_date()
    today = asof.isoformat()
    target = previous_session_date(asof)
    target_s = target.isoformat()
    existing_dates: set[tuple[str, str]] = set()
    first_write = not os.path.exists(path)
    if not first_write:
        try:
            prev = pd.read_csv(path, usecols=["date", "ticker"])
            existing_dates = set(zip(prev["date"].astype(str), prev["ticker"].astype(str)))
        except Exception:
            pass

    rows = []
    failed = []
    # Always pull the full window: rv20_pctile ranks against 252 trailing RV
    # obs (needs >=60), and rv60 needs 60 bars — a 3mo pull left every
    # incremental rv20_pctile nan and rv60 one holiday away from nan.
    period = force_period or f"{backfill_years}y"
    cutoff = (asof - dt.timedelta(days=10)).isoformat()
    keep_all_history = first_write or force_period is not None

    for t in tickers:
        try:
            px = yf.Ticker(t).history(period=period, auto_adjust=False)
            if px is None or px.empty:
                failed.append((t, "no history")); continue
            px = px.copy()
            px["rv20_yz"] = yang_zhang_vol(px, 20)
            px["rv60_yz"] = yang_zhang_vol(px, 60)
            px["rv20_c2c"] = close_to_close_vol(px["Close"], 20)

            for idx, r in px.iterrows():
                d = index_session_date(idx).isoformat()
                if d >= today:
                    # never persist the in-progress / unpublished session
                    continue
                if (d, t) in existing_dates:
                    continue
                # on incremental runs, only keep recent bars (avoid re-pulling years)
                if not keep_all_history and d < cutoff:
                    continue
                rv20 = float(r["rv20_yz"]) if np.isfinite(r["rv20_yz"]) else float("nan")
                pct = float("nan")
                try:
                    sub = px.loc[:idx, "rv20_yz"]
                    pct = vol_percentile(sub)
                except Exception:
                    pass
                rows.append({
                    "date": d, "ticker": t,
                    "open": round(float(r["Open"]), 4),
                    "high": round(float(r["High"]), 4),
                    "low": round(float(r["Low"]), 4),
                    "close": round(float(r["Close"]), 4),
                    "volume": _safe_int(r.get("Volume")),
                    "rv20_yz": None if not np.isfinite(rv20) else round(rv20, 6),
                    "rv60_yz": None if not np.isfinite(r["rv60_yz"]) else round(float(r["rv60_yz"]), 6),
                    "rv20_c2c": None if not np.isfinite(r["rv20_c2c"]) else round(float(r["rv20_c2c"]), 6),
                    "rv20_pctile": None if not np.isfinite(pct) else round(pct, 2),
                })
        except Exception as ex:
            failed.append((t, str(ex)[:60]))

    n_target = sum(1 for r in rows if r["date"] == target_s)
    already = sum(1 for d, _t in existing_dates if d == target_s)
    print(f"underlying T-1 {target_s}: {n_target} new, {already} already stored "
          f"(skipping in-session {today})")

    if not rows:
        print(f"underlying: nothing new through {target_s}")
        return 0

    df = pd.DataFrame(rows)[UNDERLYING_SCHEMA]
    n = _append_dedupe(path, df, UNDERLYING_SCHEMA, ["date", "ticker"])
    print(f"wrote {n} underlying rows -> {path}" + (" (incl. backfill)" if first_write else ""))
    if failed:
        print(f"underlying failed: {failed}")
    return n


def _next_earnings_date(tk) -> dt.date | None:
    try:
        cal = tk.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            ed = cal["Earnings Date"]
            ed = ed[0] if isinstance(ed, list) else ed
            if hasattr(ed, "date"):
                return ed.date()
            if isinstance(ed, dt.date):
                return ed
        if hasattr(cal, "empty") and not cal.empty:
            val = cal.iloc[0, 0]
            return val.date() if hasattr(val, "date") else val
    except Exception:
        return None
    return None


def snapshot_earnings(tickers, path="earnings_calendar.csv"):
    """Persist next earnings date as known on each asof day (forward-looking snapshot)."""
    import yfinance as yf
    asof = session_date()
    rows, failed = [], []
    for t in tickers:
        try:
            ed = _next_earnings_date(yf.Ticker(t))
            if ed is None:
                failed.append((t, "no earnings date")); continue
            rows.append({
                "asof": asof.isoformat(),
                "ticker": t,
                "earnings_date": ed.isoformat(),
                "days_to_earnings": (ed - asof).days,
            })
        except Exception as ex:
            failed.append((t, str(ex)[:60]))

    if not rows:
        print("earnings: nothing written")
        return 0

    df = pd.DataFrame(rows)[EARNINGS_SCHEMA]
    n = _append_dedupe(path, df, EARNINGS_SCHEMA, ["asof", "ticker"])
    if n == 0:
        print("earnings already snapshotted today")
    else:
        print(f"wrote {n} earnings rows -> {path}")
    if failed:
        print(f"earnings failed: {failed}")
    return n


def _github_output(**kv) -> None:
    """Expose step outputs to the Actions workflow (no-op locally)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


def run_all(tickers):
    now = dt.datetime.now(dt.timezone.utc)
    captured, _ = chain_capture_status(now)
    _github_output(chains_captured=str(captured).lower(), session=session_date(now).isoformat())
    n_chain = snapshot_chains(tickers, now=now)
    n_und = snapshot_underlyings(tickers)
    n_earn = snapshot_earnings(tickers)
    return n_chain, n_und, n_earn


if __name__ == "__main__":
    from screener import UNIVERSE
    run_all(UNIVERSE)
