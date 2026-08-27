"""
snapshot.py — daily market data capture (not model outputs).

Stores:
  chain_history.csv      raw quotes + own IV/greeks across a wider surface
                         (same-session; cannot be backfilled)
  underlying_history.csv OHLC + realized vol for the *previous* completed
                         session (Yahoo's daily bar is reliable on T+1)
  earnings_calendar.csv  next earnings date as known each asof day

yfinance's impliedVolatility is frequently stale; we still keep yf_iv but also
solve our own IV from mid and store BS greeks. You cannot backfill option
chains, so capture them from day one. Underlying OHLC *can* be backfilled.
"""

from __future__ import annotations

import os
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from screener import close_to_close_vol, yang_zhang_vol, vol_percentile
from spread_eval import implied_vol, bs_greeks

NY = ZoneInfo("America/New_York")


def session_date(now: dt.datetime | None = None) -> dt.date:
    """US equity session date (America/New_York), not the runner's UTC date."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(NY).date()


def index_session_date(idx) -> dt.date:
    """Map a yfinance DatetimeIndex label to the NY session date."""
    ts = pd.Timestamp(idx)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY)
    return ts.date()


def previous_session_date(asof: dt.date | None = None) -> dt.date:
    """Last weekday before asof (Fri if asof is Mon). Ignores market holidays."""
    d = (asof or session_date()) - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


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


def _one_expiry(tk, ticker, expiry, spot, today, ts, band):
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
        df = df[df["moneyness"].abs() <= band]
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


def snapshot_chains(tickers, path="chain_history.csv",
                    target_dtes=(14, 30, 45, 60, 90), band=0.25,
                    dte_window=(7, 120)):
    """
    Append today's option surface for each ticker.
    band=0.25 -> strikes within +/-25% of spot.
    target_dtes -> expiry nearest each target (term structure + wings).
    """
    import yfinance as yf
    today = session_date()
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
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

            if not chosen:
                failed.append((t, "no expiry in window")); continue

            for e in sorted(chosen):
                all_rows += _one_expiry(tk, t, e, spot, today, ts, band)

        except Exception as ex:
            failed.append((t, str(ex)[:60]))

    if not all_rows:
        raise RuntimeError(f"snapshot captured 0 rows. failures: {failed}")

    df = pd.DataFrame(all_rows)[CHAIN_SCHEMA]
    n = _append_dedupe(path, df, CHAIN_SCHEMA, ["date", "ticker"])
    if n == 0:
        print("already snapshotted today; nothing written")
    else:
        print(f"wrote {n} chain rows / {df['ticker'].nunique()} tickers -> {path}")
    if failed:
        print(f"failed: {failed}")
    return n


def snapshot_underlyings(tickers, path="underlying_history.csv", backfill_years: int = 2):
    """
    Append completed daily OHLC + realized vol. Incremental runs persist the
    previous session (T-1), not today — Yahoo often omits the current daily
    bar until late evening, and a mid-day 1d quote is not a real OHLC.
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
    period = f"{backfill_years}y" if first_write else "3mo"
    cutoff = (asof - dt.timedelta(days=10)).isoformat()

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
                if not first_write and d < cutoff:
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


def run_all(tickers):
    n_chain = snapshot_chains(tickers)
    n_und = snapshot_underlyings(tickers)
    n_earn = snapshot_earnings(tickers)
    return n_chain, n_und, n_earn


if __name__ == "__main__":
    from screener import UNIVERSE
    run_all(UNIVERSE)
