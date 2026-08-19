"""
snapshot.py — replaces snapshot_iv() in screener.py

Stores RAW QUOTES (bid/ask/strike/spot/dte), not derived IV. yfinance's
impliedVolatility field is frequently stale or wrong; storing raw quotes means
you can recompute IV correctly later with your own solver. You cannot backfill
this, so capture it from day one.

Captures both calls AND puts: put skew is the informative wing, and you need
both to reconstruct the surface.
"""

from __future__ import annotations
import os, datetime as dt
import numpy as np
import pandas as pd

SCHEMA = ["date", "ts_utc", "ticker", "spot", "expiry", "dte", "type",
          "strike", "moneyness", "bid", "ask", "mid", "spread_pct",
          "volume", "open_interest", "yf_iv"]


def _safe_int(x) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        return int(x)
    except (TypeError, ValueError):
        return 0


def _one_expiry(tk, ticker, expiry, spot, today, ts, band):
    try:
        ch = tk.option_chain(expiry)
    except Exception:
        return []
    dte = (dt.date.fromisoformat(expiry) - today).days
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
            rows.append({
                "date": today.isoformat(), "ts_utc": ts, "ticker": ticker,
                "spot": round(spot, 4), "expiry": expiry, "dte": dte,
                "type": kind, "strike": float(r["strike"]),
                "moneyness": round(float(r["moneyness"]), 5),
                "bid": bid, "ask": ask, "mid": round(mid, 4),
                "spread_pct": round((ask - bid) / mid, 4) if mid > 0 else None,
                "volume": _safe_int(r.get("volume")),
                "open_interest": _safe_int(r.get("openInterest")),
                "yf_iv": float(r.get("impliedVolatility") or np.nan),
            })
    return rows


def snapshot_chains(tickers, path="chain_history.csv",
                    target_dtes=(30, 60), band=0.10, dte_window=(7, 90)):
    """
    Append today's near-the-money chain for each ticker.
    band=0.10 -> keep strikes within +/-10% of spot.
    target_dtes -> grab the expiry nearest each target (term structure).
    """
    import yfinance as yf
    today = dt.date.today()
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

    df = pd.DataFrame(all_rows)[SCHEMA]

    # idempotence: don't double-write if the job reruns the same day
    if os.path.exists(path):
        try:
            prev = pd.read_csv(path, usecols=["date", "ticker"])
            have = set(zip(prev["date"], prev["ticker"]))
            df = df[~df.apply(lambda r: (r["date"], r["ticker"]) in have, axis=1)]
        except Exception:
            pass

    if df.empty:
        print("already snapshotted today; nothing written")
        return 0

    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"wrote {len(df)} rows / {df['ticker'].nunique()} tickers -> {path}")
    if failed:
        print(f"failed: {failed}")
    return len(df)


if __name__ == "__main__":
    from screener import UNIVERSE
    snapshot_chains(UNIVERSE)