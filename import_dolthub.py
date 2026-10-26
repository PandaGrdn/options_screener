"""
import_dolthub.py — free historical option chains for backtesting.

Source: DoltHub's post-no-preference/options (public SQL API, no key). Writes
historical/chain/YYYY-MM.csv in the same columns as chain_history/ (plus
`vendor_iv`), and historical/underlying.csv + historical/splits.csv from
yfinance. The live chain_history/ is never touched — keep the two apart so the
paper-trading scorecard stays out-of-sample.

What this data is (checked 2026-10-25):
  - daily EOD bid/ask per strike + vendor IV/greeks; NO volume / open interest
  - ~3-4 expiries per day, ~2 weeks apart (~14-60 DTE); an expiry drops out
    about 2 weeks before it expires, so a held spread can't be marked to expiry
    from this data — backtest.py settles at expiry from the underlying close
  - strikes ~±30% of spot
  - 17/23 UNIVERSE names (no SHOP SMH ARKK GDX QQQ IWM); COIN/MRVL start later
  - NVDA 2026-10-23 bid/ask match our own chain_history to the penny
  - strikes are as-traded, Yahoo closes are split-adjusted: spot is rebuilt by
    un-adjusting the close with the split history, and any ticker-date whose
    strikes aren't centred on that spot is dropped (status spot_mismatch)

`yf_iv`, `volume`, `open_interest`, `ts_utc` are blank; `iv`/greeks are solved
from mid exactly like snapshot.py, so live and historical IV are comparable.

historical/ is gitignored (~200MB, re-downloadable). Resumable: finished
(date, ticker) pairs are logged in historical/chain/_done.csv, so a rerun only
fetches what's missing. ~0.5s per request; 4 workers do 2 years in ~30 min.

    python import_dolthub.py                 # 2 years, whole UNIVERSE
    python import_dolthub.py --years 1 --tickers NVDA AMD
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from screener import UNIVERSE
from snapshot import (
    CHAIN_SCHEMA, RISK_FREE, chain_month_path, nyse, snapshot_underlyings,
)
from spread_eval import bs_greeks, implied_vol

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
ROOT = Path("historical")
CHAIN_DIR = ROOT / "chain"
DONE = CHAIN_DIR / "_done.csv"
UNDERLYING = ROOT / "underlying.csv"
SPLITS = ROOT / "splits.csv"
HIST_SCHEMA = CHAIN_SCHEMA + ["vendor_iv"]
# median strike / spot outside this range => spot and strikes disagree (split
# mis-adjustment or bad close); drop rather than store a wrong moneyness
CENTRED = (0.80, 1.25)


# --- DoltHub ------------------------------------------------------------------

def _query(sql: str, retries: int = 4, timeout: int = 60) -> list[dict]:
    url = API + "?" + urllib.parse.urlencode({"q": sql})
    err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                d = json.load(resp)
            if d.get("query_execution_status") == "Success":
                return d.get("rows", [])
            err = d.get("query_execution_message")
        except Exception as ex:  # network / timeout / bad JSON
            err = str(ex)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"DoltHub query failed after {retries} tries: {err}")


def fetch_day(date: str, ticker: str) -> pd.DataFrame:
    rows = _query(
        "SELECT date, act_symbol, expiration, strike, call_put, bid, ask, vol "
        f"FROM option_chain WHERE date='{date}' AND act_symbol='{ticker}'"
    )
    return pd.DataFrame(rows)


# --- underlying + splits ------------------------------------------------------

def fetch_underlying(tickers, years: int = 5) -> None:
    """5y OHLC/RV (so rv_percentile's 120-obs gate is live from the first
    backtest date) + split history. Cheap; refetched on every run."""
    import yfinance as yf

    ROOT.mkdir(parents=True, exist_ok=True)
    if UNDERLYING.exists():
        UNDERLYING.unlink()
    snapshot_underlyings(tickers, path=str(UNDERLYING), force_period=f"{years}y")
    rows = []
    for t in tickers:
        s = yf.Ticker(t).splits
        for idx, ratio in s.items():
            rows.append({"ticker": t, "date": pd.Timestamp(idx).date().isoformat(),
                         "ratio": float(ratio)})
    pd.DataFrame(rows, columns=["ticker", "date", "ratio"]).to_csv(SPLITS, index=False)


def unadjusted_closes(und: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's close is split-adjusted; option strikes are not. Before each
    split's ex-date, multiply by its ratio. Returns und + `close_raw`."""
    und = und.copy()
    und["close_raw"] = und["close"].astype(float)
    for _, s in splits.iterrows():
        m = (und["ticker"] == s["ticker"]) & (und["date"].astype(str) < str(s["date"]))
        und.loc[m, "close_raw"] *= float(s["ratio"])
    return und


# --- conversion ---------------------------------------------------------------

def to_chain_rows(raw: pd.DataFrame, ticker: str, date: str, spot: float) -> pd.DataFrame:
    """DoltHub rows -> HIST_SCHEMA, IV/greeks solved from mid as snapshot.py does."""
    d0 = dt.date.fromisoformat(date)
    out = []
    for _, r in raw.iterrows():
        bid = float(r["bid"]) if r["bid"] is not None else 0.0
        ask = float(r["ask"]) if r["ask"] is not None else 0.0
        if ask <= 0:
            continue
        expiry = str(r["expiration"])[:10]
        dte = (dt.date.fromisoformat(expiry) - d0).days
        kind = "C" if str(r["call_put"]).lower().startswith("c") else "P"
        strike = float(r["strike"])
        mid = (bid + ask) / 2.0
        T = max(dte, 1) / 365.0
        iv = implied_vol(mid, spot, strike, T, r=RISK_FREE, kind=kind)
        g = bs_greeks(spot, strike, T, RISK_FREE, iv, kind=kind)
        fin = lambda x, n: None if not np.isfinite(x) else round(float(x), n)  # noqa: E731
        out.append({
            "date": date, "ts_utc": "", "ticker": ticker, "spot": round(spot, 4),
            "expiry": expiry, "dte": dte, "type": kind, "strike": strike,
            "moneyness": round(strike / spot - 1.0, 5),
            "bid": bid, "ask": ask, "mid": round(mid, 4),
            "spread_pct": round((ask - bid) / mid, 4) if mid > 0 else None,
            "volume": None, "open_interest": None, "yf_iv": None,
            "iv": fin(iv, 6), "delta": fin(g["delta"], 6), "gamma": fin(g["gamma"], 8),
            "vega": fin(g["vega"], 6), "theta": fin(g["theta"], 6),
            "vendor_iv": None if r.get("vol") is None else float(r["vol"]),
        })
    return pd.DataFrame(out, columns=HIST_SCHEMA)


def is_centred(rows: pd.DataFrame, spot: float) -> bool:
    if rows.empty:
        return True
    ratio = float(rows["strike"].median()) / spot
    return CENTRED[0] <= ratio <= CENTRED[1]


# --- driver -------------------------------------------------------------------

def sessions(start: dt.date, end: dt.date) -> list[str]:
    return [d.date().isoformat() for d in nyse().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]


def _done() -> set[tuple[str, str]]:
    if not DONE.exists():
        return set()
    d = pd.read_csv(DONE, dtype=str)
    return set(zip(d["date"], d["ticker"]))


def _write(rows: pd.DataFrame, status: list[dict]) -> None:
    for month, part in rows.groupby(rows["date"].str[:7]):
        out = chain_month_path(CHAIN_DIR, dt.date.fromisoformat(f"{month}-01"))
        part[HIST_SCHEMA].to_csv(out, mode="a", header=not out.exists(), index=False)
    pd.DataFrame(status, columns=["date", "ticker", "rows", "status"]).to_csv(
        DONE, mode="a", header=not DONE.exists(), index=False)


def run(tickers=UNIVERSE, years: int = 2, end: dt.date | None = None,
        workers: int = 4, refresh_underlying: bool = True) -> None:
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    if refresh_underlying or not UNDERLYING.exists():
        fetch_underlying(tickers)
    und = unadjusted_closes(pd.read_csv(UNDERLYING), pd.read_csv(SPLITS))
    spot = {(r.date, r.ticker): r.close_raw for r in und.itertuples()}

    end = end or dt.date.fromisoformat(str(und["date"].max()))
    start = end - dt.timedelta(days=round(365.25 * years))
    have = _done()
    todo = [(d, t) for d in sessions(start, end) for t in tickers if (d, t) not in have]
    print(f"import_dolthub: {start}..{end}, {len(todo)} (date, ticker) requests to go")

    def job(d, t):
        s = spot.get((d, t))
        if s is None or not np.isfinite(s):
            return d, t, None, "no_spot"
        raw = fetch_day(d, t)
        if raw.empty:
            return d, t, None, "empty"
        rows = to_chain_rows(raw, t, d, float(s))
        if not is_centred(rows, float(s)):
            return d, t, None, "spot_mismatch"
        return d, t, rows, "ok"

    buf, status, failed, t0 = [], [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, d, t) for d, t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                d, t, rows, st = f.result()
            except Exception as e:  # leave out of _done so a rerun retries it
                failed += 1
                print(f"  fetch failed: {e}")
                continue
            status.append({"date": d, "ticker": t, "rows": 0 if rows is None else len(rows), "status": st})
            if rows is not None:
                buf.append(rows)
            if len(status) >= 200 or i == len(futs):
                _write(pd.concat(buf) if buf else pd.DataFrame(columns=HIST_SCHEMA), status)
                buf, status = [], []
                print(f"  {i}/{len(futs)} done, {failed} failed, {time.time() - t0:.0f}s")
    summary()


def summary() -> None:
    d = pd.read_csv(DONE)
    by = d.pivot_table(index="ticker", columns="status", values="date", aggfunc="count", fill_value=0)
    print(by.to_string())
    ok = d[d["status"] == "ok"]
    if not ok.empty:
        print("first/last ok date per ticker:")
        print(ok.groupby("ticker")["date"].agg(["min", "max"]).to_string())


def validate_against_live(live_root="chain_history") -> pd.Series:
    """Overlap check: bid/ask agreement with our own snapshots on shared dates."""
    from paper import read_chain_history
    live = read_chain_history(Path(live_root))
    hist = read_chain_history(CHAIN_DIR)
    key = ["date", "ticker", "expiry", "type", "strike"]
    j = live[key + ["bid", "ask", "spot"]].merge(hist[key + ["bid", "ask", "spot"]], on=key,
                                                 suffixes=("_live", "_hist"))
    exact = (j["bid_live"] == j["bid_hist"]) & (j["ask_live"] == j["ask_hist"])
    out = pd.Series({
        "overlap_rows": len(j),
        "bid_ask_exact": round(float(exact.mean()), 4) if len(j) else np.nan,
        "spot_median_abs_diff_pct": round(float((j["spot_hist"] / j["spot_live"] - 1).abs().median() * 100), 4)
        if len(j) else np.nan,
    })
    print(out.to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--tickers", nargs="*", default=UNIVERSE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-refresh-underlying", action="store_true")
    a = ap.parse_args()
    run(a.tickers, years=a.years, workers=a.workers, refresh_underlying=not a.no_refresh_underlying)
    validate_against_live()
