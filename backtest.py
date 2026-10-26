"""
backtest.py — hold-to-expiry backtest of the live spread picker on historical/.

For each ticker, every `step` sessions, run the live `evaluate()` exactly as the
cron does (market default: ATM IV, zero drift, 21-day horizon) and settle every
candidate spread AT EXPIRY from the underlying close: value = intrinsic, which
is exact and needs no option quotes after entry (historical/chain can't mark a
spread to expiry anyway — expiries drop out ~2 weeks early; see import_dolthub).

What is scored:
  model_prob_exp  P(S_T > breakeven) under GBM at ATM IV, zero edge drift —
                  the market-default *hold-to-expiry* probability. This is the
                  one comparable with the realized hold-to-expiry outcome.
  live_prob       evaluate()'s path-dependent TP/SL/time-stop probability.
                  Recorded for reference only: it scores a different exit
                  policy, so it is never calibrated against these outcomes.

Entries are the cheapness_gate / earnings / verdict-agnostic universe; the
report splits by each so you can see whether any filter selects winners.
Overlapping holds (step < hold length) make neighbouring trades correlated —
n overstates the effective sample.

Vol-forecast mode (`--vol-forecast`, after a normal run): re-decides every
entry under alternative vol forecasts built only from realized-vol history as
of the entry date (Phase 2 rules: vol-only, drift 0, non-overlapping fit
windows, no ML frameworks) and compares them to the market default (ATM IV):
forecast accuracy vs realized vol to expiry, calibration, and P&L of the spreads
each forecast would TRADE (best candidate by hold-to-expiry log growth > 0).

    python import_dolthub.py      # first
    python backtest.py [--step 5]
    python backtest.py --vol-forecast
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import signals
from import_dolthub import CHAIN_DIR, ROOT, SPLITS, UNDERLYING, unadjusted_closes
from paper import read_chain_history
from paper.entry import atm_iv_from_chain as atm_iv_30
from paper.score import auc, brier, calibration_table
from snapshot import RISK_FREE, nyse
from spread_eval import FEE_PER_CONTRACT, MAX_PER_TRADE_PCT, evaluate

HORIZON = 21  # paper.auto.DEFAULT_HORIZON
FEES = 4 * FEE_PER_CONTRACT / 100.0  # 2 legs, open + close, per share (as simulate())
TRADES_OUT = ROOT / "backtest_trades.csv"
REPORT_OUT = ROOT / "backtest_report.txt"
VOL_REPORT_OUT = ROOT / "vol_forecast_report.txt"
VOL_TRADES_OUT = ROOT / "vol_forecast_trades.csv"
EARNINGS = ROOT / "earnings.csv"
# DoltHub quotes are sometimes taken intraday, not at the close (put-call parity
# implies a spot 0.4-2% off the close on ~15% of ticker-dates, mostly 2026).
# Pairing those quotes with the close misstates moneyness — skip the entry.
PARITY_MAX_GAP_PCT = 0.5
ETFS = frozenset({"XLE", "XLF", "XBI", "SMH", "ARKK", "GDX", "SPY", "QQQ", "IWM"})


# --- model + settlement -------------------------------------------------------

def expiry_model(spot: float, k1: float, k2: float, dte: int, debit: float, vol: float,
                 drift: float = 0.0, r: float = RISK_FREE, n: int = 20_000, seed: int = 7,
                 frac: float = MAX_PER_TRADE_PCT) -> dict:
    """Hold-to-expiry call debit spread under GBM. prob is closed-form
    P(S_T > k1 + debit + fees); ev / log_growth from seeded terminal samples.
    T = dte/365 — the basis snapshot.py solves IV on.

    log_growth is Kelly at the position size actually taken: E[log(1 + frac*r)]
    with frac = 4% of the portfolio at risk. (Whole-stake E[log(1+r)] is ~-3 for
    any spread held to expiry — half the paths lose 100% — so it can't gate.)"""
    T = max(int(dte), 1) / 365.0
    sig = max(float(vol), 1e-4)
    mu = r + drift
    be = k1 + debit + FEES
    if be >= k2:
        prob = 0.0
    else:
        prob = float(norm.cdf((math.log(spot / be) + (mu - 0.5 * sig * sig) * T) / (sig * math.sqrt(T))))
    z = np.random.default_rng(seed).standard_normal(n)
    st = spot * np.exp((mu - 0.5 * sig * sig) * T + sig * math.sqrt(T) * z)
    pnl = np.clip(st - k1, 0.0, k2 - k1) - debit - FEES
    rets = np.clip(pnl / (debit + FEES), -0.999, 10.0)
    return {"prob": prob, "ev": float(pnl.mean()), "log_growth": float(np.mean(np.log1p(frac * rets)))}


def settle(k1: float, k2: float, st: float, debit: float) -> tuple[float, float]:
    """(pnl per share, return on debit+fees) holding to expiry."""
    pnl = min(max(st - k1, 0.0), k2 - k1) - debit - FEES
    return pnl, pnl / (debit + FEES)


def parity_gap_pct(day: pd.DataFrame, r: float = RISK_FREE) -> float:
    """Median (C - P + K e^{-rT}) / spot - 1, in %, over near-ATM strikes with
    both a call and a put. ~0 when the quotes were taken at the close the
    spot comes from; nan if no pairs."""
    k = ["expiry", "strike"]
    c = day[day["type"] == "C"][k + ["mid", "dte", "moneyness", "spot"]]
    p = day[day["type"] == "P"][k + ["mid"]]
    j = c.merge(p, on=k, suffixes=("_c", "_p"))
    j = j[j["moneyness"].abs() < 0.03]
    if j.empty:
        return float("nan")
    s = j["mid_c"] - j["mid_p"] + j["strike"] * np.exp(-r * j["dte"] / 365.0)
    return float((s / j["spot"] - 1).median() * 100)


def settle_session(expiry: str) -> str:
    """Last NYSE session on/before expiry (Good Friday expiries settle Thursday)."""
    return nyse().date_to_session(pd.Timestamp(expiry), direction="previous").date().isoformat()


# --- inputs -------------------------------------------------------------------

def load_earnings(tickers) -> pd.DataFrame:
    """Past earnings dates from yfinance (cached). Empty frame if unavailable."""
    if EARNINGS.exists():
        return pd.read_csv(EARNINGS)
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=16)
        except Exception:
            continue
        if ed is None:
            continue
        for idx in ed.index:
            rows.append({"ticker": t, "earnings_date": pd.Timestamp(idx).date().isoformat()})
    df = pd.DataFrame(rows, columns=["ticker", "earnings_date"]).drop_duplicates()
    df.to_csv(EARNINGS, index=False)
    return df


def _iv_rank_series(chain: pd.DataFrame) -> dict[str, pd.Series]:
    """Per ticker: daily ATM IV30 (live recipe) and its trailing IV rank,
    nan below signals.MIN_IV_RANK_DAYS — same rule as signals.iv_rank."""
    out = {}
    for t, g in chain.groupby("ticker"):
        iv = pd.Series({d: atm_iv_30(day, float(day["spot"].iloc[0]))
                        for d, day in g.groupby("date")}).sort_index().dropna()
        rank = pd.Series(np.nan, index=iv.index)
        for i in range(len(iv)):
            if i + 1 >= signals.MIN_IV_RANK_DAYS:
                s = iv.iloc[max(0, i + 1 - 252): i + 1]
                rank.iloc[i] = float((s < s.iloc[-1]).mean() * 100)
        out[t] = pd.DataFrame({"iv30": iv, "iv_rank": rank})
    return out


# --- run ----------------------------------------------------------------------

def run(step: int = 5) -> pd.DataFrame:
    chain = read_chain_history(CHAIN_DIR)
    chain["date"] = chain["date"].astype(str)
    und = unadjusted_closes(pd.read_csv(UNDERLYING), pd.read_csv(SPLITS))
    close = {(r.ticker, str(r.date)): r.close_raw for r in und.itertuples()}
    last_px = str(und["date"].max())
    splits = pd.read_csv(SPLITS)
    tickers = sorted(chain["ticker"].unique())
    earn = load_earnings(tickers)
    ivr = _iv_rank_series(chain)
    signals.underlying_history_path = lambda: UNDERLYING  # gate reads historical RV

    rows, skipped = [], {"open": 0, "split_in_hold": 0, "no_candidates": 0, "quotes_off_close": 0}
    for t, g in chain.groupby("ticker"):
        dates = sorted(g["date"].unique())[::step]
        tsplits = splits.loc[splits["ticker"] == t, "date"].astype(str).tolist()
        tearn = earn.loc[earn["ticker"] == t, "earnings_date"].astype(str).tolist()
        for d in dates:
            day = g[g["date"] == d]
            spot = float(day["spot"].iloc[0])
            gap = parity_gap_pct(day)
            if np.isfinite(gap) and abs(gap) > PARITY_MAX_GAP_PCT:
                skipped["quotes_off_close"] += 1
                continue
            res = evaluate(t, spot, day, None, 0.0, HORIZON)
            cands = res.get("candidates") or []
            if not cands:
                skipped["no_candidates"] += 1
                continue
            expiry = str(res["expiry"])
            sd = settle_session(expiry)
            if sd > last_px:
                skipped["open"] += 1
                continue
            if any(d < s <= sd for s in tsplits):
                skipped["split_in_hold"] += 1
                continue
            st = float(close[(t, sd)])
            rvp = signals.rv_percentile(t, asof=d)
            rv20 = signals._latest_rv(t, asof=d)
            iv30 = ivr[t]["iv30"].get(d, np.nan)
            rank = ivr[t]["iv_rank"].get(d, np.nan)
            ivrv = iv30 - rv20 if np.isfinite(iv30) and np.isfinite(rv20) else np.nan
            gate = (np.isfinite(rvp) and np.isfinite(ivrv) and rvp <= signals.RV_PERCENTILE_MAX
                    and ivrv <= signals.IV_RV_SPREAD_MAX
                    and (not np.isfinite(rank) or rank <= signals.IV_RANK_MAX))
            # ETFs have no earnings; an empty list for a stock means the fetch failed
            earn_in_hold = (any(d < e <= sd for e in tearn) if tearn
                            else (False if t in ETFS else None))
            for c in cands:
                k1, k2 = float(c["long_strike"]), float(c["short_strike"])
                debit, mid = float(c["entry_debit"]), float(c["entry_mid"])
                m = expiry_model(spot, k1, k2, c["dte"], debit, res["forecast_vol"])
                pnl, ret = settle(k1, k2, st, debit)
                pnl_mid, _ = settle(k1, k2, st, mid)
                rows.append({
                    "date": d, "ticker": t, "spot": spot, "parity_gap_pct": gap,
                    "expiry": expiry, "dte": c["dte"],
                    "long_strike": k1, "short_strike": k2, "entry_debit": debit, "entry_mid": mid,
                    "chosen": k1 == float(res["long_strike"]) and k2 == float(res["short_strike"]),
                    "market_iv": res["forecast_vol"],
                    "live_verdict": c["verdict"], "live_prob": c["prob_profit"],
                    "live_log_growth": c["log_growth"],
                    "model_prob_exp": m["prob"], "model_ev_exp": m["ev"], "model_lg_exp": m["log_growth"],
                    "rv_pctile": rvp, "iv_rv_spread": ivrv, "iv_rank": rank, "gate_pass": bool(gate),
                    "earnings_in_hold": earn_in_hold,
                    "settle_date": sd, "settle_px": st,
                    "pnl": pnl, "ret": ret, "win": int(pnl > 0), "pnl_mid_fill": pnl_mid,
                })
        print(f"  {t}: {sum(r['ticker'] == t and r['chosen'] for r in rows)} entries")
    df = pd.DataFrame(rows)
    ROOT.mkdir(exist_ok=True)
    df.to_csv(TRADES_OUT, index=False)
    print(f"skipped: {skipped}")
    report(df, step, skipped)
    return df


# --- report -------------------------------------------------------------------

def _outcomes(df: pd.DataFrame) -> str:
    if df.empty:
        return "n=0"
    return (f"n={len(df):<5} hit={df['win'].mean():6.1%}  mean ret={df['ret'].mean():+7.1%}  "
            f"median ret={df['ret'].median():+7.1%}  $/contract={df['pnl'].mean() * 100:+7.2f}  "
            f"model p={df['model_prob_exp'].mean():.3f}")


def report(df: pd.DataFrame, step: int, skipped: dict) -> str:
    ch = df[df["chosen"]].copy()
    p, y = ch["model_prob_exp"].tolist(), ch["win"].tolist()
    base = float(np.mean(y)) if y else float("nan")
    L = [
        "=== HOLD-TO-EXPIRY BACKTEST (historical/, DoltHub EOD) ===",
        f"generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  step={step} sessions",
        f"entries (chosen spread) n={len(ch)}  candidates n={len(df)}  tickers={ch['ticker'].nunique()}  "
        f"{ch['date'].min()}..{ch['date'].max()}",
        f"skipped: {skipped}",
        "",
        "1. Calibration — market-default hold-to-expiry model prob vs realized (chosen spreads)",
        f"   Brier model={brier(p, y)}  base-rate={np.mean((base - np.asarray(y)) ** 2) if y else None}  "
        f"AUC={auc(p, y)}",
    ]
    L += [f"   {r['bin']:<8} n={r['n']:<5} pred={r['pred']!s:<6} actual={r['actual']}"
          for r in calibration_table(p, y)]
    L += ["", "2. Outcomes (conservative fills + fees, held to expiry)",
          f"   all chosen      {_outcomes(ch)}",
          f"   mid fills       mean $/contract={ch['pnl_mid_fill'].mean() * 100:+.2f}  "
          f"(fill assumption costs {(ch['pnl_mid_fill'] - ch['pnl']).mean() * 100:.2f}/contract)",
          f"   all candidates  {_outcomes(df)}",
          "", "3. By filter (chosen spreads)"]
    for name, mask in [
        ("cheapness gate PASS", ch["gate_pass"]),
        ("cheapness gate FAIL", ~ch["gate_pass"]),
        ("live verdict TRADE", ch["live_verdict"] == "TRADE"),
        ("live verdict SKIP", ch["live_verdict"] == "SKIP"),
        ("earnings in hold", ch["earnings_in_hold"] == True),  # noqa: E712
        ("no earnings in hold", ch["earnings_in_hold"] == False),  # noqa: E712
    ]:
        L.append(f"   {name:<22} {_outcomes(ch[mask])}")
    L += ["", "4. By ticker (chosen spreads)"]
    for t, g in ch.groupby("ticker"):
        L.append(f"   {t:<6} {_outcomes(g)}")
    L += ["", "5. Implied vs realized vol (ATM IV30 at entry vs realized close-to-close vol to expiry)"]
    vrp = _iv_vs_realized(ch)
    for t, r in vrp.iterrows():
        L.append(f"   {t:<6} n={int(r['n']):<4} IV={r['iv']:.3f}  RV={r['rv']:.3f}  IV-RV={r['iv'] - r['rv']:+.3f}")
    L += ["", "Caveats: entries whose quotes weren't taken at the close (|parity gap| > 0.5%) are skipped; overlapping holds (n overstates independent samples); one 2-year window; "
          "hold-to-expiry only (no TP/SL — data can't mark to expiry); flat ATM vol in the model "
          "(no skew); no volume/OI liquidity filter; 17/23 names."]
    text = "\n".join(L)
    REPORT_OUT.write_text(text + "\n")
    print(text)
    return text


def _iv_vs_realized(ch: pd.DataFrame) -> pd.DataFrame:
    und = pd.read_csv(UNDERLYING)
    und["r"] = np.log(und["close"] / und.groupby("ticker")["close"].shift())
    out = []
    for t, g in ch.groupby("ticker"):
        u = und[und["ticker"] == t].set_index(und.loc[und["ticker"] == t, "date"].astype(str))["r"]
        for _, row in g.iterrows():
            seg = u[(u.index > row["date"]) & (u.index <= row["settle_date"])].dropna()
            if len(seg) >= 5:
                out.append({"ticker": t, "iv": row["market_iv"], "rv": float(seg.std() * math.sqrt(252))})
    d = pd.DataFrame(out)
    if d.empty:
        return pd.DataFrame(columns=["n", "iv", "rv"])
    return d.groupby("ticker").agg(n=("iv", "size"), iv=("iv", "mean"), rv=("rv", "mean"))


# --- vol-forecast mode ----------------------------------------------------------

FORECASTERS = ("market", "rv20", "blend", "har")
FWD = 15            # sessions in a HAR fit window (~21 calendar days, the hold)
HAR_MIN_ROWS = 200  # pooled non-overlapping windows needed before HAR is used


def rv_features(und: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, date): rv20_yz, rv60_yz, rv252 (trailing 252-obs mean of
    rv20_yz) — all known at that close — plus fwd_rv, the close-to-close vol
    over the next FWD sessions, and fwd_end, the date it becomes known. fwd_*
    are fit targets only; a fit at date d may use a row only if fwd_end <= d."""
    out = []
    for t, g in und.sort_values("date").groupby("ticker"):
        g = g.copy()
        g["rv252"] = g["rv20_yz"].rolling(252, min_periods=120).mean()
        r = np.log(g["close"] / g["close"].shift())
        g["fwd_rv"] = r.rolling(FWD).std().shift(-FWD) * math.sqrt(252)
        g["fwd_end"] = g["date"].shift(-FWD)
        g["phase"] = np.arange(len(g)) % FWD == 0  # non-overlapping windows
        out.append(g[["ticker", "date", "rv20_yz", "rv60_yz", "rv252", "fwd_rv", "fwd_end", "phase"]])
    return pd.concat(out, ignore_index=True)


def _har_x(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), np.log(df["rv20_yz"]),
                            np.log(df["rv60_yz"]), np.log(df["rv252"])])


def har_fit(feats: pd.DataFrame, asof: str) -> tuple[np.ndarray, float] | None:
    """Pooled log-HAR on non-overlapping windows whose target is fully known by
    `asof`. Returns (beta, residual variance) or None if too little data."""
    tr = feats[feats["phase"] & (feats["fwd_end"].astype(str) <= str(asof))]
    tr = tr.dropna(subset=["rv20_yz", "rv60_yz", "rv252", "fwd_rv"])
    tr = tr[(tr[["rv20_yz", "rv60_yz", "rv252", "fwd_rv"]] > 0).all(axis=1)]
    if len(tr) < HAR_MIN_ROWS:
        return None
    X, y = _har_x(tr), np.log(tr["fwd_rv"].to_numpy())
    with np.errstate(all="ignore"):  # macOS Accelerate BLAS emits spurious matmul warnings
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
    if not np.isfinite(beta).all():
        return None
    return beta, float(resid.var())


def forecast_vols(entries: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, ticker) entry with a vol per forecaster (nan if the
    features aren't available yet)."""
    f = feats.set_index(["ticker", "date"])
    fits: dict[str, tuple | None] = {}
    rows = []
    for e in entries.itertuples():
        x = f.loc[(e.ticker, e.date)] if (e.ticker, e.date) in f.index else None
        v = {"date": e.date, "ticker": e.ticker, "market": e.market_iv,
             "rv20": np.nan, "blend": np.nan, "har": np.nan}
        if x is not None and np.isfinite(x["rv20_yz"]):
            v["rv20"] = float(x["rv20_yz"])
            if np.isfinite(x["rv60_yz"]) and np.isfinite(x["rv252"]):
                v["blend"] = float(0.5 * x["rv20_yz"] + 0.3 * x["rv60_yz"] + 0.2 * x["rv252"])
                if e.date not in fits:
                    fits[e.date] = har_fit(feats, e.date)
                fit = fits[e.date]
                if fit is not None:
                    beta, s2 = fit
                    xx = _har_x(pd.DataFrame([x]))
                    v["har"] = float(np.exp(xx @ beta + 0.5 * s2)[0])
        rows.append(v)
    return pd.DataFrame(rows)


def decide(cands: pd.DataFrame, vol: float) -> dict | None:
    """Best candidate by hold-to-expiry log growth under `vol`; TRADE if > 0."""
    if not np.isfinite(vol) or vol <= 0:
        return None
    best = None
    for c in cands.itertuples():
        m = expiry_model(c.spot, c.long_strike, c.short_strike, c.dte, c.entry_debit, vol)
        if best is None or m["log_growth"] > best["log_growth"]:
            best = {**m, "row": c}
    best["trade"] = best["log_growth"] > 0
    return best


def realized_to_expiry(entries: pd.DataFrame, und: pd.DataFrame) -> pd.Series:
    u = und.sort_values("date").copy()
    u["r"] = np.log(u["close"] / u.groupby("ticker")["close"].shift())
    by = {t: g.set_index(g["date"].astype(str))["r"] for t, g in u.groupby("ticker")}
    out = []
    for e in entries.itertuples():
        seg = by[e.ticker][(by[e.ticker].index > e.date) & (by[e.ticker].index <= e.settle_date)].dropna()
        out.append(float(seg.std() * math.sqrt(252)) if len(seg) >= 5 else np.nan)
    return pd.Series(out, index=entries.index)


def run_vol_forecasts() -> pd.DataFrame:
    cands = pd.read_csv(TRADES_OUT)
    cands["date"] = cands["date"].astype(str)
    und = pd.read_csv(UNDERLYING)
    und["date"] = und["date"].astype(str)
    entries = cands.drop_duplicates(["date", "ticker"])[
        ["date", "ticker", "market_iv", "settle_date"]].reset_index(drop=True)
    vols = forecast_vols(entries, rv_features(und))
    vols["realized"] = realized_to_expiry(entries, und)

    rows = []
    groups = {k: g for k, g in cands.groupby(["date", "ticker"])}
    for v in vols.itertuples():
        g = groups[(v.date, v.ticker)]
        for name in FORECASTERS:
            d = decide(g, getattr(v, name))
            if d is None:
                continue
            c = d["row"]
            rows.append({"date": v.date, "ticker": v.ticker, "forecaster": name,
                         "vol": getattr(v, name), "realized_vol": v.realized,
                         "market_iv": v.market, "long_strike": c.long_strike,
                         "short_strike": c.short_strike, "entry_debit": c.entry_debit,
                         "prob": d["prob"], "log_growth": d["log_growth"], "trade": d["trade"],
                         "win": c.win, "ret": c.ret, "pnl": c.pnl, "pnl_mid_fill": c.pnl_mid_fill})
    df = pd.DataFrame(rows)
    df.to_csv(VOL_TRADES_OUT, index=False)
    vol_report(df)
    return df


def vol_report(df: pd.DataFrame) -> str:
    # compare every forecaster on the same entries (those where all exist)
    full = df.groupby(["date", "ticker"])["forecaster"].transform("nunique") == len(FORECASTERS)
    d = df[full]
    n_e = d.groupby(["date", "ticker"]).ngroups
    L = ["=== VOL-FORECAST MODE (hold-to-expiry, historical/) ===",
         f"generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  "
         f"entries with every forecaster n={n_e}  {d['date'].min()}..{d['date'].max()}",
         "forecasters: market=ATM IV | rv20=Yang-Zhang 20d | blend=.5 rv20+.3 rv60+.2 rv252mean | "
         "har=walk-forward pooled log-HAR (non-overlapping 15-session windows, fit on data known by entry date)",
         "", "A. Vol forecast vs realized close-to-close vol to expiry (log error = log(forecast/realized))"]
    a = d[d["realized_vol"] > 0].copy()
    a["le"] = np.log(a["vol"] / a["realized_vol"])
    mkt = a[a["forecaster"] == "market"].set_index(["date", "ticker"])["le"].abs()
    for name in FORECASTERS:
        x = a[a["forecaster"] == name].set_index(["date", "ticker"])
        beats = (x["le"].abs() < mkt.reindex(x.index)).mean()
        L.append(f"   {name:<7} bias={x['le'].mean():+.3f}  RMSE={np.sqrt((x['le'] ** 2).mean()):.3f}  "
                 f"corr={np.corrcoef(np.log(x['vol']), np.log(x['realized_vol']))[0, 1]:.3f}  "
                 f"closer than IV {beats:5.1%}")
    L += ["", "B. Calibration of each forecaster's prob on the spread it picks (all entries)"]
    for name in FORECASTERS:
        x = d[d["forecaster"] == name]
        p, y = x["prob"].tolist(), x["win"].tolist()
        base = np.mean((np.mean(y) - np.asarray(y)) ** 2)
        L.append(f"   {name:<7} n={len(x):<5} mean p={np.mean(p):.3f}  hit={np.mean(y):.3f}  "
                 f"Brier={brier(p, y):.4f} (base {base:.4f})  AUC={auc(p, y):.3f}")
    L += ["", "C. Spreads each forecaster would TRADE (log growth > 0), conservative fills + fees"]
    for name in FORECASTERS:
        x = d[(d["forecaster"] == name) & d["trade"]]
        if x.empty:
            L.append(f"   {name:<7} n=0 trades")
            continue
        L.append(f"   {name:<7} {_trade_line(x)}")
    L += ["", "D. Stability: TRADE subsets by year of entry"]
    for name in FORECASTERS:
        x = d[(d["forecaster"] == name) & d["trade"]]
        for yr, g in x.groupby(x["date"].str[:4]):
            L.append(f"   {name:<7} {yr}  {_trade_line(g)}")
    L += ["", "Caveats: overlapping holds (effective n ~1/4 of n); one 2-year window; hold-to-expiry "
          "decisions, not the live TP/SL engine; the live paper scorecard is the out-of-sample test."]
    text = "\n".join(L)
    VOL_REPORT_OUT.write_text(text + "\n")
    print(text)
    return text


def _trade_line(x: pd.DataFrame) -> str:
    return (f"n={len(x):<5} hit={x['win'].mean():6.1%}  mean ret={x['ret'].mean():+7.1%}  "
            f"$/contract={x['pnl'].mean() * 100:+8.2f}  total=${x['pnl'].sum() * 100:+9.0f}  "
            f"(mid fills ${x['pnl_mid_fill'].mean() * 100:+.2f}/contract)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5, help="sessions between entries per ticker")
    ap.add_argument("--vol-forecast", action="store_true",
                    help="compare RV-history vol forecasts vs market default on the last run's entries")
    a = ap.parse_args()
    run_vol_forecasts() if a.vol_forecast else run(a.step)
