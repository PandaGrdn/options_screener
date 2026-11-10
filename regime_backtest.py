"""
regime_backtest.py — regime-conditional strategies on historical/, hold to expiry.

Research basis (see AGENT_CONTEXT §5c for links):
  - Variance risk premium: implied > realized most of the time; index put
    writing (CBOE PUT) has earned equity-like returns at lower vol but with
    strong negative skew (Bondarenko 2016; Israelov & Nielsen 2015).
  - Term structure: the VIX curve slope predicts variance-premium returns
    (Johnson 2017); backwardation marks stress.
  - Trend: practitioner tests of "sell puts only above the 200-day" are mixed.
  - Regimes: 2-state Markov-switching vol models (calm / turbulent) are the
    standard statistical regime detector.
  - Option momentum exists gross but mostly vanishes after bid-ask (Heston et al).

Every rule below was fixed BEFORE looking at results, has no fitted threshold
(0, 1, 200 days, 0.5 are the textbook defaults), and uses only information
known at the entry close. Eight strategies are compared, so one of them
clearing |t| >= 2 by luck is likely (~1 in 3) — read the table with that in mind.

Trades (all grid points, every `step` sessions per ticker):
  put_credit   credit_backtest.candidates(): short 2/4/6% OTM, width 2/4%, near+far tenor
  call_debit   long 0/2% OTM, short ~4% above, near+far tenor
Each (date, ticker, kind) is collapsed to one entry = mean over its grid, so a
strategy's result doesn't depend on picking a structure.

Strategies (action per date/ticker: put_credit | call_debit | cash):
  always_put     always put_credit
  always_call    always call_debit
  trend          put_credit if close > 200d SMA
  term           put_credit if VIX / VIX3M < 1 (contango)
  vrp            put_credit if ATM IV - RV20 > 0
  hmm            put_credit if walk-forward P(calm) > 0.5 (2-state Gaussian HMM on SPY returns)
  switch_rules   stress (VIX/VIX3M >= 1 or SPY < its 200d) -> cash;
                 else uptrend & IV <= RV20 -> call_debit; uptrend -> put_credit; else cash
  switch_hmm     same, with stress = P(calm) <= 0.5

    python regime_backtest.py [--step 5] [--rebuild]
"""

from __future__ import annotations

import argparse
import datetime as dt
import math

import numpy as np
import pandas as pd

from backtest import ETFS, FEES, PARITY_MAX_GAP_PCT, ROOT, load_rates, parity_gap_pct, settle, settle_session
from credit_backtest import TENORS, _leg_ok, candidates, settle_put_credit
from import_dolthub import CHAIN_DIR, SPLITS, UNDERLYING, unadjusted_closes
from paper import read_chain_history
from snapshot import RISK_FREE
from spread_eval import debit_entry_fill

LONG_OTM = (0.0, 0.02)
CALL_WIDTH = 0.04
VIX_PATH = ROOT / "vix.csv"
TRADES_OUT = ROOT / "regime_trades.csv"
REPORT_OUT = ROOT / "regime_report.txt"
STRATEGIES = ("always_put", "always_call", "trend", "term", "vrp", "hmm", "switch_rules", "switch_hmm")
HMM_MIN_OBS = 250


# --- 2-state Gaussian HMM (numpy only) ------------------------------------------

def _emis(r: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    z = (r[:, None] - mu[None, :]) / sig[None, :]
    return np.exp(-0.5 * z * z) / (sig[None, :] * math.sqrt(2 * math.pi)) + 1e-300


def hmm_filter(r: np.ndarray, params: dict) -> np.ndarray:
    """Filtered P(state | r_1..r_t) for each t — forward pass only, so the
    value at t never depends on later returns. Column 0 = calm (lower vol)."""
    B = _emis(r, params["mu"], params["sig"])
    A, a = params["A"], params["pi"]
    out = np.empty_like(B)
    for t in range(len(r)):
        a = (a if t == 0 else a @ A) * B[t]
        a = a / a.sum()
        out[t] = a
    return out


def hmm_fit(r: np.ndarray, iters: int = 30, init: dict | None = None) -> dict:
    """Baum-Welch for a 2-state Gaussian HMM. States are ordered calm, turbulent."""
    r = np.asarray(r, dtype=float)
    if init is None:
        hi = np.abs(r - r.mean()) > np.median(np.abs(r - r.mean()))
        p = {"mu": np.array([r[~hi].mean(), r[hi].mean()]),
             "sig": np.array([r[~hi].std() + 1e-6, r[hi].std() + 1e-6]),
             "A": np.array([[0.98, 0.02], [0.05, 0.95]]), "pi": np.array([0.7, 0.3])}
    else:
        p = {k: v.copy() for k, v in init.items()}
    T = len(r)
    for _ in range(iters):
        B = _emis(r, p["mu"], p["sig"])
        alpha, c = np.empty((T, 2)), np.empty(T)
        a = p["pi"] * B[0]
        for t in range(T):
            if t:
                a = (alpha[t - 1] @ p["A"]) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            beta[t] = (p["A"] @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * p["A"][None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        p["A"] = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        w = gamma.sum(axis=0)
        p["mu"] = (gamma * r[:, None]).sum(axis=0) / w
        p["sig"] = np.sqrt((gamma * (r[:, None] - p["mu"]) ** 2).sum(axis=0) / w) + 1e-8
        p["pi"] = gamma[0]
    if p["sig"][0] > p["sig"][1]:  # keep state 0 = calm
        p = {"mu": p["mu"][::-1].copy(), "sig": p["sig"][::-1].copy(),
             "A": p["A"][::-1, ::-1].copy(), "pi": p["pi"][::-1].copy()}
    return p


def walk_forward_calm(dates: list[str], r: np.ndarray, min_obs: int = HMM_MIN_OBS) -> pd.Series:
    """P(calm) per date. Parameters are refit at each new quarter on returns
    strictly before it; the probability for date d filters returns through d
    (forward pass only), so nothing after d is used."""
    quarter = [(d[:4], (int(d[5:7]) - 1) // 3) for d in dates]
    out, params = {}, None
    i = min_obs
    while i < len(dates):
        j = i
        while j < len(dates) and quarter[j] == quarter[i]:
            j += 1
        params = hmm_fit(r[:i], iters=30 if params is None else 10, init=params)
        probs = hmm_filter(r[:j], params)[:, 0]
        for k in range(i, j):
            out[dates[k]] = float(probs[k])
        i = j
    return pd.Series(out, dtype=float)


# --- regime features --------------------------------------------------------------

def fetch_vix() -> pd.DataFrame:
    if VIX_PATH.exists():
        return pd.read_csv(VIX_PATH, dtype={"date": str})
    import yfinance as yf
    cols = {}
    for sym, name in (("^VIX", "vix"), ("^VIX3M", "vix3m")):
        h = yf.Ticker(sym).history(period="10y", auto_adjust=False)
        cols[name] = pd.Series(h["Close"].to_numpy(), index=[pd.Timestamp(i).date().isoformat() for i in h.index])
    df = pd.DataFrame(cols).dropna().rename_axis("date").reset_index()
    df.to_csv(VIX_PATH, index=False)
    return df


def features(und: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Per (date, ticker), all known at that close: trend_up, rv20, spy_trend_up,
    ts_ratio (VIX/VIX3M), p_calm."""
    und = und.sort_values(["ticker", "date"]).copy()
    und["date"] = und["date"].astype(str)
    und["sma200"] = und.groupby("ticker")["close"].transform(lambda s: s.rolling(200).mean())
    und["trend_up"] = np.where(und["sma200"].notna(), und["close"] > und["sma200"], np.nan)
    f = und[["date", "ticker", "trend_up", "rv20_yz"]].rename(columns={"rv20_yz": "rv20"})
    spy = und[und["ticker"] == "SPY"]
    f = f.merge(spy[["date", "trend_up"]].rename(columns={"trend_up": "spy_trend_up"}), on="date", how="left")
    v = vix.copy()
    v["ts_ratio"] = v["vix"] / v["vix3m"]
    f = f.merge(v[["date", "ts_ratio"]], on="date", how="left")
    r = np.log(spy["close"] / spy["close"].shift()).to_numpy()[1:]
    calm = walk_forward_calm(spy["date"].tolist()[1:], r)
    return f.merge(calm.rename("p_calm").rename_axis("date").reset_index(), on="date", how="left")


def decide(strategy: str, x) -> str:
    """Action for one (date, ticker) feature row. Missing inputs -> cash."""
    ok = lambda *v: all(pd.notna(i) for i in v)  # noqa: E731
    if strategy == "always_put":
        return "put_credit"
    if strategy == "always_call":
        return "call_debit"
    if strategy == "trend":
        return "put_credit" if ok(x.trend_up) and x.trend_up else "cash"
    if strategy == "term":
        return "put_credit" if ok(x.ts_ratio) and x.ts_ratio < 1 else "cash"
    if strategy == "vrp":
        return "put_credit" if ok(x.atm_iv, x.rv20) and x.atm_iv - x.rv20 > 0 else "cash"
    if strategy == "hmm":
        return "put_credit" if ok(x.p_calm) and x.p_calm > 0.5 else "cash"
    if strategy in ("switch_rules", "switch_hmm"):
        if strategy == "switch_rules":
            if not ok(x.ts_ratio, x.spy_trend_up) or x.ts_ratio >= 1 or not x.spy_trend_up:
                return "cash"
        elif not ok(x.p_calm) or x.p_calm <= 0.5:
            return "cash"
        if not ok(x.trend_up) or not x.trend_up:
            return "cash"
        if ok(x.atm_iv, x.rv20) and x.atm_iv <= x.rv20:
            return "call_debit"
        return "put_credit"
    raise ValueError(strategy)


# --- trades -----------------------------------------------------------------------

def call_candidates(day: pd.DataFrame, spot: float) -> list[dict]:
    calls, out = day[day["type"] == "C"], []
    for tenor, (lo, hi, target) in TENORS.items():
        w = calls[(calls["dte"] >= lo) & (calls["dte"] <= hi)]
        if w.empty:
            continue
        expiry = w.iloc[(w["dte"] - target).abs().argsort().iloc[0]]["expiry"]
        book = w[w["expiry"] == expiry].sort_values("strike").reset_index(drop=True)
        seen = set()
        for m in LONG_OTM:
            l = book.iloc[(book["strike"] - spot * (1 + m)).abs().argsort().iloc[0]]
            above = book[book["strike"] > l["strike"]]
            if above.empty:
                continue
            s = above.iloc[(above["strike"] - l["strike"] * (1 + CALL_WIDTH)).abs().argsort().iloc[0]]
            key = (l["strike"], s["strike"])
            if key in seen or not (_leg_ok(l) and _leg_ok(s)):
                continue
            seen.add(key)
            debit, mid = debit_entry_fill(l["ask"], l["bid"], s["bid"], s["ask"])
            width = float(s["strike"] - l["strike"])
            if debit <= 0 or debit + FEES >= width:
                continue
            out.append({"tenor": tenor, "expiry": expiry, "dte": int(l["dte"]),
                        "long_strike": float(l["strike"]), "short_strike": float(s["strike"]),
                        "debit": debit, "debit_mid": mid})
    return out


def atm_iv(day: pd.DataFrame, spot: float) -> float:
    c = day[(day["type"] == "C") & day["iv"].notna()]
    if c.empty:
        return float("nan")
    exp = c.iloc[(c["dte"] - 30).abs().argsort().iloc[0]]["expiry"]
    b = c[c["expiry"] == exp]
    return float(b.iloc[(b["strike"] - spot).abs().argsort().iloc[0]]["iv"])


def build_trades(step: int = 5) -> pd.DataFrame:
    chain = read_chain_history(CHAIN_DIR)
    chain["date"] = chain["date"].astype(str)
    und = unadjusted_closes(pd.read_csv(UNDERLYING), pd.read_csv(SPLITS))
    close = {(r.ticker, str(r.date)): r.close_raw for r in und.itertuples()}
    last_px = str(und["date"].max())
    splits = pd.read_csv(SPLITS)
    rates = load_rates()
    rows = []
    for t, g in chain.groupby("ticker"):
        tsplits = splits.loc[splits["ticker"] == t, "date"].astype(str).tolist()
        for d in sorted(g["date"].unique())[::step]:
            day = g[g["date"] == d]
            spot = float(day["spot"].iloc[0])
            gap = parity_gap_pct(day, rates.get(d, RISK_FREE))
            if np.isfinite(gap) and abs(gap) > PARITY_MAX_GAP_PCT:
                continue
            iv = atm_iv(day, spot)
            base = {"date": d, "ticker": t, "spot": spot, "atm_iv": iv}

            def settled(expiry):
                sd = settle_session(expiry)
                if sd > last_px or any(d < s <= sd for s in tsplits) or (t, sd) not in close:
                    return None
                return float(close[(t, sd)])

            for c in candidates(day, spot)[0]:
                st = settled(c["expiry"])
                if st is None:
                    continue
                pnl, ret = settle_put_credit(c["short_strike"], c["long_strike"], st, c["credit"])
                pm, _ = settle_put_credit(c["short_strike"], c["long_strike"], st, c["credit_mid"])
                rows.append({**base, "kind": "put_credit", "tenor": c["tenor"], "expiry": c["expiry"],
                             "pnl": pnl, "ret": ret, "pnl_mid": pm,
                             "risk": c["width"] - c["credit"] + FEES})
            for c in call_candidates(day, spot):
                st = settled(c["expiry"])
                if st is None:
                    continue
                pnl, ret = settle(c["long_strike"], c["short_strike"], st, c["debit"])
                pm, _ = settle(c["long_strike"], c["short_strike"], st, c["debit_mid"])
                rows.append({**base, "kind": "call_debit", "tenor": c["tenor"], "expiry": c["expiry"],
                             "pnl": pnl, "ret": ret, "pnl_mid": pm, "risk": c["debit"] + FEES})
    df = pd.DataFrame(rows)
    df.to_csv(TRADES_OUT, index=False)
    return df


# --- evaluation ---------------------------------------------------------------------

def entries(trades: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, ticker, kind): grid-mean outcome + regime features."""
    e = trades.groupby(["date", "ticker", "kind"]).agg(
        ret=("ret", "mean"), win=("pnl", lambda s: float((s > 0).mean())),
        pnl=("pnl", "sum"), pnl_mid=("pnl_mid", "sum"), risk=("risk", "sum"),
        atm_iv=("atm_iv", "first")).reset_index()
    return e.merge(feats, on=["date", "ticker"], how="left")


def run_strategy(e: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Entries the strategy takes (its action matches the entry's kind)."""
    act = np.array([decide(strategy, x) for x in e.itertuples()])
    return e[act == e["kind"].to_numpy()]


def stats(x: pd.DataFrame, n_opps: int) -> dict:
    if x.empty:
        return {"n": 0}
    m = x.groupby(x["date"].str[:7])["ret"].mean()
    t = m.mean() / (m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 2 and m.std(ddof=1) > 0 else float("nan")
    cum = x.groupby(x["date"].str[:7])["ret"].sum().cumsum()
    return {"n": len(x), "traded": len(x) / max(n_opps, 1), "hit": x["win"].mean(),
            "ret": x["ret"].mean(), "t": t, "ret_w": x["pnl"].sum() / x["risk"].sum(),
            "ret_w_mid": x["pnl_mid"].sum() / x["risk"].sum(),
            "total": x["ret"].sum(), "maxdd": float((cum - cum.cummax().clip(lower=0)).min()),
            "worst_month": m.min(), "calls": float((x["kind"] == "call_debit").mean())}


def _fmt(name: str, s: dict) -> str:
    if not s["n"]:
        return f"   {name:<13} n=0"
    return (f"   {name:<13} n={s['n']:<5} traded={s['traded']:4.0%}  hit={s['hit']:5.1%}  "
            f"ret/risk={s['ret']:+6.1%} t={s['t']:+5.2f}  $-wtd={s['ret_w']:+6.1%} (mid {s['ret_w_mid']:+6.1%})  "
            f"total={s['total']:+7.1f}R  maxDD={s['maxdd']:+6.1f}R  worst mo={s['worst_month']:+5.0%}  "
            f"calls={s['calls']:3.0%}")


def report(e: pd.DataFrame) -> str:
    L = ["=== REGIME STRATEGY BACKTEST (hold to expiry, conservative fills + fees) ===",
         f"generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  "
         f"entries n={len(e)}  {e['date'].min()}..{e['date'].max()}",
         "ret/risk = mean return on max risk per entry; t = month-clustered; $-wtd = total pnl / total risk; "
         "total/maxDD in R (1R = one entry's max risk); 8 strategies tested -> expect ~1 false |t|>=2 in 3 tries",
         ""]
    etf = e[e["ticker"].isin(ETFS)]
    panels = [("A. ETFs, full history", etf),
              ("B. ETFs, 2020-02..2024-10 (older data: COVID, 2022 bear, SVB)", etf[etf["date"] < "2024-10-23"]),
              ("C. ETFs, 2024-10.. (the window studied before)", etf[etf["date"] >= "2024-10-23"]),
              ("D. Single names, 2024-10..", e[~e["ticker"].isin(ETFS)]),
              ("E. SPY only, full history", e[e["ticker"] == "SPY"])]
    for title, x in panels:
        if x.empty:
            continue
        n_opps = x.groupby(["date", "ticker"]).ngroups
        L.append(f"{title}  (opportunities n={n_opps})")
        for s in STRATEGIES:
            L.append(_fmt(s, stats(run_strategy(x, s), n_opps)))
        L.append("")
    L.append("F. ETFs by year: always_put vs switch_rules vs switch_hmm (ret/risk, n)")
    for yr, x in etf.groupby(etf["date"].str[:4]):
        parts = []
        for s in ("always_put", "trend", "term", "hmm", "switch_rules", "switch_hmm"):
            st = stats(run_strategy(x, s), 1)
            parts.append(f"{s}={st['ret']:+.1%}/{st['n']}" if st["n"] else f"{s}=-/0")
        L.append(f"   {yr}  " + "  ".join(parts))
    cal = e.drop_duplicates("date")
    L += ["", f"Regime mix over entry dates: P(calm)>0.5 on {(cal['p_calm'] > 0.5).mean():.0%}, "
          f"VIX/VIX3M<1 on {(cal['ts_ratio'] < 1).mean():.0%}, SPY>200d on {(cal['spy_trend_up'] == 1).mean():.0%}",
          "Caveats: overlapping holds; 4 ETFs before 2024-10; hold-to-expiry only; no early assignment; "
          "rules are untuned but the comparison set is 8 strategies on one history."]
    text = "\n".join(L)
    REPORT_OUT.write_text(text + "\n")
    print(text)
    return text


def run(step: int = 5, rebuild: bool = False) -> pd.DataFrame:
    trades = build_trades(step) if rebuild or not TRADES_OUT.exists() else pd.read_csv(TRADES_OUT, dtype={"date": str})
    feats = features(pd.read_csv(UNDERLYING), fetch_vix())
    e = entries(trades, feats)
    report(e)
    return e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true", help="regenerate trades from historical/chain")
    a = ap.parse_args()
    run(a.step, a.rebuild)
