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

    python import_dolthub.py      # first
    python backtest.py [--step 5]
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
from spread_eval import FEE_PER_CONTRACT, evaluate

HORIZON = 21  # paper.auto.DEFAULT_HORIZON
FEES = 4 * FEE_PER_CONTRACT / 100.0  # 2 legs, open + close, per share (as simulate())
TRADES_OUT = ROOT / "backtest_trades.csv"
REPORT_OUT = ROOT / "backtest_report.txt"
EARNINGS = ROOT / "earnings.csv"
ETFS = frozenset({"XLE", "XLF", "XBI", "SMH", "ARKK", "GDX", "SPY", "QQQ", "IWM"})


# --- model + settlement -------------------------------------------------------

def expiry_model(spot: float, k1: float, k2: float, dte: int, debit: float, vol: float,
                 drift: float = 0.0, r: float = RISK_FREE, n: int = 20_000, seed: int = 7) -> dict:
    """Hold-to-expiry call debit spread under GBM. prob is closed-form
    P(S_T > k1 + debit + fees); ev / log_growth from seeded terminal samples.
    T = dte/365 — the basis snapshot.py solves IV on."""
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
    return {"prob": prob, "ev": float(pnl.mean()), "log_growth": float(np.mean(np.log1p(rets)))}


def settle(k1: float, k2: float, st: float, debit: float) -> tuple[float, float]:
    """(pnl per share, return on debit+fees) holding to expiry."""
    pnl = min(max(st - k1, 0.0), k2 - k1) - debit - FEES
    return pnl, pnl / (debit + FEES)


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

    rows, skipped = [], {"open": 0, "split_in_hold": 0, "no_candidates": 0}
    for t, g in chain.groupby("ticker"):
        dates = sorted(g["date"].unique())[::step]
        tsplits = splits.loc[splits["ticker"] == t, "date"].astype(str).tolist()
        tearn = earn.loc[earn["ticker"] == t, "earnings_date"].astype(str).tolist()
        for d in dates:
            day = g[g["date"] == d]
            spot = float(day["spot"].iloc[0])
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
                    "date": d, "ticker": t, "spot": spot, "expiry": expiry, "dte": c["dte"],
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
    L += ["", "Caveats: overlapping holds (n overstates independent samples); one 2-year window; "
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5, help="sessions between entries per ticker")
    run(ap.parse_args().step)
