"""
credit_backtest.py — hold-to-expiry backtest of bull PUT CREDIT spreads on historical/.

Why: backtest.py found buying call spreads ~breakeven before costs, and ATM IV
running ~2-3 vol points above subsequent realized vol on the sector ETFs. If
there is a premium, the seller collects it. This tests that directly — research
only; the live system stays call-debit-spreads-only (AGENT_CONTEXT §2).

Structure: sell a put ~m below spot, buy a put ~w further below (defined risk).
Grid: short moneyness m in SHORT_OTM, width w in WIDTHS, two tenors (near:
nearest 21 DTE in 14-30; far: nearest 42 DTE in 31-50). Every `step` sessions
per ticker. Same conventions as backtest.py:
  - fills: sell at bid + 40% toward mid, buy at ask - 40% toward mid; fees
    $0.65 x 2 legs x open+close (FEES); mid-fill P&L kept for comparison
  - settle at expiry from the underlying close: loss = clip(K_short - S_T, 0, width)
  - skip entries whose quotes weren't at the close (parity gap > 0.5%), and
    legs quoted wider than MAX_LEG_SPREAD of mid (bad prints — XLE 0.01/1.33)
  - model_prob: P(S_T > breakeven) under GBM at the SHORT put's own IV (skew
    included), zero drift — the market-implied odds of keeping the credit

Ignores early assignment (American puts deep ITM) and pin risk — both make
real results slightly worse than this.

    python credit_backtest.py [--step 5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest import ETFS, FEES, PARITY_MAX_GAP_PCT, ROOT, load_rates, parity_gap_pct, settle_session
from import_dolthub import CHAIN_DIR, SPLITS, UNDERLYING, unadjusted_closes
from paper import read_chain_history
from paper.score import auc, brier
from snapshot import RISK_FREE
from spread_eval import fill_toward_mid

SHORT_OTM = (0.02, 0.04, 0.06)
WIDTHS = (0.02, 0.04)
TENORS = {"near": (14, 30, 21), "far": (31, 50, 42)}
MAX_LEG_SPREAD = 0.50
TRADES_OUT = ROOT / "credit_trades.csv"
REPORT_OUT = ROOT / "credit_report.txt"


def credit_entry_fill(short_bid, short_ask, long_bid, long_ask) -> tuple[float, float]:
    """(conservative credit, mid credit) per share for selling the spread."""
    s_mid, l_mid = (short_bid + short_ask) / 2, (long_bid + long_ask) / 2
    return fill_toward_mid(short_bid, s_mid) - fill_toward_mid(long_ask, l_mid), s_mid - l_mid


def settle_put_credit(k_short: float, k_long: float, st: float, credit: float) -> tuple[float, float]:
    """(pnl per share, return on max risk) holding to expiry."""
    width = k_short - k_long
    pnl = credit - min(max(k_short - st, 0.0), width) - FEES
    risk = max(width - credit + FEES, 1e-6)
    return pnl, pnl / risk


def model_prob(spot: float, k_short: float, credit: float, dte: int, iv: float,
               r: float = RISK_FREE) -> float:
    """P(S_T > k_short - (credit - fees)) under GBM at the short leg's IV."""
    be = k_short - (credit - FEES)
    if be <= 0 or not np.isfinite(iv) or iv <= 0:
        return float("nan")
    T = max(int(dte), 1) / 365.0
    return float(norm.cdf((math.log(spot / be) + (r - 0.5 * iv * iv) * T) / (iv * math.sqrt(T))))


def _leg_ok(row) -> bool:
    mid = (row["bid"] + row["ask"]) / 2
    return row["bid"] > 0 and mid > 0 and (row["ask"] - row["bid"]) / mid <= MAX_LEG_SPREAD


def candidates(day: pd.DataFrame, spot: float) -> tuple[list[dict], int]:
    """Every grid spread available on this ticker-date, and how many grid
    points were dropped for bad leg quotes."""
    puts = day[day["type"] == "P"]
    out, bad = [], 0
    for tenor, (lo, hi, target) in TENORS.items():
        w = puts[(puts["dte"] >= lo) & (puts["dte"] <= hi)]
        if w.empty:
            continue
        expiry = w.iloc[(w["dte"] - target).abs().argsort().iloc[0]]["expiry"]
        book = w[w["expiry"] == expiry].sort_values("strike").reset_index(drop=True)
        seen = set()
        for m in SHORT_OTM:
            s = book.iloc[(book["strike"] - spot * (1 - m)).abs().argsort().iloc[0]]
            if s["strike"] >= spot:
                continue
            below = book[book["strike"] < s["strike"]]
            for wd in WIDTHS:
                if below.empty:
                    continue
                l = below.iloc[(below["strike"] - s["strike"] * (1 - wd)).abs().argsort().iloc[0]]
                key = (s["strike"], l["strike"])
                if key in seen:
                    continue  # coarse strikes: two grid points map to one spread
                seen.add(key)
                if not (_leg_ok(s) and _leg_ok(l)):
                    bad += 1
                    continue
                credit, credit_mid = credit_entry_fill(s["bid"], s["ask"], l["bid"], l["ask"])
                if credit <= FEES:
                    continue
                out.append({
                    "tenor": tenor, "expiry": expiry, "dte": int(s["dte"]),
                    "short_otm_target": m, "width_target": wd,
                    "short_strike": float(s["strike"]), "long_strike": float(l["strike"]),
                    "short_otm": 1 - float(s["strike"]) / spot,
                    "width": float(s["strike"] - l["strike"]),
                    "credit": credit, "credit_mid": credit_mid, "short_iv": s["iv"],
                })
    return out, bad


def run(step: int = 5) -> pd.DataFrame:
    chain = read_chain_history(CHAIN_DIR)
    chain["date"] = chain["date"].astype(str)
    und = unadjusted_closes(pd.read_csv(UNDERLYING), pd.read_csv(SPLITS))
    close = {(r.ticker, str(r.date)): r.close_raw for r in und.itertuples()}
    last_px = str(und["date"].max())
    splits = pd.read_csv(SPLITS)
    rates = load_rates()
    earn_path = ROOT / "earnings.csv"
    earn = pd.read_csv(earn_path) if earn_path.exists() else pd.DataFrame(columns=["ticker", "earnings_date"])

    rows = []
    skipped = {"quotes_off_close": 0, "bad_leg_quotes": 0, "open": 0, "split_in_hold": 0}
    for t, g in chain.groupby("ticker"):
        tsplits = splits.loc[splits["ticker"] == t, "date"].astype(str).tolist()
        tearn = earn.loc[earn["ticker"] == t, "earnings_date"].astype(str).tolist()
        for d in sorted(g["date"].unique())[::step]:
            day = g[g["date"] == d]
            spot = float(day["spot"].iloc[0])
            gap = parity_gap_pct(day, rates.get(d, RISK_FREE))
            if np.isfinite(gap) and abs(gap) > PARITY_MAX_GAP_PCT:
                skipped["quotes_off_close"] += 1
                continue
            cands, bad = candidates(day, spot)
            skipped["bad_leg_quotes"] += bad
            for c in cands:
                sd = settle_session(c["expiry"])
                if sd > last_px:
                    skipped["open"] += 1
                    continue
                if any(d < s <= sd for s in tsplits):
                    skipped["split_in_hold"] += 1
                    continue
                st = float(close[(t, sd)])
                pnl, ret = settle_put_credit(c["short_strike"], c["long_strike"], st, c["credit"])
                pnl_mid, _ = settle_put_credit(c["short_strike"], c["long_strike"], st, c["credit_mid"])
                rows.append({
                    "date": d, "ticker": t, "etf": t in ETFS, "spot": spot, **c,
                    "model_prob": model_prob(spot, c["short_strike"], c["credit"], c["dte"], c["short_iv"]),
                    "earnings_in_hold": (any(d < e <= sd for e in tearn) if tearn
                                         else (False if t in ETFS else None)),
                    "settle_date": sd, "settle_px": st, "move": st / spot - 1,
                    "pnl": pnl, "ret": ret, "win": int(pnl > 0), "pnl_mid_fill": pnl_mid,
                    "max_loss": -(c["width"] - c["credit"] + FEES),
                })
    df = pd.DataFrame(rows)
    df.to_csv(TRADES_OUT, index=False)
    print(f"skipped: {skipped}")
    report(df, step, skipped)
    return df


# --- report -------------------------------------------------------------------

def month_t(x: pd.DataFrame, col: str = "ret") -> float:
    """t-stat of the mean, clustered by entry month (overlapping holds and
    same-week entries across tickers are not independent)."""
    m = x.groupby(x["date"].str[:7])[col].mean()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float("nan")
    return float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m))))


def _line(x: pd.DataFrame) -> str:
    if x.empty:
        return "n=0"
    p, y = x["model_prob"].dropna().tolist(), x.loc[x["model_prob"].notna(), "win"].tolist()
    return (f"n={len(x):<5} hit={x['win'].mean():6.1%} (model {np.mean(p) if p else float('nan'):.3f})  "
            f"ret/risk={x['ret'].mean():+6.1%}  t={month_t(x):+5.2f}  "
            f"$/contract={x['pnl'].mean() * 100:+8.2f}  mid=${x['pnl_mid_fill'].mean() * 100:+7.2f}  "
            f"worst={x['ret'].min():+.0%}")


def report(df: pd.DataFrame, step: int, skipped: dict) -> str:
    etf = df[df["etf"]]
    single = df[~df["etf"]]
    L = ["=== PUT CREDIT SPREAD BACKTEST (hold to expiry, historical/) ===",
         f"generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  step={step}  "
         f"trades n={len(df)}  {df['date'].min()}..{df['date'].max()}",
         f"skipped: {skipped}",
         "ret/risk = pnl / (width - credit + fees); t = month-clustered t-stat of ret/risk "
         "(|t| >= 2 is the bar)",
         "", "1. ETFs vs single names (all grid points)",
         f"   ETFs ({', '.join(sorted(etf['ticker'].unique()))})  {_line(etf)}",
         f"   single names             {_line(single)}",
         f"   singles, no earnings     {_line(single[single['earnings_in_hold'] == False])}",  # noqa: E712
         "", "2. ETFs by structure (tenor / short OTM target / width target)"]
    for (tn, m, w), g in etf.groupby(["tenor", "short_otm_target", "width_target"]):
        L.append(f"   {tn:<4} short {m:.0%} OTM width {w:.0%}   {_line(g)}")
    L += ["", "3. ETFs by ticker"]
    for t, g in etf.groupby("ticker"):
        L.append(f"   {t:<5} {_line(g)}")
    L += ["", "4. ETFs by year of entry"]
    for yr, g in etf.groupby(etf["date"].str[:4]):
        L.append(f"   {yr}  {_line(g)}")
    L += ["", "5. ETF tail: worst entry months (mean ret/risk across all ETF trades entered that month)"]
    mm = etf.groupby(etf["date"].str[:7])["ret"].agg(["mean", "size"]).sort_values("mean")
    for mo, r in mm.head(5).iterrows():
        L.append(f"   {mo}  mean ret/risk={r['mean']:+.1%}  n={int(r['size'])}")
    L.append(f"   share of months with positive mean: {(mm['mean'] > 0).mean():.0%} of {len(mm)}")
    p, y = etf["model_prob"].dropna().tolist(), etf.loc[etf["model_prob"].notna(), "win"].tolist()
    L += ["", "6. Calibration (ETFs): market-implied P(keep credit) vs realized",
          f"   Brier={brier(p, y)}  AUC={auc(p, y)}  mean p={np.mean(p):.3f}  hit={np.mean(y):.3f}"]
    L += ["", "Caveats: one 2-year window (includes Apr-2025 crash); overlapping holds; 4 ETFs "
          "(DoltHub lacks QQQ/IWM/SMH/GDX/ARKK); coarse strikes; no early assignment / pin risk; "
          "1 lot of a $5k book can exceed the 4% risk cap on SPY widths."]
    text = "\n".join(L)
    REPORT_OUT.write_text(text + "\n")
    print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5)
    run(ap.parse_args().step)
