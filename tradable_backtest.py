"""
tradable_backtest.py — can a $5k account capture the SPY put-spread premium?

regime_backtest.py found SPY put credit spreads earning ~+3.9% of max risk per
entry (2020-2026), but at 2-4% widths = $1.5-3k risk per lot vs the $200
per-trade cap. This tests one-lot spreads $1 / $2 / $5 wide with real fees.

Problem: historical/ SPY strikes are ~2% ($6-10) apart, so narrow spreads are
not quoted there. The long leg at K-w is therefore priced by LINEAR
INTERPOLATION in strike between the short strike K and the next quoted strike
below it (bid and ask separately). Put prices are convex in strike, so this
overstates the long put and UNDERSTATES the credit — a conservative bias.
`validate_live()` measures that bias on chain_history/ (our own snapshots,
$1 strikes): actual narrow credit vs the interpolated one.

The actual adjacent-strike spread ("adj", no interpolation) is reported
alongside as the anchor.

Per lot: pnl$ = 100 * (credit - min(max(K - S_T, 0), w)) - 2.60 fees
(4 x $0.65; charged even when the spread expires worthless — conservative).
Fills: 40% toward mid on both legs, same as everywhere else.

Not modelled: assignment. SPY options are American and physically settled —
a short put in the money at expiry puts 100 shares (~$75k) into a $5k
account. In practice the spread must be closed before expiry (extra cost),
or traded on XSP (cash-settled, European; not in this data).

    python tradable_backtest.py
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from backtest import PARITY_MAX_GAP_PCT, ROOT, load_rates, parity_gap_pct, settle_session
from credit_backtest import TENORS, _leg_ok
from import_dolthub import CHAIN_DIR, SPLITS, UNDERLYING, unadjusted_closes
from paper import PORTFOLIO, MAX_PER_TRADE_PCT, read_chain_history
from snapshot import RISK_FREE
from spread_eval import FEE_PER_CONTRACT, fill_toward_mid

TICKER = "SPY"
WIDTHS = (1.0, 2.0, 5.0)
SHORT_OTM = (0.02, 0.04, 0.06)
FEES_LOT = 4 * FEE_PER_CONTRACT          # $ per one-lot spread, open + close
RISK_CAP = PORTFOLIO * MAX_PER_TRADE_PCT  # $200
TRADES_OUT = ROOT / "tradable_trades.csv"
REPORT_OUT = ROOT / "tradable_report.txt"


def interp_quote(short, lower, w: float) -> tuple[float, float]:
    """(bid, ask) of a put at short.strike - w, linear in strike between two quoted puts."""
    f = w / (short["strike"] - lower["strike"])
    return (short["bid"] + f * (lower["bid"] - short["bid"]),
            short["ask"] + f * (lower["ask"] - short["ask"]))


def conservative_credit(s_bid, s_ask, l_bid, l_ask) -> float:
    return fill_toward_mid(s_bid, (s_bid + s_ask) / 2) - fill_toward_mid(l_ask, (l_bid + l_ask) / 2)


def lot_pnl(k_short: float, w: float, credit: float, st: float) -> float:
    """$ P&L of one spread held to expiry."""
    return 100.0 * (credit - min(max(k_short - st, 0.0), w)) - FEES_LOT


def lot_risk(w: float, credit: float) -> float:
    return 100.0 * (w - credit) + FEES_LOT


def _books(day: pd.DataFrame):
    """(tenor, expiry, put book sorted by strike) for each tenor present."""
    puts = day[day["type"] == "P"]
    for tenor, (lo, hi, target) in TENORS.items():
        x = puts[(puts["dte"] >= lo) & (puts["dte"] <= hi)]
        if x.empty:
            continue
        expiry = x.iloc[(x["dte"] - target).abs().argsort().iloc[0]]["expiry"]
        yield tenor, expiry, x[x["expiry"] == expiry].sort_values("strike").reset_index(drop=True)


def spreads_for_day(day: pd.DataFrame, spot: float, coarse_gap: float | None = None) -> list[dict]:
    """Candidate one-lot spreads. coarse_gap=None: the lower strike is the next
    quoted one (historical data). coarse_gap=g: the lower strike is the one
    nearest K*(1-g), to mimic the historical grid on dense live chains."""
    out = []
    for tenor, expiry, book in _books(day):
        for m in SHORT_OTM:
            s = book.iloc[(book["strike"] - spot * (1 - m)).abs().argsort().iloc[0]]
            below = book[book["strike"] < s["strike"]]
            if s["strike"] >= spot or below.empty:
                continue
            lower = (below.iloc[-1] if coarse_gap is None else
                     below.iloc[(below["strike"] - s["strike"] * (1 - coarse_gap)).abs().argsort().iloc[0]])
            if not (_leg_ok(s) and _leg_ok(lower)):
                continue
            gap = float(s["strike"] - lower["strike"])
            base = {"tenor": tenor, "expiry": expiry, "dte": int(s["dte"]), "otm": m,
                    "short_strike": float(s["strike"]), "gap": gap}
            out.append({**base, "width": "adj", "w": gap,
                        "credit": conservative_credit(s["bid"], s["ask"], lower["bid"], lower["ask"])})
            for w in WIDTHS:
                if w >= gap:
                    continue
                lb, la = interp_quote(s, lower, w)
                row = {**base, "width": f"${w:g}", "w": w,
                       "credit": conservative_credit(s["bid"], s["ask"], lb, la)}
                actual = book[np.isclose(book["strike"], s["strike"] - w)]
                if coarse_gap is not None and not actual.empty and _leg_ok(actual.iloc[0]):
                    a = actual.iloc[0]
                    row["credit_actual"] = conservative_credit(s["bid"], s["ask"], a["bid"], a["ask"])
                out.append(row)
    return out


def validate_live() -> pd.DataFrame:
    """On our own $1-strike snapshots: interpolated vs actual narrow credit."""
    live = read_chain_history()
    live = live[live["ticker"] == TICKER]
    rows = []
    for d, day in live.groupby(live["date"].astype(str)):
        for r in spreads_for_day(day, float(day["spot"].iloc[0]), coarse_gap=0.02):
            if "credit_actual" in r:
                rows.append({"date": d, **r})
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    chain = read_chain_history(CHAIN_DIR)
    chain = chain[chain["ticker"] == TICKER]
    chain["date"] = chain["date"].astype(str)
    und = unadjusted_closes(pd.read_csv(UNDERLYING), pd.read_csv(SPLITS))
    close = {str(r.date): r.close_raw for r in und[und["ticker"] == TICKER].itertuples()}
    last_px, rates, rows = max(close), load_rates(), []
    for d, day in chain.groupby("date"):
        spot = float(day["spot"].iloc[0])
        gap = parity_gap_pct(day, rates.get(d, RISK_FREE))
        if np.isfinite(gap) and abs(gap) > PARITY_MAX_GAP_PCT:
            continue
        for r in spreads_for_day(day, spot):
            sd = settle_session(r["expiry"])
            if sd > last_px or sd not in close or r["credit"] * 100 <= FEES_LOT:
                continue
            pnl, risk = lot_pnl(r["short_strike"], r["w"], r["credit"], close[sd]), lot_risk(r["w"], r["credit"])
            rows.append({"date": d, "spot": spot, **r, "settle_px": close[sd],
                         "pnl": pnl, "risk": risk, "ret": pnl / risk, "win": int(pnl > 0)})
    df = pd.DataFrame(rows)
    df.to_csv(TRADES_OUT, index=False)
    return df


def month_t(x: pd.DataFrame, col: str = "ret") -> float:
    m = x.groupby(x["date"].str[:7])[col].mean()
    return float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))) if len(m) > 2 and m.std(ddof=1) > 0 else float("nan")


def weekly_book(x: pd.DataFrame, bias: float = 1.0) -> dict:
    """$5k book: first entry of each ISO week, as many lots as fit under the
    $200 risk cap (skip if one lot doesn't fit). P&L only — cash interest excluded."""
    x = x.sort_values("date").copy()
    x["wk"] = pd.to_datetime(x["date"]).dt.strftime("%G-%V")
    w = x.groupby("wk").first()
    lots = np.floor(RISK_CAP / w["risk"]).astype(int)
    w, lots = w[lots > 0], lots[lots > 0]
    if w.empty:
        return {"n": 0}
    pnl = (w["pnl"] * lots).to_numpy()
    eq = PORTFOLIO + np.cumsum(pnl)
    yrs = (pd.Timestamp(w["date"].iloc[-1]) - pd.Timestamp(w["date"].iloc[0])).days / 365.25
    peak = np.maximum.accumulate(np.concatenate([[PORTFOLIO], eq]))[1:]
    return {"n": len(w), "lots": float(lots.mean()), "per_year": pnl.sum() / PORTFOLIO / yrs,
            "end": float(eq[-1]), "maxdd": float((eq - peak).min()), "worst": float(pnl.min())}


def rescale(df: pd.DataFrame, ratios: dict[str, float]) -> pd.DataFrame:
    """Narrow-width trades re-priced with credit x ratio (the live
    actual/interpolated median) — the central estimate rather than the lower bound."""
    g = df[df["width"].isin(ratios)].copy()
    g["credit"] = np.minimum(g["credit"] * g["width"].map(ratios), g["w"] * 0.95)
    g["pnl"] = [lot_pnl(k, w, c, st) for k, w, c, st in zip(g["short_strike"], g["w"], g["credit"], g["settle_px"])]
    g["risk"] = [lot_risk(w, c) for w, c in zip(g["w"], g["credit"])]
    g["ret"], g["win"] = g["pnl"] / g["risk"], (g["pnl"] > 0).astype(int)
    return g


def report(df: pd.DataFrame, live: pd.DataFrame) -> str:
    L = ["=== TRADABLE-SIZE SPY PUT CREDIT SPREADS (one lot, hold to expiry) ===",
         f"generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  "
         f"{df['date'].min()}..{df['date'].max()}  entry dates={df['date'].nunique()}",
         "narrow widths use an interpolated long leg (understates credit); adj = actual adjacent strikes",
         "", f"1. Interpolation check on live $1-strike snapshots ({live['date'].nunique()} sessions)"]
    for w, g in live.groupby("width"):
        ratio = (g["credit_actual"] / g["credit"]).replace([np.inf, -np.inf], np.nan).dropna()
        L.append(f"   {w:<4} n={len(g):<5} actual credit ${g['credit_actual'].mean():.3f} vs interpolated "
                 f"${g['credit'].mean():.3f}  (actual/interp median {ratio.median():.2f}; "
                 f"actual higher on {(g['credit_actual'] >= g['credit']).mean():.0%})")
    L += ["", "2. All entries by width (every session, all tenors and OTM targets)"]
    for w in ["$1", "$2", "$5", "adj"]:
        g = df[df["width"] == w]
        if g.empty:
            continue
        L.append(f"   {w:<4} n={len(g):<5} hit={g['win'].mean():5.1%}  credit=${g['credit'].mean() * 100:6.2f}  "
                 f"risk/lot=${g['risk'].mean():7.2f}  $/lot={g['pnl'].mean():+7.2f}  "
                 f"ret/risk={g['ret'].mean():+6.1%} t={month_t(g):+5.2f}  fees={FEES_LOT / (g['credit'].mean() * 100):4.0%} of credit")
    L += ["", "3. By tenor and short strike (ret/risk, t, n) — sensitivity, not a menu to pick the best from"]
    for w in ["$1", "$2", "$5", "adj"]:
        parts = []
        for (tn, m), g in df[df["width"] == w].groupby(["tenor", "otm"]):
            parts.append(f"{tn} {m:.0%}: {g['ret'].mean():+5.1%} t={month_t(g):+4.1f} n={len(g)}")
        L.append(f"   {w:<4} " + " | ".join(parts))
    L += ["", "4. By year (ret/risk, n)"]
    for w in ["$1", "$2", "$5", "adj"]:
        g = df[df["width"] == w]
        L.append(f"   {w:<4} " + "  ".join(f"{y}:{x['ret'].mean():+5.1%}/{len(x)}" for y, x in g.groupby(g["date"].str[:4])))
    L += ["", f"5. $5k book, one entry per week, lots = floor(${RISK_CAP:.0f} / risk per lot), near tenor, 4% OTM "
          "(fixed in advance: the live system's 21-day horizon and the middle strike)"]
    for w in ["$1", "$2", "$5", "adj"]:
        b = weekly_book(df[(df["width"] == w) & (df["tenor"] == "near") & (df["otm"] == 0.04)])
        if not b["n"]:
            L.append(f"   {w:<4} one lot exceeds the ${RISK_CAP:.0f} cap — not tradable")
            continue
        L.append(f"   {w:<4} weeks={b['n']:<4} avg lots={b['lots']:.1f}  P&L/yr={b['per_year']:+5.1%} of book  "
                 f"end=${b['end']:,.0f}  maxDD=${b['maxdd']:,.0f}  worst week=${b['worst']:,.0f}")
    ratios = {w: float((g["credit_actual"] / g["credit"]).median()) for w, g in live.groupby("width")}
    adj = rescale(df, ratios)
    L += ["", "6. Central estimate: narrow credits scaled by the live actual/interpolated median "
          f"({', '.join(f'{w} x{r:.2f}' for w, r in ratios.items())}; measured on {live['date'].nunique()} "
          "sessions in one volatility regime)"]
    for w in ["$1", "$2", "$5"]:
        g = adj[adj["width"] == w]
        if g.empty:
            continue
        line = (f"   {w:<4} $/lot={g['pnl'].mean():+7.2f}  ret/risk={g['ret'].mean():+6.1%} t={month_t(g):+5.2f}  "
                f"risk/lot=${g['risk'].mean():6.2f}")
        for tn in ("near", "far"):
            b = weekly_book(g[(g["tenor"] == tn) & (g["otm"] == 0.04)])
            line += (f"  | book {tn} 4%: {b['per_year']:+5.1%}/yr maxDD ${b['maxdd']:,.0f}" if b["n"]
                     else f"  | book {tn} 4%: over cap")
        L.append(line)
    L += ["", "Caveats: narrow credits are estimates (conservative per section 1); assignment / closing before "
          "expiry not modelled; quotes on ~3 sessions a week before 2024-10; one 7-year history."]
    text = "\n".join(L)
    REPORT_OUT.write_text(text + "\n")
    print(text)
    return text


if __name__ == "__main__":
    report(build(), validate_live())
