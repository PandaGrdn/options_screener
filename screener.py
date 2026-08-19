"""
screener.py — Finding underlyings where BUYING options is least structurally penalized.

Run locally:  pip install yfinance pandas numpy scipy
              python screener.py

HONEST FRAMING FIRST
--------------------
There is no list of tickers with a persistent, stable edge for call buyers. If
one existed it would be arbitraged away. What actually exists is CONDITIONAL
edge: specific underlyings, in specific vol regimes, where the headwind you're
fighting is smaller or briefly negative.

The single biggest lever is NOT which ticker you pick. It's WHEN you buy --
specifically, whether you're buying cheap vol or expensive vol. A mediocre
ticker at IV rank 15 beats a great ticker at IV rank 85, every time.

WHY SINGLE NAMES BEAT SPY FOR BUYERS
------------------------------------
Index variance premium decomposes into (a) average single-stock variance
premium plus (b) a CORRELATION risk premium. Index options embed extra premium
because they insure against everything crashing together -- and correlation
spikes exactly when you need the hedge. That correlation component is large and
it is the reason SPY is so hostile to buyers.

Single stocks carry only component (a), which is much smaller and for some
names is approximately zero or negative. That is the structural reason to leave
SPY if you insist on being a buyer.

WHAT TO SCREEN FOR (in rough order of importance)
-------------------------------------------------
 1. IV RANK / PERCENTILE LOW.  Buying at IV rank <30 vs >70 is most of the
    game. High IV rank = you're paying peak premium and eating the crush.
 2. NEGATIVE HISTORICAL VRP.  Names where realized vol has tended to EXCEED
    implied. Requires IV history (see DATA note). This is the closest thing
    to a real structural edge for a buyer.
 3. VOL CLUSTERING / EXPANSION.  Realized vol is autocorrelated. Buying as
    vol expands from a compressed base is when long gamma pays.
 4. OPTION LIQUIDITY.  Penny or nickel-wide markets only. Illiquid names
    give back the entire edge in slippage -- this filter kills most tickers.
 5. NO EARNINGS INSIDE THE HOLD.  Pre-earnings IV ramp then crush is the
    single most reliable way retail call buyers lose. Either avoid it, or
    trade it deliberately as a different strategy.
 6. TREND ALIGNMENT.  If buying calls, you want a name with positive drift,
    not one you hope reverses.

DATA NOTE
---------
yfinance gives you price history (free) and a CURRENT option chain, but NOT
implied vol history. Without IV history you cannot compute item 2 properly.
Options: (a) subscribe to ORATS / IVolatility / CBOE DataShop, or (b) run the
snapshot function below daily and build your own IV history over a few months.
Do (b) starting today regardless -- it costs nothing and in 3 months you'll
have the dataset that actually answers the question.
"""

from __future__ import annotations
import math, os, json, datetime as dt
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Realized vol estimators
# ---------------------------------------------------------------------------

def close_to_close_vol(close: pd.Series, window: int) -> pd.Series:
    r = np.log(close / close.shift(1))
    return r.rolling(window).std() * math.sqrt(TRADING_DAYS)


def yang_zhang_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """
    More efficient than close-to-close: uses OHLC and handles overnight gaps.
    Matters because option IV prices the FULL 24h move, while close-to-close
    understates it. Comparing IV to close-to-close vol biases you toward
    thinking options are expensive when they aren't.

    CAVEAT: verified on synthetic data only (no market data access when this
    was written). YZ read ~13% low there, which I believe is an artifact of
    the simulated high/low understating true path extremes, not an estimator
    bug. Close-to-close verified unbiased but ~2x noisier. Sanity-check both
    against a name whose vol you know. The PERCENTILE is robust either way,
    and the percentile is what the screen actually keys on.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    log_ho, log_lo, log_co = np.log(h/o), np.log(l/o), np.log(c/o)
    log_oc = np.log(o / c.shift(1))
    rs = log_ho*(log_ho-log_co) + log_lo*(log_lo-log_co)          # Rogers-Satchell
    close_var = log_co.rolling(window).var()
    open_var = log_oc.rolling(window).var()
    rs_var = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window+1)/(window-1))
    return np.sqrt(open_var + k*close_var + (1-k)*rs_var) * math.sqrt(TRADING_DAYS)


def vol_percentile(series: pd.Series, lookback: int = 252) -> float:
    """Where does today's vol sit in its own trailing distribution? 0-100."""
    s = series.dropna().tail(lookback)
    if len(s) < 60:
        return float("nan")
    return float((s < s.iloc[-1]).mean() * 100)


# ---------------------------------------------------------------------------
# Option chain metrics
# ---------------------------------------------------------------------------

def chain_metrics(tk, target_dte=30, max_expiries=6):
    """Pull the live chain, find the expiry nearest target_dte, summarize."""
    import yfinance as yf
    try:
        exps = tk.options[:max_expiries]
    except Exception:
        return None
    if not exps:
        return None
    today = dt.date.today()
    best, best_gap = None, 1e9
    for e in exps:
        d = (dt.date.fromisoformat(e) - today).days
        if 7 <= d <= 75 and abs(d - target_dte) < best_gap:
            best, best_gap = e, abs(d - target_dte)
    if best is None:
        return None
    try:
        ch = tk.option_chain(best)
    except Exception:
        return None
    calls = ch.calls.copy()
    if calls.empty:
        return None

    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    calls["moneyness"] = calls["strike"] / spot - 1.0
    near = calls[calls["moneyness"].abs() < 0.10].copy()
    if near.empty:
        return None

    near["mid"] = (near["bid"] + near["ask"]) / 2.0
    near = near[near["mid"] > 0.05]
    if near.empty:
        return None
    near["spread_pct"] = (near["ask"] - near["bid"]) / near["mid"]

    atm = near.iloc[(near["moneyness"].abs()).argmin()]
    return {
        "expiry": best,
        "dte": (dt.date.fromisoformat(best) - today).days,
        "spot": spot,
        "atm_iv": float(atm.get("impliedVolatility", np.nan)),
        "median_spread_pct": float(near["spread_pct"].median()),
        "total_oi": int(near["openInterest"].fillna(0).sum()),
        "total_vol": int(near["volume"].fillna(0).sum()),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_ticker(ticker: str, target_dte: int = 30, verbose: bool = False):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        px = tk.history(period="2y", auto_adjust=False)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}
    if px is None or len(px) < 300:
        return {"ticker": ticker, "error": "insufficient price history"}

    rv20 = yang_zhang_vol(px, 20)
    rv60 = yang_zhang_vol(px, 60)
    rv_now, rv_slow = float(rv20.iloc[-1]), float(rv60.iloc[-1])
    rv_pctile = vol_percentile(rv20)

    ch = chain_metrics(tk, target_dte=target_dte)
    if ch is None or not np.isfinite(ch.get("atm_iv", np.nan)):
        return {"ticker": ticker, "error": "no usable option chain"}

    iv = ch["atm_iv"]

    # Proxy VRP: current IV vs recent realized. Imperfect (forward vs backward)
    # but it's what's obtainable free. Positive = options rich = buyer headwind.
    vrp_proxy = iv - rv_now
    vrp_pts = vrp_proxy * 100

    # vol expansion: short-window vol rising above long-window = regime turning
    vol_expansion = rv_now / rv_slow - 1.0

    # trend (call buyers want positive drift)
    c = px["Close"]
    trend = float(c.iloc[-1] / c.iloc[-63] - 1.0)

    # earnings inside the window?
    earn_days = None
    try:
        cal = tk.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            ed = cal["Earnings Date"]
            ed = ed[0] if isinstance(ed, list) else ed
            earn_days = (ed - dt.date.today()).days
        elif hasattr(cal, "empty") and not cal.empty:
            earn_days = (cal.iloc[0, 0].date() - dt.date.today()).days
    except Exception:
        pass

    # ---- composite buyer-friendliness score ----
    flags = []
    s = 0.0
    # cheap vol is the dominant term
    if np.isfinite(rv_pctile):
        s += (50.0 - rv_pctile) * 0.6
        if rv_pctile > 70: flags.append(f"vol already elevated (pctile {rv_pctile:.0f})")
    # negative VRP proxy is the real prize
    s += -vrp_pts * 2.0
    if vrp_pts > 6: flags.append(f"options rich: IV {iv:.0%} vs RV {rv_now:.0%}")
    if vrp_pts < 0: flags.append(f"IV BELOW realized vol ({vrp_pts:+.1f} pts) — buyer-favorable")
    # expansion
    s += vol_expansion * 30.0
    # trend
    s += np.clip(trend, -0.3, 0.3) * 40.0
    # liquidity is a hard gate, not a score term
    if ch["median_spread_pct"] > 0.10:
        flags.append(f"ILLIQUID: median spread {ch['median_spread_pct']:.0%} — DISQUALIFY")
        s -= 100
    if ch["total_oi"] < 2000:
        flags.append(f"thin OI ({ch['total_oi']}) — DISQUALIFY")
        s -= 100
    if earn_days is not None and 0 <= earn_days <= target_dte:
        flags.append(f"EARNINGS in {earn_days}d — inside your hold window")
        s -= 40

    return {
        "ticker": ticker, "spot": ch["spot"], "dte": ch["dte"],
        "atm_iv": iv, "rv20": rv_now, "rv60": rv_slow,
        "vrp_pts": vrp_pts, "rv_pctile": rv_pctile,
        "vol_expansion": vol_expansion, "trend_3m": trend,
        "spread_pct": ch["median_spread_pct"], "oi": ch["total_oi"],
        "earn_days": earn_days, "score": s, "flags": flags,
    }


def screen(tickers, target_dte=30):
    rows = []
    for t in tickers:
        r = score_ticker(t, target_dte)
        rows.append(r)
        if "error" in r:
            print(f"  {t:<8} skipped: {r['error']}")
        else:
            print(f"  {t:<8} scored {r['score']:+7.1f}")
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'tkr':<7}{'spot':>9}{'IV':>7}{'RV20':>7}{'VRP':>7}{'RVpct':>7}"
          f"{'expand':>8}{'trend':>8}{'spr':>7}{'score':>8}")
    print("-" * 82)
    for r in ok:
        print(f"{r['ticker']:<7}{r['spot']:9.2f}{r['atm_iv']:7.1%}{r['rv20']:7.1%}"
              f"{r['vrp_pts']:+7.1f}{r['rv_pctile']:7.0f}{r['vol_expansion']:+8.1%}"
              f"{r['trend_3m']:+8.1%}{r['spread_pct']:7.1%}{r['score']:+8.1f}")
    print("\nFLAGS")
    for r in ok:
        if r["flags"]:
            print(f"  {r['ticker']}: " + "; ".join(r["flags"]))
    return ok


# ---------------------------------------------------------------------------
# Build your own IV history — start this today
# ---------------------------------------------------------------------------

def snapshot_iv(tickers, path="iv_history.csv"):
    """
    Append today's ATM IV for each ticker. Run daily (cron / Task Scheduler).
    After ~3 months you can compute each name's TRUE realized VRP:
        VRP(t) = IV30(t) - realized_vol(t, t+30)
    and screen on names whose VRP is consistently near zero or negative.
    That is the only version of this that's a real edge rather than a proxy.
    """
    import yfinance as yf
    today = dt.date.today().isoformat()
    rows = []
    for t in tickers:
        try:
            ch = chain_metrics(yf.Ticker(t))
            if ch and np.isfinite(ch.get("atm_iv", np.nan)):
                rows.append({"date": today, "ticker": t, "spot": ch["spot"],
                             "dte": ch["dte"], "atm_iv": ch["atm_iv"]})
        except Exception:
            continue
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"appended {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------

# Starting universe: liquid enough to have penny/nickel option markets, and
# enough idiosyncratic vol to be worth a buyer's time. This is a STARTING
# POINT, not a recommendation -- run the screen and let it disqualify names.
UNIVERSE = [
    # high-idio-vol large caps with deep option markets
    "NVDA", "TSLA", "AMD", "META", "AVGO", "MU", "COIN", "PLTR",
    "NFLX", "CRM", "UBER", "SHOP", "MRVL", "SMCI",
    # sector ETFs: lower correlation premium than SPY, still liquid
    "XLE", "XLF", "XBI", "SMH", "ARKK", "GDX",
    # reference points
    "QQQ", "IWM", "SPY",
]

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "screen"
    try:
        if cmd == "snapshot":
            path = sys.argv[2] if len(sys.argv) > 2 else "iv_history.csv"
            snapshot_iv(UNIVERSE, path=path)
        else:
            print("Screening for buyer-friendly conditions (30 DTE)...\n")
            results = screen(UNIVERSE, target_dte=30)
            print("\nReminder: a high score means the headwind is SMALLER, not that the")
            print("trade is positive-EV. Feed the winners into spread_eval.py with real")
            print("quotes before risking anything.")
    except ImportError:
        print("Install first:  pip install yfinance pandas numpy")
        raise