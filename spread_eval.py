"""
spread_eval.py — Black-Scholes IV, debit-spread simulation, sizing, evaluate().

Used by the paper CLI after a forecast is already logged. Never call this from
`paper forecast` — that command must stay model-blind.
"""

from __future__ import annotations

import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

TRADING_DAYS = 252
FEE_PER_CONTRACT = 0.65
PORTFOLIO = 5000.0
MAX_PER_TRADE_PCT = 0.04
MAX_DEPLOYED_PCT = 0.20

# Decision knobs (the Monte Carlo “model” you actually tweak)
MIN_DTE = 14                 # 2-week floor
MAX_DTE = 30                 # 30-day ceiling
LOG_GROWTH_MIN = 0.0         # Kelly: require E[log(1+r)] above this
IV_RANK_MAX = 30.0           # screen: skip if IV rank is richer than this (legacy; see signals.py)

# simulate() versions: old trades.csv/forecasts.csv rows were scored under
# the biased terminal-only MC (see AGENT_CONTEXT Patch 1). Never rewritten —
# tagged so score.py can headline only the corrected model.
MODEL_VERSION = "mc_path_v2"
MODEL_VERSION_LEGACY = "mc_terminal_v1"


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_price(S, K, T, r, sigma, kind="C"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "C" else max(K - S, 0.0)
        return float(intrinsic)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "C":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def implied_vol(mid, S, K, T, r=0.05, kind="C", lo=1e-4, hi=5.0):
    """Solve BS IV from a mid quote. Returns nan on failure."""
    if mid is None or not np.isfinite(mid) or mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    intrinsic = max(S - K, 0.0) if kind == "C" else max(K - S, 0.0)
    if mid < intrinsic * 0.99:
        return float("nan")

    def obj(sig):
        return bs_price(S, K, T, r, sig, kind) - mid

    try:
        return float(brentq(obj, lo, hi, maxiter=100))
    except (ValueError, RuntimeError):
        return float("nan")


def bs_greeks(S, K, T, r, sigma, kind="C"):
    """
    Black-Scholes greeks. Vega is per 1 vol point; theta is per calendar day.
    Returns dict with delta, gamma, vega, theta (nans if undefined).
    """
    nan = float("nan")
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or not np.isfinite(sigma):
        return {"delta": nan, "gamma": nan, "vega": nan, "theta": nan}
    sqrt_t = math.sqrt(T)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt_t
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * sqrt_t)
    vega = S * pdf * sqrt_t / 100.0
    disc = math.exp(-r * T)
    if kind == "C":
        delta = float(norm.cdf(d1))
        theta = (-S * pdf * sigma / (2.0 * sqrt_t) - r * K * disc * norm.cdf(d2)) / 365.0
    else:
        delta = float(norm.cdf(d1) - 1.0)
        theta = (-S * pdf * sigma / (2.0 * sqrt_t) + r * K * disc * norm.cdf(-d2)) / 365.0
    return {
        "delta": delta,
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


# ---------------------------------------------------------------------------
# Fill model (shared with paper)
# ---------------------------------------------------------------------------

def fill_toward_mid(natural, mid, frac: float = 0.4):
    """natural ± 40% of the way to mid. Same formula for debit entry / credit exit.

    Works elementwise: scalar float in -> float out, ndarray in -> ndarray out
    (the path simulator calls this vectorized across all paths for a given day).
    """
    return natural - frac * (natural - mid)


def debit_entry_fill(long_ask, long_bid, short_bid=None, short_ask=None) -> tuple[float, float]:
    """
    Returns (conservative_debit, mid_debit) per share.
    Long call: pay ask side, 40% toward mid.
    Credit on short leg (spread): receive bid side, 40% toward mid.
    """
    long_mid = (long_bid + long_ask) / 2.0
    long_fill = fill_toward_mid(long_ask, long_mid)
    if short_bid is None:
        return long_fill, long_mid
    short_mid = (short_bid + short_ask) / 2.0
    short_fill = fill_toward_mid(short_bid, short_mid)  # credit received
    return long_fill - short_fill, long_mid - short_mid


def credit_exit_fill(long_bid, long_ask, short_ask=None, short_bid=None,
                     stop: bool = False) -> tuple[float, float]:
    """Exit credit (what you receive). Conservative is worse than mid."""
    long_mid = (long_bid + long_ask) / 2.0
    long_fill = fill_toward_mid(long_bid, long_mid)
    width = long_ask - long_bid
    if stop:
        long_fill -= 0.20 * width
    if short_ask is None:
        return max(long_fill, 0.0), long_mid
    short_mid = (short_bid + short_ask) / 2.0
    # closing short = buy back at ask, 40% toward mid, stop worse
    short_fill = fill_toward_mid(short_ask, short_mid)
    sw = short_ask - short_bid
    if stop:
        short_fill += 0.20 * sw
    return max(long_fill - short_fill, 0.0), max(long_mid - short_mid, 0.0)


# ---------------------------------------------------------------------------
# Simulation + sizing
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    structure: str
    expiry: str
    dte: int
    long_strike: float
    short_strike: Optional[float]
    entry_debit: float
    entry_mid: float
    prob_profit: float
    ev: float
    log_growth: float
    contracts: int
    capital_at_risk: float
    tp_level: float
    sl_level: float
    verdict: str  # TRADE / SKIP
    skip_reason: str = ""
    prob_tp: float = 0.0
    prob_sl: float = 0.0
    prob_time_stop: float = 0.0
    mean_return: float = 0.0
    median_return: float = 0.0
    model_version: str = MODEL_VERSION
    returns: Optional[np.ndarray] = None


def size_position(entry_debit: float, portfolio: float = PORTFOLIO,
                  max_pct: float = MAX_PER_TRADE_PCT,
                  already_deployed: float = 0.0) -> tuple[int, float]:
    """4% max per trade, 20% max total deployed. Returns (contracts, capital_at_risk)."""
    if entry_debit <= 0:
        return 0, 0.0
    per_contract = entry_debit * 100 + 2 * FEE_PER_CONTRACT  # open both legs worst-case fee pad
    # long call: 1 leg open; spread: 2 legs — pad is fine
    max_dollars = min(portfolio * max_pct, portfolio * MAX_DEPLOYED_PCT - already_deployed)
    if max_dollars <= 0:
        return 0, 0.0
    n = int(max_dollars // per_contract)
    return n, float(n * per_contract)


def _bs_price_vec(S: np.ndarray, K: float, T: float, r: float, sigma: float,
                  kind: str = "C") -> np.ndarray:
    """Vectorized Black-Scholes price over an array of spots; K/T/r/sigma scalar."""
    S = np.asarray(S, dtype=float)
    intrinsic = np.maximum(S - K, 0.0) if kind == "C" else np.maximum(K - S, 0.0)
    if T <= 0 or sigma <= 0:
        return intrinsic
    sqrt_t = math.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-r * T)
    if kind == "C":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def simulate(spot: float, long_strike: float, short_strike: Optional[float],
             dte: int, entry_debit: float, pred_vol: float, drift_annual: float,
             structure: str = "long_call", n_paths: int = 40_000,
             already_deployed: float = 0.0, r: float = 0.05, q: float = 0.0,
             seed: int = 42,
             long_entry_iv: Optional[float] = None, short_entry_iv: Optional[float] = None,
             long_iv_exit: Optional[float] = None, short_iv_exit: Optional[float] = None,
             long_spread_frac: float = 0.15, short_spread_frac: float = 0.15,
             hold_days: Optional[int] = None) -> SimResult:
    """
    Path-dependent daily Monte Carlo. TP (2x debit) and SL (0.5x debit) are
    FIRST-TOUCH events on the conservative close price — not terminal-value
    thresholds. A spread that touches TP mid-hold and fades by expiry counts
    as a TP exit, the way it would in real trading. Survivors close at
    time_stop (day hold_days-5) on that day's conservative mark.

    Each leg is Black-Scholes marked at its OWN IV (may differ from the
    underlying's forecast vol used to drive the GBM path), linearly
    interpolated from entry IV to iv_at_exit (default: unchanged) across the
    hold. Bid/ask around each day's BS mid is synthesized by holding the
    leg's entry-day relative bid/ask width constant, then run through the
    same conservative-fill math as `credit_exit_fill` (inlined here, fully
    vectorized over paths — no per-path Python loop; the only loop is over
    the ~dte calendar days, which is required for first-touch logic).
    """
    rng = np.random.default_rng(seed)
    dte = max(int(dte), 1)
    n_steps = int(max(hold_days if hold_days is not None else dte, 1))
    time_stop_step = max(n_steps - 5, 1)
    dt_ = 1.0 / 365.0
    sig = max(pred_vol, 1e-4)
    mu = r - q + drift_annual

    has_short = short_strike is not None
    long_iv0 = long_entry_iv if long_entry_iv is not None else sig
    long_iv1 = long_iv_exit if long_iv_exit is not None else long_iv0
    if has_short:
        short_iv0 = short_entry_iv if short_entry_iv is not None else sig
        short_iv1 = short_iv_exit if short_iv_exit is not None else short_iv0

    z = rng.standard_normal((n_paths, n_steps))
    log_rets = (mu - 0.5 * sig * sig) * dt_ + sig * math.sqrt(dt_) * z
    log_path = np.cumsum(log_rets, axis=1)
    S = spot * np.exp(log_path)  # S[:, i] = spot after day i+1

    tp_level = entry_debit * 2.0   # 100% gain on debit -> value = 2x debit
    sl_level = entry_debit * 0.50  # 50% loss -> value = 0.5x debit

    alive = np.ones(n_paths, dtype=bool)
    exit_value = np.zeros(n_paths, dtype=float)
    exit_reason = np.full(n_paths, "time_stop", dtype=object)
    cons_normal = np.full(n_paths, entry_debit, dtype=float)  # fallback if n_steps loop is skipped

    for i in range(1, n_steps + 1):
        if not alive.any():
            break
        col = i - 1
        frac = i / n_steps if n_steps > 0 else 1.0
        T_remaining = max(dte - i, 0) / TRADING_DAYS

        long_iv_t = long_iv0 + (long_iv1 - long_iv0) * frac
        long_mid = _bs_price_vec(S[:, col], long_strike, T_remaining, r, long_iv_t, kind="C")
        long_bid = long_mid * (1 - long_spread_frac / 2)
        long_ask = long_mid * (1 + long_spread_frac / 2)
        long_fill = fill_toward_mid(long_bid, long_mid)

        if has_short:
            short_iv_t = short_iv0 + (short_iv1 - short_iv0) * frac
            short_mid = _bs_price_vec(S[:, col], short_strike, T_remaining, r, short_iv_t, kind="C")
            short_bid = short_mid * (1 - short_spread_frac / 2)
            short_ask = short_mid * (1 + short_spread_frac / 2)
            short_fill = fill_toward_mid(short_ask, short_mid)
            cons_normal = np.maximum(long_fill - short_fill, 0.0)
        else:
            cons_normal = np.maximum(long_fill, 0.0)

        tp_touch = alive & (cons_normal >= tp_level)
        sl_touch = alive & (cons_normal <= sl_level) & ~tp_touch

        if sl_touch.any():
            long_w = long_ask - long_bid
            long_fill_stop = long_fill - 0.20 * long_w
            if has_short:
                short_w = short_ask - short_bid
                short_fill_stop = short_fill + 0.20 * short_w
                cons_stop = np.maximum(long_fill_stop - short_fill_stop, 0.0)
            else:
                cons_stop = np.maximum(long_fill_stop, 0.0)
        else:
            cons_stop = cons_normal

        is_time_stop_day = (i == time_stop_step)
        time_stop_touch = (alive & ~(tp_touch | sl_touch)) if is_time_stop_day else np.zeros(n_paths, dtype=bool)

        exit_value = np.where(tp_touch, cons_normal, exit_value)
        exit_value = np.where(sl_touch, cons_stop, exit_value)
        exit_value = np.where(time_stop_touch, cons_normal, exit_value)
        exit_reason[tp_touch] = "tp"
        exit_reason[sl_touch] = "sl"
        exit_reason[time_stop_touch] = "time_stop"

        alive = alive & ~(tp_touch | sl_touch | time_stop_touch)

    if alive.any():
        # shouldn't happen (time_stop_step <= n_steps forces closure) — safety net
        exit_value = np.where(alive, cons_normal, exit_value)
        exit_reason[alive] = "time_stop"

    legs = 1 if not has_short else 2
    fees = legs * 2 * FEE_PER_CONTRACT / 100.0  # per share, open+close
    pnl = exit_value - entry_debit - fees
    prob = float((pnl > 0).mean())
    ev = float(pnl.mean())
    risk = max(entry_debit + fees, 1e-6)
    rets = np.clip(pnl / risk, -0.999, 10.0)
    log_g = float(np.mean(np.log1p(rets)))

    prob_tp = float((exit_reason == "tp").mean())
    prob_sl = float((exit_reason == "sl").mean())
    prob_time_stop = float((exit_reason == "time_stop").mean())

    contracts, capital = size_position(entry_debit, already_deployed=already_deployed)

    verdict = "TRADE"
    reason = ""
    if contracts == 0:
        verdict, reason = "SKIP", "contracts==0 (size/cap)"
    elif log_g <= LOG_GROWTH_MIN:
        verdict, reason = "SKIP", f"log_growth {log_g:.4f} <= {LOG_GROWTH_MIN}"

    return SimResult(
        structure=structure,
        expiry="",
        dte=dte,
        long_strike=long_strike,
        short_strike=short_strike,
        entry_debit=entry_debit,
        entry_mid=entry_debit,  # caller overwrites with true mid
        prob_profit=prob,
        ev=ev,
        log_growth=log_g,
        contracts=contracts,
        capital_at_risk=capital,
        tp_level=tp_level,
        sl_level=sl_level,
        verdict=verdict,
        skip_reason=reason,
        prob_tp=prob_tp,
        prob_sl=prob_sl,
        prob_time_stop=prob_time_stop,
        mean_return=float(np.mean(rets)),
        median_return=float(np.median(rets)),
        model_version=MODEL_VERSION,
        returns=rets,
    )


def atm_iv_from_chain(chain_rows: "pd.DataFrame", spot: float) -> float:
    """Nearest-to-spot call IV from a chain snapshot. NaN if unusable."""
    calls = chain_rows[chain_rows["type"] == "C"].copy()
    if calls.empty or "iv" not in calls.columns:
        return float("nan")
    iv = calls["iv"].astype(float)
    calls = calls.loc[np.isfinite(iv)]
    if calls.empty:
        return float("nan")
    gap = (calls["strike"].astype(float) - float(spot)).abs()
    return float(calls.loc[gap.idxmin(), "iv"])


def evaluate(ticker: str, spot: float, chain_rows: "pd.DataFrame",
             pred_vol_annual: Optional[float], pred_move_pct: float, horizon_days: int,
             already_deployed: float = 0.0,
             forecast_vol: Optional[float] = None,
             edge_drift_annual: Optional[float] = None) -> dict:
    """
    Call debit spreads only, 14–30 DTE. Monte Carlo under the forecast;
    Kelly (log_growth) is the TRADE gate.

    Market default: forecast_vol / pred_vol_annual is None → ATM IV, and
    edge_drift_annual is 0. That should SKIP after fees and conservative fills.
    A TRADE requires a forecast that beats those market numbers on log growth.
    """
    import pandas as pd  # local to keep import light for IV-only callers

    market_iv = atm_iv_from_chain(chain_rows, spot)
    vol = forecast_vol if forecast_vol is not None else pred_vol_annual
    if vol is None or not np.isfinite(vol):
        vol = market_iv
    if vol is None or not np.isfinite(vol) or vol <= 0:
        return {"verdict": "SKIP", "skip_reason": "no forecast_vol / ATM IV", "candidates": []}

    if edge_drift_annual is not None:
        drift = float(edge_drift_annual)
    else:
        drift = float(pred_move_pct) * (TRADING_DAYS / max(horizon_days, 1))

    calls = chain_rows[(chain_rows["type"] == "C")].copy()
    if calls.empty:
        return {"verdict": "SKIP", "skip_reason": "no calls in chain", "candidates": []}

    window = calls[(calls["dte"] >= MIN_DTE) & (calls["dte"] <= MAX_DTE)]
    if window.empty:
        return {"verdict": "SKIP", "skip_reason": f"no expiry in {MIN_DTE}–{MAX_DTE} DTE",
                "candidates": []}

    window = window.copy()
    window["dte_gap"] = (window["dte"] - horizon_days).abs()
    expiry = window.sort_values("dte_gap").iloc[0]["expiry"]
    book = window[window["expiry"] == expiry].sort_values("strike").reset_index(drop=True)
    dte = int(book["dte"].iloc[0])
    max_debit = (PORTFOLIO * MAX_PER_TRADE_PCT - 2 * FEE_PER_CONTRACT) / 100.0

    def _leg_iv(row, fallback):
        try:
            v = float(row.get("iv"))
        except (TypeError, ValueError):
            return fallback
        return v if np.isfinite(v) and v > 0 else fallback

    def _spread_frac(bid, ask, mid, fallback=0.15):
        if mid is None or not np.isfinite(mid) or mid <= 0:
            return fallback
        return max((ask - bid) / mid, 1e-4)

    def try_spread(long_row, short_row):
        long_k = float(long_row["strike"])
        short_k = float(short_row["strike"])
        if short_k <= long_k:
            return None
        debit, mid = debit_entry_fill(
            float(long_row["ask"]), float(long_row["bid"]),
            float(short_row["bid"]), float(short_row["ask"]),
        )
        if debit <= 0 or not np.isfinite(debit):
            return None
        long_mid_q = (float(long_row["bid"]) + float(long_row["ask"])) / 2.0
        short_mid_q = (float(short_row["bid"]) + float(short_row["ask"])) / 2.0
        sim = simulate(
            spot, long_k, short_k, dte, debit, vol, drift,
            structure="call_debit_spread", already_deployed=already_deployed,
            long_entry_iv=_leg_iv(long_row, vol), short_entry_iv=_leg_iv(short_row, vol),
            long_spread_frac=_spread_frac(float(long_row["bid"]), float(long_row["ask"]), long_mid_q),
            short_spread_frac=_spread_frac(float(short_row["bid"]), float(short_row["ask"]), short_mid_q),
        )
        sim.expiry = str(expiry)
        sim.entry_mid = mid
        return sim

    candidates = []
    for target_m in (0.0, 0.02, 0.04):
        idx = (book["strike"] - spot * (1 + target_m)).abs().idxmin()
        long_row = book.loc[idx]
        for width in (0.03, 0.05, 0.08):
            short_target = float(long_row["strike"]) * (1 + width)
            above = book[book["strike"] > float(long_row["strike"])]
            if above.empty:
                continue
            sidx = (above["strike"] - short_target).abs().idxmin()
            sim = try_spread(long_row, book.loc[sidx])
            if sim:
                candidates.append(sim)

    if not candidates:
        return {"verdict": "SKIP", "skip_reason": "no viable debit spreads", "candidates": []}

    trades = [c for c in candidates if c.verdict == "TRADE"]
    if trades:
        fit = [c for c in trades if c.entry_debit <= max_debit + 1e-9]
        pool = fit or trades
        best = max(pool, key=lambda c: c.log_growth)
    else:
        best = max(candidates, key=lambda c: c.log_growth)

    out = asdict(best)
    out["candidates"] = [asdict(c) for c in candidates]
    out["drift_annual"] = drift
    out["forecast_vol"] = vol
    out["market_iv"] = market_iv if np.isfinite(market_iv) else None
    out["max_debit_budget"] = max_debit
    out["spot"] = spot
    return out


def breakeven_forecast(spread: dict, hold_days: Optional[int] = None,
                       seed: int = 123) -> dict:
    """
    What a TRADE verdict requires you to believe — market data + the fill
    model, NOT a model opinion. Safe to show BEFORE a human forecast without
    violating the blind-forecast rule: it doesn't suggest a view, it just
    states the minimum directional or vol claim that already flips the
    market-default (ATM IV, zero drift) verdict from SKIP to TRADE.

    `spread` is an evaluate() result (or one of its `candidates`) — needs
    spot, long_strike, short_strike, entry_debit, dte, market_iv.
    Bisects the Patch-1 path-dependent simulator: first edge_drift_annual
    with vol held at market IV, then forecast_vol with drift held at zero.
    """
    spot = float(spread["spot"])
    long_k = float(spread["long_strike"])
    short_k = spread.get("short_strike")
    short_k = float(short_k) if short_k not in (None, "") else None
    dte = int(spread["dte"])
    entry_debit = float(spread["entry_debit"])
    market_iv = float(spread["market_iv"])
    hold = int(hold_days) if hold_days is not None else dte
    structure = "call_debit_spread" if short_k is not None else "long_call"

    def log_growth_at(vol: float, drift: float) -> float:
        sim = simulate(spot, long_k, short_k, dte, entry_debit, vol, drift,
                       structure=structure, n_paths=8000, seed=seed,
                       long_entry_iv=market_iv, short_entry_iv=market_iv,
                       hold_days=hold)
        return sim.log_growth

    market_log_growth = log_growth_at(market_iv, 0.0)

    def _bisect(f, lo, hi, iters=30):
        """f monotonically increasing in x; find smallest x with f(x) > 0."""
        if f(lo) > 0:
            return lo
        if f(hi) <= 0:
            return float("nan")  # not achievable even at the search ceiling
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        return hi

    drift_be = _bisect(lambda d: log_growth_at(market_iv, d), 0.0, 5.0)
    T = hold / TRADING_DAYS
    move_pct = float(math.exp(drift_be * T) - 1.0) if np.isfinite(drift_be) else float("nan")

    vol_be = _bisect(lambda v: log_growth_at(v, 0.0), max(market_iv, 1e-4), 5.0)

    return {
        "move_pct": move_pct,
        "vol_forecast": vol_be,
        "market_log_growth": market_log_growth,
        "hold_days": hold,
        "market_iv": market_iv,
    }
