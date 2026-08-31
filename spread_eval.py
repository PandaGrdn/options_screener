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
IV_RANK_MAX = 30.0           # screen: skip if IV rank is richer than this


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

def fill_toward_mid(natural: float, mid: float, frac: float = 0.4) -> float:
    """natural ± 40% of the way to mid. Same formula for debit entry / credit exit."""
    return float(natural - frac * (natural - mid))


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


def simulate(spot: float, long_strike: float, short_strike: Optional[float],
             dte: int, entry_debit: float, pred_vol: float, drift_annual: float,
             structure: str = "long_call", n_paths: int = 8000,
             already_deployed: float = 0.0, r: float = 0.05,
             seed: int = 42) -> SimResult:
    """
    Terminal-value Monte Carlo under user's vol/drift forecast.
    TP = 100% of debit, SL = 50% of debit (spread value).
    """
    rng = np.random.default_rng(seed)
    T = max(dte, 1) / TRADING_DAYS
    mu = drift_annual
    sig = max(pred_vol, 1e-4)
    z = rng.standard_normal(n_paths)
    ST = spot * np.exp((mu - 0.5 * sig * sig) * T + sig * math.sqrt(T) * z)

    long_pay = np.maximum(ST - long_strike, 0.0)
    if short_strike is not None:
        short_pay = np.maximum(ST - short_strike, 0.0)
        payoff = long_pay - short_pay
        width = short_strike - long_strike
        payoff = np.minimum(payoff, width)
    else:
        payoff = long_pay

    # fees: open + close, 1 or 2 legs
    legs = 1 if short_strike is None else 2
    fees = legs * 2 * FEE_PER_CONTRACT / 100.0  # per share
    pnl = payoff - entry_debit - fees
    prob = float((pnl > 0).mean())
    ev = float(pnl.mean())
    # log-growth on fraction of capital at risk; avoid -inf on total loss paths
    risk = max(entry_debit + fees, 1e-6)
    rets = np.clip(pnl / risk, -0.999, 10.0)
    log_g = float(np.mean(np.log1p(rets)))

    contracts, capital = size_position(entry_debit, already_deployed=already_deployed)
    tp = entry_debit * 2.0          # 100% gain on debit → value = 2x debit
    sl = entry_debit * 0.50        # 50% loss → value = 0.5x debit

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
        tp_level=tp,
        sl_level=sl,
        verdict=verdict,
        skip_reason=reason,
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
        sim = simulate(spot, long_k, short_k, dte, debit, vol, drift,
                       structure="call_debit_spread", already_deployed=already_deployed)
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
    return out
