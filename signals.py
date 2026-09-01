"""
signals.py — the cheapness gate that replaces the raw IV-rank screen.

Problem: IV_RANK_MAX=30 (spread_eval.py) needs ~6-12 months of IV history to
mean anything. With only weeks of chain_history, IV rank is noise, and it was
deciding which tickers are allowed to trade. This module scores "cheapness"
from what we actually have enough of (RV history, backfillable 2y) and lets
IV rank kick in on its own once there's enough of it — no manual flag flip.

Four features, no more:
  rv_percentile   — Yang-Zhang RV percentile vs the ticker's own history
  iv_rv_spread    — ATM IV (30d, recomputed from chain mid) minus current RV20
  iv_rank         — existing calc, nan below min_history_days
  cheapness_gate  — combines the three; the nan iv_rank term is ignored until
                    there's enough history, so the gate silently strengthens
                    over time without a code change.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from paper import underlying_history_path

RV_PERCENTILE_MAX = 40.0
IV_RV_SPREAD_MAX = 0.03
IV_RANK_MAX = 30.0
MIN_RV_OBS = 120
MIN_IV_RANK_DAYS = 180


def _underlying_series(ticker: str, asof: Optional[str] = None) -> pd.DataFrame:
    path = underlying_history_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if asof is not None:
        df = df[df["date"].astype(str) <= str(asof)]
    return df.sort_values("date")


def rv_percentile(ticker: str, asof: Optional[str] = None, window: int = 20,
                  lookback: int = 504) -> float:
    """Yang-Zhang RV percentile from underlying_history (backfillable 2y).
    Returns 0-100, nan if < 120 obs."""
    col = "rv20_yz" if window <= 20 else "rv60_yz"
    df = _underlying_series(ticker, asof)
    if df.empty or col not in df.columns:
        return float("nan")
    s = df[col].astype(float).dropna()
    if len(s) < MIN_RV_OBS:
        return float("nan")
    s = s.tail(lookback)
    if len(s) < MIN_RV_OBS:
        return float("nan")
    return float((s < s.iloc[-1]).mean() * 100)


def _latest_rv(ticker: str, asof: Optional[str] = None, window: int = 20) -> float:
    col = "rv20_yz" if window <= 20 else "rv60_yz"
    df = _underlying_series(ticker, asof)
    if df.empty or col not in df.columns:
        return float("nan")
    s = df[col].astype(float).dropna()
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


def _atm_iv_30(ticker: str, asof: Optional[str] = None) -> float:
    """ATM IV nearest 30 DTE, recomputed from chain_history mid (NOT yf_iv)."""
    from paper.entry import load_chain, atm_iv_from_chain  # local: avoid import cycle at module load

    try:
        df = load_chain(ticker)
    except FileNotFoundError:
        return float("nan")
    if asof is not None:
        df = df[df["date"].astype(str) <= str(asof)]
    if df.empty:
        return float("nan")
    latest = df["date"].astype(str).max()
    day = df[df["date"].astype(str) == latest]
    spot = float(day["spot"].iloc[0])
    return atm_iv_from_chain(day, spot)


def iv_rv_spread(ticker: str, asof: Optional[str] = None) -> float:
    """(ATM IV nearest 30 DTE) minus current RV20. Vol points, e.g. +0.04."""
    iv30 = _atm_iv_30(ticker, asof)
    rv20 = _latest_rv(ticker, asof, window=20)
    if not np.isfinite(iv30) or not np.isfinite(rv20):
        return float("nan")
    return float(iv30 - rv20)


def iv_rank(ticker: str, asof: Optional[str] = None,
           min_history_days: int = MIN_IV_RANK_DAYS, lookback: int = 252) -> float:
    """Existing IV-rank calc (paper.entry), but nan below min_history_days —
    it stays inert until there's enough IV history to mean something."""
    from paper.entry import iv_series_for_ticker  # local: avoid import cycle at module load

    s = iv_series_for_ticker(ticker).dropna()
    if asof is not None:
        s = s[s.index.astype(str) <= str(asof)]
    if len(s) < min_history_days:
        return float("nan")
    s = s.tail(lookback)
    return float((s < s.iloc[-1]).mean() * 100)


def cheapness_gate(ticker: str, asof: Optional[str] = None) -> tuple[bool, str]:
    """
    Passes when:
      rv_percentile <= 40          (vol compressed vs own history)
      AND iv_rv_spread <= +0.03    (not paying a fat premium over realized)
      AND iv_rank <= 30 IF iv_rank is not nan (activates automatically as
                                    history matures — no manual flag flip)
    Returns (passed, human-readable reason string).
    """
    ticker = ticker.upper()
    rvp = rv_percentile(ticker, asof)
    if not np.isfinite(rvp):
        return False, f"insufficient RV history for {ticker} (<{MIN_RV_OBS} obs)"

    ivrv = iv_rv_spread(ticker, asof)
    if not np.isfinite(ivrv):
        return False, f"insufficient IV/RV history for {ticker} (iv_rv_spread n/a)"

    ivr = iv_rank(ticker, asof)

    reasons = []
    if rvp > RV_PERCENTILE_MAX:
        reasons.append(f"RV percentile {rvp:.0f} > {RV_PERCENTILE_MAX:.0f}")
    if ivrv > IV_RV_SPREAD_MAX:
        reasons.append(f"IV-RV spread {ivrv:+.3f} > {IV_RV_SPREAD_MAX:+.3f}")
    if np.isfinite(ivr) and ivr > IV_RANK_MAX:
        reasons.append(f"IV rank {ivr:.0f} > {IV_RANK_MAX:.0f}")

    if reasons:
        return False, "; ".join(reasons)

    ivr_note = f"IV rank {ivr:.0f}" if np.isfinite(ivr) else "IV rank n/a (<180d history, not yet active)"
    return True, f"cheap: RV%ile {rvp:.0f}, IV-RV {ivrv:+.3f}, {ivr_note}"
