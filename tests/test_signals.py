"""Patch 2: cheapness_gate replaces the raw (data-starved) IV-rank screen."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import signals
import paper.entry as entry
from spread_eval import bs_price, TRADING_DAYS


def _dates(n, end=dt.date(2026, 8, 31)):
    return [(end - dt.timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


@pytest.fixture
def tmp_market(tmp_path, monkeypatch):
    und_path = tmp_path / "underlying_history.csv"
    chain_path = tmp_path / "chain_history.csv"
    monkeypatch.setattr(signals, "underlying_history_path", lambda: und_path)
    monkeypatch.setattr(entry, "chain_history_path", lambda: chain_path)
    return und_path, chain_path


def _write_underlying(path, ticker, n, rv_values):
    rows = []
    for d, rv in zip(_dates(n), rv_values):
        rows.append({"date": d, "ticker": ticker, "open": 100, "high": 101, "low": 99,
                     "close": 100, "volume": 1_000_000, "rv20_yz": rv, "rv60_yz": rv,
                     "rv20_c2c": rv, "rv20_pctile": np.nan})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_chain(path, ticker, asof, spot, iv, dte=30, spread_frac=0.10):
    T = dte / TRADING_DAYS
    rows = []
    for k in (spot - 5, spot, spot + 5):
        price = max(bs_price(spot, k, T, 0.05, iv, kind="C"), 0.02)
        bid = round(price * (1 - spread_frac / 2), 4)
        ask = round(price * (1 + spread_frac / 2), 4)
        rows.append({
            "date": asof, "ts_utc": asof, "ticker": ticker, "spot": spot,
            "expiry": (dt.date.fromisoformat(asof) + dt.timedelta(days=dte)).isoformat(),
            "dte": dte, "type": "C", "strike": k, "moneyness": (k - spot) / spot,
            "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4), "spread_pct": spread_frac,
            "volume": 100, "open_interest": 100, "yf_iv": iv, "iv": iv,
            "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cheapness_gate_fails_on_high_rv_percentile(tmp_market):
    und_path, chain_path = tmp_market
    asof = "2026-08-31"
    # 129 quiet days then a vol spike on the asof day -> RV percentile ~99
    rv_values = [0.20] * 129 + [0.90]
    _write_underlying(und_path, "AAA", 130, rv_values)
    _write_chain(chain_path, "AAA", asof, spot=100.0, iv=0.91)  # close to rv20 -> iv_rv_spread passes

    passed, reason = signals.cheapness_gate("AAA", asof=asof)
    assert passed is False
    assert "RV percentile" in reason


def test_rv_percentile_nan_below_min_obs(tmp_market):
    und_path, chain_path = tmp_market
    _write_underlying(und_path, "BBB", 50, [0.2] * 50)  # < MIN_RV_OBS (120)
    assert np.isnan(signals.rv_percentile("BBB"))
    passed, reason = signals.cheapness_gate("BBB")
    assert passed is False
    assert "insufficient" in reason.lower()


def test_iv_rank_nan_below_180_days_gate_still_functions(tmp_market):
    und_path, chain_path = tmp_market
    asof = "2026-08-31"
    # plenty of RV history, all quiet -> RV percentile low, gate should be
    # decidable even though only one day of chain (IV) history exists.
    _write_underlying(und_path, "CCC", 200, [0.20] * 199 + [0.20])
    _write_chain(chain_path, "CCC", asof, spot=100.0, iv=0.20)

    ivr = signals.iv_rank("CCC", asof=asof)
    assert np.isnan(ivr)  # only 1 day of IV history, far below 180

    passed, reason = signals.cheapness_gate("CCC", asof=asof)
    assert passed is True  # nan iv_rank term must not block a pass
    assert "IV rank n/a" in reason


def test_backfill_idempotent_when_already_sufficient(tmp_path, monkeypatch):
    import backfill_underlyings as bf

    path = tmp_path / "underlying_history.csv"
    _write_underlying(path, "DDD", 130, [0.2] * 130)

    need_before = bf._tickers_needing_backfill(["DDD"], path=str(path))
    assert need_before == []  # already >= MIN_RV_OBS -> nothing to pull

    # backfill() must short-circuit to 0 without touching the network when
    # nothing needs pulling — safe to call repeatedly.
    n1 = bf.backfill(tickers=["DDD"], path=str(path))
    n2 = bf.backfill(tickers=["DDD"], path=str(path))
    assert n1 == 0
    assert n2 == 0
