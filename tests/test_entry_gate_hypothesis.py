"""Patch 2 (cheapness_gate wiring) + Patch 3 (hypothesis field) in paper/entry.py."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import paper.entry as entry
import paper.models as models
import signals
from spread_eval import bs_price, TRADING_DAYS, MODEL_VERSION


def _dates(n, end=dt.date(2026, 8, 31)):
    return [(end - dt.timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


def _write_underlying(path, ticker, n, rv_values):
    rows = [
        {"date": d, "ticker": ticker, "open": 100, "high": 101, "low": 99, "close": 100,
         "volume": 1_000_000, "rv20_yz": rv, "rv60_yz": rv, "rv20_c2c": rv, "rv20_pctile": np.nan}
        for d, rv in zip(_dates(n), rv_values)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_chain(path, ticker, asof, spot, iv, dte=21, spread_frac=0.10):
    T = dte / TRADING_DAYS
    rows = []
    for k in (spot - 5, spot, spot + 5, spot + 10):
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


def _write_earnings(path, ticker, asof, days_out=200):
    ed = (dt.date.fromisoformat(asof) + dt.timedelta(days=days_out)).isoformat()
    pd.DataFrame([{"asof": asof, "ticker": ticker, "earnings_date": ed,
                   "days_to_earnings": days_out}]).to_csv(path, index=False)


@pytest.fixture
def market(tmp_path, monkeypatch):
    und = tmp_path / "underlying_history.csv"
    chain = tmp_path / "chain_history.csv"
    earn = tmp_path / "earnings_calendar.csv"
    data = tmp_path / "data"
    data.mkdir()

    monkeypatch.setattr(entry, "chain_history_path", lambda: chain)
    monkeypatch.setattr(entry, "underlying_history_path", lambda: und)
    monkeypatch.setattr(entry, "earnings_calendar_path", lambda: earn)
    monkeypatch.setattr(signals, "underlying_history_path", lambda: und)
    monkeypatch.setattr(models, "FORECASTS", data / "forecasts.csv")
    monkeypatch.setattr(models, "TRADES", data / "trades.csv")
    monkeypatch.setattr(models, "ensure_data_dir", lambda: data.mkdir(exist_ok=True))

    asof = "2026-08-31"
    return {"und": und, "chain": chain, "earn": earn, "asof": asof}


def test_decide_forces_skip_when_gate_fails(market):
    asof = market["asof"]
    # RV percentile pinned high -> gate fails regardless of the model's Kelly verdict
    _write_underlying(market["und"], "NVDA", 130, [0.20] * 129 + [0.90])
    _write_chain(market["chain"], "NVDA", asof, spot=100.0, iv=0.91)
    _write_earnings(market["earn"], "NVDA", asof)

    out = entry.decide("NVDA", horizon_days=21, noninteractive={
        "pred_vol_annual": 0.91, "pred_move_pct": 3.0,  # huge drift -> model would say TRADE
        "decision": "skip", "skip_reason": "gate says skip",
        "rationale": "x" * 25, "hypothesis": "trend",
    })
    assert out["forecast"]["decision"] == "skip"
    assert "RV percentile" in out["forecast"]["gate_reason"]
    assert out["forecast"]["hypothesis"] == "trend"
    assert out["forecast"]["model_version"] == MODEL_VERSION


def test_decide_rejects_invalid_hypothesis(market):
    asof = market["asof"]
    _write_underlying(market["und"], "NVDA", 130, [0.20] * 130)
    _write_chain(market["chain"], "NVDA", asof, spot=100.0, iv=0.20)
    _write_earnings(market["earn"], "NVDA", asof)

    with pytest.raises(ValueError, match="hypothesis"):
        entry.decide("NVDA", horizon_days=21, noninteractive={
            "pred_vol_annual": 0.20, "pred_move_pct": 0.0,
            "decision": "skip", "skip_reason": "n/a", "rationale": "x" * 25,
            "hypothesis": "not_a_real_hypothesis",
        })


def test_prompt_forecast_requires_hypothesis(market):
    asof = market["asof"]
    _write_underlying(market["und"], "NVDA", 130, [0.20] * 130)
    _write_chain(market["chain"], "NVDA", asof, spot=100.0, iv=0.20)
    _write_earnings(market["earn"], "NVDA", asof)

    row = entry.prompt_forecast("NVDA", noninteractive={
        "horizon_days": 21, "direction": "up", "pred_move_pct": 0.02,
        "pred_vol_annual": 0.25, "pred_prob_profit": 0.4,
        "rationale": "x" * 25, "decision": "skip", "skip_reason": "just logging",
        "hypothesis": "vol_expansion",
    })
    assert row["hypothesis"] == "vol_expansion"
    assert row["model_version"] == ""  # blind forecast never calls the model

    with pytest.raises(ValueError, match="hypothesis"):
        entry.prompt_forecast("NVDA", noninteractive={
            "horizon_days": 21, "direction": "up", "pred_move_pct": 0.02,
            "pred_vol_annual": 0.25, "pred_prob_profit": 0.4,
            "rationale": "x" * 25, "decision": "skip", "skip_reason": "just logging",
            "hypothesis": "bogus",
        })
