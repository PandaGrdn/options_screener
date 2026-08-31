"""Debit-spread Monte Carlo: Kelly gate, 14–30 DTE, no naked calls."""

from __future__ import annotations

import pandas as pd

from spread_eval import LOG_GROWTH_MIN, evaluate, simulate


def _chain(spot=100.0, dte=21, iv=0.35):
    expiry = "2026-09-21"
    rows = []
    for k, bid, ask in (
        (100.0, 0.80, 0.95),
        (103.0, 0.35, 0.50),
        (105.0, 0.18, 0.28),
        (108.0, 0.08, 0.16),
    ):
        rows.append({
            "type": "C", "strike": k, "bid": bid, "ask": ask,
            "dte": dte, "expiry": expiry, "iv": iv,
        })
    return pd.DataFrame(rows)


def test_evaluate_spreads_only_and_market_skips():
    out = evaluate("TEST", 100.0, _chain(), pred_vol_annual=None, pred_move_pct=0.0,
                   horizon_days=21)
    assert out["structure"] == "call_debit_spread"
    assert out["short_strike"] is not None
    assert all(c["structure"] == "call_debit_spread" for c in out["candidates"])
    assert out["verdict"] == "SKIP"
    assert "log_growth" in out["skip_reason"]


def test_evaluate_rejects_expiry_outside_window():
    out = evaluate("TEST", 100.0, _chain(dte=60), None, 0.0, 21)
    assert out["verdict"] == "SKIP"
    assert "14" in out["skip_reason"] and "30" in out["skip_reason"]


def test_kelly_gate_on_simulate():
    # Debit too rich vs 3-wide max payoff → negative log growth
    sim = simulate(100.0, 100.0, 103.0, dte=21, entry_debit=0.80,
                   pred_vol=0.35, drift_annual=0.0, structure="call_debit_spread")
    assert sim.log_growth <= LOG_GROWTH_MIN
    assert sim.verdict == "SKIP"


def test_strong_drift_forecast_can_trade():
    out = evaluate("TEST", 100.0, _chain(), pred_vol_annual=0.35, pred_move_pct=0.0,
                   horizon_days=21, edge_drift_annual=3.0)
    assert out["verdict"] == "TRADE"
    assert out["log_growth"] > LOG_GROWTH_MIN
    assert out["structure"] == "call_debit_spread"
