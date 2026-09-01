"""Debit-spread Monte Carlo: Kelly gate, 14–30 DTE, no naked calls."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spread_eval import (
    LOG_GROWTH_MIN, TRADING_DAYS, bs_price, breakeven_forecast, evaluate, simulate,
)


def _chain(spot=100.0, dte=21, iv=0.35, r=0.05, spread_frac=0.12):
    """Self-consistent synthetic chain: bid/ask are BS-priced FROM `iv`, the
    way real chain_history rows are (snapshot.py solves `iv` from that same
    row's mid). A fixture whose quoted price doesn't match its own iv field
    creates an artificial day-1 repricing jump once the simulator marks legs
    off `iv` (Patch 1) — this mirrors real data instead.
    """
    expiry = "2026-09-21"
    T = dte / TRADING_DAYS
    rows = []
    for k in np.arange(90, 121, 1.0):
        price = max(bs_price(spot, float(k), T, r, iv, kind="C"), 0.02)
        bid = round(max(price * (1 - spread_frac / 2), 0.01), 4)
        ask = round(price * (1 + spread_frac / 2), 4)
        rows.append({
            "type": "C", "strike": float(k), "bid": bid, "ask": ask,
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
    assert "log_growth" in out["skip_reason"] or "contracts" in out["skip_reason"]


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


# ---------------------------------------------------------------------------
# Patch 1 — path-dependent Monte Carlo
# ---------------------------------------------------------------------------

def test_path_dependence_tp_touch_exceeds_terminal_only():
    """TP is a first-touch event: high vol + a TP close to entry should be hit
    mid-hold far more often than terminal value alone would ever cross it."""
    spot, long_k, short_k, dte, debit = 100.0, 100.0, 108.0, 21, 1.0
    sim = simulate(
        spot, long_k, short_k, dte, debit, pred_vol=1.2, drift_annual=0.0,
        structure="call_debit_spread", long_entry_iv=1.2, short_entry_iv=1.2,
        long_spread_frac=0.05, short_spread_frac=0.05, n_paths=20_000, seed=11,
    )
    tp_level = debit * 2.0

    # terminal-only baseline: intrinsic payoff at expiry only, same process
    rng = np.random.default_rng(11)
    T = dte / TRADING_DAYS
    z = rng.standard_normal(20_000)
    ST = spot * np.exp((0.05 - 0.5 * 1.2 ** 2) * T + 1.2 * math.sqrt(T) * z)
    payoff = np.clip(np.maximum(ST - long_k, 0.0) - np.maximum(ST - short_k, 0.0),
                     None, short_k - long_k)
    terminal_prob_tp = float((payoff >= tp_level).mean())

    assert sim.prob_tp > terminal_prob_tp + 0.05


def test_zero_vol_degenerate_all_time_stop_deterministic():
    # Process vol -> 0 (underlying barely moves) but legs still mark at a
    # real IV, consistent with entry_debit — decoupled from the degenerate
    # GBM driver, the way real leg IV is decoupled from forecast vol.
    spot, long_k, short_k, dte, iv = 100.0, 100.0, 105.0, 21, 0.35
    T = dte / TRADING_DAYS
    entry_debit = bs_price(spot, long_k, T, 0.05, iv) - bs_price(spot, short_k, T, 0.05, iv)
    sim = simulate(spot, long_k, short_k, dte, entry_debit, pred_vol=1e-6, drift_annual=0.0,
                   structure="call_debit_spread", long_entry_iv=iv, short_entry_iv=iv,
                   n_paths=500, seed=1)
    assert sim.prob_time_stop == pytest.approx(1.0)
    assert sim.prob_tp == 0.0
    assert sim.prob_sl == 0.0
    assert np.std(sim.returns) < 1e-3


def test_exit_reason_probabilities_sum_to_one():
    sim = simulate(100.0, 100.0, 105.0, 21, 1.0, pred_vol=0.4, drift_annual=0.0,
                   structure="call_debit_spread", n_paths=5000, seed=3)
    assert sim.prob_tp + sim.prob_sl + sim.prob_time_stop == pytest.approx(1.0, abs=1e-9)


def test_reproducible_same_seed():
    kwargs = dict(structure="call_debit_spread", n_paths=2000, seed=5)
    a = simulate(100.0, 100.0, 105.0, 21, 1.0, 0.4, 0.0, **kwargs)
    b = simulate(100.0, 100.0, 105.0, 21, 1.0, 0.4, 0.0, **kwargs)
    assert a.log_growth == b.log_growth
    assert a.prob_tp == b.prob_tp
    assert a.prob_sl == b.prob_sl
    assert np.array_equal(a.returns, b.returns)


# ---------------------------------------------------------------------------
# Patch 4 — breakeven forecast display
# ---------------------------------------------------------------------------

def test_breakeven_drift_fed_back_yields_zero_log_growth():
    out = evaluate("TEST", 100.0, _chain(), pred_vol_annual=None, pred_move_pct=0.0,
                   horizon_days=21)
    assert out["verdict"] == "SKIP"
    be = breakeven_forecast(out)
    assert np.isfinite(be["move_pct"])

    T = be["hold_days"] / TRADING_DAYS
    drift_be = math.log1p(be["move_pct"]) / T
    sim = simulate(out["spot"], out["long_strike"], out["short_strike"], out["dte"],
                   out["entry_debit"], be["market_iv"], drift_be,
                   structure="call_debit_spread", n_paths=8000, seed=123,
                   long_entry_iv=be["market_iv"], short_entry_iv=be["market_iv"],
                   hold_days=be["hold_days"])
    assert abs(sim.log_growth) < 0.02  # bisection converges tight; loose gate for MC noise


def test_breakeven_move_strictly_positive_for_debit_spread_with_fees():
    out = evaluate("TEST", 100.0, _chain(), pred_vol_annual=None, pred_move_pct=0.0,
                   horizon_days=21)
    assert out["log_growth"] <= 0
    be = breakeven_forecast(out)
    assert be["market_log_growth"] <= 0
    assert be["move_pct"] > 0


def test_market_default_skip_regression():
    """Market-default (ATM IV, zero drift) must still SKIP after fees/fills —
    the core invariant Patch 1 must not break (AGENT_CONTEXT §10)."""
    out = evaluate("TEST", 100.0, _chain(), pred_vol_annual=None, pred_move_pct=0.0,
                   horizon_days=21)
    assert out["verdict"] == "SKIP"
    assert out["log_growth"] <= 0
