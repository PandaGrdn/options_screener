"""Put credit spread backtest: fills, settlement, model odds, strike grid."""

from __future__ import annotations

import pandas as pd
import pytest

import credit_backtest as cb
from backtest import FEES


def test_credit_fill_is_worse_than_mid():
    cons, mid = cb.credit_entry_fill(1.00, 1.10, 0.40, 0.50)
    assert mid == pytest.approx(0.60)
    assert cons == pytest.approx((1.00 + 0.4 * 0.05) - (0.50 - 0.4 * 0.05))  # 0.54
    assert cons < mid


def test_settle_put_credit_keep_partial_and_max_loss():
    # short 95 / long 90, credit 1.00
    assert cb.settle_put_credit(95, 90, 100, 1.0)[0] == pytest.approx(1.0 - FEES)
    assert cb.settle_put_credit(95, 90, 94.5, 1.0)[0] == pytest.approx(0.5 - FEES)
    pnl, ret = cb.settle_put_credit(95, 90, 50, 1.0)
    assert pnl == pytest.approx(1.0 - 5.0 - FEES)
    assert ret == pytest.approx(-1.0)  # lost the whole max risk


def test_model_prob_rises_as_short_strike_moves_out():
    near = cb.model_prob(100, 98, 1.0, 21, 0.20)
    far = cb.model_prob(100, 90, 0.2, 21, 0.20)
    assert 0.5 < near < far < 1.0


def _day(spot=100.0, bad_strike=None):
    rows = []
    for k in range(80, 101):
        bid, ask = max(0.05, (100 - k) * 0.0 + 0.02 * (k - 79)), None
        ask = bid + 0.05
        if k == bad_strike:
            bid, ask = 0.01, 1.50
        rows.append({"type": "P", "strike": float(k), "bid": bid, "ask": ask, "dte": 21,
                     "expiry": "2026-11-20", "iv": 0.3})
    return pd.DataFrame(rows)


def test_candidates_are_otm_and_dedupe_coarse_strikes():
    cands, bad = cb.candidates(_day(), 100.0)
    assert bad == 0 and cands
    assert all(c["short_strike"] < 100 and c["long_strike"] < c["short_strike"] for c in cands)
    keys = [(c["short_strike"], c["long_strike"]) for c in cands]
    assert len(keys) == len(set(keys))


def test_candidates_drop_bad_leg_quotes():
    _, bad = cb.candidates(_day(bad_strike=98), 100.0)
    assert bad > 0
