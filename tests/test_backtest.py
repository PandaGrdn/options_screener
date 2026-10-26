"""DoltHub importer conversion/split handling + hold-to-expiry backtest math."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import backtest
import import_dolthub as imp
from spread_eval import FEE_PER_CONTRACT

FEES = 4 * FEE_PER_CONTRACT / 100.0


def test_unadjusted_closes_undo_split_before_ex_date():
    und = pd.DataFrame({"ticker": ["NFLX"] * 3 + ["NVDA"],
                        "date": ["2025-11-14", "2025-11-17", "2025-11-18", "2025-11-14"],
                        "close": [115.0, 112.0, 110.0, 190.0]})
    splits = pd.DataFrame({"ticker": ["NFLX"], "date": ["2025-11-17"], "ratio": [10.0]})
    out = imp.unadjusted_closes(und, splits)
    assert out["close_raw"].tolist() == [1150.0, 112.0, 110.0, 190.0]


def test_to_chain_rows_matches_live_schema_and_solves_iv():
    raw = pd.DataFrame([
        {"expiration": "2026-11-06", "strike": "210.00", "call_put": "Call", "bid": "6.90", "ask": "7.05", "vol": "0.3820"},
        {"expiration": "2026-11-06", "strike": "200.00", "call_put": "Put", "bid": "2.10", "ask": "2.20", "vol": "0.4000"},
        {"expiration": "2026-11-06", "strike": "300.00", "call_put": "Call", "bid": "0.00", "ask": "0.00", "vol": None},
    ])
    rows = imp.to_chain_rows(raw, "NVDA", "2026-10-23", 210.96)
    assert list(rows.columns) == imp.HIST_SCHEMA
    assert len(rows) == 2  # ask<=0 dropped, like snapshot.py
    c = rows[rows["type"] == "C"].iloc[0]
    assert c["dte"] == 14 and c["mid"] == pytest.approx(6.975)
    assert c["iv"] == pytest.approx(0.382, abs=0.01)  # own solve ~ vendor IV
    assert rows["volume"].isna().all()


def test_is_centred_rejects_split_mismatch():
    rows = pd.DataFrame({"strike": [1100.0, 1150.0, 1200.0]})
    assert imp.is_centred(rows, 1157.5)
    assert not imp.is_centred(rows, 115.75)  # split-adjusted spot vs as-traded strikes


def test_settle_is_capped_intrinsic_minus_debit_and_fees():
    assert backtest.settle(100, 105, 90, 2.0)[0] == pytest.approx(-2.0 - FEES)
    assert backtest.settle(100, 105, 103, 2.0)[0] == pytest.approx(1.0 - FEES)
    pnl, ret = backtest.settle(100, 105, 200, 2.0)
    assert pnl == pytest.approx(3.0 - FEES) and ret == pytest.approx(pnl / (2.0 + FEES))


def test_expiry_model_prob_matches_its_own_terminal_samples():
    m = backtest.expiry_model(100, 100, 105, 21, 2.0, 0.40, n=200_000)
    # closed-form P(S_T > breakeven) vs fraction of sampled terminal pnl > 0
    T = 21 / 365
    z = np.random.default_rng(7).standard_normal(200_000)
    st = 100 * np.exp((0.05 - 0.08) * T + 0.40 * math.sqrt(T) * z)
    mc = float(((np.clip(st - 100, 0, 5) - 2.0 - FEES) > 0).mean())
    assert m["prob"] == pytest.approx(mc, abs=0.005)


def test_expiry_model_zero_when_breakeven_above_short_strike():
    assert backtest.expiry_model(100, 100, 102, 21, 2.5, 0.4)["prob"] == 0.0


def test_settle_session_rolls_holiday_expiry_back():
    assert backtest.settle_session("2026-04-03") == "2026-04-02"  # Good Friday
    assert backtest.settle_session("2026-11-06") == "2026-11-06"


def test_parity_gap_flags_quotes_not_taken_at_the_close():
    from spread_eval import bs_price

    def day(spot_quoted, spot_stored):
        rows = []
        for k in (98.0, 100.0, 102.0):
            for kind in ("C", "P"):
                m = bs_price(spot_quoted, k, 21 / 365, backtest.RISK_FREE, 0.4, kind)
                rows.append({"expiry": "2026-11-13", "strike": k, "type": kind, "mid": m,
                             "dte": 21, "spot": spot_stored, "moneyness": k / spot_stored - 1})
        return pd.DataFrame(rows)

    assert backtest.parity_gap_pct(day(100.0, 100.0)) == pytest.approx(0.0, abs=1e-6)
    assert backtest.parity_gap_pct(day(101.0, 100.0)) == pytest.approx(1.0, abs=1e-6)
