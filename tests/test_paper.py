"""Tests for paper trading constraints — append-only forecasts, fills, exits, sizing."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

import paper.models as models
from paper.exit import check_exit, apply_close
from paper.mark import price_spread
from paper.score import brier, calibration_table, auc, MIN_N
from spread_eval import (
    fill_toward_mid, debit_entry_fill, credit_exit_fill, size_position,
    PORTFOLIO, MAX_DEPLOYED_PCT,
)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(models, "FORECASTS", data / "forecasts.csv")
    monkeypatch.setattr(models, "TRADES", data / "trades.csv")
    monkeypatch.setattr(models, "MARKS", data / "marks.csv")
    monkeypatch.setattr(models, "ensure_data_dir", lambda: data.mkdir(exist_ok=True))
    return data


def test_forecasts_append_only_never_mutates_prior_rows(tmp_data):
    models.append_forecast({
        "forecast_id": "a", "ts_utc": "t1", "ticker": "NVDA", "horizon_days": 30,
        "direction": "up", "pred_move_pct": 0.05, "pred_vol_annual": 0.4,
        "pred_prob_profit": 0.45, "iv_at_forecast": 0.4, "iv_rank": 20,
        "rationale": "x" * 20, "decision": "skip", "skip_reason": "too rich",
        "earnings_trade": "false",
    })
    before = (tmp_data / "forecasts.csv").read_text()
    first_lines = before.strip().splitlines()
    assert len(first_lines) == 2  # header + 1

    models.append_forecast({
        "forecast_id": "b", "ts_utc": "t2", "ticker": "NVDA", "horizon_days": 30,
        "direction": "up", "pred_move_pct": 0.03, "pred_vol_annual": 0.35,
        "pred_prob_profit": 0.5, "iv_at_forecast": 0.38, "iv_rank": 25,
        "rationale": "y" * 20, "decision": "trade", "skip_reason": "",
        "earnings_trade": "false",
    })
    after = (tmp_data / "forecasts.csv").read_text()
    # prior content (header + first row) must be an unchanged prefix
    assert after.startswith(before.rstrip("\n") + "\n") or after.startswith(before)
    rows = models.read_forecasts()
    assert len(rows) == 2
    assert rows[0]["forecast_id"] == "a"
    assert rows[0]["pred_prob_profit"] == "0.45"
    # no edit helper exists
    assert not hasattr(models, "edit_forecast")
    assert not hasattr(models, "update_forecast")
    assert not hasattr(models, "delete_forecast")


def test_fill_model_entry_worse_than_mid_exit_worse_than_mid():
    # long call: bid 1.00 ask 1.20 mid 1.10
    debit, mid = debit_entry_fill(1.20, 1.00)
    assert mid == pytest.approx(1.10)
    assert debit > mid  # pay more than mid
    assert debit < 1.20  # but better than full ask
    assert debit == pytest.approx(fill_toward_mid(1.20, 1.10))

    # exit: receive less than mid
    credit, mid_x = credit_exit_fill(1.00, 1.20)
    assert mid_x == pytest.approx(1.10)
    assert credit < mid_x
    assert credit > 1.00


def test_stop_fill_worse_than_normal_exit():
    normal, _ = credit_exit_fill(1.00, 1.20, stop=False)
    stop, _ = credit_exit_fill(1.00, 1.20, stop=True)
    assert stop < normal


def test_exit_triggers_tp_sl_time():
    trade = {
        "tp_level": 2.0,
        "sl_level": 0.5,
        "time_stop_date": "2026-09-01",
        "expiry": "2026-09-18",
        "entry_debit": 1.0,
        "contracts": 1,
        "capital_at_risk": 100,
        "short_strike": "",
    }
    assert check_exit(trade, 2.05, "2026-08-20") == "tp"
    assert check_exit(trade, 0.40, "2026-08-20") == "sl"
    assert check_exit(trade, 1.0, "2026-09-01") == "time_stop"
    assert check_exit(trade, 1.0, "2026-08-20") is None

    closed = apply_close(trade, 2.05, "tp")
    assert closed["status"] == "closed"
    assert closed["exit_reason"] == "tp"
    # immutable exit rules
    assert closed["tp_level"] == trade["tp_level"]
    assert closed["sl_level"] == trade["sl_level"]
    assert closed["time_stop_date"] == trade["time_stop_date"]


def test_position_sizing_respects_caps():
    n, cap = size_position(1.0, already_deployed=0.0)  # $100+ fees per contract
    assert n >= 1
    assert cap <= PORTFOLIO * 0.04 + 1e-6

    # already near 20% deployed
    n2, cap2 = size_position(1.0, already_deployed=PORTFOLIO * MAX_DEPLOYED_PCT - 50)
    assert n2 * (1.0 * 100) <= PORTFOLIO * MAX_DEPLOYED_PCT - (PORTFOLIO * MAX_DEPLOYED_PCT - 50) + 200
    n3, _ = size_position(1.0, already_deployed=PORTFOLIO * MAX_DEPLOYED_PCT)
    assert n3 == 0


def test_missing_strike_flag_not_interpolate():
    import pandas as pd
    day = pd.DataFrame([
        {"expiry": "2026-09-18", "type": "C", "strike": 200.0, "bid": 10.0, "ask": 10.4, "mid": 10.2},
    ])
    trade = {"expiry": "2026-09-18", "long_strike": 210.0, "short_strike": ""}
    cons, mid, missing = price_spread(day, trade)
    assert missing is True
    assert cons is None and mid is None


def test_brier_and_calibration_synthetic():
    rng = np.random.default_rng(0)
    # well-calibrated: outcome ~ Bern(p)
    probs = list(rng.uniform(0.2, 0.8, size=80))
    outcomes = [int(rng.random() < p) for p in probs]
    # temporarily lower threshold via direct formula
    p = np.asarray(probs)
    y = np.asarray(outcomes, dtype=float)
    score = float(np.mean((p - y) ** 2))
    assert 0.1 < score < 0.3  # calibrated-ish

    # perfect forecasts
    perfect_p = [0.0] * 40 + [1.0] * 40
    perfect_y = [0] * 40 + [1] * 40
    assert brier(perfect_p, perfect_y) == pytest.approx(0.0)

    table = calibration_table(perfect_p, perfect_y)
    # edge bins should match
    assert table[0]["actual"] == 0.0
    assert table[-1]["actual"] == 1.0
    assert auc(perfect_p, perfect_y) == pytest.approx(1.0)

    assert brier([0.5] * 10, [0, 1] * 5) is None  # below MIN_N


def test_dashboard_html_insufficient_sample(tmp_data, monkeypatch):
    import paper.dashboard as dash
    from paper.dashboard import write_dashboard

    monkeypatch.setattr(dash, "MARKS", tmp_data / "marks.csv")
    models.append_forecast({
        "forecast_id": "f1", "ts_utc": "2026-08-19T00:00:00+00:00", "ticker": "NVDA",
        "horizon_days": 30, "direction": "up", "pred_move_pct": 0.0,
        "pred_vol_annual": 0.4, "pred_prob_profit": 0.36, "iv_at_forecast": 0.3,
        "iv_rank": "", "rationale": "auto cron model=TRADE p=0.360 vol=0.40",
        "decision": "trade", "skip_reason": "", "earnings_trade": "false", "source": "model",
    })
    models.append_trade({
        "trade_id": "t1", "forecast_id": "f1", "opened_utc": "2026-08-19T00:00:00+00:00",
        "ticker": "NVDA", "structure": "call_debit_spread", "expiry": "2026-09-18",
        "dte_at_entry": 30, "long_strike": 180, "short_strike": 185,
        "entry_debit": 1.2, "entry_mid": 1.1, "contracts": 1, "capital_at_risk": 120,
        "model_prob_profit": 0.36, "model_ev": 0.05, "model_log_growth": -3.0,
        "tp_level": 2.4, "sl_level": 0.6, "time_stop_date": "2026-09-18",
        "status": "closed", "closed_utc": "2026-08-19T00:00:03+00:00",
        "exit_credit": 0.3, "exit_reason": "sl", "pnl": -90, "return_pct": -0.75,
        "override": "false", "override_reason": "", "earnings_trade": "false",
    })
    out = tmp_data / "dashboard.html"
    write_dashboard(out)
    text = out.read_text()
    assert "Too early to score the model" in text
    assert "NVDA" in text
    assert "n=1" in text
    assert "same-day SL" in text
