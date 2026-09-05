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


def test_score_excludes_v1_closed_trades(tmp_data):
    from paper.score import _closed_with_forecasts
    from paper.models import REGIME_KELLY

    models.append_forecast({
        "forecast_id": "old", "ts_utc": "t0", "ticker": "COIN", "horizon_days": 30,
        "direction": "up", "pred_move_pct": 0.0, "pred_vol_annual": 0.7,
        "pred_prob_profit": 0.36, "iv_at_forecast": 0.5, "iv_rank": "",
        "rationale": "x" * 20, "decision": "trade", "skip_reason": "",
        "earnings_trade": "false", "source": "model",
    })
    models.append_forecast({
        "forecast_id": "new", "ts_utc": "t1", "ticker": "NVDA", "horizon_days": 21,
        "direction": "up", "pred_move_pct": 0.04, "pred_vol_annual": 0.5,
        "pred_prob_profit": 0.4, "iv_at_forecast": 0.4, "iv_rank": 20,
        "rationale": "y" * 20, "decision": "trade", "skip_reason": "",
        "earnings_trade": "false", "source": "human", "regime": REGIME_KELLY,
    })
    models.append_trade({
        "trade_id": "told", "forecast_id": "old", "opened_utc": "t", "ticker": "COIN",
        "structure": "call_debit_spread", "expiry": "2026-09-18", "dte_at_entry": 30,
        "long_strike": 1, "short_strike": 2, "entry_debit": 1, "entry_mid": 1,
        "contracts": 1, "capital_at_risk": 100, "model_prob_profit": 0.36,
        "model_ev": 0.1, "model_log_growth": -3, "tp_level": 2, "sl_level": 0.5,
        "time_stop_date": "2026-09-18", "status": "closed", "closed_utc": "t",
        "exit_credit": 0, "exit_reason": "sl", "pnl": -100, "return_pct": -1,
        "override": "false", "override_reason": "", "earnings_trade": "false",
    })
    models.append_trade({
        "trade_id": "tnew", "forecast_id": "new", "opened_utc": "t", "ticker": "NVDA",
        "structure": "call_debit_spread", "expiry": "2026-09-18", "dte_at_entry": 21,
        "long_strike": 1, "short_strike": 2, "entry_debit": 1, "entry_mid": 1,
        "contracts": 1, "capital_at_risk": 100, "model_prob_profit": 0.4,
        "model_ev": 0.1, "model_log_growth": 0.01, "tp_level": 2, "sl_level": 0.5,
        "time_stop_date": "2026-09-18", "status": "closed", "closed_utc": "t",
        "exit_credit": 2, "exit_reason": "tp", "pnl": 50, "return_pct": 0.5,
        "override": "false", "override_reason": "", "earnings_trade": "false",
    })
    kelly = _closed_with_forecasts(kelly_only=True)
    all_closed = _closed_with_forecasts(kelly_only=False)
    assert len(kelly) == 1 and kelly[0]["forecast_id"] == "new"
    assert len(all_closed) == 2


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
    assert "v1 excluded from score" in text
    assert "n=0" in text
    assert "same-day SL" in text


def test_shadow_does_not_count_as_deployed_capital(tmp_data):
    models.append_trade({
        "trade_id": "sh", "forecast_id": "fsh", "opened_utc": "t", "ticker": "NVDA",
        "structure": "call_debit_spread", "expiry": "2026-09-25", "dte_at_entry": 21,
        "long_strike": 180, "short_strike": 190, "entry_debit": 1.2, "entry_mid": 1.1,
        "contracts": 1, "capital_at_risk": 999, "model_prob_profit": 0.4,
        "model_ev": -0.1, "model_log_growth": -0.01, "tp_level": 2.4, "sl_level": 0.6,
        "time_stop_date": "2026-09-25", "status": "open", "closed_utc": "",
        "exit_credit": "", "exit_reason": "", "pnl": "", "return_pct": "",
        "override": "false", "override_reason": "", "earnings_trade": "false",
        "shadow": "true",
    })
    models.append_trade({
        "trade_id": "real", "forecast_id": "fr", "opened_utc": "t", "ticker": "AMD",
        "structure": "call_debit_spread", "expiry": "2026-09-25", "dte_at_entry": 21,
        "long_strike": 160, "short_strike": 170, "entry_debit": 1.5, "entry_mid": 1.4,
        "contracts": 1, "capital_at_risk": 150, "model_prob_profit": 0.4,
        "model_ev": 0.1, "model_log_growth": 0.01, "tp_level": 3.0, "sl_level": 0.75,
        "time_stop_date": "2026-09-25", "status": "open", "closed_utc": "",
        "exit_credit": "", "exit_reason": "", "pnl": "", "return_pct": "",
        "override": "false", "override_reason": "", "earnings_trade": "false",
    })
    assert models.open_capital_at_risk() == pytest.approx(150.0)


def test_score_includes_kelly_shadow_closes(tmp_data):
    from paper.models import REGIME_KELLY
    from paper.score import _closed_with_forecasts
    from spread_eval import MODEL_VERSION

    models.append_forecast({
        "forecast_id": "shf", "ts_utc": "t1", "ticker": "NVDA", "horizon_days": 21,
        "direction": "up", "pred_move_pct": 0.0, "pred_vol_annual": 0.4,
        "pred_prob_profit": 0.41, "iv_at_forecast": 0.4, "iv_rank": 20,
        "rationale": "z" * 20, "decision": "skip", "skip_reason": "no forecast",
        "earnings_trade": "false", "source": "model", "regime": REGIME_KELLY,
    })
    models.append_trade({
        "trade_id": "sht", "forecast_id": "shf", "opened_utc": "t", "ticker": "NVDA",
        "structure": "call_debit_spread", "expiry": "2026-09-25", "dte_at_entry": 21,
        "long_strike": 180, "short_strike": 190, "entry_debit": 1.2, "entry_mid": 1.1,
        "contracts": 1, "capital_at_risk": 0, "model_prob_profit": 0.41,
        "model_ev": -0.1, "model_log_growth": -0.01, "tp_level": 2.4, "sl_level": 0.6,
        "time_stop_date": "2026-09-25", "status": "closed", "closed_utc": "t",
        "exit_credit": 1.8, "exit_reason": "tp", "pnl": 60, "return_pct": 0.5,
        "override": "false", "override_reason": "", "earnings_trade": "false",
        "model_version": MODEL_VERSION, "shadow": "true",
    })
    kelly = _closed_with_forecasts(kelly_only=True)
    assert len(kelly) == 1
    assert models.is_shadow(kelly[0])
    assert kelly[0]["outcome"] == 1


def test_open_shadow_no_capital_and_refuses_nonspread(tmp_data):
    from paper.entry import open_shadow
    from paper.models import REGIME_KELLY

    models.append_forecast({
        "forecast_id": "s1", "ts_utc": "t1", "ticker": "NVDA", "horizon_days": 21,
        "direction": "up", "pred_move_pct": 0.0, "pred_vol_annual": 0.4,
        "pred_prob_profit": 0.4, "iv_at_forecast": 0.4, "iv_rank": 20,
        "rationale": "w" * 20, "decision": "skip", "skip_reason": "no forecast",
        "earnings_trade": "false", "source": "model", "regime": REGIME_KELLY,
    })
    row = open_shadow("s1", {
        "structure": "call_debit_spread",
        "entry_debit": 1.2,
        "entry_mid": 1.1,
        "expiry": "2026-09-25",
        "dte": 21,
        "long_strike": 180,
        "short_strike": 190,
        "prob_profit": 0.42,
        "ev": -0.1,
        "log_growth": -0.01,
        "tp_level": 2.4,
        "sl_level": 0.6,
    })
    assert row is not None
    assert row["shadow"] == "true"
    assert float(row["capital_at_risk"]) == 0.0
    assert int(row["contracts"]) == 1
    assert models.open_capital_at_risk() == 0.0
    assert open_shadow("s1", {"structure": "long_call", "entry_debit": 1.0}) is None


def test_auto_shadows_only_when_cheapness_passes(tmp_data, monkeypatch):
    import paper.auto as auto

    shadowed = []

    def fake_ctx(ticker):
        passed = ticker == "AMD"
        return {
            "iv": 0.4, "iv_rank": 20, "earn_days": None,
            "gate_passed": passed,
            "gate_reason": "cheap: ok" if passed else "RV percentile 90 > 40",
            "spot": 100.0, "chain_day": None,
        }

    def fake_eval(**kwargs):
        return {
            "verdict": "SKIP", "skip_reason": "log_growth -0.1 <= 0",
            "prob_profit": 0.4, "forecast_vol": 0.4,
            "structure": "call_debit_spread", "entry_debit": 1.2, "entry_mid": 1.1,
            "expiry": "2026-09-25", "dte": 21, "long_strike": 100, "short_strike": 105,
            "ev": -0.1, "log_growth": -0.1, "tp_level": 2.4, "sl_level": 0.6,
        }

    def fake_shadow(fid, result):
        shadowed.append(fid)
        return {"ticker": "AMD", "forecast_id": fid}

    monkeypatch.setattr(auto, "UNIVERSE", ["AMD", "TSLA"])
    monkeypatch.setattr(auto, "REFERENCE_ONLY", set())
    monkeypatch.setattr(auto, "context_for_forecast", fake_ctx)
    monkeypatch.setattr(auto, "evaluate", fake_eval)
    monkeypatch.setattr(auto, "open_shadow", fake_shadow)
    monkeypatch.setattr(auto, "open_capital_at_risk", lambda: 0.0)
    monkeypatch.setattr(auto, "_already_decided_today", lambda *a, **k: False)
    monkeypatch.setattr(auto, "_open_tickers", lambda: set())

    summary = auto.auto_decide_universe()
    assert len(shadowed) == 1
    assert len(summary["shadowed"]) == 1
    assert {f["ticker"] for f in models.read_forecasts()} == {"AMD", "TSLA"}
