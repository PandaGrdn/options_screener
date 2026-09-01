"""Patch 1 migration note: model_version backfill touches ONLY that column,
only on rows a model actually scored (kelly regime, pred_prob_profit set),
and is idempotent. forecasts.csv stays append-only in every other respect."""

from __future__ import annotations

import pytest

import paper.models as models
import migrate_model_version as mig
from spread_eval import MODEL_VERSION_LEGACY


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for mod in (models, mig):
        monkeypatch.setattr(mod, "FORECASTS", data / "forecasts.csv")
        monkeypatch.setattr(mod, "TRADES", data / "trades.csv")
    monkeypatch.setattr(models, "ensure_data_dir", lambda: data.mkdir(exist_ok=True))
    return data


def _forecast_row(**overrides):
    row = {
        "forecast_id": "f1", "ts_utc": "t", "ticker": "NVDA", "horizon_days": 21,
        "direction": "up", "pred_move_pct": 0.0, "pred_vol_annual": 0.4,
        "pred_prob_profit": 0.4, "iv_at_forecast": 0.4, "iv_rank": 10,
        "rationale": "x" * 20, "decision": "skip", "skip_reason": "market default",
        "earnings_trade": "false", "source": "model", "regime": "kelly",
    }
    row.update(overrides)
    return row


def test_backfills_only_kelly_rows_with_a_model_probability(tmp_data):
    models.append_forecast(_forecast_row(forecast_id="a", regime="kelly", pred_prob_profit=0.4))
    models.append_forecast(_forecast_row(forecast_id="b", regime="", pred_prob_profit=0.5))  # v1 legacy
    models.append_forecast(_forecast_row(forecast_id="c", regime="kelly", pred_prob_profit=""))  # blind, no model

    n = mig.migrate_forecasts()
    assert n == 1

    rows = {r["forecast_id"]: r for r in models.read_forecasts()}
    assert rows["a"]["model_version"] == MODEL_VERSION_LEGACY
    assert rows["b"]["model_version"] == ""
    assert rows["c"]["model_version"] == ""


def test_forecast_migration_touches_only_model_version_column(tmp_data):
    row = _forecast_row(forecast_id="a")
    models.append_forecast(row)
    before = models.read_forecasts()[0]

    mig.migrate_forecasts()

    after = models.read_forecasts()[0]
    for k in row:
        assert after[k] == before[k], k
    assert after["model_version"] == MODEL_VERSION_LEGACY


def test_migration_idempotent(tmp_data):
    models.append_forecast(_forecast_row(forecast_id="a"))
    models.append_trade({
        "trade_id": "t1", "forecast_id": "a", "opened_utc": "t", "ticker": "NVDA",
        "structure": "call_debit_spread", "expiry": "2026-09-18", "dte_at_entry": 21,
        "long_strike": 1, "short_strike": 2, "entry_debit": 1, "entry_mid": 1,
        "contracts": 1, "capital_at_risk": 100, "model_prob_profit": 0.4,
        "model_ev": 0.1, "model_log_growth": 0.01, "tp_level": 2, "sl_level": 0.5,
        "time_stop_date": "2026-09-18", "status": "open",
    })

    n1f, n1t = mig.migrate_forecasts(), mig.migrate_trades()
    n2f, n2t = mig.migrate_forecasts(), mig.migrate_trades()
    assert (n1f, n1t) == (1, 1)
    assert (n2f, n2t) == (0, 0)
