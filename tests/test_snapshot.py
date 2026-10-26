"""Snapshot session dates — underlyings are T-1, chains are same-session."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from snapshot import index_session_date, previous_session_date, session_date


def test_session_date_uses_new_york_not_utc():
    now = dt.datetime(2026, 8, 27, 21, 55, tzinfo=dt.timezone.utc)
    assert session_date(now) == dt.date(2026, 8, 27)
    late = dt.datetime(2026, 8, 28, 2, 0, tzinfo=dt.timezone.utc)
    assert session_date(late) == dt.date(2026, 8, 27)


def test_index_session_date_converts_eastern_tz():
    idx = pd.Timestamp("2026-08-27 16:00:00", tz="America/New_York")
    assert index_session_date(idx) == dt.date(2026, 8, 27)
    utc = pd.Timestamp("2026-08-27 20:00:00", tz="UTC")
    assert index_session_date(utc) == dt.date(2026, 8, 27)


def test_previous_session_skips_weekend():
    thursday = dt.date(2026, 8, 27)
    assert previous_session_date(thursday) == dt.date(2026, 8, 26)
    monday = dt.date(2026, 8, 24)
    assert previous_session_date(monday) == dt.date(2026, 8, 21)
    sunday = dt.date(2026, 8, 23)
    assert previous_session_date(sunday) == dt.date(2026, 8, 21)


def test_previous_session_skips_nyse_holidays():
    # Tuesday after Labor Day 2026-09-07 -> Friday 09-04
    assert previous_session_date(dt.date(2026, 9, 8)) == dt.date(2026, 9, 4)
    # Friday after Thanksgiving 2026-11-26 -> Wednesday 11-25
    assert previous_session_date(dt.date(2026, 11, 27)) == dt.date(2026, 11, 25)


# --- closed-market guard -----------------------------------------------------

from snapshot import chain_capture_status  # noqa: E402


def _utc(*a):
    return dt.datetime(*a, tzinfo=dt.timezone.utc)


def test_capture_refused_on_holiday():
    ok, why = chain_capture_status(_utc(2026, 9, 7, 21, 55))  # Labor Day
    assert not ok and "not an NYSE session" in why


def test_capture_refused_before_close_buffer():
    assert not chain_capture_status(_utc(2026, 10, 23, 19, 0))[0]   # 3:00pm ET
    assert not chain_capture_status(_utc(2026, 10, 23, 20, 10))[0]  # 4:10pm ET (ETF options open)
    assert chain_capture_status(_utc(2026, 10, 23, 20, 20))[0]      # 4:20pm ET


def test_capture_refused_overnight_before_session():
    # The 2026-08-28 05:50 UTC run: 1:50am ET stored 08-27 quotes as "08-28".
    assert not chain_capture_status(_utc(2026, 8, 28, 5, 50))[0]


def test_capture_allowed_after_early_close():
    # Day after Thanksgiving closes 1:00pm ET
    assert chain_capture_status(_utc(2026, 11, 27, 18, 20))[0]


# --- snapshot_chains with a fake yfinance ------------------------------------

import sys  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import snapshot  # noqa: E402
from paper import read_chain_history  # noqa: E402

NOW = _utc(2026, 10, 23, 20, 30)  # Fri 4:30pm ET
EXPIRIES = ("2026-10-30", "2026-11-06", "2026-11-13", "2026-11-20", "2026-12-18")


class _Chain:
    def __init__(self, shift):
        strikes = np.arange(80.0, 135.0, 5.0)
        base = pd.DataFrame({
            "strike": strikes,
            "volume": 10, "openInterest": 100, "impliedVolatility": 0.4,
        })
        self.calls = base.assign(bid=np.maximum(100 - strikes, 0) + 1.0 + shift,
                                 ask=np.maximum(100 - strikes, 0) + 1.2 + shift)
        self.puts = base.assign(bid=np.maximum(strikes - 100, 0) + 1.0 + shift,
                                ask=np.maximum(strikes - 100, 0) + 1.2 + shift)


def _fake_yf(monkeypatch, shift=0.0):
    calls = []

    class Ticker:
        options = EXPIRIES

        def __init__(self, sym):
            calls.append(sym)

        def history(self, period="1d", **kw):
            return pd.DataFrame({"Close": [100.0]})

        def option_chain(self, e):
            return _Chain(shift)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    return calls


def test_snapshot_captures_held_legs_outside_band_and_target(tmp_path, monkeypatch):
    _fake_yf(monkeypatch)
    root = tmp_path / "chain_history"
    # 10-30 is 7 DTE (not nearest any target); 125/130 are +25%/+30% (outside ±10%)
    legs = {"NVDA": {"2026-10-30": {125.0, 130.0}}}
    n = snapshot.snapshot_chains(["NVDA", "AMD"], path=root, trade_legs=legs, now=NOW)
    assert n > 0
    assert (root / "2026-10.csv").exists()
    df = read_chain_history(root)
    held = df[(df.ticker == "NVDA") & (df.expiry == "2026-10-30") & (df.type == "C")]
    assert {125.0, 130.0} <= set(held.strike)
    # untouched name keeps the normal surface: no 10-30, nothing outside the band
    amd = df[df.ticker == "AMD"]
    assert "2026-10-30" not in set(amd.expiry)
    assert amd.moneyness.abs().max() <= 0.10 + 1e-9


def test_snapshot_skips_holiday_without_fetching(tmp_path, monkeypatch):
    calls = _fake_yf(monkeypatch)
    n = snapshot.snapshot_chains(["NVDA"], path=tmp_path / "ch", trade_legs={},
                                 now=_utc(2026, 9, 7, 21, 55))
    assert n == 0 and calls == []
    assert not (tmp_path / "ch").exists()


def test_snapshot_refuses_frozen_chain(tmp_path, monkeypatch):
    root = tmp_path / "ch"
    tickers = ["NVDA", "AMD", "TSLA"]
    monkeypatch.setattr(snapshot, "STALE_MIN_OVERLAP", 50)
    _fake_yf(monkeypatch)
    snapshot.snapshot_chains(tickers, path=root, trade_legs={}, now=_utc(2026, 10, 22, 20, 30))
    with pytest.raises(RuntimeError, match="identical"):
        snapshot.snapshot_chains(tickers, path=root, trade_legs={}, now=NOW)
    _fake_yf(monkeypatch, shift=0.05)  # quotes moved -> a real session
    assert snapshot.snapshot_chains(tickers, path=root, trade_legs={}, now=NOW) > 0


def test_open_trade_legs_includes_shadows_excludes_closed():
    trades = [
        {"ticker": "nvda", "status": "open", "shadow": "true", "expiry": "2026-11-06",
         "long_strike": "245.0", "short_strike": "255.0"},
        {"ticker": "AMD", "status": "closed", "expiry": "2026-11-06",
         "long_strike": "620", "short_strike": "665"},
        {"ticker": "NVDA", "status": "open", "expiry": "2026-11-13",
         "long_strike": "250", "short_strike": ""},
    ]
    assert snapshot.open_trade_legs(trades) == {
        "NVDA": {"2026-11-06": {245.0, 255.0}, "2026-11-13": {250.0}},
    }


# --- partition + rv20_pctile repair ------------------------------------------

def test_partition_splits_by_month_and_drops_holiday(tmp_path):
    from partition_chain_history import partition
    from snapshot import CHAIN_SCHEMA

    legacy = tmp_path / "chain_history.csv"
    rows = [{c: 1 for c in CHAIN_SCHEMA} | {"date": d} for d in
            ("2026-08-31", "2026-09-04", "2026-09-07", "2026-10-01")]
    pd.DataFrame(rows)[CHAIN_SCHEMA].to_csv(legacy, index=False)
    root = tmp_path / "chain_history"
    assert partition(legacy, root) == 3
    assert not legacy.exists()
    assert sorted(p.name for p in root.glob("*.csv")) == ["2026-08.csv", "2026-09.csv", "2026-10.csv"]
    assert "2026-09-07" not in set(read_chain_history(root)["date"].astype(str))
    assert partition(legacy, root) == 0  # idempotent


def test_repair_rv_pctile_fills_only_nan_and_is_idempotent(tmp_path):
    from backfill_underlyings import repair_rv_pctile
    from snapshot import UNDERLYING_SCHEMA

    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: 1.0 for c in UNDERLYING_SCHEMA}, index=range(n))
    df["date"] = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    df["ticker"] = "NVDA"
    df["rv20_yz"] = rng.uniform(0.2, 0.6, n)
    df["rv20_pctile"] = np.nan
    df.loc[:279, "rv20_pctile"] = 12.34  # pre-existing values must not change
    path = tmp_path / "u.csv"
    df[UNDERLYING_SCHEMA].to_csv(path, index=False)

    assert repair_rv_pctile(path) == 20
    out = pd.read_csv(path)
    assert (out.loc[:279, "rv20_pctile"] == 12.34).all()
    assert out["rv20_pctile"].notna().all()
    assert repair_rv_pctile(path) == 0
