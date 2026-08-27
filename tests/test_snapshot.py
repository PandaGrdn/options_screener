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
