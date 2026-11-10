"""Regime backtest: HMM has no lookahead, strategy rules, call spread grid."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import regime_backtest as rb


def _two_regime(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 0.006, 400), rng.normal(0, 0.03, 150),
                           rng.normal(0, 0.006, 300)])


def test_hmm_separates_calm_from_turbulent():
    r = _two_regime()
    p = rb.hmm_fit(r)
    assert p["sig"][0] < p["sig"][1]
    calm = rb.hmm_filter(r, p)[:, 0]
    assert calm[100:380].mean() > 0.9
    assert calm[430:540].mean() < 0.1


def test_hmm_filter_ignores_future_returns():
    r = _two_regime()
    p = rb.hmm_fit(r)
    a = rb.hmm_filter(r, p)[:500]
    r2 = r.copy()
    r2[500:] = 0.2
    assert np.allclose(a, rb.hmm_filter(r2, p)[:500])


def test_walk_forward_calm_has_no_lookahead():
    r = _two_regime()
    dates = pd.bdate_range("2021-01-01", periods=len(r)).strftime("%Y-%m-%d").tolist()
    a = rb.walk_forward_calm(dates, r)
    r2 = r.copy()
    r2[600:] = 0.2  # rewrite the future
    b = rb.walk_forward_calm(dates, r2)
    early = [d for d in a.index if d < dates[600]]
    assert len(early) > 100
    assert np.allclose(a[early], b[early])
    assert a.index.min() == dates[rb.HMM_MIN_OBS]


def _x(**kw):
    base = dict(trend_up=1.0, spy_trend_up=1.0, ts_ratio=0.9, p_calm=0.9, atm_iv=0.30, rv20=0.25)
    return SimpleNamespace(**{**base, **kw})


def test_decide_rules():
    assert rb.decide("always_put", _x()) == "put_credit"
    assert rb.decide("always_call", _x()) == "call_debit"
    assert rb.decide("trend", _x(trend_up=0.0)) == "cash"
    assert rb.decide("term", _x(ts_ratio=1.05)) == "cash"
    assert rb.decide("vrp", _x(atm_iv=0.2)) == "cash"
    assert rb.decide("hmm", _x(p_calm=0.3)) == "cash"
    # switcher: stress -> cash; calm uptrend rich vol -> sell puts; cheap vol -> buy calls
    assert rb.decide("switch_rules", _x()) == "put_credit"
    assert rb.decide("switch_rules", _x(atm_iv=0.2)) == "call_debit"
    assert rb.decide("switch_rules", _x(ts_ratio=1.1)) == "cash"
    assert rb.decide("switch_rules", _x(spy_trend_up=0.0)) == "cash"
    assert rb.decide("switch_rules", _x(trend_up=0.0)) == "cash"
    assert rb.decide("switch_hmm", _x(p_calm=0.2)) == "cash"
    assert rb.decide("switch_hmm", _x(ts_ratio=1.1)) == "put_credit"  # hmm variant ignores VIX curve


def test_decide_missing_inputs_go_to_cash():
    nan = float("nan")
    for s in ("trend", "term", "vrp", "hmm", "switch_rules", "switch_hmm"):
        assert rb.decide(s, _x(trend_up=nan, ts_ratio=nan, p_calm=nan, atm_iv=nan, spy_trend_up=nan)) == "cash"


def test_call_candidates_are_debit_spreads_inside_width():
    rows = [{"type": "C", "strike": float(k), "bid": max(100 - k, 0) + 1.0, "ask": max(100 - k, 0) + 1.1,
             "dte": 21, "expiry": "2026-11-20", "iv": 0.3} for k in range(90, 115)]
    c = rb.call_candidates(pd.DataFrame(rows), 100.0)
    assert c
    for x in c:
        assert x["short_strike"] > x["long_strike"]
        assert 0 < x["debit"] < x["short_strike"] - x["long_strike"]
        assert x["debit"] > x["debit_mid"]  # conservative fill pays more than mid


def test_stats_month_clustered():
    e = pd.DataFrame({"date": ["2025-01-02", "2025-01-09", "2025-02-03", "2025-03-03"],
                      "ret": [0.1, 0.3, -0.2, 0.2], "win": [1, 1, 0, 1], "pnl": [1, 3, -2, 2],
                      "pnl_mid": [1, 3, -2, 2], "risk": [10, 10, 10, 10], "kind": ["put_credit"] * 4})
    s = rb.stats(e, 8)
    assert s["n"] == 4 and s["traded"] == 0.5
    assert s["ret_w"] == pytest.approx(0.1)
    assert s["worst_month"] == pytest.approx(-0.2)
