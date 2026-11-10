"""One-lot SPY put spreads: interpolation, lot P&L, book sizing."""

from __future__ import annotations

import pandas as pd
import pytest

import tradable_backtest as tb
from spread_eval import bs_price


def test_interp_quote_is_linear_and_overstates_a_convex_put():
    s = {"strike": 100.0, "bid": 2.00, "ask": 2.10}
    l = {"strike": 90.0, "bid": 0.50, "ask": 0.60}
    assert tb.interp_quote(s, l, 5.0) == pytest.approx((1.25, 1.35))
    p = lambda k: bs_price(105, k, 30 / 365, 0.04, 0.2, "P")  # noqa: E731
    lin = p(100) + 0.2 * (p(90) - p(100))
    assert lin > p(98)  # interpolated long put is dearer than the real one -> credit understated


def test_lot_pnl_and_risk():
    assert tb.lot_pnl(100, 1.0, 0.20, 105) == pytest.approx(20 - tb.FEES_LOT)      # keeps credit
    assert tb.lot_pnl(100, 1.0, 0.20, 99.5) == pytest.approx(-30 - tb.FEES_LOT)    # partial
    assert tb.lot_pnl(100, 1.0, 0.20, 80) == pytest.approx(-80 - tb.FEES_LOT)      # max loss
    assert tb.lot_risk(1.0, 0.20) == pytest.approx(80 + tb.FEES_LOT)
    assert tb.lot_pnl(100, 1.0, 0.20, 80) == pytest.approx(-tb.lot_risk(1.0, 0.20))


def _day(step):
    rows = []
    for k in [x * step for x in range(int(80 / step), int(101 / step))]:
        px = max(bs_price(100, k, 21 / 365, 0.04, 0.25, "P"), 0.03)
        rows.append({"type": "P", "strike": float(k), "bid": round(px * 0.97, 3), "ask": round(px * 1.03, 3),
                     "dte": 21, "expiry": "2026-11-20"})
    return pd.DataFrame(rows)


def test_spreads_for_day_skips_widths_wider_than_the_strike_gap():
    out = tb.spreads_for_day(_day(2.0), 100.0)
    assert {r["width"] for r in out} == {"adj", "$1"}  # $2 and $5 are not narrower than the $2 gap
    assert all(0 < r["credit"] < r["w"] for r in out)


def test_live_mode_attaches_actual_credit_and_interp_is_lower():
    out = [r for r in tb.spreads_for_day(_day(1.0), 100.0, coarse_gap=0.05) if "credit_actual" in r]
    assert out
    assert all(r["credit_actual"] >= r["credit"] - 1e-9 for r in out)


def test_weekly_book_sizes_lots_under_the_cap_and_skips_oversize():
    x = pd.DataFrame({"date": ["2025-01-06", "2025-01-07", "2025-01-13"],
                      "pnl": [10.0, 99.0, -80.0], "risk": [90.0, 90.0, 90.0]})
    b = tb.weekly_book(x)
    assert b["n"] == 2 and b["lots"] == 2           # floor(200 / 90), first entry of each week
    assert b["end"] == pytest.approx(5000 + 20 - 160)
    assert tb.weekly_book(x.assign(risk=900.0))["n"] == 0


def test_rescale_raises_credit_and_reprices():
    df = pd.DataFrame({"width": ["$1", "adj"], "w": [1.0, 8.0], "credit": [0.10, 1.0],
                       "short_strike": [100.0, 100.0], "settle_px": [105.0, 105.0]})
    g = tb.rescale(df, {"$1": 1.5})
    assert list(g["width"]) == ["$1"]
    assert g["credit"].iloc[0] == pytest.approx(0.15)
    assert g["pnl"].iloc[0] == pytest.approx(15 - tb.FEES_LOT)
