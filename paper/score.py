"""Calibration-first scoring. Model is the primary signal; P&L last and noisy."""

from __future__ import annotations

from typing import Optional

import numpy as np

from paper.models import read_forecasts, read_trades

MIN_N = 30


def _closed_with_forecasts() -> list[dict]:
    fcs = {f["forecast_id"]: f for f in read_forecasts()}
    out = []
    for t in read_trades():
        if t.get("status") != "closed":
            continue
        fc = fcs.get(t["forecast_id"])
        if not fc:
            continue
        row = {**fc, **t}
        row["outcome"] = 1 if float(t["pnl"]) > 0 else 0
        out.append(row)
    return out


def brier(probs: list[float], outcomes: list[int]) -> Optional[float]:
    if len(probs) < MIN_N:
        return None
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_table(probs: list[float], outcomes: list[int], n_bins: int = 10) -> list[dict]:
    if not probs:
        return []
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "pred": "", "actual": ""})
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "pred": round(float(p[mask].mean()), 3),
            "actual": round(float(y[mask].mean()), 3),
        })
    return rows


def auc(probs: list[float], outcomes: list[int]) -> Optional[float]:
    if len(probs) < MIN_N:
        return None
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    correct = 0.0
    for a in pos:
        correct += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(correct / (len(pos) * len(neg)))


def _print_cal_table(probs, outcomes, n):
    if n < MIN_N:
        print(f"   insufficient sample n={n}")
        return
    print(f"   {'bin':<12}{'n':>5}{'pred':>8}{'actual':>8}")
    for r in calibration_table(probs, outcomes):
        if r["n"] == 0:
            print(f"   {r['bin']:<12}{0:>5}{'':>8}{'':>8}")
        else:
            print(f"   {r['bin']:<12}{r['n']:>5}{r['pred']:>8.3f}{r['actual']:>8.3f}")


def report() -> None:
    closed = _closed_with_forecasts()
    forecasts = read_forecasts()
    trades = read_trades()

    print("\n=== PAPER REPORT (model calibration first) ===\n")

    taken = list(closed)
    n = len(taken)
    print(f"closed trades n={n} (need {MIN_N} for Brier/calibration/AUC)")

    if n:
        y = [c["outcome"] for c in taken]
        model_p = [float(c["model_prob_profit"]) for c in taken]

        bm = brier(model_p, y)
        print("\n1. Model Brier score (lower better) — primary metric")
        print(f"   model: {bm:.4f} n={n}" if bm is not None else f"   model: insufficient sample n={n}")

        print("\n2. Model calibration — predicted vs actual hit rate")
        _print_cal_table(model_p, y, n)

        print("\n3. Model discrimination (AUC)")
        am = auc(model_p, y)
        print(f"   model: {am:.3f} n={n}" if am is not None else f"   model: insufficient sample n={n}")

        human = [c for c in taken if c.get("source", "human") != "model"]
        if human:
            hp = [float(c["pred_prob_profit"]) for c in human]
            hy = [c["outcome"] for c in human]
            print(f"\n   (optional human-forecast subset n={len(human)})")
            bh = brier(hp, hy)
            print(f"   human Brier: {bh:.4f}" if bh is not None else "   human Brier: insufficient sample")
    else:
        print("\n1–3. Model Brier / calibration / AUC — insufficient sample n=0")

    print("\n4. Skip analysis")
    skips = [f for f in forecasts if f.get("decision") == "skip"]
    takes = [f for f in forecasts if f.get("decision") == "trade"]
    model_src = [f for f in forecasts if f.get("source") == "model"]
    print(f"   forecasts: trade={len(takes)} skip={len(skips)} (model-sourced={len(model_src)})")
    if taken:
        print(f"   taken hit rate: {np.mean([c['outcome'] for c in taken]):.1%} n={len(taken)}")

    print("\n5. Override analysis")
    ov = [c for c in taken if str(c.get("override", "")).lower() in ("true", "1", "yes")]
    ap = [c for c in taken if str(c.get("override", "")).lower() not in ("true", "1", "yes")]

    def _hit(rows):
        if not rows:
            return None
        return float(np.mean([r["outcome"] for r in rows]))

    print(f"   model-approved n={len(ap)} hit={_hit(ap)}")
    print(f"   override n={len(ov)} hit={_hit(ov)}")

    print("\n6. P&L (NOISY — do not use for decisions before n≈100)")
    if not taken:
        print("   insufficient sample n=0")
    else:
        pnls = [float(c["pnl"]) for c in taken]
        print(f"   total P&L     ${sum(pnls):+.2f} n={len(pnls)}")
        print(f"   mean P&L      ${np.mean(pnls):+.2f}")
        if len(taken) < MIN_N:
            print(f"   insufficient sample for inference n={len(taken)}")

    open_n = sum(1 for t in trades if t.get("status") == "open")
    print(f"\nopen trades: {open_n}")
