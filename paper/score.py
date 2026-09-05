"""Calibration-first scoring. Model is the primary signal; P&L last and noisy."""

from __future__ import annotations

from typing import Optional

import numpy as np

from paper.models import (
    read_forecasts, read_trades, is_kelly_regime, is_shadow,
    REGIME_KELLY, REGIME_V1, HYPOTHESES,
)
from spread_eval import MODEL_VERSION, MODEL_VERSION_LEGACY

MIN_N = 30
MIN_N_HYPOTHESIS = 10


def _closed_with_forecasts(kelly_only: bool = True) -> list[dict]:
    fcs = {f["forecast_id"]: f for f in read_forecasts()}
    out = []
    for t in read_trades():
        if t.get("status") != "closed":
            continue
        fc = fcs.get(t["forecast_id"])
        if not fc:
            continue
        if kelly_only and not is_kelly_regime(fc):
            continue
        row = {**fc, **t}
        row["outcome"] = 1 if float(t["pnl"]) > 0 else 0
        out.append(row)
    return out


def _model_version(row: dict) -> str:
    """trades.csv rows written before Patch 1 have no model_version cell —
    they were scored under the biased terminal-only MC. Tag them so they
    never leak into the corrected model's headline numbers."""
    v = str(row.get("model_version") or "").strip()
    return v if v else MODEL_VERSION_LEGACY


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
    closed = _closed_with_forecasts(kelly_only=True)
    closed_v1 = _closed_with_forecasts(kelly_only=False)
    v1_closed = [c for c in closed_v1 if not is_kelly_regime(c)]
    forecasts_all = read_forecasts()
    forecasts = [f for f in forecasts_all if is_kelly_regime(f)]
    v1_forecasts = [f for f in forecasts_all if not is_kelly_regime(f)]
    trades = read_trades()

    print("\n=== PAPER REPORT (Kelly regime) ===\n")
    print(f"regime={REGIME_KELLY}  (v1 {REGIME_V1} kept on disk, excluded from score)")
    print(f"excluded v1: forecasts n={len(v1_forecasts)} closed trades n={len(v1_closed)}")

    all_taken = list(closed)
    taken = [c for c in all_taken if _model_version(c) == MODEL_VERSION]
    legacy = [c for c in all_taken if _model_version(c) != MODEL_VERSION]
    n = len(taken)
    print(f"model_version={MODEL_VERSION}  ({MODEL_VERSION_LEGACY} kept on disk, "
          f"excluded from headline — biased terminal-only MC, see AGENT_CONTEXT Patch 1)")
    print(f"excluded {MODEL_VERSION_LEGACY}: closed trades n={len(legacy)}")
    shadows = [c for c in taken if is_shadow(c)]
    taken_real = [c for c in taken if not is_shadow(c)]
    print(f"closed trades n={n} (shadow={len(shadows)} taken={len(taken_real)}; "
          f"need {MIN_N} for Brier/calibration/AUC)")

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

    print("\n3b. Per model_version (informational — headline above is mc_path_v2 only)")
    for mv in (MODEL_VERSION, MODEL_VERSION_LEGACY):
        rows = [c for c in all_taken if _model_version(c) == mv]
        if not rows:
            print(f"   {mv:<16} n=0")
            continue
        yv = [c["outcome"] for c in rows]
        pv = [float(c["model_prob_profit"]) for c in rows]
        bv = brier(pv, yv)
        hv = float(np.mean(yv))
        print(f"   {mv:<16} n={len(rows):<4} hit_rate={hv:.1%}  "
              f"Brier={f'{bv:.4f}' if bv is not None else 'insufficient sample'}")

    print("\n3c. Per hypothesis (Patch 3 — localizes calibration by trade idea)")
    for h in HYPOTHESES:
        rows = [c for c in taken if str(c.get("hypothesis", "")).strip() == h]
        nh = len(rows)
        if nh < MIN_N_HYPOTHESIS:
            print(f"   {h:<15} n={nh:<3} insufficient sample (need {MIN_N_HYPOTHESIS})")
            continue
        yh = [c["outcome"] for c in rows]
        ph = [float(c["model_prob_profit"]) for c in rows]
        bh2 = brier(ph, yh) if nh >= MIN_N else None
        hit = float(np.mean(yh))
        mean_pred = float(np.mean(ph))
        bstr = f"{bh2:.4f}" if bh2 is not None else f"n<{MIN_N} for formal Brier"
        print(f"   {h:<15} n={nh:<3} hit_rate={hit:.1%}  mean_pred={mean_pred:.3f}  Brier={bstr}")

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

    print("\n6. P&L (NOISY — do not use for decisions before n≈100; taken only, no shadows)")
    if not taken_real:
        print("   insufficient sample n=0")
    else:
        pnls = [float(c["pnl"]) for c in taken_real]
        print(f"   total P&L     ${sum(pnls):+.2f} n={len(pnls)}")
        print(f"   mean P&L      ${np.mean(pnls):+.2f}")
        if len(taken_real) < MIN_N:
            print(f"   insufficient sample for inference n={len(taken_real)}")

    open_real = [t for t in trades if t.get("status") == "open" and not is_shadow(t)]
    open_shadow = [t for t in trades if t.get("status") == "open" and is_shadow(t)]
    print(f"\nopen trades: {len(open_real)}  shadows: {len(open_shadow)}")
