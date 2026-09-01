# Options Screener — Agent Context

Hand this file to other agents working on **this repo only** (`options_screener`).  
Ignore anything under sibling projects (e.g. timeseriesfm playground).

Repo: https://github.com/PandaGrdn/options_screener  
Local path: `options_screener/` (standalone git repo).

---

## 1. What this system is

A **paper-trading + data-capture** stack for single-name / sector-ETF **call debit spreads**.

It does **not** place live broker orders. It:

1. Snapshots option chains, underlyings, and earnings daily (GitHub Actions).
2. Scores the universe under a **Kelly / log-growth** model.
3. Marks open paper trades to market and auto-closes on TP / SL / time-stop.
4. Scores **calibration** (Brier, calibration curve, AUC) — P&L is reported last and labeled noisy.

**Primary question the system answers:** when the model assigns probability \(p\) to profit, does the trade win ~\(p\) of the time?

---

## 2. Hard constraints (do not “helpfully” violate)

| Constraint | Why |
|---|---|
| `data/forecasts.csv` is **append-only** | No edit/update/delete API. A wrong forecast honestly logged beats a corrected one. Do **not** add an edit command. |
| Exit rules (`tp_level`, `sl_level`, `time_stop_date`) immutable after open | Only exit fields may change on close. |
| Fills at **40% toward mid**, never mid | `natural - 0.4*(natural - mid)`. Stops get an extra 20% of width against you. |
| Fees | `$0.65` / contract / leg / side |
| Portfolio | `$5,000`; **4%** max per trade; **20%** max deployed |
| Reference-only (no entries) | `SPY`, `QQQ`, `IWM` |
| Structures | **Call debit spreads only** (buy call + sell higher call). No naked long calls in current `evaluate()`. |
| DTE window | **14–30** days |
| Cheapness screen | `signals.cheapness_gate()` — RV percentile ≤40 AND IV-RV spread ≤+0.03, AND IV rank ≤30 once ≥180d of IV history exists (nan term ignored until then). Replaced the raw `IV_RANK_MAX=30` check (Patch 2) — IV rank alone is noise on weeks of chain_history. |
| TRADE gate | Kelly: `log_growth > 0` (plus size/cap). Not raw EV / prob thresholds alone. Computed by a **path-dependent** daily Monte Carlo (`spread_eval.simulate()`, Patch 1) — TP/SL are first-touch events, not terminal-value thresholds. |
| Score only `regime=kelly` | Older `v1_rv_pev` / blank-regime rows stay on disk but are **excluded** from Brier/calibration. |
| Cron **does not open** trades | Auto logs market-default skips; a TRADE needs a human forecast that beats market on Kelly. |

Spec source of truth (original design): `papertradingspec.md` — some behaviors have evolved (Kelly, no auto-open); **prefer this file + code** when they disagree.

---

## 3. Layout

```
options_screener/
├── screener.py              # UNIVERSE, vol estimators, live buyer screen, legacy snapshot_iv
├── snapshot.py              # Daily capture: chains + underlyings + earnings
├── spread_eval.py           # BS IV/greeks, fills, path-dependent MC simulate(), evaluate(), breakeven_forecast(), sizing
├── signals.py                # cheapness_gate(): rv_percentile, iv_rv_spread, iv_rank (data-gated)
├── backfill_underlyings.py   # one-time idempotent 2y OHLC backfill (underlying_history is backfillable; chains are not)
├── migrate_model_version.py  # one-time idempotent model_version backfill (sanctioned append-only exception, Patch 1)
├── paper/
│   ├── __init__.py          # Paths, PORTFOLIO caps, REFERENCE_ONLY
│   ├── __main__.py          # python -m paper
│   ├── cli.py               # argparse entrypoint
│   ├── models.py            # CSV schemas; append_forecast (append-only)
│   ├── entry.py             # forecast / decide / evaluate / open
│   ├── mark.py              # Daily MTM from chain_history
│   ├── exit.py              # TP/SL/time_stop/expiry checks + apply_close
│   ├── score.py             # Brier, calibration, AUC, report()
│   ├── auto.py              # Cron: score universe (no opens) → mark → report → dashboard
│   └── dashboard.py         # data/dashboard.html generator
├── data/
│   ├── forecasts.csv        # APPEND-ONLY
│   ├── trades.csv
│   ├── marks.csv
│   ├── latest_report.txt
│   ├── report_history.txt
│   └── dashboard.html
├── chain_history.csv        # Repo-root market data (Actions commits these)
├── underlying_history.csv
├── earnings_calendar.csv
├── iv_history.csv           # Legacy ATM-IV-only (superseded by chain_history)
├── tests/
│   ├── test_paper.py
│   ├── test_snapshot.py
│   └── test_spread_eval.py
├── .github/workflows/iv-snapshot.yml
├── requirements.txt         # yfinance pandas numpy scipy pytest
├── run_snapshot.sh          # Local fallback (prefer Actions)
└── papertradingspec.md
```

---

## 4. Universe (`screener.UNIVERSE`)

```
NVDA TSLA AMD META AVGO MU COIN PLTR NFLX CRM UBER SHOP MRVL SMCI
XLE XLF XBI SMH ARKK GDX
QQQ IWM SPY          # reference only — refuse paper entries
```

---

## 5. Daily data capture (`snapshot.py`)

Entry: `python snapshot.py` → `run_all(UNIVERSE)`.

| File | What | Notes |
|---|---|---|
| `chain_history.csv` | Raw bid/ask + own BS `iv`/greeks + `yf_iv` | Same-session; **not backfillable**. Calls+puts, ~30d & ~60d targets, ±10% moneyness band. Idempotent by `(date, ticker)`. |
| `underlying_history.csv` | OHLC + YZ/C2C RV | **Previous** completed NY session (Yahoo daily bar reliable T+1). Can backfill. |
| `earnings_calendar.csv` | Next earnings as known on `asof` | Idempotent by `(asof, ticker)`. |

Session dates use **America/New_York**, not the runner’s UTC date (`session_date()`, `previous_session_date()`).

Chain schema includes:  
`date, ts_utc, ticker, spot, expiry, dte, type, strike, moneyness, bid, ask, mid, spread_pct, volume, open_interest, yf_iv, iv, delta, gamma, vega, theta`.

---

## 6. Model (`spread_eval.py`)

- **IV:** solve BS from mid (`implied_vol`); greeks via `bs_greeks`.
- **Fills:** `fill_toward_mid` / `debit_entry_fill` / `credit_exit_fill` (40% to mid; stops worse).
- **Simulate (Patch 1, path-dependent):** daily GBM walk (`n_steps=hold_days`, default = dte).
  Each leg marked by Black-Scholes at its OWN IV (`long_entry_iv`/`short_entry_iv`, linearly
  interpolated toward `*_iv_exit`, default unchanged), decoupled from the forecast vol driving
  the underlying's path. TP (2× debit) / SL (0.5× debit) are **first-touch** events on the
  conservative mark, not terminal-value thresholds — survivors close at `time_stop`
  (day `hold_days-5`). Returns `prob_tp`/`prob_sl`/`prob_time_stop` plus the full `returns`
  array. `n_paths=40_000` default, vectorized, seeded (reproducible).
  Old `trades.csv`/`forecasts.csv` rows were scored under the prior **terminal-only** MC —
  tagged `model_version=mc_terminal_v1` (vs `mc_path_v2` now) and never rewritten; `score.py`
  headlines only `mc_path_v2`. See `migrate_model_version.py`.
- **`breakeven_forecast()` (Patch 4):** bisects the simulator to find the minimum directional
  move (vol held at market IV) and minimum vol (drift held at zero) that flip a spread's
  verdict from SKIP to TRADE. Market data + the fill model, not a model opinion — `paper decide`
  shows it **before** the forecast prompt without violating the blind-forecast rule.
- **`evaluate()`:** call debit spreads only, 14–30 DTE, pick best by `log_growth`.
  - Market default: vol = ATM IV, drift = 0 → should SKIP after fees/conservative fills.
  - TRADE requires a forecast that beats market on Kelly log-growth.
- Knobs: `MIN_DTE`, `MAX_DTE`, `LOG_GROWTH_MIN`, `PORTFOLIO`, size caps.
  `IV_RANK_MAX` still exists but the live screen is `signals.cheapness_gate()` (Patch 2).

---

## 7. Paper trading (`paper/`)

### CLI (`python -m paper <cmd>`)

| Command | Role |
|---|---|
| `decide --ticker X` | Interactive: enter forecast vol/move → Kelly verdict → trade/skip → optional open |
| `forecast --ticker X` | Optional **blind** human forecast (no model shown) |
| `evaluate --forecast-id` | Re-run model on logged forecast |
| `open --forecast-id [--override]` | Open trade; override needs reason |
| `mark [--asof]` | Mark opens; auto-close exits |
| `run-daily` | Cron path: market-default score all names (**no opens**) → mark → report → dashboard |
| `report` | Print Kelly-regime scorecard |
| `dashboard` | Regenerate `data/dashboard.html` |

There is **no** `edit` / `revise` forecast command.

### Schemas (`paper/models.py`)

**forecasts.csv** (append-only):  
`forecast_id, ts_utc, ticker, horizon_days, direction, pred_move_pct, pred_vol_annual, pred_prob_profit, iv_at_forecast, iv_rank, rationale, decision, skip_reason, earnings_trade, source, regime, gate_reason, hypothesis, model_version`

- `source`: `model` | `human`
- `regime`: `kelly` (scored) | blank/`v1_rv_pev` (legacy, excluded)
- `gate_reason` (Patch 2): human-readable `cheapness_gate()` output, whatever the decision
- `hypothesis` (Patch 3): `trend|mean_reversion|vol_expansion|post_earnings|catalyst|other` —
  **required** on `paper decide`/`paper forecast` (`other` needs ≥20-char rationale); blank on
  old rows and on cron-auto rows (no trade idea to tag), excluded from per-hypothesis tables
- `model_version` (Patch 1): `mc_terminal_v1` | `mc_path_v2` | blank (blind `paper forecast` —
  no model was invoked). Backfilled once on old kelly-regime rows by
  `migrate_model_version.py` — the **only** sanctioned exception to append-only, and it touches
  only this column.

**trades.csv:**  
open/closed positions linked by `forecast_id`; stores structure, strikes, conservative `entry_debit`, model metrics, TP/SL/time_stop, exit fields, override flags, `model_version`.

**marks.csv:**  
`mark_date, trade_id, spot, spread_mid, spread_conservative, unrealized_pnl, dte_left, carried_forward, flag`  
Missing strike → **carry forward last mark + flag**; never interpolate.

### Exits (`paper/exit.py` + `mark.py`)

Reasons: `tp`, `sl`, `time_stop`, `expiry`.  
SL fills use stop haircut. Marks store both mid and conservative so fill assumption can be measured.

### Scoring (`paper/score.py`)

Order: **Model Brier → calibration table → AUC → per-model_version → per-hypothesis → skips → overrides → P&L (noisy)**.  
Suppress interpretation below **n=30** (“insufficient sample”); per-hypothesis suppresses below **n=10**.  
Kelly-only for headline metrics, and within Kelly, `model_version=mc_path_v2` only (Patch 1) —
`mc_terminal_v1` rows are shown in an informational per-version breakdown but never headline.

### Dashboard (`paper/dashboard.py`)

Static HTML at `data/dashboard.html` from latest CSVs (open MTM, skips, closed PnL, sample warnings).

---

## 8. GitHub Actions (the daily cron)

File: `.github/workflows/iv-snapshot.yml`  
Name: **Daily paper trading**

```yaml
schedule:
  - cron: "55 21 * * 1-5"   # 2:55pm Pacific (PDT); after US cash close
workflow_dispatch: true      # manual run from Actions tab
```

**Job steps:**

1. Checkout `main`, Python 3.12, `pip install -r requirements.txt`
2. `python snapshot.py` (chains + underlyings + earnings)
3. Fail loudly if today’s chain rows are zero / underlyings stale
4. `python -m paper run-daily --max-new 1` (scores + marks + report + dashboard; **does not open**)
5. Fail if `data/latest_report.txt` or `data/dashboard.html` empty
6. Commit & push: `chain_history.csv`, `underlying_history.csv`, `earnings_calendar.csv`, `data/`

**Known quirk:** GitHub `schedule` often slips hours (or rare miss). On-time proof exists historically; delays are GHA, not a wrong cron string. Use **Run workflow** as backup.

Workflow is **active**. Permissions: `contents: write` (bot commits).

---

## 9. Local commands

```bash
cd options_screener
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# One-shot market snapshot
.venv/bin/python snapshot.py

# Buyer-friendly screen (live yfinance; proxy VRP)
.venv/bin/python screener.py

# Paper
.venv/bin/python -m paper decide --ticker NVDA
.venv/bin/python -m paper run-daily
.venv/bin/python -m paper report
.venv/bin/python -m paper dashboard

# Tests
.venv/bin/python -m pytest tests/ -q
```

---

## 10. Current design intent (evolved)

1. **Data flywheel:** build irreversible option-chain history for real IV / VRP later.
2. **Market-default should SKIP** (IV + 0 drift after conservative fills/fees).
3. **Opens require an edge forecast** that improves Kelly log-growth vs market — entered via `paper decide` (or forecast→evaluate→open). Cron only maintains data + marks + scorecard.
4. **Legacy v1 trades** (early auto-opens under RV/flat model) remain in CSVs for history but are **not** the Kelly scorecard.

Default horizon in CLI/auto: **21** DTE.

---

## 11. Tests worth knowing

- Append-only forecasts (no mutate/delete helpers)
- Fill model worse than mid; stops worse than normal exit
- Exit triggers + immutable TP/SL/time_stop
- Position sizing caps
- Missing strike → carry forward, no interpolate
- Brier/calibration on synthetic data
- Score excludes v1 closed trades
- `evaluate` spreads-only; Kelly gate; expiry window
- Snapshot session dates use New York
- **Patch 1:** path-dependence (first-touch TP beats terminal-only), zero-vol determinism,
  exit-reason probabilities sum to 1, seeded reproducibility, market-default SKIP regression
- **Patch 2:** `cheapness_gate` fails on high RV percentile; `iv_rank` nan below 180d and the
  gate still resolves; `backfill_underlyings.py` idempotent (0 new rows on rerun)
- **Patch 3:** wired through the Patch-2 integration tests (`hypothesis` required/validated)
- **Patch 4:** feeding `breakeven_forecast()`'s drift back into `simulate()` ≈ zero log_growth;
  breakeven move is strictly positive for a debit spread with fees
- **Migration:** `migrate_model_version.py` touches only the `model_version` cell, only on
  kelly-regime rows a model actually scored, and is idempotent

---

## 12. What not to build

- Live order routing / broker APIs
- Real-time streaming UI beyond the static dashboard
- Mid fills or “edit forecast”
- Auto-opening from cron without a forecast that beats market
- Entries in SPY/QQQ/IWM
- Treating noisy early P&L as proof of edge (need ~30 for calibration, ~100+ for P&L)
- A drift/direction prediction model before Phase 2's data gate (≥60 calendar days of
  `chain_history`) — as of this writing there are only ~8 days of chain_history, nowhere near
  the gate. Phase 2 (vol-only forecaster) and Phase 3 (`paper verdict`, no-edge exit criterion)
  are specified but intentionally **not built yet** — see `papertradingspec.md` history /
  ask the user for the Patch Spec v2 doc if picking this up. Even once built: vol-only
  (drift hardcoded to 0), non-overlapping fit windows only, never rewrite historical
  `model_prob*` values, no new TA indicators/ML frameworks, never soften the phase-3
  no-edge message.

---

## 13. Quick file→responsibility map

| Need | Look in |
|---|---|
| Add ticker to scan list | `screener.UNIVERSE` |
| Change fill / Kelly / DTE knobs | `spread_eval.py` |
| Change the cheapness/vol screen | `signals.py` |
| Change cron schedule or commit paths | `.github/workflows/iv-snapshot.yml` |
| Change what cron scores/opens | `paper/auto.py` |
| Forecast/open UX, breakeven display | `paper/entry.py`, `paper/cli.py` |
| Mark/exit logic | `paper/mark.py`, `paper/exit.py` |
| Metrics definition | `paper/score.py` |
| CSV columns | `paper/models.py`, `snapshot.py` schemas |
| Path constants | `paper/__init__.py` |
| One-time backfill/migration scripts | `backfill_underlyings.py`, `migrate_model_version.py` |
