# Paper Trading System — Build Spec

Hand this to Cursor as-is. Existing repo already has `screener.py`, `spread_eval.py`,
`snapshot.py`, and a growing `chain_history.csv` from a daily GitHub Actions cron.

---

## 0. What this system actually tests

**Not "did I make money."** At 30–50 trades with ~40% win rates, P&L is almost pure
noise and will mislead in either direction.

The system tests one thing: **is my forecast calibrated?** When I say a trade has a
45% chance of profit, does it hit ~45% of the time? Calibration converges far faster
than P&L and is the only thing that tells you whether to risk real capital.

Everything below exists to make forecasts recorded, immutable, and scoreable.

**Non-goals:** live order routing, broker integration, real-time streaming, a UI
beyond CLI, optimizing returns. Do not build these.

---

## 1. Repo structure

```
options_screener/
├── screener.py          # exists
├── spread_eval.py       # exists
├── snapshot.py          # exists
├── paper/
│   ├── __init__.py
│   ├── models.py        # dataclasses + schema constants
│   ├── entry.py         # forecast capture + trade open
│   ├── mark.py          # daily mark-to-market from chain_history
│   ├── exit.py          # exit rule evaluation
│   ├── score.py         # calibration + performance metrics
│   └── cli.py           # argparse entrypoint
├── data/
│   ├── chain_history.csv    # exists, from cron
│   ├── forecasts.csv        # append-only, IMMUTABLE
│   ├── trades.csv           # open + closed positions
│   └── marks.csv            # daily MTM per open trade
└── tests/
```

---

## 2. Schemas

### `forecasts.csv` — append-only, never edited or deleted

Written **before** any model output is displayed. This ordering is the entire point.

| field | type | notes |
|---|---|---|
| `forecast_id` | str | uuid4 |
| `ts_utc` | iso8601 | |
| `ticker` | str | |
| `horizon_days` | int | matches intended hold |
| `direction` | enum | `up` / `down` / `flat` |
| `pred_move_pct` | float | your point estimate of underlying move |
| `pred_vol_annual` | float | your realized-vol forecast |
| `pred_prob_profit` | float | 0–1, your gut probability **before** running the model |
| `iv_at_forecast` | float | recomputed from `chain_history`, not yfinance's field |
| `iv_rank` | float | 0–100 |
| `rationale` | str | free text, min 20 chars, enforced |
| `decision` | enum | `trade` / `skip` |
| `skip_reason` | str | required when `decision=skip` |

**Log skipped setups too.** Trades you rejected are data — if your skips would have
won and your entries lost, that's the finding.

### `trades.csv`

| field | type | notes |
|---|---|---|
| `trade_id` | str | uuid4 |
| `forecast_id` | str | FK, required, must exist |
| `opened_utc` | iso8601 | |
| `ticker`, `structure` | str | `long_call` / `call_debit_spread` |
| `expiry`, `dte_at_entry` | date, int | |
| `long_strike`, `short_strike` | float | short null for long_call |
| `entry_debit` | float | see fill model §4 |
| `contracts` | int | from `spread_eval.size_position` |
| `capital_at_risk` | float | |
| `model_prob_profit` | float | from `spread_eval.simulate` |
| `model_ev`, `model_log_growth` | float | |
| `tp_level`, `sl_level`, `time_stop_date` | float, float, date | set at entry, never changed |
| `status` | enum | `open` / `closed` |
| `closed_utc`, `exit_credit`, `exit_reason`, `pnl`, `return_pct` | | null while open |

### `marks.csv`

`mark_date, trade_id, spot, spread_mid, spread_conservative, unrealized_pnl, dte_left`

---

## 3. Workflow

### `paper forecast --ticker NVDA`
1. Load latest `chain_history` row for ticker. Recompute IV from bid/ask with the
   BS solver in `spread_eval.py`. Compute IV rank from the ticker's own history.
2. Show **only**: spot, IV, IV rank, realized vol, earnings date. **Do not show any
   model output, EV, or probability.**
3. Prompt for all forecast fields. Enforce `rationale` length.
4. Append to `forecasts.csv`. Print `forecast_id`.

### `paper evaluate --forecast-id <id>`
1. Now run `spread_eval.evaluate()` with the user's `pred_vol_annual` and a drift
   derived from `pred_move_pct` annualized.
2. Print the model verdict and sizing.
3. Print the delta between `pred_prob_profit` and `model_prob_profit` — this gap,
   tracked over time, shows whether the user or the model is better calibrated.

### `paper open --forecast-id <id> [--override]`
- Refuse if the model says SKIP unless `--override` is passed **and** an override
  reason is logged. Overrides get tagged and scored separately.
- Refuse if total open `capital_at_risk` would exceed 20% of $5,000.
- Refuse if `contracts == 0`.

### `paper mark` — run daily via the same Actions cron, after `snapshot_chains`
- For each open trade, price both legs off today's `chain_history` rows.
- Append to `marks.csv`.
- Auto-close any trade hitting TP, SL, or time stop; write exit fields.
- If a strike is missing from the chain that day, carry forward the last mark and
  flag it. Never silently interpolate.

### `paper report`
Calibration and performance output (§5).

---

## 4. Fill model — do not use mid

This is where paper trading lies to you most.

- **Entry:** `natural - 0.4 * (natural - mid)`. You get 40% of the way to mid, not 100%.
- **Exit:** same haircut, applied against you.
- **Fees:** $0.65 per contract per leg per side.
- **Slippage on stops:** additional 20% of the bid-ask width — stops fill worse.

Store both `spread_mid` and `spread_conservative` in marks so you can measure how
much the fill assumption is worth. If your edge only exists at mid fills, you have
no edge.

---

## 5. Scoring — `paper report`

Primary metrics, in this order:

1. **Brier score** on `pred_prob_profit` vs realized binary outcome.
   Also compute for `model_prob_profit`. Lower is better. Compare the two.
2. **Calibration curve** — bucket forecasts into deciles, plot predicted vs actual
   hit rate. Print as a text table. This is the headline output.
3. **Discrimination** — AUC of forecast probability vs outcome. Tells you whether
   your forecasts rank trades correctly even if the absolute levels are off.
4. **Skip analysis** — hit rate of trades taken vs the counterfactual outcome of
   skipped setups.
5. **Override analysis** — performance of `--override` trades vs model-approved.
6. P&L, realized log growth vs `model_log_growth`. **Report last, labeled as noisy.**

Every metric must print `n=` alongside. Suppress interpretation below n=30 —
print "insufficient sample" rather than a number that invites over-reading.

---

## 6. Constraints

- `forecasts.csv` append-only. Add a test asserting no row is ever mutated or removed.
- `trade_id` requires a valid `forecast_id`; reject orphan trades.
- Exit rules immutable after open. Add a test.
- Portfolio $5,000; 4% max per trade; 20% max total deployed.
- Universe: single names and sector ETFs from `screener.UNIVERSE`. **Reject SPY,
  QQQ, IWM for entries** — they're reference points only, and the index variance
  premium makes them structurally hostile to buyers.
- Refuse entry if earnings falls inside the hold window unless explicitly flagged
  as an earnings trade (separate tag, scored separately).

---

## 7. Tests

- Append-only enforcement on forecasts.
- Fill model: entry debit strictly worse than mid, exit credit strictly worse than mid.
- Exit triggers fire at correct thresholds; time stop fires on the right date.
- Position sizing never exceeds caps, including across multiple open trades.
- Missing-strike day carries forward and flags rather than interpolating.
- Brier and calibration verified against a synthetic set with known probabilities.

---

## 8. Discipline notes for the human

- Forecast **before** `evaluate`. The CLI enforces the order; don't work around it.
- Never edit a past forecast. A wrong forecast honestly recorded is worth more than
  a right one edited in.
- 2–4 trades/week. Faster means forcing setups.
- First review at 3 months. Do not draw conclusions before n=30.