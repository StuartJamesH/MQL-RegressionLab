# Phase 1 — Labelling Redesign (labels.py + training notebook label cell)

Status: IMPLEMENTED (this session) — see verification results at the bottom of this file.

Depends on: Phase 0 (uses `Learn/train_utils.py` metric helpers for calibration diagnostics only;
otherwise independent).

## Background
Current primary label pipeline (`ModelWorkbench/Learn/labels.py`):
- `_compute_outcomes_capped_nb` (numba kernel): entry = prior candle High (buy) / Low (sell) —
  this correctly matches production buy-stop/sell-stop mechanics. TP/SL = entry +/- mult*ATR.
  Scans forward up to `max_horizon` bars. Produces `buy_outcome`/`sell_outcome` (1.0=TP, -1.0=SL,
  NaN=unresolved) and `buy_MFE`/`buy_MAE`/`sell_MFE`/`sell_MAE` (normalized to TP/SL distance,
  unidirectional tracking post-resolution — this part is sound, do not change it).
- `calculate_trade_outcomes_capped(df, atr_window=14, tp_mult=4.0, sl_mult=2.0, max_horizon=30)` —
  the primary entrypoint used by the training notebook. Keep this function name and signature.

Problem being fixed: the training notebook's label-computation cell (`2.1 Train LGBM Regression
Model.ipynb`, cell titled "Load data and compute target variables") builds `long_quality`/
`short_quality` via `log1p(MFE/MAE)`, then runs an O(n × max_horizon) pure-Python nested loop
(`bars_to_exit_buy`/`bars_to_exit_sell`) that scans forward to find the first resolved bar and
applies `exp(-TIME_PENALTY_LAMBDA * bars_to_exit / max_horizon)` multiplicative shaping to the
target. This is slow, mixes "trade quality" with "speed of resolution" in a way that isn't clearly
useful, and unresolved trades are currently `fillna(0.0)`'d — i.e. treated as zero-quality even
though "unresolved within horizon" is different from "genuinely bad setup".

## Decision (already agreed with user)
Two-head design: a classifier predicts {SL, TP, timeout} per side; a separate magnitude regressor
predicts `log1p(MFE/MAE)` trained ONLY on resolved (TP or SL) rows. Drop the time-penalty loop
entirely. Extend/calibrate `max_horizon` to shrink the timeout fraction rather than dropping
timeout rows from the classifier (they become the third class).

`TIME_PENALTY_LAMBDA` must remain declared in the training notebook's config cell (do not delete
any existing config variable) — add a one-line comment noting it is no longer used by the label
pipeline, retained for backward compatibility / potential future use.

## Steps

### 1. Add horizon calibration helper to `labels.py`
Add a new function:
```python
def resolution_rate_by_horizon(df, atr_window, tp_mult, sl_mult, horizon_candidates):
    """
    For each candidate max_horizon, compute the fraction of buy/sell trades that resolve
    (hit TP or SL) within that horizon. Reuses the existing outcome kernel — do not
    duplicate the numba kernel logic, just call calculate_trade_outcomes_capped once per
    candidate horizon (largest horizon first is fine; correctness over speed here since this
    runs once per training session).

    Returns a DataFrame with columns: horizon, buy_resolved_rate, sell_resolved_rate.
    """
```
Implementation: for each `h` in `horizon_candidates`, call
`calculate_trade_outcomes_capped(df, atr_window=atr_window, tp_mult=tp_mult, sl_mult=sl_mult, max_horizon=h)`
and compute `outcomes['buy_outcome'].notna().mean()` / same for sell. Return as a tidy DataFrame.

### 2. Extend `calculate_trade_outcomes_capped` output with explicit class columns
In the same function (or immediately as a small wrapper — prefer editing the function directly
since it's the sole call site of interest), add two new integer columns to the returned DataFrame:
- `buy_class`: 1 where `buy_outcome == 1.0` (TP), 0 where `buy_outcome == -1.0` (SL), 2 where
  `buy_outcome` is NaN (timeout/unresolved).
- `sell_class`: same mapping using `sell_outcome`.

Keep all existing returned columns (`buy_outcome`, `sell_outcome`, `buy_MFE`, `buy_MAE`,
`sell_MFE`, `sell_MAE`, and any others already present) unchanged — this is purely additive so
any other consumer (e.g. `1.1 Regression Lab.ipynb`, `MFE_filter_outcomes`) keeps working.

### 3. Remove the time-penalty loop from the training notebook
In `2.1 Train LGBM Regression Model.ipynb`, cell "Load data and compute target variables":
- Delete the `bars_to_exit_buy`/`bars_to_exit_sell` nested-loop block and the
  `buy_time_penalty`/`sell_time_penalty` computation and multiplication into `long_q`/`short_q`.
- `long_quality`/`short_quality` become plain `np.clip(np.log1p(MFE/(MAE+eps)), 0.0, TARGET_CLIP_MAX)`
  (unchanged formula otherwise).
- Add `df["buy_class"] = outcomes["buy_class"]` and `df["sell_class"] = outcomes["sell_class"]`.
- Keep `buy_win`/`sell_win`/`signed_win` computation as-is (still useful, still additive).
- Add a comment near `TIME_PENALTY_LAMBDA` in the config cell noting it is currently inert.

### 4. Add horizon calibration cell (new cell, placed right after data load, before label
computation)
- Uses new config vars `MAX_HORIZON_CANDIDATES` (e.g. `[20, 30, 45, 60]`) and
  `TARGET_RESOLUTION_RATE` (e.g. `0.90`) — add these to the config cell, do not remove/rename
  `outcome_params['max_horizon']`.
- Calls `resolution_rate_by_horizon`, displays the table, picks the smallest horizon candidate
  whose `min(buy_resolved_rate, sell_resolved_rate) >= TARGET_RESOLUTION_RATE`; if none qualify,
  use the largest candidate and print a warning.
- Assigns the chosen horizon into `outcome_params['max_horizon']` (the existing dict/key — do not
  rename), so all downstream cells keep working unchanged.

### 5. Filtering resolved rows for the magnitude regressor (later, in Phase 3)
Do NOT filter rows inside `labels.py` or inside `preprocess.py` — filtering by
`buy_class != 2` / `sell_class != 2` happens in the training notebook when building the
regressor's train/holdout arrays (Phase 3, step 5). This phase only needs to make sure
`buy_class`/`sell_class` end up as columns on `df` so that filtering is possible later.

## Verification
- After editing, run (in a Python shell, from `ModelWorkbench/`):
  - Load a small slice of one CSV (e.g. `EURUSD_M1_520weeks.csv`, `n_rows=20_000`), call
    `calculate_trade_outcomes_capped` with the existing default params, and assert:
    - Every row with `buy_class == 1` has `buy_MFE > 0` (a TP-resolved trade favors the reward
      side).
    - Every row with `buy_class == 0` has `buy_MAE > 0`.
    - `(outcomes['buy_class'] == 2).mean()` roughly matches `outcomes['buy_outcome'].isna().mean()`
      from the unmodified function (sanity check the new column derivation).
  - Call `resolution_rate_by_horizon` with `horizon_candidates=[20, 30, 45, 60]` and confirm
    resolved rate increases monotonically (or nearly so) with horizon.
- Do not proceed to Phase 3 until this file's checks pass on real data.

## Verification results (this session)
- Unit-level check on EURUSD_M1_520weeks.csv (20k row slice, atr_window=60, tp_mult=3.0, sl_mult=2.0,
  max_horizon=30): all TP rows have `buy_MFE > 0`, all SL rows have `buy_MAE > 0`,
  `(buy_class==2).mean()` exactly matches `buy_outcome.isna().mean()`. `resolution_rate_by_horizon`
  over [20,30,45,60] gave monotonically increasing buy/sell resolved rates (0.768/0.872/0.938/0.965
  buy; similar for sell).
- Full end-to-end run of the training notebook's data-load + horizon-calibration + label cells on
  the full EURUSD dataset (N_ROWS=500,000): horizon calibrated to 45 bars (smallest candidate
  clearing the 90% TARGET_RESOLUTION_RATE for both sides — actual rates ~94.4%/94.7%). Resulting
  class balance: buy {SL: 0.667, TP: 0.278, timeout: 0.055}, sell {SL: 0.665, TP: 0.282,
  timeout: 0.053}. No time-penalty loop remains; label cell now completes without the previous
  O(n×horizon) Python loop.
- NOTE: this full run (through feature selection + label computation) took ~30 minutes wall-clock on
  500k rows, dominated by `select_features_pipeline`'s mutual-information ranking, not by anything
  in this phase. Future phases should avoid re-running the whole notebook repeatedly; use a reduced
  `N_ROWS` smoke test while iterating on Phase 3/4 code, and only do a full run at the end.
