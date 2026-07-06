# Phase 3 — Training Notebook Rewrite (`2.1 Train LGBM Regression Model.ipynb`)

Status: NOT STARTED

Depends on: Phase 0 (train_utils), Phase 1 (labels.py buy_class/sell_class + horizon calibration),
Phase 2 (feature_codegen.py).

## Hard constraint
Do NOT remove or rename any existing config variable from the notebook's config cell:
`DS_NAME, N_ROWS, TP_MULT, SL_MULT, TARGET_CLIP_MAX, TRAIN_FRACTION, MAX_TUNING_ROWS, WFO_FOLDS,
WFO_MIN_TRAIN_FRAC, WFO_TEST_FRAC, WFO_GAP_ROWS, EARLY_STOPPING_ROUNDS, ROBUSTNESS_PENALTY,
USE_FEATURE_SELECTION, SELECTED_FEATURE_COUNT, USE_TRADEABILITY_WEIGHTING,
TRADEABILITY_N_FOLDS, TRADEABILITY_MIN_TRAIN, TRADEABILITY_TEST_SIZE, TARGET_COLS,
CLASSIFICATION_TARGETS, PRIMARY_TARGET, TIME_PENALTY_LAMBDA, TRAIN_CLASSIFIER, ENSEMBLE_SIZE,
ENSEMBLE_SEEDS, regime_params, outcome_params, LGBM_BASE_PARAMS, LGBM_CANDIDATES`.
New variables are added alongside, never in place of existing ones. If a variable becomes inert
(e.g. `TIME_PENALTY_LAMBDA`), keep it declared with a one-line comment explaining why.

## New config variables to add
- `MAX_HORIZON_CANDIDATES = [20, 30, 45, 60]`
- `TARGET_RESOLUTION_RATE = 0.90`
- `MODEL_CANDIDATES = ["lgbm", "mlp"]`
- `DL_HIDDEN_SIZES = (128, 64)`
- `DL_EPOCHS = 60`
- `DL_LR = 1e-3`
- `DL_BATCH_SIZE = 4096`

## Step-by-step

### 1. Imports cell
Replace inline helper definitions with:
```python
from Learn.train_utils import (
    load_ohlcv, safe_spearman, reg_metrics, lgbm_spearman_eval,
    walk_forward_folds, describe_target_frame, tune_lgbm_by_spearman, tune_lgbm_classifier,
)
from Learn.feature_codegen import write_generated_feature_module, load_generated_feature_module
```
(`tune_lgbm_classifier` is a new function added to `Learn/train_utils.py` in step 6 below — add it
there, not inline in the notebook.)

### 2. Horizon calibration cell (new, placed right after the raw data load, before feature
engineering / label computation)
- Load raw OHLCV via `load_ohlcv(DS_NAME, n_rows=N_ROWS)`.
- Call `resolution_rate_by_horizon(df, atr_window=outcome_params['atr_window'], tp_mult=TP_MULT,
  sl_mult=SL_MULT, horizon_candidates=MAX_HORIZON_CANDIDATES)`, display the table.
- Pick the smallest candidate meeting `TARGET_RESOLUTION_RATE` on both sides (fallback: largest
  candidate + printed warning). Assign into `outcome_params['max_horizon']`.

### 3. Label computation cell — apply Phase 1 changes
- Use `calculate_trade_outcomes_capped(df, **outcome_params)` (now includes `buy_class`/
  `sell_class`).
- `long_quality`/`short_quality` = `np.clip(np.log1p(MFE/(MAE+eps)), 0, TARGET_CLIP_MAX)` — no time
  penalty loop.
- Assign `df["buy_class"]`, `df["sell_class"]` from `outcomes`.
- Keep `buy_win`/`sell_win`/`signed_win`/`signed_quality` computation as today.

### 4. Feature selection + generated-module export (moved earlier than before)
- Keep `USE_FEATURE_SELECTION` branch calling `select_features_pipeline(...)` to get
  `selected_features` — unchanged.
- Immediately call:
  ```python
  ds_title = DS_NAME.split("/")[-1].split(".")[0]
  feature_file_path, feature_file_hash = write_generated_feature_module(
      selected_features, output_dir="ModelPacks/generated_features", dataset_slug=ds_title,
  )
  compute_features = load_generated_feature_module(feature_file_path, feature_file_hash)
  ```
- Replace the current "re-generate features from FRESH raw data with only selected columns"
  block with a call to `compute_features(df_raw_fresh, include_mtf=True,
  regime_params=regime_params)` — this becomes the single source of truth used for both training
  and (via the pack) for backtest/live.
- If `USE_FEATURE_SELECTION` is False, still call `write_generated_feature_module` using whatever
  the full/legacy feature column list is (so the pack always has a `feature_file_path` — do not
  special-case pack export around a missing generated module).

### 5. Preprocess/split cell — build classifier + regressor arrays
After the existing `preprocess_ohlcv` call producing `X_train`/`X_holdout` and the existing
`TARGET_COLS`/`CLASSIFICATION_TARGETS` arrays:
- Build resolved-only masks: `train_resolved_long = df_train["buy_class"] != 2`,
  `train_resolved_short = df_train["sell_class"] != 2` (and the equivalent for holdout), aligned to
  the same row order as `X_train`/`X_holdout` (they are, since `preprocess_ohlcv` preserves order
  and only drops NaN rows the same way for both — verify row counts match before indexing; if
  `preprocess_ohlcv` drops rows, filter the class arrays through the same `dropna` mask it uses,
  which is available via `proc_df_train`/`proc_df_holdout`'s index).
- Regressor training data for `long_quality`: `X_train[train_resolved_long.values]`,
  `target_arrays["long_quality"]["train"][train_resolved_long.values]` (mirror for short/holdout).
- Classifier training data for each side: full `X_train`/`X_holdout`, target = `buy_class`/
  `sell_class` (3-class int array), NOT filtered.

### 6. Add `tune_lgbm_classifier` to `Learn/train_utils.py`
Mirror `tune_lgbm_by_spearman`'s structure (same walk-forward loop, same `candidate_params`/
`base_params` shape) but:
- Fit `LGBMClassifier(objective="multiclass", num_class=3, **{**base_params, **cfg})` instead of
  `LGBMRegressor`.
- Score each fold with macro `average_precision_score` computed one-vs-rest for the TP class (class
  label 1), or macro log-loss — pick ONE metric and document it in the function's docstring; macro
  PR-AUC for the TP class is recommended since it best reflects "how well can we identify winning
  setups", which is what drives `pred_long`/`pred_short` at inference.
- Return the same `(results_df, best_cfg_dict)` shape as `tune_lgbm_by_spearman` so the training
  notebook can reuse it symmetrically.

### 7. Tuning cell
- Keep calling `tune_lgbm_by_spearman` for the two magnitude regressors (resolved-only rows from
  step 5).
- Add calls to `tune_lgbm_classifier` for the two 3-class classifiers (full rows).

### 8. Final LGBM training cell
- Train `classifier_long`, `classifier_short` (3-class LGBMClassifier, tuned config from step 7) on
  full train rows.
- Train `regressor_long`, `regressor_short` (LGBMRegressor, tuned config) on resolved-only train
  rows.
- Evaluate holdout: classifier via macro PR-AUC / per-class precision-recall; regressor via
  `reg_metrics` on resolved-only holdout rows.
- Leave the existing `TRAIN_CLASSIFIER`/binary `buy_win`/`sell_win`/`signed_win` LGBMClassifier
  block in place as a secondary/legacy comparison (do not delete — no-deprecation rule) but add a
  short markdown note that it is retained for comparison and the 3-class classifiers above are now
  primary.

### 9. New DL candidate cell (PyTorch MLP, two-head)
- Requires `torch` — add `torch` (CPU wheel, e.g. `torch==2.x` matching whatever is
  pip-installable for the environment's Python version) to `requirements.txt`. Note: `Learn/Util.py`
  already imports `torch` today even though it is absent from `requirements.txt` and not installed
  in this environment — this phase is what finally makes that import valid; run `pip install torch`
  (or the pinned version added to requirements.txt) before executing this cell.
- Architecture: for each side (long/short) independently (matches the LGBM structure — do not
  build one combined 6-output model, keep it simple/symmetric): a small `nn.Module` with a shared
  trunk (`Linear -> ReLU -> Linear -> ReLU`, sizes from `DL_HIDDEN_SIZES`) branching into:
  - a 3-unit linear classification head (logits for SL/TP/timeout, trained with
    `nn.CrossEntropyLoss`),
  - a 1-unit linear regression head (trained with `nn.MSELoss`, masked to resolved rows only within
    the same batch — multiply the per-sample loss by a `resolved_mask` before averaging, or simply
    only include resolved rows' regression loss term; simplest correct approach: compute regression
    loss only over the resolved subset of each batch, skip the term for batches with zero resolved
    rows).
- Train with `DL_EPOCHS`/`DL_LR`/`DL_BATCH_SIZE` from config, plain Adam optimizer, on the same
  `X_train`/holdout arrays already built (`torch.from_numpy(...).float()`).
- Evaluate with the same `reg_metrics`/`safe_spearman` (magnitude head on resolved holdout rows) and
  macro PR-AUC (classification head, all holdout rows) so results are directly comparable to LGBM's
  in the same units/metrics.

### 10. Model comparison cell
- Build a small comparison table: rows = {lgbm, mlp}, columns = {holdout Spearman (magnitude),
  holdout macro PR-AUC (classifier)}, per side.
- Pick `model_choice["long"]` / `model_choice["short"]` = whichever of `"lgbm"`/`"mlp"` has the
  higher holdout Spearman for that side (document this as the selection rule; PR-AUC as a
  tie-breaker if Spearman values are within 0.01 of each other). Store both models in the pack
  regardless of which wins.

### 11. Export cell — new pack schema
Replace the current `feature_function`/`feature_function_source`/pickled-closure fields with:
- `feature_file_path` (string), `feature_file_hash` (string), `selected_features` (list, kept for
  inspection).
- `classifiers = {"long": {"model": classifier_long, "type": "lgbm"}, "short": {...}}` (add the MLP
  variants alongside if trained, e.g. `classifiers["long"]["mlp_model"]` — keep both LGBM and MLP
  objects retrievable regardless of `model_choice`).
- `regressors = {"long": {"model": regressor_long, "type": "lgbm"}, "short": {...}}` with the same
  pattern for MLP.
- `model_choice = {"long": "lgbm"|"mlp", "short": ...}`.
- `horizon_calibration_table` (the DataFrame from step 2, as a dict via `.to_dict(orient="records")`).
- Keep everything else already exported (`model_info`, `preprocess_function`, `preprocess_args`,
  `scaler`, `regime_params`, `outcome_params`, `data_split`, `validation` block) — these are
  unaffected by this refactor and should remain for parity with the backtest notebook's existing
  expectations, updated only where the schema changes above require it (e.g. `validation` should
  now report both classifier and regressor holdout metrics per side/per model).

## Verification
- Run the notebook end-to-end on `../data/EURUSD_M1_520weeks.csv` (or whichever `DS_NAME` is
  currently configured).
- Confirm a `.pkl` is written to `ModelWorkbench/ModelPacks/` and a generated feature file exists
  under `ModelWorkbench/ModelPacks/generated_features/`.
- Confirm the model comparison table shows both `lgbm` and `mlp` rows with sane (non-NaN, non-inf)
  metrics for both `long` and `short` sides.
- Sanity check `classifiers["long"]["model"].predict_proba(X_holdout)[:, 1]` (TP-class probability,
  assuming class label 1 = TP as encoded in Phase 1) returns values in [0, 1].
