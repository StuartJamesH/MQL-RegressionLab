# Phase 4 — Backtest Notebook Rewrite (`3.1 Backtest LGBM Regression.ipynb`)

Status: NOT STARTED

Depends on: Phase 3 (new pack schema must exist — this phase consumes packs produced by the
rewritten training notebook; it will NOT work against old-schema `.pkl` files).

## Combined prediction scheme (implement exactly this, so downstream cells need no changes)
Per side (`long`/`short`):
- `pred_long` = P(TP | long) from the long classifier (class label 1 = TP, per Phase 1 encoding).
- `pred_short` = P(TP | short) from the short classifier.
- `quality_long` / `quality_short` = magnitude regressor output for that side (only meaningful
  when a trade resolves, but the regressor always returns a numeric prediction).
- `expected_long = pred_long * quality_long - P(SL | long) * 1.0` (SL loss defined as −1R by
  construction; `P(SL|long)` is the classifier's class-0 probability).
- `expected_short` computed the same way for the short side.
- `pred_signed = expected_long - expected_short`.
- `quality` (used by `QUALITY_THRESHOLD` downstream) = `np.abs(pred_signed)`, matching the existing
  `prediction_to_signal` cell's `quality = abs(row['pred_signed'])` line — no change needed there.

Keep the column names `pred_long`, `pred_short`, `pred_signed`, `quality` identical to today's
`pred_df` so `prediction_to_signal`, `simulate_production_trades`, the threshold sweep, and all
charting cells require ZERO changes.

## Steps

### 1. Remove the `__main__` FEATURES shim and lambda-detection fallback
Delete the notebook's first cell content that defines `FEATURES` in `__main__` for pickle
resolution, and delete the "Reconstruct a reliable feature function..." / lambda-detection block
in the "Load Model Pack" cell. These existed only to work around pickled-closure fragility, which
Phase 2/3 eliminates.

### 2. Load features via the generated module
In the "Load Model Pack" cell, after loading `pack`:
```python
from Learn.feature_codegen import load_generated_feature_module

feature_file_path = pack["feature_file_path"]
feature_file_hash = pack["feature_file_hash"]
compute_features = load_generated_feature_module(feature_file_path, feature_file_hash)
```
Use `compute_features` exactly where `feature_fn` was previously used (the "Load and Prepare
Evaluation Data" cell: `df_feat = compute_features(df_raw.copy(), include_mtf=True,
regime_params=regime_params)`). Remove the `pack_features_are_processed` schema-guessing logic and
the "enforce raw feature parity" column-dropping block — the generated module's output is always
exactly `['Time','Open','High','Low','Close','Volume'] + selected_features` by construction, so
this defensive code is no longer needed. Keep the ATR-compatibility patch cell only if `'atr'` can
legitimately be a selected feature that the generated module might omit under some edge case —
otherwise remove it too; verify by checking whether `'atr'` ever appears in a real
`selected_features` list before deleting.

### 3. Replace `_predict_pack_outputs` with the new classifier+regressor dispatch
```python
def _predict_side(side, X):
    clf_info = pack["classifiers"][side]
    reg_info = pack["regressors"][side]
    choice = pack["model_choice"][side]

    clf_model = clf_info["model"] if choice == "lgbm" else clf_info["mlp_model"]
    reg_model = reg_info["model"] if choice == "lgbm" else reg_info["mlp_model"]

    if choice == "lgbm":
        proba = clf_model.predict_proba(X)  # columns ordered by class label: 0=SL,1=TP,2=timeout
        p_sl, p_tp = proba[:, 0], proba[:, 1]
        quality = reg_model.predict(X)
    else:
        # MLP path: forward pass, softmax the classification head, take scalar regression head.
        # Implement using the same model class/forward signature defined in the training notebook's
        # DL candidate cell (Phase 3, step 9) — import or redefine that class identically here so
        # the pickled state_dict (or full model object, whichever the pack stores) can be used.
        ...

    return p_tp, p_sl, quality


pred_long, p_sl_long, quality_long = _predict_side("long", X_eval)
pred_short, p_sl_short, quality_short = _predict_side("short", X_eval)
expected_long = pred_long * quality_long - p_sl_long
expected_short = pred_short * quality_short - p_sl_short
pred_signed = expected_long - expected_short
```
Adjust class-index assumptions (`proba[:, 0]` = SL, `proba[:, 1]` = TP) to match however
`LGBMClassifier` orders `classes_` after training on labels `{0, 1, 2}` (verify via
`clf_model.classes_` rather than assuming order — LightGBM sorts classes ascending by default for
integer labels, so `[0, 1, 2]` order should hold, but confirm this explicitly in code with an
assertion rather than assuming).

Build `pred_df` with the same columns as today (`Time, pred_long, pred_short, pred_signed`), adding
`quality_long`/`quality_short` as extra diagnostic columns if useful, but the four original columns
must be present with the same meanings consumed downstream.

### 4. Everything else unchanged
`prediction_to_signal`, `simulate_production_trades`, the P&L analysis cell, the threshold sweep,
the recommended-thresholds cell, and the charting cell all read `pred_long`/`pred_short`/
`pred_signed`/`quality` by column name only — leave them exactly as they are.

## Verification
- Run this notebook against a pack freshly exported by the rewritten Phase 3 training notebook.
- Confirm no `__main__`/lambda-rebinding code paths are hit (they should no longer exist in the
  file at all).
- Confirm `pred_long`/`pred_short` are within `[0, 1]` and look like a genuine probability
  distribution (e.g. histogram roughly matches the classifier's holdout PR-AUC characteristics from
  Phase 3's validation output) rather than the old unbounded regression-score behavior.
- Confirm trades/P&L/threshold-sweep cells execute without errors and produce a non-empty
  `trades` DataFrame on a reasonably sized `N_ROWS`.
