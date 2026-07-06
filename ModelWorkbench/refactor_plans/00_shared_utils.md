# Phase 0 — Shared Utilities Consolidation

Status: IMPLEMENTED (this session)

## Goal
Remove duplicated helper code that is copy-pasted almost identically at the top of both
`2.1 Train LGBM Regression Model.ipynb` and `3.1 Backtest LGBM Regression.ipynb`, and centralize
it in `ModelWorkbench/Learn/train_utils.py` so a fix in one place benefits both notebooks.

## What moved
From the training notebook's first code cell into `Learn/train_utils.py` (unchanged behavior,
copied verbatim):
- `load_ohlcv(ds_name, n_rows=None)`
- `safe_spearman(y_true, y_pred)`
- `reg_metrics(y_true, y_pred)`
- `lgbm_spearman_eval(y_true, y_pred)`
- `walk_forward_folds(n_samples, min_train_size, test_size, gap_size, n_folds)`
- `describe_target_frame(frame, target_cols)`
- `tune_lgbm_by_spearman(X, y, folds, candidate_params, base_params, sample_weight=None, early_stopping_rounds=200, robustness_penalty=0.25)`

Only the training notebook's first cell duplicated these functions inline. It now does:
```python
from Learn.train_utils import (
    load_ohlcv, safe_spearman, reg_metrics,
    walk_forward_folds, describe_target_frame, tune_lgbm_by_spearman,
)
```
instead of redefining these functions inline. (`lgbm_spearman_eval` is imported into
`train_utils.py` from nowhere else — it's defined there and used internally by
`tune_lgbm_by_spearman`; the notebook itself never calls it directly, so it is intentionally NOT
re-imported into the notebook namespace.)

The backtest notebook (`3.1`) never duplicated these helpers (it only defines a `FEATURES` shim for
feature engineering, addressed separately in Phase 2/4), so no change was needed there for Phase 0.

## Verification performed
- `python -c "from Learn import train_utils"` import check from `ModelWorkbench/` directory.
- Confirmed function signatures/bodies match the originals exactly (no behavior change).
- Notebooks were NOT executed end-to-end in this pass (requires the real CSV datasets / LightGBM
  training run, which is expensive) — a future implementer should run both notebooks top-to-bottom
  once before trusting this phase fully, and diff holdout metrics against a pre-refactor run if one
  is available.

## Notes for next phases
- `Learn/train_utils.py` is the designated home for any further shared training/eval helper
  functions (e.g. the new `tune_lgbm_classifier` function required by Phase 3).
- Do not duplicate these functions back into notebooks.
