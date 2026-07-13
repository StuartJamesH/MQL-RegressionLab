# Implementation Plan: TP_MULT / SL_MULT Grid Search Notebook

## Goal
Create a new Jupyter notebook at `ModelWorkbench/2.2 SL TP Grid Search.ipynb` that tests a range of `SL_MULT` and `TP_MULT` values, evaluates LGBM regression model performance for each combination using walk-forward cross-validation and holdout evaluation, then reports the optimal combination.

**Constraint:** `SL_MULT >= 2.0` always.

---

## 1. Notebook Structure (6 cells)

### Cell 1: Markdown title + description
```
# SL × TP Multiplier Grid Search
Tests a grid of SL_MULT and TP_MULT values to find the optimal stop-loss / take-profit
multiplier combination for LGBM regression models. Evaluates each combination via
walk-forward cross-validation Spearman correlation and holdout metrics.
```

### Cell 2: Imports
```python
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
from lightgbm import LGBMRegressor
from talib import ATR

from Learn.features import add_feature_library
from Learn.labels import calculate_trade_outcomes_capped, resolution_rate_by_horizon
from Learn.preprocess import preprocess_ohlcv
from Learn.train_utils import (
    load_ohlcv, safe_spearman, reg_metrics,
    walk_forward_folds, describe_target_frame, tune_lgbm_by_spearman,
)

warnings.filterwarnings("ignore")
```

### Cell 3: Configuration + Grid Definition
```python
# ========== CORE CONFIGURATION ==========
DS_NAME = "../data/XAUUSD_M1_520weeks.csv"
N_ROWS = 1_000_000  # Use fewer for speed (e.g. 500_000) if needed

# --- Walk-forward validation settings ---
MAX_TUNING_ROWS = 350_000
WFO_FOLDS = 4
WFO_MIN_TRAIN_FRAC = 0.55
WFO_TEST_FRAC = 0.10
WFO_GAP_ROWS = 60
EARLY_STOPPING_ROUNDS = 200
ROBUSTNESS_PENALTY = 0.25

# --- Horizon calibration ---
MAX_HORIZON_CANDIDATES = [20, 30, 45, 60]
TARGET_RESOLUTION_RATE = 0.90

# --- Train/holdout split ---
TRAIN_FRACTION = 0.90

# --- Target columns ---
TARGET_COLS = ["long_quality", "short_quality", "signed_quality"]
PRIMARY_TARGET = "signed_quality"

# --- Feature engineering ---
SELECTED_FEATURE_COUNT = 80  # Set to None to skip feature selection
USE_FEATURE_SELECTION = True  # Set to False to skip for speed

# --- Regime params (must match 2.1) ---
regime_params = {
    "ma_period": 90,
    "slope_smoothness": 50,
    "regime_min_duration": 0,
    "atr_window": 60,
    "atr_lookback": 720,
    "atr_percentile": 0.0,
    "slope_threshold": 0,
}

# --- LGBM base params (same as 2.1) ---
LGBM_BASE_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "max_depth": -1,
    "subsample_freq": 1,
}

LGBM_CANDIDATES = [
    dict(learning_rate=0.03, num_leaves=63, min_child_samples=60, subsample=0.80, colsample_bytree=0.80, reg_alpha=0.10, reg_lambda=0.50, min_split_gain=0.00),
    dict(learning_rate=0.02, num_leaves=127, min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.20, reg_lambda=1.00, min_split_gain=0.00),
    dict(learning_rate=0.01, num_leaves=255, min_child_samples=120, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.50, reg_lambda=3.00, min_split_gain=0.00),
    dict(learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=5.00, min_split_gain=0.00),
    dict(learning_rate=0.025, num_leaves=63, min_child_samples=40, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.10, reg_lambda=2.00, min_split_gain=0.01),
    dict(learning_rate=0.02, num_leaves=95, min_child_samples=50, subsample=0.80, colsample_bytree=0.75, reg_alpha=0.00, reg_lambda=1.00, min_split_gain=0.00),
]

# ========== GRID DEFINITION ==========
# SL_MULT must be >= 2.0
SL_MULT_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0]
TP_MULT_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Generate all combinations where SL_MULT >= 2.0
grid = [(tp, sl) for sl in SL_MULT_VALUES for tp in TP_MULT_VALUES if sl >= 2.0]

# --- Speed mode: if True, use a single LGBM candidate only (fastest) ---
# If False, runs full hyperparameter tuning (6 candidates × 4 folds) per grid point
SPEED_MODE = False  # Set to True for faster grid scan (~1 model per combo)

print(f"Dataset: {DS_NAME}")
print(f"Rows: {N_ROWS}")
print(f"Grid size: {len(grid)} combinations")
print(f"SL_MULT range: {SL_MULT_VALUES}")
print(f"TP_MULT range: {TP_MULT_VALUES}")
print(f"Speed mode: {SPEED_MODE} ({'1 LGBM config' if SPEED_MODE else '6-candidate tuning'} per combo)")
```

### Cell 4: Load Data + Generate Features (ONCE, cached)
```python
# ========== LOAD & FEATURE ENGINEER (ONE-TIME) ==========
# This cell loads data and generates features once.
# Features are independent of TP/SL, so they're reusable across all grid points.

from Learn.features import add_feature_library, select_features_pipeline
from Learn.feature_codegen import write_generated_feature_module, load_generated_feature_module

def load_and_feature_engineer(ds_name, n_rows, regime_params, selected_feature_count, use_feature_selection):
    """Load OHLCV data and generate features. Returns df_raw, feature_func, selected_features list."""
    print(f"Loading {n_rows} rows from {ds_name}...")
    df_raw = load_ohlcv(ds_name, n_rows=n_rows)
    ds_title = ds_name.split("/")[-1].split(".")[0]
    print(f"Loaded {len(df_raw):,} rows: {df_raw['Time'].min()} -> {df_raw['Time'].max()}")

    print("Generating feature library...")
    df_feat = add_feature_library(df_raw.copy(), include_mtf=True, regime_params=regime_params)
    print(f"Full feature library: {len(df_feat.columns)} columns")

    if use_feature_selection and selected_feature_count and selected_feature_count > 0:
        print(f"Running feature selection (target={selected_feature_count})...")
        imported = False
        selected_features = select_features_pipeline(
            df_feat.tail(min(500_000, len(df_feat))),
            target_col=None,  # will use a temporary target
            n_features=selected_feature_count,
            correlation_threshold=0.95,
            include_mtf=True,
            regime_params=regime_params,
        )
        print(f"Selected {len(selected_features)} features")
    else:
        _ohlcv_cols = {"Time", "Open", "High", "Low", "Close", "Volume"}
        selected_features = [c for c in df_feat.columns if c not in _ohlcv_cols]
        print(f"Feature selection disabled; keeping all {len(selected_features)} library features")

    return df_raw, selected_features

print("=" * 60)
print("LOADING DATA & GENERATING FEATURES (one-time)")
print("=" * 60)
df_raw, selected_features = load_and_feature_engineer(
    DS_NAME, N_ROWS, regime_params, SELECTED_FEATURE_COUNT, USE_FEATURE_SELECTION
)
print(f"Feature matrix will have {len(selected_features)} columns")

# Store raw data for reuse - each grid point will:
# 1. Start from df_raw
# 2. Build features (using selected_features list)
# 3. Compute ATR and labels with its own TP/SL params
```

**IMPORTANT IMPLEMENTATION NOTES:**
- The `select_features_pipeline` call above needs a temporary target. Since we don't know the final TP/SL yet, use the approach from 2.1 lines 438-458: compute a quick interim target with default TP=2.5, SL=2.5 for feature selection purposes only. The code should:
  1. Compute ATR on df_feat: `df_feat["atr"] = ATR(df_feat["High"], df_feat["Low"], df_feat["Close"], timeperiod=14)`
  2. Compute temporary outcomes with `tp_mult=2.5, sl_mult=2.5, max_horizon=30`: `outcomes_fs = calculate_trade_outcomes_capped(df_feat, atr_window=60, tp_mult=2.5, sl_mult=2.5, max_horizon=30)`
  3. Derive temporary signed_quality: `long_q_fs = np.log1p(outcomes_fs["buy_MFE"] / (outcomes_fs["buy_MAE"] + 1e-8))` etc.
  4. Call `select_features_pipeline(fs_subset, target_col='signed_quality', ...)` with the temp target

### Cell 5: Grid Search Loop (main computation)
```python
# ========== GRID SEARCH ==========
# For each (TP_MULT, SL_MULT) pair:
#   a) Compute trade outcomes (labels) with those multipliers
#   b) Calibrate max_horizon
#   c) Derive quality targets
#   d) Preprocess features
#   e) Tune/train LGBM with walk-forward folds
#   f) Evaluate on holdout
#   g) Record metrics

results = []

for idx, (tp_mult, sl_mult) in enumerate(grid):
    print(f"\n{'='*60}")
    print(f"Grid point {idx+1}/{len(grid)}: TP={tp_mult}, SL={sl_mult}  (RR={tp_mult/sl_mult:.2f})")
    print(f"{'='*60}")

    # === a) Horizon calibration ===
    outcomes_calib = calculate_trade_outcomes_capped(
        df_raw, atr_window=60, tp_mult=tp_mult, sl_mult=sl_mult, max_horizon=30,  # use 30 for calibration
    )
    horizon_table = resolution_rate_by_horizon(
        df_raw, atr_window=60, tp_mult=tp_mult, sl_mult=sl_mult,
        horizon_candidates=MAX_HORIZON_CANDIDATES,
    )
    _qualifying = horizon_table[
        (horizon_table["buy_resolved_rate"] >= TARGET_RESOLUTION_RATE)
        & (horizon_table["sell_resolved_rate"] >= TARGET_RESOLUTION_RATE)
    ]
    if len(_qualifying) > 0:
        chosen_horizon = int(_qualifying.index.min())
    else:
        chosen_horizon = int(horizon_table.index.max())

    outcome_params = {
        "atr_window": 60,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "max_horizon": chosen_horizon,
    }
    # Re-compute outcomes with calibrated max_horizon
    outcomes = calculate_trade_outcomes_capped(df_raw, **outcome_params)

    # === b) Derive quality targets ===
    # Build features on df_raw, add ATR, compute targets
    # NOTE: we rebuild features from df_raw each time but only using selected_features.
    # The function add_feature_library is called on fresh df_raw, then we subset to selected_features.
    # This mirrors the 2.1 pattern.
    df_feat = add_feature_library(df_raw.copy(), include_mtf=True, regime_params=regime_params)
    df_feat["atr"] = ATR(df_feat["High"], df_feat["Low"], df_feat["Close"], timeperiod=14)
    
    # Build quality targets
    TARGET_CLIP_MAX = max(tp_mult, sl_mult) * 1.5
    eps = 1e-8
    long_q = np.log1p(outcomes["buy_MFE"] / (outcomes["buy_MAE"] + eps))
    short_q = np.log1p(outcomes["sell_MFE"] / (outcomes["sell_MAE"] + eps))
    long_q = np.clip(long_q, 0.0, TARGET_CLIP_MAX)
    short_q = np.clip(short_q, 0.0, TARGET_CLIP_MAX)

    df_feat["long_quality"] = long_q
    df_feat["short_quality"] = short_q
    df_feat["signed_quality"] = df_feat["long_quality"] - df_feat["short_quality"]

    # Keep only selected features + targets + Time for splitting
    keep_cols = selected_features + TARGET_COLS + ["Time"]
    df_model = df_feat[keep_cols].copy()

    # === c) Train/holdout split ===
    split_idx = int(len(df_model) * TRAIN_FRACTION)
    df_train = df_model.iloc[:split_idx].copy().reset_index(drop=True)
    df_holdout = df_model.iloc[split_idx:].copy().reset_index(drop=True)

    # === d) Preprocess ===
    preprocess_ohlcv_args = {
        "target_col": TARGET_COLS,
        "shift": 0,
        "onehot_prefixes": ["OH_"],
        "price_prefixes": ["PR_"],
    }
    X_train, y_train, scaler, features, _, _ = preprocess_ohlcv(
        df_train, **preprocess_ohlcv_args, scaler=None, return_df=True,
    )
    X_holdout, y_holdout, _, _, _, _ = preprocess_ohlcv(
        df_holdout, **preprocess_ohlcv_args, scaler=scaler, return_df=True,
    )

    # Build target arrays
    target_arrays = {
        target: {"train": y_train[:, i], "holdout": y_holdout[:, i]}
        for i, target in enumerate(TARGET_COLS)
    }

    # === e) Setup tuning fold indices ===
    tune_rows = min(len(X_train), MAX_TUNING_ROWS)
    X_tune = X_train[-tune_rows:]
    tune_targets = {k: v["train"][-tune_rows:] for k, v in target_arrays.items()}

    tune_test_size = max(20_000, int(tune_rows * WFO_TEST_FRAC))
    tune_min_train = max(100_000, int(tune_rows * WFO_MIN_TRAIN_FRAC))
    tune_min_train = min(tune_min_train, tune_rows - tune_test_size - WFO_GAP_ROWS)
    while tune_min_train + tune_test_size + WFO_GAP_ROWS > tune_rows and tune_test_size > 10_000:
        tune_test_size = int(tune_test_size * 0.8)
        tune_min_train = min(tune_min_train, tune_rows - tune_test_size - WFO_GAP_ROWS)

    wf_folds = walk_forward_folds(
        n_samples=tune_rows,
        min_train_size=tune_min_train,
        test_size=tune_test_size,
        gap_size=max(WFO_GAP_ROWS, chosen_horizon),
        n_folds=WFO_FOLDS,
    )

    # === f) Tune/train & evaluate per target ===
    grid_results = {"tp_mult": tp_mult, "sl_mult": sl_mult, "rr": round(tp_mult / sl_mult, 4),
                     "max_horizon": chosen_horizon}

    for target_name in TARGET_COLS:
        print(f"  Tuning: {target_name}...", end=" ")
        if SPEED_MODE:
            # Fast: use a single config (first candidate) with fixed n_estimators
            single_cfg = LGBM_CANDIDATES[0].copy()
            single_cfg["cfg_id"] = 1
            candidates = [single_cfg]
        else:
            candidates = LGBM_CANDIDATES

        results_df, best_cfg = tune_lgbm_by_spearman(
            X=X_tune,
            y=tune_targets[target_name],
            folds=wf_folds,
            candidate_params=candidates,
            base_params=LGBM_BASE_PARAMS,
            sample_weight=None,  # no tradeability weighting in grid search (simplification)
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            robustness_penalty=ROBUSTNESS_PENALTY,
        )

        best_n_estimators = int(best_cfg["median_best_iteration"])
        cv_mean_spearman = float(best_cfg["mean_spearman"])
        cv_robust = float(best_cfg["robust_score"])

        # Train final model on full training set
        final_params = {
            **LGBM_BASE_PARAMS,
            **{k: v for k, v in best_cfg.items() if k in LGBM_CANDIDATES[0]},
            "n_estimators": best_n_estimators,
        }
        model = LGBMRegressor(**final_params)
        model.fit(X_train, target_arrays[target_name]["train"])
        pred_holdout = model.predict(X_holdout)
        holdout_metrics = reg_metrics(target_arrays[target_name]["holdout"], pred_holdout)

        # Store
        grid_results[f"{target_name}_cv_spearman"] = cv_mean_spearman
        grid_results[f"{target_name}_cv_robust"] = cv_robust
        grid_results[f"{target_name}_holdout_spearman"] = float(holdout_metrics["Spearman"])
        grid_results[f"{target_name}_holdout_r2"] = float(holdout_metrics["R2"])
        grid_results[f"{target_name}_holdout_rmse"] = float(holdout_metrics["RMSE"])
        grid_results[f"{target_name}_n_estimators"] = best_n_estimators

        print(f"CV Spearman={cv_mean_spearman:.4f}  Holdout Spearman={holdout_metrics['Spearman']:.4f}")

    results.append(grid_results)
    print(f"  -> signed_quality CV Spearman={grid_results['signed_quality_cv_spearman']:.4f}")

print(f"\n{'='*60}")
print("GRID SEARCH COMPLETE")
print(f"{'='*60}")
```

**CRITICAL IMPLEMENTATION NOTE:** The feature selection cell above uses a TEMPORARY target. But the approach in cell 4 should follow EXACTLY the pattern from 2.1 lines 437-458:
1. Add features to df_raw: `df_feat = add_feature_library(df_raw.copy(), include_mtf=True, regime_params=regime_params)`
2. Add ATR: `df_feat["atr"] = ATR(df_feat["High"], df_feat["Low"], df_feat["Close"], timeperiod=14)`
3. Compute temp outcomes with default TP/SL: `outcomes_fs = calculate_trade_outcomes_capped(df_feat, atr_window=60, tp_mult=2.5, sl_mult=2.5, max_horizon=30)`
4. Derive temp targets, then call `select_features_pipeline(..., target_col='signed_quality', ...)`

The returned `selected_features` list is reused for all grid points. Each grid point rebuilds features from `df_raw` but only keeps those columns.

**PERFORMANCE NOTE:** In the grid loop, `add_feature_library(df_raw.copy(), ...)` is called for EVERY grid point. This is O(203 columns) each time and is the bottleneck. To optimize:
- Option A (complex): Cache the full feature dataframe from cell 4, then in the loop just subset columns. However, we need targets merged in, and `preprocess_ohlcv` expects raw-ish columns (OHLCV + features + targets + Time). So we could build the full feature matrix once, then in the loop: recompute ATR + labels, replace target columns.
- Option B (simple, recommended for v1): Just rebuild features each iteration. With 30 grid points this takes a few minutes but is safer and simpler.

### Cell 6: Results Summary & Visualization
```python
# ========== RESULTS ==========
results_df = pd.DataFrame(results)

# Sort by primary metric: signed_quality holdout Spearman
results_df = results_df.sort_values("signed_quality_holdout_spearman", ascending=False).reset_index(drop=True)

print("=" * 60)
print("GRID SEARCH RESULTS (sorted by signed_quality holdout Spearman)")
print("=" * 60)

# Display top results
display_cols = [
    "tp_mult", "sl_mult", "rr", "max_horizon",
    "signed_quality_cv_spearman", "signed_quality_cv_robust",
    "signed_quality_holdout_spearman", "signed_quality_holdout_r2",
    "long_quality_holdout_spearman", "short_quality_holdout_spearman",
]
display(results_df[display_cols].round(4))

# === Best combination ===
best = results_df.iloc[0]
print(f"\n{'='*60}")
print(f"OPTIMAL COMBINATION: TP_MULT={best['tp_mult']}, SL_MULT={best['sl_mult']}")
print(f"  R:R = {best['tt']:.2f}")
print(f"  Signed quality holdout Spearman = {best['signed_quality_holdout_spearman']:.4f}")
print(f"  Signed quality CV robust score = {best['signed_quality_cv_robust']:.4f}")
print(f"  Long quality holdout Spearman  = {best['long_quality_holdout_spearman']:.4f}")
print(f"  Short quality holdout Spearman = {best['short_quality_holdout_spearman']:.4f}")
print(f"  Max horizon = {int(best['max_horizon'])}")
print(f"{'='*60}")

# === Heatmap visualization ===
# Pivot table for signed_quality holdout Spearman
pivot = results_df.pivot_table(
    values="signed_quality_holdout_spearman",
    index="sl_mult",
    columns="tp_mult",
    aggfunc="first",
)

fig = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=pivot.index,
    colorscale="RdYlGn",
    text=[[f"{v:.4f}" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    textfont={"size": 12},
    colorbar=dict(title="Holdout Spearman"),
))
fig.update_layout(
    title="Signed Quality Holdout Spearman by SL_MULT × TP_MULT",
    xaxis_title="TP_MULT",
    yaxis_title="SL_MULT",
    height=450,
    width=600,
)
fig.show()

# === Secondary heatmap: CV robust score ===
pivot_cv = results_df.pivot_table(
    values="signed_quality_cv_robust",
    index="sl_mult",
    columns="tp_mult",
    aggfunc="first",
)

fig2 = go.Figure(data=go.Heatmap(
    z=pivot_cv.values,
    x=pivot_cv.columns,
    y=pivot_cv.index,
    colorscale="RdYlGn",
    text=[[f"{v:.4f}" for v in row] for row in pivot_cv.values],
    texttemplate="%{text}",
    textfont={"size": 12},
    colorbar=dict(title="CV Robust Score"),
))
fig2.update_layout(
    title="Signed Quality CV Robust Score by SL_MULT × TP_MULT",
    xaxis_title="TP_MULT",
    yaxis_title="SL_MULT",
    height=450,
    width=600,
)
fig2.show()
```

---

## 2. Key Design Decisions

### 2.1 What to simplify vs the full 2.1 notebook
The full notebook includes: feature selection, tradeability weighting, two-head (classifier + regressor), MLP candidate, ensembles, model pack export. For a grid search, we simplify to:
| Feature | Included? | Why |
|---|---|---|
| Feature engineering (`add_feature_library`) | Yes | Needed for any model |
| Feature selection | Yes (optional) | Reuse selected features across grid |
| Horizon calibration | Yes | Each TP/SL yields different max_horizon |
| Walk-forward fold generation | Yes | Core evaluation methodology |
| LGBM hyperparameter tuning | Yes (or speed mode) | Core to fair comparison |
| Tradeability weighting | No | Adds significant time; not critical for comparison |
| Two-head model (classifier + regressor) | No | Complexity not needed for grid comparison |
| MLP candidate | No | Grid would be prohibitively slow |
| Ensemble | No | Not needed for relative comparison |
| Model pack export | No | This is analysis, not production training |

### 2.2 Metric hierarchy
The primary metric for ranking combinations is `signed_quality_holdout_spearman`. Secondary metrics:
- `signed_quality_cv_robust` (CV robust score)
- `long_quality_holdout_spearman`
- `short_quality_holdout_spearman`

### 2.3 SL_MULT >= 2.0 constraint
The grid loop already filters: `for sl in SL_MULT_VALUES ... if sl >= 2.0`. This is enforced programmatically.

### 2.4 Speed considerations
With `SPEED_MODE=True`: ~1 LGBM model per target per grid point → ~90 fits for 30 grid points.  
With `SPEED_MODE=False`: ~6 candidates × 4 folds = 24 fits per target per grid point → ~2160 fits for 30 grid points.  
Set `SPEED_MODE=False` in the notebook by default so it can be changed by the user.

---

## 3. File Path
Create the notebook at:
```
/home/stuart/code/MQL-RegressionLab/ModelWorkbench/2.2 SL TP Grid Search.ipynb
```

---

## 4. Notes for the Coding Agent

1. **Use the `nbformat` library** to construct the `.ipynb` JSON programmatically in Python, or write it directly as a JSON file matching the structure of `2.1 Train LGBM Regression Model.ipynb`. The 2.1 notebook uses `nbformat` v4 format (standard Jupyter).

2. **Every cell** needs `"execution_count": null` since it's a fresh notebook.

3. **The feature selection in Cell 4** needs a temporary target. Follow the exact pattern from 2.1 lines 437-458 (in the "Load Data, Generate Features, and Compute Targets" cell). See the IMPORTANT IMPLEMENTATION NOTES in Cell 4 above for exact code.

4. **In Cell 5**, call `add_feature_library(df_raw.copy(), ...)` for each grid point. This is needed because labels (targets) are recomputed per grid point. While it's redundant to regenerate 203 features each time, it's the simplest approach. If the implementer wants to optimize, they can cache the full feature dataframe and just replace target columns.

5. **The `walk_forward_folds` function** signature is: `walk_forward_folds(n_samples, min_train_size, test_size, gap_size, n_folds)`. It returns a list of `(train_indices, val_indices)` arrays. See `Learn/train_utils.py` for exact implementation.

6. **The `tune_lgbm_by_spearman` function** signature is: `tune_lgbm_by_spearman(X, y, folds, candidate_params, base_params, sample_weight=None, early_stopping_rounds=200, robustness_penalty=0.25)`. Returns `(results_df, best_cfg_dict)`. See `Learn/train_utils.py`.

7. **`results_df`** from `tune_lgbm_by_spearman` has columns: `cfg_id, mean_spearman, std_spearman, robust_score, mean_r2, mean_rmse, median_best_iteration` plus all the hyperparameter columns.

8. **`reg_metrics`** returns a dict with keys: `MAE, RMSE, R2, Spearman`.

9. **import `resolution_rate_by_horizon` from `Learn.labels`**. It takes `(df, atr_window, tp_mult, sl_mult, horizon_candidates)` and returns a DataFrame.

10. **No model pack export** is needed. This is pure analysis.

11. **`add_feature_library`** from `Learn.features` takes `(df, include_mtf=True, regime_params=None)` and returns a DataFrame with added feature columns.

12. **`select_features_pipeline`** from `Learn.features` takes `(df, target_col, n_features=80, correlation_threshold=0.95, include_mtf=True, regime_params=None)` and returns a list of selected feature column names.

13. **`preprocess_ohlcv`** from `Learn.preprocess` takes `(df, target_col, shift=0, onehot_prefixes=['OH_'], price_prefixes=['PR_'], scaler=None, return_df=False, outcomes_col=None)` and returns `(X, y, scaler, feature_names, [proc_df])`. The last element is only returned when `return_df=True`. Check the exact return signature in `Learn/preprocess.py`.

14. Each cell must be a valid `nbformat` cell object with `"cell_type"`, `"metadata"`, and `"source"`. Markdown cells use `"cell_type": "markdown"`, code cells use `"cell_type": "code"` with `"outputs": []` and `"execution_count": null`.

15. **The notebook should be runnable as-is** from the ModelWorkbench directory, matching the same working directory assumptions as 2.1 (i.e., `from Learn.features import ...` imports from `ModelWorkbench/Learn/`).
