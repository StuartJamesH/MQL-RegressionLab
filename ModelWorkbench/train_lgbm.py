import argparse
import inspect
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from Learn.features import add_feature_library, select_features_pipeline
from Learn.labels import calculate_trade_outcomes_capped, generate_tradeability_scores, resolution_rate_by_horizon
from Learn.preprocess import preprocess_ohlcv
from Learn.train_utils import (
    load_ohlcv, safe_spearman, reg_metrics,
    walk_forward_folds, describe_target_frame, tune_lgbm_by_spearman, tune_lgbm_classifier,
)
from Learn.feature_codegen import write_generated_feature_module, load_generated_feature_module
from Learn.Util import TwoHeadMLP, train_two_head_mlp, predict_two_head_mlp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds-name", type=str, required=True, help="Path to OHLCV CSV dataset")
    parser.add_argument("--n-rows", type=int, default=1_000_000, help="Number of rows to load")
    parser.add_argument("--tp-mult", type=float, default=2.5, help="Take-profit multiplier")
    parser.add_argument("--sl-mult", type=float, default=2.5, help="Stop-loss multiplier")
    args = parser.parse_args()

    # ========== CORE CONFIGURATION ==========
    DS_NAME = args.ds_name
    N_ROWS = args.n_rows

    # --- Asset-specific settings ---
    TP_MULT = args.tp_mult
    SL_MULT = args.sl_mult
    TARGET_CLIP_MAX = max(TP_MULT, SL_MULT) * 1.5
    TRAIN_FRACTION = 0.90

    # --- Walk-forward validation ---
    MAX_TUNING_ROWS = 350_000
    WFO_FOLDS = 4
    WFO_MIN_TRAIN_FRAC = 0.55
    WFO_TEST_FRAC = 0.10
    WFO_GAP_ROWS = max(60, 30 * 2)  # At least 60 bars purge gap
    EARLY_STOPPING_ROUNDS = 200
    ROBUSTNESS_PENALTY = 0.25

    # --- Feature engineering ---
    USE_FEATURE_SELECTION = True
    SELECTED_FEATURE_COUNT = 80

    # --- Tradeability weighting ---
    USE_TRADEABILITY_WEIGHTING = True
    TRADEABILITY_N_FOLDS = 5
    TRADEABILITY_MIN_TRAIN = 100_000
    TRADEABILITY_TEST_SIZE = 50_000

    # --- Targets ---
    TARGET_COLS = ["long_quality", "short_quality", "signed_quality"]
    CLASSIFICATION_TARGETS = ["buy_win", "sell_win", "signed_win"]
    PRIMARY_TARGET = "signed_quality"
    TIME_PENALTY_LAMBDA = 0.1  # NOTE: no longer applied to targets (see two-head label redesign); kept for backward compatibility

    # --- Horizon calibration (picks outcome_params['max_horizon']) ---
    MAX_HORIZON_CANDIDATES = [20, 30, 45, 60]
    TARGET_RESOLUTION_RATE = 0.90

    # --- Two-head model candidates (classifier + resolved-only magnitude regressor) ---
    MODEL_CANDIDATES = ["lgbm", "mlp"]
    DL_HIDDEN_SIZES = (128, 64)
    DL_EPOCHS = 60
    DL_LR = 1e-3
    DL_BATCH_SIZE = 4096

    # --- Optional model enhancements ---
    TRAIN_CLASSIFIER = True
    ENSEMBLE_SIZE = 5
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 1111]

    # --- Regime params ---
    regime_params = {
        "ma_period": 90,
        "slope_smoothness": 50,
        "regime_min_duration": 0,
        "atr_window": 60,
        "atr_lookback": 720,
        "atr_percentile": 0.0,
        "slope_threshold": 0,
    }

    # --- Outcome params (must match backtest) ---
    outcome_params = {
        "atr_window": 60,
        "tp_mult": TP_MULT,
        "sl_mult": SL_MULT,
        "max_horizon": 30,
    }

    # --- LGBM base params ---
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

    # FEATURES starts as the generic full-library builder (Learn.features.add_feature_library).
    # After feature selection below, it is replaced with the generated, pack-specific
    # compute_features() produced by Learn.feature_codegen.write_generated_feature_module - see
    # the "Load data and compute target variables" cell.
    FEATURES = add_feature_library

    print("Device: CPU (tree model)")
    print("Dataset:", DS_NAME)
    print("Targets:", TARGET_COLS)
    print("Feature function:", getattr(FEATURES, "__name__", str(FEATURES)))
    print(f"R:R = {TP_MULT/SL_MULT:.2f} (TP_MULT={TP_MULT}, SL_MULT={SL_MULT})")
    print(f"Purge gap: {WFO_GAP_ROWS} bars")

    # ---------------------------------------------------------------
    # Calibrate max_horizon
    # ---------------------------------------------------------------
    _df_calib = load_ohlcv(DS_NAME, n_rows=N_ROWS)
    horizon_table = resolution_rate_by_horizon(
        _df_calib,
        atr_window=outcome_params["atr_window"],
        tp_mult=outcome_params["tp_mult"],
        sl_mult=outcome_params["sl_mult"],
        horizon_candidates=MAX_HORIZON_CANDIDATES,
    )
    print(horizon_table.to_string())

    _qualifying = horizon_table[
        (horizon_table["buy_resolved_rate"] >= TARGET_RESOLUTION_RATE)
        & (horizon_table["sell_resolved_rate"] >= TARGET_RESOLUTION_RATE)
    ]
    if len(_qualifying) > 0:
        chosen_horizon = int(_qualifying.index.min())
    else:
        chosen_horizon = int(horizon_table.index.max())
        print(
            f"WARNING: no candidate horizon reached TARGET_RESOLUTION_RATE={TARGET_RESOLUTION_RATE}; "
            f"falling back to the largest candidate ({chosen_horizon})."
        )

    outcome_params["max_horizon"] = chosen_horizon
    print(f"Calibrated max_horizon = {chosen_horizon} bars")
    del _df_calib

    # ---------------------------------------------------------------
    # Load data and compute target variables
    # ---------------------------------------------------------------
    from talib import ATR

    df = load_ohlcv(DS_NAME, n_rows=N_ROWS)
    print(f"Loaded {len(df):,} rows")
    print(df["Time"].min(), "->", df["Time"].max())
    ds_title = DS_NAME.split("/")[-1].split(".")[0]

    # Step 1: Generate the full generic feature library
    df = FEATURES(df.copy(), include_mtf=True, regime_params=regime_params)
    print(f'Added features: {len(df.columns)} columns')

    # Step 2: Feature selection (if enabled)
    if USE_FEATURE_SELECTION:
        print("Running automated feature selection...")
        # Compute targets first on the full feature set for selection
        df_for_selection = df.copy()
        df_for_selection["atr"] = ATR(df_for_selection["High"], df_for_selection["Low"], df_for_selection["Close"], timeperiod=14)
        outcomes_fs = calculate_trade_outcomes_capped(df_for_selection, **outcome_params)
        eps_fs = 1e-8
        long_q_fs = np.log1p(outcomes_fs["buy_MFE"] / (outcomes_fs["buy_MAE"] + eps_fs))
        short_q_fs = np.log1p(outcomes_fs["sell_MFE"] / (outcomes_fs["sell_MAE"] + eps_fs))
        if TARGET_CLIP_MAX is not None:
            long_q_fs = np.clip(long_q_fs, 0.0, TARGET_CLIP_MAX)
            short_q_fs = np.clip(short_q_fs, 0.0, TARGET_CLIP_MAX)
        df_for_selection["signed_quality"] = long_q_fs - short_q_fs

        # Use a subset for speed (last 500k rows)
        fs_subset = df_for_selection.tail(min(500_000, len(df_for_selection)))
        selected_features = select_features_pipeline(
            fs_subset,
            target_col='signed_quality',
            n_features=SELECTED_FEATURE_COUNT,
            correlation_threshold=0.95,
            include_mtf=True,
            regime_params=regime_params,
        )
        print(f"Selected {len(selected_features)} features")
    else:
        # No automated selection: keep every generic library feature (minus OHLCV/Time).
        _ohlcv_cols = {"Time", "Open", "High", "Low", "Close", "Volume"}
        selected_features = [c for c in df.columns if c not in _ohlcv_cols]
        print(f"Feature selection disabled; keeping all {len(selected_features)} library features")

    # Generate a standalone, hash-named feature module for this pack (see Learn/feature_codegen.py)
    # and use it as the single source of truth for both training and export - no more pickled
    # closures or instrument-specific functions hardcoded in Learn/features.py.
    feature_file_path, feature_file_hash = write_generated_feature_module(
        selected_features, output_dir="ModelPacks/generated_features", dataset_slug=ds_title,
    )
    compute_features = load_generated_feature_module(feature_file_path, feature_file_hash)
    print(f"Generated feature module: {feature_file_path} (hash {feature_file_hash[:10]}...)")

    # Re-generate features from FRESH raw data via the generated module.
    # IMPORTANT: Use the original raw df (before any feature engineering) to avoid
    # _x/_y suffixed duplicate columns that occur when add_feature_library is
    # called on a dataframe that already has MTF columns.
    df_raw_fresh = load_ohlcv(DS_NAME, n_rows=N_ROWS)
    df = compute_features(df_raw_fresh, include_mtf=True, regime_params=regime_params)
    print(f"Feature matrix: {len(df)} rows x {len(selected_features)} features")

    # Update FEATURES for model pack export (single-pass, uses the generated module)
    FEATURES = compute_features

    # Step 3: Compute ATR (needed for label computation)
    df["atr"] = ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

    # Step 4: Compute trade outcomes using horizon-capped label function
    outcomes = calculate_trade_outcomes_capped(df, **outcome_params)
    print(f"Calculated trade outcomes for {len(outcomes):,} rows")

    eps = 1e-8
    long_q = np.log1p(outcomes["buy_MFE"] / (outcomes["buy_MAE"] + eps))
    short_q = np.log1p(outcomes["sell_MFE"] / (outcomes["sell_MAE"] + eps))
    print(f"Computed long and short quality metrics for {len(long_q):,} rows")

    if TARGET_CLIP_MAX is not None:
        long_q = np.clip(long_q, 0.0, TARGET_CLIP_MAX)
        short_q = np.clip(short_q, 0.0, TARGET_CLIP_MAX)
    print(f"Clipped long and short quality metrics to max {TARGET_CLIP_MAX}")

    df["long_quality"] = long_q
    df["short_quality"] = short_q
    df["signed_quality"] = df["long_quality"] - df["short_quality"]

    # 3-class win/loss/timeout labels (1=TP, 0=SL, 2=timeout/unresolved within max_horizon).
    # The magnitude regressor (long_quality/short_quality) is trained only on resolved rows
    # (buy_class != 2 / sell_class != 2) later in the preprocessing step; the classifier is
    # trained on all rows including timeouts as the third class.
    df["buy_class"] = outcomes["buy_class"].values
    df["sell_class"] = outcomes["sell_class"].values

    # Binary classification targets
    df['buy_win'] = (outcomes['buy_outcome'] == 1.0).astype(float)
    df['sell_win'] = (outcomes['sell_outcome'] == 1.0).astype(float)
    df['signed_win'] = np.where(
        df['signed_quality'] > 0, df['buy_win'],
        np.where(df['signed_quality'] < 0, df['sell_win'], 0.0)
    )

    print("Target summary:")
    print(describe_target_frame(df, TARGET_COLS).to_string())

    print(
        f"Zero share | long={(df['long_quality'] <= 1e-12).mean():.3f} "
        f"short={(df['short_quality'] <= 1e-12).mean():.3f} "
        f"signed={(df['signed_quality'] == 0).mean():.3f}"
    )
    print(
        f"Class balance | buy TP/SL/timeout={df['buy_class'].value_counts(normalize=True).sort_index().round(3).to_dict()} "
        f"sell TP/SL/timeout={df['sell_class'].value_counts(normalize=True).sort_index().round(3).to_dict()}"
    )

    # ---------------------------------------------------------------
    # Generate Tradeability Scores
    # ---------------------------------------------------------------
    if USE_TRADEABILITY_WEIGHTING:
        print("Generating tradeability scores...")
        tradeability_y = pd.Series(
            (~outcomes['buy_outcome'].isna() | ~outcomes['sell_outcome'].isna()).astype(int),
            name='tradeable'
        )

        # Use top features by variance for the tradeability classifier
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = set(TARGET_COLS + CLASSIFICATION_TARGETS + ['tradeability_score', 'atr', 'buy_class', 'sell_class'])
        feature_candidates = [c for c in numeric_cols if c not in exclude_cols]

        if len(feature_candidates) > 50:
            variances = df[feature_candidates].var().sort_values(ascending=False)
            top_features = variances.head(50).index.tolist()
        else:
            top_features = feature_candidates

        tradeability_scores, tradeability_metrics = generate_tradeability_scores(
            X=df[top_features].values,
            y=tradeability_y.values,
            timestamps=df['Time'],
            n_folds=TRADEABILITY_N_FOLDS,
            min_train_size=TRADEABILITY_MIN_TRAIN,
            test_size=TRADEABILITY_TEST_SIZE,
            gap_size=outcome_params['max_horizon'],
        )

        df['tradeability_score'] = tradeability_scores['tradeability_score'].values
        # Rows never assigned to a test fold will have NaN; default to neutral 0.5
        df['tradeability_score'] = df['tradeability_score'].fillna(0.5)
        print(f"Tradeability score range: [{df['tradeability_score'].min():.4f}, {df['tradeability_score'].max():.4f}]")
        print(f"Tradeability ROC AUC: {tradeability_metrics['overall_roc_auc']:.4f}")
        print(f"Tradeability PR AUC: {tradeability_metrics['overall_pr_auc']:.4f}")
    else:
        df['tradeability_score'] = 1.0

    # ---------------------------------------------------------------
    # Preprocess features and create train / holdout splits
    # ---------------------------------------------------------------
    preprocess_ohlcv_args = {
        "target_col": TARGET_COLS + ["tradeability_score"] + (CLASSIFICATION_TARGETS if TRAIN_CLASSIFIER else []),
        # buy_class/sell_class are the two-head classifier targets (see labelling cell above) - they
        # must NEVER be columns in X_train/X_holdout, only read back out of proc_df_train/proc_df_holdout
        # for building the resolved-only masks below. Passing them as outcomes_col makes
        # preprocess_ohlcv add them to its internal exclusion set (label_exclude) so they are dropped
        # from the scaled/passthrough/one-hot feature groups without changing the shape of y_train/
        # y_holdout (outcomes_col is returned separately, not appended to y).
        "outcomes_col": ["buy_class", "sell_class"],
        "shift": 0,
        "onehot_prefixes": ["OH_"],
        "price_prefixes": ["PR_"],
    }

    split_idx = int(len(df) * TRAIN_FRACTION)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_holdout = df.iloc[split_idx:].copy().reset_index(drop=True)

    X_train, y_train, scaler, features, _, proc_df_train = preprocess_ohlcv(
        df_train, **preprocess_ohlcv_args, scaler=None, return_df=True,
    )
    X_holdout, y_holdout, _, _, _, proc_df_holdout = preprocess_ohlcv(
        df_holdout, **preprocess_ohlcv_args, scaler=scaler, return_df=True,
    )

    # Split target arrays for regression targets
    regression_target_count = len(TARGET_COLS)
    target_arrays = {
        target: {
            "train": y_train[:, i],
            "holdout": y_holdout[:, i],
        }
        for i, target in enumerate(TARGET_COLS)
    }

    # Classification target arrays
    if TRAIN_CLASSIFIER:
        clf_target_arrays = {
            target: {
                "train": y_train[:, regression_target_count + 1 + i],
                "holdout": y_holdout[:, regression_target_count + 1 + i],
            }
            for i, target in enumerate(CLASSIFICATION_TARGETS)
        }

    # Tradeability weight is at index len(TARGET_COLS) (the tradeability_score column)
    tradeability_idx = len(TARGET_COLS)
    train_weight = None if not USE_TRADEABILITY_WEIGHTING else y_train[:, tradeability_idx]

    tune_rows = min(len(X_train), MAX_TUNING_ROWS)
    X_tune = X_train[-tune_rows:]
    tune_targets = {k: v["train"][-tune_rows:] for k, v in target_arrays.items()}
    tune_weight = None if train_weight is None else train_weight[-tune_rows:]

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
        gap_size=max(WFO_GAP_ROWS, outcome_params["max_horizon"]),
        n_folds=WFO_FOLDS,
    )

    print(f"Train rows   : {len(X_train):,}")
    print(f"Holdout rows : {len(X_holdout):,}")
    print(f"Tuning rows  : {len(X_tune):,}")
    print(f"Walk-forward folds: {len(wf_folds)}")
    for i, (tr, va) in enumerate(wf_folds, start=1):
        print(f"  fold {i}: train={len(tr):,} val={len(va):,}")

    # --- Two-head design: build classifier (all rows) + resolved-only regressor arrays ---
    # proc_df_train/proc_df_holdout retain every original column (buy_class/sell_class included)
    # in the exact row order used to build X_train/X_holdout, since preprocess_ohlcv only drops
    # NaN rows uniformly - it does not reorder or subset columns beyond the feature matrix itself.
    buy_class_train = proc_df_train["buy_class"].values.astype(int)
    sell_class_train = proc_df_train["sell_class"].values.astype(int)
    buy_class_holdout = proc_df_holdout["buy_class"].values.astype(int)
    sell_class_holdout = proc_df_holdout["sell_class"].values.astype(int)

    assert len(buy_class_train) == len(X_train), "buy_class_train misaligned with X_train"
    assert len(sell_class_train) == len(X_train), "sell_class_train misaligned with X_train"
    assert len(buy_class_holdout) == len(X_holdout), "buy_class_holdout misaligned with X_holdout"
    assert len(sell_class_holdout) == len(X_holdout), "sell_class_holdout misaligned with X_holdout"

    train_resolved_long = buy_class_train != 2
    train_resolved_short = sell_class_train != 2
    holdout_resolved_long = buy_class_holdout != 2
    holdout_resolved_short = sell_class_holdout != 2

    print(
        f"Resolved rows (train)   | long={train_resolved_long.sum():,}/{len(train_resolved_long):,} "
        f"short={train_resolved_short.sum():,}/{len(train_resolved_short):,}"
    )
    print(
        f"Resolved rows (holdout) | long={holdout_resolved_long.sum():,}/{len(holdout_resolved_long):,} "
        f"short={holdout_resolved_short.sum():,}/{len(holdout_resolved_short):,}"
    )

    # ---------------------------------------------------------------
    # Tune hyper-parameters for each target
    # ---------------------------------------------------------------
    tuning_results = {}
    best_config_by_target = {}
    best_iteration_by_target = {}

    for target_name in TARGET_COLS:
        print(f"\nTuning target: {target_name}")
        results_df, best_cfg = tune_lgbm_by_spearman(
            X=X_tune,
            y=tune_targets[target_name],
            folds=wf_folds,
            candidate_params=LGBM_CANDIDATES,
            base_params=LGBM_BASE_PARAMS,
            sample_weight=tune_weight,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            robustness_penalty=ROBUSTNESS_PENALTY,
        )
        tuning_results[target_name] = results_df
        best_config_by_target[target_name] = best_cfg
        best_iteration_by_target[target_name] = int(best_cfg["median_best_iteration"])

        print("Top candidates:")
        print(results_df[[
            "cfg_id", "mean_spearman", "std_spearman", "robust_score",
            "mean_r2", "mean_rmse", "median_best_iteration",
            "learning_rate", "num_leaves", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        ]].head(5).round(4).to_string())

    tuning_summary = pd.DataFrame([
        {
            "target": target,
            "best_cfg_id": best_config_by_target[target]["cfg_id"],
            "best_mean_spearman": best_config_by_target[target]["mean_spearman"],
            "best_std_spearman": best_config_by_target[target]["std_spearman"],
            "best_robust_score": best_config_by_target[target]["robust_score"],
            "best_n_estimators": best_iteration_by_target[target],
        }
        for target in TARGET_COLS
    ]).set_index("target")

    print("\nSelected tuning summary:")
    print(tuning_summary.round(4).to_string())

    # ---------------------------------------------------------------
    # Two-head tuning: 3-class win/loss/timeout classifier + resolved-only magnitude regressor
    # ---------------------------------------------------------------
    tune_buy_class = buy_class_train[-tune_rows:]
    tune_sell_class = sell_class_train[-tune_rows:]
    tune_resolved_long = tune_buy_class != 2
    tune_resolved_short = tune_sell_class != 2

    wf_folds_long = [
        (tr[tune_resolved_long[tr]], va[tune_resolved_long[va]]) for tr, va in wf_folds
    ]
    wf_folds_short = [
        (tr[tune_resolved_short[tr]], va[tune_resolved_short[va]]) for tr, va in wf_folds
    ]

    two_head_tuning_results = {}
    two_head_best_config = {}
    two_head_best_iteration = {}

    for side, y_class_tune, clf_folds, y_reg_key, reg_folds in [
        ("long", tune_buy_class, wf_folds, "long_quality", wf_folds_long),
        ("short", tune_sell_class, wf_folds, "short_quality", wf_folds_short),
    ]:
        print(f"\nTuning 3-class classifier: {side}")
        clf_results_df, clf_best_cfg = tune_lgbm_classifier(
            X=X_tune,
            y=y_class_tune,
            folds=clf_folds,
            candidate_params=LGBM_CANDIDATES,
            base_params=LGBM_BASE_PARAMS,
            sample_weight=tune_weight,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            robustness_penalty=ROBUSTNESS_PENALTY,
        )
        two_head_tuning_results[f"classifier_{side}"] = clf_results_df
        two_head_best_config[f"classifier_{side}"] = clf_best_cfg
        two_head_best_iteration[f"classifier_{side}"] = int(clf_best_cfg["median_best_iteration"])
        print(f"  best mean TP PR-AUC: {clf_best_cfg['mean_tp_pr_auc']:.4f}")

        print(f"Tuning resolved-only magnitude regressor: {side}")
        reg_results_df, reg_best_cfg = tune_lgbm_by_spearman(
            X=X_tune,
            y=tune_targets[y_reg_key],
            folds=reg_folds,
            candidate_params=LGBM_CANDIDATES,
            base_params=LGBM_BASE_PARAMS,
            sample_weight=tune_weight,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            robustness_penalty=ROBUSTNESS_PENALTY,
        )
        two_head_tuning_results[f"regressor_{side}"] = reg_results_df
        two_head_best_config[f"regressor_{side}"] = reg_best_cfg
        two_head_best_iteration[f"regressor_{side}"] = int(reg_best_cfg["median_best_iteration"])
        print(f"  best mean Spearman: {reg_best_cfg['mean_spearman']:.4f}")

    # ---------------------------------------------------------------
    # Train final models on full training set and evaluate on holdout
    # ---------------------------------------------------------------
    final_models = {}
    holdout_results = {}

    for target_name in TARGET_COLS:
        best_cfg = best_config_by_target[target_name]
        final_params = {
            **LGBM_BASE_PARAMS,
            **{k: v for k, v in best_cfg.items() if k in LGBM_CANDIDATES[0]},
            "n_estimators": int(best_iteration_by_target[target_name]),
        }

        print(f"Training final model for {target_name} with {final_params['n_estimators']} trees")
        model = LGBMRegressor(**final_params)
        fit_kwargs = {"X": X_train, "y": target_arrays[target_name]["train"]}
        if train_weight is not None:
            fit_kwargs["sample_weight"] = train_weight
        model.fit(**fit_kwargs)

        pred_holdout = model.predict(X_holdout)
        final_models[target_name] = model
        holdout_results[target_name] = reg_metrics(
            target_arrays[target_name]["holdout"], pred_holdout
        )
        holdout_results[target_name]["best_n_estimators"] = int(final_params["n_estimators"])

    signed_pred = final_models["signed_quality"].predict(X_holdout)
    long_pred = final_models["long_quality"].predict(X_holdout)
    short_pred = final_models["short_quality"].predict(X_holdout)
    spread_pred = long_pred - short_pred

    holdout_results["spread_proxy"] = reg_metrics(
        target_arrays["signed_quality"]["holdout"], spread_pred
    )
    holdout_results["spread_proxy"]["best_n_estimators"] = int(
        np.median([
            best_iteration_by_target["long_quality"],
            best_iteration_by_target["short_quality"],
        ])
    )

    holdout_df = pd.DataFrame(holdout_results).T
    print("\nHoldout performance:")
    print(holdout_df.round(4).to_string())

    cv_best = pd.DataFrame([
        {
            "target": target,
            "cv_mean_spearman": tuning_summary.loc[target, "best_mean_spearman"],
            "cv_std_spearman": tuning_summary.loc[target, "best_std_spearman"],
            "cv_robust_score": tuning_summary.loc[target, "best_robust_score"],
            "holdout_spearman": holdout_results[target]["Spearman"],
            "holdout_r2": holdout_results[target]["R2"],
        }
        for target in TARGET_COLS
    ]).set_index("target")

    print("\nCV vs holdout:")
    print(cv_best.round(4).to_string())

    # ---------------------------------------------------------------
    # Train final two-head LGBM models: 3-class classifier + magnitude regressor
    # ---------------------------------------------------------------
    lgbm_classifiers = {}
    lgbm_regressors = {}
    lgbm_two_head_holdout = {}

    for side, y_class_col, resolved_train_mask, resolved_holdout_mask, y_reg_key in [
        ("long", buy_class_train, train_resolved_long, holdout_resolved_long, "long_quality"),
        ("short", sell_class_train, train_resolved_short, holdout_resolved_short, "short_quality"),
    ]:
        clf_cfg = two_head_best_config[f"classifier_{side}"]
        clf_params = {
            **LGBM_BASE_PARAMS,
            **{k: v for k, v in clf_cfg.items() if k in LGBM_CANDIDATES[0]},
            "n_estimators": int(two_head_best_iteration[f"classifier_{side}"]),
            "objective": "multiclass",
            "num_class": 3,
        }
        print(f"Training 3-class classifier for {side} with {clf_params['n_estimators']} trees")
        clf_model = LGBMClassifier(**clf_params)
        clf_model.fit(X_train, y_class_col, sample_weight=train_weight)
        lgbm_classifiers[side] = clf_model

        reg_cfg = two_head_best_config[f"regressor_{side}"]
        reg_params_side = {
            **LGBM_BASE_PARAMS,
            **{k: v for k, v in reg_cfg.items() if k in LGBM_CANDIDATES[0]},
            "n_estimators": int(two_head_best_iteration[f"regressor_{side}"]),
        }
        X_train_resolved = X_train[resolved_train_mask]
        y_train_resolved = target_arrays[y_reg_key]["train"][resolved_train_mask]
        reg_fit_kwargs = {"X": X_train_resolved, "y": y_train_resolved}
        if train_weight is not None:
            reg_fit_kwargs["sample_weight"] = train_weight[resolved_train_mask]
        print(
            f"Training resolved-only magnitude regressor for {side} with "
            f"{reg_params_side['n_estimators']} trees on {len(X_train_resolved):,} resolved rows"
        )
        reg_model = LGBMRegressor(**reg_params_side)
        reg_model.fit(**reg_fit_kwargs)
        lgbm_regressors[side] = reg_model

        # Holdout evaluation
        X_holdout_resolved = X_holdout[resolved_holdout_mask]
        y_holdout_resolved = target_arrays[y_reg_key]["holdout"][resolved_holdout_mask]
        reg_pred_holdout = reg_model.predict(X_holdout_resolved)
        reg_metrics_holdout = reg_metrics(y_holdout_resolved, reg_pred_holdout)

        clf_proba_holdout = clf_model.predict_proba(X_holdout)
        y_class_holdout = buy_class_holdout if side == "long" else sell_class_holdout
        tp_pr_auc = float(average_precision_score((y_class_holdout == 1).astype(int), clf_proba_holdout[:, 1]))
        pr_aucs = []
        for c in range(3):
            y_true_c = (y_class_holdout == c).astype(int)
            if y_true_c.sum() > 0:
                pr_aucs.append(average_precision_score(y_true_c, clf_proba_holdout[:, c]))
        macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else float("nan")

        lgbm_two_head_holdout[side] = {
            "regressor": reg_metrics_holdout,
            "classifier": {"tp_pr_auc": tp_pr_auc, "macro_pr_auc": macro_pr_auc},
        }
        print(
            f"  {side}: holdout Spearman={reg_metrics_holdout['Spearman']:.4f}  "
            f"TP PR-AUC={tp_pr_auc:.4f}  macro PR-AUC={macro_pr_auc:.4f}"
        )

    # ---------------------------------------------------------------
    # Train PyTorch two-head MLP candidates
    # ---------------------------------------------------------------
    device = torch.device("cpu")
    mlp_models = {}
    mlp_two_head_holdout = {}

    for side, y_class_col, resolved_train_mask, resolved_holdout_mask, y_reg_key in [
        ("long", buy_class_train, train_resolved_long, holdout_resolved_long, "long_quality"),
        ("short", sell_class_train, train_resolved_short, holdout_resolved_short, "short_quality"),
    ]:
        print(f"Training MLP two-head model for {side}")
        y_reg_full_train = target_arrays[y_reg_key]["train"]
        mlp_model = train_two_head_mlp(
            X_train, y_class_col, y_reg_full_train, resolved_train_mask,
            hidden_sizes=DL_HIDDEN_SIZES, epochs=DL_EPOCHS, lr=DL_LR, batch_size=DL_BATCH_SIZE,
            device=device, verbose=True,
        )
        mlp_models[side] = mlp_model

        proba_holdout, reg_pred_holdout_full = predict_two_head_mlp(mlp_model, X_holdout, device=device)
        y_class_holdout = buy_class_holdout if side == "long" else sell_class_holdout

        tp_pr_auc = float(average_precision_score((y_class_holdout == 1).astype(int), proba_holdout[:, 1]))
        pr_aucs = []
        for c in range(3):
            y_true_c = (y_class_holdout == c).astype(int)
            if y_true_c.sum() > 0:
                pr_aucs.append(average_precision_score(y_true_c, proba_holdout[:, c]))
        macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else float("nan")

        y_reg_holdout_resolved = target_arrays[y_reg_key]["holdout"][resolved_holdout_mask]
        reg_pred_resolved = reg_pred_holdout_full[resolved_holdout_mask]
        mlp_reg_metrics_side = reg_metrics(y_reg_holdout_resolved, reg_pred_resolved)

        mlp_two_head_holdout[side] = {
            "regressor": mlp_reg_metrics_side,
            "classifier": {"tp_pr_auc": tp_pr_auc, "macro_pr_auc": macro_pr_auc},
        }
        print(
            f"  {side}: holdout Spearman={mlp_reg_metrics_side['Spearman']:.4f}  "
            f"TP PR-AUC={tp_pr_auc:.4f}  macro PR-AUC={macro_pr_auc:.4f}"
        )

    # ---------------------------------------------------------------
    # Compare LGBM vs MLP per side, pick model_choice
    # ---------------------------------------------------------------
    model_choice = {}
    comp_rows = []
    for side in ["long", "short"]:
        lgbm_reg_spearman = lgbm_two_head_holdout[side]["regressor"]["Spearman"]
        mlp_reg_spearman = mlp_two_head_holdout[side]["regressor"]["Spearman"]
        lgbm_clf_macro_pr = lgbm_two_head_holdout[side]["classifier"]["macro_pr_auc"]
        mlp_clf_macro_pr = mlp_two_head_holdout[side]["classifier"]["macro_pr_auc"]

        comp_rows.append({"side": side, "model": "lgbm", "holdout_spearman": lgbm_reg_spearman, "holdout_macro_pr_auc": lgbm_clf_macro_pr})
        comp_rows.append({"side": side, "model": "mlp", "holdout_spearman": mlp_reg_spearman, "holdout_macro_pr_auc": mlp_clf_macro_pr})

        spearman_diff = lgbm_reg_spearman - mlp_reg_spearman
        if abs(spearman_diff) <= 0.01:
            model_choice[side] = "lgbm" if lgbm_clf_macro_pr >= mlp_clf_macro_pr else "mlp"
        else:
            model_choice[side] = "lgbm" if lgbm_reg_spearman > mlp_reg_spearman else "mlp"

    comp_df = pd.DataFrame(comp_rows)
    print("Model comparison (LGBM vs MLP, two-head):")
    print(comp_df.round(4).to_string())
    print("Model choice:", model_choice)

    # ---------------------------------------------------------------
    # Train Classification Models
    # ---------------------------------------------------------------
    if TRAIN_CLASSIFIER:
        final_classifiers = {}
        classifier_holdout_results = {}

        for target_name in CLASSIFICATION_TARGETS:
            print(f"Training classifier for {target_name}")

            clf = LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                class_weight='balanced',
            )

            clf.fit(
                X_train,
                clf_target_arrays[target_name]['train'],
                sample_weight=train_weight,
            )

            pred_proba = clf.predict_proba(X_holdout)[:, 1]
            y_true_clf = clf_target_arrays[target_name]['holdout']

            classifier_holdout_results[target_name] = {
                'ROC_AUC': float(roc_auc_score(y_true_clf, pred_proba)),
                'PR_AUC': float(average_precision_score(y_true_clf, pred_proba)),
            }
            final_classifiers[target_name] = clf

        print("\nClassifier holdout performance:")
        print(pd.DataFrame(classifier_holdout_results).T.round(4).to_string())

    # ---------------------------------------------------------------
    # Train Ensemble Models
    # ---------------------------------------------------------------
    if ENSEMBLE_SIZE > 1:
        ensemble_models = {target: [] for target in TARGET_COLS}

        for target_name in TARGET_COLS:
            best_cfg = best_config_by_target[target_name]
            base_params_for_target = {
                **LGBM_BASE_PARAMS,
                **{k: v for k, v in best_cfg.items() if k in LGBM_CANDIDATES[0]},
            }

            for seed in ENSEMBLE_SEEDS[:ENSEMBLE_SIZE]:
                params = {
                    **base_params_for_target,
                    'random_state': seed,
                    'n_estimators': int(best_iteration_by_target[target_name]),
                }

                model = LGBMRegressor(**params)
                model.fit(X_train, target_arrays[target_name]['train'])
                ensemble_models[target_name].append(model)

            print(f"Trained {len(ensemble_models[target_name])} models for {target_name}")

        # Evaluate ensemble on holdout
        ensemble_holdout_results = {}
        for target_name in TARGET_COLS:
            ensemble_preds = np.column_stack([
                m.predict(X_holdout) for m in ensemble_models[target_name]
            ])
            pred_mean = np.mean(ensemble_preds, axis=1)
            ensemble_holdout_results[target_name] = reg_metrics(
                target_arrays[target_name]['holdout'], pred_mean
            )

        print("\nEnsemble holdout performance:")
        print(pd.DataFrame(ensemble_holdout_results).T.round(4).to_string())

        # Compare single vs ensemble
        comparison = pd.DataFrame({
            'single': [holdout_results[t]['Spearman'] for t in TARGET_COLS],
            'ensemble': [ensemble_holdout_results[t]['Spearman'] for t in TARGET_COLS],
        }, index=TARGET_COLS)
        comparison['improvement'] = comparison['ensemble'] - comparison['single']
        print("\nSingle vs Ensemble Spearman:")
        print(comparison.round(4).to_string())

    # ---------------------------------------------------------------
    # Trading-style proxy analysis on the signed target
    # ---------------------------------------------------------------
    y_true_signed = target_arrays["signed_quality"]["holdout"]
    y_pred_signed = signed_pred
    thresholds = np.quantile(np.abs(y_pred_signed), np.linspace(0.50, 0.95, 10))

    sweep_rows = []
    for thr in thresholds:
        pred_sign = np.zeros(len(y_pred_signed), dtype=int)
        pred_sign[y_pred_signed >= thr] = 1
        pred_sign[y_pred_signed <= -thr] = -1
        taken = pred_sign != 0

        if taken.sum() == 0:
            sweep_rows.append({
                "threshold": float(thr),
                "coverage": 0.0,
                "taken": 0,
                "long_count": 0,
                "short_count": 0,
                "sign_hit_rate": np.nan,
                "aligned_signed_mean": np.nan,
                "taken_spearman": np.nan,
            })
            continue

        sign_hit = np.mean(np.sign(y_true_signed[taken]) == pred_sign[taken])
        aligned_signed = np.where(
            pred_sign[taken] == 1, y_true_signed[taken], -y_true_signed[taken]
        )
        sweep_rows.append({
            "threshold": float(thr),
            "coverage": float(taken.mean()),
            "taken": int(taken.sum()),
            "long_count": int((pred_sign == 1).sum()),
            "short_count": int((pred_sign == -1).sum()),
            "sign_hit_rate": float(sign_hit),
            "aligned_signed_mean": float(np.mean(aligned_signed)),
            "taken_spearman": float(safe_spearman(y_true_signed[taken], y_pred_signed[taken])),
        })

    threshold_df = pd.DataFrame(sweep_rows)
    print("Signed-model threshold sweep:")
    print(threshold_df.round(4).to_string())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df["aligned_signed_mean"],
        mode="lines+markers",
        name="Aligned signed mean",
    ))
    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df["sign_hit_rate"],
        mode="lines+markers",
        name="Sign hit rate",
    ))
    fig.update_layout(
        title="Trading Proxy vs Confidence Threshold",
        xaxis_title="Absolute prediction threshold",
        yaxis_title="Score",
        height=420,
    )
    print("Threshold sweep plot omitted (CLI mode)")

    # ---------------------------------------------------------------
    # Export the model pack
    # ---------------------------------------------------------------
    today = pd.Timestamp.now().strftime("%Y%m%d")
    model_name = "_".join([ds_title, "LightGBM", today, "spearman_prod", "TP" + str(TP_MULT), "SL" + str(SL_MULT)])

    model_pack_dir = Path("../ModelWorkbench/ModelPacks")
    model_pack_dir.mkdir(parents=True, exist_ok=True)
    model_pack_path = model_pack_dir / f"{model_name}_model.pkl"

    model_info = {
        "dataset_name": ds_title,
        "dataset_dir": DS_NAME,
        "date_trained": today,
        "model_type": "LightGBMRegressionPack",
        "task": "regression",
        "primary_target": PRIMARY_TARGET,
        "targets": TARGET_COLS,
        "cv_scheme": {
            "type": "expanding_walk_forward",
            "folds": len(wf_folds),
            "gap_rows": max(WFO_GAP_ROWS, outcome_params["max_horizon"]),
            "tuning_rows": int(len(X_tune)),
            "holdout_fraction": TRAIN_FRACTION,
            "robustness_penalty": ROBUSTNESS_PENALTY,
        },
    }

    pack = {
        "model": final_models[PRIMARY_TARGET],
        "primary_model": final_models[PRIMARY_TARGET],
        "aux_models": {k: v for k, v in final_models.items() if k != PRIMARY_TARGET},
        "model_class": final_models[PRIMARY_TARGET].__class__,
        "model_class_source": inspect.getsource(final_models[PRIMARY_TARGET].__class__),
        "model_params": {
            target: {
                **LGBM_BASE_PARAMS,
                **{k: v for k, v in best_config_by_target[target].items() if k in LGBM_CANDIDATES[0]},
                "n_estimators": int(best_iteration_by_target[target]),
            }
            for target in TARGET_COLS
        },
        "model_info": model_info,
        "features": features,
        "selected_features": selected_features if "selected_features" in globals() else None,

        "feature_count": X_train.shape[1],
        # NOTE: the actual function object is intentionally NOT stored here (unlike the pre-refactor
        # schema) - FEATURES is now Learn.feature_codegen's dynamically-loaded compute_features, whose
        # module was never registered in sys.modules, so pickle cannot serialize it
        # (PicklingError: import of module '_generated_features_...' failed). The source text is still
        # picklable/useful for inspection; the load-bearing mechanism for reconstructing features at
        # eval/live time is feature_file_path + feature_file_hash below, loaded via
        # Learn.feature_codegen.load_generated_feature_module - not this field.
        "feature_function_source": inspect.getsource(FEATURES),
        "preprocess_function": preprocess_ohlcv,
        "preprocess_function_source": inspect.getsource(preprocess_ohlcv),
        "preprocess_args": preprocess_ohlcv_args,
        "scaler": scaler,
        "label_function": calculate_trade_outcomes_capped,
        "label_function_source": inspect.getsource(calculate_trade_outcomes_capped),
        "label_params": None,
        "regime_params": regime_params,
        "outcome_params": outcome_params,
        "trading_hours": None,
        "input_shape": None,
        "data_split": {
            "train_rows": len(X_train),
            "holdout_rows": len(X_holdout),
            "train_fraction": TRAIN_FRACTION,
            "max_tuning_rows": MAX_TUNING_ROWS,
        },
        "validation": {
            "cv_summary": tuning_summary.to_dict(orient="index"),
            "cv_results": {
                target: tuning_results[target].head(10).to_dict(orient="records")
                for target in TARGET_COLS
            },
            "holdout": {
                target: {k: float(v) for k, v in holdout_results[target].items()}
                for target in holdout_results
            },
            "threshold_sweep": threshold_df.to_dict(orient="records"),
        },
    }

    # Add classifiers to pack if trained
    if TRAIN_CLASSIFIER:
        pack['classifiers'] = final_classifiers
        pack['classifier_validation'] = classifier_holdout_results

    # Add ensemble models to pack if trained
    if ENSEMBLE_SIZE > 1:
        pack['ensemble_models'] = ensemble_models
        pack['ensemble_size'] = ENSEMBLE_SIZE
        pack['ensemble_seeds'] = ENSEMBLE_SEEDS[:ENSEMBLE_SIZE]
        pack['ensemble_validation'] = ensemble_holdout_results

    # --- Two-head design (classifier + resolved-only magnitude regressor, per side) ---
    # Stored under distinct keys from the legacy 'classifiers'/'aux_models' fields above so both
    # schemas coexist in the same pack without collision. feature_file_path/feature_file_hash
    # replace the old pickled feature_function closure - see Learn/feature_codegen.py.
    pack['feature_file_path'] = str(feature_file_path)
    pack['feature_file_hash'] = feature_file_hash
    pack['horizon_calibration_table'] = horizon_table.reset_index().to_dict(orient='records')

    pack['two_head_classifiers'] = {
        "long": {"model": lgbm_classifiers["long"], "type": "lgbm", "mlp_model": mlp_models["long"]},
        "short": {"model": lgbm_classifiers["short"], "type": "lgbm", "mlp_model": mlp_models["short"]},
    }
    pack['two_head_regressors'] = {
        "long": {"model": lgbm_regressors["long"], "type": "lgbm", "mlp_model": mlp_models["long"]},
        "short": {"model": lgbm_regressors["short"], "type": "lgbm", "mlp_model": mlp_models["short"]},
    }
    pack['model_choice'] = model_choice
    pack['two_head_validation'] = {
        "lgbm": lgbm_two_head_holdout,
        "mlp": mlp_two_head_holdout,
        "comparison": comp_df.to_dict(orient='records'),
    }

    with open(model_pack_path, "wb") as fh:
        pickle.dump(pack, fh)

    print(f"Saved model pack to {model_pack_path}")
