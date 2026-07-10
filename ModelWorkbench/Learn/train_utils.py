"""
Shared training/evaluation helper functions used by the model-training and backtest notebooks.

These were previously copy-pasted near-identically into the first code cell of both
`2.1 Train LGBM Regression Model.ipynb` and `3.1 Backtest LGBM Regression.ipynb`. They now live
here so a fix in one place benefits every notebook that imports them.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    average_precision_score, log_loss,
)


def load_ohlcv(ds_name, n_rows=None):
    """Load and prepare OHLCV data from a CSV file."""
    df = pd.read_csv(ds_name)
    if n_rows is not None:
        df = df.tail(n_rows)
    df = df.sort_values("Time").reset_index(drop=True)
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def safe_spearman(y_true, y_pred):
    """Compute Spearman rank correlation with safe fallback to 0.0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.nanstd(y_true) == 0.0 or np.nanstd(y_pred) == 0.0:
        return 0.0
    value = stats.spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return 0.0 if value is None or np.isnan(value) else float(value)


def reg_metrics(y_true, y_pred):
    """Compute a dict of standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Spearman": safe_spearman(y_true, y_pred),
    }


def lgbm_spearman_eval(y_true, y_pred):
    """Custom LightGBM eval function tracking Spearman (higher is better)."""
    return "spearman", safe_spearman(y_true, y_pred), True


def walk_forward_folds(n_samples, min_train_size, test_size, gap_size, n_folds):
    """
    Generate expanding-window walk-forward train/val indices.

    Each fold uses all data up to train_end for training, then
    a contiguous test block separated by a gap.
    """
    folds = []
    train_end = int(min_train_size)
    for _ in range(int(n_folds)):
        test_start = train_end + int(gap_size)
        if test_start >= n_samples:
            break
        test_end = min(test_start + int(test_size), n_samples)
        if test_end <= test_start:
            break
        train_idx = np.arange(0, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        if len(train_idx) == 0 or len(test_idx) == 0:
            break
        folds.append((train_idx, test_idx))
        train_end = test_end
        if train_end >= n_samples:
            break
    return folds


def describe_target_frame(frame, target_cols):
    """Return a summary DataFrame of target distributions."""
    rows = []
    for col in target_cols:
        s = pd.Series(frame[col])
        rows.append({
            "target": col,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p01": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.50)),
            "p99": float(s.quantile(0.99)),
        })
    return pd.DataFrame(rows).set_index("target")


def tune_lgbm_by_spearman(
    X,
    y,
    folds,
    candidate_params,
    base_params,
    sample_weight=None,
    early_stopping_rounds=200,
    robustness_penalty=0.25,
):
    """
    Tune hyper-parameter candidates via walk-forward cross-validation.

    Ranks candidates by a robust score = mean Spearman - penalty * std Spearman,
    then mean Spearman, then mean R². Returns (results_df, best_cfg_dict).
    """
    rows = []
    for cfg_i, cfg in enumerate(candidate_params, start=1):
        fold_rows = []
        best_iters = []
        for tr_idx, va_idx in folds:
            model = LGBMRegressor(**{**base_params, **cfg})
            fit_kwargs = {
                "X": X[tr_idx],
                "y": y[tr_idx],
                "eval_set": [(X[va_idx], y[va_idx])],
                "eval_metric": lgbm_spearman_eval,
                "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
            }
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight[tr_idx]
            model.fit(**fit_kwargs)

            best_iter = getattr(model, "best_iteration_", None)
            if best_iter is None or best_iter <= 0:
                best_iter = model.n_estimators
            best_iters.append(int(best_iter))

            pred = model.predict(X[va_idx], num_iteration=best_iter)
            fold_rows.append(reg_metrics(y[va_idx], pred))

        fold_df = pd.DataFrame(fold_rows)
        rows.append({
            **cfg,
            "cfg_id": cfg_i,
            "folds": len(fold_rows),
            "mean_spearman": float(fold_df["Spearman"].mean()),
            "std_spearman": float(fold_df["Spearman"].std(ddof=0)),
            "mean_r2": float(fold_df["R2"].mean()),
            "mean_rmse": float(fold_df["RMSE"].mean()),
            "mean_mae": float(fold_df["MAE"].mean()),
            "robust_score": float(
                fold_df["Spearman"].mean()
                - robustness_penalty * fold_df["Spearman"].std(ddof=0)
            ),
            "median_best_iteration": int(np.median(best_iters)),
            "fold_metrics": fold_rows,
        })

    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "mean_spearman", "mean_r2"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    return results, best


def tune_lgbm_classifier(
    X,
    y,
    folds,
    candidate_params,
    base_params,
    sample_weight=None,
    early_stopping_rounds=200,
    robustness_penalty=0.25,
):
    """
    Tune hyper-parameter candidates for the 3-class win/loss/timeout classifier
    (0=SL, 1=TP, 2=timeout) via walk-forward cross-validation.

    Scores each fold by average_precision_score for the TP class (label 1,
    one-vs-rest), since predicting "does this setup resolve as a winning trade"
    is what ultimately feeds pred_long/pred_short at inference time. Ranks
    candidates by a robust score = mean TP-PR-AUC - penalty * std TP-PR-AUC, then
    mean TP-PR-AUC. Returns (results_df, best_cfg_dict) with the same shape as
    tune_lgbm_by_spearman so callers can use both functions symmetrically.
    """
    rows = []
    for cfg_i, cfg in enumerate(candidate_params, start=1):
        fold_rows = []
        best_iters = []
        for tr_idx, va_idx in folds:
            model = LGBMClassifier(**{**base_params, **cfg, "objective": "multiclass", "num_class": 3})
            fit_kwargs = {
                "X": X[tr_idx],
                "y": y[tr_idx],
                "eval_set": [(X[va_idx], y[va_idx])],
                "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
            }
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight[tr_idx]
            model.fit(**fit_kwargs)

            best_iter = getattr(model, "best_iteration_", None)
            if best_iter is None or best_iter <= 0:
                best_iter = model.n_estimators
            best_iters.append(int(best_iter))

            proba = model.predict_proba(X[va_idx], num_iteration=best_iter)
            y_true_tp = (y[va_idx] == 1).astype(int)
            tp_pr_auc = (
                float(average_precision_score(y_true_tp, proba[:, 1]))
                if y_true_tp.sum() > 0 else 0.0
            )
            fold_log_loss = float(log_loss(y[va_idx], proba, labels=[0, 1, 2]))
            fold_rows.append({"tp_pr_auc": tp_pr_auc, "log_loss": fold_log_loss})

        fold_df = pd.DataFrame(fold_rows)
        rows.append({
            **cfg,
            "cfg_id": cfg_i,
            "folds": len(fold_rows),
            "mean_tp_pr_auc": float(fold_df["tp_pr_auc"].mean()),
            "std_tp_pr_auc": float(fold_df["tp_pr_auc"].std(ddof=0)),
            "mean_log_loss": float(fold_df["log_loss"].mean()),
            "robust_score": float(
                fold_df["tp_pr_auc"].mean()
                - robustness_penalty * fold_df["tp_pr_auc"].std(ddof=0)
            ),
            "median_best_iteration": int(np.median(best_iters)),
            "fold_metrics": fold_rows,
        })

    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "mean_tp_pr_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    return results, best
