"""
metrics.py — Training metrics tracking and validation-time metric computation.

Provides ``TrainingMetricsTracker`` — a logging abstraction that writes to
Weights & Biases, TensorBoard, or a plain in-memory dictionary, plus a suite
of validation metrics including Spearman per horizon, directional accuracy,
calibration error, regime F1, and trading proxy metrics (Sharpe ratio,
max drawdown).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def _tensorboard_available() -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TrainingMetricsTracker
# ---------------------------------------------------------------------------

class TrainingMetricsTracker:
    """Unified logging interface for training metrics.

    Automatically detects and uses Weights & Biases if ``wandb`` is installed
    and a run is active, or TensorBoard if ``tensorboard`` is available and
    ``log_dir`` is provided.  Falls back to an in-memory Python dictionary.

    Usage::

        tracker = TrainingMetricsTracker(log_dir="./runs/experiment_1")
        for step in range(steps):
            tracker.log_scalar("loss", loss_val, step)
        tracker.finalize()
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            log_dir: Directory for TensorBoard logs (if using TB).
            use_wandb: If True and wandb is installed, use Weights & Biases.
            wandb_project: W&B project name.
            wandb_run_name: W&B run display name.
            wandb_config: Config dict to log to W&B.
        """
        self._use_wandb = False
        self._use_tb = False
        self._memory_log: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._writer = None
        self._step_counter: int = 0

        # ---- W&B ----------------------------------------------------------
        if use_wandb and _wandb_available():
            import wandb
            run = wandb.run
            if run is None and wandb_project is not None:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config or {},
                    dir=log_dir,
                )
                run = wandb.run
            if run is not None:
                self._use_wandb = True
                logger.info("W&B logging active: %s", run.name or run.id)

        # ---- TensorBoard --------------------------------------------------
        if not self._use_wandb and log_dir is not None and _tensorboard_available():
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
            self._use_tb = True
            logger.info("TensorBoard logging active: %s", log_dir)

        if not self._use_wandb and not self._use_tb:
            logger.info(
                "No W&B or TensorBoard detected; logging to in-memory dict."
            )

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_scalar(
        self,
        name: str,
        value: Union[float, int],
        step: Optional[int] = None,
    ) -> None:
        """Log a scalar metric value.

        Args:
            name: Metric name (e.g. ``"loss/train"``).
            value: Scalar value.
            step: Global step index.  Auto-incremented if None.
        """
        if step is None:
            step = self._step_counter
            self._step_counter += 1

        if self._use_wandb:
            import wandb
            wandb.log({name: value}, step=step)
        elif self._use_tb:
            self._writer.add_scalar(name, value, step)

        # Always keep in-memory as fallback
        self._memory_log[name].append({"step": step, "value": value})

    def log_scalars(
        self,
        name_prefix: str,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ) -> None:
        """Log a set of related scalars under a common prefix.

        Args:
            name_prefix: Prefix prepended to each metric key (e.g. ``"val"``
                → ``"val/spearman_mean"``).
            metrics: Dict of metric_name → value.
            step: Global step index.
        """
        if step is None:
            step = self._step_counter
            self._step_counter += 1

        log_dict = {}
        for k, v in metrics.items():
            full_name = f"{name_prefix}/{k}" if name_prefix else k
            log_dict[full_name] = v
            self._memory_log[full_name].append({"step": step, "value": v})

        if self._use_wandb:
            import wandb
            wandb.log(log_dict, step=step)
        elif self._use_tb:
            for k, v in log_dict.items():
                self._writer.add_scalar(k, v, step)

    # ------------------------------------------------------------------
    # Histogram logging
    # ------------------------------------------------------------------

    def log_histogram(
        self,
        name: str,
        values: Union[np.ndarray, torch.Tensor, Sequence],
        step: Optional[int] = None,
    ) -> None:
        """Log a histogram (distribution) of values.

        Args:
            name: Histogram name.
            values: Array-like of values to histogram.
            step: Global step index.
        """
        if step is None:
            step = self._step_counter
            self._step_counter += 1

        # Convert to numpy
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values_np = np.asarray(values, dtype=np.float32).ravel()
        values_np = values_np[np.isfinite(values_np)]

        if len(values_np) == 0:
            return

        if self._use_wandb:
            import wandb
            wandb.log({name: wandb.Histogram(values_np)}, step=step)
        elif self._use_tb:
            self._writer.add_histogram(name, values_np, step)

        # Store summary stats in memory
        self._memory_log[name].append({
            "step": step,
            "count": len(values_np),
            "mean": float(np.mean(values_np)),
            "std": float(np.std(values_np)),
            "min": float(np.min(values_np)),
            "max": float(np.max(values_np)),
        })

    # ------------------------------------------------------------------
    # Model graph
    # ------------------------------------------------------------------

    def log_model_graph(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
    ) -> None:
        """Log the model computation graph.

        For TensorBoard, adds the graph.  For W&B, calls ``wandb.watch``.

        Args:
            model: The PyTorch model.
            sample_input: A sample input tensor (on the correct device).
        """
        if self._use_wandb:
            import wandb
            wandb.watch(model, log="all", log_freq=100)
            logger.info("W&B watch() called on model.")
        elif self._use_tb:
            try:
                self._writer.add_graph(model, sample_input)
                logger.info("TensorBoard graph logged.")
            except Exception as exc:
                logger.warning("Failed to log model graph: %s", exc)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Close any active logging sessions."""
        if self._use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if self._use_tb and self._writer is not None:
            self._writer.close()
            self._writer = None
        logger.info("Metrics tracker finalized.")

    # ------------------------------------------------------------------
    # Convenience: compute full validation metrics
    # ------------------------------------------------------------------

    def compute_and_log_validation(
        self,
        targets: Dict[str, np.ndarray],
        predictions: Dict[str, np.ndarray],
        step: int,
        prefix: str = "val",
    ) -> Dict[str, float]:
        """Compute a comprehensive validation metrics suite and log it.

        Args:
            targets: Dict of ground-truth arrays:
                ``"distribution"`` — (N, n_horizons) target values.
                ``"direction"`` — (N, n_horizons) binary direction labels.
                ``"volatility"`` — (N, n_horizons) or (N,).
                ``"regime"`` — (N,) integer class indices.
                ``"close"`` — (N,) close prices (for Sharpe/MDD proxy).
            predictions: Dict of predicted arrays:
                ``"mu"`` — (N, n_horizons) predicted mean.
                ``"log_sigma"`` — (N, n_horizons).
                ``"direction"`` — (N, n_horizons) logits or sigmoid probs.
                ``"volatility"`` — (N, n_horizons) or (N,).
                ``"regime"`` — (N, n_classes) logits.
            step: Global step index.
            prefix: Prefix for logged metric names.

        Returns:
            Dict of computed metrics.
        """
        metrics: Dict[str, float] = {}

        # ---- Spearman per horizon ----------------------------------------
        if "mu" in predictions and "distribution" in targets:
            mu = np.asarray(predictions["mu"])
            dist_tgt = np.asarray(targets["distribution"])
            n_h = min(mu.shape[1], dist_tgt.shape[1])
            spearman_vals = []
            for h in range(n_h):
                m = mu[:, h]
                t = dist_tgt[:, h]
                valid = np.isfinite(m) & np.isfinite(t)
                if valid.sum() < 2 or np.std(m[valid]) == 0 or np.std(t[valid]) == 0:
                    sp = 0.0
                else:
                    sp = spearmanr(m[valid], t[valid]).statistic
                    sp = 0.0 if sp is None or np.isnan(sp) else float(sp)
                metrics[f"spearman_h{h}"] = sp
                spearman_vals.append(sp)
            metrics["spearman_mean"] = float(np.mean(spearman_vals)) if spearman_vals else 0.0

        # ---- Directional accuracy ----------------------------------------
        if "direction" in predictions and "direction" in targets:
            dir_pred = np.asarray(predictions["direction"])
            dir_tgt = np.asarray(targets["direction"])
            # If logits, apply sigmoid then threshold
            if dir_pred.dtype not in (np.float16, np.float32, np.float64):
                pass  # assume already binary
            dir_pred_bin = (1.0 / (1.0 + np.exp(-dir_pred))) > 0.5
            dir_tgt_bin = dir_tgt > 0.5
            acc = (dir_pred_bin == dir_tgt_bin).mean()
            metrics["direction_accuracy"] = float(acc)

        # ---- Calibration error (ECE) -------------------------------------
        if "mu" in predictions and "distribution" in targets:
            metrics["calibration_error"] = _expected_calibration_error(
                np.asarray(predictions["mu"]),
                np.asarray(targets["distribution"]),
            )

        # ---- Regime F1 ---------------------------------------------------
        if "regime" in predictions and "regime" in targets:
            regime_pred = np.asarray(predictions["regime"])
            regime_tgt = np.asarray(targets["regime"])
            if regime_pred.ndim > 1:
                regime_pred = regime_pred.argmax(axis=-1)
            if regime_tgt.ndim > 1:
                regime_tgt = regime_tgt.argmax(axis=-1)
            try:
                f1 = f1_score(regime_tgt, regime_pred, average="macro", zero_division=0)
            except ValueError:
                f1 = 0.0
            metrics["regime_f1"] = float(f1)

        # ---- Trading proxy metrics ---------------------------------------
        if "mu" in predictions and "close" in targets:
            trading = _compute_trading_proxy(
                np.asarray(predictions["mu"]),
                np.asarray(targets["close"]),
            )
            metrics.update(trading)

        # ---- NLL (if log_sigma available) --------------------------------
        if "mu" in predictions and "log_sigma" in predictions and "distribution" in targets:
            mu = np.asarray(predictions["mu"])
            log_sig = np.asarray(predictions["log_sigma"])
            tgt = np.asarray(targets["distribution"])
            log_sig = np.clip(log_sig, -20, 20)
            nll = np.mean(log_sig + 0.5 * ((tgt - mu) / np.exp(log_sig)) ** 2)
            metrics["gaussian_nll"] = float(nll)

        # ---- Log all -----------------------------------------------------
        if metrics:
            self.log_scalars(prefix, metrics, step)

        return metrics

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self, name: str) -> List[Dict[str, Any]]:
        """Return the in-memory log history for a given metric name."""
        return list(self._memory_log.get(name, []))

    def get_all_metric_names(self) -> List[str]:
        """Return all metric names that have been logged."""
        return sorted(self._memory_log.keys())


# ---------------------------------------------------------------------------
# Standalone metric computation helpers
# ---------------------------------------------------------------------------

def _expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for regression.

    Bins predictions by quantile, compares mean prediction to mean target
    within each bin.  Returns a single ECE value averaged over all horizons.

    Args:
        predictions: (N, H) predicted means.
        targets: (N, H) ground-truth values.
        n_bins: Number of calibration bins.

    Returns:
        Mean ECE across horizons.
    """
    N, H = predictions.shape
    ece_per_horizon = []
    for h in range(H):
        pred_h = predictions[:, h]
        tgt_h = targets[:, h]
        valid = np.isfinite(pred_h) & np.isfinite(tgt_h)
        if valid.sum() < n_bins:
            ece_per_horizon.append(np.nan)
            continue

        pred_valid = pred_h[valid]
        tgt_valid = tgt_h[valid]

        bin_edges = np.percentile(pred_valid, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(pred_valid, bin_edges[1:-1])  # 0..n_bins-1

        ece = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            bin_diff = np.abs(pred_valid[mask].mean() - tgt_valid[mask].mean())
            ece += (mask.sum() / len(pred_valid)) * bin_diff
        ece_per_horizon.append(ece)

    ece_vals = [v for v in ece_per_horizon if not np.isnan(v)]
    return float(np.mean(ece_vals)) if ece_vals else 0.0


def _compute_trading_proxy(
    mu: np.ndarray,
    close_prices: np.ndarray,
    transaction_cost_pct: float = 0.0005,
) -> Dict[str, float]:
    """Quick backtest proxy to estimate Sharpe and max drawdown from predictions.

    Uses a simple position rule:
        if predicted mu[:, 0] > 0 → long, else → flat.
        (For a proper backtest, use horizon-weighted ensemble — this is a
        lightweight proxy for training monitoring.)

    Args:
        mu: (N, H) predicted means — we use the first horizon column.
        close_prices: (N,) close price series (aligned to mu rows).
        transaction_cost_pct: Fractional transaction cost per trade.

    Returns:
        Dict with ``"proxy_sharpe"`` and ``"proxy_max_drawdown"``.
    """
    if mu.ndim == 2:
        signal = mu[:, 0]  # first horizon only
    else:
        signal = mu

    if len(close_prices) < 2:
        return {"proxy_sharpe": 0.0, "proxy_max_drawdown": 0.0}

    # Daily/bar returns
    returns = np.diff(close_prices) / (close_prices[:-1] + 1e-12)

    # Position: +1 if signal > 0, else 0 (no short for proxy)
    positions = (signal[:-1] > 0).astype(float)

    # Transaction cost for position changes
    pos_changes = np.diff(np.concatenate([[0], positions]))
    costs = np.abs(pos_changes) * transaction_cost_pct

    strategy_returns = positions * returns - costs

    # Sharpe (annualised proxy — just mean/std * sqrt(252) if daily, else raw)
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = float(mean_ret / (std_ret + 1e-12)) if std_ret > 1e-12 else 0.0

    # Max drawdown
    cumulative = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / (running_max + 1e-12)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "proxy_sharpe": sharpe,
        "proxy_max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Convenience: compute Spearman per horizon (standalone)
# ---------------------------------------------------------------------------

def compute_spearman_per_horizon(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute Spearman rank correlation for each forecast horizon.

    Args:
        predictions: (N, H) predicted values.
        targets: (N, H) ground-truth values.

    Returns:
        Dict of ``"spearman_h{i}"`` and ``"spearman_mean"``.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    H = min(predictions.shape[1], targets.shape[1])
    spearman_vals = []
    metrics = {}
    for h in range(H):
        p = predictions[:, h]
        t = targets[:, h]
        valid = np.isfinite(p) & np.isfinite(t)
        if valid.sum() < 2 or np.std(p[valid]) == 0 or np.std(t[valid]) == 0:
            sp = 0.0
        else:
            sp = spearmanr(p[valid], t[valid]).statistic
            sp = 0.0 if sp is None or np.isnan(sp) else float(sp)
        metrics[f"spearman_h{h}"] = sp
        spearman_vals.append(sp)
    metrics["spearman_mean"] = float(np.mean(spearman_vals)) if spearman_vals else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Convenience: compute directional accuracy per horizon
# ---------------------------------------------------------------------------

def compute_directional_accuracy_per_horizon(
    direction_logits: np.ndarray,
    direction_targets: np.ndarray,
) -> Dict[str, float]:
    """Compute binary classification accuracy for each forecast horizon.

    Args:
        direction_logits: (N, H) raw logits.
        direction_targets: (N, H) binary (0/1) targets.

    Returns:
        Dict of ``"dir_acc_h{i}"`` and ``"dir_acc_mean"``.
    """
    direction_logits = np.asarray(direction_logits, dtype=np.float64)
    direction_targets = np.asarray(direction_targets, dtype=np.float64)

    H = min(direction_logits.shape[1], direction_targets.shape[1])
    metrics = {}
    accs = []
    for h in range(H):
        preds = (1.0 / (1.0 + np.exp(-direction_logits[:, h]))) > 0.5
        tgt = direction_targets[:, h] > 0.5
        acc = float(np.mean(preds == tgt)) if len(preds) > 0 else 0.0
        metrics[f"dir_acc_h{h}"] = acc
        accs.append(acc)
    metrics["dir_acc_mean"] = float(np.mean(accs)) if accs else 0.0
    return metrics
