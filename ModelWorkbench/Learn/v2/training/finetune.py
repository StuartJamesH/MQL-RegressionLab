"""
finetune.py — Distributional Fine-Tuning (Phase 2 of three-phase curriculum).

Implements ``DistributionalFinetuning`` — supervised fine-tuning on distributional
targets with multi-task composite loss, cosine LR schedule with warmup, gradient
clipping, and early stopping.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from Learn.v2.training.losses import composite_loss, gaussian_nll_loss, pinball_loss

logger = logging.getLogger(__name__)


class DistributionalFinetuning:
    """Phase 2 trainer: supervised fine-tuning on distributional targets.

    Supports Gaussian NLL and Pinball loss modes.  Uses cosine LR schedule
    with warmup, gradient clipping, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: Optional[torch.device] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        loss_mode: str = "gaussian",
        pinball_quantiles: Optional[List[float]] = None,
        warmup_steps: int = 1000,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        max_steps: int = 100000,
        grad_clip_norm: float = 1.0,
        patience: int = 10,
        patience_metric: str = "val_spearman_mean",
        patience_mode: str = "max",
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.config = config

        self.loss_weights = loss_weights or {
            "distribution": 1.0, "direction": 0.3,
            "volatility": 0.2, "regime": 0.1,
        }
        self.loss_mode = loss_mode
        self.pinball_quantiles = pinball_quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

        # LR schedule params
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.grad_clip_norm = grad_clip_norm

        # Early stopping
        self.patience = patience
        self.patience_metric = patience_metric
        self.patience_mode = patience_mode
        self._best_metric: float = float("-inf") if patience_mode == "max" else float("inf")
        self._epochs_without_improvement: int = 0
        self._stopped: bool = False

        # Training state
        self.global_step: int = 0
        self.current_lr: float = 0.0
        self._train_losses: List[float] = []
        self._val_metrics_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def _get_lr(self, step: int) -> float:
        """Cosine annealing with linear warmup."""
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

    def _apply_lr(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        lr = self._get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        self.current_lr = lr
        return lr

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> float:
        """Run a single fine-tuning epoch. Returns average total loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch[0].to(self.device)
            targets = self._unpack_targets(batch)

            lr = self._apply_lr(optimizer, self.global_step)
            self.global_step += 1

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                model_output = self.model(x)

                if self.loss_mode == "pinball" and hasattr(model_output, "quantiles"):
                    total, components = self._compute_pinball_composite(
                        model_output, targets,
                    )
                else:
                    total, components = composite_loss(
                        model_output, targets, self.loss_weights,
                    )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()

            total_loss += total.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self._train_losses.append(avg_loss)
        logger.info(
            "Finetune epoch %d: train_loss=%.6f, lr=%.2e, step=%d",
            len(self._train_losses), avg_loss, self.current_lr, self.global_step,
        )
        return avg_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation, compute Spearman per horizon, dir accuracy, etc."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_mu: List[torch.Tensor] = []
        all_dist_target: List[torch.Tensor] = []
        all_dir_logits: List[torch.Tensor] = []
        all_dir_target: List[torch.Tensor] = []
        all_vol_pred: List[torch.Tensor] = []
        all_vol_target: List[torch.Tensor] = []
        all_regime_logits: List[torch.Tensor] = []
        all_regime_target: List[torch.Tensor] = []

        for batch in dataloader:
            x = batch[0].to(self.device)
            targets = self._unpack_targets(batch)

            model_output = self.model(x)
            total, _ = composite_loss(model_output, targets, self.loss_weights)
            total_loss += total.item()
            n_batches += 1

            if hasattr(model_output, "distribution") and model_output.distribution is not None:
                all_mu.append(model_output.distribution.mu.detach().cpu())
                if "distribution_target" in targets:
                    all_dist_target.append(targets["distribution_target"].detach().cpu())

            if hasattr(model_output, "direction") and model_output.direction is not None:
                all_dir_logits.append(model_output.direction.detach().cpu())
                if "direction_target" in targets:
                    all_dir_target.append(targets["direction_target"].detach().cpu())

            if hasattr(model_output, "volatility") and model_output.volatility is not None:
                all_vol_pred.append(model_output.volatility.detach().cpu())
                if "volatility_target" in targets:
                    all_vol_target.append(targets["volatility_target"].detach().cpu())

            if hasattr(model_output, "regime") and model_output.regime is not None:
                all_regime_logits.append(model_output.regime.detach().cpu())
                if "regime_target" in targets:
                    all_regime_target.append(targets["regime_target"].detach().cpu())

        metrics: Dict[str, float] = {}
        metrics["val_loss"] = total_loss / max(n_batches, 1)

        # ---- Spearman per horizon ----
        if all_mu and all_dist_target:
            mu_cat = torch.cat(all_mu, dim=0).numpy()
            target_cat = torch.cat(all_dist_target, dim=0).numpy()
            n_h = min(mu_cat.shape[1], target_cat.shape[1])
            spearman_vals = []
            for h in range(n_h):
                m = mu_cat[:, h]
                t = target_cat[:, h]
                valid = np.isfinite(m) & np.isfinite(t)
                if valid.sum() < 2 or np.std(m[valid]) == 0 or np.std(t[valid]) == 0:
                    sp = 0.0
                else:
                    sp = spearmanr(m[valid], t[valid]).statistic
                    sp = 0.0 if sp is None or np.isnan(sp) else float(sp)
                metrics[f"spearman_h{h}"] = sp
                spearman_vals.append(sp)
            metrics["spearman_mean"] = float(np.mean(spearman_vals)) if spearman_vals else 0.0

        # ---- Directional accuracy ----
        if all_dir_logits and all_dir_target:
            dl = torch.cat(all_dir_logits, dim=0)
            dt = torch.cat(all_dir_target, dim=0)
            preds = (torch.sigmoid(dl) > 0.5).float()
            acc = (preds.view(-1) == dt.view(-1)).float().mean().item()
            metrics["direction_accuracy"] = acc

        # ---- Volatility MSE ----
        if all_vol_pred and all_vol_target:
            vp = torch.cat(all_vol_pred, dim=0)
            vt = torch.cat(all_vol_target, dim=0)
            metrics["volatility_mse"] = F.mse_loss(vp, vt).item()

        # ---- Regime F1 ----
        if all_regime_logits and all_regime_target:
            from sklearn.metrics import f1_score
            rl = torch.cat(all_regime_logits, dim=0)
            rt = torch.cat(all_regime_target, dim=0)
            if rt.dim() > 1 and rt.shape[-1] > 1:
                rt = rt.argmax(dim=-1)
            rt = rt.long()
            rp = rl.argmax(dim=-1)
            try:
                f1 = f1_score(rt.numpy(), rp.numpy(), average="macro", zero_division=0)
            except ValueError:
                f1 = 0.0
            metrics["regime_f1"] = float(f1)

        # ---- Early stopping ----
        self._val_metrics_history.append(metrics)
        metric_val = metrics.get(self.patience_metric, 0.0)
        improved = (
            metric_val > self._best_metric if self.patience_mode == "max"
            else metric_val < self._best_metric
        )
        if improved:
            self._best_metric = metric_val
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.patience:
            self._stopped = True
            logger.info("Early stopping triggered (patience=%d)", self.patience)

        return metrics

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def should_stop(self) -> bool:
        return self._stopped

    @property
    def best_metric(self) -> float:
        return self._best_metric

    @property
    def val_metrics_history(self) -> List[Dict[str, float]]:
        return self._val_metrics_history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "best_metric": self._best_metric,
            "epochs_without_improvement": self._epochs_without_improvement,
            "train_losses": self._train_losses,
            "val_metrics_history": self._val_metrics_history,
            "loss_weights": self.loss_weights,
            "loss_mode": self.loss_mode,
        }
        torch.save(checkpoint, path)
        logger.info("Finetune checkpoint saved to '%s'", path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self._best_metric = checkpoint.get("best_metric", self._best_metric)
        self._epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self._train_losses = checkpoint.get("train_losses", [])
        self._val_metrics_history = checkpoint.get("val_metrics_history", [])
        self.loss_weights = checkpoint.get("loss_weights", self.loss_weights)
        self.loss_mode = checkpoint.get("loss_mode", self.loss_mode)
        logger.info("Checkpoint loaded from '%s' (step=%d, best=%.4f)", path, self.global_step, self._best_metric)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_targets(self, batch: tuple) -> Dict[str, torch.Tensor]:
        """Convert dataloader batch into targets dict for composite_loss."""
        if len(batch) >= 2 and isinstance(batch[1], dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch[1].items()}

        targets: Dict[str, torch.Tensor] = {}
        if len(batch) >= 2 and batch[1] is not None:
            targets["distribution_target"] = batch[1].to(self.device)
        if len(batch) >= 3 and batch[2] is not None:
            targets["direction_target"] = batch[2].to(self.device)
        if len(batch) >= 4 and batch[3] is not None:
            targets["volatility_target"] = batch[3].to(self.device)
        if len(batch) >= 5 and batch[4] is not None:
            targets["regime_target"] = batch[4].to(self.device)
        return targets

    def _compute_pinball_composite(
        self, model_output, targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute composite loss with pinball quantile loss."""
        from Learn.v2.training.losses import focal_loss
        components: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device)

        if "distribution_target" in targets and hasattr(model_output, "quantiles"):
            qt = model_output.quantiles
            if qt is not None:
                qt_levels = torch.tensor(self.pinball_quantiles, device=self.device, dtype=qt.dtype)
                q_loss = pinball_loss(qt, targets["distribution_target"])
                w = self.loss_weights.get("distribution", 1.0)
                components["quantile"] = q_loss.detach()
                total = total + w * q_loss

        if "direction_target" in targets and hasattr(model_output, "direction"):
            dir_loss = focal_loss(model_output.direction, targets["direction_target"])
            w = self.loss_weights.get("direction", 0.3)
            components["direction"] = dir_loss.detach()
            total = total + w * dir_loss

        if "volatility_target" in targets and hasattr(model_output, "volatility"):
            vol_loss = F.mse_loss(model_output.volatility, targets["volatility_target"])
            w = self.loss_weights.get("volatility", 0.2)
            components["volatility"] = vol_loss.detach()
            total = total + w * vol_loss

        if "regime_target" in targets and hasattr(model_output, "regime"):
            rt = targets["regime_target"]
            if rt.dim() > 1 and rt.shape[-1] > 1:
                rt = rt.argmax(dim=-1)
            rt = rt.long()
            regime_loss = F.cross_entropy(
                model_output.regime.view(-1, model_output.regime.shape[-1]),
                rt.view(-1),
            )
            w = self.loss_weights.get("regime", 0.1)
            components["regime"] = regime_loss.detach()
            total = total + w * regime_loss

        return total, components
