"""
losses.py — Specialized loss functions for distributional regression and
composite multi-task training.

Provides:
    gaussian_nll_loss  — Negative log-likelihood under a Gaussian
    pinball_loss       — Multi-horizon multi-quantile pinball loss
    quantile_loss      — Pinball loss with explicit quantile_levels
    focal_loss         — Binary focal loss for direction prediction
    composite_loss     — Weighted multi-objective loss combining all heads
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gaussian NLL
# ---------------------------------------------------------------------------

def gaussian_nll_loss(
    mu: Tensor,
    log_sigma: Tensor,
    target: Tensor,
    clamp_log_sigma: float = -20.0,
) -> Tensor:
    """Negative log-likelihood under a per-horizon Gaussian distribution.

    L = 0.5 * log(2*pi) + log_sigma + 0.5 * ((target - mu) / exp(log_sigma))^2

    Args:
        mu: Predicted mean, shape (B, n_horizons).
        log_sigma: Predicted log-standard-deviation, same shape as mu.
        target: Ground-truth target, same shape as mu.
        clamp_log_sigma: Floor for log_sigma to prevent numerical issues.

    Returns:
        Scalar loss averaged over batch and horizons.

    Raises:
        ValueError: If shape mismatch between mu, log_sigma, target.
    """
    if mu.shape != log_sigma.shape or mu.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: mu={tuple(mu.shape)}, log_sigma={tuple(log_sigma.shape)}, "
            f"target={tuple(target.shape)} — all must match."
        )

    log_sigma = torch.clamp(log_sigma, min=clamp_log_sigma)
    sigma = torch.exp(log_sigma)
    nll = log_sigma + 0.5 * ((target - mu) / sigma) ** 2
    nll = nll + 0.5 * math.log(2.0 * math.pi)

    return nll.mean()


# ---------------------------------------------------------------------------
# Pinball / Quantile loss — multi-horizon, multi-quantile
# ---------------------------------------------------------------------------

def pinball_loss(quantiles: Tensor, target: Tensor) -> Tensor:
    """Pinball (quantile) loss for multi-horizon, multi-quantile predictions.

    L_τ(y, q) = max(τ·(y−q), (τ−1)·(y−q))

    Quantile levels τ are inferred automatically by splitting the unit
    interval uniformly over the last dimension of ``quantiles``.

    Args:
        quantiles: Predicted quantiles, shape (B, n_horizons, n_quantiles).
        target: Ground truth, shape (B, n_horizons).

    Returns:
        Scalar loss averaged over batch, horizons, and quantile levels.
    """
    B, H, K = quantiles.shape
    if target.dim() == 2:
        target = target.unsqueeze(-1)  # (B, H) -> (B, H, 1)

    if target.shape[:2] != (B, H):
        raise ValueError(
            f"Target shape mismatch: expected first two dims ({B}, {H}), "
            f"got {tuple(target.shape)}"
        )

    # Infer quantile levels: evenly spaced [1/(K+1), K/(K+1)]
    tau = torch.linspace(
        1.0 / (K + 1), K / (K + 1), K,
        device=quantiles.device, dtype=quantiles.dtype,
    ).view(1, 1, -1)

    errors = target - quantiles  # (B, H, K)
    loss = torch.maximum(tau * errors, (tau - 1.0) * errors)
    return loss.mean()


def quantile_loss(
    predictions: Tensor,
    targets: Tensor,
    quantile_levels: Tensor,
) -> Tensor:
    """Pinball loss with explicitly provided quantile levels.

    Args:
        predictions: Predicted values, shape (B, n_horizons, n_quantiles).
        targets: Ground truth, shape (B, n_horizons).
        quantile_levels: 1-D tensor of quantile levels in [0, 1] in
            strictly increasing order, shape (n_quantiles,).

    Returns:
        Scalar loss averaged over batch, horizons, and quantile levels.
    """
    B, H, K = predictions.shape
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    if quantile_levels.dim() != 1 or quantile_levels.shape[0] != K:
        raise ValueError(
            f"quantile_levels shape must be ({K},), got {tuple(quantile_levels.shape)}"
        )

    tau = quantile_levels.to(
        device=predictions.device, dtype=predictions.dtype,
    ).view(1, 1, -1)

    errors = targets - predictions
    loss = torch.maximum(tau * errors, (tau - 1.0) * errors)
    return loss.mean()


# ---------------------------------------------------------------------------
# Focal loss (direction / classification)
# ---------------------------------------------------------------------------

def focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Binary cross-entropy with focal weighting for direction prediction.

    FL(p_t) = −α_t * (1−p_t)^γ * log(p_t)

    Args:
        logits: Raw logits, shape (B, n_horizons) or (B, n_horizons, 1).
        targets: Binary targets (0/1), broadcastable to logits shape.
        gamma: Focusing parameter — higher values down-weight easy examples.
        alpha: Class-balancing weight for the positive class.

    Returns:
        Scalar loss averaged over all elements.
    """
    if logits.shape != targets.shape:
        targets = targets.view_as(logits)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits)
    # p_t for the *true* class: if target == 1 use p, else use 1−p
    p_t_correct = p_t * targets + (1.0 - p_t) * (1.0 - targets)
    focal_weight = (1.0 - p_t_correct) ** gamma

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * focal_weight * bce

    return loss.mean()


# ---------------------------------------------------------------------------
# Composite loss (multi-head weighted combination)
# ---------------------------------------------------------------------------

def composite_loss(
    model_output,                                   # ModelOutput
    targets: Dict[str, Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute total loss as a weighted sum across all prediction heads.

    Args:
        model_output: ``ModelOutput`` namedtuple/dataclass with fields:
            ``distribution`` (has ``mu``, ``log_sigma``),
            ``direction`` (logits), ``volatility``, ``regime``,
            and optionally ``quantiles``.
        targets: Dict mapping head names to ground-truth tensors:
            ``"distribution_target"`` — (B, n_horizons),
            ``"direction_target"`` — (B, n_horizons) binary,
            ``"volatility_target"`` — (B, n_horizons),
            ``"regime_target"`` — (B,) class indices or (B, C) one-hot.
        loss_weights: Optional dict overriding default weights.
            Defaults: distribution=1.0, direction=0.3, volatility=0.2, regime=0.1.

    Returns:
        (total_loss, per_component_dict) where per_component_dict maps
        head name → scalar loss tensor (detached) for logging.
    """
    if loss_weights is None:
        loss_weights = {
            "distribution": 1.0,
            "direction":    0.3,
            "volatility":   0.2,
            "regime":       0.1,
        }

    components: Dict[str, Tensor] = {}
    total = torch.tensor(0.0, device=_get_device(model_output))

    # ---- distribution loss ------------------------------------------------
    if "distribution_target" in targets and loss_weights.get("distribution", 0) != 0:
        dist_target = targets["distribution_target"]
        mu = model_output.distribution.mu
        log_sigma = model_output.distribution.log_sigma

        dist_loss = gaussian_nll_loss(mu, log_sigma, dist_target)
        components["distribution"] = dist_loss.detach()
        total = total + loss_weights["distribution"] * dist_loss

    # ---- quantile loss (if quantile head present) -------------------------
    if "distribution_target" in targets and hasattr(model_output, "quantiles"):
        quantiles_tensor = model_output.quantiles
        if quantiles_tensor is not None:
            dist_target = targets["distribution_target"]
            q_loss = pinball_loss(quantiles_tensor, dist_target)
            q_weight = loss_weights.get("quantile", 0.5)
            components["quantile"] = q_loss.detach()
            total = total + q_weight * q_loss

    # ---- direction loss (binary focal) ------------------------------------
    if "direction_target" in targets and loss_weights.get("direction", 0) != 0:
        dir_target = targets["direction_target"]
        dir_logits = model_output.direction
        dir_loss = focal_loss(dir_logits, dir_target)
        components["direction"] = dir_loss.detach()
        total = total + loss_weights["direction"] * dir_loss

    # ---- volatility loss (MSE) --------------------------------------------
    if "volatility_target" in targets and loss_weights.get("volatility", 0) != 0:
        vol_target = targets["volatility_target"]
        vol_pred = model_output.volatility
        vol_loss = F.mse_loss(vol_pred, vol_target)
        components["volatility"] = vol_loss.detach()
        total = total + loss_weights["volatility"] * vol_loss

    # ---- regime loss (cross-entropy) --------------------------------------
    if "regime_target" in targets and loss_weights.get("regime", 0) != 0:
        regime_target = targets["regime_target"]
        regime_logits = model_output.regime
        # Handle both class-index and one-hot targets
        if regime_target.dim() > 1 and regime_target.shape[-1] > 1:
            regime_target = regime_target.argmax(dim=-1)
        regime_target = regime_target.long()
        regime_loss = F.cross_entropy(
            regime_logits.view(-1, regime_logits.shape[-1]),
            regime_target.view(-1),
        )
        components["regime"] = regime_loss.detach()
        total = total + loss_weights["regime"] * regime_loss

    return total, components


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_device(model_output) -> torch.device:
    """Infer device from the first relevant tensor in model_output."""
    if hasattr(model_output, "distribution") and model_output.distribution is not None:
        return model_output.distribution.mu.device
    if hasattr(model_output, "direction") and model_output.direction is not None:
        return model_output.direction.device
    return torch.device("cpu")
