"""
pretrain.py — Masked Patch Pretraining (Phase 1 of three-phase curriculum).

Implements ``MaskedPatchPretraining`` — an MAE-style self-supervised trainer
that randomly masks input patches and trains a lightweight decoder to
reconstruct the masked OHLCV bars.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight MAE decoder
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """A compact 2-layer transformer decoder reconstructing masked OHLCV patches."""

    def __init__(
        self,
        d_model: int,
        n_features: int,
        dim_feedforward: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.output_proj = nn.Linear(d_model, n_features)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "mask_token" in name:
                continue
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(encoded)
        out = self.output_proj(out)
        return out


# ---------------------------------------------------------------------------
# MaskedPatchPretraining
# ---------------------------------------------------------------------------

class MaskedPatchPretraining:
    """MAE-style masked-patch pretraining trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: Optional[torch.device] = None,
        mask_ratio: float = 0.5,
        decoder_dim_feedforward: int = 128,
        decoder_nhead: int = 4,
        decoder_dropout: float = 0.1,
        grad_clip_norm: float = 1.0,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.mask_ratio = mask_ratio
        self.grad_clip_norm = grad_clip_norm

        # Extract metadata from config
        d_model = getattr(config, "d_model", getattr(config, "hidden_dim", 128))
        n_features = getattr(config, "n_features", getattr(config, "input_dim", 5))
        self.patch_size = getattr(config, "patch_size", 8)

        # Build decoder
        self.decoder = MAEDecoder(
            d_model=d_model,
            n_features=n_features,
            dim_feedforward=decoder_dim_feedforward,
            nhead=decoder_nhead,
            dropout=decoder_dropout,
        ).to(device)

        self.current_lr: float = 0.0
        self._epoch_losses: list = []
        self._best_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    def _generate_random_mask(
        self, batch_size: int, seq_len: int, patch_size: int
    ) -> torch.Tensor:
        """Generate per-patch boolean mask: True = masked."""
        n_patches = (seq_len + patch_size - 1) // patch_size
        mask = torch.rand(batch_size, n_patches, device=self.device) < self.mask_ratio
        return mask

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> float:
        """Run a single pretraining epoch. Returns average reconstruction loss."""
        self.model.train()
        self.decoder.train()

        total_loss = 0.0
        n_batches = 0
        patch_size = self.patch_size

        for batch in dataloader:
            x = batch[0].to(self.device)
            target = batch[1].to(self.device)
            B, S, F = x.shape
            n_patches = (S + patch_size - 1) // patch_size

            patch_mask = self._generate_random_mask(B, S, patch_size)  # (B, n_patches)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # Forward through encoder with mask
                try:
                    encoded = self.model.forward_with_mask(x, mask=~patch_mask)
                except (AttributeError, TypeError):
                    bar_mask = _patch_mask_to_bar_mask(patch_mask, patch_size, S)
                    x_masked = x.clone()
                    x_masked[bar_mask] = 0.0
                    encoded = self.model(x_masked)

                # Ensure patch-level representation
                if encoded.shape[1] != n_patches:
                    encoded = encoded[:, :n_patches * patch_size, :]
                    encoded = encoded.reshape(B, n_patches, -1, encoded.shape[-1])
                    encoded = encoded.mean(dim=2)

                # Insert mask tokens
                decoded_input = self._apply_mask_tokens(encoded, patch_mask)
                recon_patches = self.decoder(decoded_input, patch_mask)

                # Loss on masked patches only
                target_padded = _pad_to_multiple(target, patch_size)
                target_patches = target_padded.reshape(B, n_patches, patch_size, F)
                recon_reshaped = recon_patches.reshape(B, n_patches, -1)
                recon_padded = _pad_to_multiple(recon_reshaped, F)
                recon_padded = recon_padded.reshape(B, n_patches, -1, F)[:, :, :patch_size, :]

                masked_patches = patch_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_size, F)
                loss = F.mse_loss(recon_padded[masked_patches], target_patches[masked_patches])

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.decoder.parameters()),
                    self.grad_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.decoder.parameters()),
                    self.grad_clip_norm,
                )
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if optimizer.param_groups:
                self.current_lr = optimizer.param_groups[0]["lr"]

        avg_loss = total_loss / max(n_batches, 1)
        self._epoch_losses.append(avg_loss)
        if avg_loss < self._best_loss:
            self._best_loss = avg_loss

        _log_gpu_memory(logger)
        logger.info(
            "Pretrain epoch complete: recon_loss=%.6f, lr=%.2e, best=%.6f",
            avg_loss, self.current_lr, self._best_loss,
        )
        return avg_loss

    # ------------------------------------------------------------------
    # Mask token insertion
    # ------------------------------------------------------------------

    def _apply_mask_tokens(
        self, encoded_patches: torch.Tensor, patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Insert learned mask tokens at masked positions."""
        B, N, D = encoded_patches.shape
        mask_tokens = self.decoder.mask_token.expand(B, N, D)
        visible_mask = (~patch_mask).unsqueeze(-1)
        out = encoded_patches * visible_mask + mask_tokens * (~visible_mask)
        return out

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save encoder, decoder, and training state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "epoch_losses": self._epoch_losses,
            "best_loss": self._best_loss,
            "current_lr": self.current_lr,
            "mask_ratio": self.mask_ratio,
        }
        torch.save(checkpoint, path)
        logger.info("Pretrain checkpoint saved to '%s'", path)

    def load_checkpoint(self, path: str, load_decoder: bool = True) -> None:
        """Load encoder (and optionally decoder) weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_decoder and "decoder_state_dict" in checkpoint:
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self._epoch_losses = checkpoint.get("epoch_losses", [])
        self._best_loss = checkpoint.get("best_loss", float("inf"))
        self.current_lr = checkpoint.get("current_lr", 0.0)
        logger.info("Checkpoint loaded from '%s' (best=%.6f)", path, self._best_loss)

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def epoch_losses(self) -> list:
        return self._epoch_losses


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _patch_mask_to_bar_mask(
    patch_mask: torch.Tensor, patch_size: int, seq_len: int,
) -> torch.Tensor:
    """Expand a per-patch boolean mask to a per-bar boolean mask."""
    B, n_patches = patch_mask.shape
    bar_mask = patch_mask.unsqueeze(-1).expand(-1, -1, patch_size)
    bar_mask = bar_mask.reshape(B, n_patches * patch_size)
    return bar_mask[:, :seq_len].contiguous()


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad the sequence dimension of x to a multiple of `multiple`."""
    S = x.shape[1]
    pad_len = (multiple - S % multiple) % multiple
    if pad_len == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_len))


def _log_gpu_memory(log: logging.Logger) -> None:
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        log.debug("GPU memory: allocated=%.2fGB, reserved=%.2fGB", allocated, reserved)
