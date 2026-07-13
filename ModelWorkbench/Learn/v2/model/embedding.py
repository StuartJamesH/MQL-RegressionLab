"""
Patch-based embedding modules for the Causal Patch Transformer.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Conv1d-based patch projection with learned position encodings and [CLS] token.

    Forward: (B, seq_len, total_channels) -> (B, n_patches + 1, d_model)
    """

    def __init__(self, seq_len: int, in_channels: int = 5, session_channels: int = 4,
                 patch_len: int = 16, patch_stride: int = 8, d_model: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.total_channels = in_channels + session_channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.seq_len = seq_len
        self.session_channels = session_channels

        if seq_len >= patch_len:
            self.max_n_patches = (seq_len - patch_len) // patch_stride + 1
        else:
            self.max_n_patches = 0

        self.conv = nn.Conv1d(in_channels=self.total_channels, out_channels=d_model,
                              kernel_size=patch_len, stride=patch_stride)
        if self.max_n_patches > 0:
            self.pos_embed = nn.Parameter(torch.randn(1, self.max_n_patches, d_model) * 0.02)
        else:
            self.pos_embed = nn.Parameter(torch.empty(1, 0, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _num_patches(self, seq_len: int) -> int:
        if seq_len < self.patch_len:
            return 0
        return (seq_len - self.patch_len) // self.patch_stride + 1

    def forward(self, x_raw: torch.Tensor, x_session: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x_raw.shape
        if x_session is None:
            x_session = torch.zeros(B, L, self.session_channels, device=x_raw.device, dtype=x_raw.dtype)
        x = torch.cat([x_raw, x_session], dim=-1)
        n_patches = self._num_patches(L)
        if n_patches == 0:
            raise ValueError(f"Sequence length {L} < patch_len {self.patch_len}")
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        pos_embed = self.pos_embed[:, :n_patches, :]
        x = x + pos_embed
        x = self.dropout(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x

    def extra_repr(self) -> str:
        return (f"seq_len={self.seq_len}, d_model={self.d_model}, "
                f"patch_len={self.patch_len}, patch_stride={self.patch_stride}, "
                f"max_n_patches={self.max_n_patches}")


class TimeframeEmbedding(nn.Module):
    """Learned per-timeframe embedding for multi-timeframe fusion."""

    def __init__(self, n_timeframes: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_timeframes, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, timeframe_idx: torch.Tensor) -> torch.Tensor:
        if timeframe_idx.dim() == 0:
            timeframe_idx = timeframe_idx.unsqueeze(0)
        return self.embed(timeframe_idx)

    def extra_repr(self) -> str:
        return f"n_timeframes={self.embed.num_embeddings}, d_model={self.embed.embedding_dim}"
