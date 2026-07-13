"""
ModelConfig — single source of truth for the Causal Patch Transformer architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for the Causal Patch Transformer model."""

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    in_channels: int = 5
    session_channels: int = 5
    patch_len: int = 16
    patch_stride: int = 8

    max_seq_len: int = 512

    n_horizons: int = 6
    n_regimes: int = 4
    n_quantiles: int = 5

    n_timeframes: int = 5

    use_mixed_precision: bool = True
    weight_decay: float = 1e-4
    quantized: bool = False

    use_quantile_head: bool = False
    use_mtf_fusion: bool = False

    @property
    def total_channels(self) -> int:
        return self.in_channels + self.session_channels

    @property
    def max_n_patches(self) -> int:
        if self.max_seq_len < self.patch_len:
            return 0
        return (self.max_seq_len - self.patch_len) // self.patch_stride + 1

    def n_patches_for_len(self, seq_len: int) -> int:
        if seq_len < self.patch_len:
            return 0
        return (seq_len - self.patch_len) // self.patch_stride + 1

    def to_dict(self) -> dict:
        """Serialize config to a JSON-compatible dictionary."""
        d = asdict(self)
        d["total_channels"] = self.total_channels
        d["max_n_patches"] = self.max_n_patches
        return d
