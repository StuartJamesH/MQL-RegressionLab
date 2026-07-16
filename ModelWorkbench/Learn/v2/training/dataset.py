"""
dataset.py — Memory-efficient PyTorch Dataset for supervised finetuning.

Provides ``FinetuneDataset``, which loads, normalises, and indexes OHLCV
CSVs for distributional finetuning.  Only 2-D normalised arrays are stored;
sliding windows are sliced on-the-fly in ``__getitem__`` — no pre-expansion
to 3-D window tensors.

This is the counterpart to ``MultiInstrumentDataset`` (pretraining path) but
uses the same normalisation / label scheme as ``train_transformer.py``
(log-ratio OHLCV, cyclical session features, directional return labels).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Learn.v2.data import normalize_ohlcv, SessionFeatureEncoder
from Learn.v2.labels import (
    compute_directional_return_distribution,
    compute_forward_excursion_surface,
)
from Learn.train_utils import load_ohlcv

DEFAULT_HORIZONS = [5, 10, 20, 40, 60, 120]


def _compute_atr_normalized_targets(df: pd.DataFrame, horizons: List[int], atr_window: int = 14) -> np.ndarray:
    """
    Compute ATR-normalized forward excursion scores as targets.

    For each bar t and horizon h, computes:
        score = (buy_MFE - buy_MAE) / max(buy_MFE + buy_MAE, 1e-8)

    This produces a signed score in [-1, 1] where:
        +1 = price only moved favorably (pure win)
        -1 = price only moved adversely (pure loss)
         0 = equal movement or no movement

    The MFE/MAE values are in ATR units, making the target scale-invariant.
    """
    excursion = compute_forward_excursion_surface(df, horizons, atr_window=atr_window)
    buy_mfe = excursion[:, :, 0, 0]
    buy_mae = excursion[:, :, 0, 1]

    denom = np.maximum(buy_mfe + buy_mae, 1e-8)
    score = (buy_mfe - buy_mae) / denom
    score = np.nan_to_num(score, nan=0.0)
    return score.astype(np.float32)


class FinetuneDataset(Dataset):
    """Memory-efficient PyTorch Dataset for transformer finetuning.

    Stores only 2-D arrays per instrument (normalised OHLCV, session
    features, labels) and slices sliding windows on-the-fly during
    ``__getitem__``.  This avoids the ~10 GB peak memory cost of
    pre-computing all 3-D window tensors, making it feasible to train
    on datasets of arbitrary size.

    Parameters
    ----------
    ds_paths : list of str
        Paths to OHLCV CSV files.
    n_rows : int
        Maximum rows to read per CSV (pass to ``load_ohlcv``).
    seq_len : int
        Sliding window length in bars.
    target_type : str
        ``"log_return"`` (directional return distribution) or
        ``"atr_score"`` (ATR-normalised MFE/MAE score).
    horizons : list of int
        Forward horizons (bars) for label computation.
    """

    def __init__(
        self,
        ds_paths: List[str],
        n_rows: int,
        seq_len: int,
        target_type: str = "log_return",
        horizons: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.target_type = target_type
        self.horizons = horizons or DEFAULT_HORIZONS

        self._raw: List[np.ndarray] = []
        self._sess: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._window_starts: List[np.ndarray] = []

        self._num_datasets = 0
        self._total_windows = 0

        for ds_path in ds_paths:
            self._add_dataset(ds_path, n_rows)

        if self._num_datasets == 0:
            raise RuntimeError("No valid datasets loaded.")

        self._flat_starts = np.concatenate(self._window_starts).astype(np.int64)
        self._flat_instrument_ids = np.concatenate([
            np.full(len(starts), i, dtype=np.int64)
            for i, starts in enumerate(self._window_starts)
        ])
        self._total_windows = len(self._flat_starts)

        print(f"FinetuneDataset: {self._num_datasets} instrument(s), "
              f"{self._total_windows:,} total windows, {self._raw[0].shape[1]} OHLCV channels, "
              f"{self._sess[0].shape[1]} session channels, "
              f"{len(self.horizons)} label horizons")

    # ------------------------------------------------------------------
    # Internal: load & index one CSV
    # ------------------------------------------------------------------

    def _add_dataset(self, ds_path: str, n_rows: int) -> None:
        df = load_ohlcv(ds_path, n_rows=n_rows)
        n_bars = len(df)
        print(f"  Loaded {ds_path}: {n_bars:,} rows")

        if n_bars < self.seq_len:
            print(f"  Skipping {ds_path}: only {n_bars} bars (need >= {self.seq_len})")
            return

        inst_id = self._num_datasets

        X_raw = normalize_ohlcv(df).astype(np.float32)

        encoder = SessionFeatureEncoder()
        times = pd.to_datetime(df["Time"])
        X_sess = encoder.encode(times, include_gap=True).astype(np.float32)

        if self.target_type == "atr_score":
            labels = _compute_atr_normalized_targets(df, self.horizons).astype(np.float32)
        else:
            labels = compute_directional_return_distribution(df, self.horizons).astype(np.float32)

        max_h = max(self.horizons)
        n_windows_raw = max(0, n_bars - self.seq_len)
        starts_all = np.arange(n_windows_raw, dtype=np.int64)

        # Filter windows whose label bar lands in the NaN tail
        label_indices = starts_all + self.seq_len - 1
        label_vals = labels[label_indices]
        valid = ~np.isnan(label_vals).any(axis=1)

        starts = starts_all[valid]
        if len(starts) == 0:
            print(f"  Skipping {ds_path}: all windows filtered")
            return

        self._raw.append(X_raw)
        self._sess.append(X_sess)
        self._labels.append(labels)
        self._window_starts.append(starts)
        self._num_datasets += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_datasets(self) -> int:
        return self._num_datasets

    @property
    def total_windows(self) -> int:
        return self._total_windows

    @property
    def n_labels(self) -> int:
        return self._labels[0].shape[1] if self._labels else 0

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self._flat_starts[idx])
        inst_id = int(self._flat_instrument_ids[idx])

        raw = self._raw[inst_id][start : start + self.seq_len].copy()
        sess = self._sess[inst_id][start : start + self.seq_len].copy()
        label = self._labels[inst_id][start + self.seq_len - 1].copy()

        return (
            torch.from_numpy(raw),
            torch.from_numpy(sess),
            torch.from_numpy(label),
        )
