"""
pretrain_data.py — Multi-instrument dataset for self-supervised pretraining.

Provides ``MultiInstrumentDataset``, which loads raw OHLCV CSVs, normalises
each instrument by rolling-ATR, creates causally-masked sliding windows, and
samples them with probability proportional to instrument size.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation utility
# ---------------------------------------------------------------------------

def _normalize_ohlcv_by_atr(
    df: pd.DataFrame,
    ohlc_cols: Tuple[str, str, str, str] = ("Open", "High", "Low", "Close"),
    volume_col: Optional[str] = None,
    atr_window: int = 20,
    clip: float = 10.0,
) -> np.ndarray:
    """Normalise OHLC (and optionally volume) by rolling ATR of the Close.

    Args:
        df: DataFrame with columns Open, High, Low, Close.
        ohlc_cols: Column names for OHLC.
        volume_col: Optional volume column name.
        atr_window: Window for rolling ATR and EMA.
        clip: Maximum absolute value after normalisation.

    Returns:
        ndarray of shape (N, n_features) with normalised values.
    """
    o, h, l, c = ohlc_cols

    # Rolling true-range
    prev_close = df[c].shift(1)
    tr = np.maximum(
        df[h] - df[l],
        np.maximum(
            (df[h] - prev_close).abs(),
            (df[l] - prev_close).abs(),
        ),
    )
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    atr = atr.replace(0.0, np.nan)

    # Centre each column with EMA
    ema_open = df[o].ewm(span=atr_window, adjust=False).mean()
    ema_high = df[h].ewm(span=atr_window, adjust=False).mean()
    ema_low = df[l].ewm(span=atr_window, adjust=False).mean()
    ema_close = df[c].ewm(span=atr_window, adjust=False).mean()

    norm_open = (df[o] - ema_open) / atr
    norm_high = (df[h] - ema_high) / atr
    norm_low = (df[l] - ema_low) / atr
    norm_close = (df[c] - ema_close) / atr

    features_list = [
        norm_open.clip(-clip, clip),
        norm_high.clip(-clip, clip),
        norm_low.clip(-clip, clip),
        norm_close.clip(-clip, clip),
    ]

    if volume_col is not None and volume_col in df.columns:
        ema_vol = df[volume_col].ewm(span=atr_window, adjust=False).mean()
        vol_std = df[volume_col].rolling(window=atr_window, min_periods=atr_window).std()
        vol_std = vol_std.replace(0.0, np.nan)
        norm_vol = (df[volume_col] - ema_vol) / vol_std
        features_list.append(norm_vol.clip(-clip, clip))

    result = np.column_stack([s.to_numpy(dtype=np.float32) for s in features_list])
    mask = ~np.isnan(result).any(axis=1)
    return result[mask]


# ---------------------------------------------------------------------------
# Helper: infer timeframe from filename
# ---------------------------------------------------------------------------

def _infer_timeframe_id(filename: str) -> int:
    """Infer a coarse timeframe ID from the filename."""
    name_upper = os.path.basename(filename).upper()
    mapping = {
        "M1": 0, "M5": 1, "M15": 2, "M30": 3,
        "H1": 4, "H4": 5, "D1": 6, "W1": 7, "MN1": 8,
        "D": 6, "W": 7,
    }
    for key, value in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if key in name_upper:
            return value
    logger.warning("Could not infer timeframe from '%s', defaulting to 0.", filename)
    return 0


# ---------------------------------------------------------------------------
# MultiInstrumentDataset
# ---------------------------------------------------------------------------

class MultiInstrumentDataset(Dataset):
    """A PyTorch Dataset that loads, normalises, and windows multiple
    instrument CSVs for self-supervised pretraining.
    """

    _CSV_REQUIRED_COLS = {"Time", "Open", "High", "Low", "Close"}

    def __init__(
        self,
        dataset_paths: List[str],
        seq_len: int = 512,
        max_horizon: int = 60,
        slide_step: int = 1,
        atr_window: int = 20,
        ohlc_cols: Tuple[str, str, str, str] = ("Open", "High", "Low", "Close"),
        volume_col: Optional[str] = "TickVolume",
        clip: float = 10.0,
        cache_normalized: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        self.window_len = seq_len + max_horizon
        self.slide_step = slide_step
        self.atr_window = atr_window
        self.ohlc_cols = ohlc_cols
        self.volume_col = volume_col
        self.clip = clip
        self.cache_normalized = cache_normalized

        # Resolve paths (handle globs)
        resolved: List[str] = []
        for p in dataset_paths:
            if "*" in p or "?" in p:
                import glob as _glob
                resolved.extend(sorted(_glob.glob(p)))
            else:
                resolved.append(p)

        if not resolved:
            raise ValueError(f"No datasets found matching: {dataset_paths}")

        self._data: List[np.ndarray] = []
        self._window_starts: List[np.ndarray] = []
        self._instrument_ids: List[np.ndarray] = []
        self._timeframe_ids: List[np.ndarray] = []
        self._n_features: Optional[int] = None
        self._file_paths = resolved
        self._cache: Dict[int, np.ndarray] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Load each CSV, normalise, and build a flat window index."""
        total_windows = 0
        self._data.clear()
        self._window_starts.clear()
        self._instrument_ids.clear()
        self._timeframe_ids.clear()

        for inst_id, fp in enumerate(self._file_paths):
            df = pd.read_csv(fp)
            missing = self._CSV_REQUIRED_COLS - set(df.columns)
            if missing:
                logger.warning("Skipping '%s': missing columns %s", fp, missing)
                continue

            if "Time" in df.columns:
                df["Time"] = pd.to_datetime(df["Time"])
                df = df.sort_values("Time").reset_index(drop=True)

            num_bars = len(df)
            if num_bars < self.window_len + self.atr_window:
                logger.warning(
                    "Skipping '%s': only %d bars, need at least %d",
                    fp, num_bars, self.window_len + self.atr_window,
                )
                continue

            norm = _normalize_ohlcv_by_atr(
                df,
                ohlc_cols=self.ohlc_cols,
                volume_col=self.volume_col,
                atr_window=self.atr_window,
                clip=self.clip,
            )
            if len(norm) < self.window_len:
                logger.warning(
                    "Skipping '%s': after normalisation only %d valid bars",
                    fp, len(norm),
                )
                continue

            if self._n_features is None:
                self._n_features = norm.shape[1]
            elif norm.shape[1] != self._n_features:
                logger.warning(
                    "Skipping '%s': feature count %d != expected %d",
                    fp, norm.shape[1], self._n_features,
                )
                continue

            self._data.append(norm)
            if self.cache_normalized:
                self._cache[inst_id] = norm

            num_windows = (len(norm) - self.window_len) // self.slide_step + 1
            starts = np.arange(0, num_windows * self.slide_step, self.slide_step, dtype=np.int64)
            self._window_starts.append(starts)

            tf_id = _infer_timeframe_id(fp)
            inst_ids = np.full(len(starts), inst_id, dtype=np.int64)
            tf_ids = np.full(len(starts), tf_id, dtype=np.int64)
            self._instrument_ids.append(inst_ids)
            self._timeframe_ids.append(tf_ids)

            total_windows += len(starts)
            logger.info(
                "Loaded '%s': %d bars -> %d windows (norm %d features, tf_id=%d)",
                os.path.basename(fp), num_bars, len(starts), self._n_features, tf_id,
            )

        if total_windows == 0:
            raise RuntimeError(
                "No valid datasets after loading. Check CSV columns, "
                "bar counts, and atr_window settings."
            )

        self._flat_starts = np.concatenate(self._window_starts)
        self._flat_instrument_ids = np.concatenate(self._instrument_ids)
        self._flat_timeframe_ids = np.concatenate(self._timeframe_ids)

        self._sample_weights = np.ones(len(self._flat_starts), dtype=np.float64)
        self._sample_weights /= self._sample_weights.sum()

        logger.info(
            "MultiInstrumentDataset built: %d instruments, %d total windows, %d features",
            len(self._data), total_windows, self._n_features,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("Dataset not initialised (no data loaded).")
        return self._n_features

    @property
    def num_instruments(self) -> int:
        return len(self._data)

    @property
    def total_windows(self) -> int:
        return len(self._flat_starts)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._flat_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        start = int(self._flat_starts[idx])
        inst_id = int(self._flat_instrument_ids[idx])
        tf_id = int(self._flat_timeframe_ids[idx])

        if self.cache_normalized and inst_id in self._cache:
            data = self._cache[inst_id]
        else:
            data = self._data[inst_id]

        window = data[start : start + self.window_len].copy()
        full = torch.from_numpy(window)
        mask = torch.zeros(self.window_len, dtype=torch.bool)

        return full, full.clone(), mask, inst_id, tf_id

    def sample_weighted_index(self) -> int:
        """Return a single index sampled proportionally to instrument size."""
        return int(np.random.choice(len(self._flat_starts), p=self._sample_weights))

    def sample_weighted_batch_indices(self, batch_size: int) -> List[int]:
        """Return a list of indices sampled with replacement, weighted by instrument size."""
        return [self.sample_weighted_index() for _ in range(batch_size)]
