"""
folds.py — Purged walk-forward cross-validation splitter.

Implements ``PurgedWalkForwardSplit`` following the methodology described in
López de Prado's *Advances in Financial Machine Learning* (Chapter 7).  The
splitter supports both expanding-window and rolling-window modes, with
configurable purge gaps to prevent information leakage across fold
boundaries.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PurgedWalkForwardSplit
# ---------------------------------------------------------------------------

class PurgedWalkForwardSplit:
    """Purged walk-forward cross-validation with expanding or rolling windows.

    In an expanding-window scheme, fold *i* trains on the union of blocks
    0 through *i* (i.e. all history up to the test block), and tests on
    block *i+1*.  In rolling-window mode, each fold trains only on the
    most recent ``lookback`` bars before the test block.

    Crucially, a **purge gap** of ``gap_size`` bars on either side of the
    test block is excluded from the training set.  This prevents
    information leakage from the test-period labels (e.g. triple-barrier
    labels that look ``gap_size`` bars into the future) into the training
    feature windows.

    Usage::

        splitter = PurgedWalkForwardSplit(
            n_folds=5, min_train_size=100_000, test_size=50_000, gap_size=60,
        )
        X = np.random.randn(300_000, 20)          # or a DataFrame
        timestamps = pd.date_range("2020-01-01", periods=300_000, freq="1min")

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, timestamps)
        ):
            X_train, X_val = X[train_idx], X[val_idx]
            ...train and evaluate...

    Attributes:
        n_folds: Number of validation folds.
        min_train_size: Minimum number of bars required in the first
            training block.
        test_size: Number of bars in each test block.
        gap_size: Purge gap (bars on each side of the test block excluded
            from training).
        rolling: If True, use a fixed-size rolling lookback instead of
            expanding windows.
        lookback: When ``rolling=True``, the lookback window size in bars.
            Defaults to ``min_train_size`` if not provided.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_size: int = 100_000,
        test_size: int = 50_000,
        gap_size: int = 60,
        rolling: bool = False,
        lookback: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_folds: Desired number of validation folds.  Fewer folds may
                be produced if data is insufficient.
            min_train_size: Minimum bars for the initial training window.
            test_size: Number of bars per test block.
            gap_size: Purge gap (bars excluded from train on each side
                of the test block).  Use a value ≥ the maximum forward-
                looking label horizon to avoid leakage.
            rolling: If True, each fold uses a fixed-size rolling lookback
                instead of the full expanding history.
            lookback: Lookback window size for rolling mode.  If None,
                falls back to ``min_train_size``.
        """
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if gap_size < 0:
            raise ValueError(f"gap_size must be >= 0, got {gap_size}")

        self.n_folds = int(n_folds)
        self.min_train_size = int(min_train_size)
        self.test_size = int(test_size)
        self.gap_size = int(gap_size)
        self.rolling = bool(rolling)
        self.lookback = int(lookback) if lookback is not None else min_train_size

        if self.rolling and self.lookback < self.test_size + self.gap_size:
            logger.warning(
                "lookback (%d) is smaller than test_size + gap_size (%d); "
                "training windows may be smaller than test blocks.",
                self.lookback, self.test_size + self.gap_size,
            )

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame, Sequence],
        timestamps: Optional[Union[pd.Series, pd.DatetimeIndex, Sequence]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate ``(train_indices, test_indices)`` tuples.

        The indices are integer arrays suitable for indexing ``X`` directly
        (i.e. ``X[train_indices]``).  They are **not** fancy-index arrays
        — they are plain flat integer positions into the 0-th axis of ``X``.

        If ``timestamps`` are provided, the splitter includes metadata in
        the log output but does *not* use timestamps to determine fold
        boundaries — only integer positions matter (the caller is
        responsible for providing time-ordered data).

        Args:
            X: Feature array (or any sequence that can report ``len``).
                Used only to determine the total number of samples.
            timestamps: Optional time series aligned with X for logging.

        Yields:
            ``(train_indices, test_indices)`` — each a 1-D ``int64``
            ``ndarray`` of positions into ``X``.
        """
        n_samples = len(X)
        if n_samples == 0:
            logger.warning("Empty dataset; no folds to yield.")
            return

        # Compute fold boundary positions
        fold_boundaries = self._compute_boundaries(n_samples)
        if len(fold_boundaries) < 2:
            logger.warning(
                "Insufficient data for walk-forward splitting: "
                "need at least %d bars, have %d.",
                self.min_train_size + self.test_size + self.gap_size,
                n_samples,
            )
            return

        for fold_idx in range(min(self.n_folds, len(fold_boundaries) - 2)):
            train_start, train_end, test_start, test_end = (
                self._fold_indices(fold_boundaries, fold_idx)
            )

            if test_start >= n_samples or test_end <= test_start:
                logger.info(
                    "Fold %d: test block exceeds data bounds; stopping.",
                    fold_idx,
                )
                break

            train_indices = np.arange(train_start, train_end, dtype=np.int64)
            test_indices = np.arange(test_start, test_end, dtype=np.int64)

            # Log fold info
            if timestamps is not None:
                ts_arr = np.asarray(timestamps)
                try:
                    train_t0 = ts_arr[train_start] if train_start < len(ts_arr) else "?"
                    train_t1 = ts_arr[train_end - 1] if train_end <= len(ts_arr) else "?"
                    test_t0 = ts_arr[test_start] if test_start < len(ts_arr) else "?"
                    test_t1 = ts_arr[test_end - 1] if test_end <= len(ts_arr) else "?"
                except (IndexError, TypeError):
                    train_t0, train_t1, test_t0, test_t1 = "?", "?", "?", "?"
                logger.info(
                    "Fold %d: train [%s → %s] (%d bars), test [%s → %s] (%d bars)",
                    fold_idx, train_t0, train_t1, len(train_indices),
                    test_t0, test_t1, len(test_indices),
                )
            else:
                logger.info(
                    "Fold %d: train [%d:%d] (%d bars), test [%d:%d] (%d bars)",
                    fold_idx, train_start, train_end, len(train_indices),
                    test_start, test_end, len(test_indices),
                )

            yield train_indices, test_indices

    # ------------------------------------------------------------------
    # Boundary calculation
    # ------------------------------------------------------------------

    def _compute_boundaries(self, n_samples: int) -> List[int]:
        """Compute test-block boundary positions.

        Returns a list of boundary positions [b0, b1, b2, ...] where each
        adjacent pair defines a test block, and the purge-gap-adjusted
        training region is everything before ``b_i - gap_size``.

        For expanding-window mode:
            Train on 0 .. (boundary_i − gap_size)
            Test on  boundary_i .. boundary_{i+1}

        For rolling-window mode:
            Train on max(0, boundary_i − lookback − gap_size) .. (boundary_i − gap_size)
            Test  on boundary_i .. boundary_{i+1}
        """
        boundaries = [self.min_train_size]
        for i in range(self.n_folds):
            next_boundary = boundaries[-1] + self.test_size
            if next_boundary > n_samples:
                break
            boundaries.append(next_boundary)
        return boundaries

    def _fold_indices(
        self,
        boundaries: List[int],
        fold_idx: int,
    ) -> Tuple[int, int, int, int]:
        """Compute (train_start, train_end, test_start, test_end) for a fold.

        Args:
            boundaries: List of test-block boundary positions.
            fold_idx: Zero-based fold index.

        Returns:
            Tuple of four integer positions.
        """
        test_start = boundaries[fold_idx]
        test_end = min(boundaries[fold_idx + 1], boundaries[-1])

        if self.rolling:
            # Rolling window: look back from just before the test block
            train_end = max(0, test_start - self.gap_size)
            train_start = max(0, train_end - self.lookback)
        else:
            # Expanding window: everything from the start up to gap before test
            train_start = 0
            train_end = max(0, test_start - self.gap_size)

        return train_start, train_end, test_start, test_end

    # ------------------------------------------------------------------
    # Convenience: get all splits at once
    # ------------------------------------------------------------------

    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, Sequence]] = None,
    ) -> int:
        """Return the number of splits this instance will produce for a
        dataset of length ``len(X)``.  If ``X`` is None, returns the
        configured ``n_folds`` as an upper bound.
        """
        if X is None:
            return self.n_folds
        n_samples = len(X)
        boundaries = self._compute_boundaries(n_samples)
        return max(0, len(boundaries) - 2)  # need at least one test block

    # ------------------------------------------------------------------
    # Train / test time ranges (for logging / display)
    # ------------------------------------------------------------------

    def get_train_test_times(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Return a list of ``(train_start, train_end, test_start, test_end)``
        timestamps for each fold.

        Args:
            timestamps: ``pd.DatetimeIndex`` aligned to the data.

        Returns:
            List of timestamp tuples.
        """
        n = len(timestamps)
        boundaries = self._compute_boundaries(n)
        result = []
        for fold_idx in range(min(self.n_folds, len(boundaries) - 2)):
            train_start, train_end, test_start, test_end = self._fold_indices(
                boundaries, fold_idx,
            )
            if test_start >= n or test_end <= test_start:
                break
            ts_train_start = timestamps[train_start] if train_start < n else None
            ts_train_end = timestamps[train_end - 1] if train_end <= n and train_end > 0 else None
            ts_test_start = timestamps[test_start] if test_start < n else None
            ts_test_end = timestamps[test_end - 1] if test_end <= n and test_end > 0 else None
            result.append((ts_train_start, ts_train_end, ts_test_start, ts_test_end))
        return result

    def __repr__(self) -> str:
        return (
            f"PurgedWalkForwardSplit(n_folds={self.n_folds}, "
            f"min_train_size={self.min_train_size}, test_size={self.test_size}, "
            f"gap_size={self.gap_size}, rolling={self.rolling}"
            + (f", lookback={self.lookback}" if self.rolling else "")
            + ")"
        )
