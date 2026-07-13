"""
data.py — Data normalisation, windowing, and session feature encoding for v2.

Provides normalisation for OHLCV inputs (log-ratio pricing + volume scaling),
causal sliding-window construction, and cyclical time-feature encoding.

All operations are strictly causal — no future data leaks into the context
window at time t.

Public API
----------
  normalize_ohlcv          — Convert OHLCV DataFrame to (n_bars, 5) float32 array
  create_sliding_windows   — Build overlapping (seq_len, n_channels) windows with labels
  SessionFeatureEncoder    — Cyclical sin/cos encoding of hour and weekday
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit


# ============================================================================
# Numba-compiled kernels
# ============================================================================


@njit(cache=True)
def _build_windows_nb(
    data: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """
    Build overlapping sliding windows from a 2-D feature array.

    Parameters
    ----------
    data : float32 or float64 (n_bars, n_channels)
        Normalised feature matrix.
    seq_len : int
        Sequence length (number of bars per window).

    Returns
    -------
    windows : float64 (n_windows, seq_len, n_channels)
        windows[i] = data[i : i + seq_len]
    """
    n_bars = data.shape[0]
    n_channels = data.shape[1]
    n_windows = n_bars - seq_len + 1

    if n_windows <= 0:
        raise ValueError(
            f"Cannot create windows: data has {n_bars} bars but "
            f"seq_len={seq_len} requires at least {seq_len} bars."
        )

    windows = np.empty((n_windows, seq_len, n_channels), dtype=np.float64)
    for w in range(n_windows):
        windows[w] = data[w : w + seq_len]

    return windows


@njit(cache=True)
def _extract_labels_nb(
    label_array: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """
    Extract the label for each window (the last bar in the window).

    Parameters
    ----------
    label_array : float64 (n_bars, n_dims)
        Label tensor aligned 1:1 with the data rows.
    seq_len : int
        Sequence length.

    Returns
    -------
    labels : float64 (n_windows, n_dims)
        labels[w] = label_array[w + seq_len - 1]
    """
    n_bars = label_array.shape[0]
    n_dims = label_array.shape[1]
    n_windows = n_bars - seq_len + 1

    labels = np.empty((n_windows, n_dims), dtype=np.float64)
    for w in range(n_windows):
        labels[w] = label_array[w + seq_len - 1]

    return labels


# ============================================================================
# Public functions
# ============================================================================


def normalize_ohlcv(df: pd.DataFrame) -> np.ndarray:
    """
    Normalise OHLCV data into a (n_bars, 5) float32 array suitable for
    model input.

    Transformation
    --------------
    * **Open / High / Low / Close**: log-ratio to the *previous* bar's close.
      This makes the representation scale-free and strictly causal — the
      normalisation at time ``t`` uses only ``Close[t-1]``.
    * **Volume**: divided by the rolling median volume over the last 252
      bars (or fewer at the start of the series), producing a dimensionless
      relative-volume feature.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

    Returns
    -------
    np.ndarray of shape ``(n_bars, 5)``, dtype float32.
        Columns: [Open, High, Low, Close, Volume] — all normalised.

    Notes
    -----
    - The first row will have NaN Open/High/Low/Close because there is no
      previous close; these are filled with 0.0 (neutral log-return).
    - Volume normalisation uses an expanding window of at least 1 bar, so
      the very first bar gets its own volume as the median (ratio = 1.0).
    """
    if df.empty:
        raise ValueError("DataFrame is empty — cannot normalise.")

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    eps = 1e-9
    n_bars = len(df)
    normed = np.zeros((n_bars, 5), dtype=np.float32)

    # ---- OHLC: log-ratio to previous close ----
    prev_close = df["Close"].shift(1).values.astype(np.float64)

    for i, col in enumerate(["Open", "High", "Low", "Close"]):
        vals = df[col].values.astype(np.float64)
        ratio = vals / (prev_close + eps)
        normed[:, i] = np.log(np.maximum(ratio, eps)).astype(np.float32)

    # ---- Volume: divided by rolling median (252-bar, causal) ----
    vol = df["Volume"].values.astype(np.float64)
    vol_median = (
        pd.Series(vol)
        .rolling(window=252, min_periods=1)
        .median()
        .values
        .astype(np.float64)
    )
    normed[:, 4] = (vol / (vol_median + eps)).astype(np.float32)

    # Fill first-row NaN (no previous close) with 0.0
    if np.isnan(normed[0, :4]).any():
        normed[0, :4] = 0.0
    # Safety: fill any remaining NaNs/infs
    normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    return normed


def create_sliding_windows(
    data: np.ndarray,
    seq_len: int,
    label_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create overlapping sliding windows from normalised OHLCV data and
    corresponding label arrays.

    Each window covers bars ``[i, i+seq_len-1]`` and maps to the labels
    at bar ``i+seq_len-1`` (the most recent bar in the window). This is
    strictly causal: the model sees only bars up to the prediction point.

    Parameters
    ----------
    data : np.ndarray of shape ``(n_bars, n_channels)``
        Normalised feature matrix (e.g. output of :func:`normalize_ohlcv`).
    seq_len : int
        Number of bars in each sliding window.
    label_dict : dict of str → np.ndarray, optional
        Dictionary mapping label names to label arrays. Each array must
        have the same first-dimension length as ``data`` (``n_bars``).
        Labels can be 1-D (scalar per bar) or 2-D (vector per bar).

    Returns
    -------
    X : np.ndarray of shape ``(n_windows, seq_len, n_channels)``, float64
    y : dict of str → np.ndarray
        Same keys as ``label_dict``. Each value has shape
        ``(n_windows, ...)`` — the first dimension matches ``X``, and the
        remaining dimensions capture the label structure.

    Raises
    ------
    ValueError
        If ``data`` has fewer rows than ``seq_len``, or if any label array
        length does not match ``n_bars``.

    Notes
    -----
    - ``n_windows = n_bars - seq_len + 1``
    - The window at position ``w`` corresponds to bars ``[w, w+seq_len-1]``.
      Its labels are taken from bar ``w+seq_len-1``.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(f"data must be a 2-D numpy array, got shape {getattr(data, 'shape', None)}")

    n_bars, n_channels = data.shape

    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if n_bars < seq_len:
        raise ValueError(
            f"Not enough bars: data has {n_bars} rows, seq_len={seq_len} requires >= {seq_len}"
        )

    # Build feature windows
    X = _build_windows_nb(data.astype(np.float64), seq_len)

    # Extract labels
    y: Dict[str, np.ndarray] = {}
    if label_dict is not None:
        for name, label_arr in label_dict.items():
            if not isinstance(label_arr, np.ndarray):
                raise TypeError(
                    f"Label '{name}' must be a numpy array, got {type(label_arr).__name__}"
                )
            if len(label_arr) != n_bars:
                raise ValueError(
                    f"Label '{name}' has {len(label_arr)} rows but data has {n_bars}. "
                    "All labels must be aligned 1:1 with data."
                )

            label_2d: np.ndarray
            if label_arr.ndim == 1:
                label_2d = label_arr[:, np.newaxis]
            elif label_arr.ndim == 2:
                label_2d = label_arr
            else:
                raise ValueError(
                    f"Label '{name}' has {label_arr.ndim} dimensions; "
                    "only 1-D or 2-D label arrays are supported."
                )

            y_raw = _extract_labels_nb(label_2d.astype(np.float64), seq_len)

            # Squeeze back to 1-D if original was 1-D
            if label_arr.ndim == 1:
                y[name] = y_raw[:, 0]
            else:
                y[name] = y_raw

    return X, y


# ============================================================================
# SessionFeatureEncoder
# ============================================================================


class SessionFeatureEncoder:
    """
    Encode hour-of-day and day-of-week as cyclical sin/cos features.

    This yields 4 continuous features that preserve the circular nature of
    time — e.g. 23:00 is close to 00:00, and Sunday is close to Monday.

    Optionally includes a binary ``has_gap`` flag (5th channel) that marks
    bars following a temporal gap (e.g. weekend close → reopen).  This lets
    the model distinguish continuous trading from session-boundary bars.

    Parameters
    ----------
    None

    Attributes
    ----------
    _2pi : float
        Cached 2π constant.

    Example
    -------
    >>> encoder = SessionFeatureEncoder()
    >>> timestamps = pd.to_datetime(["2024-01-01 06:30", "2024-01-01 23:45"])
    >>> features = encoder.encode(timestamps)
    >>> features.shape
    (2, 4)

    >>> features_gap = encoder.encode(timestamps, include_gap=True)
    >>> features_gap.shape
    (2, 5)
    """

    def __init__(self) -> None:
        self._2pi = 2.0 * np.pi

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gaps(
        ts_idx: pd.DatetimeIndex,
        threshold_multiplier: float = 3.0,
    ) -> np.ndarray:
        """
        Return a float32 array where 1.0 marks bars that follow a temporal gap.

        A "gap" is defined as a time delta exceeding ``threshold_multiplier``
        × the median inter-bar interval (computed from the non-zero deltas
        present in the series).

        Parameters
        ----------
        ts_idx : pd.DatetimeIndex
            Sorted timestamps (ascending).
        threshold_multiplier : float
            Multiple of the median bar interval above which a delta is
            considered a gap.  Default 3.0 works well for M1 (flags gaps
            > ~3 min) and M5 (flags gaps > ~15 min) while ignoring small
            timing jitter.

        Returns
        -------
        np.ndarray of shape ``(n_bars,)``, dtype float32.
            gap[0] is always 0.0 (no previous bar).
            gap[i] = 1.0 if ``ts_idx[i] - ts_idx[i-1] > threshold``.
        """
        n = len(ts_idx)
        if n < 2:
            return np.zeros(n, dtype=np.float32)

        # Nanosecond deltas
        deltas_ns = (ts_idx[1:].view("int64") - ts_idx[:-1].view("int64")).astype(np.float64)

        # Median of non-zero deltas (ignore NaT-derived zeros)
        positive = deltas_ns[deltas_ns > 0]
        if len(positive) == 0:
            return np.zeros(n, dtype=np.float32)

        median_delta = np.median(positive)
        threshold = median_delta * threshold_multiplier

        gaps = np.zeros(n, dtype=np.float32)
        gaps[1:] = (deltas_ns > threshold).astype(np.float32)
        return gaps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        timestamps: Union[pd.Series, np.ndarray],
        include_gap: bool = False,
        gap_threshold_multiplier: float = 3.0,
    ) -> np.ndarray:
        """
        Encode a sequence of timestamps into cyclical features.

        Parameters
        ----------
        timestamps : pd.Series or np.ndarray
            Array of datetime-like values (strings, datetime64, or
            pandas Timestamps).
        include_gap : bool
            If True, append a 5th column ``has_gap`` (0.0 or 1.0) that
            flags bars following a temporal gap (e.g. weekend reopen).
        gap_threshold_multiplier : float
            Only used when ``include_gap=True``.  Multiplier applied to
            the median bar interval to define the gap threshold.

        Returns
        -------
        np.ndarray of shape ``(n_bars, 4)`` or ``(n_bars, 5)``, dtype float32.
            Columns (always the first 4):
                0 — ``hour_sin``  : sin(2π * hour / 24)
                1 — ``hour_cos``  : cos(2π * hour / 24)
                2 — ``dow_sin``   : sin(2π * dayofweek / 7)
                3 — ``dow_cos``   : cos(2π * dayofweek / 7)
            Optional 5th column (when ``include_gap=True``):
                4 — ``has_gap``   : 1.0 if this bar follows a temporal gap
            Where ``dayofweek`` is 0=Monday, 6=Sunday.

        Raises
        ------
        ValueError
            If ``timestamps`` is empty or contains only NaT values.
        """
        # Normalise to DatetimeIndex (handles Series, list, ndarray, Index)
        ts_idx = pd.DatetimeIndex(pd.to_datetime(timestamps))

        if len(ts_idx) == 0:
            raise ValueError("timestamps must not be empty.")

        if ts_idx.isna().all():
            raise ValueError("timestamps contains only NaT values.")

        # Fill NaT with a sentinel so that .hour / .dayofweek work;
        # we mask the corresponding output features back to NaN afterward.
        nat_mask = ts_idx.isna()
        ts_filled = ts_idx.fillna(pd.Timestamp("1970-01-05"))  # Monday epoch

        hours = ts_filled.hour.values.astype(np.float64)
        dows = ts_filled.dayofweek.values.astype(np.float64)  # Monday=0

        n_cols = 5 if include_gap else 4
        features = np.zeros((len(ts_idx), n_cols), dtype=np.float32)

        features[:, 0] = np.sin(self._2pi * hours / 24.0).astype(np.float32)
        features[:, 1] = np.cos(self._2pi * hours / 24.0).astype(np.float32)
        features[:, 2] = np.sin(self._2pi * dows / 7.0).astype(np.float32)
        features[:, 3] = np.cos(self._2pi * dows / 7.0).astype(np.float32)

        # Optional gap flag
        if include_gap:
            features[:, 4] = self._detect_gaps(ts_idx, gap_threshold_multiplier)

        # NaN-out features for NaT timestamps
        features[nat_mask, :] = np.nan

        return features
