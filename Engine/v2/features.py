"""
Engine/v2/features.py — Live causal feature preprocessing wrappers.

Exports thin wrappers around ``Learn.v2.data`` so the live runtime uses
exactly the same normalization and session encoding as training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from Learn.v2.data import normalize_ohlcv, SessionFeatureEncoder


def normalize_live_ohlcv(df: pd.DataFrame) -> np.ndarray:
    """
    Causally normalize a live OHLCV DataFrame to a ``(n_bars, 5)`` float32 array.

    Wraps ``Learn.v2.data.normalize_ohlcv`` so the live path stays identical to
    the training path.  Columns must include ``Open``, ``High``, ``Low``,
    ``Close``, and ``Volume``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV bars in chronological order.

    Returns
    -------
    np.ndarray of shape ``(n_bars, 5)``, dtype float32.
    """
    return normalize_ohlcv(df)


def encode_live_session_features(
    timestamps: pd.DatetimeIndex,
    include_gap: bool = True,
) -> np.ndarray:
    """
    Encode bar timestamps into cyclical session features.

    Wraps ``Learn.v2.data.SessionFeatureEncoder.encode`` with the same default
    gap-awareness used during training.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Bar timestamps in chronological order.
    include_gap : bool, optional
        If ``True`` (default), append the ``has_gap`` flag as a 5th column.
        This must match ``config.session_channels`` for the loaded model.

    Returns
    -------
    np.ndarray of shape ``(n_bars, 4)`` or ``(n_bars, 5)``, dtype float32.
    """
    encoder = SessionFeatureEncoder()
    return encoder.encode(timestamps, include_gap=include_gap)
