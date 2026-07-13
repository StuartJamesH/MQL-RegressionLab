"""
labels.py — Label engineering for v2 training pipeline.

Provides forward-excursion surfaces, directional return distributions,
optimal-exit labels (triple-barrier with continuous outcomes), volatility
regime classification, and an HDF5-backed label cache.

All label computations are strictly causal — no future information leaks
into the label at time t.

Public API
----------
  compute_forward_excursion_surface    — MFE/MAE in ATR units at multiple horizons
  compute_directional_return_distribution — forward log returns at each horizon
  compute_optimal_exit_labels          — triple-barrier TP/SL/timeout outcomes
  compute_volatility_regime_labels     — 4-regime classification from rolling stats
  LabelStore                           — HDF5-backed cache for precomputed tensors
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from numba import njit
from talib import ATR


# ============================================================================
# Numba-compiled kernels (module-level for cache=True)
# ============================================================================


@njit(cache=True)
def _compute_excursion_surface_nb(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """
    Numba-compiled inner loop for compute_forward_excursion_surface.

    For each bar t (up to n - max_horizon - 1), computes MFE and MAE at
    every horizon for both buy and sell perspectives, normalised by ATR.

    Parameters
    ----------
    highs, lows, closes : float64 (n_bars,)
    atrs : float64 (n_bars,)
    horizons : int64 (n_horizons,)

    Returns
    -------
    surface : float64 (n_bars, n_horizons, 2, 2)
        surface[t, h, 0, 0] = buy  MFE in ATR units
        surface[t, h, 0, 1] = buy  MAE in ATR units
        surface[t, h, 1, 0] = sell MFE in ATR units
        surface[t, h, 1, 1] = sell MAE in ATR units
        Last max(horizons) rows are NaN (no forward data).
    """
    n_bars = len(highs)
    n_horizons = len(horizons)
    max_horizon = int(horizons[-1])
    surface = np.full((n_bars, n_horizons, 2, 2), np.nan, dtype=np.float64)

    for t in range(n_bars - max_horizon):
        close_t = closes[t]
        atr_t = atrs[t]

        if np.isnan(close_t) or close_t <= 0.0:
            continue
        if np.isnan(atr_t) or atr_t <= 0.0:
            continue

        atr_pct = atr_t / close_t
        if atr_pct <= 0.0:
            continue

        for h_idx in range(n_horizons):
            h_val = int(horizons[h_idx])
            end = t + h_val + 1  # t+1 .. t+h inclusive

            # rolling max/min over forward window (manual — numba-safe)
            fwd_high = highs[t + 1]
            fwd_low = lows[t + 1]
            for j in range(t + 2, end):
                if highs[j] > fwd_high:
                    fwd_high = highs[j]
                if lows[j] < fwd_low:
                    fwd_low = lows[j]

            # ---- Buy side ----
            buy_mfe_raw = (fwd_high / close_t) - 1.0
            buy_mae_raw = 1.0 - (fwd_low / close_t)
            surface[t, h_idx, 0, 0] = buy_mfe_raw / atr_pct  # buy MFE
            surface[t, h_idx, 0, 1] = buy_mae_raw / atr_pct  # buy MAE

            # ---- Sell side ----
            sell_mfe_raw = 1.0 - (fwd_low / close_t)
            sell_mae_raw = (fwd_high / close_t) - 1.0
            surface[t, h_idx, 1, 0] = sell_mfe_raw / atr_pct  # sell MFE
            surface[t, h_idx, 1, 1] = sell_mae_raw / atr_pct  # sell MAE

    return surface


@njit(cache=True)
def _compute_directional_returns_nb(
    closes: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """
    Numba-compiled forward log-return computation.

    Parameters
    ----------
    closes : float64 (n_bars,)
    horizons : int64 (n_horizons,)

    Returns
    -------
    rets : float64 (n_bars, n_horizons)
        rets[t, h_idx] = ln(close[t + h] / close[t])
        NaN where insufficient forward data.
    """
    n_bars = len(closes)
    n_horizons = len(horizons)
    max_horizon = int(horizons[-1])
    rets = np.full((n_bars, n_horizons), np.nan, dtype=np.float64)

    for t in range(n_bars - max_horizon):
        ct = closes[t]
        if np.isnan(ct) or ct <= 0.0:
            continue
        for h_idx in range(n_horizons):
            h_val = int(horizons[h_idx])
            cth = closes[t + h_val]
            if not np.isnan(cth) and cth > 0.0:
                rets[t, h_idx] = np.log(cth / ct)

    return rets


@njit(cache=True)
def _compute_optimal_exit_nb(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    tp_atr_mult: float,
    sl_atr_mult: float,
    max_horizon: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Numba-compiled triple-barrier simulator for both buy and sell at every bar.

    Parameters
    ----------
    highs, lows, closes, atrs : float64 (n_bars,)
    tp_atr_mult, sl_atr_mult : float
        Take-profit / stop-loss distances in ATR multiples of the entry price.
    max_horizon : int
        Maximum holding period in bars.

    Returns
    -------
    Tuple of 8 float64 arrays, each (n_bars,):
        buy_outcome, sell_outcome  —  1=TP, -1=SL, 0=timeout, NaN if no data
        buy_duration, sell_duration — bars to exit (or max_horizon for timeout)
        buy_mfe, buy_mae             — MFE/MAE in ATR units (non-negative)
        sell_mfe, sell_mae
    """
    n = len(highs)
    buy_outcome = np.full(n, np.nan, dtype=np.float64)
    sell_outcome = np.full(n, np.nan, dtype=np.float64)
    buy_duration = np.full(n, np.nan, dtype=np.float64)
    sell_duration = np.full(n, np.nan, dtype=np.float64)
    buy_mfe = np.full(n, np.nan, dtype=np.float64)
    buy_mae = np.full(n, np.nan, dtype=np.float64)
    sell_mfe = np.full(n, np.nan, dtype=np.float64)
    sell_mae = np.full(n, np.nan, dtype=np.float64)

    for t in range(n - 1):
        close_t = closes[t]
        atr_t = atrs[t]

        if np.isnan(close_t) or close_t <= 0.0:
            continue
        if np.isnan(atr_t) or atr_t <= 0.0:
            continue

        atr_pct = atr_t / close_t
        if atr_pct <= 0.0:
            continue

        # Barrier levels as multipliers of entry price
        tp_buy_mult = 1.0 + tp_atr_mult * atr_pct
        sl_buy_mult = 1.0 - sl_atr_mult * atr_pct
        tp_sell_mult = 1.0 - tp_atr_mult * atr_pct
        sl_sell_mult = 1.0 + sl_atr_mult * atr_pct

        buy_resolved = False
        sell_resolved = False

        buy_max_exc = 0.0   # MFE in ATR units
        buy_min_exc = 0.0   # MAE in ATR units
        sell_max_exc = 0.0
        sell_min_exc = 0.0

        horizon_end = min(t + max_horizon + 1, n)

        for t1 in range(t + 1, horizon_end):
            h_val = highs[t1]
            l_val = lows[t1]

            h_ratio = h_val / close_t
            l_ratio = l_val / close_t

            # -- Buy excursions (ATR units) --
            buy_curr_mfe = max(0.0, (h_ratio - 1.0) / atr_pct)
            buy_curr_mae = max(0.0, (1.0 - l_ratio) / atr_pct)
            if buy_curr_mfe > buy_max_exc:
                buy_max_exc = buy_curr_mfe
            if buy_curr_mae > buy_min_exc:
                buy_min_exc = buy_curr_mae

            # -- Sell excursions (ATR units) --
            sell_curr_mfe = max(0.0, (1.0 - l_ratio) / atr_pct)
            sell_curr_mae = max(0.0, (h_ratio - 1.0) / atr_pct)
            if sell_curr_mfe > sell_max_exc:
                sell_max_exc = sell_curr_mfe
            if sell_curr_mae > sell_min_exc:
                sell_min_exc = sell_curr_mae

            # -- Barrier checks --
            if not buy_resolved:
                if h_ratio >= tp_buy_mult:
                    buy_outcome[t] = 1.0
                    buy_duration[t] = float(t1 - t)
                    buy_resolved = True
                elif l_ratio <= sl_buy_mult:
                    buy_outcome[t] = -1.0
                    buy_duration[t] = float(t1 - t)
                    buy_resolved = True

            if not sell_resolved:
                if l_ratio <= tp_sell_mult:
                    sell_outcome[t] = 1.0
                    sell_duration[t] = float(t1 - t)
                    sell_resolved = True
                elif h_ratio >= sl_sell_mult:
                    sell_outcome[t] = -1.0
                    sell_duration[t] = float(t1 - t)
                    sell_resolved = True

            if buy_resolved and sell_resolved:
                break

        # Timeout
        if not buy_resolved:
            buy_outcome[t] = 0.0
            buy_duration[t] = float(max_horizon)
        if not sell_resolved:
            sell_outcome[t] = 0.0
            sell_duration[t] = float(max_horizon)

        buy_mfe[t] = buy_max_exc
        buy_mae[t] = buy_min_exc
        sell_mfe[t] = sell_max_exc
        sell_mae[t] = sell_min_exc

    return (
        buy_outcome, sell_outcome,
        buy_duration, sell_duration,
        buy_mfe, buy_mae,
        sell_mfe, sell_mae,
    )


# ============================================================================
# Public label functions
# ============================================================================


def compute_forward_excursion_surface(
    df: pd.DataFrame,
    horizons: List[int],
    atr_window: int = 14,
) -> np.ndarray:
    """
    Compute MFE/MAE excursion surface at multiple forward horizons.

    For each bar ``t`` and each horizon ``h``, computes the maximum favourable
    and adverse excursion over bars [t+1, t+h] for both a hypothetical long
    (buy) and short (sell) position entered at ``Close[t]``. All excursions
    are normalised to ATR units so that values are comparable across time and
    volatility regimes.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``Time``, ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume``.
    horizons : list of int
        Forward horizon lengths in bars, e.g. ``[5, 10, 20, 40, 60, 120]``.
        Must be sorted ascending.
    atr_window : int, default=14
        ATR lookback period.

    Returns
    -------
    np.ndarray of shape ``(n_bars, len(horizons), 2, 2)``, dtype float64.
        * ``[..., 0, 0]`` — buy  MFE (ATR units)
        * ``[..., 0, 1]`` — buy  MAE (ATR units)
        * ``[..., 1, 0]`` — sell MFE (ATR units)
        * ``[..., 1, 1]`` — sell MAE (ATR units)
        The last ``max(horizons)`` rows are NaN because there is not enough
        forward data to compute excursions at the largest horizon.

    Notes
    -----
    - Strictly causal: excursion at bar ``t`` uses only prices from
      ``t+1`` to ``t+h``.
    - Buy MFE = (max_high / close) - 1, normalised by ATR/close.
    - Buy MAE = 1 - (min_low / close), normalised by ATR/close.
    - Sell side is the mirror.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — cannot compute excursion surface.")

    required = {"High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError(
            "horizons must be a non-empty list of positive integers."
        )

    sorted_horizons = sorted(horizons)
    max_h = sorted_horizons[-1]

    if len(df) <= max_h:
        raise ValueError(
            f"DataFrame has {len(df)} rows — too short for max horizon {max_h}."
        )

    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)
    closes = df["Close"].values.astype(np.float64)

    atrs = ATR(df["High"], df["Low"], df["Close"], timeperiod=atr_window)
    atrs_arr = atrs.values.astype(np.float64)

    horizons_arr = np.array(sorted_horizons, dtype=np.int64)

    return _compute_excursion_surface_nb(highs, lows, closes, atrs_arr, horizons_arr)


def compute_directional_return_distribution(
    df: pd.DataFrame,
    horizons: List[int],
) -> np.ndarray:
    """
    Compute forward log-returns at multiple horizons for every bar.

    For each bar ``t`` and horizon ``h``, returns ``ln(Close[t+h] / Close[t])``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Close`` column.
    horizons : list of int
        Forward horizon lengths in bars. Sorted ascending internally.

    Returns
    -------
    np.ndarray of shape ``(n_bars, len(horizons))``, dtype float64.
        * ``rets[t, h_idx]`` = forward log return at horizon ``horizons[h_idx]``
          from bar ``t``.
        * The last ``max(horizons)`` rows are NaN.

    Notes
    -----
    - Strictly causal: the return at bar ``t`` uses ``Close[t+h]`` which is
      in the future relative to ``t``, but this is the label — the feature
      window at ``t`` does not see ``t+h``.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    if "Close" not in df.columns:
        raise KeyError("DataFrame missing required column: Close")

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError(
            "horizons must be a non-empty list of positive integers."
        )

    sorted_horizons = sorted(horizons)
    max_h = sorted_horizons[-1]

    if len(df) <= max_h:
        raise ValueError(
            f"DataFrame has {len(df)} rows — too short for max horizon {max_h}."
        )

    closes = df["Close"].values.astype(np.float64)
    horizons_arr = np.array(sorted_horizons, dtype=np.int64)

    return _compute_directional_returns_nb(closes, horizons_arr)


def compute_optimal_exit_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.5,
    sl_atr_mult: float = 2.5,
    max_horizon: int = 60,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Triple-barrier exit labels with continuous MFE/MAE outcomes.

    For every bar, simulates a hypothetical long (buy) and short (sell) trade
    entered at ``Close[t]``. The trade exits when either the take-profit or
    stop-loss barrier is hit, or after ``max_horizon`` bars (timeout). The
    barriers are expressed as multiples of the current ATR.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``High``, ``Low``, ``Close``.
    tp_atr_mult : float, default=2.5
        Take-profit distance in ATR multiples.
    sl_atr_mult : float, default=2.5
        Stop-loss distance in ATR multiples.
    max_horizon : int, default=60
        Maximum holding period in bars.
    atr_window : int, default=14
        ATR lookback window.

    Returns
    -------
    pd.DataFrame indexed the same as ``df``, with columns:
        * ``buy_outcome``  — 1=TP, -1=SL, 0=timeout, NaN if no data
        * ``sell_outcome`` — 1=TP, -1=SL, 0=timeout, NaN if no data
        * ``buy_duration``  — bars held (or max_horizon for timeout)
        * ``sell_duration`` — bars held (or max_horizon for timeout)
        * ``buy_mfe`` — max favourable excursion in ATR units (>= 0)
        * ``buy_mae`` — max adverse excursion in ATR units (>= 0)
        * ``sell_mfe``
        * ``sell_mae``

    Notes
    -----
    MFE and MAE are always non-negative and measured in ATR units — a value of
    1.5 means the price moved 1.5×ATR in that direction at some point during the
    trade. Strictly causal: the outcome at bar ``t`` is determined solely by
    prices from ``t+1`` onward.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    required = {"High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive.")
    if tp_atr_mult <= 0 or sl_atr_mult <= 0:
        raise ValueError("tp_atr_mult and sl_atr_mult must be positive.")

    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)
    closes = df["Close"].values.astype(np.float64)
    atrs = ATR(df["High"], df["Low"], df["Close"], timeperiod=atr_window)
    atrs_arr = atrs.values.astype(np.float64)

    (
        buy_out, sell_out,
        buy_dur, sell_dur,
        buy_mfe, buy_mae,
        sell_mfe, sell_mae,
    ) = _compute_optimal_exit_nb(
        highs, lows, closes, atrs_arr,
        float(tp_atr_mult), float(sl_atr_mult), int(max_horizon),
    )

    return pd.DataFrame({
        "buy_outcome": buy_out,
        "sell_outcome": sell_out,
        "buy_duration": buy_dur,
        "sell_duration": sell_dur,
        "buy_mfe": buy_mfe,
        "buy_mae": buy_mae,
        "sell_mfe": sell_mfe,
        "sell_mae": sell_mae,
    }, index=df.index)


def compute_volatility_regime_labels(
    df: pd.DataFrame,
    lookback: int = 20,
    n_regimes: int = 4,
) -> np.ndarray:
    """
    Assign each bar to a volatility regime (0–3) using rolling statistics.

    Computes two causal volatility proxies:
    1. Rolling realised volatility — std of log returns over ``lookback``.
    2. Rolling spread proxy — mean of (High-Low)/Close over ``lookback``.

    Each proxy is standardised into a z-score using a longer rolling window
    (100 bars), then summed to form a composite score. The composite score
    is binned into ``n_regimes`` equal-frequency buckets using global
    quantiles derived from the entire available dataset.

    Regime 0 = lowest volatility, regime 3 = highest volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``High``, ``Low``, ``Close`` columns.
    lookback : int, default=20
        Rolling window for the base volatility statistics.
    n_regimes : int, default=4
        Number of volatility regimes to produce.

    Returns
    -------
    np.ndarray of shape ``(n_bars,)``, dtype int32.
        Integer labels in [0, n_regimes-1]. Early bars where the rolling
        statistics are not yet available will have value -1 (invalid).

    Notes
    -----
    - Causal: regime at time ``t`` uses only data available at or before ``t``.
      The rolling windows (std, mean) are right-aligned.
    - Global quantile thresholds are computed from the full dataset. For strict
      walk-forward deployment, pre-compute thresholds on training data and
      pass them as fixed bin edges.
    - Returns -1 for bars where the composite score is NaN (insufficient history
      for the rolling z-score normalisation).
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    required = {"High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
    if lookback < 2:
        raise ValueError("lookback must be at least 2.")

    close = df["Close"]
    eps = 1e-9

    # ---- Proxy 1: rolling realised volatility (std of log returns) ----
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(window=lookback, min_periods=lookback).std()

    # ---- Proxy 2: rolling spread (High-Low)/Close ----
    spread = (df["High"] - df["Low"]) / (close + eps)
    spread_ma = spread.rolling(window=lookback, min_periods=lookback).mean()

    # ---- Standardise each proxy into z-scores (causal: 100-bar rolling) ----
    rv_z = (rv - rv.rolling(100, min_periods=100).mean()) / (
        rv.rolling(100, min_periods=100).std() + eps
    )
    spread_z = (spread_ma - spread_ma.rolling(100, min_periods=100).mean()) / (
        spread_ma.rolling(100, min_periods=100).std() + eps
    )

    # Composite score — equal weighting
    composite = (rv_z + spread_z).values

    # ---- Quantile-based binning ----
    valid_mask = np.isfinite(composite)
    if valid_mask.sum() < n_regimes:
        labels = np.full(len(df), -1, dtype=np.int32)
        labels[valid_mask] = 0
        return labels

    valid_composite = composite[valid_mask]
    quantile_edges = np.percentile(
        valid_composite,
        np.linspace(0, 100, n_regimes + 1),
    )
    # np.digitize produces bins 1..n_regimes; shift to 0..n_regimes-1
    labels = np.full(len(df), -1, dtype=np.int32)
    labels[valid_mask] = np.digitize(
        composite[valid_mask], quantile_edges[1:-1], right=True
    )

    return labels


# ============================================================================
# LabelStore — HDF5-backed tensor cache
# ============================================================================


class LabelStore:
    """
    HDF5-backed cache for precomputed label tensors.

    Avoids recomputing expensive label arrays (e.g. excursion surfaces,
    optimal-exit outcomes) every time a training script runs.  Stores
    arrays in a single HDF5 file keyed by a SHA-256 hash derived from the
    dataset identity and parameter configuration.

    Parameters
    ----------
    base_dir : str or pathlib.Path, default="ModelWorkbench/data/labels"
        Directory where the HDF5 file is stored.

    Attributes
    ----------
    h5_path : pathlib.Path
        Full path to the ``labels_cache.h5`` file.
    """

    def __init__(
        self,
        base_dir: str = "ModelWorkbench/data/labels",
    ) -> None:
        self._base_dir = pathlib.Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self._base_dir / "labels_cache.h5"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _df_fingerprint(df: pd.DataFrame) -> str:
        """
        Produce a short, stable hash of a DataFrame for cache-key purposes.

        Hashes the head (first 5 rows), tail (last 5 rows), shape, and
        column names so that different datasets produce different keys.
        """
        payload = {
            "n_rows": len(df),
            "columns": sorted(df.columns.tolist()),
            "head": (
                df.head(5)
                .select_dtypes(include=[np.number])
                .fillna(0.0)
                .round(6)
                .values.tolist()
            ),
            "tail": (
                df.tail(5)
                .select_dtypes(include=[np.number])
                .fillna(0.0)
                .round(6)
                .values.tolist()
            ),
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _make_key(dataset_name: str, params: Dict[str, Any]) -> str:
        """
        Derive an HDF5 group key from dataset name and parameters.

        Parameters
        ----------
        dataset_name : str
            Human-readable dataset identifier (e.g. ``"BTCUSD_M5_260weeks"``).
        params : dict
            Dictionary of parameter name → value pairs that affect the
            computation.  Only scalar/int/float/str values are supported.

        Returns
        -------
        str
            SHA-256 hex digest, truncated to 32 characters for readability
            while retaining practically zero collision probability.
        """
        payload = {
            "dataset": dataset_name,
            "params": params,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[..., np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Retrieve a cached tensor or compute and store it.

        Parameters
        ----------
        key : str
            Unique key identifying this label computation (use
            :meth:`_make_key` to generate one from dataset name + params).
        compute_fn : callable
            A function ``f(*args, **kwargs) -> np.ndarray`` that computes
            the label array. Only called if the key is not already in
            the cache.
        *args, **kwargs
            Forwarded to ``compute_fn``.

        Returns
        -------
        np.ndarray
            The label tensor, either loaded from cache or freshly computed.
        """
        with h5py.File(str(self.h5_path), "a") as h5f:
            if key in h5f:
                return h5f[key][:]

            result = compute_fn(*args, **kwargs)
            if not isinstance(result, np.ndarray):
                raise TypeError(
                    f"compute_fn must return np.ndarray, "
                    f"got {type(result).__name__}"
                )

            h5f.create_dataset(
                key, data=result, compression="gzip", compression_opts=4
            )
            return result
