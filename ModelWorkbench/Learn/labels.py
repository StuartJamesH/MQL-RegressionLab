from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit
from talib import ATR, EMA
from typing import Any, Dict, List, Tuple
from numpy.lib.stride_tricks import sliding_window_view
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def super_smoother(series, period):
    """
    Ehlers Super Smoother (2-pole IIR) - robust to NaNs.
    """
    import numpy as np

    if period <= 0:
        raise ValueError("period must be > 0")

    # Protect against NaNs by forward/back filling for calculation
    s = series.astype(float).ffill().bfill().to_numpy()

    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    filt = np.zeros_like(s, dtype=float)
    if len(s) == 0:
        return pd.Series(filt, index=series.index)

    filt[0] = s[0]
    if len(s) > 1:
        filt[1] = c1 * (s[1] + s[0]) / 2 + c2 * filt[0]

    for i in range(2, len(s)):
        filt[i] = (
            c1 * (s[i] + s[i - 1]) / 2
            + c2 * filt[i - 1]
            + c3 * filt[i - 2]
        )

    return pd.Series(filt, index=series.index)


def causal_market_regime(df, ma_period=21, slope_smoothness=1, regime_min_duration=1,
                         slope_threshold=0, atr_window=14, atr_lookback=100, atr_percentile=20,
                         slope_lookback=200, slope_percentile=30):
    """
    Calculates market regime using purely causal (trailing) indicators.
    Returns a Series with values: 1 (Uptrend), 0 (Range), -1 (Downtrend)

    Differences from get_market_regime:
      - Uses trailing SMA of Close (no centered shift)
      - Uses trailing smoothed slope via rolling mean
      - Uses rolling quantile of past slope magnitudes for thresholds
      - Enforces regime_min_duration causally (requires sustained evidence)
    """
    df = df.copy()

    # 1. Trailing Moving Average (causal)
    ma = EMA(df['Close'], timeperiod=ma_period)

    # 2. Causal slope with smoothing (trailing) — normalized by MA level for price-scale invariance
    slope = ma.diff() / ma
    slope_sm = super_smoother(slope, period=slope_smoothness)

    # 3. Directional regime by slope magnitude vs threshold
    regime = pd.Series(0, index=df.index, dtype=int)
    regime[slope_sm > 0] = 1
    regime[slope_sm < 0] = -1

    # 4. Forward-fill zero values with latest trend value (causal)
    vals = regime.values.copy()
    last_trend = 0
    for i in range(len(vals)):
        if vals[i] == 0:
            vals[i] = last_trend
        else:
            last_trend = vals[i]
    regime_ff = pd.Series(vals, index=regime.index, dtype=int)

    # 5. Causal enforcement of minimum duration: require sustained evidence
    vals2 = regime_ff.values
    runlen = np.zeros_like(vals2, dtype=int)
    for i in range(1, len(vals2)):
        if vals2[i] == vals2[i-1]:
            runlen[i] = runlen[i-1] + 1
        else:
            runlen[i] = 0

    regime_causal = regime_ff.copy()
    min_run = max(int(regime_min_duration) - 1, 0)
    short_mask = runlen < min_run

    # 7. Filter flat slope regimes to range (0)
    slope_adaptive_thresh = (
        slope_sm.abs()
        .rolling(window=slope_lookback, min_periods=slope_lookback)
        .quantile(slope_percentile / 100.0)
    )
    flat_slope = slope_sm.abs() < slope_adaptive_thresh.fillna(0)
    if slope_threshold > 0:
        flat_slope = flat_slope | (slope_sm.abs() < slope_threshold)
    regime_causal[flat_slope] = 0

    return regime_causal


def causal_triple_barrier_hilow_trend_labeler(
    df,
    z_window=14,
    z_thresh=1.0,
    z_limit=2.5,
    atr_window=14,
    tp_mult=4.0,
    sl_mult=2.0,
    max_horizon=60,
    trend_pullback_thresh=0.0,
    regime_params=None,
    skip_range=False
):
    """
    Enhanced labeler that uses Market Regime to filter signals.
    - Regime 0 (Range): Uses standard Mean Reversion (Buy Oversold, Sell Overbought)
    - Regime 1 (Uptrend): Only Buys, preferably on dips (Z < trend_pullback_thresh)
    - Regime -1 (Downtrend): Only Sells, preferably on rallies (Z > -trend_pullback_thresh)
    """
    df = df.copy()

    # 1. Get Regime
    if regime_params is None:
        df['Regime'] = causal_market_regime(df)
    else:
        df['Regime'] = causal_market_regime(df, **regime_params)

    # 2. Compute Z-Score
    df["mean"] = df["Close"].rolling(z_window).mean()
    df["std"] = df["Close"].rolling(z_window).std()
    df["z"] = (df["Close"] - df["mean"]) / df["std"]

    # 3. Generate Signals based on Regime
    signals = pd.Series(index=df.index, dtype=float)

    # --- Range Logic (Regime 0) ---
    if skip_range:
        pass
    else:
        mask_range = df['Regime'] == 0
        signals[mask_range & (df["z"] < -z_thresh) & (df["z"] > -z_limit)] = 1  # Long
        signals[mask_range & (df["z"] > +z_thresh) & (df["z"] < +z_limit)] = -1  # Short

    # --- Uptrend Logic (Regime 1) ---
    mask_uptrend = df['Regime'] == 1
    signals[mask_uptrend & (df["z"] < -trend_pullback_thresh) & (df["z"] > -z_limit)] = 1  # Long on dip

    # --- Downtrend Logic (Regime -1) ---
    mask_downtrend = df['Regime'] == -1
    signals[mask_downtrend & (df["z"] > trend_pullback_thresh) & (df["z"] < z_limit)] = -1  # Short on rally

    signals = signals.dropna()

    # 4. Triple Barrier Loop
    df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
    events = []

    for t0, side in signals.items():
        if t0 >= len(df) - 1:
            continue

        # Entry Price Logic
        if side == +1:
            entry_price = df.loc[t0, "High"]
        else:
            entry_price = df.loc[t0, "Low"]

        atr = df.loc[t0, "atr"]
        if np.isnan(atr) or atr == 0:
            continue

        # Barriers
        if side == +1:  # Long
            tp = entry_price + tp_mult * atr
            sl = entry_price - sl_mult * atr
        else:  # Short
            tp = entry_price - tp_mult * atr
            sl = entry_price + sl_mult * atr

        t_end = min(t0 + max_horizon, df.index[-1])
        label = 0
        end_time = t_end

        for t in range(t0 + 1, t_end + 1):
            high = df.loc[t, "High"]
            low = df.loc[t, "Low"]

            if side == +1:  # Long
                if high >= tp:
                    label = 1
                    end_time = t
                    break
                if low <= sl:
                    label = -1
                    end_time = t
                    break
            else:  # Short
                if low <= tp:
                    label = 1
                    end_time = t
                    break
                if high >= sl:
                    label = -1
                    end_time = t
                    break

        events.append({
            "t0": t0,
            "side": side,
            "z": df.loc[t0, "z"],
            "regime": df.loc[t0, "Regime"],
            "tp": tp,
            "sl": sl,
            "t_end": end_time,
            "label": label
        })

    return pd.DataFrame(
        events,
        columns=["t0", "side", "z", "regime", "tp", "sl", "t_end", "label"]
    ).set_index("t0")


@njit(cache=True)
def _compute_outcomes_nb(highs, lows, atrs, tp_mult, sl_mult):
    """
    Numba-compiled inner kernel for calculate_trade_outcomes_all_candles.
    Scans forward to end-of-data for every bar with no horizon cap, giving
    outcomes that match production behaviour (trades run until TP or SL).
    Returns eight float64 arrays: buy_outcomes, sell_outcomes,
    buy_exit_prices, sell_exit_prices, buy_mfe, buy_mae, sell_mfe, sell_mae.
    Unresolved bars stay NaN for outcomes and exit prices; excursions are
    measured only up to the first exit for that direction so that only one
    side can reach 1.0 on a resolved trade.
    """
    n = len(highs)
    buy_outcomes = np.full(n, np.nan)
    sell_outcomes = np.full(n, np.nan)
    buy_exit_prices = np.full(n, np.nan)
    sell_exit_prices = np.full(n, np.nan)
    buy_mfe = np.full(n, np.nan)
    buy_mae = np.full(n, np.nan)
    sell_mfe = np.full(n, np.nan)
    sell_mae = np.full(n, np.nan)

    for t0 in range(n - 1):
        atr = atrs[t0]
        if np.isnan(atr) or atr == 0.0:
            continue

        entry_buy = highs[t0]
        tp_buy = entry_buy + tp_mult * atr
        sl_buy = entry_buy - sl_mult * atr
        tp_buy_dist = tp_buy - entry_buy
        sl_buy_dist = entry_buy - sl_buy

        entry_sell = lows[t0]
        tp_sell = entry_sell - tp_mult * atr
        sl_sell = entry_sell + sl_mult * atr
        tp_sell_dist = entry_sell - tp_sell
        sl_sell_dist = sl_sell - entry_sell

        buy_resolved = False
        sell_resolved = False
        buy_exit_type = 0
        sell_exit_type = 0
        buy_max_high = entry_buy
        buy_min_low = entry_buy
        sell_max_high = entry_sell
        sell_min_low = entry_sell

        for t1 in range(t0 + 1, n):
            h = highs[t1]
            l = lows[t1]

            if not buy_resolved:
                if h >= tp_buy:
                    buy_outcomes[t0] = 1.0
                    buy_exit_prices[t0] = tp_buy
                    buy_resolved = True
                    buy_exit_type = 1
                elif l <= sl_buy:
                    buy_outcomes[t0] = -1.0
                    buy_exit_prices[t0] = sl_buy
                    buy_resolved = True
                    buy_exit_type = -1

            if not sell_resolved:
                if l <= tp_sell:
                    sell_outcomes[t0] = 1.0
                    sell_exit_prices[t0] = tp_sell
                    sell_resolved = True
                    sell_exit_type = 1
                elif h >= sl_sell:
                    sell_outcomes[t0] = -1.0
                    sell_exit_prices[t0] = sl_sell
                    sell_resolved = True
                    sell_exit_type = -1

            if not np.isnan(h):
                if not buy_resolved or buy_exit_type == 1:
                    if h > buy_max_high:
                        buy_max_high = h
                if not sell_resolved or sell_exit_type == -1:
                    if h > sell_max_high:
                        sell_max_high = h
            if not np.isnan(l):
                if not buy_resolved or buy_exit_type == -1:
                    if l < buy_min_low:
                        buy_min_low = l
                if not sell_resolved or sell_exit_type == 1:
                    if l < sell_min_low:
                        sell_min_low = l

            if buy_resolved and sell_resolved:
                break

        if tp_buy_dist > 0.0:
            buy_mfe[t0] = min(max((buy_max_high - entry_buy) / tp_buy_dist, 0.0), 1.0)
        if sl_buy_dist > 0.0:
            buy_mae[t0] = min(max((entry_buy - buy_min_low) / sl_buy_dist, 0.0), 1.0)
        if tp_sell_dist > 0.0:
            sell_mfe[t0] = min(max((entry_sell - sell_min_low) / tp_sell_dist, 0.0), 1.0)
        if sl_sell_dist > 0.0:
            sell_mae[t0] = min(max((sell_max_high - entry_sell) / sl_sell_dist, 0.0), 1.0)

    return (
        buy_outcomes,
        sell_outcomes,
        buy_exit_prices,
        sell_exit_prices,
        buy_mfe,
        buy_mae,
        sell_mfe,
        sell_mae,
    )


def calculate_trade_outcomes_all_candles(
    df,
    atr_window=14,
    tp_mult=4.0,
    sl_mult=2.0,
    max_horizon=None,
):
    """
    Calculate trade outcomes for BOTH buy and sell at every candle.
    Returns DataFrame with columns: ['buy_outcome', 'sell_outcome', 'buy_exit_price', 'sell_exit_price',
    'buy_MFE', 'buy_MAE', 'sell_MFE', 'sell_MAE']

    Outcome encoding:
      1: Take Profit hit
     -1: Stop Loss hit
     NaN: Neither TP nor SL reached before end of dataset (only affects the
          last few bars). Callers should fillna(0.0).

    MFE/MAE are normalized to [0, 1] using the TP and SL distances from the
    entry price. They are measured only until the first exit for that side,
    so resolved trades have exactly one of MFE/MAE reaching 1.0.

    Looks forward to the end of the dataset with no time cap, matching
    production behaviour where a trade runs until TP or SL is hit.
    Uses a Numba-compiled inner loop for O(n) average-case performance.
    """
    df = df.copy()
    df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)

    highs = df['High'].values.astype(np.float64)
    lows = df['Low'].values.astype(np.float64)
    atrs = df['atr'].values.astype(np.float64)

    (
        buy_out,
        sell_out,
        buy_exit,
        sell_exit,
        buy_mfe,
        buy_mae,
        sell_mfe,
        sell_mae,
    ) = _compute_outcomes_nb(
        highs, lows, atrs, float(tp_mult), float(sl_mult)
    )

    return pd.DataFrame({
        'buy_outcome': buy_out,
        'sell_outcome': sell_out,
        'buy_exit_price': buy_exit,
        'sell_exit_price': sell_exit,
        'buy_MFE': buy_mfe,
        'buy_MAE': buy_mae,
        'sell_MFE': sell_mfe,
        'sell_MAE': sell_mae,
    }, index=df.index)


@njit(cache=True)
def _compute_outcomes_capped_nb(highs, lows, atrs, tp_mult, sl_mult, max_horizon):
    """
    Numba-compiled inner kernel for calculate_trade_outcomes_capped.
    Scans forward up to max_horizon bars only, matching production behaviour.
    Returns eight float64 arrays: buy_outcomes, sell_outcomes,
    buy_exit_prices, sell_exit_prices, buy_mfe, buy_mae, sell_mfe, sell_mae.
    Unresolved bars stay NaN for outcomes and exit prices.
    """
    n = len(highs)
    buy_outcomes = np.full(n, np.nan)
    sell_outcomes = np.full(n, np.nan)
    buy_exit_prices = np.full(n, np.nan)
    sell_exit_prices = np.full(n, np.nan)
    buy_mfe = np.full(n, np.nan)
    buy_mae = np.full(n, np.nan)
    sell_mfe = np.full(n, np.nan)
    sell_mae = np.full(n, np.nan)

    for t0 in range(n - 1):
        atr = atrs[t0]
        if np.isnan(atr) or atr == 0.0:
            continue

        entry_buy = highs[t0]
        tp_buy = entry_buy + tp_mult * atr
        sl_buy = entry_buy - sl_mult * atr
        tp_buy_dist = tp_buy - entry_buy
        sl_buy_dist = entry_buy - sl_buy

        entry_sell = lows[t0]
        tp_sell = entry_sell - tp_mult * atr
        sl_sell = entry_sell + sl_mult * atr
        tp_sell_dist = entry_sell - tp_sell
        sl_sell_dist = sl_sell - entry_sell

        buy_resolved = False
        sell_resolved = False
        buy_exit_type = 0
        sell_exit_type = 0
        buy_max_high = entry_buy
        buy_min_low = entry_buy
        sell_max_high = entry_sell
        sell_min_low = entry_sell

        horizon = min(t0 + 1 + max_horizon, n)
        for t1 in range(t0 + 1, horizon):
            h = highs[t1]
            l = lows[t1]

            if not buy_resolved:
                if h >= tp_buy:
                    buy_outcomes[t0] = 1.0
                    buy_exit_prices[t0] = tp_buy
                    buy_resolved = True
                    buy_exit_type = 1
                elif l <= sl_buy:
                    buy_outcomes[t0] = -1.0
                    buy_exit_prices[t0] = sl_buy
                    buy_resolved = True
                    buy_exit_type = -1

            if not sell_resolved:
                if l <= tp_sell:
                    sell_outcomes[t0] = 1.0
                    sell_exit_prices[t0] = tp_sell
                    sell_resolved = True
                    sell_exit_type = 1
                elif h >= sl_sell:
                    sell_outcomes[t0] = -1.0
                    sell_exit_prices[t0] = sl_sell
                    sell_resolved = True
                    sell_exit_type = -1

            if not np.isnan(h):
                if not buy_resolved or buy_exit_type == 1:
                    if h > buy_max_high:
                        buy_max_high = h
                if not sell_resolved or sell_exit_type == -1:
                    if h > sell_max_high:
                        sell_max_high = h
            if not np.isnan(l):
                if not buy_resolved or buy_exit_type == -1:
                    if l < buy_min_low:
                        buy_min_low = l
                if not sell_resolved or sell_exit_type == 1:
                    if l < sell_min_low:
                        sell_min_low = l

            if buy_resolved and sell_resolved:
                break

        if tp_buy_dist > 0.0:
            buy_mfe[t0] = min(max((buy_max_high - entry_buy) / tp_buy_dist, 0.0), 1.0)
        if sl_buy_dist > 0.0:
            buy_mae[t0] = min(max((entry_buy - buy_min_low) / sl_buy_dist, 0.0), 1.0)
        if tp_sell_dist > 0.0:
            sell_mfe[t0] = min(max((entry_sell - sell_min_low) / tp_sell_dist, 0.0), 1.0)
        if sl_sell_dist > 0.0:
            sell_mae[t0] = min(max((sell_max_high - entry_sell) / sl_sell_dist, 0.0), 1.0)

    return (
        buy_outcomes,
        sell_outcomes,
        buy_exit_prices,
        sell_exit_prices,
        buy_mfe,
        buy_mae,
        sell_mfe,
        sell_mae,
    )


def calculate_trade_outcomes_capped(
    df,
    atr_window=14,
    tp_mult=4.0,
    sl_mult=2.0,
    max_horizon=30,
):
    """
    Calculate trade outcomes for BOTH buy and sell at every candle,
    capped at max_horizon bars forward (matching production behaviour).

    Outcome encoding:
      1: Take Profit hit
     -1: Stop Loss hit
     NaN: Neither TP nor SL reached within max_horizon bars

    MFE/MAE are normalized to [0, 1] using the TP and SL distances.
    """
    df = df.copy()
    df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)

    highs = df['High'].values.astype(np.float64)
    lows = df['Low'].values.astype(np.float64)
    atrs = df['atr'].values.astype(np.float64)

    (
        buy_out,
        sell_out,
        buy_exit,
        sell_exit,
        buy_mfe,
        buy_mae,
        sell_mfe,
        sell_mae,
    ) = _compute_outcomes_capped_nb(
        highs, lows, atrs, float(tp_mult), float(sl_mult), int(max_horizon)
    )

    return pd.DataFrame({
        'buy_outcome': buy_out,
        'sell_outcome': sell_out,
        'buy_exit_price': buy_exit,
        'sell_exit_price': sell_exit,
        'buy_MFE': buy_mfe,
        'buy_MAE': buy_mae,
        'sell_MFE': sell_mfe,
        'sell_MAE': sell_mae,
    }, index=df.index)


def create_quality_targets(
    df: pd.DataFrame,
    lookahead_bars: int,
    atr_col: str = "atr"
) -> pd.DataFrame:
    """
    Generate regression targets for dual-head opportunity ranking.

    Computes:
      - long_mfe / long_mae
      - short_mfe / short_mae
      - long_quality = ln(1 + long_mfe_atr) - ln(1 + long_mae_atr)
      - short_quality = ln(1 + short_mfe_atr) - ln(1 + short_mae_atr)

    where MFE and MAE are computed over a fixed lookahead horizon, and
    expressed as multiples of the current ATR.

    This provides a continuous quality score instead of a discrete TP/SL label.

    Vectorized implementation to avoid slow loops.
    """
    highs = df["High"]
    lows = df["Low"]
    closes = df["Close"]
    atrs = df[atr_col]

    future_max_high = highs.rolling(window=lookahead_bars, min_periods=1).max().shift(-lookahead_bars)
    future_min_low = lows.rolling(window=lookahead_bars, min_periods=1).min().shift(-lookahead_bars)

    long_mfe = (future_max_high - closes).clip(lower=0.0)
    long_mae = (closes - future_min_low).clip(lower=0.0)

    short_mfe = (closes - future_min_low).clip(lower=0.0)
    short_mae = (future_max_high - closes).clip(lower=0.0)

    safe_atr = atrs.replace(0, np.nan)

    long_mfe_atr = long_mfe / safe_atr
    long_mae_atr = long_mae / safe_atr

    short_mfe_atr = short_mfe / safe_atr
    short_mae_atr = short_mae / safe_atr

    long_quality = np.log1p(long_mfe_atr) - np.log1p(long_mae_atr)
    short_quality = np.log1p(short_mfe_atr) - np.log1p(short_mae_atr)

    return pd.DataFrame({
        "long_quality": long_quality.fillna(0.0),
        "short_quality": short_quality.fillna(0.0),
        "long_mfe_atr": long_mfe_atr.fillna(0.0),
        "long_mae_atr": long_mae_atr.fillna(0.0),
        "short_mfe_atr": short_mfe_atr.fillna(0.0),
        "short_mae_atr": short_mae_atr.fillna(0.0)
    }, index=df.index)


def MFE_filter_outcomes(df, mfe_thresh=1, mfa_thresh=0.30, label_params=None, outcome_params=None):
    """
    Filter outcomes to identify "high-quality" trades with MFE above mfe_thresh
    and MAE below mae_thresh.

    buy/sell_signal are 1 for bars where the respective side had an outcome
    meeting the thresholds, else 0. This allows identifying bars with high-quality
    outcomes on one or both sides.
    """

    def filter_targets(t, b, s):
        """Helper function to filter original target based on buy/sell signals."""
        if abs(t) == b or abs(t) == s:
            return t
        return 0

    if label_params is not None:
        df_signals = causal_triple_barrier_hilow_trend_labeler(df, **label_params)
    else:
        df_signals = causal_triple_barrier_hilow_trend_labeler(df)

    winning_trades = df_signals[df_signals['label'] == 1]
    df['target'] = 0
    df.loc[winning_trades.index, 'target'] = winning_trades['side']

    if outcome_params is not None:
        df_outcomes = calculate_trade_outcomes_all_candles(df, **outcome_params)
    else:
        df_outcomes = calculate_trade_outcomes_all_candles(df)

    buy_signal = ((df_outcomes['buy_MFE'] >= mfe_thresh) & (df_outcomes['buy_MAE'] <= mfa_thresh)).astype(int)
    sell_signal = ((df_outcomes['sell_MFE'] >= mfe_thresh) & (df_outcomes['sell_MAE'] <= mfa_thresh)).astype(int)

    filtered_outcomes = pd.DataFrame({
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'buy_exit_price': df_outcomes['buy_exit_price'],
        'sell_exit_price': df_outcomes['sell_exit_price'],
        'buy_MFE': df_outcomes['buy_MFE'],
        'buy_MAE': df_outcomes['buy_MAE'],
        'sell_MFE': df_outcomes['sell_MFE'],
        'sell_MAE': df_outcomes['sell_MAE'],
    }, index=df_outcomes.index)

    cols = df.columns
    output = df.merge(filtered_outcomes, left_index=True, right_index=True)

    output['target'] = output.apply(lambda row: filter_targets(row['target'], row['buy_signal'], row['sell_signal']), axis=1)
    return output[cols], df_outcomes


def build_flattened_sequence_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    sequence_length: int = 128,
    include_ohlc: bool = True,
    time_col: str = "Time",
    prefix: str = "sq",
    dtype: str = "float32",
    drop_invalid_windows: bool = True,
) -> pd.DataFrame:
    """
    Build one row per anchor timestep containing a flattened trailing sequence.

    Each output row corresponds to anchor t and contains window [t-seq_len+1 : t].
    Flattened columns use compact numeric names like sq_l000_f000.

    Returns a dataframe suitable for LightGBM where each row is tabular.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame")

    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")

    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    ordered_cols: list[str] = []
    if include_ohlc:
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                ordered_cols.append(col)

    for col in feature_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    if not ordered_cols:
        raise ValueError("No usable feature columns found in dataframe")

    missing_requested = [c for c in feature_cols if c not in df.columns]
    if missing_requested:
        print(f"Warning: dropped missing feature columns: {missing_requested}")

    n_rows = len(df)
    if n_rows < sequence_length:
        raise ValueError(
            f"Not enough rows for sequence_length={sequence_length}. Rows={n_rows}"
        )

    working = df[ordered_cols + [label_col]].copy()
    if time_col in df.columns:
        working[time_col] = df[time_col].values

    finite_cols = ordered_cols + [label_col]
    valid_row = np.isfinite(working[finite_cols].to_numpy(dtype=np.float64)).all(axis=1)
    valid_counts = np.convolve(
        valid_row.astype(np.int32),
        np.ones(sequence_length, dtype=np.int32),
        mode="valid",
    )
    valid_anchor = valid_counts == sequence_length

    np_dtype = np.float32 if dtype == "float32" else np.float64
    X = np.ascontiguousarray(working[ordered_cols].to_numpy(dtype=np_dtype))
    y = working[label_col].to_numpy()

    windows = sliding_window_view(X, window_shape=sequence_length, axis=0)
    flat_matrix = windows.reshape(windows.shape[0], -1)

    n_features = len(ordered_cols)
    flat_cols = [
        f"{prefix}_l{lag:03d}_f{feat:03d}"
        for lag in range(sequence_length)
        for feat in range(n_features)
    ]

    anchor_index = np.arange(sequence_length - 1, n_rows)

    out = pd.DataFrame(flat_matrix, columns=flat_cols)
    out["anchor_index"] = anchor_index
    out[label_col] = y[anchor_index]

    if time_col in working.columns:
        out["anchor_time"] = working[time_col].iloc[anchor_index].to_numpy()

    out["sequence_length"] = int(sequence_length)
    out["source_feature_count"] = int(n_features)

    if drop_invalid_windows:
        out = out.loc[valid_anchor].reset_index(drop=True)

    return out


def _coerce_inputs(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    timestamps: pd.Series | np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalize and validate feature, target, and timestamp inputs.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Feature matrix in chronological order.
    y : np.ndarray | pd.Series
        Binary target vector where 1 = trade and 0 = no trade.
    timestamps : pd.Series | np.ndarray
        Timestamp series aligned 1:1 with X and y.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        Normalized (X_df, y_series, ts_series) objects with aligned lengths.

    Raises
    ------
    ValueError
        If lengths mismatch, timestamps are invalid, or y is not binary-like.
    """
    if isinstance(X, pd.DataFrame):
        X_df = X
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        X_df = pd.DataFrame(X_arr)

    y_series = pd.Series(y, copy=False).reset_index(drop=True)
    ts_series = pd.Series(timestamps, copy=False).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    n_samples = len(X_df)
    if len(y_series) != n_samples or len(ts_series) != n_samples:
        raise ValueError(
            "X, y, and timestamps must have identical lengths. "
            f"Got len(X)={n_samples}, len(y)={len(y_series)}, len(timestamps)={len(ts_series)}."
        )

    unique_y = pd.Series(y_series.dropna().unique())
    if not unique_y.isin([0, 1]).all():
        raise ValueError("y must be binary and encoded as 0/1.")

    ts_dt = pd.to_datetime(ts_series, errors="coerce")
    if ts_dt.isna().any():
        raise ValueError("timestamps contains non-parsable values.")

    if not ts_dt.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in chronological ascending order.")

    return X_df, y_series.astype(np.int8), ts_dt


def _build_expanding_window_folds(
    n_samples: int,
    n_folds: int,
    min_train_size: int,
    test_size: int,
    gap_size: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window walk-forward train/test index folds."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if n_folds <= 0:
        raise ValueError("n_folds must be > 0.")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be > 0.")
    if test_size <= 0:
        raise ValueError("test_size must be > 0.")
    if gap_size < 0:
        raise ValueError("gap_size must be >= 0.")

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_size

    for _ in range(n_folds):
        test_start = train_end + gap_size
        if test_start >= n_samples:
            break

        test_end = min(test_start + test_size, n_samples)
        if test_end <= test_start:
            break

        train_idx = np.arange(0, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)

        if len(train_idx) == 0 or len(test_idx) == 0:
            break

        folds.append((train_idx, test_idx))
        train_end = test_end

        if train_end >= n_samples:
            break

    return folds


def _safe_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Compute ROC AUC and PR AUC, returning NaN when undefined."""
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def generate_tradeability_scores(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    timestamps: pd.Series | np.ndarray,
    n_folds: int = 10,
    min_train_size: int = 100_000,
    test_size: int = 50_000,
    gap_size: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate out-of-fold tradeability probabilities via expanding-window LightGBM.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Feature matrix in chronological order.
    y : np.ndarray | pd.Series
        Binary target where 1 = trade signal and 0 = no trade.
    timestamps : pd.Series | np.ndarray
        Chronological timestamps aligned to rows in X.
    n_folds : int, default=10
        Maximum number of walk-forward folds.
    min_train_size : int, default=100_000
        Initial training window size.
    test_size : int, default=50_000
        Target test window size per fold; final fold may be partial.
    gap_size : int, default=0
        Number of rows to skip between train and test windows.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - scores_df: columns [timestamp, tradeability_score, target]
        - metrics_dict: overall metrics and per-fold diagnostics/metrics
    """
    X_df, y_series, ts_series = _coerce_inputs(X=X, y=y, timestamps=timestamps)
    n_samples = len(X_df)

    folds = _build_expanding_window_folds(
        n_samples=n_samples,
        n_folds=n_folds,
        min_train_size=min_train_size,
        test_size=test_size,
        gap_size=gap_size,
    )
    if not folds:
        raise ValueError(
            "No valid folds were generated. "
            "Check min_train_size/test_size/gap_size against dataset length."
        )

    scores = np.full(n_samples, np.nan, dtype=np.float32)
    fold_metrics: List[Dict[str, Any]] = []

    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        print(f"Processing fold {fold_number}/{len(folds)}: "
              f"train [{train_idx[0]}:{train_idx[-1]}], "
              f"test [{test_idx[0]}:{test_idx[-1]}]")
        X_train = X_df.iloc[train_idx]
        y_train = y_series.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_test = y_series.iloc[test_idx]

        model = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1].astype(np.float32, copy=False)
        scores[test_idx] = y_prob

        fold_roc_auc, fold_pr_auc = _safe_auc_metrics(y_test.to_numpy(), y_prob)
        train_pos_rate = float(y_train.mean()) if len(y_train) else float("nan")
        test_pos_rate = float(y_test.mean()) if len(y_test) else float("nan")

        train_start_ts = ts_series.iloc[int(train_idx[0])]
        train_end_ts = ts_series.iloc[int(train_idx[-1])]
        test_start_ts = ts_series.iloc[int(test_idx[0])]
        test_end_ts = ts_series.iloc[int(test_idx[-1])]

        fold_metrics.append(
            {
                "fold": fold_number,
                "train_start": train_start_ts,
                "train_end": train_end_ts,
                "test_start": test_start_ts,
                "test_end": test_end_ts,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_positive_rate": train_pos_rate,
                "test_positive_rate": test_pos_rate,
                "roc_auc": fold_roc_auc,
                "pr_auc": fold_pr_auc,
            }
        )

    valid_mask = np.isfinite(scores)
    y_valid = y_series.to_numpy()[valid_mask]
    scores_valid = scores[valid_mask]
    overall_roc_auc, overall_pr_auc = _safe_auc_metrics(y_valid, scores_valid)

    scores_df = pd.DataFrame(
        {
            "timestamp": ts_series,
            "tradeability_score": scores,
            "target": y_series,
        }
    )

    metrics_dict: Dict[str, Any] = {
        "overall_roc_auc": overall_roc_auc,
        "overall_pr_auc": overall_pr_auc,
        "fold_metrics": fold_metrics,
        "n_scored_rows": int(valid_mask.sum()),
        "n_total_rows": int(n_samples),
    }
    return scores_df, metrics_dict


def plot_tradeability_distribution(
    scores_df: pd.DataFrame,
    score_col: str = "tradeability_score",
    target_col: str = "target",
    bins: int = 50,
) -> Tuple[go.Figure, go.Figure]:
    """Plot score distributions overall and by class using Plotly."""
    required_cols = {score_col, target_col}
    missing = required_cols.difference(scores_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_df = scores_df[[score_col, target_col]].dropna(subset=[score_col]).copy()

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=plot_df[score_col],
            nbinsx=bins,
            name="All Scores",
            marker_color="#1f77b4",
            opacity=0.8,
        )
    )
    fig_hist.update_layout(
        title="Tradeability Score Distribution",
        xaxis_title="Tradeability Score",
        yaxis_title="Count",
        bargap=0.05,
    )

    fig_by_class = go.Figure()
    class_colors = {0: "#636EFA", 1: "#EF553B"}
    for class_value in sorted(plot_df[target_col].dropna().unique()):
        class_mask = plot_df[target_col] == class_value
        fig_by_class.add_trace(
            go.Histogram(
                x=plot_df.loc[class_mask, score_col],
                nbinsx=bins,
                name=f"Class {class_value}",
                opacity=0.55,
                marker_color=class_colors.get(int(class_value), None),
            )
        )

    fig_by_class.update_layout(
        title="Tradeability Score Distribution by Class",
        xaxis_title="Tradeability Score",
        yaxis_title="Count",
        barmode="overlay",
        bargap=0.05,
    )

    fig_hist.show()
    fig_by_class.show()
    return fig_hist, fig_by_class


def plot_precision_recall_by_percentile(
    scores_df: pd.DataFrame,
    score_col: str = "tradeability_score",
    target_col: str = "target",
    n_buckets: int = 20,
) -> Tuple[pd.DataFrame, go.Figure]:
    """Plot actual trade rate by descending score percentile bucket."""
    if n_buckets <= 0:
        raise ValueError("n_buckets must be > 0.")

    required_cols = {score_col, target_col}
    missing = required_cols.difference(scores_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_df = scores_df[[score_col, target_col]].dropna(subset=[score_col, target_col]).copy()
    if plot_df.empty:
        raise ValueError("No rows available after dropping NaN score/target values.")

    plot_df = plot_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_rows = len(plot_df)

    bucket_idx = (np.arange(n_rows) * n_buckets) // n_rows
    bucket_idx = np.minimum(bucket_idx, n_buckets - 1)
    plot_df["percentile_bucket"] = bucket_idx.astype(int)

    bucket_summary = (
        plot_df.groupby("percentile_bucket", as_index=False)
        .agg(
            bucket_size=(target_col, "size"),
            actual_trade_rate=(target_col, "mean"),
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            score_mean=(score_col, "mean"),
        )
        .sort_values("percentile_bucket")
        .reset_index(drop=True)
    )

    pct_per_bucket = 100.0 / n_buckets
    bucket_summary["percentile_start"] = bucket_summary["percentile_bucket"] * pct_per_bucket
    bucket_summary["percentile_end"] = (bucket_summary["percentile_bucket"] + 1.0) * pct_per_bucket
    bucket_summary["percentile_label"] = bucket_summary.apply(
        lambda r: f"{r['percentile_start']:.0f}-{r['percentile_end']:.0f}%", axis=1
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bucket_summary["percentile_label"],
            y=bucket_summary["actual_trade_rate"],
            marker_color="#00CC96",
            name="Actual Trade Rate",
        )
    )
    fig.update_layout(
        title="Actual Trade Rate by Descending Score Percentile Bucket",
        xaxis_title="Percentile Bucket (0% = highest scores)",
        yaxis_title="Actual Trade Rate",
    )
    fig.show()

    return bucket_summary, fig


def create_purified_labels(
    df: pd.DataFrame,
    score_col: str = "tradeability_score",
    label_col: str = "label",
    keep_top_percent: float = 50,
    flat_label: int = 0,
    timestamp_col: str | None = "timestamp",
) -> pd.DataFrame:
    """Create purified labels with a chronological walk-forward filter.

    Logic
    -----
    - Preserve the input row order as the walk-forward order.
    - If timestamp_col is present, require chronological ascending timestamps.
    - Keep all FLAT labels unchanged.
    - Process trade labels oldest first and newest last.
        - For each trade row with a finite score, keep it only if its score sits
            inside the top keep_top_percent of scored trade rows seen so far,
            including the current row.
        - Trade rows with NaN/non-finite scores are treated as unscored and are
            discarded to FLAT.
    - Convert discarded trade labels to FLAT.
    """
    from bisect import insort_right

    if not (0 <= keep_top_percent <= 100):
        raise ValueError("keep_top_percent must be in [0, 100].")

    required_cols = {score_col, label_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if timestamp_col is not None and timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if ts.isna().any():
            raise ValueError(f"{timestamp_col} contains non-parsable values.")
        if not ts.is_monotonic_increasing:
            raise ValueError(
                f"{timestamp_col} must be sorted in chronological ascending order "
                "(oldest first, newest last)."
            )

    out = df.copy()
    out["is_trade_label"] = out[label_col] != flat_label
    out["keep_trade_label"] = False
    out["walk_forward_keep_threshold"] = np.nan
    out["walk_forward_seen_trade_count"] = 0

    trade_idx = out.index[out["is_trade_label"]].tolist()
    if not trade_idx:
        out["purified_label"] = out[label_col]
        return out

    keep_fraction = keep_top_percent / 100.0
    sorted_seen_scores: list[float] = []
    seen_scored_trade_count = 0

    for row_idx in trade_idx:
        score = out.at[row_idx, score_col]
        is_scored = pd.notna(score) and np.isfinite(float(score))

        if is_scored:
            score_value = float(score)
            insort_right(sorted_seen_scores, score_value)
            seen_scored_trade_count += 1
        else:
            score_value = np.nan

        n_seen = seen_scored_trade_count
        n_keep = int(np.ceil(keep_fraction * n_seen))
        out.at[row_idx, "walk_forward_seen_trade_count"] = n_seen

        if (not is_scored) or n_keep <= 0 or n_seen <= 0:
            threshold = np.nan
            keep_current = False
        else:
            threshold = sorted_seen_scores[n_seen - n_keep]
            keep_current = score_value >= threshold

        out.at[row_idx, "walk_forward_keep_threshold"] = threshold if np.isfinite(threshold) else np.nan
        out.at[row_idx, "keep_trade_label"] = keep_current

    out["purified_label"] = out[label_col]
    discard_mask = out["is_trade_label"] & (~out["keep_trade_label"])
    out.loc[discard_mask, "purified_label"] = flat_label

    return out