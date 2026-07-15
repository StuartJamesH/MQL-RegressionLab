"""
Engine/v2/strategy.py — v2 distributional signal strategy.

Builds causal input windows from a ring buffer of completed bars, runs the
v2 transformer inference, converts distributional outputs into scalar signals
in ``[-1, 1]``, and emits ``DataHandler.Order`` objects sized by the v2 risk
manager and Kelly position sizer.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from Engine.DataHandler import Order
from Engine.TicketBook import TicketBook
from Engine.v2.features import encode_live_session_features, normalize_live_ohlcv
from Engine.v2.inference import V2InferenceEngine
from Learn.v2.position_sizing import KellyPositionSizer
from Learn.v2.risk_manager import RiskConfig, RiskManager
from Learn.v2.signals import DistributionalSignalGenerator

_LOG = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return a naive UTC datetime (consistent with TicketBook internals)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class V2RiskConfig:
    """Runtime risk parameters for the v2 strategy."""

    risk_per_trade: float = 50.0          # Max account-currency risk per trade
    max_position_lots: float = 0.5        # Hard lot cap
    account_equity: float = 10_000.0      # Used by Kelly sizing
    signal_threshold: float = 0.1         # Minimum |signal| to trade
    patience_bars: int = 5                # Pending-order expiry in bars/minutes
    temperature: float = 1.0              # Signal temperature
    primary_horizon_idx: int = 2          # 20-bar horizon by default
    extreme_regime_idx: int = 3           # Class index considered extreme
    order_type: str = "stop"              # "stop" or "market"
    trailing_stop_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0
    atr_window: int = 14


class V2SignalStrategy:
    """
    v2 transformer signal strategy.

    Parameters
    ----------
    symbol : str
        Instrument ticker, e.g. ``'EURUSD'``.
    pack : dict
        Loaded model pack from :class:`~Engine.v2.model_pack.ModelPackLoader`.
    inference_engine : V2InferenceEngine
        Initialised ONNX or PyTorch inference engine.
    risk_config : V2RiskConfig
        Runtime risk parameters.
    ticket_book : TicketBook
        Shared order-state journal.
    mt5_executor : Any
        Execution handler providing ``get_point_value(symbol)``.
    **kwargs
        Forward compatibility overrides (ignored).
    """

    def __init__(
        self,
        symbol: str,
        pack: Dict[str, Any],
        inference_engine: V2InferenceEngine,
        risk_config: V2RiskConfig,
        ticket_book: TicketBook,
        mt5_executor: Any,
        **kwargs: Any,
    ) -> None:
        self.symbol = symbol
        self.pack = pack
        self.config = pack["config"]
        self.inference_engine = inference_engine
        self.risk_config = risk_config
        self.ticket_book = ticket_book
        self.mt5_executor = mt5_executor
        self.order_type = risk_config.order_type

        self._signal_generator = DistributionalSignalGenerator(
            temperature=risk_config.temperature,
            signal_threshold=risk_config.signal_threshold,
            extreme_regime_gate=True,
            regime_idx=risk_config.extreme_regime_idx,
            primary_horizon_idx=risk_config.primary_horizon_idx,
        )

        self._risk_manager = RiskManager(
            RiskConfig(
                trailing_stop_atr_mult=risk_config.trailing_stop_atr_mult,
                take_profit_atr_mult=risk_config.take_profit_atr_mult,
                atr_window=risk_config.atr_window,
            )
        )
        self._position_sizer = KellyPositionSizer(
            max_position_pct=0.05, half_kelly=True
        )

        self._buffer: pd.DataFrame = pd.DataFrame(
            columns=["Time", "Open", "High", "Low", "Close", "Volume"]
        )
        self._bar_count: int = 0
        self._last_signal: float = 0.0
        self._last_inference_latency_ms: Optional[float] = None
        self._last_error: Optional[str] = None
        self._last_bar_time: Optional[datetime] = None
        self._live_ready: bool = False

        # CSV trade log
        self._trade_log_path = self._init_trade_log(pack)
        self._trade_log_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Trade log setup
    # ------------------------------------------------------------------

    def _init_trade_log(self, pack: Dict[str, Any]) -> str:
        model_name = pack.get("model_info", {}).get("model_name", "v2_model")
        log_dir = os.path.join("Engine", "v2", "Trade Logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{self.symbol}_{model_name}_{timestamp}.csv")
        return path

    def _flush_trade_log(self) -> None:
        if not self._trade_log_rows:
            return
        df = pd.DataFrame(self._trade_log_rows)
        header = not os.path.exists(self._trade_log_path)
        df.to_csv(self._trade_log_path, mode="a", index=False, header=header)
        self._trade_log_rows = []

    def close(self) -> None:
        """Flush any pending trade-log rows."""
        self._flush_trade_log()

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _append_bar(self, bar) -> None:
        new_row = pd.DataFrame(
            [
                {
                    "Time": getattr(bar, "Time", None),
                    "Open": float(bar.Open),
                    "High": float(bar.High),
                    "Low": float(bar.Low),
                    "Close": float(bar.Close),
                    "Volume": float(bar.Volume),
                }
            ]
        )
        if self._buffer.empty:
            self._buffer = new_row.copy()
        else:
            self._buffer = pd.concat([self._buffer, new_row], ignore_index=True)
        # Keep slightly more than max_seq_len so ATR / normalisation windows have
        # a small history cushion at the front.
        max_keep = self.config.max_seq_len + self.risk_config.atr_window + 10
        if len(self._buffer) > max_keep:
            self._buffer = self._buffer.iloc[-max_keep:].reset_index(drop=True)

    def _build_input_tensor(self) -> tuple[np.ndarray, np.ndarray]:
        """Build ``x_raw`` and ``x_session`` from the internal ring buffer."""
        window = self._buffer.iloc[-self.config.max_seq_len :].copy()
        x_raw = normalize_live_ohlcv(window)  # (seq_len, 5)
        x_session = encode_live_session_features(
            pd.DatetimeIndex(window["Time"]),
            include_gap=(self.config.session_channels == 5),
        )
        return x_raw, x_session

    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
        """Return the latest ATR value from a trailing window of bars."""
        if len(df) < window + 1:
            # Not enough history: use the average bar range as a fallback.
            ranges = df["High"] - df["Low"]
            return float(ranges.mean()) if len(ranges) > 0 else 0.0

        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean().iloc[-1]
        return float(atr)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _signal_from_outputs(self, outputs: Dict[str, np.ndarray]) -> float:
        """
        Port of ``DistributionalSignalGenerator.generate`` to work with the
        dict returned by :class:`V2InferenceEngine`.
        """
        h = self.risk_config.primary_horizon_idx

        mu = outputs["mu"][:, h]
        log_sigma = outputs["log_sigma"][:, h]
        direction = outputs["direction"][:, h]
        regime = outputs["regime"]

        sigma = np.exp(log_sigma) + 1e-6
        s = mu / sigma

        p_up = 1.0 / (1.0 + np.exp(-direction))
        c = 2.0 * np.abs(p_up - 0.5)

        signal = np.sign(s) * np.tanh(np.abs(s) * c / self.risk_config.temperature)

        if self.risk_config.extreme_regime_idx is not None:
            regime_pred = np.argmax(regime, axis=1)
            signal[regime_pred == self.risk_config.extreme_regime_idx] = 0.0

        signal[np.abs(signal) < self.risk_config.signal_threshold] = 0.0
        return float(signal[0])

    # ------------------------------------------------------------------
    # Order construction helpers
    # ------------------------------------------------------------------

    def _get_point_value(self) -> float:
        try:
            return float(self.mt5_executor.get_point_value(self.symbol))
        except Exception as exc:
            _LOG.warning(
                "Could not determine point value for %s: %s. Falling back to 1.0.",
                self.symbol,
                exc,
            )
            return 1.0

    def _size_position(
        self,
        signal: float,
        close: float,
        sl_distance: float,
        tp_distance: float,
        point_value: float,
    ) -> float:
        """Return position size in lots."""
        if sl_distance <= 0 or point_value <= 0:
            return 0.0

        # Directional win probability and expected payoff from model confidence.
        p_up = 1.0 / (1.0 + np.exp(-abs(self._last_raw_direction)))
        win_prob = float(p_up)

        avg_win = tp_distance * point_value
        avg_loss = sl_distance * point_value

        kelly_size_currency = self._position_sizer.compute_size(
            win_prob=win_prob,
            avg_win=max(avg_win, 1e-6),
            avg_loss=max(avg_loss, 1e-6),
            account_equity=self.risk_config.account_equity,
        )

        # Convert currency to lots and cap by risk-per-trade.
        max_lots_by_risk = self.risk_config.risk_per_trade / (sl_distance * point_value)
        lots = kelly_size_currency / (sl_distance * point_value)
        lots = min(lots, max_lots_by_risk, self.risk_config.max_position_lots)

        # Broker volume step clamping (best-effort when symbol info is available).
        lots = self._clamp_to_volume_step(lots)
        return max(float(lots), 0.0)

    def _clamp_to_volume_step(self, lots: float) -> float:
        """Best-effort clamp to MT5 symbol volume constraints."""
        try:
            import MetaTrader5 as mt5

            info = mt5.symbol_info(self.symbol)
            if info is None:
                return lots
            volume_min = getattr(info, "volume_min", 0.01)
            volume_max = getattr(info, "volume_max", 1000.0)
            volume_step = getattr(info, "volume_step", 0.01)
            lots = max(lots, volume_min)
            lots = min(lots, volume_max)
            lots = round(lots / volume_step) * volume_step
            # Round to a sensible number of decimals to avoid float noise.
            decimals = max(0, int(round(-np.log10(volume_step))))
            lots = round(lots, decimals)
            return lots
        except Exception:
            return lots

    # ------------------------------------------------------------------
    # Main per-bar handler
    # ------------------------------------------------------------------

    def on_bar(self, bar) -> List[Order]:
        """
        Process one completed bar and return zero or one :class:`Order`.

        During warm-up (fewer than ``config.max_seq_len`` bars in the buffer
        OR the initial historical-burst phase still in progress) an empty list
        is returned.  The burst phase is detected by tracking inter-bar
        wall-clock gaps: bars arriving within 5 seconds of each other are
        considered part of the warm-up burst and suppressed.

        If a pending order or open position exists for the symbol, inference
        is skipped and an empty list is returned.
        """
        self._append_bar(bar)
        self._bar_count += 1

        bar_time = pd.to_datetime(getattr(bar, "Time", _utc_now()))
        close = float(bar.Close)

        # Detect the transition from historical warm-up burst to live bars.
        # During the initial poll the data handler yields hundreds of completed
        # bars in a tight loop (milliseconds apart).  Once live, bars arrive
        # with natural timeframe gaps.  A 5-second inter-arrival threshold
        # cleanly separates the two phases for all practical timeframes.
        now = _utc_now()
        if self._last_bar_time is not None:
            if (now - self._last_bar_time).total_seconds() > 5.0:
                if not self._live_ready:
                    _LOG.info(
                        "Live mode detected after %d bars (gap=%.1fs)",
                        self._bar_count,
                        (now - self._last_bar_time).total_seconds(),
                    )
                self._live_ready = True
        self._last_bar_time = now

        # Health log every 100 bars.
        if self._bar_count % 100 == 0:
            _LOG.info(
                "V2 health — bar_count=%d buffer=%d signal=%.4f "
                "inference_ms=%s last_error=%s",
                self._bar_count,
                len(self._buffer),
                self._last_signal,
                self._last_inference_latency_ms,
                self._last_error,
            )

        # Warm-up guard: suppress orders until buffer is full AND the initial
        # historical-burst warm-up has been fully consumed (live bars detected).
        if len(self._buffer) < self.config.max_seq_len or not self._live_ready:
            action = "warmup" if len(self._buffer) < self.config.max_seq_len else "warmup_burst"
            self._log_bar(
                bar_time=bar_time,
                bar=bar,
                signal=0.0,
                outputs=None,
                side=None,
                entry=None,
                sl=None,
                tp=None,
                lots=None,
                action=action,
            )
            return []

        # State guard: only one signal / position at a time per symbol.
        has_pending = self.ticket_book.has_pending_order(self.symbol)
        has_open = self.ticket_book.has_open_position(self.symbol)
        if has_pending or has_open:
            self._log_bar(
                bar_time=bar_time,
                bar=bar,
                signal=0.0,
                outputs=None,
                side=None,
                entry=None,
                sl=None,
                tp=None,
                lots=None,
                action="skipped_existing_exposure",
            )
            return []

        # Build tensors.
        try:
            x_raw, x_session = self._build_input_tensor()
        except Exception as exc:
            self._last_error = f"feature_build_failed:{exc}"
            _LOG.warning("Failed to build input tensor: %s", exc)
            return []

        if not np.isfinite(x_raw).all() or not np.isfinite(x_session).all():
            self._last_error = "non_finite_input"
            _LOG.warning("Skipping inference: non-finite values in input tensor")
            return []

        # Inference.
        t0 = _utc_now()
        try:
            outputs = self.inference_engine.predict(x_raw, x_session)
            self._last_error = None
        except Exception as exc:
            self._last_error = f"inference_failed:{exc}"
            _LOG.warning("Inference failed: %s", exc)
            return []
        self._last_inference_latency_ms = (
            _utc_now() - t0
        ).total_seconds() * 1000.0

        # Signal.
        self._last_raw_direction = float(outputs["direction"][0, self.risk_config.primary_horizon_idx])
        signal = self._signal_from_outputs(outputs)
        self._last_signal = signal

        if abs(signal) < self.risk_config.signal_threshold:
            self._log_bar(
                bar_time=bar_time,
                bar=bar,
                signal=signal,
                outputs=outputs,
                side=None,
                entry=None,
                sl=None,
                tp=None,
                lots=None,
                action="below_threshold",
            )
            return []

        # Build order.
        side = "buy" if signal > 0 else "sell"
        direction = 1 if side == "buy" else -1

        current_atr = self._compute_atr(self._buffer, self.risk_config.atr_window)
        if current_atr <= 0:
            self._last_error = "invalid_atr"
            _LOG.warning("Cannot compute ATR; skipping signal")
            return []

        tp_price, sl_price, _ = self._risk_manager.compute_exit_levels(
            entry_price=close,
            direction=direction,
            current_atr=current_atr,
            account_equity=self.risk_config.account_equity,
        )

        # Stop-entry price slightly beyond the completed bar's close to emulate
        # a momentum stop order on the next bar.
        entry_buffer = max(current_atr * 0.1, close * 0.0001)
        entry = close + direction * entry_buffer

        sl_distance = abs(entry - sl_price)
        tp_distance = abs(tp_price - entry)
        point_value = self._get_point_value()
        lots = self._size_position(
            signal=signal,
            close=close,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            point_value=point_value,
        )

        if lots <= 0:
            self._log_bar(
                bar_time=bar_time,
                bar=bar,
                signal=signal,
                outputs=outputs,
                side=side,
                entry=entry,
                sl=sl_price,
                tp=tp_price,
                lots=0.0,
                action="zero_size",
            )
            return []

        expiration = bar_time + timedelta(minutes=int(self.risk_config.patience_bars))

        order = Order(
            symbol=self.symbol,
            side=side,
            entry=entry,
            qty=lots,
            entry_time=bar_time.isoformat(),
            expiration=expiration,
            sl=sl_price,
            tp=tp_price,
        )

        self._log_bar(
            bar_time=bar_time,
            bar=bar,
            signal=signal,
            outputs=outputs,
            side=side,
            entry=entry,
            sl=sl_price,
            tp=tp_price,
            lots=lots,
            action="submit_stop_order",
        )
        return [order]

    # ------------------------------------------------------------------
    # Trailing-stop update
    # ------------------------------------------------------------------

    def apply_trailing_stops(self, bar) -> None:
        """
        Compute and apply trailing-stop updates for all open positions.

        Called once per bar after order processing.  For each open position,
        computes a trailing SL using the current bar's High/Low and the
        configured ATR multiplier, then sends the updated SL to MT5 via the
        executor if it has moved in the favourable direction.

        Parameters
        ----------
        bar :
            Current completed bar with ``High``, ``Low`` fields (as yielded
            by the data handler).
        """
        if self.ticket_book is None:
            return

        open_positions = list(self.ticket_book.get_open_positions())
        if not open_positions:
            return

        current_atr = self._compute_atr(self._buffer, self.risk_config.atr_window)
        if current_atr <= 0:
            return

        for record in open_positions:
            direction = 1 if record.side == "buy" else -1
            current_sl = record.sl or 0.0

            current_bar = {
                "High": float(bar.High),
                "Low": float(bar.Low),
                "Close": float(bar.Close),
                "atr": current_atr,
            }

            position_info = {
                "direction": direction,
                "trailing_sl": current_sl,
                "entry_price": record.entry_price,
            }

            new_sl = self._risk_manager.update_trailing_stop(
                position_info, current_bar
            )

            if new_sl != current_sl:
                _LOG.info(
                    "Trailing stop updated for ticket=%d: %.5f -> %.5f",
                    record.ticket, current_sl, new_sl,
                )
                self.mt5_executor.modify_position_sl(record.ticket, new_sl)

    # ------------------------------------------------------------------
    # Trade-log row helper
    # ------------------------------------------------------------------

    def _log_bar(
        self,
        bar_time: Any,
        bar,
        signal: float,
        outputs: Optional[Dict[str, np.ndarray]],
        side: Optional[str],
        entry: Optional[float],
        sl: Optional[float],
        tp: Optional[float],
        lots: Optional[float],
        action: str,
    ) -> None:
        h = self.risk_config.primary_horizon_idx
        row: Dict[str, Any] = {
            "bar_time": bar_time,
            "open": float(bar.Open),
            "high": float(bar.High),
            "low": float(bar.Low),
            "close": float(bar.Close),
            "volume": float(bar.Volume),
            "signal": signal,
            "mu_h2": float(outputs["mu"][0, h]) if outputs else None,
            "sigma_h2": float(np.exp(outputs["log_sigma"][0, h])) if outputs else None,
            "direction_prob": (
                float(1.0 / (1.0 + np.exp(-outputs["direction"][0, h])))
                if outputs
                else None
            ),
            "regime": int(np.argmax(outputs["regime"], axis=1)[0]) if outputs else None,
            "side": side,
            "entry": entry,
            "stop": sl,
            "take": tp,
            "position_size_lots": lots,
            "pending_order": self.ticket_book.has_pending_order(self.symbol),
            "open_position": self.ticket_book.has_open_position(self.symbol),
            "action_taken": action,
        }
        self._trade_log_rows.append(row)
        # Flush periodically to keep the CSV close to real-time.
        if len(self._trade_log_rows) >= 10:
            self._flush_trade_log()
