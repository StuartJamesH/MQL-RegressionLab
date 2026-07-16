"""
Engine/v2/engine.py — v2 live trading orchestrator.

Wires a data handler, strategy, and executor into the proven per-bar loop and
provides graceful shutdown hooks.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from Engine.Engine import _bar_time_as_utc

_LOG = logging.getLogger(__name__)


class V2LiveEngine:
    """
    Top-level orchestrator for the v2 transformer runtime.

    Parameters
    ----------
    data_handler :
        Any object exposing ``get_next_bar()`` (e.g. :class:`V2MT5DataHandler`
        or :class:`Engine.DataHandler.DataHandler`).
    strategy :
        Strategy implementing ``on_bar(bar) -> list[Order]`` and an
        ``order_type`` attribute (``'market'`` or ``'stop'``).
    executor :
        Execution handler with ``execute_market_order``, ``submit_stop_order``,
        ``process_pending_batch``, and ``process_position_updates_batch``.
    """

    def __init__(self, data_handler, strategy, executor) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.executor = executor
        self.order_type: str = getattr(strategy, "order_type", "stop")
        self._shutdown_requested: bool = False
        self._bars_processed: int = 0

    def run(self) -> Dict[str, Any]:
        """
        Start the per-bar trading loop.

        Returns a status dict with ``bars_processed`` when the generator is
        exhausted (replay / CSV mode) or when ``shutdown()`` is called.
        """
        _LOG.info("V2LiveEngine started — order_type=%s", self.order_type)
        self._bars_processed = 0

        for bar in self.data_handler.get_next_bar():
            if self._shutdown_requested:
                _LOG.info("V2LiveEngine: shutdown requested; exiting loop")
                break

            self._bars_processed += 1
            orders = self.strategy.on_bar(bar)

            for order in orders:
                if self.order_type == "market":
                    self.executor.execute_market_order(order)
                elif self.order_type == "stop":
                    self.executor.submit_stop_order(order)
                else:
                    raise ValueError(f"Unknown order_type: {self.order_type}")

            bar_time = _bar_time_as_utc(bar)
            self.executor.process_pending_batch(bar_time)
            self.executor.process_position_updates_batch(bar_time)
            self.strategy.apply_trailing_stops(bar)

        result = {
            "bars_processed": self._bars_processed,
            "shutdown_requested": self._shutdown_requested,
        }
        _LOG.info("V2LiveEngine stopped — %s", result)
        return result

    def shutdown(self) -> None:
        """
        Request the engine to stop at the next bar boundary and clean up
        executor / strategy resources.
        """
        self._shutdown_requested = True
        if hasattr(self.data_handler, "stop"):
            self.data_handler.stop()
        if hasattr(self.executor, "shutdown"):
            self.executor.shutdown()
        if hasattr(self.strategy, "close"):
            self.strategy.close()
