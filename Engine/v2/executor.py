"""
Engine/v2/executor.py — MT5 live execution handler for the v2 runtime.

Wraps the proven :class:`Engine.Executor.MT5LiveExecutionHandler` and adds the
``shutdown()`` hook required by the v2 engine.
"""
from __future__ import annotations

from typing import Optional

from Engine.Executor import MT5LiveExecutionHandler
from Engine.TicketBook import TicketBook

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover — package absent in non-MT5 environments
    mt5 = None


class V2MT5LiveExecutionHandler(MT5LiveExecutionHandler):
    """
    Live MT5 execution handler for the v2 runtime.

    Inherits the full order lifecycle from :class:`MT5LiveExecutionHandler`
    (market order execution, pending stop-order submission, pending-batch
    processing, and position-update batch processing).  Adds a ``shutdown()``
    method that cleanly closes the MT5 connection.

    Parameters
    ----------
    deviation : int, optional
        Maximum allowed price deviation in points.  Defaults to ``0``.
    magic : int, optional
        Expert Advisor magic number.  Defaults to ``234000``.
    ticket_book : TicketBook, optional
        Shared order-state journal.
    """

    def __init__(
        self,
        deviation: int = 0,
        magic: int = 234000,
        ticket_book: Optional[TicketBook] = None,
    ) -> None:
        super().__init__(
            deviation=deviation,
            magic=magic,
            ticket_book=ticket_book,
        )

    def shutdown(self) -> None:
        """Close the MT5 terminal connection gracefully."""
        try:
            if mt5 is not None:
                mt5.shutdown()
        except Exception:
            pass
