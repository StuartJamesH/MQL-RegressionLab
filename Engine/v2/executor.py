"""
Engine/v2/executor.py — MT5 live execution handler for the v2 runtime.

Wraps the proven :class:`Engine.Executor.MT5LiveExecutionHandler` and adds
trailing-stop management and the ``shutdown()`` hook required by the v2 engine.
"""
from __future__ import annotations

import logging
from typing import Optional

from Engine.Executor import MT5LiveExecutionHandler
from Engine.TicketBook import TicketBook

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover — package absent in non-MT5 environments
    mt5 = None

_LOG = logging.getLogger(__name__)


class V2MT5LiveExecutionHandler(MT5LiveExecutionHandler):
    """
    Live MT5 execution handler for the v2 runtime.

    Inherits the full order lifecycle from :class:`MT5LiveExecutionHandler`
    (market order execution, pending stop-order submission, pending-batch
    processing, and position-update batch processing).  Adds a ``shutdown()``
    method that cleanly closes the MT5 connection and a
    ``modify_position_sl()`` method for trailing-stop updates.

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

    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """
        Send an SL-modification request to MT5 for an open position.

        Parameters
        ----------
        ticket : int
            MT5 position ticket.
        new_sl : float
            New stop-loss price (must be on the correct side of current price).

        Returns
        -------
        bool
            ``True`` if the modification was accepted by the terminal.
        """
        if mt5 is None:
            _LOG.warning("MT5 not available — cannot modify SL for ticket=%d", ticket)
            return False

        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            _LOG.warning("Position %d not found in MT5 — skipping SL update", ticket)
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            _LOG.warning(
                "SL modification failed for ticket=%d: retcode=%s error=%s",
                ticket,
                getattr(result, "retcode", None) if result else None,
                mt5.last_error(),
            )
            return False

        # Update the journal in-memory and in SQLite.
        if self.ticket_book is not None and ticket in self.ticket_book._tickets:
            record = self.ticket_book._tickets[ticket]
            record.sl = new_sl
            try:
                self.ticket_book._update_in_db(record)
            except Exception:
                pass

        _LOG.info("Trailing SL updated: ticket=%d new_sl=%.5f", ticket, new_sl)
        return True

    def shutdown(self) -> None:
        """Close the MT5 terminal connection gracefully."""
        try:
            if mt5 is not None:
                mt5.shutdown()
        except Exception:
            pass
