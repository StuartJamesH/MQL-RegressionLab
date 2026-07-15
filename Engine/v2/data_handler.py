"""
Engine/v2/data_handler.py — Live and replay bar feeds for the v2 runtime.

Provides ``V2MT5DataHandler``, which is the production MT5-backed feed, and a
lightweight ``V2CSVDataHandler`` for offline replay testing when MT5 is not
available.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Iterator, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover — package absent in non-MT5 environments
    mt5 = None

from Engine.DataHandler import MT5DataHandler

_LOG = logging.getLogger(__name__)


class V2MT5DataHandler(MT5DataHandler):
    """
    MT5-backed bar feed for the v2 runtime.

    Inherits the robust live/replay logic from :class:`Engine.DataHandler.MT5DataHandler`
    and adds a warm-up fetch of at least ``max_seq_len + 100`` historical bars on
    construction so the strategy buffer is ready as soon as the live loop begins.

    Parameters
    ----------
    symbol : str
        MT5 instrument ticker.
    timeframe : str, optional
        Timeframe string (``'M1'``, ``'M5'``, etc.).  Defaults to ``'M1'``.
    mode : str, optional
        ``'live'`` or ``'replay'``.  Defaults to ``'live'``.
    poll_interval : float, optional
        Seconds between MT5 polls in live mode.  Defaults to ``1.0``.
    history_bars : int, optional
        Number of bars to fetch for warm-up.  Defaults to ``1024``.
    max_seq_len : int, optional
        Model sequence length; warm-up fetch is at least ``max_seq_len + 100``.
        Defaults to ``512``.
    start : str, optional
        Replay start (ISO-8601).  Required for ``mode='replay'``.
    end : str, optional
        Replay end (ISO-8601).  Defaults to now.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "M1",
        mode: str = "live",
        poll_interval: float = 1.0,
        history_bars: int = 1024,
        max_seq_len: int = 512,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        # Ensure we fetch enough history for the strategy buffer.
        self.poll_interval = poll_interval
        self.history_bars = max(history_bars, max_seq_len + 100)
        self.max_seq_len = max_seq_len

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            mode=mode,
            start=start,
            end=end,
            max_bars=self.history_bars,
            poll_interval=self.poll_interval,
        )

        # In live mode, perform the warm-up fetch immediately so the strategy
        # has historical context before the first yielded bar.
        if self.mode == "live":
            self._warm_up_bars = self._fetch_latest_bars(self.history_bars)
            _LOG.info(
                "V2MT5DataHandler warm-up fetched %d bars for %s",
                len(self._warm_up_bars),
                self.symbol,
            )
        else:
            self._warm_up_bars = pd.DataFrame()

    def _fetch_latest_bars(self, count: int) -> pd.DataFrame:
        """Fetch the most recent ``count`` bars from MT5 as a DataFrame."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not available")

        rates = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(
                columns=["Time", "Open", "High", "Low", "Close", "Volume"]
            )

        df = pd.DataFrame(rates)
        df["Time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
            },
            inplace=True,
        )
        return df[["Time", "Open", "High", "Low", "Close", "Volume"]].sort_values(
            "Time"
        )

    def get_next_bar(self) -> Iterator:
        """
        Yield completed bars.

        In replay mode this replays the historical range loaded by the parent
        class.  In live mode the parent generator is used unchanged; the warm-up
        bars are stored on ``self._warm_up_bars`` for the strategy to consume on
        startup if desired.
        """
        yield from super().get_next_bar()


class V2CSVDataHandler:
    """
    CSV-backed replay handler for offline v2 testing.

    Mirrors the ``get_next_bar()`` interface of :class:`V2MT5DataHandler` and
    yields bars as namedtuples with fields ``Time``, ``Open``, ``High``,
    ``Low``, ``Close``, ``Volume``.

    Parameters
    ----------
    csv_path : str
        Path to a CSV with columns ``Time``, ``Open``, ``High``, ``Low``,
        ``Close``, ``Volume``.
    time_col : str, optional
        Name of the timestamp column.  Defaults to ``'Time'``.
    max_bars : int, optional
        If set, only replay the last ``max_bars`` rows.
    """

    def __init__(
        self,
        csv_path: str,
        time_col: str = "Time",
        max_bars: Optional[int] = None,
    ) -> None:
        df = pd.read_csv(csv_path, parse_dates=[time_col])
        df.rename(columns={"Date": "Time"}, errors="ignore", inplace=True)
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df[["Time", "Open", "High", "Low", "Close", "Volume"]]
        if max_bars is not None:
            df = df.iloc[-max_bars:].reset_index(drop=True)
        self.data = df

    def get_next_bar(self) -> Iterator:
        for row in self.data.itertuples():
            yield row
