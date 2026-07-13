"""
Engine/run_v2.py — Launcher for the v2 transformer live trading runtime.

Run from the repo root with the virtual environment activated:

    .venv/bin/python Engine/run_v2.py \
        --symbol EURUSD \
        --pack-dir ModelWorkbench/ModelPacks/transformers/my_model \
        --magic 234001 \
        --db-path Engine/v2/ticketbook_EURUSD.db \
        --backend onnx

Replay mode (no live order submission; useful for validation):

    .venv/bin/python Engine/run_v2.py \
        --symbol EURUSD \
        --pack-dir ModelWorkbench/ModelPacks/transformers/my_model \
        --magic 234001 \
        --db-path Engine/v2/ticketbook_EURUSD.db \
        --mode replay \
        --start "2026-06-01" \
        --end "2026-06-02"
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure ModelWorkbench is importable when launching from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_WORKBENCH = _REPO_ROOT / "ModelWorkbench"
if str(_MODEL_WORKBENCH) not in sys.path:
    sys.path.insert(0, str(_MODEL_WORKBENCH))

from Engine.Engine import configure_logging
from Engine.TicketBook import TicketBook
from Engine.v2.config import V2RuntimeConfig
from Engine.v2.data_handler import V2CSVDataHandler, V2MT5DataHandler
from Engine.v2.engine import V2LiveEngine
from Engine.v2.executor import V2MT5LiveExecutionHandler
from Engine.v2.inference import V2InferenceEngine
from Engine.v2.model_pack import ModelPackLoader
from Engine.v2.strategy import V2RiskConfig, V2SignalStrategy

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the v2 transformer live trading runtime."
    )
    parser.add_argument("--symbol", required=True, help="MT5 instrument ticker")
    parser.add_argument(
        "--pack-dir", required=True, help="Path to v2 model pack directory"
    )
    parser.add_argument(
        "--magic", type=int, required=True, help="Unique EA magic number"
    )
    parser.add_argument(
        "--db-path", required=True, help="Unique SQLite journal path"
    )
    parser.add_argument("--timeframe", default="M1", help="Bar timeframe")
    parser.add_argument(
        "--risk", type=float, default=50.0, help="Max account-currency risk per trade"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Pending-order expiry in bars/minutes"
    )
    parser.add_argument(
        "--backend", default="onnx", choices=["onnx", "pytorch"], help="Inference backend"
    )
    parser.add_argument("--device", default="cpu", help="Inference device")
    parser.add_argument(
        "--mode", default="live", choices=["live", "replay"], help="Feed mode"
    )
    parser.add_argument("--start", default=None, help="Replay start (ISO-8601)")
    parser.add_argument("--end", default=None, help="Replay end (ISO-8601)")
    parser.add_argument(
        "--csv", default=None, help="CSV path for CSV replay (skips MT5)"
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.1,
        help="Minimum |signal| to emit a trade",
    )
    parser.add_argument(
        "--max-position-lots",
        type=float,
        default=0.5,
        help="Hard cap on position size in lots",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    configure_logging(log_file="trading.log", cloud_log=True)

    args = parse_args()

    runtime_config = V2RuntimeConfig(
        symbol=args.symbol,
        magic=args.magic,
        pack_dir=args.pack_dir,
        db_path=args.db_path,
        timeframe=args.timeframe,
        risk_per_trade=args.risk,
        max_position_lots=args.max_position_lots,
        signal_threshold=args.signal_threshold,
        patience_bars=args.patience,
        backend=args.backend,
        device=args.device,
    )

    _LOG.info("Loading model pack: %s", runtime_config.pack_dir)
    pack = ModelPackLoader.load(runtime_config.pack_dir)

    _LOG.info(
        "Initialising inference — backend=%s device=%s model=%s",
        runtime_config.backend,
        runtime_config.device,
        pack["model_info"].get("model_name", "unknown"),
    )
    inference_engine = V2InferenceEngine(
        pack,
        backend=runtime_config.backend,
        device=runtime_config.device,
    )

    ticket_book = TicketBook(
        db_path=runtime_config.db_path,
        use_memory_only=False,
    )

    executor = V2MT5LiveExecutionHandler(
        deviation=0,
        magic=runtime_config.magic,
        ticket_book=ticket_book,
    )

    if args.csv is not None:
        _LOG.info("CSV replay mode: %s", args.csv)
        data_handler = V2CSVDataHandler(
            csv_path=args.csv,
            max_bars=pack["config"].max_seq_len + 500,
        )
    elif args.mode == "replay":
        _LOG.info("MT5 replay mode: %s %s to %s", args.symbol, args.start, args.end)
        data_handler = V2MT5DataHandler(
            symbol=runtime_config.symbol,
            timeframe=runtime_config.timeframe,
            mode="replay",
            max_seq_len=pack["config"].max_seq_len,
            start=args.start,
            end=args.end,
        )
    else:
        _LOG.info("Live mode: %s %s", args.symbol, runtime_config.timeframe)
        data_handler = V2MT5DataHandler(
            symbol=runtime_config.symbol,
            timeframe=runtime_config.timeframe,
            mode="live",
            max_seq_len=pack["config"].max_seq_len,
        )

    risk_config = V2RiskConfig(
        risk_per_trade=runtime_config.risk_per_trade,
        max_position_lots=runtime_config.max_position_lots,
        account_equity=runtime_config.account_equity,
        signal_threshold=runtime_config.signal_threshold,
        patience_bars=runtime_config.patience_bars,
        temperature=runtime_config.temperature,
        primary_horizon_idx=runtime_config.primary_horizon_idx,
        extreme_regime_idx=runtime_config.extreme_regime_idx,
        order_type=runtime_config.order_type,
    )

    strategy = V2SignalStrategy(
        symbol=runtime_config.symbol,
        pack=pack,
        inference_engine=inference_engine,
        risk_config=risk_config,
        ticket_book=ticket_book,
        mt5_executor=executor,
    )

    engine = V2LiveEngine(data_handler, strategy, executor)

    # Graceful shutdown on SIGINT / SIGTERM.
    def _signal_handler(signum, frame):
        _LOG.info("Received signal %s — requesting graceful shutdown", signum)
        engine.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        engine.run()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
