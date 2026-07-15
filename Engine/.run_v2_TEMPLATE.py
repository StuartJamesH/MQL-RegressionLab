"""
Hidden per-symbol launcher template for the v2 transformer runtime.

Copy this file to a git-ignored location (e.g. ``Engine/run_v2_EURUSD.py``),
replace the placeholder values, and run from the repo root:

    .venv/bin/python Engine/run_v2_EURUSD.py

IMPORTANT:
  * ``MAGIC`` must be unique per running instance.
  * ``DB_PATH`` must be unique per running instance.
  * Do not commit copies of this file to version control.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_WORKBENCH = _REPO_ROOT / "ModelWorkbench"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MODEL_WORKBENCH) not in sys.path:
    sys.path.insert(1, str(_MODEL_WORKBENCH))

from Engine.Engine import configure_logging
from Engine.TicketBook import TicketBook
from Engine.v2.config import V2RuntimeConfig
from Engine.v2.data_handler import V2MT5DataHandler
from Engine.v2.engine import V2LiveEngine
from Engine.v2.executor import V2MT5LiveExecutionHandler
from Engine.v2.inference import V2InferenceEngine
from Engine.v2.model_pack import ModelPackLoader
from Engine.v2.strategy import V2RiskConfig, V2SignalStrategy

# ---------------------------------------------------------------------------
# User-configurable values — change these for each symbol instance.
# ---------------------------------------------------------------------------
SYMBOL = "EURUSD"                        # MT5 instrument ticker
MAGIC = 234001                           # Unique EA magic number
PACK_DIR = "ModelWorkbench/ModelPacks/transformers/MY_MODEL"  # v2 pack path
DB_PATH = "Engine/v2/ticketbook_EURUSD.db"  # Unique SQLite journal
TIMEFRAME = "M1"
BACKEND = "onnx"                         # "onnx" or "pytorch"
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    configure_logging(log_file=f"trading_{SYMBOL}.log", cloud_log=True)
    log = logging.getLogger(__name__)

    runtime_config = V2RuntimeConfig(
        symbol=SYMBOL,
        magic=MAGIC,
        pack_dir=PACK_DIR,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        backend=BACKEND,
    )

    log.info("Loading pack: %s", PACK_DIR)
    pack = ModelPackLoader.load(PACK_DIR)

    inference_engine = V2InferenceEngine(pack, backend=BACKEND, device="cpu")
    ticket_book = TicketBook(db_path=DB_PATH, use_memory_only=False)
    executor = V2MT5LiveExecutionHandler(
        deviation=0, magic=MAGIC, ticket_book=ticket_book
    )
    data_handler = V2MT5DataHandler(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        mode="live",
        max_seq_len=pack["config"].max_seq_len,
    )

    risk_config = V2RiskConfig()
    strategy = V2SignalStrategy(
        symbol=SYMBOL,
        pack=pack,
        inference_engine=inference_engine,
        risk_config=risk_config,
        ticket_book=ticket_book,
        mt5_executor=executor,
    )

    engine = V2LiveEngine(data_handler, strategy, executor)

    def _signal_handler(signum, frame):
        log.info("Received signal %s — shutting down", signum)
        engine.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        engine.run()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
