---
goal: Live Trading Runtime for Learn/v2 Transformer Model Packs
version: 1.0
date_created: 2026-07-13
last_updated: 2026-07-13
owner: MQL-RegressionLab
status: 'Planned'
tags: [feature, live-trading, mt5, v2, transformer, onnx, deployment]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan designs and implements a self-contained live trading runtime that consumes `ModelWorkbench/Learn/v2` transformer model packs, ingests real-time OHLCV bars from a local MetaTrader 5 (MT5) terminal via the official `MetaTrader5` Python API, and emits managed trading signals. The runtime must be reproducible on a fresh instance: given only a model pack directory and an MT5 connection, it can reconstruct the inference pipeline and begin producing signals without retraining or access to the original training data.

The architecture reuses the proven separation of concerns from `Engine/` (DataHandler → Strategy → Executor → TicketBook) but replaces the legacy LSTM/TCN/PyTorch inference path with the v2 Causal Patch Transformer pipeline, using ONNX Runtime as the primary inference backend to stay aligned with the MQL5 deployment path.

## 1. Requirements & Constraints

- **REQ-001**: The runtime must load a v2 model pack produced by `ModelWorkbench/Learn/v2/deploy.py` and reconstruct the inference pipeline without the original training dataset.
- **REQ-002**: The runtime must ingest live bars from MT5 using the `MetaTrader5` Python package and yield each completed bar exactly once.
- **REQ-003**: The runtime must transform model outputs (distribution, direction, volatility, regime) into scalar trade signals in `[-1, 1]` and emit `Order` objects for execution.
- **REQ-004**: The runtime must support both ONNX Runtime inference (primary) and PyTorch fallback inference from `model.pt`.
- **REQ-005**: The runtime must enforce risk limits, compute SL/TP levels, and size positions using the v2 `RiskManager` and `KellyPositionSizer` semantics.
- **REQ-006**: The runtime must submit orders to MT5, track pending/open state via `TicketBook`, and detect fills/closures through the per-bar lifecycle batch.
- **REQ-007**: The runtime must be configurable per symbol/instance with unique `MAGIC` numbers and isolated SQLite journal paths.
- **REQ-008**: The runtime must support graceful shutdown on `SIGINT`/`SIGTERM`, closing log files and calling `mt5.shutdown()`.
- **SEC-001**: No strategy code may call the MT5 API directly; all broker interaction is centralized in the execution handler.
- **CON-001**: Model packs are immutable deployment artifacts; the runtime must never modify pack contents.
- **CON-002**: Feature normalization is causal only; live normalization must use the same log-ratio and volume scaling logic as `Learn.v2.data.normalize_ohlcv`.
- **CON-003**: The runtime must operate on completed bars only; no intra-bar tick inference is required.
- **CON-004**: The runtime must run from the repo root using the existing `.venv`.
- **GUD-001**: Reuse `Engine/DataHandler.py`, `Engine/Executor.py`, `Engine/TicketBook.py`, and `Engine/Engine.py` wherever their interfaces match the v2 requirements.
- **GUD-002**: Keep the v2 runtime in a dedicated `Engine/v2/` package so v1 and v2 engines can coexist during migration.
- **PAT-001**: The bar loop follows the established pattern: `get_next_bar()` → `strategy.on_bar(bar)` → submit orders → `process_pending_batch()` → `process_position_updates_batch()`.

## 2. Implementation Steps

### Implementation Phase 1 — Foundation & Reuse Audit

- GOAL-001: Audit existing `Engine/` components and establish the v2 runtime foundation.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Verify that `Engine/DataHandler.py::Order`, `Engine/Executor.py::MT5LiveExecutionHandler`, `Engine/TicketBook.py::TicketBook`, and `Engine/Engine.py::Live_Engine` can be reused with a v2 strategy. Record any interface mismatches in this plan's notes. |  |  |
| TASK-002 | Confirm the v2 model pack schema by reading `ModelWorkbench/Learn/v2/deploy.py`: required files are `model.onnx`, `config.json`, `normalizer.json`, `feature_spec.json`, `model_info.json`, and optional `model.pt`. |  |  |
| TASK-003 | Ensure `MetaTrader5` and `onnxruntime` are listed in `requirements.txt`; if missing, add them with minimum versions `MetaTrader5>=5.0.45` and `onnxruntime>=1.16.0`. |  |  |
| TASK-004 | Create the `Engine/v2/` package directory and add an empty `__init__.py`. |  |  |

### Implementation Phase 2 — Model Pack Loader & Inference Engine

- GOAL-002: Implement a self-contained v2 model pack loader and inference engine that can reconstruct the model from pack artifacts alone.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Create `Engine/v2/model_pack.py` containing `class ModelPackLoader` with method `load(pack_dir: str) -> dict`. The loader must read `config.json`, `normalizer.json`, `feature_spec.json`, `model_info.json`, and `model.pt`; return a dict with keys `config` (`Learn.v2.model.config.ModelConfig`), `normalizer`, `feature_spec` (`Learn.v2.feature_spec.FeatureSpec`), `model_info`, and `pytorch_state_dict`. |  |  |
| TASK-006 | Create `Engine/v2/inference.py` containing `class V2InferenceEngine` with `__init__(self, pack: dict, backend: str = "onnx", device: str = "cpu")`. The class must initialize an ONNX InferenceSession from `model.onnx` when `backend == "onnx"`, or reconstruct `TradeForecastTransformer` from `config` and load `pytorch_state_dict` when `backend == "pytorch"`. |  |  |
| TASK-007 | Implement `V2InferenceEngine.predict(x_raw: np.ndarray, x_session: np.ndarray) -> dict` returning keys `mu`, `log_sigma`, `direction`, `volatility`, `regime`, `quantiles` with shapes `(1, n_horizons)` for vector outputs. The ONNX path must feed `{"x_raw": x_raw, "x_session": x_session}` and map output names; the PyTorch path must call `model.forward_features` is not used — call full `forward` and unwrap `ModelOutput`. |  |  |
| TASK-008 | Add a parity helper `Engine/v2/inference.py::compare_backends(pack_dir: str, sample_csv: str, atol: float = 1e-4) -> bool` that runs the same input through ONNX and PyTorch and asserts max absolute difference is within tolerance. |  |  |
| TASK-009 | Create `Engine/v2/features.py` exporting `normalize_live_ohlcv(df: pd.DataFrame) -> np.ndarray` and `encode_live_session_features(timestamps: pd.DatetimeIndex) -> np.ndarray`. Both must reuse `Learn.v2.data.normalize_ohlcv` and `Learn.v2.data.SessionFeatureEncoder.encode(include_gap=True)` respectively. |  |  |

### Implementation Phase 3 — Signal Generation, Risk, and Order Construction

- GOAL-003: Integrate v2 distributional signal generation with risk management and produce `Order` objects compatible with the executor.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Create `Engine/v2/strategy.py` containing `class V2SignalStrategy` with constructor signature `(symbol: str, pack: dict, inference_engine: V2InferenceEngine, risk_config: RiskConfig, ticket_book: TicketBook, mt5_executor: Any, **kwargs)`. |  |  |
| TASK-011 | Implement `V2SignalStrategy._build_input_tensor()` that constructs `x_raw` of shape `(1, max_seq_len, 5)` and `x_session` of shape `(1, max_seq_len, 5)` from the internal ring buffer, using `Engine/v2/features.py` for normalization and session encoding. |  |  |
| TASK-012 | Port the signal logic from `Learn.v2.signals.DistributionalSignalGenerator` into `V2SignalStrategy._signal_from_outputs(outputs: dict) -> float`, using primary horizon index 2 (20 bars), default `temperature=1.0`, default `signal_threshold=0.1`, and extreme regime class 3. |  |  |
| TASK-013 | Integrate `Learn.v2.risk_manager.RiskManager` to compute SL/TP/trailing-stop levels and `Learn.v2.position_sizing.KellyPositionSizer` to compute position size in account-currency units; translate currency units to lots using `mt5_executor.get_point_value(symbol)`. |  |  |
| TASK-014 | Implement `V2SignalStrategy.on_bar(bar) -> list[Order]` that appends the bar to ring buffers, skips inference when `ticket_book.has_pending_order(symbol)` or `ticket_book.has_open_position(symbol)` is true, and returns a single `Engine.DataHandler.Order` (side `'buy'` or `'sell'`, entry as stop price, SL/TP, expiry `bar_time + timedelta(minutes=patience)`) when `abs(signal) >= signal_threshold`. |  |  |
| TASK-015 | Add CSV trade logging to `Engine/v2/strategy.py` writing to `Engine/v2/Trade Logs/<symbol>_<model_name>_<timestamp>.csv` with columns: `bar_time`, `open`, `high`, `low`, `close`, `volume`, `signal`, `mu_h2`, `sigma_h2`, `direction_prob`, `regime`, `side`, `entry`, `stop`, `take`, `position_size_lots`, `pending_order`, `open_position`, `action_taken`. |  |  |

### Implementation Phase 4 — Live Data Feed, Execution, and Orchestration

- GOAL-004: Wire the live MT5 data feed and execution layer into the v2 runtime.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-016 | Create `Engine/v2/data_handler.py` containing `class V2MT5DataHandler` with constructor `(symbol: str, timeframe: str = "M1", poll_interval: float = 1.0, history_bars: int = 1024)` and generator `get_next_bar()` yielding completed bars as namedtuples with fields `Time`, `Open`, `High`, `Low`, `Close`, `Volume`. On construction, fetch at least `max_seq_len + 100` historical bars to warm the strategy buffer. |  |  |
| TASK-017 | Create `Engine/v2/executor.py` containing `class V2MT5LiveExecutionHandler`. Reuse `Engine/Executor.py::MT5LiveExecutionHandler` by inheritance or composition; ensure it supports `submit_stop_order`, `execute_market_order`, `process_pending_batch`, and `process_position_updates_batch` with identical signatures. |  |  |
| TASK-018 | Create `Engine/v2/engine.py` containing `class V2LiveEngine` with constructor `(data_handler, strategy, executor)` and `run()` method implementing the per-bar loop: fetch bar → `strategy.on_bar(bar)` → route orders by `strategy.order_type` → `executor.process_pending_batch(bar_time)` → `executor.process_position_updates_batch(bar_time)`. |  |  |
| TASK-019 | Add `V2LiveEngine.shutdown()` that calls `executor.shutdown()` (which calls `mt5.shutdown()`) and closes strategy log files. Register `signal.signal(signal.SIGINT, ...)` and `signal.signal(signal.SIGTERM, ...)` in the launcher to invoke `shutdown()` and exit cleanly. |  |  |
| TASK-020 | Implement replay mode in `V2MT5DataHandler` by accepting `mode: str = "live"` and optional `start`/`end` datetimes; when `mode == "replay"`, fetch the range once and yield bars sequentially for walk-forward validation. |  |  |

### Implementation Phase 5 — Launcher, Configuration, and Documentation

- GOAL-005: Provide runnable launchers and configuration templates for one or more symbol instances.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-021 | Create `Engine/v2/config.py` containing `dataclass V2RuntimeConfig` with fields: `symbol: str`, `magic: int`, `pack_dir: str`, `db_path: str`, `timeframe: str = "M1"`, `risk_per_trade: float = 50.0`, `max_position_lots: float = 0.5`, `signal_threshold: float = 0.1`, `patience_bars: int = 5`, `backend: str = "onnx"`, `device: str = "cpu"`, `log_dir: str = "Engine/v2/Trade Logs"`. |  |  |
| TASK-022 | Create `Engine/run_v2.py` launcher that: loads `.env`, parses CLI args (`--symbol`, `--pack-dir`, `--magic`, `--db-path`, `--risk`, `--patience`, `--backend`), constructs `V2RuntimeConfig`, wires `V2MT5DataHandler`, `TicketBook`, `V2MT5LiveExecutionHandler`, `V2InferenceEngine`, `V2SignalStrategy`, and `V2LiveEngine`, then calls `engine.run()`. |  |  |
| TASK-023 | Add hidden per-symbol launcher template `Engine/.run_v2_TEMPLATE.py` showing unique `MAGIC` and `DB_PATH` placeholders, and document that production launchers are git-ignored runtime configuration. |  |  |
| TASK-024 | Create `Engine/v2/README.md` documenting: directory layout, how to train a pack, how to launch live trading, how to run replay validation, and the requirement that `MAGIC`/`DB_PATH` must be unique per instance. |  |  |
| TASK-025 | Add `Engine/v2/.gitignore` entries for `Trade Logs/*.csv` and `ticketbook_*.db`. |  |  |

### Implementation Phase 6 — Validation, Tests, and Hardening

- GOAL-006: Validate the end-to-end signal path and harden the runtime for production use.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-026 | Create `Engine/tests/test_v2_loader.py` with unit tests for `ModelPackLoader.load` verifying all required keys exist and `ModelConfig` is reconstructed correctly. |  |  |
| TASK-027 | Create `Engine/tests/test_v2_inference.py` with tests for `V2InferenceEngine.predict` shape assertions and the ONNX/PyTorch parity check. |  |  |
| TASK-028 | Create `Engine/tests/test_v2_strategy.py` with tests for `V2SignalStrategy.on_bar` returning empty lists during warm-up and producing a valid `Order` when signal threshold is exceeded. |  |  |
| TASK-029 | Add an integration smoke test `Engine/tests/test_v2_replay.py` that runs `V2MT5DataHandler` in replay mode (or a CSV-backed mock) through `V2LiveEngine` with `TicketBook(use_memory_only=True)` and confirms no exceptions and correct bar count. |  |  |
| TASK-030 | Implement runtime health logging: log buffer length, last inference latency, last signal value, and last error every 100 bars to `trading.log`. |  |  |
| TASK-031 | Add defensive checks in `V2SignalStrategy.on_bar`: if `x_raw`/`x_session` contains NaN/Inf, skip inference and log a warning; if `mt5_executor.get_point_value` fails, fall back to `point_value=1.0` and log a sizing warning. |  |  |

## 3. Alternatives

- **ALT-001**: Embed the v2 runtime inside the existing `Engine/Strategy.py` rather than creating `Engine/v2/`. This was rejected because the v2 pipeline (patch embeddings, distribution heads, ONNX export, log-ratio normalization) is structurally different from the legacy `model_pack` dict expected by `TripleBarrierHiLowMulticlass`; a dedicated package prevents interface collisions and allows v1 and v2 engines to run side by side during validation.
- **ALT-002**: Use PyTorch as the default inference backend instead of ONNX Runtime. This was rejected because the v2 deployment path already exports `model.onnx` for MQL5; using ONNX in Python aligns the Python runtime with the MQL5 runtime and avoids the larger PyTorch memory footprint in production.
- **ALT-003**: Implement a new order-state machine instead of reusing `Engine/TicketBook.py`. This was rejected because `TicketBook` is already battle-tested, persists to SQLite, and provides the exact `has_pending_order`/`has_open_position` interface the strategy needs.

## 4. Dependencies

- **DEP-001**: `MetaTrader5` Python package must be installed and a local MT5 terminal must be running with the target symbol visible in Market Watch.
- **DEP-002**: `onnxruntime` must be installed for ONNX inference; PyTorch is required only for the optional PyTorch fallback and parity tests.
- **DEP-003**: The v2 model pack directory must contain `model.onnx`, `config.json`, `normalizer.json`, `feature_spec.json`, and `model_info.json`; `model.pt` is required only for the PyTorch backend.
- **DEP-004**: `Learn.v2` modules (`model.config`, `model.full_model`, `data`, `signals`, `risk_manager`, `position_sizing`, `feature_spec`) must be importable; scripts run from repo root with `ModelWorkbench` on `sys.path` or via `PYTHONPATH`.
- **DEP-005**: The existing `Engine/DataHandler.py::Order` dataclass, `Engine/Executor.py::MT5LiveExecutionHandler`, `Engine/TicketBook.py::TicketBook`, and `Engine/Engine.py::Live_Engine` must remain stable; any breaking changes to those files must be mirrored in `Engine/v2/`.

## 5. Files

- **FILE-001**: `Engine/v2/__init__.py` — package marker.
- **FILE-002**: `Engine/v2/model_pack.py` — `ModelPackLoader` for reading v2 deployment artifacts.
- **FILE-003**: `Engine/v2/inference.py` — `V2InferenceEngine` with ONNX Runtime and PyTorch backends.
- **FILE-004**: `Engine/v2/features.py` — causal live feature normalization and session encoding wrappers.
- **FILE-005**: `Engine/v2/strategy.py` — `V2SignalStrategy` signal generator, risk integration, and order construction.
- **FILE-006**: `Engine/v2/data_handler.py` — `V2MT5DataHandler` live/replay bar feed.
- **FILE-007**: `Engine/v2/executor.py` — `V2MT5LiveExecutionHandler` MT5 order lifecycle wrapper.
- **FILE-008**: `Engine/v2/engine.py` — `V2LiveEngine` per-bar orchestrator.
- **FILE-009**: `Engine/v2/config.py` — `V2RuntimeConfig` dataclass.
- **FILE-010**: `Engine/run_v2.py` — main launcher script.
- **FILE-011**: `Engine/.run_v2_TEMPLATE.py` — template for per-symbol hidden launchers.
- **FILE-012**: `Engine/v2/README.md` — runtime documentation.

## 6. Testing

- **TEST-001**: `Engine/tests/test_v2_loader.py` — assert `ModelPackLoader.load(pack_dir)` returns a dict with keys `config`, `normalizer`, `feature_spec`, `model_info`, `pytorch_state_dict`; assert `config.max_seq_len` matches `config.json`.
- **TEST-002**: `Engine/tests/test_v2_inference.py` — assert `V2InferenceEngine.predict` returns `mu` of shape `(1, 6)` and `direction` of shape `(1, 6)`; assert ONNX and PyTorch outputs match within `atol=1e-4`.
- **TEST-003**: `Engine/tests/test_v2_strategy.py` — assert `V2SignalStrategy.on_bar` returns `[]` until the ring buffer holds at least `max_seq_len` bars; assert a bar producing `signal >= threshold` returns a single `Order` with `side`, `entry`, `sl`, `tp`, and `qty` populated.
- **TEST-004**: `Engine/tests/test_v2_replay.py` — run `V2LiveEngine` over a CSV-backed `DataHandler` (or replay-mode handler with a fixed date range) and assert the engine processes the expected number of bars without raising.
- **TEST-005**: Manual integration test — launch `Engine/run_v2.py` against a live MT5 terminal in replay mode first, verify trade log CSV contains expected columns and no duplicate bars, then switch to live mode with minimum lot size for a supervised live session.

## 7. Risks & Assumptions

- **RISK-001**: ONNX Runtime provider availability varies by platform (CPU vs CUDA vs DirectML); the default `CPUExecutionProvider` is safe but may be slower than PyTorch on GPU. Mitigation: expose `--backend pytorch` for GPU inference and benchmark both.
- **RISK-002**: Live MT5 timezone offsets and bar-time alignment can cause duplicate or skipped bars if the data handler does not track `last_yield_time` correctly. Mitigation: reuse the timestamp-dedup logic from `Engine/DataHandler.py::MT5DataHandler` and log warnings on jumps.
- **RISK-003**: The v2 model expects a fixed `max_seq_len` (default 512); if MT5 returns fewer historical bars on startup, the strategy will silently warm up and begin inference only when the buffer is full, which may delay the first signal. Mitigation: fetch `max_seq_len + 100` bars on handler construction and log warm-up progress.
- **RISK-004**: `MetaTrader5` Python API can return `None` for `symbol_info_tick` or `orders_get` during brief disconnections; the executor must handle `None` without crashing. Mitigation: wrap all MT5 calls in try/except and retry once after a 1-second sleep.
- **ASSUMPTION-001**: The MT5 account supports the target symbol, hedging is allowed, and lot sizes comply with the broker's `volume_min`/`volume_max`/`volume_step`; the runtime uses `symbol_info.volume_min` to clamp `qty`.
- **ASSUMPTION-002**: The v2 model pack was trained with `max_seq_len`, `n_horizons`, and input channels identical to the live configuration; mismatches are detected at load time and raise `ValueError`.
- **ASSUMPTION-003**: The operator runs the launcher from the repo root with the virtual environment activated so `from Learn.v2...` and `from Engine...` imports resolve correctly.

## 8. Related Specifications / Further Reading

- `Engine/ARCHITECTURE.md` — component diagram and order-lifecycle state machine for the legacy engine.
- `Engine/README.md` — detailed deep-dive on `Live_Engine`, `MT5DataHandler`, `MT5LiveExecutionHandler`, and `TicketBook`.
- `ModelWorkbench/Learn/v2/README.md` — v2 transformer architecture, training entry point, and deployment pack contents.
- `ModelWorkbench/Learn/v2/deploy.py` — deployment packaging logic and expected model pack file schema.
- `ModelWorkbench/Learn/v2/signals.py` — `DistributionalSignalGenerator` reference implementation.
- `ModelWorkbench/Learn/v2/risk_manager.py` — `RiskConfig` and `RiskManager` reference implementation.
- `ModelWorkbench/Learn/v2/position_sizing.py` — `KellyPositionSizer` reference implementation.
- `ModelWorkbench/Learn/v2/data.py` — `normalize_ohlcv` and `SessionFeatureEncoder` reference implementation.
