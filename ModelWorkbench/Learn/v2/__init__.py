"""
Learn/v2 — Next-Generation Profitable Trading Model.

Causal Patch Transformer with distributional regression, three-phase curriculum
training (self-supervised pre-training → multi-task fine-tuning → RL optimization),
and MQL5 ONNX deployment.

Subpackages:
    model/      — Model architecture (config, embedding, transformer, heads, full model)
    training/   — Training pipelines (pretrain, finetune, rl_finetune, losses, metrics)
"""

from Learn.v2.model.config import ModelConfig
from Learn.v2.model.full_model import TradeForecastTransformer, ModelOutput
from Learn.v2.labels import (
    compute_forward_excursion_surface,
    compute_directional_return_distribution,
    compute_optimal_exit_labels,
    compute_volatility_regime_labels,
    LabelStore,
)
from Learn.v2.data import normalize_ohlcv, create_sliding_windows, SessionFeatureEncoder
from Learn.v2.signals import DistributionalSignalGenerator
from Learn.v2.position_sizing import KellyPositionSizer
from Learn.v2.risk_manager import RiskManager, RiskConfig
from Learn.v2.signal_evaluator import SignalEvaluator
from Learn.v2.backtest import VectorizedBacktester, Trade
from Learn.v2.backtest_metrics import BacktestMetrics
from Learn.v2.walk_forward_backtest import WalkForwardBacktest
from Learn.v2.deploy import DeploymentPackager
from Learn.v2.feature_spec import FeatureSpec, FeatureDef
from Learn.v2.parity_check import check_python_mql5_parity, export_test_bars

__all__ = [
    "ModelConfig",
    "TradeForecastTransformer",
    "ModelOutput",
    "compute_forward_excursion_surface",
    "compute_directional_return_distribution",
    "compute_optimal_exit_labels",
    "compute_volatility_regime_labels",
    "LabelStore",
    "normalize_ohlcv",
    "create_sliding_windows",
    "SessionFeatureEncoder",
    "DistributionalSignalGenerator",
    "KellyPositionSizer",
    "RiskManager",
    "RiskConfig",
    "SignalEvaluator",
    "VectorizedBacktester",
    "Trade",
    "BacktestMetrics",
    "WalkForwardBacktest",
    "DeploymentPackager",
    "FeatureSpec",
    "FeatureDef",
    "check_python_mql5_parity",
    "export_test_bars",
]
