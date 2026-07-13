"""
feature_spec.py — FeatureSpec

Formal specification of all input features for the transformer model.
Replaces ad-hoc feature computation with a declarative approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class FeatureDef:
    """Definition of a single input feature."""

    name: str                 # Human-readable feature name
    formula: str              # Human-readable formula/description
    lookback: int = 0         # Bars of history needed
    normalization: str = "zscore"  # "none", "zscore", "minmax", "log"


class FeatureSpec:
    """
    Formal specification of input features for the transformer.

    Usage:
        spec = FeatureSpec()
        # Access features: spec.features
        # Validate DataFrame: spec.validate(df)
        # Serialize: spec.to_dict()
        # Generate MQL5 code: spec.to_mql5_include()
    """

    def __init__(self):
        self.features: List[FeatureDef] = []
        self._register_defaults()

    def _register_defaults(self):
        """Register OHLCV + session + normalized price features."""
        # Raw price features
        self.features.append(FeatureDef("log_return", "ln(Close / Close[-1])", lookback=1))
        self.features.append(FeatureDef("O_rel", "(Open - Close) / Close", lookback=0))
        self.features.append(FeatureDef("H_rel", "(High - Close) / Close", lookback=0))
        self.features.append(FeatureDef("L_rel", "(Low - Close) / Close", lookback=0))

        # Session features (cyclic time encoding)
        self.features.append(FeatureDef("hour_sin", "sin(2*pi*hour/24)", normalization="none"))
        self.features.append(FeatureDef("hour_cos", "cos(2*pi*hour/24)", normalization="none"))
        self.features.append(FeatureDef("dow_sin", "sin(2*pi*day_of_week/7)", normalization="none"))
        self.features.append(FeatureDef("dow_cos", "cos(2*pi*day_of_week/7)", normalization="none"))

        # Temporal gap flag (session boundary awareness)
        self.features.append(FeatureDef("has_gap", "1 if time delta > 3x median bar interval else 0", lookback=1, normalization="none"))

        # Volatility features
        self.features.append(FeatureDef("ret_vol_20", "rolling_std(log_return, 20)", lookback=20))
        self.features.append(FeatureDef("ret_vol_60", "rolling_std(log_return, 60)", lookback=60))
        self.features.append(FeatureDef("ret_vol_scaled", "log_return / ret_vol_20", lookback=20))

        # Momentum features
        self.features.append(FeatureDef("mom_5", "Close / Close[-5] - 1", lookback=5))
        self.features.append(FeatureDef("mom_20", "Close / Close[-20] - 1", lookback=20))
        self.features.append(FeatureDef("mom_60", "Close / Close[-60] - 1", lookback=60))

        # Range features
        self.features.append(FeatureDef("hl_range", "(High - Low) / Close", lookback=0))
        self.features.append(FeatureDef("hl_range_20", "avg(hl_range, 20)", lookback=20))

    def add_feature(self, name: str, formula: str, lookback: int = 0,
                    normalization: str = "zscore") -> None:
        """Register a custom feature."""
        self.features.append(FeatureDef(name, formula, lookback, normalization))

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "n_features": len(self.features),
            "total_lookback": max((f.lookback for f in self.features), default=0),
            "features": [
                {
                    "name": f.name,
                    "formula": f.formula,
                    "lookback": f.lookback,
                    "normalization": f.normalization,
                }
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        """Deserialize from dict."""
        spec = cls()
        spec.features = []  # Clear defaults
        for fd in d.get("features", []):
            spec.features.append(FeatureDef(
                name=fd["name"],
                formula=fd.get("formula", ""),
                lookback=fd.get("lookback", 0),
                normalization=fd.get("normalization", "zscore"),
            ))
        return spec

    def to_mql5_include(self) -> str:
        """
        Generate MQL5 include file with feature computation code.

        Returns a .mqh file content as a string.
        """
        lines = [
            "//+------------------------------------------------------------------+",
            "//| TransformerFeatureSpec.mqh                                      |",
            "//| Auto-generated feature computation specification                 |",
            "//+------------------------------------------------------------------+",
            f"//| Total features: {len(self.features):<50} |",
            f"//| Minimum lookback: {max(f.lookback for f in self.features):<50} |",
            "//+------------------------------------------------------------------+",
            "",
            "// Feature array: one row per bar, columns in declared order",
            f"#define N_FEATURES {len(self.features)}",
            "",
        ]

        for i, f in enumerate(self.features):
            lines.append(f"// [{i:03d}] {f.name:<30s} | {f.formula}")

        lines.append("")
        lines.append("// Use in OnCalculate:")
        lines.append("//   double features[];")
        lines.append("//   ArrayResize(features, N_FEATURES);")
        lines.append("//   ComputeFeatures(bar_index, open, high, low, close, features);")
        lines.append("")

        return "\n".join(lines)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Check that all required feature columns can be computed from the DataFrame.

        Returns True if the DataFrame has the minimum required columns.
        """
        required_raw = {"Open", "High", "Low", "Close", "Time"}
        missing = required_raw - set(df.columns)
        if missing:
            print(f"Missing required columns: {missing}")
            return False
        return True

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def total_lookback(self) -> int:
        return max((f.lookback for f in self.features), default=0)
