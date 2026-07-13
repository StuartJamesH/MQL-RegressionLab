"""
parity_check.py — Python vs MQL5 ONNX inference parity checking.

Ensures that the ONNX model deployed to MQL5 produces outputs identical
to the PyTorch model within floating-point tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def check_python_mql5_parity(
    python_outputs: dict,
    mql5_outputs: dict,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare Python and MQL5 ONNX inference outputs.

    Args:
        python_outputs: Dict of head_name → numpy array from PyTorch/ONNX.
        mql5_outputs: Dict of head_name → numpy array from MQL5 test output.
        tolerance: Maximum allowed absolute difference.

    Returns:
        True if all outputs match within tolerance.
    """
    common_keys = set(python_outputs.keys()) & set(mql5_outputs.keys())
    if not common_keys:
        print("ERROR: No common keys between Python and MQL5 outputs.")
        return False

    all_ok = True
    print(f"{'Head':<25s} {'Max Abs Diff':>15s} {'Status':>10s}")
    print("-" * 52)

    for key in sorted(common_keys):
        py_val = np.asarray(python_outputs[key], dtype=np.float64)
        mq_val = np.asarray(mql5_outputs[key], dtype=np.float64)

        if py_val.shape != mq_val.shape:
            print(f"{key:<25s} {'shape mismatch':>15s} {'FAIL':>10s}")
            print(f"  Python shape: {py_val.shape}, MQL5 shape: {mq_val.shape}")
            all_ok = False
            continue

        max_diff = float(np.max(np.abs(py_val - mq_val)))
        status = "OK" if max_diff <= tolerance else "FAIL"
        if status != "OK":
            all_ok = False

        print(f"{key:<25s} {max_diff:>15.8f} {status:>10s}")

    if all_ok:
        print("\n✓ All outputs match within tolerance.")
    else:
        print(f"\n✗ Some outputs exceed tolerance of {tolerance}.")

    return all_ok


def export_test_bars(
    df: pd.DataFrame,
    output_path: str,
    n_bars: int = 100,
    feature_cols: Optional[list] = None,
) -> str:
    """
    Export a small subset of bars as CSV for MQL5 testing.

    Args:
        df: OHLCV DataFrame with feature columns.
        output_path: Path for the output CSV file.
        n_bars: Number of bars to export (most recent).
        feature_cols: Specific columns to export. Default: OHLCV + all numeric.

    Returns:
        Absolute path to the exported CSV.
    """
    if feature_cols is None:
        ohlcv = ["Time", "Open", "High", "Low", "Close", "Volume"]
        feature_cols = [c for c in ohlcv if c in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols.extend([c for c in numeric_cols if c not in feature_cols])
        feature_cols = list(dict.fromkeys(feature_cols))  # Dedupe preserving order

    subset = df[feature_cols].tail(n_bars).copy()
    output_path = str(Path(output_path).resolve())
    subset.to_csv(output_path, index=False)
    print(f"Exported {len(subset)} bars to {output_path}")
    print(f"Columns: {', '.join(feature_cols)}")
    return output_path
