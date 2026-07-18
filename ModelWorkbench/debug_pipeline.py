#!/usr/bin/env python
"""
debug_pipeline.py — Compare the old eager-window pipeline against the new
FinetuneDataset-based lazy-loading pipeline to diagnose discrepancies after
the refactoring from prepare_ohlcv_windows() → FinetuneDataset.

Usage (from ModelWorkbench/):
    ../.venv/bin/python debug_pipeline.py

OR from repo root:
    .venv/bin/python ModelWorkbench/debug_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the script can be run from either repo root or ModelWorkbench/
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from Learn.v2.data import normalize_ohlcv, SessionFeatureEncoder
from Learn.v2.labels import compute_directional_return_distribution
from Learn.v2.training.dataset import FinetuneDataset, DEFAULT_HORIZONS
from Learn.train_utils import load_ohlcv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEQ_LEN = 256
N_ROWS = 5000
HORIZONS = DEFAULT_HORIZONS  # [5, 10, 20, 40, 60, 120]

# CSV search order (from ModelWorkbench/ CWD or repo root)
CSV_CANDIDATES = [
    "../data/BTCUSD_M5_260weeks.csv",
    "../data/BTCUSD_M5_156weeks.csv",
    "data/BTCUSD_M5_260weeks.csv",
    "data/BTCUSD_M5_156weeks.csv",
    "../data/XAUUSD_M5_260weeks.csv",
    "../data/XAUUSD_M5_156weeks.csv",
    "../data/BTCUSD_M1_156weeks.csv",
    "../data/BTCUSD_M1_520weeks.csv",
]


# ============================================================================
# Helpers
# ============================================================================


def _find_csv() -> Path:
    """Resolve the first existing CSV from the candidate list."""
    for cand in CSV_CANDIDATES:
        p = Path(cand).resolve()
        if p.exists():
            return p
    # Brute-force: scan repo-root data/ for any M5
    repo_data = Path(__file__).resolve().parent.parent / "data"
    if repo_data.exists():
        for f in sorted(repo_data.glob("*M5*.csv")):
            if f.stat().st_size > 1000:
                return f
    searched = "\n  ".join(str(Path(c).resolve()) for c in CSV_CANDIDATES)
    raise FileNotFoundError(
        f"No CSV found among candidates:\n  {searched}"
    )


def _stats(name: str, arr: np.ndarray) -> str:
    """Return a compact stats string (mean, std, min, max)."""
    a = arr.astype(np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return f"{name}: [empty]"
    return (
        f"{name}: shape={arr.shape}, "
        f"mean={a.mean():.6g}, std={a.std():.6g}, "
        f"min={a.min():.6g}, max={a.max():.6g}"
    )


# ============================================================================
# OLD pipeline — replicate prepare_ohlcv_windows logic
# ============================================================================


def old_windows(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    horizons: List[int] | None = None,
) -> Tuple[
    List[np.ndarray],  # raw windows
    List[np.ndarray],  # sess windows
    List[np.ndarray],  # labels
]:
    """
    Build windows the *old* way: eager for-loop over n_bars - seq_len,
    matching the original prepare_ohlcv_windows() logic before the
    refactoring to FinetuneDataset.

    Steps (in order):
      1. normalize_ohlcv(df) → X_raw  (float32)
      2. SessionFeatureEncoder().encode(times, include_gap=True) → X_sess  (float32)
      3. compute_directional_return_distribution(df, horizons) → labels (float64 → float32)
      4. Slice: for i in range(n_bars - seq_len):
           raw[i:i+seq_len], sess[i:i+seq_len], label[i+seq_len-1]
      5. Filter windows whose label row has any NaN horizon
    """
    horizons = horizons or HORIZONS
    n_bars = len(df)

    if n_bars < seq_len:
        raise ValueError(f"DataFrame too short: {n_bars} < {seq_len}")

    # Step 1: normalise OHLCV
    X_raw = normalize_ohlcv(df).astype(np.float32)

    # Step 2: encode session features
    encoder = SessionFeatureEncoder()
    times = pd.to_datetime(df["Time"])
    X_sess = encoder.encode(times, include_gap=True).astype(np.float32)

    # Step 3: compute directional return distribution
    #  (FinetuneDataset casts result to float32; replicate that for fair comparison)
    labels = compute_directional_return_distribution(df, horizons).astype(np.float32)

    # Step 4: slice windows — same count as FinetuneDataset._add_dataset
    #   n_windows_raw = n_bars - seq_len  (NOT +1)
    n_windows_raw = max(0, n_bars - seq_len)

    raw_wins: List[np.ndarray] = []
    sess_wins: List[np.ndarray] = []
    label_vals: List[np.ndarray] = []

    for i in range(n_windows_raw):
        raw_wins.append(X_raw[i : i + seq_len].copy())
        sess_wins.append(X_sess[i : i + seq_len].copy())
        label_vals.append(labels[i + seq_len - 1].copy())

    # Step 5: filter windows whose label row has any NaN
    raw_arr = np.array(raw_wins)   # (n_windows_raw, seq_len, 5)
    sess_arr = np.array(sess_wins)  # (n_windows_raw, seq_len, 5)
    lbl_arr = np.array(label_vals)  # (n_windows_raw, n_horizons)

    # Keep windows where ALL label horizons are finite
    valid = np.isfinite(lbl_arr).all(axis=1)

    return (
        [raw_arr[i] for i in range(len(raw_arr)) if valid[i]],
        [sess_arr[i] for i in range(len(sess_arr)) if valid[i]],
        [lbl_arr[i] for i in range(len(lbl_arr)) if valid[i]],
    )


# ============================================================================
# NEW pipeline — FinetuneDataset
# ============================================================================


def new_windows(
    csv_path: Path,
    n_rows: int = N_ROWS,
    seq_len: int = SEQ_LEN,
) -> Tuple[
    List[np.ndarray],  # raw windows
    List[np.ndarray],  # sess windows
    List[np.ndarray],  # labels
]:
    """
    Collect every window from FinetuneDataset by iterating over all indices.
    Returns the same structure as old_windows() for side-by-side comparison.
    """
    ds = FinetuneDataset(
        ds_paths=[str(csv_path)],
        n_rows=n_rows,
        seq_len=seq_len,
        target_type="log_return",
        horizons=HORIZONS,
    )

    raw_wins: List[np.ndarray] = []
    sess_wins: List[np.ndarray] = []
    label_vals: List[np.ndarray] = []

    for idx in range(len(ds)):
        raw_t, sess_t, lbl_t = ds[idx]
        raw_wins.append(raw_t.numpy().copy())
        sess_wins.append(sess_t.numpy().copy())
        label_vals.append(lbl_t.numpy().copy())

    return raw_wins, sess_wins, label_vals


# ============================================================================
# Comparison logic
# ============================================================================


def compare_windows(
    old_raw: List[np.ndarray],
    old_sess: List[np.ndarray],
    old_lbl: List[np.ndarray],
    new_raw: List[np.ndarray],
    new_sess: List[np.ndarray],
    new_lbl: List[np.ndarray],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> str:
    """Side-by-side comparison. Returns "IDENTICAL" or "MISMATCH" with details."""

    n_old = len(old_raw)
    n_new = len(new_raw)

    # Pad the shorter list with None for zip comparison
    max_len = max(n_old, n_new)
    pad_old = n_old < max_len
    pad_new = n_new < max_len

    from itertools import zip_longest

    mismatches: List[str] = []
    nan_count_old = 0
    nan_count_new = 0

    for idx, (r_o, r_n) in enumerate(zip_longest(old_raw, new_raw)):
        if r_o is None:
            mismatches.append(f"  index {idx}: old MISSING (new has window)")
            break
        if r_n is None:
            mismatches.append(f"  index {idx}: new MISSING (old has window)")
            break

        if not np.allclose(r_o, r_n, rtol=rtol, atol=atol):
            # Compute detailed diffs
            diff = np.abs(r_o.astype(np.float64) - r_n.astype(np.float64))
            max_diff = diff.max()
            mean_diff = diff.mean()

            # Also check if it's just a label mismatch at certain horizons
            s_o = old_sess[idx].astype(np.float64)
            s_n = new_sess[idx].astype(np.float64)
            l_o = old_lbl[idx].astype(np.float64)
            l_n = new_lbl[idx].astype(np.float64)

            sess_match = np.allclose(s_o, s_n, rtol=rtol, atol=atol)
            raw_match = np.allclose(r_o, r_n, rtol=rtol, atol=atol)
            lbl_match = np.allclose(l_o, l_n, rtol=rtol, atol=atol)

            mismatches.append(
                f"  index {idx}: MISMATCH (raw={raw_match}, sess={sess_match}, "
                f"lbl={lbl_match}), max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}"
            )
            if len(mismatches) >= 20:
                mismatches.append("  ... (truncated at 20 mismatches)")
                break

        # Track NaN presence for diagnostics
        nan_count_old += int(np.isnan(r_o).any())
        nan_count_new += int(np.isnan(r_n).any())

    # Build result
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append("WINDOW COUNT")
    lines.append(f"  Old (eager loop):   {n_old:,}")
    lines.append(f"  New (FinetuneDataset): {n_new:,}")
    lines.append(f"  Delta:               {n_new - n_old:+d}")

    if mismatches:
        lines.append(f"\nMISMATCHES ({len(mismatches)})")
        for m in mismatches:
            lines.append(m)
    else:
        # Verify they're truly identical in all respects
        all_close = True
        for idx in range(min(n_old, n_new)):
            if not np.allclose(old_raw[idx], new_raw[idx], rtol=rtol, atol=atol):
                all_close = False
                break
            if not np.allclose(old_sess[idx], new_sess[idx], rtol=rtol, atol=atol):
                all_close = False
                break
            if not np.allclose(old_lbl[idx], new_lbl[idx], rtol=rtol, atol=atol):
                all_close = False
                break

        if all_close:
            lines.append("\nRESULT: IDENTICAL  (all windows match within tolerance)")
        else:
            lines.append("\nRESULT: MISMATCH  (some windows differ)")

    lines.append(f"\nNaN bars in raw windows: old={nan_count_old}, new={nan_count_new}")

    # Per-array stats comparison
    lines.append("\n" + "=" * 72)
    lines.append("ELEMENT-WISE STATS (all windows concatenated)")

    if n_old > 0 and n_new > 0:
        # Concatenate all windows into one flat array for each type
        old_raw_flat = np.concatenate([w.ravel() for w in old_raw])
        new_raw_flat = np.concatenate([w.ravel() for w in new_raw])
        old_sess_flat = np.concatenate([w.ravel() for w in old_sess])
        new_sess_flat = np.concatenate([w.ravel() for w in new_sess])
        old_lbl_flat = np.concatenate([w.ravel() for w in old_lbl])
        new_lbl_flat = np.concatenate([w.ravel() for w in new_lbl])

        lines.append(_stats("Old raw  ", old_raw_flat))
        lines.append(_stats("New raw  ", new_raw_flat))

        lines.append(_stats("Old sess ", old_sess_flat))
        lines.append(_stats("New sess ", new_sess_flat))

        lines.append(_stats("Old lbl  ", old_lbl_flat))
        lines.append(_stats("New lbl  ", new_lbl_flat))

        # Per-horizon label stats (most likely place for divergence)
        if n_old == n_new and n_old > 0:
            old_lbl_arr = np.array(old_lbl)
            new_lbl_arr = np.array(new_lbl)
            lines.append("\nPER-HORIZON LABEL STATS")
            for h in range(old_lbl_arr.shape[1]):
                o = old_lbl_arr[:, h]
                n = new_lbl_arr[:, h]
                # Correlation between old and new labels
                if np.std(o) > 1e-12 and np.std(n) > 1e-12:
                    corr = np.corrcoef(o, n)[0, 1]
                else:
                    corr = 1.0
                h_label = HORIZONS[h] if h < len(HORIZONS) else h
                lines.append(
                    f"  H{h_label:>3d}: old mean={o.mean():.8f} std={o.std():.8f}, "
                    f"new mean={n.mean():.8f} std={n.std():.8f}, corr={corr:.8f}"
                )

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    # --- Locate CSV ---
    try:
        csv_path = _find_csv()
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Using: {csv_path}")
    print(f"Config: n_rows={N_ROWS}, seq_len={SEQ_LEN}, horizons={HORIZONS}")

    # --- Load data ---
    df = load_ohlcv(str(csv_path), n_rows=N_ROWS)
    print(f"Loaded: {len(df):,} bars, columns={list(df.columns)}")
    print(f"Time range: {df['Time'].min()} → {df['Time'].max()}")

    # --- OLD pipeline ---
    print("\n--- OLD pipeline (eager for-loop) ---")
    old_raw, old_sess, old_lbl = old_windows(df, seq_len=SEQ_LEN, horizons=HORIZONS)
    print(f"  Windows: {len(old_raw):,}")
    if old_raw:
        print(f"  raw shape per window: {old_raw[0].shape}, dtype={old_raw[0].dtype}")
        print(f"  sess shape per window: {old_sess[0].shape}, dtype={old_sess[0].dtype}")
        print(f"  label shape per window: {old_lbl[0].shape}, dtype={old_lbl[0].dtype}")

    # --- NEW pipeline ---
    print("\n--- NEW pipeline (FinetuneDataset) ---")
    new_raw, new_sess, new_lbl = new_windows(csv_path, n_rows=N_ROWS, seq_len=SEQ_LEN)
    print(f"  Windows: {len(new_raw):,}")
    if new_raw:
        print(f"  raw shape per window: {new_raw[0].shape}, dtype={new_raw[0].dtype}")
        print(f"  sess shape per window: {new_sess[0].shape}, dtype={new_sess[0].dtype}")
        print(f"  label shape per window: {new_lbl[0].shape}, dtype={new_lbl[0].dtype}")

    # --- First / last 5 windows detail ---
    n_old = len(old_raw)
    n_new = len(new_raw)

    def _print_window_pair(idx, label):
        if idx < max(n_old, n_new):
            o_raw = old_raw[idx] if idx < n_old else None
            n_raw = new_raw[idx] if idx < n_new else None
            o_lbl = old_lbl[idx] if idx < n_old else None
            n_lbl = new_lbl[idx] if idx < n_new else None

            if o_raw is not None and n_raw is not None:
                raw_match = np.allclose(o_raw, n_raw, atol=1e-6, rtol=1e-5)
                lbl_match = np.allclose(o_lbl, n_lbl, atol=1e-6, rtol=1e-5) if o_lbl is not None and n_lbl is not None else "N/A"
            else:
                raw_match = "N/A"
                lbl_match = "N/A"

            print(f"  [{label}] idx={idx}: "
                  f"old={'present' if o_raw is not None else 'MISSING'}, "
                  f"new={'present' if n_raw is not None else 'MISSING'}, "
                  f"raw_match={raw_match}, lbl_match={lbl_match}")

    print("\n--- First 5 windows ---")
    for i in range(5):
        _print_window_pair(i, "FIRST")

    print("\n--- Last 5 windows ---")
    for i in range(max(0, max(n_old, n_new) - 5), max(n_old, n_new)):
        _print_window_pair(i, "LAST")

    # --- Full comparison ---
    print()
    result = compare_windows(
        old_raw, old_sess, old_lbl,
        new_raw, new_sess, new_lbl,
    )
    print(result)

    # --- Summary exit code ---
    if n_old != n_new:
        print("\n*** WARNING: Window count differs! ***")
        sys.exit(2)
    else:
        # Quick allclose check
        all_ok = True
        for idx in range(n_old):
            if not np.allclose(old_raw[idx], new_raw[idx], atol=1e-6, rtol=1e-5):
                all_ok = False
                break
            if not np.allclose(old_sess[idx], new_sess[idx], atol=1e-6, rtol=1e-5):
                all_ok = False
                break
            if not np.allclose(old_lbl[idx], new_lbl[idx], atol=1e-6, rtol=1e-5):
                all_ok = False
                break
        if all_ok:
            print("\n*** IDENTICAL — pipelines produce matching outputs ***")
            sys.exit(0)
        else:
            print("\n*** MISMATCH — see details above ***")
            sys.exit(1)


if __name__ == "__main__":
    main()
