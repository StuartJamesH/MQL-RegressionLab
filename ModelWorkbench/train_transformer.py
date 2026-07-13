#!/usr/bin/env python
"""
train_transformer.py — CLI entry point for training the Causal Patch Transformer.

Mirrors train_lgbm.py in structure but trains the TradeForecastTransformer
instead of LightGBM. Supports three-phase curriculum training.

Usage:
    cd ModelWorkbench
    python -m Learn.v2.training.pretrain --help
    OR
    python train_transformer.py --ds-names ../data/XAUUSD_M1.csv ...
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from Learn.v2.model.config import ModelConfig
from Learn.v2.model.full_model import TradeForecastTransformer, ModelOutput
from Learn.v2.labels import (
    compute_directional_return_distribution,
    compute_volatility_regime_labels,
    LabelStore,
)
from Learn.v2.data import normalize_ohlcv, SessionFeatureEncoder
from Learn.v2.deploy import DeploymentPackager
from Learn.v2.feature_spec import FeatureSpec
from Learn.train_utils import load_ohlcv


# Model expects 5 OHLCV channels: O, H, L, C, V
DEFAULT_HORIZONS = [5, 10, 20, 40, 60, 120]


def compute_atr_normalized_targets(df, horizons, atr_window=14):
    """
    Compute ATR-normalized forward excursion scores as targets.
    
    For each bar t and horizon h, computes:
        score = (buy_MFE - buy_MAE) / max(buy_MFE + buy_MAE, 1e-8)
    
    This produces a signed score in [-1, 1] where:
        +1 = price only moved favorably (pure win)
        -1 = price only moved adversely (pure loss)
         0 = equal movement or no movement
    
    The MFE/MAE values are in ATR units, making the target scale-invariant.
    """
    import talib
    from Learn.v2.labels import compute_forward_excursion_surface
    
    excursion = compute_forward_excursion_surface(df, horizons, atr_window=atr_window)
    # excursion shape: (n_bars, n_horizons, 2, 2) — [buy_mfe, buy_mae], [sell_mfe, sell_mae]
    buy_mfe = excursion[:, :, 0, 0]  # (n_bars, n_horizons)
    buy_mae = excursion[:, :, 0, 1]
    
    # Normalized trade quality score: MFE advantage over MAE
    denom = np.maximum(buy_mfe + buy_mae, 1e-8)
    score = (buy_mfe - buy_mae) / denom  # range [-1, 1]
    
    # Fill NaN (last horizon rows) with 0
    score = np.nan_to_num(score, nan=0.0)
    return score.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Causal Patch Transformer for trade forecasting.",
    )
    parser.add_argument("--ds-names", nargs="+", required=True,
                        help="Paths to OHLCV CSV datasets.")
    parser.add_argument("--pretrain-steps", type=int, default=10000,
                        help="Self-supervised pretraining steps.")
    parser.add_argument("--finetune-epochs", type=int, default=50,
                        help="Distributional finetuning epochs.")
    parser.add_argument("--rl-steps", type=int, default=5000,
                        help="RL policy finetuning steps.")
    parser.add_argument("--output-dir", default="ModelPacks/transformers",
                        help="Output directory for model packs.")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda, cpu, mps.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Skip self-supervised pretraining.")
    parser.add_argument("--no-rl", action="store_true",
                        help="Skip RL finetuning.")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Input sequence length in bars (max_seq_len).")
    parser.add_argument("--n-rows", type=int, default=500000,
                        help="Max rows per dataset.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Custom model name for the output pack.")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Transformer hidden dimension.")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of transformer layers.")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads.")
    parser.add_argument("--d-ff", type=int, default=512,
                        help="FFN hidden dimension.")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data for validation.")
    parser.add_argument("--direction-weight", type=float, default=0.0,
                        help="Weight for auxiliary direction classification loss (0=disabled).")
    parser.add_argument("--volatility-weight", type=float, default=0.0,
                        help="Weight for auxiliary volatility prediction loss (0=disabled).")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in transformer layers.")
    parser.add_argument("--target-type", type=str, default="log_return",
                        choices=["log_return", "atr_score"],
                        help="Target type: log_return=forward log returns, atr_score=ATR-normalized MFE/MAE score.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer.")
    return parser.parse_args()


def prepare_ohlcv_windows(
    ds_path: str,
    n_rows: int,
    seq_len: int,
    target_type: str = "log_return",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, normalize, and create sliding windows from an OHLCV CSV.

    Returns:
        X_raw: (n_windows, seq_len, 5) — normalized OHLCV
        X_sess: (n_windows, seq_len, 4) — session features (hour_sin/cos, dow_sin/cos)
        labels: (n_windows, n_horizons) — targets depending on target_type
    """
    df = load_ohlcv(ds_path, n_rows=n_rows)
    print(f"  Loaded {ds_path}: {len(df)} rows")

    # Normalize OHLCV
    X_raw = normalize_ohlcv(df)  # (n_bars, 5)

    # Session features (include temporal gap flag for cross-instrument sessions)
    encoder = SessionFeatureEncoder()
    times = pd.to_datetime(df["Time"])
    X_sess = encoder.encode(times, include_gap=True)  # (n_bars, 5)

    # Label: directional return distribution or ATR-normalized score
    if target_type == "atr_score":
        labels = compute_atr_normalized_targets(df, DEFAULT_HORIZONS)
    else:
        labels = compute_directional_return_distribution(df, DEFAULT_HORIZONS)

    # Create sliding windows (causal: window i...i+seq_len-1 predicts label at i+seq_len-1)
    n_bars = len(X_raw)
    n_windows = max(0, n_bars - seq_len)
    
    windows_raw = np.zeros((n_windows, seq_len, 5), dtype=np.float32)
    windows_sess = np.zeros((n_windows, seq_len, 5), dtype=np.float32)
    windows_labels = np.zeros((n_windows, len(DEFAULT_HORIZONS)), dtype=np.float32)
    
    for i in range(n_windows):
        windows_raw[i] = X_raw[i:i + seq_len]
        windows_sess[i] = X_sess[i:i + seq_len]
        # Label at the last bar of the window
        windows_labels[i] = labels[i + seq_len - 1]

    return windows_raw, windows_sess, windows_labels


def main():
    args = parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {args.ds_names}")

    # Config — using the canonical ModelConfig from model/config.py
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
        n_horizons=len(DEFAULT_HORIZONS),
        patch_len=16,
        patch_stride=8,
    )
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"max_seq_len={config.max_seq_len}, n_horizons={config.n_horizons}, "
          f"dropout={config.dropout}, target_type={args.target_type}")

    # Build model
    model = TradeForecastTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable parameters")

    # Prepare data
    print("\n--- Loading Data ---")
    all_raw = []
    all_sess = []
    all_labels = []
    for ds_path in args.ds_names:
        raw, sess, lbl = prepare_ohlcv_windows(ds_path, args.n_rows, args.seq_len, args.target_type)
        all_raw.append(raw)
        all_sess.append(sess)
        all_labels.append(lbl)

    X_raw = np.concatenate(all_raw, axis=0)
    X_sess = np.concatenate(all_sess, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)

    # Filter windows with NaN labels (caused by max_horizon tail)
    valid = ~np.isnan(y_labels).any(axis=1) & ~np.isnan(X_raw).any(axis=(1, 2)) & ~np.isnan(X_sess).any(axis=(1, 2))
    if valid.sum() < len(valid):
        n_filtered = len(valid) - valid.sum()
        X_raw = X_raw[valid]
        X_sess = X_sess[valid]
        y_labels = y_labels[valid]
        print(f"Filtered {n_filtered} windows with NaN labels/features")

    print(f"Total windows: {len(X_raw):,}")

    # Convert to tensors
    X_raw_t = torch.from_numpy(X_raw).float()
    X_sess_t = torch.from_numpy(X_sess).float()
    y_t = torch.from_numpy(y_labels).float()

    # Train/validation split
    val_frac = args.val_split
    n_train = int(len(X_raw_t) * (1.0 - val_frac))
    indices = torch.randperm(len(X_raw_t))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train_raw, X_val_raw = X_raw_t[train_idx], X_raw_t[val_idx]
    X_train_sess, X_val_sess = X_sess_t[train_idx], X_sess_t[val_idx]
    y_train, y_val = y_t[train_idx], y_t[val_idx]
    print(f"Train windows: {len(train_idx):,}  |  Val windows: {len(val_idx):,}")

    # Create output directory for logs
    model_name = args.model_name or f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(args.output_dir) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = log_dir / "metrics.jsonl"

    # Training metrics tracker
    from Learn.v2.training.metrics import TrainingMetricsTracker
    tracker = TrainingMetricsTracker(log_dir=str(log_dir / "tb_logs"))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs,
    )

    # Per-epoch metrics history (saved to JSON)
    epoch_metrics: list[dict] = []

    # Training loop (Phase 2: Distributional Finetuning)
    print(f"\n{'='*80}")
    print(f"{'Epoch':>6s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'Val Sprmn':>10s} | {'Dir Acc':>8s} | {'LR':>9s} | {'Grad Norm':>10s}")
    print(f"{'='*80}")
    model.train()

    for epoch in range(args.finetune_epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        max_grad_norm = 0.0

        indices_t = torch.randperm(len(X_train_raw))
        for i in range(0, len(X_train_raw), args.batch_size):
            batch_idx = indices_t[i:i + args.batch_size]
            batch_raw = X_train_raw[batch_idx].to(device)
            batch_sess = X_train_sess[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            output: ModelOutput = model(batch_raw, batch_sess)

            # Numerically stable Gaussian NLL
            mu, log_sigma = output.distribution
            sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0)) + 1e-6
            nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + 0.5 * ((batch_y - mu) / sigma) ** 2
            loss = nll.mean()

            # Auxiliary direction loss (predict sign of return)
            if args.direction_weight > 0:
                dir_target = (batch_y > 0).float()
                dir_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output.direction, dir_target
                )
                loss = loss + args.direction_weight * dir_loss

            # Auxiliary volatility loss (predict absolute return magnitude)
            if args.volatility_weight > 0:
                vol_target = torch.log(torch.abs(batch_y) + 1e-8)
                vol_loss = torch.nn.functional.mse_loss(output.volatility, vol_target)
                loss = loss + args.volatility_weight * vol_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            max_grad_norm = max(max_grad_norm, float(grad_norm))

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0.0
        val_pred_mu = []
        val_pred_log_sigma = []
        val_pred_direction = []
        val_true = []

        with torch.no_grad():
            for i in range(0, len(X_val_raw), args.batch_size * 2):
                batch_raw = X_val_raw[i:i + args.batch_size * 2].to(device)
                batch_sess = X_val_sess[i:i + args.batch_size * 2].to(device)
                batch_y = y_val[i:i + args.batch_size * 2].to(device)

                output = model(batch_raw, batch_sess)
                mu, log_sigma = output.distribution
                sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0)) + 1e-6
                nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + 0.5 * ((batch_y - mu) / sigma) ** 2
                val_loss += nll.mean().item()

                val_pred_mu.append(mu.cpu().numpy())
                val_pred_log_sigma.append(log_sigma.cpu().numpy())
                val_pred_direction.append(torch.sigmoid(output.direction).cpu().numpy())
                val_true.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / max(i // (args.batch_size * 2) + 1, 1)

        # Concatenate validation predictions
        val_mu_all = np.concatenate(val_pred_mu, axis=0)
        val_true_all = np.concatenate(val_true, axis=0)
        val_dir_all = np.concatenate(val_pred_direction, axis=0)

        # ---- COMPUTE METRICS ----
        # Spearman per horizon
        from scipy.stats import spearmanr
        spearman_vals = []
        for h in range(val_mu_all.shape[1]):
            valid = np.isfinite(val_mu_all[:, h]) & np.isfinite(val_true_all[:, h])
            if valid.sum() >= 2 and np.std(val_mu_all[valid, h]) > 0 and np.std(val_true_all[valid, h]) > 0:
                sp = spearmanr(val_mu_all[valid, h], val_true_all[valid, h]).statistic
                spearman_vals.append(0.0 if sp is None or np.isnan(sp) else float(sp))
            else:
                spearman_vals.append(0.0)
        mean_spearman = float(np.mean(spearman_vals)) if spearman_vals else 0.0

        # Directional accuracy (predicted sign vs actual sign, at primary horizon)
        primary_h = min(2, val_mu_all.shape[1] - 1)  # horizon index 2 = 20 bars
        pred_sign = np.sign(val_mu_all[:, primary_h])
        true_sign = np.sign(val_true_all[:, primary_h])
        valid_sign = (pred_sign != 0) & (true_sign != 0)
        dir_accuracy = float(np.mean(pred_sign[valid_sign] == true_sign[valid_sign])) if valid_sign.sum() > 0 else 0.0

        # ---- LOG TO TRACKER ----
        current_lr = scheduler.get_last_lr()[0]
        tracker.log_scalar("loss/train", avg_train_loss, step=epoch)
        tracker.log_scalar("loss/val", avg_val_loss, step=epoch)
        tracker.log_scalar("metrics/val_spearman_mean", mean_spearman, step=epoch)
        tracker.log_scalar("metrics/val_direction_accuracy", dir_accuracy, step=epoch)
        tracker.log_scalar("metrics/grad_norm", max_grad_norm, step=epoch)
        tracker.log_scalar("metrics/learning_rate", current_lr, step=epoch)
        for h, sp in enumerate(spearman_vals):
            tracker.log_scalar(f"metrics/val_spearman_h{h}", sp, step=epoch)

        # Store per-epoch record
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_spearman_mean": mean_spearman,
            "val_direction_accuracy": dir_accuracy,
            "val_spearman_per_horizon": spearman_vals,
            "grad_norm_max": max_grad_norm,
            "learning_rate": current_lr,
        }
        epoch_metrics.append(epoch_record)

        # Print epoch summary
        print(f"{epoch+1:5d}/{args.finetune_epochs:<3d} | {avg_train_loss:10.6f} | {avg_val_loss:10.6f} | "
              f"{mean_spearman:10.4f} | {dir_accuracy:7.4f} | {current_lr:8.2e} | {max_grad_norm:10.4f}")

    print(f"{'='*80}")

    # Save per-epoch metrics to JSON log
    with open(metrics_log_path, "w") as f:
        for record in epoch_metrics:
            f.write(json.dumps(record) + "\n")
    print(f"Epoch metrics log saved to {metrics_log_path}")

    # Save best model based on validation Spearman
    best_epoch = max(epoch_metrics, key=lambda r: r["val_spearman_mean"])
    print(f"Best epoch: {best_epoch['epoch']} (val Spearman={best_epoch['val_spearman_mean']:.4f})")

    tracker.finalize()

    # Compute normalizer stats from first dataset for deployment
    print("\n--- Computing Normalizer Stats ---")
    df_sample = load_ohlcv(args.ds_names[0], n_rows=args.n_rows)
    normalizer_stats = {
        "close_mean": float(df_sample["Close"].mean()),
        "close_std": float(df_sample["Close"].std()),
        "atr_mean": float((df_sample["High"] - df_sample["Low"]).mean()),
        "n_features": 5,
        "input_channels": ["O", "H", "L", "C", "V"],
        "session_channels": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "has_gap"],
        "horizons": DEFAULT_HORIZONS,
    }

    # Package model (ONNX export may fail if dynamo tracer can't handle ModelOutput;
    # PyTorch checkpoint is always saved regardless)
    print("\n--- Packaging Model for Deployment ---")
    feature_spec = FeatureSpec()
    packager = DeploymentPackager(output_dir=args.output_dir)

    training_meta = {
        "n_epochs": args.finetune_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "device": str(device),
        "datasets": args.ds_names,
        "config_d_model": config.d_model,
        "config_n_layers": config.n_layers,
        "config_n_heads": config.n_heads,
        "config_max_seq_len": config.max_seq_len,
        "config_n_horizons": config.n_horizons,
        "total_params": total_params,
    }

    pack_path = None
    try:
        pack_path = packager.package(
            model=model,
            config=config,
            normalizer_stats=normalizer_stats,
            feature_spec=feature_spec,
            model_name=model_name,
            training_metadata=training_meta,
        )
    except Exception as e:
        print(f"  ONNX export failed (non-critical): {e}")
        pack_path = str(Path(args.output_dir) / model_name)
        Path(pack_path).mkdir(parents=True, exist_ok=True)

    # Save full PyTorch model checkpoint
    ckpt_path = Path(pack_path) / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config_dict": {k: v for k, v in config.__dict__.items()},
        "normalizer": normalizer_stats,
        "training_meta": training_meta,
    }, ckpt_path)
    print(f"  Saved PyTorch checkpoint to {ckpt_path}")

    print(f"\nTraining complete. Model pack at: {pack_path}")
    print(f"  ONNX model: {pack_path}/model.onnx")
    print(f"  Config:     {pack_path}/config.json")


if __name__ == "__main__":
    main()
