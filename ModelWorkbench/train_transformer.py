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

import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")

from Learn.v2.model.config import ModelConfig
from Learn.v2.model.full_model import TradeForecastTransformer, ModelOutput
from Learn.v2.training.dataset import FinetuneDataset, DEFAULT_HORIZONS
from Learn.v2.deploy import DeploymentPackager
from Learn.v2.feature_spec import FeatureSpec
from Learn.train_utils import load_ohlcv


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
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers (0=main process only).")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience in epochs (0=disabled). Tracks val_spearman_mean.")
    return parser.parse_args()


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

    # Prepare data — uses memory-efficient FinetuneDataset (2D arrays only,
    # windows sliced on-the-fly in __getitem__)
    print("\n--- Loading Data ---")
    dataset = FinetuneDataset(
        ds_paths=args.ds_names,
        n_rows=args.n_rows,
        seq_len=args.seq_len,
        target_type=args.target_type,
        horizons=DEFAULT_HORIZONS,
    )

    # Train/validation split
    val_frac = args.val_split
    n_total = len(dataset)
    n_train = int(n_total * (1.0 - val_frac))
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Total windows: {n_total:,}  |  Train: {n_train:,}  |  Val: {n_val:,}")

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

    # Early stopping state
    best_spearman = -float("inf")
    patience_counter = 0
    best_state_dict = None

    # Training loop (Phase 2: Distributional Finetuning)
    print(f"\n{'='*80}")
    print(f"{'Epoch':>6s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'Val Sprmn':>10s} | {'Dir Acc':>8s} | {'LR':>9s} | {'Grad Norm':>10s}")
    if args.patience > 0:
        print(f"  Patience: {args.patience} (best Spearman: -inf)")
    print(f"{'='*80}")

    for epoch in range(args.finetune_epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        max_grad_norm = 0.0

        for batch_raw, batch_sess, batch_y in train_loader:
            batch_raw = batch_raw.to(device)
            batch_sess = batch_sess.to(device)
            batch_y = batch_y.to(device)

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
        val_n_batches = 0
        val_pred_mu = []
        val_pred_log_sigma = []
        val_pred_direction = []
        val_true = []

        with torch.no_grad():
            for batch_raw, batch_sess, batch_y in val_loader:
                batch_raw = batch_raw.to(device)
                batch_sess = batch_sess.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_raw, batch_sess)
                mu, log_sigma = output.distribution
                sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0)) + 1e-6
                nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + 0.5 * ((batch_y - mu) / sigma) ** 2
                val_loss += nll.mean().item()
                val_n_batches += 1

                val_pred_mu.append(mu.cpu().numpy())
                val_pred_log_sigma.append(log_sigma.cpu().numpy())
                val_pred_direction.append(torch.sigmoid(output.direction).cpu().numpy())
                val_true.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / max(val_n_batches, 1)

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

        # ---- EARLY STOPPING ----
        if args.patience > 0:
            if mean_spearman > best_spearman:
                improved_by = mean_spearman - best_spearman
                best_spearman = mean_spearman
                patience_counter = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                print(f"  >> New best Spearman: {best_spearman:.4f} (+{improved_by:.6f}), patience reset")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  >> Early stopping triggered at epoch {epoch+1} "
                          f"(patience={args.patience}, best Spearman={best_spearman:.4f})")
                    break

        # Print epoch summary
        print(f"{epoch+1:5d}/{args.finetune_epochs:<3d} | {avg_train_loss:10.6f} | {avg_val_loss:10.6f} | "
              f"{mean_spearman:10.4f} | {dir_accuracy:7.4f} | {current_lr:8.2e} | {max_grad_norm:10.4f}")

    print(f"{'='*80}")

    # Restore best model if early stopping was used
    if args.patience > 0 and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val Spearman={best_spearman:.4f})")
    elif args.patience > 0:
        print(f"No improvement recorded; using final model weights")

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
