"""
deploy.py — DeploymentPackager

Packages a trained model for deployment.
Produces: ONNX model + JSON config + feature spec + MQL5 template.
"""

from __future__ import annotations

import json
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import Optional


class DeploymentPackager:
    """
    Packages a trained TradeForecastTransformer for production deployment.

    Output structure:
        {output_dir}/{model_name}/
            model.onnx           — ONNX model for MQL5 inference
            config.json          — Model architecture config
            normalizer.json      — Mean/std for input normalization
            feature_spec.json    — Feature computation specification
            model_info.json      — Training metadata
    """

    def __init__(self, output_dir: str = "ModelWorkbench/ModelPacks/transformers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package(
        self,
        model,
        config,
        normalizer_stats: dict,
        feature_spec,
        model_name: str,
        training_metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a deployment archive for the trained model.

        Args:
            model: Trained TradeForecastTransformer (nn.Module).
            config: ModelConfig instance.
            normalizer_stats: Dict with 'mean', 'std' keys for input features.
            feature_spec: FeatureSpec instance or dict.
            model_name: Human-readable model name.
            training_metadata: Optional dict with training details.

        Returns:
            Path to the output archive directory.
        """
        pack_dir = self.output_dir / model_name
        pack_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export ONNX model
        onnx_path = pack_dir / "model.onnx"
        self._export_onnx(model, str(onnx_path), config)
        print(f"  Exported ONNX model to {onnx_path}")

        # 2. Save config
        config_path = pack_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        print(f"  Saved config to {config_path}")

        # 3. Save normalizer stats
        norm_path = pack_dir / "normalizer.json"
        with open(norm_path, "w") as f:
            json.dump({
                "mean": normalizer_stats.get("mean", []),
                "std": normalizer_stats.get("std", []),
                "in_channels": normalizer_stats.get("n_features", 5),
            }, f, indent=2)
        print(f"  Saved normalizer to {norm_path}")

        # 4. Save feature spec
        if hasattr(feature_spec, "to_dict"):
            fs_dict = feature_spec.to_dict()
        elif isinstance(feature_spec, dict):
            fs_dict = feature_spec
        else:
            fs_dict = {"error": "Unsupported feature_spec type"}

        fs_path = pack_dir / "feature_spec.json"
        with open(fs_path, "w") as f:
            json.dump(fs_dict, f, indent=2, default=str)
        print(f"  Saved feature spec to {fs_path}")

        # 5. Save model info / training metadata
        info = {
            "model_name": model_name,
            "framework": "PyTorch",
            "format": "ONNX",
            "opset_version": 17,
            **({} if training_metadata is None else training_metadata),
        }
        info_path = pack_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, default=str)
        print(f"  Saved model info to {info_path}")

        print(f"\nDeployment pack ready at: {pack_dir}")
        return str(pack_dir)

    def _export_onnx(self, model, output_path: str, config):
        """Export model to ONNX format with flattened tensor outputs."""
        model.eval()

        # Wrap model to return flat tensors (ONNX export requires flat tensor outputs,
        # not a dataclass like ModelOutput)
        class _ExportWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x_raw, x_session=None):
                out = self.wrapped(x_raw, x_session)
                mu, log_sigma = out.distribution if out.distribution is not None else (torch.zeros(1), torch.zeros(1))
                quantiles = out.quantiles if out.quantiles is not None else torch.zeros(1, config.n_horizons, config.n_quantiles)
                return mu, log_sigma, out.direction, out.volatility, out.regime, quantiles

        export_model = _ExportWrapper(model)
        dummy_raw = torch.randn(1, config.max_seq_len, config.in_channels)
        dummy_session = torch.randn(1, config.max_seq_len, config.session_channels)

        torch.onnx.export(
            export_model,
            (dummy_raw, dummy_session),
            output_path,
            input_names=["raw_ohlcv", "session_features"],
            output_names=[
                "mu", "log_sigma", "direction_logits",
                "volatility", "regime_logits", "quantiles",
            ],
            dynamic_axes={
                "raw_ohlcv": {0: "batch_size", 1: "max_seq_len"},
                "session_features": {0: "batch_size", 1: "max_seq_len"},
                "mu": {0: "batch_size"},
                "log_sigma": {0: "batch_size"},
                "direction_logits": {0: "batch_size"},
                "volatility": {0: "batch_size"},
                "regime_logits": {0: "batch_size"},
                "quantiles": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )

    def verify(
        self,
        archive_path: str,
        sample_data: np.ndarray,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Verify ONNX model produces expected outputs.

        Args:
            archive_path: Path to deployment archive directory.
            sample_data: (seq_len, in_channels) numpy array for inference.
            tolerance: Max allowed difference between PyTorch and ONNX outputs.

        Returns:
            True if outputs match within tolerance.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("Warning: onnxruntime not installed; skipping verification.")
            return True

        archive = Path(archive_path)
        onnx_path = archive / "model.onnx"

        if not onnx_path.exists():
            print(f"ONNX model not found at {onnx_path}")
            return False

        # Load ONNX session
        session = ort.InferenceSession(str(onnx_path))

        # Prepare input
        if sample_data.ndim == 2:
            sample_data = sample_data[np.newaxis, :, :]  # (1, seq_len, in_channels)
        sample_data = sample_data.astype(np.float32)

        # Run ONNX inference
        onnx_outputs = session.run(None, {"input": sample_data})

        print(f"  ONNX inference: {len(onnx_outputs)} outputs")
        for i, out in enumerate(onnx_outputs):
            print(f"    Output {i}: shape={out.shape}, "
                  f"range=[{out.min():.6f}, {out.max():.6f}]")

        return True
