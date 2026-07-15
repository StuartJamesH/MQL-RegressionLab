"""
One-shot deployment script: generates ONNX + JSON artifacts for run15_100ep
from the existing model.pt checkpoint.

Run from ModelWorkbench/:
    ../.venv/bin/python deploy_run15.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Learn.v2.model.config import ModelConfig
from Learn.v2.model.full_model import TradeForecastTransformer
from Learn.v2.feature_spec import FeatureSpec
from Learn.v2.deploy import DeploymentPackager


def main():
    pack_dir = Path("ModelPacks/transformers/run15_100ep")
    pt_path = pack_dir / "model.pt"

    print(f"Loading checkpoint from {pt_path}")
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    # 1. Reconstruct ModelConfig
    config_dict = ckpt["config_dict"]
    valid_fields = set(ModelConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ModelConfig(**filtered)
    print(f"ModelConfig: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"max_seq_len={config.max_seq_len}, n_horizons={config.n_horizons}")

    # 2. Save config.json
    with open(pack_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print("  -> config.json")

    # 3. FeatureSpec
    spec = FeatureSpec()
    with open(pack_dir / "feature_spec.json", "w") as f:
        json.dump(spec.to_dict(), f, indent=2, default=str)
    print("  -> feature_spec.json")

    # 4. Normalizer (format expected by Engine/v2)
    normalizer = {
        "mean": [],
        "std": [],
        "in_channels": config.in_channels,
    }
    with open(pack_dir / "normalizer.json", "w") as f:
        json.dump(normalizer, f, indent=2)
    print("  -> normalizer.json")

    # 5. Model info
    meta = ckpt.get("training_meta", {})
    model_info = {
        "model_name": "run15_100ep",
        "framework": "PyTorch",
        "format": "ONNX",
        "opset_version": 17,
        **meta,
    }
    with open(pack_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2, default=str)
    print("  -> model_info.json")

    # 6. Build model and load state dict
    print("Building model...")
    model = TradeForecastTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # 7. Export ONNX
    packager = DeploymentPackager(output_dir="ModelPacks/transformers")
    onnx_path = pack_dir / "model.onnx"
    print(f"Exporting ONNX to {onnx_path} ...")
    packager._export_onnx(model, str(onnx_path), config)
    print("  -> model.onnx")

    # 8. Quick verification
    print("\nVerifying ONNX parity (random input)...")
    x_raw = torch.randn(1, config.max_seq_len, config.in_channels)
    x_session = torch.randn(1, config.max_seq_len, config.session_channels)

    with torch.no_grad():
        pt_out = model(x_raw, x_session)
    mu_pt, logsigma_pt = pt_out.distribution if pt_out.distribution else (None, None)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = {"raw_ohlcv": x_raw.numpy().astype("float32"),
                "session_features": x_session.numpy().astype("float32")}
        onnx_out = sess.run(None, feed)
    except ImportError:
        print("  onnxruntime not installed; skipping ONNX verification")
        return

    # output names: mu, log_sigma, direction_logits, volatility, regime_logits, quantiles
    def max_diff(a, b):
        return float(abs(a - b).max())

    print("  Max absolute differences:")
    names = ["mu", "log_sigma", "direction", "volatility", "regime", "quantiles"]
    for i, name in enumerate(names):
        if i == 0 and mu_pt is not None:
            diff = max_diff(mu_pt.numpy(), onnx_out[i])
        elif i == 1 and logsigma_pt is not None:
            diff = max_diff(logsigma_pt.numpy(), onnx_out[i])
        elif i == 2:
            diff = max_diff(pt_out.direction.numpy(), onnx_out[i])
        elif i == 3:
            diff = max_diff(pt_out.volatility.numpy(), onnx_out[i])
        elif i == 4:
            diff = max_diff(pt_out.regime.numpy(), onnx_out[i])
        else:
            diff = max_diff(pt_out.quantiles.numpy() if pt_out.quantiles is not None else 0, onnx_out[i])
        status = "OK" if diff < 1e-4 else f"WARNING: {diff:.2e}"
        print(f"    {name}: {diff:.6f}  {status}")

    print("\nDone. Pack contents:")
    for f in sorted(pack_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
