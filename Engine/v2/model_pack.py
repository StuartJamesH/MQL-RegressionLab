"""
Engine/v2/model_pack.py — v2 deployment pack loader.

Loads the immutable artifact directory produced by
``ModelWorkbench/Learn/v2/deploy.py`` and reconstructs the Python objects
required for live inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


class ModelPackLoader:
    """
    Self-contained loader for Learn.v2 transformer deployment packs.

    A valid pack contains the following files:
      * ``model.onnx``      — ONNX Runtime inference graph (required)
      * ``config.json``     — ``ModelConfig`` serialized by ``deploy.py``
      * ``normalizer.json`` — mean/std statistics used at training time
      * ``feature_spec.json`` — ``FeatureSpec`` serialization
      * ``model_info.json`` — training metadata
      * ``model.pt``        — PyTorch state dict (optional; required for
        the PyTorch fallback backend)

    The loader never modifies pack contents.
    """

    REQUIRED_FILES = ["config.json", "normalizer.json", "feature_spec.json", "model_info.json"]
    ONNX_FILE = "model.onnx"
    PYTORCH_FILE = "model.pt"

    @classmethod
    def load(cls, pack_dir: str) -> Dict[str, Any]:
        """
        Read a v2 model pack directory and return its artifacts as Python objects.

        Parameters
        ----------
        pack_dir : str
            Path to the deployment pack directory.

        Returns
        -------
        dict
            Keys:
              * ``config`` — ``Learn.v2.model.config.ModelConfig``
              * ``normalizer`` — dict from ``normalizer.json``
              * ``feature_spec`` — ``Learn.v2.feature_spec.FeatureSpec``
              * ``model_info`` — dict from ``model_info.json``
              * ``pytorch_state_dict`` — PyTorch state dict or ``None``
              * ``onnx_path`` — absolute path to ``model.onnx``
              * ``pack_dir`` — absolute path to the pack directory
        """
        pack_path = Path(pack_dir).expanduser().resolve()
        if not pack_path.is_dir():
            raise NotADirectoryError(f"Model pack directory not found: {pack_path}")

        # ONNX is mandatory for the primary inference backend.
        onnx_path = pack_path / cls.ONNX_FILE
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing required ONNX model: {onnx_path}")

        for fname in cls.REQUIRED_FILES:
            fpath = pack_path / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required pack file: {fpath}")

        # config.json -> ModelConfig
        with open(pack_path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        from Learn.v2.model.config import ModelConfig

        # Only keep keys that belong to ModelConfig so stale/derived fields
        # from older exports do not break reconstruction.
        valid_fields = set(ModelConfig.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = ModelConfig(**filtered_config)

        # normalizer.json
        with open(pack_path / "normalizer.json", "r", encoding="utf-8") as f:
            normalizer = json.load(f)

        # feature_spec.json -> FeatureSpec
        with open(pack_path / "feature_spec.json", "r", encoding="utf-8") as f:
            feature_spec_dict = json.load(f)

        from Learn.v2.feature_spec import FeatureSpec

        feature_spec = FeatureSpec.from_dict(feature_spec_dict)

        # model_info.json
        with open(pack_path / "model_info.json", "r", encoding="utf-8") as f:
            model_info = json.load(f)

        # model.pt is optional; only required for PyTorch fallback.
        pt_path = pack_path / cls.PYTORCH_FILE
        pytorch_state_dict = None
        if pt_path.exists():
            raw_checkpoint = torch.load(
                pt_path, map_location="cpu", weights_only=False
            )
            # Training checkpoints are often saved as a wrapper dict containing
            # 'model_state_dict'; extract it when present.
            if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
                pytorch_state_dict = raw_checkpoint["model_state_dict"]
            else:
                pytorch_state_dict = raw_checkpoint

        return {
            "config": config,
            "normalizer": normalizer,
            "feature_spec": feature_spec,
            "model_info": model_info,
            "pytorch_state_dict": pytorch_state_dict,
            "onnx_path": str(onnx_path),
            "pack_dir": str(pack_path),
        }
