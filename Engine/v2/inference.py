"""
Engine/v2/inference.py — ONNX primary + PyTorch fallback inference engine.

Reconstructs a Learn.v2 TradeForecastTransformer from a deployment pack and
exposes a unified ``predict`` interface.  ONNX Runtime is the default backend
so the Python runtime stays aligned with the MQL5 deployment path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover — optional in non-ONNX environments
    ort = None

from Learn.v2.model.config import ModelConfig
from Learn.v2.model.full_model import TradeForecastTransformer


class V2InferenceEngine:
    """
    Inference engine for a v2 transformer model pack.

    Parameters
    ----------
    pack : dict
        Output of :class:`~Engine.v2.model_pack.ModelPackLoader.load`.
    backend : str, optional
        ``"onnx"`` (default) or ``"pytorch"``.
    device : str, optional
        PyTorch device or ONNX provider hint.  Defaults to ``"cpu"``.
    """

    def __init__(self, pack: Dict[str, Any], backend: str = "onnx", device: str = "cpu"):
        self.pack = pack
        self.config: ModelConfig = pack["config"]
        self.backend = (backend or "onnx").lower()
        self.device = device

        self._session: Optional[Any] = None
        self._model: Optional[TradeForecastTransformer] = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

        if self.backend == "onnx":
            self._init_onnx()
        elif self.backend == "pytorch":
            self._init_pytorch()
        else:
            raise ValueError(f"Unknown inference backend: {backend!r}")

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_onnx(self) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with `pip install onnxruntime` "
                "or use backend='pytorch'."
            )

        onnx_path = Path(self.pack["onnx_path"])
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device != "cpu"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    def _init_pytorch(self) -> None:
        state_dict = self.pack.get("pytorch_state_dict")
        if state_dict is None:
            raise ValueError(
                "PyTorch backend requires model.pt in the pack. "
                "Re-export the pack with the checkpoint or use backend='onnx'."
            )

        self._model = TradeForecastTransformer(self.config)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        x_raw: np.ndarray,
        x_session: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference and return structured model outputs.

        Parameters
        ----------
        x_raw : np.ndarray
            Normalised OHLCV tensor.  Accepts ``(seq_len, 5)`` or
            ``(batch, seq_len, 5)``.
        x_session : np.ndarray
            Session feature tensor.  Accepts ``(seq_len, C)`` or
            ``(batch, seq_len, C)`` where ``C`` matches ``config.session_channels``.

        Returns
        -------
        dict
            Keys: ``mu``, ``log_sigma``, ``direction``, ``volatility``,
            ``regime``, ``quantiles``.  All values are NumPy arrays with a
            leading batch dimension of size 1.
        """
        if x_raw.ndim == 2:
            x_raw = x_raw[np.newaxis, :, :]
        if x_session.ndim == 2:
            x_session = x_session[np.newaxis, :, :]

        x_raw = np.ascontiguousarray(x_raw, dtype=np.float32)
        x_session = np.ascontiguousarray(x_session, dtype=np.float32)

        # Guard against channel mismatch caused by stale configs or different
        # session encoders.  Slice or pad to the config's expected shapes.
        x_raw = self._align_channels(x_raw, target_channels=self.config.in_channels)
        x_session = self._align_channels(
            x_session, target_channels=self.config.session_channels
        )

        if self.backend == "onnx":
            return self._predict_onnx(x_raw, x_session)
        return self._predict_pytorch(x_raw, x_session)

    @staticmethod
    def _align_channels(x: np.ndarray, target_channels: int) -> np.ndarray:
        """Slice or zero-pad the last axis to match ``target_channels``."""
        current = x.shape[-1]
        if current == target_channels:
            return x
        if current > target_channels:
            return x[..., :target_channels]
        padding = np.zeros(
            (*x.shape[:-1], target_channels - current), dtype=x.dtype
        )
        return np.concatenate([x, padding], axis=-1)

    def _predict_onnx(self, x_raw: np.ndarray, x_session: np.ndarray) -> Dict[str, np.ndarray]:
        feed: Dict[str, np.ndarray] = {}
        for name in self._input_names:
            if "session" in name:
                feed[name] = x_session
            else:
                feed[name] = x_raw

        outputs = self._session.run(None, feed)

        # Map by declared output order; fallback to positional mapping if names
        # differ from the export template.
        def _find(prefix: str, default_idx: int) -> np.ndarray:
            for idx, out_name in enumerate(self._output_names):
                if out_name.startswith(prefix):
                    return outputs[idx]
            return outputs[default_idx]

        return {
            "mu": _find("mu", 0),
            "log_sigma": _find("log_sigma", 1),
            "direction": _find("direction", 2),
            "volatility": _find("volatility", 3),
            "regime": _find("regime", 4),
            "quantiles": _find("quantile", 5),
        }

    def _predict_pytorch(
        self, x_raw: np.ndarray, x_session: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_raw_t = torch.from_numpy(x_raw).to(self.device)
        x_session_t = torch.from_numpy(x_session).to(self.device)

        with torch.no_grad():
            out = self._model(x_raw_t, x_session_t)

        mu, log_sigma = out.distribution if out.distribution is not None else (None, None)
        quantiles = (
            out.quantiles
            if out.quantiles is not None
            else torch.zeros(
                x_raw.shape[0],
                self.config.n_horizons,
                self.config.n_quantiles,
                device=self.device,
            )
        )

        return {
            "mu": mu.detach().cpu().numpy(),
            "log_sigma": log_sigma.detach().cpu().numpy(),
            "direction": out.direction.detach().cpu().numpy(),
            "volatility": out.volatility.detach().cpu().numpy(),
            "regime": out.regime.detach().cpu().numpy(),
            "quantiles": quantiles.detach().cpu().numpy(),
        }


# ------------------------------------------------------------------------------
# Parity helper
# ------------------------------------------------------------------------------

def compare_backends(
    pack_dir: str,
    sample_csv: Optional[str] = None,
    atol: float = 1e-4,
) -> bool:
    """
    Run the same input through the ONNX and PyTorch backends and assert parity.

    Parameters
    ----------
    pack_dir : str
        Path to a v2 model pack directory.
    sample_csv : str, optional
        CSV of OHLCV bars used to build realistic input.  If omitted, random
        Gaussian data is used (sufficient for graph parity checks).
    atol : float, optional
        Maximum allowed absolute difference.  Defaults to ``1e-4``.

    Returns
    -------
    bool
        ``True`` if all output tensors match within ``atol``.

    Raises
    ------
    AssertionError
        If any output differs by more than ``atol``.
    RuntimeError
        If model.pt is missing or a backend cannot be initialised.
    """
    from Engine.v2.model_pack import ModelPackLoader

    pack = ModelPackLoader.load(pack_dir)
    config = pack["config"]

    if sample_csv is not None:
        import pandas as pd
        from Engine.v2.features import normalize_live_ohlcv, encode_live_session_features

        df = pd.read_csv(sample_csv, parse_dates=["Time"])
        if len(df) < config.max_seq_len:
            raise ValueError(
                f"Sample CSV has {len(df)} bars but model expects max_seq_len={config.max_seq_len}"
            )
        df = df.iloc[-config.max_seq_len :].reset_index(drop=True)
        x_raw = normalize_live_ohlcv(df)
        x_session = encode_live_session_features(
            pd.DatetimeIndex(df["Time"]),
            include_gap=(config.session_channels == 5),
        )
    else:
        rng = np.random.default_rng(42)
        x_raw = rng.standard_normal(
            (config.max_seq_len, config.in_channels), dtype=np.float32
        )
        x_session = rng.standard_normal(
            (config.max_seq_len, config.session_channels), dtype=np.float32
        )

    engine_onnx = V2InferenceEngine(pack, backend="onnx", device="cpu")
    engine_torch = V2InferenceEngine(pack, backend="pytorch", device="cpu")

    out_onnx = engine_onnx.predict(x_raw, x_session)
    out_torch = engine_torch.predict(x_raw, x_session)

    for key in ("mu", "log_sigma", "direction", "volatility", "regime", "quantiles"):
        a = out_onnx[key]
        b = out_torch[key]
        if a.shape != b.shape:
            raise AssertionError(
                f"Shape mismatch for {key}: ONNX {a.shape} vs PyTorch {b.shape}"
            )
        diff = float(np.max(np.abs(a - b)))
        if diff > atol:
            raise AssertionError(
                f"Backend parity failed for {key}: max abs diff={diff:.6e} > atol={atol}"
            )

    return True
