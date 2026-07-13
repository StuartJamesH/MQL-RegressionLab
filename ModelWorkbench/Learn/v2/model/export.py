"""ONNX export for the TradeForecastTransformer."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
from Learn.v2.model.full_model import TradeForecastTransformer, ModelOutput

_logger = logging.getLogger(__name__)


def export_to_onnx(
    model: TradeForecastTransformer,
    sample_input: Tuple[torch.Tensor, Optional[torch.Tensor]],
    path: str,
    opset_version: int = 18,
    verify: bool = True,
) -> str:
    """Export a TradeForecastTransformer to ONNX and optionally verify."""
    model.eval()
    x_raw, x_session = sample_input
    B, L, C = x_raw.shape

    if x_raw.dtype != torch.float32:
        x_raw = x_raw.to(torch.float32)
    if x_session is not None and x_session.dtype != torch.float32:
        x_session = x_session.to(torch.float32)

    if x_session is not None:
        model_args = (x_raw, x_session)
        input_names = ["x_raw", "x_session"]
        dynamic_axes_dict = {"x_raw": {0: "batch_size"}, "x_session": {0: "batch_size"}}
    else:
        model_args = (x_raw,)
        input_names = ["x_raw"]
        dynamic_axes_dict = {"x_raw": {0: "batch_size"}}

    output_names = ["mu", "log_sigma", "direction", "volatility", "regime", "quantiles"]
    dynamic_axes: dict = {}
    for name, axes in dynamic_axes_dict.items():
        dynamic_axes[name] = axes
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    class _ONNXWrapper(nn.Module):
        def __init__(self, model: TradeForecastTransformer):
            super().__init__()
            self.model = model

        def forward(self, x_raw: torch.Tensor, x_session: Optional[torch.Tensor] = None):
            output: ModelOutput = self.model(x_raw, x_session)
            if output.distribution is not None:
                mu, log_sigma = output.distribution
            else:
                mu = torch.zeros(B, self.model.config.n_horizons, device=x_raw.device)
                log_sigma = torch.zeros(B, self.model.config.n_horizons, device=x_raw.device)
            quantiles = output.quantiles
            if quantiles is None:
                quantiles = torch.zeros(
                    B, self.model.config.n_horizons, self.model.config.n_quantiles,
                    device=x_raw.device)
            return mu, log_sigma, output.direction, output.volatility, output.regime, quantiles

    wrapper = _ONNXWrapper(model)
    wrapper.eval()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, model_args, str(output_path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=opset_version,
            do_constant_folding=True,
        )

    _logger.info("ONNX model exported to %s (opset %d)", output_path, opset_version)
    if verify:
        _verify_onnx_output(wrapper, model_args, str(output_path))
    return str(output_path)


def _verify_onnx_output(pytorch_model: nn.Module, inputs: tuple, onnx_path: str,
                        atol: float = 1e-4) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        _logger.warning("onnx/onnxruntime not installed -- skipping verification.")
        return

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    with torch.no_grad():
        pytorch_outputs = pytorch_model(*inputs)
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"x_raw": inputs[0].cpu().numpy()}
    if len(inputs) > 1 and inputs[1] is not None:
        ort_inputs["x_session"] = inputs[1].cpu().numpy()
    ort_outputs = ort_session.run(None, ort_inputs)
    output_names = ["mu", "log_sigma", "direction", "volatility", "regime", "quantiles"]
    for i, (pt_out, ort_out) in enumerate(zip(pytorch_outputs, ort_outputs)):
        max_diff = float(abs(pt_out.detach().cpu().numpy() - ort_out).max())
        if max_diff >= atol:
            raise RuntimeError(
                f"ONNX verification FAILED for '{output_names[i]}': "
                f"max diff = {max_diff:.2e} (tol = {atol})")
    _logger.info("ONNX verification passed -- all %d outputs match within %.0e tolerance.",
                  len(pytorch_outputs), atol)
