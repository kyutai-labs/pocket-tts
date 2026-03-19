"""
Dynamic int8 quantization for pocket-tts.

Uses torchao if available (torch 2.10+ with C++ extensions), otherwise
falls back to torch.ao.quantization (deprecated but functional on torch 2.5-2.9).
"""

import logging
import platform

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# The recommended quantization config: best balance of speed and quality.
# - x86 (FBGEMM): ~27% faster than baseline
# - ARM (QNNPACK via torchao): ~16% faster than baseline
# - Runtime memory: ~48% reduction
# - No measurable impact on speech quality (WER unchanged)
RECOMMENDED_CONFIG = {"attention", "ffn"}

# All available quantization configs for evaluation/benchmarking.
# Groups correspond to functional parts of the FlowLM model:
#   "attention" — Q/K/V/output projections in transformer layers (~25M params)
#   "ffn"       — feed-forward linear1/linear2 in transformer layers (~50M params)
#   "flow_net"  — MLP sampler / flow matching network (~7M params)
CONFIGS = {
    "baseline": set(),
    "all": {"attention", "ffn", "flow_net"},
    "attention_ffn": {"attention", "ffn"},
    "attention": {"attention"},
    "ffn": {"ffn"},
    "flow_net": {"flow_net"},
    "flow_net_attention": {"flow_net", "attention"},
    "ffn_flow_net": {"ffn", "flow_net"},
}


def _get_backend():
    """Detect the best available quantization backend.

    Returns "torchao" if torchao is installed with working C++ extensions,
    otherwise returns "torch.ao".
    """
    try:
        import importlib.util

        if importlib.util.find_spec("torchao") is None:
            return "torch.ao"

        import torchao

        if hasattr(torchao, "_C") or not getattr(torchao, "_SKIPPED_CPP_EXTENSIONS", False):
            return "torchao"
    except ImportError:
        pass
    return "torch.ao"


def _quantize_module_torchao(module: nn.Module):
    """Apply int8 dynamic quantization using torchao."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    quantize_(module, Int8DynamicActivationInt8WeightConfig())


def _ensure_quantization_engine():
    """Set the quantization engine for torch.ao (QNNPACK for ARM, FBGEMM for x86)."""
    if platform.machine() in ("arm64", "aarch64"):
        torch.backends.quantized.engine = "qnnpack"
    elif torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "fbgemm"


def apply_dynamic_int8(flow_lm: nn.Module, quantize_groups: set[str]) -> nn.Module:
    """
    Apply dynamic int8 quantization to the specified layer groups of a FlowLM model.

    Automatically selects the best available backend:
    - torchao (torch 2.10+): optimized C++ kernels, faster on both ARM and x86
    - torch.ao (torch 2.5-2.9): deprecated but functional fallback

    Args:
        flow_lm: The FlowLM model (model.flow_lm)
        quantize_groups: Set of group keys to quantize.
            Valid keys: "attention", "ffn", "flow_net"

    Returns:
        The quantized model (modified in-place).
    """
    if not quantize_groups:
        logger.info("No quantization groups specified, returning model unchanged.")
        return flow_lm

    backend = _get_backend()
    logger.info("Using quantization backend: %s", backend)

    quantized_count = 0

    if backend == "torchao":
        quantized_count = _apply_torchao(flow_lm, quantize_groups)
    else:
        quantized_count = _apply_torch_ao(flow_lm, quantize_groups)

    logger.info("Quantized %d Linear modules (groups: %s)", quantized_count, quantize_groups)
    return flow_lm


def _apply_torchao(flow_lm: nn.Module, quantize_groups: set[str]) -> int:
    """Apply quantization using torchao backend."""
    quantized_count = 0

    if "flow_net" in quantize_groups:
        _quantize_module_torchao(flow_lm.flow_net)
        quantized_count += sum(
            1 for _, m in flow_lm.flow_net.named_modules() if isinstance(m, nn.Linear)
        )
        logger.info("Quantized flow_net (MLP sampler)")

    if "attention" in quantize_groups or "ffn" in quantize_groups:
        for layer in flow_lm.transformer.layers:
            if "attention" in quantize_groups:
                _quantize_module_torchao(layer.self_attn)
                quantized_count += 2

            if "ffn" in quantize_groups:
                wrapper1 = nn.Sequential(layer.linear1)
                wrapper2 = nn.Sequential(layer.linear2)
                _quantize_module_torchao(wrapper1)
                _quantize_module_torchao(wrapper2)
                layer.linear1 = wrapper1[0]
                layer.linear2 = wrapper2[0]
                quantized_count += 2

    return quantized_count


def _apply_torch_ao(flow_lm: nn.Module, quantize_groups: set[str]) -> int:
    """Apply quantization using deprecated torch.ao backend (fallback)."""
    from torch.ao.quantization import quantize_dynamic

    _ensure_quantization_engine()
    quantized_count = 0

    if "flow_net" in quantize_groups:
        quantize_dynamic(flow_lm.flow_net, {nn.Linear}, dtype=torch.qint8, inplace=True)
        quantized_count += sum(
            1 for _, m in flow_lm.flow_net.named_modules() if isinstance(m, nn.Linear)
        )
        logger.info("Quantized flow_net (MLP sampler)")

    if "attention" in quantize_groups or "ffn" in quantize_groups:
        for layer in flow_lm.transformer.layers:
            if "attention" in quantize_groups:
                quantize_dynamic(layer.self_attn, {nn.Linear}, dtype=torch.qint8, inplace=True)
                quantized_count += 2

            if "ffn" in quantize_groups:
                layer.linear1 = quantize_dynamic(
                    nn.Sequential(layer.linear1), {nn.Linear}, dtype=torch.qint8
                )[0]
                layer.linear2 = quantize_dynamic(
                    nn.Sequential(layer.linear2), {nn.Linear}, dtype=torch.qint8
                )[0]
                quantized_count += 2

    return quantized_count


def get_model_size_mb(model: nn.Module) -> float:
    """Returns model size in MB by serializing to a byte buffer.

    This correctly accounts for quantized weights stored in packed params,
    which don't appear in .parameters().
    """
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024**2)
