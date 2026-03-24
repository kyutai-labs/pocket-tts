"""
Dynamic int8 quantization for pocket-tts.

Uses torchao if available (torch 2.10+ with C++ extensions), otherwise
falls back to torch.ao.quantization (deprecated but functional on torch 2.5-2.9).

Quantizes attention (Q/K/V/output projections) and FFN (linear1/linear2) layers
in the FlowLM transformer. The flow matching network and Mimi VAE decoder
remain in float32.
"""

import logging
import platform

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Used by load_model(quantize=True) to specify which layer groups to quantize.
RECOMMENDED_CONFIG = {"attention", "ffn"}


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

    if backend == "torchao":
        _apply_torchao(flow_lm, quantize_groups)
    else:
        _apply_torch_ao(flow_lm, quantize_groups)

    return flow_lm


def _apply_torchao(flow_lm: nn.Module, quantize_groups: set[str]) -> None:
    """Apply quantization using torchao backend."""
    if "flow_net" in quantize_groups:
        _quantize_module_torchao(flow_lm.flow_net)

    for layer in flow_lm.transformer.layers:
        if "attention" in quantize_groups:
            _quantize_module_torchao(layer.self_attn)

        if "ffn" in quantize_groups:
            wrapper1 = nn.Sequential(layer.linear1)
            wrapper2 = nn.Sequential(layer.linear2)
            _quantize_module_torchao(wrapper1)
            _quantize_module_torchao(wrapper2)
            layer.linear1 = wrapper1[0]
            layer.linear2 = wrapper2[0]


def _apply_torch_ao(flow_lm: nn.Module, quantize_groups: set[str]) -> None:
    """Apply quantization using deprecated torch.ao backend (fallback)."""
    from torch.ao.quantization import quantize_dynamic

    _ensure_quantization_engine()

    if "flow_net" in quantize_groups:
        quantize_dynamic(flow_lm.flow_net, {nn.Linear}, dtype=torch.qint8, inplace=True)

    for layer in flow_lm.transformer.layers:
        if "attention" in quantize_groups:
            quantize_dynamic(layer.self_attn, {nn.Linear}, dtype=torch.qint8, inplace=True)

        if "ffn" in quantize_groups:
            layer.linear1 = quantize_dynamic(
                nn.Sequential(layer.linear1), {nn.Linear}, dtype=torch.qint8
            )[0]
            layer.linear2 = quantize_dynamic(
                nn.Sequential(layer.linear2), {nn.Linear}, dtype=torch.qint8
            )[0]
