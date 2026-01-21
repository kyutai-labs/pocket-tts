"""Int8 quantization utilities for pocket-tts model.

This module provides functionality to quantize model components to int8
for reduced memory footprint and potentially faster inference.

Note: Quantization may affect audio quality. Use quality validation before
deploying quantized models in production.

Example:
    >>> from pocket_tts.utils.quantization import quantize_model
    >>> model = TTSModel.load_model()
    >>> quantized_model = quantize_model(model)
"""

import logging
from typing import Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)


def quantize_linear_layers(module: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Apply dynamic int8 quantization to all Linear layers in a module.

    Uses PyTorch's dynamic quantization which quantizes weights to int8
    at load time and quantizes activations dynamically during inference.

    Args:
        module: PyTorch module to quantize.
        dtype: Quantization dtype (default torch.qint8).

    Returns:
        Quantized module (modified in-place for efficiency).
    """
    try:
        quantized = torch.quantization.quantize_dynamic(module, {nn.Linear}, dtype=dtype)
        return quantized
    except Exception as e:
        logger.warning("Dynamic quantization failed: %s", e)
        return module


def quantize_model(model, components: Literal["all", "flow-lm", "mimi"] = "all") -> nn.Module:
    """Quantize pocket-tts model components to int8 for reduced memory.

    This applies dynamic int8 quantization to the specified components,
    reducing memory footprint by ~4x for quantized layers.

    Args:
        model: TTSModel instance to quantize.
        components: Which components to quantize - "all", "flow-lm", or "mimi".

    Returns:
        Model with quantized components (modified in-place).

    Warning:
        Quantization may affect audio quality. Always validate output
        quality after quantization before deploying.

    Example:
        >>> model = TTSModel.load_model()
        >>> model = quantize_model(model, components="flow-lm")
        >>> # Model now uses ~50% less memory for flow_lm
    """
    # Normalize component name
    components = components.replace("_", "-").lower()

    if components in ("all", "flow-lm"):
        logger.info("Quantizing FlowLM to int8...")
        original_params = sum(p.numel() for p in model.flow_lm.parameters())
        model.flow_lm = quantize_linear_layers(model.flow_lm)
        logger.info("FlowLM quantized (original params: %d)", original_params)

    if components in ("all", "mimi"):
        logger.info("Quantizing Mimi to int8...")
        # Only quantize the transformer parts, not the audio codec
        model.mimi.encoder_transformer = quantize_linear_layers(model.mimi.encoder_transformer)
        model.mimi.decoder_transformer = quantize_linear_layers(model.mimi.decoder_transformer)
        logger.info("Mimi transformers quantized")

    return model


def estimate_memory_savings(
    model, quantize_components: Literal["all", "flow-lm", "mimi"] = "all"
) -> dict[str, int]:
    """Estimate memory savings from int8 quantization.

    Args:
        model: TTSModel instance.
        quantize_components: Which components to estimate savings for.

    Returns:
        Dictionary with 'original_bytes', 'estimated_quantized_bytes', and 'savings_bytes'.
    """

    def count_bytes(module: nn.Module) -> int:
        total = 0
        for p in module.parameters():
            total += p.numel() * p.element_size()
        return total

    def estimate_quantized_bytes(module: nn.Module) -> int:
        """Estimate bytes after quantization (Linear layers go from 4 bytes to 1 byte per weight)."""
        total = 0
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear):
                # Linear layers compressed to 1 byte per weight (int8)
                total += child.weight.numel() * 1
                if child.bias is not None:
                    total += child.bias.numel() * child.bias.element_size()
            else:
                # Non-linear layers keep original size
                for p in child.parameters(recurse=False):
                    total += p.numel() * p.element_size()
        return total

    quantize_components = quantize_components.replace("_", "-").lower()
    original_bytes = 0
    estimated_bytes = 0

    if quantize_components in ("all", "flow-lm"):
        original_bytes += count_bytes(model.flow_lm)
        estimated_bytes += estimate_quantized_bytes(model.flow_lm)

    if quantize_components in ("all", "mimi"):
        original_bytes += count_bytes(model.mimi.encoder_transformer)
        original_bytes += count_bytes(model.mimi.decoder_transformer)
        estimated_bytes += estimate_quantized_bytes(model.mimi.encoder_transformer)
        estimated_bytes += estimate_quantized_bytes(model.mimi.decoder_transformer)

    return {
        "original_bytes": original_bytes,
        "estimated_quantized_bytes": estimated_bytes,
        "savings_bytes": original_bytes - estimated_bytes,
        "savings_percent": round((original_bytes - estimated_bytes) / original_bytes * 100, 1)
        if original_bytes > 0
        else 0,
    }
